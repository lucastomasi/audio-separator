"""Fine-tune one RVC voice from dry recordings.

The result is an inference .pth plus an optional .index in rvc_models/.
The discriminator stays out of that folder so it is not offered as a voice.
Speaker id stays 0, which is what conversion uses.
"""
import os
import re

import numpy as np

from app_paths import data_dir
from audio_io import load, resample
from rvc_engine import RVC_DIR, _hubert_weight_file, local_hubert_path, local_rmvpe_path

SAMPLE_RATE = 40000
HUBERT_RATE = 16000
HOP = 400
N_FFT = 2048
WIN = 2048
N_MELS = 125
SEGMENT_FRAMES = 32
PHONE_DIM = 768
F0_MIN = 50.0
F0_MAX = 1100.0
EPOCHS_DEFAULT = 20
EPOCHS_MAX = 200

CONFIG_LIST = [
    N_FFT // 2 + 1,
    SEGMENT_FRAMES,
    192,
    192,
    768,
    2,
    6,
    3,
    0,
    "1",
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [10, 10, 2, 2],
    512,
    [16, 16, 4, 4],
    109,
    256,
    SAMPLE_RATE,
]

_MODEL = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "spk_embed_dim": 109,
    "gin_channels": 256,
}


def voice_name(name):
    cleaned = re.sub(r"[^\w.-]+", "_", str(name or ""), flags=re.UNICODE).strip("._")
    if not cleaned:
        raise ValueError("Poné un nombre para la voz.")
    return cleaned[:60]


def slice_voice(wave, sample_rate, min_seconds=2.0, max_seconds=8.0):
    """Return voiced pieces. Silence and very short bits are left out."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size == 0 or sample_rate <= 0:
        return []
    frame = max(1, int(0.02 * sample_rate))
    rms = []
    for start in range(0, max(wave.size - frame, 0) + 1, frame):
        chunk = wave[start : start + frame]
        if chunk.size == 0:
            break
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not rms:
        return []
    threshold = max(0.01, float(np.median(rms)) * 0.35)
    voiced = [level >= threshold for level in rms]
    spans = []
    begin = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and begin is None:
            begin = index
        if begin is not None and (not is_voiced or index == len(voiced) - 1):
            end = index if not is_voiced else index + 1
            if end - begin >= 3:
                spans.append((begin * frame, min(end * frame, wave.size)))
            begin = None
    pieces = []
    min_samples = int(min_seconds * sample_rate)
    max_samples = int(max_seconds * sample_rate)
    for start, end in spans:
        length = end - start
        if length < min_samples:
            continue
        cursor = start
        while cursor < end:
            stop = min(cursor + max_samples, end)
            if stop - cursor >= min_samples:
                pieces.append(wave[cursor:stop].copy())
            cursor = stop
    if not pieces and wave.size >= min_samples and float(np.max(np.abs(wave))) >= 0.01:
        for cursor in range(0, wave.size, max_samples):
            piece = wave[cursor : cursor + max_samples]
            if piece.size >= min_samples:
                pieces.append(piece.copy())
    return pieces


def quantize_f0(f0):
    """Map Hz to the 1..255 bins the pitch embedding expects."""
    values = np.asarray(f0, dtype=np.float32).reshape(-1)
    coarse = np.ones(values.shape, dtype=np.int64)
    voiced = values > 1.0
    if not np.any(voiced):
        return coarse
    mel = 1127.0 * np.log(1.0 + values[voiced] / 700.0)
    mel_min = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
    mel_max = 1127.0 * np.log(1.0 + F0_MAX / 700.0)
    bins = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0
    bins = np.rint(np.clip(bins, 1, 255)).astype(np.int64)
    coarse[voiced] = bins
    return coarse


def export_inference_checkpoint(state, dest, epochs):
    """Write the small checkpoint conversion already knows how to load."""
    weight = {}
    for key, value in state.items():
        if "enc_q" in key:
            continue
        tensor = value.detach().cpu().float()
        if tensor.is_floating_point():
            tensor = tensor.half()
        weight[key] = tensor
    packed = {
        "weight": weight,
        "config": list(CONFIG_LIST),
        "f0": 1,
        "version": "v2",
        "sr": "40k",
        "info": "%sepoch" % int(epochs),
    }
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    import torch

    torch.save(packed, dest)
    return dest


def _report(on_progress, frac, message):
    if on_progress is None:
        return
    on_progress(max(0.0, min(1.0, float(frac))), message)


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _mel_basis(n_fft, device, dtype):
    import torch

    try:
        from librosa.filters import mel as librosa_mel

        basis = librosa_mel(
            sr=SAMPLE_RATE, n_fft=n_fft, n_mels=N_MELS, fmin=0.0, fmax=None
        )
    except Exception:
        basis = _numpy_mel(SAMPLE_RATE, n_fft, N_MELS)
    return torch.from_numpy(np.asarray(basis, dtype=np.float32)).to(device=device, dtype=dtype)


def _numpy_mel(sample_rate, n_fft, n_mels):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    fmax = sample_rate / 2.0
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    basis = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for freq in range(left, center):
            if 0 <= freq < basis.shape[1]:
                basis[index, freq] = (freq - left) / (center - left)
        for freq in range(center, right):
            if 0 <= freq < basis.shape[1]:
                basis[index, freq] = (right - freq) / (right - center)
    return basis


def _spectrogram(wave, device):
    import torch
    import torch.nn.functional as F

    audio = torch.as_tensor(wave, dtype=torch.float32, device=device).view(1, -1)
    pad = int((N_FFT - HOP) / 2)
    audio = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(WIN, device=device, dtype=torch.float32)
    spec = torch.stft(
        audio,
        N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 2e-7)


def _mel_from_spec(spec):
    import torch

    basis = _mel_basis(N_FFT, spec.device, spec.dtype)
    return torch.log(torch.clamp(basis @ spec, min=2e-6))


def _mel_from_wave(wave):
    spec = _spectrogram(wave.squeeze(1) if wave.dim() == 3 else wave, wave.device)
    return _mel_from_spec(spec)


def _feature_loss(real_maps, fake_maps):
    import torch

    loss = 0.0
    for real, fake in zip(real_maps, fake_maps):
        for real_layer, fake_layer in zip(real, fake):
            loss = loss + torch.mean(torch.abs(real_layer.float().detach() - fake_layer.float()))
    return loss * 2


def _discriminator_loss(real_outputs, fake_outputs):
    import torch

    loss = 0.0
    for real, fake in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((1 - real.float()) ** 2)
        loss = loss + torch.mean(fake.float() ** 2)
    return loss


def _generator_loss(fake_outputs):
    import torch

    loss = 0.0
    for fake in fake_outputs:
        loss = loss + torch.mean((1 - fake.float()) ** 2)
    return loss


def _kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    import torch

    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask).clamp_min(1.0)


def _load_hubert(folder):
    import torch

    try:
        from transformers import HubertConfig, HubertModel
    except Exception as exc:
        raise ValueError("No pude cargar HuBERT para entrenar.") from exc

    class HubertWithProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            size = getattr(config, "classifier_proj_size", None) or 256
            self.final_proj = torch.nn.Linear(config.hidden_size, size)

    try:
        return HubertWithProj.from_pretrained(folder)
    except Exception:
        config = HubertConfig.from_pretrained(folder)
        if not getattr(config, "classifier_proj_size", None):
            config.classifier_proj_size = 256
        model = HubertWithProj(config)
        weights = _hubert_weight_file(folder)
        if not weights:
            raise ValueError("HuBERT no tiene los pesos. Volvé a bajar los modelos.")
        if str(weights).endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(weights)
        else:
            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model


def _hubert_frames(model, audio_16k, device):
    import torch

    source = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).view(1, -1).to(device)
    with torch.no_grad():
        output = model(source, output_hidden_states=True)
        states = output.hidden_states
        if not states or len(states) <= 12:
            raise ValueError(
                "HuBERT no devolvió la capa de la voz. Volvé a bajar los modelos."
            )
        hidden = states[12]
    frames = hidden.squeeze(0).float().cpu().numpy()
    return np.repeat(frames, 2, axis=0)


def _pitch_of(rmvpe, audio_16k):
    f0 = np.asarray(rmvpe.infer_from_audio(np.asarray(audio_16k, dtype=np.float32)), dtype=np.float32)
    f0[(f0 < F0_MIN) | (f0 > F0_MAX)] = 0
    return f0


def _clip_example(wave_40k, hubert, rmvpe, device):
    wave_16k = np.asarray(resample(wave_40k, SAMPLE_RATE, HUBERT_RATE), dtype=np.float32).reshape(-1)
    phone = _hubert_frames(hubert, wave_16k, device)
    pitchf = _pitch_of(rmvpe, wave_16k)
    spec = _spectrogram(wave_40k, "cpu").squeeze(0)
    frames = min(phone.shape[0], pitchf.shape[0], spec.shape[1])
    if frames < SEGMENT_FRAMES + 2 or phone.shape[1] != PHONE_DIM:
        return None
    samples = frames * HOP
    return {
        "phone": phone[:frames],
        "pitch": quantize_f0(pitchf[:frames]),
        "pitchf": pitchf[:frames],
        "spec": spec[:, :frames].cpu(),
        "wave": np.asarray(wave_40k[:samples], dtype=np.float32),
    }


def _ensure_rvc_namespace():
    """Make the model files importable.

    The Mac app can import the package normally. If that entrypoint fails,
    load the model files without it so training still starts. A later
    conversion in the same process keeps the real package when it loaded.
    """
    import sys
    import types
    from importlib.machinery import PathFinder

    name = "infer_rvc_python"
    existing = sys.modules.get(name)
    if existing is not None and (
        getattr(existing, "__file__", None) or getattr(existing, "__path__", None)
    ):
        return
    try:
        __import__(name)
    except Exception:
        sys.modules.pop(name, None)
    loaded = sys.modules.get(name)
    if loaded is not None and getattr(loaded, "__file__", None):
        return
    spec = PathFinder.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("No pude cargar el entrenamiento. Falta infer-rvc-python.")
    module = types.ModuleType(name)
    module.__path__ = list(spec.submodule_search_locations)
    module.__package__ = name
    sys.modules[name] = module


def _build_models(device):
    _ensure_rvc_namespace()
    from infer_rvc_python.lib.infer_pack.models import (
        MultiPeriodDiscriminatorV2,
        SynthesizerTrnMs768NSFsid,
    )

    generator = SynthesizerTrnMs768NSFsid(
        N_FFT // 2 + 1,
        SEGMENT_FRAMES,
        **_MODEL,
        sr=SAMPLE_RATE,
        is_half=False,
    )
    discriminator = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
    return generator.to(device), discriminator.to(device)


def _load_pretrained(generator, discriminator, generator_path, discriminator_path):
    import torch

    saved_g = torch.load(generator_path, map_location="cpu")
    saved_d = torch.load(discriminator_path, map_location="cpu")
    if isinstance(saved_g, dict) and "model" in saved_g:
        saved_g = saved_g["model"]
    if isinstance(saved_d, dict) and "model" in saved_d:
        saved_d = saved_d["model"]
    if not isinstance(saved_g, dict) or "emb_g.weight" not in saved_g:
        raise ValueError("No pude cargar el modelo base de la voz. Volvé a bajarlo.")
    if not isinstance(saved_d, dict):
        raise ValueError("No pude cargar el modelo base de la voz. Volvé a bajarlo.")
    try:
        missing, _unexpected = generator.load_state_dict(saved_g, strict=False)
        discriminator.load_state_dict(saved_d, strict=False)
    except Exception as exc:
        raise ValueError(
            "No pude cargar el modelo base de la voz. Volvé a bajarlo."
        ) from exc
    if "emb_g.weight" in missing:
        raise ValueError("No pude cargar el modelo base de la voz. Volvé a bajarlo.")


def training_step(generator, discriminator, optim_g, optim_d, example, device):
    """One generator and discriminator update. Returns the mel loss as a float."""
    import torch
    import torch.nn.functional as F

    phone = torch.as_tensor(example["phone"], dtype=torch.float32, device=device).unsqueeze(0)
    phone_lengths = torch.tensor([phone.shape[1]], dtype=torch.long, device=device)
    pitch = torch.as_tensor(example["pitch"], dtype=torch.long, device=device).unsqueeze(0)
    pitchf = torch.as_tensor(example["pitchf"], dtype=torch.float32, device=device).unsqueeze(0)
    spec = example["spec"].to(device=device, dtype=torch.float32).unsqueeze(0)
    spec_lengths = torch.tensor([spec.shape[-1]], dtype=torch.long, device=device)
    wave = torch.as_tensor(example["wave"], dtype=torch.float32, device=device).view(1, 1, -1)
    sid = torch.zeros(1, dtype=torch.long, device=device)

    generator.train()
    discriminator.train()
    y_hat, ids_slice, _x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = generator(
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
    )
    mel = _mel_from_spec(spec)
    y_mel = _slice_frames(mel, ids_slice, SEGMENT_FRAMES)
    y_hat_mel = _mel_from_wave(y_hat.float())
    wave_slice = _slice_samples(wave, ids_slice * HOP, SEGMENT_FRAMES * HOP)

    real_out, fake_out, _, _ = discriminator(wave_slice, y_hat.detach())
    loss_d = _discriminator_loss(real_out, fake_out)
    optim_d.zero_grad()
    loss_d.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1000.0)
    optim_d.step()

    _real_out, fake_out, real_maps, fake_maps = discriminator(wave_slice, y_hat)
    loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * 45
    loss_kl = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
    loss_fm = _feature_loss(real_maps, fake_maps)
    loss_gen = _generator_loss(fake_out)
    loss_g = loss_gen + loss_fm + loss_mel + loss_kl
    optim_g.zero_grad()
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1000.0)
    optim_g.step()
    return float(loss_mel.detach().cpu())


def _slice_frames(spec, ids, frames):
    import torch

    gathered = []
    for index, start in enumerate(ids):
        start = int(start)
        gathered.append(spec[index, :, start : start + frames])
    return torch.stack(gathered)


def _slice_samples(wave, starts, samples):
    import torch

    gathered = []
    for index, start in enumerate(starts):
        start = int(start)
        gathered.append(wave[index, :, start : start + samples])
    return torch.stack(gathered)


def _write_index(features, dest):
    if not features:
        return None
    try:
        import faiss
    except Exception:
        return None
    stacked = np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float32)
    if stacked.ndim != 2 or stacked.shape[1] != PHONE_DIM or stacked.shape[0] < 1:
        return None
    index = faiss.IndexFlatL2(PHONE_DIM)
    index.add(stacked)
    faiss.write_index(index, dest)
    return dest


def _prepare_examples(paths, hubert, rmvpe, device, on_progress):
    examples = []
    index_rows = []
    total = max(1, len(paths))
    for index, path in enumerate(paths):
        _report(on_progress, 0.08 + 0.25 * index / total, "Leyendo las tomas…")
        wave, _sr = load(path, mono=True, sr=SAMPLE_RATE)
        for piece in slice_voice(wave, SAMPLE_RATE):
            example = _clip_example(piece, hubert, rmvpe, device)
            if example is None:
                continue
            examples.append(example)
            index_rows.append(example["phone"][::2])
    real_count = len(examples)
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    silent = _clip_example(silence, hubert, rmvpe, device)
    if silent is not None:
        examples.append(silent)
    if real_count < 1:
        raise ValueError(
            "Hace falta más audio de esa voz, sola, sin música. "
            "Unos segundos cantados o hablados alcanzan para empezar."
        )
    return examples, index_rows, real_count


def _run_epochs(generator, discriminator, examples, epochs, device, on_progress):
    import torch

    optim_g = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.8, 0.99), eps=1e-9)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.8, 0.99), eps=1e-9)
    order = list(range(len(examples)))
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        for step, index in enumerate(order):
            training_step(
                generator, discriminator, optim_g, optim_d, examples[index], device
            )
            done = (epoch - 1) * len(order) + step + 1
            total = epochs * len(order)
            _report(
                on_progress,
                0.35 + 0.6 * done / total,
                f"Entrenando, vuelta {epoch} de {epochs}…",
            )
        for optim in (optim_g, optim_d):
            for group in optim.param_groups:
                group["lr"] = max(group["lr"] * 0.999875, 1e-6)


def train_voice(audio_paths, name, epochs=EPOCHS_DEFAULT, on_progress=None):
    """Train one voice. Return (pth_path, note)."""
    import model_fetch

    label = voice_name(name)
    paths = []
    for path in audio_paths or []:
        if path and os.path.isfile(path):
            paths.append(os.path.abspath(path))
    if not paths:
        raise ValueError("Subí tomas de esa voz, o usá la voz que ya separaste.")
    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        epochs = EPOCHS_DEFAULT
    epochs = min(EPOCHS_MAX, max(1, epochs))

    _report(on_progress, 0.02, "Bajando HuBERT, el pitch y el modelo base…")
    model_fetch.ensure_rvc_support(RVC_DIR, on_progress=on_progress)
    _downloaded, generator_path, discriminator_path = model_fetch.ensure_training_bases(
        RVC_DIR, on_progress=on_progress
    )
    hubert_path = local_hubert_path()
    rmvpe_path = local_rmvpe_path()
    if not hubert_path or not os.path.isdir(hubert_path):
        raise ValueError("Falta HuBERT. Tocá Bajá los modelos y volvé a entrenar.")
    if not rmvpe_path:
        raise ValueError("Falta el estimador de pitch. Tocá Bajá los modelos y volvé a entrenar.")

    device = _pick_device()
    try:
        return _train_on(
            paths, label, epochs, device, hubert_path, rmvpe_path,
            generator_path, discriminator_path, on_progress,
        )
    except ValueError:
        raise
    except Exception as exc:
        if device == "cpu":
            raise ValueError(f"No pude entrenar la voz: {exc}") from exc
        _report(on_progress, 0.05, "El chip no pudo entrenar. Sigo en CPU…")
        try:
            return _train_on(
                paths, label, epochs, "cpu", hubert_path, rmvpe_path,
                generator_path, discriminator_path, on_progress, cpu_note=True,
            )
        except ValueError:
            raise
        except Exception as retry_exc:
            raise ValueError(f"No pude entrenar la voz: {retry_exc}") from retry_exc


def _train_on(
    paths, label, epochs, device, hubert_path, rmvpe_path,
    generator_path, discriminator_path, on_progress, cpu_note=False,
):
    import torch

    _ensure_rvc_namespace()
    from infer_rvc_python.lib.rmvpe import RMVPE

    hubert = _load_hubert(hubert_path).to(device).eval()
    rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)
    examples, index_rows, real_count = _prepare_examples(
        paths, hubert, rmvpe, device, on_progress
    )
    del hubert, rmvpe
    if device == "cuda":
        torch.cuda.empty_cache()

    generator, discriminator = _build_models(device)
    _load_pretrained(generator, discriminator, generator_path, discriminator_path)
    _run_epochs(generator, discriminator, examples, epochs, device, on_progress)

    os.makedirs(RVC_DIR, exist_ok=True)
    dest = os.path.join(RVC_DIR, f"{label}.pth")
    export_inference_checkpoint(generator.state_dict(), dest, epochs)
    index_path = _write_index(index_rows, os.path.join(RVC_DIR, f"{label}.index"))
    note = f"Modelo v2 a 40 kHz. {real_count} tomas, {epochs} vueltas."
    if index_path:
        note += " Índice listo."
    else:
        note += " Sin índice."
    if cpu_note:
        note += " El chip no pudo entrenar. Seguí en CPU."
    elif device == "cpu":
        note += " Entrenó en CPU."
    else:
        note += " Entrenó en el chip."
    _report(on_progress, 1.0, "Listo.")
    return dest, note
