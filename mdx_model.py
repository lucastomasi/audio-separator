"""MDX ONNX core. Imported by uvr_runtime."""
import os
import gc
import hashlib
import queue
import threading
import numpy as np
import soundfile as sf
import audio_io as librosa
from utils import logger

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

torch = None
ort = None
tqdm = None
_ORT_SESSIONS = {}
_MODEL_HASHES = {}


def _ensure_ml():
    global torch, ort, tqdm
    if torch is None:
        import torch as _torch
        import onnxruntime as _ort
        from tqdm import tqdm as _tqdm
        torch = _torch
        ort = _ort
        tqdm = _tqdm
    return torch, ort, tqdm


def _ort_session(model_path, providers):
    _ensure_ml()
    key = (os.path.abspath(model_path), tuple(providers))
    session = _ORT_SESSIONS.get(key)
    if session is not None:
        return session
    options = ort.SessionOptions()
    # One thread: Intel Mac + Gradio already share the core; dual ORT workers
    # thrash during Separar without a real speed win on this hardware.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=providers
    )
    _ORT_SESSIONS[key] = session
    return session


def clear_ort_sessions():
    """Drop cached ONNX sessions so Separar does not pin RAM until quit."""
    for key in list(_ORT_SESSIONS):
        session = _ORT_SESSIONS.pop(key, None)
        del session
    _MODEL_HASHES.clear()
    gc.collect()


stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless",
}


class MDXModel:
    def __init__(
        self,
        device,
        dim_f,
        dim_t,
        n_fft,
        hop=1024,
        stem_name=None,
        compensation=1.000,
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation
        _ensure_ml()

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(
            window_length=self.n_fft, periodic=True
        ).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(
        self, model_path: str, params: MDXModel, processor=0
    ):
        _ensure_ml()
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )

        self.model = params
        self.ort = _ort_session(model_path, self.provider)
        self.process = lambda spec: self.ort.run(
            None, {"input": spec.cpu().numpy()}
        )[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        path = os.path.abspath(model_path)
        cached = _MODEL_HASHES.get(path)
        if cached:
            return cached
        with open(path, "rb") as handle:
            try:
                handle.seek(-10000 * 1024, 2)
            except OSError:
                handle.seek(0)
            model_hash = hashlib.md5(handle.read()).hexdigest()
        _MODEL_HASHES[path] = model_hash
        return model_hash

    @staticmethod
    def segment(
        wave,
        combine=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        margin_size=DEFAULT_MARGIN_SIZE,
    ):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array.
                If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            # Initializing as None instead of [] for later numpy array concatenation
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if sample_count <= 0:
                return []

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(
                range(0, sample_count, chunk_size)
            ):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate(
            (
                np.zeros((2, trim), dtype=np.float32),
                wave,
                np.zeros((2, pad), dtype=np.float32),
                np.zeros((2, trim), dtype=np.float32),
            ),
            1,
        )

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        # torch.tensor(list_of_arrays) copies window by window and stalls Separar.
        stacked = np.stack(mix_waves, axis=0).astype(np.float32, copy=False)
        mix_waves = torch.from_numpy(stacked).to(self.device)

        return mix_waves, pad, trim

    def _infer_windows(self, mix_waves, trim, pad, on_chunk=None):
        """Run STFT → ONNX → iSTFT on this thread. Returns (2, n_sample)."""
        windows = mix_waves.split(1)
        total = len(windows)
        if total == 0:
            raise RuntimeError("MDX no armó ventanas de audio.")
        pw = []
        with torch.no_grad():
            for index, mix_wave in enumerate(windows, start=1):
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(
                    processed_spec.to(self.device)
                )
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
                if on_chunk is not None:
                    on_chunk(index, total)
        processed_signal = np.concatenate(pw, axis=-1)
        if pad:
            processed_signal = processed_signal[:, :-pad]
        return processed_signal

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        """GPU path only: one worker per chunk group. CPU stays inline."""
        try:
            q.put({_id: self._infer_windows(mix_waves, trim, pad)})
        except Exception as exc:
            q.put({_id: exc})

    def process_wave(self, wave: np.array, mt_threads=1, on_chunk=None):
        """
        Process the wave array.

        On CPU (mt_threads <= 1) inference stays on the caller thread.
        The ONNX session is created on that thread; Run() from a worker
        deadlocks on Intel Mac and the Separar progress bar never moves.
        """
        if getattr(wave, "ndim", 0) != 2 or wave.shape[0] != 2:
            raise ValueError("MDX necesita audio estéreo.")
        if wave.shape[-1] == 0:
            return np.zeros_like(wave)
        mt_threads = max(1, int(mt_threads or 1))
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        prepared = []
        for batch in waves:
            prepared.append(self.pad_wave(batch))

        # One group on CPU: do not spawn a thread.
        if mt_threads <= 1 or len(prepared) <= 1:
            total_windows = sum(mix.shape[0] for mix, _pad, _trim in prepared)
            cursor = 0
            processed_batches = []
            for mix_waves, pad, trim in prepared:
                base = cursor

                def _one(done, _total, start=base):
                    if on_chunk is not None and total_windows:
                        on_chunk(start + done, total_windows)

                processed_batches.append(
                    self._infer_windows(mix_waves, trim, pad, on_chunk=_one)
                )
                cursor += mix_waves.shape[0]
            return self.segment(processed_batches, True, chunk)

        q = queue.Queue()
        threads = []
        for c, (mix_waves, pad, trim) in enumerate(prepared):
            thread = threading.Thread(
                target=self._process_wave, args=(mix_waves, trim, pad, q, c)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [
            list(item.values())[0]
            for item in sorted(
                processed_batches, key=lambda d: list(d.keys())[0]
            )
        ]
        if len(processed_batches) != len(waves):
            raise RuntimeError(
                "MDX no terminó todas las ventanas. Reintentá Separar."
            )
        for item in processed_batches:
            if isinstance(item, Exception):
                raise item
        return self.segment(processed_batches, True, chunk)


def _as_stereo(wave):
    """MDX STFT expects two channels. Do not mutate the caller's array."""
    wave = np.asarray(wave)
    if wave.ndim == 1:
        wave = np.stack([wave, wave], axis=0)
    elif wave.ndim == 2 and wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)
    elif wave.ndim == 2 and wave.shape[0] > 2:
        wave = wave[:2]
    if wave.ndim != 2 or wave.shape[0] != 2:
        raise ValueError("MDX necesita audio estéreo.")
    return np.ascontiguousarray(wave, dtype=np.float32)


def _peak_norm(wave):
    """Copy normalized to peak 1. The mix used for the instrumental stays intact."""
    if wave.size == 0:
        return wave.copy(), 0.0
    peak = float(np.max(np.abs(wave)))
    if peak > 1e-6:
        return (wave / np.float32(peak)).astype(np.float32, copy=False), peak
    return wave.copy(), 0.0


def _restore_peak(processed, peak):
    if peak > 1e-6:
        return processed * np.float32(peak)
    return processed


def _report_chunks(progress, span, done, total, desc):
    if progress is None or not total:
        return
    start, end = span
    frac = start + (end - start) * (float(done) / float(total))
    if frac < 0.0:
        frac = 0.0
    if frac > 0.99:
        frac = 0.99
    try:
        progress(float(frac), desc=f"{desc} {int(done)}/{int(total)}")
    except Exception:
        return


def _execute_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main,
    exclude_inversion,
    suffix,
    invert_suffix,
    denoise,
    keep_orig,
    m_threads,
    device,
    processor_num,
    progress,
    progress_span,
    progress_desc,
):
    loaded, sr = librosa.load(filename, mono=False, sr=44100)
    # Keep the original mix. Dividing it in place makes the instrumental
    # residual (mix - voz) use the wrong gain, so the voice never leaves.
    wave = _as_stereo(loaded)
    if wave.shape[-1] == 0:
        raise ValueError("El audio está vacío.")

    torch, _ort, _tqdm = _ensure_ml()
    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    if not mp or "mdx_dim_f_set" not in mp:
        raise ValueError(
            "Falta la ficha de este modelo UVR. Pulsá Completar instalación."
        )
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    norm_wave, peak = _peak_norm(wave)

    def _run(audio, span):
        def on_chunk(done, total):
            _report_chunks(progress, span, done, total, progress_desc)

        return mdx_sess.process_wave(audio, m_threads, on_chunk=on_chunk)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if denoise:
            mid = (progress_span[0] + progress_span[1]) / 2.0
            wave_processed = -_run(-norm_wave, (progress_span[0], mid)) + _run(
                norm_wave, (mid, progress_span[1])
            )
            wave_processed *= 0.5
        else:
            wave_processed = _run(norm_wave, progress_span)
    finally:
        torch.set_num_threads(previous_threads)
    wave_processed = _restore_peak(wave_processed, peak)
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = (
            stem_naming.get(stem_name)
            if invert_suffix is None
            else invert_suffix
        )
        stem_name = (
            f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        )
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(
            invert_filepath,
            (-wave_processed.T * model.compensation) + wave.T,
            sr,
        )

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return main_filepath, invert_filepath


@spaces.GPU(duration=(40/1.5))
def run_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
    device_base="cuda",
    progress=None,
    progress_span=(0.2, 0.9),
    progress_desc="Separando…",
):

    torch, _ort, _tqdm = _ensure_ml()
    if device_base == "cuda":
        device = torch.device("cuda:0")
        processor_num = 0
        device_properties = torch.cuda.get_device_properties(device)
        vram_gb = device_properties.total_memory / 1024**3
        m_threads = 1 if vram_gb < 8 else (8 if vram_gb > 32 else 2)
        duration = librosa.get_duration(filename=filename)
        if duration < 60:
            m_threads = 1
        logger.info(f"threads: {m_threads} vram: {vram_gb}")
    else:
        device = torch.device("cpu")
        processor_num = -1
        m_threads = 1

    return _execute_mdx(
        model_params,
        output_dir,
        model_path,
        filename,
        exclude_main,
        exclude_inversion,
        suffix,
        invert_suffix,
        denoise,
        keep_orig,
        m_threads,
        device,
        processor_num,
        progress,
        progress_span,
        progress_desc,
    )


def run_mdx_beta(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
    device_base="",
    progress=None,
    progress_span=(0.2, 0.9),
    progress_desc="Separando…",
):

    torch, _ort, _tqdm = _ensure_ml()
    m_threads = 1
    logger.info("threads: 1 (CPU)")
    return _execute_mdx(
        model_params,
        output_dir,
        model_path,
        filename,
        exclude_main,
        exclude_inversion,
        suffix,
        invert_suffix,
        denoise,
        keep_orig,
        m_threads,
        torch.device("cpu"),
        -1,
        progress,
        progress_span,
        progress_desc,
    )


