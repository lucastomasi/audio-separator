import os
import shutil
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
import gc
import hashlib
import queue
import threading
import json
import shlex
import sys
import subprocess
import audio_io as librosa
from audio_text import (
    STEM_AMBAS,
    STEM_SOLO_INST,
    STEM_SOLO_VOZ,
    stem_choice_to_list,
)
from exports import copy_to_downloads, open_exports_dir
from ui_status import (
    KIND_ERROR,
    KIND_OK,
    KIND_RUN,
    MSG_SEPARATE,
    RUN_SEPARATE,
    fail,
    status_update,
)
import numpy as np
import soundfile as sf
from utils import (
    remove_directory_contents,
    create_directories,
)
import random
from utils import logger
import warnings
import gradio as gr
import time
import traceback

warnings.filterwarnings("ignore")
IS_COLAB = "google.colab" in sys.modules
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

from mdx_model import (
    MDX,
    MDXModel,
    run_mdx,
    run_mdx_beta,
    _ensure_ml,
    _MODEL_HASHES,
)

UVR_MODELS = [
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mdxnet_models_dir = os.path.join(BASE_DIR, "mdx_models")
output_dir = os.path.join(
    __import__("app_env").data_dir(), "Trabajos", "Separar"
)


def _audio_path(value):
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if isinstance(value, (list, tuple)) and value:
        return _audio_path(value[0])
    return str(value) if value else None


def convert_to_stereo_and_wav(audio_path):
    """UVR/MDX expects 44.1 kHz stereo WAV. YouTube work files are 48 kHz."""
    audio_path = _audio_path(audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta el archivo de audio.")
    try:
        info = sf.info(audio_path)
        if (
            str(audio_path).lower().endswith(".wav")
            and info.channels == 2
            and int(info.samplerate) == 44100
        ):
            return audio_path
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    stereo_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_44100_stereo.wav"
    stereo_path = os.path.join(output_dir, stereo_name)
    from youtube_lib import ffmpeg_binary

    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg.")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ac",
        "2",
        "-ar",
        "44100",
        "-f",
        "wav",
        stereo_path,
    ]
    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }
    process_wav = subprocess.Popen(command, **sub_params)
    _out, errors = process_wav.communicate()
    if process_wav.returncode != 0 or not os.path.isfile(stereo_path):
        err = (errors or b"").decode("utf-8", "ignore")[:300]
        logger.error("ffmpeg stereo: %s", err)
        raise ValueError("No pude pasar el audio a WAV 44.1 kHz estéreo.")
    return stereo_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


def ensure_uvr_model(filename, progress=None):
    """Use ONNX already on disk or in the local can. Never download mid-job."""
    dest = os.path.join(mdxnet_models_dir, filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        from pathlib import Path

        from install_rvc_assets import _link_or_copy, can_root

        can_file = can_root() / "mdx_models" / filename
        if can_file.is_file() and can_file.stat().st_size > 0:
            _tick(progress, 0.05, f"Usando modelo UVR local: {filename}")
            _link_or_copy(can_file, Path(dest))
            return dest
    except Exception:
        pass
    raise ValueError(
        f"Falta el modelo UVR ({filename}). Pulsá Completar instalación."
    )


def random_sleep():
    sleep_time = 0.1
    if IS_ZERO_GPU:
        sleep_time = round(random.uniform(3.2, 5.9), 1)
    time.sleep(sleep_time)


def _tick(progress, frac, desc):
    if progress is None:
        return
    try:
        progress(frac, desc=desc)
    except Exception:
        pass


def process_uvr_task(
    orig_song_path: str = "aud_test.mp3",
    main_vocals: bool = False,
    dereverb: bool = True,
    song_id: str = "mdx",  # folder output name
    only_voiceless: bool = False,
    remove_files_output_dir: bool = False,
    progress=None,
):

    torch, ort, _tqdm = _ensure_ml()
    device_base = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_base}")

    if remove_files_output_dir:
        remove_directory_contents(output_dir)

    with open(os.path.join(mdxnet_models_dir, "data.json")) as infile:
        mdx_model_params = json.load(infile)

    song_output_dir = os.path.join(output_dir, song_id)
    create_directories(song_output_dir)
    _tick(progress, 0.08, "Pasando a WAV 44.1 kHz…")
    orig_song_path = convert_to_stereo_and_wav(orig_song_path)

    logger.info(f"onnxruntime device >> {ort.get_device()}")

    if only_voiceless:
        logger.info("Voiceless Track Separation...")
        _tick(progress, 0.35, "Separando instrumental…")

        process = run_mdx(
            mdx_model_params,
            song_output_dir,
            ensure_uvr_model("UVR-MDX-NET-Inst_HQ_4.onnx", progress),
            orig_song_path,
            suffix="Voiceless",
            denoise=False,
            keep_orig=True,
            exclude_inversion=True,
            device_base=device_base,
        )

        _tick(progress, 1.0, "Listo")
        return process

    logger.info("Vocal Track Isolation...")
    _tick(progress, 0.25, "Separando voz…")
    vocals_path, instrumentals_path = run_mdx(
        mdx_model_params,
        song_output_dir,
        ensure_uvr_model("UVR-MDX-NET-Voc_FT.onnx", progress),
        orig_song_path,
        denoise=False,
        keep_orig=True,
        device_base=device_base,
    )

    if main_vocals:
        random_sleep()
        msg_main = "Separando la voz principal de los coros…"
        logger.info(msg_main)
        _tick(progress, 0.55, msg_main)
        gr.Info(msg_main)
        try:
            backup_vocals_path, main_vocals_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                ensure_uvr_model("UVR_MDXNET_KARA_2.onnx", progress),
                vocals_path,
                suffix="Backup",
                invert_suffix="Main",
                denoise=True,
                device_base=device_base,
            )
        except Exception as e:
            backup_vocals_path, main_vocals_path = run_mdx_beta(
                mdx_model_params,
                song_output_dir,
                ensure_uvr_model("UVR_MDXNET_KARA_2.onnx", progress),
                vocals_path,
                suffix="Backup",
                invert_suffix="Main",
                denoise=True,
                device_base=device_base,
            )
    else:
        backup_vocals_path, main_vocals_path = None, vocals_path

    if dereverb:
        random_sleep()
        msg_dereverb = "Quitando reverb de la voz…"
        logger.info(msg_dereverb)
        _tick(progress, 0.75, msg_dereverb)
        gr.Info(msg_dereverb)
        try:
            _, vocals_dereverb_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                ensure_uvr_model("Reverb_HQ_By_FoxJoy.onnx", progress),
                main_vocals_path,
                invert_suffix="DeReverb",
                exclude_main=True,
                denoise=True,
                device_base=device_base,
            )
        except Exception as e:
            _, vocals_dereverb_path = run_mdx_beta(
                mdx_model_params,
                song_output_dir,
                ensure_uvr_model("Reverb_HQ_By_FoxJoy.onnx", progress),
                main_vocals_path,
                invert_suffix="DeReverb",
                exclude_main=True,
                denoise=True,
                device_base=device_base,
            )
    else:
        vocals_dereverb_path = main_vocals_path

    _tick(progress, 0.95, "Guardando pistas…")
    return (
        vocals_path,
        instrumentals_path,
        backup_vocals_path,
        main_vocals_path,
        vocals_dereverb_path,
    )


def add_vocal_effects(input_file, output_file, reverb_room_size=0.6, vocal_reverb_dryness=0.8, reverb_damping=0.6, reverb_wet_level=0.35,
                      delay_seconds=0.4, delay_mix=0.25,
                      compressor_threshold_db=-25, compressor_ratio=3.5, compressor_attack_ms=10, compressor_release_ms=60,
                      gain_db=3):
    from pedalboard import Pedalboard, Reverb, Delay, Compressor, Gain, HighpassFilter
    from pedalboard.io import AudioFile

    effects = [HighpassFilter()]

    effects.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level, dry_level=vocal_reverb_dryness))

    effects.append(Compressor(threshold_db=compressor_threshold_db, ratio=compressor_ratio,
                              attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))

    if delay_seconds > 0 or delay_mix > 0:
        effects.append(Delay(delay_seconds=delay_seconds, mix=delay_mix))
        # print("delay applied")
    # effects.append(Chorus())

    if gain_db:
        effects.append(Gain(gain_db=gain_db))
        # print("added gain db")

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


def add_instrumental_effects(input_file, output_file, highpass_freq=100, lowpass_freq=12000,
                             reverb_room_size=0.5, reverb_damping=0.5, reverb_wet_level=0.25,
                             compressor_threshold_db=-20, compressor_ratio=2.5, compressor_attack_ms=15, compressor_release_ms=80,
                             gain_db=2):
    from pedalboard import Pedalboard, Reverb, Compressor, Gain, HighpassFilter, LowpassFilter
    from pedalboard.io import AudioFile

    effects = [
        HighpassFilter(cutoff_frequency_hz=highpass_freq),
        LowpassFilter(cutoff_frequency_hz=lowpass_freq),
    ]
    if reverb_room_size > 0 or reverb_damping > 0 or reverb_wet_level > 0:
        effects.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level))

    effects.append(Compressor(threshold_db=compressor_threshold_db, ratio=compressor_ratio,
                              attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))

    if gain_db:
        effects.append(Gain(gain_db=gain_db))

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


COMMON_SAMPLE_RATES = [8000, 16000, 22050, 32000, 44100, 48000, 96000]


def save_audio(audio_opt: np.ndarray, final_sr: int, output_audio_path: str, target_format: str) -> str:
    """
    Save audio with automatic handling of unsupported sample rates for non-WAV formats.
    """
    ext = os.path.splitext(output_audio_path)[1].lower()

    try:
        if ext == ".wav":
            sf.write(output_audio_path, audio_opt, final_sr, format=target_format)
        else:
            target_sr = min(COMMON_SAMPLE_RATES, key=lambda altsr: abs(altsr - final_sr))
            if target_sr != final_sr:
                logger.warning(f"Resampling from {final_sr} -> {target_sr} for {ext}")
                audio_opt = librosa.resample(audio_opt, orig_sr=final_sr, target_sr=target_sr)
            sf.write(output_audio_path, audio_opt, target_sr, format=target_format)
    except Exception as e:
        logger.error(e)
        logger.error(f"Error saving {output_audio_path}, performing fallback to WAV")
        output_audio_path = output_audio_path.replace(f"_converted.{target_format}", ".wav")

    return output_audio_path


def convert_format(file_paths, media_dir, target_format):
    """
    Convert a list of audio files to the target format with automatic safe sample rates.

    WAV files are returned as-is; non-WAV files are resampled if needed to a supported rate.
    """
    target_format = target_format.lower()
    if target_format == "wav":
        return file_paths  # No conversion needed for WAV

    suffix = "_converted"
    converted_files = []

    for fp in file_paths:
        # Absolute paths and base filename
        abs_fp = os.path.abspath(fp)
        file_name, _ = os.path.splitext(os.path.basename(abs_fp))
        file_ext = f".{target_format}"
        out_name = file_name + suffix + file_ext
        out_path = os.path.join(media_dir, out_name)

        # Load audio with librosa (handles many formats)
        audio, sr = sf.read(abs_fp)

        # Save using safe resampling
        saved_path = save_audio(audio, sr, out_path, target_format)
        converted_files.append(saved_path)

        # print(f"Converted: {abs_fp} -> {saved_path}")

    return converted_files


READY_STATUS = "Audio listo. Elegí qué extraer y pulsá Separar."
RUN_STATUS = RUN_SEPARATE
DONE_STATUS = "Listo. Las pistas están en Descargas/Audio Separator."


def _drop_separate_work(paths):
    """Delete Trabajos/Separar/<song> after the stems were copied out."""
    root = os.path.abspath(output_dir)
    song_dirs = set()
    for path in paths or []:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        prefix = root + os.sep
        if not abs_path.startswith(prefix):
            continue
        song = os.path.relpath(abs_path, root).split(os.sep)[0]
        if song and song != ".":
            song_dirs.add(os.path.join(root, song))
    for song_dir in song_dirs:
        shutil.rmtree(song_dir, ignore_errors=True)


def _separate_error_message(error):
    return fail("uvr", error, MSG_SEPARATE)

def unlock_run_button():
    return gr.update(interactive=True, value="Separar audio")


def lock_run_button():
    return gr.update(interactive=False, value="Separando…"), RUN_STATUS


def sound_separate(
    media_file, stem, main, dereverb, vocal_effects=True, background_effects=True,
    vocal_reverb_room_size=0.6, vocal_reverb_damping=0.6, vocal_reverb_dryness=0.8, vocal_reverb_wet_level=0.35,
    vocal_delay_seconds=0.4, vocal_delay_mix=0.25,
    vocal_compressor_threshold_db=-25, vocal_compressor_ratio=3.5, vocal_compressor_attack_ms=10, vocal_compressor_release_ms=60,
    vocal_gain_db=4,
    background_highpass_freq=120, background_lowpass_freq=11000,
    background_reverb_room_size=0.5, background_reverb_damping=0.5, background_reverb_wet_level=0.25,
    background_compressor_threshold_db=-20, background_compressor_ratio=2.5, background_compressor_attack_ms=15, background_compressor_release_ms=80,
    background_gain_db=3,
    target_format="WAV",
    progress=gr.Progress(track_tqdm=True),
):
    yield (
        None,
        None,
        None,
        status_update(KIND_RUN, RUN_STATUS),
        gr.update(interactive=False, value="Separando…"),
    )
    try:
        import occupancy

        occupancy.acquire(occupancy.HOLD_UVR)
        try:
            vocal, background, files, status, button = _sound_separate(
                media_file, stem, main, dereverb, vocal_effects, background_effects,
                vocal_reverb_room_size, vocal_reverb_damping, vocal_reverb_dryness, vocal_reverb_wet_level,
                vocal_delay_seconds, vocal_delay_mix,
                vocal_compressor_threshold_db, vocal_compressor_ratio, vocal_compressor_attack_ms, vocal_compressor_release_ms,
                vocal_gain_db,
                background_highpass_freq, background_lowpass_freq,
                background_reverb_room_size, background_reverb_damping, background_reverb_wet_level,
                background_compressor_threshold_db, background_compressor_ratio, background_compressor_attack_ms, background_compressor_release_ms,
                background_gain_db,
                target_format,
                progress=progress,
            )
        finally:
            occupancy.release(occupancy.HOLD_UVR)
        yield vocal, background, files, status_update(KIND_OK, status), button
    except Exception as error:
        message = _separate_error_message(error)
        yield (
            None,
            None,
            None,
            status_update(KIND_ERROR, message),
            unlock_run_button(),
        )


def _sound_separate(
    media_file, stem, main, dereverb, vocal_effects, background_effects,
    vocal_reverb_room_size, vocal_reverb_damping, vocal_reverb_dryness, vocal_reverb_wet_level,
    vocal_delay_seconds, vocal_delay_mix,
    vocal_compressor_threshold_db, vocal_compressor_ratio, vocal_compressor_attack_ms, vocal_compressor_release_ms,
    vocal_gain_db,
    background_highpass_freq, background_lowpass_freq,
    background_reverb_room_size, background_reverb_damping, background_reverb_wet_level,
    background_compressor_threshold_db, background_compressor_ratio, background_compressor_attack_ms, background_compressor_release_ms,
    background_gain_db,
    target_format,
    progress=None,
):
    media_file = _audio_path(media_file)
    if not media_file or not os.path.isfile(media_file):
        raise ValueError("Falta el archivo de audio.")

    stem = stem_choice_to_list(stem)
    if not stem:
        raise ValueError("Elige voz, instrumental, o ambos.")

    hash_audio = str(get_hash(media_file))
    media_dir = os.path.dirname(media_file)

    outputs = []
    instrumentals_from_vocal = None

    try:
        duration_base_ = librosa.get_duration(filename=media_file)
        print("Duration audio:", duration_base_)
    except Exception as e:
        print(e)

    start_time = time.time()

    if "vocal" in stem:
        try:
            _, instrumentals_from_vocal, _, _, vocal_audio = process_uvr_task(
                orig_song_path=media_file,
                song_id=hash_audio + "mdx",
                main_vocals=main,
                dereverb=dereverb,
                remove_files_output_dir=False,
                progress=progress,
            )

            if vocal_effects:
                suffix = '_effects'
                file_name, file_extension = os.path.splitext(os.path.abspath(vocal_audio))
                out_effects = file_name + suffix + file_extension
                out_effects_path = os.path.join(media_dir, out_effects)
                add_vocal_effects(vocal_audio, out_effects_path,
                                  reverb_room_size=vocal_reverb_room_size, reverb_damping=vocal_reverb_damping, vocal_reverb_dryness=vocal_reverb_dryness, reverb_wet_level=vocal_reverb_wet_level,
                                  delay_seconds=vocal_delay_seconds, delay_mix=vocal_delay_mix,
                                  compressor_threshold_db=vocal_compressor_threshold_db, compressor_ratio=vocal_compressor_ratio, compressor_attack_ms=vocal_compressor_attack_ms, compressor_release_ms=vocal_compressor_release_ms,
                                  gain_db=vocal_gain_db
                                  )
                vocal_audio = out_effects_path

            outputs.append(vocal_audio)
        except Exception:
            logger.exception("process_uvr_task vocal failed")
            raise

    if "background" in stem:
        if instrumentals_from_vocal and os.path.isfile(instrumentals_from_vocal):
            background_audio = instrumentals_from_vocal
        else:
            background_audio, _ = process_uvr_task(
                orig_song_path=media_file,
                song_id=hash_audio + "voiceless",
                only_voiceless=True,
                remove_files_output_dir=False,
                progress=progress,
            )

        if background_effects:
            suffix = '_effects'
            file_name, file_extension = os.path.splitext(os.path.abspath(background_audio))
            out_effects = file_name + suffix + file_extension
            out_effects_path = os.path.join(media_dir, out_effects)
            # print(file_name, file_extension, out_effects, out_effects_path)
            add_instrumental_effects(background_audio, out_effects_path,
                                     highpass_freq=background_highpass_freq, lowpass_freq=background_lowpass_freq,
                                     reverb_room_size=background_reverb_room_size, reverb_damping=background_reverb_damping, reverb_wet_level=background_reverb_wet_level,
                                     compressor_threshold_db=background_compressor_threshold_db, compressor_ratio=background_compressor_ratio, compressor_attack_ms=background_compressor_attack_ms, compressor_release_ms=background_compressor_release_ms,
                                     gain_db=background_gain_db
                                     )
            background_audio = out_effects_path

        outputs.append(background_audio)

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")

    if not outputs:
        raise ValueError("No se pudo separar el audio.")

    files = convert_format(outputs, media_dir, target_format)
    want_vocal = "vocal" in stem
    want_bg = "background" in stem
    vocal_out = files[0] if want_vocal and files else None
    background_out = None
    if want_bg:
        if want_vocal:
            background_out = files[1] if len(files) > 1 else None
        else:
            background_out = files[0] if files else None
    labels = []
    export_paths = []
    if vocal_out:
        labels.append("voz")
        export_paths.append(vocal_out)
    if background_out:
        labels.append("instrumental")
        export_paths.append(background_out)
    export_dir, copied = copy_to_downloads(export_paths, labels)
    if copied:
        if want_vocal:
            vocal_out = copied[0]
        if want_bg:
            background_out = copied[-1] if want_vocal and len(copied) > 1 else copied[0]
        _drop_separate_work(export_paths)
    if vocal_out and os.path.isfile(vocal_out):
        try:
            import library

            library.set_session_meta(last_vocal_path=os.path.abspath(vocal_out))
        except Exception:
            logger.exception("no guardé last_vocal_path")
    status = (
        f"Listo. Las pistas están en {export_dir}"
        if copied
        else DONE_STATUS
    )
    return vocal_out, background_out, copied or files, status, unlock_run_button()
