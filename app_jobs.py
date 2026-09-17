import os
import gradio as gr
from utils import logger
from exports import copy_to_downloads, open_exports_dir
from uvr_runtime import (
    IS_ZERO_GPU,
    unlock_run_button,
    lock_run_button,
    sound_separate,
)
from ui_widgets import FORMAT_OPTIONS

IDLE_STATUS = "1 Canción → 2 Extraer → 3 Resultado → 4 Voz (RVC) → 5 Unir."
DEMO_SONG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.mp3")
DEMO_STATUS = "Demo cargada (test.mp3). Pulsá Separar. Podés cambiar el archivo cuando quieras."


def demo_song_path():
    return DEMO_SONG if os.path.isfile(DEMO_SONG) else None


def ensure_demo_voice():
    """Register bundled smoke_voice.pth if present (does not download)."""
    import library

    library.ensure_dirs()
    folder = library.PATHS["rvc_voices"]
    pth = os.path.join(folder, "smoke_voice.pth")
    if not os.path.isfile(pth):
        return None
    library.register("rvc_voices", pth, "smoke_voice.pth")
    index = os.path.join(folder, "smoke_voice.index")
    if os.path.isfile(index):
        library.register("rvc_voices", index, "smoke_voice.index")
    return pth


def load_demo_bundle():
    """Preload demo song + refresh RVC lists. Never disables other inputs."""
    ensure_demo_voice()
    rvc_upd, voice_upd, tts_upd = refresh_library_ui()
    song = demo_song_path()
    run = unlock_run_button() if song else gr.update()
    status = DEMO_STATUS if song else IDLE_STATUS
    return song, run, rvc_upd, voice_upd, tts_upd, status


def _install_status_line():
    try:
        from install_rvc_assets import missing_rvc_assets, rvc_assets_ready

        if rvc_assets_ready():
            return IDLE_STATUS
        missing = missing_rvc_assets()
        return (
            f"Faltan pesos RVC ({len(missing)}). "
            "Pulsá «Completar instalación» (una vez, ~700 MB públicos)."
        )
    except Exception:
        return IDLE_STATUS


def install_rvc_job():
    try:
        from install_rvc_assets import install_rvc_assets, missing_rvc_assets

        lines = []

        def _log(msg):
            lines.append(msg)

        written = install_rvc_assets(log=_log)
        try:
            from rvc_engine import ensure_vc_engine

            _log("Preparando motor de conversión…")
            ensure_vc_engine()
            written.append("motor")
        except Exception as exc:
            _log(f"Motor: {exc}")
            raise
        left = missing_rvc_assets()
        if left:
            return (
                "Instalación incompleta: " + ", ".join(left),
                "\n".join(lines[-12:]),
            )
        song, run, rvc_upd, voice_upd, tts_upd, demo_status = load_demo_bundle()
        note = (
            f"Listo ({len(written)} archivos). {demo_status}"
            if song
            else f"Listo ({len(written)} archivos). Ya podés Entrenar / Convertir."
        )
        log = "\n".join(lines[-12:]) or note
        return note, log, song, run, rvc_upd, voice_upd, tts_upd
    except Exception as error:
        logger.error(str(error))
        gr.Warning(str(error))
        return (
            f"Falló la descarga: {error}",
            str(error),
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
READY_STATUS = "Audio listo. Elegí qué extraer y pulsá Separar."



def lock_download_button():
    return gr.update(interactive=False, value="Descargando…"), "Descargando audio…"


def unlock_download_button():
    return gr.update(interactive=True, value="Descargar")


def audio_downloader(url_media, with_video=True):
    unlock = unlock_download_button()
    empty_video = None
    if IS_ZERO_GPU and url_media and "youtube.com" in url_media:
        gr.Info("Esta opción no está disponible en Hugging Face.")
        return None, empty_video, gr.update(), "YouTube no está disponible aquí.", unlock
    from youtube_lib import download_media

    try:
        path, video_path, reused, note = download_media(
            url_media, with_video=bool(with_video)
        )
    except ValueError as error:
        gr.Warning(str(error))
        return None, empty_video, gr.update(), str(error), unlock
    if reused:
        status = "Audio (WAV 48 kHz) ya estaba. Listo para separar."
    else:
        status = "Audio WAV 48 kHz listo. Extraé y Separá."
    if video_path:
        status += " Video MP4 también listo para remux."
    elif with_video:
        status += " (sin video)"
    if note:
        status = f"{note} {status}"
    try:
        import library

        library.set_session_meta(
            last_audio_path=path,
            last_video_path=video_path,
            last_youtube_url=url_media,
        )
    except Exception:
        pass
    return path, video_path, unlock_run_button(), status, unlock


def clip_for_clone(source_path, start, end):
    from youtube_lib import clip_audio
    from exports import copy_to_downloads
    import library

    try:
        path = clip_audio(source_path, start, end)
        library.register("voices", path)
        _, copied = copy_to_downloads([path], ["ref_clon"])
        saved = copied[0] if copied else path
        return saved, f"Recorte listo ({start}–{end})."
    except ValueError as error:
        gr.Warning(str(error))
        return None, str(error)


def on_audio_ready(path):
    if path:
        return unlock_run_button(), READY_STATUS
    return gr.update(interactive=False, value="Separar audio"), IDLE_STATUS


def reset_job():
    return (
        None,
        None,
        None,
        None,
        IDLE_STATUS,
        gr.update(interactive=False, value="Separar audio"),
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "",
        None,
    )


def refresh_library():
    import library

    library.ensure_dirs()
    rvc = library.dropdown_choices(library.list_rvc_voices())
    voices = library.dropdown_choices(library.list_voices())
    rvc_upd = gr.update(choices=rvc, value=(rvc[0][1] if rvc else None))
    voice_upd = gr.update(choices=voices, value=(voices[0][1] if voices else None))
    return rvc_upd, voice_upd


def refresh_library_ui():
    """Same as refresh_library plus the Text→RVC model dropdown."""
    rvc_upd, voice_upd = refresh_library()
    return rvc_upd, voice_upd, rvc_upd


def import_voice_into_library(voice_file):
    import library

    path = _gradio_path(voice_file)
    if not path:
        return gr.update(), None, "Elegí un audio de referencia."
    item = library.register("voices", path)
    voices = library.dropdown_choices(library.list_voices())
    return (
        gr.update(choices=voices, value=item["path"]),
        item["path"],
        f"Voz guardada en biblioteca: {item['name']}.",
    )


def _gradio_path(file_obj):
    """Normalize Gradio File/Audio values to a local filesystem path."""
    if not file_obj:
        return None
    if isinstance(file_obj, str):
        return file_obj if os.path.exists(file_obj) else file_obj
    if isinstance(file_obj, dict):
        for key in ("path", "name", "orig_name"):
            value = file_obj.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if isinstance(file_obj, (list, tuple)) and file_obj:
        return _gradio_path(file_obj[0])
    return getattr(file_obj, "name", None) or getattr(file_obj, "path", None)


def load_rvc_into_library(
    hubert_file, rmvpe_file, model_file, index_file, g_file=None, d_file=None
):
    import library
    import rvc_engine

    loaded = []
    model_path = None
    hubert = _gradio_path(hubert_file)
    if hubert:
        library.register("rvc", hubert, "hubert_base.pt")
        loaded.append("hubert")
    rmvpe = _gradio_path(rmvpe_file)
    if rmvpe:
        library.register("rvc", rmvpe, "rmvpe.pt")
        loaded.append("rmvpe")
    g_path = _gradio_path(g_file)
    if g_path:
        library.register("rvc", g_path, "f0G40k.pth")
        loaded.append("f0G40k")
    d_path = _gradio_path(d_file)
    if d_path:
        library.register("rvc", d_path, "f0D40k.pth")
        loaded.append("f0D40k")
    model = _gradio_path(model_file)
    if model:
        item = library.register("rvc_voices", model)
        model_path = item["path"]
        loaded.append(".pth")
    index = _gradio_path(index_file)
    if index and model_path:
        index_name = os.path.splitext(os.path.basename(model_path))[0] + ".index"
        library.register("rvc_voices", index, index_name)
        loaded.append(".index")
    elif index and not model_path:
        library.register("rvc_voices", index)
        loaded.append(".index")
    rvc_engine._converter = None
    rvc_upd, voice_upd, tts_upd = refresh_library_ui()
    if not loaded:
        return rvc_upd, voice_upd, tts_upd, "Elegí archivos y pulsá Cargar."
    hubert_ok = "sí" if rvc_engine.local_hubert_path() else "no"
    rmvpe_ok = "sí" if rvc_engine.local_rmvpe_path() else "no"
    return (
        rvc_upd,
        voice_upd,
        tts_upd,
        f"Cargado: {', '.join(loaded)}. Soporte → hubert: {hubert_ok}, rmvpe: {rmvpe_ok}.",
    )


def train_rvc_job(exp_name, dataset_files):
    try:
        from rvc_train import train_voice

        files = dataset_files or []
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        pth, index = train_voice(exp_name, files)
        import library

        rvc = library.dropdown_choices(library.list_rvc_voices())
        _, voice_upd = refresh_library()
        note = f"Modelo listo: {os.path.basename(pth)}"
        if index:
            note += " (+index)"
        note += ". Ya podés Convertir / Texto→RVC."
        rvc_upd = gr.update(choices=rvc, value=pth)
        return rvc_upd, voice_upd, note, rvc_upd
    except ValueError as error:
        gr.Warning(str(error))
        rvc_upd, voice_upd, tts_upd = refresh_library_ui()
        return rvc_upd, voice_upd, str(error), tts_upd
    except Exception as error:
        logger.error(str(error))
        gr.Warning("Falló el entrenamiento.")
        rvc_upd, voice_upd, tts_upd = refresh_library_ui()
        return rvc_upd, voice_upd, f"Falló el entrenamiento: {error}", tts_upd


def rvc_job(audio_path, library_model, model_file, index_file):
    import library
    from rvc_engine import convert_voice

    audio_path = _gradio_path(audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        msg = (
            "Falta la pista de voz. Primero Separá (paso 2–3) "
            "o usá el paso 6 Texto → habla."
        )
        gr.Warning(msg)
        return None, None, msg

    voices_root = os.path.abspath(library.PATHS["rvc_voices"])
    model_path = library_model
    if model_file:
        uploaded = _gradio_path(model_file)
        if uploaded:
            item = library.register("rvc_voices", uploaded)
            model_path = item["path"]
    # Convert only uses library/models/rvc_voices/ (never assets/weights directly).
    if model_path and os.path.abspath(model_path).startswith(voices_root + os.sep):
        pass
    elif model_path and os.path.isfile(model_path):
        item = library.register("rvc_voices", model_path)
        model_path = item["path"]
    if not model_path:
        msg = "Elegí un modelo RVC en la biblioteca (paso 4)."
        gr.Warning(msg)
        return None, None, msg
    index_path = library.find_index_for_model(model_path) if model_path else None
    if index_file:
        uploaded_index = _gradio_path(index_file)
        if uploaded_index and model_path:
            index_name = os.path.splitext(os.path.basename(model_path))[0] + ".index"
            item = library.register("rvc_voices", uploaded_index, index_name)
            index_path = item["path"]
    try:
        out_path = convert_voice(audio_path, model_path, index_path=index_path)
        note = " (+index)" if index_path else ""
        from vc_runner import infer_log_path, last_infer_tail

        tail = last_infer_tail(6)
        extra = f" Log: {infer_log_path()}"
        if tail:
            extra += " | " + tail[-1]
        return (
            out_path,
            out_path,
            f"Voz convertida{note}. Está lista para unir.{extra}",
        )
    except ValueError as error:
        gr.Warning(str(error))
        return None, None, str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudo convertir la voz.")
        return None, None, "No se pudo convertir la voz."


def clone_job(text, speaker_wav):
    """Legacy XTTS path (pesos opcionales). Preferí tts_rvc_job."""
    try:
        from clone_engine import clone_voice, missing_xtts_files

        missing = missing_xtts_files()
        if missing:
            msg = (
                "XTTS no está instalado. Usá «Texto → habla (Edge + RVC)» "
                "con un modelo entrenado del paso 4."
            )
            gr.Warning(msg)
            return None, msg
        out_path = clone_voice(text, speaker_wav)
        return out_path, "Voz clonada (XTTS). Archivo en Descargas."
    except ValueError as error:
        gr.Warning(str(error))
        return None, str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudo clonar la voz.")
        return None, "No se pudo clonar la voz."


def tts_rvc_job(text, rvc_model, edge_voice, pitch):
    try:
        from tts_rvc_engine import resolve_edge_voice, speak_with_rvc

        voice_id = resolve_edge_voice(edge_voice)
        out = speak_with_rvc(
            text,
            rvc_model,
            edge_voice=voice_id,
            pitch=int(pitch or 0),
        )
        return out, out, f"Listo (Edge {voice_id} → RVC). {out}"
    except ValueError as error:
        gr.Warning(str(error))
        return None, None, str(error)
    except Exception as error:
        logger.error(str(error))
        msg = f"Falló texto→RVC: {error}"
        gr.Warning(msg)
        return None, None, msg


def detect_voices_job(audio_path):
    try:
        from diarize import detect_speakers

        import library

        paths = detect_speakers(audio_path)
        for path in paths:
            library.register("voices", path)
        labels = [f"Voz {i}" for i in range(1, len(paths) + 1)]
        voice_choices = library.dropdown_choices(library.list_voices())
        return (
            gr.update(choices=list(zip(labels, paths)), value=paths[0], visible=True),
            paths[0],
            gr.update(choices=voice_choices, value=paths[0]),
            f"Encontré {len(paths)} voz/voces. Guardadas en la biblioteca local.",
        )
    except ValueError as error:
        gr.Warning(str(error))
        return gr.update(visible=False), None, gr.update(), str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudieron detectar las voces.")
        return gr.update(visible=False), None, gr.update(), "No se pudieron detectar las voces."


def remix_job(
    voice_path,
    instrumental_path,
    delay_milliseconds,
    match_duration,
    voice_db,
    instrumental_db,
    target_format,
):
    try:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remix_output")
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, "remix.wav")
        from remix import remix_to_wav

        remix_to_wav(
            voice_path,
            instrumental_path,
            wav_path,
            delay_milliseconds=delay_milliseconds or 0,
            match_duration=bool(match_duration),
            voice_db=voice_db or 0,
            instrumental_db=instrumental_db or 0,
        )
        files = convert_format([wav_path], out_dir, target_format or "WAV")
        final = files[0]
        export_dir, copied = copy_to_downloads([final], ["remix"])
        saved = copied[0] if copied else final
        return saved, saved, f"Pistas unidas (WAV). Archivo en {export_dir}"
    except ValueError as error:
        gr.Warning(str(error))
        return None, None, str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudo armar el remix.")
        return None, None, "No se pudo armar el remix."


def remux_job(video_path, audio_path):
    try:
        video_path = _gradio_path(video_path)
        audio_path = _gradio_path(audio_path)
        from video_remux import remux_audio_onto_video

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remix_output")
        os.makedirs(out_dir, exist_ok=True)
        raw = os.path.join(out_dir, "remux.mp4")
        remux_audio_onto_video(video_path, audio_path, raw, shortest=True)
        _, copied = copy_to_downloads([raw], ["video_nuevo_audio"])
        saved = copied[0] if copied else raw
        return saved, f"Video + audio nuevo (AAC 320k, video copy). {saved}"
    except ValueError as error:
        gr.Warning(str(error))
        return None, str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudo pegar el audio al video.")
        return None, "No se pudo pegar el audio al video."


def cover_job(title, artist, audio_path, artistic):
    try:
        from album_cover import generate_cover, save_cover_with_audio

        audio_path = _gradio_path(audio_path)
        cover = generate_cover(title or "Audio Separator", artist or "", artistic=bool(artistic))
        saved = save_cover_with_audio(cover, audio_path)
        return saved, f"Portada lista (no toca el audio): {saved}"
    except ValueError as error:
        gr.Warning(str(error))
        return None, str(error)
    except Exception as error:
        logger.error(str(error))
        gr.Warning("No se pudo generar la portada.")
        return None, str(error)

