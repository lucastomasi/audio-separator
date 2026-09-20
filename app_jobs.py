import os
import gradio as gr
from utils import logger
from exports import copy_to_downloads, open_exports_dir
from uvr_runtime import (
    IS_ZERO_GPU,
    unlock_run_button,
    lock_run_button,
    sound_separate,
    convert_format,
)
from ui_widgets import FORMAT_OPTIONS
from ui_status import (
    IDLE as IDLE_STATUS,
    KIND_ERROR,
    KIND_OK,
    KIND_RUN,
    MSG_CONVERT,
    MSG_COVER,
    MSG_INSTALL,
    MSG_REMIX,
    MSG_REMUX,
    MSG_TRAIN,
    MSG_TTS,
    READY as READY_STATUS,
    fail,
    status_update,
)

DEMO_SONG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.mp3")
DEMO_STATUS = "Demo cargada (test.mp3). Pulsá Separar. Podés cambiar el archivo cuando quieras."


def _ok(text):
    return status_update(KIND_OK, text)


def _err(where, exc, default):
    return status_update(KIND_ERROR, fail(where, exc, default))


def demo_song_path():
    return DEMO_SONG if os.path.isfile(DEMO_SONG) else None


def ensure_demo_voice():
    """Register bundled smoke_voice.pth into the live library dir."""
    import library
    from app_env import package_dir

    library.ensure_dirs()
    folder = library.PATHS["rvc_voices"]
    pth = os.path.join(folder, "smoke_voice.pth")
    bundled = os.path.join(
        package_dir(), "library", "models", "rvc_voices", "smoke_voice.pth"
    )
    src = pth if os.path.isfile(pth) else bundled
    if not os.path.isfile(src):
        return None
    item = library.register("rvc_voices", src, "smoke_voice.pth")
    index_src = os.path.splitext(src)[0] + ".index"
    if os.path.isfile(index_src):
        library.register("rvc_voices", index_src, "smoke_voice.index")
    return item["path"]


def load_demo_bundle():
    """Preload demo song + refresh RVC lists. Never disables other inputs."""
    ensure_demo_voice()
    rvc_upd, tts_upd = refresh_library_ui()
    song = demo_song_path()
    run = unlock_run_button() if song else gr.update()
    status = _ok(DEMO_STATUS if song else IDLE_STATUS)
    return song, run, rvc_upd, tts_upd, status


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


def install_rvc_job(progress=gr.Progress()):
    try:
        from install_rvc_assets import install_rvc_assets, missing_rvc_assets

        lines = []

        def _log(msg):
            lines.append(msg)
            try:
                progress((min(len(lines), 8)) / 10, desc=str(msg)[:80])
            except Exception:
                pass

        written = install_rvc_assets(log=_log)
        try:
            from rvc_engine import ensure_vc_engine

            _log("Preparando motor de conversión…")
            ensure_vc_engine()
            written.append("motor")
        except Exception as exc:
            _log("Motor: falló")
            raise
        left = missing_rvc_assets()
        if left:
            return (
                _ok("Instalación incompleta. Pulsá Completar instalación de nuevo."),
                "\n".join(lines[-12:]),
                None,
                gr.update(),
                gr.update(),
                gr.update(),
            )
        song, run, rvc_upd, tts_upd, demo_status = load_demo_bundle()
        note = (
            f"Listo ({len(written)} archivos). Ya podés Entrenar / Convertir."
        )
        log = "\n".join(lines[-12:]) or note
        return _ok(note), log, song, run, rvc_upd, tts_upd
    except Exception as error:
        return (
            _err("install_rvc_job", error, MSG_INSTALL),
            MSG_INSTALL,
            None,
            gr.update(),
            gr.update(),
            gr.update(),
        )



def lock_download_button():
    return (
        gr.update(interactive=False, value="Descargando…"),
        status_update(KIND_RUN, "Descargando audio…"),
    )


def unlock_download_button():
    return gr.update(interactive=True, value="Descargar")


def audio_downloader(url_media, with_video=True, progress=gr.Progress()):
    unlock = unlock_download_button()
    try:
        progress(0.1, desc="Descargando de YouTube…")
    except Exception:
        pass
    empty_video = None
    if IS_ZERO_GPU and url_media and "youtube.com" in url_media:
        gr.Info("Esta opción no está disponible en Hugging Face.")
        return (
            None,
            empty_video,
            gr.update(),
            _ok("YouTube no está disponible aquí."),
            unlock,
        )
    from youtube_lib import download_media

    try:
        path, video_path, reused, note = download_media(
            url_media, with_video=bool(with_video)
        )
        try:
            progress(0.85, desc="Preparando WAV 48 kHz…")
        except Exception:
            pass
    except ValueError as error:
        return None, empty_video, gr.update(), _err("audio_downloader", error, "No pude descargar."), unlock
    except Exception as error:
        return None, empty_video, gr.update(), _err("audio_downloader", error, "No pude descargar."), unlock
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
    return path, video_path, unlock_run_button(), _ok(status), unlock


def clip_for_clone(source_path, start, end):
    from youtube_lib import clip_audio
    from exports import copy_to_downloads
    import library

    try:
        path = clip_audio(source_path, start, end)
        library.register("voices", path)
        _, copied = copy_to_downloads([path], ["ref_clon"])
        saved = copied[0] if copied else path
        return saved, _ok(f"Recorte listo ({start}–{end}).")
    except ValueError as error:
        return None, _err("clip_for_clone", error, "No pude recortar.")
    except Exception as error:
        return None, _err("clip_for_clone", error, "No pude recortar.")


def on_audio_ready(path):
    if path:
        return unlock_run_button(), _ok(READY_STATUS)
    return gr.update(interactive=False, value="Separar audio"), _ok(IDLE_STATUS)


def reset_job():
    return (
        None,
        None,
        None,
        None,
        _ok(IDLE_STATUS),
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
    )


def refresh_library():
    import library

    library.ensure_dirs()
    rvc = library.dropdown_choices(library.list_rvc_voices())
    return gr.update(choices=rvc, value=(rvc[0][1] if rvc else None))


def refresh_library_ui():
    rvc_upd = refresh_library()
    return rvc_upd, rvc_upd


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
        if os.path.isdir(hubert) and rvc_engine._is_transformers_hubert_dir(hubert):
            loaded.append("hubert (carpeta Transformers; no se pisa)")
        elif str(hubert).lower().endswith((".pt", ".pth")):
            loaded.append(
                "hubert ignorado (hace falta carpeta hubert_base/ "
                "con config.json + model.safetensors, no un .pth de voz)"
            )
        else:
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
    rvc_upd, tts_upd = refresh_library_ui()
    if not loaded:
        return rvc_upd, tts_upd, _ok("Elegí archivos y pulsá Cargar.")
    hubert_ok = "sí" if rvc_engine.local_hubert_path() else "no"
    rmvpe_ok = "sí" if rvc_engine.local_rmvpe_path() else "no"
    return (
        rvc_upd,
        tts_upd,
        _ok(
            f"Cargado: {', '.join(loaded)}. Soporte → hubert: {hubert_ok}, rmvpe: {rmvpe_ok}."
        ),
    )


def train_rvc_job(
    exp_name,
    dataset_files,
    epochs=10,
    hire_gpu=False,
    runpod_key=None,
    progress=gr.Progress(),
):
    try:
        from rvc_train import train_voice

        try:
            progress(0.05, desc="Entrenando (CPU, puede tardar)…")
        except Exception:
            pass

        files = dataset_files or []
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        elif not isinstance(files, (list, tuple)):
            files = [files]
        files = [_gradio_path(item) or item for item in files]
        n_files = len([item for item in files if item])
        from runpod_train import estimate_copy

        try:
            progress(0.05, desc=estimate_copy(n_files, int(epochs or 10)))
        except Exception:
            pass
        if hire_gpu and runpod_key:
            from gpu_secrets import save as save_secrets

            save_secrets({"RUNPOD_API_KEY": runpod_key})
        pth, index = train_voice(
            exp_name, files, epochs=epochs, progress=progress
        )
        import library

        rvc_upd, tts_upd = refresh_library_ui()
        rvc_upd = gr.update(
            choices=library.dropdown_choices(library.list_rvc_voices()),
            value=pth,
        )
        note = f"Modelo listo: {os.path.basename(pth)}"
        if index:
            note += " (+index)"
        note += ". Ya podés Convertir / Texto→RVC."
        bar = _ok(note)
        return rvc_upd, bar, tts_upd, bar
    except ValueError as error:
        rvc_upd, tts_upd = refresh_library_ui()
        text = str(error)
        if "sigue entrenando" in text or "Convertir está bloqueado" in text:
            bar = status_update(KIND_RUN, text)
        else:
            bar = _err("train_rvc_job", error, MSG_TRAIN)
        return rvc_upd, bar, tts_upd, bar
    except Exception as error:
        rvc_upd, tts_upd = refresh_library_ui()
        bar = _err("train_rvc_job", error, MSG_TRAIN)
        return rvc_upd, bar, tts_upd, bar


def rvc_job(audio_path, library_model, model_file, index_file, progress=gr.Progress()):
    import library
    from rvc_engine import convert_voice

    try:
        progress(0.08, desc="Preparando motor de conversión…")
    except Exception:
        pass

    audio_path = _gradio_path(audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        msg = (
            "Falta la pista de voz. Primero Separá (paso 2–3) "
            "o usá el paso 6 Texto → habla."
        )
        return None, None, _ok(msg)

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
        return None, None, _ok(msg)
    index_path = library.find_index_for_model(model_path) if model_path else None
    if index_file:
        uploaded_index = _gradio_path(index_file)
        if uploaded_index and model_path:
            index_name = os.path.splitext(os.path.basename(model_path))[0] + ".index"
            item = library.register("rvc_voices", uploaded_index, index_name)
            index_path = item["path"]
    try:
        try:
            progress(0.35, desc="Convirtiendo voz…")
        except Exception:
            pass
        out_path = convert_voice(audio_path, model_path, index_path=index_path)
        try:
            progress(1.0, desc="Listo")
        except Exception:
            pass
        note = " (+index)" if index_path else ""
        return (
            out_path,
            out_path,
            _ok(f"Voz convertida{note}. Está lista para unir."),
        )
    except ValueError as error:
        return None, None, _err("rvc_job", error, MSG_CONVERT)
    except Exception as error:
        return None, None, _err("rvc_job", error, MSG_CONVERT)


def tts_rvc_job(text, rvc_model, edge_voice, pitch, progress=gr.Progress()):
    try:
        from tts_rvc_engine import resolve_edge_voice, speak_with_rvc

        try:
            progress(0.15, desc="Edge TTS…")
        except Exception:
            pass

        voice_id = resolve_edge_voice(edge_voice)
        try:
            progress(0.45, desc="Convirtiendo con RVC…")
        except Exception:
            pass
        out, src = speak_with_rvc(
            text,
            rvc_model,
            edge_voice=voice_id,
            pitch=int(pitch or 0),
        )
        try:
            progress(1.0, desc="Listo")
        except Exception:
            pass
        return out, out, _ok(f"Listo ({src} → RVC).")
    except ValueError as error:
        return None, None, _err("tts_rvc_job", error, MSG_TTS)
    except Exception as error:
        return None, None, _err("tts_rvc_job", error, MSG_TTS)


def remix_job(
    voice_path,
    instrumental_path,
    delay_milliseconds,
    match_duration,
    voice_db,
    instrumental_db,
    target_format,
    progress=gr.Progress(),
):
    try:
        try:
            progress(0.3, desc="Uniendo pistas…")
        except Exception:
            pass
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
        return saved, saved, _ok(f"Pistas unidas (WAV). Archivo en {export_dir}")
    except ValueError as error:
        return None, None, _err("remix_job", error, MSG_REMIX)
    except Exception as error:
        return None, None, _err("remix_job", error, MSG_REMIX)


def remux_job(video_path, audio_path, progress=gr.Progress()):
    try:
        video_path = _gradio_path(video_path)
        audio_path = _gradio_path(audio_path)
        from video_remux import remux_audio_onto_video

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remix_output")
        os.makedirs(out_dir, exist_ok=True)
        raw = os.path.join(out_dir, "remux.mp4")
        try:
            progress(0.4, desc="Pegando audio al video (copy)…")
        except Exception:
            pass
        remux_audio_onto_video(video_path, audio_path, raw, shortest=True)
        _, copied = copy_to_downloads([raw], ["video_nuevo_audio"])
        saved = copied[0] if copied else raw
        return saved, _ok("Video + audio nuevo (AAC 320k, video copy).")
    except ValueError as error:
        return None, _err("remux_job", error, MSG_REMUX)
    except Exception as error:
        return None, _err("remux_job", error, MSG_REMUX)


def cover_job(title, artist, audio_path, artistic):
    try:
        from album_cover import generate_cover, save_cover_with_audio

        audio_path = _gradio_path(audio_path)
        cover = None
        gemini_note = ""
        if artistic:
            try:
                from gemini_cover import available as gemini_on, generate_png
                from exports import unique_path, exports_dir

                if gemini_on():
                    dest = unique_path(exports_dir(), "portada_gemini.png")
                    cover = str(generate_png(title or "", artist or "", dest))
            except Exception:
                cover = None
                gemini_note = " Gemini no anduvo; portada local."
        if not cover:
            cover = generate_cover(title or "Audio Separator", artist or "", artistic=bool(artistic))
        saved = save_cover_with_audio(cover, audio_path)
        used = "Gemini" if cover and "portada_gemini" in cover else "local"
        return saved, _ok(f"Portada lista ({used}).{gemini_note}")
    except ValueError as error:
        return None, _err("cover_job", error, MSG_COVER)
    except Exception as error:
        return None, _err("cover_job", error, MSG_COVER)

