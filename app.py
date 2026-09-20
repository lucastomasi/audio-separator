import os
import gradio as gr

from uvr_runtime import (
    IS_COLAB,
    IS_ZERO_GPU,
    sound_separate,
    unlock_run_button,
)
from exports import open_exports_dir
from ui_widgets import (
    url_media_conf,
    url_button_conf,
    audio_conf,
    out_audio,
    out_file,
    stem_conf,
    main_conf,
    dereverb_conf,
    vocal_effects_conf,
    background_effects_conf,
    vocal_reverb_room_size_conf,
    vocal_reverb_damping_conf,
    vocal_reverb_wet_level_conf,
    vocal_reverb_dryness_level_conf,
    vocal_delay_seconds_conf,
    vocal_delay_mix_conf,
    vocal_compressor_threshold_db_conf,
    vocal_compressor_ratio_conf,
    vocal_compressor_attack_ms_conf,
    vocal_compressor_release_ms_conf,
    vocal_gain_db_conf,
    background_highpass_freq_conf,
    background_lowpass_freq_conf,
    background_reverb_room_size_conf,
    background_reverb_damping_conf,
    background_reverb_wet_level_conf,
    background_compressor_threshold_db_conf,
    background_compressor_ratio_conf,
    background_compressor_attack_ms_conf,
    background_compressor_release_ms_conf,
    background_gain_db_conf,
    button_conf,
    output_conf,
    show_vocal_components,
    format_conf,
)
from app_jobs import (
    IDLE_STATUS,
    DEMO_SONG,
    DEMO_STATUS,
    READY_STATUS,
    demo_song_path,
    ensure_demo_voice,
    load_demo_bundle,
    install_rvc_job,
    audio_downloader,
    clip_for_clone,
    on_audio_ready,
    reset_job,
    refresh_library_ui,
    load_rvc_into_library,
    train_rvc_job,
    rvc_job,
    tts_rvc_job,
    remix_job,
    remux_job,
    cover_job,
    lock_download_button,
    _install_status_line,
)

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    radius_size=gr.themes.sizes.radius_md,
).set(
    button_primary_background_fill="#007AFF",
    button_primary_background_fill_hover="#0066d6",
    button_primary_text_color="white",
    button_secondary_background_fill="#E8E8ED",
    button_secondary_background_fill_hover="#DCDCE0",
    block_background_fill="#FFFFFF",
    background_fill_primary="#F2F2F7",
    border_color_primary="#D1D1D6",
    checkbox_label_text_weight="500",
)

UI_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.css")
with open(UI_CSS_PATH, encoding="utf-8") as css_file:
    UI_CSS = css_file.read()

def get_gui():
    with gr.Blocks(
        title="Audio Separator",
        fill_width=False,
        fill_height=False,
        delete_cache=(3200, 10800),
        theme=APP_THEME,
        css=UI_CSS,
    ) as app:
        gr.Markdown("# Audio Separator", elem_classes=["app-header"])
        gr.Markdown(
            "Trabaja en este Mac. YouTube, Edge y Completar instalación "
            "(Hugging Face) usan internet. Sin los pesos en disco, Separar / "
            "Entrenar / Convertir no arrancan offline.",
            elem_classes=["lede"],
        )
        gr.HTML(
            '<nav class="stepper" aria-label="Pasos">'
            "<span>1 Canción</span><span>2 Extraer</span>"
            "<span>3 Resultado</span><span>4 Voz</span>"
            "<span>5 Unir</span><span>6 Texto</span>"
            "</nav>",
            elem_classes=["stepper-wrap"],
        )
        status = gr.Markdown(_install_status_line(), elem_id="job-status")
        import library as _lib_ui

        _lib_ui.ensure_dirs()
        ensure_demo_voice()
        _rvc_choices = _lib_ui.dropdown_choices(_lib_ui.list_rvc_voices())
        _rvc_value = _rvc_choices[0][1] if _rvc_choices else None
        try:
            from install_rvc_assets import rvc_assets_ready as _rvc_ready

            _need_install = not _rvc_ready()
        except Exception:
            _need_install = True
        with gr.Accordion(
            "Instalación (una vez)",
            open=_need_install,
            elem_id="install-acc",
        ):
            install_btn = gr.Button(
                "Completar instalación",
                variant="secondary",
                elem_id="install-btn",
            )
            install_log = gr.Textbox(
                label="Log",
                interactive=False,
                lines=2,
                placeholder="Solo la primera vez: pesos públicos (~700 MB).",
            )

        with gr.Row(equal_height=False, elem_classes=["top-row"]):
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["step"]):
                    gr.Markdown("## 1. Canción", elem_classes=["panel-title"])
                    with gr.Tabs():
                        with gr.Tab("Archivo"):
                            gr.Markdown(
                                "Arrastrá el audio al reproductor o cargá el demo.",
                                elem_classes=["hint"],
                            )
                        with gr.Tab("YouTube"):
                            with gr.Row():
                                url_media_gui = url_media_conf()
                                url_button_gui = url_button_conf()
                            want_video = gr.Checkbox(
                                True,
                                label="También bajar el video (para pegarlo después)",
                            )
                            last_video = gr.State(value=None)
                    aud = audio_conf()
                    with gr.Row():
                        clip_start = gr.Textbox(
                            label="Inicio",
                            placeholder="0:15",
                            scale=1,
                        )
                        clip_end = gr.Textbox(
                            label="Fin",
                            placeholder="0:25",
                            scale=1,
                        )
                        clip_btn = gr.Button("Recortar", scale=1)
                    with gr.Row(elem_classes=["action-row"]):
                        demo_btn = gr.Button("Cargar demo", variant="secondary")
                    gr.Examples(
                        examples=[[DEMO_SONG]] if os.path.isfile(DEMO_SONG) else [],
                        inputs=[aud],
                        label="Ejemplo",
                        examples_per_page=1,
                    )

            with gr.Column(scale=5):
                with gr.Group(elem_classes=["step"]):
                    gr.Markdown("## 2. Extraer", elem_classes=["panel-title"])
                    stem_gui = stem_conf()
                    target_format_gui = format_conf()
                    button_base = button_conf()
                    gr.Markdown(
                        "En Intel tarda minutos. El botón queda en «Separando…»; "
                        "no cierres la ventana.",
                        elem_classes=["hint"],
                    )

        with gr.Group(elem_classes=["step"]):
            gr.Markdown("## 3. Resultado", elem_classes=["panel-title"])
            with gr.Row():
                vocal_out = out_audio("Voz")
                background_out = out_audio("Instrumental")
            output_base = output_conf()
            gr.Markdown(
                "Salida en Descargas. Usá **Abrir Descargas** (no el ícono ↓).",
                elem_classes=["hint"],
            )
            with gr.Row(elem_classes=["action-row"]):
                open_folder_btn = gr.Button("Abrir Descargas", variant="secondary")
                nueva_btn = gr.Button("Nueva canción", variant="secondary")

        with gr.Group(elem_classes=["step"], elem_id="step-voice"):
            gr.Markdown(
                "## 4. Voz (RVC)",
                elem_classes=["panel-title"],
            )
            gr.Markdown(
                "Entrená una voz o convertí con un modelo de la biblioteca.",
                elem_classes=["hint"],
            )
            with gr.Tabs():
                with gr.Tab("Entrenar"):
                    train_name = gr.Textbox(label="Nombre", placeholder="mi_voz")
                    train_dataset = gr.File(
                        label="Audios o videos de la persona",
                        file_count="multiple",
                        file_types=[
                            ".wav",
                            ".mp3",
                            ".flac",
                            ".m4a",
                            ".mp4",
                            ".mov",
                            ".mkv",
                            ".webm",
                            ".avi",
                        ],
                    )
                    train_epochs = gr.Slider(
                        5,
                        50,
                        value=10,
                        step=1,
                        label="Epochs (CPU Intel: 10 de prueba)",
                    )
                    gr.Markdown(
                        "Si subís un video, se extrae el audio y se entrena en este Mac. "
                        "10 epochs de prueba: minutos u horas según el largo. "
                        "El .pth aparece en Convertir al terminar. "
                        "No lo compartas si la voz no es tuya.",
                        elem_classes=["hint"],
                    )
                    train_btn = gr.Button(
                        "Entrenar", variant="primary", elem_id="train-btn"
                    )
                    train_status = gr.Textbox(
                        label="Estado",
                        interactive=False,
                        placeholder="Al terminar, el modelo aparece en Convertir.",
                    )
                    with gr.Accordion("Opcional: key RunPod", open=False):
                        gr.Markdown(
                            "El entrenamiento es en este Mac. Esto solo guarda "
                            "la key; no alquila GPU.",
                            elem_classes=["hint"],
                        )
                        hire_gpu = gr.Checkbox(
                            False,
                            label="Guardar key RunPod",
                        )
                        runpod_key = gr.Textbox(
                            label="RunPod API key",
                            type="password",
                            placeholder="Opcional. No alquila GPU.",
                        )
                with gr.Tab("Convertir"):
                    rvc_pick = gr.Dropdown(
                        label="Modelo en biblioteca",
                        choices=_rvc_choices,
                        value=_rvc_value,
                        info="Si está vacío, entrená una voz o cargá un .pth abajo.",
                    )
                    rvc_btn = gr.Button(
                        "Convertir voz", variant="primary", elem_id="rvc-btn"
                    )
                    rvc_audio = out_audio("Voz convertida")
            with gr.Accordion("Pesos / biblioteca", open=False):
                gr.Markdown(
                    "Si Convertir está vacío, entrená una voz o cargá un .pth. "
                    "Los pesos de apoyo (hubert, rmvpe) se instalan una vez arriba.",
                    elem_classes=["hint"],
                )
                with gr.Row():
                    rvc_hubert = gr.File(
                        label="hubert (.pt o carpeta zip)",
                        file_types=[".pt", ".pth"],
                    )
                    rvc_rmvpe = gr.File(
                        label="rmvpe.pt", file_types=[".pt", ".pth"]
                    )
                with gr.Row():
                    rvc_g = gr.File(
                        label="f0G40k.pth", file_types=[".pth", ".pt"]
                    )
                    rvc_d = gr.File(
                        label="f0D40k.pth", file_types=[".pth", ".pt"]
                    )
                with gr.Row():
                    rvc_model = gr.File(
                        label="Modelo .pth", file_types=[".pth", ".pt"]
                    )
                    rvc_index = gr.File(
                        label="Índice .index", file_types=[".index"]
                    )
                with gr.Row(elem_classes=["action-row"]):
                    load_rvc_btn = gr.Button(
                        "Cargar en biblioteca", variant="secondary"
                    )
                    refresh_lib_btn = gr.Button(
                        "Actualizar listas", variant="secondary"
                    )

        with gr.Group(elem_classes=["step"]):
            gr.Markdown("## 5. Unir", elem_classes=["panel-title"])
            with gr.Row():
                remix_voice = gr.Audio(
                    label="Voz", type="filepath", sources=["upload"]
                )
                remix_inst = gr.Audio(
                    label="Instrumental", type="filepath", sources=["upload"]
                )
            with gr.Row():
                remix_voice_db = gr.Slider(
                    -20, 12, value=0, step=1, label="Volumen voz (dB)"
                )
                remix_inst_db = gr.Slider(
                    -20, 12, value=0, step=1, label="Volumen instrumental (dB)"
                )
            remix_btn = gr.Button(
                "Unir voz + instrumental", variant="primary", elem_id="join-btn"
            )
            remix_audio = out_audio("Unión")
            remix_file = out_file("Archivo unido")
            with gr.Accordion("Video y portada", open=False):
                with gr.Row():
                    remux_video_in = gr.Video(
                        label="Video original",
                        sources=["upload"],
                    )
                    remux_audio_in = gr.Audio(
                        label="Audio nuevo",
                        type="filepath",
                        sources=["upload"],
                        buttons=[],
                    )
                remux_btn = gr.Button("Pegar audio al video", variant="secondary")
                remux_file = out_file("MP4 unido")
                with gr.Row():
                    cover_title = gr.Textbox(
                        label="Título", placeholder="Nombre del tema"
                    )
                    cover_artist = gr.Textbox(label="Artista", placeholder="Opcional")
                with gr.Row():
                    cover_btn = gr.Button("Portada local", variant="secondary")
                    cover_ai_btn = gr.Button(
                        "Portada artística", variant="secondary"
                    )
                cover_preview = gr.Image(label="Portada", type="filepath")
            with gr.Accordion("Si la voz es otra grabación", open=False):
                remix_delay = gr.Slider(
                    -2000,
                    2000,
                    value=0,
                    step=10,
                    label="Retraso de la voz (ms)",
                )
                remix_match = gr.Checkbox(
                    False,
                    label="Igualar duración al instrumental",
                )

        with gr.Group(elem_classes=["step"], elem_id="step-tts"):
            gr.Markdown("## 6. Texto → voz", elem_classes=["panel-title"])
            gr.Markdown(
                "ElevenLabs si hay key; si no, Edge (internet). Después aplica "
                "tu modelo RVC de la biblioteca.",
                elem_classes=["hint"],
            )
            tts_text = gr.Textbox(
                label="Texto",
                lines=3,
                placeholder="Escribí lo que tiene que decir la voz…",
            )
            with gr.Row():
                import tts_rvc_engine as _tts_rvc_ui

                tts_edge = gr.Dropdown(
                    label="Voz Edge (idioma base)",
                    choices=_tts_rvc_ui.EDGE_VOICES,
                    value="es-AR-ElenaNeural",
                    allow_custom_value=False,
                )
                tts_pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch RVC")
            tts_rvc_pick = gr.Dropdown(
                label="Modelo RVC (biblioteca)",
                choices=_rvc_choices,
                value=_rvc_value,
                info="El mismo que en el paso 4.",
            )
            tts_btn = gr.Button(
                "Generar voz", variant="primary", elem_id="tts-rvc-btn"
            )
            tts_audio = out_audio("Salida")

        with gr.Accordion("Opciones avanzadas", open=False):
            with gr.Row():
                main_gui = main_conf()
                dereverb_gui = dereverb_conf()
            with gr.Row():
                vocal_effects_gui = vocal_effects_conf()
                background_effects_gui = background_effects_conf()
            with gr.Accordion("Efectos de voz", open=False, visible=False) as vocal_acc:
                with gr.Row():
                    vocal_reverb_room_size_gui = vocal_reverb_room_size_conf()
                    vocal_reverb_damping_gui = vocal_reverb_damping_conf()
                with gr.Row():
                    vocal_reverb_dryness_gui = vocal_reverb_dryness_level_conf()
                    vocal_reverb_wet_level_gui = vocal_reverb_wet_level_conf()
                with gr.Row():
                    vocal_delay_seconds_gui = vocal_delay_seconds_conf()
                    vocal_delay_mix_gui = vocal_delay_mix_conf()
                with gr.Row():
                    vocal_gain_db_gui = vocal_gain_db_conf()
                with gr.Row():
                    vocal_compressor_threshold_db_gui = vocal_compressor_threshold_db_conf()
                    vocal_compressor_ratio_gui = vocal_compressor_ratio_conf()
                with gr.Row():
                    vocal_compressor_attack_ms_gui = vocal_compressor_attack_ms_conf()
                    vocal_compressor_release_ms_gui = vocal_compressor_release_ms_conf()
            with gr.Accordion("Efectos de instrumental", open=False, visible=False) as background_acc:
                with gr.Row():
                    background_highpass_freq_gui = background_highpass_freq_conf()
                    background_lowpass_freq_gui = background_lowpass_freq_conf()
                with gr.Row():
                    background_reverb_room_size_gui = background_reverb_room_size_conf()
                    background_reverb_damping_gui = background_reverb_damping_conf()
                with gr.Row():
                    background_reverb_wet_level_gui = background_reverb_wet_level_conf()
                    background_gain_db_gui = background_gain_db_conf()
                with gr.Row():
                    background_compressor_threshold_db_gui = background_compressor_threshold_db_conf()
                    background_compressor_ratio_gui = background_compressor_ratio_conf()
                with gr.Row():
                    background_compressor_attack_ms_gui = background_compressor_attack_ms_conf()
                    background_compressor_release_ms_gui = background_compressor_release_ms_conf()

        install_btn.click(
            install_rvc_job,
            outputs=[
                status,
                install_log,
                aud,
                button_base,
                rvc_pick,
                tts_rvc_pick,
            ],
            show_progress="full",
            concurrency_limit=1,
        )
        demo_btn.click(
            load_demo_bundle,
            outputs=[aud, button_base, rvc_pick, tts_rvc_pick, status],
        )
        url_button_gui.click(
            lock_download_button,
            outputs=[url_button_gui, status],
        ).then(
            audio_downloader,
            [url_media_gui, want_video],
            [aud, last_video, button_base, status, url_button_gui],
            show_progress="full",
            concurrency_limit=1,
        )
        last_video.change(lambda p: p, last_video, remux_video_in)
        aud.change(on_audio_ready, aud, [button_base, status])
        stem_gui.change(
            show_vocal_components,
            [stem_gui],
            [main_gui, dereverb_gui, vocal_effects_gui, background_effects_gui],
        )
        vocal_effects_gui.change(
            lambda active: gr.update(visible=active),
            vocal_effects_gui,
            vocal_acc,
        )
        background_effects_gui.change(
            lambda active: gr.update(visible=active),
            background_effects_gui,
            background_acc,
        )
        def _open_folder():
            from ui_status import KIND_OK, status_update

            return status_update(KIND_OK, f"Carpeta abierta: {open_exports_dir()}")

        open_folder_btn.click(
            _open_folder,
            outputs=[status],
        )
        nueva_btn.click(
            reset_job,
            outputs=[
                aud,
                vocal_out,
                background_out,
                output_base,
                status,
                button_base,
                url_media_gui,
                remix_inst,
                remix_voice,
                remix_audio,
                remix_file,
                rvc_audio,
                rvc_model,
                tts_audio,
                tts_text,
            ],
        )
        clip_btn.click(
            clip_for_clone,
            inputs=[aud, clip_start, clip_end],
            outputs=[aud, status],
            show_progress="full",
        )
        refresh_lib_btn.click(
            refresh_library_ui, outputs=[rvc_pick, tts_rvc_pick]
        )
        tts_btn.click(
            tts_rvc_job,
            inputs=[tts_text, tts_rvc_pick, tts_edge, tts_pitch],
            outputs=[tts_audio, remix_voice, status],
            show_progress="full",
            concurrency_limit=1,
        )
        vocal_out.change(lambda path: path, vocal_out, remix_voice)
        background_out.change(lambda path: path, background_out, remix_inst)
        remix_audio.change(lambda path: path, remix_audio, remux_audio_in)
        remux_btn.click(
            remux_job,
            inputs=[remux_video_in, remux_audio_in],
            outputs=[remux_file, status],
            show_progress="full",
            concurrency_limit=1,
        )
        cover_btn.click(
            lambda t, a, p: cover_job(t, a, p, False),
            inputs=[cover_title, cover_artist, remux_audio_in],
            outputs=[cover_preview, status],
        )
        cover_ai_btn.click(
            lambda t, a, p: cover_job(t, a, p, True),
            inputs=[cover_title, cover_artist, remux_audio_in],
            outputs=[cover_preview, status],
        )
        load_rvc_btn.click(
            load_rvc_into_library,
            inputs=[rvc_hubert, rvc_rmvpe, rvc_model, rvc_index, rvc_g, rvc_d],
            outputs=[rvc_pick, tts_rvc_pick, status],
        )
        rvc_btn.click(
            rvc_job,
            inputs=[vocal_out, rvc_pick, rvc_model, rvc_index],
            outputs=[rvc_audio, remix_voice, status],
            show_progress="full",
            concurrency_limit=1,
        )
        train_btn.click(
            train_rvc_job,
            inputs=[train_name, train_dataset, train_epochs, hire_gpu, runpod_key],
            outputs=[rvc_pick, status, tts_rvc_pick, train_status],
            show_progress="full",
            concurrency_limit=1,
        )
        def _boot_ui():
            import library

            last_vid = library.get_session_meta().get("last_video_path")
            if last_vid and not os.path.isfile(last_vid):
                last_vid = None
            last_audio = library.get_session_meta().get("last_audio_path")
            if last_audio and not os.path.isfile(last_audio):
                last_audio = None
            ensure_demo_voice()
            rvc_upd, tts_upd = refresh_library_ui()
            song = last_audio or demo_song_path()
            run = unlock_run_button() if song else gr.update()
            from occupancy import HOLD_TRAIN, snapshot
            from ui_status import KIND_OK, KIND_RUN, RUN_TRAIN, status_update

            occ = snapshot()
            if occ is not None and occ.holder == HOLD_TRAIN:
                status_txt = status_update(KIND_RUN, RUN_TRAIN)
            elif song and song == demo_song_path() and not last_audio:
                status_txt = status_update(KIND_OK, DEMO_STATUS)
            elif not song:
                status_txt = status_update(KIND_OK, _install_status_line())
            else:
                status_txt = status_update(KIND_OK, READY_STATUS)
            return rvc_upd, tts_upd, last_vid, last_vid, song, run, status_txt

        app.load(
            _boot_ui,
            outputs=[
                rvc_pick,
                tts_rvc_pick,
                last_video,
                remux_video_in,
                aud,
                button_base,
                status,
            ],
        )
        remix_btn.click(
            remix_job,
            inputs=[
                remix_voice,
                remix_inst,
                remix_delay,
                remix_match,
                remix_voice_db,
                remix_inst_db,
                target_format_gui,
            ],
            outputs=[remix_audio, remix_file, status],
            show_progress="full",
            concurrency_limit=1,
        )
        button_base.click(
            sound_separate,
            inputs=[
                aud,
                stem_gui,
                main_gui,
                dereverb_gui,
                vocal_effects_gui,
                background_effects_gui,
                vocal_reverb_room_size_gui, vocal_reverb_damping_gui, vocal_reverb_dryness_gui, vocal_reverb_wet_level_gui,
                vocal_delay_seconds_gui, vocal_delay_mix_gui, vocal_compressor_threshold_db_gui, vocal_compressor_ratio_gui,
                vocal_compressor_attack_ms_gui, vocal_compressor_release_ms_gui, vocal_gain_db_gui,
                background_highpass_freq_gui, background_lowpass_freq_gui, background_reverb_room_size_gui,
                background_reverb_damping_gui, background_reverb_wet_level_gui, background_compressor_threshold_db_gui,
                background_compressor_ratio_gui, background_compressor_attack_ms_gui, background_compressor_release_ms_gui,
                background_gain_db_gui, target_format_gui,
            ],
            outputs=[vocal_out, background_out, output_base, status, button_base],
            show_progress="full",
            concurrency_limit=1,
        )

    return app


def build_server():
    demo = get_gui()
    demo.queue(default_concurrency_limit=1)
    return demo


def launch_kwargs(**overrides):
    settings = dict(
        max_threads=4,
        share=IS_COLAB,
        show_error=True,
        quiet=False,
        debug=IS_COLAB,
        ssr_mode=False,
        theme=APP_THEME,
        css=UI_CSS,
        footer_links=[],
        inbrowser=False,
        server_name=__import__("app_env").host(),
        server_port=int(
            os.environ.get("AUDIO_SEPARATOR_PORT")
            or __import__("app_env").preferred_port()
        ),
        allowed_paths=[
            __import__("app_env").home(),
            __import__("app_env").data_dir(),
            os.path.join(os.path.expanduser("~"), "Downloads"),
        ],
    )
    settings.update(overrides)
    return settings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the app with optional sharing")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--theme", type=str, default="NoCrypt/miku")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.share:
        IS_COLAB = True
    build_server().launch(**launch_kwargs(inbrowser=args.open))
