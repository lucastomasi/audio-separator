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
    _gradio_path,
    _install_status_line,
)

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
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


def _convert_interactive():
    from occupancy import HOLD_TRAIN, snapshot

    occ = snapshot()
    on = not (occ is not None and occ.holder == HOLD_TRAIN)
    return gr.update(interactive=on), gr.update(interactive=on)


def _join_ready(voice, inst):
    voice_path = _gradio_path(voice)
    inst_path = _gradio_path(inst)
    has_voice = bool(voice_path and os.path.isfile(str(voice_path)))
    has_inst = bool(inst_path and os.path.isfile(str(inst_path)))
    if has_voice and has_inst:
        text = "Listo para unir."
    elif not has_voice and not has_inst:
        text = "Falta la voz y el instrumental."
    elif not has_voice:
        text = "Falta la voz."
    else:
        text = "Falta el instrumental."
    return text, gr.update(interactive=has_voice and has_inst)


def _use_recent(path):
    import os

    from ui_status import KIND_ERROR, KIND_OK, status_update

    if not path or not os.path.isfile(str(path)):
        return gr.update(), status_update(KIND_ERROR, "Ese archivo ya no está.")
    return [path], status_update(
        KIND_OK, f"Elegido: {os.path.basename(str(path))}"
    )


def _continue_last():
    import os

    from train_run import latest_job
    from ui_status import KIND_ERROR, KIND_OK, status_update

    job = latest_job()
    if not job:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            status_update(KIND_ERROR, "No hay un entrenamiento anterior."),
        )
    files = [path for path in job.get("files") or [] if os.path.isfile(path)]
    if not files:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            status_update(KIND_ERROR, "El último entrenamiento no tiene archivos."),
        )
    epochs = int(job.get("epochs") or 10)
    return (
        job["exp"],
        files,
        epochs,
        status_update(KIND_OK, f"Último: {job['exp']}. Apretá Entrenar para seguir."),
    )


def _on_train(name, dataset, epochs, progress=gr.Progress()):
    name_s = (name or "").strip()
    if isinstance(dataset, (list, tuple)):
        files = [item for item in dataset if item]
    elif dataset:
        files = [dataset]
    else:
        files = []
    if not name_s or not files:
        from ui_status import KIND_ERROR, status_update

        if not name_s and not files:
            text = "Poné un nombre y subí al menos un audio de la voz."
        elif not name_s:
            text = "Poné un nombre para la voz."
        else:
            text = "Subí al menos un audio de la voz a entrenar."
        rvc_upd, tts_upd = refresh_library_ui()
        btn_a, btn_b = _convert_interactive()
        return rvc_upd, status_update(KIND_ERROR, text), tts_upd, btn_a, btn_b
    rvc_upd, bar, tts_upd = train_rvc_job(
        name_s, files, epochs, False, None, progress
    )
    btn_a, btn_b = _convert_interactive()
    return rvc_upd, bar, tts_upd, btn_a, btn_b


def get_gui():
    with gr.Blocks(
        title="Audio Separator",
        fill_width=True,
        fill_height=False,
        delete_cache=(3200, 10800),
        theme=APP_THEME,
        css=UI_CSS,
    ) as app:
        gr.Markdown("# Audio Separator", elem_classes=["app-header"])
        gr.Markdown(
            "Separá, cambiá la voz y uní en este Mac.",
            elem_classes=["lede"],
        )
        gr.HTML(
            '<p class="stepper">'
            "<span>Canción</span><span>Separar</span>"
            "<span>Convertir</span><span>Unir</span></p>",
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

        last_video = gr.State(value=None)
        with gr.Tabs(
            elem_id="main-tabs",
            selected="ajustes" if _need_install else "cancion",
        ):
            with gr.Tab("1 Canción", id="cancion"):
                with gr.Group(elem_classes=["step"]):
                    with gr.Tabs():
                        with gr.Tab("Archivo"):
                            gr.Markdown(
                                "Arrastrá el audio al reproductor o cargá el demo.",
                                elem_classes=["hint"],
                            )
                            aud = audio_conf()
                            with gr.Row():
                                clip_start = gr.Textbox(
                                    label="Inicio (m:ss)",
                                    placeholder="0:15",
                                    scale=1,
                                )
                                clip_end = gr.Textbox(
                                    label="Fin (m:ss)",
                                    placeholder="0:25",
                                    scale=1,
                                )
                                clip_btn = gr.Button("Recortar", scale=1)
                            with gr.Row(elem_classes=["action-row"]):
                                demo_btn = gr.Button(
                                    "Cargar demo", variant="secondary"
                                )
                        with gr.Tab("YouTube"):
                            with gr.Row():
                                url_media_gui = url_media_conf()
                                url_button_gui = url_button_conf()
                            want_video = gr.Checkbox(
                                True,
                                label="También bajar el video (para pegarlo después)",
                            )

            with gr.Tab("2 Separar", id="separar"):
                with gr.Group(elem_classes=["step"]):
                    stem_gui = stem_conf()
                    target_format_gui = format_conf()
                    button_base = button_conf()
                    gr.Markdown(
                        "Por defecto saca voz e instrumental. "
                        "En Intel tarda minutos; no cierres la ventana.",
                        elem_classes=["hint"],
                    )
                    with gr.Row():
                        vocal_out = out_audio("Voz")
                        background_out = out_audio("Instrumental")
                    output_base = output_conf()
                    gr.Markdown(
                        "Salida en Descargas. Usá **Mostrar en Finder** (no el ícono ↓).",
                        elem_classes=["hint"],
                    )
                    with gr.Row(elem_classes=["action-row"]):
                        open_folder_btn = gr.Button(
                            "Mostrar en Finder", variant="secondary"
                        )
                        nueva_btn = gr.Button(
                            "Nueva canción", variant="secondary"
                        )

            with gr.Tab("3 Convertir", id="convertir"):
                with gr.Group(elem_classes=["step"]):
                    rvc_in = gr.Audio(
                        label="Audio a convertir",
                        type="filepath",
                        sources=["upload"],
                        buttons=[],
                    )
                    gr.Markdown(
                        "La voz separada entra sola. El modelo solo cambia el timbre.",
                        elem_classes=["hint"],
                    )
                    rvc_pick = gr.Dropdown(
                        label="Buscar modelo",
                        choices=_rvc_choices,
                        value=_rvc_value,
                        info="Si está vacío, entrená o cargá un .pth en Biblioteca.",
                        elem_classes=["model-search"],
                        filterable=True,
                    )
                    rvc_btn = gr.Button(
                        "Convertir voz",
                        variant="primary",
                        elem_id="rvc-btn",
                    )
                    rvc_audio = out_audio("Voz convertida")
                    open_rvc_btn = gr.Button(
                        "Mostrar en Finder", variant="secondary"
                    )
                    with gr.Accordion("Si no trajo la voz", open=False):
                        use_sep_btn = gr.Button(
                            "Usar voz separada", variant="secondary"
                        )
                        rvc_same = gr.Checkbox(
                            False,
                            label="Este archivo está en Entrenar",
                        )

            with gr.Tab("4 Unir", id="unir"):
                with gr.Group(elem_classes=["step"]):
                    with gr.Row():
                        remix_voice = gr.Audio(
                            label="Voz",
                            type="filepath",
                            sources=["upload"],
                            buttons=[],
                        )
                        remix_inst = gr.Audio(
                            label="Instrumental",
                            type="filepath",
                            sources=["upload"],
                            buttons=[],
                        )
                    join_ready = gr.Markdown(
                        "Falta la voz y el instrumental.",
                        elem_classes=["hint"],
                    )
                    with gr.Row():
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
                    with gr.Row():
                        remix_voice_db = gr.Slider(
                            -20, 12, value=0, step=1, label="Volumen voz (dB)"
                        )
                        remix_inst_db = gr.Slider(
                            -20,
                            12,
                            value=0,
                            step=1,
                            label="Volumen instrumental (dB)",
                        )
                    remix_btn = gr.Button(
                        "Unir",
                        variant="primary",
                        elem_id="join-btn",
                        interactive=False,
                    )
                    remix_audio = out_audio("Unión")
                    remix_file = out_file("Archivo unido")
                    open_join_btn = gr.Button(
                        "Mostrar en Finder", variant="secondary"
                    )
                    with gr.Accordion("Video y portada", open=False):
                        with gr.Row():
                            remux_video_in = gr.File(
                                label="Video o imagen",
                                file_types=[
                                    ".mp4",
                                    ".mov",
                                    ".mkv",
                                    ".webm",
                                    ".avi",
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                    ".webp",
                                    ".gif",
                                    ".bmp",
                                ],
                            )
                            remux_audio_in = gr.Audio(
                                label="Audio nuevo",
                                type="filepath",
                                sources=["upload"],
                                buttons=[],
                            )
                        remux_btn = gr.Button(
                            "Pegar audio al video o a la imagen", variant="secondary"
                        )
                        remux_file = out_file("MP4 unido")
                        with gr.Row():
                            cover_title = gr.Textbox(
                                label="Título", placeholder="Nombre del tema"
                            )
                            cover_artist = gr.Textbox(
                                label="Artista", placeholder="Opcional"
                            )
                        with gr.Row():
                            cover_btn = gr.Button(
                                "Portada local", variant="secondary"
                            )
                            cover_ai_btn = gr.Button(
                                "Portada artística", variant="secondary"
                            )
                        cover_preview = gr.Image(
                            label="Portada",
                            type="filepath",
                            buttons=[],
                        )

            with gr.Tab("Entrenar", id="entrenar"):
                with gr.Group(elem_classes=["step"]):
                    train_name = gr.Textbox(
                        label="Nombre", placeholder="mi_voz"
                    )
                    _recent_choices = _lib_ui.recent_media()
                    train_recent = gr.Dropdown(
                        label="Últimos en esta Mac",
                        choices=_recent_choices,
                        value=_recent_choices[0][1] if _recent_choices else None,
                    )
                    with gr.Row():
                        use_recent_btn = gr.Button(
                            "Elegir este", variant="secondary"
                        )
                        continue_btn = gr.Button(
                            "Continuar el último", variant="secondary"
                        )
                    train_dataset = gr.File(
                        label="Elegir archivo",
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
                        "El entrenamiento es en este Mac. "
                        "Separá solo la voz, sin quitar reverb. "
                        "10 epochs de prueba. "
                        "Cerrar la ventana no corta el entrenamiento; Separar y Convertir sí. "
                        "No lo compartas si la voz no es tuya.",
                        elem_classes=["hint"],
                    )
                    train_btn = gr.Button(
                        "Entrenar",
                        variant="primary",
                        elem_id="train-btn",
                    )

            with gr.Tab("Ajustes", id="ajustes"):
                with gr.Tabs():
                    with gr.Tab("Instalación"):
                        with gr.Group(elem_classes=["step"]):
                            gr.Markdown(
                                "Una vez: pesos públicos (~700 MB, Hugging Face).",
                                elem_classes=["hint"],
                            )
                            install_btn = gr.Button(
                                "Completar instalación",
                                variant="secondary",
                                elem_id="install-btn",
                            )
                            install_log = gr.Textbox(
                                label="Registro de instalación",
                                interactive=False,
                                lines=2,
                                placeholder="Solo la primera vez.",
                            )
                    with gr.Tab("Biblioteca"):
                        with gr.Group(elem_classes=["step"]):
                            gr.Markdown(
                                "Si Convertir está vacío, entrená una voz o cargá un .pth. "
                                "hubert y rmvpe se instalan una vez en Instalación.",
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
                    with gr.Tab("Texto"):
                        with gr.Group(elem_classes=["step"]):
                            gr.Markdown(
                                "ElevenLabs si hay key; si no, Edge (internet). "
                                "Después aplica tu modelo de la biblioteca.",
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
                                tts_pitch = gr.Slider(
                                    -12, 12, value=0, step=1, label="Tono"
                                )
                            tts_rvc_pick = gr.Dropdown(
                                label="Buscar modelo",
                                choices=_rvc_choices,
                                value=_rvc_value,
                                info="El mismo que en Convertir.",
                                elem_classes=["model-search"],
                                filterable=True,
                            )
                            tts_btn = gr.Button(
                                "Generar voz",
                                variant="primary",
                                elem_id="tts-rvc-btn",
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
            show_progress="minimal",
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
            show_progress="minimal",
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
        open_rvc_btn.click(
            _open_folder,
            outputs=[status],
        )
        open_join_btn.click(
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
            show_progress="minimal",
        )
        refresh_lib_btn.click(
            refresh_library_ui, outputs=[rvc_pick, tts_rvc_pick]
        )
        tts_btn.click(
            tts_rvc_job,
            inputs=[tts_text, tts_rvc_pick, tts_edge, tts_pitch],
            outputs=[tts_audio, remix_voice, status],
            show_progress="minimal",
            concurrency_limit=1,
        )
        vocal_out.change(
            lambda path: (path, path),
            vocal_out,
            [remix_voice, rvc_in],
        )
        background_out.change(lambda path: path, background_out, remix_inst)
        remix_voice.change(
            _join_ready,
            [remix_voice, remix_inst],
            [join_ready, remix_btn],
        )
        remix_inst.change(
            _join_ready,
            [remix_voice, remix_inst],
            [join_ready, remix_btn],
        )
        remix_audio.change(lambda path: path, remix_audio, remux_audio_in)
        remux_btn.click(
            remux_job,
            inputs=[remux_video_in, remux_audio_in],
            outputs=[remux_file, status],
            show_progress="minimal",
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
        def _pull_separated(vocal):
            from ui_status import KIND_ERROR, KIND_OK, status_update

            path = vocal.get("path") if isinstance(vocal, dict) else vocal
            if not path or not os.path.isfile(str(path)):
                return None, status_update(
                    KIND_ERROR, "Separar no tiene una voz para traer."
                )
            name = os.path.basename(str(path))
            return path, status_update(
                KIND_OK, f"Melodía: {name} (desde Separar)."
            )

        use_sep_btn.click(
            _pull_separated,
            inputs=[vocal_out],
            outputs=[rvc_in, status],
        )
        rvc_btn.click(
            rvc_job,
            inputs=[rvc_in, rvc_pick, rvc_model, rvc_index, train_dataset, rvc_same],
            outputs=[rvc_audio, remix_voice, status],
            show_progress="minimal",
            concurrency_limit=1,
        )
        use_recent_btn.click(
            _use_recent,
            inputs=[train_recent],
            outputs=[train_dataset, status],
        )
        continue_btn.click(
            _continue_last,
            outputs=[train_name, train_dataset, train_epochs, status],
        )
        train_btn.click(
            _on_train,
            inputs=[train_name, train_dataset, train_epochs],
            outputs=[rvc_pick, status, tts_rvc_pick, rvc_btn, tts_btn],
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
            from ui_status import KIND_ERROR, KIND_OK, KIND_RUN, RUN_TRAIN, status_update

            occ = snapshot()
            rvc_on, tts_on = _convert_interactive()
            from train_run import boot_status

            persisted = boot_status()
            if occ is not None and occ.holder == HOLD_TRAIN:
                status_txt = status_update(KIND_RUN, persisted or RUN_TRAIN)
            elif persisted and persisted.startswith("Modelo listo:"):
                status_txt = status_update(KIND_OK, persisted)
            elif song and song == demo_song_path() and not last_audio:
                status_txt = status_update(KIND_OK, DEMO_STATUS)
            elif not song:
                status_txt = status_update(KIND_OK, _install_status_line())
            else:
                status_txt = status_update(KIND_OK, READY_STATUS)
            return (
                rvc_upd,
                tts_upd,
                last_vid,
                last_vid,
                song,
                run,
                status_txt,
                rvc_on,
                tts_on,
            )

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
                rvc_btn,
                tts_btn,
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
            show_progress="minimal",
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
            show_progress="minimal",
            concurrency_limit=1,
        )

    return app


def build_server():
    demo = get_gui()
    demo.queue(default_concurrency_limit=1)
    return demo


def _allowed_paths():
    import app_security

    return app_security.allowed_paths()


def _blocked_paths():
    import app_security

    return app_security.blocked_paths()


def auth_dependency(request):
    import app_security

    return app_security.auth_dependency(request)


def _ui_error(exc):
    import app_security

    return app_security.ui_error(exc)


def _mdx_config(model_path, session):
    import app_security

    return app_security.mdx_config(model_path, session)


TOKEN_ENV = "AUDIO_SEPARATOR_TOKEN"
TOKEN_HEADER = "x-audio-separator-token"
TOKEN_QUERY = "access_token"


def launch_kwargs(**overrides):
    import inspect

    import app_security

    settings = dict(
        max_threads=4,
        share=False if not IS_COLAB else True,
        show_error=False,
        quiet=False,
        debug=IS_COLAB,
        ssr_mode=False,
        theme=APP_THEME,
        css=UI_CSS,
        head=app_security._HEAD_TOKEN_JS,
        footer_links=[],
        inbrowser=False,
        server_name=__import__("app_env").host(),
        server_port=int(
            os.environ.get("AUDIO_SEPARATOR_PORT")
            or __import__("app_env").preferred_port()
        ),
        allowed_paths=app_security.allowed_paths(),
        blocked_paths=app_security.blocked_paths(),
        max_file_size=app_security.MAX_UPLOAD,
        strict_cors=True,
        enable_monitoring=False,
        mcp_server=False,
        auth_dependency=auth_dependency,
        app_kwargs={
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        },
    )
    settings.update(overrides)
    supported = set(inspect.signature(gr.Blocks.launch).parameters)
    return {key: value for key, value in settings.items() if key in supported}


if __name__ == "__main__":
    import argparse

    import app_security

    parser = argparse.ArgumentParser(description="Run the app with optional sharing")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--theme", type=str, default=None)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.share:
        IS_COLAB = True
    app_security.ensure_token()
    build_server().launch(**launch_kwargs(inbrowser=args.open))
