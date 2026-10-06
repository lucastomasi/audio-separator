import os
import gradio as gr
from audio_text import STEM_AMBAS, STEM_SOLO_VOZ, STEM_SOLO_INST

FORMAT_OPTIONS = ["MP3", "WAV", "FLAC"]

def url_media_conf():
    return gr.Textbox(
        value="",
        label="Enlace de YouTube",
        placeholder="https://www.youtube.com/watch?v=…",
        lines=1,
        scale=4,
    )


def url_button_conf():
    return gr.Button(
        "Descargar",
        variant="primary",
        scale=1,
    )


def audio_conf():
    return gr.Audio(
        label="Canción",
        type="filepath",
        sources=["upload"],
        # No download/share: in pywebview ↓ opens a dead-end player window.
        buttons=[],
    )


def out_audio(label: str):
    """Playback-only audio (files already copied to ~/Downloads/Audio Separator)."""
    return gr.Audio(
        label=label,
        type="filepath",
        interactive=False,
        buttons=[],
    )


def out_file(label: str, file_count: str = "single"):
    return gr.File(
        label=label,
        file_count=file_count,
        interactive=False,
        buttons=[],
    )


def stem_conf():
    return gr.Radio(
        choices=[
            ("Solo voz", STEM_SOLO_VOZ),
            ("Solo instrumental", STEM_SOLO_INST),
            ("Las dos", STEM_AMBAS),
        ],
        value=STEM_AMBAS,
        label="Qué extraer",
    )


def main_conf():
    return gr.Checkbox(
        False,
        label="Solo voz principal",
        info="Intenta dejar atrás coros. Tarda más.",
    )


def dereverb_conf():
    return gr.Checkbox(
        False,
        label="Quitar reverb",
        info="Tarda más.",
        visible=True,
    )


def vocal_effects_conf():
    return gr.Checkbox(
        False,
        label="Efectos de voz",
        visible=True,
    )


def background_effects_conf():
    return gr.Checkbox(
        False,
        label="Efectos de instrumental",
        visible=True,
    )


def vocal_reverb_room_size_conf():
    return gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Eco de sala")


def vocal_reverb_damping_conf():
    return gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Amortiguación")


def vocal_reverb_wet_level_conf():
    return gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Eco")


def vocal_reverb_dryness_level_conf():
    return gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Sonido directo")


def vocal_delay_seconds_conf():
    return gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Retraso (s)")


def vocal_delay_mix_conf():
    return gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Mezcla de eco")


def vocal_compressor_threshold_db_conf():
    return gr.Slider(-60, 0, value=-15, step=1, label="Umbral (dB)")


def vocal_compressor_ratio_conf():
    return gr.Slider(0, 20, value=4.0, step=0.1, label="Compresión")


def vocal_compressor_attack_ms_conf():
    return gr.Slider(0, 1000, value=1.0, step=1, label="Ataque (ms)")


def vocal_compressor_release_ms_conf():
    return gr.Slider(0, 3000, value=100, step=1, label="Soltar (ms)")


def vocal_gain_db_conf():
    return gr.Slider(-40, 40, value=0, step=1, label="Volumen (dB)")


def background_highpass_freq_conf():
    return gr.Slider(0, 1000, value=120, step=1, label="Quitar graves (Hz)")


def background_lowpass_freq_conf():
    return gr.Slider(0, 20000, value=11000, step=1, label="Quitar agudos (Hz)")


def background_reverb_room_size_conf():
    return gr.Slider(0.0, 1.0, value=0.1, step=0.1, label="Eco de sala")


def background_reverb_damping_conf():
    return gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Amortiguación")


def background_reverb_wet_level_conf():
    return gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Eco")


def background_compressor_threshold_db_conf():
    return gr.Slider(-60, 0, value=-15, step=1, label="Umbral (dB)")


def background_compressor_ratio_conf():
    return gr.Slider(0, 20, value=4.0, step=0.1, label="Compresión")


def background_compressor_attack_ms_conf():
    return gr.Slider(0, 1000, value=15, step=1, label="Ataque (ms)")


def background_compressor_release_ms_conf():
    return gr.Slider(0, 3000, value=60, step=1, label="Soltar (ms)")


def background_gain_db_conf():
    return gr.Slider(-40, 40, value=0, step=1, label="Volumen (dB)")


def button_conf():
    return gr.Button(
        "Separar audio",
        variant="primary",
        elem_id="run-btn",
        interactive=False,
    )


def output_conf():
    return out_file("Archivos en Descargas", file_count="multiple")


def show_vocal_components(value_name):
    v_ = value_name in (STEM_SOLO_VOZ, STEM_AMBAS, "vocal") or (
        isinstance(value_name, (list, tuple, set)) and "vocal" in value_name
    )
    b_ = value_name in (STEM_SOLO_INST, STEM_AMBAS, "background") or (
        isinstance(value_name, (list, tuple, set)) and "background" in value_name
    )
    return (
        gr.update(visible=v_),
        gr.update(visible=v_),
        gr.update(visible=v_),
        gr.update(visible=b_),
    )


def format_conf():
    return gr.Radio(
        choices=FORMAT_OPTIONS,
        value="WAV",
        label="Formato",
    )

