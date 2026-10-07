import os
import unittest
from unittest import mock

import app_env


class AppEnvTests(unittest.TestCase):
    def test_data_dir_respects_env(self):
        with mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_DATA": "/tmp/as-data-test"}):
            path = app_env.data_dir()
        self.assertTrue(path.endswith("as-data-test"))

    def test_host_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_SEPARATOR_HOST", None)
            self.assertEqual(app_env.host(), "127.0.0.1")

    def test_no_user_path_in_module(self):
        import inspect

        src = inspect.getsource(app_env)
        self.assertNotIn("lucastomasi", src)

    def test_ui_copy_keeps_youtube_drops_tts(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "app.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("YouTube", text)
        self.assertNotIn("Edge", text)
        self.assertNotIn("ElevenLabs", text)
        self.assertNotIn("sin subir audio", text.lower())
        self.assertNotIn("Alquilar GPU RunPod", text)
        self.assertNotIn("RunPod", text)
        self.assertNotIn("hire_gpu", text)
        self.assertNotIn("se apaga a los 45 min", text)
        self.assertIn("el entrenamiento es en este mac", text.lower())
        self.assertIn('gr.Tab("1 Canción", id="cancion")', text)
        self.assertIn('gr.Tab("2 Separar", id="separar")', text)
        self.assertIn('gr.Tab("3 Convertir", id="convertir")', text)
        self.assertIn('gr.Tab("4 Unir", id="unir")', text)
        self.assertIn('gr.Tab("Entrenar", id="entrenar")', text)
        self.assertIn('gr.Tab("Ajustes", id="ajustes")', text)
        self.assertLess(
            text.index('gr.Tab("4 Unir", id="unir")'),
            text.index('gr.Tab("Entrenar", id="entrenar")'),
        )
        self.assertLess(
            text.index('gr.Tab("Entrenar", id="entrenar")'),
            text.index('gr.Tab("Ajustes", id="ajustes")'),
        )
        self.assertNotIn('gr.Tab("Texto")', text)
        self.assertNotIn("lock_tts_button", text)
        self.assertNotIn("tts_rvc_job", text)
        self.assertNotIn('gr.Tab("2 Extraer")', text)
        self.assertNotIn('gr.Tab("3 Resultado")', text)
        self.assertNotIn('gr.Tab("4 Voz")', text)
        self.assertIn("Audio a convertir", text)
        self.assertIn("inputs=[rvc_in, rvc_pick", text)
        self.assertNotIn("inputs=[vocal_out, rvc_pick", text)
        self.assertNotIn("el resto corre offline", text.lower())
        self.assertIn("modelos uvr", text.lower())
        self.assertIn("solo baja lo que falte", text.lower())
        self.assertIn("Listo para unir", text)
        self.assertIn("Falta la voz y el instrumental", text)
        self.assertNotIn("stepper-wrap", text)
        self.assertGreater(
            text.index('gr.Accordion("Opciones avanzadas"'),
            text.index('gr.Tab("2 Separar", id="separar")'),
        )
        self.assertGreater(
            text.index('gr.Tab("3 Convertir", id="convertir")'),
            text.index('gr.Accordion("Opciones avanzadas"'),
        )
        self.assertIn('elem_classes=["app-chrome"]', text)
        self.assertIn('elem_classes=["panel-title"]', text)
        self.assertIn("lock_convert_button", text)
        self.assertIn("lock_train_button", text)
        self.assertIn("lock_join_button", text)
        self.assertIn("fill_height=True", text)
        self.assertNotIn("fill_height=False", text)

    def test_agents_md_locks_offline_core(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Product (hard rule)", text)
        self.assertIn("offline", text.lower())
        self.assertIn("Completar instalación", text)
        self.assertIn("never download mid-job", text.lower())
        self.assertIn("YouTube", text)
        self.assertIn("never reintroduce Edge/ElevenLabs", text)

    def test_rvc_job_missing_model_is_error(self):
        import inspect

        import app_jobs

        src = inspect.getsource(app_jobs.rvc_job)
        self.assertIn("KIND_ERROR", src)
        self.assertNotIn("return None, None, _ok(msg)", src)

    def test_remix_status_uses_target_format(self):
        import inspect

        import app_jobs

        src = inspect.getsource(app_jobs.remix_job)
        self.assertIn("target_format", src)
        self.assertNotIn("Pistas unidas (WAV)", src)

    def test_join_ready_requires_both_files(self):
        import tempfile

        import app as app_mod

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        missing, btn = app_mod._join_ready(None, None)
        self.assertIn("Falta", missing)
        self.assertFalse(btn.get("interactive", True))
        voice = os.path.join(tmp.name, "v.wav")
        inst = os.path.join(tmp.name, "i.wav")
        open(voice, "wb").close()
        open(inst, "wb").close()
        ready, btn = app_mod._join_ready(voice, inst)
        self.assertEqual(ready, "Listo para unir.")
        self.assertTrue(btn.get("interactive", False))

    def test_mp3_copy_not_320(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "ui_widgets.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("MP3 320", text)


if __name__ == "__main__":
    unittest.main()
