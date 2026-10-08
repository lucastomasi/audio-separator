"""GUI wiring, reset, and copy. Visual QA stays on the native Mac window."""
import os
import socket
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import app
import app_jobs


ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _get(url, token=None):
    headers = {}
    if token:
        headers[app.TOKEN_HEADER] = token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class GuiWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = app.get_gui()

    def test_event_arity_matches_handlers(self):
        expected = {
            "lock_install_button": (0, 2),
            "install_rvc_job": (0, 7),
            "load_demo_bundle": (0, 5),
            "lock_download_button": (0, 2),
            "audio_downloader": (2, 6),
            "on_audio_ready": (1, 3),
            "show_vocal_components": (1, 4),
            "reset_job": (0, 19),
            "lock_clip_button": (0, 2),
            "clip_for_clone": (3, 3),
            "refresh_library_ui": (0, 1),
            "_join_ready": (2, 2),
            "remux_job": (2, 2),
            "load_rvc_into_library": (6, 2),
            "_pull_separated": (1, 2),
            "lock_convert_button": (0, 2),
            "rvc_job": (6, 4),
            "_use_recent": (1, 2),
            "_continue_last": (0, 4),
            "lock_train_button": (0, 2),
            "_on_train": (3, 4),
            "_boot_ui": (0, 8),
            "lock_chords_button": (0, 2),
            "chords_job": (1, 4),
            "lock_join_button": (0, 2),
            "remix_job": (7, 4),
            "sound_separate": (28, 5),
        }
        seen = {}
        for fn in self.demo.fns.values():
            name = fn.fn.__name__ if fn.fn is not None else ""
            if name in expected:
                seen[name] = (len(fn.inputs or []), len(fn.outputs or []))
        self.assertEqual(seen, expected)

    def test_no_tts_handlers(self):
        names = {fn.fn.__name__ for fn in self.demo.fns.values() if fn.fn}
        self.assertNotIn("tts_rvc_job", names)
        self.assertNotIn("lock_tts_button", names)

    def test_reset_job_clears_convert_and_join(self):
        out = app_jobs.reset_job()
        self.assertEqual(len(out), 19)
        self.assertIsNone(out[12])
        self.assertEqual(out[14], "Falta la voz y el instrumental.")
        self.assertFalse(out[15].get("interactive", True))
        self.assertEqual(out[16], "")
        self.assertIsNone(out[17])
        self.assertFalse(out[18].get("interactive", True))
        self.assertEqual(out[18].get("value"), "Estimar acordes")


class GuiCssTests(unittest.TestCase):
    def test_css_drops_tts_and_does_not_force_install_primary(self):
        css = (ROOT / "ui.css").read_text(encoding="utf-8")
        self.assertNotIn("tts-rvc-btn", css)
        self.assertIn("#run-btn", css)
        self.assertIn("#chords-btn button", css)
        self.assertIn("#run-btn button", css)
        self.assertNotIn("#install-btn button", css)


class GuiHeadlessCopyTests(unittest.TestCase):
    def setUp(self):
        self._old_token = os.environ.get(app.TOKEN_ENV)
        os.environ[app.TOKEN_ENV] = "g" * 32
        self.token = os.environ[app.TOKEN_ENV]
        self.port = _free_port()
        self.demo = app.build_server()
        from pwa_server import launch_with_pwa

        launch_with_pwa(
            self.demo,
            app.launch_kwargs(
                prevent_thread_lock=True,
                quiet=True,
                server_name="127.0.0.1",
                server_port=self.port,
            ),
        )

    def tearDown(self):
        self.demo.close()
        if self._old_token is None:
            os.environ.pop(app.TOKEN_ENV, None)
        else:
            os.environ[app.TOKEN_ENV] = self._old_token

    def test_tabs_and_no_texto(self):
        status, body = _get(f"http://127.0.0.1:{self.port}/", token=self.token)
        self.assertEqual(status, 200)
        text = body.decode("utf-8", errors="replace")
        for label in (
            "1 Canción",
            "2 Separar",
            "3 Convertir",
            "4 Unir",
            "Acordes",
            "Entrenar",
            "Ajustes",
        ):
            self.assertIn(label, text)
        self.assertIn("Estimar acordes", text)
        self.assertIn("no es la tablatura de la grabación", text)
        self.assertNotIn(">Texto<", text)
        self.assertNotIn("tts-rvc-btn", text)
        self.assertNotIn("ElevenLabs", text)
        self.assertIn("job-status", text)


if __name__ == "__main__":
    unittest.main()
