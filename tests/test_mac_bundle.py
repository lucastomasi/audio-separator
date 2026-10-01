"""The Mac .app ships the downloader. The window is desktop.py, not a browser."""
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "dist", "Audio Separator.app", "Contents", "Resources", "app")


class MacBundleTests(unittest.TestCase):
    def test_layout_includes_the_downloader_and_the_desktop_window(self):
        subprocess.check_call(["bash", "macos/build_app.sh", "--layout-only"], cwd=ROOT)
        desktop = os.path.join(APP_DIR, "desktop.py")
        self.assertTrue(os.path.isfile(os.path.join(APP_DIR, "model_fetch.py")))
        self.assertTrue(os.path.isfile(desktop))
        with open(desktop, encoding="utf-8") as handle:
            window = handle.read()
        self.assertIn("webview.create_window", window)
        self.assertIn("inbrowser=False", window)
        launcher = os.path.join(ROOT, "macos", "launcher.c")
        with open(launcher, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("/app/desktop.py", source)
        with tempfile.TemporaryDirectory() as home:
            env = os.environ.copy()
            env["AUDIO_SEPARATOR_HOME"] = home
            env["PYTHONPATH"] = APP_DIR
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    "import app, model_fetch; app.build_server(); "
                    "assert model_fetch.VOCAL_ONNX_NAME.endswith('.onnx')",
                ],
                cwd=APP_DIR,
                env=env,
            )
