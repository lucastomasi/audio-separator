import ast
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bundle_py import BUNDLE_PY


ROOT = Path(__file__).resolve().parents[1]


class FullZipIsCannedTests(unittest.TestCase):
    def test_standalone_requires_uvr_onnx(self):
        text = (ROOT / "scripts" / "build_standalone.sh").read_text(encoding="utf-8")
        self.assertIn("UVR-MDX-NET-Voc_FT.onnx", text)
        self.assertIn("el zip full es un enlatado", text)
        self.assertIn("no baja ONNX en el Mac de destino", text)
        self.assertIn("no baja RVC en el Mac de destino", text)
        self.assertIn("third_party/vc", text)
        self.assertNotIn("RVC-WebUI", text)
        self.assertIn("import gradio, torch", text)
        self.assertNotIn("import av, gradio, torch", text)

    def test_cloud_install_does_not_download_models(self):
        text = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
        self.assertNotIn("TRvlvr", text)
        self.assertNotIn("huggingface", text.lower())
        self.assertNotIn("curl ", text)
        self.assertIn("/opt/audio-separator-models", text)
        self.assertIn("No bajes nada", text)


class BundlePyTests(unittest.TestCase):
    def test_required_modules_listed(self):
        names = set(BUNDLE_PY)
        for name in (
            "occupancy.py",
            "train_run.py",
            "ui_status.py",
            "gpu_secrets.py",
            "eleven_tts.py",
            "desktop.py",
        ):
            self.assertIn(name, names)

    def test_covers_local_imports(self):
        names = set(BUNDLE_PY)
        for mod in ("app.py", "app_jobs.py", "desktop.py"):
            tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                top = node.module.split(".")[0] + ".py"
                if (ROOT / top).is_file():
                    self.assertIn(top, names, f"{mod} imports {top}")


class SeedSupportTests(unittest.TestCase):
    def test_seed_copies_once(self):
        import library

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name) / "home"
        data = Path(tmp.name) / "data"
        hubert = home / "library" / "models" / "rvc" / "hubert_base"
        hubert.mkdir(parents=True)
        (hubert / "config.json").write_text("from-bundle", encoding="utf-8")
        env = {
            "AUDIO_SEPARATOR_HOME": str(home),
            "AUDIO_SEPARATOR_DATA": str(data),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            library.seed_support_weights()
            dest = data / "Voces" / "models" / "rvc" / "hubert_base" / "config.json"
            self.assertEqual(dest.read_text(encoding="utf-8"), "from-bundle")
            dest.write_text("keep", encoding="utf-8")
            library.seed_support_weights()
            self.assertEqual(dest.read_text(encoding="utf-8"), "keep")


class RelocateVenvTests(unittest.TestCase):
    def test_relative_python_and_cfg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "python" / "bin").mkdir(parents=True)
        py = root / "python" / "bin" / "python3.12"
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        venv = root / "venv-vc"
        (venv / "bin").mkdir(parents=True)
        script = ROOT / "scripts" / "relocate_venv.sh"
        subprocess.check_call(["bash", str(script), str(venv)])
        link = os.readlink(venv / "bin" / "python")
        self.assertEqual(link, "../../python/bin/python3.12")
        cfg = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        self.assertIn("home = ../python/bin", cfg)
        self.assertNotIn("lucastomasi", cfg)
        script = venv / "bin" / "f2py"
        script.write_text(
            "#!/Users/someone/project/.venv-vc/bin/python\nprint('ok')\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        subprocess.check_call(["bash", str(ROOT / "scripts" / "relocate_venv.sh"), str(venv)])
        self.assertEqual(script.read_text(encoding="utf-8").splitlines()[0], "#!/usr/bin/env python3")
