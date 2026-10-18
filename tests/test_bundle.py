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
        self.assertIn("bootstrap_macos.sh", text)
        self.assertNotIn("xtts", text)
        self.assertNotIn("clone_engine", text)
        self.assertNotIn("eleven_tts", text)
        self.assertNotIn("tts_rvc_engine", text)
        self.assertNotIn("edge-tts", text)

    def test_applio_embedder_does_not_wget(self):
        text = (ROOT / "third_party" / "vc" / "rvc" / "lib" / "utils.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("wget.download", text)
        self.assertNotIn("huggingface.co/IAHispano", text)
        self.assertIn("Completar instalación", text)

    def test_separar_does_not_download_onnx(self):
        text = (ROOT / "uvr_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("MDX_DOWNLOAD_LINK", text)
        self.assertIn("Pulsá Completar instalación", text)

    def test_cloud_install_does_not_download_models(self):
        install = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
        link = (ROOT / "scripts" / "link_can_weights.sh").read_text(encoding="utf-8")
        text = install + "\n" + link
        self.assertNotIn("TRvlvr", text)
        self.assertNotIn("huggingface", text.lower())
        self.assertNotIn("curl ", text)
        self.assertIn("/opt/audio-separator-models", text)
        self.assertIn("No bajes nada", install)
        self.assertIn("Esta instalación no baja pesos", install)
        self.assertIn("link_can_weights.sh", install)
        self.assertIn("pack_can.sh", link)
        self.assertIn("exit 1", link)
        self.assertIn("from app import build_server, launch_app", install)
        self.assertIn("from remix import remix_to_wav", install)

    def test_standalone_weight_rsync_dereferences_can_links(self):
        text = (ROOT / "scripts" / "build_standalone.sh").read_text(encoding="utf-8")
        self.assertIn(
            'rsync -aL --delete --exclude \'__pycache__\' "$ROOT/mdx_models/"',
            text,
        )
        self.assertIn(
            'rsync -aL --delete \\\n  --exclude \'*.bak\'',
            text,
        )

    def _run_link(self, root: Path, can: Path):
        env = os.environ.copy()
        env["AUDIO_SEPARATOR_CAN"] = str(can)
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / "link_can_weights.sh"), str(root)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_link_can_fails_closed_without_weights(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "repo"
        can = Path(tmp.name) / "can"
        root.mkdir()
        can.mkdir()
        result = self._run_link(root, can)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Esta instalación no baja pesos", result.stderr)
        self.assertIn("pack_can.sh", result.stderr)
        self.assertIn("UVR-MDX-NET-Voc_FT.onnx", result.stderr)
        self.assertIn("f0G40k.pth", result.stderr)

    def test_link_can_copies_real_bytes_and_keeps_checkout(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        can = base / "can"
        root = base / "repo"
        onnx_names = (
            "UVR-MDX-NET-Voc_FT.onnx",
            "UVR_MDXNET_KARA_2.onnx",
            "Reverb_HQ_By_FoxJoy.onnx",
            "UVR-MDX-NET-Inst_HQ_4.onnx",
        )
        (can / "mdx_models").mkdir(parents=True)
        for name in onnx_names:
            (can / "mdx_models" / name).write_bytes(b"onnx-" + name.encode())
        real = base / "real-rmvpe.pt"
        real.write_bytes(b"rmvpe-bytes")
        rvc = can / "rvc"
        (rvc / "hubert_base").mkdir(parents=True)
        (rvc / "rmvpe.pt").symlink_to(real)
        (rvc / "f0G40k.pth").write_bytes(b"g40")
        (rvc / "f0D40k.pth").write_bytes(b"d40")
        (rvc / "hubert_base" / "model.safetensors").write_bytes(b"sft")
        (rvc / "hubert_base" / "config.json").write_bytes(b"cfg")
        root.mkdir()
        result = self._run_link(root, can)
        self.assertEqual(result.returncode, 0, result.stderr)
        linked = root / "mdx_models" / onnx_names[0]
        self.assertTrue(linked.is_symlink())
        self.assertEqual(linked.read_bytes(), b"onnx-" + onnx_names[0].encode())
        copied = root / "library" / "models" / "rvc" / "rmvpe.pt"
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), b"rmvpe-bytes")
        self.assertEqual(
            (root / "library" / "models" / "rvc" / "f0D40k.pth").read_bytes(),
            b"d40",
        )
        (root / "library" / "models" / "rvc" / "rmvpe.pt").write_bytes(b"keep")
        (can / "rvc" / "f0G40k.pth").write_bytes(b"replaced")
        again = self._run_link(root, can)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(copied.read_bytes(), b"keep")
        self.assertEqual(
            (root / "library" / "models" / "rvc" / "f0G40k.pth").read_bytes(),
            b"g40",
        )


class BundlePyTests(unittest.TestCase):
    def test_required_modules_listed(self):
        names = set(BUNDLE_PY)
        for name in (
            "occupancy.py",
            "chords.py",
            "train_run.py",
            "ui_status.py",
            "gpu_secrets.py",
            "desktop.py",
        ):
            self.assertIn(name, names)
        self.assertNotIn("eleven_tts.py", names)
        self.assertNotIn("tts_rvc_engine.py", names)
        self.assertNotIn("clone_engine.py", names)
        self.assertFalse((ROOT / "eleven_tts.py").exists())
        self.assertFalse((ROOT / "tts_rvc_engine.py").exists())
        self.assertFalse((ROOT / "clone_engine.py").exists())

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
