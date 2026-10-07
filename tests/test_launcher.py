import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class LauncherScriptTests(unittest.TestCase):
    def test_bootstrap_is_executable_and_creates_venvs(self):
        text = (ROOT / "scripts" / "bootstrap_macos.sh").read_text(encoding="utf-8")
        self.assertIn("requirements-macos.txt", text)
        self.assertIn("requirements-vc.txt", text)
        self.assertIn("python3.12", text)
        self.assertIn("import gradio, torch, webview", text)
        self.assertIn("import torch, transformers, librosa", text)
        mode = (ROOT / "scripts" / "bootstrap_macos.sh").stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR)

    def test_macos_launcher_bootstraps_then_starts_desktop(self):
        text = (ROOT / "scripts" / "macos_launcher.sh").read_text(encoding="utf-8")
        self.assertIn("bootstrap_macos.sh", text)
        self.assertIn("desktop.py", text)
        self.assertIn("AUDIO_SEPARATOR_VC_PYTHON", text)
        self.assertIn("env -i", text)
        self.assertIn("Resources/app/desktop.py", text)
        self.assertIn(".venv-vc", text)

    def test_ensure_vc_venv_can_pip_install(self):
        text = (ROOT / "scripts" / "ensure_vc_venv.sh").read_text(encoding="utf-8")
        self.assertIn("bootstrap_macos.sh", text)
        self.assertNotIn("No instalo pip en el Mac de destino", text)
        self.assertNotIn("este zip está incompleto", text)

    def test_build_launcher_app_writes_info_plist(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "Audio Separator.app"
        subprocess.check_call(
            ["bash", str(ROOT / "scripts" / "build_launcher_app.sh"), str(dest)]
        )
        exe = dest / "Contents" / "MacOS" / "Audio Separator"
        self.assertTrue(exe.is_file())
        self.assertTrue(os.access(exe, os.X_OK))
        plist = (dest / "Contents" / "Info.plist").read_text(encoding="utf-8")
        self.assertIn("com.lucastomasi.audioseparator", plist)
        self.assertIn("13.0", plist)
        body = exe.read_text(encoding="utf-8")
        self.assertIn("bootstrap_macos.sh", body)
        self.assertIn("desktop.py", body)

    def test_committed_app_matches_launcher_script(self):
        committed = ROOT / "Audio Separator.app" / "Contents" / "MacOS" / "Audio Separator"
        self.assertTrue(committed.is_file(), "commit Audio Separator.app next to desktop.py")
        self.assertEqual(
            committed.read_text(encoding="utf-8"),
            (ROOT / "scripts" / "macos_launcher.sh").read_text(encoding="utf-8"),
        )
        plist = (ROOT / "Audio Separator.app" / "Contents" / "Info.plist").read_text(
            encoding="utf-8"
        )
        self.assertIn("com.lucastomasi.audioseparator", plist)

    def test_thin_app_next_to_repo_resolves_desktop(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "audio-separator"
        repo.mkdir()
        (repo / "desktop.py").write_text("# stub\n", encoding="utf-8")
        dest = repo / "Audio Separator.app"
        subprocess.check_call(
            ["bash", str(ROOT / "scripts" / "build_launcher_app.sh"), str(dest)]
        )
        script = dest / "Contents" / "MacOS" / "Audio Separator"
        # Dry-run the path detection without exec'ing python.
        snippet = r"""
set -euo pipefail
HERE="%s"
if [[ "$(basename "$HERE")" == "MacOS" ]]; then
  APP_BUNDLE="$(cd "$HERE/../.." && pwd)"
  REPO="$(cd "$APP_BUNDLE/.." && pwd)"
  test -f "$REPO/desktop.py"
  echo "$REPO"
fi
""" % script.parent
        out = subprocess.check_output(["bash", "-c", snippet], text=True)
        self.assertEqual(out.strip(), str(repo.resolve()))


class DesktopWarmupTests(unittest.TestCase):
    def test_warmup_starts_vc_worker(self):
        import desktop

        src = Path(desktop.__file__).read_text(encoding="utf-8")
        self.assertIn("warmup_vc_worker", src)
        self.assertIn("ensure_vc_engine", src)
        self.assertIn("threading.Thread(target=warmup_vc_worker", src)

        with mock.patch("vc_runner.ensure_vc_engine") as ensure:
            with mock.patch("vc_runner._get_worker") as get:
                desktop.warmup_vc_worker()
        ensure.assert_called_once()
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
