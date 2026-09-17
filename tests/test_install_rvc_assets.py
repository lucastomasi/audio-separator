import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install_rvc_assets


class InstallRvcAssetsTests(unittest.TestCase):
    def test_missing_when_empty(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "hubert_base").mkdir()
        with mock.patch.object(install_rvc_assets, "_root", return_value=root):
            missing = install_rvc_assets.missing_rvc_assets()
            self.assertTrue(any("hubert_base.pt" in m for m in missing))
            self.assertTrue(any("rmvpe.pt" in m for m in missing))
            self.assertFalse(install_rvc_assets.rvc_assets_ready())

    def test_link_or_copy_reuses_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "src.bin"
        dest = Path(tmp.name) / "dest.bin"
        src.write_bytes(b"hello-cache")
        install_rvc_assets._link_or_copy(src, dest)
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_bytes(), b"hello-cache")

    def test_ready_when_present(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        hubert = root / "hubert_base"
        hubert.mkdir()
        (root / "hubert_base.pt").write_bytes(b"x" * 10)
        (root / "rmvpe.pt").write_bytes(b"x" * 10)
        (root / "f0G40k.pth").write_bytes(b"x" * 10)
        (root / "f0D40k.pth").write_bytes(b"x" * 10)
        (hubert / "config.json").write_text("{}")
        (hubert / "preprocessor_config.json").write_text("{}")
        (hubert / "model.safetensors").write_bytes(b"x" * 10)
        with mock.patch.object(install_rvc_assets, "_root", return_value=root):
            with mock.patch.object(
                install_rvc_assets, "_hubert_pt_is_fairseq", return_value=True
            ):
                self.assertEqual(install_rvc_assets.missing_rvc_assets(), [])
                self.assertTrue(install_rvc_assets.rvc_assets_ready())


if __name__ == "__main__":
    unittest.main()
