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

    def test_torch_load_requires_weights_only(self):
        with mock.patch("torch.load", return_value={"model": 1}) as loader:
            install_rvc_assets._torch_load(Path("/tmp/x.pt"))
        self.assertTrue(loader.call_args.kwargs.get("weights_only") is True)

    def test_hubert_fairseq_does_not_unpickle(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "hubert_base.pt"
        path.write_bytes(b"x" * 80_000_001)
        with mock.patch.object(
            install_rvc_assets, "_torch_load", side_effect=RuntimeError("weights")
        ) as loader:
            self.assertTrue(install_rvc_assets._hubert_pt_is_fairseq(path))
        loader.assert_called_once()

    def test_ensure_rvc_webui_errors_when_git_fails(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "RVC-WebUI"
        fake = mock.Mock(returncode=1, stderr="network", stdout="")
        with mock.patch.object(install_rvc_assets, "rvc_webui_root", return_value=dest):
            with mock.patch("subprocess.run", return_value=fake):
                with self.assertRaises(RuntimeError) as ctx:
                    install_rvc_assets.ensure_rvc_webui()
        self.assertIn("Entrenar queda apagado", str(ctx.exception))

    def test_ensure_rvc_webui_skips_if_present(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "RVC-WebUI"
        (dest / "train").mkdir(parents=True)
        (dest / "train" / "train.py").write_text("# train\n")
        with mock.patch.object(install_rvc_assets, "rvc_webui_root", return_value=dest):
            self.assertIsNone(install_rvc_assets.ensure_rvc_webui())


if __name__ == "__main__":
    unittest.main()
