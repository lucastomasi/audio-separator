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

    def test_link_or_copy_resolves_relative_blob_symlink(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        blobs = Path(tmp.name) / "blobs"
        blobs.mkdir()
        payload = blobs / "abc"
        payload.write_bytes(b"real-weight")
        snapshot = Path(tmp.name) / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        cached = snapshot / "rmvpe.pt"
        cached.symlink_to("../../blobs/abc")
        dest = Path(tmp.name) / "library" / "rmvpe.pt"
        install_rvc_assets._link_or_copy(cached, dest)
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_bytes(), b"real-weight")

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

    def test_ensure_safetensors_allows_full_bin(self):
        with mock.patch.object(
            install_rvc_assets, "_torch_load", return_value={"w": mock.Mock()}
        ) as loader:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            hubert = Path(tmp.name)
            (hubert / "pytorch_model.bin").write_bytes(b"x")
            with mock.patch("safetensors.torch.save_file"):
                install_rvc_assets._ensure_safetensors(hubert)
        self.assertEqual(loader.call_args.kwargs.get("weights_only"), False)

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

    def test_missing_uvr_when_empty(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        with mock.patch.object(install_rvc_assets, "uvr_models_dir", return_value=root):
            missing = install_rvc_assets.missing_uvr_assets()
        self.assertEqual(set(missing), set(install_rvc_assets.UVR_MODELS))
        self.assertFalse(install_rvc_assets.uvr_assets_ready())

    def test_install_uvr_uses_can_and_skips_present(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "mdx"
        can = Path(tmp.name) / "can" / "mdx_models"
        dest.mkdir()
        can.mkdir(parents=True)
        present = install_rvc_assets.UVR_MODELS[0]
        (dest / present).write_bytes(b"already")
        for name in install_rvc_assets.UVR_MODELS[1:]:
            (can / name).write_bytes(b"from-can-" + name.encode())
        logs = []
        with mock.patch.object(install_rvc_assets, "uvr_models_dir", return_value=dest):
            with mock.patch.object(
                install_rvc_assets, "can_root", return_value=can.parent
            ):
                with mock.patch.object(
                    install_rvc_assets, "_http_download", side_effect=AssertionError("no net")
                ):
                    written = install_rvc_assets.install_uvr_assets(log=logs.append)
        self.assertEqual(set(written), set(install_rvc_assets.UVR_MODELS[1:]))
        self.assertEqual((dest / present).read_bytes(), b"already")
        for name in install_rvc_assets.UVR_MODELS[1:]:
            self.assertTrue((dest / name).is_file())
            self.assertGreater((dest / name).stat().st_size, 0)

    def test_first_install_only_fills_gaps(self):
        logs = []
        with mock.patch.object(
            install_rvc_assets, "missing_uvr_assets", return_value=[]
        ):
            with mock.patch.object(
                install_rvc_assets, "missing_rvc_assets", return_value=["rmvpe.pt"]
            ):
                with mock.patch.object(
                    install_rvc_assets, "install_uvr_assets"
                ) as uvr:
                    with mock.patch.object(
                        install_rvc_assets,
                        "install_rvc_assets",
                        return_value=["rmvpe.pt"],
                    ) as rvc:
                        written = install_rvc_assets.install_first_time_assets(
                            log=logs.append
                        )
        uvr.assert_not_called()
        rvc.assert_called_once()
        self.assertEqual(written, ["rmvpe.pt"])
        self.assertTrue(any("OK modelos UVR" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
