import os
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

    def test_hf_hub_download_is_hub1_compatible(self):
        import inspect

        src = inspect.getsource(install_rvc_assets._download)
        self.assertNotIn("resume_download", src)
        self.assertIn("hf_hub_download", src)
        self.assertIn("_allow_hub_download", src)

    def test_download_works_when_desktop_sets_hub_offline(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "rmvpe.pt"
        cached = Path(tmp.name) / "cached.pt"
        cached.write_bytes(b"weight")
        seen = {}

        def fake_hub(**_kwargs):
            import huggingface_hub.constants as hub_constants

            seen["offline_flag"] = hub_constants.HF_HUB_OFFLINE
            seen["env"] = os.environ.get("HF_HUB_OFFLINE")
            return str(cached)

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.addCleanup(lambda: os.environ.pop("HF_HUB_OFFLINE", None))
        self.addCleanup(lambda: os.environ.pop("TRANSFORMERS_OFFLINE", None))

        import huggingface_hub.constants as hub_constants

        previous = hub_constants.HF_HUB_OFFLINE
        hub_constants.HF_HUB_OFFLINE = True
        self.addCleanup(lambda: setattr(hub_constants, "HF_HUB_OFFLINE", previous))

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_hub):
            install_rvc_assets._download("rmvpe.pt", dest, Path(tmp.name) / "cache")

        self.assertFalse(seen["offline_flag"])
        self.assertIsNone(seen["env"])
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_bytes(), b"weight")
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertTrue(hub_constants.HF_HUB_OFFLINE)

    def test_allow_hub_download_restores_offline_after_error(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.addCleanup(lambda: os.environ.pop("HF_HUB_OFFLINE", None))
        import huggingface_hub.constants as hub_constants

        previous = hub_constants.HF_HUB_OFFLINE
        hub_constants.HF_HUB_OFFLINE = True
        self.addCleanup(lambda: setattr(hub_constants, "HF_HUB_OFFLINE", previous))

        with self.assertRaises(RuntimeError):
            with install_rvc_assets._allow_hub_download():
                raise RuntimeError("hub down")

        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertTrue(hub_constants.HF_HUB_OFFLINE)

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

    def test_ensure_rvc_webui_errors_when_missing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "vc"
        with mock.patch.object(install_rvc_assets, "rvc_webui_root", return_value=dest):
            with self.assertRaises(RuntimeError) as ctx:
                install_rvc_assets.ensure_rvc_webui()
        self.assertIn("third_party/vc", str(ctx.exception))

    def test_ensure_rvc_webui_skips_if_present(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "vc"
        (dest / "rvc" / "train").mkdir(parents=True)
        (dest / "rvc" / "train" / "train.py").write_text("# train\n")
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
