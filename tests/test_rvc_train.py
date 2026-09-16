import os
import tempfile
import unittest
from unittest import mock

import rvc_train


class RvcTrainTests(unittest.TestCase):
    def test_missing_webui(self):
        with mock.patch.object(
            rvc_train, "RVC_ROOT", rvc_train.Path("/no/such/RVC-WebUI")
        ):
            with self.assertRaises(ValueError):
                rvc_train.require_rvc_webui()

    def test_missing_assets(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake_root = os.path.join(tmp.name, "rvc")
        os.makedirs(fake_root, exist_ok=True)
        with mock.patch("library.PATHS", {**__import__("library").PATHS, "rvc": fake_root}):
            with mock.patch.object(rvc_train, "rvc_support_dir", return_value=fake_root):
                with self.assertRaises(ValueError) as ctx:
                    rvc_train.require_train_assets()
        self.assertIn("Faltan pesos para entrenar", str(ctx.exception))
        self.assertIn("hubert_base/", str(ctx.exception))

    def test_classic_hubert_pt_not_enough_for_train(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake_root = os.path.join(tmp.name, "rvc")
        os.makedirs(fake_root, exist_ok=True)
        for name in ("hubert_base.pt", "rmvpe.pt", "f0G40k.pth", "f0D40k.pth"):
            with open(os.path.join(fake_root, name), "wb") as handle:
                handle.write(b"x" * 10)
        with mock.patch.object(rvc_train, "rvc_support_dir", return_value=fake_root):
            with self.assertRaises(ValueError) as ctx:
                rvc_train.require_train_assets()
        self.assertIn("Transformers", str(ctx.exception))

    def test_config_template_40k_uses_v1(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        exp = rvc_train.Path(tmp.name) / "exp"
        for name in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"):
            (exp / name).mkdir(parents=True)
        (exp / "0_gt_wavs" / "a.wav").write_bytes(b"x")
        (exp / "3_feature768" / "a.npy").write_bytes(b"x")
        (exp / "2a_f0" / "a.wav.npy").write_bytes(b"x")
        (exp / "2b-f0nsf" / "a.wav.npy").write_bytes(b"x")
        rvc_train._write_filelist_and_config(exp)
        config = (exp / "config.json").read_text(encoding="utf-8")
        self.assertIn("40000", config)

    def test_prepare_dataset_copies_files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "a.wav")
        with open(src, "wb") as handle:
            handle.write(b"wav")
        with mock.patch.object(rvc_train, "APP_ROOT", rvc_train.Path(tmp.name)):
            out = rvc_train._prepare_dataset([src], "demo")
        self.assertTrue(out.is_dir())
        self.assertEqual(len(list(out.glob("sample_*"))), 1)

    def test_find_small_weight_skips_G_checkpoints(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = rvc_train.Path(tmp.name)
        weights = root / "assets" / "weights"
        logs = root / "logs" / "demo"
        weights.mkdir(parents=True)
        logs.mkdir(parents=True)
        big = logs / "G_2333333.pth"
        big.write_bytes(b"x" * (400 * 1024 * 1024))
        small = weights / "demo.pth"
        small.write_bytes(b"y" * (55 * 1024 * 1024))
        with mock.patch.object(rvc_train, "RVC_ROOT", root):
            found = rvc_train._find_small_weight("demo")
        self.assertEqual(found, small)

    def test_find_index_prefers_added(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = rvc_train.Path(tmp.name)
        logs = root / "logs" / "demo"
        logs.mkdir(parents=True)
        trained = logs / "trained_IVF_demo.index"
        added = logs / "added_IVF_demo.index"
        trained.write_bytes(b"t" * 1000)
        added.write_bytes(b"a" * 50000)
        with mock.patch.object(rvc_train, "RVC_ROOT", root):
            found = rvc_train._find_index("demo")
        self.assertEqual(found, added)


if __name__ == "__main__":
    unittest.main()
