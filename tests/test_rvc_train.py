import os
import sys
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

    def test_prepare_dataset_accepts_gradio_dicts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "a.wav")
        extra = os.path.join(tmp.name, "b.mp3")
        for path in (src, extra):
            with open(path, "wb") as handle:
                handle.write(b"wav")
        with mock.patch.object(rvc_train, "APP_ROOT", rvc_train.Path(tmp.name)):
            out = rvc_train._prepare_dataset(
                [{"path": src, "orig_name": "a.wav"}, extra],
                "demo",
            )
        names = sorted(p.name for p in out.glob("sample_*"))
        self.assertEqual(len(names), 2)

    def test_prepare_dataset_extracts_video(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "talk.mp4")
        with open(src, "wb") as handle:
            handle.write(b"mp4")

        def fake_extract(path, dest, sample_rate=40000, mono=True):
            with open(dest, "wb") as handle:
                handle.write(b"RIFF")
            return dest

        with mock.patch.object(rvc_train, "APP_ROOT", rvc_train.Path(tmp.name)):
            with mock.patch(
                "youtube_lib.extract_audio_from_media", side_effect=fake_extract
            ):
                out = rvc_train._prepare_dataset([src], "from_video")
        wavs = list(out.glob("sample_*.wav"))
        self.assertEqual(len(wavs), 1)
        self.assertGreater(wavs[0].stat().st_size, 0)

    def test_prepare_dataset_rejects_empty_gradio_dicts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rvc_train, "APP_ROOT", rvc_train.Path(tmp.name)):
            with self.assertRaises(ValueError) as ctx:
                rvc_train._prepare_dataset([{"orig_name": "a.wav"}], "demo")
        self.assertIn("al menos un audio", str(ctx.exception))

    def test_features_ready_needs_npy(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        exp = rvc_train.Path(tmp.name)
        self.assertFalse(rvc_train._features_ready(exp))
        feat = exp / "3_feature768"
        feat.mkdir(parents=True)
        self.assertFalse(rvc_train._features_ready(exp))
        (feat / "a.npy").write_bytes(b"x")
        self.assertTrue(rvc_train._features_ready(exp))

    def test_snapshot_copies_generator_ckpt(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = rvc_train.Path(tmp.name)
        logs = root / "logs" / "demo"
        logs.mkdir(parents=True)
        src = logs / "G_2333333.pth"
        src.write_bytes(b"x" * 100)
        with mock.patch.object(rvc_train, "RVC_ROOT", root):
            with mock.patch.object(rvc_train, "APP_ROOT", root):
                copied = rvc_train.snapshot_checkpoints("demo")
        dest = root / "library" / "train_runs" / "demo" / "ckpt" / "G_2333333.pth"
        self.assertTrue(dest.is_file())
        self.assertTrue(any(p.name == "G_2333333.pth" for p in copied))

    def test_train_running_reads_ps_command_line(self):
        fake = (
            "python -m train.train -e PELA1 -sr 40k -f0 1\n"
            "python -m train.train -e other -sr 40k\n"
        )
        with mock.patch("subprocess.check_output", return_value=fake):
            self.assertIn("PELA1", rvc_train.train_running("PELA1") or "")
            self.assertIsNone(rvc_train.train_running("nope"))

    def test_save_every_epoch_is_one(self):
        self.assertEqual(rvc_train.SAVE_EVERY_EPOCH, 1)

    def test_epochs_default_is_ten(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RVC_TRAIN_EPOCHS", None)
            self.assertEqual(rvc_train._epochs(), 10)

    def test_ensure_savee_absolute_rewrites_relative_torch_save(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = rvc_train.Path(tmp.name)
        ckpt = root / "train" / "process_ckpt.py"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text(
            "i18n = I18nAuto()\n"
            'torch.save(opt, "assets/weights/%s.pth" % name)\n',
            encoding="utf-8",
        )
        with mock.patch.object(rvc_train, "RVC_ROOT", root):
            rvc_train._ensure_savee_absolute()
        text = ckpt.read_text(encoding="utf-8")
        self.assertIn("def inference_weight_path", text)
        self.assertIn("inference_weight_path(name)", text)
        self.assertNotIn('"assets/weights/%s.pth"', text)

    def test_inference_weights_dir_is_absolute(self):
        prev = os.getcwd()
        sys_path = list(sys.path)
        os.chdir(str(rvc_train.RVC_ROOT))
        sys.path.insert(0, str(rvc_train.RVC_ROOT))
        try:
            from train.process_ckpt import inference_weights_dir

            path = inference_weights_dir()
        finally:
            os.chdir(prev)
            sys.path[:] = sys_path
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(path.endswith(os.path.join("assets", "weights")))
        self.assertTrue(os.path.isdir(path))

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

    def test_ensure_inference_weight_no_pickle_fallback(self):
        import inspect
        import types

        src = inspect.getsource(rvc_train._ensure_inference_weight)
        self.assertIn("weights_only=True", src)
        self.assertNotIn("weights_only=False", src)
        self.assertIn("de forma segura", src)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = rvc_train.Path(tmp.name)
        logs = root / "logs" / "demo"
        logs.mkdir(parents=True)
        (logs / "G_1.pth").write_bytes(b"x" * 100)
        (logs / "config.json").write_text("{}", encoding="utf-8")
        train_pkg = types.ModuleType("train")
        train_pkg.__path__ = []
        utils = types.ModuleType("train.utils")
        ckpt = types.ModuleType("train.process_ckpt")
        utils.HParams = mock.Mock(return_value=mock.Mock())
        ckpt.savee = mock.Mock()
        with mock.patch.dict(
            sys.modules,
            {
                "train": train_pkg,
                "train.utils": utils,
                "train.process_ckpt": ckpt,
            },
        ):
            with mock.patch.object(rvc_train, "RVC_ROOT", root):
                with mock.patch("rvc_engine._scan_model"):
                    with mock.patch(
                        "torch.load", side_effect=RuntimeError("unsafe")
                    ) as loader:
                        with self.assertRaises(ValueError) as ctx:
                            rvc_train._ensure_inference_weight("demo")
        self.assertIn("segura", str(ctx.exception))
        self.assertTrue(
            all(call.kwargs.get("weights_only") is True for call in loader.call_args_list)
        )

    def test_replace_with_link_does_not_copy_bytes(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = rvc_train.Path(tmp.name) / "src.bin"
        dest = rvc_train.Path(tmp.name) / "dest.bin"
        src.write_bytes(b"payload")
        rvc_train._replace_with_link(src, dest)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_bytes(), b"payload")

    def test_library_root_uses_data_dir_when_unpatched(self):
        with mock.patch(
            "train_run.library_root",
            return_value=rvc_train.Path("/tmp/as-data-lib"),
        ):
            root = rvc_train._library_root()
        self.assertEqual(root, rvc_train.Path("/tmp/as-data-lib"))

    def test_train_voice_spawns_supervisor(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "a.wav")
        with open(src, "wb") as handle:
            handle.write(b"wav")
        fake_proc = mock.Mock()
        with mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_DATA": tmp.name}):
            with mock.patch.object(rvc_train, "require_rvc_webui"):
                with mock.patch.object(rvc_train, "require_train_assets"):
                    with mock.patch.object(rvc_train, "train_running", return_value=None):
                        with mock.patch("occupancy.snapshot", return_value=None):
                            with mock.patch(
                                "train_run.spawn_supervisor", return_value=fake_proc
                            ) as spawn:
                                with mock.patch(
                                    "train_run.wait_supervisor",
                                    return_value=("/tmp/x.pth", None),
                                ):
                                    out = rvc_train.train_voice(
                                        "demo", [src], epochs=1
                                    )
                                    self.assertEqual(out[0], "/tmp/x.pth")
                                    spawn.assert_called_once()

    def test_train_voice_blocked_when_occupied(self):
        import occupancy

        occ = occupancy.Occupancy("convert")
        with mock.patch.object(rvc_train, "require_rvc_webui"):
            with mock.patch.object(rvc_train, "require_train_assets"):
                with mock.patch.object(rvc_train, "train_running", return_value=None):
                    with mock.patch("occupancy.snapshot", return_value=occ):
                        with self.assertRaises(ValueError) as ctx:
                            rvc_train.train_voice("demo", ["a.wav"], epochs=1)
        self.assertIn("conversión", str(ctx.exception))

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
