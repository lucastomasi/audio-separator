import inspect
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rvc_train
import train_run


class TrainRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.patch_data = mock.patch("app_env.data_dir", return_value=str(self.root))
        self.patch_data.start()
        self.addCleanup(self.patch_data.stop)

    def test_paths_use_data_dir(self):
        log = train_run.log_path("demo")
        self.assertTrue(str(log).startswith(str(self.root)))
        self.assertTrue(str(log).endswith(os.path.join("Voces", "demo", "trabajo", "train.log")))
        data = train_run.train_data_dir("demo")
        self.assertIn(os.path.join("Voces", "demo", "entrada"), str(data))
        self.assertFalse((self.root / "library").exists())

    def test_write_and_read_published(self):
        train_run.write_published("demo", ok=True, pth="/tmp/a.pth", index=None)
        got = train_run.read_published("demo")
        self.assertTrue(got["ok"])
        self.assertEqual(got["pth"], "/tmp/a.pth")

    def test_spawn_supervisor_is_new_session(self):
        job = self.root / "job.json"
        job.write_text("{}", encoding="utf-8")
        fake = mock.Mock()
        with mock.patch("subprocess.Popen", return_value=fake) as popen:
            proc = train_run.spawn_supervisor(job)
        self.assertIs(proc, fake)
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs.get("start_new_session"))
        self.assertNotEqual(kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertNotEqual(kwargs.get("stderr"), subprocess.DEVNULL)
        self.assertNotIn("DEVNULL", inspect.getsource(train_run.spawn_supervisor))
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[1:3], ["-m", "train_run"])
        self.assertEqual(cmd[3], "supervise")

    def test_execute_train_uses_applio_scripts(self):
        src = inspect.getsource(rvc_train.execute_train)
        src += inspect.getsource(rvc_train.finish_train_publish)
        self.assertIn('"preprocess.py"', src)
        self.assertIn('"extract.py"', src)
        self.assertIn('"train.py"', src)
        self.assertIn('"extract_index.py"', src)
        self.assertIn('"rvc"', src)
        self.assertNotIn("train.dataset.extract_f0", src)
        self.assertNotIn("train.dataset.extract_hubert_feature", src)
        self.assertIn("_TRAIN_OK_CODES", inspect.getsource(rvc_train._run))
        self.assertIn(2333333, rvc_train._TRAIN_OK_CODES)

    def test_export_weight_prefers_vc_python(self):
        src = inspect.getsource(train_run.export_weight)
        self.assertIn("_engine_python", src)
        self.assertIn("PYTHONPATH", src)

    def test_execute_train_does_not_import_torch(self):
        src = inspect.getsource(rvc_train.execute_train)
        src += inspect.getsource(rvc_train.finish_train_publish)
        self.assertNotIn("import torch", src)
        self.assertIn("finish_train_publish", inspect.getsource(rvc_train.execute_train))

    def test_run_does_not_pipe_stdout(self):
        src = inspect.getsource(rvc_train._run)
        self.assertNotIn("subprocess.PIPE", src)
        self.assertIn("stdout=log", src)

    def test_run_sets_intel_mac_train_env(self):
        src = inspect.getsource(rvc_train._run)
        self.assertIn("KMP_DUPLICATE_LIB_OK", src)
        self.assertIn("USE_LIBUV", src)
        self.assertIn("RVC_AUDIO_FORCE_CPU", src)
        self.assertIn("OMP_NUM_THREADS", src)
        self.assertIn("ffmpeg_binary", src)

    def test_supervisor_inherits_openmp_env(self):
        src = inspect.getsource(train_run.spawn_supervisor)
        self.assertIn("KMP_DUPLICATE_LIB_OK", src)
        self.assertIn("USE_LIBUV", src)

    def test_latest_job_is_the_newest(self):
        train_run.write_job("vieja", ["/tmp/a.wav"], 5)
        train_run.write_job("nueva", ["/tmp/b.wav"], 12)
        old = train_run.job_path("vieja")
        new = train_run.job_path("nueva")
        os.utime(old, (1_000, 1_000))
        os.utime(new, (2_000, 2_000))
        job = train_run.latest_job()
        self.assertEqual(job["exp"], "nueva")
        self.assertEqual(job["epochs"], 12)

    def test_boot_status_names_published_without_training(self):
        train_run.write_published("demo", ok=True, pth="/tmp/Voces/demo/demo.pth")
        with mock.patch("occupancy.snapshot", return_value=None):
            with mock.patch("rvc_train.train_voice") as train:
                text = train_run.boot_status()
        train.assert_not_called()
        self.assertEqual(text, "Modelo listo: demo.pth")

    def test_boot_status_shows_live_log_line(self):
        import occupancy

        log = train_run.log_path("demo")
        log.write_text("Training epoch: 3\n", encoding="utf-8")
        occ = occupancy.Occupancy(occupancy.HOLD_TRAIN, exp="demo")
        with mock.patch("occupancy.snapshot", return_value=occ):
            text = train_run.boot_status()
        self.assertIn("Training epoch: 3", text)
