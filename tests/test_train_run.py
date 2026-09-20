import inspect
import json
import os
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
        self.assertTrue(str(log).endswith(os.path.join("train_runs", "demo", "train.log")))
        data = train_run.train_data_dir("demo")
        self.assertIn("train_data", str(data))

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
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[1:3], ["-m", "train_run"])
        self.assertEqual(cmd[3], "supervise")

    def test_execute_train_does_not_import_torch(self):
        src = inspect.getsource(rvc_train.execute_train)
        self.assertNotIn("import torch", src)
        self.assertIn("export_weight", src)

    def test_run_does_not_pipe_stdout(self):
        src = inspect.getsource(rvc_train._run)
        self.assertNotIn("subprocess.PIPE", src)
        self.assertIn("stdout=log", src)
