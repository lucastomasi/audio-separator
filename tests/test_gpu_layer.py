import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gpu_secrets
import runpod_train


class GpuSecretsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.patch = mock.patch("gpu_secrets.data_dir", return_value=tmp.name)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_roundtrip(self):
        gpu_secrets.save({"GEMINI_API_KEY": "sk-test"})
        self.assertEqual(gpu_secrets.get("GEMINI_API_KEY"), "sk-test")
        path = Path(gpu_secrets.secrets_path())
        self.assertTrue(path.is_file())
        self.assertIn("sk-test", path.read_text(encoding="utf-8"))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class RunpodTrainTests(unittest.TestCase):
    def test_estimate_grows_with_files(self):
        small = runpod_train.estimate_cpu_minutes(3, 10)
        big = runpod_train.estimate_cpu_minutes(300, 15)
        self.assertGreater(big, small)
        copy = runpod_train.estimate_copy(3, 10)
        self.assertIn("Mac", copy)
        self.assertNotIn("GPU", copy)
        self.assertNotIn("45", copy)
        self.assertNotIn("alquiler", copy)

    def test_start_without_image_raises(self):
        with mock.patch.object(runpod_train, "get", return_value="key"):
            with mock.patch.object(runpod_train, "check_key", return_value="key"):
                with mock.patch.dict("os.environ", {}, clear=False):
                    os.environ.pop("RUNPOD_TRAIN_IMAGE", None)
                    with self.assertRaises(ValueError) as ctx:
                        runpod_train.start_train_pod()
        self.assertIn("Mac", str(ctx.exception))

    def test_stop_pod_clears_id(self):
        http = mock.Mock()
        http.delete.return_value = mock.Mock(status_code=200)
        with mock.patch.object(runpod_train, "get", side_effect=lambda k: "x"):
            with mock.patch.object(runpod_train, "save") as saver:
                runpod_train.stop_pod("pod1", session=http)
        http.delete.assert_called()
        saver.assert_called_with({"RUNPOD_POD_ID": ""})


class TrainRvcJobGpuTests(unittest.TestCase):
    def test_hire_gpu_saves_key_without_starting_pod(self):
        import app_jobs

        saver = mock.Mock()
        start = mock.Mock()
        rvc_upd = mock.Mock()
        with mock.patch("rvc_train.train_voice", return_value=("/tmp/m.pth", None)):
            with mock.patch("train_prep.assert_channel_matches"):
                with mock.patch(
                    "train_prep.prepare_for_train",
                    side_effect=lambda files, progress=None: files,
                ):
                    with mock.patch(
                        "app_jobs.refresh_library_ui", return_value=rvc_upd
                    ):
                        with mock.patch("library.dropdown_choices", return_value=[]):
                            with mock.patch("library.list_rvc_voices", return_value=[]):
                                with mock.patch("gpu_secrets.save", saver):
                                    with mock.patch(
                                        "runpod_train.start_train_pod", start
                                    ):
                                        app_jobs.train_rvc_job(
                                            "demo",
                                            ["/tmp/a.wav"],
                                            epochs=10,
                                            hire_gpu=True,
                                            runpod_key="rp-test",
                                            progress=mock.Mock(),
                                        )
        start.assert_not_called()
        saver.assert_called_with({"RUNPOD_API_KEY": "rp-test"})


if __name__ == "__main__":
    unittest.main()
