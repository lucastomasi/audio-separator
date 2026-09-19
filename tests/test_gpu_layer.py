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
        gpu_secrets.save({"ELEVENLABS_API_KEY": "sk-test"})
        self.assertEqual(gpu_secrets.get("ELEVENLABS_API_KEY"), "sk-test")
        path = Path(gpu_secrets.secrets_path())
        self.assertTrue(path.is_file())
        self.assertIn("sk-test", path.read_text(encoding="utf-8"))


class RunpodTrainTests(unittest.TestCase):
    def test_estimate_grows_with_files(self):
        small = runpod_train.estimate_cpu_minutes(3, 10)
        big = runpod_train.estimate_cpu_minutes(300, 15)
        self.assertGreater(big, small)
        self.assertIn("Mac", runpod_train.estimate_copy(3, 10))
        self.assertIn("45", runpod_train.estimate_copy(3, 10))

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


class ElevenTtsTests(unittest.TestCase):
    def test_available_false_without_key(self):
        with mock.patch("eleven_tts.get", return_value=None):
            import eleven_tts

            self.assertFalse(eleven_tts.available())

    def test_speak_writes_bytes(self):
        import eleven_tts

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = Path(tmp.name) / "a.mp3"
        resp = mock.Mock(status_code=200, content=b"x" * 200)
        with mock.patch("eleven_tts.get", return_value="ek"):
            with mock.patch("eleven_tts.requests.post", return_value=resp):
                eleven_tts.speak_to_mp3("hola", dest)
        self.assertGreater(dest.stat().st_size, 100)

    def test_quota_maps_to_valueerror(self):
        import eleven_tts

        resp = mock.Mock(status_code=429, content=b"")
        with mock.patch("eleven_tts.get", return_value="ek"):
            with mock.patch("eleven_tts.requests.post", return_value=resp):
                with self.assertRaises(ValueError) as ctx:
                    eleven_tts.speak_to_mp3("hola", Path("/tmp/x.mp3"))
        self.assertIn("Cupo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
