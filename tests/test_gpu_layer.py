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
        tts_upd = mock.Mock()
        with mock.patch("rvc_train.train_voice", return_value=("/tmp/m.pth", None)):
            with mock.patch(
                "app_jobs.refresh_library_ui", return_value=(rvc_upd, tts_upd)
            ):
                with mock.patch("library.dropdown_choices", return_value=[]):
                    with mock.patch("library.list_rvc_voices", return_value=[]):
                        with mock.patch("gpu_secrets.save", saver):
                            with mock.patch("runpod_train.start_train_pod", start):
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


class SpeakWithRvcSourceTests(unittest.TestCase):
    def _run_speak(self, *, eleven_on):
        import tts_rvc_engine

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path(tmp.name)
        out = work / "out.wav"
        out.write_bytes(b"x" * 20)

        def fake_to_wav(_src, dest):
            dest.write_bytes(b"RIFF")
            return dest

        with mock.patch.object(tts_rvc_engine, "WORK", work):
            with mock.patch.object(
                tts_rvc_engine, "resolve_voice_model", return_value=("m.pth", None)
            ):
                with mock.patch.object(
                    tts_rvc_engine, "convert_voice", return_value=str(out)
                ):
                    with mock.patch.object(
                        tts_rvc_engine, "_to_wav", side_effect=fake_to_wav
                    ):
                        with mock.patch("eleven_tts.available", return_value=eleven_on):
                            with mock.patch("eleven_tts.speak_to_mp3"):
                                with mock.patch.object(
                                    tts_rvc_engine, "_edge_tts_to_file"
                                ) as edge:
                                    path, src = tts_rvc_engine.speak_with_rvc(
                                        "hola", "m.pth"
                                    )
        return path, src, edge

    def test_reports_elevenlabs_when_used(self):
        _path, src, edge = self._run_speak(eleven_on=True)
        self.assertEqual(src, "ElevenLabs")
        edge.assert_not_called()

    def test_reports_edge_when_eleven_unavailable(self):
        _path, src, edge = self._run_speak(eleven_on=False)
        self.assertTrue(src.startswith("Edge"), src)
        edge.assert_called()


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
