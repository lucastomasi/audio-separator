import json
import os
import tempfile
import unittest
from unittest import mock

import vc_runner


class VcRunnerWorkerTests(unittest.TestCase):
    def test_run_vc_infer_uses_worker_json(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "in.wav")
        model = os.path.join(tmp.name, "m.pth")
        for path in (audio, model):
            with open(path, "wb") as handle:
                handle.write(b"RIFF" + b"\x00" * 80)

        class FakeProc:
            def __init__(self):
                self.stdin_lines = []
                self._reply_path = None

            def poll(self):
                return None

            class _In:
                def __init__(self, outer):
                    self.outer = outer

                def write(self, data):
                    self.outer.stdin_lines.append(data)
                    job = json.loads(data)
                    out = job["output"]
                    with open(out, "wb") as handle:
                        handle.write(b"RIFF" + b"\x00" * 80)
                    self.outer._reply_path = out

                def flush(self):
                    pass

            @property
            def stdin(self):
                return self._In(self)

            class _Out:
                def __init__(self, outer):
                    self.outer = outer

                def readline(self):
                    return (
                        json.dumps({"ok": True, "path": self.outer._reply_path}) + "\n"
                    )

            @property
            def stdout(self):
                return self._Out(self)

        fake = FakeProc()
        with mock.patch("app_env.data_dir", return_value=tmp.name):
            with mock.patch.object(vc_runner, "_get_worker", return_value=fake):
                path = vc_runner.run_vc_infer(audio, model, pitch=1, index_rate=0.7)
        self.assertTrue(path.startswith(os.path.join(tmp.name, "rvc_output")))
        self.assertTrue(os.path.isfile(path))
        job = json.loads(fake.stdin_lines[0])
        self.assertEqual(job["cmd"], "infer")
        self.assertEqual(job["output"], path)
        self.assertEqual(job["pitch"], 1)
        self.assertEqual(job["index_rate"], 0.7)
        self.assertTrue(job["split_audio"])

    def test_infer_output_is_exportable(self):
        import exports

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch("app_env.data_dir", return_value=tmp.name):
            path = vc_runner._infer_output_path()
        self.assertTrue(path.startswith(os.path.join(tmp.name, "rvc_output")))
        with open(path, "wb") as handle:
            handle.write(b"wav")
        with mock.patch.object(exports, "data_dir", return_value=tmp.name):
            with mock.patch.object(
                exports,
                "exports_dir",
                return_value=os.path.join(tmp.name, "Downloads"),
            ):
                self.assertTrue(exports.is_exportable(path))

    def test_ensure_vc_engine_writes_log_when_python_missing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = os.path.join(tmp.name, "vc_infer.log")
        missing_py = os.path.join(tmp.name, "no-python")
        paths = {
            "app": tmp.name,
            "log": log,
            "pid": os.path.join(tmp.name, "vc_worker.pid"),
            "py": missing_py,
            "root": tmp.name,
        }
        with mock.patch.object(vc_runner, "_paths", return_value=paths):
            with self.assertRaises(ValueError) as ctx:
                vc_runner.ensure_vc_engine()
        self.assertTrue(os.path.isfile(log))
        with open(log, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Completar instalación", str(ctx.exception))
        self.assertIn("python", text.lower())

    def test_spawn_worker_writes_log_when_script_missing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = os.path.join(tmp.name, "vc_infer.log")
        py = os.path.join(tmp.name, "python")
        with open(py, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(py, 0o755)
        paths = {
            "app": tmp.name,
            "log": log,
            "pid": os.path.join(tmp.name, "vc_worker.pid"),
            "py": py,
            "root": os.path.join(tmp.name, "vc"),
        }
        os.makedirs(paths["root"], exist_ok=True)
        with mock.patch.object(vc_runner, "_paths", return_value=paths):
            with self.assertRaises(ValueError) as ctx:
                vc_runner._spawn_worker()
        self.assertTrue(os.path.isfile(log))
        self.assertIn("Completar instalación", str(ctx.exception))
        with open(log, encoding="utf-8") as handle:
            self.assertIn("infer_worker", handle.read())


if __name__ == "__main__":
    unittest.main()
