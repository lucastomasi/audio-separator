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
        produced = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, produced):
            with open(path, "wb") as handle:
                handle.write(b"RIFF" + b"\x00" * 80)

        class FakeProc:
            def __init__(self):
                self.stdin_lines = []
                self.poll_n = None

            def poll(self):
                return None

            class _In:
                def __init__(self, outer):
                    self.outer = outer

                def write(self, data):
                    self.outer.stdin_lines.append(data)

                def flush(self):
                    pass

            @property
            def stdin(self):
                return self._In(self)

            class _Out:
                def readline(self):
                    return json.dumps({"ok": True, "path": produced}) + "\n"

            @property
            def stdout(self):
                return self._Out()

        fake = FakeProc()
        with mock.patch.object(vc_runner, "_get_worker", return_value=fake):
            path = vc_runner.run_vc_infer(audio, model, pitch=1, index_rate=0.7)
        self.assertEqual(path, os.path.abspath(produced))
        job = json.loads(fake.stdin_lines[0])
        self.assertEqual(job["cmd"], "infer")
        self.assertEqual(job["pitch"], 1)
        self.assertEqual(job["index_rate"], 0.7)
        self.assertTrue(job["split_audio"])


if __name__ == "__main__":
    unittest.main()
