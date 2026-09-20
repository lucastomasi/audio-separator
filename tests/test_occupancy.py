import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import occupancy


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.patch_data = mock.patch(
            "app_env.data_dir", return_value=self.root
        )
        self.patch_data.start()
        self.addCleanup(self.patch_data.stop)

    def test_acquire_and_snapshot(self):
        with mock.patch.object(occupancy, "_ps_commands", return_value=""):
            occupancy.acquire(occupancy.HOLD_CONVERT)
            snap = occupancy.snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.holder, occupancy.HOLD_CONVERT)
        occupancy.release(occupancy.HOLD_CONVERT)
        with mock.patch.object(occupancy, "_ps_commands", return_value=""):
            self.assertIsNone(occupancy.snapshot())

    def test_convert_blocked_while_train_ps(self):
        ps = "python -m train.train -e voz_20260919 -sr 40k -f0 1\n"
        with mock.patch.object(occupancy, "_ps_commands", return_value=ps):
            with self.assertRaises(ValueError) as ctx:
                occupancy.acquire(occupancy.HOLD_CONVERT)
        self.assertIn("entrenamiento", str(ctx.exception))
        self.assertIn("Convertir", str(ctx.exception))

    def test_stale_lock_cleared_when_pid_dead(self):
        path = occupancy.lock_path()
        path.write_text(
            '{"holder": "train", "exp": "x", "pid": 99999999, "started": 1}',
            encoding="utf-8",
        )
        with mock.patch.object(occupancy, "_ps_commands", return_value=""):
            with mock.patch.object(occupancy, "_pid_alive", return_value=False):
                self.assertIsNone(occupancy.snapshot())
                self.assertFalse(path.is_file())

    def test_lock_path_under_data_dir(self):
        path = occupancy.lock_path()
        self.assertTrue(str(path).startswith(self.root))
        self.assertEqual(path.name, "occupancy.json")
