"""Bound ownership around NVML queries without hiding genuinely foreign work."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from remote_run import foreign_gpu_pids


class ProcessOwnershipTests(unittest.TestCase):
    def snapshot(self, pids):
        return {"gpu_processes": {"returncode": 0, "stdout": "".join(f"{pid}, python, 100 MiB\n" for pid in pids), "stderr": ""}}

    def test_exiting_owned_child_during_query_is_not_foreign(self):
        self.assertEqual(foreign_gpu_pids(self.snapshot([83758]), {1, 83758}, {1}), [])

    def test_new_owned_child_and_foreign_pid_remain_distinct(self):
        self.assertEqual(foreign_gpu_pids(self.snapshot([42, 99]), {1}, {1, 42}), [99])
        self.assertEqual(foreign_gpu_pids(self.snapshot([99]), {1}, {1}), [99])

    def test_missing_gpu_query_is_not_silently_accepted(self):
        invalid = self.snapshot([])
        invalid["gpu_processes"]["returncode"] = 1
        invalid["gpu_processes"]["stderr"] = "query failed"
        with self.assertRaises(RuntimeError):
            foreign_gpu_pids(invalid, {1}, {1})


if __name__ == "__main__":
    unittest.main()
