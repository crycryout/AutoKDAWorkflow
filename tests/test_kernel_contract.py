"""Check task boundaries and manifest/compile numerical contracts without a GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thor_kda import build, semantics


class KernelContractTests(unittest.TestCase):
    def manifest(self, directory, **changes):
        data = json.loads((ROOT / 'templates/kernel-candidate.json').read_text())
        data.update(changes)
        path = directory / 'candidate.json'
        path.write_text(json.dumps(data))
        return path

    def test_manifest_prevents_filename_escape_and_bias_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'filename-safe'):
                semantics.load(self.manifest(directory, name='../outside'))
            with self.assertRaisesRegex(ValueError, 'BIAS'):
                semantics.load(self.manifest(directory, bias=True, defines={'BIAS': 0}))

    def test_compile_keeps_arch_lineinfo_and_declared_defines(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(build, 'assert_root'):
                cmd = build.cutlass_command(directory / 'kernel.cu', directory / 'kernel.so', directory / 'cutlass', {'BIAS': 0})
            self.assertIn('-arch=sm_110a', cmd)
            self.assertIn('-lineinfo', cmd)
            self.assertIn('-DBIAS=0', cmd)
            self.assertNotIn('--use_fast_math', cmd)

    def test_missing_cutlass_dependency_fails_before_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.manifest(directory)
            with patch.object(build, 'ROOT', directory), patch.dict(os.environ, {}, clear=True), patch.object(build.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'CUTLASS_ROOT'):
                    build.build_cutlass(path)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
