"""CPU-only coverage of connection, boundary, identity, and evidence guarantees."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from thor_kda.config import PACKAGE_ID, RuntimeConfig, load_controller
import new_task
import remote_run
import thor


def gpu_state(pids=(), name="NVIDIA Thor"):
    return dict(gpu=dict(returncode=0, stdout=name),
                gpu_processes=dict(returncode=0, stdout="".join(f"{pid}, worker, 1 MiB\n" for pid in pids)))


class ConfigurationTests(unittest.TestCase):
    def fixture(self, base, **updates):
        data = dict(ssh_target="thor-lab", remote_root="/srv/tasks/workflow", allowed_root="/srv/tasks")
        data.update(updates)
        path = base / "connection.json"
        path.write_text(json.dumps(data))
        return path

    def test_strict_host_keys_and_private_transport_stays_local(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.fixture(Path(temp), ssh_prefix=["container-exec", "-i", "relay"],
                                  ssh_identity_file="/local/keys/identity", ssh_known_hosts_file="/local/keys/hosts")
            config = load_controller(source)
            ssh = config.ssh_command()
            self.assertIn("StrictHostKeyChecking=yes", ssh)
            self.assertIn("BatchMode=yes", ssh)
            self.assertEqual(ssh[:3], ["container-exec", "-i", "relay"])
            runtime = config.runtime.executor_environment()
            self.assertEqual(config.runtime.python, "python3")
            self.assertFalse(any("ssh" in key.lower() for key in runtime))
            self.assertNotIn("identity", " ".join(runtime.values()))
            self.assertEqual(RuntimeConfig.from_environ(runtime), config.runtime)

    def test_env_config_and_config_argument_work_before_or_after_subcommand(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.fixture(Path(temp))
            self.assertEqual(load_controller(environ={"THOR_KDA_CONFIG": str(source)}).ssh_target, "thor-lab")
            parser = thor.build_parser()
            for argv in (("--config", str(source), "doctor"), ("doctor", "--config", str(source))):
                self.assertEqual(parser.parse_args(argv).config, source)

    def test_unsafe_connection_paths_and_ssh_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for updates in (dict(allowed_root="/"), dict(remote_root="/srv/tasks"),
                            dict(remote_root="/srv/tasks/../outside"), dict(remote_root="relative"),
                            dict(ssh_target="-Fmalicious"), dict(ssh_target="two hosts"),
                            dict(profiler_sudo="false"), dict(expected_gpu="")):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    load_controller(self.fixture(base, **updates))

    def test_runtime_requires_explicit_write_boundary(self):
        config = RuntimeConfig.from_environ({})
        with self.assertRaisesRegex(ValueError, "THOR_KDA_ALLOWED_ROOT"):
            remote_run.bounded_root("/tmp/task", config)


class DeploymentTests(unittest.TestCase):
    def test_archive_omits_local_configuration_generated_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            files = ("tools/run.py", "config/thor.example.json", "config/thor.local.json",
                     "config/other-private.json", "tasks/sample/task.json", "tasks/sample/results/run.json",
                     "tasks/sample/cache/data", "tools/__pycache__/module.pyc", "knowledge/topic.md", "docs/setup.md")
            for name in files:
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            (base / "tools/link").symlink_to(base / "config/thor.local.json")
            with patch.object(thor, "ROOT", base):
                payload = thor.package_bytes(extra_files=("config/thor.local.json",))
                runtime = thor.package_bytes(runtime_only=True)
                with self.assertRaises(ValueError):
                    thor.package_bytes(extra_files=("../outside",))
                with self.assertRaises(ValueError):
                    thor.package_bytes(extra_files=("docs/../config/thor.local.json",))
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                self.assertEqual(set(archive.getnames()), {
                    "tools/run.py", "config/thor.example.json", "tasks/sample/task.json", "knowledge/topic.md", "docs/setup.md",
                })
            with tarfile.open(fileobj=io.BytesIO(runtime), mode="r:gz") as archive:
                self.assertNotIn("knowledge/topic.md", archive.getnames())

    def archive(self, filename="tools/sample.py"):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            member = tarfile.TarInfo(filename)
            content = b"print('sample')\n"
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        return stream.getvalue()

    def test_deploy_identity_boundary_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            allowed, root = Path(temp), Path(temp) / "workflow"

            def invoke(destination, payload=None):
                return subprocess.run([sys.executable, "-B", "-c", thor.DEPLOY_CODE,
                                       str(destination), str(allowed), PACKAGE_ID],
                                      input=self.archive() if payload is None else payload, capture_output=True)

            self.assertNotEqual(invoke(allowed).returncode, 0)
            root.mkdir()
            (root / "foreign.txt").write_text("owned by another package")
            self.assertNotEqual(invoke(root).returncode, 0)
            (root / "foreign.txt").unlink()
            self.assertNotEqual(invoke(root, self.archive("../escaped.py")).returncode, 0)
            self.assertFalse((allowed / "escaped.py").exists())
            self.assertEqual(invoke(root).returncode, 0)
            self.assertEqual(json.loads((root / "workflow.package.json").read_text())["package"], PACKAGE_ID)
            self.assertEqual(invoke(root).returncode, 0)

    def test_predeployment_doctor_stream_is_self_contained(self):
        # An unrelated working directory has neither the installed package nor SSH configuration.
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, THOR_KDA_PROFILER_SUDO="0", PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-B", "-", "--doctor"], input=thor.doctor_source(),
                                    capture_output=True, cwd=temp, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertFalse(json.loads(result.stdout)["profiler_capture_verified"])
            self.assertEqual(list(Path(temp).iterdir()), [])


class ExecutorTests(unittest.TestCase):
    def config(self, allowed, **changes):
        return replace(RuntimeConfig(allowed_root=Path(allowed), python=sys.executable), **changes)

    def test_boundary_resolves_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            allowed = Path(temp) / "allowed"
            allowed.mkdir()
            outside = Path(temp) / "outside"
            outside.mkdir()
            (allowed / "alias").symlink_to(outside, target_is_directory=True)
            config = self.config(allowed)
            self.assertEqual(remote_run.bounded_root(allowed / "task", config), allowed / "task")
            for path in (allowed, outside, allowed / "alias/output"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    remote_run.bounded_root(path, config)

    def test_busy_gpu_prevents_execution_and_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "task"
            with patch.object(remote_run.platform, "machine", return_value="aarch64"), \
                 patch.object(remote_run, "snapshot", return_value=gpu_state([99])), \
                 patch.object(remote_run.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                    remote_run.execute(root, "test", ["true"], config=self.config(temp))
                run.assert_not_called()
                self.assertFalse(root.exists())

    def test_hardware_and_optional_host_identity_prevent_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "task"
            cases = (("x86_64", "unselected", "NVIDIA Thor", None),
                     ("aarch64", "unselected", "Other GPU", None),
                     ("aarch64", "unexpected", "NVIDIA Thor", "selected"))
            for arch, host, gpu, required_host in cases:
                with self.subTest(arch=arch, gpu=gpu, host=host), \
                     patch.object(remote_run.platform, "machine", return_value=arch), \
                     patch.object(remote_run.platform, "node", return_value=host), \
                     patch.object(remote_run, "snapshot", return_value=gpu_state(name=gpu)), \
                     self.assertRaises(RuntimeError):
                    remote_run.execute(root, "test", ["true"], config=self.config(temp, expected_hostname=required_host))
                self.assertFalse(root.exists())

    def test_local_execution_retains_logs_and_environment_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "task"
            with patch.object(remote_run.platform, "machine", return_value="aarch64"), \
                 patch.object(remote_run, "snapshot", side_effect=lambda: gpu_state()):
                record, directory = remote_run.execute(
                    root, "local-cpu", ["{python}", "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                    cpu_only=True, config=self.config(temp),
                )
            self.assertEqual(record["returncode"], 0)
            self.assertFalse(record["interference_observed"])
            self.assertTrue((directory / "stdout.log").is_file())
            self.assertTrue((directory / "stderr.log").is_file())
            saved = json.loads((directory / "run.json").read_text())
            self.assertEqual(saved["profile"], "none")
            self.assertEqual(saved["command"][0], sys.executable)
            self.assertTrue(Path(saved["environment"]["TMPDIR"]).is_relative_to(root))
            self.assertIn("before", saved)
            self.assertIn("after", saved)

    def test_missing_profiler_report_never_counts_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            for profile, field in (("ncu", "ncu_capture_import_passed"), ("nsys", "nsys_capture_export_passed")):
                with self.subTest(profile=profile), \
                     patch.object(remote_run.platform, "machine", return_value="aarch64"), \
                     patch.object(remote_run, "snapshot", side_effect=lambda: gpu_state()), \
                     patch.object(remote_run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                    record, directory = remote_run.execute(
                        Path(temp) / "task", profile, ["true"], profile=profile, config=self.config(temp),
                    )
                self.assertFalse(record[field])
                self.assertTrue((directory / "run.json").is_file())

    def test_profiler_cache_is_per_run_and_home_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "task"
            config = self.config(temp)
            env, first = remote_run.environment(root, root / "results/ncu/scratch", config)
            _, second = remote_run.environment(root, root / "results/nsys/scratch", config)
            self.assertEqual(env.get("HOME"), os.environ.get("HOME"))
            self.assertNotIn("HOME", first)
            for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "CUTE_DSL_CACHE_DIR"):
                self.assertNotEqual(first[key], second[key])
                self.assertTrue(Path(second[key]).is_relative_to(root))
            with self.assertRaises(ValueError):
                remote_run.environment(root, Path(temp) / "other-task", config)

    def test_gpu_process_ownership_handles_child_exit_and_foreign_work(self):
        self.assertEqual(remote_run.foreign_gpu_pids(gpu_state([42]), {1, 42}, {1}), [])
        self.assertEqual(remote_run.foreign_gpu_pids(gpu_state([42, 99]), {1}, {1, 42}), [99])
        state = gpu_state()
        state["gpu_processes"].update(returncode=1, stderr="unavailable")
        with self.assertRaises(RuntimeError):
            remote_run.foreign_gpu_pids(state, {1}, {1})


class TaskTests(unittest.TestCase):
    def test_generic_task_requires_real_baseline_and_acceptance(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(new_task, "ROOT", Path(temp)):
            path = new_task.create("sample-model", "model-id", "Reduce measured end-to-end latency")
            contract = json.loads((path / "task.json").read_text())
            self.assertFalse(contract["workload_frozen"])
            self.assertIsNone(contract["baseline_manifest"])
            self.assertIsNone(contract["numerical_acceptance"])
            self.assertEqual(contract["remote_relative_root"], "tasks/sample-model")
            self.assertNotIn("host", contract["target"])
            self.assertTrue((path / "bench").is_dir())
            with self.assertRaises(FileExistsError):
                new_task.create("sample-model", "model-id", "Overwrite")
            with self.assertRaises(ValueError):
                new_task.create("../escape", "model-id", "Escape")


if __name__ == "__main__":
    unittest.main()
