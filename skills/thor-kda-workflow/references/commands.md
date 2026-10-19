# Commands

Run commands from the repository root. Use a fresh tag/output path for each
measurement. Read each command's `--help` for the exact configured options.
Values in angle brackets are task-specific placeholders.

- [Prepare dependencies and a task](#prepare-public-dependencies-and-a-task)
- [Configure and inspect Thor](#configure-and-inspect-thor)
- [Execute and analyze evidence](#execute-a-bounded-task-runner)
- [Verify CPU tools](#verify-cpu-tools)
- [Optional GEMM ABI checker](#optional-gemm-abi-checker)

## Prepare public dependencies and a task

```bash
python3 tools/bootstrap_corpus.py
python3 tools/install_links.py --destination "$HOME/.codex/skills"
python3 tools/install_links.py --destination "$HOME/.codex/skills" --apply
python3 tools/new_task.py --name model-e2e --model MODEL --objective 'Reduce synchronized inference latency within the frozen quality contract'
```

Bootstrap runs on a CPU analysis machine. Prepare model weights, datasets and
the compatible Torch/CUDA environment on Thor separately. Complete task.json
and the numerical/model-quality contract before using measurements to decide
adoption. Task preparation does not launch an agent or GPU workload.

Add `--with-kda` to fetch the pinned public KDA and KDA-Pilot sources as well.
Use `python3 tools/bootstrap_corpus.py --verify-only` to check prepared
dependencies without downloading them.

## Configure and inspect Thor

```bash
cp config/thor.example.json config/thor.local.json
python3 tools/thor.py --config config/thor.local.json doctor
python3 tools/thor.py --config config/thor.local.json deploy
```

Fill the local configuration before connecting. Review the configured target
and deployment directory. A new environment needs successful build, execution
and profiler capture checks in addition to doctor output.

## Execute a bounded task runner

```bash
python3 tools/thor.py --config config/thor.local.json run --task-root '<REMOTE_TASK_ROOT>' --tag baseline-001 -- '{python}' bench/runner.py
python3 tools/thor.py --config config/thor.local.json run --task-root '<REMOTE_TASK_ROOT>' --tag nsys-001 --profile nsys -- '{python}' bench/runner.py
python3 tools/thor.py --config config/thor.local.json run --task-root '<REMOTE_TASK_ROOT>' --tag ncu-001 --profile ncu -- '{python}' bench/kernel_probe.py
```

Create the task-owned runner before these commands. Preserve its actual input
preparation, stream and Graph semantics. Use a short representative profiler
window. Collect unprofiled paired performance in separate runs, preserving raw
samples and A/A calibration. Fetch artifacts using `tools/thor.py ... fetch`;
inspect `fetch --help` for source and destination arguments.

For native Torch capture, instrument an already warmed task harness:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, with_stack=True,
) as prof:
    with torch.profiler.record_function("model/verified_request"):
        output = model_call(frozen_input)
        torch.cuda.synchronize()
prof.export_chrome_trace(str(task_root / "profile" / "request.trace.json"))
```

This snippet defines a capture boundary; integrate it with the model's existing
initialization and runner. It does not define the unprofiled performance test.

## Analyze evidence

```bash
python3 tools/nsys_to_chrome.py --input '<REPORT.sqlite>' --output '<NEW_TRACE.json>' --framework thor --window-regex '<VERIFIED_REQUEST_REGEX>' --window-index 0
python3 infra/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py --framework thor --input '<NEW_TRACE.json>'
python3 knowledge/KernelWiki/scripts/query.py 'small M GEMM' --limit 3
python3 tools/ncu_summary.py --input '<REPORT.ncu-rep>' --output '<NEW_METRICS.json>'
python3 tools/thor_metrics.py compare --input '<PAIRED_SAMPLES.json>' --output '<NEW_COMPARISON.json>'
```

NCU inventory requires NVIDIA's `ncu_report` Python binding in the analysis
environment. Preserve report provenance and unknown metrics. For layer/phase
analysis, supply source-verified boundaries instead of inferring layers from
divisible kernel counts. The paired sample format is documented in
`templates/paired-e2e.schema-example.json`; fill it with actual observations.

## Verify CPU tools

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s infra/tests -v
```

CPU results do not establish GPU correctness or performance. Run the task's
numerical, stream, Graph and model-quality checks on Thor before adoption.

## Optional GEMM ABI checker

`templates/kernel-candidate.json`, `thor_kda` and `tools/smoke_gemm.py` provide
an optional manifest-driven BF16 GEMM interface. Supply your own kernel source
and binary implementing the declared C ABI, fill the candidate manifest, then
run the build/check commands shown by `python3 -m thor_kda --help` and
`python3 tools/smoke_gemm.py --help` in the configured Thor task environment.
`tools/validate_cutlass.py` exposes the same checker through a small CLI bridge.

The function signatures and tensor layout are documented in
[the GEMM ABI reference](../../../docs/kernel-abi.md). Set the template's
numerical tolerance from the task contract before running it.

This checker covers its declared layout, bias, numerics, nondefault stream and
changed-input Graph replay contract. It is not an arbitrary kernel verifier
and cannot establish model E2E speed or model/task quality. Extend the task's
validation harness when its operator or numerical contract differs.
