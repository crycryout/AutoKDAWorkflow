# Thor execution and acceptance

## Execution environment

Use the task's local configuration for the SSH target, remote package/task
directories, Python, CUDA and profiler paths. Keep host-key checking enabled.
Use a task-owned directory and validate resolved paths before writing, deploying
or fetching. Keep caches, temporary files and build outputs inside that scope;
do not repurpose `HOME`.

Configure `ssh_target`, optional `ssh_prefix`, `remote_root`, `allowed_root`,
`python` and `cuda_home` in `config/thor.local.json`. Configure profiler sudo,
NCU/NSYS paths and NCU section directories for the actual environment. Use
optional hostname checks only when the task requires a specific host; hardware
and architecture checks remain required.

Verify ARM64 (`aarch64`), NVIDIA Thor and compute capability 11.0 on the actual
execution target. Compile the supported SM110 path for `sm_110a` and set
`CUTE_DSL_ARCH=sm_110a` when using CuTe DSL. Record actual CUDA, driver, Torch,
CUTLASS and CuTe versions together with source and binary hashes. Treat an
unsupported instruction, cluster geometry or backend as a capability gap.

Run builds and inference as the configured normal user. Use the profiler
privilege permitted by that environment. Verify NCU capture and report import;
an installed executable or a successful `--version` is insufficient. Retain
profiler failures and investigate supported permissions and metrics. Never
request or embed a password in a benchmark command.

Defer GPU measurements when foreign compute activity is present. Monitor
interference during a window and mark affected samples invalid. Do not stop
another process. Record power mode, GPU/EMC devfreq, memory pressure and cache
conditions when comparing windows; changing clocks or power mode changes the
experiment contract.

## Frozen model and workload contract

Record model and weight revisions, framework/backend, source hashes, dtype,
accumulation/quantization boundaries, batch, shapes, layout, bias, masks, seeds,
input source, preprocessing, iteration counts, stream, CUDA Graph mode, warmup
and the exact timing region. Record whether shape, values, addresses and control
flow may change independently. Preserve the actual accepted baseline.

Use synchronized inference/action-chunk latency for policy and diffusion
workloads. Use tokens/s, TTFT/TPOT, prefill/decode or KV-cache capacity only for
the autoregressive serving paths where those metrics apply. Include all work
required by the task inside E2E; document preprocessing, capture, compilation
and request handling separately when their cost is outside the declared region.

Match baseline and candidate allocation, output, stream, launch and timing
semantics. Validate tails and input changes, initialize outputs with poison
values to expose unwritten elements, check nondefault streams, and replay
Graphs after changing inputs. Use sanitizers when new memory or synchronization
behavior warrants them. A kernel tolerance does not establish model quality.

## Performance acceptance

Establish adjacent A/A runs before interpreting A/B. Keep raw unprofiled paired
samples, at least 20 valid pairs for the default protocol, exclusions with
reasons, and an independent confirmation window. Predeclare the improvement
target and accepted variation. Increase samples or report uncertainty when
the signal cannot be separated from the A/A noise.

Use instrumented traces and NCU reports for attribution and diagnosis. Measure
candidate speed separately without instrumentation. A fast warm-weight micro-
benchmark may lose its gain under the model's actual cache and execution context.
Keep kernel-duration sums, interval unions, GPU spans and synchronized model
E2E separate. Compute contribution percentages from the same trace/window.

Adopt a candidate only after all numerical, stream, Graph and task-quality gates
pass and the declared E2E condition is met. Record rejected candidates too,
including the failed premise and the evidence needed to reconsider them.

## Thor-specific interpretation

Thor uses shared system RAM for CPU and GPU. `MemAvailable`, process RSS and
Torch allocated/reserved memory are overlapping views; do not add them as
independent CPU and GPU capacities. Report point-in-time headroom and actual
runtime allocations rather than importing an HBM-based capacity model.

Treat GPU/EMC devfreq values as platform diagnostics. They do not alone prove
stable effective memory behavior or equal cache conditions. Check run-to-run
variance and representative memory pressure alongside sampled clocks.

Discover NCU metrics from the actual report. Missing or unsupported counters
remain unknown, including DRAM and PM-sampling observations. Preserve
`--clock-control none --cache-control none` when diagnosing the application's
current context; record any deliberately different capture policy. Match any
peak-throughput denominator to dtype, power mode and a cited or measured source.
Leave MFU unset when no justified peak is available.
