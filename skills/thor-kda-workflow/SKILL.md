---
name: thor-kda-workflow
description: Optimize and analyze NVIDIA Thor ARM64/SM110 model inference, CUDA kernels, and runtime execution with KDA knowledge, profiler evidence, frozen numerical contracts, and paired end-to-end measurements. Use for Thor performance investigations, model integration, kernel candidates, trace analysis, and correctness or performance regressions.
---

# Thor KDA workflow

Resolve the repository root two directories above this skill directory. Run
commands from that root. Read [execution and acceptance](references/thor-contract.md)
before executing GPU work. Read [tool routing](references/tool-map.md) only for
the task's current question and [commands](references/commands.md) when running
the selected tool. Use [the workflow guide](../../docs/workflow.md) for detailed
experiment design.

Prepare missing public tools with `python3 tools/bootstrap_corpus.py` on the
CPU analysis host. Use pins from `sources.lock.json` and the reviewed Thor
adaptations in `patches/`. Load only relevant knowledge pages. Treat SM90/SM100
designs as hypotheses until compiled and tested on SM110.

## Establish the experiment

1. Create or open one task and freeze its measurable objective, baseline source
   and weight revisions, actual workload, input variability, precision,
   numerical and model quality limits, Graph/stream mode, and timing boundary.
2. Establish an unprofiled baseline and adjacent A/A pairs under the intended
   cache, warmup, concurrency, power and memory conditions. Resolve drift and
   foreign GPU activity before attributing a small change to an optimization.
3. Profile a bounded representative window. Keep model E2E, CPU launch time,
   kernel-duration sum, interval union and GPU span distinct. Attribute phases
   using source/configuration evidence, preserving unknown ownership.

## Form and validate a candidate

Choose a single bottleneck-supported change. Prefer framework changes when
avoidable work, launch overhead, synchronization, allocation or data movement
dominate. Use KDA/KernelWiki and Nsight Compute when a material kernel cost
justifies a candidate. Preserve the accepted baseline and write a short
hypothesis, changed variable, expected effect, and rejection condition.

Keep baseline and candidate harnesses symmetric in inputs, allocation, stream,
capture and timing. Verify basic numerics during development, then validate
real input ranges, high precision or boundary cases, tails, changed inputs,
poisoned outputs, nondefault streams, and changed-input CUDA Graph replay as
applicable. Distinguish fixed shape from fixed values. A fixed-shape route
must still compute changed inputs unless the task explicitly fixes values.

Keep numerical acceptance active throughout experiments. Enable the full
model quality gate before adoption. Do not loosen tolerances, move work outside
the timing boundary, introduce output reuse, or change precision as an implicit
optimization. Record and evaluate any requested contract change separately.

## Decide using end-to-end evidence

Measure adjacent or interleaved unprofiled baseline/candidate model pairs.
Use at least 20 valid pairs for the default comparison protocol, retain A/A
calibration, and confirm the result in an independent measurement window.
Increase sampling or report an inconclusive result when noise exceeds the
claimed effect. Preserve every excluded sample with its reason.

Adopt only when the frozen correctness and quality gates pass and model E2E
meets the task's acceptance condition. A standalone kernel score or a profiler
estimate cannot make this decision. Record adoption or rollback, parent/source
hashes, commands, raw timings, Graph/stream evidence, platform state and remaining
limits. After adoption, update the baseline and reprofile when the bottleneck
may have changed.

Use the configured `tools/thor.py` entry point for bounded Thor execution.
Do not launch a goal, agent, server, or unrelated GPU job merely because the
skill was loaded. Continue the explicitly requested optimization within its
task scope and report measured outcomes without inventing missing evidence.
