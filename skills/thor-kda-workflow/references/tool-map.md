# Tool routing

Resolve paths relative to the repository root. Run `tools/bootstrap_corpus.py`
before loading `infra/` or `knowledge/`. Open the selected upstream skill and
apply [the Thor contract](thor-contract.md) to the actual runtime.

| Question | Tool or reference | Required interpretation |
|---|---|---|
| Is the Thor environment ready? | `tools/thor.py ... doctor` | Verify actual architecture, packages, occupancy and paths; version checks do not prove GPU capture. |
| How should a task be isolated? | `tools/new_task.py` | Fill the objective, frozen baseline/workload and quality gates before experiments. |
| Where does the request spend time? | `infra/skills/llm-torch-profiler-analysis/` | Select the actual framework; preserve unknown module ownership. |
| Which stage or layer owns an interval? | `infra/skills/llm-pipeline-analysis/`, `infra/skills/torch-profiler-layer-track/` | Supply verified anchors, counts and component boundaries; navigation intervals are not automatically exclusive layer times. |
| How can Nsight be analyzed offline? | `tools/nsys_to_chrome.py` | Retain original reports and mapping metadata; converted events cannot create absent Python stacks. |
| Which kernel design is worth trying? | `knowledge/KernelWiki/`, pinned KDA references | Read relevant pages only; validate architecture-specific assumptions on Thor. |
| Which kernel cost explains the result? | `knowledge/ncu-report-skill/`, `tools/ncu_summary.py` | Discover supported report metrics and keep instrumented time separate from speed measurements. |
| Did the candidate improve model E2E? | `tools/thor_metrics.py compare` | Supply matched-contract unprofiled pairs, A/A and independent-window evidence; the report does not adopt a candidate. |
| What compute and memory budget was observed? | `tools/thor_metrics.py compute` / `capacity` | Use observed calls and unified-memory headroom; leave missing counts, verified peak and runtime-specific capacity unset. |
| What integration constraints need coverage? | `infra/skills/model-optimization/sglang-model-day0-support/` | Reuse dependency and validation structure for the actual framework; record unsupported backends and fallback paths. |

## Runtime-specific routes

Use serving benchmark, capacity and incident tools under `infra/skills/` only
when their deployed runtime and input formats match the task. Do not assume
that a policy model exposes an HTTP profiler endpoint or that a diffusion model
has autoregressive KV-cache semantics. Check ARM64 packaging and SM110 backend
support before reusing an upstream launch command.

Use `infra/model-pr-optimization-history/` and the public review corpus as
design references when a concrete implementation question needs them. Verify
the cited change, callsite, shape and backend before applying its lesson.

## Candidate interface

Use `thor_kda` for task-local CUTLASS builds, manifests and source/binary evidence.
Freeze a candidate's layout, bias, dtype, accumulation, output ownership,
workspace, stream and Graph semantics. Provide an explicit model harness that
calls the accepted baseline and candidate through symmetric wrappers. Kernel
search is one step in the E2E workflow, and the integration quality gate belongs
to the actual model task.
