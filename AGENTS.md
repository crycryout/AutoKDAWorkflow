# Thor KDA workflow

Read `skills/thor-kda-workflow/SKILL.md` before changing or measuring a Thor
model, runtime, or kernel. Resolve the repository root from this file; keep
commands independent of the checkout's absolute location.

Use `python3 tools/bootstrap_corpus.py` on the analysis host to prepare pinned
public dependencies before using `infra/` or `knowledge/`. Preserve their
licenses and the versions in `sources.lock.json`.

Freeze the task's model, accepted baseline, inputs, precision, Graph/stream
behavior, timing boundary, and quality contract. Establish A/A noise before
interpreting small gains. Select framework or kernel changes from measured
E2E bottlenecks, then change one testable variable at a time.

Build and validate ARM64 / SM110 code on the configured Thor. CPU analysis is
useful for traces and tests but cannot establish Thor kernel correctness or
performance. Read the local configuration rather than inventing a host,
interpreter, dependency path, or profiler privilege. Do not alter clocks,
drivers, power modes, other tasks, or running services to obtain a result.

Keep each task's caches, outputs, baseline and candidate isolated. Validate
numerics, changed inputs, output coverage, nondefault streams, and changed-input
Graph replay before comparing candidates. Use unprofiled paired model E2E and
the actual model quality requirement for adoption; preserve raw samples and
source hashes. Profiler timings and standalone scores are diagnostic evidence.

Run `python3 -m unittest discover -s tests -v` for local tool changes. After
bootstrap, run the relevant upstream tests, including
`python3 -m unittest discover -s infra/tests -v` when affected. State the exact
validation performed and leave untested GPU behavior explicitly unverified.
