# Third-party notices

AutoKDAWorkflow combines a Thor workflow with integrations for the public projects
listed below. Each upstream project's original copyright notices, attribution,
license files and file-level terms continue to apply. This repository does not
relicense third-party work or expand permissions granted by its authors.

| Component and attribution | Pinned source | Installed location |
| --- | --- | --- |
| AI-Infra-Auto-Driven-SKILLS — BBuf and contributors | [v0.1.5, 6dc9c66](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/6dc9c66a008daded66f214022919ff88b2186252) | `infra/` |
| KernelWiki — MIT HAN Lab and contributors | [76d27b5](https://github.com/mit-han-lab/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75) | `knowledge/KernelWiki/` |
| ncu-report-skill — MIT HAN Lab and contributors | [d188794](https://github.com/mit-han-lab/ncu-report-skill/tree/d1887948c7d53690cfe6605f59c1329b8a1c6bb5) | `knowledge/ncu-report-skill/` |
| Kernel Design Agents — NVIDIA CORPORATION & AFFILIATES and contributors | [ef6ce61](https://github.com/NVlabs/kda/tree/ef6ce617693ef0782b3ecb9f37e39bbf10226a90) | `upstream/kda/` (optional) |
| KDA-Pilot — BBuf and contributors | [bac966b](https://github.com/BBuf/KDA-Pilot/tree/bac966ba09fe22f6220ff527c7c3555e539d5fe6) | `upstream/kda-pilot/` (optional) |

KDA's pinned `LICENSE` states that documentation, skills and assets use
CC-BY-4.0, while source code uses Apache-2.0. Its original `LICENSE`,
`THIRD_PARTY_NOTICES.md`, `third_party_licenses/` and source headers are retained
by bootstrap. See those files for the complete terms and component-specific
exceptions. Git submodule contents are not included in GitHub source archives;
KDA's external knowledge dependencies are installed separately from their own
pins.

The other pinned source archives do not provide a top-level license file.
Their upstream terms and any file-level notices remain applicable; public
availability is not a new license grant from this repository. Check the
respective upstream project for permission before redistributing its material.

The reviewed Thor integration changes for AI-Infra, KernelWiki and
ncu-report-skill are distributed as patches under `patches/`. Their patch hashes
and touched files are recorded in `patches/manifest.json`; full public commit
IDs and archive hashes are recorded in `sources.lock.json`. Bootstrap preserves
all files, notices and source headers from each pinned archive.
