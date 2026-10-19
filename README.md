# AutoKDAWorkflow

面向 NVIDIA Thor（ARM64 / SM110）的 KDA（Kernel Design Agents）优化工作流：从真实端到端（End-to-End，E2E）瓶颈出发，结合框架分析、内核设计知识和受控实验，形成可复查的优化闭环。

工作流支持模型推理、算子与运行时优化。它通过工具和验收契约协助 Codex、Claude Code 等 agent 工作；任务目标、精度要求和采用条件由具体任务定义。

## 快速开始

1. 安装分析机上的 Python 3.10+ 工具。

   ```bash
   git clone https://github.com/crycryout/AutoKDAWorkflow.git
   cd AutoKDAWorkflow
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements-dev.txt
   ```

2. 获取固定版本的公开知识库和分析工具。

   ```bash
   .venv/bin/python tools/bootstrap_corpus.py
   ```

   该步骤可在 CPU 分析机执行。需要 KDA 和 KDA-Pilot 源码时加 `--with-kda`；离线检查已准备的依赖可用 `--verify-only`。上游版本见 [sources.lock.json](sources.lock.json)，Thor 适配见 [patches/](patches/)。

3. 配置 Thor 执行环境。

   ```bash
   cp config/thor.example.json config/thor.local.json
   ```

   在本地配置中填写 `ssh_target`、`remote_root`、`allowed_root`、`python`、`cuda_home` 与 profiler 路径。默认使用 SSH alias `thor`，可以按环境配置 `ssh_prefix`、identity 和 known-hosts 文件。Thor 上的模型运行环境、权重、数据和可用 profiler 权限需要单独准备。

4. 检查连接并建立独立任务。

   ```bash
   .venv/bin/python tools/thor.py --config config/thor.local.json doctor
   .venv/bin/python tools/new_task.py --name model-e2e --model MODEL --objective 'Reduce synchronized inference latency within the frozen quality contract'
   ```

   开始测量前填写任务中的 baseline、workload、数值与质量要求。`doctor` 的版本输出不能代替真实编译、推理或 profiler capture 验证。

## 优化闭环

**冻结契约 → 建立 baseline 与 A/A → profile 定位瓶颈 → 单变量优化 → 验证数值、stream 与 Graph → 未插桩配对 E2E → 采用或回退。**

先根据 E2E 瓶颈选择框架优化或算子优化。框架开销、重复计算、数据拷贝、分配与同步占主导时，优先处理执行路径；主要时间集中在特定算子时，再进入 KDA 的内核设计与搜索。每次改动后重新检查 E2E，避免沿用已经变化的瓶颈判断。

完整方法见 [工作流说明](docs/workflow.md)，agent 入口见 [Thor KDA skill](skills/thor-kda-workflow/SKILL.md)，命令见 [运行参考](skills/thor-kda-workflow/references/commands.md)。

## 目录与职责

| 路径 | 用途 |
|---|---|
| `skills/thor-kda-workflow/` | agent 工作流、工具路由与 Thor 验收契约 |
| `tools/` | 任务创建、执行、知识库准备、trace 转换与性能分析 |
| `thor_kda/` | Thor 平台发现、构建、候选语义与证据记录 |
| `config/`、`templates/` | 本地环境配置示例与任务输入模板 |
| `infra/`、`knowledge/` | bootstrap 获取的固定版本上游工具和知识库 |

`tasks/` 下创建独立任务，分别维护 baseline、candidate、benchmark 与结果。构建产物、trace、模型权重、数据集和本地环境配置由使用者按项目需要管理。

## Agent 接入与验证

直接让 agent 阅读 `skills/thor-kda-workflow/SKILL.md`，或先预览技能链接，再显式安装：

```bash
.venv/bin/python tools/install_links.py --destination "$HOME/.codex/skills"
.venv/bin/python tools/install_links.py --destination "$HOME/.codex/skills" --apply
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s infra/tests -v
```

Claude Code 可将目标目录换为 `$HOME/.claude/skills`。工具应保留已有 skill，遇到同名冲突时先核对路径。

CPU 测试检查工具行为；ARM64 / SM110 构建、数值正确性、CUDA Graph、stream、profiler 与模型质量需要在实际 Thor 环境验证。依赖与许可见 [第三方说明](THIRD_PARTY_NOTICES.md)。
