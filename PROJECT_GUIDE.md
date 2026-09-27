# Long-Term Benchmark 项目说明

## 1. 项目定位

本项目是一个运行在 Isaac Sim 上的长程室内导航与交互评测 harness。它读取
CL_session_builder 生成的冻结 `scene_package_v2`，按 session 和 task 顺序驱动 agent，
在运行时合成 mutation 与前序状态变化，并输出任务、session 和 capability 指标。

项目不负责重新生成场景、筛选对象、规划任务或编译 ground truth。那些工作属于场景
生成/构建流程。`generation/` 中的 Actor SDG 只是可选的数据生成入口，和 benchmark
运行时保持边界分离。

## 2. 推荐仓库结构

```text
longterm_benchmark/
├── benchmark/
│   ├── agents/              # Protocol v1、子进程和 JSONL 通信
│   ├── scenes/              # 数据加载、状态合成和 Isaac Sim 环境
│   ├── runtime/             # session/task 编排和运行审计
│   ├── evaluation/          # 成功判定、指标、报告和可选 judge
│   └── viewer/              # viewer overlay
├── generation/
│   ├── actor_sdg.py         # 可选 Actor SDG 启动器
│   └── configs/              # Replicator Agent 配置
├── examples/                 # random、SimpleNav、HoloAgent 适配器
├── scripts/
│   ├── bootstrap.sh          # 主机 Python 环境
│   ├── check_env.sh          # 数据、numpy、Isaac Sim 预检
│   ├── run_smoke.sh          # 最小随机 agent 评测
│   ├── validate_dataset.py   # scene_package_v2 契约检查
│   ├── make_checksums.py     # 发布校验
│   └── analyze_result.py     # 结果分析
├── dataset/                  # 示例数据或发布清单
├── tests/                    # CPU 可运行测试
├── docs/                     # 协议、指标、数据卡和 agent 集成
├── run_benchmark.sh          # Isaac Sim 评测入口
├── README.md                 # 用户使用指南
└── PROJECT_GUIDE.md          # 本文件
```

`results/`、`.venv/`、缓存、模型和生成输出不属于源代码发布内容。

## 3. 运行时数据流

```text
dataset_manifest.json
        ↓
BenchmarkDataset
        ↓
canonical state + mutation + previous state changes
        ↓
IsaacVLNEnv
        ↓
SessionRunner ⇄ AgentProcess / Protocol v1
        ↓
action/state logs → metrics → summary
```

`BenchmarkDataset` 只读取冻结数据；`StateComposer` 合并隐藏 mutation 和前序实际状态；
`IsaacVLNEnv` 负责 USD 场景、导航栅格、相机、移动、碰撞以及高层 pick/place；
`SessionRunner` 负责 session 边界、任务顺序、agent I/O、超时和审计文件。

agent 不会收到 ground truth 或 mutation。QA 和 manipulation 的能力由 runner 根据
协议响应和环境状态判断；SimpleNav 只实现导航能力，其余任务按失败记录但不阻断后续
session。

## 4. 文件职责

| 文件或目录 | 职责 |
| --- | --- |
| `benchmark/scenes/dataset_loader.py` | 严格读取 `scene_package_v2` 和 manifest |
| `benchmark/scenes/state_composer.py` | 合成 mutation、canonical state 和实际状态变化 |
| `benchmark/scenes/isaac_env.py` | Isaac Sim 场景、相机、移动、碰撞、门和 pick/place |
| `benchmark/runtime/session_runner.py` | session/task 编排、协议交互、日志和结果落盘 |
| `benchmark/agents/agent_api.py` | agent 子进程、超时、stderr 和 JSONL 管道 |
| `benchmark/agents/protocol.py` | Protocol v1 请求/响应校验 |
| `benchmark/evaluation/success.py` | navigation、manipulation、QA 成功判定 |
| `benchmark/evaluation/report.py` | task/session/capability 汇总 |
| `benchmark/viewer/` | viewer 任务面板和运行状态展示 |
| `examples/` | 外部 agent 到 Protocol v1 的适配，不放入评测核心 |
| `generation/` | Actor SDG；需要独立的 Replicator Agent 环境 |
| `scripts/` | 发布、检查、smoke 和结果工具 |

## 5. 环境分层

### 主机 Python

用于数据验证、dry-run 和 CPU 测试。`scripts/bootstrap.sh` 创建虚拟环境并安装
`requirements.txt`，当前核心依赖为 numpy。

### Isaac Sim Python

`run_benchmark.sh` 使用 `ISAACSIM_PYTHON` 或 `${ISAACSIM_ROOT}/python.sh` 启动真正的
仿真评测。Isaac Sim 不由 pip 或本仓库自动安装；它必须由使用者预先安装并满足 GPU、
驱动和许可证要求。

### 外部 agent 环境

SimpleNav、HoloAgent、ROS 2、checkpoint 和本地 planner 各自维护自己的环境。适配器
只定义通信边界，不将这些项目复制进 benchmark 仓库。

### Actor SDG 环境

`generation/actor_sdg.py` 需要 `EXP_PATH` 指向 Isaac Sim experience 目录，并加载
`isaacsim.exp.action_and_event_data_generation.base.kit`。它使用 Replicator Agent
扩展和 Isaac 内置资产，仅用于生成数据，不应成为运行 benchmark 的隐含前置步骤。

## 6. 一键运行约定

仓库支持以下两级入口：

```bash
# 无 Isaac Sim：安装主机依赖并做数据/协议检查
./scripts/bootstrap.sh
source .venv/bin/activate
./scripts/check_env.sh
./run_benchmark.sh --dry-run

# 有 Isaac Sim：随机 agent headless smoke
export ISAACSIM_ROOT=/path/to/isaacsim
./scripts/run_smoke.sh
```

`run_smoke.sh` 只验证 benchmark、场景加载、相机和 Protocol v1 的闭环，不代表外部
模型质量。完整 SimpleNav/HoloAgent 评测仍需要用户启动相应服务并传递 `--agent-cmd`。

## 7. 数据和发布策略

数据集必须通过 `scripts/validate_dataset.py`，发布时使用
`scripts/make_checksums.py` 生成 SHA-256 清单。源数据不可被运行时修改，运行状态只写入
`results/`。

当前工作区的示例 USDZ 资产大于 GitHub 普通单文件限制，且场景总数据和历史结果较大。
正式仓库应采取以下策略：

1. Git 中只保留代码、schema、manifest、校验和、最小 smoke 数据或下载元数据；
2. 大型 USDZ 通过 Git LFS、GitHub Release 或对象存储发布；
3. 不提交 `results/`、模型 checkpoint、缓存和个人环境；
4. 发布前核对 `NOTICE` 中的场景资产授权；
5. 为每个数据集固定 `dataset_id`、版本、下载地址和 SHA-256。

## 8. 扩展规则

- 新 agent：新增 `examples/<agent>_agent.py`，遵守 Protocol v1；只有共享协议能力才进入
  `benchmark/agents/`。
- 新场景格式：修改 `benchmark/scenes/` 和 validator，同时增加契约测试；不要在 runner
  中加入旧 schema 兼容分支。
- 新指标：放在 `benchmark/evaluation/`，同时更新 `docs/METRICS.md` 和结果 schema。
- 新数据集：先生成冻结 scene package，再运行 validator、dry-run、单元测试和短步数
  Isaac smoke。
- 新数据生成能力：放在 `generation/`，明确其外部 Isaac Sim/资产依赖，不让评测入口
  隐式调用生成流程。

## 9. 发布前检查清单

```bash
./scripts/check_env.sh --dataset /path/to/dataset
python3 -m unittest discover -s tests -v
python3 -m benchmark.run --dataset /path/to/dataset --dry-run
python3 scripts/make_checksums.py /path/to/dataset
```

如果 Isaac Sim 可用，再执行：

```bash
DATASET_DIR=/path/to/dataset ./scripts/run_smoke.sh
```

最后确认 Git 状态中没有 `results/`、模型、缓存和未授权的大型资产。
