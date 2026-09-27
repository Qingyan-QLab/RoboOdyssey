# Long-Term Indoor Benchmark

面向 Isaac Sim 的长程视觉语言导航与场景交互评测仓库。仓库同时保留一个可选的
Actor SDG 数据生成入口，但数据生成不属于 benchmark 运行时依赖。

## 能力边界

- benchmark 消费 `scene_package_v2` 数据集，按 session 顺序执行 navigation、
  manipulation 和 QA，并记录可复现实验结果；
- `examples/random_agent.py` 用于 smoke 和协议联调；
- `examples/simplenav_agent.py` 适配 SimpleNav 导航策略；
- `examples/holoagent_*.py` 适配 HoloAgent/ROS/HTTP 网关；
- `generation/actor_sdg.py` 通过 Isaac Replicator Agent 生成行为数据，独立于评测；
- Isaac Sim、模型 checkpoint、SimpleNav/HoloAgent 和相关许可证资产不随 Python
  依赖自动安装。

## Clone 后的最短路径

### 1. 安装主机侧依赖

```bash
git clone <repository-url>
cd longterm_benchmark
./scripts/bootstrap.sh
source .venv/bin/activate
```

脚本创建 `.venv` 并安装 `requirements.txt`。若已有可用 Python 环境，可使用
`./scripts/bootstrap.sh --skip-install`，或设置 `BENCHMARK_VENV` 指定位置。

### 2. 不启动 Isaac Sim 的数据和协议检查

```bash
./scripts/check_env.sh
python -m benchmark.run --dry-run
python -m unittest discover -s tests -v
```

`run_benchmark.sh --dry-run` 也可以在未安装 Isaac Sim 的机器上执行：

```bash
./run_benchmark.sh --dry-run
```

### 3. Isaac Sim smoke benchmark

先安装与当前项目兼容的 Isaac Sim，并设置其安装目录：

```bash
export ISAACSIM_ROOT=/absolute/path/to/isaacsim
./scripts/run_smoke.sh
```

默认使用 `dataset/scene000430_new`、`random_agent.py` 和每个任务 20 步。可以通过
`SMOKE_MAX_STEPS`、`DATASET_DIR` 或额外的 benchmark 参数调整：

```bash
SMOKE_MAX_STEPS=50 ./scripts/run_smoke.sh --viewer
DATASET_DIR=/path/to/dataset ./scripts/run_smoke.sh
```

`--viewer` 需要图形环境；服务器或无显示环境使用默认 headless 模式。完整运行入口
仍为：

```bash
./run_benchmark.sh --max-steps 2000 \
  --agent-cmd "python3 ${PWD}/examples/random_agent.py"
```

## 使用外部 agent

SimpleNav 需要先启动其 policy server，并提供 checkpoint：

```bash
./run_benchmark.sh \
  --scene scene000430 --max-steps 2000 \
  --agent-cmd "python3 ${PWD}/examples/simplenav_agent.py \
    --url ws://127.0.0.1:10093 --checkpoint /path/to/checkpoint.pt"
```

HoloAgent 需要先启动 benchmark gateway 或 planner。仓库提供的适配器只负责 Protocol
v1 转换，不会伪造 `/benchmark/step` 服务，也不会把 ROS 命令发布成功误认为任务完成。
具体接口和在线 OVO/FSR-VLN 配置见 [`docs/AGENT_INTEGRATION.md`](docs/AGENT_INTEGRATION.md)。

## 数据集

数据集根目录必须包含 `dataset_manifest.json`，并遵循 `scene_package_v2`：

```text
dataset_root/
├── dataset_manifest.json
└── scenes/<scene_id>/
    ├── scene.usdz
    ├── scene_metainfo.json
    └── sessions/
        ├── scene_build.json
        ├── navigation.npz
        ├── manipulation.json
        └── session_NNN/
            ├── session.json
            ├── mutation.json
            └── ground_truth.json
```

提交新数据前运行：

```bash
python3 scripts/validate_dataset.py /path/to/dataset
python3 scripts/make_checksums.py /path/to/dataset
```

当前工作区包含多个示例数据集。发布到 GitHub 时不要把大型 USDZ 和运行结果直接当作
普通 Git blob：单个 `scene.usdz` 约 228 MB，超过 GitHub 普通文件限制。建议使用
Git LFS 或 GitHub Release/对象存储，并在发布包中提供 manifest、SHA-256 和下载说明。
场景资产是否可以重新分发还需遵守 [`NOTICE`](NOTICE)。

## 结果

每次运行在 `--output` 下创建时间戳目录：

```text
results/YYYYMMDD_HHMMSS_<agent>/
├── run_config.json
├── run.log
├── summary.json
└── scenes/<scene>/sessions/<session>/
    ├── action_changelog.jsonl
    ├── agent_io.jsonl
    ├── session_meta.json
    ├── state_changelog.json
    └── metrics.json
```

`results/`、缓存、虚拟环境和生成输出默认被 `.gitignore` 排除。结果分析：

```bash
python3 scripts/analyze_result.py results/<run-directory>
```

## 目录速览

```text
longterm_benchmark/
├── benchmark/       # 数据加载、Isaac 环境、session runner、评测和协议
├── generation/      # 可选 Actor SDG 入口和配置
├── examples/        # agent 适配器和 smoke agent
├── scripts/         # bootstrap、环境检查、smoke、数据校验和结果分析
├── dataset/         # 示例数据或发布清单
├── tests/           # 不依赖 GPU 的测试
└── docs/            # 协议、指标、数据卡和 agent 集成说明
```

核心设计、扩展边界和发布流程见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)。
