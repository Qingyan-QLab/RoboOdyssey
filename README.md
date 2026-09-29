# Long-Term Indoor Benchmark

面向 Isaac Sim 的长程视觉语言导航与场景交互评测仓库。仓库同时保留一个可选的
Actor SDG 数据生成入口，但数据生成不属于 benchmark 运行时依赖。

## 能力边界

- benchmark 消费 `scene_package_v2` 数据集，按 session 顺序执行 navigation、
  manipulation 和 QA，并记录可复现实验结果；
- `examples/random_agent.py` 用于 smoke 和协议联调；
- `examples/simplenav_agent.py` 适配 SimpleNav 导航策略；
- `examples/holoagent_*.py` 适配 HoloAgent/ROS/HTTP 网关；
- `examples/openai_compatible_vlm_agent.py` 适配 OpenAI-compatible 视觉模型服务；
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
python -m benchmark doctor
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

默认使用 `dataset/scene000430_after_changes`、`random_agent.py` 和每个任务 20 步。可以通过
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

也可以使用统一入口；`--official` 会锁定完整数据集、headless、相机和步数等正式配置，
并把包括 session 起点策略在内的配置哈希写入结果。当前仍允许 `inherit` 和 `explicit`
两种起点模式，两者应按不同配置口径比较：

```bash
python -m benchmark run --official \
  --agent-cmd "python3 ${PWD}/examples/my_agent.py"
```

数据发布后可通过统一下载入口安装并校验归档：

```bash
python -m benchmark download --url <dataset-archive-url> \
  --sha256 <archive-sha256> --output dataset/scene000430-preview-v1
```

## 使用外部 agent

通用 OpenAI-compatible VLM 服务可以通过薄适配器接入：

```bash
VLM_BASE_URL=http://127.0.0.1:8000/v1 VLM_MODEL=my-vlm \
./run_benchmark.sh --agent-cmd \
  "python3 ${PWD}/examples/openai_compatible_vlm_agent.py --model my-vlm"
```

模型服务只接收协议中的图像和文本，不需要导入 Isaac Sim。

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

完整能力使用组合 server：FSR/在线 OVO 负责导航，OpenAI-compatible 视觉模型负责操作、
QA 和跨 session 记忆更新：

```bash
python examples/holoagent_benchmark_server.py \
  --live-ovo --mapping-root /path/to/HoloAgent/agentic_robot/fsr_vln \
  --vlm-endpoint http://127.0.0.1:8000/v1 --vlm-model my-vlm

./run_benchmark.sh --holoagent-sensors \
  --agent-cmd "python3 ${PWD}/examples/holoagent_agent.py \
    --endpoint http://127.0.0.1:8090/benchmark/step"
```

本机已有 HoloAgent 环境和高层 VLM 权重时，可直接运行完整评测：

```bash
./scripts/run_holoagent.sh
```

该命令使用 HoloAgent 原生 OVO/SAM2/SigLIP/VanillaMapper 进行在线 RGB-D
语义建图和导航；高层 VLM 仅处理 QA、操作决策和记忆。可通过
`HOLOAGENT_VLM_MODEL`、`HOLOAGENT_MAPPING_DEVICE`、`HOLOAGENT_CONDA_ENV`
和 `HOLOAGENT_VLM_CONDA_ENV` 覆盖本机配置。

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

`results/`、缓存、虚拟环境和生成输出默认被 `.gitignore` 排除。汇总分母固定为数据集定义的
全部任务；未执行、不支持、超时、崩溃和缺失结果均按失败计。结果分析：

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
