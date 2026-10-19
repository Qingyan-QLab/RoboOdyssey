# Long-Term Indoor Benchmark

面向 Isaac Sim 的长程视觉语言导航、操作和问答评测工具。一次运行按数据集中的
session 顺序执行任务，并将动作、状态和指标保存到结果目录。

## 快速开始

### 1. 安装依赖

需要 Python 3.10 或更高版本。下面的脚本会在项目目录创建 `.venv`，并安装项目所需的
主机侧依赖；不要求用户预先准备虚拟环境：

```bash
git clone <repository-url>
cd longterm_benchmark
./scripts/bootstrap.sh
```

后文命令使用项目内的 `.venv/bin/python`，因此不依赖当前 shell 是否执行过
`source .venv/bin/activate`。如果希望直接输入 `python`，可以额外执行该命令激活环境。

主机 Python 环境用于数据校验和 dry-run；真正运行 benchmark 还需要单独安装兼容版本的
Isaac Sim，并设置：

```bash
export ISAACSIM_ROOT=/absolute/path/to/isaacsim
```

也可以直接设置 Isaac Sim 启动器：`export ISAACSIM_PYTHON=/path/to/isaacsim/python.sh`。

### 2. 放置场景数据

将每个数据集放在 `dataset/<dataset-name>/` 下。目录根部必须有
`dataset_manifest.json`，manifest 中的路径相对于该目录解析：

```text
dataset/<dataset-name>/
├── dataset_manifest.json
└── scenes/<scene-id>/
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

仓库自带示例数据 `dataset/scene000430_after_changes`，也是默认数据集。外部数据可直接
放在其他目录，并通过 `--dataset` 或 `DATASET_DIR` 指定。先校验数据：

```bash
./.venv/bin/python scripts/validate_dataset.py dataset/<dataset-name>
# 可选：生成 SHA-256 清单
./.venv/bin/python scripts/make_checksums.py dataset/<dataset-name>
```

### 3. 检查并预览任务

以下命令不启动 Isaac Sim：

```bash
./.venv/bin/python -m benchmark doctor --dataset dataset/<dataset-name>
./.venv/bin/python -m benchmark run --dataset dataset/<dataset-name> --dry-run
# 等价写法
./run_benchmark.sh --dataset dataset/<dataset-name> --dry-run
```

### 4. 运行 benchmark

最小 smoke 运行（默认使用 `random_agent.py`，每个任务最多 20 步）：

```bash
./scripts/run_smoke.sh
```

指定数据集、agent 和步数：

```bash
./run_benchmark.sh \
  --dataset dataset/<dataset-name> \
  --agent-cmd "python3 ${PWD}/examples/random_agent.py" \
  --max-steps 2000
```

正式评测使用完整数据集配置：

```bash
./run_benchmark.sh --official \
  --dataset dataset/<dataset-name> \
  --agent-cmd "python3 ${PWD}/examples/my_agent.py"
```

服务器无显示器时保持默认 headless 模式；本地查看 Isaac Sim 窗口时追加 `--viewer`。
常用筛选参数有 `--scene scene000430`、`--session session_000` 和
`--task-type navigation|manipulation|qa`。运行参数完整列表：

```bash
./.venv/bin/python -m benchmark run --help
```

## Agent 接入

Agent 通过 JSONL protocol 与 benchmark 通信。可直接使用：

- `examples/random_agent.py`：协议 smoke 测试；
- `examples/simplenav_agent.py`：SimpleNav；
- `examples/openai_compatible_vlm_agent.py`：OpenAI-compatible VLM；
- `examples/holoagent_agent.py`：HoloAgent 网关。

例如连接 OpenAI-compatible VLM：

```bash
VLM_BASE_URL=http://127.0.0.1:8000/v1 \
./run_benchmark.sh --agent-cmd \
  "python3 ${PWD}/examples/openai_compatible_vlm_agent.py --model my-vlm"
```

HoloAgent 的传感器、网关和服务配置见 [`docs/AGENT_INTEGRATION.md`](docs/AGENT_INTEGRATION.md)。

## 结果与分析

结果默认写入 `results/YYYYMMDD_HHMMSS_<agent>/`，包括 `run_config.json`、`run.log`、
`summary.json` 以及每个 session 的 JSONL 和指标文件。分析结果：

```bash
./.venv/bin/python scripts/analyze_result.py results/<run-directory>
```

`results/`、虚拟环境、缓存和大型数据资产默认被 `.gitignore` 排除。发布大型
`scene.usdz` 时请使用 Git LFS 或对象存储，并同时提供 manifest、SHA-256 和下载说明。

## 目录

```text
benchmark/   # 数据加载、Isaac Sim 环境、session runner、评测和协议
examples/    # agent 适配器
scripts/     # 环境安装、校验、运行和结果分析
dataset/     # 示例数据集或数据清单
docs/        # 协议、指标、数据卡和集成说明
tests/       # 不依赖 GPU 的测试
```

协议和扩展边界见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)。
