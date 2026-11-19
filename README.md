# Long-Term Indoor Benchmark

基于 Isaac Sim 的长程室内任务评测框架，支持导航、物体搬运和问答。
同一序列中的任务与 Session 连续执行，根据实际世界状态和执行经历修补请求与 GT，
记录观测、动作和状态变化，生成任务评分、审计报告和诊断视频。

正式数据单独发布，Benchmark 直接读取 `OSS/<scene_id>/session.tar` 或解包目录。
安装、运行和测试均不依赖 Task Builder／Session Builder 源码。

## 1. 安装

主机环境需要 Python 3.10+，用于数据校验、运行预检、结果分析和 CPU 测试。

```bash
git clone https://github.com/Qingyan-QLab/CL_NavBenchmark.git longterm_benchmark
cd longterm_benchmark
./scripts/bootstrap.sh
./.venv/bin/python -m benchmark --help
```

安装脚本创建 `.venv` 并安装 numpy、Pillow。下文命令均从项目根目录执行。

真实任务执行还需要单独安装 Isaac Sim 及其 GPU 运行环境；模型源码、权重和推理依赖安装在各自的模型环境中。
SimpleNav 接入需要准备源码、checkpoint 及配套的 `dataset_statistics.json`。
录像需要带 `libx264` 编码器的 ffmpeg，完整 CPU 测试还需要 ffprobe。

## 2. 准备数据

取得单独发布的数据包，按以下结构放置；示例路径可替换为实际位置：

```text
workspace/
├── OSS/
│   └── scene000010/
│       └── session.tar
└── longterm_benchmark/
```

校验数据与主机依赖，再预览要执行的序列。这些命令均在 CPU 上运行：

```bash
./.venv/bin/python -m benchmark validate ../OSS/scene000010/session.tar
./.venv/bin/python -m benchmark doctor --dataset ../OSS/scene000010/session.tar
./.venv/bin/python -m benchmark run \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 --dry-run
```

`seq_0001` 应替换为数据中的实际序列 ID。重复 `--sequence` 可选择多条序列，
使用 `--all-sequences` 可执行包内全部序列；校验整个数据包无需选择序列。
当前发布数据采用 schema 5、`contextual_requests` v3，包内结构见 [数据格式](docs/DATA_FORMAT.md)。

无需手动解包 tar。首次读取会按归档 SHA-256 解包到
`${XDG_CACHE_HOME:-~/.cache}/longterm-indoor-benchmark/datasets/<SHA-256>/`，
后续读取复用缓存并校验 manifest 和文件哈希。原始数据保持只读，结果写入独立目录。
需要更换缓存位置时，在运行前设置：

```bash
export BENCHMARK_DATA_CACHE=/path/to/dataset-cache
```

## 3. 运行

### SimpleNav

复制配置模板，并填写本机路径与设备设置：

```bash
cp configs/simplenav/managed.example.json configs/simplenav/managed.local.json
```

| 配置项 | 填写内容 |
|---|---|
| `dataset`、`sequences` | 数据包路径和实际序列 ID；执行全部序列时改用 `all_sequences: true` |
| `simplenav_root`、`checkpoint` | SimpleNav 源码目录和模型权重路径 |
| `agent_python`、`isaac_python` | 模型环境的 Python 和 Isaac Sim 的 `python.sh` |
| `output` | 实验结果的保存父目录 |
| `url`、`unnorm_key` | 模型服务的 WebSocket 地址和动作统计键 |
| `gpu`、`model_service.gpu_uuid` | Isaac/Vulkan 设备索引和模型使用的 CUDA GPU UUID |

配置中的相对路径按配置文件所在目录解析。将模板中的占位路径和 GPU UUID 替换为实际值，
并按需设置 `viewer`、`record_video`。本机配置使用 `*.local.json`，不纳入版本控制。

默认 `model_service.mode=managed`，由实验入口启动并管理 SimpleNav 模型服务。
连接已启动的服务时设为 `external`。GUI 所需的 `DISPLAY`、`XAUTHORITY` 可通过 `environment` 配置。
完整字段说明见 [实验配置](docs/LAUNCH.md)，模型能力与输入方式见 [SimpleNav 接入](docs/SIMPLENAV_V5.md)。

先进行 CPU 预检：

```bash
./.venv/bin/python -m benchmark preflight --config configs/simplenav/managed.local.json
```

预检检查数据、资产、路径、动作统计、编码器和输出目录，不启动模型或 Isaac。
通过后执行真实实验：

```bash
./.venv/bin/python -m benchmark experiment --config configs/simplenav/managed.local.json --execute
```

实验入口保存配置与来源信息，按所选序列启动 Isaac 和 Agent，记录运行状态，结束后生成报告并清理进程。
不带 `--execute` 时仅做预检。运行中可按 `Ctrl-C`，或在实验根目录创建 `STOP` 文件请求终止。

### 其他 Agent

按照 [Agent 接入说明](docs/AGENT_INTEGRATION.md) 实现统一协议，通过通用入口执行。
将以下 Isaac 启动器、Agent Python 和脚本路径替换为实际路径：

```bash
/path/to/isaacsim/python.sh -m benchmark run \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 \
  --agent-cmd "/path/to/agent/python /absolute/path/to/my_agent.py" \
  --max-steps 2000 --viewer --record-video --output results
```

通用入口的全部参数见 `python -m benchmark run --help`。
正式评测的 `--official` 配置要求、执行连续性和预算见 [运行与评分规则](docs/RULES.md)。

## 4. 查看结果

实验入口会打印本次输出目录，默认在 `results/` 下创建独立实验目录。
保持配置中的 `auto_analyze: true`，完成后用浏览器打开该目录的 `reports/index.html`。
主要文件如下：

```text
results/experiment_<time>_<id>/
├── config.json / preflight.json / provenance.json  # 配置、预检和来源
├── status.json / progress.json                    # 完成状态和运行进度
├── summary.json                                  # 序列汇总与有效性
├── logs/                                         # 模型和 Isaac 日志
├── sequences/<sequence_id>/
│   ├── summary.json / status.json
│   ├── sessions/<session_id>/
│   │   ├── metrics.json                           # 逐任务结果
│   │   ├── traces/                                # 请求、GT、动作和实际状态记录
│   │   ├── evaluation/                            # 评分参考与校验信息
│   │   └── video/                                 # review.mp4 与事件索引
│   └── reports/                                  # 请求修补审计
└── reports/index.html                            # 实验报告入口
```

报告包含任务评分、完整成功率、修补后的请求与 GT、导航进度、搬运成本、路径、碰撞和 reset 明细，
并提供 HTML、JSON、CSV 及录像跳转。启用录像时，视频保存在各 Session 的 `video/` 下。

任务分使用 `progress_path_v3`，统一为 0–100：导航按最终有序进度和路径效率计分，
搬运按交付成本进展和执行成本效率计分，问答按答案正确性计 100／0。
查看分数时同时核对 `status.json`、`summary.json` 中的完成状态、`evaluation_valid` 和指标完整性。
评分定义见 [运行与评分规则](docs/RULES.md)，文件字段见 [输出说明](docs/OUTPUTS.md)。

需要重新生成报告时，可对实验目录或单次运行目录进行 CPU 分析。
将下面的占位路径替换为实际目录，输出目录须尚不存在：

```bash
./.venv/bin/python -m benchmark analyze 'results/<run-or-experiment>' \
  --output 'results/<new-analysis>'
```

分析完成后打开新目录中的 `index.html`。报告可从完整原始日志重建；
保存或移动结果时应一并保留日志、评分参考和视频。

## 检查与开发参考

运行完整 CPU 测试：

```bash
./.venv/bin/python -m benchmark check
```

测试使用仓库内的合成样例，无需外部数据、模型或 Isaac Sim。
模型配置统一放在 `configs/<model>/`，当前提供 SimpleNav 模板；适配实现与协议入口位于 `benchmark/agents/` 和 `examples/`。

[项目架构](PROJECT_GUIDE.md) · [文件清单](docs/FILE_MAP.md) · [数据格式](docs/DATA_FORMAT.md) ·
[实验配置](docs/LAUNCH.md) · [Agent 接入](docs/AGENT_INTEGRATION.md) · [协议](docs/PROTOCOL.md) ·
[API Agent](docs/OPENAI_API_AGENT.md) · [运行与评分规则](docs/RULES.md) · [输出说明](docs/OUTPUTS.md)
