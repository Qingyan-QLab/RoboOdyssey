# RoboOdyssey｜万境长行

RoboOdyssey 是面向具身智能体的长期、多场景仿真与评测项目。“万境”指家庭、工业、商场、医院及室外等不同环境；“长行”强调智能体在持续执行中积累记忆、适应变化，并将经验带到后续任务。

## 设立背景

机器人需要在不断变化的真实场景中长期工作：导航、取物、问询等任务会跨越不同地点、时间和情境。一次性的单场景成功率难以反映它能否记住经历、理解环境变化、持续适应并完成物理执行。RoboOdyssey 因此以多环境、长时序任务为方向，建立可复现的执行、取证和评测流程，为研究长期记忆与持续适应提供基础。

当前仓库实现的是以 Isaac Sim 为基础的室内长程任务评测主流程；家庭、工业、商场、医院和室外等更广场景是项目的扩展方向，不代表现有数据集已全部覆盖。

在 Isaac Sim 中执行长程导航、搬运和问答任务。当前主流程直接读取 **schema 5／contextual_requests v3** 发布数据，
按真实世界状态修补请求和 GT，连续执行各 Session，再输出任务评分、运行审计和诊断视频。

数据生成端负责场景理解、任务生成、冻结与审核；Benchmark 独立负责数据读取、真实执行、证据采集、修补和评分。
运行顺序、正式配置、修补、评分与版本差异统一见 [运行与评分规则](docs/RULES.md)。

## 1. 安装与目录约定

Python 3.10+。版本记录见 [变更日志](CHANGELOG.md)。

从源码目录使用主机 Python 做预检和分析：

```bash
cd RoboOdyssey
./scripts/bootstrap.sh
./.venv/bin/python -m roboodyssey --help
```

Python 调用方使用 `roboodyssey` 包名。下列核心模块可在主机 Python 中导入：

```python
import roboodyssey
from roboodyssey.runtime.environment import RequestEnvironment
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
```

已有脚本请将 `benchmark` 导入前缀和 `python -m benchmark` 分别迁移到
`roboodyssey` 和 `python -m roboodyssey`。

Benchmark 的安装、校验、执行和测试均不依赖 Task Builder／Session Builder 源码。
数据独立发布，推荐布局为：

```text
tools/
├── OSS/scene000010/session.tar
└── RoboOdyssey/
```

Isaac Sim、SimpleNav 和模型权重分别安装在自己的环境中；主机环境只需 numpy、Pillow。
录像需要提供 CPU libx264 的 ffmpeg，视频回归还需 ffprobe。不要用主机 pip 安装仿真或模型全套依赖。
没有 Isaac／GPU 也能进行数据校验、预检、回归和已有结果分析。

## 2. 校验与预览数据（CPU）

本仓库不内置 dataset，也不生成任务或场景资产。请使用发布的 `OSS/<scene_id>/session.tar` 或已解包的场景集合目录，
`run` 和 `doctor` 必须显式提供 `--dataset`；`run_benchmark.sh` 也可通过 `DATASET_DIR` 指定。
把下面的数据路径替换成实际路径，场景资产和冻结目录保持只读。

```bash
./.venv/bin/python -m roboodyssey validate ../OSS/scene000010/session.tar
./.venv/bin/python -m roboodyssey doctor --dataset ../OSS/scene000010/session.tar
./.venv/bin/python -m roboodyssey run --dataset ../OSS/scene000010/session.tar --sequence seq_0001 --dry-run
```

首次读取 tar 时，在独立缓存解包，原始归档保持只读。默认缓存为
`${XDG_CACHE_HOME:-~/.cache}/longterm-indoor-benchmark/datasets/<归档SHA-256>/`，
该缓存路径沿用旧项目标识；可用 `BENCHMARK_DATA_CACHE` 指定有足够空间的位置；每次加载仍校验 manifest 和文件哈希。
支持的包内层级、版本和私有数据边界见 [发布数据契约](docs/DATA_FORMAT.md)。

场景集合可使用 `--sequence seq_0001`、重复 `--sequence` 或 `--all-sequences`；
校验整个集合不需要选择序列。选择范围和连续性要求见 [执行规则](docs/RULES.md#execution)。

## 3. 准备 SimpleNav 配置（CPU）

完整实验使用统一配置与管理入口；无需复制历史结果目录中的 supervisor。

```bash
cp configs/simplenav/managed.example.json configs/simplenav/managed.local.json
# 编辑数据、源码、checkpoint、Python、Isaac、GPU、端口和显示设置。
./.venv/bin/python -m roboodyssey preflight --config configs/simplenav/managed.local.json
```

- `model_service.mode=managed`：由本次实验启动模型、等待端口就绪、监控退出并清理自己创建的进程。
- `model_service.mode=external`：连接已经运行的服务，不启动或关闭该模型服务。
- `gpu` 是 Isaac/Vulkan 设备索引；`model_service.gpu_uuid` 是模型的 CUDA 设备 UUID。
- GUI 可继承当前 `DISPLAY`／`XAUTHORITY`，或在 `environment` 中显式配置，不硬编码本机桌面路径。
- `sequences: ["seq_0001", "seq_0002"]` 或 `all_sequences: true` 用于场景集合。
- 相对文件路径按配置文件目录解析。示例中的占位路径必须替换；预检失败不会启动进程。

预检只验证归档、资产、源码、包可发现性、动作统计、编码器和输出目录；不会启动 Isaac、
导入模型、连接服务或占用 GPU。外部服务的实际权重身份由服务管理者保证。
配置细节、启动边界和复现信息见 [实验管理](docs/LAUNCH.md)。

## 4. 执行实验（真实 GPU）

```bash
./.venv/bin/python -m roboodyssey experiment --config configs/simplenav/managed.local.json --execute
```

不带 `--execute` 时只做 CPU 预检。真实执行会保存配置／来源、按需启动模型、逐序列启动 Isaac 与 Agent，
结束后生成报告并清理进程。`Ctrl-C`／SIGTERM 或在实验根目录创建 `STOP` 文件请求终止。
SimpleNav 的能力与录像方式见 [接入说明](docs/SIMPLENAV_V5.md)。

其他 Agent 可直接使用通用入口，Isaac 启动器路径需替换：

```bash
/path/to/isaacsim/python.sh -m roboodyssey run \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 \
  --agent-cmd "python3 /absolute/path/to/my_agent.py" \
  --max-steps 2000 --viewer --record-video --output results
```

通用入口可使用 `--official`，约束和适用范围见 [正式配置](docs/RULES.md#official)。
所有选项见 `python -m roboodyssey run --help`；`run_benchmark.sh` 仍支持旧启动方式。

## 5. 查看与重建报告（CPU）

受管理实验默认生成 `reports/index.html`。任何时候可从完整实验根目录、单次运行目录重建到新目录：

```bash
./.venv/bin/python -m roboodyssey analyze results/<run-or-experiment> --output results/<new-analysis>
```

也支持旧 `baseline/<timestamp>_simplenav` 运行目录；不修改原日志或原完成状态。
统一入口按格式生成修补审计、逐任务 GT／任务分、导航距离／进度、路径／碰撞和逐次 reset 对照，
含 HTML、JSON、CSV 和视频跳转。缺失证据会列为分析错误，不悄悄减少分母。

新运行使用 `progress_path_v3`：任务分统一为 0–100，导航按最终有序进度×路径效率计分，
搬运按剩余交付成本的进展×执行成本效率计分，问答只按答案正确性计 100／0，没有问答过程分。
完整成功率独立统计。最终进度、最佳进度各占一列。旧运行保留原评分版本，具体定义见下方规则文档。

明确需要用新版重算历史运行时，使用同一入口：

```bash
./.venv/bin/python -m roboodyssey analyze results/<single-run> --rescore \
  --reference-dataset /path/to/original-frozen-dataset --output results/<new-rescored-report>
```

已有评分地图时可省略 `--reference-dataset`。历史地图重建的验证范围和限制见 [重算规则](docs/RULES.md#rescoring)。

评分口径见 [运行与评分规则](docs/RULES.md)，文件位置见 [输出文件契约](docs/OUTPUTS.md)。

## 6. 输出结构

受管理实验使用结果布局 v2：

```text
results/experiment_<time>_<id>/
├── manifest.json             # 格式版本及序列目录索引
├── config.json / preflight.json / provenance.json
├── status.json / progress.json / processes.json / runs.json
├── summary.json              # 按序列汇总，异常不可被忽略
├── logs/                     # 模型、各序列 Isaac 启动日志
├── sequences/<sequence_id>/
│   ├── manifest.json / run_config.json / provenance.json
│   ├── status.json / summary.json / logs/
│   ├── sessions/<session_id>/
│   │   ├── metrics.json / name_ledger.json
│   │   ├── traces/           # 请求、GT、动作、历史、mutation、错误
│   │   ├── evaluation/       # 每任务真实碰撞地图及校验哈希，供 CPU 复算
│   │   └── video/            # review.mp4、索引和编码记录
│   └── reports/              # 自动修补审计
└── reports/                  # 统一报告索引及每序列评分诊断
```

通用单次 `run` 也使用 v2，但 Session 位于该运行的 `sequences/<sequence_id>/sessions/`。
旧格式仍按 `scenes/<scene_id>/sessions/` 读取；旧实验文件不会被迁移或覆盖。
原始证据长期保留，`reports/` 可重新生成。字段来源与版本兼容见 [OUTPUTS.md](docs/OUTPUTS.md)。

## 7. 验证和扩展

```bash
./.venv/bin/python -m roboodyssey check
```

该命令禁止导入 torch、isaacsim、omni 等模型／仿真 SDK，仅执行 CPU 回归。
测试使用仓库内冻结的合成样例和临时目录，不需要外部数据或生成工具；完整回归需要 ffmpeg／ffprobe。
检查器同时禁止导入两套 Builder，CI 执行完整 CPU 回归与文件清单检查。
`python -m roboodyssey check --core` 保留为快速子集，完整 v5、SimpleNav 与启动管理验收使用默认 `check`。
真实渲染取证检查和受控失败诊断需另行运行，见 [SimpleNav 专题](docs/SIMPLENAV_V5.md)。

| 目录 | 职责 |
|---|---|
| `roboodyssey/launch/` | 配置、预检、服务管理、多序列启动、来源记录 |
| `roboodyssey/data/`、`contracts/` | 发布归档、manifest、冻结语言、实际状态修补及几何计算 |
| `roboodyssey/artifacts/` | 输出版本、目录索引、兼容路径与证据引用 |
| `roboodyssey/validation/` | 冻结数据校验 |
| `roboodyssey/agents/`、`examples/` | 协议、Agent 进程及模型适配 |
| `roboodyssey/scenes/`、`viewer/` | 数据读取、Isaac 环境、取证和显示 |
| `roboodyssey/runtime/` | 请求执行、连续历史、事件与录像 |
| `roboodyssey/evaluation/` | 完整 GT 判定、汇总与诊断计算 |
| `roboodyssey/reporting/` | HTML／CSV／JSON、修补审计与报告索引 |
| `roboodyssey/legacy/` | 旧 v4 修补和历史契约 |
| `scripts/` | 安装、HoloAgent 启动及不能由普通运行代替的专项检查 |
| `tests/` | CPU 回归，覆盖协议、评分、修补、历史、录像和启动管理 |

完整目录及每个文件的内容/作用见 [项目文件清单](docs/FILE_MAP.md)。新增或移动文件后执行：

```bash
python -m roboodyssey inventory          # 生成清单
python -m roboodyssey inventory --check  # 检查遗漏/过期，不修改文件
```

常用流程统一使用 `python -m roboodyssey`。原验证、预检和单报告包装脚本已移除，分别使用
`validate`、`preflight / experiment` 和 `analyze`。`generation/` 已移除，生成职责由外部 Builder 承担。
文件用途集中维护于清单生成器，无需重复维护目录说明。

内部辅助模块已按职责合并：语言／日历／名称位于 `contracts/language.py`，
对象与历史指代位于 `contracts/references.py`，路线与连通性位于 `contracts/navigation.py`，
计分分母和新旧汇总统一位于 `evaluation/report.py`。历史导入路径由包入口集中映射到同一实现，
完整映射见 [文件清单](docs/FILE_MAP.md#保留的导入路径)。模型接口、数据与评分版本、命令和检查流程继续保留。

`scripts/` 仅保留以下 6 个有独立用途的入口：

| 脚本 | 保留用途 | 执行环境 |
|---|---|---|
| `bootstrap.sh` | 创建最小主机环境 | CPU |
| `check_simplenav_cpu.py` | `benchmark check` 使用的模型/仿真导入隔离检查器 | CPU |
| `check_runtime_tasks.py` | 对外部旧 schema 4 归档进行三种失败情形重放并输出审计 | CPU |
| `check_request_evidence.py` | 检查真实渲染的新帧、可见像素和世界状态 | Isaac / GPU |
| `run_request_diagnostics.py` | 使用完整发布序列注入拿起后失败、原地释放、初始已满足的受控实验 | prepare-only 为 CPU；实跑为 GPU |
| `run_holoagent.sh` | HoloAgent / VLM 服务与仿真联合启动 | 模型 / Isaac / GPU |

历史实验及整理验证的说明随对应 `results/` 保存，`docs/` 只保留当前使用和契约文档。

模型适配保留 `roboodyssey/agents/` 实现、`examples/` 协议入口、`configs/<model>/` 参数配置的结构。
当前仅发布 `configs/simplenav/` 下的配置模板；以后新增模型配置也统一放在 `configs/<model>/`，本机配置使用 `*.local.json`。
SimpleNav 的适配器和 managed／external 服务启动均保留；新增模型按 [Agent 接入](docs/AGENT_INTEGRATION.md) 扩展，
数据契约、任务修补和评分由 Benchmark 统一维护。

[架构说明](PROJECT_GUIDE.md) · [Agent 接入](docs/AGENT_INTEGRATION.md) · [协议](docs/PROTOCOL.md) ·
[运行与评分规则](docs/RULES.md) · [发布数据契约](docs/DATA_FORMAT.md)

`results/`、本机 `.local.json` 配置、虚拟环境、缓存、权重和大型资产不纳入源码。
来源哈希可核对版本，不能保证不同驱动、物理引擎或随机模型的结果逐位相同。

## 第三方许可与资产

第三方组件与外部资产的声明见 [NOTICE](NOTICE)。从生成端移入的运行时辅助代码保留原有的 [MIT 许可](roboodyssey/contracts/LICENSE)。
Isaac Sim、外部模型与另行发布的场景数据适用各自的许可和分发条款。
