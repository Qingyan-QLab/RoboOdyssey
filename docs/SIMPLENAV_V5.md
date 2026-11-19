# SimpleNav v5 接入与全过程审阅

新实验推荐 [README](../README.md) 与 [统一实验管理](LAUNCH.md)：
`python -m benchmark preflight --config ...` →
`python -m benchmark experiment --config ... --execute` →
`python -m benchmark analyze <运行目录>`。支持 managed／external 模型服务、参数化 GPU、进程清理和报告。
本页只说明 SimpleNav 适配和专项取证；通用 [运行与评分规则](RULES.md) 及 [输出文件契约](OUTPUTS.md) 分别维护。

## 当前接入行为

`examples/simplenav_agent.py` 使用 `benchmark/agents/simplenav.py` 的可注入状态机。
旧版 `navigation` 与 v5 `long_horizon` 都能进入导航预测。v5 依据公开
`capabilities` 路由，不读取私有 GT、模板、房间／对象 ID、修补结果或 overview。

入口共用 `benchmark/agents/stream.py`，接收线程在模型预测期间持续收图。
每张已收到的图像进入 Agent 的视觉历史，默认按 history_frames 保留最近 8 帧，
后续模型调用使用该历史快照和当前图；WebSocket 预测仍一次一调用，不会为每张新图排队推理。
这是 SimpleNav 的记忆／输入选择策略，Benchmark 按 [观测规则](RULES.md#observations) 逐帧发送。

SimpleNav 当前只支持导航；包含 QA 或 manipulation 的请求返回 `agent_status: unsupported`。
连接失败、预测异常或格式错误返回／触发 `integration_error`，两者的计分和有效性处理见
[结果分类](RULES.md#exclusions)。它无法单独完成全部 41 条模板对应的能力。

同一场景跨 Session 保留图像历史、公开名称定义以及最近 32 条已下发请求的文本和
情境时间；请求记录明确不代表成功经历。场景变化清空这些内容。新序列使用新 Agent
进程／隔离记忆。v5 不用附近物体清单自动开门；旧版的自动开门行为仅保留在旧协议。
碰撞恢复默认连续 5 次受阻后尝试 reset，可通过 collision_reset_threshold 调整（0 关闭），并遵守 allowed_actions。

名称账本和状态修补遵守 [统一修补规则](RULES.md#repair)。

## CPU 准备流程

在 `longterm_benchmark/` 目录，用普通主机 Python 执行；不要为这些步骤使用 Isaac
启动器。主机依赖为 numpy、Pillow，视频测试另需 ffmpeg／ffprobe 和 CPU libx264。

```bash
cp configs/simplenav/review.example.json configs/simplenav/review.local.json
# 编辑 local 配置中的 checkpoint、Python、SimpleNav 源码和数据集路径。
CUDA_VISIBLE_DEVICES='' python -B -m benchmark preflight \
  --config configs/simplenav/review.local.json --report /tmp/simplenav_preflight.json
CUDA_VISIBLE_DEVICES='' python -B -m benchmark check
```

相对路径按配置文件所在目录解析。示例 checkpoint 是占位路径，预检会明确失败；
不得把占位配置的失败当成已具备模型运行条件。`--report` 要求新文件，避免覆盖记录。

默认预检只做冻结数据契约及完整性校验、资产哈希、源码和统计文件检查、主机 Python 包可发现性、
CPU 编码器及输出空间检查。统计文件选择与 SimpleNav `ActionCodec` 一致，多数据键
需配置 `unnorm_key`。检查包时不导入 torch。预检不启动 Isaac、不连接 WebSocket／
显示服务、不加载权重，也不验证服务可用性或渲染质量；输出包含 `not_executed`。
CPU 回归脚本额外阻止导入 torch、isaacsim、omni 等库。

真实运行使用 `python -m benchmark experiment --config ... --execute`。模型服务已由外部启动时，
配置 `model_service.mode=external`；否则使用 managed 配置。此步骤会启动 Isaac / Agent 并使用 GPU。
配置默认开启 GUI 和录像；选择范围遵守运行规则。

`gpu` 对应 Isaac / Vulkan 设备索引；模型 GPU 使用 `model_service.gpu_uuid`，
显示使用 DISPLAY / XAUTHORITY。两种设备编号不作自动换算，详细配置见 [LAUNCH](LAUNCH.md)。

私有受控诊断可独立运行：

```bash
# 只校验并选择现有完整序列，不加载 Isaac 或模型。
python -B scripts/run_request_diagnostics.py \
  --source ../OSS/scene000010/session.tar --sequence seq_0001 \
  --output results/request_diagnostics_new --prepare-only
# 真实 GUI／GPU；输出目录中的各 case 必须尚不存在。
../../python.sh -B scripts/run_request_diagnostics.py \
  --source ../OSS/scene000010/session.tar --sequence seq_0001 \
  --output results/request_diagnostics_new --gpu 0
```

诊断在原始完整序列中选择首个 Session 的单目标搬运，注入拿起后失败、原地放下或初始已满足三种情况；
其他任务返回 unsupported。须提供含合法放置几何的相应数据，案例不符合预期时明确失败。
脚本使用私有状态控制初始位置和动作，不生成或裁剪任务，不作为模型成绩。CPU 的实际历史／fallback 回归另由 `benchmark check` 覆盖。

## 可视化与视频

runner 向 Viewer 和私有事件日志发送 Session 开始、mutation、修补决策、任务开始、
等待 Agent、每步动作、reset、任务结果及接入错误。reset 的允许范围和统计见 [诊断规则](RULES.md#diagnostics)。

运行选项：`--record-video`、`--video-layout front_overview|front`、`--video-fps 10`、
`--video-event-seconds 0.5`、`--ffmpeg /usr/bin/ffmpeg`。不开录像时不创建额外 overview 传感器。

每个 Session 输出一个 `review.mp4`，画面为第一视角、可选 overview、事件和实际任务
文本。overview 只用于私有审阅。编码固定使用 CPU libx264；真实运行的相机渲染仍需要
Isaac 的 GPU。录像模式下 GUI 保留第一视角窗口，overview 使用独立传感器录入视频，
避免额外 GUI viewport 的重复渲染。观察取证使用零时间步同步渲染，不推进物理世界；
帧新鲜度由完成渲染序号判断，不依赖暂停时保持不变的仿真时间戳。

这是**事件采样视频**：状态事件读取相机帧，文字事件复用最近帧，每个事件保持指定
时长；模型等待会被压缩，不是按墙钟时间连续录屏。`video_index.jsonl` 将事件 ID、
Session／Task、真实 UTC／单调时钟、帧区间、视频秒数以及是否采到新帧关联起来。
`video_manifest.json` 明示 `timeline: event_sampled`、完成状态和时长。

普通异常及 Ctrl-C 会尝试收尾视频、保留中断状态并关闭资源。编码失败写入
`video_encoder.log` 并使运行无效；强制杀进程可能只留下 `review.partial.mp4` 和索引，
不能保证可播放。缺少所选相机帧会报错，不用空白帧冒充真实相机。

## 失败、状态变化与后续替换的证据

新 v5 运行在 `reports/` 自动生成私有 `request_audit.json` 和 `request_audit.html`；旧布局位于运行根目录。也可在 CPU 上重建：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m benchmark analyze results/<run-directory> --output results/<new-analysis>
```

报告按 Task 串联原请求、修补后请求、实际尝试下发内容、原 GT／运行时 GT、来源任务
及其真实变化、修补原因、后续执行结果和视频时间链接。下发记录来自 Agent I/O 日志，
表示已尝试调用 Agent，不保证远端收到；相机初始化失败前只有修补结果的任务不会被
错误标为已下发。排除和失败因果要求见运行规则。

原始证据逐文件说明见 [OUTPUTS](OUTPUTS.md#session-原始证据)。
报告的 incomplete_tasks 统计已产生决策但没有完成结果的任务，包括中断记录；
未到达的后续任务不在该项计数中，不能用它替代评分分母。损坏／截断日志列为审计错误。

## 专项验证范围

CPU 回归覆盖 v5 导航分流、公开名称和跨 Session 记忆、能力不支持、后端错误、
拿起后失败、放错区域、因果排除、来源已满足、mutation 回读、编码失败和中断收尾。
合成相机帧通过 CPU 编码／解码验证视频及索引；这不证明真实渲染正常。

真实帧取证需在 Isaac 环境显式执行：

```bash
/path/to/isaacsim/python.sh scripts/check_request_evidence.py \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 --gpu 0
```

该脚本检查相机新帧、可见像素、观测不改变世界状态，以及归档对应的实际 USD 状态。
受控诊断检查失败后的修补和评分，不代表模型能力。各次实验观察、使用的旧命令和
验证计数随对应 `results/` 保存，不作为当前使用规则。
