# 结果文件契约

本页只维护文件格式、位置和证据引用；执行与评分定义见 [RULES](RULES.md)。
输入 dataset schema、运行时任务契约和**结果目录布局**是三个独立版本。
新 v5 运行根部 `manifest.json` 使用 `artifact_schema=benchmark_results`、`schema_version=2`。
没有该文件的历史运行按旧 `scenes/<scene>/sessions/<session>` 布局读取，不改变旧文件。
旧 schema 1／4 执行仍写旧结果布局。

## 根目录类型

| kind | 来源 | 子目录 |
|---|---|---|
| `experiment` | `benchmark experiment --execute` | `sequences/<id>/` 是每条序列的独立运行根目录 |
| `collection` | `benchmark run --all-sequences`／重复 `--sequence` | 同上，不管理外部模型服务 |
| `run` | 单次运行或实验子运行 | `sequences[].sessions_root` 指向各场景的 Session 目录 |

实验的旧独立 v5 归档没有 sequence_id 时使用 `default_000`；普通单次运行使用 `default`。
不要通过目录名推断场景或契约，使用 manifest 中的身份与相对路径。
实验／集合的 `runs` 列出所有计划子运行，未完成的子运行也不能从统计中消失。

## 配置、来源与状态

| 文件 | 含义 |
|---|---|
| `config.json` | 实验解析后的绝对路径、模型和执行配置 |
| `preflight.json` | CPU 检查及未执行的检查项；不是 GPU 可运行性的证明 |
| `run_config.json` | 子运行的数据哈希、序列身份、相机／动作配置、observation_policy、metric_policy、正式配置标记和缓存路径 |
| `launch_config.json` | 单次运行在导入 Isaac 前记录的 argv、数据身份和 Python；初始化失败时也能追溯 |
| `provenance.json` | 代码 revision／dirty 状态／文件哈希、Python 包；实验额外记录模型源码、权重和动作统计哈希 |
| `processes.json` | 自己启动的模型进程或外部服务声明 |
| `runs.json` | 实验各子运行的 argv、PID、退出码、summary 和目录 |
| `status.json` | phase、execution、summary／analysis、cleanup 及各阶段错误 |
| `progress.json` | 从完整事件行增量统计的进度；不是最终评分依据 |
| `summary.json` | 任务／序列汇总及有效性；分母和聚合方式见 [汇总规则](RULES.md#aggregation) |
| `logs/` | 模型、Isaac 启动器、Benchmark 运行日志 |

启动前配置和预检先落盘。模型／仿真启动失败、汇总失败和分析失败分别保留；不能因为任务
已经处理完就把整个实验标为成功。Isaac 的 `app.close()` 可能直接退出解释器，子运行
`cleanup=simulator_close_requested` 表示请求收尾，父管理器的 `cleanup=completed` 表示已回收所管理进程。
未受管理的单次运行保留此区别。强制 SIGKILL 无法保证完整收尾。

`provenance.json` 保存核对信息，不自动打包模型或源代码；复现时仍须取得对应资产、代码和环境。
dirty 源码哈希不能替代保存该源码版本。已有外部模型服务的权重身份不能仅由 TCP 端口证明。

## Session 原始证据

| 路径 | 职责 |
|---|---|
| `metrics.json` | 紧凑的逐任务、逐目标结果与 Session 有效性 |
| `name_ledger.json` | 已实际下发的名称账本 |
| `traces/runtime_tasks.jsonl` | 原请求／GT、实际下发前状态、运行时 GT、修补步骤和来源 |
| `traces/agent_io.jsonl` | 实际请求尝试、响应、错误及采样／逐帧发送事件，图像保留审计摘要 |
| `traces/action_changelog.jsonl` | 每个动作的前后位姿、结果、状态变化、来源与事件 ID |
| `traces/runtime_history.jsonl` | 真实任务边界快照、观察／位置经历、未下发决策 |
| `traces/runtime_mutations.jsonl` | mutation 计划、实际施加与回读结果 |
| `traces/runtime_events.jsonl` | 带 UTC／单调时间的显示与录像事件 |
| `traces/request_outcomes.jsonl` | 成功、失败、排除或中断及实际变化 |
| `traces/integration_errors.jsonl` | 接入或收尾错误；按需创建 |
| `evaluation/<task_id>.npz` | 真实碰撞网格、原点、分辨率与半径；搬运还保存可移除目标占据、房间标签、真实对象原点、抓放参数及初始状态，用于 CPU 评分复算 |
| `video/review.mp4` | 事件采样视频，不是墙钟连续录屏 |
| `video/video_index.jsonl` | 事件 ID、视频时间、帧区间和真实时间映射 |
| `video/video_manifest.json` | 录像参数、时间线类型、完成状态 |
| `video/video_encoder.log` | 编码日志；异常时可能只有 partial.mp4 |

没有执行动作、mutation 或错误时，相应文件可以不存在。任务日志、结果与指标仍须足够证明
完成状态；分析器不会把缺少动作记录默认为零步成功。

新 runtime_tasks 的 metric_reference 包含上述快照的相对路径、SHA-256 和 grid_geodesic_8 距离类型。
request_outcomes 和 metrics.task_results 的 evaluation 保存评分版本、score／score_max、逐目标分、
网格路径基准、进度、路径效率、reset 计费及 measurements；task_timing 保存单调时钟边界、请求次数与耗时。
搬运的 metric_reference 标注 `kind=transfer_cost`、成本单位与标准移动步长；
evaluation.transfers 保存逐目标 J0、J、C、进展、效率、得分、动作后时间线和抓放尝试数。
私有权威对象状态的 interaction_position 是实际 USD 原点；action_changelog.state_changes
记录该字段与包围盒、持物状态的前后变化，不发送给 Agent。
搬运动作另存 transfer_collision_before_sha256／transfer_collision_after_sha256，
核验移除／放回目标后重建的占据和通行半径与实际环境相同；不为每步重复保存完整地图。
action_changelog 另存每次 request_seconds 和 action_seconds。
离线分析校验参考哈希、动作／状态变化和结果中的评价一致性。旧结果没有这些字段时保留原二元评分，
不把旧 reference_steps 或前序全成功位置变成新的实际基准。

汇总 score_100 为 0–100；task_score 为相同得分的 0–1 表示，task_success_rate 为独立完整成功率。
metric_version、metrics_complete 和 tasks_missing_metric 必须随分数读取；部分任务指标缺失不能减少分母。
step_efficiency_score／time_efficiency_score 缺少已验证的 GT 成本时为 null，不补成 0 或 100。

`agent_io.jsonl` 的 `direction=request/response/error` 保持原义。
`direction=observation`、`request_type=observation` 的记录包括：`event=sample` 的 frame_id、
采样时刻、延迟和漏采数；`event=send_attempt` 的实际 observation 消息（图像为摘要）；
`event=sent` 的 frame_id 和 sent_monotonic_s，表示完整消息已写入管道。
新流程没有 drop 事件或 latest_only 缓存；旧日志中的这些字段保留原义。
独立 observation 不计动作步数，不触发新的动作响应。发送失败以 integration_errors 和运行错误收尾。
真实请求的 `payload.observation.timing` 对应使用的采样帧及下发时刻，字段见 [协议](PROTOCOL.md#观测与时间)。
结合 sample 事件计算实际采样率，合并 request 中的初始帧和 sent 中的持续观测计算下发率；配置 fps 不能替代实测值。
响应／v5 动作日志的 observation_frame_id 限定该次评分可使用的观察证据截止帧；request_id 关联请求和响应。

新 `trace_schema=2` 将结果／历史中的重复 `actions` 替换为：

```json
{"action_evidence":{"scope":"session","file":"traces/action_changelog.jsonl","task_id":"task_000","count":2000}}
```

路径相对 Session 根目录，以 task_id 过滤，并核对数量与 task_step 顺序。
权威 before/after 快照和观察证据继续保留；内存里的运行历史不压缩。
动作依然可能作为带时间戳的事件出现于 runtime_events，这是录像／进度索引的用途，不可随意删除。

## API Agent 配套产物

API Agent 将配套日志写入配置的 `log_dir/<instance_id>/`，例如
`results/gpt6_astra_api_diagnostic/agent/<instance_id>/`；对应 benchmark 输出可位于同一实验父目录的
`benchmark/<自动运行目录>/`。它不属于结果 manifest 的 Session 证据路径，不改变现有评分契约。
启动时 Agent 将绝对日志目录写入 stderr，运行器的运行日志保留该行，作为外部产物关联入口。

| 文件 | 内容 |
|---|---|
| `config.json` / `prompt.txt` | 实际解析配置与系统提示词；配置只记录密钥环境变量名。 |
| `events.jsonl` | Session 握手与接入错误，含可用的 run_id 和请求身份。 |
| `calls.jsonl` | 每次调用的实例／请求身份、父响应 ID、配置／提示词／输入哈希、唯一当前帧号及 RGB 哈希、prompt／初始状态是否本轮上传、上传文本字符数、延迟、错误、响应标识、模型名和 usage；有效响应保存动作／答案。 |
| `summary.json` | Agent 客户端状态、输入策略／反馈档位、调用和图像上传计数、prompt／初始状态上传次数、响应链 run_id、已知用量累计、缺失用量次数、调用累计延迟和未核实费用 `null`。 |

帧清单只含本次当前帧，不再保存显式记忆；日志不保存发送的完整 PNG 或服务端错误正文。
`run_id + scene_id/session_id/task_id/request_id` 用于关联公开请求；旧归档缺少 run_id 时结合实例目录和启动日志定位。
Agent 的 `client_status=closed` 仅代表客户端收尾，不证明 benchmark 有效或完成。
缺失用量不能按零费用解释。usage 累计的是 API 返回的计量 token，可能包括再次计量的会话历史；
上传字符数和上传图像计数不能代替计费 token。deadline 到达后远端请求可能仍计费，
迟到结果不更改响应链或日志。此模式不自动重试，避免图像和未知完成状态的请求被重复发送。
费用估算及启动方式见 [API Agent](OPENAI_API_AGENT.md)。

## 可重建报告

`python -m benchmark analyze <run-or-experiment> --output <新目录>` 在 CPU 上生成：

- `index.html`、`analysis.json`：入口、来源与分析错误。
- `repairs/request_audit.html|json`：原文、实际请求、GT、失败来源和视频。
- `scores/report.html`、`scores.json`、`summary.md`：任务分与导航诊断。
- `scores/tasks.csv`、`navigation_goals.csv`、`transfer_goals.csv`、`resets.csv`：可导出明细。
- 集合的 `sequence_NNN/` 与 `aggregate.json`：逐序列报告和原汇总副本。

自动修补报告位于子运行 `reports/request_audit.*`；统一分析可位于实验 `reports/` 或显式新目录。
新旧运行都使用统一 analyze 入口重建到新目录，避免覆盖已发布报告。
报告可重建，原始日志不可由分析器改写。移动结果时同时保留视频相对目录；来源 JSON 保留原绝对路径供追溯。

显式 `--rescore` 报告另存 rescored、original_task_score／original_metric_version，
summary 含逐 Session／任务类型分组、网格评分与记录耗时汇总。
使用 `--reference-dataset` 时，`scores/references/<scene>/<session>/evaluation/<task>.npz`
保存离线重建地图，reference_reconstruction 说明验证与来源，reference.original_live_snapshot 为 false。
历史时间另存 recorded_task_span_seconds、recorded_agent_roundtrip_seconds；缺失动作计时不补写。

新版启动/目录由 CPU 夹具验证。真实历史运行的离线数值回归不等价于重新运行模型或物理实验。
不同契约的分析支持范围见 [旧版本兼容](RULES.md#legacy)。
