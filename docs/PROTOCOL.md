# Agent Protocol v1.1

runner 和 Agent 通过 stdin/stdout 交换 JSONL，协议版本为 1.1。每行 stdout 是一个 JSON 对象，
日志写入 stderr。完整响应 schema 见 [protocol_schema.json](../benchmark/agents/protocol_schema.json)。
本页维护消息结构；执行顺序、公开信息边界、动作限制和结果处理统一见 [运行与评分规则](RULES.md)。

## 请求与响应

| 请求类型 | 主要内容 | 响应 |
|---|---|---|
| session_start | Session 上下文、Agent memory；包含 instruction／scene_briefing／sequence_context | 可返回 memory_update 和 sequence_reset_ack |
| step | 当前任务文本、task_id、observation、memory；包含 capabilities、response_mode、name_introductions、allowed_actions | action；可带 done、answer、memory_update、agent_status |
| observation | scene_id、session_id、task_id、最近的 request_id、observation；按固定时钟独立发送 | 无响应；接收端须在推理期间持续读入 |

session_start 携带 `observation_stream: {fps: 2.0, mode: "push"}`，Agent 返回
`observation_stream_ack: true`。所有 session_start／step 响应必须原样返回 request_id；
step 另须返回 `observation_frame_id`，表示本次推理已纳入的观测截止帧。
它必须是同一 Task 内实际已发送的帧，且不早于本次请求自带的帧。不得把随后仅由接收线程读入的帧宣称为已纳入当前推理。
observation 消息的 request_id 仅用于关联上下文，不触发额外动作响应。

Agent 必须持续接收观测；缺少流式确认会在 Session 握手时报错。

v5 `contextual_requests` 的所有完整请求都使用 step 循环，response_mode 为 actions 或 actions_and_answer。
例如：

```json
{"request_id":"2","observation_frame_id":1,"action":"no_op","answer":"study","done":true}
```

Agent 应按公开 capabilities 区分导航、操作和问答，不能把所有 long_horizon 请求都视为不支持。
动作编码示例见 [Agent 接入指南](AGENT_INTEGRATION.md)。v5 抓取使用公开 target 名称，
如 `{"action":"pick","target":"reading book"}`，不接受私有 object_id／room_id。
问答动作与计分见 [成功条件](RULES.md#scoring)：仅按答案正确性计分，不要求观察取证。
不在 allowed_actions 中的抓放、开关门及 reset 会被拦截反馈，但不因该尝试强制判错。

## 观测与时间

v5 observation 可包含 images、depth、pose、reset_applied、holding_object、
已介绍物体的 held_object_name、普通 action_feedback 和 timing。私有取证数据不在该结构中。

任务观测的 `observation.timing` 字段如下。
调度与慢 Agent 的处理规则统一见 [观测规则](RULES.md#observations)。

| 字段 | 含义 |
|---|---|
| frame_id、slot | 本运行采样序号（从 1 开始）和固定时钟槽号（从 0 开始）；跨 Task 连续 |
| fps、timebase | 配置值 2.0、wall_monotonic；不是测得的实际处理帧率 |
| scheduled_monotonic_s | 该帧计划采样的单调时钟时刻 |
| sample_started_monotonic_s、captured_monotonic_s | 实际开始采样和完成复制帧的时刻 |
| dispatched_monotonic_s、age_at_dispatch_seconds | 开始提交请求／观测消息的时刻及距采样完成的时间；不代表 Agent 已读到 stdin |
| capture_seconds、lateness_seconds | 采样耗时及开始相对计划时刻的延迟 |
| missed_slots、missed_slots_total | 本次及累计错过的采样时刻数；不包括任务间停采时间 |

单调时间只用于同一运行内比较，不能当作情境日历或跨机器 UTC。
timing 不含目标可见性、对象 ID 或 GT。Agent 无需响应 observation，每次 step 仍只回复一条消息。

示例共用 [stream.serve](../benchmark/agents/stream.py)：接收线程持续读消息，可通过
on_observation 回调即时处理每一帧；独立工作线程执行模型调用。handler 的 observation_history
是 Agent 自己保存的先前图像快照，排除当前帧，推理期间的新图用于后续调用。
此字段由示例接收器在 Agent 内部添加，不由 Benchmark 筛选后发送。
自定义流式模型可以自行实现接收器，按实际使用情况返回 observation_frame_id。

带情境日历的数据在 session_start 提供
`time_context: {kind: "scenario_calendar", starts_at: ...}`，在 step 提供
current_time（含时区的 ISO 8601 时间）。其含义见 [执行规则](RULES.md#execution)。

## 状态字段

agent_status 可为 ok、unsupported、integration_error，缺省为 ok。
后两者必须附非空 reason_code；integration_error 可附 error。
unsupported 的 step 响应仍需符合 schema，例如返回 no_op。
计分及运行是否继续由 [结果分类](RULES.md#exclusions) 定义，不由 Agent 自报成功决定。

## 序列身份与重置握手

每个独立序列在 session_start 携带：

```text
sequence_context: {sequence_id, run_id, reset_memory}
```

step 携带相同身份；run_id 每次运行唯一，首个 Session 的 reset_memory 为 true，后续为 false。
完成 [序列记忆重置](RULES.md#execution) 后，Agent 返回 `sequence_reset_ack: <同一 run_id>`。
无确认／错误确认按接入错误处理。

SimpleNav 适配器清空本地观察／指令历史，其远端预测请求自行携带完整输入；
OpenAI-compatible VLM 适配器使用无状态 chat-completions 请求。
独立 `openai_api_agent.py` 使用 Responses 响应链，新序列丢弃父响应引用后确认 reset，
同序列跨 Session 保持会话；不维护或返回显式记忆。其输入与服务端存储策略见
[API Agent 专题](OPENAI_API_AGENT.md)。
HoloAgent 网关须实际处理并原样确认 run_id，适配器不会替远端伪造确认。
