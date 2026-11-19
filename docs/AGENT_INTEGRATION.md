# Agent 接入指南

模型进程通过 stdin/stdout 使用 JSONL Protocol v1.1。每行 stdout 必须是一个 JSON
对象，日志写入 stderr。公开输入边界和计分见 [运行与评分规则](RULES.md)。

Gateway 会递归拒绝 `ground_truth`、`expected_outcome`、参考轨迹等私有字段。公开请求只含
任务文本、相机观测、agent pose、动作反馈和 agent 自己此前写入的 memory。环境也不会在
缺少对象参数时用隐藏任务目标补全 `pick`；可以发送视觉识别得到的语义名称，例如
`{"action":"pick","target":"vase"}`，环境只在当前可达对象中解析该名称。

## 请求

- session_start：包含 scene_id、session_id、session instruction、task IDs 和
  agent 自己维护的 memory；
- step：包含当前 task、视觉 observation、当前 memory、上一动作和结果；
- qa：包含问题、视觉 observation、当前 memory 和 answer type。

QA 按任务列表即时发送，不会在 session 末尾集中发送。
推理期间还有独立 observation 消息，接收器必须持续读取；流式握手、关联字段及共用接收器见
[PROTOCOL](PROTOCOL.md#请求与响应)。下述响应示例只展示动作／答案主体，接收器须补齐 request_id 和 observation_frame_id。

## 响应

step 响应：

~~~json
{"action":{"action":"move_forward","distance":0.25},"done":false,"memory_update":{}}
~~~

移动距离和转向角度可由 agent 自定义：

~~~json
{"action":{"action":"move_forward","distance":0.8}}
{"action":{"action":"turn_left","angle_degrees":45}}
~~~

环境按二维半径 `0.1 m` 的圆形 agent footprint 进行路径碰撞检测。若超长移动
遇到障碍，环境会在碰撞前约 `0.1 m` 停止，并在下一次请求的
`observation.action_feedback` 与 `previous_result` 中返回 `collision`、
`executed_distance` 和 `reason`。SimpleNav 的恢复策略见 [适配说明](SIMPLENAV_V5.md)。

旧版 pick/place 可携带 object_id、target 或 room_id：

~~~json
{"action":{"action":"pick","object_id":"corridor1_vase002"}}
{"action":{"action":"place","room_id":"kitchen1"}}
~~~

v5 不公开或接受 object_id／room_id 操作句柄。pick 使用已介绍的公开名称，例如
`{"action":"pick","target":"reading book"}`；place 在当前位置释放持物。
公开字段和完整请求循环见 [协议](PROTOCOL.md)。

QA 响应：

~~~json
{"answer":"kitchen"}
~~~

合法动作是 no_op、move_forward、move_backward、move_left、move_right、turn_left、turn_right、
look_up、look_down、pick、place、open、close 和 reset。SimpleNav 不直接预测
open/close；旧版接近并碰撞关闭的门时适配器会自动发出 open，并在日志中标记
`action_source=adapter_auto`、`agent_requested=false`。未知动作、无效 JSON、进程退出
和超时的处理见 [结果分类](RULES.md#exclusions)。

v5 没有私有附近物体清单，因此不启用上述自动开门。SimpleNav 只支持导航；
QA／操作返回 unsupported，后端错误使用 integration_error。
代码准备、CPU 预检、GUI／录像启动和失败修补报告见
[SimpleNav v5](SIMPLENAV_V5.md)。

## API 视觉 Agent

`examples/openai_api_agent.py --config /absolute/path/to/agent.local.json` 是直接调用 API 的 v5
Agent 入口，使用 `benchmark/agents/openai_api.py` 和 Responses 响应链。
序列首轮上传 prompt 和初始机器人状态，每轮仅上传当前 RGB；任务文本变更时追加，
不重传历史帧、不维护显式记忆。模型自行依靠连续会话记忆，默认仅保留最小执行反馈。
本仓库仅发布 SimpleNav 配置模板；API 实验使用自行准备的 `configs/openai_api/agent.local.json`，不传给 SimpleNav 实验管理器。
模型 ID、API 参数、离线检查、诊断／正式命令及费用记录见 [API Agent 专题](OPENAI_API_AGENT.md)。

旧 `openai_compatible_vlm_agent.py` 仍被 HoloAgent 组合服务导入，两条入口独立。

## Manipulation

pick 只能作用于 builder 的 pick/place candidate，且对象必须在 avatar 可达范围内。
持有对象会显示在第一人称画面右下方，并随 avatar 移动。place 必须发生在 agent
当前所在房间；对象随后解除附着，写入 session 的实际 state change。

## HoloAgent 组合服务

`holoagent_benchmark_server.py` 可以组合两个后端：

- FSR-VLN 或 live OVO：导航目标定位与移动动作；
- OpenAI-compatible VLM：视觉操作、QA，以及 `memory_update`。

至少配置一个后端。完整能力运行时同时提供导航后端和 `--vlm-endpoint/--vlm-model`。
没有对应后端的能力返回明确的 `*_not_configured` 状态。

## 新增模型适配

SimpleNav 接口继续保留：`benchmark/agents/simplenav.py` 实现模型状态与动作转换，
`examples/simplenav_agent.py` 为 JSONL 入口，`configs/simplenav/*.example.json` 配置 managed／external 服务。
模型配置统一放在 `configs/<model>/`；当前仅保留 SimpleNav 共享模板，本机配置使用 `*.local.json`，不提交版本控制。
后续模型采用同样的实现／入口／配置分层，并复用 `agents/stream.py` 持续接收观测。

适配器声明能力和不支持的任务，按协议确认新序列记忆重置，关联实际使用的观测帧。
模型依赖单独安装，不加入 Benchmark 的主机依赖。适配器仅接收公开消息，
发布数据由 Benchmark 的独立读取器加载，GT、名称解析、真实历史修补和评分继续由 Benchmark 负责。
