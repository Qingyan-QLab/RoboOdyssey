# Agent 接入指南

模型进程通过 stdin/stdout 使用 JSONL Protocol v1。每行 stdout 必须是一个 JSON
对象，日志写入 stderr。模型不能读取数据集中的 ground truth 文件。

## 请求

- session_start：包含 scene_id、session_id、session instruction、task IDs 和
  agent 自己维护的 memory；
- step：包含当前 task、视觉 observation、当前 memory、上一动作和结果；
- qa：包含问题、视觉 observation、当前 memory 和 answer type。

QA 按任务列表即时发送，不会在 session 末尾集中发送。

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
`executed_distance` 和 `reason`。SimpleNav 连续五次被阻挡后发送 `reset`，不会自动
改变转向。

pick/place 可携带 object_id、target 或 room_id：

~~~json
{"action":{"action":"pick","object_id":"corridor1_vase002"}}
{"action":{"action":"place","room_id":"kitchen1"}}
~~~

QA 响应：

~~~json
{"answer":"kitchen"}
~~~

合法动作是 no_op、move_forward、move_backward、move_left、move_right、turn_left、turn_right、
look_up、look_down、pick、place、open、close 和 reset。SimpleNav 不直接预测
open/close；接近并碰撞关闭的门时适配器会自动发出 open，并在日志中标记
`action_source=adapter_auto`、`agent_requested=false`。未知动作、无效 JSON、进程退出
和超时会使当前响应失败。

## Manipulation

pick 只能作用于 builder 的 pick/place candidate，且对象必须在 avatar 可达范围内。
持有对象会显示在第一人称画面右下方，并随 avatar 移动。place 必须发生在 agent
当前所在房间；对象随后解除附着，写入 session 的实际 state change。
