# Agent Protocol v1

runner 和 agent 通过 stdin/stdout 交换 JSONL。协议版本为 1.0，请求类型为
session_start、step 和 qa。

每个 session 的任务按 session.json.tasks[] 顺序发送。GT、mutation 和 canonical
state 不出现在请求中；agent 只能通过视觉 observation、任务文本和自己维护的
memory 推断状态。

step 返回 action，可选 done 和 memory_update。qa 返回 answer 或 prediction，也
可携带 memory_update。reset 会把 avatar 放回安全可行走点，记录在 action log 中
并降低 reset-aware score。

完整响应 schema 位于 `benchmark/agents/protocol_schema.json`。
