# 通用交互协议

`protocol.py` 和 `protocol_schema.json` 定义评测端的 JSONL 请求／响应与动作校验；`agent_api.py` 管理 Agent 子进程通信、超时及退出；`stream.py` 是可复用的 Agent 侧持续收帧实现。这里不包含任何模型权重、模型服务协议或被评 Agent 的决策实现。完整消息字段和公开／私有边界见 [PROTOCOL](../../docs/PROTOCOL.md)。
