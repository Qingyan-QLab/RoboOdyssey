# RoboOdyssey 架构与维护

本页记录必须保持的架构边界。第一次使用请从 [README](README.md) 开始；模块调用关系与目标目录见 [ARCHITECTURE](docs/ARCHITECTURE.md)，独立输入格式见 [SCENE_SESSION](docs/SCENE_SESSION.md)，逐文件职责见自动生成的 [FILE_MAP](docs/FILE_MAP.md)。

## 依赖方向

- `roboodyssey/` 是环境、交互、评测及直接支撑它们的数据与证据核心。它不能导入 `agent_integrations/` 或 `synthesis/` 中的模型／生成器实现。
- 被评 Agent 的适配器、模型侧协议、配置和案例位于独立仓库 https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations。通用 JSONL 协议属于 `roboodyssey/interaction/`。
- https://github.com/Qingyan-QLab/RoboOdyssey-DataSynth 使用核心环境与私有 Oracle 策略导出新 Scene／Session；正式评测仅读取冻结输入，不调用生成器。

## 运行与数据

调用方加载 Scene 目录或 tar、Session 目录或 tar，再组合执行。资源 manifest 列出 3D 资产、场景状态、导航、任务、GT 等依赖；加载器拒绝缺失、越界和不匹配的资源。原有发布 `session.tar` 由兼容读取器处理，不在运行中重编译任务。

同一序列内的 Session 按顺序执行并保留世界状态与 Agent 记忆；不同序列隔离运行。被评 Agent 只收到公开请求、观测与动作反馈。Oracle、私有 GT、参考地图及评分证据不得进入公开消息。评分使用真实执行证据，离线分析保持原日志只读。

## 验证

```bash
python3 -m roboodyssey check
python3 -m roboodyssey inventory --check
```

变更 Scene／Session 契约时更新数据格式与加载测试；变更评分时验证在线和离线结果一致；变更模型接入时确保核心库仍可在无模型、无 GPU 的主机 Python 中导入。有 Isaac Sim 环境时再运行物理冒烟测试。正式规则见 [RULES](docs/RULES.md)。
