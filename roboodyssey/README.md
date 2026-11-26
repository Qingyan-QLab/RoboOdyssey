# 核心库 `roboodyssey`

本目录只包含环境、交互、评测、Oracle 接口，以及直接服务于这些能力的数据读取、证据和报告代码。核心库不导入 Agent Integrations 或 DataSynth。

## 模块

| 模块 | 职责 |
|---|---|
| `data/`、`scenes/dataset_loader.py` | 独立 Scene／Session 目录或 tar、现有发布包、资源校验 |
| `scenes/`、`runtime/environment.py` | Isaac 仿真环境、状态、观测和可注入传感器桥接 |
| `interaction/`、`runtime/` | 通用 JSONL 协议、Agent 子进程、连续 Session 请求循环与证据记录 |
| `evaluation/` | 成功判定、数值评分和私有评分参考 |
| `artifacts/`、`reporting/` | 结果布局、哈希、离线诊断与报告 |
| `launch/`、`cli.py`、`run.py` | 通用启动、序列隔离和命令包装 |

公共入口为 `load_scene(source)`、`load_session(source)` 和 `BenchmarkSuite(scene, sessions)`；`source` 是目录或 tar 包。`BenchmarkSuite.interact` 接受通用 Agent 对象和环境工厂，`evaluate` 汇总已记录的证据。已有发布包通过 `load_published_dataset` 读取。

具体资源格式和 Python 示例见 [独立输入指南](../docs/SCENE_SESSION.md)，完整调用链见 [架构说明](../docs/ARCHITECTURE.md)。
