# 架构与调用关系

RoboOdyssey 分为核心库、被评 Agent 集成仓库和数据合成仓库三个依赖层。核心库负责环境、交互、评测和私有评分参考。外围包可以调用核心，核心不依赖具体模型或生成器。

```text
Scene 目录/tar + Session 目录/tar ─┐
                                  ├─→ data 加载与校验 → BenchmarkSuite
现有发布 session.tar ──────────────┘                       │
                              ┌───────────────────────────┼───────────────────────────┐
                              ↓                           ↓                           ↓
                    environment + interaction      evaluation + reference       synthesis 调用核心接口
                              ↓                           ↓                           ↓
                    Agent 公开观测与动作             私有证据与评分参考           新 Scene/Session 包
                              ↓                           ↓
                         运行证据 ─────────────────→ 结果与报告
```

## 核心模块

- `data/sources.py`：独立 Scene／Session 目录或 tar 的公共加载器和 `BenchmarkSuite`；`data/archive.py` 负责安全缓存。`scenes/dataset_loader.py` 保留现有发布包适配。
- `scenes/` 与 `runtime/environment.py`：Isaac 场景、导航、抓放、状态和观测；可选传感器由外部插件工厂注入。
- `interaction/`：语言无关的 JSONL 协议、Agent 子进程和持续收帧实现。`runtime/` 执行 Session 和请求循环，记录动作、状态与修补证据。
- `contracts/`：实际状态、冻结语言、请求修补和几何规则，供环境与评测共同使用。
- `evaluation/`：成功判定、版本化指标、汇总和离线诊断。`reference.py` 提供私有评分参考入口；可执行 Oracle Policy 位于 DataSynth。
- `artifacts/`、`reporting/`：结果 manifest、哈希、日志路径及报告。
- `launch/`、`cli.py`、`run.py`：通用序列进程管理和命令入口。SimpleNav 专用服务管理位于 [Agent Integrations](https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations/tree/main/agent_integrations/simplenav)。

## 依赖与信息边界

[Agent Integrations](https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations) 保存模型实现、模型侧协议、配置和案例；[DataSynth](https://github.com/Qingyan-QLab/RoboOdyssey-DataSynth) 保存生成器接口及导出。两仓库依赖 `roboodyssey`，不得反向导入。核心 JSONL 协议允许公开请求、普通观测与动作反馈；GT、私有对象 ID、Oracle 动作和评分参考只在评测／合成的私有路径内使用。

同一序列的 Session 按顺序运行，环境状态与 Agent 记忆连续；不同序列分进程隔离。正式数据只读，运行证据和合成产物写入新目录。更多细节见 [独立输入](SCENE_SESSION.md)、[协议](PROTOCOL.md)、[规则](RULES.md) 和 [逐文件清单](FILE_MAP.md)。
