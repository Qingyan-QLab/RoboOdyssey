# Benchmark 架构与维护

使用入口见 [README](README.md)。本页描述代码职责，输出契约见 [OUTPUTS](docs/OUTPUTS.md)。

## 主流程

```text
session.tar / 解包目录 → data → validation / launch → BenchmarkDataset → SessionRunner
                                             ├─ v5 request_runner
                                             └─ legacy session/task 路径
     → Isaac 环境与 Agent 协议 → 原始证据 → evaluation → reporting
```

发布数据提供不可变场景／序列、名称、模板、参考 GT 和状态缓存。
`data/` 按 manifest 与哈希独立解析，`contracts/` 按冻结契约和真实状态修补请求，
记录真实经历、下发名称账本和评分证据。运行及测试不导入生成工具，也不重新编译任务。
归档中的 builder 字段仅作来源信息；数据契约见 [DATA_FORMAT](docs/DATA_FORMAT.md)。

## 模块边界

- `data/` 负责 tar 缓存、场景版本、共享资源、序列选择与完整性校验；源数据不写入实验结果。
- `contracts/` 负责冻结语言渲染、历史绑定、实际状态修补、mutation 几何与动态导航；保留迁入代码的 MIT 声明。
  `language.py` 集中日历、公开表达、名称和语言目录；`references.py` 集中对象与历史指代；
  `navigation.py` 集中路径、接近位置和连通性。栅格、放置、门和实际状态仍按职责独立维护。
- `launch/` 负责启动前配置、CPU 预检、外部模型服务的生命周期、多序列隔离与来源信息；不实现评分。
- `artifacts/layout.py` 是输出路径的单一入口。新读写逻辑用它，不能再次硬编码 `scenes/*/sessions/*`。
- `runtime/request_runner.py` 执行 v5 完整请求；`session_runner.py` 保留公共 Agent 通信及旧版执行分流。
- `legacy/` 只承载原 schema 4 修补／历史。旧模块路径是兼容别名，只有一份业务实现。
- `evaluation/` 负责成功判定、汇总和数值诊断；`reporting/` 负责报告展示，不修改评分。
  `evaluation/report.py` 共用计分分母并汇总新旧运行；动作名称读取复用 `agents/protocol.py`，
  数值诊断复用评分器的距离计算。`artifacts/layout.py` 共用 JSONL 读取，正式计分严格报错，审阅报告保留错误行。
- `scenes/isaac_env.py` 负责实际相机、碰撞、门、pick/place 和权威状态；`request_evidence.py` 提供私有取证。
- `viewer/viewport.py` 管理 Isaac 视口；`viewer_overlay.py` 展示事件，不为 Agent 提供 GT。
- `agents/` 放模型适配实现，`examples/` 放 JSONL 薄入口，配置统一放在 `configs/<model>/`。
  当前仅发布 SimpleNav 模板；本机配置采用同目录的 `*.local.json`，不提供旧配置路径的转发或副本。
  SimpleNav 与 API 适配器保留；模型源码、权重和推理环境独立安装。新增模型复用统一协议，
  不在适配器中读取 GT、改变修补或评分规则。

## 兼容策略

旧 `benchmark.runtime.simplenav_launch`／`sequence_runs`、旧报告路径及 task_repair/task_history
使用模块别名转到新实现；合并辅助模块的旧导入也继续保留。别名集中声明于各包 `__init__.py` 的
`MODULE_ALIASES`，由顶层公共函数注册，不再逐个维护转发文件。旧路径与规范路径指向同一模块对象，
保留已有 import 和 mock/patch；内部代码只导入规范实现。完整映射由文件清单静态提取。
命令统一使用 `python -m benchmark`。v5 输出布局升级与输入任务契约版本无关。
旧运行保持只读，分析器同时识别布局 1 与 2。历史实验 `supervise.py` 只作证据保留，
新实验使用 `python -m benchmark experiment --config ... --execute`。

新格式删除持久化 goal_results／history 中重复的动作数组，改为 Session 范围的动作证据引用。
实际运行历史和权威 before/after 快照不变，不实现日志恢复运行或跨运行续接。

## 验证要求

运行 `python -m benchmark check`，覆盖新旧数据、名称、修补、完整任务评分、序列隔离、
视频编码、报告、结构化目录、预检不启动服务和受管理子进程清理。
测试数据位于 `tests/fixtures/`，只选择已冻结样例，不调用生成器；检查器禁止导入 Builder 或 GPU SDK。
CI 使用同一完整 CPU 入口。新增数据版本同时补 tar／目录路径、完整性拒绝和状态修补回归。
改动评分／证据读取后，应对既有运行做只读复算；不能以改写旧日志消除差异。
当前流程整理使用 CPU 和历史日志核验，真实新启动流程仍需另一次经授权的 GPU 验证。

数据与场景生成在本项目之外完成；本仓库仅附带合成测试样例，正式资产通过发布数据提供。
当前支持源码目录安装；`check` 与 `inventory` 使用仓库内脚本／文档，不宣称可由独立 wheel 完成开发检查。
