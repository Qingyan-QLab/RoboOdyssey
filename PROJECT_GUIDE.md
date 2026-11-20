# Benchmark 架构与维护

使用入口见 [README](README.md)，数据字段见 [DATA_FORMAT](docs/DATA_FORMAT.md)，输出字段见 [OUTPUTS](docs/OUTPUTS.md)。

## 运行链路

```text
session.tar / 解包目录 → 数据校验与序列选择 → BenchmarkDataset → SessionRunner
  → request_runner → Isaac 世界与 Agent 协议 → 执行证据 → 评分与报告
```

发布数据使用 dataset schema 5、layout 2、contextual_requests v3；结果使用 benchmark_results v2。
数据及结果均按 manifest 解析，不推测版本或目录布局；不支持的契约明确报错。
运行和测试独立于 Task Builder／Session Builder，不生成或重新编译任务。

## 模块职责

- `data/`：tar 缓存、场景集合、共享场景版本、冻结序列及文件哈希校验。
- `contracts/`：冻结语言、公开名称、真实执行历史、请求与 GT 修补、mutation 验证、导航及放置几何。
  `contextual.py` 是当前运行契约；`state.py` 处理实际对象状态；`references.py` 解析指代；
  `language.py` 校验并渲染语言。迁入代码的授权声明保存在该目录的 LICENSE 中。
- `launch/`：配置、CPU 预检、模型服务管理、序列进程隔离、终止清理和来源记录。
- `runtime/`：Session 连续执行、Agent 通信、2 FPS 观测、请求循环、状态与动作证据和录像。
- `runtime/environment.py` 定义请求运行器使用的环境接口，同一序列内跨 Session 复用环境。
- `scenes/isaac_backend.py` 管理 Isaac 应用、场景配置、相机预热和视口接入。
- `scenes/`、`viewer/`：Isaac 场景、相机、碰撞、抓放、权威状态、显示和私有取证。
- `evaluation/`：完整成功判定、progress_path_v3 评分、计分分母、汇总与数值诊断。
- `reporting/`：修补审计及 HTML／CSV／JSON 展示；复用评分实现，保持原始日志只读。
- `artifacts/layout.py`：结果路径、manifest、JSON／JSONL、哈希及动作证据引用的统一入口。
- `agents/`、`examples/`：模型适配实现和 JSONL 协议入口。配置集中在 `configs/<model>/`，
  当前发布 SimpleNav 模板，本机参数使用 `*.local.json`。模型源码、权重及环境单独安装。

模块使用实际定义所在路径导入。公共动作名称、距离计算、JSONL 读取和文件哈希均复用同一实现。
逐文件职责由 [FILE_MAP](docs/FILE_MAP.md) 统一维护；新增、移动或删除文件后更新用途登记并生成清单。

## 执行与证据边界

同一序列只初始化一次世界，跨任务／Session 保留实际状态、机器人位姿、名称账本和 Agent 记忆。
不同序列使用独立进程，并要求 Agent 确认记忆重置。依赖顺序、预算和评分见 [RULES](docs/RULES.md)。
模型仅接收公开请求、普通观测和允许反馈，不能读取 GT、私有对象 ID、修补记录或评分参考。

持久化的目标结果与历史通过 Session 动作引用关联日志，权威 before／after 快照完整保存。
分析可以在 CPU 上复算已记录的执行证据；缺失证据明确列出，不能补成零成本或成功。

## 验证与维护

```bash
python -m roboodyssey check
python -m roboodyssey inventory --check
```

CPU 测试覆盖发布数据读取与完整性、状态修补、真实历史、名称、导航与搬运评分、问答、
序列隔离、模型适配、录像、报告和进程生命周期。测试只使用冻结合成样例，禁止导入 Builder 或 GPU SDK。
真实物理执行和模型推理需在对应 Isaac／模型环境中验证。

新增数据版本须覆盖 tar／目录读取、版本拒绝、哈希、任务顺序和修补回归；修改评分须验证在线与离线结果一致。
当前开发检查使用源码目录中的脚本和文档；模型、场景资产、实验结果及本机环境单独管理。
