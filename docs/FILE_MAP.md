# 项目文件清单

由 `python -m roboodyssey inventory` 生成；`python -m roboodyssey inventory --check` 检查遗漏和过期。
只维护生成器中的文件用途；目录、链接和主要定义从源码提取。无需同步改写 PROJECT_GUIDE。

本页逐项列出 **163 个源码、配置、文档和测试文件**。
不导入被检查模块，不读取模型、视频或场景资产内容，不启动 Isaac/GPU。
运行产物/本地文件按末尾目录模板说明，不逐项列出每次实验产生的重复文件。

推荐使用：`python -m roboodyssey validate / preflight / experiment / run / analyze / check`。
scripts 只保留安装、联合启动及专项检查；重复包装入口已移除。
数据由发布 session.tar 或解包目录提供；运行与测试不依赖生成项目；统一 analyze 写入新目录。

## 完整源码目录

```text
RoboOdyssey/
├── .github/
│   └── workflows/
│       └── test.yml
├── configs/
│   └── simplenav/
│       ├── managed.example.json
│       └── review.example.json
├── docs/
│   ├── validation/
│   │   └── pr_integration_20261126.json
│   ├── AGENT_INTEGRATION.md
│   ├── BATCH.md
│   ├── DATA_FORMAT.md
│   ├── EXPECTATIONS_AND_PLAN.md
│   ├── FILE_MAP.md
│   ├── LAUNCH.md
│   ├── MOTION.md
│   ├── OPENAI_API_AGENT.md
│   ├── OUTPUTS.md
│   ├── PROTOCOL.md
│   ├── PUBLIC_API.md
│   ├── ROBOTS.md
│   ├── RULES.md
│   └── SIMPLENAV_V5.md
├── examples/
│   ├── holoagent_agent.py
│   ├── holoagent_benchmark_server.py
│   ├── holoagent_sim_gateway.py
│   ├── openai_api_agent.py
│   ├── openai_compatible_vlm_agent.py
│   ├── random_agent.py
│   └── simplenav_agent.py
├── roboodyssey/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_api.py
│   │   ├── openai_api.py
│   │   ├── openai_api_cli.py
│   │   ├── protocol.py
│   │   ├── protocol_schema.json
│   │   ├── random_agent.py
│   │   ├── simplenav.py
│   │   ├── simplenav_cli.py
│   │   └── stream.py
│   ├── artifacts/
│   │   ├── __init__.py
│   │   └── layout.py
│   ├── contracts/
│   │   ├── LICENSE
│   │   ├── __init__.py
│   │   ├── contextual.py
│   │   ├── doors.py
│   │   ├── language.py
│   │   ├── navigation.py
│   │   ├── navigation_io.py
│   │   ├── placement.py
│   │   ├── planned_history.py
│   │   ├── planning.py
│   │   ├── raster.py
│   │   ├── references.py
│   │   ├── requests.py
│   │   ├── snapshots.py
│   │   └── state.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── archive.py
│   │   └── frozen.py
│   ├── evaluation/
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   └── qa.py
│   │   ├── __init__.py
│   │   ├── navigation.py
│   │   ├── report.py
│   │   ├── requests.py
│   │   ├── rescore.py
│   │   ├── task_diagnostics.py
│   │   ├── task_metrics.py
│   │   └── transfer.py
│   ├── launch/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   ├── experiment.py
│   │   ├── provenance.py
│   │   ├── sequences.py
│   │   └── simplenav.py
│   ├── reference_oracles/
│   │   ├── ORIGIN.md
│   │   ├── __init__.py
│   │   ├── answers.py
│   │   ├── navigation.py
│   │   ├── reference.py
│   │   └── types.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── analyze.py
│   │   ├── compare.py
│   │   ├── request_audit.py
│   │   └── task_diagnostics_report.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── capture.py
│   │   ├── environment.py
│   │   ├── observations.py
│   │   ├── policy.py
│   │   ├── recording.py
│   │   ├── request_events.py
│   │   ├── request_runner.py
│   │   └── session_runner.py
│   ├── scenes/
│   │   ├── robot_assets/
│   │   │   ├── profiles/
│   │   │   │   └── stretch3.json
│   │   │   ├── usd/
│   │   │   │   └── stretch3/
│   │   │   │       ├── articulated/
│   │   │   │       │   └── robot/
│   │   │   │       │       ├── payloads/
│   │   │   │       │       │   ├── Physics/
│   │   │   │       │       │   │   ├── mujoco.usda
│   │   │   │       │       │   │   ├── physics.usda
│   │   │   │       │       │   │   └── physx.usda
│   │   │   │       │       │   ├── base.usda
│   │   │   │       │       │   ├── geometries.usd
│   │   │   │       │       │   ├── instances.usda
│   │   │   │       │       │   ├── materials.usda
│   │   │   │       │       │   └── robot.usda
│   │   │   │       │       └── robot.usda
│   │   │   │       └── avatar.usdc
│   │   │   ├── LICENSE-stretch_urdf.md
│   │   │   ├── README.md
│   │   │   └── sources.lock.json
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── dataset_loader.py
│   │   ├── holoagent_sensor_bridge.py
│   │   ├── isaac_backend.py
│   │   ├── isaac_env.py
│   │   ├── lighting.py
│   │   ├── request_evidence.py
│   │   └── robot_body.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── check.py
│   │   ├── dataset.py
│   │   └── inventory.py
│   ├── viewer/
│   │   ├── __init__.py
│   │   ├── viewer_overlay.py
│   │   └── viewport.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── run.py
├── scripts/
│   ├── bootstrap.sh
│   ├── check_request_evidence.py
│   ├── check_simplenav_cpu.py
│   ├── run_holoagent.sh
│   └── run_request_diagnostics.py
├── tests/
│   ├── fixtures/
│   │   └── contextual.json
│   ├── frozen_data.py
│   ├── test_batch.py
│   ├── test_benchmark.py
│   ├── test_comparison.py
│   ├── test_contextual_requests.py
│   ├── test_evaluation_metrics.py
│   ├── test_observations.py
│   ├── test_openai_api_agent.py
│   ├── test_planned_projection.py
│   ├── test_policy_runtime.py
│   ├── test_project_workflow.py
│   ├── test_public_data.py
│   ├── test_published_data.py
│   ├── test_reference_oracles.py
│   ├── test_robot_body.py
│   ├── test_scene_camera.py
│   ├── test_scene_sequences.py
│   ├── test_simplenav_v5.py
│   ├── test_task_diagnostics.py
│   └── test_transfer_metrics.py
├── .gitattributes
├── .gitignore
├── AGENTS.md
├── CHANGELOG.md
├── NOTICE
├── PROJECT_GUIDE.md
├── README.md
├── environment.yml
├── pyproject.toml
├── requirements.txt
└── run_benchmark.sh
```

## 文件内容与职责

### 根目录

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [.gitattributes](../.gitattributes) | 保留固定 SHA-256 的生成 USD 层字节，仅豁免这些资产的文件末尾空行检查。 | — |
| [.gitignore](../.gitignore) | 实验结果、本地配置、环境、缓存和资产的版本控制排除规则。 | — |
| [AGENTS.md](../AGENTS.md) | 仓库贡献和验证约定。 | — |
| [CHANGELOG.md](../CHANGELOG.md) | 发行版本变更与迁移说明。 | — |
| [NOTICE](../NOTICE) | 第三方代码及授权声明。 | — |
| [PROJECT_GUIDE.md](../PROJECT_GUIDE.md) | 稳定的架构边界和维护原则；逐文件清单集中在本页。 | — |
| [README.md](../README.md) | 安装、数据验证、实验执行、结果分析和扩展的使用入口。 | — |
| [environment.yml](../environment.yml) | 可选 Conda 主机环境依赖。 | — |
| [pyproject.toml](../pyproject.toml) | Python 包元数据、依赖、打包与测试配置。 | — |
| [requirements.txt](../requirements.txt) | 主机 Python 的最小依赖。 | — |
| [run_benchmark.sh](../run_benchmark.sh) | Bash 启动器：选择 Isaac/主机 Python，准备 ROS 环境并调用通用 run。 | — |

### .github/workflows

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [test.yml](../.github/workflows/test.yml) | GitHub Actions 的完整独立 CPU 回归、视频工具安装与清单检查。 | — |

### configs/simplenav

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [managed.example.json](../configs/simplenav/managed.example.json) | 受管理实验配置示例，含模型服务、GPU、序列和自动报告。 | — |
| [review.example.json](../configs/simplenav/review.example.json) | SimpleNav 单次 GUI/录像复核配置示例。 | — |

### docs

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [AGENT_INTEGRATION.md](../docs/AGENT_INTEGRATION.md) | 第三方 Agent 接入和适配示例。 | — |
| [BATCH.md](../docs/BATCH.md) | 通用批次配置、新尝试重试/停止及严格配对比较用法。 | — |
| [DATA_FORMAT.md](../docs/DATA_FORMAT.md) | session.tar 发布层级、缓存、版本、共享引用、校验和公开/私有边界。 | — |
| [EXPECTATIONS_AND_PLAN.md](../docs/EXPECTATIONS_AND_PLAN.md) | 待审阅的职责与改动规划：三库数据约定、公共模块、状态边界及迁移验收。 | — |
| [FILE_MAP.md](../docs/FILE_MAP.md) | 自动生成的完整源码目录、文件用途。 | — |
| [LAUNCH.md](../docs/LAUNCH.md) | 实验配置、预检、进程管理、终止和来源记录。 | — |
| [MOTION.md](../docs/MOTION.md) | 运动策略现状审计与 Stretch 预期设计：动作适配、碰撞、抓放、评分及分阶段验收。 | — |
| [OPENAI_API_AGENT.md](../docs/OPENAI_API_AGENT.md) | API Agent 精简输入、连续会话、反馈公平性、启动方式和上传/计费 token 区别。 | — |
| [OUTPUTS.md](../docs/OUTPUTS.md) | 实验/运行/Session 的输出文件、证据引用和当前布局契约。 | — |
| [PROTOCOL.md](../docs/PROTOCOL.md) | Agent 请求/响应、动作、图像及记忆协议。 | — |
| [PUBLIC_API.md](../docs/PUBLIC_API.md) | 三库当前公共 API、来源/状态/参考边界及独立 wheel 用法。 | — |
| [ROBOTS.md](../docs/ROBOTS.md) | 圆柱与六种机器人选择、资产路径、运动边界及配置用法。 | — |
| [RULES.md](../docs/RULES.md) | 运行与评分规则的唯一维护入口：连续性、预算、正式配置、修补、成功/排除、聚合、导航诊断。 | — |
| [SIMPLENAV_V5.md](../docs/SIMPLENAV_V5.md) | SimpleNav v5 接入、受控诊断、取证及录像复现。 | — |

### docs/validation

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [pr_integration_20261126.json](../docs/validation/pr_integration_20261126.json) | 当前 PR 的 CPU、隔离安装和三库 Isaac 联合验收；含版本、资产身份与可复现命令。 | — |

### examples

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [holoagent_agent.py](../examples/holoagent_agent.py) | HoloAgent HTTP 服务到 Benchmark 协议的适配。 | `normalize_response`, `HoloAgentAdapter`, `main` |
| [holoagent_benchmark_server.py](../examples/holoagent_benchmark_server.py) | HoloAgent/VLM 服务、规划与动作桥接。 | `FsrVlnPlanner`, `LiveOvoPlanner`, `DirectOvoPlanner`, `assert_public_agent_request`, `ManipulationStateMachine`, `CompositeHoloAgentPlanner`, `RemoteProtocolAdapter`, `PlannerGatewayHandler` 等 9 个定义 |
| [holoagent_sim_gateway.py](../examples/holoagent_sim_gateway.py) | HoloAgent 与仿真侧交互网关。 | `parse_relative_command`, `relative_to_actions`, `TargetPose`, `SimGatewayState`, `GatewayHandler`, `RosBridgeNode`, `main` |
| [openai_api_agent.py](../examples/openai_api_agent.py) | API Agent 的 JSONL 薄入口，提供配置检查和持续收帧。 | — |
| [openai_compatible_vlm_agent.py](../examples/openai_compatible_vlm_agent.py) | OpenAI 兼容视觉语言服务的 Agent 示例。 | `Adapter`, `LocalTransformersAdapter`, `main` |
| [random_agent.py](../examples/random_agent.py) | 随机动作 Agent，用于协议/仿真冒烟验证。 | — |
| [simplenav_agent.py](../examples/simplenav_agent.py) | SimpleNav JSONL 协议入口，复用 agents/simplenav.py。 | — |

### roboodyssey

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/__init__.py) | Python 包入口。 RoboOdyssey simulation, evaluation and shared runtime package. | — |
| [__main__.py](../roboodyssey/__main__.py) | python -m roboodyssey 的入口，转发 CLI。 | — |
| [cli.py](../roboodyssey/cli.py) | 统一命令分发、环境检查与数据下载；按需导入各功能。 | `main` |
| [run.py](../roboodyssey/run.py) | 通用评测参数、数据分流、Isaac/Agent 生命周期和结果汇总。 | `main` |

### roboodyssey/agents

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/agents/__init__.py) | Python 包入口。 Agent process integration and the JSONL protocol. | — |
| [agent_api.py](../roboodyssey/agents/agent_api.py) | 通过 JSONL 子进程协议与 Agent 交互，管理超时和退出。 | `AgentProcess` |
| [openai_api.py](../roboodyssey/agents/openai_api.py) | API 视觉 Agent：连续响应链、单帧增量输入、最小反馈、动作校验和用量日志。 | `load_config`, `APIError`, `post_json`, `OpenAIAPIAgent` |
| [openai_api_cli.py](../roboodyssey/agents/openai_api_cli.py) | 随 wheel 分发的视觉 API JSONL 命令入口。 | `main` |
| [protocol.py](../roboodyssey/agents/protocol.py) | Agent 响应校验、协议版本、允许动作及统一动作名称读取。 | `action_name`, `required_action_name`, `validate_response` |
| [protocol_schema.json](../roboodyssey/agents/protocol_schema.json) | Agent JSON 消息的机器可读结构。 | — |
| [random_agent.py](../roboodyssey/agents/random_agent.py) | 随 wheel 分发的确定性协议冒烟 Agent。 | `SmokeAgent`, `main` |
| [simplenav.py](../roboodyssey/agents/simplenav.py) | SimpleNav 请求和动作适配、历史图像及不支持任务的响应。 | `instruction_text`, `simple_action`, `SimpleNavAgent` |
| [simplenav_cli.py](../roboodyssey/agents/simplenav_cli.py) | 随 wheel 分发的 SimpleNav JSONL 命令入口。 | `main` |
| [stream.py](../roboodyssey/agents/stream.py) | Agent 侧持续接收观测与独立推理线程、请求关联及示例视觉历史。 | `serve` |

### roboodyssey/artifacts

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/artifacts/__init__.py) | Python 包入口。 Artifacts interfaces. | — |
| [layout.py](../roboodyssey/artifacts/layout.py) | 结果布局唯一解析入口；共用 JSON/JSONL、文件与内容哈希、状态和证据引用。 | `sha256_file`, `json_digest`, `read_json_lines`, `write_json`, `read_json`, `local_path`, `safe_id`, `manifest` 等 18 个定义 |

### roboodyssey/contracts

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [LICENSE](../roboodyssey/contracts/LICENSE) | 迁入运行契约与合成测试样例代码的原 MIT 版权和许可。 | — |
| [__init__.py](../roboodyssey/contracts/__init__.py) | Python 包入口。 Benchmark-owned frozen-data runtime contracts; see LICENSE. | — |
| [contextual.py](../roboodyssey/contracts/contextual.py) | v3 实际状态归一化、名称/历史/GT 修补、失败因果和 mutation 决策。 | `RuntimeContractError`, `normalize_state`, `distance_to_box`, `completed`, `bound_record`, `snapshot_room`, `introduction_reference`, `failed_causes` 等 11 个定义 |
| [doors.py](../roboodyssey/contracts/doors.py) | 冻结开门策略、USD 门几何应用与回读验证。 | `profile_digest`, `door_records`, `collision_bounds`, `apply_open_doors`, `verify_open_doors` |
| [language.py](../roboodyssey/contracts/language.py) | 情境日历、历史时间、公开语言、名称和冻结语言目录的统一校验/渲染。 | `HistoryError`, `parse_time`, `relative_time`, `historical_reference`, `validate_public_wording`, `room_name`, `clauses_for`, `clause_options` 等 13 个定义 |
| [navigation.py](../roboodyssey/contracts/navigation.py) | 地面网格、对象标签、接近位置、路径和 mutation 连通性校验。 | `entity_label`, `cell_world`, `nearest_free`, `navigation_task`, `navigation_components`, `validate_connectivity` |
| [navigation_io.py](../roboodyssey/contracts/navigation_io.py) | 无 pickle 的导航 NPZ 读取与确定性测试数据写入。 | `write_navigation_npz`, `read_navigation_npz` |
| [placement.py](../roboodyssey/contracts/placement.py) | 实际状态下地面放置、碰撞和可接近性几何计算。 | `PlacementError`, `translated_entity`, `floor_placement` |
| [planned_history.py](../roboodyssey/contracts/planned_history.py) | 预计事件与位置账本；明确不代表执行观察。 | `HistoryError`, `PlannedHistory` |
| [planning.py](../roboodyssey/contracts/planning.py) | 复用请求修补与几何的 CPU 预计序列推演；不可冒充正式参考。 | `PlannedProjection`, `project_sequence` |
| [raster.py](../roboodyssey/contracts/raster.py) | 房间/对象几何与开门策略的动态碰撞栅格。 | `derive_navigation` |
| [references.py](../roboodyssey/contracts/references.py) | 元数据对象身份、区域名称、真实历史和位置变化的统一指代校验/修补。 | `same_location`, `RepairUnavailable`, `area_name`, `object_reference` |
| [requests.py](../roboodyssey/contracts/requests.py) | 完整请求的实际历史记录、旧契约修补和 mutation 放置校验。 | `history_event`, `prepare_mutation` |
| [snapshots.py](../roboodyssey/contracts/snapshots.py) | 计划/实际来源隔离、状态版本、快照分支与缓存失效。 | `state_version`, `StateSnapshot`, `StateBranch` |
| [state.py](../roboodyssey/contracts/state.py) | 对象真实几何、位置签名、冻结参考缓存和哈希。 | `apply_object_deltas`, `label`, `attributes`, `geometry`, `location_signature`, `build_runtime_cache`, `same_location` |

### roboodyssey/data

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/data/__init__.py) | Python 包入口。 Published dataset archives and immutable manifest readers. | — |
| [api.py](../roboodyssey/data/api.py) | 当前格式场景、独立 sequence、集合与绑定的公共读取入口。 | `load_scene`, `load_sequence`, `validate_binding` |
| [archive.py](../roboodyssey/data/archive.py) | 发布 tar 的只读源解析、独立哈希缓存、包内路径与文件类型检查。 | `discover_root`, `dataset_root` |
| [frozen.py](../roboodyssey/data/frozen.py) | 场景/版本/序列 manifest、共享资源、冻结契约校验和最小 mutation 上下文；不编译任务。 | `read`, `identity`, `inside`, `check_file`, `inventory`, `collection`, `select_sequences`, `SceneVersion` 等 9 个定义 |

### roboodyssey/evaluation

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/evaluation/__init__.py) | Python 包入口。 Task scoring, numerical diagnostics and result aggregation. | — |
| [navigation.py](../roboodyssey/evaluation/navigation.py) | 实际任务起点网格的带权最短路、机器人半径、目标区域和有序路线动态规划。 | `GridDistance`, `box_distance`, `OrderedRoute` |
| [report.py](../roboodyssey/evaluation/report.py) | 请求/Session/序列及能力统计的唯一汇总实现。 | `write_summary` |
| [requests.py](../roboodyssey/evaluation/requests.py) | v5 完整请求的运行时 GT 判定与任务计分。 | `evaluate_request`, `request_totals` |
| [rescore.py](../roboodyssey/evaluation/rescore.py) | 显式历史重算、验证归档及真实状态重建参考地图、历史耗时和分组统计。 | `final_state`, `ReconstructedReference`, `historical_measurements`, `metric_breakdown` |
| [task_diagnostics.py](../roboodyssey/evaluation/task_diagnostics.py) | 从真实 GT/动作日志复算逐任务评分、距离、路径与 reset；不渲染报告。 | `xy`, `target_box`, `is_applied_reset`, `motion_metrics`, `progress_fraction`, `navigation_diagnostics`, `reset_diagnostics`, `compact_gt` 等 13 个定义 |
| [task_metrics.py](../roboodyssey/evaluation/task_metrics.py) | 版本化导航进展×路径效率、搬运/问答部分分、真实耗时和评分完整性。 | `prepare_reference`, `load_reference`, `navigation_metric`, `evaluate_metrics`, `metric_totals` |
| [transfer.py](../roboodyssey/evaluation/transfer.py) | 搬运剩余交付成本、真实抓放约束、可移除占据网格与进展乘效率评分。 | `collision_digest`, `TransferReference`, `transfer_metric` |

### roboodyssey/evaluation/metrics

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/evaluation/metrics/__init__.py) | Python 包入口。 Benchmark metric functions. | — |
| [qa.py](../roboodyssey/evaluation/metrics/qa.py) | 问答归一化、公开区域名称及布尔答案判定。 | `normalize_answer`, `contextual_answer_correct` |

### roboodyssey/launch

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/launch/__init__.py) | Python 包入口。 Launch interfaces. | — |
| [batch.py](../roboodyssey/launch/batch.py) | 通用串行实验清单、独立attempt、有限基础设施重试与取消清理。 | `build_plan`, `execute`, `main` |
| [experiment.py](../roboodyssey/launch/experiment.py) | 管理模型服务和序列进程、读取进度、终止清理及自动分析。 | `service_command`, `port_ready`, `Progress`, `execute` |
| [provenance.py](../roboodyssey/launch/provenance.py) | 记录源码/数据/权重哈希、Git 状态和 Python 环境。 | `repository`, `capture_provenance` |
| [sequences.py](../roboodyssey/launch/sequences.py) | 序列选择、独立进程运行、共用进程组清理与集合汇总。 | `stop_owned`, `selected_paths`, `child_arguments`, `aggregate_runs`, `run_collection` |
| [simplenav.py](../roboodyssey/launch/simplenav.py) | 配置解析、命令构建、CPU 预检和共用启动 CLI；预检和 managed/external 实验入口。 | `load_config`, `commands`, `preflight`, `selected_datasets`, `prepare`, `main` |

### roboodyssey/reference_oracles

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [ORIGIN.md](../roboodyssey/reference_oracles/ORIGIN.md) | 从 DataSynth 迁入的实验参考代码来源与依赖边界。 | — |
| [__init__.py](../roboodyssey/reference_oracles/__init__.py) | Python 包入口。 Optional, explicitly versioned reference models; not official scoring.

Install RoboOdyssey[reference] for restools navigation.
Importing this package or answers does not import restools or Isaac. | — |
| [answers.py](../roboodyssey/reference_oracles/answers.py) | 基于显式场景与历史来源的共享房间答案计算。 | `RoomSnapshot`, `TaskHistory`, `AnswerResult`, `room_answer` |
| [navigation.py](../roboodyssey/reference_oracles/navigation.py) | 独立版本 restools 点到点参考、快照和搜索预算。 | `point_xy`, `NavigationSnapshot`, `SearchLimits`, `SearchBudget`, `PathResult`, `PointNavigator` |
| [reference.py](../roboodyssey/reference_oracles/reference.py) | 实验性有限候选任务参考，区别预算耗尽与不可达。 | `TargetPose`, `GoalRegion`, `CandidateGoal`, `Interaction`, `ReferenceStage`, `ReferenceLeg`, `TaskReference`, `ReferenceOracle` |
| [types.py](../roboodyssey/reference_oracles/types.py) | 实验参考的轻量状态来源和结果类型，不导入 restools。 | — |

### roboodyssey/reporting

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/reporting/__init__.py) | Python 包入口。 Reports from recorded benchmark execution evidence. | — |
| [analyze.py](../roboodyssey/reporting/analyze.py) | 统一离线分析分发：实验/集合递归、当前评分诊断和 HTML 索引。 | `analyze`, `main` |
| [compare.py](../roboodyssey/reporting/compare.py) | 严格按数据与评测条件配对比较，保留失败、缺失与独立重复样本。 | `compare_runs`, `main` |
| [request_audit.py](../roboodyssey/reporting/request_audit.py) | v5 请求、GT、来源失败和视频的修补审计。 | `lines`, `write_request_audit` |
| [task_diagnostics_report.py](../roboodyssey/reporting/task_diagnostics_report.py) | 逐任务分数/导航/reset 的 HTML、CSV、JSON 和 Markdown 展示；共用评分报告入口。 | `esc`, `fmt`, `pct`, `ref`, `table`, `detail`, `video_link`, `distance_chart` 等 11 个定义 |

### roboodyssey/runtime

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/runtime/__init__.py) | Python 包入口。 Continuous request execution, observations and evidence logging. | — |
| [capture.py](../roboodyssey/runtime/capture.py) | 可选原始 RGB/depth、实际状态、位姿与帧引用归档。 | `ObservationArchive` |
| [environment.py](../roboodyssey/runtime/environment.py) | 当前请求运行器的环境接口。 | `RequestEnvironment` |
| [observations.py](../roboodyssey/runtime/observations.py) | 真实时间 2 FPS 采样时钟与漏采审计；采到的帧立即交给发送方，不缓冲。 | `capture_observation`, `ObservationClock` |
| [policy.py](../roboodyssey/runtime/policy.py) | 私有策略适配与公共 sequence 执行、真实环境工厂及来源记录。 | `InProcessPolicyAgent`, `make_isaac_env_factory`, `run_sequence` |
| [recording.py](../roboodyssey/runtime/recording.py) | CPU ffmpeg 录像、帧时间索引和编码器收尾。 | `RecordingError`, `validate_video_options`, `rgb_image`, `wrap`, `SessionRecorder` |
| [request_events.py](../roboodyssey/runtime/request_events.py) | 任务/动作/reset/mutation 事件，GUI 和录像联动。 | `RequestEvents`, `state_changes`, `source_context` |
| [request_runner.py](../roboodyssey/runtime/request_runner.py) | v5 请求修补、真实状态、名称账本、连续历史和任务执行。 | `append`, `actual_state`, `public_observation`, `resolve_action`, `load_runtime_context`, `apply_boundary`, `run_requests` |
| [session_runner.py](../roboodyssey/runtime/session_runner.py) | Agent 通信、观测流及完整请求执行入口。 | `SessionRunner` |

### roboodyssey/scenes

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/scenes/__init__.py) | Python 包入口。 Scene package loading, state composition, and Isaac Sim environment. | — |
| [camera.py](../roboodyssey/scenes/camera.py) | 延迟加载 Isaac 的共用场景打开与持久相机采集。 | `open_scene_stage`, `create_camera`, `SceneCamera` |
| [dataset_loader.py](../roboodyssey/scenes/dataset_loader.py) | 发布场景集合、Session 和冻结资源的统一读取。 | `canonical_state_from_metadata`, `SessionRecord`, `BenchmarkDataset` |
| [holoagent_sensor_bridge.py](../roboodyssey/scenes/holoagent_sensor_bridge.py) | HoloAgent ROS 传感器发布桥接。 | `HoloAgentSensorTopics`, `pose_message_fields`, `IsaacSimHoloAgentSensorBridge` |
| [isaac_backend.py](../roboodyssey/scenes/isaac_backend.py) | Isaac 应用启停、场景工厂、相机预热和视口接入。 | `make_env_factory`, `IsaacBackend` |
| [isaac_env.py](../roboodyssey/scenes/isaac_env.py) | Isaac 场景、相机、导航/搬运、碰撞、reset 及权威世界状态。 | `EnvConfig`, `IsaacVLNEnv` |
| [lighting.py](../roboodyssey/scenes/lighting.py) | 准备与执行共用的显式基准照明配置，记录参数并拒绝静默失败。 | `LightingConfig`, `apply_scene_lighting` |
| [request_evidence.py](../roboodyssey/scenes/request_evidence.py) | 面向评分器的当前帧可见性与检查取证。 | `runtime_state`, `observe`, `apply_mutation` |
| [robot_body.py](../roboodyssey/scenes/robot_body.py) | 本体选择、CPU 列表/校验、USD 配置与资产身份；高层身体位置/朝向同步。 | `robot_assets_root`, `validate_robot_selection`, `RobotSelection`, `resolve_robot`, `main`, `RobotBodyConfig`, `load_robot_profile`, `default_robot_body` 等 14 个定义 |

### roboodyssey/scenes/robot_assets

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [LICENSE-stretch_urdf.md](../roboodyssey/scenes/robot_assets/LICENSE-stretch_urdf.md) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [README.md](../roboodyssey/scenes/robot_assets/README.md) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [sources.lock.json](../roboodyssey/scenes/robot_assets/sources.lock.json) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/scenes/robot_assets/profiles

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [stretch3.json](../roboodyssey/scenes/robot_assets/profiles/stretch3.json) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/scenes/robot_assets/usd/stretch3

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [avatar.usdc](../roboodyssey/scenes/robot_assets/usd/stretch3/avatar.usdc) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [robot.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/robot.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [base.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/base.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [geometries.usd](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/geometries.usd) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [instances.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/instances.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [materials.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/materials.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [robot.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/robot.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/Physics

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [mujoco.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/Physics/mujoco.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [physics.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/Physics/physics.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |
| [physx.usda](../roboodyssey/scenes/robot_assets/usd/stretch3/articulated/robot/payloads/Physics/physx.usda) | 独立 wheel 内的固定 Stretch3 profile、USD 依赖、来源记录或上游许可；用于默认本体与资产完整性校验。 | — |

### roboodyssey/validation

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/validation/__init__.py) | Python 包入口。 Validation interfaces. | — |
| [check.py](../roboodyssey/validation/check.py) | 源码 CPU 回归与独立 wheel 冒烟检查，禁止仿真/模型导入。 | `NoAcceleratorImports`, `installed_smoke`, `main` |
| [dataset.py](../roboodyssey/validation/dataset.py) | 发布数据归档／集合／序列校验；文件与场景资产完整性检查。 | `validate`, `main` |
| [inventory.py](../roboodyssey/validation/inventory.py) | 静态扫描文件、用途和主要定义，生成/检查本页；不导入被扫描模块。 | `source_files`, `describe`, `render`, `main` |

### roboodyssey/viewer

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [__init__.py](../roboodyssey/viewer/__init__.py) | Python 包入口。 Isaac Sim viewer overlays and visualization helpers. | — |
| [viewer_overlay.py](../roboodyssey/viewer/viewer_overlay.py) | GUI 任务、状态、成功/失败和 reset 信息覆盖层。 | `ViewerOverlay` |
| [viewport.py](../roboodyssey/viewer/viewport.py) | Isaac 视口相机绑定、窗口布局与渲染同步。 | — |

### scripts

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [bootstrap.sh](../scripts/bootstrap.sh) | 创建/安装最小主机 Python 环境。 | — |
| [check_request_evidence.py](../scripts/check_request_evidence.py) | 真实 Isaac 当前帧可见性取证检查；需要 GPU。 | `main` |
| [check_simplenav_cpu.py](../scripts/check_simplenav_cpu.py) | 禁止导入模型/仿真 SDK 的 CPU 回归执行器。 | — |
| [run_holoagent.sh](../scripts/run_holoagent.sh) | HoloAgent 模型/VLM 服务与仿真联动启动。 | — |
| [run_request_diagnostics.py](../scripts/run_request_diagnostics.py) | 完整发布序列上的私有失败注入诊断；--prepare-only 仅校验选择，实际执行需要 Isaac。 | `prepare`, `main` |

### tests

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [frozen_data.py](../tests/frozen_data.py) | 选择仓库内已冻结合成样例并写入临时测试归档；不调用任务生成器。 | `key`, `goal`, `task`, `fixture_v3`, `select_scenario`, `runtime_cache`, `write_dataset`, `prepare_scene` 等 10 个定义 |
| [test_batch.py](../tests/test_batch.py) | 真实CPU子进程故障注入，验证重试保留、失败不筛选、超时和取消终态。 | `BatchTests` |
| [test_benchmark.py](../tests/test_benchmark.py) | 发布数据、显式数据路径 CLI 及模型接口基础回归。 | `MetricTests`, `DatasetTests` |
| [test_comparison.py](../tests/test_comparison.py) | 比较条件拒混、缺失/失败、重复 attempt 与配对汇总回归。 | `make_run`, `ComparisonTests` |
| [test_contextual_requests.py](../tests/test_contextual_requests.py) | v5 请求、实际 GT、名称/指代与评分契约回归。 | `Env`, `Agent`, `RequestRunnerTests` |
| [test_evaluation_metrics.py](../tests/test_evaluation_metrics.py) | 部分得分、绕路/墙体/目标顺序、reset 计费、参考哈希及新旧评分回归。 | `pose`, `truth`, `trajectory`, `nav_score`, `PathMetricTests`, `SemanticMetricTests`, `ReproducibilityTests` |
| [test_observations.py](../tests/test_observations.py) | 固定时钟、慢 Agent 持续收帧、阻塞接收/渲染失败和 JSONL 通信 CPU 回归。 | `ManualTime`, `ObservationTests` |
| [test_openai_api_agent.py](../tests/test_openai_api_agent.py) | 模拟 API 的 prompt/图像去重、响应链隔离、输入边界、最小反馈和计量回归。 | `observation`, `start`, `step`, `output`, `result`, `delta`, `APIAgentTests` |
| [test_planned_projection.py](../tests/test_planned_projection.py) | 预计语义修补、历史边界、来源拒混与失败原子性回归。 | `PlannedProjectionTests` |
| [test_policy_runtime.py](../tests/test_policy_runtime.py) | 实际执行、私有输入、原始采集引用及状态分支隔离回归。 | `PolicyRuntimeTests` |
| [test_project_workflow.py](../tests/test_project_workflow.py) | 结构化输出、统一入口、源码清单、分析和托管进程生命周期回归。 | `ArtifactWorkflowTests`, `ExperimentWorkflowTests`, `ReusedEntryTests` |
| [test_public_data.py](../tests/test_public_data.py) | 独立场景、分离 sequence 与身份/版本/哈希绑定回归。 | `PublicDataTests` |
| [test_published_data.py](../tests/test_published_data.py) | 发布 tar/包装目录、缓存/篡改拒绝、任务顺序、独立导入及真实 mutation 几何回归。 | `PublishedDataTests` |
| [test_reference_oracles.py](../tests/test_reference_oracles.py) | 可选 restools 参考算法迁入回归，缺 extra 时明确跳过。 | `snapshot`, `candidates`, `NavigationTests`, `TaskReferenceTests`, `AnswerTests` |
| [test_robot_body.py](../tests/test_robot_body.py) | 默认 Stretch3、官方身份、场景半径匹配、多本体选择、资产完整性和相机/身体分离回归。 | `RobotBodyTests` |
| [test_scene_camera.py](../tests/test_scene_camera.py) | 共享相机矩形视野比例与资源关闭回归。 | `SceneCameraTests` |
| [test_scene_sequences.py](../tests/test_scene_sequences.py) | 场景多序列选择、隔离进程及集合汇总回归。 | `ResetAgent`, `SequenceIntegrationTests` |
| [test_simplenav_v5.py](../tests/test_simplenav_v5.py) | SimpleNav、预检、录像、取证与资源清理回归。 | `SimpleNavProtocolTests`, `RecordingTests`, `CapturingEnv`, `V5TraceTests`, `IsaacBackendTests`, `PreflightTests` |
| [test_task_diagnostics.py](../tests/test_task_diagnostics.py) | 正式分、目标顺序、导航距离、路径及 reset 诊断回归。 | `pose`, `truth_at`, `action`, `outcome`, `NavigationDiagnosticsTests`, `TaskScoreReportTests` |
| [test_transfer_metrics.py](../tests/test_transfer_metrics.py) | 搬运连续进展、抓放范围/偏移、错误放置、绕路/reset、参考保存和证据完整性回归。 | `pose`, `object_at`, `fixture`, `change`, `run_cost`, `TransferCostTests` |

### tests/fixtures

| 文件 | 内容 / 作用 | 主要函数 / 类 |
|---|---|---|
| [contextual.json](../tests/fixtures/contextual.json) | 合成回归数据：场景、已冻结场景变体、模板/GT 和语言目录。 | — |

## 动态目录与本地文件

| 目录 / 文件 | 内容与维护方式 |
|---|---|
| `results/<experiment-or-run>/` | 实验配置、来源、状态、序列、Session 原始证据、视频和可重建报告；逐文件定义见 [OUTPUTS.md](OUTPUTS.md)。历史结果保留原格式，实验/整理说明随各自结果归档。 |
| 外部数据目录（可选本地 `dataset/`、`dataset_downloads/`） | 数据由使用者提供，通过 --dataset/配置指定；不随源码分发或生成。 |
| `configs/<model>/*.local.json` | 按模型集中存放的本机路径、GPU/端口配置；不纳入共享源码。 |
| `.venv/`、`*.egg-info/`、`build/`、`dist/` | 可重建的安装/打包文件。 |
| `__pycache__/`、`.pytest_cache/`、`.cache/`、`*.py[cod]` | 可重建缓存，不属于项目接口。 |
| `.git/` | Git 元数据和 LFS 对象。 |

新增文件需补充 `roboodyssey/validation/inventory.py` 中的用途（包入口自动识别），
然后运行 `python -m roboodyssey inventory`。文件新增/删除、主要定义变化或清单被手工改动后，
`--check` 会返回非零；生成不带时间戳，不因重复执行产生无意义差异。
