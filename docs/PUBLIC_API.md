# 三库公共 API（0.2.1）

安装包名 `RoboOdyssey`，导入名 `roboodyssey`；消费者依赖 `>=0.2.1,<0.3`。
`roboodyssey.PUBLIC_API_VERSION` 与 `roboodyssey.data.API_VERSION` 当前均为 `1`。
兼容范围是三库**当前输出**：sequence schema 5、layout 2、`contextual_requests` v3。
不承诺旧 Builder 格式转换、旧编译器重放或任意历史评分重算。

## 冻结场景和任务

```python
from roboodyssey.data import load_scene, load_sequence, validate_binding

scene = load_scene('/data/scene_versions/r_123')
sequence = load_sequence('/data/detached_sequence', scene=scene)
identity = validate_binding(scene, sequence)
```

场景入口接受 collection/tar、scene-only collection、独立 revision 目录；sequence 入口接受
collection/tar 或 sequence 目录。多 sequence 集合须传 `sequence_id`，脱离原布局的 sequence 必须显式传
`scene`。绑定检查包含场景 ID、revision 与 manifest 哈希。读取校验冻结文件，不调用 Builder 编译器。
加载场景不校验无关 sequence；加载 sequence 必须校验自身及所绑定场景。嵌入资产不得逃逸包目录。
生成事务、场景理解、任务内容仍由 Builder 负责。

## 预计状态与公共语义

`roboodyssey.contracts` 维护名称、状态归一化、语言、历史修补、碰撞栅格、放置候选等共同规则。
`instantiate_request_v3(..., state_source='actual')` 是默认运行模式；它拒绝计划历史。
显式 `state_source='planned'` 使用 `assumed_dispatch`、`assumed_success` 等假设字段，输出
`planned_predispatch_state`，不能生成执行事件或正式评分证据。

`StateBranch(cache, state, source='planned')` 提供版本、分支、提交、历史和状态变化时缓存失效。
实际分支仅接受 `readback`；实际运行器当前继续由环境回读与 `execution_events_v3` 唯一维护历史，
并未整体改为 `StateBranch`。`state_version(normalized_state, source=...)` 同时用于快照和采集记录：
它是包含来源的规范化状态内容哈希，不是递增序号或文件 SHA；`state_sha256` 才是状态文件 SHA。

```python
from roboodyssey.contracts.planning import project_sequence
plan = project_sequence(sequence)
assert plan['state_source'] == 'planned'
assert plan['formal_score_reference'] is False
```

CPU 投影模型为 `grid_aabb_expected_success_v1`，按实际起始状态重新计算预计路径、放置和答案，
共用请求修补、already-satisfied 判断及历史绑定。历史 before/after 始终指完整请求边界。
保守的栅格/AABB 放置与 avatar 固定前向偏移放置不是同一种模型；投影不能证明物理可执行，不能替代正式参考。
路线/放置/历史不确定时返回 `indeterminate`、停止位置及原因，已完成前缀保留。
问答正确性不自动建立预计现场观察；依赖该观察的环境变化会明确停止，而非伪造观察经历。

## 公共执行与真实采集

```python
from roboodyssey.runtime.policy import run_sequence, make_isaac_env_factory
# app 为调用方已创建的 SimulationApp；CPU import 不启动 Isaac。
factory = make_isaac_env_factory(sequence, app, config=env_config)
summary = run_sequence(sequence, policy, '/data/new_attempt', env_factory=factory)
```

`policy` 是可信 Python 对象，提供 `decide(task, public_observation, context)` 和可选 `reset()`。
默认 `permission_profile='private_state_oracle'` 接收私有 context：实际 GT、状态、历史、名称、任务起始参考及实际动作反馈。
声明 `permission_profile='perception_policy'` 则仅收到公共任务、原始 numpy 观测、scene/session 身份和白名单公开反馈；不交付私有上下文。
观测交给策略前复制，在线噪声不覆盖原始采集帧。该权限分流是接口约定，不是对恶意 Python 的进程沙箱。
外部被测模型仍使用现有公共 JSONL 协议，不能收到这些字段。策略响应使用现有动作协议；标准 runner 决定
实际成功、排除、修补、连续 Session 历史和评分。整个 sequence 新建结果目录；失败与异常不删除原产物。

`run_config.json` 标记 `official=false` 和策略声明的 execution_profile，记录数据/规则/观测/策略身份、环境配置、
环境来源哈希，以及 `isaac_sim` 或 `custom_or_test` 后端。测试环境的成功不能作为真实 Isaac 证据。
`capture=False` 可关闭原始存储；默认开启不改变正式评测的默认采集设置。

原始 RGB/depth 以禁用 pickle 的 `.npy` 存储在 `observations/`；`frames.jsonl` 同时记录帧号、
实际状态文件及 SHA、camera/base pose、时钟、数组 shape/dtype 和媒体 SHA。
动作通过 `observation_frame_id` 关联帧。采集耗时属于当前同步观察流程的真实开销，不补帧、不将录像当原始数据。
真实回读碰撞与参考地图采用同一 USD 几何来源；字段新增、显式 null 和删除都进入动作差异，以支持严格重放。

`roboodyssey.scenes.camera.SceneCamera` 复用场景打开、开门设置与相机；视点选择、图片命名和审核属于 Builder。
必须在已有 SimulationApp 中构造；关闭相机不关闭调用方 app。`capture` 返回 RGB 数组，
`last_capture_metadata` 提供递增帧号、UTC/monotonic 时间、world 相机位姿（XYZW）、尺寸/数据类型/图像 SHA，
以及 asset/environment setup/verified environment 的准备输入身份。此 `state_source=preparation` 和
`state_identity.kind=prepared_asset_and_environment` 明确区别实际执行状态；不保证检测构造后任意外部 USD 编辑。
静态拍照显式使用零物理时间步，矩形分辨率保持声明的水平 FOV。
准备与执行共用 `roboodyssey.scenes.lighting.LightingConfig`：默认保留原执行环境的 Dome 3000、Fill 800 与光照方向；
`SceneCamera(..., lighting=...)` 或 `EnvConfig(lighting=...)` 可显式选择 authored/不同强度。
参数与版本进入准备帧身份和正式运行 profile。全黑准备帧明确失败，不因尺寸/哈希完整而列为成功，也不填充像素掩盖问题。

## 参考模型

正式 `progress_path_v3` 继续使用现有 `GridDistance` 与 `TransferReference`。
`GridDistance.next_waypoint` 和 `TransferReference.guidance` 给策略提供同一参考场的下一步查询，
不代替实际抓放反馈或成功判定。`guidance` 的 phase 为 `approach`、`carry`、`completed`；
能否抓取与可放置朝向另由 `can_pick`、`placement_yaw_degrees` 返回。

实验性 restools 参考位于 `roboodyssey.reference_oracles`，通过 `[reference]` extra 显式安装；
算法版本、候选枚举与预算独立，不切换正式评分。预算用尽是 `indeterminate`，不能当作不可达。

## 独立安装和检查

```bash
python -m pip install ./roboodyssey-0.2.1-py3-none-any.whl
python -m roboodyssey check
python -m roboodyssey validate --help
python -m roboodyssey.agents.simplenav_cli --help
python -m roboodyssey.agents.openai_api_cli --help
```

wheel 的 `check` 执行公共 CPU 模块、协议资源与碰撞查询 smoke check；源码树的同一命令运行完整回归，
`--core` 选择源码快速子集。两种检查均禁止导入 Isaac/模型/生成库，不混称完整测试。
默认随机协议 Agent 已在包内；源 `examples/` 入口只转发。Isaac SDK 和场景资产仍由运行环境单独提供。

通用实验批次、有限重试及严格结果比较见 [BATCH](BATCH.md)。

## Stretch3 身份与默认配置（0.2.1）

`roboodyssey.scenes.robot_body` 提供 `default_robot_body()`、`default_robot_identity()`、
`resolve_robot()`、`robot_identity_for_config(config)`、`scene_robot_identity(scene)` 和
`validate_scene_robot(scene, runtime_identity, require_declared=False)`。
默认使用 bundled Stretch3；最后一个函数拒绝声明的场景与运行身份错配，并检查导航半径。
场景可为 SceneVersion、SequenceData、BenchmarkDataset 或 manifest 字典；字典不包含磁盘导航，
只有传完整场景对象才能校验 NPZ。缺少声明不会推断为 Stretch3，官方传 `require_declared=True`。

Builder 应将 `default_robot_identity()` 写入 `scene_version.json.robot_profile` 和
`navigation.robot_profile`，半径取其中 `robot_radius_m`；公共 NPZ 读写保留该字段。
`EnvConfig()` 使用 canonical Stretch3；`EnvConfig(robot_body=None)` 显式选择历史圆柱，
自定义 scalar 半径也须明确 `robot_body=None`，不能覆盖身体 profile 却保留其身份。
共享计划、放置与派生导航保留声明；计划仍是预计状态，不能充当实际运行证据。

## 当前 PR 联合验收

2026-11-26，迁移后的 RoboOdyssey 0.2.1 安装包在源码目录外完成三库联调：
Stretch3 合成场景 5/5 任务成功、46 帧、1 个完整 sequence 训练样本；公开核心独立官方入口及离线重评分通过。
[可复现条件和完整验收记录](validation/pr_integration_20261126.json)区分私有状态 Oracle 闭环与官方脚本验证，
并列出 CPU 条件跳过和物理控制边界。
