# 发布数据与独立运行契约

Benchmark 直接消费 `OSS/<scene_id>/session.tar`。安装、校验、执行、修补、评分和 CPU 测试均不需要
Task Builder／Session Builder；manifest 的 `builder` 字段只记录数据来源。

## 入口与缓存

```bash
python -m roboodyssey validate ../OSS/scene000010/session.tar
python -m roboodyssey doctor --dataset ../OSS/scene000010/session.tar
python -m roboodyssey run --dataset ../OSS/scene000010/session.tar --sequence seq_0001 --dry-run
python -m roboodyssey run --dataset ../OSS/scene000010/session.tar --all-sequences --dry-run
```

`run`、`doctor`、`validate`、SimpleNav 的 `preflight / experiment` 和显式历史重算均支持 tar。
也接受已解包的 `session/`、`session/preparation/collection/` 或单条序列目录。
多序列包须显式选择 `--sequence` 或 `--all-sequences`；校验整个包不需要选择。

首次读取按归档 SHA-256 解包到独立缓存，源 tar 保持只读，不在源数据旁生成文件。
默认位置为 `${XDG_CACHE_HOME:-~/.cache}/longterm-indoor-benchmark/datasets/<sha256>/`；
用 `BENCHMARK_DATA_CACHE=/path/to/cache` 指定容量足够的位置。解包使用临时目录完成后发布，
拒绝越界路径、重复文件、符号／硬链接及特殊文件。后续可复用解包结果，但仍校验 manifest 和文件哈希；
损坏缓存可删除对应哈希目录后重新读取。实验状态和报告写入独立结果目录。

## 发布包层级

```text
session.tar
└── session/
    ├── preparation/collection/
    │   ├── scene_manifest.json
    │   ├── scene_versions/<revision>/
    │   │   ├── scene_version.json
    │   │   ├── scene.usdz
    │   │   ├── scene_metainfo.json / scene_understanding.json
    │   │   ├── navigation.npz / manipulation.json / environment_setup.json
    │   │   └── names.json / public_language_profile.json / initial_state.json
    │   └── sequences/<sequence_id>/
    │       ├── dataset_manifest.json
    │       ├── sessions/<session_id>/session.json
    │       └── private/
    │           ├── scene/runtime_scene_cache.json
    │           ├── build/scenario_plan.json / language_catalog.json
    │           └── sessions/<session_id>/ground_truth.json / templates.json / mutations.json
    ├── blueprints/、generation/、gt/ …
    └── statistics.json / verification.json …
```

图中省略了审阅、生成审计、历史等辅助文件。归档可包含它们；读取器校验 manifest 清单内所有文件，
执行仅使用冻结请求、真实运行所需的场景资料和契约，不从生成报告或计划成功历史推断实际经历。
包内 `gt/*/*_physx.json` 作为随包参考证据保留，不替换当次运行回读得到的实际 GT／成功结果。

## 版本与解析

| 文件 | 当前契约 | 作用 |
|---|---|---|
| `scene_manifest.json` | `cl_scene_collection_v1`，schema 1 | 场景身份、共享版本和序列索引，各自 manifest SHA-256 |
| `scene_version.json` | `cl_scene_version_v1`，schema 1 | 固定资产及共享资料、初始状态、逐文件 SHA-256 |
| `dataset_manifest.json` | `cl_task_sequence_v1`，schema 5、layout 2 | Session 顺序、公开／私有文件路径、共享版本绑定及文件 SHA-256 |
| `runtime_contract` | `contextual_requests`，version 3 | 冻结语言、目标、名称、真实历史和修补语义 |

共享资源使用 `{"scope":"scene_version","key":"navigation"}` 等显式引用；场景资产固定为版本内
嵌入的相对路径。读取器验证场景、序列和版本身份、Session／任务／GT／模板顺序、名称、
语言目录、初始状态和环境设置的一致性。未知版本、文件缺失或哈希不符明确失败。

校验检查发布文件与运行契约，不重新调用生成端编译器验证“计划全部成功”的离线生成过程。
生成端验收报告可随数据发布，Benchmark 的真实运行有效性另按 [RULES](RULES.md) 判定。
读取器仅接受上述发布版本；不支持的 schema、layout 或 runtime_contract 明确报错。

## 运行时职责与公开边界

`roboodyssey/data/` 只读取数据，`roboodyssey/contracts/` 提供冻结语言渲染、真实历史绑定、状态修补、
mutation 可行性、动态导航和放置几何。mutation 的最小上下文直接来自冻结房间、导航和可搬运对象表，
不构造生成器或重新规划任务。运行失败后的请求／GT 按实际状态修补，原归档保持不变。

Agent 仅通过统一协议收到公开任务、名称介绍、普通观测和允许反馈。私有缓存、GT、模板、
scene_understanding、导航参考、overview 和修补审计不作为模型输入。SimpleNav 和其他适配器
共享这一边界，不需要直接理解 tar 或 manifest。适配方式见 [AGENT_INTEGRATION](AGENT_INTEGRATION.md)。
