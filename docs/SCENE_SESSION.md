# 独立 Scene 与 Session 输入

公共加载入口接收**目录或 tar 包**。每个包包含一个资源 manifest；文件可以在包内子目录，但所有引用必须留在包内。tar 首次读取后按内容哈希解包到独立缓存，拒绝越界路径、重复文件、链接和特殊文件。可用 `BENCHMARK_DATA_CACHE` 指定缓存根目录。原包始终只读。

## Scene 包

```text
scene_bundle/
├── scene_resources.json
├── scene.usd                         # 也可为 .usda/.usdc/.usdz
├── scene_metainfo.json
├── navigation.npz
├── manipulation.json
├── environment_setup.json
├── initial_state.json
└── runtime_scene_cache.json
```

`scene_resources.json` 示例：

```json
{
  "schema_version": 1,
  "kind": "roboodyssey_scene",
  "scene_id": "scene000010",
  "asset": "scene.usd",
  "asset_sha256": "<scene.usd 的 64 位 SHA-256>",
  "resources": {
    "scene_metainfo": "scene_metainfo.json",
    "navigation": "navigation.npz",
    "manipulation": "manipulation.json",
    "environment_setup": "environment_setup.json",
    "initial_state": "initial_state.json",
    "runtime_scene_cache": "runtime_scene_cache.json"
  },
  "sha256": {
    "scene_metainfo": "<文件 SHA-256>",
    "navigation": "<文件 SHA-256>",
    "manipulation": "<文件 SHA-256>",
    "environment_setup": "<文件 SHA-256>",
    "initial_state": "<文件 SHA-256>",
    "runtime_scene_cache": "<文件 SHA-256>"
  }
}
```

`asset_sha256` 和 `sha256` 必须填写实际内容哈希；上方尖括号仅为说明占位。读取器验证文件存在、引用不越界、资产格式、哈希和场景身份。资源本身沿用[发布数据格式](DATA_FORMAT.md)中的相应字段契约。

## Session 包

```text
session_bundle/
├── session_resources.json
├── session.json                    # 公开任务与上下文
├── ground_truth.json              # 私有 GT
├── templates.json                 # 私有模板
└── mutations.json                 # Session 边界变化
```

`session_resources.json` 示例：

```json
{
  "schema_version": 1,
  "kind": "roboodyssey_session",
  "scene_id": "scene000010",
  "sequence_id": "seq_0001",
  "session_id": "session_0001",
  "session": "session.json",
  "session_sha256": "<session.json 的 SHA-256>",
  "runtime_contract": {"name": "contextual_requests", "version": 3},
  "resources": {
    "ground_truth": "ground_truth.json",
    "templates": "templates.json",
    "mutations": "mutations.json"
  },
  "sha256": {
    "ground_truth": "<文件 SHA-256>",
    "templates": "<文件 SHA-256>",
    "mutations": "<文件 SHA-256>"
  }
}
```

`session.json` 沿用当前公开 Session 契约，包含 `scene_id`、`session_id`、`context`、`time_context` 和有序 `tasks`。GT 与模板数量必须与任务数量一致。多个 Session 按传入顺序执行，必须指向同一 `scene_id` 和 `sequence_id`，且 Session ID 不得重复。包内私有资源不会发送给 Agent。

## 公共入口与冻结场景

`roboodyssey.load_scene` 和 `roboodyssey.data.load_scene` 继续将本页资源包加载为 `SceneSpec`。
相同公共入口也接受冻结发布场景，按 manifest 返回 `SceneVersion`；不会转换现有资源格式。
若需要固定返回类型，可从 `roboodyssey.data` 显式导入 `load_scene_spec` 或 `load_scene_version`。
`revision=` 仅供冻结场景选择使用，资源包不支持该参数。混合两类 manifest 的目录或 tar 必须使用显式入口。

资源 manifest 的 `schema_version: 1` 与冻结 sequence 的 schema 5/layout 2 是不同契约。
`load_sequence`、`validate_binding` 和进程内策略 `run_sequence` 面向冻结数据；资源包仍通过
`BenchmarkSuite` 加载有序 Session。`BenchmarkSuite.from_published` 保留从发布数据进入同一评测器的方式。

## Python 调用

```python
from roboodyssey import BenchmarkSuite, load_scene, load_session

scene = load_scene("/data/scene_bundle")
sessions = [load_session("/data/session_1.tar"), load_session("/data/session_2")]
suite = BenchmarkSuite(scene, sessions)
suite.validate("interaction")

# env_factory: (scene_id, initial_state, session_id) -> RequestEnvironment
# agent: 遵守通用 request/send_observation 接口的对象
output = suite.interact(agent, "/results/run_001", env_factory)
summary = suite.evaluate(output)
```

数据合成使用同一组已加载输入，由外围生成器实现任务生成：

```python
from synthesis import synthesize

# generator.generate(scene, session, oracle) 返回 SessionDraft。
# oracle 是调用方提供的私有策略；Oracle 策略由 DataSynth 或调用方提供。
new_session = synthesize(generator, scene, sessions[0], oracle, "/data/generated_session")
```

当前仓库提供导出与校验流程，未内置 Actor SDG 生成器或完整动作 Oracle。`new_session` 可直接交给 `BenchmarkSuite`，不覆盖原 Session。

真实 Isaac 运行可通过 CLI 使用同一加载器。以下 Agent 命令要求先安装 Agent Integrations 仓库：

```bash
/path/to/isaacsim/python.sh -m roboodyssey run \
  --scene-source /data/scene_bundle \
  --session-source /data/session_1.tar \
  --agent-cmd "python3 -m agent_integrations.random.agent" \
  --output results
```

先加 `--dry-run` 检查输入与任务；实际运行需要场景内相机及物理资源符合当前 Isaac 环境契约。`--scene` 仍是旧发布包入口的场景 ID 筛选参数，因此独立输入使用 `--scene-source`。正式 `--official` 模式目前只接受完整冻结发布包。

## 旧发布包

`BenchmarkSuite.from_published(path, sequence_id=...)`、`load_published_dataset(path, sequence_id=...)` 和 `--dataset path` 继续读取原有 `session.tar`／解包目录。它保留完整发布包的版本、哈希、冻结契约和序列校验；独立 Scene／Session 包是新增入口，不自动把任意资源组合标记为正式发布数据。导出新包的工具见 [DataSynth](https://github.com/Qingyan-QLab/RoboOdyssey-DataSynth)。
