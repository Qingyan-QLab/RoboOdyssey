# 通过 API 运行视觉 Agent

`examples/openai_api_agent.py` 使用 Responses API 的连续会话执行 v5 任务。
每个独立序列建立新的响应链，同序列跨 Task／Session 继续使用；模型依靠会话上下文自行记忆。
本地只保存请求关联、去重哈希及上一个动作类别，不整理事实、摘要、地点或轨迹记忆。

GPT6 Astra 的真实 model ID、账户权限、上下文上限、图像计量与单价仍需核实。
2026-11-17 官方模型／会话／缓存文档访问返回 403 或超时；目前用模拟 API 检查请求结构和行为，
不声称已确认目标服务兼容性。服务端必须支持 Responses、`previous_response_id`、消息继承与所选输出格式。

## 每轮发送什么

| 内容 | 发送策略 |
|---|---|
| 系统 prompt | 序列首次决策作为 developer 消息上传一次，后续通过响应链保留；不使用需要反复设置的顶层 `instructions`。 |
| 当前图像 | 每次决策只上传当前 `front` RGB 一张，不重传历史帧；相同帧号不能重复调用。 |
| 机器人状态 | 序列第一次决策发送一次公开位姿、持物标志及公开持物名称；后续由模型推导，不提供精确状态纠正。 |
| Session 介绍 | Session 开始时暂存公开文本，下次决策追加到会话；已发送的相同名称介绍只记去重哈希。 |
| 任务文本／情境时间／允许动作 | 新任务或契约变化时发送一次；任务内部不反复发送。 |
| 动作反馈 | 默认最小反馈，见下节；不重发上一动作，其输出已在会话内。 |
| 显式记忆／决策上下文 | 不发送，也不要求模型返回 `memory_update`。 |
| 协议标识、帧时间和像素哈希 | 本地审计保存，不额外发送给模型。 |

输入使用白名单；不读取数据集 `private/`、`scene_understanding.json`、GT、私有对象／区域 ID、
目标坐标、深度、overview、精确执行距离或权威区域归属。
输出格式约束仍通过 API 的 `text.format` 每轮设置，它属于接口参数，不能靠历史设置推定继承。

`stream.serve(history_frames=0)` 持续读取 Benchmark 的 2 FPS 通知；独立 observation 不调用 API，
不缓存为下一轮历史图。一次调用仅使用本次动作请求携带的当前帧；推理期间的新帧不会用于当前答案。
因此“每张图只发送一次”在这里表示每张用于决策的图最多上传一次，**不是将所有 2 FPS 图都传给 API**。
这符合现有规则中由 Agent 自行决定模型使用哪些观测的边界，输入策略会保存于 Agent 配套汇总。

## 动作反馈与公平性

默认 `feedback="minimal"`，只在有相应公开字段时发送：

- 平移动作：`{"collision": true/false}`。
- pick／place／open／close：`{"success": true/false}`，只表示该动作执行结果。
- reset：公开 `reset_applied` 转为 `{"success": true/false}`，不提供恢复后的坐标。
- 转向、观察、no_op：不附加反馈；缺失字段不推定成功。

保留理由：碰撞与抓放执行失败不能仅凭“发出了动作”可靠确定。没有反馈时模型更容易积累错误状态，
增加无效步数；这些标志提供执行层反馈，不泄露目标身份、目标方向、距离或完整任务是否成功。
精确 `executed_distance` 会提供额外里程计信息，原因文本可能提供多余语义，均不发送。

当前公共协议允许普通动作反馈；SimpleNav 的适配器也读取碰撞、在指令中提示受阻并据此恢复，
因此最小反馈有现有基线依据。但这只说明没有越过公开输入边界，不代表已经证明所有模型的传感器信息完全一致。
严格比较应记录并对齐各模型实际使用的信息；原 SimpleNav 还可能使用持续位姿等信息。
若评测目标为纯视觉推断，可设 `feedback="none"`，并将其作为单独输入配置比较，不能把两种配置混算。

## 会话、输出与错误

Session 握手本地处理，不发 API 请求。序列首次决策不带父响应；之后每次带
`previous_response_id=<上一成功响应 ID>`，并设置 `store=true`。
新 run_id 必须有 reset_memory 握手；开始新序列时丢弃父响应引用，确认后建立独立链。
这是逻辑隔离，不代表删除服务端旧记录。真实 API 请求会在服务端存储会话内容。

输出只含 action、answer、done：

```json
{
  "action": {"action": "move_forward", "distance": 0.25, "angle_degrees": null, "target": null},
  "answer": null,
  "done": false
}
```

v5 问答通过 step / `actions_and_answer`，最终回答可使用 no_op 与 `done=true`。
抓取使用公开 target，place 在当前位置释放；动作仍受 `allowed_actions` 限制，成功由 Benchmark 判定。
本地不改写模型动作、不自动生成恢复动作。`request_id` 和 `observation_frame_id` 由共用接收器补齐。

配置 `truncation="disabled"`：上下文用满、会话过期或服务不支持继承时报接入错误；
不自动清空会话、重发历史、概括记忆或截断序列来声称完整成绩。
HTTP 错误、限流、超时、无效输出、拒答和身份异常均作为 `integration_error`，**不自动重试**，
避免未知完成状态下重复上传图像。调用总等待默认 115 秒，为 Benchmark 120 秒留余量。
超时后不采用迟到响应，后续调用停止；已经发出的远端请求仍可能计费。

## 配置与启动

本仓库仅发布 SimpleNav 配置模板。API 实验需自行准备 `configs/openai_api/agent.local.json`，
必填字段为 `model`、`log_dir`，其他可选字段及默认值见
[`load_config` 与 `DEFAULTS`](../benchmark/agents/openai_api.py)。本机配置不纳入版本控制。
在 `longterm_benchmark/` 执行：

```bash
./scripts/bootstrap.sh
# 先准备本机配置，并核实 model、endpoint、推理设置、feedback 和 log_dir。
./.venv/bin/python examples/openai_api_agent.py \
  --config "$PWD/configs/openai_api/agent.local.json" --check-config
```

模型 ID 必须经服务端确认，默认输出预算 2048，`reasoning_effort=null` 表示不传该参数；
不能据此假定无推理 token。`image_detail=auto`，按目标服务确认图像计量后固定。
`--check-config` 不读取密钥、不连接网络或创建日志。密钥仅从 `api_key_env` 指定的环境变量读取。
所有相对文件路径按配置文件目录解析，此 JSON 只由 Agent 读取，不传给 SimpleNav 实验管理器。

旧版 `history_frames`、记忆预算、重试配置及 `api=chat_completions` 已移除；本机配置须通过 `--check-config`。
不自动降级为无状态调用，否则无法满足系统 prompt／图像只上传一次以及跨任务记忆要求。

以下实跑命令不属于离线检查，需 API 凭据与 Isaac：

```bash
API_AGENT_CONFIG="$PWD/configs/openai_api/agent.local.json"
API_AGENT_CMD="$(./.venv/bin/python -c 'import shlex,sys; print(shlex.join(sys.argv[1:]))' \
  "$PWD/.venv/bin/python" "$PWD/examples/openai_api_agent.py" --config "$API_AGENT_CONFIG")"
ISAACSIM_PYTHON="/path/to/isaacsim/python.sh"
./.venv/bin/python -m benchmark validate ../OSS/scene000010/session.tar
./.venv/bin/python -m benchmark run --dataset ../OSS/scene000010/session.tar --sequence seq_0001 --dry-run
```

小预算诊断仍保留完整序列，只降低每任务步数：

```bash
"$ISAACSIM_PYTHON" -m benchmark run \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 \
  --agent-cmd "$API_AGENT_CMD" --max-steps 20 --agent-timeout 120 \
  --output results/gpt6_astra_api_diagnostic/benchmark
```

正式执行前换新的配置／输出目录、核实真实 usage 与上下文容量，再从初始状态运行：

```bash
"$ISAACSIM_PYTHON" -m benchmark run \
  --dataset ../OSS/scene000010/session.tar --sequence seq_0001 \
  --agent-cmd "$API_AGENT_CMD" --official \
  --output results/gpt6_astra_api_official/benchmark
```

正式配置的完整序列、2000 步／任务、120 秒／请求、256×256、100° FOV 等规则仍由通用运行器执行。
检查运行完成状态、evaluation_valid、metrics_complete 和 metric_version；Agent 输入策略另见配套日志。
`benchmark analyze <运行目录> --output <新报告目录>` 生成任务评分。

## Token 与费用：上传一次和计费一次不同

`previous_response_id` 避免客户端重复上传，但历史消息、图像和模型输出仍可能被服务端再次纳入
后续上下文并计为输入 token。缓存命中可降单价，不等于历史免费；不能按“本轮只有一张图”推定账单。
费用按每次 API 返回的 input_tokens、cached_input_tokens、output_tokens 累计，
不要用本地上传字符数代替。输出 usage 已包含的 reasoning_tokens 不再相加一次。

以 82 个任务作**参数化演算**，不冒充 GPT6 Astra 的实测 tokenizer：一次性 prompt／介绍等 1000 token，
每决策新增图像与短反馈等 300 token、当前输出 Schema 开销 200 token、输出 100 token（含推理）。
图像计量、Schema 实际开销、输出推理长度、缓存与单价均须实测校准。

| 平均决策／任务 | 调用 N | 新上传输入（含每轮 Schema） | 输出 | 新增输入＋输出 |
|---|---:|---:|---:|---:|
| 20 | 1,640 | 82.1 万 | 16.4 万 | 98.5 万 |
| 50 | 4,100 | 205.1 万 | 41 万 | 246.1 万 |
| 200 | 16,400 | 820.1 万 | 164 万 | 984.1 万 |
| 500 | 41,000 | 2050.1 万 | 410 万 | 2460.1 万 |

如果服务端完整保留并重复计量历史，设一次性内容 S、新增输入 I、每轮 Schema K、输出 O：

```text
新上传输入 ≈ S + (I + K) × N
累计计量输入 ≈ (S + K) × N + I × N × (N + 1) / 2 + O × N × (N - 1) / 2
最终上下文规模 ≈ S + (I + O) × N
费用 ≈ (非缓存输入 × 输入单价 + 缓存输入 × 缓存单价 + 输出 × 输出单价) / 1,000,000
```

按上表假设，N=4100 的完整历史累计输入约 33.67 亿，最终上下文约 164.1 万；
N=16400 时最终上下文约 656.1 万。是否能完整执行取决于目标模型的上下文容量，不能假定都能装下。
这些是完整历史保留的数学情形；真实 API 是否压缩及如何计量尚未核实。
如果假设输入 $5／百万、缓存输入 $0.5／百万、输出 $30／百万，并理想缓存所有可复用前缀，
N=4100 仍约 $1707；把每段内容错误地只计一次会得到约 $23，二者不能混淆。
上述价格均为演算假设，**不是 GPT6 Astra 官方报价或实际账单预测**。

离线回归：`python -m benchmark check --core` / `python -m benchmark check`。
配套日志定义见 [OUTPUTS](OUTPUTS.md#api-agent-配套产物)。本次修改未启动真实 API 或 Isaac。
