# 配置与可复现实验入口

独立 Scene／Session 目录或 tar 的用法见 [SCENE_SESSION](SCENE_SESSION.md)。本页主要说明现有发布包与 SimpleNav 实验管理。

核心入口是 `python -m roboodyssey`；SimpleNav 实验入口是 `python -m agent_integrations.simplenav`。安装 Agent Integrations 后从其仓库根目录调用 SimpleNav 命令；可以用 `.venv/bin/python` 替代 `python`。
本页只维护配置与进程管理；序列执行、预算和评分约束见 [RULES](RULES.md)。

| 命令 | 作用 | 会启动 GPU 工作吗 |
|---|---|---|
| `validate <dataset>` | 归档／场景集合校验 | 否 |
| `doctor --dataset <dataset>` | 数据与主机依赖检查 | 否；`--require-isaac` 另检查 SDK |
| `run --dataset <dataset> --dry-run` | 序列与任务预览 | 否 |
| `agent_integrations.simplenav preflight --config <json>` | SimpleNav 配置、源码、数据、动作统计、CPU 编码器预检 | 否 |
| `agent_integrations.simplenav experiment --config <json>` | 同预检 | 否 |
| `agent_integrations.simplenav experiment --config <json> --execute` | 管理模型与 Isaac、记录状态、自动分析和清理 | 是 |
| `run --dataset ... --agent-cmd ...` | 用 Isaac Python 执行通用 Agent | 是 |
| `analyze <run> --output <新目录>` | 运行与实验结果的离线分析 | 否 |
| `batch <config> --output <新目录>` | 通用串行批次计划；执行参数见 [BATCH](BATCH.md) | 默认否 |
| `compare <config> --output <新目录>` | 相同评测条件下严格配对比较，参数见 [BATCH](BATCH.md) | 否 |
| `robots` | 列出本体与本地资产状态；`--check` 另检查全部六类外部资产 | 否 |
| `check` | 独立完整 CPU 回归；禁止 Builder／模型／仿真导入 | 否 |
| `check --core` | 快速基础测试子集 | 否 |

## 配置

复制 `agent_integrations/simplenav/managed.example.json` 为同目录的 `managed.local.json`；已有模型服务时可使用
`agent_integrations/simplenav/review.example.json`。当前仅发布 SimpleNav 配置模板，模型专用配置与适配器集中放在 `agent_integrations/<model>/`。
本机配置不提交版本控制，但会复制到实验输出供复核。

dataset 可直接指定发布 tar；首次加载按 [数据契约](DATA_FORMAT.md) 解包到独立缓存并校验，全部 CLI 共用同一解析器。

必填路径为 dataset、simplenav_root、checkpoint、agent_python、isaac_python、output。
相对路径基于 JSON 所在目录，`~` 会展开。output 是输出父目录，每次运行创建唯一新目录。
checkpoint 可是权重文件或模型目录，动作统计按 SimpleNav ActionCodec 的路径规则读取；多数据键需 unnorm_key。

通用字段包括 max_steps、agent_timeout（默认预算及含义见 [执行规则](RULES.md#execution)）、history_frames（8）、
collision_reset_threshold（5，0 为关闭）、url、gpu、viewer、record_video、video_layout、
video_fps、video_event_seconds、ffmpeg、auto_analyze（true）。
`sequences` 是不重复的序列 ID 列表，与 `all_sequences` 互斥。
场景集合中只有一条序列时可省略选择，多条时必须明确选择。

`environment` 设置 Isaac／Agent 子进程环境，可指定 DISPLAY、XAUTHORITY 等。
管理器不把自身 CUDA_VISIBLE_DEVICES 或 CONDA_PREFIX 隐式带入 Isaac；需要时由 environment 显式指定。
Isaac/Vulkan 的 `gpu` 索引和模型的 CUDA UUID 属于不同编号空间，不作自动换算。

模型服务配置示例：

```json
{
  "model_service": {
    "mode": "managed",
    "gpu_uuid": "GPU-actual-device-uuid",
    "startup_timeout": 600,
    "extra_args": ["--use_bf16", "--idle_timeout", "3600"],
    "environment": {"OMP_NUM_THREADS": "4"}
  }
}
```

managed 只支持本地地址，启动 SimpleNav 的 `deployment.model_server.server_policy`。
checkpoint 与端口分别从主配置 checkpoint 和 url 取得，extra_args 不能覆盖这两个字段。
先检查端口未占用，再启动；TCP 可连接只代表端口就绪，实际协议／推理错误由 Agent 接入路径记录。
external 不启动或停止模型，仅在显式执行时检查连接；不会根据 PID／端口清理他人的服务。

## 生命周期与复现信息

1. CPU 预检失败则返回非零，不启动模型或 Isaac。
2. 新建实验目录，保存解析配置、预检、代码来源和主机／模型 Python 包列表、checkpoint／统计文件哈希。
3. 按模式启动或连接模型，启动各序列 Benchmark／Agent 进程。
4. runner 按 [执行规则](RULES.md#execution) 运行，记忆重置字段见 [协议](PROTOCOL.md)。
5. 持续增量读取完整事件行，生成 progress.json；记录所有子进程退出码，异常序列不能从汇总中删除。
6. 默认自动生成报告。原始执行、汇总和离线分析结果分别保存，分析失败不会伪装成模型失败。
7. SIGINT／SIGTERM、STOP 或普通异常进入收尾，先 TERM 后必要时 KILL，只处理本次创建的进程组并回收。

强制杀死管理器不能保证模型／视频收尾。中断后的重跑要求见 [执行规则](RULES.md#execution)。
录像和所有来源信息可复核，但不同 GPU、驱动、物理／模型随机性仍可能导致结果变化。
记录了代码哈希不等于自动保存 dirty 源文件；分发复现实验须同时保留对应源码与依赖。

## 入口与专项检查

发布数据从 `OSS/<scene_id>/session.tar` 或解包目录传入，使用 `--dataset`；独立输入使用 `--scene-source` 和一个或多个 `--session-source`。Bash 启动器也接受 `DATASET_DIR`。
数据校验和报告使用核心 CLI；SimpleNav 预检与实验使用其集成包入口。

- `validate <dataset>`：校验归档或场景集合。
- `python -m agent_integrations.simplenav preflight --config ...`：CPU 检查；不启动服务。
- `python -m agent_integrations.simplenav experiment --config ... --execute`：管理实验。已有模型服务使用 `model_service.mode=external`。
- `run --dataset ...` 或 `run --scene-source ... --session-source ...`：通用 Agent 执行；Bash 入口还负责 Isaac / ROS 环境。
- `analyze <run> --output <新目录>`：生成请求修补、评分诊断与报告索引；日志保持只读。
- [Agent Integrations](https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations) 中的 `scripts/run_holoagent.sh --dataset ...`：HoloAgent 服务组合；启动模型前先做 CPU 数据预览。
- `scripts/check_request_evidence.py` 和 `scripts/run_request_diagnostics.py`：真实取证／受控失败诊断；后者 `--prepare-only` 为 CPU，见 [SimpleNav 专题](https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations/blob/main/docs/SIMPLENAV_V5.md)。

真实运行的最小测试直接复用 `run --dataset ... --agent-cmd "<外部 Agent 命令>" --max-steps 20`，选择范围遵守执行规则。
普通 `analyze` 也可分析这些诊断结果。

## 公共执行与独立安装

安装 wheel 后，`python -m roboodyssey check` 检查包内公共模块和 `interaction` 协议资源；源码目录运行该命令则执行完整 CPU 回归。真实执行始终显式传入 `--agent-cmd`，核心不包含被评 Agent 实现。

默认本体为 `stretch3_kinematic_circle_v1`（0.29 m 圆代理、1.50 m 虚拟相机）。`--official` 要求冻结的本体、预算和场景声明；旧未声明场景仍可非官方运行。独立 Scene/Session 输入同样校验本体，`--sensor-plugin` 保持通用插件接入。详情见 [ROBOTS](ROBOTS.md) 和 [RULES](RULES.md)。
