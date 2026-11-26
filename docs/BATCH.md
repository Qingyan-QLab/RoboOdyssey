# 通用实验批次与比较

`python -m roboodyssey batch` 接受任意兼容 JSONL 协议的 Agent 命令。每条 job 选择一个冻结 sequence；
CPU 规划不启动模型/Isaac、不探测端口。外部模型服务由调用方管理，SimpleNav 托管服务入口位于外部 Agent Integrations 仓库。

```json
{
  "schema_version": 1,
  "python": "/path/to/isaac/python.sh",
  "agent_argv": ["/path/to/host/python", "/path/to/agent.py"],
  "jobs": [
    {"id": "sequence_A_seed_1", "dataset": "/data/collection", "sequence_id": "sequence_A"}
  ],
  "run_options": {"max_steps": 2000, "agent_timeout": 120, "official": true, "gpu": 1},
  "max_attempts": 2,
  "attempt_timeout_seconds": 86400,
  "environment": {}
}
```

```bash
python -m roboodyssey batch experiment.json --output plan_review
python -m roboodyssey batch experiment.json --output new_experiment --execute
```

两个 output 均须为新目录。规划冻结输入 ID/revision/hash、完整 argv、运行目录、预算与重试策略。
不含模拟资产的 CPU 合成场景会产生明确失败的预检结果；不能启动假运行。每次真实启动前重新校验全部冻结内容。
Agent 参数中的数据/配置路径建议使用绝对路径；相对路径基于 experiment.json 所在目录。
模型内部随机种子由 Agent 参数明确指定；job ID 中的 seed 标签不会自动设置模型或物理引擎种子。

执行写入 `plan.json`、`jobs.json`、`manifest.json`、`logs/` 与 `attempts/<job>/<编号>/`。
每次 attempt 都从规范初态和新模型记忆开始，由公共运行器执行完整 sequence；不能恢复物理中间状态。
仅进程异常、超时或无效执行证据可触发有限重试；已完成的任务失败、低分和缺失参考不会因重试挑选更好结果。
数据/身份/格式契约错误不可重试；每次失败都保留，首个执行有效的完成结果被选中。
即使完成后 SDK 清理导致非零退出，已有完整有效结果也不重跑，清理问题单列；不计算 best-of-retry 分数。
启动前失败也有独立 attempt_status.json。environment 只用于非敏感配置；凭据从父进程环境继承，避免写入归档的配置文件。

创建 `new_experiment/STOP` 或向批次进程发 SIGINT/SIGTERM 会终止自有子进程组，并将未开始 job 标为 cancelled。
超时会终止并回收自有进程组；所有计划 job 有 completed/failed/cancelled 终态。
退出码：0 全部完成，1 基础设施失败，130 取消。`summary.json` 的 formal_score 始终为空；
批次管理不把重试混入正式评分分母，比较要显式选择各 job 的 attempt。

## 比较保存结果

```json
{
  "baseline": "model_A",
  "groups": {
    "model_A": [{"run": "/data/A/attempts/job/001", "replicate": "seed_1"}],
    "model_B": [{"run": "/data/B/attempts/job/002", "replicate": "seed_1"}]
  }
}
```

```bash
python -m roboodyssey compare compare.json --output new_comparison
```

按 scene/sequence/replicate 配对；同一组重复 attempt 必须显式选择，不自动取最优或最后一次。
数据哈希、runtime/成功规则、评分/参考模型、抓放参数、本体、动作、观测与预算必须一致。
旧结果缺少这些身份或配置字段时只报告不可比，不能仅凭包版本、指标名或相同文件名混算。
缺失、执行失败、任务群体差异均保留；存在无效配对时整体均值为 null，而不是删除失败后统计。
输出 `comparison.json` 与 `pairs.csv`，逐任务给出差值、排除原因、任务类型/能力分组；
历史跨度基于保存的 source_context，证据缺失或两侧不一致为 unknown。
重复样本的 replicate 是实验标签，不证明仿真确定性；当前不计算置信区间。
