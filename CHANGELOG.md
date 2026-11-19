# 变更日志

## 0.0.1（待发布）

- 提供 `roboodyssey` Python 包和 `python -m roboodyssey` 命令。已有脚本需更新 `benchmark` 导入前缀和命令模块名。
- 支持读取外部发布的场景与任务数据，执行连续 Session，记录证据并生成评分报告。
- 将 Isaac Sim 生命周期与场景初始化集中到后端模块，允许在主机 Python 中导入评测核心。
