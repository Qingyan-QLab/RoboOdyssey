# 指标

- Navigation：目标位置误差、作者轨迹长度、agent 路径长度、SPL、碰撞数；
- Manipulation：GT 要求的对象关系是否全部满足；
- QA：规范化 exact match 和 accepted answers；
- Memory capability：带 memory capability 的 task 成功率，按来源 session horizon
  聚合；
- Task score：session 内所有 task 的平均成功率；
- Session success：session 内所有 task 成功；
- Reset-aware score：task_score / (1 + reset_count)。

缺失 task、非法响应、超时和未完成任务都计为失败。指标只读取运行时
metrics.json，不修改源数据。
