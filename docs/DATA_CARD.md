# Data Card

当前示例数据集：dataset/scene000430。

- 1 个住宅场景；
- 10 个 parent-linked sessions；
- 每个 session 至少包含 navigation、manipulation 和 QA；
- 任务、导航栅格、对象候选和 ground truth 由 CL_session_builder 编译；
- manipulation 是高层逻辑 pick/place，不代表机械臂物理抓取性能；
- 评测数据是 synthetic scene package，暂无 train/validation/test split；
- 语言任务文本为英文，场景分布和任务难度尚未完成公开基准校准。

场景 USD、纹理和上游 ArchVis 资产的再分发必须经过逐项许可证审查。发布前使用
scripts/make_checksums.py 生成 hash，并同步维护 NOTICE、来源和许可证清单。
