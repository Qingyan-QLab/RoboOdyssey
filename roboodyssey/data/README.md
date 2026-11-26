# Scene 与 Session 数据接口

`sources.py` 提供 `load_scene`、`load_session`、`BenchmarkSuite` 和发布包兼容入口；`archive.py` 安全解析目录／tar 并使用独立缓存；`frozen.py` 校验既有发布契约。资源清单显式列出场景资产、导航、初始状态、公开任务及私有 GT，所有文件按 SHA-256 验证。格式、调用示例和能力边界见 [SCENE_SESSION](../../docs/SCENE_SESSION.md)。
