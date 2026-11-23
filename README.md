# RoboOdyssey｜万境长行

RoboOdyssey 是面向长期具身任务的仿真交互与评测框架。核心库负责加载场景和任务、运行环境、记录交互证据并评分；被评 Agent 与数据合成代码分别位于独立仓库。当前仿真后端使用 Isaac Sim，支持导航、搬运和问答任务。

## 从哪里开始

| 需求 | 文档 |
|---|---|
| 了解模块和调用关系 | [架构说明](docs/ARCHITECTURE.md)、[核心库](roboodyssey/README.md) |
| 加载 Scene／Session 目录或 tar | [独立输入指南](docs/SCENE_SESSION.md) |
| 使用已有发布数据包运行评测 | [运行指南](docs/LAUNCH.md)、[数据格式](docs/DATA_FORMAT.md) |
| 接入被评 Agent | [Agent Integrations](https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations)、[协议](docs/PROTOCOL.md) |
| 生成或导出新任务数据 | [DataSynth](https://github.com/Qingyan-QLab/RoboOdyssey-DataSynth) |
| 查看指标与结果 | [评分规则](docs/RULES.md)、[输出说明](docs/OUTPUTS.md) |
| 按改造计划继续工作 | [11.23 框架改进说明](11.23重要框架改进说明.md) |

## 快速使用

使用 Python 3.10+。主机环境可运行数据校验、CPU 测试和结果分析；实际仿真另需 Isaac Sim。模型及权重由各 Agent 集成单独提供。

```bash
./scripts/bootstrap.sh
python3 -m roboodyssey --help
python3 -m roboodyssey validate /path/to/session.tar
python3 -m roboodyssey run --dataset /path/to/session.tar --sequence seq_0001 --dry-run
```

独立 Scene 与 Session 均以**目录或 tar 包**传入，包内 manifest 列出 3D 文件和其他资源：

```bash
python3 -m roboodyssey run \
  --scene-source /path/to/scene_bundle \
  --session-source /path/to/session_bundle.tar \
  --dry-run
```

实际运行可通过 `--agent-cmd` 指定 Agent，并使用 Isaac Sim 的 Python 启动；参数、输入格式和示例见[独立输入指南](docs/SCENE_SESSION.md)。运行结果写入独立的 `results/` 子目录。

## 开发检查

```bash
python3 -m roboodyssey check
python3 -m roboodyssey inventory --check
```

完整文档见 [docs 索引](docs/README.md)，逐文件清单见 [FILE_MAP](docs/FILE_MAP.md)。场景资产、发布数据、模型源码及权重不随本仓库分发；第三方声明见 [NOTICE](NOTICE)。
