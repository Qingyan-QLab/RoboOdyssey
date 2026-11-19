"""Generate/check a source file map without importing the inspected modules."""
import argparse
import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = 'docs/FILE_MAP.md'
# This is the one maintained purpose registry. Tree, links, symbols and aliases
# are derived from files, so documentation cannot silently omit a new source.
PURPOSES = {
    '.gitignore': '实验结果、本地配置、环境、缓存和资产的版本控制排除规则。',
    '.github/workflows/test.yml': 'GitHub Actions 的完整独立 CPU 回归、视频工具安装与清单检查。',
    'NOTICE': '第三方代码及授权声明。',
    'README.md': '安装、数据验证、实验执行、结果分析和扩展的使用入口。',
    'PROJECT_GUIDE.md': '稳定的架构边界和维护原则；逐文件清单集中在本页。',
    'environment.yml': '可选 Conda 主机环境依赖。',
    'requirements.txt': '主机 Python 的最小依赖。',
    'pyproject.toml': 'Python 包元数据、依赖、打包与测试配置。',
    'run_benchmark.sh': '兼容 Bash 启动器：选择 Isaac/主机 Python，准备 ROS 环境并调用通用 run。',
    'configs/simplenav/managed.example.json': '受管理实验配置示例，含模型服务、GPU、序列和自动报告。',
    'configs/simplenav/review.example.json': 'SimpleNav 单次 GUI/录像复核配置示例。',
    'docs/AGENT_INTEGRATION.md': '第三方 Agent 接入和适配示例。',
    'docs/OPENAI_API_AGENT.md': 'API Agent 精简输入、连续会话、反馈公平性、启动方式和上传/计费 token 区别。',
    'docs/PROTOCOL.md': 'Agent 请求/响应、动作、图像及记忆协议。',
    'docs/RULES.md': '运行与评分规则的唯一维护入口：连续性、预算、正式配置、修补、成功/排除、聚合、导航诊断及旧版差异。',
    'docs/SIMPLENAV_V5.md': 'SimpleNav v5 接入、受控诊断、取证及录像复现。',
    'docs/LAUNCH.md': '实验配置、预检、进程管理、终止和来源记录。',
    'docs/OUTPUTS.md': '实验/运行/Session 的输出文件、证据引用和新旧布局契约。',
    DOCUMENT: '自动生成的完整源码目录、文件用途和兼容模块索引。',
    'benchmark/__main__.py': 'python -m benchmark 的入口，转发 CLI。',
    'benchmark/cli.py': '统一命令分发、环境检查与数据下载；按需导入各功能。',
    'benchmark/run.py': '通用评测参数、数据分流、Isaac/Agent 生命周期和结果汇总。',
    'benchmark/agents/agent_api.py': '通过 JSONL 子进程协议与 Agent 交互，管理超时和退出。',
    'benchmark/agents/stream.py': 'Agent 侧持续接收观测与独立推理线程、请求关联及示例视觉历史。',
    'benchmark/agents/protocol.py': 'Agent 响应校验、协议版本、允许动作及统一动作名称读取。',
    'benchmark/agents/protocol_schema.json': 'Agent JSON 消息的机器可读结构。',
    'benchmark/agents/simplenav.py': 'SimpleNav 请求和动作适配、历史图像及不支持任务的响应。',
    'benchmark/agents/openai_api.py': 'API 视觉 Agent：连续响应链、单帧增量输入、最小反馈、动作校验和用量日志。',
    'benchmark/artifacts/layout.py': '结果布局唯一解析入口；共用 JSON/JSONL、文件与内容哈希、状态和证据引用。',
    'benchmark/evaluation/requests.py': 'v5 完整请求的运行时 GT 判定与任务计分。',
    'benchmark/evaluation/navigation.py': '实际任务起点网格的带权最短路、机器人半径、目标区域和有序路线动态规划。',
    'benchmark/evaluation/transfer.py': '搬运剩余交付成本、真实抓放约束、可移除占据网格与进展乘效率评分。',
    'benchmark/evaluation/task_metrics.py': '版本化导航进展×路径效率、搬运/问答部分分、真实耗时和评分完整性。',
    'benchmark/evaluation/rescore.py': '显式历史重算、验证归档及真实状态重建参考地图、历史耗时和分组统计。',
    'benchmark/evaluation/success.py': '旧版导航/搬运/问答成功判定与已满足检测。',
    'benchmark/evaluation/report.py': '共用计分分母、新旧请求/Session/序列及能力统计的唯一汇总实现。',
    'benchmark/evaluation/task_diagnostics.py': '从真实 GT/动作日志复算逐任务评分、距离、路径与 reset；不渲染报告。',
    'benchmark/evaluation/metrics/qa.py': '问答归一化、精确匹配和 token F1。',
    'benchmark/evaluation/judges/local_llm.py': '可选本地语言模型问答裁判；非默认 CPU 流程。',
    'benchmark/launch/simplenav.py': '配置解析、命令构建、CPU 预检和共用启动 CLI；预检和 managed/external 实验入口。',
    'benchmark/launch/experiment.py': '管理模型服务和序列进程、读取进度、终止清理及自动分析。',
    'benchmark/launch/sequences.py': '序列选择、独立进程运行、共用进程组清理与集合汇总。',
    'benchmark/launch/provenance.py': '记录源码/数据/权重哈希、Git 状态和 Python 环境。',
    'benchmark/legacy/task_repair.py': '旧 schema 4 的实际状态修补实现。',
    'benchmark/legacy/task_history.py': '旧 schema 4 的真实执行历史实现。',
    'benchmark/reporting/analyze.py': '统一离线分析分发：实验/集合递归、新旧契约选择和 HTML 索引。',
    'benchmark/reporting/request_audit.py': 'v5 请求、GT、来源失败和视频的修补审计。',
    'benchmark/reporting/task_diagnostics_report.py': '逐任务分数/导航/reset 的 HTML、CSV、JSON 和 Markdown 展示；共用评分报告入口。',
    'benchmark/reporting/legacy.py': '旧版任务统计、轨迹、HTML 报告及运行时修补审阅。',
    'benchmark/runtime/session_runner.py': '公共 Agent 交互、旧版 Session 执行与 v5 请求分流。',
    'benchmark/runtime/observations.py': '真实时间 2 FPS 采样时钟与漏采审计；采到的帧立即交给发送方，不缓冲。',
    'benchmark/runtime/request_runner.py': 'v5 请求修补、真实状态、名称账本、连续历史和任务执行。',
    'benchmark/runtime/request_events.py': '任务/动作/reset/mutation 事件，GUI 和录像联动。',
    'benchmark/runtime/recording.py': 'CPU ffmpeg 录像、帧时间索引和编码器收尾。',
    'benchmark/runtime/instruction.py': '旧版任务介绍与语言指令组装。',
    'benchmark/runtime/reference.py': '旧版基于实际状态的导航参考与可达性。',
    'benchmark/scenes/dataset_loader.py': '新旧数据集、场景集合、Session 和冻结资源的统一读取。',
    'benchmark/scenes/state_composer.py': '初始状态与 Session 状态变更合成。',
    'benchmark/scenes/isaac_env.py': 'Isaac 场景、相机、导航/搬运、碰撞、reset 及权威世界状态。',
    'benchmark/scenes/request_evidence.py': '面向评分器的当前帧可见性与检查取证。',
    'benchmark/scenes/holoagent_sensor_bridge.py': 'HoloAgent ROS 传感器发布桥接。',
    'benchmark/validation/dataset.py': '外部旧格式和 v4/v5 归档/集合校验；文件与场景资产完整性检查。',
    'benchmark/validation/inventory.py': '静态扫描文件、用途和主要定义，生成/检查本页；不导入被扫描模块。',
    'benchmark/viewer/viewport.py': 'Isaac 视口相机绑定、窗口布局与渲染同步。',
    'benchmark/viewer/viewer_overlay.py': 'GUI 任务、状态、成功/失败和 reset 信息覆盖层。',
    'examples/random_agent.py': '随机动作 Agent，用于协议/仿真冒烟验证。',
    'examples/simplenav_agent.py': 'SimpleNav JSONL 协议入口，复用 agents/simplenav.py。',
    'examples/openai_compatible_vlm_agent.py': 'OpenAI 兼容视觉语言服务的 Agent 示例。',
    'examples/openai_api_agent.py': 'API Agent 的 JSONL 薄入口，提供配置检查和持续收帧。',
    'examples/holoagent_agent.py': 'HoloAgent HTTP 服务到 Benchmark 协议的适配。',
    'examples/holoagent_benchmark_server.py': 'HoloAgent/VLM 服务、规划与动作桥接。',
    'examples/holoagent_sim_gateway.py': 'HoloAgent 与仿真侧交互网关。',
    'scripts/bootstrap.sh': '创建/安装最小主机 Python 环境。',
    'scripts/run_holoagent.sh': 'HoloAgent 模型/VLM 服务与仿真联动启动。',
    'scripts/check_simplenav_cpu.py': '禁止导入模型/仿真 SDK 的 CPU 回归执行器。',
    'scripts/check_runtime_tasks.py': '旧 schema 4 修补与历史连续性的合成验证。',
    'scripts/check_request_evidence.py': '真实 Isaac 当前帧可见性取证检查；需要 GPU。',
    'scripts/run_request_diagnostics.py': '完整发布序列上的私有失败注入诊断；--prepare-only 仅校验选择，实际执行需要 Isaac。',
    'tests/test_benchmark.py': '临时合成数据、协议、环境、动作、评分及显式数据路径 CLI 基础回归。',
    'tests/test_observations.py': '固定时钟、慢 Agent 持续收帧、阻塞接收/渲染失败和 JSONL 通信 CPU 回归。',
    'tests/test_openai_api_agent.py': '模拟 API 的 prompt/图像去重、响应链隔离、输入边界、最小反馈和计量回归。',
    'tests/test_contextual_requests.py': 'v5 请求、实际 GT、名称/指代与评分契约回归。',
    'tests/test_predispatch.py': '下发前已满足、排除、记分与报告回归。',
    'tests/test_project_workflow.py': '新旧输出、入口兼容、源码清单、分析和托管进程生命周期回归。',
    'tests/test_runtime_repair.py': '旧版实际状态修补与连续执行回归。',
    'tests/test_scene_sequences.py': '场景多序列选择、隔离进程及集合汇总回归。',
    'tests/test_simplenav_v5.py': 'SimpleNav、预检、录像、取证与资源清理回归。',
    'tests/test_task_diagnostics.py': '正式分、目标顺序、导航距离、路径及 reset 诊断回归。',
    'tests/test_evaluation_metrics.py': '部分得分、绕路/墙体/目标顺序、reset 计费、参考哈希及新旧评分回归。',
    'tests/test_transfer_metrics.py': '搬运连续进展、抓放范围/偏移、错误放置、绕路/reset、参考保存和证据完整性回归。',
    'tests/test_task_repair.py': '旧 schema 4 修补规则和历史因果回归。',
    'benchmark/data/archive.py': '发布 tar 的只读源解析、独立哈希缓存、包内路径与文件类型检查。',
    'benchmark/data/frozen.py': '场景/版本/序列 manifest、共享资源、冻结契约校验和最小 mutation 上下文；不编译任务。',
    'docs/DATA_FORMAT.md': 'session.tar 发布层级、缓存、版本、共享引用、校验和公开/私有边界。',
    'tests/frozen_data.py': '选择仓库内已冻结合成样例并写入临时测试归档；不调用任务生成器。',
    'tests/fixtures/contextual.json': '合成回归数据：场景、已冻结场景变体、模板/GT 和语言目录。',
    'tests/test_published_data.py': '发布 tar/包装目录、缓存/篡改拒绝、任务顺序、独立导入及真实 mutation 几何回归。',
    'benchmark/contracts/LICENSE': '迁入运行契约与合成测试样例代码的原 MIT 版权和许可。',
    'benchmark/contracts/language.py': '情境日历、历史时间、公开语言、名称和冻结语言目录的统一校验/渲染。',
    'benchmark/contracts/contextual.py': 'v3 实际状态归一化、名称/历史/GT 修补、失败因果和 mutation 决策。',
    'benchmark/contracts/doors.py': '冻结开门策略、USD 门几何应用与回读验证。',
    'benchmark/contracts/legacy_templates.py': '旧 schema 4 模板读取校验及重放浮点/结构比较。',
    'benchmark/contracts/navigation_io.py': '无 pickle 的导航 NPZ 读取与确定性测试数据写入。',
    'benchmark/contracts/placement.py': '实际状态下地面放置、碰撞和可接近性几何计算。',
    'benchmark/contracts/raster.py': '房间/对象几何与开门策略的动态碰撞栅格。',
    'benchmark/contracts/references.py': '元数据对象身份、区域名称、真实历史和位置变化的统一指代校验/修补。',
    'benchmark/contracts/requests.py': '完整请求的实际历史记录、旧契约修补和 mutation 放置校验。',
    'benchmark/contracts/navigation.py': '地面网格、对象标签、接近位置、路径和 mutation 连通性校验。',
    'benchmark/contracts/state.py': '对象真实几何、位置签名、冻结参考缓存和哈希。',
}
SKIP_DIRS = {'.git', '.venv', '.cache', '.pytest_cache', '__pycache__', 'results',
             'outputs', 'dataset', 'dataset_downloads', 'node_modules', 'build', 'dist'}


def source_files(root):
    """Prune generated/local trees before descending; also works without Git."""
    def walk(folder):
        for path in sorted(folder.iterdir()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or path.name in SKIP_DIRS or path.name.endswith('.egg-info'):
                continue
            if path.is_dir():
                yield from walk(path)
            elif not (path.suffix in {'.pyc', '.pyo'} or '.local.' in path.name):
                yield relative
    return sorted(set(walk(root)) | {DOCUMENT})


def describe(root, name):
    path = root/name
    definitions = []
    if path.suffix == '.py':
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=name)
        definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and not node.name.startswith('_')]
        # Detect both module aliases and compatibility script forwards.
        doc = ast.get_docstring(tree) or ''
        if doc.startswith(('Compatibility ', 'Legacy report entry;')):
            target = next((node.args[0].value for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == 'import_module' and node.args
                           and isinstance(node.args[0], ast.Constant)), None)
            if target is None:
                target = next((node.module for node in tree.body if isinstance(node, ast.ImportFrom)
                               and (node.module or '').startswith('benchmark.')), None)
            if target:
                return '兼容入口，转发到 `' + target + '`；不维护第二份业务实现。', definitions
        if path.name == '__init__.py':
            return 'Python 包入口。' + ((' ' + doc) if doc else ''), definitions
    purpose = PURPOSES.get(name)
    if purpose is None:
        raise ValueError('请补充文件用途 PURPOSES: ' + name)
    return purpose, definitions


def module_aliases(root, files):
    """Read package alias declarations statically and validate their targets."""
    aliases = {}
    for name in files:
        if not name.endswith('/__init__.py'):
            continue
        tree = ast.parse((root/name).read_text(encoding='utf-8'), filename=name)
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'MODULE_ALIASES'
                            for target in node.targets)):
                package = str(Path(name).parent).replace('/', '.')
                for alias, target in ast.literal_eval(node.value).items():
                    destination = target.replace('.', '/') + '.py'
                    if destination not in files:
                        raise ValueError('兼容模块目标不存在: ' + target)
                    aliases[package + '.' + alias] = (target, destination)
    return aliases


def render(root=ROOT):
    root = Path(root)
    files = source_files(root)
    stale = set(PURPOSES) - set(files)
    if stale:
        raise ValueError('文件用途登记已过期: ' + ', '.join(sorted(stale)))
    groups = defaultdict(list)
    tree = {}
    for name in files:
        purpose, definitions = describe(root, name)
        symbols = ', '.join('`' + s + '`' for s in definitions[:8])
        if len(definitions) > 8:
            symbols += f' 等 {len(definitions)} 个定义'
        groups[str(Path(name).parent)].append(
            f'| [{Path(name).name}](../{name}) | {purpose} | {symbols or "—"} |')
        branch = tree
        for part in Path(name).parts:
            branch = branch.setdefault(part, {})
    lines = ['# 项目文件清单', '',
             '由 `python -m benchmark inventory` 生成；`python -m benchmark inventory --check` 检查遗漏和过期。',
             '只维护生成器中的文件用途；目录、链接、主要定义和兼容转发目标从源码提取。无需同步改写 PROJECT_GUIDE。', '',
             f'本页逐项列出 **{len(files)} 个源码、配置、文档和测试文件**。',
             '不导入被检查模块，不读取模型、视频或场景资产内容，不启动 Isaac/GPU。',
             '运行产物/本地文件按末尾目录模板说明，不逐项列出每次实验产生的重复文件。', '',
             '推荐使用：`python -m benchmark validate / preflight / experiment / run / analyze / check`。',
             'scripts 只保留安装、联合启动及专项检查；重复包装入口已移除。',
             '数据由发布 session.tar 或解包目录提供；运行与测试不依赖生成项目；统一 analyze 写入新目录。', '',
             '## 完整源码目录', '', '```text', 'RoboOdyssey/']
    def emit(branch, prefix=''):
        items = sorted(branch.items(), key=lambda item: (not bool(item[1]), item[0]))
        for i, (name, children) in enumerate(items):
            last = i == len(items)-1
            lines.append(prefix + ('└── ' if last else '├── ') + name + ('/' if children else ''))
            if children:
                emit(children, prefix + ('    ' if last else '│   '))
    emit(tree)
    lines += ['```', '', '## 文件内容与职责', '']
    for group, rows in sorted(groups.items(), key=lambda item: (item[0] != '.', item[0])):
        lines += ['### ' + ('根目录' if group == '.' else group), '',
                  '| 文件 | 内容 / 作用 | 主要函数 / 类 |', '|---|---|---|', *rows, '']
    aliases = module_aliases(root, files)
    if aliases:
        lines += ['## 保留的导入路径', '',
                  '以下路径由各包 `__init__.py` 的 `MODULE_ALIASES` 集中注册；不再单独生成转发文件。',
                  '旧路径和规范路径指向同一模块对象，保留旧导入和 mock/patch；内部代码使用规范路径。', '',
                  '| 保留导入路径 | 唯一实现 |', '|---|---|']
        for alias, (target, destination) in sorted(aliases.items()):
            lines.append(f'| `{alias}` | [{target}](../{destination}) |')
        lines.append('')
    lines += ['## 动态目录与本地文件', '',
              '| 目录 / 文件 | 内容与维护方式 |', '|---|---|',
              '| `results/<experiment-or-run>/` | 实验配置、来源、状态、序列、Session 原始证据、视频和可重建报告；逐文件定义见 [OUTPUTS.md](OUTPUTS.md)。历史结果保留原格式，实验/整理说明随各自结果归档。 |',
              '| `outputs/runtime_repair_validation/` | 旧修补合成验证输出；由检查脚本重建。 |',
              '| 外部数据目录（可选本地 `dataset/`、`dataset_downloads/`） | 数据由使用者提供，通过 --dataset/配置指定；不随源码分发或生成。 |',
              '| `configs/<model>/*.local.json` | 按模型集中存放的本机路径、GPU/端口配置；不纳入共享源码。 |',
              '| `.venv/`、`*.egg-info/`、`build/`、`dist/` | 可重建的安装/打包文件。 |',
              '| `__pycache__/`、`.pytest_cache/`、`.cache/`、`*.py[cod]` | 可重建缓存，不属于项目接口。 |',
              '| `.git/` | Git 元数据和 LFS 对象。 |', '',
              '新增文件需补充 `benchmark/validation/inventory.py` 中的用途（包入口和兼容转发自动识别），',
              '然后运行 `python -m benchmark inventory`。文件新增/删除、主要定义变化或清单被手工改动后，',
              '`--check` 会返回非零；生成不带时间戳，不因重复执行产生无意义差异。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the file map is incomplete or stale; do not write')
    args = parser.parse_args(argv)
    try:
        text = render()
    except (ValueError, SyntaxError) as exc:
        parser.exit(1, str(exc) + '\n')
    path = ROOT/DOCUMENT
    if args.check:
        if not path.is_file() or path.read_text(encoding='utf-8') != text:
            parser.exit(1, 'File map is stale; run python -m benchmark inventory\n')
        print('File map is complete and current.')
    else:
        path.write_text(text, encoding='utf-8')
        print(path)
    return 0
