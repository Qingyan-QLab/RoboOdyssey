# Stretch3 默认本体与评测身份

默认使用随 Benchmark wheel 分发的 **Stretch3 `stretch3_kinematic_circle_v1`**。
场景准备、实际碰撞、导航／搬运参考和正式评测使用同一 profile；不需要私有仓库或同级 `robot/` 目录。
显式 `--robot cylinder` 保留原半径 0.10 m 的无网格逻辑本体，属于非官方实验。

## 固定几何与运行边界

| 项目 | 当前值／定义 |
|---|---|
| 碰撞圆心 | USD 机器人根坐标的 XY |
| 可见网格最大 XY 距离 | 0.2764972449 m，固定收臂姿态，共 36 个网格、117175 个顶点 |
| 安全余量与取整 | 加 0.01 m，再向上取整至 0.01 m |
| 默认碰撞半径 | **0.29 m** |
| 可见身体宽／高 | 0.3701 m／1.4098 m |
| 虚拟评测相机高度 | **1.50 m**，独立于身体高度 |
| 动作／碰撞／抓放 | `high_level_xy_yaw`／`2d_circle`／`logical_pick_place` |

旧 0.36 m 来自轴对齐包围盒角点估计，已由实际顶点包围圆替换。
0.1850 m 是身体宽度的一半，根坐标并非轮廓中心，不能用它代替根坐标包围圆。
本版使用对所有水平朝向保守的圆形代理，因此会拒绝部分真实轮廓可直行通过的窄门。
未来若改成带朝向轮廓，需同步改变场景派生、动作扫掠和评分参考，并发布新的评测身份。

本版仍支持高层横移、直接转向和逻辑抓放；没有差速底盘、机械臂可达性或动力学抓取。
固定身体圆不包含持物范围，不代表持物、伸臂、桌底／顶部碰撞已经完整验证。
相机并非真机传感器安装复现。运动和持物边界见 [MOTION](MOTION.md)。

## 统一场景和运行身份

新版场景的 `scene_version.json.robot_profile` 保存完整本体身份；导航 NPZ 的
`robot_profile_json` 同步保存身份，`robot_radius` 必须等于其中的半径。
身份包括版本、profile 与资产哈希、半径、相机高度、落地偏移及动作／碰撞／抓放模式。
公共冻结读取检查导航与 manifest 一致性；运行工厂检查场景与所选本体一致性。

`--official` 仅接受命名的 Stretch3、代码固定的 profile／资产哈希和声明相同身份的场景。
直接修改 bundled JSON 不能生成新的官方身份。旧未声明本体的当前格式数据仍可普通读取／运行，
但不能自动成为 Stretch3 官方场景；应重新准备一个新 revision。
起点容纳不下所选本体时实际启动报错，不移动冻结起点。
`--dry-run` 校验文件、身份和任务，未加载 USD，不能证明几何可达。
新 Stretch3 和旧圆柱结果不能合并，比较器按完整评测身份分组。

## 运行与非默认机器人

```bash
/path/to/isaacsim/python.sh -m roboodyssey run \
  --dataset /path/to/collection --sequence seq_0001 \
  --agent-cmd '/path/to/agent/python /path/to/agent.py' --official --output results
```

默认不需 `--robot` 或资产目录。环境变量 `BENCHMARK_ROBOT_ASSETS` 可覆盖 CLI 命名选择的资产根目录；
正式运行仍校验固定哈希。直接 `EnvConfig()` 始终用 bundled canonical profile，不受环境变量影响。

其他可选命名本体为 `tiago`、`turtlebot3_burger`、`turtlebot4`、`go2`、`g1`，
须通过 `--robot-assets /path/to/robot` 提供其 `profiles/` 与 `usd/` 目录。
`--robot-profile /path/to/profile.json` 为自定义非官方入口，与 `--robot`、`--robot-assets` 互斥。

```bash
python -m roboodyssey robots
python -m roboodyssey robots --robot-assets /path/to/all-robot-assets --check
python -m roboodyssey run --dataset /path/to/legacy-data --robot cylinder --dry-run
```

`robots --check` 检查全部六种资产；仅安装默认 wheel 时其余五种会报告缺失，不能理解为 Stretch3 损坏。
不会自动下载。资产许可与来源随包保存在 `roboodyssey/scenes/robot_assets/`；
转换后的 Stretch USD 使用上游 Clear BSD 许可，保持逐文件完整性校验。
