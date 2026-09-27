# Actor SDG

这里的脚本是可选的数据生成工具，不是 long-term benchmark 的运行时依赖。

运行前需要安装包含 Replicator Agent 扩展的 Isaac Sim，并设置 experience：

```bash
export ISAACSIM_ROOT=/path/to/isaacsim
export EXP_PATH="${ISAACSIM_ROOT}/apps"
"${ISAACSIM_ROOT}/python.sh" generation/actor_sdg.py \
  --config_file generation/configs/sample_config.yaml
```

也可以显式传入 `--experience`。配置中的 Isaac 内置资产路径必须在当前 Isaac Sim
安装中存在。生成出的数据应放到外部数据目录，经过整理、授权确认和校验后再作为
benchmark dataset 发布；不要让评测入口自动触发 Actor SDG。
