# 重新训练

当前机器已有全部数据和权重，正常体验不需要重新执行。以下命令会覆盖同名研究数据、报告与权重，建议在新的项目副本运行，并先阅读数据许可。没有启动外部服务或上传数据。

在项目根目录依次执行：

```sh
mkdir -p data/processed artifacts reports
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/train.py --name v1 --epochs 12
.venv/bin/python scripts/train.py --name v2 --epochs 35 --resume artifacts/v1.pt --augment
.venv/bin/python scripts/train_relations.py
.venv/bin/python scripts/mine_negatives.py
.venv/bin/python scripts/train.py --name v3 --epochs 30 --resume artifacts/v2.pt --augment --hard-negatives
.venv/bin/python scripts/prepare_trajectories.py
.venv/bin/python scripts/prepare_context.py
.venv/bin/python scripts/train_context.py --name v4 --epochs 40
.venv/bin/python scripts/train_context.py --name v5 --epochs 28 --dual
.venv/bin/python scripts/rsi.py --models v4,v5 --limit 0 --promote
```

v3 用于历史对照，v4 的初始化来自 v2，而不是 v3。双头候选初始化来自 v4。固定随机种子降低实验波动；MPS 和依赖版本变化仍可能导致不同权重，不承诺逐位一致。训练/数据准备脚本是研究脚本；只有含 argparse 的命令提供 `--help`，不要把 `--help` 传给无参数的数据准备脚本。

验证选型后冻结模型、关系权重和配置，并保存 SHA256。只有最终验收才运行预留测试；不要将其放入循环调参：

```sh
# 这对应当前冻结模型的配置；新一轮应读取新 selected.json 的 edge_mode
.venv/bin/python scripts/evaluate.py --checkpoint artifacts/best.pt --edge-mode odds --split test --limit 0 --name final-test
```

本项目已测过当前 353 条预留集。后续如果针对这份结果改算法，应建立新的、从未用于迭代的盲测集，不能继续称同一份测试为独立最终验收。

数据准备在划分前固定整个归档的符号词表（保留至少 20 次的标签，另加 reject），因此这是固定词表任务，未评估发现未知数学符号的能力。测试标签用于离线生成真值/诊断和固定词表，没有进入优化器或模型/阈值选择。

当前 `captures/` 只负责存储人工采样，不会自动合并进训练。进入下一轮前需要人工核对标签，并补充每个数学符号对应的笔画分组和 MathML 结构；仅有整式 LaTeX 还不能直接供当前分组/关系监督脚本训练。优先按真实作者、设备、采集会话先划分，再做增强和负例挖掘。
