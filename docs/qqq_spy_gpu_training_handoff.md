# QQQ / SPY Agent：GPU 训练前交接

## 当前边界

CPU 数据与环境闭环已经完成。正式训练之前，系统具备：

- Longbridge 分批、缓存、前复权日线取数。
- 29 个只向后看的市场特征。
- 收盘观察、下一交易日开盘执行的数据契约。
- 原仓位隔夜、新仓位日内的组合账本。
- 2002-H1 至 2026-H1 的 49 个完整半年度 Episode。
- 按 `target_date` 切分的训练、验证和封存测试集。
- 只在训练集拟合的特征标准化器。
- 15 个离散目标仓位和 Gymnasium / SB3 接口。
- 固定策略基线和 CPU PPO 冒烟训练。

## 冻结切分

| 数据集 | 时间范围 | Episode |
|---|---|---:|
| 训练 | 2002-H1 至 2018-H2 | 34 |
| 验证 | 2019-H1 至 2022-H2 | 8 |
| 测试 | 2023-H1 至 2026-H1 | 7 |

测试集不得用于选择随机种子、超参数、奖励系数或特征。

## 正式训练入口

先在云服务器安装与 RTX 5090 对应的 CUDA 版 PyTorch，并确认：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

随后只做配置与环境检查：

```bash
python scripts/train_qqq_spy_ppo_multiseed.py --dry-run
```

确认后才启动冻结的多随机种子训练：

```bash
python scripts/train_qqq_spy_ppo_multiseed.py
```

配置文件：

```text
configs/qqq_spy_agent/ppo_gpu_v1.json
```

## 晋级规则

正式脚本只在验证集上选择候选模型，并要求：

- 平均终局分数超过最佳固定策略至少 0.002。
- 最差半年度最大回撤不超过 35%。
- 平均半年度换手率不超过 10。
- 所有指标均为有限值。
- 测试集保持封存。

通过后只获得 `validation_challenger` 身份，不能自动成为实盘
Champion。测试集评估、模拟盘、风险审查和人工批准仍是后续独立步骤。

## GPU 的现实作用

当前模型是小型 MLP，单次 PPO 在 CPU 上已经很快，RTX 5090 不一定比
CPU 更高效。GPU 的主要价值在于多随机种子、更大网络和后续序列模型，
不是证明策略有效。不得把 GPU 训练完成等同于获得稳定 Alpha。
