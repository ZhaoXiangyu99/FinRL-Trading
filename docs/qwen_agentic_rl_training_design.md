# Qwen Agentic-RL 训练设计

## 当前结论

Qwen3-8B 用作受约束的策略接口，经典投资组合环境继续负责资产、交易成本、
半年结算奖励和回测。首个阶段采用 4-bit QLoRA SFT，只训练模型读取工具结果并
输出合法动作的能力；它不是收益优化，也不能证明策略存在超额收益。

## 为什么先 SFT，后 Agentic-RL

如果模型还不能稳定遵守 JSON、15 个离散动作、目标权重和人工批准字段，直接
进行 GRPO/PPO 会把大量算力浪费在格式错误上。训练顺序固定为：

1. QLoRA SFT：学习观察接口、动作空间和约束。
2. validation 格式评估：JSON 合法率、动作/权重一致率、人工批准字段。
3. Agentic-RL：在模拟环境中生成完整半年轨迹，用冻结奖励函数优化 adapter。
4. 时间外 validation 选择；test 只在方案冻结后开启一次。
5. paper trading；达到成熟门槛后才讨论小额实盘。

SFT 教师是透明的当前状态规则，只使用已有 120 日趋势、60 日相对强弱和
252 日回撤。标签用途明确标记为 `format_bootstrap_not_alpha`，禁止把教师一致
率解释为交易能力。

## Agent 每日可见状态

模型收到两个只读工具结果：

- `observe_market_state`：29 个仅使用 `feature_date` 收盘及之前数据构建的特征，
  数值使用只在训练集拟合的标准化参数；
- `observe_portfolio_state`：当前 QQQ、SPY、现金权重。

模型同时看到 `feature_date` 和计划执行的下一交易日 `execution_date`。提示中
不得出现隔夜收益、日内收益、收盘到收盘收益、半年奖励或基准未来收益。

## 动作与安全边界

输出只能是 `qqq_spy_cash_action_v1` JSON，动作 ID 为 0–14，并且目标权重必须
与环境的 25% 网格完全一致。每个输出必须包含：

- 低或中置信度；
- 非自由文本的理由代码；
- `requires_human_approval=true`。

模型只提出目标仓位，不连接下单接口。独立风险引擎和人工批准不可被 Qwen、
SFT 或后续 RL 绕过。

## 数据边界

- train：目标日期不晚于 2018-12-31；
- validation：2019-01-01 至 2022-12-31；
- test：2023-01-01 至 2026-06-30，SFT 和模型选择阶段完全封存。

切分字段是 `target_date`，每个半年 episode 只能属于一个切分。标准化器只在
train 拟合。生成文件的 SHA256、行数、日期范围和 test 使用状态写入元数据。

## 4090 配置

- 基座：`Qwen/Qwen3-8B`，固定 revision；
- 量化：4-bit NF4、double quant、BF16 compute；
- LoRA：rank 16、alpha 32、dropout 0.05、`all-linear`；
- batch size 1、gradient accumulation 16、长度 1536；
- 只保存 adapter，最多保留 2 个 checkpoint；
- 正式首轮仅 1 epoch。

RTX 4090 的 1-step smoke 必须在正式训练前通过，并记录峰值显存。验证集只评估
格式和教师接口一致性；收益、回撤、换手和跑赢 QQQ/SPY 的能力属于后续
Agentic-RL 与封存评估阶段。

## 后续 Agentic-RL 奖励

后续奖励仍以完整半年 episode 为结算单位。核心项为投资组合相对 SPY 与 QQQ
的分层表现，并保留回撤、换手和成本约束。不能用不完整半年给终局奖励，也不能
每天把半年结果稠密化造成目标错配。

单卡 24GB 首版优先使用 LoRA adapter 的离线轨迹优化，并严格限制每次生成数量；
GRPO 的多样本生成和参考模型显存必须在独立 smoke 后再决定。任何从真实市场
新增的数据先进入可回放日志，不能让模型在实盘中直接在线更新并立即控制下单。

## 第一版半年奖励 GRPO 的精确定义

第一版不是让 8B 模型在 GPU 上逐日生成完整的 34 个半年轨迹。该做法在单张
24GB 4090 上成本过高，也难以调试。项目先离线建立反事实 reward table：

1. 用格式教师重放每个 train/validation 半年中已经发生的组合前缀。
2. 在每个决策状态分别尝试 15 个合法动作。
3. 每个候选动作从当日开始维持同一目标仓位至该半年末。
4. 用既有三资产环境结算组合收益、SPY/QQQ 基准、0.1% 交易成本、最大回撤和
   换手正则。
5. Qwen 对同一提示生成 4 个候选 JSON，GRPO 用对应的终局分数组内优化。

这相当于带固定 continuation policy 的离线 counterfactual policy
improvement。它是真实 outcome reward，而不是未来收益监督标签；但它仍不是
“模型自由滚动整段半年”的最终形态。其局限必须保留在实验结论中，不能简称为
已经解决了长期信用分配。

奖励表允许未来收益进入 evaluator，因为强化学习必须看到结果；但以下内容永远
不能进入 prompt：

- 当日之后的隔夜、日内或收盘到收盘收益；
- 候选动作的终局分数；
- 哪个动作是事后最优动作；
- validation 或 test 的结果摘要。

GRPO 仅使用 train reward table。validation 通过固定随机种子的 64 条生成样本
比较 SFT 和 GRPO：

- 严格 JSON 合法率；
- 平均反事实半年奖励；
- 相对 SFT 教师的差值；
- 相对 15 动作 oracle 的遗憾值。

只有格式合法且 validation 奖励高于 SFT adapter，GRPO adapter 才能成为下一
阶段候选。该比较不是完整组合回测，不能触发 test、paper trading 或实盘晋级。
