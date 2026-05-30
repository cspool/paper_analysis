## RLVR (Reinforcement Learning with Verifiable Rewards)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RLVR (Reinforcement Learning with Verifiable Rewards) 是一类使用可自动验证的奖励函数进行强化学习训练的范式，无需人工标注偏好数据。核心思想：利用任务本身定义的、可通过程序自动计算的规则化奖励（如数学题答案正确性、代码执行结果、时间定位 IoU），替代需要人工或 LLM 裁判的偏好奖励。RLVR 的关键特征：(1) 奖励函数是确定性的、可编程验证的，而非学习得到的 reward model；(2) 通常与 GRPO 等 policy-gradient 算法结合使用；(3) 奖励信号是序列级（sequence-level）而非 token 级的；(4) 可避免 reward hacking，因为奖励基于客观事实而非主观判断。代表工作：DeepSeek-R1（数学推理的 RLVR）、TimeLens（视频时间定位的 RLVR）、VideoSSR（自监督视频 RLVR）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 TimeLens 中，RLVR 用于视频时间定位（VTG）任务的训练：
```
# Thinking-free RLVR for VTG with GRPO
for each (video v, query q, ground_truth S*) in training_batch:
    # 1. 采样 G 个 responses（无需 thinking）
    for g in 1..G:  # G=8
        y^(g) = π_θ(v, q)  # 直接生成 "(t_start, t_end)"
        # y^(g) 不含 thinking 过程
    
    # 2. 计算可验证奖励（仅用 IoU）
    for g in 1..G:
        Ŝ^(g) = parse_time_segment(y^(g))
        r^(g) = IoU(Ŝ^(g), S*)  # 确定性、可编程验证
    
    # 3. GRPO 策略更新（组内 relative advantage）
    r_mean = mean(r^(1..G))
    for g in 1..G:
        A^(g) = r^(g) - r_mean
    L = -(1/G) Σ A^(g) log π_θ(y^(g) | v, q)
    θ = θ - lr * ∇L
```

RLVR vs SFT 对比（TimeLens Tab. 3）：
| 训练范式 | 训练时间 | Charades-TimeLens mIoU |
|---------|---------|----------------------|
| SFT (100K) | 2.4× | 48.6 |
| Thinking-based RLVR | 1.9× | 42.7 |
| Thinking-free RLVR | 1.0× | **48.3** |

关键结论：在 VTG 这类感知主导型任务上，RLVR 不需要 thinking 过程，且仅用 IoU 单一可验证奖励即可超越 SFT 和 thinking-based RLVR。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RLVR 通常通过 verl (Volcano Engine RL)、TRL 或自定义 GRPO 训练脚本实现。在 verl 框架中：(1) 定义 reward function 为 Python callable，接收模型输出和 ground truth，返回标量奖励；(2) 配置 GRPO trainer（G 个 samples per prompt, KL coefficient, learning rate）；(3) rollout engine 使用 SGLang 或 vLLM 进行高效采样；(4) 训练 loop 中 auto-regressive 采样 → 计算 reward → 计算 advantage → policy gradient update。TimeLens 中 RLVR 的关键实践：early stopping 当 reward plateau 时（~310 steps）、difficulty-aware data sampling（选择 difficulty d_i = 1 - IoU(Ŝ_i, S*_i) ≈ 0.95 的样本）、不使用 KL 正则化（β=0）。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
