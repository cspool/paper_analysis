## GRPO (Group Relative Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO (Group Relative Policy Optimization) 是一种在线强化学习算法，由 DeepSeek (Shao et al., 2024, DeepSeekMath) 提出，后用于 DeepSeek-R1 训练推理模型。核心创新：对每个 prompt 生成一组（group of）候选输出，通过比较组内各输出的 reward 相对于组均值和标准差来计算 advantage，从而消除对独立 critic/value network 的需求（PPO 需要）。GRPO 是一种 KL-regularized policy optimization 方法：在最大化期望 reward 的同时，通过 KL 散度约束策略保持在 reference model π_ref 附近。在 EVA 中，GRPO 用于三阶段训练的第三阶段（最后阶段），通过 Data-Enhanced GRPO 变体在多轮 video agent 交互中进行在线策略优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GRPO Training (per batch)
# 输入: 一批 prompts {x_i}, reference model π_ref

# Step 1: 对每个 prompt 采样 G 个候选输出
for x_i in batch:
    {y_i^(1), ..., y_i^(G)} ~ π_θ_old(·|x_i)  # G=8 in EVA

# Step 2: 计算每个候选输出的 reward
for each y_i^(g):
    R_i^(g) = w_acc * r_acc(y_i^(g)) + w_fmt * r_fmt(y_i^(g))
    # EVA: r_acc ∈ {r_csv (MCQ) | r_rouge (open-ended)}
    # r_fmt = 0.05 if tool_call ∧ answer_incorrect else 0

# Step 3: Group-relative advantage
μ_i = (1/G) * Σ_g R_i^(g)
σ_i = std({R_i^(1), ..., R_i^(G)})
A_i^(g) = (R_i^(g) - μ_i) / σ_i  # normalized advantage

# Step 4: Policy update with KL constraint
L_GRPO = -E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
         + λ * KL(π_θ || π_ref)

# ratio = π_θ(y|x) / π_θ_old(y|x)
```

EVA 的 GRPO 设置：
- batch size=64, rollouts per sample=8, lr=1e-6, 1 epoch
- 数据: 90% open-ended QA + 10% MCQ
- reward: r_acc (ROUGE/CSV) + r_fmt (format reward=0.05)
- 训练硬件: 32 H100 GPU

GRPO vs PPO 对比：
| 特性 | PPO | GRPO |
|------|-----|------|
| Value Model | 需要 (critic network) | 不需要 |
| Advantage | 学习得到 (critic) | 组内相对 (mean/std) |
| 模型数量 | 4 (policy, value, reward, ref) | 3 (无 value model) |
| KL 惩罚 | 加入 reward | 直接加入 loss |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：TRL (`GRPOTrainer`)、veRL (by volcengine)、Open-R1 (HuggingFace)。GRPO 已被 DeepSeek-R1、DeepSeek-V3 等模型使用。核心优势：(1) 减少 GPU 内存（无需 critic）；(2) 组内 whitening 提供自适应加权——罕见成功获得更多信用，常见成功对错误惩罚更重；(3) 适合 verifiable rewards（数学答案匹配、代码执行、格式约束）。局限性：当组内所有响应都错误时无学习信号（all-negative-sample problem），可能需要 step-wise judge (SGPO) 或 Dr.GRPO 等改进。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

SAGE 中的 GRPO 使用方式（any-horizon agent RL post-training）：
SAGE 使用 GRPO 训练 orchestrator VLM (SAGE-MM) 的 any-horizon reasoning 能力。配置：batch size=16, 8 rollouts/sample, lr=1e-6 (cosine decay), KL coeff=0.005, 训练480 steps。关键创新：
(a) **Multi-reward design**：R_i = Σ s_j + a_N，其中 step-level rewards (s_format +0.05/-0.10, s_reasonable-tool +0.10/-0.10, s_args-repeat -0.05×√rep, s_args-valid -0.1/0) 和 accuracy reward a_N (LLM-Judge GPT-4o binary: +1.25 正确+visual tools, +1.0 正确无tools, -0.5 错误, -2.0 JSON无效) 被 uniform 赋给 trajectory 中所有 actions。
(b) **Curricular horizon scheduling**：前100步 N_max=6（短 trajectory 稳定训练），之后 N_max=11（允许更复杂多轮推理）。
(c) **Visual tool bonus**：正确且使用 extract-video-parts 或 ground-event 时额外 +0.25 reward，鼓励利用视觉信息。
(d) **SFT pre-requisite**：RL 直接应用于 base model 失败（collapse to single-turn），需要 SFT cold-start 建立 any-horizon 行为基础。训练硬件：16×H100 GPUs。

Mirage 中的 GRPO 使用方式：
Mirage 在两阶段 SFT 之后使用 GRPO (via VERL) 进行第三阶段 RL fine-tuning，进一步增强多模态推理能力。配置：rollout num=5, mini batch=8, lr=1e-6, batch size=32, grad accum=4, epochs=15。奖励函数 = σ_c·r_acc + σ_f·r_format（σ_c=0.9, σ_f=0.1），r_acc = 1(binary correctness), r_fmt = 0.1（若输出含 <think>...</think> 和 \boxed{}）。KL penalty λ_kl=0.01 仅应用于 text tokens，latent visual tokens 排除 KL 正则（让 latent tokens 自由探索）。Entropy regularization disabled (λ_en=0.0)。关键创新点：Mirage 的 GRPO 在 latent visual token 存在下进行——这些连续 embedding 接收梯度但不参与 KL penalty，允许 RL 阶段同时优化 text trajectory 和 interleaved latent visual representations 的生成模式。训练硬件：单 NVIDIA H100 GPU。

TwigVLM++ 中的 GRPO 使用方式：
TwigVLM++ 在 Stage-2 RL 训练中使用 GRPO 直接优化 P-Head 的参数，将 visual token pruning 建模为 sequential decision process。与标准 GRPO 的区别：(a) Action space 是选择 R 个 visual token 位置（无放回采样，而非生成文本序列）；(b) Reward 使用 reference-based reward（pruned 输入下 target model 生成参考答案的 mean log-probability，Eq.13），而非 verifiable reward（数学结果/代码执行）；(c) 由于严格 on-policy（一次采样→一次梯度更新），importance ratio = 1，clipping 机制不激活，loss 简化为 L_stage2 = (1/G)·Σ Â_i·log π_θ(a_i)（Eq.15）；(d) 配置 G=32, lr=2e-5, 50K training samples, 候选集 R̄={64..192}；(e) 配合 Dynamic Pruning-Ratio Schedule 使单一模型支持多种 pruning ratio。这展示了 GRPO 在非文本生成场景（离散 token 选择优化）中的适用性。

HORNet 中的 GRPO 使用方式（概念创新：首次将 GRPO 从优化 VLM 输出重定向到优化 VLM 输入）：
HORNet 使用 GRPO 训练一个轻量级 frame selection policy（MLP, <1M params, frozen VLM），选择哪些视频帧送入 frozen VLM 进行 QA。与标准 GRPO 的区别：(a) Action space 是选择 T 帧的子集（binary mask b ∈ {0,1}^T，factorized Bernoulli distribution），而非文本生成；(b) Reward 使用 task-grounded QA reward = 0.1·F1_token + 0.9·EditSim（lemmatized），直接测量帧选择质量；(c) Candidate generation 策略：K=8 = 7 top-k sweep（deterministic） + 1 Bernoulli（stochastic exploration），平衡利用与探索；(d) Log-probability 计算为独立 Bernoulli 乘积：log π(b|F) = Σ[b_t·log p_t + (1-b_t)·log(1-p_t)]；(e) 两阶段训练：Stage 1（短视频+F1-Lev reward）学"有用帧识别"，Stage 2（长视频+MCQ accuracy reward）学"因果/时序推理帧选择"；(f) 关键发现：GRPO 的 OOD generalization 优于 PPO 和 SFT——仅训练 MSVD-QA，MSRVTT-QA OOD 上 GRPO 保留 94% baseline（vs PPO 92%, SFT 90%），证明 group-relative advantage 学到的选择策略更可迁移。这代表了 GRPO 的一个新的应用方向：将 GRPO 从"优化模型生成什么"（output optimization）转向"优化模型看到什么"（input optimization），且 policy 可跨 VLM answerer transfer 无需 retraining。

SPIKE-RL 中的 GRPO 使用方式（Belief Optimization via GRPO）：
SPIKE-RL 将 GRPO 创新性地应用于优化中间信念假设（belief hypotheses）生成——而非直接优化最终输出。配置：M=3 trajectories/视频, N=3 hypotheses/时间步, batch_size=4, lr=1e-6, GRPO β=0.1, epochs=1, 4×H100 + DeepSpeed ZeRO-3 offload, PEFT (LoRA) 微调 Qwen2.5-VL-7B-Instruct。关键创新：
(a) **Sequence-level belief optimization**：GRPO 的 action 不是模型输出 token，而是每个时间步 t 的完整信念假设 B_t = {b_{t,1}, b_{t,2}, b_{t,3}}（textual hypotheses），整条 trajectory 的信念序列作为 sequence-level action。
(b) **Caption-based reward propagation**：reward 来自最终 caption 与 ground truth 的 LLM-Match 评分（0-1 连续值），而非中间信念的直接监督。通过 GRPO 的 group-relative advantage 将 caption 质量的 credit 反向分配给各时间步的信念假设：A^{(r)} = (R^{(r)} - μ_R) / σ_R。
(c) **Policy objective**: L = -1/M Σ_r A^{(r)} Σ_t Σ_k log p_θ(b_{t,k}^{(r)} | H_t, W_t)，增加高 advantage 轨迹中信念的似然，抑制低 advantage 轨迹中的信念。
(d) **训练数据设计**：2000 视频混合 30% surprising（Oops! 训练集）+ 70% unsurprising（ActivityNet Captions），让策略同时学习"信念稳定期的预测"和"信念转折点的适应"。
(e) **效果**：SPIKE-RL 生成的假设多样性从 33.5% 提升至 40.3%（逆余弦相似度衡量），与人类判断的 Spearman 相关性从 0.84 提升至 0.87。这代表了 GRPO 的又一个新应用方向：将 GRPO 从优化模型输出转向优化模型的内部推理过程（intermediate belief trajectory），通过最终结果的 reward 隐式训练中间推理步骤。

TSPO 中的 GRPO 使用方式（Joint Keyframe-Language Optimization via GRPO）：
TSPO 将 GRPO 创新性地应用于联合优化关键帧选择策略和语言生成——将离散帧采样和自回归文本生成建模为统一决策过程。配置：G 组采样/query（论文未明确说明 G 值），batch size=1，lr=5×10⁻⁴，1 epoch，8×A800 GPUs，DeepSpeed 分布式训练。关键创新：
(a) **联合策略建模**：将 temporal sampling policy π_ts 和 language policy π_l 建模为联合策略 π(o,V_s|q,V_c) = π_l(o|q,V_s,V_c) · π_ts(V_s|q,V_c)。Video-MLLM (π_l) 冻结（ratio=1），仅通过 GRPO 优化 Temporal Agent (π_ts)。
(b) **Action space 为帧索引选择**：Gumbel-Softmax + TopK 产生 T_s 个关键帧索引 I={i_1,...,i_T_s} 及对应概率 P={p_1,...,p_T_s}，log-probability = Σ log P_i。与 HORNet 的 Bernoulli mask 不同，TSPO 使用 Gumbel-based categorical 采样。
(c) **Rule-based dual reward**：R_A = 1(预测选项==正确选项)（多选题准确性），R_T = T_t/T_a（采样帧中目标视频帧占比，仅 Needle-in-a-Haystack 数据使用）。总 reward = R_A + 1（Comprehensive Temporal）或 R_A + R_T（Needle-in-a-Haystack）。
(d) **Temperature annealing for exploration**：τ 从 0.025 逐步退火至 0.01，早期高温（Gumbel noise 大）鼓励探索不同帧组合，后期低温收敛至确定性关键段。
(e) **TSPO objective**：J*_tspo(θ) = E[1/G Σ (π_ts/π_ts_old) · A_i]，其中 A_i = (R_i - mean(R)) / std(R)，无 KL penalty（因仅训练轻量 agent 且冻结 MLLM）。
(f) **与标准 GRPO 的核心差异**：标准 GRPO 优化 LLM output tokens（文本生成），TSPO 优化 frame indices（离散子集选择）——输入优化而非输出优化。这代表了 GRPO 的"input optimization"范式：通过最终回答的 reward 隐式训练输入选择策略。

VideoAuto-R1 中的 GRPO 使用方式（Dual-Answer Reward GRPO）：
VideoAuto-R1 将标准 GRPO 的单一答案奖励扩展为双答案奖励，用于训练 answer-think-answer 模板。配置：base model Qwen2.5-VL-7B-Instruct 或 Qwen3-VL-8B-Instruct，rollout size G=16，temperature=1.0（exploration），batch size=256，lr=1e-6（constant，无 warmup），KL penalty β=0.01，weight decay=0.01，max grad norm=1.0，训练 1 epoch。训练硬件：32 H100 GPU，约35小时。关键创新：
(a) **Dual-Answer Reward Design**：$R = 0.9 R_{task}^{(1)}(a_1) + 1.1 R_{task}^{(2)}(a_2) + 1.0 R_{fmt} + 0.3 R_{fallback}$，同时监督初始和审查答案。$w_2 > w_1$（0.9:1.1）确保模型优先最终答案正确性。消融显示 uniform 权重（1:1）使"correct→wrong"和"wrong→correct"获得相同总奖励，导致训练信号模糊（Table 9/12）。
(b) **Multi-Task Reward Functions**：QA 使用 exact match / math-verify（{0,1}），Temporal Grounding 使用 max-tIoU（[0,1]），Grounding QA 使用 R_QA + R_TG 组合（[0,2]）。Format reward 通过 strict regex 检查强制 answer-think-answer 模板合规。
(c) **Fallback Reward α=0.3**：额外奖励模型在无法立刻回答时输出 fallback 字符串"Let's analyze..."而非猜测（仅当 $a_2$ 正确时）。将 fallback→correct 的总奖励从 1.1 提升至 1.4，区分"诚实 defer"和"错误猜测"（Table 12）。
(d) **训练曲线特征**：$R_{task}^{(2)}$ 始终高于 $R_{task}^{(1)}$（Figure 6），验证推理阶段确实改进答案质量。Qwen3-VL-8B 的训练 reward 始终高于 Qwen2.5-VL-7B，说明 dual-answer supervision 随模型容量 scaling。
(e) **无需 Cold-Start SFT**：直接 RL 训练即可通过 system prompt（Table 2）实现 ~100% format compliance。SFT on Video-R1-CoT data 反导致性能退化（VideoMME 66.0→60.1, Table 17），低质量 CoT 监督会扭曲强基模型的行为。
(f) **Data Filtering for GRPO**：丢弃所有 8 个 rollouts 全对或全错的 QA 样本（对 GRPO 无学习信号），仅保留部分正确/错误的样本。从 137K 过滤至 83K（Table 11），验证 filtered 配置优于 unfiltered（VideoMMMU 56.4 vs 55.4）。
(g) **与训练时 switching-policy 的 GRPO 对比**：AdaptThink 风格（per-sample think/no-think 标签 + GRPO 训练二模式）在视频域发生 mode collapse（MVBench auto mode 70.5 vs no-think 71.1, Table 7）。VideoAuto-R1 将 think/no-think 决策推迟至推理时（confidence score），训练时仅训练统一的 answer-think-answer 行为，避免了 GRPO 中的 mode selection 不稳定。
