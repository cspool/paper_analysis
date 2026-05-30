## TSPO (Temporal Sampling Policy Optimization / 时序采样策略优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TSPO (Temporal Sampling Policy Optimization) 是一个基于强化学习的可训练稀疏帧采样框架，用于 Video-MLLM 的长视频语言理解。核心思想：将关键帧选择（离散子集选择）和语言生成（自回归解码）建模为联合决策过程 π(o,V_s|q,V_c) = π_l(o|q,V_s,V_c) · π_ts(V_s|q,V_c)，通过 GRPO (Group Relative Policy Optimization) 端到端优化时序采样策略。TSPO 解决了 Video-MLLM 稀疏帧采样的两大核心挑战：(1) 无监督性——通用视频理解训练缺乏帧级标注，TSPO 通过语言级 rule-based reward（答案正确性 + 时序定位精度）隐式监督帧选择，无需帧级 ground-truth；(2) 不可微性——帧采样是离散子集选择问题，TSPO 使用 RL (GRPO) 的 policy gradient 绕过不可微性，通过期望奖励最大化优化采样策略。训练时 Video-MLLM 保持冻结，仅优化轻量 Temporal Agent (3.5M 参数, CLIP-Large 400M backbone 冻结)，推理时可迁移到不同 Video-MLLM (LLaVA-Video-7B/72B, Qwen2VL-7B, Qwen2.5VL-7B) 无需重新训练。AAAI 2026，开源：https://github.com/Hui-design/TSPO。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === TSPO 训练流程 ===
# 超参数: G 组采样/query, T_c 候选帧 (1FPS), T_s 采样帧 (训练16/推理64)
# τ 温度 (0.025→0.01 anneal), lr=5e-4, 1 epoch, 8×A800 GPUs

for each (video V, query q) in TSPO-10K:
    # Step 1: 候选帧均匀采样
    V_c = UniformSample(V, fps=1)  # T_c 帧候选

    # Step 2: CLIP 特征提取 (CLIP-Large, 冻结)
    F_f = CLIP_visual(V_c)         # [T_c, D]
    F_t = CLIP_text(q)             # [1, D]

    # Step 3: Event-aware Temporal Agent 前向传播
    F_e = LocalWindowAttention(F_f + SinusoidalPE, w=12)  # [T_c, D]
    S = CosSim(F_e, F_t) + CosSim(F_f, F_t)               # [T_c]

    # Step 4: Gumbel-Softmax 概率化采样 (G 组)
    for g in range(G):
        γ_g ~ Gumbel(0, 1)                    # [T_c], 探索噪声
        P_g = Softmax(S/τ + γ_g)              # [T_c], 概率分布
        P_top, I_g = TopK(P_g, T_s)           # 选 T_s 帧索引+概率

        # Step 5: 冻结 Video-MLLM 推理
        V_s_g = V_c[I_g]                       # 关键帧
        o_g = π_l(q, V_s_g, V_c)              # 自回归生成回答 (frozen)

        # Step 6: 计算 rule-based reward
        R_A_g = 1(predicted_option == ground_truth)  # 准确性
        R_T_g = count(I_g ∩ target_range) / T_s      # 定位精度 (仅NIAH数据)
        R_g = R_A_g + (1 or R_T_g)

    # Step 7: GRPO 优势计算
    μ_R = mean(R_1..R_G)
    σ_R = std(R_1..R_G)
    for g in range(G):
        A_g = (R_g - μ_R) / σ_R  # group-relative advantage

    # Step 8: 仅更新 Temporal Agent
    # J*_tspo = E[1/G Σ (π_ts/π_ts_old) · A_g]
    # π_l ratio = 1 (frozen MLLM)
    θ_ts ← θ_ts + lr · ∇J*_tspo(θ_ts)

    # Step 9: Temperature annealing (per epoch/step)
    τ ← anneal(τ, target=0.01)
```

TSPO vs Baseline 关键对比：
| 特性 | Uniform Sampling | Training-free Keyframe (LongVU/CoS) | TSPO |
|------|-----------------|--------------------------------------|------|
| 帧选择策略 | 均匀固定间隔 | 预训练 selector | 可训练 RL agent |
| 查询感知 | 否 | 部分 (CoS) / 否 (LongVU) | 是 (cross-modal similarity) |
| 端到端优化 | 无 (无参数) | 无 (training-free) | 是 (GRPO) |
| 帧级标注需求 | 无 | 无 | 无 (language-level reward) |
| 推理开销 | 0s | 28.4s (CoS) | 1.2s |
| 可迁移性 | N/A | N/A | 跨 Video-MLLM zero-shot |

TSPO 训练数据管道：
(1) Comprehensive Temporal Data：从 LLaVA-Video-178K 过滤>1min 的多选题，去除仅需4帧可答（太易）和64帧不可答（太难）的样本 → 保留需多关键帧的题目。
(2) Video Needle-in-a-Haystack Data：目标视频用 Qwen2.5-VL 生成事件描述→多选题，与无关视频段拼接/打乱合成 10∼60min 长视频 → 自动带目标段伪标签。
两者合并为 TSPO-10K (10,000 样本)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/Hui-design/TSPO。安装：conda env (Python 3.10) + requirements.txt + flash-attn==2.5.9.post1。训练：修改 train_deepspeed.sh 中的路径后运行。推理：demo/llava_video_tspo.py 或 demo/qwen25vl_tspo.py (需下载 LLaVA-Video-Qwen 或 Qwen2.5VL backbone + TSPO-0.4B checkpoint, ≥28GB GPU)。评估：通过 mp_tools 提取 CLIP 特征后运行 evaluation/ 下脚本。TSPO 的核心使用场景：(1) 在现有 Video-MLLM 前插入 Temporal Agent 作为可插拔帧选择器；(2) 训练的 agent 可 zero-shot 迁移到其他 Video-MLLM (如 LLaVA-Video→Qwen2.5VL 提升 5.2% LongVideoBench)；(3) 推理时可降低采样帧数 (64→32 帧) 仍超 uniform 64 帧 baseline，实现 token-效率-准确率的 trade-off。TSPO 的局限：需要 8×A800 训练；仅优化帧选择而非 token 级压缩或 LLM 本身；依赖 Video-MLLM 的 SFT 先验 (需 backbone 在均匀帧上已能回答)。

涉及论文标题：
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
