## Input-side Adaptation（输入侧自适应）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Input-side Adaptation（输入侧自适应）是 ResAdapt 提出的一种视频 MLLM 效率范式，将视觉 token 预算的分配时机从"编码后"（model-side compression）或"推理迭代中"（output-side agentic reasoning）前移到"编码前"。核心思想：传统方法接受编码器的全分辨率输入作为固定成本，在编码后才进行 token 剪枝或合并（model-side），或通过多轮检索缩放恢复覆盖（output-side）。Input-side Adaptation 则通过一个轻量级 Allocator 在编码前预测每帧的分辨率分配，让 backbone 只处理被缩放的像素——保存的像素预算可 reinvest 为更多帧的时间覆盖。该范式完全兼容 FlashAttention、vLLM 和 SGLang，无需定制 kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Input-side Adaptation 的三阶段对比伪代码：
```
# ===== Model-side (编码后压缩) =====
V_raw = load_video(n_frames=T, resolution=HxW)  # [T, 3, H, W]
tokens = vision_encoder(V_raw)                    # [T*H*W/P², D]  全量计算
tokens_pruned = token_prune(tokens)               # [K, D]  K << T*H*W/P²
answer = llm_backbone(tokens_pruned, query)       # 证据已丢失无法恢复

# ===== Output-side (迭代检索) =====
V_coarse = load_video(n_frames=T/4, resolution=HxW)
tokens_coarse = vision_encoder(V_coarse)
hint = llm_backbone(tokens_coarse, query)          # 第1次 backbone 调用
V_fine = crop_video(hint.spans)                    # 检索细粒度帧
tokens_fine = vision_encoder(V_fine)
answer = llm_backbone(concat(tokens_coarse, tokens_fine), query)  # 第2次调用

# ===== Input-side Adaptation (ResAdapt) =====
V_raw = load_video(n_frames=T, resolution=HxW)    # [T, 3, H, W]
f_coarse = lightweight_encoder(V_raw)              # SmolVLM, frozen, [T, D_coarse]
scales = allocator(f_coarse, query)                # st ∈ [0.2, 1.8] per frame
V_resized = [resize(V_raw[t], scales[t]) for t]    # 编码前缩放
tokens = vision_encoder(V_resized)                 # 仅处理缩放后像素 → 节省 token
answer = llm_backbone(tokens, query)               # 单次 backbone 调用
```

关键计算：Token Retention Ratio ρ = Σ s_t²/T。在 ρ≈0.11 时，attention FLOPs 降低为 ρ²≈0.012（约 83×）。Allocator 基于 SmolVLM (Lpred=4, Dpred=1024)，占 <3% 总 FLOPs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Allocator 使用 Beta 分布参数化连续动作空间：每帧 t，Alpha 头输出 (α_t, β_t) 通过 softplus 确保正值，at ~ Beta(α_t, β_t)，st = smin + at · (smax − smin) where smin=0.2, smax=1.8。分配策略通过 GRPO + CAPO 训练，backbone 保持 frozen（ResAdapt）或联合微调（ResAdapt-RL）。训练框架 VeRL + DeepSpeed ZeRO + vLLM。推理时取 at 的期望值代替采样。代码：https://github.com/Xnhyacinth/ResAdapt。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning

## CAPO (Cost-Aware Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CAPO（Cost-Aware Policy Optimization，成本感知策略优化）是 ResAdapt 提出的 RL 奖励塑形方法，解决 naive accuracy-cost Lagrangian penalty 导致 RL 策略向最小预算崩溃的问题。核心矛盾：若直接使用 R = Q(x,y) − λ·C(s)，任何成本降低都获得等量奖励（无论是否破坏答案），策略会无条件坍缩至 smin。CAPO 通过三项机制稳定训练：(1) Dynamic Cost Pivot τ_dyn = κ_mix·c̄_group + (1−κ_mix)·τ_fix，在组内均值和全局目标间插值；(2) Asymmetric Reward Shaping — 正确且低成本 → 中等奖励 λ_+，错误且高成本 → 强惩罚 λ_−（λ_− > λ_+ > 0）；(3) 对正确 rollout 施加正下限 ε_+，确保正确低成本 rollout 始终获得正向学习信号。消融实验证实：移除 CAPO（仅用 direct cost penalty）→ 策略坍缩至 smin；移除 cost 完全（仅 accuracy）→ 策略饱和至 smax。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CAPO 的核心计算流程（per prompt x with M=16 allocations, N=1 rollout each）：
```
# Step 1: 计算 proxy cost（用线性 proxy 避免二次方差）
c_m = (s̄_m - s_min) / (s_max - s_min)   # s̄_m = mean_t(s_m[t])

# Step 2: 动态 cost pivot（同时利用组内对比和全局锚点）
c̄_group = mean(c_1, ..., c_M)
τ_dyn = κ_mix * c̄_group + (1 - κ_mix) * τ_fix   # κ_mix=0.5, τ_fix=0.1

# Step 3: 非对称塑形（核心防止崩溃的机制）
for each rollout (m, n):
    if u_{m,n} == 1:   # 正确
        S_{m,n} = λ_+ * σ((τ_dyn - c_m) / τ_s)    # sigmoid 温度 τ_s=0.1
    else:               # 错误
        S_{m,n} = -λ_- * σ((c_m - τ_dyn) / τ_s)   # λ_- >> λ_+

# Step 4: 组合 base advantage
A_base_{m,n} = GRPO_normalize(R_task_{m,n})
Ã_{m,n} = A_base_{m,n} + λ_capo * S_{m,n} - γ * c_m   # γ: 残差全局成本压力

# Step 5: 正确 rollout 正下限保护
A_{m,n} = max(Ã_{m,n}, ε_+) if u_{m,n} == 1 else Ã_{m,n}
```
关键超参数：κ_mix 控制动态枢轴中组内 vs 全局的比例；λ_+ / λ_- 比值决定非对称程度（λ_- > λ_+ 是关键，否则策略向低成本崩溃）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAPO 作为 GRPO advantage 计算的替换层实现，在标准 GRPO 训练循环中替换 group-normalized advantage 的计算步骤。Correctness indicator u_{m,n} 定义：exact-match QA 直接使用二元结果；连续指标（ROUGE-L, temporal IoU）使用 0.35 阈值。与 GRPO 的集成：CAPO 计算 per-allocation aggregated advantage A_m_CAPO = mean_n(A_{m,n})，用于 Allocator 的 PPO update；同时各 rollout advantage A_{m,n} 用于 backbone 的 token-level PPO update（可选）。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning

## Temporal Similarity Regularization（Lsim，时序相似性正则化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Similarity Regularization（Lsim）是 ResAdapt 提出的正则化损失，用于打破 Allocator 对视觉相似相邻帧分配相同 scale 的对称性。CAPO 确定 accuracy-cost 的全局运行点，但无法阻止优化器对相似相邻帧赋予相同预算——这导致实际行为接近 FixedScale（均匀缩放），浪费了自适应分配的能力。Lsim 通过余弦相似度门控权重，仅在相邻帧超过相似度阈值时，惩罚它们的联合高预算分配。消融实验（Figure 7/13/14）显示：去除 Lsim 后 scale trace 坍缩为常数分布；恢复 Lsim 后 scale histogram 变为双峰、per-video range 扩大、Gini 系数上升——策略从退化均匀分配转型为真正的选择性分配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Lsim 的具体计算公式：
```
# 输入: per-frame coarse features f_t, predicted scales s_t
# 超参数: τ_sim (cosine threshold), γ_sim (temperature), η_sim (margin)

L_sim = 0
for t = 1 to T-1:
    # 余弦相似度门控权重（仅在相似帧上激活）
    w_t = σ((cos(f_t, f_{t+1}) - τ_sim) / γ_sim)
    # 对数尺度联合惩罚（s_t * s_{t+1} > e^{-η_sim} 时才生效）
    penalty = max(0, log(s_t) + log(s_{t+1}) + η_sim)
    L_sim += w_t * penalty

L_sim /= (T - 1)
```

门控机制的关键：当 cos(f_t, f_{t+1}) << τ_sim 时 w_t→0（帧不相似，无需惩罚），当相似度超过阈值时 w_t 渐进上升。对数尺度惩罚确保 penalty 仅在 s_t·s_{t+1} > exp(−η_sim) 时激活——即两帧的联合分配超过下界时才受惩罚，而非无条件惩罚所有分配。Total Allocator loss: L_alloc = L_θ + λ_sim·L_sim + λ_con·L_con。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Lsim 的实现：(1) 在 Allocator 的 Transformer decoder 中缓存粗粒度特征 f_t；(2) 在训练循环中计算成对余弦相似度和 scale 的联合惩罚；(3) 与 Concentration Loss（Lcon = max(0, α_t+β_t−κ_max)/T）配合使用——Lsim 打破帧间对称性，Lcon 防止 Beta 分布坍缩为确定性。代码：https://github.com/Xnhyacinth/ResAdapt。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning

涉及论文标题：
- Atlas__Multi-Scale_Attention_Improves_Long_Context_Image_Modeling
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## iMCoTT (interleaved Multimodal Chain-of-Tool-Thought)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
iMCoTT（交织多模态工具思维链）是 LongVT 论文提出的长视频推理范式。它将传统的纯文本 CoT (Chain-of-Thought) 扩展为 "推理步骤" 与 "视觉工具调用" 交织进行的循环过程。具体流程：(1) 模型首先对长视频进行全局 skim（均匀采样少量帧），形成关于证据所在时间段的粗粒度假设；(2) 模型以结构化格式 <tool_call>{"name":"crop_video","arguments":{"start_time":t_s,"end_time":t_e}}</tool_call> 调用原生视频裁剪工具，请求重采样指定时间窗口内的细粒度帧；(3) 工具返回裁剪后的视频帧（以 vision tokens 形式），模型基于新视觉证据重新 think，验证或修正假设；(4) 模型决定直接回答或进入下一轮工具调用（最多 5 轮）。iMCoTT 的核心创新在于将 LMM 的潜在 temporal grounding 能力通过 tool-integrated fine-tuning 激活，无需外部检索模型或专家模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
iMCoTT 的单次推理流程伪代码：
```
def iMCoTT(video, question, max_turns=5):
    # 全局 skimming: 均匀采样少量帧
    global_frames = uniform_sample(video, n=64)
    vision_tokens = visual_encoder(global_frames)
    context = [vision_tokens, question_text]
    
    for turn in range(max_turns):
        # Step 1: 模型思考，可能提出时间窗口
        output = llm.generate(context, stop=["</think>"])
        think_text = parse_think(output)
        
        # Step 2: 如果模型认为需要进一步检查
        if contains_tool_call(output):
            tool_args = parse_tool_call(output)
            # 调用 crop_video 工具
            cropped_frames = crop_video(video, 
                                        tool_args["start_time"],
                                        tool_args["end_time"])
            cropped_tokens = visual_encoder(cropped_frames)
            context.append(tool_response(cropped_tokens))
            continue  # 进入下一轮 think-verify
        
        # Step 3: 模型有足够证据，给出答案
        answer = parse_answer(output)
        return answer
    
    return answer
```
具体计算流程：输入 prompt = [system_prompt_with_tool_def] + [global vision tokens] + [question] → LLM decoder 逐 token 生成（最大 16384 tokens）→ 解析 <think>...</think>、<tool_call>...</tool_call>、<answer>...</answer> 标记 → 若解析到 tool_call，外部 executor 执行 crop_video → 重采样帧再编码为 vision tokens → 追加 <tool_response> 到上下文 → 继续生成下一轮。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
iMCoTT 通过三阶段训练实现：(1) Cold-Start SFT用 tool-augmented traces（由 Gemini 和 Qwen 模型蒸馏生成，含 <think>/<tool_call>/<tool_response>/<answer> 结构） 教模型工具调用范式；(2) Agentic RL (GRPO) 用联合奖励（answer accuracy + format compliance + temporal IoU）优化模型何时调用工具、裁剪多长时间、如何整合证据；(3) Agentic RFT 用 RL 阶段的高质量 rollout traces (answer 正确且 temporal IoU ≥ 0.3) 进一步微调稳定行为。推理时部署为 vLLM + MCP server 架构，通过特殊分隔标记解析多轮交互。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

## GRPO (Group Relative Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO (Group Relative Policy Optimization) 是 DeepSeek (Shao et al., 2024) 提出的一种强化学习算法，首先用于数学推理 (DeepSeekMath)，后被 DeepSeek-R1 推广至通用推理训练。GRPO 的核心思想：对每个 prompt，从当前策略采样 K 个 rollouts 组成一个 group，以组内平均奖励为 baseline 计算优势函数，而非像 PPO 那样训练一个独立的 critic/value network。优势函数 A^(k) = R^(k) - mean(R^(1..K))。优化目标是 token-level 长度归一化的 KL 正则化政策梯度：max E[1/K Σ 1/T_k Σ A^(k) log π(y_t|x,y_<t)] - β KL(π||π_ref)。GRPO 相比 PPO 的优势：(1) 无需训练和存储 value network，节省约一半 GPU 内存；(2) 组内相对比较提供的信号比绝对奖励 baseline 更稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GRPO 训练迭代伪代码（LongVT 中的 agentic RL 设置）：
```
# 超参数: K=16 rollouts, β KL coefficient, lr=1e-6
for step in range(160):
    for prompt in batch(global_batch_size=16):
        # 1. 从旧策略采样 K 个 rollouts (multi-turn tool calling)
        rollouts = []
        for k in range(K):  # K=16
            y_k = π_θ_old.generate(prompt, 
                                    max_new_tokens=16384,
                                    temperature=1.0)
            # y_k 包含多轮 <think>/<tool_call>/<answer>
            rollouts.append(y_k)
        
        # 2. 计算每个 rollout 的奖励（联合奖励函数）
        for k in range(K):
            R_acc[k] = LLM_Judge(answer_k, answer_gt)  # {0, 0.5, 1}
            R_fmt[k] = 1 if format_matches_schema(y_k) else 0
            R_time[k] = IoU(predicted_span, ground_truth_span)
            R[k] = R_acc[k] + R_fmt[k] + R_time[k]
        
        # 3. 计算组 baseline 和优势函数
        baseline = mean(R[1..K])
        for k in range(K):
            A[k] = R[k] - baseline  # 优势函数
        
        # 4. 计算 policy gradient loss + KL penalty
        for k in range(K):
            for t in range(len(y_k)):
                log_prob = log π_θ(y_t[k] | prompt, y_<t[k])
                kl = KL(π_θ(·|prompt, y_<t[k]) || π_ref(·|prompt, y_<t[k]))
                loss += A[k] * log_prob / len(y_k) - β * kl / K
        
    # 5. 梯度更新
    optimizer.step(loss)
```
该流程在 verl 框架上实现，扩展支持 multi-turn multimodal tool-augmented rollouts，通过 SGLang 作为 rollout 引擎。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GRPO 通常通过 verl (Volcano Engine RL) 或 TRL (Transformer Reinforcement Learning) 等 RL 框架实现。在 LongVT 中，GRPO 在 verl 框架上实现，扩展支持：(1) 多轮工具调用 rollouts（模型生成 → 外部工具执行 → 结果返回 → 继续生成）；(2) 多模态输入（vision tokens + text）；(3) 自定义联合奖励函数（accuracy + format + temporal IoU）。推理引擎由 SGLang 提供，支持 continuous batching 和 prefix caching。2025 年的研究发现 GRPO 可与 DPO 建立理论联系（"Your GRPO Is Secretly DPO"），且 2-GRPO（K=2）在多数任务上可匹配 K=16 的性能，大幅减少训练开销。

在 ResAdapt 中，GRPO 被创造性地应用于优化输入侧视觉分配策略：M=16 个 allocation 轨迹 × N=1 个 rollout 组成 group，CAPO（Cost-Aware Policy Optimization）替换标准 GRPO advantage 计算——将 group-normalized task reward 与 cost-aware shaping term 结合为 A_{m,n}，然后聚合为 per-allocation advantage A_m_CAPO 用于 Allocator PPO update。Allocator 和 Backbone 使用去耦的 PPO 目标交替更新。

在 TimeLens 中，GRPO 被应用于 MLLM 的 **thinking-free RLVR** 训练范式。与传统 GRPO 的关键差异：(1) 模型直接输出时间片段 `(t_start, t_end)` 而无显式 thinking 过程 —— thinking 过程在该感知主导型 VTG 任务中无益；(2) 奖励函数简化为仅包含 IoU accuracy：`r(y) = IoU(Ŝ, S*)`，无需格式奖励或 LLM Judge；(3) 每 prompt 采样 G=8 个 rollouts；(4) KL 系数 β=0，不使用 KL 正则化；(5) Early stopping：当 temporal IoU reward 和 group reward std 双 plateau 时停止训练（约 310 steps / ~2.5K samples）；(6) 配合 difficulty-aware Gaussian sampling 选择训练样本（μ=0.05, σ=0.2，即 prefer 难度 d_i ≈ 0.95 的样本）。训练在 8×H20 GPU 上完成，1.0× 训练时间约 4h10m，比 thinking-based variant 快约 2×。Qwen3-VL 上直接 RLVR 无效（因为其已含 VTG RL 数据），需先做小型 SFT "revert" 操作。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding

## Joint Answer-Temporal Grounding Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Joint Answer-Temporal Grounding Reward 是 LongVT 在 Agentic RL 阶段提出的三方联合奖励函数，用于同时优化模型的答案正确性、输出格式合规性和时间定位精度。奖励分解为三个独立可加组件：(1) Answer Accuracy R_acc：使用 LLM-as-a-Judge 将模型回答与 ground-truth 比较，分为 Fully Consistent (1.0)、Partially Consistent (0.5)、Inconsistent (0.0) 三档；(2) Format Compliance R_fmt：检查输出是否符合 <think>...</think><tool_call>...</tool_call><answer>...</answer> 的结构化模板，符合则 1、不符合则 0；(3) Temporal Overlap R_time：计算预测时间窗口 [t_s, t_e] 与 ground-truth 窗口 [t_s', t_e'] 的 Intersection over Union (IoU)，R_time = IoU ∈ [0, 1]。最终奖励 R = R_acc + R_fmt + R_time ∈ [0, 3]。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
奖励计算流程：
```
def compute_reward(rollout, ground_truth):
    # rollout = <think> time window hypothesis </think>
    #           <tool_call> crop_video </tool_call>
    #           <tool_response> frames </tool_response>
    #           <think> verify evidence </think>
    #           <answer> final answer </answer>
    
    # 1. 答案准确性 (LLM-as-a-Judge)
    answer_pred = extract_answer_tag(rollout)
    verdict = judge_llm.predict(answer_pred, ground_truth.answer)
    # verdict ∈ {"Fully Consistent", "Partially Consistent", "Inconsistent"}
    R_acc = {"Fully Consistent": 1.0, 
             "Partially Consistent": 0.5, 
             "Inconsistent": 0.0}[verdict]
    
    # 2. 格式合规性
    R_fmt = 1.0 if matches_schema(rollout) else 0.0
    
    # 3. 时间重叠 (Temporal IoU)
    t_s, t_e = extract_time_window(rollout)  # 从 tool_call 参数提取
    t_s_gt, t_e_gt = ground_truth.time_window
    intersection = max(0, min(t_e, t_e_gt) - max(t_s, t_s_gt))
    union = max(t_e, t_e_gt) - min(t_s, t_s_gt)
    R_time = intersection / union  # IoU ∈ [0, 1]
    
    return R_acc + R_fmt + R_time  # ∈ [0, 3]
```
关键设计选择：(1) 使用 IoU 而非 Recall 作为时间奖励：Recall 允许 policy 放大预测窗口来作弊（span inflation），而 IoU 通过分母 union 项隐式惩罚过度扩展；(2) 解耦 temporal grounding reward：不与 accuracy 耦合，使奖励信号更清晰可解释。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 RL 训练循环中，每个 rollout 生成后立即计算奖励。LLM-as-a-Judge 使用 Qwen3 作为评判模型，通过严格协议（输出 1/0.5/0）避免对模糊案例的奖励。时间窗口解析从 tool_call JSON 的 start_time/end_time 参数提取。该联合奖励设计避免了 prior work 中 accuracy-only 或 IoU-only 的局限，证明单一任务 RL（仅 video QA）配合 decoupled temporal grounding reward 即可在长视频推理上达到 SOTA。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

## RFT (Reinforcement Fine-Tuning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RFT (Reinforcement Fine-Tuning)，也称 ReFT (Reinforced Fine-Tuning)，是一种后训练范式：在 RL 阶段之后，将模型自身 RL policy 生成的高质量 rollout trajectories 转回监督数据，以 SFT 方式进一步微调模型。在 LongVT 的三阶段 pipeline 中，RFT 是第三阶段：从早期 RL 运行的 rollouts 中筛选同时满足 (1) final answer 正确 AND (2) predicted temporal span 与 ground-truth 的 IoU ≥ 0.3 的高质量 trajectories，转换为 <think> + <tool_call> + <tool_response> + <think> + <answer> 结构的监督训练样本，用标准 next-token prediction loss 微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RFT 数据构造和训练流程：
```
# Step 1: 从 RL rollouts 筛选高质量轨迹
rft_data = []
for rollout in early_rl_rollouts:
    pred_answer = extract_answer(rollout)
    pred_span = extract_time_span(rollout)
    gt_answer = rollout.ground_truth.answer
    gt_span = rollout.ground_truth.time_window
    
    # 双重过滤标准
    answer_correct = (judge_llm(pred_answer, gt_answer) == "Fully Consistent")
    span_accurate = (IoU(pred_span, gt_span) >= 0.3)
    
    if answer_correct and span_accurate:
        # 保留完整交互轨迹作为训练样本
        rft_data.append({
            "prompt": rollout.prompt,
            "completion": rollout.full_text  # 含 think/tool_call/answer
        })

# Step 2: RFT 训练（与 SFT 相同格式，但初始化自最佳 RL checkpoint）
model = load(best_rl_checkpoint)
for step in range(1600):  # 64 GPU, lr=5e-5, cosine schedule
    batch = stream_packing(rft_data, buffer_size=51200)
    loss = -Σ log P(completion_t | prompt, completion_<t)
    update(model, AdamW(loss, lr=5e-5))
```
RFT 的独特价值：RL 阶段通过 exploration 找到好的策略方向，但 RL 训练通常不稳定（reward hacking、policy collapse）；RFT 将成功的探索结果蒸馏为稳定的监督信号，巩固 RL 阶段获得的 temporal grounding 和 tool-calling 模式，使性能超越 RL-only plateau。在 VideoSIAH-Eval 上，RFT 相比 RL-only 有显著提升（42.0 vs 35.9）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RFT 已在多个场景中被验证有效：(1) OpenAI 的 o1 系列使用 RFT 作为后训练组件；(2) Fireworks.ai 提供 RFT API 服务；(3) 2025 年多篇论文证明 RFT 的 on-policy 数据生成本质上是其抵抗灾难性遗忘的关键因素，而非 KL 正则化或特定 RL 算法。在 LongVT 中，RFT 使用与 SFT 相同的训练基础设施（LMMs-Engine + stream packing），但 compute 从 32 GPU 扩展到 64 GPU，训练约 1600 steps。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

## Video Segment-In-A-Haystack (VideoSIAH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VideoSIAH (Video Segment-In-A-Haystack) 是 LongVT 提出的数据套件和评估范式。它模拟长视频理解中的关键挑战：答案所需的关键视觉证据（"needle"）仅存在于视频若干小时的极窄时间窗口（"haystack"）中。与传统的针-in-a-干草问题（Needle-In-A-Haystack, NIAH，测试 LLM 在长文本中检索特定事实的能力）不同，VideoSIAH 将概念扩展至视频领域：(1) 问题证据稀疏且时间上分散；(2) 证据以视觉形式存在，需要模型具备 temporal localization 和 visual reasoning 能力；(3) 采用开放式 QA 格式（而非 MCQ）避免选项记忆偏差。VideoSIAH 包含训练数据（247.9K SFT + 1.6K RL + 15.4K RFT）和评估基准 VideoSIAH-Eval（244 视频、652 QA 对，平均时长 1688s）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VideoSIAH 数据构造 pipeline（半自动 + human-in-the-loop）：
```
# 1. 场景检测与分割
segments = pixel_level_scene_detection(long_video)
segments = merge_short_segments(segments, min_duration=10s)

# 2. 视频片段描述生成
for segment in segments:
    caption = Qwen2.5-VL-72B.describe(segment)
    segment.caption = caption  # 物体、空间关系、事件演变

# 3. QA 对生成（基于 captions）
for segment in segments:
    qa_pairs = generate_qa_from_caption(segment.caption)
    # 覆盖: temporal events, spatial layouts, motion, 
    #       object attributes, scene transitions

# 4. 两阶段 QA 过滤
# Stage 1: Text-based filtering
qa_pairs = filter_by_linguistic_heuristics(qa_pairs)
qa_pairs = filter_by_model_agreement(qa_pairs)
# Stage 2: Multimodal filtering
for qa in qa_pairs:
    if not GLM-4.5V.verify(qa.answer, segment.video):
        discard(qa)

# 5. Human-in-the-loop refinement
# 人工检查少量代表性失败案例 → 改进 QA 生成/过滤 prompt 规则
# Prompt-feedback refinement loop: 提升可靠性无需全量人工标注

# 6. iMCoTT trace 生成（仅 SFT 阶段）
for qa in filtered_qa_pairs:
    imcott_trace = generate_multiround_tool_trace(
        qa, video, 
        P_multi = 1 - (L_max - clip(L_video, L_min, L_max))/(L_max - L_min)
    )
```
RL 数据额外经过 difficulty-aware filtering：对每个问题采样 K 个 rollouts，若全部正确（太易）或全部失败（太难）则丢弃，仅保留混合结果的 middle-band 样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoSIAH 数据集的构造利用了多个 SOTA LMMs（Qwen2.5-VL-72B 描述视频、GLM-4.5V 验证答案、Gemini 2.5 Flash 蒸馏 iMCoTT traces），通过半自动 + human-in-the-loop pipeline 在质量和规模间取得平衡。评估基准 VideoSIAH-Eval 通过 contamination study 验证了零泄漏（Qwen3-VL 在 "No Visual" 设置下得分为 0.00），且开放式 QA 格式天然免疫 MCQ option bias。这种 segment-in-a-haystack 范式特别适合评估需要长视频中精确定位稀疏证据的推理能力。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

## Cold-Start SFT for Tool-Aided Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cold-Start SFT for Tool-Aided Reasoning 是指在 RL 训练之前，先用包含工具调用示范的监督数据对基础模型进行微调，使其获得基本的工具使用能力。LongVT 发现，若直接对 Qwen2.5-VL-7B 进行 RL 训练（跳过 SFT），模型会崩溃：无法正确定位时间窗口、无法整合工具输出、tool-call 频率趋近于零（Figure 3b）。因此 Cold-Start SFT 是"先教范式，再优化决策"的必要前提。SFT 阶段教会模型三种基本能力：(1) 提出精确的时间窗口；(2) 基于窗口内细粒度帧进行推理；(3) 窗口不理想时自我纠正。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
Cold-Start SFT 训练设置：
```
# 模型：Qwen2.5-VL-7B-Instruct
# 框架：LMMs-Engine, 32 GPU
# 数据：228.8K non-tool samples + 19.2K tool-augmented samples
# tool-augmented 数据包括：
#   - Gemini 2.5 Flash 蒸馏的 iMCoTT traces (12.8K) for open-ended QA
#   - Qwen2.5-VL-72B 蒸馏的 temporal grounding traces (6.4K)
# 训练技术：stream packing (buffer=51200 tokens), dynamic batching
# 优化器：AdamW, lr=5e-5, cosine schedule, 300 warmup steps, 3000 total steps
```
SFT 训练目标为标准 next-token prediction loss。关键数据特征：(1) 多轮 tool calling traces 根据视频长度自适应生成（长视频 P_multi 更高）；(2) 混合 image/video reasoning 数据保持通用感知能力。SFT 阶段不需要 tool reward——仅靠模仿 tool-augmented traces 就能教会模型工具调用语义。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cold-Start SFT 的一般适用条件：当目标 RL 行为涉及基础模型完全不具备的能力（如工具调用、特定输出格式、多步交互范式）时，冷启动 SFT 是必要的。实现方式：(1) 收集/生成包含目标行为的示范数据（可通过更大模型蒸馏、人工标注、或半自动 pipeline）；(2) 用标准 SFT loss 训练，通常 1-3K steps；(3) 使用 stream packing 等训练优化以提高 GPU 利用率。LongVT 的消融实验（Table 3）证实：移除 Cold-Start SFT 后 RL-only 模型在所有 benchmark 上的表现均为最低。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

## LLM-as-a-Judge

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 是一种使用大型语言模型作为自动评估器的方法，用于评判模型生成文本的质量、正确性或一致性。在 LongVT 的 RL 训练中，LLM-as-a-Judge 用于评估 open-ended QA 的答案准确性：给定问题 Q、标准答案 A* 和模型回答 A_hat，Judge LLM 输出三级判定——Fully Consistent (语义等价，得 1.0)、Partially Consistent (包含部分正确信息但不完整/不精确，得 0.5)、Inconsistent (错误或矛盾，得 0.0)。使用严格评判协议（图 6）：仅输出 1/0.5/0 数字，避免对模糊案例提供奖励，以确保 RL 奖励信号的可靠性。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 的评判流程：
```
def judge_answer(model_answer, ground_truth, question):
    prompt = f"""
    Below are two answers to a question.
    Question is: {question}
    [Standard Answer] is: {ground_truth}
    [Model_answer] is: {model_answer}
    
    Judge how consistent the two answers are.
    Scoring rules:
    - 1 Fully consistent: convey the same meaning
    - 0.5 Partially consistent: overlap on some key points but not all
    - 0 Inconsistent: they conflict or share no essential overlap
    
    Output **only** one of: 1, 0.5, or 0.
    """
    verdict = judge_llm(prompt)  # 使用 Qwen3 作为评判模型
    return float(verdict)
```
在 GRPO 训练流程中，每个 rollout 的 answer 部分被提取后送入 Judge LLM 获得分数，该分数直接作为 R_acc 组件进入联合奖励函数 R = R_acc + R_fmt + R_time。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLM-as-a-Judge 已被广泛用于 RLHF/RL 训练中的自动评估（取代人工标注），特别适用于 open-ended 生成任务（长文本 QA、摘要、翻译）无法用规则匹配评估的场景。实现注意事项：(1) 评判模型应比训练模型更强或至少相当；(2) prompt 设计需包含明确的分级标准和输出格式约束；(3) 存在 position bias、verbosity bias 等已知偏差，可通过多轮评判或位置随机化缓解；(4) 在 LongVT 中，RL 训练设置恒定 temperature=1.0 以鼓励探索，评判严格性对防止 reward hacking 至关重要。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

SAGE 中的 LLM-as-a-Judge 使用方式（binary correctness verdict）：
SAGE 使用 GPT-4o 作为 LLM-Judge，但采用简单的 binary verdict（True/False）而非三级评分。Judge prompt: "Compare the model prediction and the ground truth and determine if they convey the same meaning for the question... respond with the verdict as 'True' if they match semantically or 'False' if they don't match." 该 binary 判定直接作为 GRPO accuracy reward a_N 的核心输入。与 LongVT 的三级评分 (1/0.5/0) 相比，SAGE 的二元判定更激进——完全正确得正奖励，部分正确也可能被判为 False 得负奖励——这迫使模型追求完整正确的答案。SAGE 同时将 LLM-as-a-Judge 用于 evaluation（对所有 DIRECT 和 AGENT baselines 统一评估），保持 training 和 evaluation 的一致性。

## Native Tool Calling in LMMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Native Tool Calling 指 LMM 通过端到端训练将工具调用策略完全内部化的能力——模型自主决定何时调用工具、传什么参数、如何整合工具返回结果，无需外部检索代理或专家模型辅助决策。"Native"（原生）的核心在于工具调用能力是通过 tool-integrated fine-tuning 内嵌于模型权重中的，而非依赖外部规则或 prompt engineering。在 LongVT 中，native tool 具体指 crop_video(start_time, end_time) 函数：模型在需要时生成结构化 JSON 调用该函数，从原始视频指定时间段重新采样细粒度帧。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
Native Tool Calling 的推理执行流程：
```
# 模型生成结构化 tool call
output = llm.generate(prompt)
# output 包含: <tool_call>
#   {"name":"crop_video","arguments":{"video_path":"...","start_time":763.0,"end_time":995.0}}
# </tool_call>

# External executor 执行工具（非模型内部）
tool_response = crop_video_executor(video_path, start_time, end_time)
# tool_response = {frames: [resampled_64_frames]}

# 工具结果注入回上下文
new_prompt = prompt + output + format_tool_response(tool_response)
# 模型基于新视觉证据继续推理
output2 = llm.generate(new_prompt)
# <think> verify evidence... </think>
# <answer> final answer </answer>
```
训练流程：SFT 通过模仿 Gemini/Qwen 蒸馏的 tool-augmented traces 教会模型工具调用语法和语义；RL 通过联合奖励优化工具调用的时机和精度；RFT 通过高质量自蒸馏轨迹巩固工具使用模式。消融实验（Figure 3b）证明：直接 RL（无 cold-start SFT）导致 tool call 频率崩溃至零；一旦通过 SFT 建立基础能力，模型在 RL 中 tool-call 频率和 accuracy 同步提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Native Tool Calling 的实现通常需要：(1) 工具定义（function name, description, parameters schema）以 system prompt 或特殊 tokens 形式注入；(2) 训练数据包含完整的 tool call → tool response → reasoning 交互轨迹；(3) 推理时通过特殊分隔符（如 <tool_call>/</tool_call>）解析工具调用、执行外部函数、将结果注入上下文。与 MCP (Model Context Protocol) 等标准化协议结合时，可通过统一的 tool server 管理多种工具。LongVT 在评估时部署 MCP server + vLLM continuous batching 架构。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

SAGE 中的 Native Tool Calling 使用方式（6-tool multi-turn agent）：
SAGE 扩展了 tool set 至 6 种工具——web-search (Serper Google Search API)、parse-website、transcribe-speech (Whisper-large-v3)、ground-event (Qwen3-VL-30B-A3B-Instruct)、extract-video-parts 和 analyze (Qwen3-VL-30B-A3B-Instruct)——使 orchestrator SAGE-MM 可进行 knowledge-driven multi-turn reasoning。与 LongVT 单一 crop_video 工具不同，SAGE 的 tool diversity 要求 SAGE-MM 学会智能选择工具：例如知道 F1 2024 赛季排名后，通过 web-search 缩小 2025 赛季视频的搜索空间。Tool calling 通过 JSON action 格式实现（而非 LongVT 的 XML tag 格式）：Stage-1 和 Stage-2 都输出推荐工具调用的 JSON 对象（含 rationale、name、arguments）。RL 训练中 s_reasonable-tool 奖励（GPT-4o judge）惩罚不合理的工具调用。消融实验（Table 10）：移除 transcribe-speech 降 5.5%（verbal 问题降 36.5%），移除 extract-video-parts 降 5.0%（visual 问题降 5.4%），移除 web-search 降 2.5%。per-tool accuracy（Table 18）：transcribe-speech 单独最高 61.1%，extract-video-parts/ground-event 最弱 50.2-50.3%（依赖其他工具做局部处理）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality Pre-fusion（模态预融合）是 LLaVA-Mini (Zhang et al., 2025) 提出的核心机制——在 LLM backbone 之前，用额外 Transformer 块将 vision token 中的视觉信息提前融合进 text token，使后续可将 vision token 极端压缩（甚至到 1 个 token）而不损失性能。动机来自论文对 LLaVA 架构的逐层注意力分析：vision token 主要在 LLM 前几层被 text token 通过 attention attend 以"吸收"视觉信息，深层中 vision token 被关注的 attention 急剧下降（80%+ 转向 instruction token），因此深层中的大量 vision token 是冗余的。Pre-fusion 模块的 N_fusion 个 Transformer decoder 块与 LLM backbone 同构（相同结构和超参数），将全部 vision token 和 text token 拼接后做 self-attention，然后仅提取 text token 对应位置的输出作为"融合 token"（fusion token），这些 text token 已携带了所需的视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pre-fusion 在 LLaVA-Mini pipeline 中的计算过程：
```
# 输入
H_v = ViT(image)  → Projection    # [576, d_h] 全部 vision token
H_q = LLM_Embed(text)             # [l_q, d_h] text token

# Pre-fusion: N_fusion 层与 LLM 同构的 Transformer decoder blocks
concat = Concat(H_v, H_q)         # [576 + l_q, d_h]
for i in range(N_fusion):         # 默认 N_fusion=4
    concat = PreFusionDecoderBlock_i(concat, causal_mask)

# 提取 text token 位置的输出作为 fusion token
H_q_fused = concat[-l_q:]         # [l_q, d_h]

# 后续: 压缩后的 vision token (甚至仅 1 个) + fusion text token 输入 LLM
```
其中 PreFusionDecoderBlock 与 LLM backbone 的 Transformer block 完全相同（包括 dimensions、heads、FFN 结构），但不共享权重。消融实验（Table 6）：N_fusion=0 时 1-token 达 VQA-v2 72.4/GQA 54.2/MMB 57.7；N_fusion=4 时提升到 77.6/60.9/65.6（1.96T FLOPs），远超只增加 vision token 数（144 token w/o pre-fusion 仅 76.9/58.9/64.9 at 2.85T FLOPs）。在相同 FLOPs 下，增加 pre-fusion 层比增加 vision token 数收益更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pre-fusion 模块实现为与 LLM backbone 完全同构的 Transformer decoder 层（如 Vicuna-7B 的 decoder block），在模型代码中直接复用 LLM 的 TransformerBlock 定义，设置层数 N_fusion=4。训练时 pre-fusion 模块参与 Stage-2 Instruction Tuning 的端到端训练。关键设计考量：(1) 放在 LLM 外部而非内部——如在 LLM 第 L 层执行 fusion，vision 经过早期层后携带上下文信息反而不利于后续压缩；放在外部也保持了 LLM backbone 不变，兼容所有 LLM 加速框架。(2) Pre-fusion 仅取 text token 位置输出——因为目的是将视觉信息融入文本，而非保留 vision token。(3) 对于视频，每帧 text token 的 fusion 结果经 pooling 聚合为视频级 fusion token。开源实现：https://github.com/ictnlp/LLaVA-Mini。

涉及论文标题：
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token

## Self-Speculative Decoding / SSD（自投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Self-Speculative Decoding (SSD) 是一种无需独立 draft model 的投机解码变体。核心思想：利用同一个模型的**浅层子网络作为 draft model**，**完整模型作为 target model**，通过"draft-then-verify"的迭代方式加速自回归解码。与传统投机解码（需要另一个独立的小模型做 draft）不同，SSD 的 draft 和 target 模型共享部分层的权重和 KV-cache。最简单的方式是 LayerSkip 的 early-exit：让 LLM 的中间层直接预测下一个 token 作为 draft，然后用完整模型验证。TwigVLM 的 SSD 设计：在 base VLM 第 K 层后附加 T 层 twig block，形成浅层子网络 Ms（前 K 层 + twig）作为 draft model，完整模型 Mb（全部 L 层）作为 target model。Draft 每次自回归生成最多 δ=5 个候选 token（配合 early-exit 机制：当预测概率 < θ=0.6 时提前停止 draft），target 通过一次并行前向验证所有候选 token。接受匹配的 draft tokens 后，target 追加一个 bonus token。由于 draft 和 target 共享前 K 层的计算和 KV-cache，SSD 的开销远低于独立双模型方案。SSD 是 lossless 的：最终输出与 target model 原生自回归解码完全一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Self-Speculative Decoding 单次迭代
# Ms: 浅层 draft model (前K层 + twig)
# Mb: 深层 target model (完整L层)
# θ: 置信度阈值, δ: max draft length

# === Draft Phase (draft model 自回归生成候选tokens) ===
draft_tokens = []
for step in range(δ):
    logits_s = Ms.forward(current_token)  # 浅层前向
    prob = softmax(logits_s)
    next_tok = argmax(prob)
    draft_tokens.append(next_tok)
    if max(prob) < θ:           # early-exit
        break
    current_token = next_tok

# === Verify Phase (target model 并行验证) ===
logits_b = Mb.forward(draft_tokens)  # 一次并行前向
# 逐个对比 draft token 与 target 预测
accepted = []
for i, (draft_tok, logits) in enumerate(zip(draft_tokens, logits_b[:-1])):
    target_tok = argmax(logits)
    if draft_tok == target_tok:
        accepted.append(draft_tok)
    else:
        break
# 追加 bonus token
accepted.append(argmax(logits_b[len(accepted)]))
```

TwigVLM 的 SSD 关键设计：
- draft model Ms = {T_1..T_K} ∪ {G_1..G_T}（前K层+twig）
- target model Mb = {T_1..T_L}（完整VLM）
- 共享前K层 KV-cache，draft 只需计算 twig 的 forward
- δ=5，θ=0.6（early-exit 阈值）

TwigVLM++ 的 Tree-based SSD 扩展：
- Draft model 构建 token tree（expansion width E=10, selection width K=10, depth D=4）
- Target model 用 tree attention（topology-aware causal mask）并行验证所有候选路径
- 从根节点遍历，接受匹配的子节点，直到某层无匹配
- 追加一个 bonus token
- 每次验证接受更多 tokens，RelSpd 从 154% 提升到 ~197%（长 response 场景）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：LayerSkip 已集成到 HuggingFace Transformers 的 `generate()` 中（`assistant_early_exit` 参数）。TwigVLM 的 SSD 在 https://github.com/MILVLG/twigvlm 开箱可用。SSD 的关键优势：(1) 无需额外存储独立 draft model（节省 GPU 内存）；(2) 共享 KV-cache 减少冗余计算；(3) Lossless：输出与 target model 完全一致；(4) 特别适合长 response 场景（decode 阶段为主要瓶颈）。局限性：draft 的 token acceptance rate（TokAR）对 speedup 至关重要，受 draft model 质量影响。当 response 较短时加速有限（prefilling 时间占比高）。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## Token Pruning / Twig-guided Token Pruning / TTP（Token剪枝 / Twig引导的Token剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Pruning 是一种通过移除冗余 token 来降低 Transformer 计算复杂度的加速方法。在 VLMs 中，视觉 token 数量通常远超文本 token 且含大量冗余信息，因此在 VLM 的早期层剪枝视觉 token 可显著减少后续层的计算量。传统方法（如 FastV）使用 VLM 早期层（如第 2 层）的 attention map，取文本 token 对视觉 token 的 attention scores 之和，选择 top-R 最重要的视觉 token 保留，丢弃其余。但早期层 attention 对多模态语义理解不充分（"attention signals in early layers are insensitive to the task"），导致剪枝后精度大幅下降。

Twig-guided Token Pruning (TTP) 是 TwigVLM 提出的改进方案：在 base VLM 第 K 层后附加 T 层 twig block，使用 twig 最后一层（深度 K+T，更靠近 prediction head）的 attention map 指导 token 剪枝。由于 twig 层更接近 loss 函数，其 attention 对多模态关系的理解更精准。TTP 流程：输入 tokens X → base VLM 前 K 层 → 得 X^(K)_Mb → twig block → 得 final twig layer attention map A^(K+T)_Ms → 选择 top-R 视觉 token → X̂^(K)_Mb = P(X^(K)_Mb, A^(K+T)_Ms, R) → 传入剩余 VLM 层。配合 FinalWipe 策略在 Kf 层后移除所有视觉 token，平均保留 token 数 R̄ = [M×K + R×(Kf-K)]/L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TTP: Twig-guided Token Pruning
# 输入: X = [X_v (visual tokens), X_q (text tokens)]
# K: pruning 位置层, T: twig 层数
# R: 保留的 visual token 数

# Step 1: 前向到共享层
X_K_mb = Mb.forward_layers(X, start=1, end=K)
           # 同时得到 twig 的 attention
X_K_ms, Attn_twig = Ms.forward_twig(X_K_mb)

# Step 2: 用 twig 最后层的 attention 计算重要性
# Attn_twig ∈ R^{(M+N)×(M+N)}: twig 最后层的 attention map
# 取 text tokens 对 visual tokens 的 attention scores
attn_scores = sum(Attn_twig[text_positions, visual_positions])
               # ∈ R^M, 每个 visual token 的重要性分数

# Step 3: Top-R 选择
keep_indices = topk(attn_scores, k=R)
X_kept_vis = X_K_mb[keep_indices]
X_kept = concat([X_kept_vis, X_K_mb[text_positions]])

# Step 4: 传入剩余层
output = Mb.forward_layers(X_kept, start=K+1, end=L)
```

与传统 Token Pruning (FastV) 的对比：
| 特性 | FastV-style Pruning | TTP (TwigVLM) |
|------|-------------------|---------------|
| 剪枝信号来源 | VLM 早期层 attention (如 layer 2) | Twig 最后层 attention (如 layer 5) |
| Attention 深度 | K (浅) | K+T (更深，更靠近 loss) |
| 信号质量 | 低 (attention 对任务不敏感) | 高 (attention 对多模态理解精准) |
| 额外参数 | 无 | T 层 twig block (~10% base VLM) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 训练时，仅训练 twig block（冻结 base VLM），使用标准 AR loss + SFT 数据（如 LLaVA-665K），训练耗时约 10% 的 base VLM 训练时间。(2) 推理时，twig block 同时在 prefilling（TTP）和 decoding（SSD）阶段使用。开源实现：https://github.com/MILVLG/twigvlm，使用 `--twig-K 2 --twig-T 3` 配置。TTP 的 R 值通过 Eq.(6) 根据目标 pruning ratio 1-R̄/M 反算。TwigVLM++ 使用 P-Head 专门计算 token 重要性 scores（替代 attention map），进一步解耦剪枝与预测任务。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
- Representation_Shift__Unifying_Token_Compression_with_FlashAttention
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

**Representation Shift-based Token Pruning**（来自 Representation Shift 论文）：
该论文提出了一种完全不依赖 attention map 的 token 重要性度量——representation shift（表示漂移），公式为 s = ||MLP(LN(x')) - x'||₂，即 token 经过 MLP 层的 L2 表示变化量。关键发现：(1) MLP 操作逐 token 独立，产生的 representation shift 比 attention-based 方法更具判别性；(2) L2 距离在所有深度上优于 L1 和 cosine distance。该方法的根本优势在于**无需 attention map**，因此可与 FlashAttention 完全兼容——FlashAttention 避免构建完整 attention map 以减少 HBM I/O，传统 attention-based pruning 无法使用。结合 FlashAttention + representation shift pruning 实现乘法级加速：FlashAttention 自身约 2.7× speedup，pruning 再额外约 2× speedup，总计 5.5× (UMT-L video-text retrieval)。方法进一步扩展到 CNN（通过行/列级表示变化剪枝）和 SSM（替换激活值基重要性分数），验证了模型无关性。

## FinalWipe Strategy（最终擦除策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FinalWipe 是 TwigVLM 提出的一种简单但有效的 token 剪枝补充策略：在 VLM 推理的某个较深层（Kf，如第 24 层）之后，**移除所有剩余的视觉 token**，后续层仅处理文本 token。其动机基于两个观察：(1) 此前研究已表明 VLM 深层（如 20 层之后）的视觉 token 对最终预测贡献极小；(2) 在固定平均保留 token 数 R̄ 下，FinalWipe 允许在前中层保留更多 token R（因为深层 token 数为 0 拉低了平均值），从而提升剪枝后的模型精度。引入 FinalWipe 后，R̄ 的计算从 R̄=[M×K+R×(L-K)]/L 变为 R̄=[M×K+R×(Kf-K)]/L，在 K 和 R̄ 固定时允许更大的 R。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FinalWipe: 在 Kf 层后移除所有 visual tokens
# Kf: FinalWipe 位置 (如 24)
# L: VLM 总层数 (如 32)

def forward_with_final_wipe(X_kept, Kf, L):
    # X_kept: 已被 TTP 剪枝后的序列 [kept_visual_tokens, text_tokens]
    # 前向到 Kf 层
    for layer in range(K+1, Kf+1):
        X_kept = transformer_layer(X_kept)
    
    # FinalWipe: 移除所有 visual tokens
    # X_kept = [visual_R, text_N] → [text_N]
    X_text_only = X_kept[text_positions]
    
    # 剩余层仅处理 text tokens
    for layer in range(Kf+1, L+1):
        X_text_only = transformer_layer(X_text_only)
    
    return X_text_only
```

效果（TwigVLM 消融实验，T=3, K=2, R̄=64）：
| FinalWipe | Kf | R | RelAcc | RelSpd |
|-----------|----|---|--------|--------|
| × | 32 | 30 | 93.1% | 154.6% |
| ✓ | 20 | 50 | 95.8% | 151.3% |
| ✓ | 24 | 41 | 96.0% | 153.6% |
| ✓ | 28 | 37 | 95.1% | 154.1% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FinalWipe 是一种纯推理时策略，无需额外训练。在 TwigVLM 的推理代码中，通过 `forward_high_layers(final_wipe=Kf)` 参数控制。Kf 的选择需要在精度（更大的 R）和速度（更少的 FFN 计算）之间权衡。消融实验表明 Kf=24 是最优平衡点。实现简单：在指定的 Kf 层后，将 attention 计算中的 KV-cache 视觉 token 部分置为不可见即可。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## PredKL and AttnKL Distillation Losses（预测级和注意力级KL蒸馏损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PredKL 和 AttnKL 是 TwigVLM++ 在第一训练阶段（蒸馏学习）中引入的两个辅助损失函数，用于提升 multi-head twig 的训练质量。

- **PredKL (Prediction-level KL Divergence Loss)**：L_PredKL = KL(p_Mb || p_Ms)，其中 p_Ms 和 p_Mb 分别是浅层 draft model Ms 和深层 target model Mb 的 next-token 预测概率分布。这是一种"强到弱"的蒸馏（strong-to-weak distillation）：用更强的 target model (Mb) 的预测分布作为 soft target，蒸馏到较弱的 draft model (Ms)。这为 twig block 提供了更丰富的监督信号，增强了其对视觉 token 的理解能力，从而间接提升 P-Head 的剪枝质量。

- **AttnKL (Attention-level KL Divergence Loss)**：L_AttnKL = KL(a_b || s)，其中 a_b ∈ R^M 是 target model Mb 某指定层的文本到视觉 token 的 attention 分布（各头平均），s ∈ R^M 是 P-Head 输出的 token 重要性分数（Eq.7）。该 loss 直接监督 P-Head，使其重要性分数与深层模型的 attention pattern 一致——而深层 attention 已被证明能提供更精准的 token 选择信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage-1 Distillation Training (TwigVLM++)
# α=0.1, γ=1.0

for batch in dataloader:
    # forward shallow model (draft)
    logits_s, p_head_scores = Ms.forward(X)  # p_head_scores = s (Eq.7)
    # forward deep model (target, frozen)
    logits_b, attn_b = Mb.forward(X)          # attn_b = a_b
    
    # Loss 1: Standard AR next-token prediction
    L_NTP = CrossEntropy(logits_s, y_true)
    
    # Loss 2: PredKL - 预测分布对齐
    p_s = softmax(logits_s)
    p_b = softmax(logits_b)
    L_PredKL = KL(p_b || p_s)  # target分布为指导
    
    # Loss 3: AttnKL - attention/重要性对齐
    L_AttnKL = KL(a_b || s)    # 深层attention为指导
    
    # 总损失
    L = L_NTP + α * L_PredKL + γ * L_AttnKL
    # α=0.1, γ=1.0
    
    # 仅更新 twig block (包括 P-Head)
    L.backward()
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PredKL 和 AttnKL 仅在 TwigVLM++ 的 Stage-1 训练中使用。消融实验表明：(a) 仅用 D-Head + L_NTP = 96.0% RelAcc；(b) D-Head + P-Head + L_NTP + L_AttnKL = 95.0%（多 head 分散训练能力导致退步）；(c) D-Head + P-Head + L_NTP + L_AttnKL + L_PredKL = 96.4%（PredKL 补偿多 head 的训练不足）。经过 Stage-2 RL 优化后，配置 (c) 达到最佳 97.7% RelAcc。AttnKL 的 teacher attention 取自与 twig 最后一层深度相同的 base VLM 层（即第 K+T 层），确保深度一致。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## Tree Attention（树注意力机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tree Attention 是一种为树状结构的 token 序列设计的因果注意力变体。在标准 self-attention 中，causal mask 保证每个 token 只能 attend 其左侧（前缀）的所有 token。在 tree-based speculative decoding 中，draft model 构建了一个 token 树（多条候选路径），需要 target model 在一次前向中并行验证整个树的所有节点。Tree Attention 使用 **topology-aware causal mask** 替代标准 causal mask：每个树节点只能 attend 其祖先节点（从根到该节点的路径上的 token），而不能 attend 树中的兄弟节点或无关分支的 token。这保证了因果性的同时允许并行处理整棵树。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tree Attention 的 topology-aware causal mask 构造
# 树结构: 每个节点有 parent_id, position_id

def build_tree_attention_mask(tree_nodes, N):
    # tree_nodes: [{id, parent_id, position_id, token}]
    # N: 总节点数
    mask = zeros(N, N)  # 全 -inf / 全 masked
    
    for i, node_i in enumerate(tree_nodes):
        for j, node_j in enumerate(tree_nodes):
            # node_j 是 node_i 的祖先（或自己）
            if is_ancestor(node_j, node_i, tree_nodes):
                mask[i][j] = 0  # 允许 attention
    
    return mask

# is_ancestor: node_j 在 node_i 的根到 node_i 路径上
def is_ancestor(node_j, node_i, tree_nodes):
    cur = node_i
    while cur is not None:
        if cur.id == node_j.id:
            return True
        cur = cur.parent
    return False
```

TwigVLM++ 的 Tree-based SSD 设置：
- Draft model 构建 token tree：expansion width E=10, selection width K=10, depth D=4
- 树结构：Level 1: E=10 children; Level 2-4: 每层选 top-K=10 节点各扩展 E=10 children
- 验证前裁剪至 Nmax=60 节点
- Target model 用 tree attention 并行验证

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tree Attention 最早由 SpecInfer (Miao et al., 2023) 和 Medusa (Cai et al., 2024) 提出。实现方式：在 FlashAttention 或标准 attention kernel 的基础上，将 topology-aware mask 传入 attention 的 mask 参数即可。关键开销权衡：(1) 树越大 → 每次接受更多 tokens → 速度更快；(2) 但验证的前向计算量也更大（需处理 Nmax 个节点）；(3) 需要在实际中获得接受 token 率和验证开销之间的平衡。TwigVLM++ 使用 E=K=10, D=4, Nmax=60 的配置，在长 response 场景下实现 ~197% RelSpd（vs 标准 SSD 的 ~154%）。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## Token Acceptance Rate / TokAR（Token 接受率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Acceptance Rate (TokAR) 是投机解码（包括 SSD）中的核心效率指标，定义为 draft model 生成的候选 token 中被 target model 接受的比例：TokAR = #accepted_tokens / #draft_tokens。TokAR 直接决定了投机解码的加速效果——接受率越高，每轮 draft-verify 迭代的有效产出的 token 越多，加速比越大。理论上，speedup ∝ TokAR × draft_length / (1 + overhead_ratio)。在 TwigVLM 中，TokAR 受 twig block 的训练质量影响：更好的 twig 初始化（从 K 层开始而非 L-T 层）可提升 TokAR 从更低水平到 57.4%，从而提升 RelSpd。TokAR 也受 twig 层数 T 的影响：增加 T 提升 TokAR 但也增加 draft 计算开销，T=3 时 TokAR 饱和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Acceptance Rate 计算
def compute_TokAR(draft_model, target_model, prefix):
    total_draft = 0
    total_accepted = 0
    current = prefix
    while not EOS:
        # Draft阶段
        draft_tokens = draft_model.generate(current, max_len=δ)
        total_draft += len(draft_tokens)
        # Verify阶段
        accepted = target_model.verify(draft_tokens)
        total_accepted += len(accepted)
        # 追加bonus token
        bonus = target_model.generate_next(accepted)
        current += accepted + [bonus]
    return total_accepted / total_draft
```

TwigVLM 消融实验中 TokAR 影响因素：
| 变量 | 配置 | TokAR | RelSpd |
|------|------|-------|--------|
| 初始化策略 | random init | 低 | 120.4% |
| 初始化策略 | VLM layers[L-T:L] | 中 | 131.4% |
| 初始化策略 | VLM layers[K:K+T] | 57.4% | 153.6% |
| Twig 层数 T | 1 | 中 | 154.1% |
| Twig 层数 T | 3 | 饱和 | 153.6% |
| Twig 层数 T | 4 | 饱和(计算增) | 145.4% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TokAR 是衡量 draft model 与 target model 对齐程度的关键指标。更高的 TokAR 需要：(1) draft model 足够强（但也不能太强，否则计算开销大抵消加速）；(2) draft 和 target 的分布一致性高。实现中，TokAR 通过在推理时统计 accepted/total draft tokens 获得，可用于动态调整 draft length 或 early-exit 阈值。TwigVLM 的训练（twig 初始化和层数选择）以最大化 TokAR 为目标间接优化。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## Dynamic Pruning-Ratio Schedule（动态剪枝率调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Pruning-Ratio Schedule 是 TwigVLM++ Stage-2 RL 训练中使用的技巧，允许单个训练好的模型在推理时支持多种 pruning ratio（R̄ 值）而无需为每种 ratio 单独重训。核心思路：在 RL 训练过程中，随机化采样 R̄ 值（从候选集 R={64, 85, 107, 128, 149, 171, 192}），并使用 curriculum-based annealing 分布逐渐偏向更激进的 pruning ratio。具体来说，R̄ 的采样概率为 P(R̄=R̄_i) = exp(-β(t)·i) / Σ_j exp(-β(t)·j)，其中 β(t) = β_max · (t/T)^p 是 annealing 参数（β_max=8.0, p=2.0），t 和 T 分别是当前和总训练步数。β(t=0)=0 时分布均匀，随着训练进行 β(t→T)=β_max 时分布集中到最小 R̄ (最激进剪枝)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Dynamic Pruning-Ratio Schedule (Stage-2 RL)
R_set = [64, 85, 107, 128, 149, 171, 192]  # 候选集, 升序排列
n = len(R_set)
β_max = 8.0
p = 2.0

for t in range(T_steps):
    # 计算 annealing 参数
    β_t = β_max * (t / T) ** p
    # 采样分布
    probs = []
    for i in range(n):
        probs_i = exp(-β_t * i) / sum(exp(-β_t * j) for j in range(n))
        probs.append(probs_i)
    # 采样 R̄
    R̄ = sample_categorical(R_set, probs)
    # 用该 R̄ 进行 RL 更新
    loss = grpo_step(R̄, G=32)
    loss.backward()
    optimizer.step()
```

效果对比（RelAcc @ R̄=64/128/192）：
| RL Strategy | @R̄=192 | @R̄=128 | @R̄=64 |
|------------|--------|--------|-------|
| static R̄=192 | 99.1% | 98.8% | 97.2% |
| static R̄=64 | 99.1% | 99.0% | 98.0% |
| dynamic schedule | 99.6% | 99.2% | 97.7% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic schedule 使 P-Head 学习到在不同 R̄ 下都有效的通用剪枝策略，避免了对单一 R̄ 的过拟合（如上表：static R̄=64 在 @R̄=192 退化到 99.1%，而 dynamic 达 99.6%）。实现方式：在每个 RL training step 开始前采样 R̄，根据 R̄ 和 K, Kf 通过 Eq.(6) 反算 R 值，然后执行 GRPO-step。推理时直接指定所需的 R̄ 即可使用训练好的单一模型。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## Multi-head Twig Architecture / P-Head and D-Head（多头Twig架构 / 剪枝头与解码头）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-head Twig Architecture 是 TwigVLM++ 对原始 TwigVLM 的核心架构改进。在原始 TwigVLM 中，twig block 的最后一个自注意力层的 attention map 同时服务于两个目的：next-token 预测（通过 AR loss 间接训练）和视觉 token 剪枝（推理时直接使用 attention scores）。这种耦合设计导致剪枝信号仅作为预测任务的副产品出现，未针对剪枝任务直接优化。TwigVLM++ 引入两个解耦的 head：(1) **D-Head (Decoding Head)**：保留标准 next-token 预测功能，复用原 twig block 的预测头；(2) **P-Head (Pruning Head)**：轻量级辅助模块，专用于计算视觉 token 重要性分数 s ∈ R^M。

P-Head 的计算 (Eq.7)：从 twig 最后一层 SA 的 Q/K 投影中提取 query vector q̃（最后 textual token 位置）和 key matrix K̃（visual token 位置），通过两个可学习的 gating 投影 G_q, G_k（Linear + nonlinear activation）调制后再计算 scaled dot-product attention，最终对各注意力头取平均：s = 1/H · Σ σ((G_q(x_q)⊙q̃)(G_k(X_k)⊙K̃)^T / √d_h)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# P-Head 重要性分数计算 (TwigVLM++)
# 输入: X^(K+T) — twig最后一层SA层输入
# H: 注意力头数, d_h: 头维度

Q = X_Wq    # ∈ R^{(M+N)×d}
K = X_Wk    # ∈ R^{(M+N)×d}

# 提取 query (最后text token位置)
q_tilde = Q[-1].reshape(H, d_h)    # ∈ R^{H×d_h}
x_q = X[-1]                        # 最后text token的hidden state

# 提取 key (visual token位置)
K_tilde = K[:M].reshape(H, M, d_h)  # ∈ R^{H×M×d_h}
X_k = X[:M]                         # visual tokens hidden states

# P-Head gating projections
gated_q = G_q(x_q).reshape(H, d_h) ⊙ q_tilde   # element-wise
gated_k = G_k(X_k).reshape(H, M, d_h) ⊙ K_tilde

# 多头的scaled dot-product attention
scores_per_head = []
for h in range(H):
    s_h = softmax(gated_q[h] @ gated_k[h].T / sqrt(d_h))
    scores_per_head.append(s_h)

s = mean(scores_per_head, dim=0)  # ∈ R^M, 归一化的token重要性
```

D-Head vs P-Head 对比：
| 特征 | D-Head | P-Head |
|------|--------|--------|
| 功能 | Next-token prediction | Visual token importance scoring |
| 输出 | Probability distribution over vocabulary | Scores s ∈ R^M |
| 训练 | L_NTP + L_PredKL | L_AttnKL + RL (Stage-2) |
| 推理使用 | SSD draft generation | TTP token pruning |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
P-Head 的实现：在 twig 最后一层 SA 之后添加两个轻量级 linear 层（G_q 和 G_k），后接非线性激活函数（如 GELU）。训练分为两个阶段：Stage-1 用蒸馏损失（L_AttnKL 监督 s 与深层 attention 对齐），Stage-2 用 GRPO-style RL 直接优化 s 以最大化剪枝后模型性能。推理时，P-Head 的输出 s 替代 attention map 用于 Eq.(8) 的 token 剪枝。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

## KV Cache Dynamic Pruning (KV Cache 动态剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Dynamic Pruning 是一种在 LLM/VLLM 推理的 decoding 阶段，动态评估并调整 KV cache 中保留的 token 集合的技术。与传统 one-shot token pruning（仅在 prefilling 阶段一次性评估 token 重要性并固定剪枝结果）不同，动态剪枝在 decoding 的每一步或每隔 N 步，重新计算当前预测 token 与 KV cache 中视觉 token 的 cross-attention 权重矩阵 A^(L) = Softmax(Q^(L) K^(L)^T / √D)，按 attention score 的 top-p% 阈值动态决定保留哪些 token 在 KV cache 中参与注意力计算。低于阈值的 token 并非永久丢弃，而是移入一个独立的 Dynamic Pruning Cache (DP Cache) 以备后续步骤重新召回。DyCoke 论文的消融实验证明：去除动态机制（改为 one-shot 剪枝）后 VideoDC benchmark 性能显著下降，验证了动态评估的必要性。相关方法包括 Lethe（AAAI 2026，layer-adaptive + recency-aware 动态剪枝）、KVzap（NVIDIA，MLP 预测重要性分数的动态剪枝）、SparK（channel-level 剪枝 + 动态重建）等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 DyCoke 的 decoding pipeline 中，动态剪枝的执行流程如下：
```
# === Decoding Stage: KV Cache Dynamic Pruning ===
# 输入: KV_cache (prefilling后填充), 超参数 L (注意力层), P (保留比例p%)
# DP_cache = {}  (存储被剪枝的token，可召回)

for decoding_step t in range(max_new_tokens):
    # Step 1: 判断是否需要重新评估注意力分布
    if t == 1 or cosine_sim(attention_dist_prev, attention_dist_curr) < threshold_sim:
        # Step 2: 在第 L 层计算当前预测 token 对视觉 token 的 cross-attention
        A = Softmax(Q_pred K_visual^T / sqrt(D))  # shape: (1, N_visual)
        
        # Step 3: 按 top-P% 阈值分离保留与剪枝 token
        threshold_tau = percentile(A, 100 - P)
        keep_idx = where(A >= threshold_tau)
        prune_idx = where(A < threshold_tau)
        
        # Step 4: KV cache 与 DP cache 双向更新
        KV_cache_visual[L] = KV_cache_visual[L][keep_idx]
        DP_cache[L] = DP_cache[L] ∪ KV_cache_full[L][prune_idx]
        
        # Step 5: 召回 DP cache 中注意力回升的 token
        A_dp = Softmax(Q_pred K_DP^T / sqrt(D))
        recall_idx = where(A_dp >= threshold_tau_new)
        KV_cache_visual[L] ∪= DP_cache[L][recall_idx]
        DP_cache[L] -= DP_cache[L][recall_idx]
    
    # Step 6: 使用压缩后的 KV cache 执行 attention 并生成下一 token
    h_t = attention(Q_pred, KV_cache_visual, KV_cache_text)
    KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 中的实现基于 LLaVA-NeXT 代码库，通过 lmms-eval 评估框架调用。关键超参数：L=3（在第 3 层评估 attention，实验证明 L>0 时对性能影响不敏感，说明动态剪枝的稳定性）、P=0.7（保留 top-70% attention 的 token，即剪枝 30%）、K=0.5~0.7（TTM 第一阶段剪枝率）。动态剪枝与 KV cache 的兼容性：DyCoke 兼容 Flash Attention，仅在特定层额外计算一次 cross-attention，复杂度远低于 prefilling 阶段。相关工具：kvpress（HuggingFace，支持 KVzap 等动态剪枝 pipeline）、TRL、PyTorch。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

## Token Temporal Merging / TTM (Token 时序合并)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Temporal Merging (TTM) 是一种在 Video LLM 推理的 prefilling 阶段，通过利用视频帧间的时序冗余（temporal redundancy）合并相似 visual token 来减少输入 token 数量的技术。核心假设是：视频中相邻帧包含大量相似或重复的视觉信息（如静态背景、连续动作的微小变化），可以通过帧间 token 级别的相似度计算来合并冗余 token。TTM 属于 training-free 方法，不需要额外训练或参数修改。DyCoke 的 TTM 采用滑动窗口（window=4 frames）、分组采样（Odd/Even 组）、余弦相似度度量和分层剪枝策略。相似方法：ToMe（Token Merging for ViT，通过 bipartite matching 合并相似 token）、TempMe（progressive merging across neighboring clips）、TESTA（temporal + spatial aggregation）、HoliTom（outer-LLM + inner-LLM holistic merging）、TTF（锚定帧 + 局部窗口相似度搜索）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DyCoke 的 TTM 在 prefilling 初始阶段执行，位于视觉编码器输出之后、LLM 输入之前：
```
# === Prefilling Stage: Token Temporal Merging (TTM) ===
# 输入: H_v' (visual tokens, shape: M_v*N_v × D)
# 超参数: K (保留比例 k%), window_size=4

# Step 1: 滑动窗口分组
for i in range(0, M_v, window_size):  # M_v=32 frames, window=4 → 8 windows
    window = H_v'[i*N_v : (i+4)*N_v]  # 4帧 × 196 tokens = 784 tokens
    # 分为 Odd 组 (帧0, 帧2) 和 Even 组 (帧1, 帧3)
    O_group = window[0*N_v:1*N_v] ∪ window[2*N_v:3*N_v]
    E_group = window[1*N_v:2*N_v] ∪ window[3*N_v:4*N_v]
    
    # Step 2: 计算 O/E 对应位置 token 余弦相似度
    for pos in range(N_v):
        S[pos] = cos_sim(O_group[pos], E_group[pos])  # Eq.3: h_i·h_j/(||h_i|| ||h_j||)
    
    # Step 3: 剪枝 E 组高相似 token (按 K 比例)
    threshold_E = percentile(S, 100 - k_E)
    E_pruned = E_group[S < threshold_E]
    
    # Step 4: O 组内 frame-0 全保留，其余帧与 frame-0 比相似度剪枝
    O_kept = O_group[0:N_v]  # 窗口第一帧全保留
    for f in [O_frame2]:
        S_o = cos_sim(O_kept[0:N_v], f)  # 与首帧比较
        O_kept ∪= f[S_o < threshold_o * K]

    merged_window = concat[O_kept, E_pruned]
    # 结果：4帧 → 约 K*4 帧等效 token 量

# Step 5: 输出压缩后的 visual tokens → concat 文本 tokens → LLM
H = concat[TTM(H_v'), H_q]  # Eq.4
# 32帧 × 196 = 6272 tokens → 约 1882 tokens (K=0.7 时保留约30%)
```
TTM 处理 32 帧输入仅需 < 10^{-3} 秒，开销可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 通过 PyTorch 实现，预计算 token 间的 cosine similarity 矩阵。使用时通过 lmms-eval 传入 dycoke=True, dycoke_k=0.5~0.7 参数启用。TTM 作为 plug-and-play 模块嵌入在 vision encoder 和 LLM projector 之间，无需修改 LLM 结构。超参数 K 应随输入帧数增加而增大（更多帧 → 更多冗余 → 可更激进压缩）。DyCoke 实验显示 K=0.7 时保留约 30% visual tokens 仍能保持或提升性能。类似工具：HoliTom (github.com/cokeshao/HoliTom)、TTF (github.com/Cominder/ttf)。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

## Dynamic Pruning Cache / DP Cache (动态剪枝缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Pruning Cache (DP Cache) 是 DyCoke 提出的一个辅助 KV cache 结构，用于在 decoding 阶段存储被动态剪枝机制暂时移除的 visual token 的 K/V 对。与传统 one-shot pruning 的"永久丢弃"策略不同，DP Cache 使得 token 可以在后续 decoding 步骤中被召回：被剪枝的 token 进入 DP Cache（而非直接释放），当模型注意力重新关注到这些 token 时（通过每隔 N 步重新计算 cross-attention 检测），它们会被动态加回主 KV cache。同时，KV cache 中注意力下降的 token 也会被移回 DP Cache。KV Cache 和 DP Cache 之间形成双向流动：KV→DP（剪枝）和 DP→KV（召回）。这种设计解决了 one-shot pruning 无法适应 decoding 过程中注意力分布变化的根本缺陷。DP Cache 中 token 不参与当前步骤的 attention 计算，因此不消耗 attention 的 O(n²) 计算成本。DyCoke 消融实验证明去除 DP Cache 后 VideoDC benchmark 性能显著下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DP Cache 在 DyCoke 两阶段压缩中的位置和运转流程：
```
# === DP Cache 运转流程 ===
# 初始状态: KV_cache = {TTM压缩后的visual KV + text KV}
#         DP_cache = {}  (空)

# 首次动态剪枝 (t=1)
A = Softmax(Q_pred K_visual^T / sqrt(D))
keep_idx = top_P_percent(A, P)   # 保留 top-P%
prune_idx = complement(keep_idx)  # 剩余低注意力 token

# 双向分流:
KV_cache[visual]  = KV_full[visual][keep_idx]   # 高注意力 → KV cache
DP_cache           = KV_full[visual][prune_idx]  # 低注意力 → DP cache (保存!)

# 后续解码 (每 N=1 步或注意力变化时)
if cosine_sim(attn_prev, attn_curr) < sim_threshold:
    # 重新评估：将 DP cache 中的 token 与 KV cache 中的 token 联合评估
    A_full = Softmax(Q_pred [K_kv; K_dp]^T / sqrt(D))
    
    # KV cache 更新
    new_keep = top_P_percent(A_full, P)
    new_prune = complement(new_keep)
    
    KV_cache[visual] = [K_kv; K_dp][new_keep]   # 包含从 DP 召回的高注意力 token
    DP_cache         = [K_kv; K_dp][new_prune]   # 包含从 KV 移出的低注意力 token

# 每步仅用 KV cache 中的 token 计算 attention
output = attention(Q_pred, KV_cache)
```
关键设计：KV cache ↔ DP cache 的双向流动性。DP cache 中 token 不参与 attention 计算以节省 FLOPS，但保留完整的 K/V 以便召回。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DP Cache 直接使用与主 KV cache 相同的 key/value 张量格式存储，实现为 PyTorch tensor 的 gather/scatter 操作（基于索引选择）。由于仅在每隔 N 步重新计算 cross-attention（当相邻迭代注意力分布余弦相似度低于阈值时），DP cache 的索引管理开销可控。DyCoke 使用 L=3 层（而非所有层）进行动态评估，进一步降低开销。与 Flash Attention 兼容。类似机制：SparK 的 channel-level recovery（通过存储分布模式重建被剪枝通道）、LeanKV 的混合精度 downgrade 路径。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

## Video Large Language Model / VLLM (视频大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Large Language Model (VLLM) 是一类能够理解和推理视频内容的多模态大语言模型。其推理 pipeline 通常包含：(1) 视频输入 → 帧采样（如 32 frames @ uniform），(2) Vision Encoder（如 CLIP ViT-L）逐帧编码为 visual tokens（每帧 ~196 tokens），(3) Projector（如 MLP）将 visual tokens 映射到 LLM 的 token embedding 空间，(4) 将 visual tokens 与 text prompt tokens 拼接送入 LLM（如 LLaMA/Qwen）进行 prefilling + decoding 生成答案。代表性 VLLM 包括：LLaVA-OneVision（统一图像/多图/视频）、VideoLLaMA 2（时空建模 + 音频）、VideoChat（chat-centric video understanding）、LLaVA-NeXT-Interleave（多图/视频/3D 统一）、VILA（预训练 for visual language）、Tarsier（视频描述）。主要挑战：数十帧 × 196 tokens/frame = 6272+ visual tokens，导致 attention 计算复杂度 O(n²) 爆炸，prefilling 和 decoding 延迟高，GPU 内存占用大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLLM 推理 pipeline（以 LLaVA-OV-7B, 32 frames 为例）：
```
# === VLLM 推理 Pipeline ===
# 模型: LLaVA-OneVision-7B (d=3584, m=18944, T=28)
# 视频输入: 32 frames

# Step 1: 帧采样与视觉编码
frames = uniform_sample(video, 32)          # 32 frames
for frame in frames:
    z_i = VisionEnc(frame)                  # CLIP ViT → 196 tokens/frame
# Z_v shape: (32, 196, d_v) → (6272, d_v)

# Step 2: Projector 映射
H_v' = Projector(Z_v)                       # MLP → token embedding space
# H_v' shape: (6272, D=3584)

# Step 3: concat 文本 tokens
H_q = TextTokenizer(prompt)                 # shape: (N_q, 3584)
H = concat[H_v', H_q]                       # shape: (6272+N_q, 3584)

# Step 4: Prefilling (所有 token 并行计算)
for l in 1..T (28 layers):
    Q = H W_Q^l, K = H W_K^l, V = H W_V^l  # Eq.1
    out = MHA(Q, K, V) + FFN(out)
    KV_cache[l] = (K, V)

# Step 5: Decoding (逐 token 自回归)
for t in 1..max_new_tokens:
    h_t = LM_Head(LLM(KV_cache))            # 仅计算当前 token 的 K/V
    KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]  # Eq.2

# 计算负载: prefilling FLOPs ≈ T(4nd² + 2n²d + 2ndm)
# n = 6272+N_q tokens → 约 41.4T FLOPs (LLaVA-OV-7B)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLLM 通常基于 image LLM 扩展：在预训练的 image MLLM 基础上添加视频数据集（如 VideoChatGPT、ActivityNet）进行 instruction tuning。主流框架：LLaVA-NeXT/OneVision（PyTorch）、VideoLLaMA（时空 Q-Former）、VILA（大规模预训练）。评估：LMMs-Eval 框架支持 VideoMME、MVBench、ActivityNet-QA、PerceptionTest、VideoDetailCaption 等 benchmark。部署：支持 Flash Attention 加速，支持 4-bit/8-bit 量化（LLaVA-1.5），训练需要 A100/H100 多卡，0.5B~72B 参数规模。token 压缩是当前 VLLM 推理加速的核心方向，包括 token merging（TTM, HoliTom, TTF）、token pruning（FastV, PruMerge）、dynamic pruning（DyCoke）等 training-free 方法。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

补充（来自 Sparrow 论文）：Sparrow 揭示了一个重要的 VLLM 训练数据效率问题——当基于 Image-LLM 通过 fine-tuning 开发 VLLM 时，简单地扩大视频数据量（如从 30K 到 200K 样本）带来的性能增益呈对数增长趋势（Video-MME 仅从 55.8 → 56.3），原因在于视频 instruction 数据的多样性不足。Sparrow 发现 ShareGemini 数据集仅使用 9 种固定模板变体（"Describe this video in detail"），而 Video-ChatGPT 的自 instruction 方式也缺乏真正的多样性。解决方案是引入文本域的长上下文数据，通过 text-to-image 合成（PIL/Pillow 将 text segments 渲染为 448×448 图像）模拟视频帧序列结构，以 2:1 比例混合真实视频数据和合成数据训练。Sparrow 用 30K 混合数据（15% 样本量）达到了与 200K 全量视频数据相当的 Video-MME 性能，GPU hours 从 276.8 降至 33.6（8.2× 效率提升），并意外地在长视频理解上获得了 +6.6 points 的提升（100K 规模，LongVideoBench/MLVU）。

## Training-Free Token Compression (免训练Token压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-Free Token Compression 指不需要对模型进行额外训练或微调、直接降低视觉 token 数量的推理加速技术。与需要训练的方法（如 xGen-MM-Vid 需要微调将多帧 token 映射到 compact 集合、VILA 需要大规模视频数据训练）不同，training-free 方法作为 plug-and-play 模块嵌入现有 VLLM 推理流程，保持模型参数完全冻结。核心优势：(1) 零训练成本——无需 GPU 小时训练；(2) 即插即用——可直接应用于不同 VLLM 架构和多规模模型；(3) 保持原始推理能力——不改变模型权重，理论上可无损恢复到 full-token 模式。主要方法包括：(a) Token Merging——基于相似度合并冗余 token（ToMe, TTM, HoliTom, TTF, TempMe）；(b) Token Pruning——基于 attention score 或 salience 剪枝不重要 token（FastV, PruMerge, PyramidDrop, VisionZip, SparseVLM）；(c) Dynamic Pruning——decoding 阶段动态调整剪枝集（DyCoke）；(d) Hierarchical Attention Pruning——基于 ViT 内部不同层 attention 语义差异的剪枝（HiPrune/HiPrune++）；(e) Test-Time Temporal Sampling (T3S)——在推理时生成 m 个短且多样化的子序列，打包到单次前向传播中处理，通过 logit 聚合输出最终预测，利用视频时间冗余同时降低 attention 复杂度（O(L²)→O(∑αᵢ²L²)）并扩展有效时间覆盖。DyCoke 在 training-free 方向上达到 SOTA：1.5× speedup + 1.4× memory reduction，性能不降反升。T3S 在 Qwen2.5-VL-7B 上实现 2.04× speedup 且准确率提升 3.1%。HiPrune 在 LLaVA-1.5 上以 1/3 token 保持 99.3% 准确率，FLOPs 减少 58.7%，并证明 training-free 方法可跨 VLM 架构（LLaVA、Qwen、Video-LLaVA）通用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Training-free 方法的共同模式——在 VLLM 推理流程中插入压缩操作而不改变模型权重：
```
# === Training-Free Token Compression 通用模式 ===
# 前提: VLLM 模型权重完全冻结 (no gradient, no fine-tuning)

# 方式 A: Pre-LLM 压缩 (在视觉 token 进入 LLM 之前)
visual_tokens = VisionEncoder(video_frames)        # (M*N_v, d)
compressed_tokens = compress(visual_tokens)          # 训练无关压缩函数
H = concat[Projector(compressed_tokens), text_tokens] # 送入 LLM

# 方式 B: In-LLM 剪枝 (在 LLM 推理过程中剪枝)
H = concat[Projector(visual_tokens), text_tokens]
KV_cache = LLM_prefill(H)
for t in decoding:
    KV_cache = prune_KV_cache(KV_cache, attention_scores)  # 动态调整
    output_t = LLM_decode(KV_cache)

# 方式 C: 混合 (DyCoke = Pre-LLM TTM + In-LLM Dynamic Pruning)
visual_tokens = VisionEncoder(video_frames)
compressed = TTM(visual_tokens, K=0.7)              # Pre-LLM merging
H = concat[Projector(compressed), text_tokens]
KV_cache = LLM_prefill(H)
for t in decoding:
    KV_cache, DP_cache = dynamic_prune(KV_cache)    # In-LLM dynamic
    output_t = LLM_decode(KV_cache)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 通过 PyTorch 实现，在 lmms-eval 中通过模型参数传入 dycoke=True 启用。关键超参数仅 3 个：K（TTM 保留比）、L（评估层）、P（动态剪枝保留比）。评估使用统一 FLOPs 指标确保与其他 training-free 方法（FastV, PruMerge）公平对比。类似工具：FastV (github.com/pkunlp-icler/FastV)、PruMerge (github.com/42Shawn/LLaVA-PruMerge)、HoliTom (github.com/cokeshao/HoliTom)。所有 training-free 方法的核心 trade-off：token 保留率 vs 性能保持，通常保留 15-35% tokens 即可保持 >99% 性能。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding

## One-Shot Token Pruning (一次性Token剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Shot Token Pruning 是一种在 LLM/VLLM 推理的 prefilling 阶段，基于单次 token 重要性评估来永久性剪枝 visual token 的技术。代表方法：(1) FastV——在 LLM 的特定层（如第 5 层）计算 prompt token 对 visual token 的 attention score，保留 top-k% token 到 KV cache，此后 decoding 阶段不再改变；(2) LLaVA-PruMerge——基于 CLIP 视觉编码器的 attention score（而非 LLM 内部 attention）选择关键 visual token，一次性剪枝。两种方法的核心缺陷（DyCoke 的核心动机）：视频输入中，不同 decoding 步骤关注的视觉 token 不同（temporal attention shift），prefilling 阶段的 attention 分布与后续 decoding 需求不一致，一次剪枝后无法纠正错误判断。DyCoke 论文 Figure 2 通过可视化证明了这一现象：某些 frame 的 attention 在 decoding 后期显著上升，而 one-shot 方法可能已在早期将其剪除。One-shot pruning 的另一局限是：被剪枝 token 的 KV cache 永久丢失，无法被召回。DyCoke 通过引入动态剪枝 + DP Cache 解决问题：每一步可重新评估并调整剪枝集，被剪枝 token 保留在 DP Cache 中可召回。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FastV（典型 one-shot pruning）的流程：
```
# === One-Shot Token Pruning (FastV) ===
# 仅一次评估，永久剪枝
layer = 5  # attention 评估层
keep_ratio = 0.35  # 保留 top-35% visual tokens

# Prefilling: 在指定层评估 attention
for l in 1..layer:
    Q, K, V = H W_Q^l, H W_K^l, H W_V^l
    H = MHA(Q, K, V) + FFN(H)  # 正常 prefill 到 layer 5

# 在第 5 层计算所有 text tokens 对 visual tokens 的平均 attention
A_layer5 = attention_weights(Q_text, K_visual)  # shape: (N_q, N_visual)
A_avg = mean(A_layer5, dim=0)                   # 对 text tokens 取平均

# 一次性剪枝
keep_idx = topk(A_avg, k=keep_ratio * N_visual)
prune_idx = complement(keep_idx)

# KV cache 永久修改
KV_cache_visual = KV_cache_visual[keep_idx]  # 保留高 attention token
# 被剪枝的 token: 永久丢弃！← 核心问题

# 后续 decoding: 使用固定剪枝后的 KV cache
for t in decoding:
    output_t = LLM_decode(KV_cache)  # 剪枝集不再改变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FastV (ECCV 2024) 开源实现：github.com/pkunlp-icler/FastV，基于 LLaVA 推理代码，通过分析 attention distribution 决定剪枝层和比例。LLaVA-PruMerge 开源实现：github.com/42Shawn/LLaVA-PruMerge，基于 CLIP attention sparsity。两者均通过减少 KV cache 中 visual token 数量来降低 decoding 阶段的 attention 计算量。DyCoke 实验证明：在相同 FLOPs 下，dynamic pruning (DyCoke) 显著优于 one-shot pruning (FastV, PruMerge)，尤其在 VideoDC、ActivityNet-QA 等需要动态理解的任务上。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

## Dynamic Compression (动态压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Compression（动态压缩）是 D-CoDe 提出的核心方法组件之一，用于解决图像预训练 VLM 扩展到视频时的感知瓶颈（Perception Bottleneck）。它是一种 training-free 的自适应视觉信息压缩策略，包含时间和空间两个维度的操作：(1) **时间维度**：先均匀采样 ⌊α·N⌋ 帧（α=0.85），再利用 CLIP 的 global feature 计算帧间余弦相似度，迭代选择与已选帧语义最不相似的 supplementary frame，直到共选 N 帧——这种"均匀覆盖 + 多样性补充"策略优先保留语义不同的关键帧；(2) **空间维度**：对每帧的 visual tokens 按 ℓ2 norm 计算 salience 分数（activation magnitude），保留 top-⌊β·M⌋ 高激活 token（β=0.625），然后在保留的 token 中按余弦相似度（阈值 τ=0.9）使用贪婪算法合并冗余 token（anchor + cluster 取平均值作为代表 token）。核心创新在于将静态、均匀、无感知的压缩策略（uniform sampling + average pooling）替换为内容感知的动态策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dynamic Compression 在 D-CoDe pipeline 中的执行流程：
```
# === 时间维度: 动态帧选择 ===
# 输入: 视频 V (T frames), 目标帧数 N, 均匀采样比 α=0.85
# Stage 1: 均匀采样
N_uniform = floor(α * N)
V_selected = uniform_sample(V, N_uniform)

# Stage 2: 基于语义多样性的补充帧选择
for k in 1..(N - N_uniform):
    # 计算每帧与已选帧的平均语义不相似度
    for each frame I_m in V \ V_selected:
        g_m = CLIP_visual(I_m)           # CLIP global feature
        avg_sim = mean(cosine_sim(g_m, g_n) for I_n in V_selected)
    # 选最不相似的帧（最大化多样性）
    I* = argmin(avg_sim)
    V_selected = V_selected ∪ {I*}

# === 空间维度: 动态 Token 压缩（每帧独立） ===
# 输入: 选中帧的 visual tokens F (M tokens × d dims)
# 参数: β=0.625 (保留比例), τ=0.9 (合并阈值)
for each frame in V_selected:
    F = VisualEnc(frame)                 # shape: (M, d)
    
    # Step 1: Salience-based Pruning
    a = [||f_i||_2 for f_i in F]        # ℓ2 norm salience
    F_pruned = TopK(F, key=a, k=floor(β*M))
    
    # Step 2: Greedy Similarity Merging
    sorted_idx = argsort(a, descending=True)
    merged = []
    merged_mask = [False] * len(F_pruned)
    for i in sorted_idx:
        if not merged_mask[i]:
            # 找与 anchor token f_i 相似度超过 τ 的未合并 token
            cluster = [i]
            for j in sorted_idx:
                if j > i and not merged_mask[j]:
                    sim = cosine_sim(F_pruned[i], F_pruned[j])
                    if sim >= τ:
                        cluster.append(j)
                        merged_mask[j] = True
            # 合并 cluster: 取平均值
            f_rep = mean(F_pruned[cluster], dim=0)
            merged.append(f_rep)
    
    frame_compressed = merged

# 最终拼接
F_final = concat([frame_compressed for frame in V_selected])
```

关键张量维度：
- CLIP global feature g_t: 由 CLIP 视觉编码器提取，维度取决于 CLIP 变体（~768 或 ~1024）
- Visual tokens F: 每帧经 LLaVA-NeXT 编码后的 token 序列，(M, d) 其中 M ~576 (336×336, 取决于 patch size)
- ℓ2 norm salience a_i: scalar，量化 token 对整体视觉表示的贡献
- 合并阈值 τ=0.9：只有 cosine similarity >= 0.9 的 token 对才被合并
- α=0.85, β=0.625：通过消融实验确定的最优超参数

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic Compression 在 D-CoDe 中作为 training-free 的前处理模块实现，不修改 LLaVA-NeXT 的任何权重。实现基于 HuggingFace Transformers + PyTorch，核心代码在 `Dcode.py` 的 `supp_frame_selection()`（帧选择）和 `token_select_and_merge()`（token 压缩）函数中。超参数通过 EgoSchema 消融实验确定：α=0.85（15帧中 ~13 帧均匀采样 + 2 帧补充）、β=0.625（保留 62.5% 高 salience token）、τ=0.9（仅合并高度相似的 token）。Dynamic Compression 单独使用将 EgoSchema accuracy 从 44.8% 提升至 51.8%（+7.0%），推理延迟从 3.927 s/sample 增加至 6.115 s/sample（+55.7%）。开源：https://github.com/hukcc/D-CoDe。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition

## Question Decomposition (问题分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Question Decomposition（问题分解）是 D-CoDe 提出的核心方法组件之一，用于解决 Token Overload 问题——即视频输入产生的 visual tokens 数量远超图像预训练 VLM 的处理容量，导致模型无法有效利用全部信息。其核心思路是：使用外部 LLM（GPT-3.5-turbo-0125，temperature t=0.5）将复杂的原始视频问题分解为一组聚焦于视频不同方面的子问题（如角色位置、动作序列、物体交互、场景转换等），每个子问题独立用压缩后的 visual tokens 推理得到子答案，然后将所有子答案拼接为辅助文本，与原始问题和压缩 visual tokens 一起送入 LLM 生成最终答案。关键洞察：子答案（而非子问题本身）提供了多样化的补充语义信息，帮助模型从不同角度"消化"大量 visual tokens——消融实验证实子答案效果远优于子问题（58.0% vs 50.4% on EgoSchema）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Question Decomposition 在 D-CoDe 中的执行流程：
```
# === Question Decomposition ===
# 输入: 原始问题 Q, 压缩 visual tokens F_final
# 参数: t=0.5 (生成温度), model=GPT-3.5-turbo-0125

# Step 1: 子问题生成
system_prompt = """
I am working on a video understanding task. Your job is to break down 
the given question into a series of subquestions that guide the model 
toward solving the problem. The subquestions should focus on temporal 
and dynamic aspects of the video, rather than just static information 
that could be answered from a single frame.
"""
Q_1, Q_2, ..., Q_n = GPT3.5(Q, system_prompt, temperature=t)
# n 不限制, 由 GPT-3.5 自主决定

# Step 2: 逐子问题推理
A_sub = []
for Q_i in [Q_1, ..., Q_n]:
    A_i = LLaVA_NeXT(F_final, Q_i)      # 独立推理
    A_sub.append(A_i)

# Step 3: 融合子答案生成最终答案
# 将子答案拼接为辅助 prompt segment
aux_text = concat(A_sub)
A_final = LLaVA_NeXT(F_final, aux_text, Q)
```

关键设计决策（消融实验验证）：
- **子答案 vs 子问题**：Table 8 显示将子问题（而非子答案）喂入模型反而降低 accuracy（50.4% vs 51.8% w/o decomposition），说明性能提升来自子答案提供的多样化中间信息，而非结构化思考过程
- **Prompt 设计**：移除任务背景解释降低 accuracy（53.2% vs 58.0%），移除"temporal and dynamic aspects"降低 accuracy（54.8% vs 58.0%），但改写措辞不影响（58.4% vs 58.0%），说明语义内容决定性能而非措辞
- **开放域 VideoQA 不适用**：对于简单问题（如 MSVD-QA 的 "What is a man sitting on?"），分解反而过度复杂化并降低 accuracy（72.4% vs 80.0% w/o decomposition），因此 D-CoDe 仅在 Multiple Choice VideoQA 上使用 Question Decomposition

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Question Decomposition 需要调用外部 LLM API（GPT-3.5-turbo-0125），因此引入了显著的推理延迟：+Question Decomposition 使延迟从 6.115 s/sample 增加到 37.395 s/sample（+511%）。轻量变体：使用更小的 CLIP（35% params）→ 35.466 s/sample；限制子问题数 = 5 → 26.273 s/sample, accuracy 56.0%；限制子问题数 = 7 → 33.704 s/sample, accuracy 57.8%。Question Decomposition 的优势场景是复杂、多步推理的问题（如 EgoSchema 的 schema 级理解、NExT-QA 的因果推理），对简单空间查询（如 MSVD-QA）效果负面。实现代码在 `Dcode.py` 的 `generate_subquestions()` 函数中，依赖 `OPENAI_API_KEY`。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition

## Perception Bottleneck (感知瓶颈)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception Bottleneck（感知瓶颈）是 D-CoDe 论文识别并命名的、在将图像预训练 VLM 扩展到视频领域时面临的核心挑战之一。它指：静态压缩策略（如均匀帧采样、空间平均池化）对所有内容等同处理，丢弃了在时间和空间维度上不均匀分布的关键视觉信息。具体表现为：(1) 时间维度——关键动作或事件可能集中在特定时间段，均匀采样可能完全跳过这些信息密集段；(2) 空间维度——平均池化对所有空间位置一视同仁，模糊了高信息量 token（物体边界、人脸、文本区域）和低信息量 token（纯色背景、模糊区域）的差异。论文通过 EgoSchema 5-frame 实验（Figure 2a）量化了这一瓶颈：uniform sampling + spatial average pooling 的 accuracy 显著低于无压缩 baseline，而 D-CoDe 的动态压缩不仅缩小了这一差距，甚至超越了 baseline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Perception Bottleneck 的产生机制：
```
# === 静态压缩（Baseline）===
# 时间维度: uniform frame sampling
frames = sample_uniform(video, N=5)   # 每 T/5 帧取一帧
# 问题: 若关键动作发生在第 0.6T 到 0.7T 之间，
# 而采样点在 0.0T, 0.2T, 0.4T, 0.6T, 0.8T，
# 则关键动作仅被 0.6T 帧部分捕获，0.8T 帧错过后续

# 空间维度: spatial average pooling
for each frame:
    tokens = VisualEnc(frame)          # (H/p × W/p, d)
    compressed = AvgPool2d(tokens)     # 所有位置等同压缩
# 问题: 人脸区域（高信息）和天空背景（低信息）被同等平均，
# 导致人脸关键特征被天空"稀释"

# === 动态压缩（D-CoDe）===
# 时间: 均匀覆盖 + 多样性补充
frames_uniform = sample_uniform(video, floor(0.85*N))
frames_supp = select_diverse_frames(video \ frames_uniform, N - floor(0.85*N))
# 基于 CLIP semantic dissimilarity，补充语义不同的关键帧

# 空间: salience pruning + similarity merging
salience = ||tokens||_2               # ℓ2 norm 作为重要性代理
tokens_kept = TopK(tokens, key=salience, k=floor(0.625*M))
tokens_merged = greedy_merge(tokens_kept, threshold=0.9)
# 保留高激活 token，合并语义冗余的相似 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Perception Bottleneck 是一个概念性术语（论文提出的问题定义），而非具体实现。D-CoDe 通过 Dynamic Compression（动态压缩）来解决这一瓶颈：时间维度用 supplementary frame selection（基于 CLIP 语义多样性的帧选择），空间维度用 salience-based pruning + cosine-similarity merging。消融实验（Table 4, EgoSchema, 15 frames）：Baseline（uniform+pooling）= 44.8% → +Dynamic Spatial Token Compression = 50.6%（+5.8%）→ +Dynamic Temporal Frame Selection = 51.8%（+1.2%）。两步分别验证了空间和时间动态压缩对缓解感知瓶颈的贡献。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition

## Token Overload (Token过载)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Overload（Token 过载）是 D-CoDe 论文识别并命名的另一个核心挑战。它指：视频输入产生的 visual tokens 数量远超静态图像（即使经过压缩），超过了图像预训练 VLM 的有效处理容量，导致模型无法全面理解这些 token 中的信息。具体表现为性能随 token 增加先提升后饱和（plateau）——多余的 token 不仅无益甚至可能引入干扰。论文通过 EgoSchema 10-frame 实验（Figure 2b）量化了这一效应：随着保留的 visual token 数量增加（通过不同 top-k activation retention ratio 控制），baseline（vanilla LLaVA-NeXT）的 accuracy 先上升后趋于平台，而 Question Decomposition 变体的 accuracy 持续增长且与 baseline 的差距不断拉大。Token Overload 的本质不是"token 太多算不动"，而是"模型无法从超量 token 中提取全部有效信息"——即模型的理解容量成为瓶颈，而非计算容量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token Overload 的量化机制：
```
# === Token Overload 现象（Figure 2b） ===
# 实验设置: EgoSchema, 10 input frames, LLaVA-NeXT 7B
# 变量: 不同 top-k retention ratio → 控制的 visual token 数量

retention_ratios = [0.1, 0.2, 0.3, ..., 1.0]
baseline_acc = [38.2, 40.1, 41.5, 42.0, 42.3, 42.3, 42.2, ...]  # 饱和
decomp_acc = [40.5, 43.2, 45.8, 47.5, 49.0, 50.2, 51.0, ...]     # 持续增长
# gap = decomp_acc - baseline_acc 随 token 增加不断扩大

# 原因分析:
# Baseline: LLaVA_NeXT(F_final, Q)
#   模型试图从大量 visual tokens 中一次性提取所有相关信息
#   → 注意力分散 → 超出模型理解容量 → 性能饱和
#
# D-CoDe: LLaVA_NeXT(F_final, A_sub_1, ..., A_sub_n, Q)
#   子问题引导模型每次关注视频的一个具体方面
#   → 注意力聚焦 → 多次 pass 覆盖全部语义 → 性能持续提升
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Token Overload 是一个概念性术语（问题定义）。D-CoDe 通过 Question Decomposition 来缓解：将复杂问题分解为聚焦子问题，引导模型每次关注视频的不同语义方面，从而在多个 pass 中"消化"超量 token，避免单次 pass 中的注意力分散。论文未讨论其他缓解 Token Overload 的方法（如增加 context length、使用 memory bank 等），因为这些方法通常需要训练或架构修改，而 D-CoDe 的目标是 training-free。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition

## Training-free Video-LLM (免训练视频大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-free Video-LLM 是一类将图像预训练的 VLM 直接扩展到视频理解的方法，其核心特点是**不做任何额外训练或微调**（与 Training-required Video-LLM 形成对比）。Training-required 方法通常需要在大规模视频数据集上微调视觉编码器、跨模态连接器或 LLM（如 Video-ChatGPT, Video-LLaVA, LLaVA-NeXT-Video 等），计算成本高。Training-free 方法利用图像和视频之间的结构相似性，通过设计推理时的压缩、采样、聚合策略来适配视频输入，同时保持预训练 VLM 的所有参数冻结。代表性方法包括：IG-VLM（构建 grid-view 图像）、FreeVA（帧级时间聚合）、SF-LLaVA（slow-fast 架构）、TS-LLaVA（thumbnail-and-sampling 策略）、D-CoDe（dynamic compression + question decomposition）。Training-free 方法的优势在于零额外训练成本、可插拔性（直接应用于不同的预训练 VLM），但通常需要仔细设计的压缩策略来平衡信息保留和 token 预算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Training-free Video-LLM 的通用 pipeline：
```
# === Training-free Video-LLM 推理流程 ===
# 输入: 视频 V = {I_1, ..., I_T}, 问题 Q
# 所有模型参数冻结（不更新）

# Step 1: 帧选择/压缩（training-free 的关键设计空间）
frames = frame_selection_strategy(V, max_frames=N)
# 策略变体:
#   - Uniform: 均匀间隔采样
#   - IG-VLM: 构造 grid-view 图像
#   - SF-LLaVA: slow (密集采样) + fast (稀疏采样) 双路径
#   - TS-LLaVA: thumbnail (全局缩略图) + sampling (均匀采样)
#   - D-CoDe: uniform + supplementary (基于语义多样性)

# Step 2: 帧编码（冻结的视觉编码器）
for frame in frames:
    tokens_frame = Frozen_ViT(frame)    # CLIP/SigLIP 等

# Step 3: Token 压缩/聚合（可选, training-free）
tokens_compressed = token_compression_strategy(all_tokens)
# 策略变体:
#   - Average Pooling: 空间平均
#   - D-CoDe: salience pruning + similarity merging

# Step 4: LLM 推理（冻结）
answer = Frozen_LLM(concat([tokens_compressed, text_emb(Q)]))
```

D-CoDe 在 training-free 方法中的定位（Table 2, EgoSchema）：
- IG-VLM: 35.8%
- SF-LLaVA: 47.2%
- TS-LLaVA: 50.2%
- D-CoDe: 58.0%（第一个超越所有 training-required 方法的 training-free 方法）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Training-free Video-LLM 的实现通常基于 HuggingFace Transformers，核心代码是帧采样器 + token 处理模块 + 冻结模型推理的 Python 脚本。D-CoDe 的实现代码在 `run_inference_multiple_choice_qa.py` 和 `run_inference_video_qa.py` 中，不涉及任何模型训练/微调代码。优势：(1) 可插拔——可直接应用于不同预训练 VLM；(2) 低成本——单卡 RTX A6000 即可运行；(3) 零数据需求——不需要视频训练数据。劣势：(1) 推理延迟通常高于 training-required 方法（Question Decomposition 引入 511% 额外延迟）；(2) 性能上限受限于基础 VLM 的能力；(3) 对频繁场景切换的视频适应性较差（D-CoDe 在 MSRVTT-QA 的频繁切换子集上从 64.2 降至 56.0%）。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition

## Curvature-Aware Scorer (CAS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curvature-Aware Scorer (CAS) 是 CurveStream 框架的核心感知模块，用于在无限长流视频中对每帧的语义转换强度进行实时评估。CAS 是一个 training-free 的评分器，使用冻结的轻量视觉编码器（DINOv2-small，~22M 参数，输出 384 维特征）提取每帧的全局特征表示 F_t ∈ R^D 并进行 L2 归一化，投影到单位超球面上。CAS 的核心创新在于同时融合**一阶运动强度**和**二阶几何曲率**来构造综合曲率分数 CS_t = M_t + λ·C_t：(1) Motion Variation M_t = 1 - cos(F_t, F_{t-1}) 衡量相邻帧之间的特征位移模长（一阶信息）；(2) Geometric Curvature C_t = 1 - cos(d1, d2) 衡量特征位移向量 d1 = F_{t-1} - F_{t-2} 与 d2 = F_t - F_{t-1} 之间的角度偏差（二阶信息）。C_t 在微分几何视角下严格等价于 1/2 ||T2 - T1||²（单位切向量变化的平方），即特征轨迹流形曲率的离散近似。这一几何特性使得 C_t 在恒速物理运动（如平滑相机平移/旋转）中自然趋近于 0，仅在特征演化方向发生突变时产生显著尖峰，实现了从低层次物理运动到高层次语义转换的数学解耦。λ 作为几何惩罚项的平衡系数（论文默认 λ=0.2），用于调节曲率分量在综合评分中的权重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CAS 在 CurveStream pipeline 中的位置：连续视频帧 I_t → DINOv2-small 编码 → L2 归一化 → CAS 计算 CS_t → 馈入 HVMM 进行记忆路由。
伪代码：
```
# 输入: t 时刻帧 I_t, 前两个时刻的特征 F_{t-1}, F_{t-2}
F_t = DINOv2_small(I_t)        # 冻结视觉编码器, shape: (D,)
F_t = F_t / ||F_t||_2          # L2归一化到单位超球面

# 一阶 Motion Variation
M_t = 1 - dot(F_t, F_{t-1})    # 余弦距离, F_t和F_{t-1}已归一化

# 二阶 Geometric Curvature
d1 = F_{t-1} - F_{t-2}         # shape: (D,)
d2 = F_t - F_{t-1}              # shape: (D,)
C_t = 1 - dot(d1,d2)/(||d1||*||d2||)  # 位移向量角度偏差

# 综合曲率分数 (λ=0.2)
CS_t = M_t + 0.2 * C_t          # scalar ∈ [0, 2.2]
```
CAS 的核心几何解释（论文 Appendix C）：将位移向量归一化为单位切向量 T1 = d1/||d1||, T2 = d2/||d2||，则 C_t = 1 - ⟨T1, T2⟩ = 1/2||T2 - T1||²。这表明 C_t 是切向量变化平方的一半，直接度量特征演化方向的变化率而非模长变化——这是区分"语义突变"（方向变）和"平滑运动"（模长大但方向不变）的数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAS 作为 training-free 模块，不需要任何训练。在 CurveStream 中，CAS 使用 DINOv2-small 作为特征提取器（论文默认使用该模型），DINOv2-small 以 ViT-S/14 架构在 142M 张无标签图像上自监督预训练，输出 384 维特征。CAS 的计算开销极低：每帧仅需一次 DINOv2-small 前向传播（~0.5G FLOPs）加上少量向量运算（dot product + L2 normalization = O(D) ≈ O(384)）。论文消融实验（Table III）验证了曲率度量的有效性：Uniform Sampling=69.04%, Cosine Similarity=73.28%, Optical Flow=46.54%, Pyramid Optical Flow=75.69%, Curvature=77.31%（训练无关方法中最优）。CAS 独立使用（无 HVMM）在 StreamingBench 上带来 +9.12% 的绝对提升（Table IX），在 OVOBench 上带来 +8.39% 的提升（Table X）。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

## Feature Manifold Curvature (Geometric Curvature C_t)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Feature Manifold Curvature（特征流形曲率）是 CurveStream 提出的几何度量 C_t，用于衡量连续视频帧在潜空间特征流形上的轨迹弯曲程度。该度量定义在视频帧经视觉编码器映射到的高维特征空间（单位超球面流形）上：将每帧 I_t 经冻结编码器编码为 F_t 并 L2 归一化后，连续时间步的帧在流形上形成一条离散参数化曲线。C_t 近似计算该曲线的二阶几何曲率（方向导数），定义为相邻时间步特征位移向量 d1 = F_{t-1} - F_{t-2} 和 d2 = F_t - F_{t-1} 之间的余弦距离：C_t = 1 - ⟨d1,d2⟩/(||d1||·||d2||)。在微分几何中等价于 C_t = 1/2 ||T2 - T1||²，其中 T1, T2 是归一化后的单位切向量（即瞬时演化方向）。C_t 的核心理论优势：(1) 免疫恒速运动噪声——平滑相机运动下特征匀速演化，T1≈T2 → C_t≈0；(2) 对语义突变正交敏感——当场景发生突变（新实体进入、镜头切换、动作边界），特征轨迹方向急剧偏转，T2 投射到近乎与 T1 正交的子空间 → C_t 急剧增大。这种"对方向敏感、对模长免疫"的特性使 C_t 成为理想的语义转换检测器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
C_t 在 CAS 内部的运动学建模（论文 Appendix C）：
```
# 运动学视角：特征位移向量 = 离散速度向量
d1 = F_{t-1} - F_{t-2}    # t-1 时刻的瞬时速度
d2 = F_t - F_{t-1}        # t 时刻的瞬时速度

# 归一化为单位切向量（方向）
T1 = d1 / ||d1||           # t-1 时刻演化方向
T2 = d2 / ||d2||           # t 时刻演化方向

# 几何曲率 = 切向量变化平方的一半
C_t = 1 - dot(T1, T2)      # = (1/2) * ||T2 - T1||²
```
典型场景对比：
- 恒速相机平推（背景整体位移）: d1 ≈ d2（模长接近，方向相同）→ T1 ≈ T2 → C_t ≈ 0 → 被 HVMM 归为 Discard 或 Blurred Memory
- 新物体突然出现（语义突变）: d1（背景运动方向）与 d2（包含新物体的特征位移）方向显著不同 → T1 ⊥ T2（近似正交）→ C_t → 1（尖峰）→ 被 HVMM 归为 Clear Memory

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
C_t 在 CurveStream 中仅需保存最近 3 帧的特征 F_t, F_{t-1}, F_{t-2}（滑动窗口），不需要全历史序列，使其适用于无限长流视频。计算复杂度和内存开销均为 O(D)，D = 384（DINOv2-small 输出维度）。C_t 与传统物理度量（cosine similarity, optical flow）的关键区别：cosine similarity 仅衡量模长变化（混淆语义突变与全局平移），optical flow 对像素噪声极度敏感（论文 Table III 中仅 46.54% accuracy），而 C_t 通过二阶方向微分天然解耦了运动幅度和方向变化。C_t 的鲁棒性验证：当 λ 在 [0.2, 1.0] 范围内变化时，accuracy 波动仅 3.33%（Table IV），证明曲率度量本身信号稳定。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

## Hierarchical Visual Memory Management (HVMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Visual Memory Management (HVMM) 是 CurveStream 的记忆调度模块，负责在固定 token 预算（N_max = 20 frames）下对 CAS 输出的曲率分数序列进行动态路由决策。HVMM 的核心机制包含两个子组件：(1) Online Manifold Distribution Estimation —— 使用 Exponential Moving Average (EMA) 在线更新曲率分数的瞬态分布参数：μ_t = γ·μ_{t-1} + (1-γ)·CS_t, σ_t² = γ·σ_{t-1}² + (1-γ)·(CS_t - μ_t)²，其中 γ ∈ (0,1) 为动量因子，控制历史观测窗口大小；(2) K-Sigma Dynamic Dual Thresholds —— 基于当前分布生成两个自适应阈值：g1 = μ_t + k1·σ_t（模糊记忆下界，k1=0.0 默认）和 g2 = μ_t + k2·σ_t（清晰记忆下界，k2=1.0 默认），k1 < k2。HVMM 的核心设计理念是将记忆管理建模为"在线分布感知"过程而非静态规则——在非平稳流视频中（如静止观察后突然剧烈奔跑），分布参数 (μ_t, σ_t²) 通过 EMA 平滑适应场景节奏变化，阈值随之动态平移，确保对加速/减速场景均能有效区分高价值和低价值帧。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HVMM 的在线更新与路由过程：
```
# 输入: 当前帧的 CS_t，历史分布参数 (μ_{t-1}, σ_{t-1}²)
# 参数: γ (EMA momentum), k1=0.0, k2=1.0, N_max=20

# Step 1: EMA 在线更新分布参数
μ_t = γ * μ_{t-1} + (1-γ) * CS_t
σ_t² = γ * σ_{t-1}² + (1-γ) * (CS_t - μ_t)²

# Step 2: 生成 K-Sigma 动态双阈值
g1 = μ_t + k1 * σ_t    # 模糊记忆下界 (k1=0.0)
g2 = μ_t + k2 * σ_t    # 清晰记忆下界 (k2=1.0)

# Step 3: 层级状态路由
if CS_t >= g2 or t == t_q:
    # 曲率尖峰 或 查询时刻帧
    s_t = Clear, r_t = High     # 保留原始分辨率
elif g1 <= CS_t < g2:
    # 中间过渡状态
    s_t = Blurred, r_t = Low    # 降采样 224×224
else:  # CS_t < g1
    # 低信息冗余
    s_t = Discard               # 直接丢弃

# Step 4: 更新记忆队列
M_t = M_{t-1}.append(I_t with (s_t, r_t))

# Step 5: FIFO 驱逐
if len(M_t) > N_max:
    evict oldest token  # 严格 FIFO 出队
```
HVMM 的 EMA 机制使得阈值能在非平稳场景中自适应用：当场景突然加速（curvature 整体抬高），μ_t 和 σ_t 通过 EMA 跟随上升，阈值 g2 也相应提高——防止正常高曲率帧被过度保留挤占 memory bank。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HVMM 在 CurveStream 中的默认超参数：γ（EMA momentum，论文未明确具体值）、k1=0.0、k2=1.0、N_max=20。HVMM 独立使用（无 CAS 动态评分，退化为均匀交替分配 Clear 和 Blurred Memory）在 StreamingBench 上带来 +9.76% 的绝对提升（Table IX），在 OVOBench 上带来 +4.69%（Table X）——表明即使无智能感知，二值层级记忆结构本身也比单层 FIFO 更有效。HVMM 的 Clear Memory 保留比例自适应维持在 ~50%（图 3b），这在 accuracy 和 token 成本之间达到最优 tradeoff（比 100% Clear 减少 ~40% token 开销同时保持或提升 accuracy）。HVMM 的 K-Sigma 超参数对性能鲁棒：当 k1, k2 在不同组合下变化时（图 4），accuracy 保持高度稳定，验证了动态阈值相对于静态阈值的优越性。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

## K-Sigma Dynamic Threshold

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
K-Sigma Dynamic Threshold 是 CurveStream 中 HVMM 模块使用的在线自适应双阈值机制，用于在非平稳流视频中动态区分高语义价值帧和低信息冗余帧。不同于任何先验静态阈值（在场景节奏剧烈变化时容易导致记忆库崩溃或关键信息丢失），K-Sigma 阈值基于曲率分数的在线分布参数实时计算：g1 = μ_t + k1·σ_t 和 g2 = μ_t + k2·σ_t，其中 (μ_t, σ_t²) 通过 EMA 递归更新。k1 和 k2 是两个固定的乘数超参数（k1 < k2），控制阈值在分布中的位置。K-Sigma 的"自校准"特性体现在：当视频从静止转为剧烈运动时，曲率分数整体均值和方差上升，阈值自动提高以避免将正常的高运动帧误判为语义突变；当视频回归平静时，均值和方差下降，阈值相应降低以保持对微弱但关键变化的敏感度。这种分布感知设计使得 CurveStream 无需针对不同视频手动调整阈值，在 OVOBench 的多样化场景下（包括自我中心、虚拟环境、监控等）均保持鲁棒。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
K-Sigma 阈值生成与 EMA 分布更新的一体化过程：
```
# 每帧到达时，先更新分布参数，再生成阈值
μ_t = γ * μ_{t-1} + (1-γ) * CS_t        # 运行均值
σ_t² = γ * σ_{t-1}² + (1-γ) * (CS_t - μ_t)²  # 运行方差
σ_t = sqrt(σ_t²)

# K-Sigma 双阈值 (k1 < k2)
g1 = μ_t + k1 * σ_t    # 下界: k1=0.0 → g1=μ_t
g2 = μ_t + k2 * σ_t    # 上界: k2=1.0 → g2=μ_t + σ_t
```
场景自适应示例：
- 长时间静止观察: CS_t 持续低值 → μ_t ≈ 0.1, σ_t ≈ 0.05 → g2 = 0.1 + 1.0·0.05 = 0.15 → 即使小幅运动（CS=0.2）也能触发 Clear Memory
- 剧烈跑动: CS_t 持续高值且波动大 → μ_t ≈ 0.8, σ_t ≈ 0.3 → g2 = 0.8 + 0.3 = 1.1 → 仅当出现方向突变（CS > 1.1）才触发 Clear Memory，一般摆动归为 Blurred/Discard
- 场景过渡（从静止到跑动）: μ_t 和 σ_t² 通过 EMA 平滑跟随变化，无突变——防止新场景的早期帧被全部归为 Clear Memory

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
K-Sigma 在 CurveStream 中默认 k1=0.0, k2=1.0。k1=0.0 意味着模糊记忆下界等于当前均值 μ_t——将约 50% 的帧归为 Discard（低于均值）、约 34% 的帧归为 Blurred Memory（低于 +1σ）、约 16% 的帧归为 Clear Memory（高于 +1σ），在标准正态假设下大致对应 50%/34%/16% 的分配比例。论文图 4 的消融显示 K-Sigma 参数对 accuracy 和帧采样率均高度鲁棒：在不同 (k1, k2) 组合下，动态机制能自动平衡 Clear 和 Blurred Memory 分配，无需繁琐的手动调参。与静态阈值的对比：若使用固定 g2=0.5，在静止场景中几乎所有帧都被归为 Discard（阈值过高），在动态场景中几乎所有帧都被归为 Clear Memory（耗尽 memory bank）。K-Sigma 的分布感知设计天然避免了这两种极端。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

## Clear/Blurred/Discard Memory States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clear/Blurred/Discard Memory States 是 CurveStream 中 HVMM 模块定义的三种层级化视觉记忆状态，对应帧的不同保留策略和分辨率。这是一个三级的 resolution-aware 记忆路由方案：(1) Clear Memory —— 最高优先级记忆，对应 CS_t ≥ g2 或触发查询的帧 t_q。帧以原始高分辨率（base MLLM 的动态高分辨率策略）保留，存储精确的空间细节以支持后续细粒度视觉推理（如 OCR、属性识别、小物体定位）。Clear Memory 帧在记忆队列中作为"语义锚点"。(2) Blurred Memory —— 中间优先级记忆，对应 g1 ≤ CS_t < g2。帧被降采样到固定的 224×224 分辨率以大幅压缩 token 开销（论文中 TRANSITION_SIZE=224），保留必要的时序因果关联和动作连贯性，同时以极低成本维持连续帧之间的平滑过渡。(3) Discard —— 最低优先级，对应 CS_t < g1，帧直接被丢弃，不占用任何 memory bank 空间。三种状态的配置设计使 CurveStream 能在严格 N_max=20 的常值内存约束下，同时保持关键语义细节的高保真（Clear Memory ~50%）、动作连贯性的低分辨率覆盖（Blurred Memory ~50%）和冗余静态背景的零成本丢弃（Discard ~50% 总帧数）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种状态的路由逻辑：
```
if CS_t >= g2 or t == t_q:
    state = Clear Memory
    resolution = High  # base model 原生动态高分辨率
elif g1 <= CS_t < g2:
    state = Blurred Memory
    resolution = Low   # 固定 224×224 (TRANSITION_SIZE)
else:  # CS_t < g1
    state = Discard
    resolution = None  # 不编码

# 仅 Clear 和 Blurred 状态存入 M_t
M_t.append(I_t encoded at resolution)
```
Clear Memory 保留比例的消融实验（图 3b）：
- 100% Clear (所有帧高分辨率): 耗尽 memory bank，触发 catastrophic forgetting → accuracy 下降
- 0% Clear (全 Blurred): 丢失关键空间细节 → accuracy 急剧下降
- ~50% Clear (自适应 hybrid): accuracy 最优，token 成本降低 ~40%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 CurveStream 中，Clear Memory 帧使用 base MLLM（如 Qwen2.5-VL-7B）的动态高分辨率编码策略——即根据图像内容自动调整分辨率和 token 数量。Blurred Memory 帧统一降采样到 224×224 后编码，每帧产生的 visual token 数量显著减少（约减少 60-80%）。当 |M_t| > N_max 时，无论帧处于何种状态，均按严格 FIFO 顺序驱逐最旧 token——这种简单的驱逐策略避免了为不同类型 token 设计复杂的 eviction priority。与其他方法的对比：(1) uniform sampling 对所有帧一视同仁（无状态区分）→ 关键帧可能被逐出；(2) HERMES 使用 KV cache 被动逐出 → 无主动信息评估；(3) FreshMem 使用频率/空间域混合记忆 → 无分辨率区分。CurveStream 的 Clear/Blurred/Discard 三层级是首个将"分辨率感知"和"语义感知"同时融入记忆状态设计的方法。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

## ECO (Episodic COmpressor)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ECO (Episodic COmpressor) 是 HERMES 长视频理解框架的核心组件之一，受人类 episodic memory（情景记忆）认知机制启发。它是一种在线、training-free 的视频帧特征压缩算法：维护一个容量上限为 E 的 memory buffer M（存储 episode prototypes），按 window 顺序处理视频帧特征 $\mathcal{W}_k$。当新 window 到达时：(1) 若 buffer 有空间，直接追加；(2) 若 buffer 溢出，将 buffer 和新 window 临时拼接为 A，迭代执行：计算 A 中所有帧对之间的 cosine similarity，找最相似帧对 $(i^*, j^*)$，合并 $A_{i^*} = (A_{i^*} + A_{j^*})/2$（元素级平均），删除 $A_{j^*}$，直到 $\|A\| \le E$。ECO 的核心创新在于 **global similarity-based merge**：不同于 MA-LMM 仅合并相邻（temporally adjacent）帧，ECO 比较 memory buffer 内所有帧之间的全局相似度，使得无论两帧在视频中相距多远，只要内容相似即可聚合到同一 episode。位置编码（PE）在 ECO 前施加于帧特征以注入 temporal locality，防止跨时间段的乱序合并且保留时序连贯性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ECO 精确伪代码（论文 Algorithm 1）：
```
A = concat(M, W_k)                              # 临时拼接 buffer 和新 window
while ||A|| > E:                                # 超过最大 episode 数
    (i*, j*) = argmax_{i≠j} (A_i · A_j) / (||A_i|| · ||A_j||)  # 全局 cosine similarity
    A_i* = (A_i* + A_j*) / 2                    # 元素级平均合并
    A = A \ A_j*                                # 删除被合并的帧
M = A                                           # 更新 buffer
```
ECO 在 HERMES pipeline 中的位置：Video → ViT-G/14 Window Encoder → ECO（维护 episode memory）→ Episodic Q-Former（cross-attention to episodes）。默认参数：N=100 frames, window=10, E=20 episodes。PE 消融：移除 PE 后 MovieChat-1k accuracy 从 78.6 降至 77.3。ECO 隐式捕获事件频率：频繁出现的事件自然在更多帧中出现，因此更可能被合并/保留为强化原型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ECO 是 training-free 设计——在任意预训练 ViT 特征上直接操作，不需要额外训练。作为 plugin 模块插入现有 VLM：(1) 替换 MA-LMM 的 memory bank → accuracy +3.4%, latency -43%（Table 5）；(2) 插入 LongVA → latency -30%, GPU memory -46%（Table 3）；(3) 插入 LLaVA-OneVision → latency -35%, accuracy +0.67%（Table 4）。ECO 的全局 merge 策略使其相比 FIFO（77.1%）和 Random（76.9%）策略提升约 1.5%（Table 6: ECO=78.6%）。22 FPS on V100（接近实时），仅需 100 帧 vs MA-LMM 2048 帧。开源：https://joslefaure.github.io/assets/html/hermes.html。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding

## SeTR (Semantics reTRiever)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SeTR (Semantics reTRiever) 是 HERMES 框架中与 ECO 互补的语义信息提取模块，受人类 semantic memory（语义记忆）认知机制启发。SeTR 的目标是从全视频帧特征中提取高层次的语义线索，而非保留每帧的时序细节。算法流程：(1) 归一化全部帧特征 $F \in \mathbb{R}^{B \times N \times T \times C}$；(2) 以 stride=k 将 N 帧分为两组：保留组 K（每 k 帧取 1 帧，得 N/k 帧）和压缩组 K̄（剩余 N-N/k 帧）；(3) 对每个 K̄ 中的帧，计算其与所有 K 帧之间的 dot-product similarity 分数；(4) 将每个 K̄ 帧按元素级平均合并到最相似的 K 帧中。最终保留 $\frac{N}{k}$ 帧作为 semantic representations。在 HERMES 中默认 keep_ratio=0.2（k=5），即从 100 帧压缩到 20 帧语义代表。SeTR 区别于 ToMe（在 ViT 内部层间合并 token）：SeTR 在帧级别（而非 token 级别）操作，保留语义最丰富、最具代表性的帧而非简单的 token 合并。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SeTR 伪代码：
```
F = concat(W_1, ..., W_{N/w})                     # 全部 window features: (B, N, T, C)
F = normalize(F)                                  # 归一化
K_indices = [0, k, 2k, ...]                       # 保留组: N/k 帧
K_bar_indices = rest                              # 压缩组: N - N/k 帧
F_K = F[:, K_indices, :, :]
F_Kbar = F[:, K_bar_indices, :, :]
for each frame f in F_Kbar:
    sim_scores = dot_product(f, F_K)              # 与每个保留帧的相似度
    j* = argmax(sim_scores)
    F_K[j*] = (F_K[j*] + f) / 2                  # 合并到最相似的保留帧
F_prime = F_K                                     # (B, N/k, T, C)
# 后续: Hierarchical Q-Former (fQFormer → vQFormer)
```
SeTR 后的 Hierarchical Q-Former：fQFormer 独立增强每帧语义 → Frame-to-Sequence Adapter (Linear) → vQFormer 全局聚合 → $Q_{sem}$。消融实验（Table 7）：移除 SeTR 导致 accuracy 下降 5%（78.6 → 73.3）；MaxPool/AvgPool 替代 SeTR 分别降至 70.4/73.3；K-Means 聚类压缩为 75.7，均低于 SeTR 的 78.6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SeTR 的 keep_ratio 是唯一超参数：20% 在 MovieChat-1k 和 Breakfast 上均为最优（Figure 5），验证 HERMES 对超参数鲁棒。SeTR 可独立作为 plugin 插入现有 VLM：(1) MA-LMM + SeTR → accuracy +3.8%, latency 仅 +1.5%（Table 5）；(2) LongVA + SeTR → accuracy +0.45%, latency -27%（Table 3）；(3) LLaVA-OneVision + SeTR → accuracy +1.04%, latency -33%（Table 4）。SeTR 与 ECO 互补：ECO 提供 episode-level temporal detail，SeTR 提供 global semantic themes——将两者 concat 后送入 LLM 实现双流理解。training-free，零额外训练。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding

## Query-based Vision Token Compression（基于可学习Query的Vision Token压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query-based Vision Token Compression 是 LMM 中一种通过可学习 query 向量与 vision token 进行 cross-attention 来压缩视觉信息的技术。LLaVA-Mini (Zhang et al., 2025) 将其推向极致——将 CLIP ViT-L 输出的 576 个 vision token 压缩到仅 1 个 token（压缩率 0.17%），同时通过 modality pre-fusion 弥补压缩带来的信息损失，性能与 LLaVA-v1.5 可比。核心机制：(1) 引入 C×C 个可学习压缩 query Q^v（默认 C=1，即 1 个 query）；(2) 对 query 和原始 vision token 施加 2D sinusoidal positional encoding 保留空间位置信息；(3) Q^v 通过 cross-attention 与全部 vision token 交互，产生注意力矩阵 A [C^2, N^2]；(4) 压缩输出 Ĥ^v = A · H^v [C^2, d_h]。相比 average pooling，query-based 压缩可自适应关注关键区域（如 OCR 文字、价格标签），仅增加 2.42G FLOPs 但带来显著的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Query-based Compression 的计算过程：
```
# 输入: H_v = [576, 4096] (576 vision tokens, d_h=4096 Vicuna-7B)
Q_v = learnable_compression_queries    # [C^2, 4096], C=1 → [1, 4096]
pos = 2D_Sinusoidal_PE()               # 2D 正弦位置编码

# Cross-attention: queries attend vision tokens
Q_with_pos = Q_v + pos(Q_v)            # [C^2, 4096]
K_with_pos = H_v + pos(H_v)            # [576, 4096]
A = Softmax(Q_with_pos @ K_with_pos.T) # [C^2, 576] 注意力矩阵

# 加权聚合压缩
H_v_compressed = A @ H_v               # [C^2, 4096]
```
关键设计：(1) 2D sinusoidal PE 保留 patch 的 2D 空间位置信息，这对图像理解至关重要——2D PE 比 1D PE 能更好地保留相邻 patch 的空间关系。(2) Cross-attention 无 causal mask——所有 query 平等 attend 所有 vision token。消融实验（Table 8）：1 token 时 query-based 77.6/60.9/65.6 (VQA-v2/GQA/MMB) vs average pooling 76.1/59.8/64.0，FLOPs 增加仅 2.42G（1.96T 总 FLOPs 的 0.12%）。可视化（Figure 12）：压缩 attention 在关键信息集中区域（文字、产品标签）聚焦明显，在主体不明确时分布更分散。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上，compression queries 为 nn.Parameter，cross-attention 复用标准 MultiHeadAttention（Q=queries, K=V=vision_tokens）。查询数 C^2 为超参数：标准分辨率 C=1 → 1 token；HD 高分辨率 C=8 → 64 tokens。在 LLaVA-Mini 的两阶段训练中，compression module 在 Stage 2 引入并端到端训练。C 值可配置以在效率-精度间 trade-off（Table 7：1 token VQA-v2 77.6, 64 tokens 78.5, 576 tokens 80.0）。相关方法对比：与 BLIP-2 Q-Former 的 32 个固定 query 不同，LLaVA-Mini 将 query 数压到 1 并配合 pre-fusion 补偿；与 PruMerge（基于 token 相似度合并）和 VoCo-LLaMA（用 LLM 压缩）也不同。开源：https://github.com/ictnlp/LLaVA-Mini。

涉及论文标题：
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token

## Q-Former (Querying Transformer) in Vision-Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Q-Former (Querying Transformer) 是 BLIP-2 (Li et al., ICML 2023) 的核心创新组件，作为连接冻结的视觉编码器与冻结的 LLM 之间的轻量级信息瓶颈（information bottleneck）。架构由两个共享 self-attention 层的 Transformer 子模块组成：(1) Image Transformer —— 通过 cross-attention 与冻结的图像编码器输出交互，从视觉特征中提取与文本语义最相关的信息；(2) Text Transformer —— 同时作为文本编码器和文本解码器使用。核心机制：32 个可学习的 query 向量（每向量 768 维）通过 self-attention 相互联系，通过 cross-attention 与冻结图像特征交互，将全图的大量 visual tokens（如 ViT 的 257 tokens）压缩为仅 32 个最具信息量的 query 输出。两阶段预训练：(1) 视觉-语言表示学习（ITC+ITM+ITG 三个联合损失）；(2) 视觉到语言的生成学习（Q-Former 输出通过线性投影馈入冻结 LLM）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准 Q-Former 前向过程：
```
Q = learnable_query_vectors                   # (32, 768)
I_emb = frozen_ViT(image)                     # (257, 1024) ViT output
# Self-Attention on queries
Q = SelfAttention(Q, causal_mask=False)
# Cross-Attention: queries attend to image features
Q = CrossAttention(Q=Q, KV=I_emb)             # (32, 768)
# 输出: 32 个信息密集的视觉 features
output = Linear(Q)                            # 投影到 LLM embedding 维度
```

HERMES 对 Q-Former 的两个扩展：
(1) **Episodic Q-Former**：在标准 Q-Former 中插入 ECO 模块——self-attention on queries → cross-attention to visual episodes M → ECO_q（在 query 空间应用与 ECO 相同的 cosine-similarity iterative merging，将跨 window 的 queries 也聚合为 query episodes）。公式：$Q = ECO_q(CA(SA(Q_0), M))$。
(2) **Hierarchical Q-Former**：两级设计——Frame Q-Former (fQFormer) 独立增强每帧语义 → Frame-to-Sequence Adapter (Linear) → Video Q-Former (vQFormer) 全局聚合所有帧信息。公式：$Q_{sem} = vQFormer(Linear(fQFormer(F')))$。消融实验（Table 8）：HQFormer=95.2% > vQFormer=94.1% > fQFormer=93.2% on Breakfast。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Q-Former 在 HERMES 中的实现：(1) 权重初始化自 InstructBLIP（已包含视觉-语言对齐）；(2) 参数在 fully-supervised 设置下可微调；(3) 作为 visual features → LLM embedding 的桥梁，输出与 SeTR semantic features concat 后馈入 Vicuna-7B；(4) Q-Former 的 32 query 设计在视频场景下通过 Episodic Q-Former 扩展为 episode-level query 组织。Q-Former 的替代方案：LLaVA 使用简单线性层（而非 Q-Former）连接 ViT 和 LLM——更简单但缺乏 Q-Former 的信息瓶颈和跨模态对齐能力。开源：BLIP-2 (https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models)。

TDC 论文对 Q-Former 的扩展使用：用于视频帧的 temporal dynamic context 压缩。具体做法：(1) 对每个视频场景 segment，首帧完整保留作为 static frame；(2) 对首帧 visual tokens 做 AvgPool 得到 K=16 个 query tokens（而非 learnable queries）；(3) 后续每帧的 visual+audio tokens 拼接后与 query tokens 做 cross-attention，同时注入 instruction text F_s 使压缩自适应于用户问题；(4) Q-Former 的 query output 作为该帧的压缩表示，形成 temporal dynamic context F_TDC。消融实验表明 AvgPool queries 优于 learned queries，且 text instruction 可提升各 benchmark 性能（MVBench +0.4, MLVU +0.2/+1.6, VideoMME +1.2）。Q-Former 由预训练 BERT 初始化。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## Episodic + Semantic Dual-Stream Long-Form Video Understanding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Episodic + Semantic Dual-Stream Video Understanding 是 HERMES 提出的受人类认知双记忆系统启发（episodic memory + semantic memory）的长视频理解框架。人类认知中，episodic memory 负责回忆特定事件/经历（"昨天下午和母亲在电话里争论了什么"），semantic memory 负责存储一般性知识/概念（"家庭关系中常见的冲突模式"）。HERMES 将这一认知框架映射到视频理解中：(1) **Episodic Stream (ECO)** —— 以 window 为粒度在线处理视频帧，通过 global cosine-similarity merging 将帧压缩为最多 E 个 episode prototypes。ECO 保存时序细节和特定事件，类比"记得电影中具体发生了什么"。Episodic Q-Former 在 query 空间也进行 episode-level 聚合；(2) **Semantic Stream (SeTR)** —— 通过 stride-based 帧分组 + similarity merging 将 N 帧压缩为 N/k 帧语义代表。SeTR 提取跨整个视频的高层次主题和概念，类比"总结电影在讲什么"。Hierarchical Q-Former 两级（frame→video）增强语义表达；(3) **Fusion** —— 将两条流的输出 concat 后经 linear projection 馈入冻结 LLM (Vicuna-7B) 生成回答。双流互补：episodic stream 回答 "what happened when"，semantic stream 回答 "what is this about overall"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HERMES 完整 dual-stream pipeline：
```
输入: long video V, instruction I
# === Episodic Stream (ECO + Episodic Q-Former) ===
windows = ViT_G_14(sample_frames(V, N=100), window=10)
M = []                                             # episode memory
for W_k in windows:
    M = ECO(M, W_k, max_episodes=20)               # online压缩
Q_0 = learned_queries                              # (32, 768)
Q_ep = Episodic_QFormer(Q_0, M)                    # episode-aware queries

# === Semantic Stream (SeTR + Hierarchical Q-Former) ===
F = concat(windows)                                # all features: (100, T, C)
F_prime = SeTR(F, keep_ratio=0.2)                  # semantic compression: (20, T, C)
Q_sem = Hierarchical_QFormer(F_prime)              # (32, 768)

# === Fusion + LLM Generation ===
U = Linear(concat([Q_ep, Q_sem]))                  # (64, LLM_dim)
answer = Vicuna_7B.generate(U, I)
```
关键设计：(1) 两条流可独立使用（作为 plugin 插入其他 VLM）；(2) 两条流均为 training-free（仅 Q-Former 和 adapter 可选微调）；(3) 仅需 100 帧（vs MA-LMM 2048 帧），22 FPS on V100。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
认知科学基础（论文 Section H.9）：基于 Tulving (1972) 的 episodic/semantic memory distinction、hippocampus 在 episodic memory consolidation 中的作用、neocortex 在 semantic knowledge 存储中的作用、event segmentation theory (Zacks et al., 2007)、gist extraction (Oliva, 2005)。在长视频理解中的优势：episodic stream 擅长捕捉角色关系变化（LVU Relationship +15.4% over S5）、semantic stream 擅长理解整体主题和场景分类。双流设计使 HERMES 在四个 benchmark 上达到 SOTA：MovieChat-1k +14.9%（zero-shot）、LVU +7.3%、Breakfast +2.2%、COIN +0.3%。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding

## Query-Aware Frame Selection (查询感知帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query-Aware Frame Selection 是一类用于长视频理解的帧采样策略，核心思想是根据输入查询（query）的内容自适应地选择最相关的视频帧，而非使用统一的固定采样策略（如 uniform sampling）。传统 uniform sampling 以固定间隔抽取帧，对所有查询一视同仁；query-aware 方法通过相关性评估机制（如 CLIPScore、LMM 评分、object detection）判断每帧对当前查询的价值，仅保留高相关度帧。DIG 论文的关键贡献在于识别出 query-aware selection 并非对所有查询类型都必要——论文定义了 global query（需要全视频理解）和 localized query（针对特定时间段），并证明：(1) 对 global query，uniform sampling 已足够有效；(2) 对 localized query，query-aware selection 才真正带来显著收益。同时，DIG 指出基于 CLIPScore 的 query-aware 方法（如 Q-Frame、AKS）在复杂推理场景中不可靠，因 CLIPScore 仅依赖浅层特征匹配，无法捕捉需要上下文推理和世界知识的查询意图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Query-Aware Frame Selection 通用 pipeline
# 输入: 视频 V, 查询 Q, 目标帧数 N

# Step 1: 候选帧生成
candidates = get_candidates(V)
# 方式: uniform sampling (Q-Frame, AKS), 或 CAFS (DIG)

# Step 2: 帧-查询相关性评估
for frame in candidates:
    relevance[frame] = score(frame, Q)
# score(): CLIPScore → cosine_sim(CLIP_I(f), CLIP_T(Q))
#          LMM Reward → LMM CoT推理 + {"reward": 0-100}

# Step 3: 基于分数的帧选择
selected = select_by_scores(candidates, relevance, N)
# 方式: Top-K, 或 iterative reward-guided (DIG, 无参数)

# Step 4: LMM 推理
answer = LMM(selected, Q)
```
三种代表性方法的对比（基于 DIG Table 1）：Q-Frame (CLIPScore+Top-K) 在 >32 frames 时退化至低于 uniform；AKS (BLIP+adaptive) 在 >64 frames 时退化；DIG (LMM Reward+iterative) 在 8-256 frames 上持续提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query-Aware Frame Selection 在 DIG 中作为 training-free pipeline 实现：对 localized query 启动 CAFS+LMM Reward+Video Refinement，对 global query 回退到 uniform sampling。查询感知的核心计算开销来自相关性评估——DIG 使用 vLLM 加速 LMM 推理（reward assignment 占总选择时间 ~70%）。开源：https://github.com/Jialuo-Li/DIG。其他 query-aware 方法包括 BOLT（LLM agent 多轮搜索）、T*（temporal search + object detection）、MDP3（list-wise frame selection）等。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## Content-Adaptive Frame Selection (CAFS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Content-Adaptive Frame Selection (CAFS) 是 DIG 提出的基于视频语义内容的自适应代表性帧选择方法，用于替代 uniform sampling 或 FPS-based sampling 作为候选帧生成策略。CAFS 利用 DINOv2 自监督视觉特征捕捉视频中的高层语义变化（如场景切换、物体出现/消失），通过检测语义边界自适应地选择每个稳定段落的代表帧（r-frames）。关键创新：(1) 基于内容密度而非固定间隔选择帧——信息密集段产生更多 r-frames，静态冗余段产生更少；(2) 使用 topographic prominence >0.1 过滤噪声峰值，排除微小帧间波动；(3) 选择段落中点帧（而非峰值帧本身）作为代表——因峰值帧位于语义边界（混合两个场景），中点帧最能代表稳定语义内容。量化指标：LoC (Localized Coverage) 评估 r-frame 局部代表性，GIC (Global Coverage) 评估 r-frame 全局覆盖性。CAFS 的非线性信息缩放特性（Figure 10）：r-frame 数量不随视频时长线性增长，证明视频语义信息密度分布不均。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CAFS 完整流程 (DIG Algorithm 1)
# 输入: 2 fps 采样 M 帧 {f_{I_i}}_{i=1}^M
# 输出: r-frame indices R_idx

# Step 1: DINOv2 特征提取
for i in 1..M:
    V_i = DINOv2(f_{I_i})              # 768-d (ViT-B)
# Step 2: 逐帧语义距离
for i in 1..M-1:
    d_i = 1 - cosine_sim(V_i, V_{i+1})  # scalar ∈ [0,2]
# Step 3: 峰值检测 (local maxima)
P = {i | d_{i-1} < d_i and d_i > d_{i+1}}
# Step 4: Topographic Prominence 过滤
P_valid = {j ∈ P | prominence(d_j) > 0.1}
# prominence: d_j - max(l_min, r_min)
#   l_min = 向左搜索到更高峰 min distance
#   r_min = 向右搜索到更高峰 min distance
# Step 5: 选相邻峰值中点
R_idx = {(I_p1 + I_p2)/2 | p1,p2 consecutive in P_valid}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAFS 在 DIG 中作为 localized query pipeline 的第一步执行，使用冻结的 DINOv2 ViT-B 作为特征提取器。计算开销较低（MLVU: 25.9 min, LVB: 20.8 min on 8×A100）——无需大模型推理。与 uniform sampling 对比消融（Figure 7）：在 DIG 中用 uniform frames 替代 CAFS r-frames 后所有 benchmark 均下降且差距随帧数增加而扩大。r-frame 统计特性（Appendix E.2）：0-10 min 视频平均 47.9 r-frames，10-20 min 视频平均 226.4 r-frames（~4.22s/帧），实现高压缩比（~99% 的帧被压缩为代表帧）。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## Query Typology: Global Query vs Localized Query

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query Typology 是 DIG 提出的视频问答查询分类框架，将查询分为：(1) Global Query (GQ) —— 需要理解和综合整个视频内容，不含特定时空指代词（如 "What title best summarizes this video?", "What is the primary focus?"），回答需要 holistic understanding；(2) Localized Query (LQ) —— 可通过关注特定时间段来回答，包含具体 referents（实体、动作、时间标记，如 "What color is the man's bike at 3:15?"）。这一分类的核心价值在于指导帧选择策略：DIG 证明 uniform sampling 在 GQ 上已足够有效且高效（Figure 5 右侧），而对 LQ 则需专门的 keyframe selection（Figure 5 左侧）。分类方法：LLM（Qwen3-Next-80B-A3B）通过 CoT 4 步推理（理解意图 → 推断视频风格 → 识别 referents → 综合判断）输出 isGlobal: true/false。LQ accuracy >90%，GQ accuracy 38-75%（误分类代价低，走错 branch 最多增加计算开销）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Query Classification + Branch Routing
classification = LLM_classify(Q)  # CoT prompt
if classification.isGlobal:
    frames = uniform_sample(V, N)  # 高效路径
else:
    r_frames = CAFS(V)             # 精准路径
    rewards = LMM_reward(r_frames, Q)
    refined = video_refinement(V, r_frames, rewards)
    frames = uniform_sample(refined, N)
answer = LMM(frames, Q)
```
判断标准：Global = 缺乏具体 referent 或虽有但需 holistic understanding；Localized = 有具体 referent 且可通过关注相关片段回答。效率增益（Table 11）：Query Identification 使 MLVU 节省 13.3% 总时间，VideoMME 节省 19.9%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query Typology 分类由任意具备推理能力的 LLM 执行，DIG 使用 Qwen3-Next-80B-A3B-Instruct。分类 prompt 采用 CoT 策略（论文 Figure 11）。benchmark 的 ground truth 标注：MLVU 通过任务结构映射（holistic→GQ, single/multi-detail→LQ），LVB 全为 LQ（referring reasoning 设计），VideoMME 通过人工标注 majority vote。DG 的误分类影响：GQ 误分为 LQ → 多耗计算但 accuracy 接近持平（Figure 5）；LQ 误分为 GQ → 回到 uniform sampling，可能丢失关键信息。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## LMM-based Reward Assignment (大模型奖励评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMM-based Reward Assignment 是 DIG 提出的帧-查询相关性评估方法，直接使用 LMM 本身对视频帧进行 relevance scoring，替代传统的 CLIPScore 或 object detection。核心流程：将候选 r-frame 和查询送入 LMM → CoT 推理 → 输出二维评分：(a) 帧对回答查询的直接有用性，(b) 帧是否暗示相邻帧包含补充信息 → reward ∈ [0, 100]。与传统 CLIPScore 的关键区别：(1) 语义深度——LMM 可理解复杂推理逻辑，非仅表面特征匹配；(2) 世界知识——利用预训练常识识别 CLIP 无法捕获的隐含关联；(3) 上下文感知——二维评分使 LMM 能评估帧的"指示价值"。DIG Table 2 证明 LMM reward (Qwen2.5-VL-7B/32B) 在所有 frame 配置下一致优于 CLIPScore，且更强的 LMM (32B vs 7B) 提供更好的 reward quality。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LMM Reward Assignment 流程
for each r_frame in r_frames:
    prompt = f"""
    Frame: <{r_frame}>; Query: <{Q}>
    1. Describe the frame, focusing on relevant elements.
    2. Assign reward 0-100 based on:
       (a) Direct usefulness for answering
       (b) Whether adjacent frames may supplement
    Output: {{"description": str, "reward": int}}
    """
    response = vLLM_inference(LMM_rewarder, r_frame, prompt)
    rewards.append(response["reward"])
```
与 CLIPScore 的对比（Table 2, 128 frames, LVB）：
- CLIPScore: 61.0% → LMM 7B: 63.1% → LMM 32B: 65.2%
- Gain 来自 LMM 的语义推理和二维评分设计

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMM Reward Assignment 使用 vLLM 后端加速，且 rewarder LMM 可与推理 LMM 解耦（如用 Qwen2.5-VL-32B 做 reward，Qwen2.5-VL-7B 做最终推理）。计算开销：reward assignment 是 DIG 中最耗时阶段（占总选择时间 ~70%）。局限性：(1) 额外推理成本；(2) 长视频 r-frames 多时耗时显著；(3) reward 准确性受限于 rewarder LMM 能力。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## Iterative Reward-Guided Selection (迭代奖励引导选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Iterative Reward-Guided Selection 是 DIG 提出的无参数帧选择方法，替代需要预设 K 的 Top-K selection。核心思想：给定 r-frames 的奖励集 {R_j}，迭代式 mean-thresholding 自动确定"显著高于平均"的相关帧：(1) 计算当前奖励均值 R̄；(2) 低于均值的置零：R'_j = max(R_j - R̄, 0)；(3) 检查正值集合是否与前一轮一致，一致则终止。优势：(1) 无预设参数——不需指定 K；(2) 自适应——不同查询的 reward 分布不同时自然产生不同选择数量；(3) 单调收敛保证——每次迭代至少移除低于均值的元素。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Iterative Reward-Guided Selection
R = {R_1, ..., R_{N-1}}; prev = None
while True:
    R_mean = mean(R)
    R_new = [max(r - R_mean, 0) for r in R]
    positives = {j | R_new[j] > 0}
    if positives == prev: break
    prev = positives; R = R_new
return S = positives  # 最终选中的 r-frame indices
```
数值示例：rewards [85,60,45,30,20,10,5] → Round 1 mean=36.4 → positives [0,1,2] → Round 2 mean=26.9 → positives [0] → Round 3 all zero → 收敛，最终选 r-frame[0]（reward=85 的帧）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 DIG 中，Iterative Selection 作为 Video Refinement 的第一步执行，之后对被选 r-frames 进行窗口合并构建 refined video。通常 2-5 轮迭代收敛。对比 Top-K：Top-K 需预设 K（5? 10? 20?），optimal K 随视频/查询/分布变化，Iterative 方法自动适应无需调参。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## CLIPScore

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CLIPScore 是基于 CLIP 模型的跨模态相似度评估指标（Hessel et al., 2021）：图像经 CLIP 视觉编码器得 v，文本得 t，CLIPScore = cosine_sim(v, t)。在视频理解中广泛用于 query-aware frame selection——对候选帧计算与查询文本的 CLIPScore，选 top-K 高分帧。DIG 揭示其关键局限：(1) 表面特征匹配——无法捕捉多步推理或世界知识；(2) 视觉偏差——倾向给含常见物体的帧高分；(3) 缺乏上下文推理——无法评估帧间关联。DIG Table 2 证明 LMM-based reward 在所有 frame count 和 benchmark 上一致优于 CLIPScore。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIPScore in query-aware selection
v_i = CLIP_vision(f_i)  # L2 normalized
t = CLIP_text(Q)        # L2 normalized
CLIPScore_i = dot(v_i, t)
selected = TopK(candidates, key=CLIPScore, k=N)
```
典型失效（DIG 分析）："Why did the character leave?" → CLIPScore 给含"人物+房间"的帧高分，但答案取决于时序因果推理而非单帧内容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用 OpenAI CLIP 或 OpenCLIP 预训练权重，单帧评分仅需 ms 级别。DIG 将 CLIPScore 作为 LMM Reward 的对比 baseline，证明在复杂推理上 LMM Reward 显著优于 CLIPScore（LVB 128 frames: +4.2%）。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

## DINOv2 for Video Frame Representation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DINOv2 是 Meta AI 的自监督视觉预训练模型（Oquab et al., 2023），基于 ViT 架构在 1.42 亿张无标签图像上通过自蒸馏训练。输出高质量的通用视觉特征（无需微调），提供 ViT-S/B/L/g 多种规模。在 DIG 的 CAFS 中，DINOv2 用于逐帧提取 global feature 计算相邻帧语义距离以检测场景边界。选择 DINOv2 的理由：(1) 自监督训练（无需标注）→ 泛化能力强；(2) 语义鲁棒性——对光照/视角不敏感，对内容变化敏感（适合场景切变检测）；(3) 计算高效——单帧特征仅需一次 ViT 前向。DIG 使用 DINOv2 ViT-B (768-d features) 的 [CLS] token 或 average pooled features 作为 frame-level 表示。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DINOv2 in CAFS
for frame in video:
    tokens = DINOv2_ViT(frame)        # patch tokens: (N, 768)
    V_i = tokens.mean(dim=0)          # global feature, 768-d
    V_i = V_i / ||V_i||_2             # L2 normalize
d_i = 1 - dot(V_i, V_{i+1})           # cosine distance for scene boundary
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
加载：`dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` 或通过 HuggingFace。在 DIG 中冻结使用，不做微调。计算开销：对 2fps 视频每帧 1 次 ViT-B 前向，CAFS 总耗时 20-30 min (8×A100)。DINOv2 也被 CurveStream 用于 CAS 模块的 curvature 特征提取。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## KTO (Kahneman-Tversky Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KTO (Kahneman-Tversky Optimization) 是一种基于前景理论（Prospect Theory）的 LLM 对齐方法，由 Ethayarajh et al. (2024, ICML 2024) 提出。KTO 的核心创新在于：仅需单样本偏好标签（"chosen" 或 "rejected"，如 👍/👎 二值信号），而不像 DPO 那样需要 pairwise preference data（chosen vs rejected 成对数据）。KTO 直接最大化生成的效用（utility），借鉴 Kahneman & Tversky (1992) 的前景理论——特别是损失厌恶（loss aversion）——人类对损失的敏感度高于等量收益。KTO 属于 HALOs（Human-Aware Losses）损失函数家族。在 EVA 中，KTO 被用于三阶段训练的第二阶段（SFT → KTO → GRPO），作用是：SFT 训练后的模型学会了 tool-call 格式但仍有典型失败模式（如视觉证据不足时猜测、欠采样、过采样），KTO 通过 63% chosen + 37% rejected 的数据让模型学习 fine-grained 策略偏好，使其偏好有效策略而避免已知失败模式。相比 DPO，KTO 不需要多轮对话共享回合的前提（这在 EVA 的多轮交互设置中会截断策略），更适合 EVA 的 multi-turn 场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KTO loss 的核心计算：
```
# KTO Loss (简化形式)
# 输入: (x, y, label) where label ∈ {chosen, rejected}
# π_θ: 当前策略, π_ref: 参考策略 (SFT模型)

# 计算 log-ratio
r_θ(x,y) = log(π_θ(y|x) / π_ref(y|x))

# KTO loss per sample
if label == chosen:
    L_KTO = -λ_chosen * σ(β * r_θ(x,y) - z_ref)
else:  # rejected
    L_KTO = -λ_rejected * σ(z_ref - β * r_θ(x,y))

# σ: sigmoid, z_ref: 参考点(人类对收益/损失的不对称感知)
# β: 控制对 reference model 偏离的惩罚强度
# λ: chosen/rejected 样本的权重超参数
```

在 EVA 中使用时：
- 63% chosen（高质量成功轨迹）+ 37% rejected（SFT 构建过程中收集的错误轨迹）
- chosen data: LLM-as-Judge 筛选推理过程有足够 visual tokens 且正确回答的轨迹
- rejected data: LLM-as-Judge 筛选 visual tokens 不足但仍强行生成答案的轨迹（guessing 模式）+ 重新采样的高质量成功轨迹
- β=0.1, lr=2e-6, 1 epoch

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KTO 在 HuggingFace TRL 库中有标准实现（`KTOTrainer`）。核心优势：(1) 不需要成对偏好数据，可以使用真实 chat logs (👍/👎)；(2) 适合 continual production fine-tuning；(3) 比 DPO 在非成对设置下更灵活。在 EVA 中，KTO 作为 GRPO 之前的"纠错阶段"，通过纠正已知 bad cases 来提升 GRPO 在线优化的收敛性、鲁棒性和稳定性。局限性：在成对偏好数据设置下 DPO 可能优于 KTO，β 超参数调优很关键。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

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

## Bayesian Surprise (贝叶斯惊奇)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bayesian Surprise 是 Itti & Baldi (2005, NIPS) 提出的一种基于信息论的注意力量化理论。其核心定义：Surprise = 新观测数据引入后，观测者对世界模型的信念分布从先验到后验的信息增益，即 KL 散度 D_KL(P_posterior || P_prior)。当新数据迫使信念发生剧烈变化时，KL 散度大，surprise 高；当新数据与先验一致时，KL 散度小，surprise 低。原论文在视觉显著性预测中证明 Bayesian Surprise 是人类注意力最强的已知吸引因子（72% 的 human gaze shifts 指向比均值更 surprising 的区域，多人一致性时为 84%）。SPIKE 论文首次将 Bayesian Surprise 引入 Video-LLM 推理：将模型的"信念"表示为对文字化假设（textual hypotheses，如 "the man will continue walking"）的概率分布，先验 P_prior 从历史文本摘要 H_t + 前序帧窗口 W_t 计算，后验 P_post 加入当前观察帧 O_t 后计算，surprise score S_t = D_KL(P_post || P_prior)（实际使用 JSD 替代 KL 以得 [0,1] 范围），用于指导后续的 surprise-weighted frame sampling。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Bayesian Surprise 在 SPIKE 中的计算流程
# 输入: 视频帧 X_{1:T}, prior_window_size W=4, hypotheses_N=3
# 输出: 每个时间步 t 的 scalar surprise score S_t ∈ [0,1]

# Step 1: 生成信念假设 (通过 Video-LLM nucleus sampling)
H_t = summarize(X_{t-C:t-W-1})              # 历史文本摘要 (BART-Large-CNN)
W_t = X_{t-W:t-1}                           # 前序帧窗口
O_t = X_t                                   # 当前观察帧

B_t = VideoLLM.generate("predict next frame", H_t, W_t, temperature, top_p)
      # B_t = {b_{t,1}: "the man walks away", 
      #         b_{t,2}: "the man trips and falls",
      #         b_{t,3}: "the man stops to look"}

# Step 2: 计算先验分布 P_prior (仅基于历史+前序帧，不包含当前帧)
for b in B_t:
    NLL_prior = -log P_M(b | H_t, W_t)       # Video-LLM 给出的负对数似然
P_prior = softmax(-[NLL_prior_i] / τ)       # τ 为温度参数

# Step 3: 计算后验分布 P_post (包含当前观察帧 O_t)
for b in B_t:
    NLL_post = -log P_M(b | H_t, W_t, O_t)  # 加入当前帧后重新评估
P_post = softmax(-[NLL_post_i] / τ)

# Step 4: Bayesian Surprise = JSD(P_post, P_prior)  (用 JSD 替代 KL)
M = 0.5 * (P_post + P_prior)
S_t = 0.5 * D_KL(P_post || M) + 0.5 * D_KL(P_prior || M)
# S_t ∈ [0, 1], 经 log_2 归一化
# S_t 大 = 当前帧 O_t 显著改变了模型对视频事件的理解
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Itti & Baldi 原实现使用 72 个空间滤波器，维护 432,000 个概率分布逐帧更新（计算量极大，2005 年硬件上 500 帧需数小时）。SPIKE 论文创新性地用 Video-LLM 的 token-level NLL 替代空间滤波器，通过文字化假设将 Bayesian Surprise 转化为语言模型可计算的量：不需要贝叶斯推断显式更新参数，只需 Video-LLM 在两个上下文（有无当前帧）下对同一假设文本做两次 forward pass 取 NLL 差值。这使 Bayesian Surprise 的计算开销降至 O(F·N)（F=帧预算，N=假设数，N=3），与推理时间 scaling 方法可比。应用场景：视频异常检测、surprise 定位、自适应帧采样、机器人异常监控等。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

## Belief Tracking (信念追踪)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Belief Tracking（信念追踪）是认知科学中 Bayesian Theory of Mind (ToM) 框架的核心机制：智能体持续维护和更新对世界状态、其他智能体意图、或未来事件预测的内部模型（信念），当新证据出现时通过贝叶斯更新修正信念。在 NLP/AI 领域，Belief Tracking 传统上用于对话系统（追踪用户意图/槽位）和 Theory of Mind 推理（追踪角色心理状态）。SPIKE 论文将其扩展到 Video-LLM：信念被显式表示为可解释的文字假设 B_t = {b_{t,1}, ..., b_{t,N}}（如 "the delivery person will hand over the package"），每个假设包含先验概率 P_prior 和后验概率 P_post，形成完整的时间追踪链 {(B_1, S_1), (B_2, S_2), ..., (B_T, S_T)}。信念更新通过 Bayesian Surprise（KL/JSD）量化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Belief Tracking 在 SPIKE 中的实现
# 核心: 维护一个随时间演化的信念假设集合 + 历史文本摘要

# 初始化
H_t = ""                                    # 滚动文本摘要 (rolling memory)
B_history = []                              # 信念轨迹

for t in range(1, T+1):
    # 1. 从 Video-LLM 生成 N 个对未来事件的文字假设
    B_t = VideoLLM.generate(
        prompt="predict what will happen next",
        memory=H_t,                         # 历史事件摘要
        prior_frames=X_{t-W:t-1},           # 前 W=4 帧
        sampling="nucleus",                 # top_p nucleus sampling
        N=3                                 # 生成 3 个假设
    )

    # 2. 计算每个假设的 prior/posterior 概率
    O_t = X_t                               # 当前观察帧
    for b in B_t:
        NLL_prior[i] = -VideoLLM.log_prob(b | H_t, X_{t-W:t-1})
        NLL_post[i]  = -VideoLLM.log_prob(b | H_t, X_{t-W:t-1}, O_t)
    
    P_prior = softmax(-NLL_prior / τ)
    P_post  = softmax(-NLL_post / τ)

    # 3. Surprise = 信念分布变化量
    S_t = JSD(P_post, P_prior)

    # 4. 更新历史摘要 (追加当前帧描述，压缩到 ~200 词以内)
    event_desc = VideoLLM.caption(O_t, H_t, X_{t-W:t-1})
    H_t = BART_Large_CNN.summarize(H_t + event_desc)   # 滚动压缩

    # 5. 记录信念轨迹（可解释、可回溯）
    B_history.append((B_t, P_prior, P_post, S_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SPIKE 中，Belief Tracking 通过以下组件实现：(1) Video-LLM（Qwen2.5-VL-7B）作为假设生成器——给定 H_t（文本内存）和 W_t（视觉前序帧），经 nucleus sampling 生成 N=3 个短假设（8-10 词）；(2) BART-Large-CNN 作为历史摘要压缩器——维持 ~200 词以内的滚动文本摘要，避免 prompt 过长；(3) 两次 forward pass（有无 O_t）获取 NLL 差值计算 surprise。这一机制的创新在于：传统 Video-LLM 无信念演化概念，将视频视为 "bag of frames"；SPIKE 赋予 Video-LLM 人类式的"预期→观察→更新"循环。未来可扩展至实时流处理、人机交互中的预期管理、异常行为预警。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

## Surprise-Weighted Frame Sampling (惊奇加权帧采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Surprise-Weighted Frame Sampling 是 SPIKE 提出的一种 query-agnostic（不依赖查询）的自适应视频帧选择策略，用于替代 Video-LLM 的标准 uniform sampling。核心思想：将固定的帧预算 F 按各时间段的 Bayesian Surprise 得分比例重新分配——高 surprise 段获得更多帧（甚至被多次采样），低 surprise（routine）段获得较少帧。采样概率 p_i = softmax(S_i / τ_s)，其中 τ_s=0.7 控制分布集中度（小 τ_s 集中于 surprise 峰值，大 τ_s 趋于均匀）。这一方法的关键洞察是：surprising 事件恰好是人类叙事理解的关键信息点，在帧预算有限时优先采样这些帧能显著提升下游任务性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Surprise-Weighted Frame Sampling
# 输入: 视频 X_{1:T}, 帧预算 F, surprise scores S_1..S_K
# 输出: 选中的 F 帧索引

# Step 1: 对视频均匀锚定 K 个采样段 (K ≤ F)
segments = uniform_sample_timesteps(T, K)

# Step 2: 每段用 SPIKE 计算 surprise score
for t in segments:
    S[t] = compute_bayesian_surprise(X, t)  # 见 Bayesian Surprise 条目

# Step 3: 将 surprise 转为采样概率 (τ_s 控制集中度)
τ_s = 0.7                                   # 论文实验值
p = softmax([S[t] / τ_s for t in segments])
# 如果所有 S_i 相等（完全无 surprise），则 p_i = 1/K (回退到 uniform)

# Step 4: 按概率采样 F 帧 (有放回，高 surprise 段可多次被选)
selected_frames = []
for f in range(F):
    seg = multinomial_sample(segments, p)    # 按 p 概率选段
    frame_idx = uniform_int_within(seg)      # 段内等概率选帧
    selected_frames.append(frame_idx)

# Step 5: 选中的帧送入 Video-LLM 进行下游任务
result = VideoLLM.infer(selected_frames, query)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SPIKE 的加权采样作为 Video-LLM 的 sampling layer 即插即用替代——不修改模型架构，仅改变帧选择策略。实现细节：(1) 时间锚点 K 也视为帧预算的一部分（K ≤ F），帧预算按视频时长按比例分配（≤1 分钟视频 8 帧基础预算，更长视频每增加 1 分钟预算翻倍）；(2) τ_s 可在推理时调节，实现从 "聚焦 surprise"（τ_s→0）到 "近似 uniform"（τ_s→∞）的连续控制；(3) 复杂度 O(F·N)，与推理时间 scaling 开销可比。与 SBD（shot boundary detection）方法（RGB Histogram、ECR、Optical Flow、Katna）的关键区别在于：SBD 依赖原始视觉变化（对相机运动/场景切换敏感），SPIKE 的 surprise score 基于语义理解（通过 Video-LLM 的先验/后验概率变化），更能对齐人类感知的关键时刻。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

## LLM-Match (LLM 匹配评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-Match 是一种基于 LLM Judge 的文本相似度评估指标，用于自动评价生成式输出（如视频 caption）与 ground truth 的语义匹配程度。不同于 n-gram 重叠指标（ROUGE、BLEU）或嵌入相似度，LLM-Match 使用一个独立的 LLM（Judge）根据精心设计的评分指令对生成结果进行 0.0-1.0 分的语义相似度评分：0.0-0.3（差，关键细节缺失）、0.4-0.6（中等，部分细节匹配）、0.7-0.9（好，大部分关键细节匹配）、1.0（完美，所有关键细节准确）。在 SPIKE-RL 中，LLM-Match 的评分被直接用作 GRPO 训练中的 **reward signal**。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LLM-Match 评分流程
# Judge LLM: Olmo-7B-hf (或类似 LLM)
# 输入: 模型生成的 caption c, ground truth caption c_gt
# 输出: scalar reward R ∈ [0.0, 1.0]

def llm_match(c_pred, c_gt, judge_model):
    prompt = f"""
    Rate how closely the content of the prediction matches the content of 
    the reference description in terms of meaning and how well it captures 
    important details regarding events in the video. Ignore the difference 
    in length. Score 0.0-1.0 where:

    0.0-0.3: Poor match (key details in the reference are missing)
    0.4-0.6: Moderate match (a few key details are captured)
    0.7-0.9: Good match (most key details are present)
    1.0: Perfect match (all key details accurately captured)

    Output only the numerical score.

    Reference: {c_gt}
    Response: {c_pred}
    Score:
    """
    score = judge_model.generate(prompt, max_tokens=10)
    return float(score)

# SPIKE-RL 训练中: R = llm_match(caption_rollout, caption_ground_truth)
# 在 GRPO group 内 Z-score 归一化: A = (R - μ_R) / σ_R
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SPIKE-RL 使用 Olmo-7B-hf 作为 LLM-Match Judge，而非使用更大的模型（如 GPT-4o）以降低训练成本。LLM-Match 在 2024-2025 年的 Video-LLM 评估中广泛使用（OpenEQA, FunQA 等），逐渐替代 ROUGE/BLEU 等 n-gram 指标。核心优势：(1) 关注语义匹配而非表面形式，不受句式、长度影响；(2) 可作为 RL 训练的密集 reward signal（连续值 0-1，而非 binary 正确/错误）；(3) 与人类判断的高相关性使其适合作为自动评估的代理。局限性：(1) Judge LLM 本身的 bias（可能偏好特定措辞风格）；(2) 表面上的"高匹配"不保证事实准确性；(3) 在某些类型误差中与人类判断存在系统性偏差。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

## Planning-before-Perception (先规划后感知)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Planning-before-Perception 是 EVA 提出的视频 Agent 核心范式，与传统的 perception-first（先感知后推理）相对。在 perception-first 范式下，MLLM 先接收均匀采样帧或完整视频作为视觉输入，然后基于这些固定视觉信息进行推理或 tool call——这使得模型被动消费可能无关的视觉 token，且早期视觉噪音可能误导后续规划。Planning-before-Perception 翻转这一流程：agent 在初始状态仅接收 textual query（无视觉输入），先基于 query 进行文本推理生成 explicit plan（明确要观察什么、何时观察、如何观察），再通过 frame_select tool 有针对性地获取视觉信息。通过迭代 summary-plan-action-reflection 循环逐步完善感知。这一范式使 MLLM 从"被动视频识别器"进化为"主动自适应自主 agentic watcher"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Planning-before-Perception 执行流程
s_0 = {q, h=[], F=[]}  # 初始状态：仅 query，无视觉信息

for round in 1..max_rounds:
    # Step 1: 基于已有信息进行 Planning
    # 无需先看视频，从 query 推理需要什么信息
    plan = MLLM.plan(q, h, F)
    # plan = {
    #   "hypothesis": "需要先获取视频全貌",
    #   "strategy": "low_res_global_scan",
    #   "estimated_action": {start:0, end:T, nframes:10, resize:0.1}
    # }
    
    # Step 2: 执行 Action（选择性获取视觉信息）
    new_frames = frame_select(V, plan.action)
    F = F ∪ new_frames
    
    # Step 3: Summary + Reflection
    summary = MLLM.summarize(new_frames)
    sufficient = MLLM.reflect(q, F, summary)
    if sufficient: break

answer = MLLM.answer(q, h, F)
```

Planning-before-Perception 相比 perception-first 的优势：
1. 避免视觉误导：uniform frames 可能包含不相关/噪音内容误导 planner
2. 节省 visual tokens：仅获取 query 真正需要的视觉信息
3. 主动感知而非被动观察：agent 显式决定需要什么、如何获取、选择性交互

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 EVA 中通过三阶段训练实现：(1) SFT stage: 用 teacher MLLM (Qwen2.5-VL-72B) 构造 planning-before-perception 格式的训练数据（Summary → Planning → Action → Reflection），冷启动训练 agent 的 tool-call 和推理格式；(2) KTO stage: 纠正 planning 策略中的典型错误（如计划获取不足 visual tokens 但仍强行回答）；(3) GRPO stage: 在线优化 exploration-exploitation 平衡，让 agent 学会根据 query 自适应调整 planning 策略。EVA 的 frame_select tool 提供 start_time/end_time/nframes/resize 四参数灵活控制。使用 vLLM 部署推理。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

## Summary-Plan-Action-Reflection Loop (摘要-规划-行动-反思循环)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Summary-Plan-Action-Reflection 是 EVA 提出的迭代视频理解推理循环范式，受 Zhang et al. (2025) 的 Agent Learning via Early Experience 启发。每个循环包含四个阶段：(1) Summary —— MLLM 对当前返回的帧生成详细内容描述，显式推动模型关注返回的视觉证据并更好理解 tool 参数和输出之间的关系；(2) Planning —— 基于当前信息（query + 历史 + summary）推理潜在 actions，估算每个 action 的 token cost 和 expected outcome，从大 action space 中选择最优策略；(3) Action —— 生成具体的 frame_select tool call JSON（start_time, end_time, nframes, resize）；(4) Reflection —— 评估当前视觉信息是否充足，若不足则生成下一步策略，若充足则终止循环并生成最终答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EVA 的 SFT 数据实例格式精确对应这四个阶段：
```
Round 1:
  Summary: "The low-resolution frames show various gameplay scenes involving
            characters and text references..."
  Planning: "To pinpoint the exact moment, I will select a segment around
             frames showing scoring attempts. The frame at [03:24] shows a
             character near a goal post. Focusing on [03:20]-[04:10] with
             higher resolution makes sense."
  Action: {"tool": "frame_select", "arguments": {
           "start_time": 200, "end_time": 250, 
           "nframes": 100, "resize": 0.4}}
  Reflection: "The increased resolution provides clearer visuals...
               Based on the gameplay analyzed so far, the trigger is evident.
               No further analysis is necessary."

Round 2:
  Planning: (evaluates that evidence is sufficient)
  Action: (no tool call — generates final answer)
  Answer: "The GOAAAAAL!! animation is triggered when the ball..."
```

循环终止条件：Reflection 阶段判断视觉信息已充足 → 直接生成答案（不再 tool call）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 EVA 中通过 Multi-Agent Data Pipeline 构造训练数据：Executor agent 分析 context 并评估 actions → Reflective Thinker 审计 tool call 参数合理性（检查 fps 是否 >1、visual budget 是否太小等规则）→ 成功轨迹存入 Experience Bank 供未来检索引导 Executor。各阶段对最终性能的贡献通过 SFT→KTO→GRPO 消融实验间接验证：SFT 学习格式但低效（多帧多轮低 accuracy）→ KTO 减少帧数和轮数提升 accuracy → GRPO 增加轮数但更精准分配 token。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

## Data-Enhanced GRPO (数据增强 GRPO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Data-Enhanced GRPO 是 EVA 对标准 GRPO 的改进。标准 GRPO 依赖静态训练数据集，模型仅在固定 query-video pairs 上通过有限 epoch 迭代学习——当模型发现自身在某一能力维度（如计数）薄弱时，只能从有限的失败 query 和视频中学习改进。Data-Enhanced GRPO 引入动态数据增强机制：在 GRPO 训练若干步后，收集当前 policy 的 failure cases，以这些 failures 作为 in-context examples 提供给 teacher MLLM (Qwen2.5-VL-72B)，让 teacher 为 HD-VILA 中的新视频生成新的 open-ended QA pairs（条件于 failure examples），然后将新 QA pairs 合并到训练集中继续训练。此机制不断扩展训练数据的多样性，帮助 agent 在更广泛的挑战中学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Data-Enhanced GRPO Training Loop
EVA_RL = {9.6k OE + 1.1k MCQ}  # 初始 RL 数据集

for grpo_step in range(total_steps):
    # 1. 标准 GRPO 训练一步
    batch = sample(EVA_RL, bs=64)
    for each (q, v) in batch:
        responses = π_θ.sample(q, v, n=8)  # 8 rollouts
        rewards = [calc_reward(r) for r in responses]
        advantages = (rewards - mean(rewards)) / std(rewards)
        update π_θ with GRPO loss
    
    # 2. 每 N 步：收集 failures + 数据增强
    if grpo_step % N == 0:
        # 收集 KTO 训练后模型的 failure cases
        failures = collect_failures(π_θ, EVA_RL)
        # Teacher MLLM 以 failures 为 in-context examples
        # 为 HD-VILA 新视频生成 open-ended QA pairs
        new_QA = teacher_MLLM(
            video=HD_VILA[v_new],
            in_context_examples=failures,
            prompt="Generate open-ended QA with concise answers"
        )
        # 增强训练集
        EVA_RL = EVA_RL ∪ new_QA
        # 在新增强的数据集上继续训练
```

选择 open-ended QA 而非 MCQ 生成的原因：(1) 缓解 reward hacking（答案猜测）；(2) 对 teacher model 更高效——设计平衡的 MCQ 选项常引入非预期的信息线索和额外复杂度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Data-Enhanced GRPO 依赖于：(1) 一个强大的 teacher MLLM (Qwen2.5-VL-72B) 用于生成高质量新数据；(2) 未标注的新视频源 (HD-VILA) 提供视频多样性；(3) in-context learning 机制让 teacher 针对模型当前的薄弱环节生成针对性训练数据。这种方法的优势在于持续扩展训练数据分布，打破标准 GRPO 的静态数据限制。实现复杂度较高，需要 teacher model inference 与 RL training 交替进行。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

## Diffusion Transformer (DiT) for Video Generation（视频生成扩散Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diffusion Transformer (DiT) 是一种将 Transformer 架构应用于扩散模型去噪过程的生成模型。在视频生成中，DiT 将视频 latent token（经 VAE 压缩的视频表示）与文本条件 token 拼接，通过多层 Transformer block（含 self-attention、cross-attention、FFN）迭代去噪，生成目标视频。与早期基于 U-Net 的视频扩散模型（如 Stable Video Diffusion）相比，DiT 的 Transformer 架构具有更好的可扩展性和生成质量。Sora 的出现证明 DiT 架构可实现高质量视频生成。EasyAnimate 使用 MMDiT 变体（对文本和视频两种模态使用独立的 FFN 和 FC 结构），结合 3D RoPE 位置编码和 rectified flow 采样，48 层 Transformer，支持 text-to-video、image-to-video、inpaint、control 等多种生成模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EasyAnimate DiT 的去噪 pipeline 如下：

```
# 输入: z_T ~ N(0,I) (视频 latent noise), c (文本条件), timesteps T..1
# 输出: z_0 (去噪后的视频 latent)

def dit_denoising_step(z_t, t, c):
    # 1. 位置编码: 对 latent token 施加 3D RoPE
    pos_enc = compute_3d_rope(z_t.shape, h_channels)

    # 2. 文本和视频 token 拼接后进入 MMDiT block
    tokens = concat([c_text, z_t + pos_enc])

    # 3. MMDiT: 两种模态共用 self-attention，但各自独立的 FFN
    for layer in 1..N_layers:
        if layer in window_layers:
            attn_out = multidirectional_swa(tokens)
        else:
            attn_out = full_3d_attention(tokens)
        tokens_video = tokens_video + ffn_video(attn_out.video_part)
        tokens_text  = tokens_text  + ffn_text(attn_out.text_part)

    # 4. 预测速度场 v(t) — rectified flow 的 ODE 向量场
    v_pred = output_proj(tokens_video)
    return v_pred
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DiT 视频模型通常通过以下方式实现：(1) VAE 压缩 —— 使用 3D causal VAE 在空间和时间维度压缩视频（如 8x 空间压缩 + 4x 时间压缩），latent 维度远小于像素空间；(2) 多阶段训练 —— 从低分辨率到高分辨率渐进训练（PixArt 策略），如 256^2 x 49f -> 512^2 x 49f -> 1024^2 x 49f；(3) 文本编码 —— 使用 Qwen2-VL-7B（EasyAnimate）、T5-XXL（CogVideoX）、CLIP+T5（SD3）等提取文本特征；(4) 联合训练 —— 图像和视频数据联合训练（如 34M video + 3M image pairs）。推理时使用 classifier-free guidance 或 rectified flow 快速采样。主要框架包括 EasyAnimate（开源，Apache 2.0）、CogVideoX、HunyuanVideo、OpenSora 等。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
- LongLive__Real-time_Interactive_Long_Video_Generation

## Hybrid Windows Attention / Multidirectional Sliding Window Attention（混合窗口注意力/多方向滑动窗口注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Windows Attention 是 EasyAnimate 提出的视频 DiT 注意力机制，通过交替使用 3D full attention 和多方向滑动窗口注意力（Multidirectional Sliding Window Attention），在降低计算复杂度的同时维持视频生成质量。核心组件 Multidirectional Sliding Window Attention 将注意力头分为 6 组，每组沿不同的 3D 维度方向重排 token 序列后执行滑动窗口注意力：(1) fhw（frame->height->width，默认顺序），(2) fwh（frame->width->height），(3) hfw（height->frame->width），(4) hwf（height->width->frame），(5) wfh（width->frame->height），(6) whf（width->height->frame）。仅需一次 FlashAttention 调用（而非 spatial-temporal decoupled attention 的多次），计算复杂度从 O(N^2) 降至 O(N x W)，其中 W 为窗口大小。在 48 层 DiT 中，中间层（12-36）使用 window attention，浅层（1-12）和深层（36-48）使用 full attention，兼顾全局上下文和计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def multidirectional_sliding_window_attention(Q, K, V, num_heads, window_size):
    # Step 1: 头分组 — 6组，每组 num_heads/6 个注意力头
    head_groups_Q = chunk(Q, 6, dim='head')
    head_groups_K = chunk(K, 6, dim='head')
    head_groups_V = chunk(V, 6, dim='head')

    # Step 2: 各方向 token 重排
    dirs = ['fhw', 'fwh', 'hfw', 'hwf', 'wfh', 'whf']
    for i, direction in enumerate(dirs):
        head_groups_Q[i] = rearrange(head_groups_Q[i], direction)
        head_groups_K[i] = rearrange(head_groups_K[i], direction)
        head_groups_V[i] = rearrange(head_groups_V[i], direction)

    # Step 3: 合并后单次 FlashAttention 调用
    Q = concat(head_groups_Q, dim='head')
    K = concat(head_groups_K, dim='head')
    V = concat(head_groups_V, dim='head')
    output = FlashAttention(Q, K, V,
        window_size_left=window_size // 2,
        window_size_right=window_size // 2)

    # Step 4: 恢复原始 token 顺序
    head_groups_out = chunk(output, 6, dim='head')
    for i, direction in enumerate(dirs):
        head_groups_out[i] = rearrange(head_groups_out[i],
                                       inverse(direction))
    return concat(head_groups_out, dim='head')
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hybrid Windows Attention 基于 FlashAttention 的 sliding_window 参数实现。FlashAttention 原生支持 local/sliding window 模式，通过设置 window_size_left/right 自动限制每个 query token 只关注窗口内的 key token。多方向的关键技巧在于通过 token 重排（rearrange）而非修改 attention 计算本身来模拟不同方向的滑动窗口。窗口大小 ablation 显示 H x W（空间分辨率对应的 latent 尺寸）是最优平衡点（FVD=352.3，推理 21.32s/iter）。在 1024 分辨率下，Hybrid Windows Attention 训练加速 22.39%，推理加速 25.53%。该设计可推广到其他需要 3D 注意力且序列较长的场景（如 point cloud、medical 3D volumes）。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## 3D Causal VAE（三维因果变分自编码器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
3D Causal VAE 是一种对视频同时在空间和时间维度进行压缩的变分自编码器，其"因果性"（causal property）体现在：编码/解码当前帧时只能依赖当前及之前的帧信息，不能看到未来帧。这与 image-based VAE（仅空间压缩逐帧处理）和 non-causal 3D VAE（可访问所有帧）形成对比。压缩率通常为 8x 空间 + 4x 或 8x 时间，将原始视频从 (T, H, W, 3) 压缩到 (T/k_t, H/k_s, W/k_s, latent_dim)。因果性的关键优势：在解码长视频时，可以缓存前帧的 latent state，连接当前帧进行增量解码，极大降低显存使用，支持生成长视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 编码器
def encode_3d_causal_vae(video_frames):
    """video_frames: (T, H, W, 3) -> z: (T//4, H//8, W//8, latent_dim)"""
    for layer in encoder_layers:
        x = causal_3d_conv(x)  # temporal padding only on past side
        x = group_norm(x)
        x = silu(x)
    mu, logvar = head_mu(x), head_logvar(x)
    z = mu + exp(0.5 * logvar) * eps  # reparameterization
    return z

# 增量解码 (利用因果性)
def decode_incremental(z_prev_cache, z_current_frame):
    z_combined = concat_causal([z_prev_cache, z_current_frame], dim='t')
    for layer in decoder_layers:
        x, cache = causal_3d_deconv(x, cache=layer.cache)
    return x, cache  # 返回当前帧像素 + 更新缓存
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
3D Causal VAE 在 EasyAnimate 中：(1) 训练阶段使用变帧间隔采样提升跨帧编解码鲁棒性；(2) 遵循 MovieGen 添加 latent encoding penalty loss 减少 speckle artifacts；(3) 使用 spatial/temporal slicing 降低长视频高分辨率解码时的 GPU 显存；(4) 在 Reward BP 中的关键作用 —— causal 属性意味着只需解码第一帧（F=1）即可通过因果关系推断后续帧质量，避免多帧 reward 导致的 dynamics 损失和 reward hacking。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## Rectified Flow / Flow Matching（整流流/流匹配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rectified Flow (Flow Matching) 是一种连续时间生成模型框架，通过学习一个 ODE 的速度场 v(x,t) 来在噪声分布和数据分布之间建立"直线"概率路径。与 DDPM/DDIM 的离散时间马尔可夫链不同，rectified flow 将生成过程建模为 dx/dt = v(x,t)，其中 x(0) 是纯噪声，x(T) 是数据。训练目标是学习神经网络预测速度场 v_theta(x_t, t) 匹配从噪声到数据的直线插值路径：x_t = (1-t) x x_0 + t x epsilon。损失函数为 L = E[||v_theta(x_t, t) - (epsilon - x_0)||^2]。相比 DDPM，rectified flow 允许更少的采样步数（路径更"直"）且训练更稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练 rectified flow
def rectified_flow_training(video_data, text_condition):
    x_0 = vae_encode(video_data)
    epsilon = torch.randn_like(x_0)
    t = torch.rand(batch_size)
    x_t = (1 - t) * x_0 + t * epsilon  # 直线插值路径
    v_target = epsilon - x_0  # 目标速度
    v_pred = dit_model(x_t, t, text_condition)
    loss = F.mse_loss(v_pred, v_target)
    return loss

# 采样: Euler 积分 ODE
def rectified_flow_sampling(text_condition, num_steps=50):
    z = torch.randn(latent_shape)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = step * dt
        v_pred = dit_model(z, t, text_condition)
        z = z + v_pred * dt  # Euler step
    return vae_decode(z)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Rectified flow 在视频生成中的关键点：(1) SD3 率先在大规模文生图中使用；(2) EasyAnimate 发现 rectified flow 下的梯度 norm 远小于 DDPM，因此 reward backpropagation 需 K=10（而非 DDPM 的 K=1）以保证训练稳定性；(3) 采样时可用 classifier-free guidance 和更高阶 ODE solver 加速。EasyAnimate 初始实验显示 rectified flow 效果优于 DDPM。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## Reward Backpropagation in Video Diffusion（视频扩散的奖励反向传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reward Backpropagation 是一种利用可微分 reward model 直接优化扩散模型采样过程的后训练方法。与 RL-based 方法（如 DDPO, DPO）将采样视为 MDP 并用 policy gradient 优化不同，reward backpropagation 直接通过 reward model 反向传播梯度到扩散模型的去噪步骤。算法核心：从文本条件 c 生成视频通过采样过程 sample(theta, c, x_T)，解码后过 reward model R 计算分数，优化目标为 L(theta) = -E_c[R(sample(theta, c, x_T), c)]。为节省显存，只对最后 K 步保留计算图。EasyAnimate 针对 rectified flow + 3D Causal VAE 的关键适配：(1) K=10（rectified flow 下梯度 norm 小于 DDPM），(2) F=1（仅解码首帧计算 reward），(3) 使用 LoRA 微调，(4) HPSv2.1 + MPS 组合最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def reward_backprop(dit_lora, vae, reward_models, prompt, K=10, F=1):
    c = qwen2vl_encode(prompt)
    z_T = torch.randn(latent_shape)

    # Phase 1: T->K detach
    z = z_T
    for t in range(T, K, -1):
        v_pred = dit_lora(z, t/T, c)
        z = (z + v_pred * (1/T)).detach()

    # Phase 2: K->0 with grad
    for t in range(K, 0, -1):
        v_pred = dit_lora(z, t/T, c)
        z = z + v_pred * (1/T)  # 保留计算图

    # Phase 3: decode first frame + reward
    video_f1 = vae.decode(z[0:1])  # F=1
    reward = reward_hps(video_f1, prompt) + reward_mps(video_f1, prompt)
    loss = -reward
    loss.backward(); lora_optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
先驱工作 AlignProp/DRaFT（文生图, K=1 DDPM）和 VADER（视频, 多帧）未适配 DiT+rectified flow。EasyAnimate 首次将该方法适配到 DiT+rectified flow+3D causal VAE 架构。消融显示：(1) K=1 时训练不稳定，reward 骤降；(2) F>1 时视频 dynamics 退化和 reward hacking（背景 artifacts）；(3) HPSv2+MPS 组合 VBench Total Score 83.42%，Aesthetic Quality 69.48 为所有模型最高。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## MMDiT (Multi-Modal Diffusion Transformer)（多模态扩散Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MMDiT 是 SD3 (Stable Diffusion 3) 提出的多模态 DiT 架构变体。核心设计：文本和视频两种模态共享 self-attention 层进行跨模态交互，但使用各自独立的全连接层和 FFN，以处理两种模态特征在数值尺度和语义空间上的差异。这种"共享注意力 + 独立 FFN/FC"的设计使模型既能实现文本-视频对齐，又能保持各自模态的特征表达能力。EasyAnimate 中文本特征经 RMSNorm + FC 变换后与视频 latent token 拼接进入 MMDiT。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MMDiTBlock(nn.Module):
    def forward(self, h_text, h_video):
        # 共享 Self-Attention
        h_combined = torch.cat([norm_text(h_text), norm_video(h_video)], dim=1)
        attn_out = self.self_attn(h_combined)
        attn_text, attn_video = split(attn_out)

        # 独立 FFN (残差连接)
        h_text  = h_text  + attn_text
        h_text  = h_text  + self.ffn_text(norm(h_text))
        h_video = h_video + attn_video
        h_video = h_video + self.ffn_video(norm(h_video))
        return h_text, h_video
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MMDiT 最初由 SD3 (Sauer et al., 2024) 提出用于文生图，EasyAnimate 扩展到视频。与标准 DiT（所有 token 共享同一 FFN）或 Cross-Attention DiT 相比，MMDiT 在模态对齐和特征表达方面取得更好平衡。EasyAnimate 的 7B 和 12B 版本均基于 MMDiT。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## 3D Rotary Position Embedding（三维旋转位置编码 / 3D RoPE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
3D RoPE 是将 Rotary Position Embedding 从 1D 序列扩展到 3D 视频数据的位置编码方法。RoPE 通过旋转矩阵将相对位置编码到 attention score 中：Q_m^T x R(m-n) x K_n。3D RoPE 将视频的时空维度 (T=temporal, H=height, W=width) 分别编码：将 hidden channels 按比例分配给三个维度（EasyAnimate 采用 3/8 temporal, 3/8 height, 2/8 width），各维度独立计算 1D RoPE 后拼接。模型可区分"同一空间不同时间"和"同一时间不同空间"的 token，捕获时空关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_3d_rope(video_tokens, d_model, frames, height, width):
    d_F = int(d_model * 3/8)  # temporal channels
    d_H = int(d_model * 3/8)  # height channels
    d_W = d_model - d_F - d_H  # width channels (2/8)

    rope_F = apply_rotary(video_tokens[:,:,:,:d_F], freqs_F, positions=t_idx)
    rope_H = apply_rotary(video_tokens[:,:,:,d_F:d_F+d_H], freqs_H, positions=h_idx)
    rope_W = apply_rotary(video_tokens[:,:,:,d_F+d_H:], freqs_W, positions=w_idx)

    return torch.cat([rope_F, rope_H, rope_W], dim=-1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
3D RoPE 继承 1D RoPE 的高效性：旋转矩阵稀疏（对角块），可通过 element-wise 乘法实现，计算开销极小。相对位置编码特性使模型对序列长度有更好外推能力。各视频 DiT（CogVideoX, HunyuanVideo, EasyAnimate）的 3D RoPE 在 channel 分配策略上可能不同，但核心机制相同。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

**VideoRoPE 论文对 3D RoPE 设计的系统分析**：VideoRoPE 提出好的 3D RoPE 应满足四个关键属性：(1) **3D Structure**——保留 (t,x,y) 时空结构而非 flatten 为 1D；(2) **Frequency Allocation**——temporal 维度应分配低频（高维），因为空间分辨率有界、仅需高频覆盖，而时间可无限增长、需要低频避免远距离"hash collision"；(3) **Spatial Symmetry**——preceding text end → visual start 的距离 ≈ visual end → subsequent text start 的距离，简化学习并减少位置偏置；(4) **Temporal Index Scaling**——temporal spacing 应不同于 spatial spacing（δ≠1），体现不同粒度的维度编码。

**Qwen2.5-VL 的 3D 位置编码**：Qwen2.5-VL 对视觉 token 使用显式三维位置编码 (x, y, t)，分别对应空间宽度、空间高度和时间维度。文本 token 的三维坐标保持一致（t 维度固定或为零），使文本 token 在空间维度上无区分。该设计允许模型在统一嵌入空间内联合推理空间、时间和语义上下文。在流式推理中，每个新到达的视频帧的视觉 token 按 (x, y, t) 坐标分配三维位置，其中 t 维度随帧序号递增。

**流式推理中的位置连续性约束问题**：Qwen2.5-VL 原生要求全局位置连续——所有 token 共享同一递增位置索引空间。在流式场景中，由于下一帧视觉 token 的起始位置依赖当前文本生成长度（不可预知），prefill 和 decode 必须串行交替执行，无法真正并行。这是本论文 Speak While Watching 识别并解决的核心瓶颈。解决方案包括三种打破连续性的位置编码策略（GDPE/OSPE/GIPE），详见对应术语条目。

## MLLM as Text Encoder for Diffusion（多模态大语言模型作为扩散模型文本编码器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
使用多模态大语言模型（如 Qwen2-VL、InternVL）替代传统 CLIP/T5 作为扩散模型的文本编码器。传统 CLIP 限制输入 77 tokens，T5 对细粒度场景理解不足。MLLM 在视觉-语言任务上预训练，具有更好文本理解能力，且统一文本-视觉 token 空间与视频生成任务天然匹配。EasyAnimate 从 Qwen2-VL-7B 倒数第二层提取 hidden features，经 RMSNorm + FC 对齐后输入 DiT。VBench 验证 Total Score 从 80.42% (T5+CLIP) 提升到 81.57% (Qwen2-VL)。由于 MLLM 特征的 L2 norm 远大于视频噪声 latent，需要 RMSNorm 归一化避免训练不稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MLLMTextEncoder(nn.Module):
    def __init__(self):
        self.mllm = load_qwen2vl("Qwen2-VL-7B")
        self.mllm.requires_grad_(False)  # 冻结
        self.rms_norm = RMSNorm(mllm_hidden_dim)
        self.fc_align = nn.Linear(mllm_hidden_dim, dit_hidden_dim)

    def forward(self, text_prompt):
        hidden_states = self.mllm(text_prompt, output_hidden_states=True)
        text_features = hidden_states[-2]  # 倒数第二层
        text_features = self.rms_norm(text_features)  # 归一化
        return self.fc_align(text_features)  # 对齐到 DiT dim
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MLLM 编码器的优势：(1) 支持多语言输入；(2) 支持远长于 77 tokens 的文本；(3) VBench 验证有效。代价是推理时额外显存和延迟（7B 参数）。MovieGen 等也在探索更强的文本编码器。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

## Multi-Armed Bandit (MAB) for Keyframe Selection / Combinatorial Pure-Exploration (CPE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Armed Bandit (MAB) 是随机决策理论中的经典框架，建模在不确定环境下分配有限资源（pulls）给多个选项（arms）以达到某种最优目标的决策问题。FOCUS 将长视频 keyframe selection 创新性地建模为 MAB 中的 Combinatorial Pure-Exploration (CPE) 子问题：视频被划分为 M 个固定时长的 clip（每个 clip 为一个 arm），目标是选出最优的 m 个 arm 子集（即最 query-relevant 的 clip），然后从这些 arm 内进一步选出 k 个 keyframes。Pure-exploration 意味着目标不是最小化 regret，而是以高置信度找到最优 arm 子集——天然匹配 keyframe selection 的"选最优帧"目标。CPE 由 Chen et al. (NeurIPS 2014) 首次提出，核心算法 CLUCB 通过 confidence bound + oracle maximization 实现对任意组合结构的 arm 子集的高效识别。FOCUS 的决策类 S 定义为所有大小为 m 的 arm 子集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 FOCUS 中，CPE bandit 建模流程：
```
# 将 keyframe selection 建模为 CPE bandit
Clips = partition(V, clip_length=l)  # M 个 clip = M 个 arm
def pull(arm_a):
    t ~ uniform(arm_a.start, arm_a.end)  # 随机采样 arm 内一帧
    return BLIP_ITM(x_t, q)              # frame-query relevance ∈ [0,1]

# CPE 目标: 找到最优 m-arm 子集
# S* = argmax_{|S|=m} sum_{a∈S} μ_a,  μ_a = E[r_t | t∈A_a]
# 决策类: S = {all size-m subsets of M arms}
```
分层设计的关键洞察：视频帧间强时间相关性（ACF > 0.5 for ~5s）意味着 clip 内帧高度相似，少量采样即可估计整个 clip 的平均 relevance。从帧级选择 C(T,k)（组合数巨大）降为 clip 级选择 × 帧级选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPE bandit 在 FOCUS 中通过 Algorithm 2（两阶段 optimistic UCB）实现。理论保证：Bernstein confidence bound 保证 |μ̂_a - μ_a| ≤ β_a 以 ≥ 1-6/n 概率成立（Theorem B.1）；Algorithm 2 以 ≥ 1-6(M-m)/n 概率返回 oracle top-m set（Theorem C.1）。CPE 框架在 FOCUS 中用于 clip 级粗筛，随后在选中的 arm 内通过 nearest-neighbor 插值 + 概率采样完成帧级 fine selection。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

## Bernstein Confidence Radius / Bernstein Confidence Bound in Bandits

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bernstein Confidence Radius 是 bandit 中使用 Bernstein 不等式构造的方差自适应置信半径，由 Audibert, Munos & Szepesvári (Theoretical Computer Science, 2009) 在 UCV 算法中引入。与标准 UCB 使用 Hoeffding 不等式（置信区间宽度仅依赖样本数，与方差无关）不同，Bernstein 版同时利用经验均值 μ̂_a 和经验方差 σ̂_a²：β_a(n) = sqrt(2·σ̂_a²·ln(n) / N_a(n)) + 3·ln(n) / N_a(n)。当 arm 方差很小时（如静态场景 clip），界宽显著小于 Hoeffding 界——因为 sqrt(σ̂_a²/N) << sqrt(1/N)。这使算法能更快排除低方差低质量 arm，将更多采样预算投向高方差高不确定性 arm。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每个 arm a 在 n 轮总采样后
N_a = max(1, pulls_of_arm_a)
μ̂_a = mean(observed_rewards_a)
σ̂_a² = variance(observed_rewards_a)

# Bernstein 置信半径 (FOCUS Eq.5)
β_a = sqrt(2 * σ̂_a² * ln(n) / N_a) + 3 * ln(n) / N_a

# 高概率保证 (Theorem B.1): P[|μ̂_a - μ_a| ≤ β_a] ≥ 1 - 6/n

# 与 Hoeffding 版对比
# Hoeffding: β = sqrt(ln(n) / (2*N_a))   — 固定宽度，忽略方差
# Bernstein: β = sqrt(2σ̂²*ln(n)/N) + 3*ln(n)/N — 方差自适应
```

FOCUS 消融（Table 8）：FOCUS-M（仅经验均值）= 63.0%, FOCUS（加 Bernstein）= 63.5% on LLaVA-Video。增益来自 Bernstein 对高不确定性 arm 的额外探索激励。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bernstein confidence radius 的实现仅需每个 arm 维护 μ̂_a 和 σ̂_a²（Welford 在线更新算法），计算开销 O(M)。在 FOCUS 中，β_a 用于构造 optimistic mean μ̃_a = μ̂_a + β_a（Stage I 后 arm 粗选），也用于判断 arm 的探索价值。效果取决于 reward 的真实方差（clip 内帧的 relevance 波动）和每 arm 采样数。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

## Two-Stage Batched Bandit Exploration for Video Keyframe Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Stage Batched Bandit Exploration 是 FOCUS 将理论上串行的 bandit 算法（Algorithm 1: iterative UCB, 每步 pull 1 arm）转化为实际可高效并行批处理的策略（Algorithm 2）。串行算法要求每轮 pull 一个 arm → 观察 reward → 更新统计 → 决定下一 arm → repeat，这在 GPU 上意味着 BLIP 以 batch_size=1 串行前向，严重浪费 GPU 利用率。FOCUS 压缩为两次并行 batch：(1) Stage I Coarse——所有 M arm 各采 q 帧，一次性 batch BLIP forward → 计算 per-arm 统计 + optimistic UCB；(2) Stage II Fine——仅对 UCB 最高的 α*m arm 各采 z 帧，再一次性 batch → 用无偏经验均值选最终 top-m arm。与 batched bandit 文献（Perchet et al., 2016; Gao et al., 2019; Jin et al., 2024 的 Tri-BBAI 三批次最优 BAI）精神一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 2: 两阶段批处理 (仅需 2 次 batch BLIP forward!)
# Stage I: Coarse Exploration
rewards = BLIP_batch(all M arms, q frames each)  # 第 1 次 batch, M*q 帧
for a in 1..M:
    μ̂_a, σ̂_a² = stats(rewards[a])
    β_a = sqrt(2*σ̂_a²*ln(n)/q) + 3*ln(n)/q
    μ̃_a = μ̂_a + β_a
A_coarse = TopM({μ̃_a}, α*m)    # optimistic UCB 粗选 α*m 个 arm

# Stage II: Fine-grained Exploitation
rewards_fine = BLIP_batch(A_coarse, z frames each)  # 第 2 次 batch
for a in A_coarse:
    update μ̂_a with rewards_fine[a]    # 合并两次采样更新经验均值
A_fine = TopM({μ̂_a}, m)              # 无偏经验均值精选 m 个 arm

# Frame Selection within A_fine
for a in A_fine:
    r̂_{a,t} = nearest_neighbor_interpolate(rewards_a)  # 插值所有帧
    p_a = softmax(r̂_{a,t})              # 构建采样分布
    K_a = sample_without_replacement(p_a, k_a)  # 不放回采样
return K = union(K_a)
```

消融（Table 7）：FOCUS-C（仅 coarse）= 62.3/58.4/62.3%, FOCUS-F（仅 fine）= 61.5/57.7/62.5%, FOCUS（两阶段）= 62.3/60.7/63.5%（Qwen2-VL/LLaVA-OV/LLaVA-Video）——两阶段互补，coarse 做全局定位，fine 做精准提取。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两阶段批处理的核心优势：仅需 2 次 batch BLIP forward（vs 串行算法需 O(M²·log(1/δ)) 次），充分利用 GPU 并行性。FOCUS 在 LongVideoBench 上处理帧数仅占总帧数的 1.6%，5.5 GPU hours。Arm 数 M = video_duration / clip_length（如 1h / 16s = 225），α=0.25 默认。q 和 z 的具体值论文未明确说明。Batched bandit 理论保证（Theorem C.1）：两阶段算法以 ≥ 1-6(M-m)/n 概率输出 oracle top-m set。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

## Frame-Query Relevance Scoring via BLIP ITM for Keyframe Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame-Query Relevance Scoring 是 FOCUS 及多数 training-free keyframe selection 方法（AKS, Q-Frame, Top-K）使用的核心信号——用预训练 vision-language encoder（默认 BLIP ITM）计算每帧与文本查询的语义相关性分数，作为该帧对回答查询的信息贡献的代理（proxy）。在 FOCUS 中，relevance r_t = cosine_similarity(BLIP.encode_image(x_t), BLIP.encode_text(q))，作为 bandit 的 reward 信号。FOCUS 理论框架建模为 r_t = y_t + ε_ψ，其中 y_t 是真值 frame-level utility（不可直接观测），ε_ψ 是 encoder 噪声（零均值，方差 σ_ψ²）——即 r_t 是 y_t 的无偏估计。BLIP (Li et al., ICML 2022) 是 Salesforce 提出的统一视觉-语言理解与生成框架，其 ITM (Image-Text Matching) 头通过 cross-attention 融合图文特征输出匹配概率，捕获细粒度对齐。BLIP-2 (Li et al., ICML 2023) 引入 Q-Former 连接冻结 ViT 和 LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BLIP ITM 用于 frame-query relevance scoring
# 输入: 帧 x_t (3×H×W), 查询文本 q

# Vision Encoder: ViT → image feature
e_img = BLIP.visual_encoder(x_t)    # shape: (d,), d≈768/1024

# Text Encoder: BERT → text feature
e_txt = BLIP.text_encoder(q)        # shape: (d,)

# ITM: cross-attention + binary classifier
# BLIP 内部: image features as K,V, text features as Q → [CLS] token → sigmoid
# 或简化版 (FOCUS 实际可能使用):
r_t = (e_img · e_txt) / (||e_img|| * ||e_txt||)  # cosine similarity ∈ [0,1]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 FOCUS 中，BLIP 参数完全冻结不做微调。FOCUS Table 10 encoder 消融：BLIP (63.5%) > SigLIP (60.9%) > CLIP (60.2%) > Uniform (58.9%)，所有 encoder 均优于 uniform baseline，框架对 encoder 选择鲁棒。每个 BLIP forward 约 10^8-10^9 FLOPs，全量评分 1h 视频需 10^11-10^12 FLOPs——即 255 GPU hours（Table 3: AKS w/o pre-filtering），所以需要 bandit 采样。BLIP ITM 的局限：(1) 对需要世界知识的复杂推理查询评分不准；(2) 无法区分同 object 不同 context 的情况。FOCUS 通过 bandit 采样仅评分 1.6% 帧，将开销降至 5.5 GPU hours。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
## MoRef Attention (Mixture-of-Reference Attention / 多参考注意力机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoRef Attention 是一种 training-free 的并行多参考注意力机制，受 MoE（Mixture-of-Experts）范式启发，用于在单次推理中让 Video-MLLM 的 LLM 并行 query 多个 vision reference chunks 并聚合为统一的 question token 激活。其核心流程为：(1) 对 N 个 parallel inference chunks 分别执行标准 causal FlashAttention，得到 O_i = [O_i^sys, O_i^vis, O_i^ques]；(2) 由于 causal attention 的单向性，各 chunk 的 O_i^sys 完全相同；(3) 保持各 chunk 的 O_i^vis 差异（保留各 reference 的视觉特征）；(4) 对 O_i^ques 执行跨 chunk 加权聚合：O_fusion = (Σ ω_i · O_i^ques).repeat(N)，ω_i = max(A[i]) / Σ max(A[j])，其中 A = softmax(Q^ques × K^vis^T) 为 query-vision 跨模态注意力图；(5) 组装最终输出 O^MoRef = [O^sys, O^vis, O_fusion]。加权系数 ω_i 表示 query 与各 reference 的相关性，使模型能自适应地从不同 reference 提取相关线索。计算复杂度约为 full attention 的 1/N（N 为 chunk 数），因为将长序列分解为 N 个短序列并行处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoRef Attention 替换 LLM 中每一层 shallow decoder 的标准 self-attention。伪代码如下：

```
for i in range(N):
    Q_i, K_i, V_i = W_Q(chunk_i), W_K(chunk_i), W_V(chunk_i)
    O_i = FlashAttention(Q_i, K_i, V_i, causal=True)

for i in range(N):
    A_i = softmax(Q_i^ques @ K_i^vis^T)  # R^{l_ques x l_vis_i}
    w_i = max(A_i) / sum(max(A_j) for j in range(N))

O_fusion = sum(w_i * O_i^ques for i in range(N))
O_fusion = O_fusion.repeat(N)

for i in range(N):
    O_i^MoRef = concat([O_i^sys, O_i^vis, O_fusion])
    chunk_i = chunk_i + O_i^MoRef
    chunk_i = chunk_i + FFN(LayerNorm(chunk_i))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 论文中 MoRef Attention 直接替换 LLaVA-Video-7B 中 Qwen2-7B 的 self-attention layers（shallow layers 0 到 L-1）。兼容标准 FlashAttention（causal 模式），额外计算仅需一次 A = softmax(Q^ques × K^vis^T)，可忽略不计。gating weights ω_i 体现 query-aware 选择。FLOPs：128 frames (N=2) 时 110.4% baseline (vs full attention 400%)，256 frames (N=4) 时 163.2% (vs 1600%)。超参数：最佳 N = input_frame_num/64, M=64。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

## Multi-Reference Partition (多参考划分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Reference Partition 是 Free-MoRef 中的 vision token 划分策略，将长 vision token 序列按时间关系划分为多个 short parallel reference chunks。过程：(1) 将 vision tokens 按时间分为 M 个 units；(2) 每个 unit 内沿时间分解为 N 个 fragments；(3) 聚合不同 unit 的相同 index fragment → N 个 reference chunks。参数 M 和 N 为手动配置：M 越大各 reference 间时间交集越多；M=1 时 chunk 时间完全独立。各 chunk 分配相同 system prompt 和 question。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
units = split_into_groups(vision_tokens, M)
for j in range(M):
    fragments[j] = split_into_groups(units[j], N)
for i in range(N):
    chunk_i = concat([fragments[j][i] for j in range(M)])
    parallel_inputs[i] = concat([system_prompt, chunk_i, question])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 固定 M=64。128 frames: N=2；256 frames: N=4；512 frames: N=8。M 影响 sparse attention pattern：M 越小各 chunk 时间连续性越强（利于 Spatial Perception），M 越大 tokens 分布越均匀（利于 Temporal Perception）。纯 token 重组操作，无额外计算开销。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

## Reference Fusion (参考融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reference Fusion 是在 LLM decoder 中间层执行的 vision token 剪枝与合并操作。基于 FastV 观察（vision tokens 在 shallow layers 均匀贡献，deep layers 集中于 question tokens）。流程：(1) 在第 L 层，基于 A ∈ R^{N × l_ques × l_vis}；(2) 沿 l_ques 平均 → E ∈ R^{N × l_vis}；(3) 每个 chunk 保留 top 1/N tokens（剪枝 1-1/N）；(4) 按时间关系聚合为 global reference；(5) system prompt/question 直接拷贝；(6) 后续 layers 仅用 global reference 做标准 attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for i in range(N):
    E_i = mean(A_i, dim=l_ques)
    kept_tokens_i = vision_chunk_i[argsort(E_i, desc)[:l_vis_i//N]]
global_vision = temporal_merge(kept_tokens_all)
global_seq = concat([system_prompt, global_vision, question])
# layers L+1..end: standard causal attention on global_seq
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 配置：128 frames → L=3 (drop 50%)，256 frames → L=6 (drop 75%)。过早 fusion (L=1) 导致信息丢失（65.4 vs L=3 的 66.3）。双重作用：减少 deep layers 计算量 + 补偿 shallow layers 缺失的跨 chunk vision 交互。消融：仅 MoRef Attention 无 Fusion 时 Overall 65.8 vs 含 Fusion 66.3。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

## Training-free Context Extension for Video-MLLM (训练无关的视频多模态大模型上下文扩展)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-free Context Extension 是 Free-MoRef 提出的无需额外训练的长上下文扩展方法。区别于：(1) LLM Context Expansion —— 通过长序列 post-training 扩大 context limit（如 LongVILA），计算负担大；(2) Token Compression —— 推理前压缩 vision tokens（如 FastV、Video-XL），高压缩率导致信息丢失；(3) Streaming Inference —— 多次调用 LLM 复用 KV Cache（如 RETAKE），延迟与上下文长成正比。Free-MoRef 通过 Partition + MoRef Attention + Reference Fusion，不训练参数、不压缩 token，实现 2x-8x 上下文扩展，FLOPs 仅 ~1/N 增长，first token latency 恒定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
vision_tokens = model.vision_encoder(sample_frames(video, N_frames))
chunks = multi_reference_partition(vision_tokens, M, N)
for layer in range(L_fusion):
    chunks = MoRef_attention(chunks)
global_ref = reference_fusion(chunks, L_fusion)
for layer in range(L_fusion, num_layers):
    global_ref = standard_layer(global_ref)
answer = decode(global_ref)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLaVA-Video-7B (Qwen2-7B) 上实现，无需修改任何模型权重。兼容 FlashAttention。可与 token compression 或 streaming inference 正交叠加。开源: https://github.com/wkfdb/Free-MoRef。在 VideoMME/MLVU/LongVideoBench 上超越需专门训练的长视频模型（LongVILA, Video-XL, RETAKE）。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

## Visual Token Sampling (VTS) / Query-Guided Visual Token Sampling（查询引导的视觉Token采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Token Sampling (VTS) 是 GroundVTS (CVPR 2026) 提出的核心模块，用于在 Vid-LLM pipeline 中对 visual tokens 进行 query-guided 细粒度采样。VTS 位于 visual encoder + multimodal projector 之后、LLM 输入之前，通过计算每个 visual token 与 text query 的语义相关性，动态选择最 informative 的 visual tokens 送入 LLM。与 uniform frame sampling（对所有帧平等分配 token 配额）和 frame-level query selection（基于外部编码器粗粒度选帧）不同，VTS 在 token 级别进行选择：同一帧内不同空间位置的 token 可因与 query 相关性不同而获得不同的保留权重。VTS 包含两个子操作：Query-Guided Token Scoring（相关性估计）和 Differentiable Top-K Selection（基于 Gumbel-Softmax STE 的可微分选择）。输出非均匀的 visual token 分布——高 query 相关性区域 token 密度高，低相关性区域 token 稀疏或为零。GroundVTS 证明，以一半的 token 预算（ρ=0.5），VTS 超越了全量 uniform baseline（Charades-STA R1@0.7: 34.2 vs 30.5）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTS 在 Vid-LLM pipeline 中的完整执行流程（GroundVTS-Q, ρ=0.5, 2 FPS）：
```
# === VTS Pipeline ===
# 输入: 视频帧 {F_t}, 文本查询 text_query
# 参数: W_v, W_q (可学习投影矩阵), τ, τ_g (温度), D_r (隐藏维度)

# 前处理
H_v = VisionEncoder({F_t})      # T frames → N_v visual tokens
V = Projector(H_v)              # MLP → R^{N_v × D}
Q = TextTokenizer(text_query)   # → R^{N_t × D}

# VTS Step 1: Query-Guided Token Scoring
V' = W_v @ V                    # W_v ∈ R^{D × D_r}
q' = W_q @ mean(Q, dim=0)      # W_q ∈ R^{D × D_r}
w = softmax(V' @ q'^T / τ)     # ∈ R^{N_v}, token-query 相关性

# VTS Step 2: Differentiable Top-K Selection
K = ceil(ρ * N_v)               # 保留 K 个 token
g_i ~ Gumbel(0, 1)
z = softmax((log w + g) / τ_g)  # Gumbel-Softmax 松弛
I_K = TopK_indices(w, K)
z_hard = 1[i ∈ I_K]
z_tilde = z_hard + z - stopgrad(z)  # STE

# Weighted Re-encoding
w_hat = exp(w/τ') * z_tilde / sum(exp(w/τ') * z_tilde)
V_selected = w_hat * MLP(V)

# 保留原始位置编码 + 送入 LLM
input = concat([V_selected + PE[I_K], Q])
answer = LLM.generate(input)
```

关键超参数: ρ=0.5 (保留 50% tokens), D_r=512(GroundVTS-Q) / 128(GroundVTS-I)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTS 作为可训练模块嵌入 Vid-LLM pipeline, 参数约 29-35M (W_v, W_q, MLP_vts)。使用三阶段训练：(1) Stage 1: VTS Warm-up (冻结 LLM, 仅训练 VTS) → (2) Stage 2: Joint LoRA Adaptation (VTS + Projector + LoRA(LLM), LLaVA-Video-178K) → (3) Stage 3: Grounding Fine-tuning (Grounding-FT 70K, VTG 任务)。Gumbel-Softmax + STE 通过 PyTorch 原生 F.gumbel_softmax(hard=True) 实现, 兼容 FlashAttention, 无需自定义 kernel。代码开源: https://github.com/Florence365/GroundVTS。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

## Video Temporal Grounding (VTG) / 视频时序定位

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Temporal Grounding (VTG) 是视频理解领域的核心任务，目标是根据自然语言查询在视频中精确定位对应事件的起止时间戳。含两个子任务：(1) Moment Retrieval (MR) — 识别与查询对应的单个时间片段，输出 start/end 时间戳，评估指标 R1@t (t∈{0.3,0.5,0.7}) 和 mIoU；(2) Highlight Detection (HD) — 输出视频中所有与查询相关的显著时刻及其 saliency scores，评估指标 mAP 和 Hit@1。标准 benchmark: Charades-STA（日常活动）、ActivityNet-Captions（网络视频事件）、QVHighlights（MR+HD 联合评估）。GroundVTS 在两个变体上验证：GroundVTS-Q (Qwen2.5VL-7B) 和 GroundVTS-I (InternVL3.5-8B)，Charades-STA mIoU 达 50.1 (+18.4 over baseline)，QVHighlights HD mAP 达 52.5 (+20.6)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MR 任务在 Vid-LLM 中的指令格式（GroundVTS Grounding-FT 数据集）：
```
# 输入
"<video>At what point in the video did the following events occur: a person takes a book off a shelf. Output the start and end timestamps."
# 输出
"from 6.0s to 12.0s"
```
HD 任务输出格式："The highlights are: important from 96.0s to 98.0s; less important from 100.0s to 102.0s"。时间戳信息不放入 text prompt，模型仅依赖 visual token 的位置编码 (PE) 推断时间信息——因此 VTS 保留原始位置编码对 VTG 精度至关重要（消融实验中去除 PE 后 mIoU 从 50.1 降至 9.5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTG 在 Vid-LLM 中通过 instruction tuning 实现：在预训练 Vid-LLM 基础上，用 VTG 数据集 (Charades-STA, ActivityNet-Captions, QVHighlights) 进行 SFT。GroundVTS 使用自建 Grounding-FT (70K, ShareGPT format, 含多样化 prompt templates)。评估使用 llm-eval 或自定义脚本计算 IoU-based metrics。代码开源: https://github.com/Florence365/GroundVTS。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
- Temporal Preference Optimization of Large Multimodal Models

ReVisionLLM 将 VTG 扩展到小时级长视频，引入递归层次化处理：(1) 顶层 hierarchy 用 sparse temporal features 扫描全视频粗定位感兴趣区段，(2) 中间层聚焦预测区域进一步细化，(3) 底层用 dense temporal features 精确定位秒级起止时间。使用 LLM 输出熵的倒数作为置信度排序（替代 CLIP 相似度），ECE 从 0.62 降至 0.46。在 MAD (1200h movies) 和 VidChapters-7M (817K videos, up to 12h) 上建立 SOTA。代码: https://github.com/Tanveer81/ReVisionLLM。

## Gumbel-Softmax Straight-Through Estimator for Differentiable Token Selection（Gumbel-Softmax STE 可微分Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gumbel-Softmax + STE 是 GroundVTS 中用于实现端到端可训练 top-K token 选择的核心技术。问题: hard top-K selection (argmax/top-K) 是非可微操作，无法通过梯度下降优化。方案: (1) Gumbel-Softmax — 向 log-probabilities 添加 Gumbel 噪声并通过 softmax 产生连续松弛，近似 categorical 采样: z_i = softmax((log w_i + g_i) / τ_g)；(2) STE — forward pass 使用 hard (0/1) mask，backward pass 通过 soft 松弛传播梯度: \tilde{z}_i = z_i^hard + z_i - stopgrad(z_i)。这使离散 token 选择可端到端训练。GroundVTS 中该技术用在 VTS 模块的 top-K selection 步骤，backward 梯度通过 z_i（连续松弛）流入 w_i → V' → W_v 和 W_q → VTS 所有可学习参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Gumbel-Softmax STE ===
# 输入: logits = log w (token 相关性)

# Gumbel 噪声采样 (reparameterization trick)
u ~ Uniform(0, 1)
g = -log(-log(u))  # Gumbel(0,1)

# 连续松弛 (Eq.4)
z_soft = softmax((logits + g) / τ_g)
# τ_g 控制松弛平滑度: τ_g→0 → z_soft→one-hot; τ_g→∞ → z_soft→uniform

# Hard Top-K (Eq.5, forward only)
z_hard = 1[i ∈ TopK(logits, K)]

# STE (Eq.6)
z_out = z_hard + z_soft - z_soft.detach()
# forward: z_out = z_hard (离散)
# backward: ∂L/∂z_out = ∂L/∂z_soft (连续梯度)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 F.gumbel_softmax(logits, tau=τ_g, hard=True) 直接实现 (hard=True 时内置 STE)。训练时使用 Gumbel noise + STE，推理时直接 hard top-K (无 noise, 确定性)。GroundVTS 中 τ_g 为可调超参数，τ_g 过小导致训练不稳定（梯度方差大），τ_g 过大导致与 hard selection 偏差大（训练-推理 gap）。与 RL-based token selection 的区别：Gumbel-Softmax STE 直接通过梯度下降优化，无需 reward 设计或 policy gradient 的方差问题。CVPR 2026。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

TSPO 中的 Gumbel-Softmax 使用方式：TSPO 使用 Gumbel-Softmax（不含 STE）进行概率化关键帧选择。流程：对 cross-modal similarity scores S ∈ R^{T_c}，注入 Gumbel(0,1) 噪声 γ，计算 softmax(S/τ + γ) 得到概率分布 P，再 TopK 选择 T_s 帧。训练时通过 τ annealing（0.025→0.01）控制探索-利用平衡；推理时去除 Gumbel 噪声，直接确定性 Softmax + TopK 采样。与 GroundVTS 的 STE 变体不同，TSPO 不通过梯度下降优化帧选择器，而是通过 GRPO 的 policy gradient 优化——Gumbel-Softmax 在此仅用于提供可探索的离散动作空间（概率化采样），梯度传播由 GRPO 的 importance sampling ratio 而非 STE 处理。

VisionSelector 的 DTS 作为替代方案：与 Gumbel-Softmax STE 不同，DTS 使用 sigmoid 连续松弛 + 隐函数微分实现可微分 Top-K，无需 Gumbel 噪声，具有确定性、严格单调性（s_i > s_j ⇔ M_i > M_j），梯度为闭式精确解（而非 STE 近似）。且 DTS 不通过 τ annealing 桥接训练-推理 gap，而是通过 Curriculum Annealing Strategy (CAS) 在损失权重空间渐进。详见 「Differentiable Top-K Selection（DTS）」条目。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs

## Progressive Optimization Strategy for VTS（VTS渐进式优化策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Optimization Strategy 是 GroundVTS 的三阶段训练范式，解决将 non-uniform visual token distribution 引入预训练 Vid-LLM 时的两个核心挑战：(1) 分布偏移 — 预训练 LLM 在均匀 token 分布上训练，直接输入非均匀 token 导致注意力不稳定；(2) 训练不稳定 — VTS 的 STE 离散选择和 LLM 的连续优化存在冲突。三阶段设计：Stage 1 (VTS Warm-up, lr=1e-5, 1 epoch) — 冻结 LLM + Projector，仅训练 VTS (W_v, W_q, MLP_vts) → Stage 2 (Joint LoRA Adaptation, lr=2e-4, 2 epochs) — LoRA (rank=8, α=16) 微调 LLM + VTS + Projector 联合训练，LLaVA-Video-178K 数据集 → Stage 3 (Grounding Fine-tuning, lr=1e-4, 3 epochs) — 同 Stage 2 配置，Grounding-FT 70K 专用 VTG 数据。消融证明每阶段必需：跳 Stage 1 (仅 2+3) R1@0.7=30.5 vs full=34.2，跳 Stage 2 (仅 1+3) R1@0.7=15.2。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === 三阶段训练 ===
# Stage 1: VTS Warm-up
freeze(LLM, Projector)
for epoch in 1..1:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # 仅更新 VTS (W_v, W_q, MLP_vts)

# Stage 2: Joint LoRA Adaptation
unfreeze(Projector); add LoRA(LLM, rank=8, α=16)
for epoch in 1..2:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # 更新 VTS + Projector + LoRA

# Stage 3: Grounding Fine-tuning
for epoch in 1..3:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # VTG-specific tuning
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
batch_size=2/GPU, gradient_accumulation=4, AdamW (β1=0.9, β2=0.999)。LoRA 作用于 LLM attention Q/V 投影矩阵。三阶段理论依据: Stage 1 预收敛 VTS 采样分布 → Stage 2 在通用视频数据上适应非均匀分布 → Stage 3 在 VTG 专用数据上精调。该策略的渐进性体现为分布稳定化 → 跨模态对齐 → 任务特化。training details 详见论文 Table 13/14。CVPR 2026。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

## Select Any Frames (SAF) / 任意帧选择

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Select Any Frames (SAF) 是 HORNet 提出的帧选择问题形式化框架，将视频帧选择解耦为独立于 VLM 推理的 RL 问题。给定视频 V={v_1,...,v_T}（T 帧均匀采样）、问题 q 和 ground-truth 答案 a，SAF 的目标是学习参数化策略 π_θ 选择一个子集 V'=π_θ(V,q) ⊆ V，最大化 frozen VLM M 的回答质量：θ* = argmax_θ E[R(M(π_θ(V,q), q), a)]。策略输出 binary mask b ∈ {0,1}^T，无时序顺序或连续性约束——策略可自由选择时间上稀疏的关键事件、短关键片段或密集运动段。策略分布分解为独立 Bernoulli：π_θ(b|V,q) = Π_t Bernoulli(b_t|p_t)，p_t 为 frame t 的选择概率。SAF 的核心贡献是将帧选择从 VLM 推理中解耦（modular policy + frozen VLM），使 frame selection policy 可独立训练、可 transfer 到不同 VLM answerer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SAF 在 HORNet 中的完整 pipeline：
```
# === SAF Pipeline (HORNet) ===
# 输入: V (T frames), q (question), a (ground-truth answer)
# M: frozen VLM (Qwen3-VL), E: trainable video encoder, π_θ: trainable MLP

# Step 1: Video Encoding
F = E(V)                    # TimeSFormer-Tiny → R^{T × D}, D=768

# Step 2: Frame Selection Policy (per-frame independent)
for t in 1..T:
    p_t = sigmoid(W_2 · GELU(W_1 · GELU(W_0 · F[t])))
# p_t ∈ (0,1), θ = {W_0, W_1, W_2}: <1M params

# Step 3: Sampling (train) / Top-K (inference)
# 训练: b_t ~ Bernoulli(p_t)
# 推理: b = TopK(p, k)

# Step 4: VLM QA
V' = V[b == 1]
a_hat = M(V', q)

# Step 5: Reward
R = 0.1 * F1_token(a_hat, a) + 0.9 * EditSim(a_hat, a)
```

SAF 与现有 frame selection 方法的关键区别：Fully learned selection (vs uniform/clip-similarity heuristics), Reward-based optimization (vs pseudo-label SFT), Frozen VLM (vs fine-tuning), Parameter efficient (<1M vs ~1B+)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 HORNet 中，SAF 通过 GRPO 训练：K=8 candidates (7 top-k sweep + 1 Bernoulli), reward = F1-Lev, group-relative advantage, Adam lr=1e-4。两阶段训练 (Stage 1: short videos + F1-Lev, Stage 2: long videos + MCQ accuracy)。推理 deterministic top-k (4-8 frames)。训练硬件：单卡 A100 40GB。开源：https://github.com/ostadabbas/HORNet。Policy 可跨 VLM answerer transfer（+8.5% relative gain with Qwen2.5-VL-3B）。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

## TimeSFormer / Factorized Spatiotemporal Attention (分解式时空注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TimeSFormer (Time-Space Transformer) 是 Facebook Research (Bertasius et al., ICML 2021) 提出的纯 Transformer 视频分类架构，核心创新是 Factorized (Divided) Spatiotemporal Attention：将 3D 联合时空注意力分解为两个独立的顺序操作——spatial self-attention（同帧内不同 patch 间）→ temporal self-attention（同 spatial position 跨帧间）。这种分解将计算复杂度从 O((T·N)²)（联合时空）降至 O(T·N² + N·T²) ≈ O(N² + T²)，且 counterintuitively 比联合注意力更准确（分离强加了有用的 inductive bias）。HORNet 使用 TimeSFormer-Tiny 作为 video encoder（patch_size=16, T=32, D=768），提取 per-frame spatiotemporal features 供 frame selection policy 使用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === TimeSFormer in HORNet ===
# 输入: V (T=32 frames, 288×288×3), P=16, D=768

# Step 1: Patchify
patches = conv2d(V, kernel=16, stride=16)  # (T, 18, 18, 768)

# Step 2: Spatial Self-Attention (per frame)
for t in 1..T:
    x_t = patches[t].flatten()          # (324, 768)
    x_t = FlashAttention(Q(x_t), K(x_t), V(x_t))  # intra-frame

# Step 3: Temporal Self-Attention (per patch position)
for (i,j) in grid(18,18):
    x_ij = [x_1[i,j], ..., x_T[i,j]]   # (T, 768)
    x_ij = FlashAttention(Q(x_ij), K(x_ij), V(x_ij))  # cross-frame

# Step 4: Spatial Average Pooling → per-frame descriptors
F = avg_pool_2d(x)                      # (T, 768)
```

复杂度：Divided Space-Time = O(TN² + NT²) vs Joint Space-Time = O(T²N²)。T=32, N=324 时，Divided ≈ 105K + 1K token pairs vs Joint ≈ 110M pairs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 提供 `TimesformerModel`（`attention_type='divided_space_time'`）。HORNet 使用 TimeSFormer-Tiny 变体从预训练权重初始化，在 GRPO 训练中与 MLP 联合微调。Spatial self-attention 用标准 `nn.MultiheadAttention` per-frame batch，temporal self-attention 用 `einops.rearrange` 重排维度后 batch attention。兼容 FlashAttention。HORNet 中 encoder + policy 共 <1M trainable params。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

## Reward-based Frame Selection for Video QA (基于奖励的帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reward-based Frame Selection 是指使用 downstream VLM 的 QA 质量作为 reward signal 来优化帧选择策略的方法范式。与 supervised frame selection（需 frame-level 标注或 pseudo-labels）和 heuristic frame selection（uniform/CLIP similarity 等固定规则）不同，reward-based 方法直接优化最终目标（QA accuracy），不需要 frame-level ground-truth。HORNet 是首次将 GRPO 用于 reward-based frame selection 的工作，其关键创新在于将 reward 信号从 VLM output 端反馈到 VLM input 端（帧选择策略），实现了"优化 VLM 看到什么"而非"优化 VLM 生成什么"。核心流程：policy 生成候选帧子集 → frozen VLM 回答问题 → reward 计算（F1-Lev = 0.1·F1_token + 0.9·EditSim）→ GRPO group-relative advantage + policy gradient 更新 policy 参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Reward-based Frame Selection via GRPO ===
for each (V, q, a):
    p = MLP(TimeSFormer(V))                # per-frame keep prob
    masks = [TopK_sweep(p, k) + Bernoulli(p)]  # K=8 candidates
    rewards = []
    for b in masks:
        a_hat = frozen_VLM(V[b==1], q)
        r = 0.1 * F1(a_hat, a) + 0.9 * EditSim(a_hat, a)
        rewards.append(r)
    r_bar, σ_r = mean(rewards), std(rewards)
    for i in 1..K:
        A_i = (rewards[i] - r_bar) / (σ_r + ε)
        loss -= A_i * log_π(b_i) / K
    update(θ_policy, θ_encoder)  # VLM never updated
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HORNet 训练配置：223,646 QA pairs, 两阶段 (MSVD+MSRVTT→NExT-QA), K=8, Adam lr=1e-4, batch_size=8。单卡 A100 40GB。VLM (Qwen3-VL-2B) 全程冻结。Reward 设计关键：lemmatized F1 + EditSim 比 exact match 对 minor lexical variations 更鲁棒。核心优势：(1) 直接优化 QA 质量；(2) 不需 frame-level annotation；(3) GRPO critic-free advantage estimation；(4) trained policy 可跨 VLM transfer。HORNet Table 4 证明 GRPO OOD generalization 优于 PPO 和 SFT。开源: https://github.com/ostadabbas/HORNet。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

## Hierarchical Attention in Vision Encoders（视觉编码器中的分层注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Attention in Vision Encoders 是 HiPrune 论文揭示的视觉编码器（Vision Transformer, ViT）内部注意力分布规律：不同深度的 ViT 层对不同语义层次的图像信息产生差异化关注——浅层注意力分散、中间层聚焦物体区域（object-centric）、深层编码全局上下文（global representation）。具体表现为三个阶段的渐进过渡：(1) 浅层（Layer 1~L/3）：注意力分布相对均匀，token 间注意力排名差异小，embedding 空间中高注意力 token 分布分散；(2) 中间层（Layer L/3~2L/3）：注意力向图像中的 main object 集中，top-10% 高注意力 token 与 COCO segmentation mask 的 IoU 达最大值（CLIP-L: 1× 在 L/2，SigLIP: 1× 在 L/2，DeiT: 1× 在 L/2，V-JEPA2: 1× 在 L/2）；(3) 深层（Layer 2L/3~L）：注意力从 object cluster 扩散至全图均匀分布，编码全局上下文信息，可作为有限 token budget 下的理想全局指标。该模式跨 CLIP、SigLIP、SigLIP2、DeiT、V-JEPA2 五种架构一致存在，与预训练数据或模型架构设计无关。t-SNE 投影显示注意力排名在相邻层之间呈现连续轨迹，证明注意力转移是渐进有序的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 token pruning 中使用分层注意力：
```
# === Hierarchical Attention for Token Selection ===
# 给定 ViT 各层 attention maps: all_attns = [A_1, ..., A_L]
# A_l: (H, N+1, N+1), H=num_heads

# 1. 从 object layer l (中间层) 提取 object-centric attention
mid_attn = all_attns[l][:, 1:, 1:].mean(0).sum(0)  # (N,)
# a_i^{[l]} = mean_h sum_n A_h[n, i]  每个 token 收到总关注度
anchor_idx = topk(mid_attn, k=N_a)               # 选物体区域 token

# 2. 从输出层提取 global attention  
deep_attn = all_attns[-1][:, 1:, 1:].mean(0).sum(0)
register_idx = topk(deep_attn, k=N_r)            # 选全局信息 token

# 3. 组合: Anchor (细节) + Buffer (空间邻居) + Register (全局)
retained = [anchor_idx, buffer(anchor_idx), register_idx]
```

Object layer l 的选择方法（dispersion-based searching）：
```
# 各候选层计算 top-k 高注意力 token 的平均 pairwise 距离
for l in candidate_layers:
    top_tokens = embeddings[l][topk(attention[l], k=K)]
    pairwise_dist[l] = mean(||t_i - t_j||_2 for all pairs)
# 选择 pairwise_dist 变化最剧烈的"临界点"层作为 object layer
# CLIP-L/14: l=9 (共 24 层), pairwise_dist 在 layer 9 处明显跃变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
分层注意力模式的使用：(1) 无需额外训练——直接从 ViT forward pass 中获取 attention map。(2) Object layer 通过 dispersion-based searching 确定（计算 top-K token pairwise distance 的拐点），一次确定后固定使用。(3) HiPrune 在 LLaVA-1.5 (CLIP-L/14, 24 layers) 中使用 layer 9，在 LLaVA-NeXT (CLIP-L/14) 中使用 layer 9，在 Qwen2.5-VL (SigLIP, 27 layers) 中使用 layer 16。(4) 该模式验证跨 CLIP-L/B、SigLIP、SigLIP2、DeiT、V-JEPA2 五类编码器，通过 COCO val2017 的 segmentation mask IoU 定量验证（Table 1: 中间层 top-10% token IoU 归一化值均为 1×，浅层/深层仅 0.26×~0.82×）。(5) 与 VAR-Turbo 中的 Learning Region/Inert Region 分区（knowledge_notes: score 44.8）形成补充——两者都揭示了 Transformer 层的 attention 行为随深度系统性地变化，但 HiPrune 聚焦于"attention 关注什么语义内容"，VAR-Turbo 聚焦于"attention 是否还有学习能力（高频信息保留程度）"。

涉及论文标题：
- HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models

## CLIP (Contrastive Language-Image Pre-training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CLIP (Contrastive Language-Image Pre-training) 是由 OpenAI 提出的跨模态基础模型，通过在数亿至数百亿 image-text pairs 上进行对比预训练，将图像和文本映射到共享表示空间。核心架构为双塔结构：Vision Encoder（通常为 ViT，如 ViT-B/16 86M、ViT-L/14 307M）和 Text Encoder（轻量自回归模型，约 1/3 ViT 参数量，上下文窗口限制为 77 tokens）。训练目标为对比损失：最大化匹配 image-text pair 的 cosine similarity，最小化非匹配 pair 的 similarity。CLIP 支持 zero-shot 分类（通过文本模板如 "a photo of the {classname}"）、图像-文本检索、跨模态特征提取。CLIP 的视觉特征被广泛用于 Multimodal LLMs (如 LLaVA、Qwen-VL) 和图像/视频生成模型 (如 Stable Diffusion 3、Wan) 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIP 训练流程
# Vision Encoder: ViT (Vision Transformer)
# Text Encoder: 轻量自回归 Transformer (~1/3 ViT params)
# 输入: N 个 (image, text) 对

def clip_training(images, texts):
    # 视觉编码
    I_f = ViT(images)          # [N, d], d=embedding_dim (e.g., 512/768/1280)
    I_e = L2_normalize(I_f)    # [N, d]

    # 文本编码 (causal attention, max 77 tokens)
    T_f = TextEncoder(texts)   # [N, d]
    T_e = L2_normalize(T_f)    # [N, d]

    # 对比损失 (双向)
    logits = I_e @ T_e.T * exp(t)  # [N, N], t 为可学习 temperature
    labels = arange(N)             # 对角线为正样本
    loss_i2t = CrossEntropy(logits, labels)
    loss_t2i = CrossEntropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss

# Zero-shot 分类推理
def clip_zero_shot_classify(image, class_names):
    I_e = L2_normalize(ViT(image))
    texts = [f"a photo of the {c}" for c in class_names]
    T_e = L2_normalize(TextEncoder(texts))
    scores = I_e @ T_e.T
    return argmax(scores)
```

Annotaions: `I_f`/`T_f` 为 raw features；`I_e`/`T_e` 为 L2 归一化后 embedding；`t` 为 temperature 参数（CLIP 原生使用可学习 logit scale，类似 temperature 的倒数）；对比损失同时计算 image→text 和 text→image 两个方向。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CLIP 的开源实现：OpenAI 官方 CLIP 在 https://github.com/openai/CLIP，预训练权重通过 HuggingFace (openai/clip-vit-base-patch16 等) 发布。后续改进包括：(1) SigLIP — 使用 sigmoid loss 替代 softmax 对比损失，支持更大 batch size；(2) EVA-CLIP/EVA02 — 改进训练技巧；(3) MetaCLIP — 数据筛选优化；(4) Long-CLIP — 扩展文本上下文长度。CLIP 的典型使用场景：(a) 作为多模态检索器的 backbone；(b) 作为 Multimodal LLM 的视觉编码器（LLaVA 系列用 CLIP-ViT-L/14-336 + MLP projector 连接 Vicuna）；(c) 作为扩散模型的文本编码器（SD3 用 CLIP 文本分支）。LLM2CLIP 论文在预训练 CLIP 基础上通过两阶段微调注入 LLM 能力，将 Text Encoder 替换为 CC fine-tuned LLM + Adaptor。

ReVisionLLM 使用 Frozen CLIP ViT-L/14 作为视频编码器，仅提取每帧 CLS token (768维) 而非全部 spatial tokens，显著降低视觉特征维度（每帧 1 token vs 257 tokens）。CLIP text encoder (12-layer) 用于提取 query 文本特征以参与 Hierarchical Adapter 的 Cross-Attention 对齐。CLIP similarity 被用作 baseline 排序方法（CONE ranking），但被 ReVisionLLM 的 LLM entropy-based 置信度取代。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Caption Contrastive Fine-tuning (CC Fine-tuning / LLM Embedding-ization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Caption Contrastive Fine-tuning (CC Fine-tuning) 是 LLM2CLIP 提出的 Stage 1 训练方法，目的是将 LLM 改造为适合 CLIP 跨模态对比训练的文本嵌入模型（即 "embedding-ization"）。核心问题：原始 LLM 的 token 输出层是 classification head（预测离散文本 token），其最后的 hidden state 对不同 caption 的语义可分离性极差——LLM2CLIP Table A1 显示 Llama3-8B 在 COCO caption-to-caption retrieval 上 Top-1 仅 5.2%，远低于 CLIP text encoder 的 25.2%。CC Fine-tuning 通过三项设计解决此问题：(1) **模型架构改造**：移除 causal attention mask → 启用 bidirectional attention；使用 average pooling 聚合所有 output tokens 获得句子嵌入；通过 LoRA (r=16, α=32) 参数高效微调。(2) **监督 SimCSE 对比损失**：使用同一图像的两个不同 caption 作为正样本对，以 in-batch 其他 caption 为负样本，最大化正样本对 cosine similarity。(3) **混合训练数据**：DreamLIP 30M captions + Echo Embeddings 1.5M 纯文本对，保证文本区分能力的泛化性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: CC Fine-tuning
# LLM: Llama 3.1 8B, LoRA (r=16, α=32)
# 移除 causal attention mask, 使用 full bidirectional attention
# 从 DreamLIP 数据中采样: (c_i, c_j) 为同一图像的两个 caption

def cc_finetune_step(llm, captions_pairs, optimizer):
    # c_i, c_j: tokenized captions, shape [B, L]
    # 构建系统 prompt: "Given a caption, retrieve a similar relevant caption."
    c_i = [system_prompt + cap for cap in c_i]
    c_j = [system_prompt + cap for cap in c_j]

    # 前向: bidirectional attention (无 causal mask)
    h_i = llm(c_i, causal_mask=False)  # [B, L, d_llm]
    h_j = llm(c_j, causal_mask=False)  # [B, L, d_llm]

    # Average pooling (而非 [EOS] token)
    e_i = h_i.mean(dim=1)  # [B, d_llm]
    e_j = h_j.mean(dim=1)  # [B, d_llm]

    # 监督 SimCSE 对比损失
    sim = cosine_similarity(e_i, e_j) / τ  # [B, B]
    labels = arange(B)  # 对角线表示 (e_i[k], e_j[k]) 为正样本对
    loss = CrossEntropy(sim, labels)

    loss.backward()
    optimizer.step()  # 仅更新 LoRA 参数
    return loss

# 训练配置: AdamW lr=2e-4, 300-step warmup, seq_len=512
#           有效 batch_size=2048, 1 epoch over 30M samples
#           32 A100 GPUs, bfloat16 + FlashAttention-2
```

Annotations: `d_llm` = 4096 (Llama 3.1 8B hidden dim)；`τ` 为 temperature 参数；in-batch negatives 利用 batch 内其他样本的 caption 作为负样本；DreamLIP 为每个图像提供多个 dense captions，因此可以采样两个不同 caption 作为正样本对。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CC Fine-tuning 的关键消融发现（LLM2CLIP Table 6/Table A5）：(1) Supervised SimCSE >> Unsupervised SimCSE >> MNTP alone；(2) Bidirectional attention 与 causal attention 性能相近（80.4 vs 80.0 Avg I2T），但 bidirectional 能更好建模文本双向关系；(3) Average pooling 优于 [EOS] token (80.4 vs 80.0)；(4) LoRA 是必需的——冻结 LLM + 仅训练 Adaptor 性能显著下降 (74.1 vs 80.4)；(5) MNTP + SimCSE 组合不优于 SimCSE alone。CC fine-tuned LLM 的分离能力超越原始 CLIP text encoder（Top-1 29.5% vs 25.2%）。CC Fine-tuning 后 LLM 的特征空间已具备充分的 caption 区分能力，可作为 Stage 2 中 CLIP 视觉编码器训练的有效文本监督信号。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## SimCSE (Simple Contrastive Learning of Sentence Embeddings)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SimCSE (Simple Contrastive Learning of Sentence Embeddings) 是一种通过对比学习训练句子嵌入的方法，由 Gao et al. (EMNLP 2021) 提出。核心思想：将同一个输入句子通过不同的 dropout mask 传递两次（unsupervised 版本），或使用标注的正样本对（supervised 版本），将这些变体作为正样本对，batch 内其他句子作为负样本，通过对比损失（NT-Xent loss）训练编码器使正样本对的嵌入彼此接近、与负样本嵌入远离。训练目标等价于最大化正样本对 cosine similarity 同时最小化与负样本的 similarity。LLM2CLIP 使用 supervised SimCSE 变体：正样本对为同一图像的两个不同 caption（由系统 prompt "Given a caption, retrieve a similar relevant caption" 构建），负样本为 batch 内其他 caption。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Supervised SimCSE for CC Fine-tuning
# 输入: batch of paired captions (c_i, c_j) from same image

def simcse_loss(encoder, batch_pairs, τ=0.05):
    # batch_pairs: [(c_1_i, c_1_j), ..., (c_B_i, c_B_j)]
    c_i_batch = [pair[0] for pair in batch_pairs]  # B captions
    c_j_batch = [pair[1] for pair in batch_pairs]  # B captions

    # 编码两个视图
    h_i = encoder(c_i_batch)  # [B, d]
    h_j = encoder(c_j_batch)  # [B, d]

    # L2 normalize
    z_i = h_i / ||h_i||_2   # [B, d]
    z_j = h_j / ||h_j||_2   # [B, d]

    # NT-Xent loss (symmetric)
    sim = z_i @ z_j.T / τ   # [B, B]
    labels = arange(B)       # (0,1,...,B-1)

    # 两个方向: i→j 和 j→i
    loss_i2j = CrossEntropy(sim, labels)
    loss_j2i = CrossEntropy(sim.T, labels)
    loss = (loss_i2j + loss_j2i) / 2

    return loss

# Unsupervised SimCSE (for comparison):
# def unsupervised_simcse(encoder, sentences):
#     # 同一句子经过两次带不同 dropout 的前向
#     z1 = encoder(sentences, dropout=True)  # 第1次 dropout
#     z2 = encoder(sentences, dropout=True)  # 第2次 dropout
#     # 其余计算相同
```

Annotations: `τ` 为 temperature (通常 0.05)；`d` 为 embedding 维度；supervised SimCSE 使用标注的正样本对（LLM2CLIP 中为同一图像的不同 caption），unsupervised 依赖 dropout 噪声产生正样本变体。LLM2CLIP Table A5 显示 supervised SimCSE (Avg I2T 80.4) 显著优于 unsupervised (59.2)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SimCSE 的使用方式：(1) Unsupervised: 仅需未标注文本，通过 dropout 增强产生正样本，适合大规模无标注数据场景。(2) Supervised: 需要标注的正样本对（如 NLI 数据、paraphrase 数据、或同一图像的不同 caption），性能优于 unsupervised。(3) 在 LLM2CLIP 中，supervised SimCSE 是 CC Fine-tuning 阶段的核心损失函数，其有效性源于：DreamLIP 为每张图像提供多个 dense captions → 天然的正样本对来源 → 训练 LLM 学习"两个相似语义的 caption 应具有相似的嵌入表示"这一能力 → LLM 特征空间获得 caption 语义可分离性。LLM2CLIP 的消融显示 SimCSE 是最关键的损失组分——仅用 MNTP 无 SimCSE 时 Avg I2T 从 80.4 降至 70.1。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA (Low-Rank Adaptation) 是 Hu et al. (ICLR 2022) 提出的参数高效微调方法，在冻结预训练权重的基础上，通过低秩分解矩阵注入可训练的 adapter 权重。核心数学：对于原始权重矩阵 W ∈ R^{d×k}，LoRA 添加低秩更新 ΔW = BA，其中 B ∈ R^{d×r}，A ∈ R^{r×k}，秩 r ≪ min(d, k)。前向计算变为 h = Wx + BAx = Wx + B(Ax)。初始化时 A 使用 Gaussian initialization，B 初始化为零矩阵，确保训练开始时 ΔW = 0 不影响原始模型输出。可应用于 Transformer 中任意线性层（通常为 Q、K、V、O 投影矩阵）。训练时仅更新 A 和 B，冻结 W，训练参数量从 d×k 降至 r×(d+k)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LoRA 在 LLM2CLIP Stage 1 中的应用
# 对 Llama 3.1 8B 的 Q、K、V、O 投影矩阵应用 LoRA

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # 冻结原始权重

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scaling = alpha / r  # LoRA scaling factor

    def forward(self, x):
        # Wx + (α/r) * B(Ax)
        frozen_out = self.linear(x)                  # [B, L, d_out]
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T  # [B, L, d_out]
        return frozen_out + self.scaling * lora_out

# LLM2CLIP Stage 1 配置:
#   r = 16, α = 32, scaling = 2.0
#   应用于 Llama 3.1 8B 的 attention Q/K/V/O projections
#   可训练参数: ~0.5% 的 LLM 总参数量
```

Annotations: `r` = rank (16)，控制低秩分解的表达能力和参数量的 trade-off；`α` = 32 控制 LoRA 更新幅度相对于原始权重的缩放；scaling factor `α/r` = 2.0。LoRA 训练完成后可通过 W' = W + (α/r)·BA 合并回原始权重，推理时无额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LoRA 在 LLM2CLIP 中应用于 Stage 1 (CC Fine-tuning)，对 Llama 3.1 8B 使用 r=16, α=32。LLM2CLIP Table A5 消融显示 LoRA 是 Stage 1 的关键：冻结 LLM + 仅训练 Adaptor 导致 Avg I2T 从 80.4 骤降至 74.1，说明 CC Fine-tuning 需要 LLM 内部权重的适配（而非仅依赖输出层的 adaptor）。在 Stage 2，LLM2CLIP 不 使用 LoRA 而是冻结全部 LLM 参数 + 添加独立 Linear Adaptor，因为：(1) 避免加载大型 LLM 到 GPU 显存中计算梯度；(2) 配合 Offline-loading 策略预计算文本嵌入；(3) Adaptor 本身提供足够的跨模态映射能力。一般使用方式：(a) 单 LoRA 微调 (r=8~64)；(b) QLoRA (4-bit 量化 + LoRA) 进一步降低显存；(c) 多个 LoRA module 对应多个下游任务实现快速切换。

LongLive 将 LoRA 应用于 Wan2.1-T2V-1.3B 的 streaming long tuning，使用 rank=256（约 350M trainable params, ~27% of 1.3B），配合 DMD 蒸馏。Table A 消融 rank 32/64/128/256/512 vs Full Model 的 VBench-Long 分数表明 rank=256 达到最佳，较全微调节省 73% 参数/优化器状态。训练在 64 H100 GPU 上进行约 12 小时（32 GPU-days）。

ReVisionLLM 使用 LoRA 进行两阶段渐进式训练：Stage 1 训练 LoRA_A（r=64, α=128）用于短片段精确边界定位，Stage 2 训练独立 LoRA_B（r=64, α=128）用于长视频层次化处理，Hierarchical Adapter 权重在 Stage 2 冻结。两个 LoRA 模块分别优化不同时间尺度的定位能力——底层 hierarchy 使用 LoRA_A（dense features, 帧级精度），上层 hierarchy 使用 LoRA_B（sparse features, 段级精度）。训练配置：8×A100 GPUs, total batch size 128, AdamW optimizer, cosine LR decay, warmup ratio 0.03, LR=1×10⁻⁴。

SlowFast-VGen 提出 TEMP-LORA 变体，将 LoRA 用于推理时训练（而非传统 fine-tuning）：在长视频生成推理过程中，每轮迭代生成新 chunk 后，将输入输出 latent 拼接加噪，通过去噪训练更新 TEMP-LORA 参数 Θ_i（rank=32），将整个生成轨迹的情节记忆存储在 LoRA 参数中。与标准 LoRA 的关键区别：(a) 推理时训练——非离线 fine-tuning，每轮推理都更新 Θ；(b) 无文本条件——训练时不输入文本 prompt，专注于轨迹记忆；(c) 逐 chunk 累积——Θ 参数随生成进度逐步编码更多场景信息，后续 chunk 生成时能回忆之前场景（如回访同一位置时场景保持一致）。推理 overhead: +6.8% 时延，+3.7% 显存。消融显示改进后的 TEMP-LORA（SCuts=0.37）优于原版 TEMP-LORA 格式（SCuts=0.55）。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
- LongLive__Real-time_Interactive_Long_Video_Generation
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

## FuseMix Adaptor (Inverted Bottleneck MLP Adaptor / 倒瓶颈MLP适配器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FuseMix Adaptor 是 LLM2CLIP Stage 2 中连接 CC fine-tuned LLM 与 CLIP Vision Encoder 的关键组件，采用 FuseMix (Vouitsis et al., CVPR 2024) 提出的 inverted bottleneck MLP 架构。由 4 层倒瓶颈线性块组成，每层结构为 Linear(d_in → d_hidden) → GeLU → Linear(d_hidden → d_in) + residual connection。最后通过一个最终投影层将 LLM 的 hidden dimension (4096 for Llama 3.1 8B) 映射到 CLIP 的 embedding dimension (1280)。总参数量约 67.1M。Adaptor 放置于 LLM 输出之后、CLIP contrastive loss 之前，作为"可学习的桥梁"，将 LLM 的文本嵌入空间对齐到 CLIP Vision Encoder 的表示空间。由于 Adaptor 完全独立于 LLM，配合 Offline-loading 策略可在训练时不加载 LLM 到 GPU 显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FuseMix Adaptor 结构 (4-layer inverted bottleneck MLP)
# 输入: LLM avg-pooled sentence embedding ∈ R^4096
# 输出: embedding ∈ R^1280 (与 CLIP visual embedding 对齐)

class FuseMixAdaptor(nn.Module):
    def __init__(self, d_in=4096, d_hidden=8192, d_out=1280, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_in, d_hidden),   # 4096 → 8192 (expand)
                nn.GELU(),
                nn.Linear(d_hidden, d_in),   # 8192 → 4096 (project back)
            ))
        self.final_proj = nn.Linear(d_in, d_out)  # 4096 → 1280

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)           # expand → activate → project
            x = x + residual        # residual connection
        x = self.final_proj(x)     # project to CLIP embedding space
        return x  # [B, 1280]

# Stage 2 训练中 Adaptor 的使用:
# LLM 冻结 → precomputed sentence embeddings[4096] → Adaptor → [1280]
#                                                                  ↓
#                                                        CLIP contrastive loss
#                                                                  ↑
#                           ViT(image) → visual embedding [1280] ─┘
```

Annotations: `d_hidden=8192` 为倒瓶颈的扩展维度（2× d_in）；residual connection 保证梯度流动和训练稳定性；Adaptor 总参数量 67.1M = 4 × (4096×8192 + 8192×4096) + 4096×1280 ≈ 4 × 67.1M + 5.2M。LLM2CLIP Table 8/Table A7 消融显示：4-layer Linear Adaptor 性能 (80.4/77.9) 与 1-layer Transformer Adaptor (80.2/77.3) 相当，选择更简单的 Linear 结构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FuseMix Adaptor 的消融发现（LLM2CLIP Table A7）：(1) 无 Adaptor → 1-layer → 2-layer → 4-layer 性能递增 (78.3→79.2→80.1→80.4 Avg I2T)；(2) 4-layer Linear Adaptor (80.4) 与 1-layer Transformer Adaptor (80.2) 性能相当，Linear 结构更简单；(3) Stage 1 中是否使用 Adaptor 对最终结果影响微小（80.4 vs 80.5）。LLM2CLIP 默认配置：Stage 1 不使用 Adaptor (直接对 LLM 输出做 avg pooling)，Stage 2 使用 4-layer Linear Adaptor。一般使用 FuseMix Adaptor 的场景：需要将一个预训练模型的 frozen embedding 投影到另一模型的 embedding space 时，通过 MLP 层级联提供足够的非线性映射能力。与 LoRA 的区别：Adaptor 是独立模块位于模型输出之后，LoRA 是注入到模型内部权重的低秩分解。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## Offline Embedding Precomputation for LLM-based Training (Offline-loading / 离线嵌入预计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Offline Embedding Precomputation (Offline-loading) 是 LLM2CLIP Stage 2 提出的训练效率优化策略：在 CLIP 跨模态对比训练开始之前，用冻结的 CC fine-tuned LLM 对所有训练数据的文本 caption 预先计算文本嵌入并存入磁盘；训练时直接从磁盘加载预计算嵌入，通过 Adaptor 参与对比损失计算，完全避免将 LLM 加载到 GPU 显存中。核心优势：(1) LLM 推理从每个训练 epoch 执行一次 → 整个训练过程仅执行一次；(2) 训练时 GPU 显存无需容纳 LLM（8B 参数模型 bf16 约 16GB + optimizer states），可将节省的显存用于增大 batch size；(3) batch size 从 LLM LoRA 时 ~704 → offline-loading 时 16384（提升 23×）；(4) 训练时间从 17h (LLM LoRA) → 1.3h (offline-loading)。该策略可行性的前提是 LLM 参数在 Stage 2 完全冻结，嵌入为固定值不随训练变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Phase 1: Offline Precomputation (一次性)
def precompute_text_embeddings(llm, all_captions):
    llm.eval()
    embeddings = {}
    with torch.no_grad():
        for idx, caption in enumerate(all_captions):
            tokens = tokenize(caption)  # [1, L], seq_len ≤ 512
            h = llm(tokens, bidirectional=True)  # [1, L, 4096]
            emb = h.mean(dim=1)  # avg pooling → [1, 4096]
            embeddings[idx] = emb.cpu()  # 存入 CPU/磁盘
    return embeddings  # 持久化到磁盘

# Phase 2: Stage 2 Training with Offline-loading
def stage2_training_with_offline(caption_emb_path, images):
    precomputed = load(caption_emb_path)  # 加载预计算嵌入

    for batch_images, batch_ids in dataloader:
        # 视觉编码: ViT 在 GPU 上运行
        v_feat = ViT(batch_images)                      # [B, 1280]

        # 文本编码: 从磁盘加载预计算 LLM 嵌入 → Adaptor
        pre_emb = precomputed[batch_ids].to(device)     # [B, 4096]
        t_feat = Adaptor(pre_emb)                        # [B, 1280]

        # LLM 完全不加载到 GPU显存
        # CLIP contrastive loss
        loss = clip_loss(v_feat, t_feat)
        loss.backward()
        optimizer.step()  # 仅更新 ViT + Adaptor
```

Annotations: 预计算使用 32 A100 GPU 对 LLM 做单次前向推理；训练时 batch size 可达 16384 (2 nodes × 8 A100 40GB)；仅 ViT (307M~428M) + Adaptor (67.1M) 在 GPU 上训练，显存占用远低于同时加载 LLM (8B)；LLM 参数、optimizer states 完全不出现在训练显存中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLM2CLIP Table A4 的 efficiency analysis 对比了三种策略的 trade-off：(1) LLM LoRA 在线训练: batch_size=704, 训练时间 17h, Avg I2T/T2I 85.4/82.5；(2) LLM Frozen + Linear Adaptor 在线: batch_size=4096, 5.5h, 83.9/82.1；(3) LLM Frozen + Adaptor + Offline-loading: batch_size=16384, 1.3h, 85.9/83.3。Offline-loading 不仅训练最快，性能也最高——更大的 batch size 对对比学习有益（更多 in-batch negatives）。适用条件：(a) 文本编码器参数冻结；(b) 训练数据在训练前完全已知（不需要在线生成文本）；(c) 存储空间足够容纳所有预计算嵌入（15M captions × 4096 × 2 bytes (bf16) ≈ 115GB，可控）。不适用场景：训练中需要实时生成或动态变化文本数据的情况。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## Bidirectional Attention for LLM-based Text Encoder (LLM文本编码器的双向注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bidirectional Attention for LLM-based Text Encoder 是 LLM2CLIP 在 CC Fine-tuning 阶段对 LLM 架构的关键修改：移除 LLM 原生的 causal (autoregressive) attention mask，改为完全的 bidirectional attention，使每个 token 可以 attend 到序列中所有其他 token（前后双向）。原始 LLM 使用 causal mask（下三角矩阵）以确保自回归生成能力——每个 token 只能看到自身及之前的 token。但在纯编码场景（文本嵌入提取），生成能力不需要，双向建模能更充分地捕获文本的双向语义关系。具体实现：将 Transformer 层的 attention mask 从 causal（lower triangular）替换为全 1 矩阵 [1, L, L]（仅保留 padding mask）。LLM2CLIP 消融显示 bidirectional 与 causal attention 性能相近（Avg I2T 80.4 vs 80.0），但 bidirectional 理论上有更好的文本理解能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Causal Attention (原始 LLM)
def causal_attention(Q, K, V):
    # Q, K, V: [B, H, L, d_h]
    scores = Q @ K.transpose(-2, -1) / sqrt(d_h)  # [B, H, L, L]

    # Causal mask: 下三角矩阵, token i 只能 attend token 0..i
    causal_mask = torch.tril(torch.ones(L, L))     # 下三角=1, 上三角=0
    scores = scores.masked_fill(causal_mask == 0, -inf)

    attn = softmax(scores, dim=-1)
    return attn @ V

# Bidirectional Attention (LLM2CLIP Stage 1 & 2)
def bidirectional_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1) / sqrt(d_h)  # [B, H, L, L]

    # 仅使用 padding mask (无 causal mask)
    # pad_mask: [B, L], 1=有效token, 0=padding
    pad_mask_expanded = pad_mask[:, None, None, :]  # [B, 1, 1, L]
    scores = scores.masked_fill(pad_mask_expanded == 0, -inf)

    attn = softmax(scores, dim=-1)  # 所有 token 互相 attend
    return attn @ V

# 在 Llama 3.1 8B 中启用 bidirectional attention:
# 将 attention_mask 从 causal + padding → padding only
# 对应的 FlashAttention-2 调用中设置 is_causal=False
```

Annotations: causal mask 限制每个 token 只能看过去（prevents information leakage from future）；bidirectional attention 让 [CLS]-like pooling 可以汇聚双向语境信息。Bidirectional attention 在前向时可能导致 attention scores 对称化，但移除 causal mask 本身不改变计算量（仍为 O(L^2)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLM2CLIP 的实现中：Stage 1 CC Fine-tuning 和 Stage 2 的 LLM 编码均使用 bidirectional attention。具体实现方式：调用 HuggingFace Transformers 的 `model.forward(input_ids, attention_mask=padding_mask)` 时，如果模型原生有 causal mask，可通过 `use_cache=False` 和设置 `output_attentions=False` 配合自定义 attention mask 覆盖。或直接修改模型 config 中的 `is_causal` 属性。LLM2CLIP 消融 (Table 6/Table A5) 显示：bidirectional vs causal 差异很小（80.4 vs 80.0 Avg I2T），原因是 caption 文本通常较短（≤ 512 tokens），且双向语义信息已在 LLM 预训练的 causal objective 中被隐式学习（通过多层堆叠间接获取反向信息）。Bidirectional attention 的使用场景：任何将自回归 LLM 转为纯编码器（embedding model）的场景，如 LLM2Vec、NV-Embed-v2 均采用此策略。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## MNTP (Masked Next Token Prediction / 掩码下一Token预测)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MNTP (Masked Next Token Prediction) 是 LLM2Vec (BehnamGhader et al., 2024) 提出的一种 LLM 训练方法，灵感来源于 BERT 的 Masked Language Modeling。核心思想：在序列中 mask 掉特定 token，让 LLM 预测被 mask token 的下一个 token（而非被 mask token 本身），以对齐 LLM 的 next-token prediction 预训练惯例。与 BERT MLM 的区别：BERT 预测被 mask 的 token 本身（基于双向上下文），MNTP 预测 mask 位置之后的 token（利用 LLM 原生 next-token prediction 能力）。LLM2CLIP 评估了 MNTP 在 CLIP 跨模态场景中的效果：单独 MNTP (Avg I2T 70.1) 远低于 SimCSE (80.4)，MNTP + SimCSE 组合 (79.7) 也不优于 SimCSE alone，因此 LLM2CLIP 默认不使用 MNTP。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MNTP 训练流程
# 输入: 文本序列 "a cat sitting on a mat"
def mntp_training(llm, text_sequences):
    # 1. Mask tokens (e.g., mask "sitting")
    # tokens:    [a, cat, [MASK], on, a, mat]
    # labels:    [-,  -,  sitting, -, -,  -]  (只计算 mask 位置)
    masked_seq, labels = apply_mask(text_sequences)

    # 2. LLM 前向 (causal attention)
    logits = llm(masked_seq)  # [B, L, vocab_size]

    # 3. 仅在 mask 位置计算 next-token prediction loss
    # token "sitting" 的预测基于其之前的 tokens [a, cat, [MASK]]
    loss = CrossEntropy(logits[mask_positions], labels[mask_positions])
    return loss

# LLM2CLIP 中 MNTP 评估:
# 与 SimCSE 对比 (Table A5):
#   MNTP alone:            Avg I2T/T2I = 70.1/67.0
#   Unsupervised SimCSE:   Avg I2T/T2I = 59.2/57.7
#   Supervised SimCSE:     Avg I2T/T2I = 80.4/77.9
#   MNTP + SimCSE:         Avg I2T/T2I = 79.7/77.2
```

Annotations: MNTP 的 mask 位置由随机选择或启发式策略确定；MNTP 的优势在纯文本任务中已被 LLM2Vec 验证有效，但在跨模态场景（需要 caption 间语义区分能力）中，SimCSE 的对比式训练更直接地针对 embedding 可分离性优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MNTP 在 LLM2Vec 中的使用：通过 MNTP 激活 LLM 的双向上下文理解能力（尽管仍用 causal attention），使 LLM 的 hidden states 对上下文更敏感。LLM2CLIP 的评估显示 MNTP 在跨模态场景效果有限：(1) MNTP 训练目标是 token-level prediction，而 CLIP 需要的是 sentence-level 语义可分离性；(2) SimCSE 通过 sentence-level 对比损失直接优化 embedding 空间的分离度，对 CLIP 跨模态对比训练更匹配；(3) MNTP + SimCSE 组合未带来额外收益（79.7 vs 80.4），说明 SimCSE 已充分激话 LLM 的文本区分能力。MNTP 可能适用于：需要 LLM 输出 token 对上下文中词级语义更敏感的场景（如信息检索、文本匹配），而在跨模态 embedding 场景中 SimCSE 是更优选择。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

## SigLIP / Sigmoid Loss for Language-Image Pre-training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP (Sigmoid Loss for Language-Image Pre-training) 是 Zhai et al. (ICCV 2023) 提出的 CLIP 训练变体，将 CLIP 原生的 softmax 对比损失替换为 sigmoid loss。核心差异：CLIP 使用 softmax + cross-entropy 对 batch 内所有 image-text pairs 进行全局归一化对比（需要大 batch size 提供足够多的负样本）；SigLIP 将每个 image-text pair 独立处理，用 sigmoid 二元分类器判断 pair 是否匹配，负样本从 batch 内其他样本中获取，免除全局 softmax 归一化。优势：(1) batch size 不再受限于 softmax 的分母精度要求，可支持更大 batch；(2) 训练更稳定；(3) 在处理大规模数据时性能更优。后续版本 SigLIP-2 (Tschannen et al., 2025) 进一步扩展至多语言支持（109 语言）、改进语义理解、定位和 dense features，使用 12B alt-text pairs 训练。LLM2CLIP 使用 SigLIP2-SO/14 (428M) 作为 SOTA baseline，在 40B image-text pairs 上预训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIP Loss (softmax-based, 用于对比)
def clip_loss(I_e, T_e, t):
    # I_e, T_e: [B, d] L2-normalized embeddings
    logits = I_e @ T_e.T * exp(t)  # [B, B]
    labels = arange(B)              # diagonal = positive pairs
    loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
    # 分母对所有 B 个 texts/images 做全局 softmax
    return loss

# SigLIP Loss (sigmoid-based, 用于对比)
def siglip_loss(I_e, T_e, t, b):
    # I_e, T_e: [B, d] L2-normalized embeddings
    logits = I_e @ T_e.T * exp(t) + b  # [B, B], b = learnable bias

    # 对角线 = positive (label=1), 非对角线 = negative (label=-1)
    labels = 2 * eye(B) - 1  # [B, B]: 对角=1, 其他=-1

    # Sigmoid binary cross-entropy, 每个 pair 独立计算
    loss = -log(sigmoid(labels * logits)).sum() / B
    # 无需全局 softmax 归一化
    return loss
```

Annotations: `t` = log-temperature (可学习)；`b` = learnable bias for sigmoid；softmax 的全局归一化使负样本数量必须充足（batch size → 大），sigmoid 的独立处理解耦了 batch size 与负样本质量的关系；SigLIP 实验中使用 batch size 32K~64K 且性能稳定，而 softmax CLIP 在小 batch 下性能退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SigLIP 的开源实现：(1) Google 官方: https://github.com/google-research/big_vision；(2) HuggingFace: google/siglip-so400m-patch14-224, google/siglip2-so400m-patch14-224；(3) OpenCLIP 复现。LLM2CLIP 使用 SigLIP2-SO/14 作为最强 baseline —— 在 40B data 预训练的基础上，LLM2CLIP 仅用 60M fine-tuning data 即实现 Flick30K +1.0/+1.9 (I2T/T2I)、long-caption +14.8/+15.8、multilingual +11.9/+15.2 的提升。SigLIP 的适用场景：大规模 CLIP 式预训练，特别是训练数据量极大（10B+ pairs）且需要稳定训练时。Sigmoid loss 在 batch size ≥ 16K 时表现出最优性能。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

## Streaming Causal Attention Masks (SCAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SCAM (Streaming Causal Attention Masks) 是 LiveStar 论文提出的流式视频-语言对齐训练策略中的核心注意力掩码机制。在标准 causal attention mask（因果注意力掩码，即每个 token 只能看到自己和之前的 token）基础上，SCAM 通过额外的掩码约束实现流式视频的增量式训练：(1) 对当前语义片段（semantic clip）中已生成的字幕 token 施加 -inf 掩码，防止模型通过"抄写"同一片段中已输出的字幕来 trivial copying；(2) 保留前一语义片段的终端字幕 token 的可见性，使模型感知场景语义边界；(3) 掩蔽之前所有非终端字幕 token，避免信息泄露。SCAM 使得模型能够在交错帧-字幕序列（interleaved frame-caption sequences）上自回归训练，逐步学习从可变长度视频前缀生成时间一致的字幕，同时保持预训练的视觉-语言对齐范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCAM 位于 LiveStar 训练 pipeline 的 attention 计算阶段，在标准 causal mask 之上叠加语义片段级别的结构约束：

```
# 输入: 交错帧-字幕序列
# C_k = 第k个语义片段, 帧 t_m..t_n 共享字幕 [Cap^k]
# 序列 = [Frm^{t_1}, Cap^1_1, ..., Frm^{t_i}, Cap^k_i, ...]

def build_scam_mask(seq_len, semantic_clips, caption_positions):
    # 标准因果mask：M[i,j] = 0 if j <= i else -inf
    mask = causal_mask(seq_len)
    
    for clip_k in semantic_clips:
        for pos in caption_positions_in_clip(clip_k):
            for other_pos in same_clip_earlier_captions(pos):
                # 掩蔽同一clip中已生成的字幕token（防止copying）
                mask[pos, other_pos] = -inf
            for other_clip in prev_clips_before(clip_k):
                for other_pos in non_terminal_captions(other_clip):
                    # 掩蔽之前clips的非终端字幕
                    mask[pos, other_pos] = -inf
            # 终端字幕不掩蔽 → 传递场景边界信息
    
    return mask

# 训练时在 attention 中使用 SCAM
Attention(Q, K, V, mask=scam_mask)
```

训练目标：max P([Cap_i^k] | [Ctx^{<t_i} {Mask^{≤t_i}}], [Frm^{t_i}])

关键设计：Mask 的稀疏模式确保 (a) 当前字幕的生成不被同一clip的已有字幕污染，(b) 可引用前一clip的终端字幕获知场景转换，(c) 所有视觉帧始终可见。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SCAM 在 HuggingFace Transformers 框架中实现：在模型 forward 时传入自定义 4D attention mask [batch, 1, seq_len, seq_len]，与标准 causal mask 叠加后送入 scaled_dot_product_attention。LiveStar 训练时使用 InternVideo2.5 的 InternViT + InternLM2.5-7B 架构，SCAM mask 在训练 loop 的每个 step 根据语义片段边界动态构建。训练配置：每序列最多 8192 tokens，仅对 assistant response tokens 计算 cross-entropy loss。M 个 paraphrase captions 池中随机采样以防止重复字幕导致的过拟合。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

## Streaming Video-Language Alignment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming Video-Language Alignment（流式视频-语言对齐）是 LiveStar 论文提出的训练范式，旨在替代传统 EOS-based 在线 Video-LLM 的训练方式。传统方法（VideoLLM-online 等）训练模型在非响应帧输出 EOS token，破坏了预训练的视觉-语言对齐（vision-language alignment，即每个视觉输入应对齐有意义的语言输出）。流式视频-语言对齐的核心创新是将训练目标重构为：对每个语义片段 C_k = {t_m, ..., t_n}，所有帧共享相同的语义字幕，训练目标为 `max P([Cap^k] | [Ctx^{<t_i}], [Frm^{t_i}])`（而非 `max P(EOS | ...)`）。这通过 SCAM（Streaming Causal Attention Masks）实现，保证每个视觉帧始终与有意义的语言内容对齐，与 pretraining 的 image-text pair 对齐范式一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两阶段训练 pipeline：
```
Phase I: Temporal Alignment Pretraining (63K samples)
  ActivityNet Captions (9K) + Shot2Story (33K) + Ego4D (20K) + MVBench (1K)
  → 构建交错帧-字幕序列
  → SCAM attention mask
  → 标准 autoregressive cross-entropy loss (仅assistant tokens)

Phase II: Multi-Task Online Adaptation (20K OmniStar samples)
  5 tasks: RNG / OTG / FDQ / COQ / MIQ
  → Task-specific adapters
  → Simultaneous multi-objective alignment
```

全微调配置：Vision Encoder (InternViT) 冻结，MLP Projector + LLM (InternLM2.5-7B) 可训练。AdamW (lr=4×10⁻⁵, β1=0.9, β2=0.999, weight decay=0.05)，cosine LR schedule with warmup ratio=0.03，effective batch size=32（per-device bs=1 × gradient accumulation 4 × 8 GPUs）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖于：(1) 细粒度语义片段标注 — 需要将视频划分为带时间戳的字幕段落（OmniStar 使用 semi-automated pipeline 进行 temporal dense annotation），每个段落的帧属于同一语义clip；(2) 释义池 — 为每个语义clip准备 M 个 paraphrased captions（M=1 默认，M=3 时 SemCor +1.57% 但 TimDiff +3.14%）；(3) SCAM mask 生成 — 需要在 DataLoader 中实时构建，随 batch 一起送入模型。训练数据格式为 chat-style interleaved format。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

## Perplexity-based Response Gate (SVeD Verification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perplexity-based Response Gate 是 SVeD (Streaming Verification Decoding) 推理框架中的核心决策机制。它通过监控当前字幕 [Dec] 在 incoming frame 上下文下的 perplexity 变化来决定是否触发新字幕生成。具体地，在时刻 t_j 收到新帧后，通过单次 forward pass 计算 PPL^{t_j}([Dec]) = sqrt[N]{1/P([Dec] | [Ctx^{≤t_j}], [Frm^{t_j}])}。若 PPL^{t_j} > α · PPL^{t_i}（α 默认 1.03），说明新帧的视觉内容与当前字幕语义不匹配（perplexity 上升），应激活解码 gate 生成更新字幕；否则保持沉默。这种方法将"响应vs沉默"的决策从离散的 EOS 分类问题转化为连续的 perplexity 变化检测问题，避免了 EOS token 的词汇表污染和全帧解码开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在推理 pipeline 中的位置：每个 incoming frame → Vision Encoding → PPL Verification → Gate Decision → (if activated) Full Decoding / (else) Silent Pass。

```
# PPL Verification伪代码
def verify_and_gate(Frm^{t_j}, Ctx, Dec, alpha=1.03):
    # 1. 追加新帧到上下文
    Ctx_new = Ctx + [Frm^{t_j}]
    
    # 2. 单次forward pass计算perplexity
    # model.forward() 输出: logits for all positions
    logits = model.forward(Ctx_new)[-len(Dec):]  # 仅取Dec位置
    # PPL = exp(-1/N Σ log P(token_i))
    log_probs = log_softmax(logits, dim=-1)
    token_log_probs = gather(log_probs, Dec_token_ids)
    PPL_new = exp(-mean(token_log_probs))
    
    # 3. Gate Decision
    if PPL_new > alpha * PPL_reference:
        return ACTIVE_DECODE  # 激活解码
    else:
        return SILENCE  # 保持沉默
```

perplexity 计算仅需一次 forward pass（约 1ms），而完整 decoding 需要逐 token 生成（约 10-50ms per token）。在 1 分钟视频 @3fps 含 5 个语义变化段的场景中，180 帧仅触发 5 次完整 decoding + 180 次 verification passes，比 EOS-based 方法（180 次完整 decoding）减少约 97% 的 decoding 开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Alpha 阈值调优：在 OmniStar-RNG 上 α ∈ [1.0, 1.1] 区间搜索，发现 1.02-1.04 最优，选 1.03 为最终配置。α 越大 → 更频繁触发解码（更敏感），TimDiff↓ 但 TimRedun↑；α 越小 → 更保守（更少解码），TimRedun↓ 但 TimDiff↑。实现时需要在 SVeD 循环中维护 PPL_cache（每次成功解码后更新 reference PPL），swap_last_two(Ctx) 操作保持沉默帧的 Dec 在上下文末尾以维持时间一致性。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

## Peak-End Memory Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Peak-End Memory Compression（峰值-终点记忆压缩）是 LiveStar 中受认知心理学 "Peak-End Rule"（峰值-终点规则，由 Kahneman 等提出）启发的长视频记忆管理策略。人类在回顾经历时倾向于优先记住"峰值时刻"（最 intense 的体验）和"终点时刻"（最近的体验），而非均匀地记住所有时刻。LiveStar 将这一原理应用于在线视频推理：利用 SVeD 预计算的每帧 perplexity 作为"语义重要性"代理指标（低 PPL = 高重要性 = "峰值"），保留每个语义片段的终端字幕作为"终点"摘要，对超出窗口 W（默认 40 帧）的旧帧按概率剪枝，删除概率正比于 PPL_relative（该帧 PPL 与所在 clip 内最小 PPL 之比）和 elapsed_time。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在推理循环中的位置：
```
def peak_end_prune(frames, PPL_cache, window_W=40, current_time):
    kept = []
    for f in frames:
        if len(frames) - f.index <= window_W:
            kept.append(f)  # 窗口内帧全部保留
        else:
            # P_delete ∝ relative_PPL × elapsed_time
            ppl_rel = f.PPL / min(PPL_cache[f.clip])
            time_factor = (current_time - f.timestamp) / total_duration
            p_delete = ppl_rel * time_factor
            if random() >= p_delete:
                kept.append(f)
    return kept
```

效果（OmniStar-RNG）：Peak-End 压缩下 SemCor=3.19, TimDiff=1.91, FPS=3.82，优于 Uniform Dropout (SemCor 3.04) 和 FIFO Forgetting (SemCor 3.07, TimDiff 2.09)。关键原因：(1) Uniform 随机删除可能丢弃关键帧（-4.70% SemCor）；(2) FIFO 丢弃最早的历史事件字幕，损害 long-range temporal reasoning（TimDiff +9.42%）；(3) Peak-End 基于语义重要性选择性保留，同时保留终端字幕维护 narrative coherence。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现需要两个先决条件：(1) SVeD 在每个 decoding step 计算 PPL 并存储，作为帧重要性评分来源；(2) 语义片段边界已知（来自训练数据或在线检测）。剪枝操作发生在 KV cache 层面：被剪枝的帧对应的 KV cache 条目从 GPU HBM 中释放，配合 Streaming KV Cache 的双级缓存机制维持 cache 一致性。配置：W=40 frames (约 13.3s @3fps)，对 10+ 分钟视频持续推理时将 KV cache 大小维持在可控范围。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

## Hybrid Mamba-Transformer Architecture / 混合Mamba-Transformer架构

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Mamba-Transformer 架构是一种混合 LLM backbone 设计，在同一个模型中将 Transformer Attention 层与 Mamba (Structured State Space Model, SSM) 层按特定比例交替排列。核心动机：Transformer attention 具有 O(N²) 的计算复杂度和 KV cache 内存消耗（N 为序列长度），长上下文场景下效率极低；Mamba SSM 具有 O(N) 的线性复杂度且无需 KV cache，但在 In-Context Learning (ICL) 和复杂检索/推理任务上能力弱于 Transformer attention。Hybrid 架构通过在层维度上混合两者，利用 Transformer attention 层保留 ICL 和上下文检索能力，利用 Mamba 层的线性复杂度降低整体计算开销，达到效率与效果的平衡。LongLLaVA 使用 4 组 hybrid stack，每组以 Attention:Mamba = 1:7 的比例排列（即每 8 层中 1 层 Transformer Attention + 7 层 Mamba SSM），配合 MoE 每隔一层集成 (16 experts, top-2 gating)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hybrid LLM Backbone (Attention:Mamba = 1:7, 4 stacks)
for stack in range(4):
    for layer in range(stack_size):
        idx = stack * stack_size + layer
        if idx % 8 == 0:                # Attention layer (1/8)
            H = RMSNorm(H); Q,K,V = W_Q(H),W_K(H),W_V(H)
            H = H + FlashAttention(Q,K,V, causal=True)
        else:                            # Mamba layer (7/8)
            H = RMSNorm(H)
            H = H + MambaBlock(H)        # selective scan
        if layer % 2 == 0:               # MoE layer
            gate = softmax(router(H)); w,idx = topk(gate,k=2)
            H = H + sum(w[i]*expert[idx[i]](H) for i in [0,1])
        else:
            H = H + SwiGLU_FFN(H)
```

Annotations: 每 8 层 1 Attention + 7 Mamba = Quasi-Linear 复杂度；仅 attention layers 需要 KV cache (12.5%)；MoE 16 experts top-2 gating。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA Table 1 效率对比 (100K tokens): Hybrid Prefill 25.5s / TP 37.6 / Mem 79.1GB vs Transformer 34.0s / 14.7 / 79.4GB vs pure Mamba 14.3s / 72.6 / 32.1GB。VL-ICL 5-shot: Hybrid 61.3 vs Mamba 53.2 vs Transformer 58.9。1:7 ratio 在 1.3B 模型上验证为最优（Table 2: 1:7 vs 1:3 性能差距极小但 1:7 更高效）。参考：Jamba 使用类似 hybrid 设计，256K tokens 仅 4GB KV cache。

**Hybrid Mamba-Transformer 在长视频理解中的设计 (TimeViper)**：TimeViper 采用 Nanov2-9B backbone（27 Mamba-2 + 4 Self-Attention + 25 MLP），Mamba:Attention ratio ≈ 17:1（27/56 Mamba + 4/56 Attention + 25/56 MLP）。Self-attention 层仅 4 层（占 7.1%），集中在特定深度位置（如第 14 层为第一个 attention 层）。这使得 LLM 整体接近 quasi-linear 复杂度，GPU 内存和 prefilling 时间随帧数近似线性增长（而非二次）。具体效果：vanilla 模型在 128 frames 即 OOM，+ToMe 扩展至 ~5K frames，+ToMe+TransV 扩展至 10K+ frames。在 4096 frames 时，TransV 减少 54.8% 内存和 15.7% prefilling time。定性分析揭示 Mamba 层 attention 模式多样性（sparsity/locality/globality），self-attention 层展示 "attention sink" 现象（大量 attention 集中在前几个 token 上），且 vision token 的 attention 随深度递减，印证了 vision-to-text information aggregation 现象。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding

## Mamba / Structured State Space Model (SSM) / 结构化状态空间模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba 是一种基于 Structured State Space Model (SSM) 的序列模型架构（Gu and Dao, 2023），作为 Transformer attention 的替代方案，以 O(N) 线性复杂度处理序列。核心机制是 selective scan：输入序列通过 input-dependent 参数 (Δ, B, C) 将连续时间 SSM 离散化，逐 token 更新 hidden state，避免了 Transformer attention 的 O(N²) 计算和 KV cache。关键创新：(1) Input-dependent 参数化——Δ, B, C 由当前输入 x_t 通过线性投影生成，使模型具备选择性（根据输入内容决定保留/丢弃哪些信息）；(2) 硬件感知算法——通过 kernel fusion 和 parallel scan 在 GPU 上高效实现；(3) 无 KV cache——状态隐式编码为固定大小 hidden state。Mamba 在长序列上效率极高，但 In-Context Learning 和复杂检索能力弱于 Transformer attention。LongLLaVA 在 Hybrid 架构中 Mamba 层占比 7/8。

**Mamba-2 (Structured State Space Duality, SSD)**：Mamba-2 (Dao and Gu, 2024) 将 SSM 重新形式化为矩阵乘法，核心公式化简为 h_t = A_t·h_{t-1} + B_t·x_t，y_t = C_t^T·h_t，其中 A_t, B_t, C_t 为离散化参数。d_state 从 16 增至 128。TimeViper 在 hybrid backbone 中使用 27 层 Mamba-2，每层通过遗忘-记忆门控将历史序列信息编码入固定大小的隐式 hidden memory。Mamba-2 的 attention pattern 可通过 row-wise L1 normalized 累乘矩阵可视化，揭示其多样化的注意力模式——sparsity（选择性关注关键 token）、locality（邻域聚焦）、globality（全局均匀关注）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mamba-2 SSM (TimeViper formulation)
# h_t = A_t * h_{t-1} + B_t * x_t
# y_t = C_t^T * h_t

def MambaBlock(x):  # x: [B,L,D]
    x_proj, z = split(Linear(x))              # each [B,L,D_inner]
    x_conv = SiLU(Conv1D(x_proj))             # local mixing
    B = Linear(x_conv); C = Linear(x_conv)    # [B,L,d_state]
    dt = softplus(Linear(x_conv) + dt_bias)   # [B,L,D_inner]
    A_bar = exp(dt ⊗ A)                       # discretize
    B_bar = dt ⊗ B
    h = selective_scan(A_bar, B_bar, x_conv)  # parallel scan O(N)
    y = (h @ C.T) * SiLU(z)                   # gated output
    return Linear(y)                           # [B,L,D]

# Mamba-2 attention pattern (for interpretability):
# y_i = sum_{j=1}^{i} C_i^T * (prod_{k=j+1}^{i} A_k) * B_j * x_j
# M'_{i,j} = |C_i^T * (prod_{k=j+1}^{i} A_k) * B_j|  -- "attention score"
```

Annotations: d_state = 16 (Mamba-1) or 128 (Mamba-2)；D_inner = D * expand_factor (2 or 4)；SiLU gate 类似 LSTM 输出门；selective scan 使用并行前缀扫描；Mamba-2 的"attention score" M'_{i,j} 非显式计算，而是从 SSM 参数推导得到，用于模型可解释性分析。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba 官方实现 https://github.com/state-spaces/mamba，HuggingFace `MambaModel`。Mamba-2 (Dao and Gu, 2024) 提出 SSD (Structured State Space Duality)，将 SSM 形式化为矩阵乘法加速，d_state 增至 128。Falcon-mamba (7.3B) 是最大纯 Mamba LLM；Jamba 首次 hybrid Mamba-Transformer + MoE；Cobra 扩展 Mamba 到多模态 LLM。LongLLaVA 中 Mamba 层无 KV cache，使 hybrid architecture 在 100K tokens 时 Throughput 37.6 (2.6× vs Transformer)。

**Mamba 在长视频理解中的独特作用**：TimeViper 发现 Mamba-2 layers 通过遗忘-记忆机制隐式建模视频的时序位置信息——即使只使用 SigLIP 的 positional embedding（无 MRoPE 等显式时间戳建模），TVG 任务 mIoU 仍达 40.5，与显式使用 MRoPE 的 Qwen2.5-VL-7B (43.6) 差距不大。Mamba-2 的 O(1) KV-cache 使 TimeViper 在 32K input tokens (≈2K frames × 16 tokens/frame)、1K output tokens、batch_size=32 时，每秒生成 token 数比 Qwen3 高 40.1%。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding

## 2D Bilinear Token Pooling / 二维双线性Token池化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D Bilinear Token Pooling 是 Multimodal LLM 中压缩视觉 token 数量的空间保持型池化方法。Vision encoder (CLIP ViT) 输出 24×24 = 576 个 patch tokens / image。为减少输入 LLM 的视觉序列长度，使用 2×2 bilinear pooling 将 token 网格从 24×24 压缩到 12×12 = 144 tokens（75% 压缩率）。与 1D pooling（直接平均池化到 144 tokens，丢失 2D 位置信息）不同，2D pooling 保持 12×12 spatial layout，使压缩 token 仍编码空间关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 2D Bilinear Token Pooling: 576 → 144 tokens
def bilinear_pool_2d(H_v):  # H_v: [576, D]
    H_grid = H_v.reshape(24, 24, D)        # [24, 24, D]
    # 2x2 avg pooling preserving spatial layout
    H_pooled = avg_pool2d(H_grid.permute(2,0,1),
                          kernel_size=2, stride=2)  # [D, 12, 12]
    return H_pooled.permute(1,2,0).reshape(144, D) # [144, D]

# Contrast 1D Pooling:
def avg_pool_1d(H_v):  # drops spatial info
    return H_v.reshape(144, 4, D).mean(dim=1)
```

Annotations: CLIP ViT-B/32 gives patch grid 24×24=576 tokens；2×2 pooling → 12×12=144；kernel_size=2, stride=2；`avg_pool2d` = bilinear downsampling with spatial awareness。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA Table 6 消融：No pooling (576 tokens) GQA 63.2/Mile 38.2；1D (144) 60.4/36.2；2D (144) 61.3/37.7。2D 在 GQA/SEED/Mile 上均优于 1D。Token compression 对细粒度任务的负面影响通过 Image Partitioning (pad to multiple of 168, split into 168×168 blocks) 缓解：V* 上 49.6%→68.5%。144 tokens/image 使单卡 A100 80GB 可处理 ~1000 张图像。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture

## Progressive Training Strategy for Multi-modal LLMs / 多模态LLM渐进式训练

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Training Strategy 是一种多阶段递进训练方法，逐步将预训练语言模型转化为多模态长上下文模型。LongLLaVA 的三阶段：Stage I (Single-image Alignment) — 仅训练 projector，冻结 vision encoder 和 LLM，~600K captions 对齐视觉-文本空间；Stage II (Single-image Instruction Tuning) — 训练 projector + LLM，~932K QA pairs 赋予单图指令跟随；Stage III (Multi-image Instruction Tuning) — 全面多图训练 (~700K instances)，配合 Replay 机制（从前阶段采样数据混入）防止 catastrophic forgetting。相比 Mixed Training（所有数据混合训练），Progressive Training 在 multi-image 上提升显著 (Mile 46.5 vs 42.2)，单图保持持平。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Three-stage Progressive Training
# Stage I: Alignment
model.freeze(vision_encoder=True, LLM=True, projector=False)
model.train(ALLaVA-Caption+ShareGPT4V, 600K captions)

# Stage II: Single-image Instruction Tuning
model.freeze(vision_encoder=True, LLM=False, projector=False)
model.train(LLaVA-1.5+Mantis-Single, 932K QA)

# Stage III: Multi-image Instruction Tuning
# + Replay (200K single-image + 50K pure-text) from Stages I/II
model.train(Mantis+VideoChat2+ShareGPT4Video+Replay+SubImage, 700K)
# packed to 176K tokens/sequence, <eos> separated, 1 epoch

# All: cosine LR, warmup 0.03, peak lr=1e-5, 3x8 A800 GPUs
```

Annotations: Stage I 仅对齐不训练 LLM；Stage III Replay 关键（w/o → SQA↓ +18.5%）；Text replay 50K 饱和；Single-image replay 随数据量持续改善。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLaVA (Li et al., 2024a) 提出 Stage I+II；LLaVA-1.5 扩展 Stage II 为 instruction tuning；LongLLaVA 增加 Stage III 专门针对 multi-image。Qwen2-VL 和 InternVL2 使用类似渐进策略（分辨率递增多阶段）。消融：Stage I+II+III (progressive) Mile 46.5 vs I&II+III 44.2 vs I&II&III 42.2 (mixed)。Replay 消融 (Appendix F) 验证了其在防止 forgetting 中的关键作用。

ReVisionLLM 的渐进式训练为两阶段：(1) Stage 1 短片段训练——先用 dense features 微调 LLM (LoRA) 学习精确边界预测，再冻结 LLM 微调 Hierarchical Adapter 生成 sparse features，引入 Contrastive Segments（不含目标事件的负样本）训练存在性判断以校准置信度。(2) Stage 2 长视频训练——冻结 Hierarchical Adapter，仅微调新 LoRA 模块，利用 sparse features 识别小时级视频中的感兴趣段。与 LongLLaVA 的 image-level 渐进策略形成互补：LongLLaVA 渐进扩展图像数量（单图→多图），ReVisionLLM 渐进扩展视频时长（短片段→小时级）。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Mixture of Experts (MoE) in Hybrid LLMs / 混合LLM中的专家混合

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture of Experts (MoE) 是一种通过稀疏激活增加模型容量但不线性增加推理计算的架构设计。在 Transformer/Mamba LLM 中，将 FFN 层替换为 E 个独立专家子网络，每个 token 通过 router (gating network) 选择 top-k 专家激活。核心优势：(1) 总参数大但推理仅激活 top-k，计算量与激活参数成正比；(2) 专家在训练中自然分化出专业化。LongLLaVA 在 hybrid 架构中每隔一层使用 MoE FFN：E=16 experts, top-2 gating per token，总参 53B，推理激活 13B。Router 为 linear projection + softmax + top-k。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MoE_FFN(h):  # h: [B,L,d], E=16, k=2
    gate = softmax(Linear(d→E)(h), dim=-1)     # [B,L,16]
    w, idx = topk(gate, k=2); w = w/sum(w)     # renormalize
    out = zeros([B,L,d])
    for i in range(2):
        out += w[:,:,i,None] * expert[idx[:,:,i]](h)
    return out
# expert: SwiGLU_FFN (d → d_ff → d)
# Load balance: L_aux = E * Σ f_e * P_e
```

Annotations: E=16, k=2 → 12.5% active parameters；MoE 遵循 Jamba 的 layer-wise pattern (every other)；LongLLaVA-9B 仅保留 Expert-0 (差异极小)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA 的 expert selection 消融 (Appendix C)：不同 expert (0/5/12/15) 和聚合方式 (arithmetic/spherical) MMLU/BBH 差异极小 (<1%)，因此 LongLLaVA-9B 仅保留 Expert-0。Jamba 首次 hybrid Mamba-Transformer + MoE。Mixtral (Jiang et al., 2024a) 8 experts top-2。DeepSeek-V2 160 experts top-6 (细粒度 MoE)。关键训练挑战：load balancing 和 expert dropping。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture

## Grouped Query Attention (GQA) / 分组查询注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GQA (Grouped Query Attention) 是介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间的注意力机制 (Ainslie et al., 2023)。MHA 为每个 head 分配独立 Q/K/V (H heads → H 组 K/V)，KV cache 大；MQA 所有 heads 共享一组 K/V，节省内存但可能损失质量。GQA 折衷：H 个 query heads 分 G 组，每组共享一组 K/V (H/G heads/group)。KV cache 从 MHA 的 2×H×L×d_head 降至 GQA 的 2×G×L×d_head。LongLLaVA 在 hybrid 架构的 Transformer attention 层中使用 GQA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GQA: H=32 query heads, G=8 KV heads, r=H/G=4
def GQA(x):
    Q = reshape(W_Q@x, [B,L,H,d])     # H heads
    K = reshape(W_K@x, [B,L,G,d])     # G heads only
    V = reshape(W_V@x, [B,L,G,d])
    # Expand K,V: G → H via repeat
    K = repeat(K, "... G d → ... (G r) d", r=H//G)  # [B,L,H,d]
    V = repeat(V, "... G d → ... (G r) d", r=H//G)
    scores = Q @ K.T / sqrt(d); attn = softmax(scores)
    return W_O @ reshape(attn @ V, [B,L,D])
```

Annotations: H=32, G=8, r=4 (common); KV cache: MHA 2*H*L*d, GQA 2*G*L*d (4× smaller); FlashAttention supports GQA natively via num_kv_heads parameter.

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GQA 通过 up-training 从 MHA checkpoint 转换 (Ainslie et al., 2023)：mean pool K/V projections 后少量步骤微调。Llama 2/3、Mistral、Qwen 等主流 LLM 均采用 GQA。FlashAttention 原生支持。LongLLaVA 中 GQA 仅用于 attention layers (12.5% layers)，配合 Mamba layers 整体 KV cache memory 大幅低于纯 Transformer。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

VideoNSA 基于 Qwen2.5-VL-7B 构建，其 LLM decoder（Qwen2.5-7B）使用 GQA 配置 28 query heads 共享 4 KV heads（group ratio = 7:1）。在 VideoNSA 的 hybrid attention 设计中，GQA 被用于 text token 的 standard attention path（保留指令跟随能力），而 vision tokens 则使用 NSA（三支路稀疏注意力）。这种 hybrid design 的关键优势：GQA 的 KV cache 复用特性天然降低了 text-side 的 KV cache 内存；NSA 的稀疏 attention 则将 vision-side 的计算从 O(L²) 降至 O(L×K_attn)。两者互补，使 VideoNSA 在 128K context 下仅需使用 3.6% 的 attention edges 且保持文本理解精度。

## KV-recache（KV重缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-recache 是 LongLive 提出的面向交互式长视频自回归（AR）生成的 KV cache 更新机制。在用户 prompt 切换时，传统的 KV cache 策略存在两种困境：(1) 丢弃全部 KV cache → 视觉断裂、时间不连续；(2) 保留全部 KV cache → 旧 prompt 语义残留在 cache 中，导致新 prompt 延迟响应或不跟随。KV-recache 通过在 prompt 切换边界重新计算 KV cache：将已生成视频前缀 x 作为视觉上下文，与新 prompt p_new 一起重新通过生成器的交叉注意力层（cross-attention: visual Q attend to text prompt K/V）和前向层计算新的 KV state。由于新 prompt 的 text embedding 替换了旧 prompt 的 text embedding，交叉注意力层中注入新的语义信号，清除旧 prompt 的残留语义；同时自注意力层的 causal attention 保留了已生成帧之间的视觉运动和外观连续性线索。每次 prompt switch 仅需一次 recache forward pass（过已生成前缀帧），随后步骤使用刷新后的 KV cache 正常进行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# KV-recache: prompt switch 时的 KV cache 刷新
def kv_recache(G_theta, video_prefix_x, old_cache_C, p_new):
    C_new = []
    for step_i, f_i in enumerate(video_prefix_x):
        kv_self = G_theta.self_attn_cache(f_i, C_new)
        kv_cross = G_theta.cross_attn_cache(f_i, p_new)  # key: K/V from NEW prompt
        C_new.append((kv_self, kv_cross))
    return C_new
```

Annotations: recache 仅需单次 forward pass（过已生成前缀），对 10s video (single switch) 额外时间开销约 6%。训练时同步集成 recache（teacher 也接收新 prompt 输出 DMD 监督），消除 train-inference mismatch。推理时支持多次 prompt switch（n+1 个 prompt → n 个 switch 边界），每边界执行一次 recache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KV-recache 基于 Wan2.1-T2V-1.3B (DiT 架构) 实现。DiT 的 cross-attention + self-attention 交替结构使 recache 关键：self-attention 传播视觉连贯性（KV cache 中的自注意力状态），cross-attention 注入 prompt 语义（K/V 来自当前 prompt embedding）。训练时集成到 streaming long tuning loop。一般使用场景：交互式视频生成（用户逐步输入新 prompt）、叙事长视频（多场景切换）、实时内容创作（streaming prompt input）。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Streaming Long Tuning（流式长调优）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming Long Tuning 是 LongLive 提出的用于 AR 视频生成的长序列训练策略。传统 AR 视频模型采用 train-short-test-long 策略（仅在 5s 短视频上训练，推理时通过滚动 KV cache rollout 长视频），导致误差累积。Streaming Long Tuning 实现 train-long-test-long 对齐：每 iteration 基于前一 iteration 存储的 KV cache 滚动生成下一个 5s clip（而非重新采样），仅对当前 clip 计算 DMD loss（teacher=Wan2.1-T2V-14B 监督），detach 历史帧梯度（仅当前 clip 产生梯度）。显存仅由当前 clip 控制 O(clip)，非全长序列 O(full_video)，避免 naive long tuning 的 OOM。DMD loss 在完整 rollout 上提供全局监督。滚动直至预设最大长度（60s 或 240s）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Streaming Long Tuning (LongLive Algorithm 1)
# l_video=60s, l_clip=5s, teacher=Wan2.1-T2V-14B
C = []; l = 0
(p, p_next) = sample_prompt_pair()
s = sample_switch_time(5s, 55s)
while l < l_video:
    p_active = p if l < s else p_next
    if l == s: C = kv_recache(G_theta, x, C, p_active)
    x_clip = generate_next_5s(G_theta, C, p_active)  # rollout
    loss = DMD_loss(x_clip, teacher_14B, p_active)
    loss.backward()  # gradient only through x_clip (history detached)
    optimizer.step(); l += l_clip
```

Annotations: 关键设计：(a) detach 历史帧 → 显存恒定；(b) teacher 对每个 5s clip 独立监督 → 始终在 teacher 能力范围内；(c) streaming rolling 模拟推理 → train-test alignment。64 H100 × 12h ≈ 32 GPU-days for 60s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Streaming Long Tuning 在 LongLive 中以 Wan2.1-T2V-1.3B + LoRA (rank=256, ~27% params trainable) + AdamW 实现。不引入额外视频数据，teacher-student DMD self-supervision。Qwen2-72B-Instruct 生成 follow-up prompt 构造 switch 训练数据。适用于：(a) AR 生成模型 train-test mismatch；(b) 自监督蒸馏场景；(c) 主流 KV cache rollout 的因果模型。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Short Window Attention for Video Generation（视频生成短窗口注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Short Window Attention 是在长视频 AR 生成中将自注意力的感受野限制在固定时间窗口 W 内的注意力机制。传统 causal attention 复杂度 O(L²)（L = 总序列长度），随视频增长不可持续。利用视频生成中的时间局部性（temporal locality）——附近帧对预测下一帧更重要——将注意力限制在最近 W 个 latent frames 内，复杂度降为 O(W·L)，KV cache 需求从 O(L) 降为 O(W)。LongLive 将 window size 设为 W=9 latent frames（配合 S=3 sink tokens）。Window size 引入 quality-efficiency trade-off：大窗口高一致性但高延迟，小窗口快但一致性下降。Frame sink 机制（见下文）可恢复此 trade-off 中的一致性损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Short Window Attention (per self-attention layer)
# W=9, S=3 sink tokens, effective KV = W+S (不随序列增长)
def short_window_attn(Q, K, V, K_sink, V_sink, window=9):
    K_eff = concat([K_sink, K[-window:]], dim=-2)  # [S+W, d_head]
    V_eff = concat([V_sink, V[-window:]], dim=-2)
    return softmax(Q @ K_eff.T / sqrt(d)) @ V_eff
# KV cache eviction: oldest non-sink tokens evicted when count > W
```

Annotations: 对 Wan2.1-T2V-1.3B @ 832x480, W=9+3 sink: 端到端计算时间降低 28%，峰值显存降低 17%（vs full attention on H100）。训练时同步使用 short window（streaming long tuning alignment），resident KV per step = O(W+T+S)（T=5s clip length）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Short window attention 通过 attention mask 实现（-inf mask for out-of-window KV）。与 FlashAttention 兼容（causal mask 内嵌 window mask）。LongLive 在 self-attention 层应用，cross-attention 保持全注意力。适用于：(a) 长视频 AR 生成推理加速；(b) 配合 frame sink 恢复一致性；(c) 任意 causal transformer 的长序列低延迟推理。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Frame Sink / Frame-Level Attention Sink（帧级注意力汇）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame Sink（帧级注意力汇）是 LongLive 在视频 AR 生成中提出的全局锚定机制，受 LLM 领域 attention sink 概念启发。在视频生成中，此前 Self-Forcing 报告 attention sink tokens 单独无法防止长 rollout collapse。LongLive 发现：一旦通过 streaming long tuning 解决了 long rollout collapse 问题，attention sink 即可生效。Frame Sink 将视频首帧 chunk（3 latent frames）固定为全局 sink token，永久保留在每层 self-attention 的 KV cache 中从不被驱逐，所有后续帧都能通过注意力访问它们。Sink tokens 作为"场景身份锚点"，缓存色调、风格、主体外观等持久视觉属性，补偿 short window attention 丢失的远距离时间上下文。实验中 W=9 local + S=3 sink 的一致性接近 W=21 full window，但保持 W=9 的速度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Frame Sink + Short Window Attention
sink_indices = [0,1,2]  # first chunk's 3 latent frames
def attn_with_sink(Q, K_all, V_all, cur_idx, window=9):
    K_sink = K_all[sink_indices]  # NEVER evicted
    V_sink = V_all[sink_indices]
    start = max(0, cur_idx - window)
    K_win = K_all[start:cur_idx]
    V_win = V_all[start:cur_idx]
    K_eff = cat([K_sink, K_win])  # sink always first
    V_eff = cat([V_sink, V_win])
    return sdpa(Q, K_eff, V_eff)
```

Annotations: S=3 sink tokens (first chunk of 3 latent frames)。20s 生成实验：Window 21 (no sink) 高一致性/慢；Window 12 (no sink) 一致性下降；Window 9 + Sink 3 一致性接近 Window 21。起效前提：streaming long tuning 必须先解决 long rollout collapse。KV recache 时不重算 sink。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时 sink tokens 在 streaming long tuning 的 short window config 中常驻 KV；推理时在帧生成开始时确定。适用于：(a) 长视频 AR 高效推理；(b) 降低 attention 复杂度但保持长程一致性；(c) 多 prompt switch（sink 维持全局视觉身份，recache 更新 prompt 语义）。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Distribution Matching Distillation (DMD) / 分布匹配蒸馏

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Matching Distillation (DMD) 由 Yin et al. (CVPR 2024, NeurIPS 2024) 提出，将多步扩散模型蒸馏为少步生成器的知识蒸馏方法。DMD 匹配 student 和 teacher 的输出分布（distribution matching）而非具体去噪值。通过辅助判别器（critic network）区分 student 和 teacher 生成样本的分布差异，以 adversarial loss 驱动 student 学习 teacher 的样本分布。LongLive 使用 DMD 将 Wan2.1-T2V-1.3B 的多步扩散生成器蒸馏为 few-step 因果 AR 生成器：student Gθ 从噪声通过单步预测 x̂_0，teacher (Wan2.1-T2V-14B) 对同样 noisy input 去噪，critic 计算分布距离。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def DMD_loss(G_theta, x_real, p_text, G_teacher, D_phi):
    eps = randn_like(x_real); t ~ U(0,1)
    x_noisy = sqrt(a_t)*x_real + sqrt(1-a_t)*eps
    x_hat_s = G_theta(x_noisy, t, p_text)     # student single-step
    x_hat_t = G_teacher(x_noisy, t, p_text)    # teacher denoising
    L_distill = mean((D_phi(x_hat_t, t) - D_phi(x_hat_s, t)) ** 2)
    L_student = mean(D_phi(x_hat_s, t))
    return L_distill + L_student
```

Annotations: LongLive DMD config: student lr=1e-5 (β1=0.0, β2=0.999), critic lr=2e-6。EMA decay=0.99 from step 200。在 streaming long tuning 中仅应用到当前 5s clip（teacher 在自身能力范围内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DMD 在 LongLive 中：(1) ODE initialization → 将 Wan2.1-T2V-1.3B 初始化为 causal AR；(2) DMD 训练 → short window + frame sink；(3) Streaming long tuning → 继续 DMD + KV-recache。适用于：(a) 扩散模型加速；(b) 自监督蒸馏（无需 ground truth）；(c) AR 模型因果化适配。论文：Yin et al. CVPR/NeurIPS 2024。LongLive 开源：https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Train-Short-Test-Long（短训练长测试问题）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Train-Short-Test-Long 是 AR 视频生成模型中普遍存在的训练-推理范式不匹配问题：训练仅在短视频片段（如 5s）上进行（硬件约束 + 长视频数据稀缺），推理时通过滚动 KV cache rollout 生成长视频。模型从未在训练中见过自生长序列上下文，推理时模型自发误差通过自循环反馈累积，导致内容漂移。LongLive 提出 train-long-test-long 解决方案：通过 streaming long tuning 在训练中模拟推理 rollout，使模型暴露于自生长序列和退化上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Train-Short-Test-Long: training only 5s clips, inference 60s
# → error accumulation because model never trained on self-generated history

# Train-Long-Test-Long (LongLive): training simulates inference rollout
# → model conditioned on imperfect self-generated history with DMD supervision
# → streaming long tuning = train as test
```

Annotations: 显存对比：Naive long tuning → OOM; Streaming long tuning (detach history) → 显存恒定。LongLive VBench-Long 30s: 83.52 vs Self-Forcing train-short-test-long: 81.59。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Train-long-test-long 实现：(a) LoRA (rank=256) 微调而非全模型 finetune；(b) DMD 自监督蒸馏无需 ground truth 长视频；(c) teacher 对每个 5s clip 独立监督；(d) 训练中集成 KV-recache + short window + frame sink（完全对齐推理配置）。适用于任何 AR 生成模型的长序列扩展训练。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation

## Five-Stage VLM Training Pipeline (LongVILA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
五阶段VLM训练pipeline是LongVILA提出的长上下文视觉语言模型训练流程，将VLM训练从传统的3阶段（对齐→预训练→SFT）扩展为5阶段：(1) Stage1 多模态对齐——冻结LLM和视觉编码器，仅训练多模态投影器（linear/MLP层），桥接视觉与语言模态；(2) Stage2 大规模预训练——冻结视觉编码器，在COYO-25M等大规模图文数据集上训练LLM和投影器，使用VILA-1.5-40B重标注数据提升质量；(3) Stage3 短监督微调（Short SFT）——全参数微调，混合图像和短视频数据（如YouCook2、ShareGPTVideo）；(4) Stage4 上下文扩展（Context Extension）——在进入长视频SFT之前，先用纯文本数据（SlimPajama 17B tokens）对LLM进行持续预训练以扩展上下文窗口，采用渐进式训练调度（8K→65K→262K tokens），配合RoPE基频增大和LoRA微调；(5) Stage5 长监督微调（Long SFT）——全参数训练，使用MM-SP系统，在15,292个长视频的LongVILA_SFT数据集上进行指令微调。Ablation证明Stage4必须在Stage5之前执行才能获得最佳性能（57.5 vs 55.3-56.0 VideoMME average w/o subtitle）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Algorithm: Five-Stage VLM Training Pipeline
Input: vision_encoder Φ, projector Ψ, LLM Θ

# Stage 1: Multi-modal Alignment
Freeze(Φ); Freeze(Θ)
for (img, caption) in D_alignment:
    v = Ψ(Φ(img))                 # vision features → projector
    loss = CE(Θ([v; text(caption)]), labels)
    Update(Ψ)                     # only projector trained

# Stage 2: Large-scale Pre-training
Freeze(Φ)
for (img, text) in D_coyo_relabeled:
    v = Ψ(Φ(img))
    loss = CE(Θ([v; text]), labels)
    Update(Ψ, Θ)                  # projector + LLM trained

# Stage 3: Short Supervised Fine-Tuning
for (img_or_frames, text) in D_short_mixed:
    v = Ψ(Φ(img_or_frames))
    loss = CE(Θ([v; text]), labels)
    Update(Φ, Ψ, Θ)               # all params

# Stage 4: Context Extension (text-only, LoRA)
SetRoPEBase(Θ, base_freq × scale) # 增大RoPE基频
for text in ProgressiveSchedule(D_slimpajama):
    # 8K → 65K → 262K progressive
    loss = CE(Θ_LoRA(text), labels)
    Update(LoRA_params)

# Stage 5: Long Video SFT (full params, MM-SP)
for (long_video_frames, text) in D_longvila_sft:
    # MM-SP distributes across GPUs
    loss = CE(Θ_SP([Ψ(Φ(frames)); text]), labels)
    Update(Φ, Ψ, Θ)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于VILA框架（HuggingFace Transformers），开源在github.com/NVlabs/VILA/tree/main/longvila。Stage4使用LongLoRA方法进行LoRA微调，约需336 GPU hours on 80GB A100。Stage5需要MM-SP系统支持，因为单个长视频样本可达1400帧（约274K tokens），超出单卡内存。数据生成方面，长视频被切分为10秒片段，每片段用VILA-1.5生成caption，再由LLM基于所有片段caption生成QA对。适用于需要处理时长数十分钟到数小时的长视频理解任务。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## Context Extension for LLMs (via RoPE Scaling + Progressive Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context Extension是扩展LLM上下文窗口的技术。LongVILA在Stage4采用纯文本持续预训练方式扩展LLM上下文：利用SlimPajama数据集共17B tokens，采用渐进式训练调度，逐步将上下文从8K扩展到65K再到262K。同时增大RoPE（Rotary Position Embedding）基频——标准的RoPE基频为10000，通过增加基频值（如扩展到更大的数），RoPE对更长距离的位置编码可以保持区分度。配合LoRA进行参数高效的微调，避免全参数训练的高昂成本。论文实证发现，在进行长视频SFT之前必须先完成上下文扩展，否则模型无法有效利用长上下文信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Context Extension via RoPE base frequency scaling
# Standard RoPE: θ_i = base^(-2i/d), base = 10000
# Extended RoPE: θ_i' = (base × scale)^(-2i/d)

def rope_extend(attention_module, scale_factor=8.0):
    """Extend RoPE base frequency for longer context"""
    original_base = attention_module.rope_theta  # e.g., 10000
    attention_module.rope_theta = original_base * scale_factor
    
def progressive_training(model, data_loader, schedule):
    """
    Progressive context extension schedule
    schedule = [(8192, 5B_tokens), (65536, 6B_tokens), (262144, 6B_tokens)]
    """
    for seq_len, num_tokens in schedule:
        model.max_position_embeddings = seq_len
        data_loader.set_sequence_length(seq_len)
        
        tokens_processed = 0
        for batch in data_loader:
            # LoRA forward/backward only
            with lora_enabled(model):
                loss = model(batch.input_ids, attention_mask)
                loss.backward()
                update_lora_params()
            tokens_processed += batch.numel()
            if tokens_processed >= num_tokens:
                break
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongVILA使用LongLoRA方法进行上下文扩展：仅训练LoRA参数（adapter层），冻结原始LLM权重，大幅降低显存和计算需求。RoPE基频缩放参考Fu et al. 2024的方法。约需336 H100 GPU hours。适用于需要将预训练LLM从较短上下文（如4K-32K）扩展到长上下文（如128K-256K）的场景，是长视频/长文档VLMs的必要前置步骤。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## Long Video Supervised Fine-Tuning (Long SFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Long Video SFT（长视频监督微调）是LongVILA的Stage5训练阶段，在完成上下文扩展后，使用长视频指令数据对VLM进行全参数监督微调。关键创新：(1) 数据生成pipeline——长视频先被切分为约10秒的短片段，每片段由VILA-1.5模型独立生成描述性caption，然后由LLM基于所有片段caption生成问答对（涵盖总结、空间、属性、动作、对象、OCR、时序等7类问题）；(2) LongVILA_SFT数据集——来自Shot2Story20k的15,292个长视频，涵盖Travel/Sports/Education等12个类别，每个视频配有1个caption question和1个QA question；(3) 使用MM-SP系统进行分布式训练，因为单个样本可达1400帧（约274K tokens），远超单GPU显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Long Video Data Generation Pipeline
def generate_long_video_sft_data(video_path):
    # Step 1: Segment long video into 10-second clips
    clips = segment_video(video_path, clip_duration=10)  # 10s each
    
    # Step 2: Generate captions per clip using VILA-1.5
    captions = []
    for clip in clips:
        frames = sample_frames(clip, num_frames=8)
        caption = VILA_15.generate_caption(frames)  # short-context VLM
        captions.append(caption)
    
    # Step 3: Generate QA pairs using LLM from all captions
    prompt = f"Based on these clip descriptions:\n{captions}\nGenerate questions about: summary, spatial relations, attributes, actions, objects, OCR, temporal events."
    qa_pairs = LLM.generate(prompt)  # text-only LLM
    
    return {"frames": sample_frames(video_path, num_frames=256), 
            "qa_pairs": qa_pairs,
            "captions": captions}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖MM-SP系统进行分布式训练（SP degree ≥ 4）。全参数微调（所有视觉编码器+投影器+LLM参数可训练），与Stage3短SFT的区别在于数据是长视频（数百到数千帧）且需要SP系统支持。Batch size设为1（受限于单序列超长）。适用于需要让VLM理解数十分钟到数小时长视频内容的应用，如体育赛事分析、电影理解、监控视频摘要等。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## Needle-in-a-Haystack Evaluation (for Long Video)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Needle-in-a-Haystack（大海捞针）评估是测试长上下文模型在极长序列中检索特定信息能力的实验方法。对于长视频VLM：构建一个极长的视频序列（如6000帧），在其中特定深度位置（如0%、25%、50%、75%、100%位置）插入"needle"（特殊设计的图像），要求模型回答与该图像相关的问题。评估指标为正确检索的准确率。LongVILA训练于2048帧，但在6000帧（超过1M tokens）测试中达99.8%准确率，证明了其上下文能力的有效扩展。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Long Video Needle-in-a-Haystack
eval_result = {}  # (depth, num_frames) → accuracy

for num_frames in [32, 64, 128, ..., 6000]:
    for depth in [0.0, 0.1, 0.2, ..., 1.0]:  # relative position
        # Create haystack video with num_frames
        base_frames = sample_video_frames(num_frames)
        
        # Insert needle image at depth position
        insert_pos = int(depth * num_frames)
        needle_image = create_needle_image(question_id)
        test_frames = base_frames[:insert_pos] + [needle_image] + base_frames[insert_pos+1:]
        
        # Query model about needle content
        prompt = f"<video>{test_frames}</video> What was shown at position {depth*100}%?"
        answer = model.generate(prompt)
        
        correct = evaluate_answer(answer, ground_truth[question_id])
        eval_result[(depth, num_frames)] = correct

# Plot heatmap: depth (y-axis) × num_frames (x-axis) × accuracy (color)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
源自LLM长上下文评估方法，扩展到视频领域。LongVILA的needle使用特殊设计的图像（如红色圆点或特定文字），模型需描述看到的内容。用于量化评估长视频VLMs在各深度位置的检索能力，是验证模型有效上下文窗口的关键实验。LongVILA是首个在1M+ token上下文中达到99%+准确率的VLM。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

TSPO 中的 Video Needle-in-a-Haystack 使用方式：TSPO 创新性地将 NIAH 从**评估范式**转化为**训练数据构建范式**。

**VideoRoPE 提出的 V-NIAH-D (with Distractors)**：VideoRoPE 发现标准 V-NIAH 的检索任务过于简单，即使位置编码有缺陷的模型也可以通过 spatial dimension（而非 temporal dimension）定位 needle。V-NIAH-D 在 V-NIAH 基础上（3000 帧 haystack, needle 插入于随机位置），在距 needle 约 200 帧处周期性插入 semantic distractor——与 needle 语义相似但与问题无关的图像（通过 Google Image Search 或 Flux 生成）。distractor 的插入周期由 RoPE 频率特性计算：2·π·10000^(32/128) ≈ 198.7 ≈ 200。该任务暴露了高频 temporal allocation 的缺陷：distractor 帧在 temporal 维度的高频旋转角下与 needle 产生相同的 temporal embedding（"hash collision"），使仅依赖 temporal 维度的模型被误导。MRoPE 在 V-NIAH→V-NIAH-D 上从 78.67% 降至 74.67%（-4.0），而 VideoRoPE 通过低频 temporal allocation 保持在 87.11%（-4.0）。传统 Video NIAH 仅用于评估模型的长程检索能力，TSPO 将其改造为 RL 训练数据管道：(1) 从 LLaVA-Video-178K 采样目标视频，使用 Qwen2.5-VL 生成详细事件描述并重格式化为多选题；(2) 在 segment 级别将目标视频与无关视频拼接/打乱，合成 10∼60 分钟超长训练视频；(3) 这些合成视频自动带有伪标签（目标视频时间边界），用于计算 Temporal Localization Reward R_T = T_t/T_a（采样帧中目标帧占比）。与评估用 NIAH 的关键区别：评估 NIAH 使用合成 needle（图像/文字）插入，TSPO 使用真实视频段作为 needle，保持自然视频分布。该管道产出的数据与 Comprehensive Temporal Data 合并为 TSPO-10K 训练集。

## Multimodal Coverage Maximization（多模态覆盖最大化 / MMTok）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multimodal Coverage Maximization 是 MMTok 提出的 training-free 视觉 token 选择方法。它将 token 选择建模为最大覆盖问题（Maximum Coverage Problem）：给定源 token 集合（n 个 vision tokens）和目标 token 集合（m 个 text tokens + n 个 vision tokens），从中选择 k 个源 token（k ≪ n）以最大化覆盖目标 token 的信息量。覆盖函数定义为子模函数 f(S; M) = (1/m) Σᵢ max_{j∈S} M_{i,j}，即对每个目标 token i，选择 S 中与之最相似的源 token j，相似度取 max，对所有目标 token 取平均。与现有 unimodal token pruning 方法（FastV 用 vision attention, SparseVLM 用 text attention, VisionZip 用 CLS attention, DivPrune 用 diversity）不同，MMTok 同时利用两个模态：text-vision（T-V）coverage 让选出的 vision token 覆盖文本查询语义；vision-vision（V-V）coverage 让选出的 vision token 覆盖全部图像信息。两者通过 softmax 温度校准后加权融合（α=0.5），贪心算法 O(kn) 获得 (1-1/e) 近似最优解。training-free，无需微调，实际开销极低（2880 tokens → 160 tokens 仅 6.4ms on A6000, 13.93 GFLOPs）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Multimodal Coverage Maximization (MMTok) Pipeline
# 输入: 图像 I, 文本查询 Q, 目标 token 数 k
# 超参数: τ_t=0.02, τ_v=0.2, α=0.5

# Step 1: Vision Encoder → vision tokens
V_raw = ViT(I)         # ∈ R^(n×d), 投影前
V_proj = MLP(V_raw)    # ∈ R^(n×d'), 投影后, 与 LLM 对齐

# Step 2: Text Embedding
T = Tokenizer(Q)       # m 个 text tokens
T_emb = Embed(T)       # ∈ R^(m×d')

# Step 3: L2 归一化
V_proj = L2_norm(V_proj, dim=-1)
V_raw  = L2_norm(V_raw, dim=-1)
T_emb  = L2_norm(T_emb, dim=-1)

# Step 4: 计算相似度矩阵
M_tv = T_emb @ V_proj.T    # ∈ R^(m×n), text-vision
M_vv = V_raw @ V_raw.T     # ∈ R^(n×n), vision-vision

# Step 5: Softmax 温度校准
M_tv' = softmax(M_tv / τ_t, dim=-1)   # 每行归一化
M_vv' = softmax(M_vv / τ_v, dim=-1)

# Step 6: 贪心覆盖选择
S = []
for i in range(k):
    best_s, best_gain = None, -inf
    for s in range(n) where s not in S:
        # 覆盖增量 = T-V 覆盖增益 + α × V-V 覆盖增益
        gain_tv = mean(max(M_tv'[t, S ∪ {s}]) for t in 1..m)
        gain_vv = mean(max(M_vv'[v, S ∪ {s}]) for v in 1..n)
        gain = gain_tv + α * gain_vv - current_f
        if gain > best_gain:
            best_s, best_gain = s, gain
    S.append(best_s)

# Step 7: LLM 推理
V_selected = V_proj[S]    # 仅保留选中的 k 个 vision token
input_llm = concat([V_selected, T_emb])
output = LLM(input_llm)
```

复杂度分析：
- M_tv 构建: O(mnd')
- M_vv 构建: O(n²d)
- 贪心选择: O(kn)，通过 incremental max 优化
- 总体开销: 2880 tokens 选 160 仅 6.4ms (A6000), 13.93 GFLOPs

与 unimodal baselines 的对比：
| 方法 | 模态 | 准则 | 理论保证 |
|------|------|------|----------|
| FastV | Vision-only | Attention ranking | 无 |
| SparseVLM | Text-only | Text→Vision attention | 无 |
| VisionZip | Vision-only | CLS attention ranking | 无 |
| DivPrune | Vision-only | Diversity maximization | 无 |
| **MMTok** | **Multimodal (T+V)** | **Coverage maximization** | **(1-1/e) 近似** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MMTok 的开源实现：https://github.com/Ironieser/mmtok。使用方式：(1) 安装依赖后加载 VLM 模型（LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL 等）；(2) 在 vision encoder 输出后、LLM 输入前插入 MMTok token selection 模块；(3) 设置目标 token 数 k 和超参数 (τ_t=0.02, τ_v=0.2, α=0.5)，论文表明这些参数不敏感，可使用默认值；(4) 可选 MMTok++ 优化：排除 CLIP-ViT padding patches 并修复 overflow bug。实现依赖 PyTorch 内置操作（matmul, softmax, max），无需自定义 CUDA kernel。适用于所有采用 vision encoder + LLM 架构的 VLMs。对于 multi-turn conversation，V-V coverage 允许复用一次选出的 vision tokens 回答多个问题。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

## Submodular Function Maximization for Token Selection（子模函数最大化用于Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
子模函数（Submodular Function）是一种具有"边际收益递减"性质的集合函数：对任意 A ⊆ B ⊆ N 和 s ∈ N\B，有 f(A ∪ {s}) - f(A) ≥ f(B ∪ {s}) - f(B)（加入同一元素，小集合的增量大于或等于大集合的增量）。MMTok 将 vision token selection 的覆盖函数 f(S; M) = (1/m) Σᵢ max_{j∈S} M_{i,j} 证明为子模函数（Leskovec et al., 2007 的设施选址函数变体）。最大化一般子模函数是 NP-hard（Khuller et al., 1999），但贪心算法可以保证解不差于最优解的 (1-1/e) ≈ 63%（Nemhauser et al., 1978）。MMTok 利用两个关键性质：(1) 覆盖函数是子模函数；(2) 两个子模函数之和仍为子模函数（加法保持子模性），因此 T-V coverage + α × V-V coverage 的合并目标仍可被贪心算法近似优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
子模覆盖贪心算法的核心步骤：
```
Algorithm: Greedy Submodular Maximization for Coverage
输入: 相似度矩阵 M ∈ R^(m×n), 目标选择数 k
输出: 选择的 token 索引集合 S, |S|=k

S = ∅
# 维护当前每个目标 token 的最佳覆盖值
coverage = zeros(m)       # coverage[i] = max_{j∈S} M[i,j]
for iter in 1..k:
    best_s, best_total = -1, -inf
    for s in 1..n, s ∉ S:
        # 计算加入 s 后的新覆盖值（增量计算）
        new_total = 0
        for i in 1..m:
            new_total += max(coverage[i], M[i,s])
        if new_total > best_total:
            best_s, best_total = s, new_total
    S.append(best_s)
    # 更新 coverage
    for i in 1..m:
        coverage[i] = max(coverage[i], M[i, best_s])
return S
```

理论性质：
- 子模性：覆盖函数满足 f(A ∪ {s}) - f(A) ≥ f(B ∪ {s}) - f(B)（A ⊆ B 时边际收益递减）
- 单调性：f(A) ≤ f(B) 当 A ⊆ B（加入更多 token 不会减少覆盖）
- 近似比：(1-1/e) ≈ 0.632（对单调子模函数 + 基数约束）
- 复杂度：O(kmn)，通过增量计算可优化至 O(kn)（利用 max 操作的结合性）

在 MMTok 中的实例化：source tokens = vision tokens (n), target tokens = text tokens (m) + vision tokens (n), 相似度矩阵 M = softmax-calibrated cosine similarity。两个覆盖子问题的和保持子模性（Corollary 1），因此 Alg. 2 的贪心算法对联合目标仍保持 (1-1/e) 近似保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
子模最大化在 MMTok 中使用 PyTorch 向量化实现：(1) 使用 `torch.matmul` 构建相似度矩阵；(2) 贪心循环使用 `torch.max` 和索引操作增量计算覆盖增益；(3) 每次迭代选增益最大的 token。关键优化：维护 running max coverage，每次迭代仅需对新候选 token 计算增量 O(m+n)，而非重新计算全部 O(kmn)。MMTok 的 PyTorch 实现使 2880 tokens 选 160 仅需 6.4ms。子模函数最大化也被广泛应用于其他领域：主动学习（传感器放置）、摘要生成（文档覆盖）、推荐系统（多样性最大化）等。在 VLM token selection 场景，其关键优势是理论保证 + 高效贪心实现 + 可组合性（多目标加权和仍保持子模性）。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

## Image Contribution (IC)（图像贡献度指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Image Contribution (IC) 是 MMTok 提出的量化评估指标，衡量视觉信息对多模态任务答案的相对贡献。定义为 IC = (Perf_All - Perf_0) / Perf_0，其中 Perf_All 是使用全部 vision tokens 时的性能，Perf_0 是完全不提供 vision tokens（text-only）时的性能。IC 越高，说明该 benchmark 越依赖视觉信息；IC 越低，则该任务主要通过语言先验/文本信息即可解决，vision token selection 的效果难以体现。MMTok 发现 LLaVA-1.5-7B 在 MMMU 上 IC=0.089、ScienceQA 上 IC=0.094 —— 即仅用文本就能达到 90%+ 的全 token 性能，因而在这些低 IC 数据集上评估 token selection 方法会严重低估差异。MMTok 据此筛选出 5 个高 IC 数据集（MMB IC=2.35, POPE IC=0.92, MME IC=0.92, SEED-I IC=0.79, GQA IC=0.64）和 LLaVA-NeXT 额外 TextVQA (IC=0.62) 用于核心评估，使实验结果更能反映 token selection 方法的真实差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IC 计算流程：
```
# 对每个 benchmark dataset D
# Step 1: Full token 性能
Perf_All = evaluate(VLM, D, use_all_vision_tokens=True)

# Step 2: Zero token (text-only) 性能
Perf_0 = evaluate(VLM, D, use_all_vision_tokens=False)
          # 仅输入 text tokens, vision 侧置零或完全移除

# Step 3: 计算 IC
IC = (Perf_All - Perf_0) / Perf_0

# Step 4: 分类 — 高 IC 数据集用于 token selection 评估
if IC > threshold:   # 论文未明确给出阈值, 从 Table 4 看 ~0.4+
    mark_as_high_IC(D)
```

应用示例（LLaVA-1.5-7B, Table 4）：
| Dataset | Perf_All | Perf_0 | IC | 分类 |
|---------|----------|--------|-----|------|
| MMB | 64.7 | 19.33 | 2.347 | High IC |
| POPE | 85.9 | 44.64 | 0.924 | High IC |
| MME | 1862 | 970.89 | 0.918 | High IC |
| SEED-I | 66.14 | 37.03 | 0.786 | High IC |
| GQA | 61.9 | 37.65 | 0.644 | High IC |
| TextVQA | 58.2 | 41.66 | 0.397 | Borderline |
| SQA | 69.5 | 63.51 | 0.094 | Low IC |
| MMMU | 36.3 | 33.33 | 0.089 | Low IC |

在 Low IC 任务上，即使保留 0 个 vision token，性能下降也不显著；因此 token selection 方法的差异被压缩。MMTok 建议仅在高 IC 任务上评估 token selection 质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IC 实现简单：对每个 benchmark，运行两次完整评估（all tokens + text-only），计算相对增益。在 Lmms-eval 框架中可通过配置 `--model_args` 控制视觉 token 的提供方式。IC 的使用场景：(1) Benchmark 筛选：在评估新 token selection 方法前，先计算各 benchmark 的 IC，仅在高 IC 任务上比较；(2) 方法诊断：若方法在低 IC 任务上也表现好，说明方法可能通过更好的 vision-unrelated 决策而非更好的 vision token selection 获得提升；(3) 任务分析：帮助研究者理解哪些 VLM 任务真正需要视觉信息，指导 VLM 架构设计。局限性：IC 依赖于具体的 VLM（不同模型对零 vision token 的鲁棒性不同），需要 per-model 计算。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

## Softmax Temperature Calibration for Cross-Modal Similarity（跨模态相似度的Softmax温度校准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Softmax Temperature Calibration 是 MMTok 中用于对齐 text-vision 和 vision-vision 相似度矩阵分布的技术。由于 T-V 相似度（基于投影后对齐 LLM embedding 的 vision tokens 与 text tokens 的内积）和 V-V 相似度（基于投影前的原始 vision tokens 的内积）具有不同的量纲和分布形状，直接相加会导致一个覆盖项主导另一个。因此 MMTok 对两个相似度矩阵分别做 temperature-scaled softmax 归一化：M'_{i,j} = exp(M_{i,j}/τ) / Σⱼ exp(M_{i,j}/τ)。温度 τ 控制分布的锐度：τ 越小，softmax 越接近 one-hot（强调最相似的 token pair）；τ 越大，分布越平滑（多个 token pair 都有显著权重）。MMTok 固定 τ_t=0.02（文本-视觉温度更低，文本查询通常关注少数相关视觉区域）和 τ_v=0.2（视觉-视觉温度更高，全图信息需要更多 token 协同覆盖）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Cross-modal Softmax Calibration
# M_tv ∈ R^(m×n): text-vision 相似度 (cosine, after projection)
# M_vv ∈ R^(n×n): vision-vision 相似度 (cosine, before projection)
# τ_t, τ_v: 温度参数

# Per-row softmax with temperature
M_tv_calibrated = softmax(M_tv / τ_t, dim=-1)
# M_tv_calibrated[i,:] 是第 i 个 text token 对所有 vision token 的概率分布
# τ_t=0.02 → 锐利分布, text 主要关注 1-2 个最相关 vision region

M_vv_calibrated = softmax(M_vv / τ_v, dim=-1)
# M_vv_calibrated[i,:] 是第 i 个 vision token 对所有 vision token 的概率分布
# τ_v=0.2 → 平滑分布, 让多个 vision token 参与覆盖全图信息

# 合并覆盖目标
f(S) = f(S; M_tv_calibrated) + α * f(S; M_vv_calibrated)
```

温度选择的直觉：
- τ_t < τ_v：因为 text-vision 语义对齐更精确（投影层专门训练用于对齐），高置信度的匹配应获得更高权重
- modality gap：vision tokens（投影前）与 vision tokens 之间的相似度天然更高（同一模态），需要更高温度调整到与 T-V 可比的量级
- 消融实验（Table 9）：将 τ_v 替换为自适应搜索策略（MMTok_Adapt），在不同温度候选 {0.05, 0.1, 0.15, 0.2} 中通过 bi-section 搜索使 f(N; M^{tv'}) ≈ f_k(N; M^{vv'}) 的 τ_v，性能几乎不变，说明方法对温度不敏感

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现使用 PyTorch 的 `F.softmax(M / tau, dim=-1)`。在 MMTok 代码中，temperature 作为可配置参数。使用建议：(1) 对于 VLM 架构不变的场景，使用默认值 τ_t=0.02, τ_v=0.2, α=0.5 即可；(2) 对于新的 VLM 架构或模态（如 video），可运行自适应温度搜索（MMTok_Adapt），在验证集上搜索最优 τ_v；(3) 温度校准使相似度矩阵行归一化为概率分布，等价于将覆盖问题从 absolute similarity maximization 转化为 relative relevance maximization。该技术的通用性使其可应用于任何需要融合异源相似度矩阵的场景（如 cross-modal retrieval, multi-view clustering 等）。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

## Latent Visual Tokens（隐空间视觉 Token）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Latent Visual Tokens 是 Mirage 框架的核心创新：在 VLM 自回归解码过程中，将模型最后一层 hidden state 直接作为紧凑的连续视觉 embedding（而非通过 LM head 映射到离散 vocabulary），插入文本 token 序列中供后续 token 的 self-attention 访问。与显式图像生成 (Anole, MVoT, Chameleon) 不同，latent visual tokens 不需要 external image decoder，也不产生 pixel-level output。它们本质上是对 VLM 内部已编码的视觉信息的高效"回放"——通过 bypass LM head、reuse hidden state，将视觉推理信息以连续向量的形式保留在 multi-modal embedding space 中，供后续 reasoning step 直接 attend。k 个 latent tokens 通过 average pooling 压缩自输入 helper image I 的 patch-level features（{e_1,...,e_n} → {ê_1,...,ê_k}），k 默认=4。这种机制受到人类 mental imagery 的启发：人类在推理时不生成照片级画面，而是构建简化的心智草图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Latent Visual Token 生成 (Mirage Inference) ===
# 模型: Qwen2.5-VL-7B, k=4

# Stage 1: 生成文本前半部分 o_pre
o_pre = VLM.generate(x, stop_at="<image_placeholder>")

# Stage 2: 生成 k 个 latent visual tokens (bypass LM head)
e = []  # list of continuous embeddings
for j in range(k):
    hidden = VLM.last_hidden_state(x, o_pre, e)  # forward through LLM
    e_j = hidden[-1]  # shape: (d_model,) = (4096,) for 7B
    e.append(e_j)
    # e_j directly used as embedding for next token position
    # NOT mapped through LM head → NOT discrete token

# Stage 3: 基于 latent tokens 生成后续文本
o_post = VLM.generate(x, o_pre, e)  # o_post attends to e_{1:k}
# self-attention: Q_text @ [K_text_pre, K_e1..K_ek, K_post_<t]^T
answer = extract_answer(o_post | e_{1:k})

# === Stage 1 训练: Joint Supervision ===
patch_feats = VLM.vision_encoder(helper_image_I)  # {e_1,...,e_n}
target_embeds = avg_pool(patch_feats, k=4)         # {ê_1,...,ê_4}
# L_visual = Σ_j cos_sim(ê_j, h_j), h_j 为模型在 latent slot 的 hidden state
# L_1 = L_visual + 0.1 * L_text

# === Stage 2 训练: Latent Relaxation ===
e_j = VLM.hidden_state(x, o_pre, e_{<j})  # 模型自回归生成
# L_2 = CE(o_pre) + CE(o_post | e_{1:k})  # 仅文本 CE loss
# 梯度通过 o_post 的 CE loss 反向传播到 e_j
```

对比 Coconut (LLM continuous thought): Coconut 在纯文本 LLM latent space 中操作，无需视觉 grounding；Mirage 在 VLM 多模态空间中操作，Stage 1 提供 visual embedding distillation 锚定机制——消融显示 w/o Stage 1 会使性能从 58% 降至 21% (VSP Spatial Planning)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现点：(1) LLM decoder loop 中检测 `<image_placeholder>` token 或通过 gating mechanism 切换 latent generation path；(2) bypass LM head: forward pass 产生 hidden state h ∈ R^d 后，h 直接作为嵌入向量缓存到 KV cache，而非映射为离散 token 的概率分布；(3) 训练时 k 个 latent slots 的 hidden states 在 Stage 1 通过 cosine similarity 对齐 target embeddings，Stage 2 自回归生成并通过 downstream CE loss 接收梯度。开源：https://github.com/UMass-Embodied-AGI/Mirage。适用场景：需要视觉想象的多模态推理任务（jigsaw, spatial planning, navigation），特别适合 bypass 图像生成、避免 unified model 推理-生成冲突的场景。局限性：k>6 时性能下降（latent sequence 误差累积），目前限于 spatial reasoning benchmarks。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

## Mental Imagery for VLM Reasoning（VLM 推理中的心智图像）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mental Imagery 源于认知心理学（Shepard & Metzler 1971 的 mental rotation 实验，Kosslyn 1996 的 imagery debate），指人类在推理时不生成照片级精确画面，而是构建和操作简化的内部心智表征（mental sketches），仅捕获任务关键信息。Mirage 将这一认知理论引入 VLM 多模态推理：通过 latent visual tokens 在隐空间内构建类似 mental images 的紧凑视觉线索，替代显式图像生成。核心类比：(1) "压缩": 人类只记住碎片轮廓而非整个房间 → Mirage 用 k=4 个 average-pooled vectors 替代全部 n 个 patch features；(2) "灵活表征": 人类的心智草图是抽象而非照片级还原 → Stage 2 允许 latent tokens 从精确 visual match 中偏离，自适应任务需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mental Imagery 的三个认知阶段在 Mirage pipeline 中的对等实现：
```
# 1. 编码 (Encoding): 感知输入 → 内部表征
#    人类: 看拼图碎片 → 提取边缘轮廓的简化特征
#    VLM: vision_encoder(image) → patch features → projection
#         → LLM hidden representations (multi-layer)

# 2. 操作 (Manipulation): 在心智中操纵表征
#    人类: 脑中旋转拼接碎片
#    VLM: self-attention 在 latent visual tokens e_{1:k} 与 text tokens
#         之间进行信息融合，latent tokens 作为 key-value 供后续查询

# 3. 提取 (Extraction): 从心智表征得出结论
#    人类: 判断匹配/不匹配
#    VLM: text_post tokens attend to e_{1:k} → LM head → answer
```

t-SNE 可视化 (Fig. 7) 验证了 mental imagery 的设计：Stage 2 后 latent tokens (red dots) 聚集在 visual cluster (yellow dots) 外侧而非内部——保持了 Stage 1 的 visual subspace 亲和性，同时体现了 Stage 2 的任务导向偏移。"simplified sketch, not photorealism."

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现通过两阶段训练：(1) Stage 1: helper image → compressed visual embeddings → cosine similarity loss 锚定 latent tokens 到 visual subspace；(2) Stage 2: 移除 cosine loss，仅文本 CE 监督，梯度反传使 latent tokens 在 visual subspace 附近自适应优化。使用场景：需要 "视觉想象" 而非 "视觉识别" 的多模态推理任务。对比纯文本 CoT：textualization 是对视觉信息的二次编码损失，mental imagery 保留了 first-order 视觉结构。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

## Interleaved Multimodal Reasoning（交织多模态推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Multimodal Reasoning 指推理轨迹中文本 token 和视觉 token（latent 或 explicit）交替排列的推理模式，而非将所有视觉信息放在序列开头或在固定位置插入。Mirage 的推理链格式：text_pre → [latent_1, latent_2, latent_3, latent_4] → text_post。与 vision-first (图像在 prompt 开头一次性输入) 和 text-only CoT 的关键区别：模型在推理过程中可以动态决定 "何时需要视觉信息"，形成 text-vision 混合的 reasoning trajectory。数据合成方式：helper image I 嵌入到 textual reasoning chain 中间 (o = o_pre ⊕ I ⊕ o_post)，训练模型学习这种交织模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Interleaved Reasoning 轨迹示例 (VSP Spatial Reasoning)
# Query: "Will the agent safely reach the goal after: Go Left, Go Down?"

# 自回归生成如下交织序列:
# text_pre (模型直接生成):
#   "Moving to [(1,3),Hole] ends the game instantly..."
#   触发 <image_placeholder> → latent generation

# latent_tokens (k=4, bypass LM head, 连续 embedding):
#   [e_1, e_2, e_3, e_4]  ← encode 路径空间信息

# text_post (attend 到 latent tokens):
#   "...making failure certain. The answer is \boxed{B}."

# Attention pattern for text_post token at position t:
# Q_t · [K_{text_pre}, K_{e1}, K_{e2}, K_{e3}, K_{e4}, K_{post_<t}]^T
# latent K/V entries 为 text_post 提供 task-specific visual cues

# GRPO RL 阶段: 模型可自由探索不同的 interleaved 模式
# latent tokens 排除于 KL penalty (λ_kl 仅应用于 text tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 数据合成 pipeline 生成 interleaved 训练样本（helper image 嵌入推理链中间，o = o_pre ⊕ I ⊕ o_post）；(2) 特殊 `<image_placeholder>` token 触发 latent generation 模式切换（bypass LM head → direct hidden state as embedding）；(3) GRPO RL 阶段 latent tokens 不受 KL constraint，允许模型探索不同交织模式。对比 unified model (Anole/MVoT) 的 interleaved generation：Mirage 的 latent visual tokens 不需要 external image decoder，避免了 pixel generation 开销和推理质量退化。适用场景：需要推理过程中多次参考视觉信息的多模态任务（spatial reasoning, jigsaw puzzle, navigation）。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

## Cosine Similarity Loss for Hidden State Alignment（隐状态对齐的余弦相似度损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 Mirage Stage 1 中用于将模型 latent slot 位置的 hidden states 锚定到 visual embedding subspace 的损失函数。损失形式：L_visual = (1/k) Σ_{j=1..k} [1 - cos_sim(ê_j, h_j)]，其中 ê_j 是从 helper image I 压缩得到的 target visual embedding（通过 average pooling），h_j 是模型在对应 latent slot 位置的 hidden state（prediction）。余弦相似度 cos_sim(a, b) = a·b / (||a||·||b||) 度量两个向量的方向对齐程度。选择 cosine similarity 而非 MSE 的理由：(1) hidden states 和 visual embeddings 处于不同 latent subspace，但方向 encode 关键语义；(2) 对向量模长不敏感，防止模型通过放大隐藏状态模长 "cheat"；(3) 在高维空间 (d=4096) 中方向比尺度更重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: Cosine Similarity Loss for Latent Grounding

# Input: helper image I, 模型 VLM f_θ, k=4

# Step 1: 提取压缩后的目标 visual embeddings
patch_feats  = f_θ.vision_encoder(I)        # [n_patches, d_model]
target_embeds = avg_pool(patch_feats, k=4)   # [4, d_model] = {ê_1,...,ê_4}

# Step 2: Forward pass 获取 hidden states at latent slots
h_1 = f_θ.hidden_state(x, o_pre)                    # 第一个 latent slot
h_2 = f_θ.hidden_state(x, o_pre, ê_1)               # 第二个, conditioned on ê_1 (teacher forcing)
h_3 = f_θ.hidden_state(x, o_pre, ê_1, ê_2)          # 第三个
h_4 = f_θ.hidden_state(x, o_pre, ê_1, ê_2, ê_3)    # 第四个

# Step 3: 计算 Cosine Similarity Loss
L_visual = 0
for j in range(1, 5):
    cos_sim_j = (ê_j · h_j) / (||ê_j|| * ||h_j||)
    L_visual += (1 - cos_sim_j)
L_visual /= 4

# Step 4: 联合文本 CE Loss
L_text = CE(o_pre) + CE(o_post | o_pre, ê_{1:k})
L_total = L_visual + 0.1 * L_text  # γ=0.1
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `F.cosine_similarity(h_j, ê_j, dim=-1)` 返回 (cos_sim+1)/2 带范围 [-1,1]，或直接 `1 - F.cosine_similarity(...)`。γ=0.1 意味着 visual alignment loss 权重约为 text CE 的 10 倍，确保 Stage 1 latent grounding 有效性。消融 (Tab 5): γ=0.1 → 87% avg, γ=0.5 → 84%, γ=1 → 83%；γ→∞ (跳过 Stage 1) → 21% (w/o Stage 2 setting)，证明 cosine loss 提供的 visual grounding 是关键初始化。使用场景：任何需要将模型内部表征对齐到特定 target embedding 的跨模态场景（multimodal alignment, knowledge distillation, latent reasoning）。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

## Vision Token Bidirectional Attention (视觉Token双向注意力)

术语解释
在VLM中让不同来源（不同视频帧、不同图像、不同crop）的vision tokens在LLM的self-attention层中互相attend的注意力策略，打破标准causal attention中vision token仅被前置token attend的单向限制。Molmo2发现启用cross-frame/cross-image双向注意力能显著提升视频理解性能（Table 8b: +0.4 QA avg, +1.0 Cap F1）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
标准VLM的LLM decoder使用causal attention mask——每个token只能attend到它之前的token。这意味着frame_2的vision tokens无法attend到frame_1的vision tokens（它们都在text prompt之前）。Vision Token Bidirectional Attention修改attention mask：所有vision tokens区域（[video_start]到[video_end]之间）设为全1（bidirectional），而text tokens保持标准causal。使不同帧/位置的vision tokens能直接交互信息（跨帧object re-identification、motion understanding），而非仅依赖text token作为中间桥梁。Gemma 3等模型也采用类似设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Custom Attention Mask for Bidirectional Vision Tokens
# Sequence: [BOS][video_start][frame1_tokens][t0.0][frame2_tokens][video_end][text]
#           |← vision token region (bidirectional) →| |← text (causal) →|

mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
# Vision region: bidirectional (all-to-all)
mask[vis_start:vis_end, vis_start:vis_end] = 0.0
# Text keeps causal, can attend to all preceding vision
# Use PyTorch SDPA (NOT FlashAttention — doesn't support arbitrary custom masks)
attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
必须使用SDPA而非FlashAttention（因不兼容custom mask）。与torch.compile兼容（static shape required）。与message tree encoding + packing的custom mask协同叠加。适用场景：任何多帧/多图VLM的LLM decoder训练和推理。计算成本：vision region ~10K tokens时增加额外~10K^2/2 pairwise attention（vs causal下三角），但Molmo2通过packing和pooling已在其他维度压缩。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

## Message Tree Encoding (消息树编码)

术语解释
Molmo2提出的多模态训练数据编码策略，允许同一视觉输入携带多个annotations（QA pairs、caption、pointing等）并打包进同一training sequence，通过custom attention mask防止不同annotation分支之间的cross-attention污染。平均每个visual input有4 annotations，与packing结合实现~15x训练效率提升。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
传统做法将每个annotation作为独立training example，同视觉tokens被重复编码多次。Message Tree Encoding以visual input tokens为root，每个(Q,A) pair为独立branch。Linearization: visual tokens出现一次→各branch顺序拼接。Custom attention mask: (1) branch之间block（防止QA A的answer泄露到QA B）；(2) 每个branch可见full visual tokens；(3) visual tokens内部bidirectional。与packing协同：message tree产生的merged sequence作为packing DP solver的输入unit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Sequence: [BOS][video][Q1][A1][Q2][A2]
#                   |← root →|← branch1 →|← branch2 →|
# Mask:  root×root→bidir; branch_i×branch_i→causal; branch_i×branch_j→BLOCK
# All branches can see root (full vision context)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于Molmo2 training code (`message_tree.py`)。Custom mask集成入PyTorch SDPA。适用场景：同一视频/图像有多个训练annotations的VLM SFT训练。Molmo2开源代码在 https://github.com/allenai/molmo2。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

## Token Weighting for Multimodal SFT Loss (多模态SFT损失加权)

术语解释
对VLM SFT阶段不同任务类型的loss token施加差异化权重，解决长输出样本（如4000+ tokens video caption）主导loss导致短输出任务（如MCQ仅1 token）性能退化的问题。Molmo2采用固定权重（caption=0.1, pointing=0.2）+ sqrt inverse weighting（其他任务 weight=4/√n）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLM SFT数据mix包含输出长度差异极大的任务。一个4000-token caption sample对总loss的贡献是1-token MCQ的4000倍。即使降低sampling rate，其token count仍主导梯度。Token Weighting在per-token CE loss上乘weight：(1) video caption weight=0.1；(2) pointing weight=0.2；(3) 其他任务 weight=4/√n_answer_tokens。n=1 (MCQ)→4.0, n=100→0.4, n=400→0.2。单个sample total weighted loss增长从O(n)降为O(√n)。Molmo2 Table 8b: 去除token weighting导致QA avg -0.8但Caption F1 +0.5——验证QA↔Caption trade-off。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
if task == 'video_caption': weight = 0.1
elif task == 'pointing':    weight = 0.2
else:                       weight = 4.0 / sqrt(n_answer_tokens)
weighted_loss = token_loss * loss_mask * weight  # Per-token weighting
# Gradient averaged over global mean tokens across all devices (not per-device)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Per-device gradient除以全局平均loss token数（非per-device own count），避免短样本隐式up-weight。适用场景：任何含输出长度差异大的多任务VLM SFT训练。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

## Point-based Spatio-temporal Grounding (基于点的时空定位)

术语解释
VLM通过生成带时间戳+空间坐标的point来定位视频中的object/action/event，而非仅输出文本/bbox。包括Video Pointing（一次性标注多个时空点，obj_id for counting）和Video Tracking（连续标注object轨迹with consistent IDs across frames）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将image grounding的2D (x,y)点标注扩展到视频的3D (x,y,t)域。核心格式：`<points coords="ts obj_id x y;...">` 或 `<tracks coords="ts obj_id x y;...">`，ts=秒（1 decimal），obj_id=unique sequential ID, (x,y)=0-1000 normalized coords。Pointing→Counting: 先point→max(obj_id)得count（"point then count" strategy）。Tracking: same obj_id跨多帧→track trajectory, HOTA评估association accuracy。Molmo2-VideoPoint: 650K human queries (8 categories, avg 2.3 points/query, 280K videos)。Molmo2-VideoTrack: 15K queries (avg 2.28 objects/query, 3.6K clips from diverse VOS + bbox datasets via SAM 2 point extraction)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Pointing output: <points coords="2.5 1 320 450;5.0 2 680 320">dogs</points>
# → Dog1 at 2.5s(x=320,y=450), Dog2 at 5.0s(x=680,y=320), count=2

# Tracking output: <tracks coords="0.0 1 635 522;0.5 1 606 490;1.0 1 515 164">person</tracks>
# → Person (obj_id=1) track: moves from (635,522)→(606,490)→(515,164) over 1s

# Evaluation: 
# Pointing F1 = point in GT mask? (2 fps sampling, window-based tolerance)
# Tracking HOTA = sqrt(DetA × AssA), DetA=binary mask hit, AssA=ID consistency
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
点坐标归一化至0-1000（image-resolution-independent）。格式选择HTML-like (token-efficient) vs JSON。Tracking数据pipeline: segmentation/bbox tracks→SAM 2 point extraction (alpha-weighted centroid+boundary distance score)→Human text query annotation+validation。训练: upsampled high-count examples + auxiliary tasks (first/last frame only, single-point tracking)。Molmo2 video pointing F1=38.4 (vs Gemini 3 Pro 20.0), tracking HOTA=57.5 (vs Gemini 3 Pro 29.1)。适用于视频搜索、机器人、安防等需要pixel级时空定位的应用。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

## SlowFast Video Encoding for VLM Inference (VLM推理SlowFast编码)

术语解释
VLM推理时对不同视频帧使用不同空间pooling ratio: 关键帧用fine-grained pooling (3×3→~81 tokens/frame)，非关键帧用coarse pooling (9×9→~9 tokens/frame)，在固定total vision token budget下覆盖更多帧。Molmo2在training-free + query-based frame selection模式下实现。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
借鉴SlowFast-LLaVA，在Molmo2 connector的MH attentional pooling层改变window size: default 3×3, fast=9×9。Frame selection策略：(1) periodic (every p-th frame slow)；(2) diff-based (相邻帧feature差异)；(3) query-based (SigLIP 2 embedding cosine similarity between query and each frame)。p selection: 动态选择p∈{1,2,3,4}使total tokens≈10.6K。Key finding: training-free query-based SlowFast在~43% fewer tokens下匹配224 frames全分辨性能；甚至training with SlowFast 10%后反而不如training-free query-based（说明Molmo2可zero-shot generalize到9×9 pooling）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Dynamic p selection based on sampled frame count
if F_t <= 128: p = 1; elif F_t <= 224: p = 2; elif F_t <= 300: p = 3; else: p = 4

# Query-based frame selection for slow pathway:
scores = cosine_sim(SigLIP2(query), SigLIP2(frames))  # [F_t]
slow_frames = select_top_global(scores, F_s//2) + select_best_per_group(scores, F_s//2 groups)

# Encode with different pooling:
slow_tokens = connector(slow_frames, pool=3)  # 81 tok/frame
fast_tokens = connector(fast_frames, pool=9)  # 9 tok/frame
# Interleave by temporal order, total ≈ 10.6K tokens
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
当FPS≥2时fallback到periodic（高FPS下frame selection不必要）。与connector的MH attentional pooling天然兼容——仅改变window size。Molmo2 training code支持此功能。适用于>2min长视频VLM推理，需在fixed token budget下覆盖更多视觉信息。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

## Temporal Dynamic Context (TDC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Dynamic Context (TDC) 是一种多模态长视频编码框架，由 Hao et al. 提出。核心思想：将视频表示分解为静态视觉特征（static visual features）和动态多模态上下文（dynamic multimodal context）。对于每个视频场景，首帧完整保留（144 visual + 50 audio tokens）作为静态参考帧；后续帧通过 Q-Former cross-attention 压缩为 K 个 context tokens（默认 K=16），这些 context tokens 聚合了帧间时序变化和视觉-音频跨模态信息。TDC 将每帧平均 token 数从 ~194 压缩至 16，使得 LLM 可在固定 context window 内处理更多帧。相比 prior work（VideoLLaMA2 仅采样 16 帧，token 简单拼接），TDC 以 1fps 密集采样所有帧，通过语义场景分割保证时序一致性，用 Q-Former 压缩替代简单采样/丢弃。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TDC 完整 Pipeline
# 输入: video (T seconds), audio, question text F_s
# 输出: compressed video representation F_TDC

# Step 1: Scene Segmentation
frames = sample(video, fps=1)                # T frames, 1fps
emb = DINOv2(frames)                         # frame-level features
sim = cosine_sim(emb[i], emb[i+1])           # inter-frame similarity
split_points = top_k_lowest(sim, S-1)        # S≤24 semantic boundaries
scenes = segment(frames, split_points)        # S semantically consistent scenes

# Step 2: Per-Scene TDC Encoding
for scene in scenes:  # sliding window length N
    # Static: first frame fully retained
    F_x1 = SigLIP(scene[0])                  # (144, D)
    F_a1 = BEATs(audio[0])                   # (50, D)

    # Query generation via AvgPool
    Q = AvgPool(F_x1)                        # (K=16, D)

    # Dynamic: compress subsequent frames
    F_TDC = [F_x1, F_a1, <Sep>]
    for i in 2..N:
        F_xi = SigLIP(scene[i]); F_ai = BEATs(audio[i])
        F_Q_i = QFormer(Q, [F_xi·F_ai], F_s)  # cross-attn → (16, D)
        F_TDC.append(F_Q_i)

# Step 3: LLM Decoding
answer = LLM(F_TDC, F_s)
```
关键设计：(1) AvgPool queries 优于 learned queries；(2) instruction text F_s 注入 Q-Former 使压缩自适应于问题；(3) <Sep> token 区分静态和动态 token。消融：S=1 (不分割) MVBench 62.7→53.5 (-9.2)；text instruction 对长视频帮助最大 (MLVU Long -1.6)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TDC 在 PyTorch + HuggingFace Transformers 上实现。Visual encoder (SigLIP+DINOv2) 和 audio encoder (BEATs) 冻结；Q-Former (BERT-initialized) 和 LLM (Qwen2-7B / LLaMA3.2-3B) 可训练。三阶段训练：Stage 1 视觉-语言对齐 (LLaVA-OneVision 3.2M)，Stage 2 视频指令微调 (2M/540K)，Stage 3 音频-视频指令微调 (300K/120K + LoRA)。开源: github.com/Hoar012/TDC-Video。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## Long Video Chain-of-Thought (LVCoT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Long Video Chain-of-Thought (LVCoT) 是 TDC 论文提出的训练无关（training-free）超长视频推理策略。当视频过长导致 LLM context window 无法容纳全部 token 时，LVCoT 将视频等分为 M 段（默认 M=3），每段独立进行 TDC 编码和推理，生成段级中间答案；所有段答案及时间戳拼接形成 "chain-of-thought" 推理链，最后基于全局视频和推理链生成最终答案。与 prior work 的区别：(1) Goldfish/StreamingLLM 通过 key frame selection 处理但破坏时序连续性；(2) VideoRecap 采用层级策略但限于 captioning；(3) VideoCoT 需训练数据且仅用于短视频。LVCoT 训练无关、任务无关，可应用于任意 MLLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LVCoT 推理流程 (training-free)
# 输入: long video V, question Q, M=3 segments
# 输出: final answer

segments = divide_equally(V, M)               # e.g., 720s → 3×240s
thoughts = []
for k, seg in enumerate(segments):
    F_TDC_seg = TDC_encode(seg)
    ans_seg = LLM(F_TDC_seg, Q)
    t_start, t_end = seg.time_range
    thoughts.append(f"From {t_start}s to {t_end}s: {ans_seg}")

# Global reasoning with accumulated CoT
F_TDC_full = TDC_encode(V)
chain = concat(thoughts)
final_answer = LLM(F_TDC_full, f"{Q}\n\nThought process:\n{chain}")
```
消融实验（Table 4e）：7B w/ LVCoT on MLVU: 63.9→64.1, VideoMME Long: 61.3→61.8 (+0.5)。增益随视频长度增加而增大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
核心开销：M+1 次 LLM forward pass（vs 1 次无 LVCoT）。适用场景：视频时长超过 LLM context window 可容纳 token 数时。局限性：有效性依赖 LLM 推理能力，模型未经 CoT 训练提升较小；论文指出未来方向包括训练模型更好利用此策略和建立更高效内存机制。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## BEATs (Bidirectional Encoder representation from Audio Transformers)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BEATs 是 Microsoft 提出的自监督音频预训练框架（Chen et al., ICML 2023 Oral），核心理念：通过迭代方式联合优化 acoustic tokenizer 和 audio SSL model。先用随机投影 tokenizer 生成离散标签（cold start），再通过知识蒸馏训练 self-distilled tokenizer，tokenizer 生成的离散标签用于下一轮 SSL 预训练，逐步抽象高层语义。架构：ViT-like 12层 Transformer encoder，convolutional relative position embedding，gated relative position bias，DeepNorm。预训练任务：Masked Audio Modeling (MAM)——随机 mask 75% 输入 patches，预测 masked positions 的离散标签。输入预处理：16kHz 重采样 → 128维 Mel filterbank (25ms window, 10ms hop) → 16×16 patches。~90M 参数，AudioSet-2M mAP 48.6%（单模型 SOTA）。TDC 论文中用作冻结的音频编码器，每 1 秒音频输出约 50 tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BEATs 在 TDC 中的使用 (frozen feature extractor)

# Preprocessing
audio_16k = resample(raw_audio, target_sr=16000)
mel_spec = mel_filterbank(audio_16k, n_mels=128, win=25ms, hop=10ms)
mel_norm = normalize(mel_spec, mean=0, std=0.5)
patches = split_into_patches(mel_norm, patch_size=16)  # 16x16 patches

# BEATs forward (frozen)
audio_tokens = BEATs_encoder(patches)          # (≈50, 768) per second

# In TDC pipeline: concat with visual tokens → Q-Former
F_ai = audio_tokens
F_xi = SigLIP(video_frame)                     # (144, D)
F_Q_i = QFormer(Q, [F_xi · F_ai], F_s)         # cross-modal compression
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BEATs 开源: github.com/microsoft/unilm/tree/master/beats (MIT License)。加载：通过 HuggingFace/fairseq 加载预训练权重。下游任务达到 SOTA: ESC-50 98.1%, AudioSet-2M 48.6%, KS 98.1%。TDC 中作为冻结特征提取器（不训练），输出约 50 Hz temporal resolution 的 patch-level audio features。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## Video Scene Segmentation via Inter-Frame Similarity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Scene Segmentation via Inter-Frame Similarity 是 TDC 论文提出的基于帧间语义相似度的场景分割方法。与传统按固定时长切分（fixed-duration clips）不同，使用 DINOv2 提取每帧 768-d embedding，计算连续帧 cosine similarity，选择 S-1 个相似度最低的帧对位置作为场景边界。每个 segment 内部语义一致，segment 间语义差异最大，保证后续 TDC 压缩在时序一致的上下文内进行。最大 segment 数 S=24。消融：S=1 (不分割) MVBench 下降 9.2 点；S=48 无额外提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Scene Segmentation Pipeline
embeddings = [normalize(DINOv2(f)) for f in frames]  # T × 768-d
sims = [(i, dot(embeddings[i], embeddings[i+1])) for i in range(T-1)]
sims.sort(key=lambda x: x[1])                        # ascending similarity
split_idx = sorted([s[0] for s in sims[:S-1]])       # S-1 lowest-sim boundaries

scenes = []
prev = 0
for idx in split_idx:
    scenes.append(frames[prev:idx+1])
    prev = idx + 1
scenes.append(frames[prev:])                          # last scene
```
DINOv2 特征对光照/视角鲁棒、对内容变化敏感。低相似度点即语义变化点（场景切换、物体出现/消失）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DINOv2 加载: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`。计算开销：T 次 ViT forward + T-1 次 cosine similarity。类似 DIG 的 CAFS 但目的不同：CAFS 选代表性帧，TDC 选分割边界。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## Instruction-Guided Video Token Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Instruction-Guided Video Token Compression 是 TDC 论文提出的将用户指令注入视频 token 压缩过程的技术。在 Q-Former cross-attention 阶段，将 instruction text tokens F_s 作为额外 key-value 输入（公式: F_Q^i = QFormer(Q, [F_xi · F_ai], F_s)），使 Q-Former 根据问题语义自适应决定每个 query token 从视觉/音频 tokens 中提取什么信息。相比无 instruction，加入 F_s 后压缩不仅捕捉帧间时序变化，还能聚焦与问题相关的细节。消融（Table 4d）：with text vs without text: MVBench 62.7 vs 62.3, MLVU Long 59.6 vs 58.0 (-1.6), VideoMME 52.7 vs 51.5 (-1.2)。instruction 对长视频帮助更大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Q-Former with Instruction-Guided Compression
# Input: Q(16 query tokens), F_xi(144 visual), F_ai(50 audio), F_s (text)

T_s = tokenize(F_s); E_s = embed(T_s)              # text embeddings
K = linear_k(concat(F_xi, F_ai, E_s))              # (144+50+L_s, d)
V = linear_v(concat(F_xi, F_ai, E_s))
Q_proj = linear_q(Q)                                # (16, d)

attn = softmax(Q_proj @ K.T / sqrt(d))              # (16, 144+50+L_s)
F_Q_i = attn @ V                                    # (16, d) compressed
```
对比 prior work: LongVU 基于 visual similarity + query relevance 压缩后再筛选，TDC 在压缩过程中（Q-Former 内部）注入 instruction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TDC 在 Q-Former forward 中增加 text_cross_attention 参数接收 instruction embeddings。instruction text 仅参与 key-value 计算，不改变输出维度（仍 K tokens）。计算开销：key-value 长度增加 L_s (通常 10-50 tokens)，可忽略。适用场景：任何 cross-attention compressor 均可加入 instruction guidance 提升压缩质量。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

## Codec Patchification（编解码器分块）

术语是什么？
Codec Patchification 是 OneVision-Encoder 提出的编解码器引导的视觉 token 输入构造策略。核心思想：HEVC/H.265 编解码器天然将视频分解为"稳定空间上下文"（I-frame 全量编码）和"稀疏时序更新"（P-frame 仅运动补偿残差），这种分解揭示了判别性视觉证据（"surprise"）仅稀疏存在于运动/变化区域。Codec Patchification 利用 HEVC 暴露的运动矢量（motion vectors）和预测残差（prediction residuals）作为 patch 级信息熵代理，仅在密集视频帧（64帧）中选择 3.1%–25% 最显著 patch 送入 ViT。包含三种形式：Dense Video-Codec Patchification（GOP结构，I-frame全量 + P-frame稀疏）、Chunk-wise Patchification（均匀分块+单帧采样）、Single-Image Spatial Patchification（静态图像行主序 patchify）。

从算法pipeline角度拆解术语：
Codec Patchification 在 ViT 编码前的数据预处理阶段运作：

```
# 64帧视频, GOP=32, token budget B=2048
# Step1: HEVC解码提取motion vectors + residuals (CPU)
for each GOP: decode I-frame(RGB) + extract mv,res per P-frame

# Step2: Patch级显著性评分
for each P-frame patch(y,x):
    saliency = sum(||mv||₂ over patch) + sum(|res| over patch)

# Step3: 全局Top-K (跨所有P-frames)
selected = topk(all_P_saliency, k=B-512)  # 512=2个I-frame全量
tokens = concat(I_patches (512), P_patches[selected] (1536))

# Step4: 3D-RoPE + ViT encoding + attentive pooling
tokens = tokens + 3D_RoPE(sparse_positions)
features = ViT(tokens) → attentive_pooling(features)
```

训练时三种模式混合：Codec 50%, Frame Sampling 37.5%, Tiling 12.5%。Token budget 固定在 clip level（非 per-GOP），确保全局最优分配。推理可灵活切换 Codec 稀疏或传统帧采样。

术语一般如何实现？如何使用？
使用 FFmpeg/libx265 提取 motion vectors 和 residuals（不重编码，直接从原始 bitstream 提取）。显著性评分在 CPU 上计算，选中的 patch indices 通过 visible_indices 传给 ViT，未选中 patch 不存储/不计算/不传梯度。Token 压缩：64帧×256 patches=16384 → 2048=87.5% reduction。限制：需 HEVC 格式存储；无法处理实时流的未来帧。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## Cluster Discrimination（聚类判别）

术语是什么？
Cluster Discrimination 是一种自监督表示学习范式：先用 frozen encoder 对大规模数据提取特征，k-means 聚类为语义中心（centroids），再用这些全局聚类中心作为 pseudo-labels 进行多标签语义判别训练。与传统 contrastive learning（CLIP/SigLIP, instance-level + batch-local negatives）和 masked modeling（MAE, pixel reconstruction）不同，Cluster Discrimination 利用全局语义结构而非局部对比。OV-Encoder 扩展到双模态：图像嵌入→2M object-level 类中心，视频嵌入（16帧 concat 特征）→400K motion-level 类中心，合并为 C_uni = C_obj ∪ C_vid（2.4M 类中心）。训练时图像仅对照 C_obj，视频仅对照 C_vid。使用 sigmoid BCE（非 softmax），因一个样本可同时属于多个语义类别（multi-label）。

从算法pipeline角度拆解术语：

```
# 离线聚类（frozen metaCLIP-H14）
e_img = metaCLIP(image)                  # [D] per image
C_obj = kmeans({e_img}, K=2M)            # 2M image centroids
e_vid = metaCLIP(uniform_16_frames)      # [16,D] → concat → [16D]
C_vid = kmeans({e_vid}, K=400K)          # 400K video centroids
per sample: assign top-10 nearest centroids as positive labels

# 在线训练
e = OV_Encoder(sample)                   # ViT + attentive pooling
sim = e @ C_m.T                          # [1, K_m]
loss = sigmoid_BCE(sim, multi_hot_labels)  # multi-label per sample
loss = loss_obj + loss_vid
```

负采样率 r=0.1（仅计算 10% 负类中心），正标签数 l=10，K_obj=2M, K_vid=400K。

术语一般如何实现？如何使用？
离线聚类使用 mini-batch k-means，类中心矩阵作为可学习参数存储于 GPU。适用场景：纯视觉预训练（无语言监督），需要建模细粒度语义结构（intra-class consistency + inter-class relationship）。相比 CLIP-style：优势是全局语义结构，劣势是依赖 frozen encoder 聚类质量。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## 3D Rotary Position Embedding (3D-RoPE)

术语是什么？
3D-RoPE 将 Rotary Position Embedding 从 1D（语言序列）扩展到 3D（视频时空）。标准 RoPE 通过旋转矩阵对 Q/K 施加位置相关旋转变换，使 attention score 仅依赖相对位置。3D-RoPE 编码 (t, x, y)，对应时间维度和两个空间维度。OV-Encoder 使用 3D-RoPE 的核心原因是 Codec Patchification 产生的 token 布局高度不规则（不同 sample 选中的 patch 来自不同帧的不同位置），绝对位置编码无法在此类稀疏布局下保持一致性。3D-RoPE 的相对方案（Δp = (t1-t2, x1-x2, y1-y2)）天然适配。

从算法pipeline角度拆解术语：
频率分配 T:H:W=4:6:6（对应 16 attention heads）。三种 Δp 定义：
- Dense Video-Codec: Δp = (t_i-t_j, x_i-x_j, y_i-y_j)
- Chunk-wise: Δp = (c_i-c_j, x_i-x_j, y_i-y_j)
- Single-Image: Δp = (0, x_i-x_j, y_i-y_j)

```
# 在 ViT self-attention 前应用
def apply_3d_rope(q, k, Δp=(dt, dx, dy)):
    q[:, :d_t], k[:, :d_t] = rope_rotate(q, k, dt, freq_t)
    q[:, d_t:d_t+d_h], k[:, :] = rope_rotate(q, k, dx, freq_h)
    q[:, d_t+d_h:], k[:, :] = rope_rotate(q, k, dy, freq_w)
    return q, k
```

术语一般如何实现？如何使用？
与 Flash Attention 2 兼容（RoPE 是 pre-attention QK 变换）。三种输入共享同一 3D-RoPE 参数，推理时自动按输入类型选择 Δp 方案。使用场景：任何需要统一处理图像和视频的共享 ViT，特别是稀疏/不规则 token 布局。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## Group of Pictures (GOP) in Visual Representation Learning

术语是什么？
GOP 是视频编解码标准中的基本编码单元：一个 I-frame（独立编码的完整帧）+ 若干 P/B-frame（依赖参考帧的预测帧）。OV-Encoder 将 GOP 引入视觉表示学习：每个 GOP 的 I-frame 全量编码（256 patches 建立空间上下文），P-frames 仅保留运动+残差显著 patches（3.1%-25%）。GOP 的关键作用不是编解码效率，而是结构化时空分解：I-frame 提供"what is where" 空间锚点，P-frame 提供"what changed" 时序更新。

从算法pipeline角度拆解术语：
OV-Encoder 配置：64帧, GOP=32 → 2个GOP。每个 GOP: 1 I-frame (全256 patches) + 31 P-frames (仅选显著)。Token budget 2048 = 2×256 (I-frames) + 1536 (P-frames top-K)。关键：budget 跨越 GOP 全局排序（非 per-GOP），确保最优分配。GOP size 超参数：太小 → I-frame 频繁刷新占 token；太大 → 长时间无空间上下文刷新，累积预测误差。

术语一般如何实现？如何使用？
GOP 结构在 HEVC 解码时自然产生（编解码器原生支持），不需额外处理。解码器直接按 GOP 输出 I/P 帧及其 motion vectors/residuals。训练和推理使用相同 GOP 配置。限制：非 HEVC 视频需先转码。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## Attentive Pooling（注意力池化）

术语是什么？
Attentive Pooling 通过可学习的 query 向量对 token 序列执行 cross-attention，将变长序列聚合为固定维度表示。与 average pooling（等权平均）和 max pooling 不同，attentive pooling 学习 token 的重要性权重，自动聚焦判别性区域。OV-Encoder 采用 multi-head attention pooling（源自 SigLIP），用少量可学习 query tokens 对 ViT 输出的全部 2048 个 spatiotemporal tokens 做 cross-attention，生成 compact embeddings 用于 cluster discrimination 损失。

从算法pipeline角度拆解术语：

```
# Multi-Head Attention Pooling
Q = learnable_query[N_queries, d]        # N_queries << M (通常1-4)
K, V = Z @ W_k, Z @ W_v                  # Z: ViT输出 [M, d]
attn = softmax(Q @ K.T / sqrt(d))        # [N_queries, M]
pooled = attn @ V                         # [N_queries, d]

# OV-Encoder 直接作用于 ViT 最后一层所有 patch tokens（无[CLS]）
# Codec Patchification 输入端筛选 + Attentive Pooling 输出端加权
```

术语一般如何实现？如何使用？
PyTorch 实现：`nn.MultiheadAttention` + 可学习 `nn.Parameter` query tokens。源自 SigLIP 的 MAP 设计。query tokens 随 ViT 一起优化。相比 [CLS] token：query 不与输入 token 耦合，可增加 N_queries 提升表达力。应用：任何需要将变长序列聚合为固定向量的 ViT/Transformer 输出端。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## Motion Vector Guided Token Selection（运动矢量引导的Token选择）

术语是什么？
利用 HEVC 解码时暴露的运动矢量（motion vectors）和预测残差（prediction residuals）作为 patch 级信息熵代理的 token 筛选策略。HEVC 的 P-frame 每个 coding unit (4×4~64×64) 关联运动矢量 d∈R² 和残差信号。运动矢量大小 ||d||₂ 反映局部运动强度，残差能量 |res| 反映外观变化不可预测性。两者相加作为 patch saliency score，从密集 P-frames 中筛选最具信息量的 patches。优势：motion vectors 是 HEVC bitstream 解码的"免费"副产品，无需额外模型推理。

从算法pipeline角度拆解术语：

```
# CU级 → Pixel级 → Patch级
mv_field[pixels] = broadcast(CU_motion_vectors)    # [H,W,2]
for each patch(i,j):
    motion_score = sum(||mv_field[patch]||₂)        # 运动强度
    residual_score = sum(|res[patch]|)               # 残差能量
    saliency[i,j] = motion_score + residual_score

global_topk_indices = topk(all_P_saliency, k=B_P)   # 跨所有P-frames
```

运动矢量需 camera motion compensation（去除全局相机运动）。saliency = motion + residual（等权相加，无学习参数）。选择是全局 Top-K（跨所有P-frames排序，非 per-frame）。

术语一般如何实现？如何使用？
FFmpeg with `-flags2 +export_mvs` 导出运动矢量，或 libx265 API 直接读取。残差信号从 YUV Y-channel 解码。CPU 上计算 saliency → 选中 patch indices → GPU tensor → ViT forward。适用场景：内容自适应的视频 token 分配；推广到 H.264/AV1/VVC 等编码标准。限制：非编码视频需先转码。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

## Omni World Model for Video Generation（视频生成全向世界模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Omni World Model (Owl-1) 是一种从世界模型视角解决长视频生成一致性问题的方法。核心思想：视频本质上是对底层演化世界的观测记录，因此长视频的一致性应从隐式世界的连贯性角度来保证，而非在像素空间进行帧间拼接。Owl-1 将世界建模为一组 latent state variables {s_t}，每个 s_t 编码当前时刻和历史所有信息。构建闭环 state-observation-dynamics 三元组模拟世界演化：(1) State Decoder D: o_t = D(s_t, o_{t-1})，将隐式状态解码为显式视频观测，s_t 负责长期一致性，o_{t-1} 负责短期平滑；(2) World Dynamics Prediction f: d_t = f(s_t, o_t)，从观测和状态预测未来世界动态（文本形式）；(3) State Update g: s_{t+1} = g(s_t, d_t)，用动态驱动状态更新。通过链式展开 s_{t+1} = h(s_0, o_0, ..., o_t)，证明 latent state 承载所有历史观测信息，解决了传统 last-frame 条件时序感受野有限的问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Owl-1 推理流程：
```
def owl1_generate(I, d_0, N_clips):
    first_frame = ImageDiffusion(I, d_0)           # SD2.1-v生成首帧
    s_0 = LMM.encode_state(I, d_0, 128 queries)    # 128 learnable queries
    o_0 = VideoDM.denoise(s_0, first_frame)        # s_0替代text condition
    for t in 1..N_clips:
        d_t = LMM.predict_dynamics(s_{t-1}, o_{t-1})    # Eq.2: next-token pred
        s_t = LMM.update_state(s_{t-1}, d_t)            # Eq.3: causal attn
        if scene_transition:
            o_t = VideoDM.denoise(s_t, prev_obs=o_{t-1})  # 仅用state条件
        else:
            o_t = VideoDM.denoise(s_t, o_{t-1}.last_frame) # state+last_frame
    return concat([o_0, ..., o_N])
```
具体计算过程：LMM (Chameleon) 以自回归序列 [I, d_0, s_0_queries, VQ(o_0_sampled), d_0_text, ...] 建模世界演化。VideoDM (DynamiCrafter-1024) 以 s_t (128×dim) 作为 cross-attention condition 替代原始 CLIP text embedding，通过 standard diffusion denoising 生成视频。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Chameleon LMM (LoRA rank=8, ~798M params) + DynamiCrafter-1024 (全参 ~1.2B params)，总可训练参数 ~2B。训练：8×NVIDIA A800 (80G)，三阶段（Alignment→Generative Pretraining→World Model Training）共约 7 天。数据：WebVid (400K) + Panda70m (2M) 用于前两阶段；ActivityNet Captions (20K) + Vript (12K) 用于第三阶段。推理：每 4s 一个 clip，可扩展到 24s+。开源：https://github.com/huang-yh/Owl。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## State-Observation-Dynamics Triplet（状态-观测-动态三元组）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
State-Observation-Dynamics Triplet 是 Owl-1 的核心数学形式化，用三个变量 (s_t, o_t, d_t) 的闭环交互来模拟世界的自回归演化。State s_t：隐式世界状态，由 128 个 learnable query embeddings 实现，无 ground truth，通过 causal self-attention 聚合所有历史观测信息。Observation o_t：从状态解码的 4s 视频片段，由 VideoDM 作为 state decoder 生成，以 s_t 为 cross-attention condition、o_{t-1} 为短期平滑参考。Dynamics d_t：文本形式的世界动态预测，从 (s_t, o_t) 推断即将发生的事件。三者构成完整闭环：s_t → o_t (解码) → d_t (预测) → s_{t+1} (更新)。链式展开：s_{t+1} = g(s_t, f(s_t, D(s_t, o_{t-1}))) = h(s_0, o_0, ..., o_t)，即当前状态包含所有历史信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMM 序列输入格式：Seq = [..., s_t_queries (128 tokens), o_t_VQ (2 frames sampled), d_t_text (tokenized), ...]。每个时间步的计算：
```
# LMM forward
s_t_queries, d_t_logits = LMM.forward([...history..., 
    s_{t-1}_queries, o_{t-1}_VQ, d_{t-1}_text])
d_t_tokens = argmax(d_t_logits)                            # 预测动态

# VideoDM denoising
z_T ~ N(0, I)
for m in T..1:
    z_{m-1} = denoise(z_m, m,
                      cross_attn_cond=s_t_queries,         # 长期一致性
                      concat_cond=o_{t-1}.last_frame)      # 短期平滑
o_t = z_0
```
s_t 训练两阶段：Stage 1 用 MSE 对齐 L_align = MSE(s_t, T(t))；Stage 2 用 L_pretrain = ||ε - ε_θ(o_{t,m}, m, s_t, o_{t-1})||²。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
o_t 的 visual tokens 由预训练 VQ-VAE 编码关键帧获得（每 4s clip 采样 2 帧）。d_t 由 LMM text tokenizer 编码，训练时用 dense captions teacher-forcing（交叉熵），推理时 LMM 自主生成。跨场景切换时丢弃 image condition，仅依赖 s_t 信息。s_t 长度 128 是固定超参数，平衡表达能力与计算开销。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## Multi-Stage Training for Video World Models（视频世界模型多阶段训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Stage Training 是 Owl-1 为解决 LMM 与 VideoDM 联合训练困难而设计的三阶段策略。挑战：(1) LMM 和 VideoDM 独立预训练，直接联合训练不稳定；(2) 长视频世界建模需长时长+dense caption 数据，此类数据稀缺不足以从零训练。三阶段从易到难：(1) Alignment: 冻结 VideoDM，MSE 对齐 s_t 与 VideoDM text encoder T(t)，仅训练 LMM；(2) Generative Pretraining: 联合微调，s_t 替代 text condition，用 diffusion denoising loss；(3) World Model Training: 引入 d_t prediction，next-token pred teacher-forcing + denoising，在少量高质量数据上微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: Alignment
L_align = MSE(s_t, T(caption))           # 冻结 VideoDM, 训练 LMM(LoRA)

# Stage 2: Generative Pretraining
o_noisy = sqrt(α_m)*o_t + sqrt(1-α_m)*ε
ε_pred = VideoDM(o_noisy, m, cross_attn(s_t), concat(o_{t-1}))
L_pretrain = ||ε - ε_pred||²            # 联合训练 LMM+VideoDM

# Stage 3: World Model Training
d_pred = LMM.predict_next_tokens(s_t, o_t)
L_dyn = CrossEntropy(d_pred, d_gt)       # teacher-forcing
L_total = L_dyn + L_pretrain             # 少量高质量数据微调
```
关键设计：Stage 1→2 丢弃 MSE loss，仅用 denoising loss，使 s_t 从"模仿 text embedding"解放为"作为 diffusion optimal condition"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练平台：8×NVIDIA A800 (80G)。Stage 1: 1天 (2.4M videos, 10K iters)，Stage 2: 5天 (2.4M videos, 10K iters)，Stage 3: 1天 (20K videos, 1K steps)。该策略将昂贵的长视频 dense caption 数据用量最小化（仅 Stage 3），大量通用短视频数据用于基础能力建设。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## World Dynamics Anticipation（世界动态预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
World Dynamics Anticipation 是 Owl-1 驱动内容多样性和可控性的机制。传统方法仅用 last frame 条件，缺乏未来预判，导致同质化内容。Owl-1 从 s_t（长期信息）和 o_t（短期参考）预测 d_t（文本）：d_t = f(s_t, o_t)。d_t 融入状态更新 s_{t+1} = g(s_t, d_t)，将未来预期编码进下一轮条件。d_t 还提供 controllability——用户可替换预测的 d_t 为自定义信号。在 LMM 中以自回归 next-token prediction 实现，训练时用 dense captions teacher-forcing。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Dynamics prediction (自回归)
d_t = ""
for k in range(max_tokens):
    logits = LMM.forward([...s_t, o_t_VQ, d_t_prev])
    next_token = sample(logits[-1])
    d_t += decode(next_token)
    if next_token == EOS: break

# State update (d_t融入s_t)
s_{t+1}_queries = LMM.causal_attention(
    queries=s_t_queries, keys_values=[text(d_t), o_t_VQ, history])
```
训练 teacher-forcing：L_dyn = -∑_i log P(d_t^(i) | s_t, o_t, d_t^(<i), θ)，d_t 由 dataset dense captions 提供。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练数据：ActivityNet Captions (100K captions, 平均 120s, 3+ events/video) + Vript (400K segments, 密集 script)。d_t 为自然语言（如 "man picks up pruning shears"）。推理：(1) 自主模式：LMM 预测 d_t；(2) 受控模式：用户指定 d_t。场景切换时丢弃 image condition，仅依赖 s_t——对 s_t 信息表达能力要求极高。Limitation：预测的 d_t 有重复性，Dynamic Degree 低于 DynamiCrafter baseline。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## Temporal Autoregressive Paradigm for Long Video Generation（时序自回归长视频生成范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Autoregressive Paradigm 是长视频生成的一类主流方法：将长视频分解为逐段生成短片段（2-4s），每轮以前序输出作为下一轮条件。代表方法：StreamingT2V（last frame + attention injection）、SEINE（transition prediction）、Phenaki（token-based autoregressive）、DynamiCrafter 迭代。优势：可任意扩展长度，复用预训练短视频模型。核心瓶颈：条件设计——大多仅用 last frame，时序感受野仅相邻 clip，导致 long-term inconsistency。Owl-1 的改进：条件从 "pixel-level last frame" 升级为 "latent state s_t（聚合所有历史的隐式表示）+ last frame（短期平滑）"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
传统范式 vs Owl-1 改进：
```
# Traditional: 仅用last frame pixel
def baseline(I, prompt, N):
    clips = []; f = I
    for t in range(N):
        c = VideoDM(f, prompt); clips.append(c); f = c.last_frame()
    return concat(clips)  # f仅含2s pixel信息，远距离漂移

# Owl-1: last_frame + latent state
def owl1(I, d_0, N):
    s = LMM.encode(I, d_0)                 # world state
    o_0 = VideoDM(state=s, image=I)
    for t in range(N):
        d = LMM.predict(s, o_{t-1}); s = LMM.update(s, d)
        o_t = VideoDM(state=s, image=o_{t-1}.last_frame)
    return concat([o_0, ..., o_N])         # s通过causal attn承载全部历史
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Clip 长度固定（如 4s），确保单次 VDM 推理在 GPU 内存内；(2) 条件类型多样：last frame latent (Phenaki)、last frame pixel (StreamingT2V)、attention features (SEINE)、latent state (Owl-1)；(3) Owl-1 的 latent state 创新在于用 LMM 的大感受野 causal attention 构建条件。适用场景：视频扩展、无限时长生成、电影生成、世界模拟。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

## Personalized Streaming Video Understanding (PSVU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSVU（个性化流式视频理解）是 PEARL 论文首次提出并正式定义的新任务。与传统的个性化图像/视频理解不同，PSVU 要求模型：(1) 接收**连续流式视频输入**（而非预录制的完整视频）；(2) 在视频流中的任意时间戳**动态定义个性化概念**（概念在运行时由用户指令创建，非预设词库）；(3) **多轮交互**地回答关于这些已定义概念的实时查询和历史查询。任务定义为：流式视频 V = [X1, X2, ...] 作为连续场景序列，用户在时间戳 tc 通过 Concept-Definition QA 注册新概念，后续在时间戳 tq ≥ tc 发出查询 Q，模型需动态构造上下文 A = M(Csub, Vcontext, Q)，其中 Csub ⊆ C 是查询相关的概念子集，Vcontext 是必要的视觉上下文。PSVU 支持两种概念类型：Frame-level（静态实体，从单帧注册，如特定人物/物体）和 Video-level（动态动作，从连续片段注册，如个性化动作序列/手势）。查询分为三类：Concept-Definition QA（注册新概念，不计入评估）、Real-Time QA（查询概念在当下的状态，需纯粹基于当前场景）、Past-Time QA（查询概念的历史状态，必须检索历史证据片段才能回答）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PSVU 任务的全栈推理流程：

```
# PSVU 任务主循环
ConceptMemory = {}       # {name: (visual_evidence, description)}
StreamingMemory = []     # [(clip_Xi, embedding_ei)]

for each arriving clip X^t at timestamp t:
    # Step 1: 场景检测与流式归档
    if scene_boundary_detected(X^t):
        e^t = multimodal_embed(X^t)      # Qwen3-VL-Embedding-2B
        StreamingMemory.append((X^t, e^t))

    # Step 2: 处理用户指令
    instruction = user_input_at(t)
    
    if instruction.type == "ConceptDefinition":
        if instruction.concept_type == "frame-level":
            evidence = X^t.last_frame
        else:  # video-level
            evidence = X^t
        description = vlm.describe_concept(evidence, instruction.name)
        # frame-level: "a young female with long black hair and oval face"
        # video-level: "the action of squatting down and then leaping forward"
        ConceptMemory[instruction.name] = (evidence, description)

    elif instruction.type in ("RealTimeQA", "PastTimeQA"):
        Q = instruction.question
        # Step 3: 概念检索
        mentioned = extract_concept_names(Q, ConceptMemory.keys())
        C_sub = {name: ConceptMemory[name] for name in mentioned}
        
        # Step 4: 查询重写 + 流式记忆检索
        Q_tilde = rewrite_query(Q, {n: desc for n, (_, desc) in C_sub.items()})
        e_Q = multimodal_embed(Q_tilde)
        similarities = [cosine_sim(e_Q, e_i) 
                        for (_, e_i) in StreamingMemory if clip_time <= t]
        top_K = top_k(similarities, K=4)
        V_context = top_K + adjacent_clips(top_K, N=1)
        
        # Step 5: VLM 生成回答
        answer = vlm.generate(concepts=C_sub, historical_clips=V_context,
                              current_clip=X^t, query=Q)
        return answer
```

具体数据流：视频以 1 FPS 采样 → PySceneDetect 基于 HSV 色彩空间像素变化（阈值 27.0）检测场景边界 → 分段为 min 1s / max 8s clips → 每个 clip 经 Qwen3-VL-Embedding-2B 编码为固定维度嵌入 → Concept-Definition QA 触发 VLM 生成概念描述 → Real-Time/Past-Time QA 触发检索 → 循环选项旋转评估（每个多选题 4 轮旋转，4/4 正确才算通过）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSVU 通过 PEARL 框架实现，代码开源（https://github.com/Yuanhong-Zheng/PEARL, CC-BY 4.0）。核心组件：`clip_memory.py`、`concept_database.py`、`concept_desc.py`、`video_scene_splitter.py`、`video_qa_inference.py`、`eval.py`。多 GPU 部署通过 `server/` 目录启动 VLM server 和 embedding server，`scripts/` 协调并行推理。评估使用 PEARL-Bench（132 个视频、2173 条精细标注）。适用场景：定制化健身教练、个性化 AI 助手、实时监控中的个性化事件检测。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model

## Concept Memory（概念记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Concept Memory 是 PEARL 双粒度记忆系统中专门存储**用户自定义概念**的记忆模块。与 Streaming Memory（存储流式视频 clip 及嵌入）不同，Concept Memory 聚焦于"谁是什么"的个性化知识。当 Concept-Definition QA 触发时，创建包含三个组件的条目：(i) 概念名（如用户定义的 "Adaliz"），(ii) 关联的视觉证据（frame-level 取当前 clip 最后一帧；video-level 取整个 clip），(iii) VLM 生成的紧凑文本描述。frame-level 描述聚焦于永久/稳定特征（性别、面部特征、发型颜色/长度、体型、年龄外观），显式排除临时元素（服装、配饰、表情/姿势、背景）；video-level 描述聚焦于核心运动学（身体运动、动作序列、涉及部位），显式排除执行者身份/外观和背景。生成的描述文本与 clip 嵌入使用相同的 Qwen3-VL-Embedding-2B 特征空间，使查询重写后的检索语义对齐。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Concept Memory 的核心操作：

```
def register_concept(Q_def, X_tc, concept_type):
    concept_name = parse_name(Q_def)
    visual_evidence = X_tc.last_frame if concept_type == "frame-level" else X_tc
    
    # VLM 生成描述（in-context prompting, 无需训练）
    description = vlm.generate(
        prompt=CONCEPT_DESC_TEMPLATE.format(concept_name, Q_def),
        visual_input=visual_evidence
    )
    # Frame-level output: "a young female with long black hair and oval face"
    # Video-level output: "the action of squatting down and then leaping forward"
    
    ConceptMemory[concept_name] = {
        "visual_evidence": visual_evidence,
        "description": description,
        "timestamp": t_c
    }

# 检索：O(1) 按键查找
def retrieve_concepts(Q):
    mentioned = [name for name in ConceptMemory if name in Q]
    return {name: ConceptMemory[name] for name in mentioned}
```

消融证据：添加 Concept Memory 使 Real-Time 准确率从 15.84% 飙升至 51.41%（+35.57%，Table 4），证明显式概念存储对个性化 VLM 理解至关重要——没有概念描述，VLM 无法将用户自定义名称链接到视觉实体。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Concept Memory 通过 VLM 的 in-context prompting 实现，无需参数更新。PEARL 代码库中 `concept_database.py` 管理存储（内存字典），`concept_desc.py` 包含 frame-level 和 video-level 两套 Prompt 模板。Frame-level Prompt 引导 VLM 忽略服装/配饰/表情/背景，聚焦性别/面部特征/发型/体型；Video-level Prompt 引导 VLM 忽略执行者身份/背景，聚焦身体运动/动作序列/涉及部位。描述格式：1 句话约 10-20 词，第三人称，简单描述性英语。适用场景：任何需要在流式视频中动态定义和识别个性化实体/动作的 VLM 应用。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model

## Concept-aware Retrieval Algorithm（概念感知检索算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Concept-aware Retrieval Algorithm 是 PEARL 的检索核心，用于在流式视频场景中精确检索与用户查询相关的个性化概念信息和历史视觉证据。四步流程：(1) **概念检索**——从查询 Q 中提取概念名，按键从 Concept Memory 检索 Csub；(2) **查询重写（Query Rewriting）**——将 Q 中概念名替换为对应文本描述，生成 Q̃（如 "What is Adaliz doing?" → "What is a young female with long black hair and oval face doing?"）；(3) **流式记忆检索**——用 Qwen3-VL-Embedding-2B 编码 Q̃ 为 e^Q，与 Streaming Memory 中所有 clip 嵌入 {ei}i≤tq 计算余弦相似度，选 Top-K；(4) **时序上下文扩展**——对每个选中 clip 扩展其相邻 N 个 clips 以捕获局部时序上下文。最终将 Csub、Vcontext、当前 clip X^tq 和 Q 送入 VLM 生成答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
检索算法的具体计算流程：

```
def concept_aware_retrieval(Q, t_q, ConceptMemory, StreamingMemory, 
                             embedding_model, vlm, K=4, N=1):
    # Phase 1: Concept Retrieval
    mentioned = [n for n in ConceptMemory if n in Q]
    C_sub = {n: ConceptMemory[n] for n in mentioned}
    rules = {n: e["description"] for n, e in C_sub.items()}
    
    # Phase 2: Query Rewriting
    Q_tilde = Q
    for name, desc in rules.items():
        Q_tilde = Q_tilde.replace(name, desc)
    
    # Phase 3: Streaming Memory Retrieval
    e_Q = embedding_model.encode(Q_tilde)          # [d_embed]
    sims = [(i, cosine_sim(e_Q, e_i)) 
            for i, (_, e_i) in enumerate(StreamingMemory) 
            if timestamp(X_i) <= t_q]
    top_K = sorted(sims, key=lambda x: x[1], reverse=True)[:K]
    
    # Phase 4: Temporal Adjacent Expansion
    V_context = set()
    for idx, _ in top_K:
        for offset in range(-N, N+1):
            if 0 <= idx + offset < len(StreamingMemory):
                V_context.add(StreamingMemory[idx + offset].clip)
    
    return C_sub, V_context
```

超参数分析（Fig.4）：K=0 时无法检索历史证据，准确率极低；K≥3 后性能趋于饱和。N=1 相比 N=0 有显著提升（捕获时序上下文），N=2 增量收益有限（噪声抵消）。默认 K=4, N=1。消融实验（Table 4）显示完整 pipeline 相比无 Query Rewriting 版本提升 4.28% Avg。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 PEARL 代码库的 `video_qa_inference.py`。使用 Qwen3-VL-Embedding-2B（MRL 训练，支持 64-2048 维嵌入）作为嵌入模型，余弦相似度通过标准向量内积实现，Query Rewriting 通过 VLM 纯文本 prompt 完成。延迟分解（Fig.5）显示核心检索和重写模块延迟极低且恒定，主要瓶颈仍是 LLM 推理。适用场景：任何需要将个性化概念与流式视频检索结合的多模态应用。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model

## Query Rewriting for Personalized Retrieval（个性化检索的查询重写）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query Rewriting 是 PEARL Concept-aware Retrieval Algorithm 中的关键预处理步骤。核心问题：通用多模态嵌入模型（Qwen3-VL-Embedding-2B）在训练时从未见过用户动态定义的个性化名称（如 "Adaliz"、"Action A"），无法将含个性化名称的查询有效编码为与视频 clip 嵌入语义对齐的向量。解决方案：在编码查询前，将查询中所有个性化概念名替换为 Concept Memory 中存储的文本描述。例如："What was Adaliz wearing when she was cooking?" → "What was a young female with long black hair and oval face wearing when she was cooking?"。重写后的查询包含嵌入模型可理解的视觉语义特征（性别、发型、面部特征），从而能与 Streaming Memory 中编码了相似视觉内容的 clip 嵌入进行有效余弦相似度匹配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Query Rewriting 的计算流程：

```
def rewrite_query(original_query, replacement_rules):
    # replacement_rules = {
    #   "Adaliz": "a young female with long black hair and oval face",
    #   "ActionA": "the action of squatting down and then leaping forward"
    # }
    prompt = f"""Rewrite the following question by replacing the 
concept names with their visual descriptions.
Keep the sentence grammatically correct and natural.

Original question: {original_query}
Replacement rules:
"""
    for name, desc in replacement_rules.items():
        prompt += f'- "{name}" → "{desc}"\n'
    prompt += "\nOutput ONLY the rewritten question, nothing else."
    
    rewritten = vlm.generate(prompt)  # 纯文本推理，低延迟
    return rewritten.strip()
```

关键设计：(a) 使用 Concept Memory 中预先存储的描述，无需额外 VLM 推理理解概念语义；(b) 替换保持语法正确性（冠词调整等）；(c) 不改变查询原始语义意图。消融实验（Table 4）证实 Query Rewriting 将 Avg 准确率从 47.96% 提升至 52.24%（+4.28%），对 Real-Time 和 Past-Time 均有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过 VLM 的纯文本推理实现（仅 token 级别的 prompt + replacement_rules → rewritten_query），延迟极低。PEARL 代码库中集成在 `video_qa_inference.py`。重写模板包含 `{query}` 和 `{replacement_instructions}` 两个占位符，输出仅为重写后的问题。适用场景：任何需要将用户自定义名称/标识符映射为嵌入模型可理解的语义描述的检索场景，特别是概念动态定义、嵌入模型训练数据中不包含这些个性化标识符时。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model

## DPSelect (Dist Peak Select / 距离峰值关键帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DPSelect (Dist Peak Select) 是 RETAKE 提出的 training-free 关键帧选择方法，用于在 VideoLLM 推理前减少视频帧间的时间冗余（temporal redundancy）。核心思想模仿人类视觉系统通过峰值刺激感知运动的机制：计算相邻帧的 token 平均余弦距离（token-averaged cosine distance），用 max pooling 识别距离的局部极大值帧作为 pivot frames（关键结构帧），再按距离值 top-k 补充剩余关键帧，最终将视频帧序列压缩到 alpha_dp 比例。与传统的均匀采样（uniform sampling）或简单 top-N 距离选择不同，DPSelect 通过峰值检测保留了视频中"变化最大的瞬间"（如场景切换、动作突发），这些帧被后续 PivotKV 模块标记为不可压缩的 pivot。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPSelect 在视觉编码器输出后、LLM 输入前执行：
```
# 输入: M (T帧, 每帧N个visual tokens, d维)
# alpha_dp: 压缩比

# Step 1: 计算帧间 token 平均余弦距离
for i in range(T-1):
    d[i] = (1/N) * sum_{j=1..N} (1 - cos(M[i,j], M[i+1,j]))

# Step 2: Max pooling 识别 pivot frames (局部峰值, window=3)
P = {i | d[i] 是 [i-1, i, i+1] 窗口内的最大值}

# Step 3: Top-k 补充关键帧至目标压缩比
k = alpha_dp * T - |P|
K = P ∪ TopK({d[i] for i not in P}, k=k)

# Step 4: 提取压缩特征并标记 pivot mask
M_hat = Flatten(M[K, :, :])
S[j] = 1 if token j 源自 P 中的帧 else 0
```
RETAKE 中 w=3（max pooling 窗口），alpha_dp 按视频自适应设置以控制总 context length <= 32K。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPSelect 完全基于 PyTorch 实现，无需训练或额外模型参数。输入为视觉编码器（如 QWen2VL 的 ViT）输出的 frame-level features，输出为压缩后的 token 序列和 binary pivot mask。与 PivotKV 解耦设计，DPSelect 可独立使用作为 keyframe selector（实验表明 DPSelect 本身在 256 帧限制下已优于 M2SM、MA-LLM 等 baseline）。代码开源在 https://github.com/SCZwangxiao/video-ReTaKe。DPSelect 的超参数 alpha_dp 与 PivotKV 的 alpha_kv 通过 trade-off 分析联合调优：固定总压缩比 0.25 时，alpha_dp/alpha_kv 在 2~3 之间取得最优性能（即更依赖知识冗余进行压缩）。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding

## PivotKV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PivotKV 是 RETAKE 提出的 training-free KV cache 压缩方法，在 VideoLLM 的 chunked prefilling 过程中，对每个视频 chunk 的 KV cache 进行 token 级剪枝。核心创新在于 pivot-guided 压缩策略：(1) DPSelect 选出的 pivot frames 的 visual tokens 被强制保留（通过在 token 重要性分数上加无穷大），保证关键低层时空细节不丢失；(2) 非 pivot frames 中，基于 LLM 层内的 self-attention 权重分布计算 token 重要性分数，低注意力 token 被剪枝——注意力分布由 LLM 的多模态高层知识隐含地确定 token 冗余性（knowledge redundancy）。因此 PivotKV 同时保留了 pivot frames 的全部信息（低层时序不变性）并通过 LLM 注意力去除非 pivot frames 的冗余 token（高层语义冗余）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PivotKV 在每个 chunk prefilling 后执行（每个 chunk 包含 tau 帧），逐层独立操作：
```
# 输入: chunk_i 的 KV cache K_i, V_i
#       pivot mask s (当前 chunk 的 pivot token 标记)
#       alpha_kv: PivotKV 压缩比

for each attention layer:
    # Step 1: 计算 chunk 内的 self-attention 权重
    A = Softmax(Q_i K_i^T / sqrt(d_h))

    # Step 2: 计算 token 重要性分数
    # 对所有 query 位置求和，对所有 head 取均值
    a_bar[j] = sum_{all queries} mean_{all heads} A[:, j]

    # Step 3: 强制保留 pivot tokens
    a_bar = a_bar + s * inf

    # Step 4: Top-k 选择
    I = ArgTopK(a_bar, k=alpha_kv * l_q)
    K_hat_i = K_i[:, I, :]
    V_hat_i = V_i[:, I, :]

    # Step 5: 更新历史 KV cache
    K = Concat(K, K_hat_i)
    V = Concat(V, V_hat_i)
```
文本 chunk（prompt tokens）不参与压缩。效率优化：使用额外 CUDA stream 将第 l 层的 PivotKV 压缩与第 l+1 层的 prefilling 重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PivotKV 基于 PyTorch 实现，以即插即用方式集成到现有 VideoLLM（QWen2VL-7B, LLaVA-Video-7B），无需额外训练。压缩比 alpha_kv 按视频自适应设置以确保总 context length <= 32K。在 A100 GPU 上，alpha_kv=0.5 时：QWen2VL 的 FLOPs 降低 9%、TPOT 降低 19%（优化后 TTFT 仅增加 8%）；LLaVA-Video 的 FLOPs 降低 18%、TPOT 降低 26%（优化后 TTFT 仅增加 11%）。消融实验证明 PivotKV 与 DPSelect 互补——仅用 DPSelect（w/o PivotKV）在低压缩比下性能显著下降，而 PivotKV 通过利用知识冗余缓解了这一问题。代码开源在 https://github.com/SCZwangxiao/video-ReTaKe。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding

## Knowledge Redundancy (in VideoLLMs / 知识冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knowledge Redundancy（知识冗余）在 VideoLLM 推理上下文中指：LLM 的多层自注意力机制内在包含的 token 级冗余信息——许多 visual tokens 即使被丢弃也不会显著影响模型对视频内容的理解，因为 LLM 的高层多模态知识可以推断出被丢弃 token 承载的信息。这与 Temporal Redundancy（时间冗余，相邻帧的像素级视觉重复）形成对比：时间冗余是"低层"的（low-level，基于帧间像素/特征距离），知识冗余是"高层"的（high-level，基于 LLM 学到的语义理解）。RETAKE 的核心洞察是：时间冗余压缩虽然计算开销低但信息损失大（仅基于帧间距离），知识冗余压缩虽然需要额外计算（需要 LLM 前向计算 attention）但信息保持更好，两者联合可取得最优的压缩比-精度 trade-off。知识冗余概念源自 H2O（Heavy-Hitter Oracle）等 LLM token pruning 工作，这些工作发现 attention scores 可以预测 token 重要性——低 attention 的 KV cache token 可被安全丢弃。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RETAKE 中，知识冗余通过 PivotKV 模块在 LLM 内部捕获和利用：
```
# 知识冗余的量化: 通过 attention 权重衡量 token 重要性
# 低 attention token = 高知识冗余 = 可安全压缩

# 在 LLM 第 l 层的 self-attention 中:
A = Softmax(QK^T / sqrt(d_h))

# token j 的重要性 = 所有 query positions 对它的总关注度
importance[j] = sum_{all queries} mean_{all heads} A[:, j]

# 高 importance -> 低知识冗余 -> 保留
# 低 importance -> 高知识冗余 -> 丢弃

# 关键设计: pivot frames 的 token 不论 attention 高低都保留
# 这确保低层时空细节（temporal structure）不丢失
```
trade-off 分析实验：固定总压缩比 0.25，变化 alpha_dp/alpha_kv 比例。alpha_dp/alpha_kv 越高 = 更依赖知识冗余压缩。最优比例在 2~3 之间，表明适度偏好知识冗余策略。但继续增大 alpha_dp/alpha_kv 会增加 FLOPs（因为 DPSelect 压缩少意味着更多 token 进入 LLM 计算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
知识冗余的利用不需要额外训练——RETAKE 直接复用 VideoLLM 已有的注意力权重（在 chunked prefilling 过程中自然计算得到）。实际使用时，用户设置总 compression budget 和 alpha_dp/alpha_kv 比例，DPSelect 先做帧级粗筛（低时间冗余），PivotKV 再做 token 级精剪（低知识冗余）。该方法对 Needle QA（需要精确定位单个关键帧中细微信息）略有精度损失（~1%），但对 Action Order、Key Information Retrieval、Temporal Grounding 等粗粒度任务反有提升，因为去除冗余信息增加了有效信息密度。适用于任何基于 Transformer attention 的 VideoLLM，无需模型修改。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding

## Temporal Redundancy (in Video Understanding / 时间冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Redundancy（时间冗余）在视频理解中指：视频的相邻帧之间由于拍摄帧率（通常 24-30 FPS）远高于场景变化速率，导致大量连续帧承载几乎相同的视觉信息（如静态背景、缓慢变化的动作）。在 VideoLLM 推理中，每帧被编码为数百个 visual tokens，时间冗余导致大量 visual tokens 承载重复信息，浪费 GPU 显存和计算。RETAKE 将其归类为"低层冗余"（low-level redundancy）——可以仅通过帧间视觉特征距离来检测，无需深层语义理解。传统解决方法包括：(a) 稀疏采样（降低 FPS）——简单但丢失关键瞬时信息；(b) 时序 token 合并（temporal token merging, TTM）——对相邻帧的 visual tokens 做池化或合并，但信息损失不可控；(c) keyframe selection——挑选代表性帧。DPSelect 属于 (c) 并引入峰值感知机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RETAKE 中时间冗余通过 DPSelect 在 visual encoder 输出后减少：
```
# 时间冗余存在于: 连续帧的 visual features M[t] 和 M[t+1]
# 量化: token-averaged cosine distance
d[t] = (1/N) * sum_j (1 - cosine(M[t,j], M[t+1,j]))

# d[t] -> 0: 高时间冗余 (几乎相同的帧)，可安全丢弃
# d[t] -> 1: 低时间冗余 (显著变化)，应保留

# DPSelect 的峰值感知策略:
# - 保留 d[t] 的局部最大值帧 (pivot): 场景切换、动作突发
# - top-k 补充高距离帧: 覆盖渐变过程
# - 丢弃 d[t] 低的帧: 静态场景中的冗余帧

# 与知识冗余的互补关系:
# 时间冗余 -> DPSelect (视觉特征距离, 低计算开销, 信息损失大)
# 知识冗余 -> PivotKV (LLM attention, 有计算开销, 信息保持好)
```
传统 keyframe selection（M2SM, A2Summ, MA-LLM）仅使用帧间距离做 top-N 选择，缺少峰值感知，容易在渐变场景中漏选关键帧。DPSelect 的 max pooling 峰值检测确保保留了每个局部变化窗口中最显著的帧。

T3S 从另一角度利用时间冗余：不试图确定性地选择关键帧，而是通过多次随机帧采样（m 个独立的随机子序列），利用概率覆盖替代精确选择。每次推理随机抽取 N 帧并进一步随机子采样 token（保留率 αᵢ），将 m 个短子序列打包到单次前向传播中并行处理，随后通过 logit 聚合获得最终预测。这一设计的核心论点："随机性是无偏的性质保证"——多试次随机采样在统计上覆盖关键时间片段，无需像学习型选择器那样需要先全量处理所有帧。T3S 同时将 self-attention 复杂度从 O(L²) 降为 O(∑αᵢ²L²)，实现了效率与覆盖的双赢。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPSelect 在 RETAKE 开源实现（https://github.com/SCZwangxiao/video-ReTaKe）中通过 PyTorch 实现，核心是 torch.nn.functional.cosine_similarity + max_pool1d。参数：window=3（适合大多数视频帧率），alpha_dp 按视频长度自适应（结合 alpha_kv 使 context length <= 32K）。实验验证：DPSelect 在 256 帧限制下性能优于 M2SM、A2Summ、MA-LLM 等 baseline（VideoMME-Long: 51.0 vs. 49.1-50.7）。使用时无需 GPU 之外的额外硬件；时间冗余压缩是最轻量的一步，发生在 LLM 推理之前。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding

## Recursive Hierarchical Video Grounding / 递归层次化视频定位

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recursive Hierarchical Video Grounding 是一种受人类搜索策略启发的长视频时序定位方法，由 ReVisionLLM (CVPR 2025) 提出。核心思想：给定小时级长视频，模型首先在顶层用压缩的稀疏特征（sparse features）扫描全视频，识别大致的感兴趣区域（如5分钟段）；然后在中间层聚焦预测区域进一步细化；最终在底层用完整时间分辨率的密集特征（dense features）精确定位事件的秒级起止时间。每层 LLM 输出 "From s to e" 或 "Not Present."，上一层的预测边界作为下一层的输入上下文，逐层缩小搜索空间。形式化：对于 L 层层次结构，输入视频特征 I^(ℓ)，第 ℓ 层预测 τ^(ℓ)，条件于先前层次预测 τ^(<ℓ)。该递归结构使模型既能高效扫描小时级视频（使用稀疏特征，段级压缩比可达 250:1），又能精确定位秒级边界（使用密集特征，250帧全部保留）。与传统的 coarse-to-fine 方法（如 CONE 两阶段：候选生成→排序）的区别在于多层级递归和 LLM 内部置信度驱动的排序。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Recursive Hierarchical Video Grounding ===
F = CLIP_ViT_encode_CLS_only(V)  # (T, 768), 仅取CLS token
C = sliding_window(F, L_w=125s, stride=25s)  # MAD: |C| ≈ 100 segments
D = DenseFeatures(C)   # 底层: (|C|, 250, 4096)
S = SparseFeatures(C, Q)  # 上层: (|C|, 768), 250:1压缩

# Hierarchy ℓ=3 (顶层): 粗粒度扫描
τ_3 = LLM([S_top100, "when can we see <event> happening?"])
# → "From 5000s to 5300s" (分钟级精度)

# Hierarchy ℓ=2 (中层): 聚焦τ_3附近约33个段
τ_2 = LLM([S_focused33, "when can we see <event> happening?"])

# Hierarchy ℓ=1 (底层): 精确边界定位
τ_1 = LLM([D_selected, "when can we see <event> happening?"])
# → "From 5123s to 5126.5s" (秒级精度)

# 置信度排序: R_i = 1 / mean(entropy(LLM_output_probs))
τ_final = topk_by_confidence(τ_1_predictions, k=1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理时视频按 sliding window 分段，每段 uniform 采样 250 帧。底层 hierarchy 使用 LoRA_A（Stage 1 训练），上层 hierarchies 使用 LoRA_B（Stage 2 训练）。对 MAD 数据集默认 3 层次，MAD segment L_w=125s/stride=25s，VidChapters-7M segment L_w=500s/stride=100s。消融：0层 R1@.1=0.0, 1层 R1@.1=8.4%, 3层 R1@.1=15.0%。代码: https://github.com/Tanveer81/ReVisionLLM。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Hierarchical Adapter for Vision-Language Models / 层次化适配器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Adapter 是 ReVisionLLM 中连接冻结 CLIP ViT-L/14 和 Vicuna-7B 的适配器模块，负责将视频帧特征转化为稀疏和密集两种时间表示。由三个轻量子模块组成：(1) Cross-Attention (2 layers, 8 heads) — 以视频段特征为 query、文本特征为 key/value 实现跨模态语义对齐；(2) Self-Attention (2 layers, 8 heads) — 将可学习 sparse token 与文本对齐段特征 concatenate 后压缩为单个 768 维向量（段级压缩比 250:1）；(3) Linear Projection — 将 CLIP CLS token (768维) 投影到 LLM embedding space (4096维) 生成密集特征。整体仅 2+2 attention layers vs CLIP 24 layers，几乎无额外计算开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class HierarchicalAdapter:
    cross_attn = MultiheadAttention(d=768, heads=8, layers=2)  # 跨模态对齐
    self_attn = MultiheadAttention(d=768, heads=8, layers=2)   # 稀疏压缩
    ffn = Sequential(Linear(768,3072), GELU, Linear(3072,768))
    linear_proj = Linear(768, 4096)  # CLIP → LLM embedding

    def forward(C_i, Q, S_learnable):
        # C_i: (250, 768), Q: (N_s, 768), S_learnable: (1, 768)
        C_tilde = cross_attn(query=C_i, key=Q, value=Q)  # 文本对齐
        attn_out = self_attn(concat([S_learnable, C_tilde]))  # 压缩
        S_i = ffn(attn_out[0])  # (768,) sparse feature
        D_i = linear_proj(C_i)  # (250, 4096) dense feature
        return S_i, D_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时，Stage 1 先冻结 Linear Projection 微调 LLM LoRA，再冻结 LoRA 微调 Cross-/Self-Attention + FFN (1 epoch, batch=32, LR=1e-3) 学习 sparse 生成。Stage 2 冻结全适配器仅微调新 LoRA。推理时全冻结。预训练 Linear Projection 使用 LCS-558K (LLaVA) 1 epoch 对齐 CLIP 和 LLM 的 embedding space。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Sparse Temporal Features / Dense Temporal Features / 稀疏与密集时间特征

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM Hierarchical Adapter 输出的两种互补视频表示。Sparse Features S^i ∈ R^768：每个视频段（250帧/125s）通过 Cross-Attn + Self-Attn + FFN 压缩为 1 个 token，压缩比 250:1，保留段级语义信息但丢失精确帧级时刻，用于上层 hierarchy 高效扫描小时级视频。Dense Features D^i ∈ R^{250×4096}：每帧独立 Linear Projection (768→4096) 映射到 LLM space，保留全部帧级时间分辨率，仅在底层 hierarchy 已缩小的搜索范围内使用以精确定位秒级边界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Sparse: 250 frame CLS tokens → 1 sparse token
C_tilde = CrossAttn(C_i, Q)      # (250, 768) 文本对齐
S_i = SelfAttn(concat([S_learnable, C_tilde]))[0]  # (768,)

# Dense: 250 frame CLS tokens → 250 LLM embeddings
D_i = Linear_768to4096(C_i)      # (250, 4096)

# 使用场景:
# Hierarchy 3: [S_1..S_100, prompt] → 100 tokens → 粗定位
# Hierarchy 1: [D_selected, prompt]  → 250 tokens → 精确边界
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
稀疏特征训练使用简化目标（Yes/No 存在性判断），密集特征训练使用完整定位目标（From s to e）。两种特征使用不同的 LoRA 模块进入 LLM。默认 ReVisionLLM 仅处理 57% 的视频帧——上层 hierarchy 使用 sparse features 大幅减少 token 数，底层仅对选定段使用 dense features。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Contrastive Segments Training for Video Calibration / 视频校准的对比段训练

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM 提出的改善 VLM 置信度校准的训练策略。传统 VLM 仅用正样本（包含目标事件的视频段）训练，从未被训练判断"事件不存在"，导致 ECE=0.62，在小时级视频中产生大量高置信度假阳性。Solution: Stage 1 从同视频中随机采样不含目标事件的段（负样本），与正样本 1:1 混合训练，目标简化为 "Does <event> happen? Yes/No."，迫使模型学习区分视觉输入中的存在与不存在。ECE 降至 0.46，+Contrastive Segments 使 R1@.1 从 1.4% 提升至 4.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1 adapter微调阶段
pos_seg = segment_containing_event(video, gt)
neg_seg = random_non_overlapping_segment(video, gt)

pos_prompt = [SparseFeatures(pos_seg), "Does <event> happen? Yes/No."]
neg_prompt = [SparseFeatures(neg_seg), "Does <event> happen? Yes/No."]

loss = CE(LLM(pos_prompt), "Yes.") + CE(LLM(neg_prompt), "No.")
# 仅更新 Hierarchical Adapter, LLM LoRA 冻结
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
负样本必须从同视频中采样（确保视觉上下文相似），不与 ground truth 时间边界重叠。与 LLM 校准论文 "LLMs must be taught to know what they don't know" (Kapoor et al. 2024) 理念一致。Stage 1 的校准效果传递到 Stage 2 和推理——良好校准的 sparse features 在上层 hierarchy 减少假阳性，使底层的 dense features 处理更高效。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## LLM Entropy-Based Confidence Calibration / LLM熵基置信度校准

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM 提出的替代 CLIP similarity ranking 的预测排序方法。对底层 hierarchy 的每个候选预测，计算 LLM 自回归生成每个词时输出概率分布的熵，取均值再取倒数作为置信度：R^i = 1 / mean_k(H_k^i)，其中 H_k^(i) = -Σ_w p(w|T_<k, D^(i)) log p(w|T_<k, D^(i))。直觉：LLM 对确信的视觉输入输出集中低熵分布→高置信度；不确时输出分散高熵分布→低置信度。按 R^i 降序排列选 Top-K。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def confidence_score(prediction):
    total_H = 0
    for k, logits_k in enumerate(prediction.logits):  # (vocab_size,) per step
        probs = softmax(logits_k)
        H_k = -sum(probs * log(probs + ε))
        total_H += H_k
    return 1.0 / (total_H / K)  # K=生成词数

ranked = argsort([confidence(p) for p in predictions], descending=True)
top_k = [predictions[i] for i in ranked[:K]]
```
Annotaions: ε 防 log(0)。仅用于底层 hierarchy (dense features)。Table 2: +Calibration (-CONE) 将 R1@.1 从 4.8% → 8.4%，ECE 从 0.6231 → 0.4614。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
有效性源于 Stage 1 contrastive training → LLM 学会对不确定输入输出高熵。与 TimeJudge (logit-space calibration offset) 不同，ReVisionLLM 依赖训练阶段的隐式校准。要求推理框架保留 LLM 原始 logits。代码: https://github.com/Tanveer81/ReVisionLLM。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

## Representation Shift（表示漂移 / Token Importance via Feature Change）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Representation Shift 是一种训练无关（training-free）、模型无关（model-agnostic）的 token 重要性度量方法，由 ICCV 2025 论文 "Representation Shift: Unifying Token Compression with FlashAttention" 提出。核心公式为 s = Δx = ||F(x) - x||₂，其中 F(·) 为某个层的变换函数（优选 MLP），Δx 量化每个 token 经过该层后的表示变化量。直观理解：对任务关键的 token 会被网络"强调"——其表示经过 MLP 后发生较大变化（大 representation shift）；冗余 token 几乎不变（小 shift）。因此可以通过剪除低 representation shift 的 token 来减少计算量。

与 attention-based token importance（如 EViT 使用 s = Softmax(q_cls K^T/√C)）的本质区别：
1. **不依赖 attention map**：可直接与 FlashAttention 配合使用。FlashAttention 为避免 HBM I/O 不构建完整 attention map，attention-based 方法因此失效。
2. **信号更可靠**：MLP 逐 token 独立操作（per-token independent），产生的 representation shift 比全局 attention（cross-token information exchange，transformation 更 diffuse）更具判别性。
3. **模型无关**：可扩展到 CNN（各 stage 后计算 feature map 变化）、SSM（替换激活值基分数）。

关键消融发现：
- 操作选择：MLP > Attention > Entire Block（Figure 5a），因 MLP 逐 token 独立使信号更具判别性
- 距离度量：L2 > L1 > Cosine（Figure 5b），L2 在所有深度上最一致；Cosine 在深层失效
- 可靠性验证：top 50% vs bottom 50% token 准确率差 26.3%（Table 8）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Representation Shift-based Token Pruning
# 超参: drop_layers = [0,1,2], drop_ratio = 0.2

for layer_idx in range(num_layers):
    # Step 1: LayerNorm + Attention (FlashAttention)
    x_norm1 = LayerNorm(x)
    x_attn = FlashAttention(x_norm1)     # 不暴露 attention map
    x = x + x_attn
    
    # Step 2: LayerNorm + MLP
    x_norm2 = LayerNorm(x)
    x_mlp = MLP(x_norm2)                 # [N, C]
    
    if layer_idx in drop_layers:
        # Step 3: 计算 representation shift (L2 distance)
        delta_x = ||x_mlp - x||_2, dim=-1  # [N]
        
        # Step 4: Top-K 保留
        num_keep = int(N * (1 - drop_ratio))
        keep_idx = topk(delta_x, k=num_keep)
        
        # Step 5: 对 token 维度剪枝
        x = x[keep_idx]                   # [N*(1-r), C]
        x_mlp = x_mlp[keep_idx]
    
    # Step 6: 残差连接
    x = x + x_mlp
```

张量计算流程（以 UMT-B, 12 frames × 224² 为例）：
- 输入 tokens x ∈ R^(12×14×14=2352, C)
- Layer 0: FlashAttention → Δ = ||MLP(LN(x')) - x'||₂ → top-80% → x ∈ R^(1881, C)
- Layer 1: 同样流程 → x ∈ R^(1505, C) 
- Layer 2: 同样流程 → x ∈ R^(1204, C)
- Layer 3-11: 1204 tokens 不变，正常 Transformer

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：加载预训练模型后直接应用，无需额外训练（training-free）。在指定层的 MLP 后插入 L2 norm 计算 + token pruning 模块。L2 norm 计算开销为 O(N × C)，可忽略（<1% total FLOPs）。使用 FlashAttention 的 fused kernel 作为标准 self-attention 后端。开源实现：https://github.com/mlvlab/Representation-Shift（MIT License），使用 `main.py --eval --use_flash True --drop_r [...]`。

配置（论文实验）：
- Video (UMT): drop_layers=[0,1,2], drop_ratio=0.2 (retrieval) / 0.1 (QA), FlashAttention enabled
- Image (DeiT): drop_layers=[1,4,7], drop_ratio=0.2
- CNN (ResNet): line-wise/token-wise pruning after stage 1/2
- SSM (ViM): 替换 ToP-ViM 的激活值基分数

累积加速效果：FlashAttention ~2.7× + pruning ~2× → 总计 5.5× (UMT-L video-text retrieval)。

涉及论文标题：
- Representation_Shift__Unifying_Token_Compression_with_FlashAttention

## Any-Horizon Reasoning（任意时长推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Any-Horizon Reasoning 是 SAGE 提出的视频推理范式，指模型能够根据任务难度和视频时长自适应地选择推理策略：对简单/短视频问题采用单轮推理直接输出答案（DIRECT模式），对复杂/长视频问题采用多轮工具调用逐步聚合信息（AGENT模式）。该概念受人类行为启发——人类看短视频会完整观看后回答，看2小时长视频则会迭代式地定位关键信息。核心机制：orchestrator VLM (SAGE-MM) 在 Stage-1 (Context VLM) 输出 video_context + query_intent + 首步action，若当前信息充足则直接输出 final_answer（单轮），否则进入 Stage-2 (Iterative Reasoner) 多步 tool-calling（最多11轮）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Any-Horizon Reasoning 决策流程
def any_horizon_reason(SAGE_MM, F, M, T, Q):
    # Stage-1: Context VLM — 首步判断
    action_1 = SAGE_MM(prompt=[T, F, Q, M])
    if action_1.final_answer is not None:
        return action_1.final_answer  # 单轮推理（短/简）
    
    # Stage-2: Iterative Reasoner — 多轮工具调用（长/难）
    history = [action_1]
    C = action_1.video_context
    for step in range(2, N_max+1):  # N_max=11
        S_j = {T, Q, M, C, history}
        action_j = SAGE_MM(prompt=S_j)
        if action_j.final_answer is not None:
            return action_j.final_answer
        tool_result = execute(action_j.recommended_tool)
        history.append(tool_result)
```

训练中通过 GRPO 多轮 reward 塑造 any-horizon 行为：单轮正确回答 +1.0，多轮+工具正确回答 +1.25（额外奖励鼓励视觉工具使用），多轮错误回答 -0.5（惩罚不必要的 tool overcalling）。RL 前100步使用 N_max=6 稳定训练，之后扩大至11步。SFT 是必须的——直接 RL 导致 collapse to single-turn（base model 的训练目标强烈偏向直接产出答案）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Any-Horizon 的核心在于训练数据设计：RL 数据中一半样本需要 tool calls，一半是 single-turn，使模型学会判别两者。表 9 展示了 any-horizon 效果：SFT 模型从 expert Gemini-2.5-Flash 蒸馏获得强 single-turn 能力（79.0% accuracy）但 overcall tools（1038 multi-turn vs expert's 885），RL 后 single-turn 升至 948 样本（+242）、multi-turn 降至 796（-242），分布更接近 expert，且 multi-turn accuracy 从 53.7% 升至 54.3%（+0.6%）。推理时模型可自主决策推理步数——表 15 显示随着视频时长增加（0-60s → 2400+s），平均 turns 从 1.74 渐变至 2.77。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

## Multi-Reward RL Post-Training for Video Agents

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Reward RL Post-Training 是 SAGE 为训练 any-horizon video agent 提出的 GRPO 多奖励设计方案。单一 accuracy reward（如 Video-R1 的 ROUGE/option-matching）对 open-ended 问题无效，而 naive 实现常导致工具调用过度或崩溃。SAGE 的解决方案：R_i = Σ step_rewards + accuracy_reward，uniformly 赋给 trajectory 中所有 actions。Step rewards 分解为四个可加组件：(1) s_format: +0.05 若 JSON 仅含必需字段，否则 -0.10；(2) s_reasonable-tool: GPT-4o 判断当前 tool call 合理→+0.10，否则 -0.10；(3) s_args-repeat: -0.05·√num_repetitions 惩罚重复参数；(4) s_args-valid: -0.10 惩罚无效参数。Accuracy reward a_N 由 GPT-4o binary judge 决定：正确+visual tools→+1.25, 正确无tools→+1.0, 错误→-0.5, JSON无效→-2.0。Step reward 总值被设计为与 accuracy reward 可比（10步累积 ≈ 1.25）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_sage_reward(trajectory, Q, ground_truth):
    step_rewards = []
    for j, step in enumerate(trajectory):
        s_format = +0.05 if valid_json(step.action) else -0.10
        s_reasonable = +0.10 if GPT4o_judge_tool(step, Q, trajectory[:j]) else -0.10
        s_repeat = -0.05 * sqrt(count_repetitions(trajectory, step.action))
        s_valid = -0.10 if invalid_args(step.action) else 0
        step_rewards.append(s_format + s_reasonable + s_repeat + s_valid)
    
    # 仅在 trajectory 结束时计算 accuracy reward
    final_action = trajectory[-1]
    if invalid_json(final_action):
        a_N = -2.0
    else:
        verdict = GPT4o_judge(final_action.answer, ground_truth)  # True/False
        if verdict:
            used_visual = any(tool in {'extract-video-parts','ground-event'} 
                            for step in trajectory)
            a_N = +1.25 if used_visual else +1.0
        else:
            a_N = -0.5 if len(trajectory) >= 1 else -2.0
    
    R_i = sum(step_rewards) + a_N  # uniform for all actions
    return R_i
```

关键设计决策：(1) 正确回答 + visual tools 额外 +0.25 bonus（鼓励视觉信息利用）；(2) 错误回答 + tool calls 惩罚 -0.5（补偿正向 step rewards，防止 tool overcalling）；(3) JSON 无效直接 -2.0（强制格式合规）。前100步 N_max=6 避免长 trajectory 的方差过大导致训练不稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 multi-reward 设计的关键工程选择：(a) Step-level rewards 累积值需与 accuracy reward 量级匹配（10步 maximal step rewards ≈ 1.25，等于 accuracy reward 上限）；(b) GPT-4o 用于 reasonable-tool 和 accuracy 评判（需 carefully designed prompt 协议）；(c) RL 数据构造：7680 样本，half tool-calls half single-turn，确保 any-horizon 学习信号平衡。与 LongVT 的 Joint Reward（R_acc + R_fmt + R_time）相比，SAGE 的奖励设计更细粒度（4种 step rewards vs 2-3种），且加入了 tool overcalling 惩罚和 visual tool bonus。SAGE 的奖励仅需 QA accuracy judge（无需 temporal IoU），简化了 reward computation pipeline。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

## Synthetic Video QnA Generation with Long-Context VLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Synthetic Video QnA Generation 是 SAGE 提出的利用长上下文 VLM（Gemini-2.5-Flash）一次性处理完整长视频（up to 2 hours）直接生成覆盖全时间跨度的 QnA pairs 的数据合成 pipeline。与传统 bottom-up pipeline（将视频切割为 10-30s subclip，分别处理后再聚合）不同，SAGE 利用 Gemini-2.5-Flash 的长上下文能力（支持数小时视频）在 single pass 中生成 10-20 个 QnA pairs。关键技术是 prompt 中的 **percent_video_parsed** 字段——要求模型为每个 QnA pair 计算已处理视频百分比，并强制至少一个问题的 percent_video_parsed ≥ 90%，确保问题覆盖视频全部时间范围。该方法成本约为人工标注的 1/100（约 $0.30/video vs $30/video），速度约为 subclip pipeline 的 10×（约 2 min/video vs 20 min/video）。人工验证 1700+ 样本仅 5% 错误率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Synthetic QnA Generation Pipeline
# Stage 1: QnA Pair Generation (single-pass长上下文处理)
def generate_qna_pairs(video_path):
    video_frames = sample(video_path, fps=1)  # 1 FPS全视频
    prompt = build_qna_prompt(video_duration)
    # prompt 要求: (1) 10-20 QnAs; (2) 混合 visual/verbal/both modality;
    #   (3) 混合 open-ended/MCQ; (4) 混合 easy/medium/hard;
    #   (5) 每个QnA含percent_video_parsed字段;
    #   (6) 至少1个QnA的percent ≥ 90%
    
    response = Gemini_2_5_Flash(video_frames + prompt + audio)
    # response 为JSON: [{index, type, difficulty, modality, 
    #   answer, question, options, start_timestamp, end_timestamp,
    #   percent_video_parsed, ...}, ...]
    
    qna_pairs = parse_json(response)  # 10-20 QnA pairs
    return qna_pairs

# Stage 2: Tool Call Trajectory Generation
def generate_tool_trajectories(qna_pairs, video_path):
    # 使用 SAGE 系统（Gemini-2.5-Flash 作为 SAGE-MM）
    # 为每个 QnA 生成 4 条 tool call trajectories
    trajectories = []
    for q in qna_pairs:
        for _ in range(4):
            traj = SAGE_system(video_path, q.question, orchestrator=Gemini)
            trajectories.append(traj)
    # 从 input-action pairs 提取 unique trajectories 构建 SFT 数据
    return trajectories
```

关键数据特征：(1) 99.1k 训练问题来自 6659 个视频；(2) 417.7k state-action pairs 用于 SFT；(3) 13 个 YouTube 频道覆盖体育、美食、喜剧、教育、旅游等 genre；(4) RL 数据 7.68k 样本（half tool-calls, half single-turn）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要求：(a) 支持超长上下文（≥2小时视频）的 VLM（如 Gemini-2.5-Flash）；(b) Carefully designed prompt 强制 percent_video_parsed 字段确保全时覆盖，否则模型倾向于仅覆盖视频开头；(c) 人工验证子集确认质量（5% 错误率已足够低）；(d) QnA pairs 可直接用于 SFT（DIRECT training），而 tool call trajectories 用于 SFT（AGENT training）。该 pipeline 可泛化到其他视频领域（论文仅用于娱乐视频，但在体育、教育视频上同样有效）。局限性：依赖闭源 Gemini-2.5-Flash 的 API 访问；开源替代品（如 Qwen3-VL）目前尚无足够的 long-context video understanding 能力来替代该 role。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

## Saliency-Coverage Oriented Token Pruning / SCOPE（显著性-覆盖度联合 Token 剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SCOPE 是一种免训练的视觉 token 剪枝算法，用于加速多模态大语言模型（MLLM）推理。其核心思想是在保留的 visual token 子集中，同时最大化**显著性（saliency）**和**语义覆盖度（coverage）**，以在 token 预算大幅缩减时仍保持语义完整性。SCOPE 将 token 选择建模为一个迭代贪心过程：每轮计算每个候选 token v 的 marginal coverage gain Δ(v; S)（v 加入当前已选集 S 后带来的额外覆盖度），然后乘以视觉 attention score A_v^α 作为显著性加权，得到 SCOPE score = Δ(v; S) · A_v^α。每轮选择 SCOPE score 最高的 token 加入 S，更新 coverage 状态，迭代 K 次得到最终 token 子集。

SCOPE 解决的关键问题：saliency-only 方法（如 FastV, VisionZip）仅按 attention 排序选 Top-K token，导致：(1) 语义完整性缺失——高 attention 的 token 集中在少数图像区域（如前景物体），背景和上下文信息被丢弃；(2) attention 分布偏斜——尾部分布的 token attention 值几乎均匀（flat tail），无法区分 informative vs redundant tokens。SCOPE 通过引入 coverage metric 和联合优化解决此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCOPE 在 MLLM pipeline 中的位置：Vision Encoder → SCOPE Token Selection → LLM。完整流程：

```
输入: 图像 I, 文本 T
1. V = CLIP_ViT(I)  → V ∈ R^{N×d}  (N=576 for LLaVA-1.5, N=2880 for LLaVA-Next)
2. A_v = Attention_CLS_to_v(V, layer=-2)  → A_v ∈ R^N  (saliency scores)
3. S_uv = cosine_sim(v_u, v_v)  → S ∈ R^{N×N}  (pairwise similarity matrix)
4. S = ∅, c_u = 0 ∀u ∈ V  (初始化: 空选集, coverage scores=0)
5. for t = 1 to K:  (K = 目标 token 数, 如 64/128/192)
     for each v ∈ V \ S:
       Δ(v; S) = Σ_{u∈V} max(S_uv, c_u) - c_u   (marginal coverage gain)
       score(v) = Δ(v; S) · A_v^α                 (SCOPE score, α=1.0)
     v* = argmax score(v)
     S = S ∪ {v*}
     c_u = max(c_u, S_{u,v*})  ∀u ∈ V            (更新 coverage)
6. LLM_input = Concat(S, Text_Tokens)
7. Output = LLM(LLM_input)  → autoregressive generation
```

复杂度：相似度矩阵 O(N²) 存储（576²≈332K 对），每轮选择 O(N²) 扫描 × K 轮 = O(K·N²)。论文报告在 4×A100 上，2880→160 tokens 时延迟从 601.9s 降至 188.8s（3.2× speedup），同时 POPE 保持 81.3%（vs full 86.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SCOPE 的实现方式：
- 集成位置：在 vision encoder 之后、LLM projector 之前插入 token selection 模块
- 显著性来源：使用 vision encoder 倒数第二层（layer -2）的 CLS token 到 visual token 的 attention scores
- 相似度：cosine similarity，预计算全量 N×N 矩阵
- 缩放因子 α：默认 1.0，通过消融实验确定最优值
- 框架：基于 lmms-evals 评估框架实现，支持 HuggingFace Transformers
- 开源代码：https://github.com/kinredon/SCOPE
- 评估：支持 LLaVA-1.5 (7B/13B), LLaVA-Next (7B/13B), Video-LLaVA, Qwen2-VL
- 与 FlashAttention 兼容（不依赖中间 attention map，仅依赖 encoder 输出 token embeddings），剪枝后 token 数减少使得后续 LLM attention 计算按 O(K²/N²) 缩放

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

## θ-Coverage / Semantic Coverage in Visual Token Selection（θ-覆盖度 / 视觉 Token 语义覆盖度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
θ-Coverage 是 SCOPE 论文提出的度量指标，用于量化选定的 visual token 子集对全量 token 集合的语义覆盖程度。定义：对于 full token set V 和 selected subset V'，一个 token v ∈ V 被 V' "覆盖"，当且仅当存在至少一个 v' ∈ V' 使 cosine similarity sim(v, v') ≥ θ（θ 为相似度阈值）。θ-coverage 即为被覆盖 token 占全量 token 的比例：

$$\operatorname{Coverage}_{\theta}(\mathcal{V}',\mathcal{V}) = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbb{I}\left(\max_{v' \in \mathcal{V}'} \operatorname{sim}(v, v') \ge \theta\right)$$

高 θ 值要求更严格的相似度标准 → 通常导致较低的 coverage 但确保保留的 token 更具语义代表性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
θ-Coverage 在 SCOPE 中作为**分析工具**而非优化目标使用。论文用它诊断 saliency-only 方法的缺陷：通过测量不同 θ 下的 coverage 曲线，发现 saliency-only 方法的 coverage 低于 random baseline，证明其语义信息丢失严重。

计算流程（分析用，非在线推理）：
```
输入: V (N tokens), V' (K selected tokens), θ
covered = 0
for each u in V:
    max_sim = max_{v in V'} cosine_sim(u, v)
    if max_sim >= θ: covered += 1
return covered / N
```

在 MME benchmark 上，当 K=64（从 576 中选）时，saliency-only 在不同 θ 下的 coverage 显著低于 SCOPE。这个分析间接验证了 coverage-aware selection 的重要性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
θ-Coverage 主要作为离线分析工具：
- 用于评估剪枝后 token 子集的语义完整性
- 支持跨方法比较（saliency-only vs coverage-only vs SCOPE vs random）
- 帮助选择合理的 token 保留数量 K
- 论文未将 θ-coverage 直接用作训练或优化目标（SCOPE 使用 soft set-coverage function f(S) = Σ_{u∈V} max_{s∈S} sim(u,s) 作为优化目标，而非硬阈值版本）

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

## Token-Coverage Gain / Marginal Gain for Submodular Coverage（Token覆盖增益 / 子模覆盖的边际增益）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-Coverage Gain（也称 Marginal Gain）是 SCOPE 的核心选择机制，量化将候选 token v 加入当前已选集 S 后带来的额外覆盖度。其理论基础是**子模函数（submodular function）**的边际增益性质。SCOPE 定义 set-coverage 函数：

$$f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \text{sim}(u, s)$$

该函数是 monotone submodular 的（满足 diminishing returns 性质）。Marginal gain 定义为：

$$\Delta(v; \mathcal{S}) = f(\mathcal{S} \cup \{v\}) - f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max(C(u, \mathcal{S}), \sin(u, v)) - C(u, \mathcal{S})$$

其中 C(u, S) = max_{s∈S} sim(u, s) 是 token u 在 S 下的当前最佳 coverage。这个边际增益刻画：v 的加入能对多少还未被良好覆盖的 token 提供更好的相似度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
边际增益在 SCOPE 贪心选择中的计算：

```
# 每轮迭代, ∀ 候选 token v ∉ S:
Δ = 0
for each u in V:  # 遍历所有 N 个 token
    current_best = c_u              # u 在 S 下的当前最佳相似度
    new_best = max(current_best, S_uv)  # 加入 v 后的新最佳相似度
    Δ += (new_best - current_best)  # 累加增益
# Δ 即为 v 的 marginal coverage gain
```

例子：假设 V = {猫头, 猫身, 背景, 香蕉}，S = {猫头}（c_u 已初始化）。候选 v = 背景 patch：猫头已有 sim=1.0，无新增益；香蕉 sim=0.2→0.3，增益 +0.1；背景 sim=0.1→1.0，增益 +0.9。Δ(背景) = 0.9。候选 v = 猫身：猫头 sim=1.0，无新增益；香蕉 sim=0.2，无新增益；背景 sim=0.2（不变），无增益。Δ(猫身) ≈ 0。因此背景会被优先选择，确保 coverage 扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Marginal gain 的实现要点：
- 需要维护 coverage score 数组 c_u (N 维)，每次选出新 token 后更新
- 计算 Δ 需要遍历 N 个 token，每轮需要 O(N²) 次比较（N 候选 × N 全量）
- 总和 K 轮，总复杂度 O(K·N²)
- 贪心选择的 (1-1/e) 近似保证：对于 monotone submodular 函数，贪心选择可以达到最优解的 (1-1/e) ≈ 63% 近似
- SCOPE 将 marginal gain 乘以 attention saliency 得到 SCOPE score，打破纯 coverage 贪心，使选择同时考虑显著性和覆盖度
- 理论参考：Iyer et al. "Submodular combinatorial information measures with applications in machine learning" (ALT 2021)

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

## Saliency-Skewed Attention Distribution in Visual Token Pruning（视觉Token剪枝中的显著性偏斜注意力分布）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Saliency-skewed attention distribution 是 SCOPE 论文揭示的 saliency-based token pruning 方法面临的关键问题：在 CLIP vision encoder 中，CLS token 对各 visual token 的 attention 分布高度偏斜——少数 token（如前景物体区域）获得极高的 attention 值，而绝大多数 token（如背景区域）的 attention 值几乎均匀地平坦分布（flat tail）。这种偏斜导致两个后果：(1) Top-K 选择几乎全部集中在前景区域，丢失背景上下文（semantic incompleteness）；(2) flat tail 区域的 token 之间 attention 差异极小，无法有效区分 informative vs redundant tokens（token indiscriminability）。

论文图 1(b) 展示了 MME benchmark 上前 128 个 token 的平均 attention 分布，显示 attention weights 迅速平坦化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
该观察在 SCOPE 中作为**动机分析**驱动算法设计，而非算法组件。其影响体现在：

1. 解释了为什么 saliency-only Top-K 方法在低 token budget 下性能急剧下降：flat tail 中即使存在 informative tokens（如"cat 旁边的地毯"可能对回答 "Where is the cat?" 有用），其 attention 与纯冗余 background token 几乎相同，Top-K 排序无法区分
2. 论证了引入 coverage metric 的必要性：coverage 不关心 attention 绝对值，而是基于 token 嵌入的语义相似度，能有效区分 semantic content
3. 支持 SCOPE score = Δ(v)·A_v^α 中 α 的设计：α=1.0 在保留高 attention token 的同时允许部分低 attention 但具有高 coverage 增益的 token 被选入

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文未提出专门"解决" attention skewness 的方法，而是通过 SCOPE 的覆盖度机制绕过了该问题。实践中，attention 偏斜程度的测量方法：
- 计算 attention 分布的 Gini 系数或熵值
- 绘制 attention 排序后的累积分布曲线（论文图 1(b)）
- 对比不同模型层级的 attention 偏斜程度（layer -2 已在 SCOPE 中被选为 saliency 来源）
- 偏斜度随 model scale 增大可能加剧（更大模型倾向于产生更集中的 attention）

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

## Submodular Set Coverage Maximization for Token Selection（子模集合覆盖最大化用于Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
子模集合覆盖最大化（Submodular Set Coverage Maximization）是 SCOPE 方法的理论基础。一个集合函数 f 是子模的（submodular），如果满足 diminishing returns 性质：向小集合添加元素带来的边际增益 ≥ 向大集合添加相同元素带来的边际增益。SCOPE 定义的 set-coverage 函数 f(S) = Σ_{u∈V} max_{s∈S} sim(u, s) 是 monotone submodular 函数，这意味着贪心选择策略可以达到 (1-1/e) ≈ 63% 的最优近似保证。

子模覆盖最大化在 ML 中的应用广泛，包括：数据摘要（document summarization）、主动学习（active learning）、特征选择（feature selection）、以及 SCOPE 中的 visual token selection。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCOPE 将 token selection 建模为 cardinality-constrained monotone submodular maximization：

$$\max_{\mathcal{S} \subseteq \mathcal{V}, |\mathcal{S}| = K} f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \operatorname{sim}(u, s)$$

标准贪心算法：每次选择边际增益 Δ(v; S) 最大的元素，重复 K 次。理论保证：
- f(∅) = 0（空集覆盖度为0）
- f 单调：S ⊆ T ⇒ f(S) ≤ f(T)（更多 token 不会降低覆盖度）
- f 子模：Δ(v; S) ≥ Δ(v; T) for S ⊆ T（diminishing returns）
- 贪心解 f(S_greedy) ≥ (1-1/e) · f(S_opt)（近似保证）

SCOPE 的独特之处：将标准 submodular maximization 的边际增益与 attention saliency 相乘，得到 SCOPE score，打破了纯子模优化的近似保证，但在实践中取得了更好的 saliency-coverage trade-off。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 贪心选择是最常用的 submodular maximization 近似算法，时间复杂度 O(K·N²)
- 更高效的实现可使用 lazy greedy（利用 diminishing returns 减少评估次数）
- SCOPE 未使用 lazy greedy（因为引入了 saliency 加权打破了单调子模性）
- 在每个 MLLM query 中在线运行（training-free）
- 子模性保证来自于 cosine similarity 的 max 聚合的数学性质，而非特定于视觉 domain
- 参考论文：Iyer et al., "Submodular combinatorial information measures with applications in machine learning", ALT 2021

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

## Visual Memory Mechanism for Long Video Understanding（长视频理解的视觉记忆机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Memory Mechanism（视觉记忆机制）是 FlexMem（CVPR 2026）提出的一种将 MLLM 的长视频理解建模为"人类观看视频"过程的训练无关方法。核心思想：MLLM 不应一次性处理所有视频帧（会导致输入上限和计算爆炸），而应像人类一样——持续观看视频内容、将关键信息压缩为视觉记忆（visual memories）、在问答时召回最相关的记忆片段来生成答案。FlexMem 的视觉记忆机制包含三个核心子模块：(1) **记忆编码（Memory Encoding）**——通过 Dual-Pathway Compression 将每个视频 clip 的视觉 KV cache 压缩为 context memory（用于跨 clip 信息传递）和 local memory（用于最终召回），local memory 写入 Visual Memory Bank；(2) **记忆存储（Memory Storage）**——Visual Memory Bank 持久存储所有 clip 的压缩 visual KV cache，每 clip 内存固定，总内存随 clip 数线性增长；(3) **记忆召回（Memory Recall）**——问答时从 M_bank 中根据 clip-问题相关性选择最相关的 na 个连续 clip 的 memory，仅将这些片段输入 MLLM 解码答案。该机制理论上支持无限长视频处理，结合了 RAG 方法（精确定位关键片段）和视觉压缩方法（保持全局理解）的优势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlexMem 的 Visual Memory Mechanism 完整 pipeline（以 LLaVA-Video 7B 为 backbone）：
```
# === Visual Memory Mechanism Pipeline (FlexMem) ===
# 输入: 长视频 V={I1,...,I_T}, 问题 Tq, MLLM backbone

# Step 1: 视频分片
clips = uniform_split(V, clip_size=8)  # N clips
M_bank = []  # Visual Memory Bank

# Step 2: 首次编码
KV_1 = MLLM.forward(clips[0], Tq_opt)   # 可选传入Tq
C_1 = context_compress(KV_1, alpha_c)    # context memory (传递)
M_1 = local_compress(KV_1, alpha_s)      # local memory (存储)
M_bank.append(M_1)

# Step 3: 迭代编码
for i in 2..N:
  ctx = [C_{i-ns}, ..., C_{i-1}]        # 前序context memory
  long_term = optional_recall(M_bank)    # 可选长期记忆
  KV_i = MLLM.forward(long_term + ctx + clips[i], Tq_opt)
  C_i = context_compress(KV_i, alpha_c)
  M_i = local_compress(KV_i, alpha_s)
  M_bank.append(M_i)

# Step 4: Memory Recall
g = [sum_attention(Tq → Vi, layers 3..L) for Vi]  # relevance scores
recalled = M_bank[topK_continuous(g, na)]           # 选na个连续clip

# Step 5: 解码
Y = MLLM.decode(recalled, Tq)  # 仅用召回片段
```
核心特点：(a) encoding 阶段逐 clip 处理，每步固定计算量；(b) 解码 token 数固定（13k / 7k），不随视频长度增长；(c) Tq 在 encoding 阶段可选——传入时可用 encoding-based reading，不传入时用 MemIndex。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexMem 以 Python 实现，基于 HuggingFace Transformers，作为 LLaVA-Video 和 LLaVA-OneVision 的 plug-and-play 模块。核心超参数：每 clip 8 帧、压缩比 α_c/α_s、context memory 窗口 ns、总采样帧 512 或 1024。在单 RTX 3090 24GB 上可处理 1024+ 帧（vs baseline 64 帧），24GB 受限下仅损失 0.5% 性能。代码开源：https://github.com/city1517/FlexMem。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## Dual-Pathway Compression / DPC（双路径压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Pathway Compression (DPC) 是 FlexMem 提出的视觉 KV cache 压缩方法，核心洞察：MLLM 的 prefill 阶段（需要历史上下文聚合）和 decoding 阶段（需要显著性保留）对"哪些 visual token 重要"有不同标准。DPC 将压缩解耦为两条路径：(a) **Context Compression**——使用 context aggregation score s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l 选择最能"聚合历史信息并传播给后续"的 token，产生 Context Memory Ci（用于迭代信息传递）；(b) **Local Compression**——使用 local saliency score ŝ_j^l = Σ_{k∈Vi} a_{kj}^l 选择 clip 内最具"显著性"的 token，产生 Local Memory Mi（写入 Memory Bank 供最终召回）。DPC 与现有 KV cache 压缩方法（如 AdaRETAKE、Video-XL）的核心区别在于区分 prefill 和 decoding 的不同目标，而非对所有 token 使用统一的重要性度量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPC 在每层 l 的具体计算（共享同一 forward pass 的 attention matrix）：
```
# === Dual-Pathway Compression (per layer l) ===
A = Attention([Q_{Vi}, Q_{Tq}], [K_C, K_{Vi}, K_{Tq}])  # cross-clip attention

# Pathway 1: Context Memory (服务于prefill的信息传递)
for token j in Vi:
  s_j = sum(A[j, :C]) + sum(A_self[h, j] for h in Vi where h>j)
  # 第1项: 从历史context聚合的信息
  # 第2项: 对后续token的因果传播
c_i^l = {K[j], V[j] for j in topK(s, alpha_c * |Vi|)}

# Pathway 2: Local Memory (服务于decoding的显著性保留)
for token j in Vi:
  ŝ_j = sum(A_self[:, j])  # clip内部影响力
m_i^l = {K[j], V[j] for j in topK(ŝ, alpha_s * |Vi|)}

Ci = [c_i^1..c_i^L];  Mi = [m_i^1..m_i^L]
```
消融实验验证：(1) Context Compression Only 在长视频上丢失局部显著信息；(2) Local Compression Only 在需要跨 clip 理解的场景下不足；(3) Dual-Pathway 在所有时长上均最优，性能增益随视频长度增加而扩大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPC 在 MLLM 的 forward pass 中自然计算——attention matrix 是 prefill 的副产物，topK 选择是纯排序操作，计算开销可忽略。compression ratio α_c 和 α_s 控制两种记忆的压缩程度，需要在"信息保留"和"内存节省"之间权衡。在 LLaVA-Video 7B 上，每 clip 8 帧经 DPC 压缩后，总解码 token 数仅 13k（vs AdaRETAKE 的 40k），同时实现了 8× baseline 的帧覆盖。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## Memory Recall via Cross-Modal Attention（基于跨模态注意力的记忆召回）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Recall via Cross-Modal Attention 是 FlexMem 中从 Visual Memory Bank 检索最相关记忆的机制。原理：在 memory encoding 阶段，如果 Tq（问题文本）被包含在 MLLM 输入中，则 MLLM 的 self-attention 会自然产生 Tq→Vi 的 cross-modal attention weights。这些权重反映了模型在理解问题时关注了哪些视觉区域，因此天然可度量 clip-问题的相关性。FlexMem 对这些 attention weights 求和作为 relevance score g_i = Σ_{l=3→L} Σ_{j∈Tq} Σ_{k∈Vi} a_{jk}^l，仅取深层（≥第3层），因为浅层 attention 分布均匀无区分力。最后选择 g_i 最高的 na 个连续 clip 的 memory 作为召回结果。该方法零额外计算（复用已有 forward pass 的 attention），但代价是必须在 encoding 阶段就传入 Tq。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Encoding-based Memory Recall ===
for i, Mi in enumerate(M_bank):
  g_i = 0
  for layer in 3..L:
    A = saved_attention_matrix[layer]  # 来自encoding的forward pass
    g_i += sum(A[j, k] for j in Tq_positions for k in Vi_positions)

start = argmax(sum(g[start:start+na]))  # 找na个连续最高分clip
Y = MLLM.decode(M_bank[start:start+na], Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该方法完全在 MLLM 的 forward pass 中实现——attention weights 是 self-attention 的中间结果，仅需在深层保存 Tq→Vi 的注意力子矩阵。论文实验（Table 1）均使用 encoding-based reading，在五个 long VideoQA benchmark 上取得 SOTA。消融（Table 5 Block 3）表明 memory recall 远优于 indiscriminate loading of all memory，验证了选择性召回的價值。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## MemIndex (Fast Memory Indexing)（快速记忆索引）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MemIndex 是 FlexMem 提出的快速记忆索引方法，解决 encoding-based reading 的"每换一个问题需重新 encoding 全视频"的限制。MemIndex 将 memory reading 与 memory encoding 解耦：(1) encoding 阶段不传入 Tq，仅做视觉 KV cache 压缩，同时以更高压缩比生成 compact visual index tensor（k×d 维，k=5 个 token，远小于原始 |Vi|×d）；(2) reading 阶段，通过轻量 statistical fitting 来近似 encoding-based reading 的 relevance score。具体使用 linear regression 学习函数 σ(r̂_i) = Σ α^l·r̂_i^l 来拟合 ground-truth g_i，并基于 learned α^l 权重选择 top-K 个最重要的 cache 层（K=3）。问题编码为 Q_{Tq}[-1]（最后一个 token 的 query embedding），视觉索引为 per-layer top-k salient key vectors。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === MemIndex: Training ===
data = []
for video, Tq in training_set:
  g_i = encoding_based_reading(M_bank, Tq)  # teacher signal
  for each layer l in 3..L:
    q = MLLM.encode(Tq)[-1]  # 问题: 最后token的Q
    K* = topK_salient_keys(Mi, k=5)  # 视觉: top-k显著keys
    r̂_i^l = dot_product_attention(q, K*)
  data.append(([r̂_i^3..r̂_i^L], g_i))

alpha = LinearRegression().fit(data)  # 学习层权重
H = topK_indices(alpha, K=3)  # 选最重要K层

# === MemIndex: Inference ===
q = MLLM.encode(Tq)[-1]
for Mi in M_bank:
  r̂_i = sum(alpha[l] * attention(q, Mi.memindex[l]) for l in H)
  relevance[i] = sigma(r̂_i)
Y = MLLM.decode(M_bank[topK(relevance, na)], Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MemIndex 适用于：(a) Streaming QA——监控/直播场景，同一视频多次提问无需重新 encoding；(b) 多问题场景——一次性 encoding 全视频后，每个问题仅需轻量匹配（单次 attention + 加权求和）。在 OVOBench streaming QA 上，FlexMem + MemIndex 达 54.4% 平均性能，显著超过 Flash-VStream (27.4%) 和 VideoLLM-online (36.1%)。劣势：需要额外训练数据（以 encoding-based reading 为教师信号），且 statistical fitting 存在一定精度损失。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## Context Memory and Local Memory in Visual KV Cache（视觉KV Cache中的上下文记忆与局部记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context Memory (C) 和 Local Memory (M) 是 FlexMem Dual-Pathway Compression 产生的两种压缩视觉 KV cache。**(a) Context Memory Ci**：度量 token 的"信息桥梁"能力——s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l，选取最能聚合历史信息并传播给后续的 token。Ci 在迭代编码中传递给后续 clip，实现跨 clip 的时序连续性。**(b) Local Memory Mi**：度量 token 的"显著性"——ŝ_j^l = Σ_{k∈Vi} a_{kj}^l，选取 clip 内被广泛关注（即"显著"）的 token。Mi 存入 M_bank 供最终召回。两者的关系：(i) 互补——C 保证时间线的信息连续性，M 保证每个时刻的视觉显著性；(ii) 消融实验（Table 5 Block 2）验证 C+M 的组合显著优于单独使用任一；(iii) 生命周期不同——C 在迭代中流式传递（生产→消费→丢弃），M 持久存储（生产→存储→召回→解码）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Context vs Local Memory 对比计算
A_cross = softmax(Q_{Vi} @ K_C^T / sqrt(d))     # [|Vi|, |C|]
A_self  = softmax(Q_{Vi} @ K_{Vi}^T / sqrt(d))  # [|Vi|, |Vi|]

# Context: 信息桥梁 (第j个token的context aggregation能力)
s_j = row_sum(A_cross[j,:]) + col_sum_upper(A_self[:,j])
#     聚合历史 ↑               传播给未来 ↑

# Local: 视觉显著性 (第j个token在clip内的受关注度)
ŝ_j = col_sum(A_self[:,j])
#     所有token对j的attention总和 ↑
```
Context 和 Local 共享相同的 attention matrix，计算无额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两种记忆的关键设计选择：Context Compression 保留能"讲好故事"的 token（连接前后文的桥梁），Local Compression 保留能"提供证据"的 token（回答问题时需要的视觉线索）。这一设计可推广到其他需要渐进式信息压缩和理解的长序列任务。压缩比 α_c 和 α_s 可在信息保留和内存效率之间权衡。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## Visual Memory Bank（视觉记忆银行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Memory Bank (M_bank) 是 FlexMem 中持久存储所有已处理 video clip 的压缩 local memory 的数据结构。每处理完一个 clip，其 local memory Mi 被追加到 M_bank。特点：(a) 固定每 clip 内存——由压缩比 α_s 决定，不随 clip 数增长；(b) 线性总内存——M_bank 大小 = N × |Mi|，随 clip 数 N 线性增长（vs 原始 KV cache 的 O(N²)；在 512 帧/64 clips 时内存约 134MB，远小于原始 KV cache）；(c) 结构化存储——按 clip 索引组织，支持按 relevance score 随机召回任意连续 clip 组；(d) 可选长期记忆召回——在迭代编码中可从 M_bank 预先调取长期记忆 `<Ml>` 作为当前处理的附加上下文。M_bank 使 FlexMem 同时具备 RAG 的精确定位（从记忆库检索）和压缩方法的全面理解（所有 clip 参与信息流）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# M_bank: 存储与召回
M_bank = []
for clip_idx in 1..N:
  M_i = local_compress(KV_i, alpha_s)  # DPC的local pathway
  M_bank.append(M_i)

# 召回 (encoding阶段可选，decoding阶段必须)
# encoding中: 召回长期记忆辅助当前处理
long_term = recall(M_bank, Tq, top_k=1)  # optional

# decoding前: 召回na个最相关连续clip
g = [relevance(M_i, Tq) for M_i in M_bank]
recalled = select_consecutive(M_bank, g, na)
Y = MLLM.decode(recalled, Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 GPU 显存受限时，M_bank 可存储在 CPU memory 中，仅在召回时将选中片段加载到 GPU。这一设计类似于 RAG 的 offline index 但更轻量——不需要额外的 embedding model，所有 visual representations 来自 MLLM 自身的 KV cache。M_bank 的检索粒度是 clip 级（连续 na 个 clip），保证了召回的时序连续性。

StreamingEval 中的 Visual Memory Bank 变体：StreamingEval 采用固定容量 FIFO 内存银行作为评估离线 VideoLLM 的统一适配器。每帧视觉编码 $z_i = g_{\theta}(v_i)$ 经投影层对齐 LLM embedding 空间后写入 memory bank，超出字节预算 M 时按 FIFO 淘汰最旧内容。字节预算公式：$\operatorname{Mem}_i(B) = B \cdot d_i \cdot s_{\text{emb}} + B \cdot 2L_i \cdot h_i^{\text{kv}} \cdot s_{\text{kv}}$，其中 $d_i$ 为投影后 visual token embedding 维度，$L_i$ 为 LLM 层数，$h_i^{\text{kv}}$ 为 per-layer KV channel width。内存银行写入规则：$M_{\tau_i^+} = \mathcal{U}(M_{\tau_i^-}, z_i; B, \pi)$，B 为容量约束，π 为淘汰策略（离线模型 FIFO，在线模型用原生策略）。与 FlexMem 的 recall-based M_bank 不同，StreamingEval 的 FIFO memory bank 是线性顺序存储的无压缩 buffer，用于公平比较而非内容检索。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

## ECRD (Evidence-Constrained Reweighting Decoding，证据约束重加权解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ECRD 是一种 training-free、plug-and-play 的解码框架，在 LVLM 推理时监督每一步 token 选择，确保生成的 token 有对应的视觉证据支持。其核心思想是：不依赖 RL 训练让模型"学会何时看图"，而是在测试时用视觉证据监督每一步。ECRD 由两个组件构成：(a) Distribution Supervisor（分布监督器）——维护一个文本证据池，计算证据诱导的 token 分布 r_i(w)，并与 base 模型的分布 p_i(w) 通过自适应权重 α_i = p_{(1)}（base 模型 top-1 概率）协商混合；(b) Visual Decider（视觉裁决器）——当混合分布 margin 不足且候选集包含多个 token 时触发，读图并生成微观察证据句，强制提交正确 token 并扩充证据池。ECRD 的命名体现了其三步流程：Evidence（积累证据）→ Constrain（约束候选）→ Reweight（重分配概率）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ECRD 解码循环（每步）
for step i in decoding:
    # 1. Base LVLM 输出 next-token 分布
    p_i = softmax(LVLM(prefix))  # [vocab_size]

    # 2. Knee Truncation: 动态选择候选集大小
    p_sorted = sort(p_i, descending=True)
    k* = argmax_k(p_sorted[k] - p_sorted[k+1])
    C_i = top_k(p_i, k*)  # 候选 token 集

    # 3. 证据评分: 对每条证据计算 mean-over-prefix 概率
    for each E_j in evidence_pool:
        q_Ej(w) = mean_{t=1..L} p_VLM(w | e_{<t})  # 式(5)
    S_i(w) = -log(mean_{j} q_Ej(w))  # 式(6)

    # 4. 证据诱导分布（仅在 C_i 内归一化）
    r_i(w) = softmax_{w in C_i}(-S_i(w))

    # 5. Mass-matching: 让 r_i 在 C_i 内的总 mass 等于 p_i
    r_tilde_i(w) = r_i(w) * sum_{C_i} p_i / sum_{C_i} r_i

    # 6. 协商混合: α_i = top-1 概率控制证据权重
    alpha = max(p_i)
    p_mix = alpha * p_i + (1-alpha) * r_tilde_i

    # 7. 不确定性检测
    margin = max(p_mix) - second_max(p_mix)
    if k* > 1 and margin <= delta:  # delta=0.08
        # 触发 Visual Decider
        w*, evidence = GRIT(image, prefix_tail, C_i)
        commit(w*)
        evidence_pool.append(evidence)
    else:
        commit(argmax(p_mix))
```

典型性能：Qwen2.5-VL-7B + ECRD 在 TreeBench 上 37.0%→47.9%（+10.9 点），超过 GPT-4o 和 Gemini-2.5-Flash；在 RH-Bench 上 RH-AUC 从 0.51→0.58。跨 LLaVA-OneVision、Qwen2.5-VL、InternVL3 三个 backbone 系列和多种 scale 一致有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ECRD 作为 decoding wrapper 包裹 frozen LVLM，不修改任何模型权重。Visual Decider（GRIT-3B）单独部署在另一 backend（FP16 on CPU），仅在触发时调用。证据评分 O(k*|E_i|) 在 CPU 上计算（k* 为个位数，|E_i| 增长缓慢）。每问题平均 decider 调用次数 r(δ) 在 δ=0.08 时处于低个位数，总延迟 T ≈ t_0 + l_0·r（l_0≈1.1-1.5s/call），overhead 控制在 20-30%。开源：github.com/uuuuZYC/See-It-Say-It-Sorted。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## VDGD (Visual Description Grounded Decoding，视觉描述接地解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VDGD 是 ECRD 的前身方法，由 Ghosh 等人于 2024 年提出（arXiv:2405.15683）。VDGD 是一种 training-free 的 LVLM 解码策略，核心思想是：先让模型生成图像的全局文本描述 d=(d1,...,dL)，然后在自回归解码的每一步，对候选 token w 计算其与描述 prefix d_{<j} 之间的 KL 散度（即 -log p_VLM(w | d_{<j})），取所有 prefix 长度 j 上的最小值作为该 token 的"接地得分"，用此得分替换 base logits 后做 softmax。VDGD 的关键洞察是：在描述图像时模型能正确"看到"的视觉细节，在推理时可能被语言先验压制——通过强制解码分布与描述一致，可恢复视觉接地。VDGD 的局限在于：(a) 使用静态单一描述，缺乏自适应能力；(b) min-over-prefix 聚合不稳定，容易受个别 prefix 波动影响；(c) 直接替换 logits 丢弃了 base 模型的校准置信度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VDGD 解码流程
d = LVLM.generate_description(image)  # 全局描述 (d1,...,dL)
k* = knee_truncation(p_i)             # 候选集大小
C_i = top_k(p_i, k*)                  # 候选 token 集

for w in C_i:
    # min-over-prefix KL 散度
    score(w) = min_{j=1..L} KL(onehot(w) || p_VLM(.|d_{<j}))
            = min_{j=1..L} -log p_VLM(w | d_{<j})

# 用 scores 替换原始 logits
p_VDGD = softmax_{w in C_i}(score(w))
x_i = argmax(p_VDGD)
```

ECRD 对 VDGD 的改进：(a) min→mean：将 min-over-prefix 替换为 mean-over-prefix，更稳定且奖励持续支持；(b) 单证据→多证据：支持证据池中多条证据的平均支持度；(c) 替换→混合：从直接替换 logits 改为与 base 分布通过自适应权重协商混合，保留 base 模型在自信步的行为。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VDGD 作为 training-free decoding wrapper，包裹任意 frozen LVLM，无需 fine-tuning。论文报告跨多个 benchmark 和 LVLM 一致提升 2%-33%。VDGD 在 ECRD 的 ablation 中作为 baseline 对比：Qwen2.5-VL-7B + VDGD 在 TreeBench 上 37.0%→39.5%（+2.5），远低于 ECRD 的 +10.9。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Visual Hallucination Propagation（视觉幻觉传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Hallucination Propagation 是多模态 CoT 推理中的一种级联失败现象：LVLM 在长链推理的某个中间步骤生成一个与图像视觉证据不一致的 token（幻觉），后续所有推理步骤——即使逻辑形式正确——都基于这个错误的中间结论，最终导致错误答案。这是"thinking more"与"seeing less"的矛盾表现：随着推理链增长，文本上下文逐渐主导 attention，视觉 token 被稀释（attention 分析表明长链中视觉 token 的注意力权重显著下降），语言先验覆盖了细粒度视觉线索，模型在关键视觉判别步产生幻觉。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
幻觉传播示例（以 TreeBench 问题为例）：
```
Step i:  候选 {"blue","red"}，模型自信度中等
         base 分布 p("red")=0.52, p("blue")=0.48
         Greedy 选 "red"（错误！实际上是 blue dress）
Step i+1: 基于 "red" 定位红色衣物，描述 red garment
Step i+2: 基于错误定位判断颜色 → 答案错误
```

关键链路：
```
单步幻觉 token → 后续 token 条件于错误 prefix → 
attention 和 logits 全部基于错误前提 → 
级联放大 → 最终答案错误
```
RH-Bench 论文（Liu et al., 2025）将此量化为 RH-AUC 指标：随推理链长度 T 增加，Perception 准确率 H_T 下降，与 Reasoning 准确率 R_T 形成 trade-off。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
应对策略分为两类：(1) RL-based 训练（DeepEyes、Pixel-Reasoner）——训练模型学会在推理中调用 visual tools（zoom/crop），但需要策划数据、设计 reward、消耗大量计算、且与特定 backbone 耦合；(2) Training-free 解码干预（ECRD、VDGD）——在推理时注入视觉证据监督 token 选择，不修改模型权重，可跨 backbone 泛化。ECRD 通过 uncertainty 检测（k*>1 且 margin≤δ）在关键步触发 visual decider 注入微观察，将级联失败打断在第一步。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Distribution Supervisor（分布监督器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Supervisor 是 ECRD 框架的第一个核心组件，负责在解码的每一步基于文本证据池对 base LVLM 的 token 分布进行重新评估和混合。其输入为：(a) base 模型当前的 next-token 分布 p_i；(b) 证据池 E_i 中的 N 条文本证据句。输出为协商混合后的分布 p_i^{mix}。核心功能：(1) 对每条证据计算 mean-over-prefix 概率 q_E(w)（替代 VDGD 的 min-over-prefix KL）；(2) 跨多条证据取平均支持度 S_i(w)；(3) 仅对 knee-selected 候选集 C_i 内归一化得到证据诱导分布 r_i(w)；(4) mass-matching：将 r_i 在 C_i 内的总 mass 缩放至与 p_i 匹配；(5) 通过自适应权重 α_i = p_{(1)} 混合 base 和 evidence 分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Distribution Supervisor 的核心计算
Input: base_dist p_i, evidence_pool E_i, candidate_set C_i

# Step 1: 证据评分
for each evidence sentence E_j in E_i:  # E_j length L
    q_Ej(w) = (1/L) * sum_{t=1..L} p_VLM(w | e_{<t})
    # 每条证据对所有候选 token 的 mean-over-prefix 条件概率
S_i(w) = -log( (1/N) * sum_{j} q_Ej(w) )
# 所有证据对 token w 的平均支持度（取负 log 转换为得分）

# Step 2: 证据诱导分布
r_i(w) = softmax_{w in C_i}(-S_i(w))  # 仅候选集内归一化

# Step 3: Mass-matching
mass_p = sum_{w in C_i} p_i(w)
mass_r = sum_{w in C_i} r_i(w)
r_tilde_i(w) = r_i(w) * (mass_p / mass_r)

# Step 4: 自适应混合
alpha_i = max(p_i)  # base 模型 top-1 概率
p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)  # w in C_i
p_mix_i(w) = alpha_i * p_i(w)                              # w not in C_i
```

关键设计：α_i = p_{(1)} 使 supervisor 在 base 模型自信时（p_{(1)} 大）保持其主导，在 base 模型犹豫时（p_{(1)} 小，即分布平坦、更易产生幻觉）给证据更大权重。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Supervisor 作为纯计算模块，运行在 CPU 上（precomputed log-likelihoods 存储为 FP16），计算复杂度 O(k*|E_i|)，k* 为个位数、|E_i| 增长缓慢，GPU 压力可忽略。Supervisor 是"始终在线"的防御层——即便不触发 visual decider，分布监督器也在每步进行证据约束重加权，提供稳定的幻觉抑制。Ablation 中 supervisor alone（无 visual decider）已在 TreeBench 上带来 +3.7 点提升（37.0%→40.7%）。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Visual Decider（视觉裁决器，ECRD 语境）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Decider 是 ECRD 框架的第二个核心组件，在 Distribution Supervisor 检测到不确定性时（k*>1 且 margin≤δ）被触发。它是一个轻量级的视觉定位模型（论文中用 GRIT-3B，基于 Qwen2.5-VL-3B），接收图像、当前文本前缀尾部、和候选 token 集 C_i，输出：(a) 一个确定的 token w*∈C_i（模型认为正确的选择）；(b) 一句人类可读的微观察证据句 E_i（包含可选的坐标标注用于可解释性，但不参与 scoring）。ECRD 强制提交 w* 并将 E_i 追加到证据池。关键设计特点：(a) decider 仅接收当前步骤的文本前缀尾部而非完整问题——因为其目标是解决当前步的潜在幻觉，而非回答整个问题；(b) 证据仅以文本形式参与后续 scoring（Eq. 5-7），坐标仅用于可解释性；(c) 稀疏触发——仅在 margin≤δ 时调用，确保计算开销按需发生。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Visual Decider 触发与执行
margin = max(p_mix) - second_max(p_mix)  # 协商分布 margin
if k* > 1 and margin <= delta:           # delta = 0.08
    # Trigger: 候选集多 token + 协商后仍不自信
    w*, evidence_sentence = GRIT.forward(
        image,            # 原始图像
        prefix_tail,      # 当前解码前缀的尾部
        C_i               # 候选 token 集 {"5", "3"}
    )
    # GRIT 内部: 视觉编码 → 定位相关区域 → 生成证据 + 选择 token
    commit(w*)                           # 强制采用 decider 选择
    evidence_pool.append(evidence_sentence)  # 追加文本证据

# 示例输出:
# w* = "3"
# evidence_sentence = "The number behind the cardboard box 
#     with the 'favorita' brand and banana illustration is '300'."
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Visual Decider 使用 GRIT-3B（基于 Qwen2.5-VL-3B + GRPO-GR 视觉定位优化）实例化，部署在独立 backend（FP16 on CPU），与 base LVLM 解耦。每次调用延迟 l_0≈1.12-1.46s（H20 GPU）。在 δ=0.08 时，每问题平均调用次数 r 在低个位数。统计表明：decider 直接输出最终答案的案例占总增益的 11.4%，decider 在中链注入视觉接地的案例占 18.2%，其余增益来自 supervisor 的重新加权和证据池的间接稳定性提升。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Evidence Pool（文本证据池，Textual Evidence Pool）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Evidence Pool 是 ECRD 中维护的一个文本证据集合 E_i = {E_1, ..., E_N}，在解码过程中动态增长。每个证据句 E_j 是一句自然语言描述，由 Visual Decider 在不确定性步生成，描述图像中与当前歧义相关的微观察（micro-observation）。证据池的关键设计特点：(a) 仅存文本——坐标存储在 GRIT 输出中用于可解释性但不参与 scoring，使得后续步骤可直接引用文本证据而无需重新编码图像；(b) 按需增长——初始化为全局图像描述 d_global（提供大范围覆盖），之后仅在 margin≤δ 时追加新证据；(c) 跨步复用——supervisor 在每一步都对证据池中所有证据计算支持度（Eq. 6），早期注入的微观察可在后续步骤中为相关 token 提供概率支撑。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Evidence Pool 生命周期
E_0 = {d_global}  # 初始化：全局图像描述

for step i in decoding:
    # 使用当前证据池评分所有候选 token
    for E_j in E_i:
        q_Ej(w) = mean_{t} p_VLM(w | E_j_prefix[0:t])
    S_i(w) = -log(mean_j q_Ej(w))
    
    # 需要时扩展证据池
    if k* > 1 and margin <= delta:
        w*, new_evidence = VisualDecider(image, prefix, C_i)
        E_{i+1} = E_i ∪ {new_evidence}
    else:
        E_{i+1} = E_i
```

证据示例：`"The first dress from the right-hand side is blue, partially hidden by the tree."`

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
证据池存储在 CPU memory 中（FP16），每步评分 O(k*|E_i|) 可忽略。文本证据的语义性质使其在 token 空间与 decoder 天然兼容——证据句子中的词直接映射到与候选 token 相同的 embedding 空间，无需额外的跨模态对齐。这与 RL-based 方法中反复编码图像裁剪的 pixel-space reasoning 形成对比：文本证据 compact、可组合、可跨步复用。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Knee Truncation（膝点截断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knee Truncation 是一种动态选择 top-k 候选 token 数量 k* 的方法。与固定 k 的 top-k sampling 不同，knee truncation 通过检测排序概率分布中的"膝点"（elbow/knee point）——即相邻排序概率之间差值最大的位置——来自适应地确定 k*：k* = argmax_k (p_{(k)} - p_{(k+1)})。其中 p_{(1)} ≥ p_{(2)} ≥ ... 为降序排列的 token 概率。直觉上，膝点之前是概率密集区（多个竞争 token），膝点之后是概率长尾区（token 概率均为极小值）。在幻觉步中，分布往往更平坦、k* 更大；在非幻觉步中，分布尖锐、k*≈1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Knee Truncation 计算
p_sorted = sort(p_i, descending=True)
diffs[k] = p_sorted[k] - p_sorted[k+1]
k* = argmax_k(diffs)
C_i = top_k(p_i, k*)

# 数值示例:
# 幻觉步: p = [0.30, 0.28, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
#   diffs = [0.02, 0.13, 0.03, 0.04, 0.04, 0.02, 0.01], k*=2
# 自信步: p = [0.95, 0.02, 0.01, 0.007, ...]
#   diffs = [0.93, 0.01, 0.003, ...], k*=1
```

k* 的双重作用：(a) 确定候选集大小——只有 C_i 内的 token 参与协商混合；(b) 参与不确定性检测——k*>1 是触发 visual decider 的必要条件之一（另一条件是 margin≤δ）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
膝点检测是一种广泛使用的自适应阈值选择方法。在 LLM 解码中，knee truncation 相比固定 k 的优势在于：(a) 不需要手动调参——k* 根据每步的概率分布形状自动确定；(b) 在自信步时 k*≈1 保持 greedy 行为；(c) 在不确定步时 k* 较大，给证据更多操作空间。论文实证（Ghosh et al., [5]）支持：幻觉步倾向于有更大的 k* 和更小的 variance。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## Negotiated Reweighting（协商重加权）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Negotiated Reweighting 是 ECRD 中将 base 模型分布 p_i 与证据诱导分布 r_i 融合的核心机制。其"协商"体现在：(a) 两个分布来源不同——base 来自 LVLM 的完整上下文推理，evidence 来自视觉证据池的纯视觉接地——在 token 选择上可能存在分歧；(b) 自适应权重 α_i = p_{(1)} 决定哪一方更有话语权——base 自信时 base 主导，base 犹豫时 evidence 主导；(c) mass-matching 确保 r_i 在候选集 C_i 内的总概率质量与 p_i 一致，仅重分配 C_i 内部的相对概率而非改变 C_i 的总 mass。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Negotiated Reweighting 具体计算
# 输入: p_i (base分布), r_i (证据诱导分布), C_i (候选集)

# Mass-matching
mass_p = sum_{w in C_i} p_i(w)     # e.g., 0.981
mass_r = sum_{w in C_i} r_i(w)     # e.g., 1.0
r_tilde_i(w) = r_i(w) * mass_p / mass_r

# 自适应协商
alpha_i = max(p_i)  # e.g., 0.498 (base 不自信)
p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)

# 数值示例:
# C_i = {"5", "3"}
# p_i: "5":0.498, "3":0.483; r_tilde_i: "5":0.503, "3":0.478
# alpha = 0.498
# p_mix("5") = 0.498*0.498 + 0.502*0.503 = 0.501
# p_mix("3") = 0.498*0.483 + 0.502*0.478 = 0.480
# margin = 0.021 ≤ 0.08 → 触发 decider
```

α_i 的设计体现了"最小干预原则"：仅在必要时介入，常规步骤保持模型自身的行为。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Negotiated Reweighting 是纯数学计算，无需额外模型调用。mass-matching 步骤（Eq. 8）确保了合理性——不改变 C_i 的总概率质量，只在内部重新分配。与 VDGD 的直接 logit 替换的本质区别：VDGD 完全丢弃 base 分布的信息，而 ECRD 保留 base 模型的校准置信度作为混合权重。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## GRIT (Grounded Reasoning with Images and Texts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRIT 是一种通过轻量级 RL（GRPO-GR）训练 MLLM 在推理链中交错生成自然语言和边界框坐标的方法，由 UC Santa Cruz 提出（NeurIPS 2025, arXiv:2505.15879）。GRIT 的核心创新：(a) 训练模型在 `<think>` `</think>` 标签内进行带有边界框坐标 `[x1,y1,x2,y2]` 的视觉推理；(b) 仅需 20 个训练样本即可实现，数据效率极高；(c) 通过三种 reward（格式奖励、计数奖励、答案准确率奖励）引导 GRPO 优化。GRIT 基于 Qwen2.5-VL-3B 和 InternVL3-2B 构建，在 VSR（空间推理）、TallyQA（计数）、GQA（组合推理）上超越 Direct Query 和 CoT baselines。在 ECRD 中，GRIT-3B 被实例化为 Visual Decider——利用其"看图生成接地文本"的能力，在解码歧义步输出 token 选择 + 含坐标的微观察证据句。代码：github.com/eric-ai-lab/GRIT。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GRIT 在 ECRD 中的使用（作为 Visual Decider）：
```
# GRIT 推理接口
w*, evidence_sentence = GRIT.forward(
    image=original_image,
    context=prefix_tail,          # 当前解码 prefix 尾部
    candidates=C_i                # 候选 token 集
)

# GRIT 内部: 视觉编码 → 定位相关区域 → 
#   <think> grounding reasoning </think>
#   <answer> token_choice + evidence_sentence </answer>
# 解析输出: 提取 w* 和 evidence_sentence
```

GRIT 的训练（GRPO-GR）使用三种 reward：
```
Reward = Format_Reward + Counting_Reward + Answer_Accuracy_Reward
# Format: 检查标签和有效坐标语法
# Counting: 生成的 bbox 数量匹配预期
# Answer: GPT-4o 评判 + BLEU 相似度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GRIT 开源在 HuggingFace：GRIT-20-Qwen2.5-VL-3B、GRIT-20-InternVL-2B。作为 Visual Decider 使用时，GRIT 接收图像、当前推理前缀、候选 token 集，输出确定的 token 选择和自然语言证据句。论文中 GRIT alone 的 TreeBench 准确率仅 30.1%（低于 Qwen2.5-VL-7B 的 37.0%），但在 ECRD 框架中作为 decider 时，通过精确触发和证据累积，将 base 7B 提升至 47.9%。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## RH-AUC (Reasoning-Hallucination Area Under Curve)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RH-AUC 是一种评估多模态推理模型的综合指标，由 Liu 等人于 2025 年在 NeurIPS 论文 "More Thinking, Less Seeing?" 中提出（arXiv:2505.21523）。RH-AUC 量化模型在不同推理长度 T 下，Reasoning 准确率 R_T 与 Perception/Hallucination 准确率 H_T 之间的权衡。计算方式：对不同 T 采样多组 (R_T, H_T)，将两者 min-max 归一化到 [0,1]，通过梯形法则计算 H 相对于 R 的曲线下面积：RH-AUC = Σ_{i=0}^{n-2} (R_{T(i+1)} - R_{T(i)})/2 · (H_{T(i+1)} + H_{T(i)})。更高的 RH-AUC 表示模型在提升推理深度的同时更好地保持视觉接地。传统指标（固定长度的准确率、幻觉率）无法捕捉推理深度与感知可靠性之间的动态 trade-off。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# RH-AUC 计算
T_values = [short, medium, long, ...]
for T in T_values:
    R_T.append(eval_reasoning(model, T))     # 500 samples
    H_T.append(eval_perception(model, T))    # 500 samples

R_norm = (R - min(R)) / (max(R) - min(R))
H_norm = (H - min(H)) / (max(H) - min(H))

RH_AUC = 0
for i in range(len(T) - 1):
    width = R_norm[i+1] - R_norm[i]
    height = (H_norm[i+1] + H_norm[i]) / 2
    RH_AUC += width * height
```

ECRD 在 Qwen2.5-VL-7B 上 RH-AUC: 0.51 → 0.58（+0.07）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RH-AUC 配套 RH-Bench（1000 samples：500 reasoning + 500 perception），reasoning 样本来自 MathVision、MathVista、MMMU、ScienceQA；perception/hallucination 样本来自 MMHalu、MMVP、HallusionBench、VMCBench。包含多项选择和开放式问题。RH-AUC 已成为评估 visually-grounded reasoning 方法的关键指标之一。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

## TEMP-LORA (Temporary Low-Rank Adaptation / 临时低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TEMP-LORA 是 SlowFast-VGen 提出的推理时快速学习方法，将情节记忆（episodic memory）存储在 LoRA 低秩参数中。原始 TEMP-LORA（Wang et al. 2024b）为长文本生成设计，逐步生成新文本 chunk 并将生成的 chunk 作为 ground-truth 训练模型。SlowFast-VGen 将其改进用于长视频生成：每轮推理迭代 i，生成新视频 chunk Y_i 后，将输入 latent X_i 和输出 latent Y_i 拼接成时序连续体 X_i' = X_i ⊕ Y_i，对全序列添加噪声得 z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε，然后用去噪 UNet 在全序列上训练 TEMP-LORA 参数 Θ_i（不含文本条件），使 LoRA 参数存储整个生成轨迹的记忆。核心改进：丢弃原始 TEMP-LORA 的 input→output 格式，对拼接全序列加噪去噪，强调记忆整个轨迹而非关注即时转换。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SlowFast-VGen 推理时 Fast Learning (TEMP-LORA)
# 输入: 冻结慢学习权重 Φ, TEMP-LORA 参数 Θ_0, 快速学习率 α
# 输出: 长视频序列 Y

X_0 = VAE_ENCODE(X_0)           # 首帧编码到 latent space
Y = X_0

for i in 0..I-1:
    if i != 0:
        X_i = Y_{i-1}            # 上一轮输出作为当前输入
    C_i = User_Input(i)           # 当前 action 文本条件
    Y_i = (Φ + Θ_i)(X_i, C_i)    # 生成当前 chunk（慢学习权重 + 快学习 LoRA）
    Y = Y ⊕ Y_i                   # 拼接到最终序列

    # 训练 TEMP-LORA 存储情节记忆
    X_i' = X_i ⊕ Y_i              # 拼接输入输出 latent
    z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε  # 全序列加噪
    loss_Θ = ||ε - ε_{Φ+Θ_i}(z_t^{i'}, t)||²    # 无文本条件，全序列去噪
    Θ_{i+1} = Θ_i - α·∇_Θ loss_Θ                # 更新 LoRA 参数

Y = VAE_DECODE(Y)                # latent 解码回像素空间
```

Annotations:
- Φ: 预训练 UNet 权重（slow learning weights），推理时冻结
- Θ_i: TEMP-LORA 低秩矩阵（fast learning weights），W' = Φ + Θ，rank=32
- X_i': 拼接的时序连续体，维度为 2·(fp+fg) 帧的 latent
- z_t^{i'}: 全序列加噪 latent（注意：不保留干净条件帧，与 slow learning 的 masked conditioning 不同）
- 训练不含文本条件 c，专注于轨迹记忆而非条件生成
- 推理 overhead: +6.8% 时延 (12.93s→13.81s)，+3.7% 显存 (9579MB→9931MB)
- 遵循 local learning rule: ΔW 仅依赖当前迭代的局部 input-output 对

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TEMP-LORA 在 SlowFast-VGen 中的实现：基于 ModelScopeT2V 的 3D UNet，在 UNet 的 target_modules 上应用 LoRA（rank=32），使用 Adam 优化器，fast learning rate=1e-4。每轮推理迭代在单张 V100 GPU 上执行 TEMP-LORA 训练（一次前向+反向传播，仅更新 LoRA 参数）。Slow-Fast Learning Loop 扩展用法：内层 fast learning 循环在每个 episode 上积累 TEMP-LORA 参数 Θ^e；外层 slow learning 循环固定 Θ^e，利用多 episode 数据更新核心权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。原始 TEMP-LORA（为文本设计）使用 input→output 格式，SlowFast-VGen 的消融实验（Table 4）显示原版 TEMP-LORA 的 SCuts=0.55 劣于改进版的 0.37，因为原版关注即时转换而非全轨迹一致性。消融还显示"无 local learning rule"变体（采样全序列训练）SCuts=0.36 表现也不错但会导致后期帧过平滑。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

## Masked Conditional Video Diffusion (掩码条件视频扩散)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Masked Conditional Video Diffusion 是 SlowFast-VGen 中慢学习（slow learning）阶段使用的条件视频生成方法，基于 Voleti et al. (MCVD, 2022) 的框架。核心思想：给定 fp 帧过去帧和 fg 帧待生成帧，过去帧的 latent 保持干净（不加噪）作为条件，仅对待生成的 fg 帧添加高斯噪声；将干净条件帧和加噪生成帧拼接后送入 UNet 去噪；在计算 loss 时 mask 掉条件帧部分，仅在生成帧上计算 MSE loss。这使得模型学会基于前序视频和语言 action 生成后续视频 chunk。该方法支持任意长度 ≤ context window（32 帧）的 fp 和 fg 组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Slow Learning: Masked Conditional Video Diffusion
# fp: 过去帧数, fg: 待生成帧数

# 前向扩散
z_{0,:fp} = z_{0,:fp}                                              # 条件帧 latent (clean)
z_{t,fp:(fp+fg)} = sqrt(ᾱ_t)·z_{0,fp:(fp+fg)} + sqrt(1-ᾱ_t)·ε    # 生成帧 latent (加噪)
z_t = concat(z_{t,:fp}, z_{t,fp:(fp+fg)})                         # 拼接送入 UNet

# UNet 去噪
ε_pred = ε_Φ(z_t, t, c)   # c = CLIP 编码的语言 action text

# Masked Loss（仅在后 fg 帧计算）
loss = ||ε - ε_pred[fp:(fp+fg)]||²
```

Annotations:
- z_{0,:fp}: 前 fp 帧的 VAE 编码 latent（clean，作为条件）
- z_{0,fp:(fp+fg)}: 后 fg 帧的 ground-truth latent
- ε: 采样的标准高斯噪声
- ᾱ_t: 扩散累积噪声系数
- c: CLIP text encoder 编码的 action 文本条件
- UNet 输出 ε_pred 包含所有 fp+fg 帧的噪声预测，但仅取 [fp:(fp+fg)] 范围计算 loss
- 条件帧的 latent 在去噪推理时保持已知（不参与 DDIM 采样）
- fp=1 时退化为单帧条件生成（如 robot 数据）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SlowFast-VGen 基于预训练 ModelScopeT2V（latent video diffusion model）实现 masked conditional video diffusion。ModelScopeT2V 使用 3D UNet 架构（spatial convolutions + temporal convolutions + attention blocks），VAE 编码视频到 latent，CLIP ViT-H/14 编码文本。慢学习在约 64 张 V100 GPU 上训练，batch size=128，slow learning rate=5e-6，冻结 VAE 和 CLIP Encoder，仅训练 UNet。训练视频长度 ≤ 32 帧（context window）。与标准 video diffusion 的区别：标准方法对全部帧加噪+去噪；MCVD 的 masking 机制使条件帧保持干净信号，引导生成帧与条件帧一致。MCVD 原论文（Voleti et al., 2022, arXiv:2205.09853）支持 prediction、generation 和 interpolation 三种任务。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

## Slow-Fast Learning Loop (慢-快学习循环)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Slow-Fast Learning Loop 是 SlowFast-VGen 提出的双速学习循环算法，将推理时快速学习（TEMP-LORA）嵌入到慢学习训练过程中，用于需要从多 episode 经验中学习的长时规划任务。其生物学动机来源于认知科学中的互补学习系统（Complementary Learning Systems）：海马体（hippocampus）支持快速编码新经验形成情节记忆，新皮层（neocortex）逐步将记忆抽象整合为通用知识。双循环结构：内层（fast learning loop）在每个 episode 上运行 TEMP-LORA 快速适配并积累数据（input, ground-truth output, TEMP-LORA 参数 Θ）；外层（slow learning loop）固定 TEMP-LORA 参数，利用多 episode 积累的数据更新核心模型权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Slow-Fast Learning Loop Algorithm
# Φ: 任务特定慢学习权重, D: 任务特定数据集, β: 慢学习率

while not converged:
    D_s = ∅                       # 准备慢学习数据集
    for each (x, episode) in D:
        # 内层: Fast Learning Loop
        初始化 TEMP-LORA 参数 Θ_0^e
        将 episode 分割为 I 个短序列: {X_i^e}_{i=0}^{I-1}
        for i in 0..I-1:
            # 收集数据点: (输入, ground-truth 输出, 当前 TEMP-LORA 参数)
            D_s = D_s ∪ {X_i^e, X_{i+1}^e, Θ_i^e}
            
            # Fast learning: 固定 Φ, 更新 Θ_i^e
            Y_i = (Φ + Θ_i^e)(X_i^e)                    # 生成
            X_i' = X_i^e ⊕ Y_i                          # 拼接输入输出
            z_t = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε       # 加噪
            loss_Θ = ||ε - ε_{Φ+Θ_i^e}(z_t, t)||²      # 去噪 loss
            Θ_{i+1}^e = Θ_i^e - α·∇_Θ loss_Θ            # 更新 TEMP-LORA

    # 外层: Slow Learning Loop
    for {X_i^e, X_{i+1}^e, Θ_i^e} in D_s:
        Φ_i^e = Φ + Θ_i^e                                # 组合慢+快学习权重
        Y_pred = Φ_i^e(X_i^e)                            # 预测输出
        loss_Φ = ||Y_pred - X_{i+1}^e||²                 # 与 ground-truth 比较
        
        # Slow learning: 固定 Θ_i^e, 更新 Φ
        Φ = Φ - β·∇_Φ loss_Φ
```

Annotations:
- Θ_i^e: episode e 中第 i 步的 TEMP-LORA 参数，存储截至该步的情节记忆
- D_s: 收集自所有 episodes 的 (input, output, Θ) 三元组
- β: 慢学习学习率，在 slow-fast loop 中用于更新 Φ
- 外层固定 Θ_i^e 仅更新 Φ：确保逐步巩固泛化知识而非覆盖特定记忆
- 该循环对完整预训练开销大，适用于特定领域/任务的 fine-tuning

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SlowFast-VGen 在两个长时规划任务上验证了 slow-fast learning loop 的效果：(1) RLBench 机器人操作——移动物体后归位，测量到先前位置的距离；(2) Minecraft 游戏导航——沿路径返回起点，测量到预定义路径点的最近距离。实验（Table 2）显示完整的 slow-fast learning loop 优于"无 loop"消融变体：RLBench Dist 0.013 vs 0.055，Minecraft Dist 1.51 vs 2.23。Loop 的关键价值在于：仅 TEMP-LORA（无 loop）能存储单 episode 记忆但无法跨 episode 泛化技能；加入 slow loop 后模型能从多次"移动-归位"或"导航-返回"经验中学习通用策略。论文指出该循环对完整预训练（200k 数据）开销过大，建议用于特定领域 fine-tuning。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

## Complementary Learning System in Video Generation (视频生成中的互补学习系统)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
互补学习系统（Complementary Learning System, CLS）是认知科学理论（McClelland et al., 1995），认为人脑由两个互补的学习机制组成：新皮层（neocortex）负责慢学习——通过大量经验逐步构建世界模型，支持泛化和决策；海马体（hippocampus）负责快学习——从单次经验快速编码情节记忆，支持快速适应和一致性保持。SlowFast-VGen 将此理论映射到视频生成系统：慢学习 = masked conditional video diffusion pre-training（类比新皮层），快学习 = TEMP-LORA 推理时训练（类比海马体）。数学对应：W' = W + ΔW = W_slow + W_fast = Φ + Θ，其中 Φ 为慢学习预训练权重，Θ 为快学习 LoRA 参数。LoRA 的 local learning rule（Δc(t) = x^μ(t)·y^μ(t)）与神经科学的 Hebbian-like local learning 对应。Slow-Fast Learning Loop 进一步模拟海马体-新皮层的记忆巩固过程：快速编码 → 离线整合 → 抽象为通用知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLS 映射到 SlowFast-VGen 的计算框架
# 
# 生物学 (McClelland et al. 1995):
#   新皮层: 慢速学习 → 结构化世界知识 → 泛化
#   海马体: 快速学习 → 情节记忆编码 → 快速适应
#   记忆巩固: 海马记忆 → 离线 → 新皮层整合
#
# SlowFast-VGen 计算类比:
#   Slow Learning (Φ):   预训练视频扩散模型 → 通用世界动力学
#   Fast Learning (Θ):   TEMP-LORA 推理时更新 → 单 episode 轨迹记忆
#   Slow-Fast Loop:      内层-快速编码 → 数据收集 → 外层-慢整合

# Local Learning Rule 对应:
# 生物学: Δc(t) = x^μ(t)·y^μ(t)   (Palm, 2013)
# LoRA:   W' = W + ΔW = Φ + Θ
#         Θ 更新: ΔΘ ∝ z_{0,i-1} ⊕ z_{0,i}  (仅依赖局部 input-output)
```

Annotations:
- CLS 理论解释了为什么仅 slow learning（预训练）不足以生成一致长视频：缺乏海马体式快速记忆机制，模型无法记忆超出当前 context window 的轨迹
- LoRA 的 local update 机制天然适配情节记忆：每个 episode step 的 ΔW 仅依赖于当前迭代的 input-output 对
- Slow-Fast Loop 的"离线整合"阶段对应：外层循环在任务完成后利用所有 episode 数据统一更新 Φ

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CLS 在 SlowFast-VGen 中作为理论框架指导系统设计，而非直接实现的计算组件。具体映射：(1) Slow Learning → 200k 视频数据上预训练 MCVD，构建通用世界模型（覆盖 Unreal/Minecraft/Kitchen/Robot/Driving 五大场景）；(2) Fast Learning → TEMP-LORA (rank=32) 推理时存储 episode 记忆，不修改慢学习权重；(3) Memory Consolidation → Slow-Fast Learning Loop 将多 episode 的 TEMP-LORA 记忆整合到核心权重 Φ。该框架的关键洞察：现有视频生成模型（WorldDreamer, Pandora, iVideoGPT 等）仅实现慢学习（预训练），缺失快学习能力，导致长视频一致性差。CLS 理论为"为什么需要双速学习"提供了认知科学基础。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

## Group-Decoupled Position Encoding (GDPE)（分组解耦位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GDPE 是 Speak While Watching 论文提出的三种打破位置连续性约束的并行流式策略中综合最优的一种。其核心思想是将 MLLM 流式推理中的视觉 token 和文本 token 分配为两个**独立的位置编码组**：视觉组从 pos_v=0 开始独立递增，文本组从 pos_a=0 开始独立递增，组间位置索引完全解耦。由于两组位置空间独立，视觉 prefill（新帧编码）和文本 decode（自回归生成）可以并行执行——新帧的位置索引仅依赖之前视觉 token 的数目（已知），不依赖当前文本生成的长度（未知）。训练时通过自定义 causal mask 确保：V_{i+1} 只 attend 到 V_{1..i}（视觉组内因果），A_i 只 attend 到 V_{1..i} 和 A_{1..i}（跨模态因果+文本组内因果）。仅需在 Qwen2.5-VL 上做少量 SFT（20K 样本），无需修改模型架构权重。GDPE 在鲁棒性和语义质量之间取得最佳平衡——BLEURT 51.53、流利度 4.56，且对调度扰动（Random schedule）表现最鲁棒（BLEURT 51.76 甚至优于 fixed 的 51.53）。

同类策略对比：
- **OSPE (Overlapped Streaming Position Encoding)**：视觉和文本从同一 max 位置共享起始索引，文本段内连续但跨段非连续 → 流利度 4.48 低于 GDPE/GIPE。
- **GIPE (Gap-Isolated Position Encoding)**：GDPE 基础上两组间加固定 gap Δ → 流利度最高 4.85（接近 Offline 4.84），但语义捕捉略低于 GDPE。
- **GDPE**：综合最优，是最推荐的默认并行流式配置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GDPE 位置编码分配（推理时视觉和文本可并行执行）
pos_v = 0  # 视觉组独立位置计数器（t 维度）
pos_a = 0  # 文本组独立位置计数器（t 维度）

for i in 1..N:  # N 轮流式推理
    # === 视觉流（可与文本流并行） ===
    V_i_tokens = vision_encoder(frame_i)    # m_i 个视觉 token
    for j in 1..m_i:
        PE_3D(V_i_tokens[j], x, y, t=pos_v)  # Qwen2.5-VL 3D RoPE
        pos_v += 1

    # === 文本流（可与视觉流并行） ===
    A_i_tokens = autoregressive_decode(causal_mask)  # 生成 k_i 个文本 token
    for j in 1..k_i:
        PE_1D(A_i_tokens[j], pos=pos_a)      # 三维坐标一致
        pos_a += 1

# Causal Mask（训练时已知完整序列）:
# V_{i+1} attend to V_{1..i} only（视觉组内因果）
# A_i attend to V_{1..i} + A_{1..i} only（跨模态因果+文本组内因果）
# 关键：A_i 不受后续 V_{i+1} 插入的影响，文本序列连续性完整

# 理论加速:
# Interleave: T_i = m_i/R_v + k_i/R_t（串行）
# GDPE/Parallel: T_i = max(m_i/R_v, k_i/R_t)（并行）
# 加速比 S = (m_i/R_v + k_i/R_t) / max(m_i/R_v, k_i/R_t) ≤ 2×
# r = (m_i/R_v)/(k_i/R_t)：r≈1 时加速最大(≈2×)；r>>1 时接近无加速
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/EIT-NLP/Speak-While-Watching。通过环境变量 `QWEN2_5_VL_VARIANT=group` 选择 GDPE 模式。代码修改 Qwen2.5-VL 的位置 ID 分配逻辑和 causal mask，无需修改模型权重。训练 `scripts/sft.sh` → checkpoint output → `eval.sh` 评估。重要说明：代码仓库不实现物理多 GPU 并行——位置编码层面的并行逻辑需配合系统级实现（双 GPU prefill/decode 流水线）才能获得完整加速。wait-K=3 配置（每帧生成约 3 个 token，匹配 PE-Video/FunQA 平均帧-文本比）。2fps 采样，20K 训练样本。适用于 Video Description 和 Video QA 流式任务。

涉及论文标题：
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models

## Pseudo-streaming（伪流式评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pseudo-streaming（伪流式）是现有流式视频理解基准中常见但存在缺陷的评估设置：在评估时将视频在 query 时间戳处截断（仅使用 query timestamp 之前的帧），但仍以离线批量方式处理——模型可一次性加载所有截断帧到 GPU、一次性编码、完整访问上下文。这与真实流式场景的根本区别在于：(1) 模型在实际推理时不需要逐帧增量编码，(2) 不需要维护跨帧 memory state，(3) 不存在帧积压和编码吞吐瓶颈。因此 pseudo-streaming 评估的 accuracy 无法反映真实流式部署中的实际表现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Pseudo-streaming Pipeline
video = load_full_video(path)          # 完整加载
t_query = query_timestamp               # query 时间戳
frames_before = video[0:t_query]        # 截断到 query 时刻
visual_tokens = vision_encoder(frames_before)  # 一次性批量编码
input_ids = concat(visual_tokens, text_tokens)
answer = model.generate(input_ids)      # 标准离线推理

# True Streaming Pipeline (StreamingEval)
for frame in stream(fps=1):            # 逐帧增量到达
    z_i = vision_encoder(frame)         # 逐帧编码
    memory_bank.update(z_i)             # 增量更新 memory
    if memory_bank.full:                # FIFO 淘汰
        memory_bank.evict_oldest()
# query 到达时仅能访问 memory_bank snapshot（不含未来帧）
answer = model.generate(memory_bank.snapshot(), query)
```

Pseudo-streaming 的核心缺陷：(a) 模型可 "提前看到" 所有截断帧，获得全局 context；(b) 不存在逐帧编码的 timing 约束；(c) 无法暴露 visual encoding 吞吐瓶颈。StreamingEval 的严格因果约束（Frame Player → Encoder-Memory Updater → Responder 三进程异步 pipeline）和 MaxFPS/TTFT 等指标正是为了修正这些缺陷。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
早期流式基准（VStream-QA、StreamingBench 的部分设置、部分 OVO-Bench 评估）采用 pseudo-streaming。StreamingEval 明确区分了 pseudo-streaming 和真实流式评估：前者只需修改数据加载方式（截断帧子集），后者需实现完整的三进程异步 pipeline 并测量系统级指标。在 StreamingEval 的统一协议下，所有模型都在严格 streaming 条件下评估，使得 pseudo-streaming 的优势（如离线模型的高 accuracy）在真实约束下体现出 latency/resource trade-off。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

## StreamingScore（流式综合评分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
StreamingScore 是 StreamingEval 提出的综合评估指标，将流式视频理解中的四个核心维度（吞吐、准确率、延迟、资源消耗）整合为单一可调权重的标量分数。定义：

$$\text{StreamingScore}(\mathbf{w}) \triangleq \frac{\text{MaxFPS}^{w_f} \cdot \text{Acc}^{w_a}}{\text{TTFT}^{w_t} \cdot M^{w_r}}$$

其中 $M \triangleq \text{Mem} \cdot \ln(\text{Params})$（结合内存占用的资源项），权重约束为 $w_f, w_a, w_t, w_r \ge 0, w_f + w_a + w_t + w_r = 1$。更高的 StreamingScore 表示模型在更高吞吐下实现更好准确率、更低首 token 延迟和更低资源消耗。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# StreamingScore 计算
# 输入: 模型在统一流式协议下的四项指标
Acc = accuracy_on_benchmark(model, dataset)     # [0, 100]
MaxFPS = max_sustainable_fps(model)              # > 0
TTFT = time_to_first_token(model)               # seconds
Mem = memory_bank_budget                        # GB
Params = model_parameter_count                  # billions

# 资源项
M_resource = Mem * ln(Params)

# 默认统一权重 (w_f=w_a=w_t=w_r=0.25)
StreamingScore = (MaxFPS^0.25 * Acc^0.25) / (TTFT^0.25 * M_resource^0.25)

# 场景感知权重示例:
# "Best Answer" (w_a=0.4, w_f=w_t=w_r=0.2)
# A_score = (MaxFPS^0.2 * Acc^0.4) / (TTFT^0.2 * M_resource^0.2)

# "Interaction First" (w_t=0.4, others=0.2)
# I_score = (MaxFPS^0.2 * Acc^0.2) / (TTFT^0.4 * M_resource^0.2)

# "Edge Resource-Saving" (w_r=0.4, others=0.2)
# R_score = (MaxFPS^0.2 * Acc^0.2) / (TTFT^0.2 * M_resource^0.4)

# "Throughput First" (w_f=0.4, others=0.2)
# T_score = (MaxFPS^0.4 * Acc^0.2) / (TTFT^0.2 * M_resource^0.2)
```

StreamingEval 实验表明：不同权重下模型排名可互换（如 Qwen3-VL 在 Best Answer 排第 1，Flash-VStream 在其余三项排第 1），但整体趋势统计稳健（Spearman ρ ∈ [0.972, 0.993]），降低了模型仅通过倾斜权重 "刷榜" 的风险。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamingScore 作为默认统一评估指标，使用均等权重（0.25, 0.25, 0.25, 0.25），同时支持四种部署场景偏好。在 StreamingEval 开源代码中通过 `compute_streaming_score(acc, maxfps, ttft, mem, params, weights)` 函数计算。局限性：(a) 权重选择是主观的——不同应用场景合理选择不同；(b) 单项指标异常值可能主导分数（如 MaxFPS=0.14 的 VideoChatOnline 在任意权重下 StreamingScore 都极低）；(c) StreamingScore 不替代单独指标分析，而是补充综合视角。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

## Attention Sink（注意力汇）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sink 是 Guangxuan Xiao et al. (ICLR 2024) 在 StreamingLLM 中首次发现并命名的现象：在 LLM 的自回归解码中，初始几个 token（通常为 system prompt / BOS token）会吸收不成比例的大量注意力分数（attention scores），成为 attention 分布的"汇"（sink）。即使这些初始 token 的语义不重要，删除它们也会导致模型输出质量急剧下降（perplexity 飙升）。Attention sink 之所以产生，是因为 softmax 要求所有注意力权重和为 1，而初始 token 的 Key 经过训练形成了较大的范数，天然吸引注意力。StreamingLLM 利用这一现象，在推理时保留少量 sink tokens（通常 4 个）+ 近期窗口 tokens，实现无限长度流式解码。StreamingVLM 将此概念扩展到多模态视频流领域，将 attention sink（Tsink=512 个初始 text tokens，包括 system prompt 和早期文本）纳入 KV Cache 保留策略，作为稳定长程视频流式推理的三要素之一（sink + text window + vision window）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Sink in Softmax Attention
# Q, K: [seq_len, d_head]
scores = Q @ K.T / sqrt(d_head)  # [seq_len, seq_len]
attn_weights = softmax(scores, dim=-1)

# 现象: attn_weights[:, 0:4] 的值显著大于其他位置
# 原因是前几个 token 的 K 范数大 ||K[:4]|| >> ||K[4:]||

# StreamingLLM 利用 Attention Sink:
# KV Cache 中永久保留 sink tokens + 近期 window tokens
# sink_indices = [0,1,2,3]  # 前 4 个 token 永不驱逐
# 推理时 attend_to = [K_sink | K_recent_window]
```

Annotations: softmax 约束 Σ attn_weights = 1 导致 attention sink 必然发生。Qwen2.5-VL 架构中，StreamingVLM 使用 Tsink=512（远大于纯文本 StreamingLLM 的 4 个），因为多模态场景中 system prompt + vision-related context 更长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 StreamingVLM 中，attention sink 的使用方式：(1) SFT 训练时，截取前 Tsink 个 previous text tokens + 后 Twindow 个 recent text tokens 作为 SFT sample 的 context；(2) 推理时，Tsink tokens 永久留在 KV cache 中不被驱逐，作为 attention 稳定性的锚点；(3) Ablation 显示 Tsink=512, Twindow=512 是最优配置（Table 5），纯 sink（Twindow=0）仍能保持较高 win rate，纯 window（Tsink=0）则性能下降。Attention sink 对 fine-tuned 模型尤其重要——ReKV 等训练无关驱逐方法破坏 sink token 布局导致输出异常（Table 2）。原始论文：Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024。代码开源：https://github.com/mit-han-lab/streaming-llm。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

VideoNSA 从 learnable sparse attention 角度分析了 attention sink 行为。论文发现 NSA 的三个稀疏注意力支路表现出截然不同的 sink 模式：(1) Compression 支路产生最多 sink（token merging 放大某些 token norm 同时抑制其他），形成沿 value norm 轴的带状集中分布；(2) Selection 支路几乎无 sink（top-k block filtering 强制平滑 value norm 分布）；(3) Sliding Window 支路在局部邻域边界产生周期性稀疏 sink peaks。通过动态 gating 加权融合三支路后，VideoNSA 整体 sink 比例仅 0.3%，远低于 dense FlashAttention。此外，dense attention 的 sink 均匀分布在全序列，而 VideoNSA 的动态稀疏机制使 sink 在时间轴上更平滑分布，缓解对序列起始位置的过度依赖。这些发现表明 learnable sparse attention 可以通过控制不同支路的 value norm 分布来主动管理 attention sink 的产生和位置分布，而非被动接受或绕过 sink 现象。

公式定义（VideoNSA 沿用 Pai et al., 2025a）：
```
Attention Sink = 1{α > 0.1 ∧ ||v|| < median(||v||) - 2*IQR(||v||)}
```
其中 α 是 key 收到的平均 attention score，||v|| 是 token 的 value norm。

**Visual Attention Sink in MLLMs from VisiPruner**：VisiPruner (EMNLP 2025) 通过系统消融实验首次揭示，MLLM 中的视觉 token 在浅层（layer 1-7）和深层（layer 26+）均表现为 attention sink——即高 attention score 不代表高信息贡献。具体证据：(1) 浅层中最高 attention 的视觉 token 在不同输入指令下保持不变（静态 attention pattern），mask 这些 token 几乎不影响性能（Tab. 1）；(2) 视觉 sink token 的 value vector L1 norm 显著低于非 sink token——因为 softmax 归一化迫使多余的 attention mass 流向 value norm 极小的 token，类似 bias term；(3) 浅层 layer 1 中视觉和文本 token 的 value vector 分布迥异（Fig. 7），需要 modality-specific sink；layer 2+ 文本 token（如 system prompt）即可替代视觉 token 作为 attention sink；(4) 移除浅层视觉 sink token 后，attention mass 会自动重新分配到文本 sink token（App. E.2），进一步证实 sink 的"结构稳定"角色而非信息传递角色。VisiPruner 利用这一发现，在 layer 1 将所有视觉 cross-attention 合并到单个随机 token 作为 attention sink，layer 2+ 完全跳过视觉 attention，保持性能不变。

**VisionSelector 对 Attention Sink 的分析与应用**：VisionSelector 通过端到端可学习的 token 重要性评估，有效规避了 attention sink 对启发式剪枝方法的负面影响。具体机制：(1) 论文假设（Sec 4.3）：attention-based baseline（FastV, VisionZip）在极端压缩率下性能崩塌（VisionZip 从 20%→10% 下降 ~14pp），原因是预训练 attention map 中的 attention sink 偏差——早期位置 token 获得不成比例的高 attention，但语义上不相关——在 10% 极低保留率下迫使模型保留这些"位置优先但语义无关"的 token，导致性能崩溃。(2) LIS 通过自己的 QK^T 全局交互计算重要性得分，而非依赖 MLLM 内部预训练 attention map，因此不受 attention sink 偏差影响（LIS 的注意力矩阵 A = QK^T/√d 在训练中被 CE loss 端到端优化，learned importance 反映语义重要性而非位置偏见）。(3) 可视化（Fig. 3/5/6/7/8）：VisionSelector 的温度图中，关键语义区域（如电话号码、文字、logo）获得高 score，而 background tokens 为低 score；VisionZip 则因 attention sink 保留了位置靠前但无信息的 background tokens。(4) 正向发现（MME 30% 保留率 100.07%）：learned selection 滤除噪声 token 后可实现增益性压缩——这与 VisiPruner 的"移除浅层视觉 sink token 不影响性能"发现一致，表明 VisionSelector 的 LIS 实际上也学会将 attention mass 从视觉 sink token 重定向到语义关键的 token。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs

## Contiguous RoPE（连续旋转位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Contiguous RoPE 是 StreamingVLM 提出的位置编码技术，用于解决流式 KV Cache 驱逐后位置索引断裂和超出训练长度的问题。在标准 RoPE（Rotary Position Embedding）中，每个 token 的位置索引从 0 开始递增，推理时位置可能增长到远超训练最大长度的值，导致 out-of-distribution 退化。Contiguous RoPE 的核心操作是：当旧 token 从 KV cache 中被驱逐后，将剩余和后续 token 的 RoPE 位置索引**左移**，使它们保持与最后保留 token 的**数值连续**。当视频长度超过总窗口尺寸后，有效 RoPE 索引停止增长，保持在有界范围（不超过训练最大长度）。对 Qwen-VL 家族的 3D RoPE（time, height, width），同样应用 contiguous 左移规则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 标准 RoPE（Native）:
# 推理时位置索引持续增长: pos = [0, 1, 2, ..., T-1]
# 当 T > L_train_max 时，索引超出训练分布

# Contiguous RoPE:
# 维护 position_offset 追踪实际起始位置
# 驱逐旧 token 后，剩余 token 索引左移
position_offset = 0  # 随着驱逐递增
available_positions = list(range(position_offset,
                                  position_offset + cache_len))
# 当 cache_len >= L_train_max 时:
# effective_positions = [position_offset % L_train_max, ...]
# 索引不再增长，在 bounded range 内循环但保持连续性

# 对 3D RoPE (Qwen-VL):
# 3D_indices = build_3d_position(time_idx, h_pos, w_pos)
# contiguous_3d_indices = left_shift(3d_indices, offset)
```

Annotations: Native RoPE 在 infinite stream 上 win rate 从 63.23 降至 25.09（vs GPT-4o, Table 4），Contiguous RoPE 维持 66.18%。100s chunking 可部分恢复 Native RoPE（63.23），但损失长程一致性。Contiguous RoPE 不改变 RoPE 的 base frequency，仅改变 position assignment。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理时实现：(1) 在每步推理前计算当前 KV cache 中每个 token 的 contiguous 位置索引；(2) 将位置索引传入模型的标准 RoPE 模块（不修改 RoPE 计算本身）；(3) 对 3D RoPE，需在构建 (t, h, w) 后统一左移 t 维度索引。训练时不需要 contiguous RoPE（训练使用 full attention with native RoPE on short chunks，W=24s 远小于训练最大长度）。适用范围：任何需要在推理时驱逐部分 KV cache、同时保持位置编码连续性的流式 Transformer 推理场景。类比技术：YaRN、LongRoPE（调整 base frequency）、NTK-aware scaling——都解决位置外推问题但策略不同；Contiguous RoPE 关注的是驱逐后连续性而非高频基频扩展。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

## Overlapped-Chunk Full-Attention Training（重叠块全注意力训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Overlapped-Chunk Full-Attention Training 是 StreamingVLM 提出的训练策略，解决"train short, test long"的分布偏移问题——训练无法使用超长视频（计算复杂度 O(T²) 和硬件限制），但推理需要在无限流式输入上稳定运行。核心思想：(1) 将长视频切分为 W=24s 的连续 chunk（重叠 O=12s）；(2) 每个 chunk 作为独立训练样本，chunk 内做 full attention（所有 token attend 到同 chunk 内所有 token）；(3) Vision 和 text tokens 在 chunk 内以 1s 间隔交错排列（非传统 VLM 的 vision-then-text 布局）；(4) 仅在 text position 计算 loss。这种设计的巧妙之处在于：chunk 内 overlapped full attention 的 effective attention pattern 天然近似推理时的 "attention sink + 近期 text 窗口 + 近期 vision 窗口" 模式（Figure 4 右侧），使模型不经特殊训练就习得 recency bias，且训练不增加额外的 attention mask 复杂性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
参数: W=24s, O=12s, T_sink=512, T_window=512

# 训练数据准备
For each training video:
    # Step 1: 切分为 overlapped chunks
    chunks = []
    for start in range(0, video_duration - W, W - O):
        chunk = video[start:start+W]  # 24s segment

    For each chunk_i:
        # Step 2: 以 1s 间隔交错采样 vision/text tokens
        V_chunk, T_chunk = [], []
        for sec in 0..W-1:
            V_chunk.append(vision_encoder(chunk_i[sec]))
            if has_commentary(chunk_i[sec]):
                T_chunk.append(tokenize(commentary[sec]))
            else:
                T_chunk.append(tokenize("..."))  # silence placeholder

        # Step 3: 取前序 previous text 的 sink + window
        prev_text = commentary_before_chunk
        prev_sink = prev_text[:T_sink]
        prev_window = prev_text[-T_window:]

        # Step 4: Full attention within chunk
        input_seq = interleave(V_chunk, T_chunk)
        # 布局: V[0],T[0], V[1],T[1], ..., V[W-1],T[W-1]
        mask = causal_full_attention  # within-chunk full attention
        loss = CE(logits[text_positions], labels)
```

Annotations: 训练不复制推理的 sink+sliding window mask，而是用 overlapped full attention 近似。W=24s, O=12s 保证每个 chunk 至少 2*W words 的 commentary label。无解说秒插入 "..." placeholder，"..." token 的 loss 也被计算（训练模型学会沉默）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamingVLM 的两阶段训练：(1) SFT：Inf-Streams-Train (525K) + Live-WhisperX-526K (526K)，overlapped-chunk full-attention 格式；(2) Annealing：14K 实时解说样本（16-64s clips，GPT-5 筛选实时解说占比 >80%）。总计算量 128 H100-days。此策略不需要极长视频训练数据或 attention mask 修改。关键 insight：训练时不需要显式地让模型"学习" sliding window——只要 chunk 间有重叠、chunk 内 full attention，模型自动学到 recency bias。验证：Table 6 显示 overlapped SFT 策略相比仅用 Live-WhisperX-526K 在 Inf-Streams-Eval 上提升 +31.29（win rate vs GPT-4o mini）。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

## Instruction Diversity in Video-LLM Training（视频-LLM训练中的指令多样性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Instruction Diversity（指令多样性）指 MLLM 训练数据中 instruction 文本的语义丰富程度，包括句式、任务类型、推理深度、领域覆盖等维度的变化。Sparrow 论文通过 t-SNE 可视化首次系统性地揭示了视频 instruction 数据的多样性不足问题：ShareGemini 数据集的 instruction 仅来自 9 种固定模板变体（如 "Describe this video in detail"），t-SNE 图中呈现 9 个清晰聚类；Video-ChatGPT 数据集的 instruction 虽然包含具体视频内容相关问题（视频摘要、内容问答、创造性任务），但由于 self-instruction 的本质——基于固定 prompting 模板由 LLM（GPT-3.5）生成——其多样性同样有限。这种不足导致的直接后果是数据效率低下：当视频样本量从 30K 扩大到 100K（3.3×），模型性能仅从 55.8 提升到 56.3（+0.5 points）呈对数增长。这一定量缺陷此前在视频-LLM 领域未被系统量化和解决。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sparrow 论文中的 instruction 多样性分析方法：
```
# Instruction 多样性评估流程
def analyze_instruction_diversity(dataset, n_samples=5000):
    # Step 1: 采样 instruction 文本
    instructions = random_sample(dataset, n_samples)

    # Step 2: 将 instruction 编码为 embedding
    # 使用 sentence transformer（如 all-MiniLM-L6-v2, d=384）
    embeddings = SentenceTransformer.encode(instructions)
    # embeddings shape: [5000, d=384]

    # Step 3: t-SNE 降维可视化
    tsne = TSNE(n_components=2, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    # Step 4: 分析聚类特征
    # ShareGemini → 9 个清晰的聚类簇（对应 9 种模板变体）
    # Video-ChatGPT → 相对分散但覆盖范围仍有限
    # Sparrow hybrid 混合后 → 分布范围显著扩展
    return reduced, cluster_labels

# Sparrow 增强 instruction 多样性的方法
def sparrow_augment_diversity(video_data, text_data, mix_ratio=2):
    """
    video_data: (ShareGemini + Video-ChatGPT, 1:1 采样)
    text_data:  (LongAlpaca + LongQLora, 1:1 采样)
    每个 text sample: (long_context, instruction, answer)
      - long_context → split by ~115 words → render as images
      - instruction: 书籍摘要、论文问答、文档理解等多样化任务
    """
    syn_samples = [text_to_images(s) for s in text_data]
    # 混合采样: video:synthetic = 2:1
    mixed_data = SampleConcat(video_data, syn_samples, ratio=mix_ratio)
    return mixed_data
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
提高 instruction 多样性有几种策略：
1. **数据源多样化**（Sparrow 方法）：引入文本域的 instruction 数据（LongAlpaca 覆盖书籍/论文的长上下文问答，LongQLora 覆盖长文档对话）。文本域 instruction 天然具有更高的提问多样性，无需额外标注，通过 text-to-image 合成转化为视觉格式混合训练。
2. **Human-in-the-loop 精炼**：人工标注者审查和丰富 instruction（如 Video-ChatGPT 的小部分数据），成本高但质量好。
3. **模板扩展**：在已有数据集上通过改写/重述扩展 instruction 模板，但覆盖范围受限于数据集固有内容。
4. **Multi-source 混合**：混合多个不同来源的数据集，利用不同数据集的自然分布差异增加多样性。

评估指标：当前主要通过 (a) t-SNE 可视化定性评估 instruction embedding 分布的覆盖范围；(b) 数据缩放实验的 learning curve 定量评估多样性改善效果——更陡峭/更持久的 scaling curve 说明多样性更足，更平的对数曲线说明多样性不足。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

## Text-to-Image Data Augmentation for Video Training / Sparrow Method（文本转图像数据增强 for 视频训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparrow Method 是一种无需调用视觉 API 的纯工程化数据增强方法，核心是将纯文本 instruction 数据转化为"类视频"的多图像序列，以桥接文本-视觉模态差异并丰富训练数据的 instruction 多样性。方法流程：(1) 从文本 instruction 数据集（LongAlpaca, LongQLora）取 (long_context, instruction, answer) 三元组；(2) NLTK 按 ~115 词分割 long_context 为多段；(3) 每段用 Pillow ImageFont 渲染为 448×448 白底黑字图像（20pt Arial Regular 字体，黑色，左右 20px margin）；(4) 生成 (images[], instruction, answer) 序列，格式与真实视频样本完全一致，可直接混合训练。与 TOPA/T3 等文本辅助方法不同：Sparrow 不提取视觉信息再转文字（信息损失），不调用 LLM API（零额外成本），而是将文字直接转为视觉表示。Sparrow 用 30K 混合数据（20K video + 10K synthetic）达到了 200K 纯视频数据相当的 Video-MME 性能（56.7 vs 56.3），GPU hours 从 276.8 降至 33.6（8.2× efficiency）。长视频理解额外提升 6.6 points（100K 规模）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Sparrow 数据增强 Pipeline ===

# Step 1: 文本 → 图像合成
def text_to_video_like(text_sample):
    # 1.1 NLTK 分词并按词数分割（~115 words/segment）
    import nltk
    from PIL import Image, ImageDraw, ImageFont
    words = nltk.word_tokenize(text_sample['long_context'])
    chunks = []
    for i in range(0, len(words), 115):
        chunks.append(" ".join(words[i:i+115]))

    # 1.2 PIL 渲染每个 chunk 为 448×448 图像
    images = []
    font = ImageFont.truetype('arial.ttf', 20)
    for chunk_text in chunks:
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        # 逐词绘制，控制换行（可用宽度 = 448 - 40 = 408px）
        y, current_line = 20, []
        for word in chunk_text.split():
            test_line = " ".join(current_line + [word])
            if len(test_line) * 10 <= 408:  # ~10px/char for 20pt
                current_line.append(word)
            else:
                draw.text((20, y), " ".join(current_line),
                          fill='black', font=font)
                y += 24
                current_line = [word]
        if current_line:
            draw.text((20, y), " ".join(current_line),
                      fill='black', font=font)
        images.append(img)
    return {'images': images, 'instruction': text_sample['instruction'],
            'answer': text_sample['answer']}

# Step 2: 混合训练（与视频数据共用同一 ViT encoder）
def sparrow_training_step(sample, image_llm):
    # 视觉编码（video frames 和 synthetic images 共用同一编码路径）
    visual_tokens = ViT(sample['images/frames'])  # [K, H*W, C]
    visual_emb = Projector(visual_tokens)          # MLP 投影
    text_emb = LLM.embed_tokens(tokenize(sample['instruction']))
    input_seq = concat([visual_emb, text_emb])
    logits = LLM(input_seq)
    loss = CE(logits[answer_pos], answer_tokens)   # 仅 answer token 计算 loss
    return loss

# 数据组织
# 文本来源: LongAlpaca (5K) + LongQLora (5K) = 10K synthetic
# 视频来源: ShareGemini (10K) + Video-ChatGPT (10K) = 20K video
# 混合比例: video:synthetic = 2:1
```

关键张量维度: InternVL-4B 每帧 256 visual tokens（单 tile 模式，关闭 patchify），max 64 frames；MiniCPM-8B 每帧 96 visual tokens，max 24 frames。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/VITA-MLLM/Sparrow (Python 3.9 + PyTorch + Flash-Attention 2)。合成数据集：https://huggingface.co/datasets/xjtupanda/Sparrow-Synthetic。

使用方法：
1. 准备文本数据（LongAlpaca, LongQLora 或任意含 long_context + QA 的数据集）
2. 运行合成脚本：NLTK 分割 → PIL 渲染 → 输出 (images[], QA) 对
3. 按 1:2 比例与真实视频数据混合
4. 标准 MLLM fine-tuning 协议训练（与 baseline 完全相同）

关键发现与约束：
- **纯文本混合失败**：直接用原始文本（不转为图像）混入训练导致 Video-MME 仅 55.8（vs Sparrow 56.7），Long 视频从 48.1 降至 47.7。Text-to-image 转换通过统一的 ViT 编码路径消除了 training-inference modality gap。
- **纯合成数据不可行**：TOPA/T3 的纯文本合成方案极易饱和甚至降级，合成数据只能作为正则化补充而非替代真实视频。
- **稠密采样帧无助于长视频**：48 帧 vs 24 帧训练无增益（短视频视觉冗余高），长上下文扩展需从 LLM backbone 层面解决（continue pretraining 扩展 context window）。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

## Image-LLM to Video-LLM Fine-tuning Paradigm（图像-LLM到视频-LLM微调范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Image-LLM to Video-LLM Fine-tuning 是当前开发视频-LLM 的两条主流路线之一：基于预训练的 Image-MLLM（通常已完成 vision-text 对齐预训练和 instruction tuning），通过额外的视频数据 fine-tuning 使其具备视频理解能力，无需从头进行 vision-text 对齐。区别于另一条路线（从 pretrained LLM 开始，先做 vision-text 对齐再做 video instruction tuning，如 VideoLLaMA 2、VITA），此路线利用 Image-LLM 中已内置的丰富视觉知识（来自大规模 image-text 数据的预训练），仅需较少的视频数据即可激活时序理解能力。Sparrow 论文使用的两个 base model 均属此范式：(1) Mini-InternVL-Chat-4B-V1.5（基于 InternLM2, 3.8B，支持最多 13 子图 patch，每子图 256 visual tokens）；(2) MiniCPM-Llama3-8B-V2.5（基于 LLaMA3-8B，最多 10 patch，每 patch 96 visual tokens）。InternVL 训练时冻结 vision encoder（保留预训练视觉知识），MiniCPM-8B 全量训练，lr=5e-6。训练时关闭动态分辨率 patchifying。关键发现：Image-LLM 的 zero-shot 视频理解能力已很强（InternVL-4B: Video-MME 52.5），甚至超过部分专用 Video-LLMs（VideoChat2 7B: 39.5），归功于大规模 image-text 预训练。Video fine-tuning 可在 zero-shot 基础上额外提升 ~3.8 points。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Image-LLM → Video-LLM Fine-tuning Pipeline ===
# 模型架构: ViT(vision_encoder) + MLP(projector) + LLM(backbone)
# 标准 MLLM 架构天然支持多帧——视频仅是多帧扩展

def video_finetune_step(image_llm, video_sample, config):
    """
    config:
      - InternVL: freeze_vit=True,  max_frames=64, tokens_per_frame=256
      - MiniCPM:  freeze_vit=False, max_frames=24, tokens_per_frame=96
    """
    # 1. 帧提取（FPS=1，超出则均匀降采样）
    frames = extract_frames(video_sample['video'], fps=1)
    if len(frames) > config.max_frames:
        frames = uniform_downsample(frames, config.max_frames)

    # 2. 逐帧视觉编码
    visual_tokens = []
    for frame in frames:
        if config.freeze_vit:
            with torch.no_grad():
                tokens = ViT(frame)  # [H_patch*W_patch, C]
        else:
            tokens = ViT(frame)
        visual_tokens.append(tokens)
    # visual_tokens: [T, num_tokens_per_frame, C]

    # 3. Projector 映射到 LLM embedding space
    visual_emb = Projector(visual_tokens)  # [T*num_tokens, d_llm]

    # 4. 与 text token 拼接 + 自回归生成
    text_emb = LLM.embed_tokens(tokenize(video_sample['instruction']))
    input_emb = concat([visual_emb, text_emb], dim=0)
    logits = LLM(input_emb)
    loss = -log P(answer | frames, instruction)

    return loss

# 训练数据: video-caption (ShareGemini 100K) + video-instruction (Video-ChatGPT 100K)
# 数据处理: 1:1 采样 video-caption 和 video-instruction
# GPU: 200K 全量 276.8 GPU hours，30K Sparrow 33.6 GPU hours
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现关键点：
1. **架构零修改**：Image-LLM 的 ViT+Projector+LLM 架构无需任何修改——视频仅是多帧输入，每帧独立通过 ViT 编码后按时间顺序拼接。这与多图 Image-LLM 推理完全一致。数据加载层将视频帧序列打包为 multiple images 即可。
2. **训练策略选择**：
   - 冻结 vision encoder（InternVL 方案）：利用预训练视觉知识，减少视频数据需求，降低训练成本。vision encoder 参数不更新，仅训练 projector + LLM。
   - 全量训练（MiniCPM 方案）：允许视觉特征适配视频域，获得更定制化的视频表示，但需要更多数据和计算。
3. **帧数与效率的平衡**：短视频为主的训练数据中，超出 24-64 帧可能引入冗余而非新信息（Sparrow 验证 48 帧无助于长视频理解）。关闭动态分辨率 patchifying 可固定每帧 token 数提升训练效率。
4. **评估**：Video-MME（短/中/长三段式评估）、MVBench（20 个视频任务）、TempCompass（时序理解）、LongVideoBench/MLVU（长视频）。评估方法推荐 exact matching + LLM matching 组合，因部分模型不严格遵守格式要求。
5. **Sparrow 增强**：在维持此范式不变的前提下，通过 text-to-image 数据增强改进数据质量（而非模型架构或训练协议），实现了 8.2× 训练效率提升。

Image-LLM zero-shot baseline 已超过部分专用 Video-LLM 的发现说明：视频理解的很大一部分基础能力（目标识别、OCR、场景理解）来自 image-text 预训练；视频 fine-tuning 主要注入时序/因果推理能力。这解释了为何 Instruction Diversity 比数据量更重要——因为模型真正从视频数据中学到的是"如何跨帧推理"，而非"如何理解单帧内容"（那已经在 image 预训练中完成了）。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

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

## Event-aware Temporal Agent（事件感知时序代理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Event-aware Temporal Agent 是 TSPO 中提出的轻量级可训练关键帧选择模块，负责从候选帧中概率化选择与文本查询最相关的关键帧子集。基于 CLIP-Large (400M 参数，冻结) 作为视觉和文本特征提取器，仅包含 3.5M 可学习参数（local window attention + MLP projector）。核心设计：(1) Event-aware 特征增强：通过 local window attention (窗口大小 w=12) 配合正弦位置编码捕获帧间时序依赖，学习"事件"级别的时序感知表示；(2) 双路径相似度融合：同时计算 event-level similarity (增强特征与文本) 和 frame-level similarity (原始特征与文本)，融合分数指导帧选择；(3) Gumbel-Softmax 概率化采样：输出帧选择概率和索引，支持 RL 探索。设计动机：现有 training-free keyframe selectors 使用大模型 (DINOv2-1B 或 MLLM-13B) 但无法针对下游 Video-MLLM 优化；Temporal Agent 保持轻量（3.5M vs 1B/13B）且可通过 GRPO 端到端训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Event-aware Temporal Agent 前向传播 ===
# 输入: V_c (T_c 帧候选视频), q (文本查询)
# 可训练参数: θ_agent = {W_q, W_k, W_v (local window attn), MLP_proj}
# 参数总量: 3.5M
# 冻结部分: CLIP-Large visual encoder, CLIP text encoder

# Step 1: CLIP 特征提取 (冻结, 无梯度)
F_f = CLIP_visual(V_c)          # [T_c, D], D 为 CLIP 特征维度
F_t = CLIP_text(q)              # [1, D]

# Step 2: 正弦位置编码 (注入时序信息)
PE = SinusoidalPositionalEncoding(T_c)  # [T_c, D]
F_f_pos = F_f + PE                      # [T_c, D]

# Step 3: Local Window Attention (事件感知)
# 每个帧 i 仅关注 [i-w/2, i+w/2] 范围内的帧
# w=12, 帮助学习局部事件依赖
F_e = LocalWindowAttention(
    Q = W_q @ F_f_pos,           # [T_c, d_k]
    K = W_k @ F_f_pos,           # [T_c, d_k]
    V = W_v @ F_f_pos,           # [T_c, d_v]
    window_size = 12,
    causal = False               # 双向窗口注意力
)                                # [T_c, D]

# Step 4: MLP 投影
F_e = MLP_proj(F_e)              # [T_c, D]

# Step 5: 双路径跨模态相似度
Sim_event = CosineSimilarity(F_e, F_t)   # [T_c], event-level
Sim_frame = CosineSimilarity(F_f, F_t)   # [T_c], frame-level
S = Sim_event + Sim_frame                # [T_c], 融合分数

# Step 6: Gumbel-Softmax 概率化采样
# 训练时: 加 Gumbel 噪声探索
γ ~ Gumbel(0, 1)                         # [T_c]
P = Softmax(S/τ + γ)                     # [T_c], 概率分布

# 推理时: 确定性采样 (无 Gumbel 噪声)
P = Softmax(S/τ)                          # [T_c]

# Step 7: TopK 选择
P_selected, I_selected = TopK(P, T_s)    # T_s 帧索引+概率
# log π_ts = Σ log P_selected[i]         # 用于 GRPO
```

Agent 的 RL 视角形式化：
- State: 输入长视频 V 和文本指令 Q
- Action: 关键帧选择，输出索引 I = {i_1,...,i_T_s} 和概率 P = {p_1,...,p_T_s}
- Policy: π_ts(V_s|q,V_c) = Π_{i=1}^{T_s} P_i(V_c, q)
- Reward: R_A + R_T (见 TSPO 术语)
- 优化: GRPO (仅更新 θ_agent)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/Hui-design/TSPO 中 src/open_tspo/ 目录下的模型实现。使用方式：(1) 下载预训练 TSPO-0.4B checkpoint (包含 CLIP-Large 权重 + 3.5M agent 参数)；(2) 通过 demo/llava_video_tspo.py 加载 agent + Video-MLLM backbone；(3) 推理流程：视频→1FPS 候选帧提取→CLIP 特征→Temporal Agent 选择 64 帧→Video-MLLM 生成回答。Agent 可跨 Video-MLLM 迁移：在 LLaVA-Video-7B 上训练的 agent 可直接用于 Qwen2.5VL-7B 等不同架构的 Video-MLLM。性能：仅 3.5M 可训练参数的 agent 使 LLaVA-Video-7B 在 LongVideoBench 提升 5.0% (58.9→63.9)，MLVU 提升 6.0% (70.3→76.3)。推理时帧提取仅需 1.2s (vs CoS 的 28.4s)。与类似模块的对比：LongVU 的 DINOv2-1B selector (1B 参数, training-free), CoS 的 MLLM-13B selector (13B, training-free), HORNet 的 MLP selector (<1M, GRPO-trained for Bernoulli mask), FrameVOYAGER 的 selector (training-based, random pre-processing sampling)。Temporal Agent 独特之处：轻量 (3.5M) + 可训练 (GRPO) + 事件感知 (local window attention) + 可迁移。

涉及论文标题：
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

## Direct Preference Optimization (DPO) / 直接偏好优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Direct Preference Optimization (DPO) 是 Rafailov et al. (2023, NeurIPS) 提出的一种用于对齐 LLM 输出与人类偏好的后训练方法。核心思想：将 RLHF 中两步流程（训练 reward model → RL 优化 policy）合并为单步直接优化。DPO 基于一个关键数学洞察——在 Bradley-Terry 偏好模型下，最优 policy π* 与 reward 函数 r 之间存在双射映射：r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)。利用此关系，可以直接在偏好数据上优化 policy，无需显式训练 reward model 或执行 RL。DPO 损失函数：L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]，其中 y_w 为 preferred response，y_l 为 dis-preferred response，π_ref 为冻结的参考模型（通常是 SFT 后的模型），β 为 KL 散度惩罚系数。σ 是 sigmoid 函数。损失函数直观含义：增大 preferred 与 dis-preferred 之间的相对对数概率差，同时 β 约束 policy 不偏离 π_ref 太远。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPO 训练 pipeline（基于 TPO 论文中的使用方式）：
```
输入: 偏好数据集 D = {(x, y_w, y_l)}, 参考模型 π_ref (SFT checkpoint)
输出: 对齐后的 policy π_θ

For each batch (x, y_w, y_l) in D:
    # 1. 前向传播计算 log-probabilities
    log_p_w = log π_θ(y_w | x)      # preferred 的对数概率
    log_p_l = log π_θ(y_l | x)      # dis-preferred 的对数概率
    log_p_w_ref = log π_ref(y_w | x)
    log_p_l_ref = log π_ref(y_l | x)

    # 2. 计算 log-ratio (implicit reward)
    ratio_w = log_p_w - log_p_w_ref
    ratio_l = log_p_l - log_p_l_ref

    # 3. DPO 损失
    L_DPO = -log σ(β * (ratio_w - ratio_l))

    # 4. 可选: SFT 辅助损失 (TPO 中使用)
    L_SFT = -log_p_w

    # 5. 联合损失
    L = L_DPO + α * L_SFT

    # 6. 反向传播
    θ ← θ - η * ∇_θ L
```
TPO 论文中 LongVA-TPO 使用 β=0.3, α=0.5, lr=4×10⁻⁶；LLaVA-Video-TPO 使用 β=0.2, α=1, lr=3×10⁻⁷。β 越大，policy 偏离 π_ref 的惩罚越重，训练越保守。训练 1 epoch，约 4 小时（8×A100, batch_size=64）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPO 的标准实现在 HuggingFace TRL 库中：`DPOTrainer` 类。使用流程：(1) 准备偏好数据集，每行包含 prompt/chosen/rejected 三个字段；(2) 加载 SFT checkpoint 作为 reference model（冻结，不参与梯度更新）；(3) 配置 β 超参数和训练参数；(4) 调用 DPOTrainer.train()。开源实现：TRL (https://github.com/huggingface/trl)。DPO 相比 RLHF+PPO 的优势：无需训练单独 reward model、无需 RL 算法（更稳定）、单步训练（更快）、内存开销更低（只需两份模型：policy + reference，而非 policy + reward + value + reference）。局限性：(1) 偏好数据质量要求高——噪声偏好对会直接误导优化方向；(2) offline 性质——无法像 online RLHF 一样从模型自身采样中学习；(3) 对 β 等超参数敏感。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models

## Reinforcement Learning from Human Feedback (RLHF) / 基于人类反馈的强化学习

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reinforcement Learning from Human Feedback (RLHF) 是一套将 LLM 输出与人类价值观和偏好对齐的训练范式，由 Ouyang et al. (2022, InstructGPT) 和 Ziegler et al. (2019) 确立。标准 RLHF 分三阶段：(1) Supervised Fine-Tuning (SFT) — 在高质量人工标注的 (prompt, response) 对上微调 base model；(2) Reward Model (RM) Training — 收集人类对同一 prompt 的多个模型输出的排序数据，基于 Bradley-Terry 模型训练 reward model r_φ，使得 P(y_w ≻ y_l) = σ(r_φ(x,y_w) - r_φ(x,y_l)) 最大化；(3) Reinforcement Learning — 使用 PPO 优化 policy π_θ，目标为 max E[r_φ(x,y)] - β·KL(π_θ || π_SFT)，KL 散度约束防止 policy 偏离 SFT 模型太远（避免 reward hacking）。RLHF 使 InstructGPT (1.3B) 在人类评估中优于 175B GPT-3，成为 ChatGPT 的核心对齐技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准 RLHF pipeline 伪代码：
```
# === Stage 1: SFT ===
For each (x, y_demo) in demonstration_data:
    L = -log π_θ(y_demo | x)
    θ ← θ - η * ∇L
π_SFT ← π_θ  # 保存 SFT 模型

# === Stage 2: Reward Model Training ===
For each (x, y_w, y_l) in comparison_data:  # y_w ≻ y_l
    r_w = r_φ(x, y_w)  # reward model 对 preferred 的评分
    r_l = r_φ(x, y_l)  # reward model 对 dis-preferred 的评分
    L = -log σ(r_w - r_l)  # Bradley-Terry 损失
    φ ← φ - η_r * ∇L

# === Stage 3: PPO Fine-tuning ===
π_θ ← π_SFT  # 初始化
For each x in prompt_data:
    y = sample(π_θ(x))      # 当前 policy 生成
    r = r_φ(x, y)           # reward model 评分
    KL_penalty = log π_θ(y|x) - log π_SFT(y|x)
    R = r - β * KL_penalty  # KL-正则化奖励
    θ ← PPO_update(θ, R)    # PPO policy gradient
```
DPO 简化了 RLHF：直接将 Stage 2+3 合并为一步优化，利用 reward-policy 双射关系 r = β log(π/π_ref) 隐式表示 reward，在偏好对数据上直接优化 policy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：TRL (https://github.com/huggingface/trl) 提供 PPOTrainer、RewardTrainer、DPOTrainer 完整 RLHF pipeline。主要变体：(1) RLAIF (RL from AI Feedback) — 用 AI（如 GPT-4）替代人类进行偏好标注，降低成本；(2) Online RLHF — 在训练过程中迭代收集新偏好数据，而非仅用固定 offline 数据集；(3) DPO/SimPO/KTO — 各种无需显式 reward model 的简化方案。TPO 论文定位在 RLHF 的 DPO 分支上，将偏好学习应用到视频时序理解领域。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models

## Video Large Multimodal Model (video-LMM) / 视频大型多模态模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Large Multimodal Model (video-LMM) 是将 LLM 从纯文本域扩展到视频域的模型架构。核心组成：(1) Visual Encoder（如 CLIP-ViT、SigLIP）— 将视频帧编码为视觉特征 token；(2) Multimodal Projector（通常是 MLP 或 cross-attention）— 将视觉特征映射到 LLM 的文本 embedding 空间；(3) LLM Backbone（如 LLaMA、Qwen）— 接收拼接后的视觉+文本 token 序列，自回归生成文本回答。video-LMM 与 image-LMM 的关键区别在于时间维度：需要处理多帧序列并建模帧间时序依赖。代表性开源模型包括 LongVA、LLaVA-Video、Video-LLaVA、NVILA、Apollo、Qwen2-VL 等。典型参数规模在 7B-72B，训练流程通常为两阶段：视觉-语言对齐预训练 → 视频指令微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
video-LMM 推理 pipeline（以 TPO 论文中使用的 LongVA-7B 为例）：
```
输入: 视频 V (时长 T 秒), 问题 Q (文本)
输出: 回答 A (文本)

# Step 1: 帧采样
F = uniform_sample(V, num_frames=128)  # 均匀采样 128 帧

# Step 2: 视觉编码 (frozen visual encoder)
for each frame f_i in F:
    v_i = VisualEncoder(f_i)  # CLIP-ViT → [N_patch, d_vis]

# Step 3: 投影到 LLM 空间 (multimodal projector)
h_vis_i = Projector(v_i)  # MLP → [N_proj, d_llm]

# Step 4: Token 拼接
H_vis = concat([h_vis_1, ..., h_vis_128])  # [128*N_proj, d_llm]
H_text = Tokenizer(Q)                       # [L_text, d_llm]
H_input = concat([H_vis, H_text])           # 视觉 token 在前

# Step 5: LLM 自回归生成
for t = 1..max_len:
    logits = LLM(H_input, A[:t-1])
    a_t = argmax / sample(logits[-1])
    if a_t == EOS: break
return A = [a_1, ..., a_t]
```
LongVA 的特点是通过语言到视觉的长上下文迁移技术支持 128 帧输入（通过扩展 LLM 的 RoPE position encoding）。LLaVA-Video 使用 96 帧（Video-MME）或 128 帧（其他 benchmark）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现分布在 HuggingFace model hub。主要评估工具：lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval)，支持 Video-MME、LongVideoBench、MLVU 等 benchmark 的统一评测。训练方面：通常使用 DeepSpeed ZeRO 或 FSDP 进行分布式训练；visual encoder 保持冻结以减少显存开销；multimodal projector 和 LLM backbone 进行 full fine-tuning 或 LoRA fine-tuning。TPO 论文中 full fine-tuning（language model + projector），8×A100 80GB，4 小时/模型。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models

## Temporal Preference Optimization (TPO) / 时序偏好优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Preference Optimization (TPO) 是 Stanford 团队提出的 video-LMM 后训练框架，通过操纵视频输入自动构建时序偏好数据，使用 DPO 训练增强模型的时序定位（temporal grounding）能力。核心创新：不需要人工标注时序标签——仅通过改变视频输入的"可见证据量"（完整帧/部分帧/不相关帧），让同一 video-LMM 对同一问题产生质量有差异的回答，自动形成 preferred vs dis-preferred 偏好对。TPO 框架分三步：(1) Temporal Preference Modeling — 采样视频帧集合 F，用 CogVLM2 生成逐帧 caption → GPT-4o-mini 基于 caption 生成问题 Q → 使用 Q + 完整帧 F 生成 preferred response r⁺ → 使用 Q + 不相关帧或部分帧生成 dis-preferred response r⁻；(2) LLM-based Post-Filtering — GPT-4o-mini 过滤三类噪声：r⁻ 优于 r⁺、r⁺ 事实错误、问题模糊；(3) DPO + SFT 联合训练 — 使用偏好数据对 (V, Q, r⁺, r⁻) 进行 DPO 优化。两种 dis-preferred 生成策略：(a) Irrelevant Information — 完全排除相关帧；(b) Incomplete Information — 仅用部分相关帧。最优数据混合比例为 5:5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TPO 完整 pipeline 伪代码：
```
输入: 视频集合, video-LMM π_θ, 参考模型 π_ref
输出: TPO 优化后的 video-LMM

# === Phase 1: 时序偏好数据生成 ===
For each video V:
    F = sample_frames(V)                   # 采样帧集合
    captions = CogVLM2.caption(each frame in F)
    Q_set = GPT-4o-mini(captions, task_prompts)

    For each q in Q_set:
        # Preferred: 使用完整相关帧
        r⁺ = video_LMM(V[F], q)

        # Dis-preferred (a): 不相关帧
        F_irr = sample(V \ F)
        r⁻_irr = video_LMM(V[F_irr], q)

        # Dis-preferred (b): 不完整帧
        F_inc = random_subset(F, ratio=0.5)
        r⁻_inc = video_LMM(V[F_inc], q)

        # Post-filtering (GPT-4o-mini, 3 条规则)
        for each (r⁺, r⁻) in [(r⁺, r⁻_irr), (r⁺, r⁻_inc)]:
            # Rule 1: r⁺ 是否优于 r⁻?
            # Rule 2: r⁺ 是否与 caption 矛盾?
            # Rule 3: r⁺ 基于 caption 是否正确的?
            if passes_all_rules:
                D.add((V, q, r⁺, r⁻))

# === Phase 2: DPO + SFT 训练 ===
For each batch (V, q, r⁺, r⁻) in D:
    # DPO loss (公式 2)
    L_DPO = -log σ(β · (log π_θ(r⁺|V,q)/π_ref(r⁺|V,q)
                      - log π_θ(r⁻|V,q)/π_ref(r⁻|V,q)))

    # SFT auxiliary loss (公式 3)
    L_SFT = -log π_θ(r⁺ | V, q)

    # Combined loss (公式 4)
    L = L_DPO + α · L_SFT

    θ ← θ - η · ∇_θ L  # full fine-tuning (LM+Projector)
```
超参数：LongVA-TPO: β=0.3, α=0.5, lr=4×10⁻⁶; LLaVA-Video-TPO: β=0.2, α=1, lr=3×10⁻⁷。8×A100 80GB, batch_size=64, 1 epoch, ~4h。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/ruili33/TPO。数据集和 checkpoint：https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10。使用流程：(1) 准备视频数据集（论文中手动 curator 200 关键词，爬取 8000 个互联网视频）；(2) 运行数据生成 pipeline（CogVLM2 captioning → GPT-4o-mini question generation → video-LMM response generation → post-filtering），生成 10K 偏好数据对；(3) 使用 DPO+SFT 联合损失训练（基于 TRL 或自定义脚本）；(4) 用 lmms-eval 评测 Video-MME/LongVideoBench/MLVU。消融发现：(a) TPO 随输入帧数增长性能持续提升（baseline 在 >64 帧退化），(b) 数据量 2K→10K 持续改善，(c) post-filtering 一致改善，(d) Incomplete:Irrelevant = 5:5 混合最优。关键优势：无需人工标注、可扩展（自动生成偏好数据）、即插即用（不修改推理架构）、可迁移到不同 video-LMM backbone。性能：LongVA-TPO 在 LongVideoBench +2.9%、MLVU +3.1%、Video-MME +2.5%。LLaVA-Video-TPO 成为 Video-MME 7B 模型 SOTA (71.5 w/ subs)。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models

## Test-Time Temporal Sampling (T3S / 测试时时间采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Test-Time Temporal Sampling (T3S) 是一种训练无关、即插即用的 MLLM 视频推理包装器。核心思想：不处理单条长视频 token 序列，而是生成 m 个短且多样化的视频子序列，将它们打包到单次前向传播中并行处理，最后通过 logit 聚合输出最终预测。整个流程分为三个阶段：(1) Multi-Trial Frame Sampling——对视频进行 m 次独立随机帧采样，每次抽取 N 帧；(2) Token Subsampling——每试次保留 αᵢ 比例的 visual token（默认使用均匀随机 patch 级采样）；(3) Multi-Subsequence Inference & Logit Aggregation——将 m 个子序列打包，使用块对角线 attention mask 确保子序列间不互相关注，推理后对各试次 logit 做均值、置信度加权或双试次交叉验证聚合。关键数学性质：self-attention 复杂度从 baseline 的 O(L²) 降为 O(∑αᵢ²L²)，当 ∑αᵢ² < 1 时获得理论加速。m=2、α₁=0.5、α₂=0.3 时 ∑αᵢ²=0.34，理论节省 66%，Qwen2.5-VL-7B 上实测加速 2.04× 同时准确率提升 3.1%（LongVideoBench）。与 FastV、AdaReTake 等 training-free 方法的核心区别：T3S 通过多试次随机采样的统计覆盖性补偿信息损失，而非依赖 attention score 的重要性排序。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === T3S Inference Pipeline ===
# 输入: 视频 V (F帧), 文本 tokens t
# 参数: N=256 (帧数/试次), m=2 (试次数), 
#        α₁=0.5, α₂=0.3 (token保留率), k=2 (top-k)

# Stage 1: Multi-Trial Frame Sampling
for i = 1 to m:
    P_i = RandomSample({1,...,F}, N)   # 随机选N帧索引
    V̂_i = V[P_i]                        # 提取子序列

# Stage 2: Vision Encoding + Token Subsampling
for i = 1 to m:
    v^(i) = VisionEncoder(V̂_i)          # |v^(i)| = L = N×M (每帧M个patch)
    idx = RandomSample({1,...,L}, ⌊αᵢL⌋)  # 均匀随机选token索引
    v̂^(i) = v^(i)[idx]                 # |v̂^(i)| = ⌊αᵢL⌋

# Stage 3: Multi-Subsequence Inference (单次前向传播)
# Pack: concat所有子序列 + block-diagonal attention mask
input_seq = concat(v̂^(1), t, v̂^(2), t)   # 或两个独立序列
{o₁, o₂} = MLLM.forward(input_seq, 
              attn_mask=BlockDiagonal(mask_size=[|v̂^(1)|+|t|, |v̂^(2)|+|t|]))
# 各子序列仅与自身tokens计算attention, 不跨子序列交互

# Stage 4: Logit Aggregation (Two-Trial Cross-Refinement, m=2)
K = TopK(o₁, k=2)                    # 试次1提出top-k候选
t* = argmax_{token∈K} o₂[token]      # 试次2在候选中重新排序

# 若 m>2, 使用均值聚合:
# o_avg = (1/m) Σ oᵢ; t* = argmax o_avg
```

时间复杂度分析：
- Baseline: O(L²) —— 单序列 self-attention
- T3S (m=2, packed): O((α₁²+α₂²)·L²) = O(0.34·L²)
- 实际打包后总序列长度 = (α₁+α₂)L，短于原始 L，进一步降低实际延迟

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/kaibinwang3/T3S。评估使用 VLMEvalKit 工具包，支持 Qwen2.5-VL-7B、LLaVA-Video-7B、Oryx-1.5-7B 等开源 MLLM。使用方式：(1) 加载预训练 MLLM；(2) 用 T3S wrapper 包裹模型推理接口，设置 m、N、αᵢ、k 参数；(3) 对每个视频采样 m 个子序列，调用一次包装后的 forward 获得聚合预测。超参数推荐：m=2（性价比最优，m>2 收益递减），α₁=0.5、α₂=0.3（平衡速度与准确率），k=2（对 k 值不敏感，2-100 范围内波动 <1%）。局限性：(1) 单 GPU 上各 chunk 计算已饱和，无法实现真正的序列级并行；(2) 每步生成 m 个不同 next-token 候选，随生成进行显存占用逐渐增加。代码已开源。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding

## Multi-Subsequence Inference with Sequence Packing (多子序列推理与序列打包)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Subsequence Inference 是 T3S 提出的推理范式：将 m 个独立采样的短视频子序列打包到一个 batch 中，通过单次 MLLM 前向传播完成推理。每一子序列包含随机采样的 N 帧视觉 token + 文本 prompt token。核心技术是 **Sequence Packing** + **Block-Diagonal Attention Mask**：各子序列在批次维度拼接，但 self-attention 计算时使用块对角线 mask——每个子序列仅与自身 token 交互，不跨子序列关注。这使得每个 attention 块的实际计算量由其子序列长度（αᵢL）决定，而非所有子序列总长度，从而将总复杂度从 O(L²) 降为 O(∑αᵢ²L²)。在 GPU 上的实际效果：打包后总 token 数 = (α₁+α₂)L < L，且每个 attention 块更小，KV cache 占用更少。与传统的 batching（多个请求独立前向）不同，T3S 的 packing 将多个子序列合并到同一 batch 中共享一次前向传播，避免了多次 kernel launch 开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Multi-Subsequence Inference with Sequence Packing ===

# 假设 m=2, L=1000 (原始视觉token数)
# α₁=0.5 → |v̂^(1)| = 500, α₂=0.3 → |v̂^(2)| = 300
# 文本 tokens |t| = 50

# 打包方案1: 将纹理子序列拼接
# Packed sequence: [v̂^(1)_1, ..., v̂^(1)_500, t_1, ..., t_50, 
#                    v̂^(2)_1, ..., v̂^(2)_300, t_1, ..., t_50]
# 总长度 = 500+50+300+50 = 900

# Block-Diagonal Attention Mask 构造:
# mask[i][j] = 
#     1  if i,j 属于同一子序列-文本块 (0 <= i,j < 550 或 550 <= i,j < 900)
#     0  otherwise (跨块不关注)
#
# 效果: QK^T 计算中，block 1 矩阵 [550×550], block 2 矩阵 [350×350]
#       总参数量 = 550² + 350², 而非 900² = 810000
#       实际节省: 1 - (302500+122500)/810000 = 47.5%

# 打包方案2: 文本tokens共享 (论文实际做法)
# 每个子序列独立拼接文本, 等价于独立推理, 
# 但通过 batch 维度并行处理
seq_1 = concat(v̂^(1), t)  # [500+50]
seq_2 = concat(v̂^(2), t)  # [300+50]
# Batch forward: MLLM([seq_1, seq_2], attn_mask=BlockDiagonal([550, 350]))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
T3S 开源实现使用 PyTorch 的 block-diagonal attention mask 实现。关键超参：m 通常设为 2（m=3/4 收益递减且计算成本递增），αᵢ 的选择需满足 ∑αᵢ² < 1 以获得理论加速。使用方式：在推理时配置 m/N/αᵢ/k 参数，调用 T3S wrapper 的 forward 方法，替代原有的单序列推理。注意：当 α₁+α₂ ≥ 1.3 时，打包后的长序列可能导致 OOM，此时需回退到两个串行前向传播。局限性：(1) 单 GPU 上序列间无真正的并行——chunk 计算已使 GPU 饱和；(2) 多 GPU 设置下可将各子序列分配到不同设备独立推理。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding

## Multi-Trial Logit Aggregation / Cross-Refinement (多试次Logit聚合与交叉验证)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Trial Logit Aggregation 是 T3S 的输出融合策略，用于将 m 个独立视频子序列推理得到的 m 个 logit 向量聚合为最终预测。论文提出三种聚合策略：(A) Mean Logits——直接对各试次 logit 取均值，参数无关且可靠性高；(B) Confidence-Weighted Aggregation——根据各试次预测分布的逆熵加权（低熵=高置信度=高权重）；(C) Two-Trial Cross-Refinement (m=2)——非对称验证方案：试次 1 提出 top-k 候选 token 集合，试次 2 在候选集上重新排序选出最优。方法 (C) 被证明在 m=2 时最优：试次 1 的采样保留率（α₁=0.5）高于试次 2（α₂=0.3），试次 1 拥有更多视觉信息适合提出候选，试次 2 用更稀疏但不同的视角验证。消融实验表明 Two-Trial Cross-Refinement 在所有 benchmark 上均优于 Mean Logits 和 Confidence-Weighted（VideoMME: 65.2 vs 65.1 vs 64.7; LongVideoBench: 62.3 vs 62.0 vs 61.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === 三种Logit聚合策略 ===
# 输入: logits o₁, o₂, ..., oₘ ∈ R^D (D=词表大小)

# (A) Mean Logits (默认, m任意)
def mean_aggregation(o_list):
    o_avg = sum(o_list) / len(o_list)
    return argmax(o_avg)

# (B) Confidence-Weighted (m任意)
def confidence_weighted(o_list):
    weights = []
    for o_i in o_list:
        pi = softmax(o_i)                    # 预测概率分布
        H_i = -sum(pi * log(pi))              # 熵 (越低越确定)
        weights.append(1.0 / H_i)             # 逆熵权重
    weights = normalize(weights)
    o_weighted = sum(w_i * o_i for w_i, o_i in zip(weights, o_list))
    return argmax(o_weighted)

# (C) Two-Trial Cross-Refinement (m=2, k=2)
def cross_refinement(o1, o2, k=2):
    # 第一阶段: 试次1 (α₁=0.5, 信息更全)提出top-k候选
    K = argsort(o1, descending=True)[:k]      # TopK(o₁, k)
    # 第二阶段: 试次2 (α₂=0.3, 但覆盖不同帧)重新排序
    t_star = argmax_{t∈K} o2[t]
    return t_star

# 直觉: 试次1做"生成"(宽覆盖), 试次2做"验证"(不同视角检验)
# 消融结果(Table 6): Two-Trial > Mean > Confidence-Weighted
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现中，聚合策略通过参数配置选择。推荐使用 Two-Trial Cross-Refinement（m=2 时）或 Mean Logits（m>2 时）。Confidence-Weighted 在 MLVU M-Avg 上表现最好（69.5），但整体略逊于 Cross-Refinement。Top-k 参数 k 对性能不敏感（2-100 范围波动 <1%），论文推荐 k=2 以获得最高的推理效率。在自回归生成中，每一步都需要执行 logit 聚合以决定下一个 token，因此聚合策略的计算开销必须极低（O(m·D)），T3S 的三种策略均满足此要求。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding

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

## VTG (Video Temporal Grounding，视频时间定位)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Temporal Grounding (VTG) 是视频理解领域的核心任务之一：给定一段未裁剪视频 v 和一个自然语言查询 q（描述某个特定事件），模型需输出该事件在视频中发生的具体时间片段 S = (t_start, t_end)。形式化定义：输入 (v, q)，输出 S = (t_start, t_end) 使得视频在 [t_start, t_end] 区间内的视觉内容与查询 q 的描述语义匹配。VTG 的核心难点：(1) 需要细粒度的时序感知能力，而非粗粒度的语义聚合；(2) 需要建模长时序视觉动态（appearance-centric features 难以标注和学习）；(3) 需要精确的边界定位（start/end boundary precision）。评估指标：R1@m（top-1 预测与 ground-truth 的 IoU 超过阈值 m 的比例，m ∈ {0.3, 0.5, 0.7}）和 mIoU（所有测试样本的平均 IoU）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTG 的 MLLM-based 处理流程（TimeLens 的 thinking-free RLVR paradigm）：
```
# Input: video v, query q (e.g., "When does the person turn off the light?")
# Output: timestamps (t_start, t_end)

# Step 1: 视频帧采样与时间戳编码
frames = sample_frames(v, fps=2)  # T frames
for i, frame in enumerate(frames):
    t_i = i / 2.0  # 绝对时间（秒）
    text_token = tokenizer(f"{t_i:.1f}s")  # e.g. "10.2s"
    visual_token = vision_encoder(frame)  # frozen ViT
    # Interleaved: timestamp before visual
    sequence.append([text_token, visual_token])

# Step 2: 追加 prompt 和 query
sequence = [prompt_tokens, query_tokens] + sequence

# Step 3: MLLM 生成时间片段
output = LLM(sequence)  # autoregressive generate
# Output: "The event happens in 5.2 - 12.7 seconds"
(t_start, t_end) = parse(output)

# Step 4: 评估（仅在训练/测试时）
IoU = intersection(t_start, t_end, t*_start, t*_end) 
    / union(t_start, t_end, t*_start, t*_end)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTG 的主流实现方式经历了三代演进：(1) 传统方法：基于 proposal-based 或 proposal-free 的专用 VTG 模型（如 2D-TAN、Moment-DETR），使用预提取的视频特征和文本特征，通过 proposal ranking 或 span prediction 定位时间边界；(2) MLLM-based 方法：利用预训练 MLLM（Qwen2.5-VL、InternVideo2 等）的多模态理解能力，通过 timestamp encoding 将时间信息注入模型，使用 SFT 或 RLVR 进行后训练优化（TimeLens、Time-R1、TRACE 等）；(3) 自监督方法：VideoSSR 等通过自监督 pretext tasks（anomaly grounding、temporal jigsaw）无需人工标注即可提升 VTG 能力。TimeLens 的核心贡献：(a) 数据质量保证（TimeLens-Bench 手动重标注 + TimeLens-100K 自动化重标注）；(b) thinking-free RLVR 训练范式；(c) interleaved textual timestamp encoding。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs

## MRoPE (Multimodal Rotary Position Embedding，多模态旋转位置嵌入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MRoPE (Multimodal Rotary Position Embedding) 是 Qwen2.5-VL 提出的多模态位置编码方案，将传统 1D RoPE 扩展为 3D 编码，同时处理文本（1D）、图像（2D）和视频（3D）的位置信息。核心设计：将 hidden dimension per head 拆分为三个 section —— 时间（T, temporal）、高度（H, height）、宽度（W, width），分别用不同的 position ID 进行旋转编码。在 Qwen2.5-VL 中，mrope_section 典型配置为 [16, 24, 24]（head dim=80）。对各模态：(1) Text：三个分量使用相同 position ID；(2) Image：T 分量恒为常数，H/W 分量使用 patch 的 2D 坐标；(3) Video：T 分量按帧递增（Qwen2.5-VL 进一步将 T 分量与绝对时间对齐，使用 second_per_grid_t 参数），H/W 同 image。注意：Qwen2.5-VL 的 ViT encoder 仅使用 2D RoPE（H+W），3D MRoPE 仅应用于 LLM backbone 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MRoPE 的 position ID 计算（以 video 为例，基于 Qwen2.5-VL 文档）：
```
# Qwen2.5-VL MRoPE position ID assignment
# Video: T frames, each frame H_patches × W_patches

second_per_grid_t = 1 / (fps * tokens_per_second)  # 默认定时分辨率
# tokens_per_second default=2

for frame_idx in range(T):
    # Temporal ID: 绝对时间对齐
    t_pos = int(frame_idx * second_per_grid_t * fps)
    for h in range(H_patches):
        for w in range(W_patches):
            pos_id = (t_pos, h, w)  # (T, H, W)
            # 分别用三组频率进行 RoPE rotation
            # dim[0:16] rotated by t_pos
            # dim[16:40] rotated by h
            # dim[40:64] rotated by w
```

在 TimeLens 论文中，MRoPE 作为 timestamp encoding 的 baseline 方案被评估。TimeLens 发现 MRoPE 在 VTG 任务上表现不理想（Charades-TimeLens mIoU 仅 36.6 vs Interleaved Textual 的 48.3），原因可能是：(1) 需要对 LLM 的 RoPE 机制进行底层修改，难以在大规模重训中实用化；(2) position embedding 方式的时间感知精度不如显式文本时间戳。TimeLens 通过将每帧作为独立 image 处理并复制两份，完全绕过 MRoPE 机制实现公平消融对比。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MRoPE 在 Qwen2.5-VL / Qwen3-VL 中直接在 LLM attention 层实现：在计算 query-key 点积之前，根据每个 token 的 (t_pos, h_pos, w_pos) 三组 position ID 分别应用不同频率的 RoPE rotation，然后拼接。HuggingFace Transformers 中的实现通过 `apply_multimodal_rotary_pos_emb` 函数完成。使用时只需配置 `mrope_section` 参数和 position ID tensor，模型自动处理。TimeLens 发现绕过 MRoPE、使用 interleaved textual timestamp encoding 既简单又有效，在 VTG 任务上无需修改 LLM 底层结构。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

**VideoRoPE 论文对 MRoPE 缺陷的分析**：VideoRoPE 识别出 MRoPE 的三个关键局限：(1) **频率分配问题**：MRoPE 将高频率（低维度 dims 0-31）分配给 temporal 维度 t，而高频率对应的旋转角 θ_n = β^{-2n/d} 具有短单调区间，cos(θ_n·t) 在远距离上周期性重复。当帧号从 0 到 3000 时，低维的 cos(θ_n·t) 多次经过零点产生"hash collision"——距离很远的位置有几乎相同的 temporal embedding，导致 V-NIAH-D 中的 distractor 帧可在 temporal 维度伪装为 needle 帧。(2) **空间非对称性**：每帧 visual token 从 (0,0) 到 (W-1,H-1) 排列，每帧最后 token 总停在 (W-1,H-1) 角形成"corner stack"，text-video 边界距离非对称。(3) **缺乏时间缩放**：所有维度 index increment=1，无法区分帧间时间间距和图像内空间间距的不同粒度。VideoRoPE 通过 LTA（低频时间分配）、DL（对角线布局）和 ATS（可调时间缩放）系统性解决这三个问题。

## Interleaved Textual Timestamp Encoding（交错文本时间戳编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 是一种将视频帧的时间信息注入 MLLM 的编码策略。核心方法：将每帧的绝对时间戳（如 "10.2s"）通过 LLM 的 text tokenizer 转换为文本 token，然后交错插入到对应帧的 visual tokens 之前，形成 "timestamp → frame" 的交替序列。这与两种替代方案形成对比：(1) Non-interleaved textual encoding：将所有时间信息放在 prompt 开头一次性声明（如 "This video samples N frames at t1, t2, ... seconds"）；(2) Visual overlay：将时间戳作为 OCR-able 文本直接渲染到帧图像上。TimeLens 的实验证明，interleaved textual encoding + raw timestamps 在所有方案中效果最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 的 token 序列构建流程：
```
# 输入: T frames, fps, query q
# 每帧作为独立 image 处理（绕过 frame merge），以便插入 text tokens

tokens = [system_prompt_tokens, query_tokens]
for i in range(T):
    t = i / fps  # 绝对秒数
    # 保留一位小数，如 "10.2s"
    timestamp_text = f"{t:.1f}s"
    # 通过 LLM text tokenizer 获取文本 token embeddings
    text_tokens = tokenizer.encode(timestamp_text)
    # Vision encoder 提取 frame visual tokens (frozen)
    # frame 被复制两份绕过 Qwen2.5-VL 的相邻帧 merge
    frame_copy1 = frame
    frame_copy2 = frame
    visual_tokens = vision_encoder(concat(frame_copy1, frame_copy2))
    # 交错: timestamp 在 visual 之前
    tokens.extend(text_tokens)
    tokens.extend(visual_tokens)

# 最终序列: [prompt, query, "0.0s", frame_0, "0.5s", frame_1, ...]
output = LLM(tokens)
```

TimeLens 消融结果（Charades-TimeLens mIoU）：
| Timestamp Encoding | Frame Index | Raw Timestamp |
|---|---|---|
| Position Embed. (MRoPE) | - | 36.6 |
| Visual Overlay | 44.0 | 46.3 |
| Non-Interleaved Textual | - | 45.8 |
| Interleaved Textual | 45.6 | **48.3** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 的实现需处理两个关键细节：(1) 对于 Qwen2.5-VL 等默认将相邻两帧 merge 的架构，需将每帧复制为两份作为独立 image 处理，以在帧间插入 text token（同时计算量等同 2 FPS 的原始实现）；(2) 时间戳格式选择：raw timestamp（如 "10.2s"）优于 frame index（如 "1, 2, 3"），因为后者忽略了帧间时间间隔（非均匀采样时尤其重要）。该方法无需修改 LLM 的 RoPE 机制，完全依赖 MLLM 已有的文本理解能力来感知时间，实现简单且有效。在推理时，只需在帧输入前拼接时间文本 token 即可，无额外计算开销。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs

## Thinking-free RLVR（无思考过程的强化学习训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Thinking-free RLVR 是 TimeLens 提出的针对感知主导型任务的强化学习训练范式。与传统 "think-then-answer" RLVR（如 DeepSeek-R1 式：模型先生成思考过程 y_thinking，再生成答案 y_answer，奖励 = accuracy + format）不同，thinking-free RLVR 让模型直接输出答案，跳过显式思考过程。奖励函数简化为仅包含任务准确度：`r(y) = r_acc(y) = IoU(Ŝ, S*)`，无需 format reward。TimeLens 证明在 VTG 这类感知主导型任务上：(1) thinking-free RLVR 性能 > thinking-based RLVR（Charades-TimeLens mIoU: 48.3 vs 42.7）；(2) 训练效率更高（1.0× vs 1.9× 训练时间）；(3) 推理更快（无需生成 thinking tokens）。论文观察到 thinking-based RLVR 在训练过程中 thinking 长度逐渐收敛至简单内容，表明模型学会了 bypass 无益的显式推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Thinking-free vs Thinking-based RLVR 对比：
```
# Thinking-based RLVR (DeepSeek-R1 style)
y = [y_thinking, y_answer]  # 两部分
r(y) = r_acc(y_answer) + r_format(y)  # 格式奖励必需
# y_thinking: "Let me analyze the video frames. At 5.2s, 
#  I observe the person reaching for the light switch..."
# y_answer: "The event happens in 5.2 - 12.7 seconds."

# Thinking-free RLVR (TimeLens)
y = y_answer  # 直接输出答案
r(y) = IoU(Ŝ, S*)  # 仅用 IoU，无需格式奖励
# y: "The event happens in 5.2 - 12.7 seconds."
```

TimeLens 中 thinking-free RLVR 的关键实践：
- Vision encoder frozen（节省显存）
- 其余参数可训练（LLM backbone + projector）
- 8×H20 GPU，batch size=8，每 prompt 8 rollouts
- lr=1×10⁻⁶，KL coefficient β=0
- Early stopping 当 reward plateau 时（~310 steps / ~2.5K samples）
- 搭配 difficulty-aware Gaussian sampling 选择训练数据

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Thinking-free RLVR 的实现简化了训练和推理流程：(1) 训练时：prompt 和 system instruction 不包含任何 "think step-by-step"或 "reason first" 指令，模型直接从输入生成最终答案；(2) 奖励计算：仅需一个确定性的、可编程验证的 accuracy reward（如 IoU、exact match 等）；(3) 推理时：生成的 token 数量大幅减少（无 thinking tokens），降低 latency 和 serving cost。适用场景：任务以感知/定位为核心（而非复杂推理），如 video temporal grounding、object counting、spatial localization 等。不适用场景：需要多步推理的复杂 QA、数学证明、代码生成等。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs

## Difficulty-aware Data Sampling for RLVR（面向RLVR的困难感知数据采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Difficulty-aware Data Sampling 是 RLVR 训练中的一种数据筛选策略：根据当前模型对训练样本的掌握程度，优先选择模型尚未掌握的困难样本进行训练。在 TimeLens 中实现为：先用待训练模型对全部训练数据进行 offline inference，计算每个样本的 difficulty score `d_i = 1 - IoU(Ŝ_i, S*_i)`（d_i 越高表示越困难），然后以 Gaussian 分布 `g(d; μ, σ²)` 为目标进行 weighted sampling。为了确保采样后的 difficulty 分布符合目标 Gaussian 而非被原始数据分布 bias，使用 density-corrected weight：`w_i = g(d_i; μ, σ²) / p̂(d_i)`，其中 `p̂(d_i)` 是原始数据中 difficulty d_i 的经验密度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TimeLens 中 difficulty-aware sampling 的完整流程：
```
# Step 1: Offline inference — 评估每个样本的 difficulty
for each (v_i, q_i, S*_i) in D_train:  # D_train = TimeLens-100K
    Ŝ_i = π_θ(v_i, q_i)  # 用待训练模型做推理
    d_i = 1 - IoU(Ŝ_i, S*_i)  # difficulty ∈ [0, 1]

# Step 2: 估计原始数据的 difficulty 经验密度 p̂(d)
# 使用直方图或 KDE 估计 p̂(d_i) for each i

# Step 3: 计算 density-corrected sampling weight
# 目标: sample difficulty ~ Gaussian(μ=0.05, σ=0.2)
# 即 prefer d_i ≈ 0.05 (IoU ≈ 0.95 的极困难样本)
for each i:
    g_i = (1 / sqrt(2πσ²)) * exp(-(d_i - μ)² / (2σ²))
    w_i = g_i / p̂(d_i)  # density correction

# Step 4: 按权重采样 ~12K 样本
D_sampled = weighted_sample(D_train, w_i, size=12000)

# Step 5: 在采样数据上做 RLVR 训练
train_GRPO(D_sampled)
```

TimeLens 通过改变 μ 值实验显示（Fig. 7）：样本难度越高（μ 越小 → d_i 越小 → IoU 越高），模型性能越好，直到难度极高时性能趋于 plateau（IoU > 0.75）。最优配置：μ=0.05, σ=0.2，即优先选择 difficulty 接近 0.05（高度困难）的样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Offline inference 需在 RLVR 训练前完成，因对全量训练数据做推理耗时较长，需计入总训练时间；(2) Density correction 是关键步骤：由于原始数据中困难样本天然偏少，直接按 Gaussian weight 采样可能导致实际采样到的仍是大量容易样本，density correction 确保采样后的 difficulty 分布跟随目标 Gaussian；(3) 该策略在 TimeLens 中贡献了显著的性能增益（Fig. 2b 中 "Difficulty Sampling" 为最终性能提升的关键组件之一）；(4) 不同于 curriculum learning（从易到难），difficulty-aware sampling 直接采样最困难的样本进行高效训练。类似策略也见于 GLM-4.1-V-Thinking、VL-Cogito 等工作。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs

## GRPO-CSV (GRPO with Completeness Self-Verification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO-CSV 是 TimeSearch-R 论文提出的一种改进 GRPO 强化学习算法，专门为长视频时序搜索任务设计。在标准 GRPO 的 outcome-only reward 基础上增加 **Completeness Self-Verification (CSV, 完备性自验证)** 阶段：在 GRPO rollout 中 policy model π_θ 生成 text-video 交错 CoT C 和最终答案 A 后，CSV 提取 C 中所有搜索到的视频帧构成动态帧集 V_c，用同一模型仅基于 V_c 重新回答问题（禁止新搜索），得到 CSV 答案 A_c。Completeness Reward: R_c = 1[Acc(A, A*) > 0.5] · Acc(A_c, A*)，仅当原始答案 A 正确时才施加 CSV reward。总奖励：R = R_c + R_fmt + R_acc。GRPO-CSV 解决标准 GRPO 的两个失败模式：(1) 搜索不充分 —— outcome-only reward 无中间搜索监督，模型可能凭部分证据或语言偏置答对而缺乏视觉 grounding；(2) 推理不一致 —— 中间推理过程可能与最终答案脱节。Ablation 显示移除 CSV 使 completeness 从 60.5% 降至 57.2%，temporal F1 从 7.8 降至 7.4，且训练约 300 step 崩塌（模型停止搜索）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GRPO-CSV 训练流程伪代码：
```
# 超参数: K=8 rollouts, β=0.005 KL coeff, lr=1e-6 AdamW
# 平台: TRL + DeepSpeed ZeRO-3 + vLLM colocate

for step in range(num_rl_steps):
    for (V, Q, A*) in batch:  # batch_size=4, grad_accum=2
        # ===== GRPO Rollout (vLLM colocate) =====
        for k in range(K):  # K=8
            C_k, A_k = π_old.interleaved_reasoning(V, Q)
        # ===== Reward Computation =====
        for k in range(K):
            R_acc[k] = 1 if A_k == A* else 0
            R_fmt[k] = 1 if valid_format(C_k, A_k) else 0
            # ===== CSV Rollout =====
            V_c = extract_all_frames(C_k)  # 收集搜索到的帧
            A_c = π_old.answer_no_search(Q, V_c)  # 禁止工具
            if Acc(A_k, A*) > 0.5:  # 仅正确轨迹
                R_c[k] = Acc(A_c, A*)
            else:
                R_c[k] = 0
            R[k] = R_c[k] + R_fmt[k] + R_acc[k]
        # ===== GRPO Update =====
        baseline = mean(R[1..K])
        for k in range(K):
            A_adv[k] = R[k] - baseline
        loss = -Σ min(r_t·A_adv, clip(r_t,1-ε,1+ε)·A_adv)
               + β·KL(π_θ||π_ref)
        optimizer.step(loss)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 基于 TRL (Transformer Reinforcement Learning) 库构建，利用 vLLM colocate 模式在训练 GPU 上同时做 rollout 推理；(2) CSV prompt 与主推理 prompt 不同：要求简短回答且可输出 "I don't know"，tools 被移除防止新搜索；(3) 仅对正确轨迹施加 CSV reward 是关键 —— 避免模型学习低质量搜索策略；(4) SFT cold-start 阶段用 GPT-4o 生成交错 CoT 数据，mask 视频 token 梯度。适用场景：需要中间步骤监督但缺乏 process annotation 的 multi-turn tool-calling RL 训练（视频搜索、网页搜索、代码搜索等）。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

## Completeness Self-Verification (CSV) in Reinforcement Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Completeness Self-Verification (CSV, 完备性自验证) 是一种 annotation-free 的中间步骤监督机制，作为 GRPO outcome-only reward 的补充。核心理念：让 policy model 自己验证中间搜索步骤是否充分 —— 搜索到的帧是否包含足够信息正确回答问题。CSV 通过将搜索帧集单独取出、禁用新搜索、要求模型重新回答，间接评估搜索质量。CSV 是弱监督过程奖励 (weakly-supervised process reward) 的一种形式：不标注"哪些帧应被搜索到"，而是通过结果（重新回答的正确性）反向验证过程的充分性。关键设计：(1) 禁止工具调用（CSV prompt 不含 tools），(2) 允许 "I don't know"（诚实承认证据不足），(3) 条件奖励（仅正确轨迹施加，避免强化错误搜索策略）。

从算法pipeline角度拆解术语：
```
# CSV 执行流程
# 输入: C (交错CoT), Q (问题), π_θ (policy model)
V_c = ∪{V_i for (T_i, V_i) in C}  # 提取所有搜索帧+时间戳

# CSV prompt (与主推理 prompt 完全不同)
csv_prompt = "You are a helpful assistant. Please answer visual 
  questions as briefly as possible. When you don't have enough 
  visual information, please say 'I don't know'."

csv_input = concat(csv_prompt, V_c_frames_with_timestamps, Q)

# CSV 推理: π_θ 仅基于 V_c 回答，禁止搜索
A_c = π_θ.generate(csv_input, blocked_tokens=[<tool_call>])

# CSV reward: 仅原始答案正确时计算
R_c = 1[Acc(A, A*) > 0.5] · Acc(A_c, A*)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CSV 不需要额外的 reward model —— 复用 policy model 自身做重新回答，保持训练简洁。CSV 仅在 RL 训练阶段使用，推理时无需额外的 CSV forward pass。Ablation 显示：无 CSV 时 GRPO 训练约 300 step 崩塌（模型停止搜索，completeness 降为零）；CSV + accuracy reward 组合实现最佳 QA（VideoMME 66.6%）。适用场景：任何 multi-turn tool-calling RL 训练，其中中间检索/搜索步骤质量难以直接评估。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

## Interleaved Text-Video Thinking（文本-视频交错思维）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Text-Video Thinking 是一种长视频推理范式，将时序搜索重新定义为文本推理与视频片段检索交错的思维过程。与传统 "先选帧后推理" 不同，模型在每个推理步 k 生成文本推理 T_k；若含搜索指令，video environment 检索视频片段 V_k 追加到 CoT，形成 "thinking → searching → thinking → ..." 循环直至输出答案或达到预算（最多 8 轮搜索/8 帧）。形式化：C_k ≜ {(T_1,V_1), ..., (T_k,V_k)}，整个推理链分解为 P_θ(A,C|Ṽ,Q) = P_θ(C|Ṽ,Q) · P_θ(A|C,Ṽ,Q)（时序搜索概率 × 答案预测概率）。该范式是 "Thinking with Images" 向长视频域的扩展（空间搜索→时间搜索），使模型从数据中端到端学习最优搜索策略。

从算法pipeline角度拆解术语，给出具体例子。
推理格式示例：
```
<think>The video shows a living room. I need to find when cooking 
starts. Let me search around 120s-300s.</think>
<tool_call>{"name":"seek_video_frames","arguments":{"query":"person 
cooking in kitchen","start_time":120,"end_time":300,"num_frames":8}}
</tool_call>
[8 frames at: 155.2s, 178.6s, 203.1s, ...]  ← 搜索返回
<think>Frames at 178.6s-224.8s show vegetable chopping. Let me 
check for earlier steps.</think>
<tool_call>...</tool_call>
[more frames]
<think>Preparation starts at 178.6s, cooking at 245.0s.</think>
<answer>B</answer>
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 两阶段训练：SFT (GPT-4o 生成交错 CoT) → RL (GRPO-CSV)；(2) 搜索用 SigLIP-400M 编码帧/query + DPP 选择帧；(3) 帧带绝对时间戳 token ("12.3s") 保持时间定位；(4) 与 "Interleaved Multimodal Reasoning" (Mirage) 区别：TimeSearch-R 处理显式视频帧检索，Mirage 处理隐式 latent token 生成。适用场景：长视频理解、视频问答、视频动作定位。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

## Determinantal Point Process (DPP) for Video Frame Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DPP (Determinantal Point Process, 行列式点过程) 是一种集合子集概率模型，选择概率 P(S) ∝ det(L_S)，其中 L 是 kernel matrix。在 TimeSearch-R 中 DPP 用作时序搜索的核心优化：在 [t_s, t_e] 内从 N 候选帧选 F≤8 帧，同时优化帧的 query relevance 和 interset diversity。Kernel 构造：L̃ = diag(r) · S · diag(r)，L̃_ij = r_i · r_j · h_i^T · h_j。r_i = SigLIP text-image 相关性分数 min-max 归一化到 [0,1]，S 是帧间 cosine similarity。DPP 的行列式编码质量和多样性权衡：高质量帧 (r_i→1) 优先被选，但相似帧被同时选中会降低行列式值被惩罚。TimeSearch-R 使用 fast greedy MAP inference (Chen et al., 2018) 近似求解 argmax det(L̃_S)，O(N·F²)。

从算法pipeline角度拆解：
```
# DPP 帧选择流程
h_i = SigLIP.encode(v_i) ∀ v_i ∈ F_cand      # [N, d]
q_emb = SigLIP.encode_text(query)             # [d]
S_ij = h_i^T · h_j                             # 帧间相似性
r_i = norm(q_emb^T · h_i)                      # query 相关性 [0,1]
L̃_ij = r_i · r_j · S_ij                       # DPP kernel
V* = greedy_MAP(L̃, F)                          # 选 F 帧最大化 det
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) SigLIP-400M 提供帧/文本嵌入；(2) Greedy MAP 是近似算法（精确 MAP 为 NP-hard），实践中效率和质量充分；(3) 比 top-K by relevance 优势：避免选连续冗余帧，DPP 通过多样性惩罚覆盖不同内容。广泛用于信息检索、推荐系统、文档摘要、视频关键帧提取。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

## TransV (Token Information Transfer via Gated Cross-Attention) / 基于门控交叉注意力的Token信息转移

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TransV 是 TimeViper 提出的 LLM 内部视觉 token 压缩模块，通过 Gated Cross-Attention 将冗余视觉 token 的信息显式转移到指令 token 中，再丢弃原始视觉 token。核心思想：不同于传统的 token dropping（不可逆丢失信息）或 token compression into new special tokens（破坏 token 身份），TransV 先通过 cross-attention 将被丢弃的 vision tokens 作为 KV、instruction tokens 作为 Q 计算信息增量，再通过门控因子 tanh(α_l) 控制转移强度，将信息"存"进 instruction tokens，最后安全地丢弃 vision tokens。TransV 增加约 100M 参数（相对于 9B backbone 约 1.1%），学习率 5e-5（高于 LLM backbone 的 1e-5）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TransV at layer l: compress and transfer vision tokens -> instruction tokens
def TransV(X_0, X_1, l):  # X_0: vision tokens [T_0, D], X_1: instruction tokens [T_1, D]
    # Step 1: Token dropping strategy
    if l == shallow_layer:  # e.g., layer 7
        X_0_kept, X_0_dropped = UniformDrop(X_0, rate=0.5)
    elif l == deep_layer:   # e.g., layer 39
        attn_scores = Attention(X_1[-1], X_0)  # last inst token as query
        keep_ids = TopK(X_0, score=-attn_scores, k=T_0 * 0.1)
        X_0_kept = X_0[keep_ids]
        X_0_dropped = X_0[~keep_ids]
    # Step 2: Gated cross-attention transfer
    X_1_tilde = CrossAttn(Q=X_1, KV=X_0_dropped)  # [T_1, D]
    # Step 3: Gated addition
    alpha = tanh(alpha_l)  # alpha_l init=0, range [-1, 1]
    X_1_new = X_1 + alpha * X_1_tilde
    return X_0_kept, X_1_new
```

Annotations: T_0 = 16×N_frames (after ToMe); shallow TransV: uniform dropping（first attention前Mamba层attention score不可靠）；deep TransV: attention-guided dropping（attention已可靠）；α_l init=0 确保初始时instruction理解不受影响；总压缩: (1-0.5)×(1-0.9) = 5% vision tokens保留。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TimeViper 配置：浅层 TransV 在第 7 层（uniform, p=50%），深层 TransV 在第 39 层（attention-guided, p=90%）。两阶段训练中 TransV 仅在 Stage 2 启用。关键消融：(1) TransV vs token dropping：TVG 上 38.1 vs 26.1；(2) 浅层位置：第 7 层 vs 第 2 层 MCQ 上 +0.6 但 VDC 上 -0.8；(3) 压缩率：50% vs 90% 导致 MCQ 从 56.7 降到 53.4。TransV 也可用于 Qwen2.5 Transformer backbone，但 Qwen 的 VDC 下降更大 (1.3 vs Nano's 0.6)。代码：https://github.com/xiaomi-research/timeviper (TBD)。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding

## Vision-to-Text Information Aggregation / 视觉到文本信息汇聚现象

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vision-to-Text Information Aggregation 是 TimeViper 在 Hybrid Mamba-Transformer MLLM 中发现的 token 信息流现象：随着 LLM 层深度增加，视觉 token 中的信息逐步从 vision tokens 汇聚到 instruction tokens（指令中心任务如 MCQ/TVG）或直接贡献到 response tokens（视觉中心任务如 VDC）。在深层 layer，vision tokens 几乎 100% 冗余——完全移除所有 vision tokens 也不影响模型性能。该现象由 information blocking 实验揭示：通过修改 attention mask 阻断 vision→instruction (V2I) 或 vision→response (V2R) 信息流，观察各层各任务的性能变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Information Blocking Experiment (3 token types: vision, instruction, response)
# Block V2I (vision -> instruction):
# [X_0^{l+1}, X_1^{l+1}, Y^{l+1}] = [[1,0,0],[0,1,0],[1,1,1]] @ [X_0^l, X_1^l, Y^l]
# Block V2R (vision -> response):
# [X_0^{l+1}, X_1^{l+1}, Y^{l+1}] = [[1,0,0],[1,1,0],[0,1,1]] @ [X_0^l, X_1^l, Y^l]
```
Annotations: Instruction-centric tasks (MCQ, TVG) → 浅层阻断 V2I 性能急剧下降，深层几乎无影响（信息已转移）；Vision-centric tasks (VDC) → 阻断 V2R 浅层急剧下降；所有任务深层完全 dropping vision tokens 无性能损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
观察到的现象直接指导 TransV 设计：(1) 浅层保留较多 vision tokens (50% uniform dropping)，因为 vision tokens 在此阶段仍重要；(2) 深层激进 dropping (90%)，信息已转移至 instruction tokens；(3) 通过 cross-attention 显式执行信息转移而非依赖模型隐式学习。这一现象与 Transformer-based MLLM 类似发现一致（LLaVA-Mini, PDrop），但 TimeViper 首次在 hybrid Mamba-Transformer 中验证并利用。方法学源自 What's in the Image (Kaduri et al., CVPR 2025)。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding

## Token Merging (ToMe) for Video MLLMs / 面向视频多模态大模型的Token合并

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Merging (ToMe) 是一种视觉 token 压缩技术（Bolya et al., 2023），通过 ViT 内部 token 相似度将冗余 token 合并为更少的 token。在视频 MLLM 中，ToMe 通常应用在 ViT 编码器与 LLM 之间的 Projector 层。每个视频帧首先由 ViT 编码为大量视觉 token（如 SigLIP 输出 768 tokens/frame @ 384×384），ToMe 基于 token 间余弦相似度合并相似 token，将每帧压缩为固定数量（如 16 tokens/frame）。TimeViper 和 VideoChat-Flash 均使用 ToMe 作为视频 MLLM projector 层压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Merging for Video: 768 -> 16 tokens/frame
def ToMe_video_frame(tokens_f):  # [768, D]
    S = cosine_similarity(tokens_f, tokens_f)  # [768, 768]
    merged = tokens_f.clone()
    for _ in range(768 - 16):
        i, j = argmax(S)  # most similar pair
        merged[i] = (merged[i] + merged[j]) / 2
        merged = remove_row(merged, j)
    return merged  # [16, D]
# In TimeViper: for each frame f_t: v_t = ToMe(SigLIP_ViT(f_t))
```
Annotations: 768 tokens = 24×24 patch grid from SigLIP @ 384×384; ToMe = 48× 帧内压缩; 与 TransV 区别：ToMe 在 LLM 外做帧内压缩，TransV 在 LLM 内做帧间+任务感知压缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ToMe 原始实现：https://github.com/facebookresearch/ToMe。在视频 MLLM 中，每帧独立处理（非跨帧 merging）以保持时序信息。TimeViper 实验：vanilla 模型 128 frames OOM；+ToMe 扩展到 ~5K frames。结合 TransV 后总 vision token 压缩比：768→16 (ToMe, 48×)→0.8 (TransV, 20×) ≈ 960×/frame。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding

## Token Variation（Vision Token Variation / Variation-aware Token Evaluation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Variation（视觉 Token 变异性）是指 LVLM 推理过程中，visual token 表示在相邻 LLM transformer 层之间发生的变化幅度。核心假设：参与 LLM 推理计算的高重要性 token 会在跨层传播时产生显著表示变化（high variation），而被动经过的 token 保持相对稳定（low variation）。V2Drop（CVPR 2026）首次系统性地从 token variation 视角研究 token 压缩，证明 variation 信号与 token 的任务相关性高度一致：high-variation tokens 对应语义重要区域（问题相关物体），low-variation tokens（"lazy tokens"）对应无关背景区域。Variation 的度量指标包括 L1 Distance、L2 Distance 和 Cosine Similarity，其中 L2 Distance 提供了最佳的 performance-efficiency 平衡。该度量与位置无关（spatial-agnostic），因此不受 attention-based 方法的位置偏见影响。

论文给出理论支撑（Theorem 1）：在一阶 Taylor 展开下，||Δf_j|| ≈ ||J_j||_op · ||Δx_j^(t)||，即 token j 对模型输出的影响与其跨层变化量 ||Δx_j^(t)|| 近似成正比——variation 是 token importance 的计算高效代理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Variation 计算发生在 LVLM decoder 的指定剪枝层。对 V2Drop 在 LLaVA-1.5-7B 上的典型配置（layers 3, 17, 22）：

```
# Step 1: Variation Computation at pruning layer l_k
for i in range(M_curr):  # M_curr = current vision token count
    # f_i^(l_k-1): token i at previous layer output
    # f_i^(l_k):   token i at current layer (after Attn+FFN+Residual)
    s[i] = ||f_i^(l_k) - f_i^(l_k-1)||_2  # L2 distance

# Step 2: Sort by variation (descending)
indices = argsort(s, descending=True)

# Step 3: Retain top-K with highest variation
F_v_retained = {f_indices[0], ..., f_indices[K_l - 1]}
# Drop: F_v_dropped = remaining tokens (low-variation lazy tokens)
```

计算开销：M=576, D'=4096 时每层约 7M FLOPs（3MD'），仅为单层 attention（32B FLOPs）的 0.022%；三层总计约 21M FLOPs（完整 forward 的 0.002%）。Variation 信息仅需简单的张量相减 + L2 norm，无需访问 attention map，天然兼容 FlashAttention。

与 Representation Shift 的区别：Representation Shift 度量 MLP 输入→输出的变化（Δ = ||MLP(LN(x)) - x||）；V2Drop 度量相邻层间完整 token 表示的变化（Δ = ||x^(l) - x^(l-1)||），更适合在多个 LLM layer 间进行渐进式剪枝。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在指定 LLM transformer layer 的 residual add 之后，读取当前层和上一层的 visual token hidden states，计算 pairwise L2 距离。在 PyTorch 中：
```python
# hook at layer l after residual connection
with torch.no_grad():
    prev_hidden = hidden_states_buffer[l-1][vision_mask]  # [M, D]
    curr_hidden = hidden_states[l][vision_mask]            # [M, D]
    var_scores = torch.norm(curr_hidden - prev_hidden, dim=-1)  # [M]
    _, topk_idx = torch.topk(var_scores, k=K_l)
    # retain only top-k vision tokens
```

使用场景：(a) 任何 ViT-Projector-LLM 架构的 LVLM 推理加速；(b) 高分辨率图像（576+ tokens）和长视频（1024+ tokens）场景；(c) 单 GPU（A100/3090/4090）部署，需降低延迟和显存；(d) 作为其他压缩方法的排序信号（替代 attention score）。开源：https://github.com/xuyang-liu16/V2Drop（Apache-2.0）。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models

## Inner-LLM Token Compression（LLM内部视觉Token压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inner-LLM Token Compression 是指在 LVLM 的 LLM decoder 内部（forward propagation 过程中）对视觉 token 进行压缩的方法类别，与 Pre-LLM Compression（在进入 LLM 之前压缩）相对。操作时机：visual token 通过 ViT 编码和 Projector 映射后进入 LLM，在 LLM 的若干选定的 transformer layer 之间进行 token 丢弃或合并。代表性方法包括：FastV（early layer 基于 cross-attention 剪枝，ECCV 2024）、SparseVLM（cross-modal attention ranking + 自适应稀疏比 + token recycling，ICML 2025）、PDrop/PyramidDrop（渐进式金字塔型剪枝，CVPR 2025）、V2Drop（variation-aware 剪枝，CVPR 2026）。

Inner-LLM 方法的优势：与模型架构无关（architecture-agnostic），plug-and-play 可插拔，无需修改 ViT encoder 或 Projector，训练无关（training-free）。主要挑战：(1) 依赖 attention weights 的评分方法不兼容 FlashAttention；(2) attention 的位置偏见（positional bias）导致丢弃语义重要 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Inner-LLM token compression 的通用 pipeline：

```
Input: Vision tokens F_v ∈ R^{M×D}, Text tokens F_t, LLM layers L
Output: Generated response Y

1. F_combined = concat([F_v, F_t])  # interleaved format
2. for l = 1 to L:
3.     h = TransformerLayer_l(h)
4.     if l in pruning_layers:
5.         # Pruning strategy varies by method:
6.         # - FastV: attn_score = mean(Attn_weights[text_tokens → vision_tokens])
7.         # - SparseVLM: attn_score = cross_modal_attention_ranking()
8.         # - V2Drop: var_score = ||f_i^(l) - f_i^(l-1)||_2  # attention-free
9.         vision_tokens = select_top_k(vision_tokens, score, K_l)
10.        h = concat([vision_tokens, text_tokens])
11. return auto_regressive_decode(h)
```

典型压缩率：Inner-LLM 方法可在 LLM 浅层剪枝 50-77.8% visual token。V2Drop 在 LLaVA-1.5 上：577→288 (layer 3)→173 (layer 17)→128 (layer 22)，即 77.8% total reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(a) 通过 PyTorch forward hook 在指定 LLM layer 的 attention 或 MLP 后拦截 hidden states；(b) 计算每个 visual token 的重要性分数（attention score / variation score / duplication score）；(c) 根据分数排序选择 top-K 保留；(d) 重组 sequence（丢弃的 visual token 不传入后续 layer）。代码通常集成在 HuggingFace Transformers 的 model forward 中，如 V2Drop 核心实现在 `llava/model/language_model/V2Drop.py`。使用：加载 LVLM checkpoint → 注册 pruning hooks → 正常推理。各方法开源情况：FastV (https://github.com/pkunlp-icler/FastV)，SparseVLM（未完全开源），PDrop (https://github.com/XingLuan/PyramidDrop)，V2Drop (https://github.com/xuyang-liu16/V2Drop)。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models

## Positional Bias in Visual Token Pruning（视觉Token剪枝的位置偏见）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Positional Bias in Visual Token Pruning 是指基于 attention weights 的 inner-LLM 视觉 token 剪枝方法系统性地对序列末尾位置的 token 赋予更高的"重要性"分数，而忽略 token 的实际语义内容。在 LVLM 中，visual token 按空间位置顺序排列（通常从上到下、从左到右扫描）进入 LLM，因此位置偏见具体表现为：过度保留图像底部位置的 visual token，而丢弃图像上部/中部的语义重要 token。V2Drop 通过定量分析（Figure 3）证实：在 LLaVA-1.5-7B 和 Qwen2-VL-7B 上，FastV 和 SparseVLM 等 attention-based 方法在剪枝 50% token 后，末尾 20% 位置的 token 保留概率远高于前部 token，形成"end-of-sequence bias"。这一偏见与 token 的实际语义内容无关（content-agnostic），严重时加剧多模态幻觉——保留不相关 token 同时丢弃关键视觉信息。

产生原因：Transformer 的 causal attention mask 和 positional encoding 使得 LLM 天然倾向于关注序列后部位置，attention weights 将这种位置偏好投射为"重要性"信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
位置偏见通过以下量化方法度量：

```
# Quantifying positional bias
# 1. Apply attention-based pruning (e.g., FastV) at layer k
# 2. Partition vision tokens into N equal intervals by position index
# 3. For each interval j: compute retention probability
#    P_retain[j] = (# tokens retained in interval j) / (total tokens in interval j)
# 4. Visualize: P_retain vs. position interval

# Expected (no bias): P_retain[j] ≈ pruning_ratio, uniform across j
# Attention-based (with bias): P_retain[j] increases monotonically with j
```

V2Drop 证明 variation-based 评分产生近乎均匀的空间保留分布——高 variation 区域可以出现在图像的任何位置（上部、中部、下部），不受 token 序列位置影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
避免位置偏见的策略：(1) 使用与位置无关的 token 重要性信号（如 variation、duplication、entropy），而非 attention weights；(2) 对 attention weights 进行位置去偏处理（如减去 position-only baseline attention）；(3) 在 positional encoding 层面引入 spatial prior（如 2D RoPE）。目前 variation-based 方法（V2Drop）和 duplication-based 方法（DART）是最有效的 position-agnostic 方案。该概念对设计新一代 token compression 方法具有指导意义——任何依赖 LLM attention weights 的 token 重要性评估都应考虑位置偏见的去偏。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models

## Progressive Token Dropping（渐进式Token丢弃）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Token Dropping 是一种多阶段 token 剪枝策略，在 LLM decoder 的多个 depth（浅层→中层→深层）逐步减少 visual token 数量，形成递减的 token 金字塔结构（M → K_a → K_b → K_c），而非在单层一次性丢弃大量 token。V2Drop 采用三层渐进式剪枝：shallow layer（如 layer 3）执行首次筛选，middle layer（如 layer 17）进一步求精，deep layer（如 layer 22）最终压缩到目标数量。渐进式策略的核心优势：早期层的粗筛保留足够 token 供后续层细化选择，避免一次性丢弃过多可能重要的 token；每层基于该层的 variation 信息重新评分，利用更深层的语义理解进行更精准的筛选。

Ablation 结果（V2Drop, LLaVA-1.5-7B, retain 192 tokens）：progressive dropping 相比 one-time dropping 在 POPE 上提升 9.3%、在 MME 上提升 5.9%，证明渐进式策略有效保留关键视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Progressive Token Dropping Schedule
# Input:  M vision tokens, pruning layers L = [l_a, l_b, l_c]
# Output: K_c retained tokens

M_curr = M  # e.g., 576 for LLaVA-1.5
for l in LLM_layers:
    h = TransformerLayer_l(h)
    if l == l_a:   # shallow: aggressive first cut
        M_curr = retain_top_k_by_variation(h, K_a)  # e.g., K_a = 288 (50%)
    elif l == l_b: # middle: further refinement
        M_curr = retain_top_k_by_variation(h, K_b)  # e.g., K_b = 173 (30%)
    elif l == l_c: # deep: final selection
        M_curr = retain_top_k_by_variation(h, K_c)  # e.g., K_c = 128 (22%)

# One-time dropping (for comparison):
# Drop all at single layer: M → K_c in one step
```

关键设计选择：(1) 剪枝层位置——V2Drop 的 ablation（Table 6）显示 (3, 17, 22) 组合最优（97.6% 原性能），但 (3, 15, 27) 等也达到 97.0%+，对层选择鲁棒；(2) 每层压缩比分配——由 performance-efficiency trade-off 调控。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在每层剪枝后重组 token sequence（丢弃的 visual token 位置被删除，缩短后续层的序列长度），直接减少后续所有 attention 和 FFN 的 FLOPs。使用场景：(a) 需要细粒度控制 compression ratio vs. performance 的部署场景；(b) 视频 LLM 场景，长 visual token sequence 需要多阶段筛选；(c) 与 variation/attention/duplication 等任意评分信号配合。PDrop (CVPR 2025) 和 V2Drop (CVPR 2026) 均采用此策略。开源实现见 https://github.com/xuyang-liu16/V2Drop 和 https://github.com/XingLuan/PyramidDrop。

**VFlowOpt 的三阶段渐进式剪枝**（来自 VFlowOpt 论文）：
VFlowOpt 将 LMM 均分为三个阶段，每阶段开始按阶段特定保留率 R=[R1, R2, R3] 保留高重要性 token。剪枝点位置取决于模型层数：LLaVA-OneVision-7B 在 LLM 前、第 9 层后、第 18 层后；LLaVA-NeXT-7B 在 LLM 前、第 10 层后、第 20 层后。每阶段先计算重要性得分，按 R_current 保留 top-k token，剩余 token 进入 recycling（按 a×a 网格加权平均融合后替代最高重要性 token 位置归入保留集合）。整体平均保留率公式：R̄ = (R1·L1 + R1·R2·L2 + R1·R2·R3·L3)/L，其中 L1/L2/L3 为三个阶段各含层数。与 V2Drop/PDrop 的关键区别：(1) 三阶段位置由均匀划分决定（非启发式选择特定层号）；(2) 每阶段执行 importance re-scoring（非仅首次评分后传递）；(3) pruned token 进入 recycling 而非直接丢弃。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization


## Attention Calibration for Token Importance Estimation（注意力校准用于Token重要性估计）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Calibration 是 VFlowOpt 提出的视觉 token 重要性估计增强技术。核心问题：使用 ViT 所有 token 的 attention 均值估计重要性时，冗余 token（如背景区域）会对同类冗余 token 分配不恰当的高 attention（attention bias），导致背景区域重要性被高估。Attention Calibration 通过两步纠正此偏差：(1) 计算全局 attention 阈值 τ = t · (1/N) Σ_i Σ_j A_{ij}，其中 t 为敏感度超参数，A_{ij} 为 ViT 层内 token i 对 token j 的 attention weight；(2) 筛选"相对重要"token 集合 K = {j | Σ_i A_{ij} > τ}，即接收总 attention 超过阈值的 token；(3) 仅用 K 中 token 的 attention 计算重要性 I_i = Σ_{k∈K} A_{ki} + α · softmax(H(V_i))。效果：排除冗余 token 的噪声 attention 信号，使重要性估计更可靠。消融实验表明移除 calibration 导致 MMStar 从 57.8→56.2、SQA 从 92.3→91.8（retain 25% tokens, LLaVA-OneVision-7B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Calibration in VFlowOpt
# Input: ViT attention matrix A ∈ R^{N×N}, sensitivity t, entropy weight α
# Output: importance scores I ∈ R^N

# Step 1: Compute global importance threshold
global_mean_attn = mean(sum(A, dim=1))           # 所有 token 接收的平均 attention
τ = t * global_mean_attn                           # 阈值

# Step 2: Identify relatively important tokens
attn_received = sum(A, dim=0)                      # ∈ R^N, 每个 token j 接收的总 attention
K = {j | attn_received[j] > τ}                     # 被高 attention 关注的 token 集合

# Step 3: Compute importance using ONLY calibrated attention + entropy
for i in 1..N:
    attn_from_K = sum(A[k, i] for k in K)         # K 中 token 对 token i 的 attention 之和
    H = compute_entropy(image_patch_i)              # 256 灰度级熵
    I_i = attn_from_K + α * softmax(H)             # 融合得分

# Contrast with uncalibrated (baseline):
# I_i_baseline = mean(sum(A[:, i])) + ...          # 包含冗余 token 的噪声 attention
```
Annotations: t 控制阈值高度——t 越大则 K 越小（越严格，仅保留最强受关注 token），t 越小则 K 越大（越宽松，接近无校准）。α 控制熵项贡献，由 Bayesian Optimization 自动搜索。与 VisionZip 的退化策略（无 [CLS] 时退化为 mean attention）相比，VFlowOpt 的 calibration 在任何 ViT 架构下均有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现不依赖 [CLS] token 或特定 ViT 架构——仅需从 ViT 最后一层的 attention matrix 中读取 A_{ij} 值（已在 ViT forward pass 中计算完成，Standard PyTorch hook 可捕获）。attention matrix 按 head 取平均后使用。对于使用 Flash Attention 的 ViT，需 output_attentions=True 以获取完整 attention matrix（增加 ~O(N²) 内存开销）。VFlowOpt 开源实现：https://github.com/sihany077/VFlowOpt，基于 LMMs-Eval + LLaVA-OneVision 框架。使用场景：任何依赖 attention 评估 token 重要性的训练无关剪枝方法均可用 calibration 改进——替换原有的 `mean(attn[:, j])` 为 `mean(attn[K, j])`。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization


## Visual Information Flow-Guided Pruning Optimization（视觉信息流引导的剪枝优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Information Flow-Guided Optimization 是 VFlowOpt 提出的自动搜索最优剪枝策略超参数的方法。理论基础：LMM 可解释性研究揭示，视觉信息从 vision tokens → query text tokens → 最后位置 last token 逐层聚合，last token 是 text-visual interaction 最具代表性的信号。VFlowOpt 将剪枝策略设计建模为优化问题：max_s f(s) = CosineSim(h_f, g_s(h_f))，其中 h_f 为无剪枝时最后 token 的表示，g_s(h_f) 为应用剪枝策略 s 后最后 token 的表示。最大化 Cosine Similarity 等价于最小化视觉信息流在剪枝前后的差异——差异越小说明剪枝对 LMM 内部信息处理的扰动越小。

关键洞察：(1) 不同 LMM 有不同的 internal information flow 特征，统一的手工策略无法最优适配；(2) 该优化仅需 30 个无标签样本 + 50 次 Bayesian Optimization 迭代，约 30 分钟完成搜索；(3) 优化目标是 task-agnostic（任务无关）的，因为在优化过程中不涉及任何下游任务标签，仅依赖 LMM 内部表示。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Bayesian Optimization for Pruning Strategy
# Search space: s = (R1, R2, R3, t, α, a)
# R1, R2 ∈ [0, 1]: stage retention ratios
# t: attention calibration sensitivity
# α: entropy weight in importance score
# a: token recycling grid size
# Constraint: R̄ = (R1*L1 + R1*R2*L2 + R1*R2*R3*L3) / L

GP = GaussianProcess(kernel=Matern52)                 # Surrogate model
acquisition = ExpectedImprovement(xi=0.01)             # Acquisition function

# Initial random sampling
X0 = uniform_sample(valid_ranges, n_init=10)
for s in X0:
    R3 = (R_target*L - R1*L1 - R1*R2*L2) / (R1*R2*L3)
    y = sum([CosineSim(LLM_last_token_no_prune(d_i),
                        LLM_last_token_with_prune(d_i, s))
             for d_i in D_unlabeled])                  # D_unlabeled = 30 samples
    data.append((s, y))

# BO iterations
for iter in 1..50:
    GP.fit(data)
    s_next = argmax ExpectedImprovement(s; GP)         # 平衡 exploration/exploitation
    R3 = solve_constraint(s_next.R1, s_next.R2, R_target)
    y_next = evaluate_f(s_next)
    data.append((s_next, y_next))

s_opt = argmax y in data                              # 最优策略
```

Annotations: 目标函数 f(s) 在每次评估时对全部 30 个无标签样本计算 cosine sim 后求和。实验证实 last token 优化优于 mean pooling（MMStar 57.8 vs 56.1）、first token（57.8 vs 54.2）、top-3 tokens（57.8 vs 56.8）。数据选择独立于任务（随机样本 vs MathV360K-GEOS 训练数据效果相当），证明优化的是模型特定信息流而非任务特定特征。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：scikit-optimize 或 BoTorch 的 Gaussian Process + Expected Improvement。LMM forward pass 需捕获最后 token 表示（hooks on final layer hidden states）。使用场景：(1) 为任何 LMM 定制剪枝策略——LLaVA-OneVision-7B、LLaVA-NeXT-7B、Qwen2-VL-7B 三者的最佳超参数经 BO 搜索后各不相同；(2) 可扩展到任何有超参数配置的 token 压缩方法（非仅 VFlowOpt 自身的剪枝）。限制：需约 30 分钟 GPU 时间 per model，对需要频繁切换模型的场景开销较高。开源实现见 https://github.com/sihany077/VFlowOpt。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization


## Token Recycling via Importance-Weighted Merging（基于重要性加权的Token回收/合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Recycling 是 VFlowOpt 提出的剪枝 token 信息保留机制。在标准 token 剪枝中，被丢弃的 token 信息永久丢失（不可逆操作）。Token Recycling 将 pruned tokens 按空间位置分组融合为 compact representations，重新纳入保留集合。流程：(1) 将图像平面划分为 a×a 正方形网格；(2) 各网格 cell 内的 pruned tokens 按重要性加权平均融合为单个 token：t_merged = Σ I_i · t_i / Σ I_i；(3) 融合 token 替换该网格内最高重要性 pruned token 的位置，归入 retained set。效果：在减少 token 数量的同时，将低重要性区域的视觉特征"压缩"保留而非丢弃。

与 ToMe (Token Merging, Bolya et al. 2023) 的关键区别：(1) ToMe 基于余弦相似度合并任意相似 token 对（无空间约束），Token Recycling 按空间网格分组（保持空间结构）；(2) ToMe 在 ViT 内部逐层执行以减少总 token 数，Token Recycling 仅在剪枝阶段执行（与保留集合并）；(3) ToMe 使用等权重平均，Token Recycling 使用重要性加权平均（高 token 贡献更多）；(4) Recycling 是剪枝的补充机制，仅在 token 被标记为 prune 后才触发。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Recycling in VFlowOpt
# Input: pruned tokens P, their features F_P, importance I_P, spatial coords C_P
#        grid size a, retained tokens R
# Output: augmented retained set R' (R + recycled tokens)

grid_cells = {}                                         # Map (p,q) -> list of (idx, feature, importance)
for each pruned token t_i with (x_i, y_i):
    p, q = floor(x_i / a), floor(y_i / a)              # Grid cell assignment
    grid_cells[(p,q)].append((i, F_P[i], I_P[i]))

for (p,q), cell_tokens in grid_cells:
    if len(cell_tokens) == 0: continue
    # Importance-weighted merging
    I_sum = sum(I_j for _, _, I_j in cell_tokens)
    t_merged = sum(I_j * F_j for _, F_j, I_j in cell_tokens) / I_sum
    # Replace position of highest-importance token in this cell
    i_max = argmax_j(I_j for _, _, I_j in cell_tokens)
    R[spatial_pos[i_max]] = t_merged                     # 归入保留集

# Total retained = R (top-k) + sum(len>0 for all cells) (recycled)
```

Annotations: 网格大小 a 由 Bayesian Optimization 搜索——a 越大则每 cell 覆盖更多 token（更多 token 融合为 1 个，更激进压缩），a 越小则保留更多细粒度空间信息。VFlowOpt 仅在 LLM 前的第一阶段剪枝后执行 Recycling（深层不再执行，因为 token 数已大幅减少后融合收益有限）。消融：移除 Recycling 导致 POPE 从 89.1→86.8（retain 25%），说明 Recycling 对 preserving coarse-grained semantics 关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 PyTorch 中为每个 grid cell 执行 `torch.sum(I.unsqueeze(-1) * F, dim=0) / I.sum()` 即可完成加权平均融合。Position IDs 在剪枝/回收后保持不变，保留原始空间结构。使用场景：(1) 任何 token 剪枝框架中作为信息保留的补充机制；(2) 对高分辨率场景（如 LLaVA-OneVision 7290 tokens）尤其有效——大量低重要性背景 token 通过 Recycling 压缩为少数代表性 token 而非全部丢弃。开源实现见 https://github.com/sihany077/VFlowOpt。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization

## Retrieval-Augmented Generation for Video（视频检索增强生成 / Video RAG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Retrieval-Augmented Generation (RAG) 是一种将信息检索（Retrieval）与生成模型（Generation）结合的框架。在视频场景中，Video RAG 将长视频切分为多个片段（clips），对每个片段提取文本描述、视觉特征或结构化信息，构建可检索的知识库。当用户提出查询时，系统先检索与查询最相关的视频片段，再将检索到的片段作为上下文输入 LVLM 生成最终答案。标准 Video RAG pipeline 包含三个阶段：(1) Indexing：将原始视频数据组织为可检索知识库——对每个 clip 调用 LVLM 生成 text description 或用 CLIP/ViT 提取 visual feature embedding；(2) Retrieval：计算 query embedding 与各 clip embedding 的 cosine similarity，返回 Top-N 最相似的 clips；(3) Generation：将检索到的 clips（video frames + subtitles）拼接后输入 LVLM 生成回答。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NaïveRAG for video 的标准 pipeline：
```
# === Indexing (offline) ===
clips = split_video(video, fps=1.0, clip_size=64)
index = []
for clip in clips:
    desc = LVLM.describe(clip)           # text description
    emb = text_encoder.encode(desc)      # BGE/CLIP text embedding
    index.append({"clip": clip, "desc": desc, "emb": emb})

# === Retrieval (online) ===
q_emb = text_encoder.encode(query)       # query embedding
scores = [cosine_sim(q_emb, item.emb) for item in index]
top_k = argsort(scores)[:N]              # Top-N clips

# === Generation (online) ===
context = concat([index[i].clip for i in top_k])
answer = LVLM.generate(query, context=context)
```
Vgent 论文中实现的 NaïveRAG baseline（遵循 GoldFish 风格）即为上述流程——每个 video clip 作为独立 plain text document 处理，检索相似 clip 后直接将视觉 frames 输入 LVLM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Video RAG 的实现通常基于开源 LVLM（如 Qwen2.5-VL, LLaVA-Video, InternVL2.5）搭配 embedding model（如 BGE, CLIP, BERT）进行检索。在 Vgent 中，NaïveRAG 使用 BAAI/bge-large-en-v1.5 进行 embedding 计算，检索 Top-N=20 个 clips 输入 LVLM 生成回答。但 NaïveRAG 的局限在于：将每个 clip 视为独立 document，破坏了跨 clip 的时序依赖和实体连续性；此外，检索到的 clips 中存在大量 hard negatives（语义相似但与问题无关的 clip），干扰 LVLM 推理。Vgent 实验显示 NaïveRAG 在 MLVU 上反而比 base model 直接推理低 3.4 个百分点（65.4 vs 68.8），验证了 NaïveRAG 在复杂长视频任务中的失效。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding

## Graph-based RAG for Video（图增强视频检索生成 / Video GraphRAG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Graph-based RAG (GraphRAG) 是 RAG 的一个变体，用图结构（而非 flat index）组织检索知识库。在视频场景中，GraphRAG 将每个 video clip 建模为图的节点（node），通过共享的语义实体（entities——人物、物体、场景、动作）在节点间建立边（edge），形成视频知识图谱 G=(V, E)。这种图表示的关键优势：(1) 保留跨 clip 的语义关系和时序依赖——同一实体在不同 clip 中出现时通过共享 entity 节点关联；(2) 支持基于实体的精准检索——query keyword 匹配 entity 后直接溯源到所有相关 clip 节点；(3) 图构建是 query-independent 的离线操作，同一视频的多个问题复用同一张图，无需重复处理视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vgent 的 GraphRAG pipeline（graph construction + graph-based retrieval）：
```
# === Phase 1: Graph Construction (offline) ===
G = Graph(vertices=empty, edges=empty)
U = set()  # global unique entities with descriptions

for i, clip in enumerate(clips):
    # Step 1: Entity extraction
    entities = LVLM.extract_entities(clip, subtitle)
    # entities = [{"name": "laptop", "desc": "silver laptop on desk"}, ...]
    
    # Step 2: Entity merging via BGE embedding similarity
    for e in entities:
        e_emb = BGE.encode(e.desc)
        sims = {u: cosine_sim(e_emb, BGE.encode(u.desc)) for u in U}
        if max(sims) > tau:  # tau = 0.7
            u_star = argmax(sims)
            merge(e, u_star)  # unify semantically equivalent entities
            add_edges(v_i, get_nodes(u_star))  # connect to nodes sharing entity
        else:
            U.add(e)  # new unique entity
    
    G.add_vertex(v_i)  # clip i as graph node

# === Phase 2: Graph-based Retrieval (online) ===
keywords = LVLM.extract_keywords(query)
R = set()
for k in keywords:
    for u in U:
        if cosine_sim(BGE.encode(k), BGE.encode(u.desc)) > theta:  # theta = 0.5
            R = R.union(get_nodes(u))

# Re-rank by avg similarity across all clip info (entities, descriptions, subtitles)
R_sorted = rank_by_avg_similarity(R, query_keywords)
R_topK = R_sorted[:20]
```
核心计算：entity merging 和 keyword-entity matching 均基于 BGE text embedding 的 cosine similarity——entity description 经 BGE 编码为 1024-d 向量后进行匹配。图的结构使得检索可以沿着 entity→node 的边直接找到所有相关 clips，而非遍历整个 index。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GraphRAG for video 作为 training-free pipeline 包裹任意开源 LVLM。离线阶段：视频以 1.0 FPS 采样，每 64 帧一个 clip，对每个 clip 调用 LVLM 提取 entities/actions/scenes 的 JSON——无需额外 object detection 或 OCR 模型。BGE embedding 用于 entity 合并 (tau=0.7) 和 keyword-entity 匹配 (theta=0.5)。Vgent 实验证实 GraphRAG 比 NaiveRAG 平均提升 2.9%，在 MLVU 上提升 4.1%——尤其是在 Count/Order 等多 clip 时序推理任务上提升显著（Count: 从 41.7→58.7）。但 GraphRAG 单独使用时提升有限——论文发现 44% 的 failure 案例中正确 clip 已在检索集内，噪声仍干扰 LVLM，这促使了 Structured Reasoning 后检索步骤的引入。与 Video-RAG（依赖 CLIP keyframe selection + external object detection/OCR）和 proprietary LLM-based 方法（VideoAgent, DrVideo 依赖 GPT-4 API）不同，GraphRAG 仅使用开源 LVLM + embedding model。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding

## Large Video Language Model（大型视频语言模型 / LVLM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Large Video Language Model (LVLM) 是多模态大语言模型 (MLLM) 在视频理解领域的扩展。LVLM 接收视频帧序列和文本作为输入，通过视觉编码器（ViT）将每帧编码为 visual tokens → Projector/MLP 将 visual tokens 映射到 LLM 的嵌入空间 → LLM decoder 进行自回归生成。部分 LVLM（如 VideoChat, Video-LLaMA）使用 Q-Former 模块（来自 BLIP-2）将 visual 和 textual features 对齐，其他（如 MiniGPT4-Video, Video-LLaVA）直接将 frame features 拼接后输入 LLM。训练通常使用 video-instruction tuning：在视频-文本配对数据上进行 SFT（Supervised Fine-Tuning），使 LLM 学会基于视频内容回答问题和推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LVLM 的标准推理 pipeline：
```
# LVLM Inference Pipeline
frames = sample_video(video, fps=1.0, max_frames=F)
visual_tokens = []
for frame in frames:
    patches = ViT.encode(frame)        # ViT: [H, W, 3] -> [N_patches, D_vit]
    projected = Projector(patches)     # MLP: [N_patches, D_vit] -> [N_patches, D_llm]
    visual_tokens.append(projected)

# Interleave visual and text tokens
input_seq = [visual_tokens, text_prompt]
# 30-min video at 1fps = 1800 frames -> 200K+ tokens

output = LLM.generate(input_seq)
```
长视频的挑战：30 分钟视频可超过 200K tokens，超出多数 LVLM 的上下文窗口（如 Qwen2.5-VL-7B 上下文约 128K）。现有解决方案包括：(a) Sparse frame sampling——无论视频多长只采样固定数量帧，但丢失细粒度时序信息；(b) Token compression/pooling——如 LongVU 的时空自适应压缩、LLaMA-VID 的 2-token-per-image 表示；(c) Memory aggregation——如 MA-LMM 的记忆增强；(d) RAG-based——如 Vgent，将长视频索引为可检索知识库。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vgent 评估的 LVLM 涵盖 2B-7B 参数范围：InternVL2.5-2B、Qwen2.5-VL-3B/7B、Qwen2-VL-2B/7B、LongVU-7B、LLaVA-Video-7B。所有模型均开源（HuggingFace），Vgent 作为 training-free pipeline 包裹任意 LVLM，不修改模型权重。在 MLVU 基准上，Vgent 在 7 种 LVLM 上一致带来 3.0%-5.4% 的绝对准确率提升。关键发现：Qwen2.5-VL-3B + Vgent 达到 70.4% MLVU，超越其 7B base model (68.8%)——说明小模型配合良好 RAG 可以匹敌甚至超越大模型的直接推理能力。LVLM 的性能上限仍是 Vgent 的瓶颈——论文在 Limitations 中指出 pipeline 性能受限于 base LVLM 的能力。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding

## Structured Query Refinement（结构化查询精炼 / Structured Reasoning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured Query Refinement（结构化查询精炼），在 Vgent 中也称为 Structured Reasoning，是一种后检索（post-retrieval）验证机制。其核心思想是：检索到的 Top-N clips 中存在 hard negatives（与 query 词义相似但实际不包含关键信息的 clips），直接将这些 clips 输入 LVLM 会稀释关键信息导致推理失败。Structured Reasoning 引入一个中间推理步骤——将原始 query 分解为一组结构化 subqueries，每个 subquery 的预期答案为二元 (yes/no) 或数值 (count)，用这些结构化的、容易验证的问题对每个检索 clip 逐一筛选，仅保留至少有一个 subquery 正向匹配的 clip，消除 hard negatives。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vgent 的 Structured Reasoning pipeline：
```
# === Step 1: Subquery Generation ===
Q_struct = LVLM.generate_subqueries(query, keywords)
# 例: query="Did I open the laptop?"
# Q_struct = [
#     {"type": "binary", "text": "Is there a laptop in the video?"},
#     {"type": "binary", "text": "Is the laptop open?"},
#     {"type": "binary", "text": "Is someone interacting with the laptop?"}
# ]

# === Step 2: Per-Clip Structured Verification ===
R_prime = []
for v_i in R_topK:  # Top-20 retrieved clips
    responses = [LVLM.answer(q.text, v_i) for q in Q_struct]
    # binary: 1(yes) or 0(no); numeric: count value
    if any(r > 0 for r in responses):
        R_prime.append(v_i)

R_prime = R_prime[:5]  # max r=5 clips retained

# === Step 3: Information Aggregation ===
summary = LVLM.aggregate(
    subquery_results, refined_clips
)

# === Step 4: Final Generation ===
answer = LVLM.generate(query, context={
    "video_clips": R_prime,
    "reasoning_summary": summary
})
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vgent 实验：(1) Structured Reasoning 在 GraphRAG 基础上额外提升 MLVU 2.6%、VideoMME 1.6%，总体比 base model 平均提升 3.4%。(2) Structured Reasoning 的效果依赖底层检索质量——应用于 NaiveRAG 时仅带来 65.4→68.6 (+3.2)，仍低于 base model 的 68.8。(3) confidence-based refinement（让模型自我反思 clip 相关性）仅带来 0.2% 提升——验证了结构化验证优于模型自反思路径。(4) r=5（最多保留 5 个 clips）在实验中取得最佳性能。(5) Count 和 Order 任务上提升最为显著——Count: 41.7→58.7 (+17.0), Order: 61.0→67.1 (+6.1)——因为这些任务最需要多 clip 信息聚合。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding

## Video Entity Extraction and Graph Construction（视频实体提取与图构建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Entity Extraction and Graph Construction 是 Vgent 中的离线预处理阶段，将原始长视频转换为可检索的结构化知识图谱。该过程分为两个子步骤：(1) Visual Entity Extraction——对每个 video clip（64 帧），调用 LVLM 提取关键语义实体（entities：物体、人物、场景）、动作（actions：交互/行为描述）和场景（scenes：地点/环境），输出为结构化 JSON。该步骤同时利用视频的视觉内容（frames）和口语内容（subtitles/ASR），形成图文对齐的实体描述。(2) Graph Construction——基于提取的实体构建视频知识图谱 G=(V, E)，通过 BGE text embedding 的 cosine similarity 识别和合并跨 clip 的语义等价实体，在共享实体的节点间建立边。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
实体提取的 JSON 输出示例（来自论文 Appendix C.1）：
```
{
  "entities": [
    {"entity name": "sailboat", "description": "A classic sailboat with white sails and wooden rigging"},
    {"entity name": "man", "description": "A man wearing a dark sweater"},
    {"entity name": "ocean", "description": "A calm ocean under a partly cloudy sky"}
  ],
  "actions": [
    {"entity name": "sailboat", "description": "sailing smoothly on the water"},
    {"entity name": "man", "description": "steering the sailboat"}
  ],
  "scenes": [
    {"location": "open sea"}
  ]
}
```
实体合并算法：
```
U = set()  # global unique entities
for clip i, entity e_j in extracted entities:
    t_j = BGE.encode(e_j.description)     # 1024-d text embedding
    scores = {u: cosine_sim(t_j, BGE.encode(u.description)) for u in U}
    u_star = argmax(scores)
    if scores[u_star] > 0.7:               # tau = 0.7 merging threshold
        merge(e_j, u_star)                 # unify as same entity
        add_edges(v_i, {v | u_star in entities(v)})
    else:
        U.add(e_j)                         # new distinct entity
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 采样率：1.0 FPS，每 64 帧一个 clip。(2) LVLM 用于实体提取的 prompt 设计（Appendix B.1）：要求 LVLM 以 JSON 格式输出 entities、actions 和 scenes，每个 entity 附带 description 字段供 embedding 编码。第一人称视频中 subject 被描述为 "me"。(3) Embedding 模型：BAAI/bge-large-en-v1.5 (1024-d) 在实验中优于 CLIP 和 BERT——BGE 是专门为语义文本相似度优化的 embedding 模型。(4) 合并阈值 tau=0.7：较高阈值偏向精准匹配，避免浅层语义相似导致的错误合并。(5) 图构建的离线性：图构建是最耗时的步骤（20.13 sec/min-video），但这是 query-independent 的一次性开销——同一视频的多个问题复用同一张图，在多问题场景下摊薄开销。Vgent 在每视频 3 个问题的 VideoMME 上实现 1.73x 加速。(6) 当前局限：仅使用 textual entity descriptions 构建图，未包含 visual embeddings 或 frame-level features——论文在 Limitations 中指出这是未来改进方向。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding

## Auto-Thinking (Adaptive Reasoning / Auto Reasoning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Auto-Thinking（也称 Adaptive Reasoning 或 Auto Reasoning）是一种推理控制策略，允许模型在推理时**动态决定是否调用链式思维（CoT）推理**，而非对所有输入都强制执行 CoT。核心理念是"必要时才推理"（reason-when-necessary）：简单/感知导向的输入直接输出答案（direct answering），复杂/推理导向的输入触发完整 CoT 推理链。相比"始终推理"（always-thinking），Auto-Thinking 旨在保持准确率的同时显著减少推理 token 消耗和延迟。VideoAuto-R1 将 Auto-Thinking 首次系统性地引入视频理解领域，揭示了视频 CoT 并非普遍有效——在感知导向 benchmark（VideoMME, MVBench）上 direct answering 通常匹配甚至超过 CoT 性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Auto-Thinking 的实现通常分为两类途径：

**(a) Training-based Auto-Thinking（训练时学习切换策略）**：在训练阶段为每个样本标注 think/no-think 标签（例如 AdaptThink），通过 SFT + RL 让模型学会对 easy/hard 样本分别输出不同格式（有/无 CoT）。VideoAuto-R1 的消融（Table 7）证明此方法在视频域易发生 mode collapse（始终 think 或始终 no-think），因为视频中"must-think"样本稀缺（VideoMMMU 上 CoT-Direct gap 仅 +1~3.4%）。

**(b) Inference-based Auto-Thinking（推理时自动决策）**：VideoAuto-R1 采用的策略。训练时不区分 think/no-think 模式，统一使用 answer→think→answer 格式。推理时通过 confidence score 决定早停或继续：

```
# Inference-Based Auto-Thinking (VideoAuto-R1)
Require: model p_θ, input (v,q), confidence threshold τ=0.97

# Step 1: Generate first answer a_1 until <think> tag detected
a_1_tokens, logprobs = p_θ.greedy_decode(v, q, stop_token="<think>")

# Step 2: Compute confidence score
L = len(a_1_tokens)
if a_1 == "Let's analyze...":   # fallback string
    s = -∞  # force continue
else:
    s = (1/L) * Σ_{ℓ=1}^{L} logprobs[ℓ]   # length-normalized mean log-prob

# Step 3: Decision
if s >= log(τ):     # e.g., τ=0.97 → log(0.97) ≈ -0.0305
    return a_1       # EARLY EXIT: direct answer (~10 tokens)
else:
    a_2 = p_θ.continue_decode()  # Continue: CoT + reviewed answer (~91 tokens)
    return a_2
```

VideoAuto-R1 中 Auto-Thinking 的行为特征（Table 8）：
- 感知导向 benchmark (MVBench): 平均 confidence 0.948, think ratio 25%, CoT gain +0.1%
- 推理导向 benchmark (VideoMMMU): 平均 confidence 0.874, think ratio 51%, CoT gain +4.0%
- Recall of think-needed samples（a_1 错误 a_2 正确的样本被路由到 CoT mode）: 94-100%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
当前文献中 Auto-Thinking 的实现方式：AdaptThink（Zhang et al. 2025b）在训练时标注 think/no-think 标签，通过 on-policy 数据平衡维持 mode 比例约 1:1；R-4B（Yang et al. 2025b）采用 Bi-Mode Annealing，先用 SFT 二模式预热再 RL 精炼；TON（Think-or-Not, 2025）使用 "thought dropout" SFT + GRPO 自由探索。VideoAuto-R1 区别于这些方法：(1) 训练时不区分模式（统一 answer-think-answer 格式），消除 per-sample 标签和 mode collapse 问题；(2) 推理时通过 token-level confidence（length-normalized log-probability）自动化决策，τ=0.97 跨数据集泛化无需调参。局限性：(1) confidence score 仅基于自回归 log-probability，未显式校准；(2) 对视频 grounding 任务 CoT 几乎无增益（初始与审查答案 mIoU 相同），说明纯语言 CoT 无法细化精确的时间边界。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## Answer-Think-Answer Template / Thinking Once, Answering Twice

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Answer-Think-Answer Template（"先答-再思-后审"，也称 Thinking Once, Answering Twice）是 VideoAuto-R1 提出的输出格式范式。模型在一次生成中按序输出三部分：(1) 第一轮 boxed 答案 $\boxed{a_1}$（简洁直接答案或 fallback 字符串）；(2) `<think>` 标签包裹的自由形式推理链 $r$；(3) 第二轮回 boxed 答案 $\boxed{a_2}$（审查后答案，可与 $a_1$ 相同或修正）。格式约束为恰好两个 `\boxed{}` 块和一个 `<think>...</think>` 块，无前后额外文本。核心理念：将"何时思考"（inference-time 早停决策）与"如何思考"（training-time RL 学到的推理行为）解耦。训练时模型学会同时输出两种答案并让两者都正确；推理时若初始答案足够自信则早停，否则继续推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
System Prompt（Table 2）精确定义了输出格式：

```
SYSTEM PROMPT:
You are a helpful assistant.

FIRST: Output your initial answer inside the first \boxed{...} without any
analysis or explanations. If you cannot determine the answer without reasoning,
output \boxed{Let's analyze the problem step by step.} instead.

THEN: Think through the reasoning as an internal monologue enclosed within
<think>...</think>...

AT LAST: Output the final answer again inside \boxed{...}. If you believe
the previous answer was correct, repeat it; otherwise, correct it.

Output format: \boxed{...}<think>...</think>\boxed{...}
```

具体输出示例（来自 VideoMMMU 数学推理题）：
```
\boxed{D}  (confidence 0.92, continue CoT reasoning)
<think>
To find P(x < 3.5 | x < 4), use conditional probability:
P(A|B) = P(A∩B)/P(B). Since A ⊂ B, P(A∩B) = P(A).
f(x) = 1/(4.5-1.5) = 1/3
F(x) = (x-1.5)/3
P(x < 3.5) = 2/3, P(x < 4) = 2.5/3
P(x < 3.5 | x < 4) = (2/3)/(2.5/3) = 2/2.5 = 0.8
Therefore: C. 0.8
</think>
\boxed{C}
```
注意：初始答案是 D（错误），推理后修正为 C（正确）。

输出格式通过 strict regex check 的 format reward 强制：$R_{fmt} \in \{0,1\}$（二进制，完全符合模板得 1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 格式通过精心设计的 system prompt（Table 2）强制，无需 cold-start SFT 教格式——VideoAuto-R1 直接 RL 训练即可获得 ~100% format compliance；(2) Fallback 机制：当问题无法立刻回答时，$a_1$ 输出 "Let's analyze the problem step by step"，避免低置信度猜测，confidence score 被设为 $-\infty$ 强制继续 CoT；(3) 训练时 $w_2 > w_1$（如 0.9:1.1）确保推理后答案优先于初始答案获得更高 rewards。此模板的优势：解耦 training objective 和 inference policy，用户可按需选择始终使用 $a_2$（高精度）或早停 $a_1$（高效率），提供灵活的精度-效率 trade-off。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## Dual-Answer Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Answer Reward（双答案奖励）是 VideoAuto-R1 中训练 "answer-think-answer" 模板的核心奖励设计。与标准 GRPO 仅奖励最终答案不同，Dual-Answer Reward 同时监督初始答案 $a_1$ 和审查答案 $a_2$，通过不对称权重 $w_2 > w_1$ 鼓励模型通过推理改进答案。总奖励公式为：

$$R = w_1 R_{task}^{(1)}(a_1) + w_2 R_{task}^{(2)}(a_2) + \lambda R_{fmt} + \alpha R_{fallback}$$

其中 $w_1=0.9, w_2=1.1, \lambda=1, \alpha=0.3$。权重不对称的关键推理（Table 12）： 若 $w_1=w_2=1$，"correct→wrong" 和 "wrong→correct" 两种模式获得相同总奖励（1），无法区分；当 $w_1=0.9, w_2=1.1$，"correct→wrong" 得 0.9，"wrong→correct" 得 1.1，明确鼓励模型通过推理纠正初始错误。训练曲线（Figure 6）显示 $R_{task}^{(2)}$ 始终高于 $R_{task}^{(1)}$，验证推理阶段确实改善答案质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Dual-Answer Reward Computation (per rollout)
# 解析输出: \boxed{a_1} <think> r </think> \boxed{a_2}

# Task reward computation (depends on task type)
if task == "QA":
    R_task^{(1)} = exact_match_or_math_verify(a_1, GT)   # {0,1}
    R_task^{(2)} = exact_match_or_math_verify(a_2, GT)   # {0,1}
elif task == "Temporal_Grounding":
    R_task^{(k)} = max_tIoU(pred_segments, GT_segments)  # [0,1]
elif task == "Grounding_QA":
    R_task^{(k)} = R_QA + R_TG                           # [0,2]

# Format reward
R_fmt = 1 if regex_match(template) else 0  # strict: exactly 2 boxes + 1 think block

# Fallback bonus
R_fallback = 1 if (a_1 == "Let's analyze...") and (R_task^{(2)} > 0) else 0

# Total reward
R = 0.9 * R_task^{(1)} + 1.1 * R_task^{(2)} + 1.0 * R_fmt + 0.3 * R_fallback
```

奖励分配矩阵（Table 12）：
| $a_1$ | $a_2$ | $w_1$:$w_2$=1:1 | 0.9:1.1 | 0.9:1.1+α=0.3 |
|-------|-------|-----------------|---------|-----------------|
| ✗ | ✗ | 0 | 0 | 0 |
| Let's analyze | ✗ | 0 | 0 | 0 |
| ✓ | ✗ | 1 | 0.9 | 0.9 |
| ✗ | ✓ | 1 | 1.1 | 1.1 |
| Let's analyze | ✓ | 1 | 1.1 | 1.4 |
| ✓ | ✓ | 2 | 2 | 2 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与标准 GRPO 的差异：标准 GRPO 仅有一个 task reward（最终答案），Dual-Answer Reward 需要解析两轮 boxed 答案并分别评估。格式 reward 通过 strict regex 强制执行 `\boxed{...}<think>...</think>\boxed{...}` 格式。Fallback reward α 仅在 $a_1$ 为 fallback 字符串且 $a_2$ 正确时激活，鼓励模型在无法立刻回答时诚实 defer 而非猜测。消融实验（Table 9）证实：(1) 不对称权重（0.9:1.1）优于均匀权重（1:1），VideoMMMU 从 56.1→56.4；(2) 添加 fallback reward α=0.3 进一步提升至 58.6。权重过度不对称（0.8:1.2）可能导致模型过度依赖推理而退化初始答案能力（VideoMME 65.8 vs 67.3）。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## Confidence-Based Early Exit

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Confidence-Based Early Exit（基于置信度的早停）是 VideoAuto-R1 在推理时使用的规则化决策机制。模型首先生成初始答案 $a_1$（通常 <10 tokens），计算其 length-normalized mean log-probability 作为置信度分数 $s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid t_{<\ell}, q)$。若 $s(a_1) \geq \log \tau$（默认 $\tau = 0.97$），则接受 $a_1$ 并提前终止解码（等效 direct answering）；否则继续生成推理链 $r$ 和审查答案 $a_2$。此机制的关键特性：(1) 仅需 $a_1$ 的 log-probability，无需额外校准器或分类头；(2) 决策完全由 test-time 信号驱动，训练时未显式优化 confidence calibration；(3) $\tau$ 提供连续可控的精度-效率 trade-off knob。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
推理流程（Algorithm 1）：

```
Require: p_θ, v, q, τ=0.97, fallback_string f

# Phase 1: Generate until first <think> tag
tokens_prefix = p_θ.greedy_decode(v, q, stop_on="<think>")
a_1 = parse_first_boxed(tokens_prefix)  # extract \boxed{...}
L = len(tokenize(a_1))

# Phase 2: Confidence computation
if a_1 == f:
    s = -1e6  # fallback forces full CoT
else:
    # Length-normalized mean log probability
    s = (1/L) * sum(logprobs_of(a_1_tokens))  # log p_θ(t_ℓ | context)

# Phase 3: Decision
if s >= log(τ):     # τ=0.97 → threshold ≈ -0.0305
    return a_1       # EARLY EXIT (~10 tokens)
else:
    remaining = p_θ.continue_decode(max_new=4096)
    r = parse_between(remaining, "<think>", "</think>")
    a_2 = parse_last_boxed(remaining)
    return a_2       # FULL CoT (~91 tokens)
```

$\tau$ 的影响（Figure 3）：
- 推理密集 benchmark (VideoMMMU): τ 从 0.86→0.98，accuracy 从 57.5%→58.7%，think ratio 从 29%→55%
- 感知导向 benchmark (VideoMME): accuracy 始终不变（diminishing returns from CoT），think ratio 仍随 τ 增加
- 默认 τ=0.97 为鲁棒选择，无需 per-dataset 调参

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上检测到 `<think>` tag 时暂停生成，提取 `\boxed{...}` 中的 tokens 计算 log-probability。由于 $a_1$ 通常仅包含答案字母/数字/短文本（<10 tokens），confidence 计算开销可忽略。$s(a_1)$ 使用标准自回归 log-probability（由 greedy decoding 的 softmax 输出），无需额外 forward pass。此机制依赖于 token-level confidence 与答案正确性的相关性（Liao et al. 2025 首次系统证明），VideoAuto-R1 在视频域验证了该相关性（Table 8: MVBench/MMVU 上 recall of think-needed samples = 100%，VideoMMMU = 94%）。局限性：(1) 训练时未显式优化 confidence calibration；(2) τ=0.97 为经验最优值，泛化到其他模型家族可能需要重新校准。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## Fallback Tolerance / Fallback Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fallback Tolerance（回退容忍）是 VideoAuto-R1 中处理"模型无法不经推理就直接回答"场景的机制。当问题过于复杂（如需要多步数学推导），模型允许在 $a_1$ 位置输出 fallback 字符串 "Let's analyze the problem step by step."（而非强制猜测），然后通过 CoT 推理产出 $a_2$。Fallback Reward $\alpha R_{fallback}$（$\alpha=0.3, R_{fallback} \in \{0,1\}$）在 $a_1$ 为 fallback 字符串且 $a_2$ 正确时提供额外奖励（总奖励从 1.1 增至 1.4），激励模型在无法立刻回答时诚实 defer 而非低置信度猜测。推理时若 $a_1$ 为 fallback 字符串，confidence score 被强制设为 $-\infty$，保证必须进入 CoT 阶段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fallback Reward 的效果（Table 12，最后一列）：

```
if a_1 == "Let's analyze the problem step by step.":
    if a_2 is correct:
        total_reward = w_2 * 1 + α * 1 = 1.1 + 0.3 = 1.4
    else:
        total_reward = 0  # 错误推理后仍错误
elif a_1 is a wrong guess (not fallback):
    if a_2 is correct:
        total_reward = w_2 * 1 = 1.1  # 没有 fallback bonus
    else:
        total_reward = 0
```

消融实验（Table 9）显示：(1) w1:w2=0.9:1.1 + fallback α=0.3 在所有 benchmark 上最优（VideoMME 67.3, VideoMMMU 58.6, MVP 39.4, Charades-STA 60.0）；(2) 仅使用不对称权重无 fallback 时 VideoMMMU 56.4（vs 58.6 with fallback），说明 fallback 对推理密集任务提升显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：fallback 字符串作为 system prompt 中定义的保留 token sequence（"Let's analyze the problem step by step."），通过 exact match 检测。训练时 fallback reward 是二元值（匹配=1, else=0），与 task reward 独立计算后加权相加。权重 $\alpha=0.3$ 的选择需平衡：过大可能导致模型过度使用 fallback（逃避初始答案），过小则激励不足。VideoAuto-R1 的经验设置（$\alpha=0.3, w_1=0.9, w_2=1.1$）通过消融确定。与训练时 confidence calibration 的差异：fallback 是**显式文本信号**（模型自己决定 defer），而非隐式低概率值——这使得决策可解释、可审计。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## Overthinking in Multimodal Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Overthinking（过思考）指推理模型对无需复杂推理的输入生成长链式思维，导致：(1) 准确率不增反降（推理链中的单步幻觉或错误推理覆写正确的初始直觉）；(2) 推理成本和延迟显著增加（生成额外数百 tokens）。VideoAuto-R1 在视频域首次系统性揭示了过思考现象：Table 1 显示 Video-R1 的 CoT 推理（386 tokens avg）在 VideoMME 上准确率 64.3% 甚至低于 direct answering 的 64.6%；Time-R1 的 CoT（138 tokens avg）在 VideoMME 上 63.8% vs direct 65.9%（-2.1%）。图 7 提供了过思考导致错误的定性示例：VideoChat-R1 的 CoT 推理链中幻觉了不存在的舞蹈动作描述，将正确的 direct answer D 覆写为错误的 E。过思考的根本原因是视频感知任务主要依赖视觉识别而非符号推理，冗长的语言推理链引入噪声而非价值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
过思考的表现形式（来自 VideoAuto-R1 的分析）：

```
# Overthinking 导致的"退化"模式
# Case 1: CoT 幻觉覆写正确直接答案（图 7）
Direct Answer: D (correct) 对 dance video 的最后一个动作
CoT Answer: D→E (incorrect) 
  推理链: 描述了不存在的舞步 → 错误推论 → 覆写正确初始判断

# Case 2: CoT 冗余验证但最终答案不变（MVBench 上 ~75% 案例）
Direct Answer: C (correct)
CoT Answer: C (correct)
  推理链: 150 tokens 逐步描述视频和对比选项 → 与 direct answer 相同结论
  → 浪费推理计算但无精度增益
```

量化证据（Table 8）：
- MVBench: confidence 0.948, think ratio 25%, CoT gain +0.1%（几乎无增益）
- MMVU: confidence 0.933, think ratio 39%, CoT gain +0.4%（边际增益）
- VideoMMMU: confidence 0.874, think ratio 51%, CoT gain +4.0%（确实需要推理的任务）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解过思考的策略：(1) Auto-Thinking（VideoAuto-R1 的核心方案）：仅在置信度低时触发 CoT，避免强行推理；(2) 难度感知数据过滤（VideoAuto-R1 的 filtering pipeline）：丢弃"所有 8 个 response 均正确"的过于简单样本，避免模型学习对琐碎问题展开复杂推理；(3) 训练时使用 Dual-Answer Reward 引导模型产出简洁初始答案。类似现象也在文本和图像域被观察到：Sui et al. (2025) 的 "Stop Overthinking" 综述、Kumar et al. (2025) 的 Overthink 攻击、Chen et al. (2024) 对 o1-like LLMs 过思考的分析。VideoAuto-R1 的启示：视频感知任务中 explicit language-based reasoning 并非普遍必要，Auto-Thinking 是更匹配视频特性的推理范式。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice

## SceneTiling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SceneTiling 是 VideoLLaMB 提出的无模型（model-free）视频语义分割算法，受 NLP 领域 TextTiling（Hearst, 1997）启发。其核心思想是：视频中相邻帧的 ViT [CLS] token 余弦相似度在场景边界处会出现显著下降（即边界两侧的语义内容差异最大），通过计算 depth score 检测这些"语义低谷"来分割视频。算法流程：(1) 计算相邻帧对 ViT [CLS] token 的余弦相似度序列 {c_1, ..., c_{n-1}}，c_i = CosineSim(ViT(v_i).cls, ViT(v_{i+1}).cls)；(2) 对每个位置 i 计算 depth score d_i = (cl_i + cr_i - 2c_i) / 2，其中 cl_i 和 cr_i 分别是 i 左侧和右侧的局部最大相似度——d_i 越大，说明 i 处的相似度相对周围越低，即语义边界越明显；(3) 计算 depth score 的均值 μ 和方差 σ，设定阈值 μ + α·σ（α 为超参数控制分割粒度），选取超过阈值的 K-1 个 depth score 对应位置作为分割点，将视频分为 K 个语义段 {s_1, ..., s_K}。SceneTiling 也可用于流式视频字幕生成：仅使用左侧相似度 d_i = (cl_i - c_i)/2 实时检测场景变化边界，无需预知完整视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SceneTiling 在 VideoLLaMB pipeline 中位于 vision encoder 之后、Memory Bridge 之前：

```
# === SceneTiling 伪代码 ===
# 输入: video V = {v_1, v_2, ..., v_n} (n frames), 超参数 α (默认值由论文经验设定)
# 输出: K 个语义段 {s_1, s_2, ..., s_K}

# Step 1: 提取帧级 CLS token
for i = 1 to n:
    f_i = ViT(v_i).cls_token  # ViT-L/14, dim=1024

# Step 2: 计算相邻帧余弦相似度
for i = 1 to n-1:
    c_i = CosineSimilarity(f_i, f_{i+1})
    # c_i ∈ [-1, 1], 值越高表示两帧越相似

# Step 3: 计算 depth score
for i = 1 to n-1:
    cl_i = max(c_1, ..., c_{i-1})  # 左侧最高相似度
    cr_i = max(c_{i+1}, ..., c_{n-1})  # 右侧最高相似度
    d_i = (cl_i + cr_i - 2 * c_i) / 2
    # d_i 高 → 帧i处的相似度显著低于周围 → 潜在场景边界

# Step 4: 确定分割阈值和分割点
μ = mean(d_1, ..., d_{n-1})
σ = std(d_1, ..., d_{n-1})
threshold = μ + α * σ
boundaries = {i | d_i > threshold}
# 选取 K-1 = len(boundaries) 个分割点

# Step 5: 分割视频
{s_1, s_2, ..., s_K} = split_by_boundaries(V, boundaries)
# 每个 s_j 是连续帧序列，内部语义一致

# 流式模式 (streaming caption):
for i = 1 to n-1:
    d_i = (cl_i - c_i) / 2  # 仅用左侧，实时检测
    if d_i > threshold:
        trigger_caption_generation()  # 场景变化时自动生成字幕
```

参数量化：n 帧视频的 SceneTiling 仅需 O(n) 次余弦计算 + O(n) 次 depth score 计算，额外开销极小。CLS token 维度为 ViT 隐藏维度（如 1024），余弦相似度计算为常数时间。α 控制分割敏感度：α 越大 → 阈值越高 → 分割点越少；α 越小 → 分割点越多但可能引入噪声分割。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SceneTiling 的实现完全基于 ViT 编码器的输出，无需训练任何额外参数。VideoLLaMB 开源实现中（github.com/bigai-nlco/VideoLLaMB），SceneTiling 作为预处理模块在特征提取后执行。核心代码使用 PyTorch 的 cosine_similarity 函数和简单的 NumPy/PyTorch 统计计算。流式模式下通过缓存左侧最大值 cl_i（而非全序列最大值）实现实时检测。局限性：(1) 依赖 ViT 编码质量，低质量/模糊帧可能导致不准确的相似度计算；(2) α 需要针对不同视频类型（快节奏 vs 慢节奏）调参；(3) 渐变场景过渡（fade/dissolve）可能不会被检测为边界，因为帧间相似度变化平缓；(4) 论文未提供 α 的最优经验值或自适应策略。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## Recurrent Memory Bridge Layers

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recurrent Memory Bridge Layers（递归记忆桥接层）是 VideoLLaMB 中的核心模型组件，位于视觉编码器（ViT）和大语言模型（LLM）之间的 bridge 位置。它由单层 Transformer（8 attention heads, hidden size=1024）构成，在输入段前 prepend 固定数量的 learnable memory tokens（32 个），通过 self-attention 同时处理 memory tokens 和当前段视觉特征：`[m_{i+1}; o_i] = BridgeLayer([m_i; s_i])`，其中 m_i 是第 i 步输入的记忆 token，s_i 是第 i 个语义段的视觉特征，m_{i+1} 是更新后的记忆 token（输出给下一步），o_i 是当前段的视觉表示（输出给 LLM）。关键设计：(1) Bridge Layer 使用标准 self-attention，不修改 ViT 和 LLM 架构，保持 plug-and-play 特性；(2) 递归处理语义段，每个段仅与当前 memory tokens 交互，计算复杂度为 O((C+M)^2) per segment，其中 C 为段内帧数、M=32 为 memory token 数；(3) memory tokens 逐步累积全视频信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory Bridge Layer 的计算流程：

```
class MemoryBridgeLayer(nn.Module):
    """单层 Transformer Bridge with Self-Attention"""
    def __init__(self):
        self.self_attn = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm1 = LayerNorm(1024)
        self.norm2 = LayerNorm(1024)
        self.ffn = FeedForward(1024, 4096)  # 标准 FFN
        # 注意：没有 cross-attention，仅 self-attention

    def forward(self, memory_tokens, segment_features):
        # memory_tokens: [32, 1024]  ← 来自前一步或初始化
        # segment_features: [C, 1024]  ← SceneTiling 分割的段
        
        # Step 1: Concat memory + segment
        x = torch.cat([memory_tokens, segment_features], dim=0)  # [32+C, 1024]
        
        # Step 2: Self-attention (所有 token 两两交互)
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        
        # Step 3: Split 回 memory 和 visual
        updated_memory = x[:32, :]        # [32, 1024] → m_{i+1}，传给下一步
        visual_output = x[32:, :]         # [C, 1024] → o_i，传给 LLM
        
        return updated_memory, visual_output
```

递归流程：
```
m_0 = nn.Parameter(torch.randn(32, 1024))  # 可学习初始化
MemoryCache = []

for i = 1 to K:  # K 个语义段
    m_i_raw, o_i = bridge_layer(m_{i-1}, s_i)
    
    # Memory Retrieval (cross-attn with cache)
    if MemoryCache:
        M_cache = torch.cat([m_0, m_1, ..., m_{i-1}], dim=0)  # [32*(i), 1024]
        m_i = cross_attn(query=m_i_raw, key=M_cache, value=M_cache)
    else:
        m_i = m_i_raw
    
    MemoryCache.append(m_i)
    llm_inputs.append(o_i)

# 最终: LLM 输入包含所有段的 o_i + 最终 memory m_K
```

训练时仅 Bridge Layer 和 LLM 的参数被更新（ViT 冻结），Bridge Layer 参数量约为 1 层 Transformer (~7M params for hidden=1024, heads=8)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用单层 Bridge Layer、32 个 memory tokens、hidden=1024、8 attention heads。训练时冻结 ViT-L/14 参数，仅训练 Bridge Layer 和 LLM（Vicuna-7B，使用 LoRA 或全参数微调，论文未明确说明 LLM 微调方式）。初始化来自 LLaVA-1.5 的权重。设计受 RMT (Recurrent Memory Transformer, Bulatov et al. 2022) 启发，但 VideoLLaMB 的关键区别在于：(1) Memory Bridge 在 vision-LLM 之间而非 LLM 内部，因此不影响 LLM 的推理能力；(2) 结合 SceneTiling 语义分割而非均匀分段；(3) 使用 cross-attention retrieval 而非简单传递。扩展性：Table 9 显示增加 Bridge Layer 层数（1→3）和 memory token 数（32→64）均可提升性能（53.8→54.6），但论文选择单层+32 tokens 的 lightweight 配置以平衡效率和性能。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## Memory Cache with Retrieval (Video Context)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Cache with Retrieval 是 VideoLLaMB 中解决递归记忆桥接层 BPTT（Backpropagation Through Time）梯度消失问题的机制。在每个 time step i，系统将所有历史 memory tokens 存储在 MemoryCache M_i = [m_1, ..., m_i] 中，使用当前 memory token m_i 作为 query、拼接的历史 cache M_i 作为 key 和 value，通过标准 multi-head cross-attention 检索历史信息并更新当前 memory：$m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$。此机制的核心优势：(1) 提供直接的跨时间步信息通路，绕过 RNN 式的逐层梯度传播，缓解梯度消失；(2) 允许当前 memory 选择性地关注历史中相关的 memory 状态（而非简单平均或全量传递）；(3) memory cache 仅存储 32-dim memory tokens per step，额外存储开销极小（128KB per step @ fp32）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory Retrieval 的计算流程：

```
class MemoryRetrieval(nn.Module):
    """单层 Cross-Attention Retrieval"""
    def __init__(self):
        self.cross_attn = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm = LayerNorm(1024)
    
    def forward(self, m_current, memory_cache):
        # m_current: [32, 1024]  ← 当前 Bridge Layer 输出的 memory
        # memory_cache: [32*K, 1024]  ← 所有历史 memory tokens 拼接
        
        # Cross-attention: query=当前memory, key/value=历史cache
        m_updated = m_current + self.cross_attn(
            query=self.norm(m_current),    # [32, 1024]
            key=memory_cache,               # [32*K, 1024]
            value=memory_cache              # [32*K, 1024]
        )
        # [32, 1024] × [32*K, 1024] → [32, 32*K] attention weights
        # → weighted sum over history → [32, 1024] updated memory
        
        return m_updated
```

检索过程的计算复杂度：O(32 * 32K * 1024) = O(M^2 * K * D) per step，其中 M=32 很小，K 随视频长度线性增长。总 Memory Retrieval 复杂度 O(M*K)。实践中 300s 视频（K≈75 at 4fps）的 retrieval 开销约 2ms，远小于 LLM 推理时间。

梯度流原理：
```
# 无 Retrieval (纯递归):
L → m_K → m_{K-1} → ... → m_1 (梯度逐层衰减，链长 K)
# 有 Retrieval (cross-attn shortcut):
L → m_K → [cross_attn: 直接读 m_1...m_{K-1}] (梯度有多条短路径)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用单层 Retrieval Layer（8 heads, hidden=1024），与 Bridge Layer 配置相同（Table 11）。实现上 memory cache 存储所有历史 memory tokens，但不需要反向传播通过整个 cache（仅通过 cross-attention 的 softmax 权重反向传播），因此避免了 BPTT 的长链问题。消融实验（Table 8）显示移除 retrieval 后 EgoSchema 性能下降 1.6 点（53.8→52.2），验证了 retrieval 对长视频理解的有效性。局限性：(1) memory cache 随视频长度线性增长（但 token 数极少），极长视频（>1000 segments）时自注意力计算可能成为瓶颈；(2) 论文未探索 retrieval 的 top-k 剪枝或稀疏注意力以进一步降低复杂度；(3) retrieval 机制引入了额外的训练参数（W_Q, W_K, W_V），增加了训练成本。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## Memory Tokens (Recurrent Video Understanding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Tokens 在 VideoLLaMB 中指在 Recurrent Memory Bridge Layers 中使用的可学习 token 向量（32 个，dim=1024），它们被 prepend 到每个视频语义段的视觉特征前，通过 self-attention 与视觉特征交互，逐步累积全视频信息。Memory tokens 的设计受 RMT (Recurrent Memory Transformer, Bulatov et al. NeurIPS 2022) 启发：RMT 将特殊 memory tokens 添加到 Transformer 输入/输出中以实现 segment-level recurrence，每个 segment 的 memory token 输出作为下一 segment 的输入。VideoLLaMB 将这一思想从纯 NLP 迁移到 video-language 设置，关键创新在于：memory tokens 在 vision-LLM bridge 位置运作（而非 LLM 内部），因此不会干扰 LLM 的语言理解能力；memory tokens 通过 self-attention 同时吸收视觉信息和历史记忆；最终 32 个 memory tokens 代表整个视频的压缩表示，与当前段视觉表示一起送入 LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory tokens 在 VideoLLaMB 中的生命周期：

```
# 1. 初始化
m_0 = nn.Parameter(torch.randn(32, 1024))  # 32个可学习token，随机初始化

# 2. 逐步更新 (遍历 K 个语义段)
for i = 1 to K:
    # Prepend memory to segment (32+C tokens total)
    input_seq = torch.cat([m_{i-1}, s_i], dim=0)  # [32+C, 1024]
    
    # Self-attention: memory 与 visual 交互
    attn_output = SelfAttn(Q=input_seq, K=input_seq, V=input_seq)
    # memory tokens attend to: (a) 其他 memory tokens, (b) 段内 visual tokens
    # visual tokens attend to: (a) memory tokens, (b) 段内其他 visual tokens
    
    # 分离 memory 和 visual
    m_i' = attn_output[:32]      # 更新后的 memory (含当前段信息)
    o_i = attn_output[32:]       # 段视觉表示 (含 memory 上下文)
    
    # Retrieval 增强
    m_i = CrossAttn(query=m_i', key=cache, value=cache)

# 3. 最终表示
video_representation = Concatenate(
    m_K,                          # 最终 memory: 全局视频压缩表示 [32, 1024]
    Projector(o_1, ..., o_K)      # 所有段视觉输出投影 [N_proj, LLM_dim]
)
# 送入 LLM: ~32+N_proj tokens (vs 原始 n×256 tokens)

# 内存分析:
# 原始方法: n frames × 256 patches × 1024 dim = n×256 tokens
# VideoLLaMB: 32 memory tokens + K×4 projected visual tokens
# n=320 frames, K=80 → 32 + 320 ≈ 352 tokens (vs 81920 tokens)
```

训练时 memory tokens m_0 作为可学习参数随 Bridge Layer 一起训练。每个 training step 中 16 帧分 4 段，memory tokens 跨 4 步递归更新。推理时可扩展到 320 帧（80 段），memory tokens 在跨段递归中持续累积信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用 32 个 memory tokens，每个 1024 维（与 ViT-L/14 输出维度一致）。初始化方式：随机初始化（nn.Parameter），随训练学习。参数量：32×1024 ≈ 33K 参数，可忽略。消融实验显示（Table 8）：(1) 使用 memory tokens only（移除 visual output o_i）→ 50.4%（-3.4 vs VideoLLaMB 53.8%），说明当前段视觉信息对 LLM 理解仍然关键；(2) 增加 memory tokens 数从 32→64（单层 Bridge）→ 53.0%（-0.8），可能因训练数据不足导致的过拟合；但 64 tokens + 3 层 Bridge → 54.6%（+0.8），说明更大容量需配合更深 Bridge 层。Memory tokens 的概念与 Perceiver (Jaegle et al. 2021) 的 latent array 和 BLIP-2 (Li et al. 2023) 的 Q-Former queries 有共通之处：都是通过少量可学习 token 压缩大量输入信息。VideoLLaMB 的差异在于 memory tokens 是递归更新的（而非一次性处理所有输入），使其能处理理论上无限长的视频流。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## TextTiling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TextTiling 是 Marti Hearst 在 1993 年 ACL 会议提出、1997 年在 Computational Linguistics 期刊上正式发表的文本分割算法。它是一种无监督、领域无关的算法，用于将说明性文本自动分割为连贯的多段子主题段落。核心思想基于词汇衔接理论（Halliday & Hasan, 1976）：当文本子主题变化时，显著比例的词汇也会随之变化。算法步骤：(1) Tokenization：将文本分为固定大小的"伪句"（pseudo-sentences，通常每 20 个 token），去除功能词；(2) 相似度计算：以滑动窗口（block size k，通常 k=6）计算相邻块间的余弦相似度，基于词频向量；(3) 边界识别：平滑相似度曲线，计算每个 gap 的 depth score（衡量相似度在 gap 两侧下降的幅度），将 depth score 最大的 gap 作为子主题边界。TextTiling 被集成在 NLTK 中，广泛应用于信息检索（段落检索）、文本摘要（选取每个子主题的代表句）和下游 NLP 任务（如词义消歧的上下文窗口选择）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TextTiling 算法伪代码（与 SceneTiling 对比）：

```
# === TextTiling (NLP) ===
# 输入: 文档 D = {w_1, ..., w_N} (词序列)
# 参数: w=20 (伪句大小), k=6 (block size)

# Step 1: Tokenization → 伪句
pseudo_sentences = group_by_size(remove_stopwords(D), w)  # 每组20个实词
# 输出: PS = {ps_1, ps_2, ..., ps_T}

# Step 2: Block 间余弦相似度
for i = 1 to T-1:
    block_left = TF_vector(ps_{i-k+1}, ..., ps_i)   # k个伪句的词频向量
    block_right = TF_vector(ps_{i+1}, ..., ps_{i+k})
    sim_i = CosineSimilarity(block_left, block_right)

# Step 3: Depth Score 计算
for i = 1 to T-1:
    d_i = (max_left_sim(i) + max_right_sim(i) - 2*sim_i) / 2
    # 识别相似度的"低谷"位置

# Step 4: 边界选取
boundaries = top_m_depth_scores(d)  # 选取 m 个最大 depth score 位置

# === SceneTiling (Video) ===  [本文，与 TextTiling 对比]
# 差异1: 用 ViT [CLS] token 替代 TF 词频向量
# 差异2: 逐帧计算而非伪句分组
# 差异3: Block size k 在视频域不适用，直接比较相邻帧
# 差异4: 阈值选取用 μ+ασ 而非固定 top-m 选取
```

TextTiling 的核心假设——"子主题变化 ≈ 词汇变化"——在视频域中被 SceneTiling 映射为"场景变化 ≈ 视觉特征 [CLS] token 变化"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TextTiling 在 Python 中可通过 NLTK 直接使用：`from nltk.tokenize import TextTilingTokenizer`。参数可配置：(1) w (pseudosentence size, default=20)；(2) k (block size, default=10)；(3) smoothing_width (default=2)；(4) cutoff_policy (hc/top/diff, 控制边界数量)。在 VideoLLaMB 中，SceneTiling 将 TextTiling 从文本域迁移到视频域，核心修改：(1) 将词频向量替换为 ViT [CLS] token 表示；(2) 将伪句概念替换为单帧；(3) 将 block-based 相似度替换为直接相邻帧相似度（因为视频的时序性远强于文本）；(4) 使用阈值 μ+ασ 替代固定 top-m 选取，使分割数自适应于视频内容。局限性：(1) TextTiling 假设线性子主题结构，不支持层级式主题组织（Hierarchical TextTiling 已提出但未用于 SceneTiling）；(2) 仅依赖词汇分布（或视觉特征），不利用语义理解或领域知识。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## NIAVH (Needle In a Video Haystack)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NIAVH (Needle In A Video Haystack) 是 VideoLLaMB 提出的长视频帧检索 benchmark，用于评估视频模型在超长视频中定位特定信息的能力。设计灵感来自 LLM 的 "Needle In A Haystack" (NIAH) 测试（将事实信息"针"插入长文档"干草堆"中测试检索能力），VideoLLaMB 将其扩展到视频多模态域。NIAVH 的 "haystack" 使用 Ego4D 数据集的 egocentric 视频拼接，长度从 1 到 320 秒；"needle" 支持三种模态：(1) 文本 needle——直接插入描述文本；(2) 图像 needle——使用 DALL-E 根据描述生成对应图像；(3) 视频 needle——使用 Sora 生成 1 秒短视频片段。Needle 插入到 haystack 的不同 depth（位置深度）和 length 组合中，评估模型回答"the young man seated on a cloud in the sky is doing what?" 的能力，LLM 对回答打分 1-10。NIAVH 区别于已有的 MM-NIAH（聚焦图像+文档混合 haystack）和 V-NIAH（纯合成视觉 benchmark），聚焦于流式视频 haystack 和多模态 needle 的组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NIAVH benchmark 的构建和评估流程：

```
# === NIAVH 基准构建 ===
# Haystack: 从 Ego4D 拼接 egocentric 视频
haystack = concat(random_ego4d_clips(total_duration=320s))
# 帧率: 1 fps → 320 frames

# Needle 生成 (基于同一描述 "young man sitting on cloud reading book"):
needle_text = "A young man is sitting on a piece of cloud in the sky, reading a book."
needle_image = DALL-E(needle_text)          # 图像模态 needle
needle_video = Sora(needle_text)            # 视频模态 needle, 1秒

# Insertion: 在指定深度插入 needle
depth = 12  # needle 位置在 haystack 的 12/40 ≈ 30% 处
haystack_with_needle[depth] = needle_video  # 替换1帧

# === 评估 ===
question = "What is the young man seated on a cloud in the sky doing?"
answer = model(haystack_with_needle, question)
score = LLM_judge(answer, ground_truth="reading a book")  # 1-10

# === 测试矩阵 ===
# X轴: 视频长度 (1-320s)
# Y轴: needle 深度 (1-40 intervals)
# 每个 (length, depth) 组合评估一次 → 热力图 (Figure 3)
```

VideoLLaMB 在 NIAVH 上的表现（Table 6, Figure 3d）:
- 320s video, depth=12: VideoLLaMB score=5.73, 推理时间=4.21s
- 对比: MA-LMM 3.39, PLLaVA 1.82, LLaVA-NeXT-Video-DPO 1.72
- 显示 VideoLLaMB 的 retrieval 机制有效保留了早期信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NIAVH 的实现依托 Ego4D、DALL-E 3、Sora 和 LLM judge。VideoLLaMB 开源代码中应包含 NIAVH 的构建和评估脚本。NIAVH 与同类 benchmark 的关系：(1) VNBench/VideoNIAH (Zhao et al. 2024, ICLR 2025)：独立的视频 NIAH 框架，支持 retrieval/ordering/counting 任务；(2) MM-NIAH (Wang et al. 2024)：多模态文档 haystack (1k-72k tokens)，支持文本+图像 needle；(3) MMNeedle (Wang et al. 2025, NAACL)：图像拼接 haystack，支持子图像检索；(4) V-NIAH (Zhang et al. 2024, LongVA)：纯合成视觉 NIAH benchmark。VideoLLaMB NIAVH 的独特优势在于支持视频 needle 模态（使用 Sora 生成）和流式视频 haystack，更贴近真实长视频理解场景。局限：(1) 当前仅单一 needle 类型和问题，缺乏多 needle、推理、counting 等多样化任务；(2) 仅用 Ego4D 作为 haystack 来源，可能引入领域偏见；(3) 评估依赖 LLM judge 的可靠性。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## Adaptive Pooling (Video Token Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Pooling（自适应池化）在视频语言模型中是一种非参数的视觉 token 压缩方法，通过 PyTorch 的 AdaptiveAvgPool3d 将任意长度的视频帧序列压缩为固定数量的视觉 token。典型实现（PLLaVA, Xu et al. 2024）：给定 n 帧视频经 ViT 编码后得到形状为 [n, H, W, D] 的特征张量，通过 AdaptiveAvgPool3d(target_shape=(T', H', W')) 池化到固定 token 数，使 LLM 输入 token 数恒定，无论原始视频多长。PLLaVA 使用 AdaptiveAvgPool3d 将视频特征池化到 [16, 12, 12] tokens（16 帧 × 144 = 2304 tokens per frame → pooled to 16×12×12）。此方法的优点：(1) 零参数，无需训练；(2) GPU 显存恒定；(3) 简单高效。缺点：(1) 平均池化丢失显著视觉细节（尤其是空间定位信息和细小物体）；(2) 当原始帧数 >> 目标帧数时，大量信息被平均化稀释；(3) 对指令内容无感知——池化与用户问题无关，可能丢弃问题关键帧。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Adaptive Pooling 在 PLLaVA 中的使用：

```
# === PLLaVA Adaptive Pooling Pipeline ===
# 输入: n frames (可变), ViT-L/14 encoder
# 输出: 固定 token 数的特征

# Step 1: ViT 编码
features = []
for frame in video_frames:  # n frames
    feat = ViT(frame)  # [257, 1024] (1 CLS + 256 patches)
    features.append(feat[1:])  # 丢弃 CLS, [256, 1024]
features = stack(features)  # [n, 256, 1024]

# Step 2: Reshape to 3D
features_3d = features.reshape(n, 16, 16, 1024)  # H=W=sqrt(256)=16

# Step 3: Adaptive 3D Pooling
# 目标: [T', H', W'] → 如 [4, 12, 12] 或 [16, 12, 12]
pooled = AdaptiveAvgPool3d(features_3d, target_size=(T', 12, 12))
# PLLaVA 训练: (16, 12, 12) → 16*12*12 = 2304 tokens
# PLLaVA 推理 (32帧): (16, 12, 12) → 仍 2304 tokens (从32帧池化)

# Step 4: 送入 Projector → LLM
tokens = Projector(pooled.reshape(16*12*12, 1024))
answer = LLM(Concat(tokens, text_tokens))
```

VideoLLaMB 的消融比较（Table 8）：
- Adaptive Pooling (PLLaVA): EgoSchema 45.6%
- Mean Pooling (uniform, 同 VideoLLaMB 设置): 51.61%
- VideoLLaMB: 53.8%
- Adaptive Pooling 甚至不如 Mean Pooling，可能因训练-推理不一致（训练时固定帧数池化与推理时自适应帧数池化的分布不匹配）

改进方向：PPLLaVA (ICLR 2026) 将 AdaptiveAvgPool3d 扩展为 Prompt-Guided Pooling——引入 CLIP 文本-视觉对齐模块计算 token 级 relevancy map，使用 relevancy 作为 3D 卷积权重进行加权池化，实现 18× token compression 同时保持 SOTA 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Adaptive Pooling 在 PyTorch 中通过 `torch.nn.AdaptiveAvgPool3d(output_size)` 实现，output_size 可以是元组 (T, H, W) 或 int（等维度）。在视频理解中，output_size 的选择是关键设计决策：较小的 output_size 减少 LLM 计算量但丢失更多信息；较大的 output_size 保留信息但增加计算成本。PLLaVA 使用 (16, 12, 12)=2304 tokens 作为默认值。Adaptive Pooling 也被 LLaVA-NeXT-Video（使用 position extrapolation + sampling）和其他 video LLM 用作 baseline。它的核心局限性——"对指令无感知的均匀压缩"——驱动了后续如 PPLLaVA（prompt-guided pooling）、VideoLLaMB（recurrent memory）等方法的提出。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges

## Native Sparse Attention (NSA) / 原生稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Native Sparse Attention (NSA, Yuan et al., 2025c) 是一种 learnable、hardware-aware 的稀疏注意力机制，替代 Transformers 中的标准 dense causal attention。NSA 不计算所有 query-key 对之间的 attention，而是对每个 query q_t 动态构建信息密集的 KV cache 子集。NSA 通过三个互补的支路实现稀疏化：(1) Token Compression Branch (CMP)：将连续 key blocks 通过 learnable MLP φ 聚合为粗粒度 block-level 表示，减少空间/时间冗余；(2) Token Selection Branch (SLC)：计算每个 KV block 的 importance score，选择 top-n 最重要的 blocks 保留细粒度信息；(3) Sliding Window Branch (SWA)：保留最近 w 个 KV pairs，确保局部上下文连续。三支路输出通过 learnable gate（两层 MLP + sigmoid）动态加权融合。NSA 的核心优势：数据依赖的稀疏性（data-dependent sparsity），即稀疏模式根据输入内容动态确定，而非使用固定的局部窗口/跨步 pattern。作为 hardware-aware 设计，NSA 的 block-level 分区与 GPU Tensor Cores 的 tile-based 计算对齐，在 H100 等硬件上可实现实际加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# NSA Attention Computation (per head, per timestep t)
# Input: q_t, K_{<=t} ∈ R^{t×d}, V_{<=t} ∈ R^{t×d}
# Hyperparams: block_size s, num_blocks n, window_size w

# Branch 1: Token Compression (CMP)
K_blocks = reshape(K[:t], [-1, s, d])
k_cmp = φ(mean(K_blocks, dim=1))         # MLP φ compresses block to vector
v_cmp = φ_v(mean(reshape(V[:t], [-1,s,d]), dim=1))
o_cmp = softmax(q_t @ k_cmp^T / sqrt(d)) @ v_cmp

# Branch 2: Token Selection (SLC)
p = importance_score(q_t, K_blocks)       # per-block importance
top_idx = topk(p, n)                      # top-n blocks
k_slc = gather(K_blocks, top_idx)
v_slc = gather(V_blocks, top_idx)
o_slc = softmax(q_t @ k_slc^T / sqrt(d)) @ v_slc

# Branch 3: Sliding Window (SWA)
k_swa = K[t-w:t]; v_swa = V[t-w:t]
o_swa = softmax(q_t @ k_swa^T / sqrt(d)) @ v_swa

# Dynamic Gating
g_cmp, g_slc, g_swa = sigmoid(MLP_gate(q_t))
o_t = g_cmp*o_cmp + g_slc*o_slc + g_swa*o_swa
```

Annotations: s=64 (block size), n=32 (selected blocks), w=256 (window) → total attention budget K_attn = 64×32 + 256 = 2304。在 L=128K context 下 γ = 2(K_attn)/(L-1) ≈ 3.6%。importance_score 通常为低秩近似。compression MLP φ: d→64→d (两层，SiLU activation)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NSA 原始实现在 FLA (Flash Linear Attention) 库（Yang & Zhang, 2024），基于 Triton 编写。VideoNSA 基于 FLA 并适配 SWIFT 训练框架（Zhao et al., 2024）。融合到 Video-LLM 时，vision tokens 使用 NSA（block size = 每帧 token 数），text tokens 保持 dense GQA。训练需联合优化 QKV + gate + compression MLP。开源：https://github.com/mdy666/Scalable-Flash-Native-Sparse-Attention (NSA 原始)；https://github.com/Espere-1119-Song/VideoNSA (VideoNSA)。

涉及论文标题：
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

## Hybrid Modality-Specific Sparse Attention / 混合模态特定稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Modality-Specific Sparse Attention 是 VideoNSA 提出的注意力分配策略：在同一 LLM decoder 的每层中，按 token 的模态（vision vs text）分配不同的注意力机制。Vision tokens 使用 NSA（三支路 learnable 稀疏注意力），Text tokens 使用标准 GQA（dense attention）。关键 insight：(1) Vision tokens 高度冗余——帧间大量重复信息，适合 aggressive sparse attention；(2) Text tokens 承载精确语义指令，dense attention 确保指令跟随不退化；(3) 分离两路 attention 避免 vision 的稀疏化噪声污染 text reasoning。每层输出：o = [o_V; o_T]。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VideoNSA Hybrid Attention (per layer)
X_V, X_T = split_by_modality(X, position_ids)
# Vision: NSA 3-branch
o_V = NSA_attention(X_V, block_size=64, n=32, w=256)
# Text: standard dense GQA
o_T = flash_attn(X_T, num_kv_heads=4)
# Merge
o = concat([o_V, o_T]); X_out = o + MLP(LayerNorm(o))
```

Annotations: 模态分离依据 position_ids。Vision block_size = 每帧 token 数（64 for Qwen2.5-VL），使每个压缩 block 对应完整一帧，归并时间冗余。28 layers 全部应用 hybrid attention（不加层选择）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现需修改 LLM decoder 的 attention forward：按 position_id 分离 token 序列 → vision path 调 NSA kernel → text path 调 FlashAttention kernel → 拼接。VideoNSA 基于 SWIFT + FLA 实现。这种 hybrid design 适用于任何视觉 token 冗余度高的多模态场景，对不同模态可选择不同 attention 策略（如音频用线性 attention，图像用 NSA）。

涉及论文标题：
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

## Low-frequency Temporal Allocation (LTA, 低频时间维度分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Low-frequency Temporal Allocation (LTA) 是 VideoRoPE 提出的 3D RoPE 频率分配策略，核心思想是将 Transformer head 的高维度（对应低频率 θ_n = β^{-2n/d}，即较宽的单调区间）分配给 temporal 维度，而将低维度（对应高频率，满足有限空间分辨率覆盖需求）分配给 spatial 维度（x 和 y）。这与 MRoPE 的分配策略相反——MRoPE 将低维（高频）分配给 temporal。LTA 的理论依据：(1) 空间维度受限于固定图像分辨率，高频足以覆盖所有空间位置的唯一编码；(2) 时间维度可无限增长（长视频），需要低频避免远距离位置产生"hash collision"——即 cos(θ_n·t) 在远距离上的周期性重复导致不同时间位置有相同 embedding。LTA 下 temporal 使用 θ_48..θ_63（d=128, β=10000），这些 θ 值极小（如 θ_63 ≈ 0.00011），在数千帧范围内几乎单调不减，确保不同时间的 temporal embedding 始终可区分。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VideoRoPE LTA dimension allocation (d=128)
# MRoPE: dims[0:32]=t(高频), dims[32:80]=x, dims[80:128]=y
# VideoRoPE LTA: dims[0:48]=x,y interleaved(高频), dims[48:64]=t(低频)

# Frequency comparison (β=10000):
# θ_n = base ** (-2*n / d)
# MRoPE t: θ_0≈1.0, θ_15≈10000^(-30/128)≈0.115
#          → cos(θ_n·t) oscillates within 3000 frames
# VideoRoPE t: θ_48≈10000^(-96/128)≈0.001, θ_63≈0.00011
#          → cos(θ_n·t) nearly monotonic within 3000 frames

def compute_videorope_rotation(q, k, t, x, y):
    # dims 0-47: spatial (x,y interleaved), higher freq
    # dims 48-63: temporal t, LOW frequency (LTA)
    q_rot = rotate_spatial(q[:,:,:48], x, y, freqs=θ[:24])
    q_rot = rotate_temporal(q[:,:,48:64], t, freqs=θ[24:32])  # LTA
    return dot(q_rot, k_rot) / sqrt(d)
```
Annotations: LTA 的核心 insight 是 RoPE 中不同维度的频率决定了捕捉的依赖范围——低维（高频）捕捉局部相对距离，高维（低频）捕捉全局长程依赖。VideoRoPE 将低频分配给需要长程建模的 temporal 维度，将高频分配给范围固定的 spatial 维度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LTA 的实现仅需修改 RoPE position embedding 的 dimension allocation：在调用 RoPE rotation 前重新排布各维度组。无需修改 RoPE 旋转计算本身或 Transformer 结构。使用时注意：(1) x 和 y 的交错排列（interleaved）优于顺序排列，因交错保持 x/y 维度频率相似性，减少空间各向异性；(2) temporal 维度数量和频率 base β 影响对不同视频长度的外推能力；(3) LTA 与 FlashAttention 等优化兼容。VideoRoPE 论文的 V-NIAH-D 结果（+12.44 点 over MRoPE）验证了 LTA 的有效性。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

## Diagonal Layout in Position Embedding (DL, 对角线布局)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diagonal Layout (DL) 是 VideoRoPE 提出的 3D 位置编码布局策略，将整个 multimodal 输入的 token 位置沿 3D 空间中的对角线排列。第 τ 帧视频的中心 patch 的 3D 坐标为 (τ, τ, τ)，其他 patch 按相对于中心的空间偏移排列——horizontal 偏移 ±(w-W/2)，vertical 偏移 ±(h-H/2)。text token 也沿同一对角线排列。核心目标是实现 spatial symmetry：T_v^start − T_pre = T_sub − T_v^end，使 visual token 从前后文本接收同等的上下文影响，减少模型对输入顺序的偏置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Diagonal Layout position index calculation
for τ in range(T_s + T_v + T_e):
    if τ < T_s:  # preceding text
        t, x, y = τ, τ, τ
    elif τ < T_s + T_v:  # video
        center = T_s + δ * (τ - T_s)
        t, x, y = center, center + w - W/2, center + h - H/2
    else:  # subsequent text
        t = τ + (δ - 1) * T_v
        x, y = t, t

# Example (Ts=10, W=H=4, δ=2, 2 frames):
# Frame 0 center: (10, 10, 10) → patch(w=0,h=0)=(10, 8, 8)
# Frame 1 center: (12, 12, 12) → patch(w=2,h=2)=(12, 12, 12)
# → All patch centers near line y=x=t (diagonal)
```
Annotations: DL 保持相邻帧间对应位置 index 增量与相邻 text token 增量一致（对角线方向），保持了 Vanilla RoPE 的索引模式。MRoPE 的 non-diagonal layout 导致 frame 0 所有 patch 共享相同 t 值，且每帧最后 token 总在 (W-1,H-1) 处形成 corner stack。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DL 实现仅需在构建输入序列时按对角线公式计算 position IDs。与标准 RoPE fine-tuning 兼容。DL 对 spatial symmetry 的保证有理论优势：简化位置关系学习，减少输入顺序偏置。在 VideoHallucer 的 Object-Relation Hallucination 子任务上 +DL 提升 18.0 点 over MRoPE，体现了 DL 对空间关系理解的增强。需注意 DL 依赖 (δ, W, H) 参数，训练推理时保持一致。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

## Adjustable Temporal Spacing (ATS, 可调时间间距)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adjustable Temporal Spacing (ATS) 是 VideoRoPE 提出的时间索引缩放机制，引入超参数 δ 控制视频帧间的 temporal index 间距。核心 insight：视频中相邻帧的时间间距与图像内相邻 patch 的空间间距本质不同——帧间变化通常更大且帧率可变。ATS 通过将 temporal index increment 设为 δ（默认 2），而 spatial/text index increment 保持 1，实现维度特定的编码粒度。公式：t = T_s + δ(τ-T_s)，使帧间 temporal jump = δ。消融实验（Table 6）：δ=0.5→56.34, δ=1.0→59.11, δ=2.0→60.92 (best), δ=3.0→59.18（三个 benchmark 平均）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ATS with δ=2
for τ in range(T_s + T_v + T_e):
    if τ < T_s:
        t_pos = τ
    elif τ < T_s + T_v:
        t_pos = T_s + δ * (τ - T_s)    # δ=2 temporal scaling
    else:
        t_pos = τ + (δ - 1) * T_v       # compensate offset
# Frame 0 to Frame 1: t jumps T_s→T_s+2 (jump=2)
# Text token 0 to token 1: t jumps 0→1 (jump=1)
```
Annotations: δ 过大（如 3.0）过度分散 temporal index 破坏与 spatial 维度的配合，δ 过小（如 0.5）压缩时间信息使相邻帧难以区分。最优 δ=2 表示相邻帧 temporal distance 为相邻 text token 距离的 2 倍，合理反映视频时间变化的粒度差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ATS 实现仅需在计算 3D position IDs 时引入 δ 因子，无需修改 Transformer 结构。ATS 与 LTA 和 DL 协同工作：LTA 决定 t 的频率范围，DL 决定 position 的布局方向，ATS 决定 t 的 index step size。δ 值应根据视频帧率和任务特性调节：高帧率可能需要更大 δ 保持时间区分度。VideoRoPE 推荐 δ=2 作为通用默认值。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

## ReAct / Think-Act-Observe Loop（推理-行动-观察循环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ReAct（Reasoning + Acting）是 2022 年 Yao et al. 提出的 LLM agent 范式，将推理（thought）与行动（action）交替交织。一个 ReAct step 包含三个子步骤：(1) Thought — LLM 基于当前 context（prompt + 历史 trajectory）推理当前状态、评估信息是否充足、规划下一步行动；(2) Action — LLM 选择并调用一个工具，指定工具名称和参数；(3) Observation — 工具执行结果被追加到 trajectory context。这三个子步骤构成的 triplet 被反复执行，直到 LLM 判断信息充足并调用 final answer action，或达到最大轮次限制。原始 ReAct 允许多工具并行调用，而 VideoSeek 约束为每轮仅一个工具，以避免 context 跳跃和过早终止。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VideoSeek 中 ReAct 风格的 agent 算法（Algorithm 1）：

```
Input: query Q, video X, system instruction I, thinking model θ, toolkit T, max_turns N

τ ← [I, Q]                    // trajectory 初始化
T ← T ∪ {answer}

for t = 1 to N:
    (z_t, a_t) ← θ(τ)         // thinking model 读完整 trajectory，输出推理 trace 和工具计划
    if a_t == [answer]:
        Y ← parseAnswer(a_t)
        break
    o_t ← callTools(a_t, X, T) // 执行工具，获取 observation
    τ ← τ ∪ [z_t, a_t, o_t]   // 追加到 trajectory

if Y == null:
    Y ← θ(τ + I_answer)       // 强制回答指令
return Y
```

VideoSeek 的关键变化：工具集为视频专用的 overview/skim/focus 三粒度工具；trajectory 的文本 token 数随轮次线性增长（LVBench 平均 49K tokens, 4.42 turns）。消融：去掉中间推理步骤（直接将 VideoSeek 选中的帧喂给单次 GPT-5）导致 LVBench 从 68.4% 降至 63.9%，说明多轮 reasoning 贡献约 4.5 pp 增益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) System prompt 定义 Thought/Action/Observation 格式和可用工具集；(2) 工具定义包含名称、用途、参数格式、使用约束；(3) Output parser 提取 action 并路由到对应工具执行器；(4) 视频帧的视觉解释由 LMM API 完成，工具返回的 observation 为文本描述。VideoSeek 的 prompt 分为六部分：Role, Environment, State, Workflow, Toolkit, Operational Rules。开源参考：github.com/jylins/videoseek（ReAct agent 框架 + video toolkit）。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Video Logic Flow（视频逻辑流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Logic Flow 是 VideoSeek 论文提出的核心概念，指视频中固有的时间顺序（temporal order）和因果结构（causal structure），包括场景转换、事件序列、叙事线索等。与逐帧密集解析依赖纯视觉信号不同，Video Logic Flow 利用视频的故事线逻辑来推断答案关键证据可能出现在哪里。例如，关于"记者离开部落时乘坐什么交通工具"的问题，逻辑流暗示出发事件应出现在视频接近结尾的部分而非开头。这一概念将视频本身的逻辑结构视为免费导航图，使 agent 可以按逻辑引导 exploration 而非盲目扫描。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Video Logic Flow 在 VideoSeek 算法 pipeline 中体现为三方面：

1. **工具选择层面**：overview 建立全局 storyline 后，agent 根据逻辑流推断答案可能出现在视频的哪个区间来指定 skim/focus 的搜索范围。例如 overview 输出显示 "1480s 两人在 mall 入口交谈"，agent 推断后续镜头转向高楼应在 1480s 之后，故指定 skim(1465s–1510s)。

2. **轨迹推理层面**：每轮 thought 显式使用 temporal order 和 causality 推理。系统 prompt 要求："Prefer internal video logic (temporal order/causality) over visual-only cues; use it to target relevant segments when frames are uninformative."

3. **实证证据**：字幕（subtitles）作为显式 textual storyline 揭示逻辑流。LVBench 加字幕后，VideoSeek 从 68.4%→76.7% 且帧数从 92.3→27.2，因为有了逻辑流后 agent 可极精确定位关键区间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现 Video Logic Flow 不需额外模型或特征提取——它是 prompt engineering 层面的设计。在 system prompt 的 Operational Rules 中明确指示 agent 使用时间/因果逻辑引导探索，并在每轮 thought 中显式推理基于已有 observation 下一步应关注哪个区间及其逻辑原因。局限性：论文承认对非结构化视频（如 anomaly detection）效果可能下降，因关键证据无法通过逻辑流预测。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Multi-Granular Video Toolkit（多粒度视频工具集）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Granular Video Toolkit 是 VideoSeek 设计的三个分层视频分析工具，分别在不同时间粒度上操作，实现从全局到局部的渐进式视频探索：(1) overview — 粗粒度全局扫描（均匀采样 16α 帧，每帧约 50 words 描述），构建视频 storyline 并标识潜在关键区间；(2) skim — 中粒度区间扫描（在候选长区间内均匀采样 4α 帧，每帧约 25 words 描述 + 约 50 words 高亮相关帧），快速确认 query-relevant 内容的位置，约束为区间长度至少 4α 秒；(3) focus — 细粒度密集检查（以 1 FPS 对短片段最多 4α 秒采样，直接回答 query 或返回 "No relevant content found"）。三个工具不是固定的 coarse-to-fine 流程，而是 agent 根据当前 trajectory 动态选择调用。α 为帧预算缩放因子（LVBench α=4，其余 benchmark α=2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

工具调用流程（LVBench case uid:860）：

```
Turn 1: agent 无先验 → overview()
  → 64 帧描述（α=4）："后期场景显示人物在车辆上离开村庄"

Turn 2: 推断离开在末尾 → skim(2800s-3148s, "Find departure moment")
  → 16 帧："3048.6s: 人物乘坐车辆离开；3098.3s: 标题卡片确认段落结束"

Turn 3: 需确认车辆类型 → focus(3044s-3056s, "Identify vehicle type")
  → 12 帧（1 FPS）："Pickup truck"
```

消融实验：移除 overview 导致最大性能下降（-13.3 pp），移除 skim（-6.0 pp），移除 focus（-4.7 pp），证明三者互补——overview 提供全局定位、skim 快速缩小范围、focus 精确验证细节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

每个工具对应一个 Python 函数，接受时间区间、query 和视频路径作为输入。帧采样用均匀采样（np.linspace），视觉解释调用 GPT-5 vision API，按工具特定 prompt 生成文本描述（如 overview 要求 JSON 格式 {"frames": [{"timestamp":..., "description":...}]}）。工具约束：每轮仅一个工具；overview 仅用于冷启动/全局问题；skim 区间至少 4α 秒；focus 区间最多 4α 秒。开源代码：github.com/jylins/videoseek。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Video Agentic Model（视频代理模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Agentic Model 是一类将视频理解任务形式化为 agent 任务的模型范式：不通过单次前向传播直接输出答案，而是通过多次迭代的 think-act-observe 循环，逐步收集证据、推理并生成最终答案。与传统的 standalone Video-LMM（如 GPT-4o、Gemini 1.5 Pro，固定帧数输入→单次推理→输出答案）不同，video agent 维护一个动态的 trajectory，每步根据此前累积的所有 observation 决定下一步工具调用。代表方法包括 VideoAgent (ECCV 2024, CLIP-based frame retrieval)、VideoTree (CVPR 2025, tree-structured search)、DrVideo (CVPR 2025, document-based retrieval)、DVD (NeurIPS 2025, multi-granular database)、VCA (ICCV 2025, curiosity-driven) 和 VideoSeek (CVPR 2026, logic-flow-guided seeking)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

两类 video agent 的算法 pipeline 对比：

**预建数据库型**（DrVideo, DVD, MR. Video）：
```
// 离线阶段（计算量正比于视频长度）
V_desc ← dense_parse(V, fps=0.2~2)  // 转为文本描述
DB ← index(V_desc)                    // 可检索索引

// 在线推理
relevant ← retrieve(DB, query)
answer ← LLM(relevant, query)
```

**主动探索型**（VideoSeek）：
```
// 无离线阶段，在线推理
τ ← [I, Q]
for t = 1 to N:
    (thought, action) ← LLM(τ)
    if action == answer: return
    observation ← execute(action, V)  // 按需获取
    τ.append(thought, action, observation)
```

关键差异：预建数据库型将感知与推理分离（感知固定在前处理阶段），VideoSeek 将二者交织——每步观察后更新对答案在哪的信念并动态调整探索方向，使 VideoSeek 可以用 1-5% 的帧数达到相当或更好的准确率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现 video agent 需要：(1) reasoning LLM（VideoSeek 用 GPT-5，也测试了 o4-mini 和 GPT-4.1）；(2) LMM/vision API 解释视觉内容；(3) 视频分析工具集；(4) trajectory manager 维护对话历史。VideoSeek 证明 reasoning model 选择至关重要——GPT-4.1（non-thinking）仅 53.0%（vs GPT-5 68.4%），因过早终止探索；框架是 model-agnostic 的——可替换任意 reasoning backbone。代码开源：github.com/jylins/videoseek。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Long-Horizon Reasoning in Video Understanding（视频理解中的长时域推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Long-Horizon Reasoning 指需要跨越较长推理步骤序列（multi-step reasoning trajectory）才能完成的复杂任务推理。在 video understanding 语境中，长时域推理要求模型：(1) 多次从视频中收集证据（而非单 pass 输入固定帧集）；(2) 每次新 observation 后重新评估已有 evidence 是否充足；(3) 基于累积的完整 trajectory（而非单一的 intermediate summary）进行推理。VideoSeek 将视频问答形式化为概率模型 p(τ, Y | X, Q) = p(τ | X, Q) × p(Y | X, Q, τ)，即先通过长时域探索构建 trajectory τ，再基于 τ 生成最终答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VideoSeek 的长时域推理 pipeline（LVBench uid:3105）：

```
初始 τ₀: [I, Q="镜头转向高楼后写了什么?"]

Turn 1: overview() → τ₁ 含全局 storyline
Turn 2: skim(1465-1510) → τ₂ 确认"1465-1497s: B1标识旁交谈; 1503s: 转向高楼"
Turn 3: focus(1499-1507) → τ₃ 确认"新年祝福语"但不够精确
Turn 4: focus(1502-1510) → τ₄ 精确读取"祝全市人民新春快乐"
Turn 5: answer → "D"
```

关键特性：(1) 完整 trajectory 作为 context（不 truncate 旧 observation），使 agent 可回溯之前发现；(2) 推理与探索交织——每步推理基于完整历史。论文分析指出：intermediate reasoning 贡献了 4.5 pp 增益（vs 将相同帧直接用于单次推理）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 维护完整对话 history 作为 trajectory，不做中间 summary；(2) 设置最大轮次限制（N=20）；(3) 若 N 轮内未触发 answer，使用 direct-answer 强制指令基于已有 evidence 回答；(4) trajectory 文本 token 数随轮次线性增长——LVBench 无字幕 49K tokens（4.42 turns）。与预建数据库方法的关键差异：tracking 状态存在于 trajectory text 本身而非预计算索引，因此状态可随新 evidence 动态调整。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Active Evidence Seeking（主动证据搜寻）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Active Evidence Seeking 是 VideoSeek 提出的视频探索策略，与密集贪婪解析（dense greedy parsing）对立。核心思想：agent 不应预先处理所有帧（exhaustive preprocessing），而应在推理过程中，基于当前已累积的 observation 推断下一步最 informative 的探索方向，仅按需检查少量帧。动机来自论文观察：LVBench 中超过 80% 的问题仅需检查不到 5% 的视频帧即可回答，因此 exhaustive parsing（如 DVD 的 8,074 帧）极其低效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

各 agent 的证据收集策略对比：

```
// Greedy Parsing（DVD, MR. Video）
frames = sample(V, fps=2)
descriptions = describe_all(frames)    // 无论 query
answer = LLM(descriptions, query)

// Active Evidence Seeking（VideoSeek）
while insufficient(τ, query):
    region = predict_region(τ, query)  // 基于 trajectory + 逻辑流
    if need_global: obs = overview(V)
    elif region_too_long: obs = skim(V, region, query)
    else: obs = focus(V, region, query)
    τ.append(obs)
```

关键差异：(1) Query-aware——工具选择和目标区间由 query 和 trajectory 决定；(2) Incremental——每步观察后重新评估充足性；(3) Logic-guided——利用时间顺序和因果缩小搜索空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现需要：(1) System prompt 明确定义评估 evidence 充足性的 thinking policy（"Before answering, list supporting evidence + timestamps... If insufficient, collect more"）；(2) 多粒度工具支持（overview/skim/focus）；(3) 完整 trajectory context（不压缩）。局限性：对需要检测意外事件（anomaly detection）的任务效果可能不佳，因关键 evidence 无法通过逻辑流预测位置。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking

## Attention Merging（注意力合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Merging 是 VisiPruner (EMNLP 2025) 提出的一种训练无关的视觉 token 压缩技术：在 MLLM 推理时，将某层的所有视觉 cross-attention weights 强制合并/聚焦到单个视觉 token 上，从而在保持 attention 分布稳定性（避免 softmax 归一化崩塌）的同时，将 N_text × N_v 的 cross-attention 计算降为 N_text × 1。具体操作：在 cross-attention matrix A ∈ R^{N_text × N_v} 上，对每一行 i（对应每个文本 token），将分散在所有视觉 token 上的 attention weights 求和后全部赋给一个随机选定的视觉 token k，其余位置的权重置零：A^{(l)}_{i,j} = Σ_{v∈V} A^{(l)}_{i,v} if j=k else 0。使用 attention merging 后，文本 token 不再需要与所有 N_v 个视觉 token 逐一交互，仅需与单个合并后的 representation 交互即可维持 attention 分布的数值稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Merging in VisiPruner (Shallow Layer 1)
# Input: Q_t [N_text, d], K_v [N_v, d], V_v [N_v, d]
# k: randomly selected visual token index

# Step 1: compute original cross-attention
scores = Q_t @ K_v.T / sqrt(d)        # [N_text, N_v]
A = softmax(scores, dim=-1)            # [N_text, N_v]

# Step 2: merge all attention weights to token k
A_merged = zeros_like(A)
for i in range(N_text):
    total_weight = sum(A[i, :])        # typically close to 1
    A_merged[i, k] = total_weight      # all weights -> single token

# Step 3: compute cross-attention output with merged attention
H_cross = A_merged @ V_v              # [N_text, d]
# 仅需 V_v[k] * total_weight，等价于用单个 token 的 value 加权
```

Annotations: k 可以是任意视觉 token index，VisiPruner 实验证实随机选择 token index=1/128/288/576 对性能几乎无影响（GQA 均保持 ~61.8），证明没有特定视觉 token 是关键的。该技术仅在 layer 1 发生——layer 2+ 可以直接跳过 cross-attention（system prompt 等文本 token 已可替代视觉 sink）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VisiPruner 在 LLaVA 推理代码中通过修改 HuggingFace LLaMA 模型的 attention mask 实现：在 layer 1 的 cross-attention 计算后，不直接使用 softmax 输出，而是先对 A 做"行求和→单列赋值"的后处理，再与 V_v 相乘。代码位于 GitHub repo `llava/cli_pruning.py` 中，通过配置 `shallow_mid_layer` 参数指定浅/中层分界。该技术的前提是经过 VisiPruner 的系统分析确认浅层视觉 token 仅起 attention sink 作用，因此可直接绕过——若无此分析直接使用可能破坏跨模态融合。Attention merging 可将浅层 cross-attention FLOPs 从 O(N_text × N_v) 降至 O(N_text)，在 LLaVA-v1.5 7B（576 visual + 74 text）中节省显著计算量。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs

## Influence-based Token Pruning（基于影响力的Token剪枝 / Value-aware Token Selection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Influence-based Token Pruning 是 VisiPruner 提出的中层视觉 token 选择方法，核心思想是：不依赖 attention weights（易受 attention sink 污染且分布分散），而是直接评估每个视觉 token mask 后对最后一个输入 token（决定首个回答 token）的 attention output 的改变程度——即该 token 对模型残差流的"实际影响力"。使用两个互补指标：(1) Cosine Similarity：衡量 mask 前后 attention output 的方向变化——cosine 越低说明该 token 对输出方向的贡献越大；(2) L2 Distance：衡量 mask 前后 attention output 的幅度变化——L2 越大说明该 token 对输出大小的贡献越大。当某层的最低 cosine similarity < 0.995 时，将该层定义为 filtering layer（跨模态融合开始的信号）；在 filtering layer 中，L2 distance < 0.2 的视觉 token 被丢弃（其对输出几乎无影响）。该方法在 LLaVA-v1.5 7B 上将 576 个视觉 token 压缩至平均 10.3 个，GQA 仅降 0.7%（62.0→61.3），远优于 attention-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Influence-based Token Selection at layer l
# Input: H_all [N_text + N_v, d], last_text_idx = -1
# Hyperparams: cosine_thresh=0.995, l2_thresh=0.2

Q, K, V = W_Q(H_all), W_K(H_all), W_V(H_all)
W = softmax(Q @ K.T / sqrt(d) + M)         # full attention weights
O = reshape(sum_over_heads(W @ V))          # [N_total, d]
O_last = O[last_text_idx]                   # attention output of last text token

for each visual token j:
    # Mask token j: set W'[last_text_idx, j] = 0 for all heads
    W_masked = W.clone()
    W_masked[last_text_idx, j, :] = 0
    O_masked = reshape(sum_over_heads(W_masked @ V))
    O_last_masked = O_masked[last_text_idx]
    
    cos_sim[j] = dot(O_last, O_last_masked) / (||O_last|| * ||O_last_masked||)
    l2_dist[j] = ||O_last - O_last_masked||_2

# Layer-level decision
if min(cos_sim) < 0.995:
    # This is a filtering layer — visual info starts contributing
    keep_mask = l2_dist >= 0.2
    H_v = H_v[keep_mask]                     # discard low-influence tokens
```

Annotations: 该方法在每层对所有视觉 token 逐个评估——每个 token 需要一次 masked attention forward（仅修改 attention weight 矩阵的一列），复杂度为 O(N_v × N_text × d)。但该方法仅在确定过滤层时执行一次（而非每层），因此在 LLaVA-1.5 7B 上平均仅需评估约 9-10 层中的一层。cosine < 0.995 是启发性阈值，从实验观察得出——对应视觉信息开始实质改变残差流方向的拐点。L2 < 0.2 同样来自实验 calibrate。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VisiPruner 在 PyTorch 中实现：通过 hook 拦截每层 attention 模块的 forward，对 attention weights 做 per-token masking 后重算 output。关键实现细节：(1) 仅需评估最后一个文本 token（idx=-1）对各视觉 token 的 cross-attention 影响——因为解码从该 token 开始，其 attention output 直接决定首个生成 token；(2) 在 attention weights 层面 mask（设 W'_{i→j}=0 across all heads），而非直接删除 token（后者会改变序列长度和 positional encoding）；(3) 评估在单层内完成——不需要 propagate 到后续层（与 leave-one-out 方法相比大幅节省计算）。该方法在 multiple MLLM architectures（LLaVA-v1.5 7B/13B, InternVL2.5 8B, Qwen2-VL 7B, MobileVLM-v2 3B）上验证有效，证明了 influence-based 选择优于 attention-based 选择的通用性。代码：https://github.com/EIT-NLP/VisiPruner。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs

## Differentiable Top-K Selection（DTS / 可微分Top-K选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Differentiable Top-K Selection (DTS) 是 VisionSelector 提出的可微分 Top-K token 选择机制。与传统 Gumbel-Softmax STE 通过随机采样实现可微分选择不同，DTS 通过 sigmoid 连续松弛 + 隐函数微分（implicit differentiation）实现确定性、单调的端到端梯度透传。Forward：给定重要性得分向量 s ∈ R^{B×N} 和保留数量 k，二分搜索阈值 t 使 Σ σ(s_i + t) ≈ k，得到 soft mask M = σ(s + t) ∈ (0,1)^N。由于 sigmoid 的严格单调性：s_i > s_j ⇔ M_i > M_j，保证高得分 token 获得更高 soft 权重，避免 Gumbel-Softmax 随机扰动导致的不稳定性和非单调性。Backward：在约束 Σ σ(s_i + t) = k 下隐式求导，得闭合形式梯度 ∂M/∂s = diag(v) − vv^T/Σv_i（v_i = M_i(1−M_i)），进一步化简为 ∂L/∂s = v⊙g − (v^T g/Σv_i)·v（g 为上游梯度）。推理时直接使用标准 Top-K 硬选择，无二分搜索或无额外开销。与 Gumbel-Softmax 的关键区别：(a) DTS 不需要随机噪声，forward 和 backward 均为确定性，(b) DTS 保持 scores→mask 的单调性，(c) DTS 通过 curriculum annealing 而非 temperature annealing 桥接训练-推理 gap。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === DTS Forward: 二分搜索 + sigmoid 连续松弛 ===
def DiffTopK_forward(s, k):
    # s ∈ R^{N} per batch, k = N * budget
    lower = -max(s) - 10   # sigmoid(-∞)=0 时保证全部 < k
    upper = -min(s) + 10   # sigmoid(+∞)=1 时保证全部 > k
    for _ in range(64):    # 64 次二分迭代达到充分精度
        mid = (lower + upper) / 2
        count = sum(sigmoid(s + mid))
        mask = (count < k)
        lower[mask] = mid[mask]     # 增大 mid → sigmoid 值增大
        upper[~mask] = mid[~mask]   # 减小 mid → sigmoid 值减小
    t = (lower + upper) / 2
    M_soft = sigmoid(s + t)  # ∈ [0,1]^N, 近似: sum(M_soft) ≈ k
    return M_soft

# === DTS Backward: 隐函数微分(闭式解) ===
def DiffTopK_backward(grad, s, t):
    M = sigmoid(s + t)
    v = M * (1 - M)       # σ'(s+t) = σ(s+t)(1-σ(s+t))
    v_sum = sum(v)
    uv = grad * v          # g ⊙ v
    uv_sum = sum(uv)
    grad_s = uv - (uv_sum / v_sum) * v  # 见论文公式(8)
    return grad_s          # ∂L/∂s 用于更新 LIS 参数

# === 对比: Gumbel-Softmax STE ===
# gumbel_softmax: z = softmax((logits + gumbel_noise) / τ)
# STE: backward = ∂L/∂z_hard ≈ ∂L/∂z_soft  (近似)
# DTS: backward = closed-form exact gradient  (精确)
```

Annotations: sigmoid: σ(x) = 1/(1+e^{-x})。二分搜索 64 次 = O(B×N×log range) 每 batch，相对 LLM forward 开销极小。显式梯度公式(8)源于论文的数学推导：对等式约束 Σ σ(s_i+t) = k 求全微分可得 ∂t/∂s_j = -v_j/Σv_i，代入 dM = σ'(s+t)·(ds + dt) 展开得到 ∂M/∂s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中实现 DTS：forward 通过 torch.sigmoid(s + t) 和二分搜索（无 autograd），backward 通过自定义 autograd.Function 注册上述闭式梯度。训练时 DTS 输出 soft mask M_soft ∈ [0,1]^N 与 token features 逐元素乘 V_pruned = M_soft ⊙ V；推理时完全跳过 DTS，直接 TopK(s, k) 得硬 mask。训练仅需在 LIS 模块上 (~12.85M 参数，Qwen2.5-VL-7B)，DTS 本身无可训练参数。二分搜索对任意 batch size 和 k 值为确定性操作，保证 sum(M_soft) 在给定精度内等于 k。VisionSelector 开源实现：https://github.com/JulietChoo/VisionSelector。与 Gumbel-Softmax 的性能对比：DTS 训练更稳定（无随机噪声从 forward 引入方差），梯度更精确（隐函数微分精确梯度 vs STE 近似），收敛更快（论文约 40 分钟 / 8 A800 训练完成）。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs

## Learnable Importance Scorer (LIS / 可学习重要性评分器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Importance Scorer (LIS) 是 VisionSelector 中用于评估视觉 token 全局重要性的轻量级模块（仅 12.85M 参数 / Qwen2.5-VL-7B）。LIS 解耦于 MLLM backbone，通过两层线性投影计算每个 token 的全局上下文感知重要性得分：给定视觉 token V ∈ R^{N×D}（N 个 token，D 维特征），投影为 Query Q = VW_q 和 Key K = VW_k（W_q, W_k ∈ R^{D×d}，d 为投影维度默认 1792 = D/2），计算简化自注意力矩阵 A = QK^T/√d ∈ R^{N×N}，每个 token 的重要性得分 s_i = (1/N)·Σ_{j=1}^{N} A_{ij}（全局平均池化）。该设计使 LIS 能同时感知所有 token 间的全局交互关系，而非仅依赖局部特征或预训练 attention map。训练时仅更新 LIS 参数（W_q, W_k），冻结 MLLM 全部参数，使用 near-zero initialization 确保初始训练稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === LIS 前向计算 ===
# 输入: V ∈ R^{N×D}, 来自 Vision Encoder + Projector
# 参数: W_q, W_k ∈ R^{D×d}, d << D (d=D/2)

V_norm = LayerNorm(V)           # 输入归一化
Q = V_norm @ W_q                # R^{N×d}
K = V_norm @ W_k                # R^{N×d}
A = Q @ K.T / sqrt(d)           # R^{N×N}, 全局 token 交互矩阵
s_i = mean(A[i, :])             # R^{N}, per-token 重要性得分

# 计算复杂度: O(N²·d), 远小于 LLM self-attention 的 O(N²·D)
# N ≈ 2000 visual tokens, d = 1792, D = 3584 (Qwen2.5-VL-7B)
# LIS FLOPs / LLM self-attn FLOPs ≈ d/D = 0.5 (单层对比)

# === 与 FastV/VisionZip 的关键区别 ===
# FastV: s_i = mean(text→vision attention scores)  (依赖 LLM 内部预训练 attn)
# VisionZip: s_i = mean(末层 vision encoder attn map)  (text-agnostic)
# LIS: s_i = mean(QK^T)  (独立于 backbone, 端到端学习)
```

Annotations: LIS 使用 Qwen2.5-VL-7B hidden_dim D=3584 的一半 (d=1792) 作为投影维度。在 Qwen2.5-VL-3B 上 d=1024 (D=2048/2)。在 LLaVA-OneVision-1.5-8B 上 d=2048。Near-zero initialization: W_q, W_k 初始化为接近零的小值，确保训练初期 s_i 接近均匀分布，LIS 不干扰 MLLM backbone 的预训练知识。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LIS 作为 plug-and-play 模块部署在 modality interface 与 LLM 之间：Vision Encoder → PatchMerger → Projector → **LIS** → DiffTopK/Mask → LLM。训练时与 MLLM backbone 完全解耦：LR=5e-5, AdamW, cosine annealing, 1 epoch (144K samples), 8 A800 GPUs, DeepSpeed ZeRO-3, 约 40 分钟完成训练。推理时 LIS 计算仅增加极小的开销（2×矩阵乘+QK^T+mean），与 FlashAttention 完全兼容。LIS 的全局交互设计使其能消除 attention sink 偏差：因为得分是从 LIS 自己学习的交互矩阵计算，而非依赖预训练 attention map 中首 token 的虚假高 attention。工作原理解释：LIS 在训练中通过下游 CE loss 学习识别对任务回答最关键的视觉 token，而非像 FastV 那样依赖可能存在 bias 的预训练 attention。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs

## Curriculum Annealing Strategy (CAS / 课程退火策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curriculum Annealing Strategy (CAS) 是 VisionSelector 提出的训练策略，用于桥接 soft token selection（训练时的 sigmoid soft mask）与 hard token selection（推理时的 Top-K binary mask）之间的 gap。总损失: L_total = L_CE + λ_t·L_constraint。其中 L_CE 是下游任务交叉熵损失，L_constraint = BCE(M_soft, M_hard) 衡量 soft mask 与 hard mask 的二值交叉熵（引导 M_soft 向 0/1 极化），λ_t 是动态权重系数。λ_t 从初始小值 λ_start 线性增加到最终值 λ_end：λ_t = λ_start + (λ_end − λ_start)·min(t_current/t_total, 1.0)。早期 λ_t 较小（如 0.1），模型优先学习下游任务；后期 λ_t 较大（如 2.0），强化 soft mask 向 hard mask 的逼近。这与 Gumbel-Softmax 的 τ (temperature) annealing 互补：CAS 操作损失权重空间，τ annealing 操作 softmax 平滑度空间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Curriculum Annealing Strategy (CAS) ===
# 参数: λ_start=0.1, λ_end=2.0, t_total (总训练步数)

for step t in 1..t_total:
    # 标准前向
    s = LIS(V)                    # 重要性得分
    M_soft = DiffTopK(s, k)       # soft mask ∈ (0,1)^N
    V_pruned = M_soft ⊙ V
    loss_ce = CE(LLM(V_pruned, text), targets)

    # 约束损失: 引导 M_soft 趋近 M_hard
    M_hard = standard_TopK(s, k)  # hard binary mask (无梯度)
    loss_constraint = BCE(M_soft, M_hard)

    # 动态权重
    λ_t = λ_start + (λ_end - λ_start) * min(t / t_total, 1.0)

    # 总损失
    loss_total = loss_ce + λ_t * loss_constraint
    loss_total.backward()
    optimizer.step()

# === 消融关键结论 ===
# Config 4 (λ_t = 固定 3.0, no annealing): Avg = 88.94%  (崩塌)
# Config 3 (λ_t = 0.1→3.0):                Avg = 95.37%
# Config 5 (λ_t = 0.1→2.0):                Avg = 95.75%  (更温和)
# VisionSelector (λ_t = 0.1→2.0):           Avg = 95.96%  (全局最优)
```

Annotations: 固定高 λ (Config 4) 导致模型过早被迫极化 token 得分而非学习下游任务，性能崩塌（88.94% vs 95.96%）。更温和的终点 λ_end=2.0 (vs 3.0) 进一步改善约 0.2pp。CAS 的核心是平衡"学习什么重要"和"学习二值化选择"两个有时冲突的目标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现完全在训练循环中：λ_t 随 step 线性插值，无需额外超参调度器。λ_start 和 λ_end 通过消融实验确定（λ_start ∈ {0.1}, λ_end ∈ {2.0, 3.0}）。与 Gumbel-Softmax τ annealing 可叠加使用但 VisionSelector 未采用 τ annealing——DTS 的 sigmoid 斜率固定，仅通过 CAS 调节 selection 硬度。CAS 的普适性：任何使用 soft-hard mismatch training 的场景（如可微分剪枝、可微分量化、可微分架构搜索等）均可使用类似策略。关键原则：早期让模型"理解任务"（低 λ），后期让模型"压缩精化"（高 λ）。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs

## Vision Exit in MLLMs（多模态大语言模型中的视觉退出）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vision Exit（视觉退出）是 VisiPruner 提出的 MLLM 深层优化策略：在 cross-modal fusion 完成后的某一层（称为 vision exit layer ℓ_exit），将所有保留的视觉 token 从后续层中移除，使深层仅做纯文本语言 refining。其核心发现是：MLLM 的深层（LLaVA-v1.5 7B 中约 layer 26+）已不再依赖视觉 token——跨模态信息已在中间层充分集成到文本表示中，继续保留视觉 token 不仅无益，反而引入噪声。实验证据：(1) 从 layer 26 起 discard 所有视觉 token，GQA 几乎不变（61.95→61.91）；(2) 但如果 skip layer 26 的视觉处理却继续在后续层处理视觉信息，性能反而下降（61.95→61.40）——说明 layer 26 的视觉处理本身在引入噪声；(3) mask 深层 cross-attention 对性能无影响（Fig. 3）；(4) 移除深层 KV cache 中的视觉信息性能略升（Tab. 2, MM-Vet 31.2→31.1 at layers 26-32）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Vision Exit Detection in VisiPruner
# After filtering layer, continuously track retained tokens' influence

filtering_layer_found = False
no_impact_counter = 0
vision_exit_layer = None

for l in range(S_mid + 1, L):
    if filtering_layer_found:
        # Track influence of currently retained visual tokens
        influence_score = compute_influence(H_v_retained, H_t)
        
        if influence_score < eps:  # negligible impact
            no_impact_counter += 1
        else:
            no_impact_counter = 0
        
        if no_impact_counter >= 2:
            vision_exit_layer = l  # exit here
            H_v = []                # remove ALL visual tokens
            break
    
    H = TransformerLayer(concat(H_v, H_t))
    # ... rest of forward pass (text-only from here on)

# Post-exit: pure language layers
for l in range(vision_exit_layer, L):
    H_t = TransformerLayer_text_only(H_t)
```

Annotations: ℓ_exit 通过"连续两层无 influence"的准则确定，避免单层波动导致过早退出。在 LLaVA-v1.5 7B 上 ℓ_exit 平均为 layer 23.9。退出后仅剩 text self-attention + FFN，compute 大幅减少。视觉退出与浅层 attention merging 互补：浅层减少 N_v×N_text attention，深层消除剩余的视觉 token 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vision exit 在 VisiPruner 推理代码中实现：在每层 forward 后检查保留视觉 token 的 cosine similarity / L2 distance 是否仍超过阈值，若连续两层都不超过则标记退出。该机制即插即用，不修改模型权重。适用于所有 decoder-only MLLM 架构（LLaVA、InternVL、Qwen2-VL 等），因为深层 focus on linguistic refinement 的现象是 MLLM 深层 encoder 行为的通用性质。可将 vision exit 机制嵌入 MLLM 训练 pipeline——如 VisiPruner Section 6 建议的"(c) Enable early exiting in deep visual layers once modality fusion is established"——通过在训练时加入 visual token dropout 或 exit loss 使模型学会自动决定退出时机。与"Confidence-Based Early Exit"（推理链提前终止）不同，Vision Exit 专指视觉 token 的退出，属于跨模态计算优化。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs

## Three-Stage Cross-Modal Interaction in MLLMs（多模态大语言模型三阶段跨模态交互）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VisiPruner (EMNLP 2025) 通过系统消融实验揭示的 MLLM 内部跨模态信息处理的三阶段规律。该框架将 MLLM 的 32 层 transformer 按功能分为三个阶段：(1) **Shallow Layers（浅层，layer 1-8）— Task Recognition**：视觉和文本 token 在浅层独立演化，无有意义的跨模态融合。文本 token 的 hidden state 经 vocabulary projection 显示浅层编码的是 task type（如"number"对应 counting、"type"对应 classification）而非视觉内容。视觉 token 在浅层仅作为 attention sink 稳定 softmax 分布，其 self-attention 也无实质作用——mask 后性能几乎不变。(2) **Middle Layers（中层，layer 9-23）— Sparse Cross-Modal Grounding**：跨模态融合在此阶段突然发生（abrupt onset），由少数关键视觉 token 驱动（~10/576）。这些关键 token 对应 instruction-relevant 的图像区域，且在不同中层间保持稳定（不需要每层重新识别）。然而 attention-based 方法无法准确选出这些 token——因为 attention sink 现象污染了 attention score 与信息贡献的对应关系。(3) **Deep Layers（深层，layer 24-32）— Linguistic Alignment**：视觉信息已集成到文本表示中，模型进入纯语言 refinement 阶段。深层将已识别的视觉答案（如"Lux"）组织为符合自然语言语法和流畅性的完整回答（如"The scene text is 'Luxmi Jewellers'"）。继续在深层处理视觉 token 会产生噪声，反而降低性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段在 MLLM 推理中的行为差异（以 LLaVA-v1.5 7B 为例）：
```
# Three-Stage Cross-Modal Processing in MLLM Inference
# N_v=576 visual tokens, N_t~74 text tokens, L=32 layers

# Stage 1: Shallow (layers 1-8) — Task Recognition
for l in 1..8:
    # Cross-attention exists but contributes NO visual content
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # cross-attn ≈ noise
    H_v = VisualSelfAttn(H_v) + FFN(H_v)            # self-attn ≈ redundant
    
    # Probing: project last text token's hidden state to vocab
    D_last = softmax(W_u @ H_t[-1])
    # Layer 7-10 outputs: "number", "type", "count" — task semantics, NOT visual

# Stage 2: Middle (layers 9-23) — Sparse Grounding  
for l in 9..23:
    # Cross-attention now fuses real visual info
    # BUT: only ~10/576 vision tokens matter
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # real fusion happens
    H_v = VisualSelfAttn(H_v) + FFN(H_v)
    
    # Top attended tokens consistently focus on instruction-relevant regions
    # e.g., "What kind of apple?" → tokens 107,108,222 (apple region)

# Stage 3: Deep (layers 24-32) — Linguistic Alignment
for l in 24..32:
    # Visual tokens no longer needed — can be safely discarded
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # cross-attn ≈ noise again
    # Probing: vocab projection shows "Lux" → "The" → "All" → grammatical refinement
```

Annotations: 阶段边界通过"mask cross-attention from shallow-to-deep / deep-to-shallow"的双向实验确定（Fig. 3）。Mask layer 1-7 跨模态 attention 性能仅轻微下降，但 mask layer 9-15 显著下降——标记中层开始。Mask layer 27-32 性能再次几乎不变——标记深层开始。该三阶段框架在 LLaVA-v1.5 7B/13B、InternVL2.5 8B、Qwen2-VL 7B、MobileVLM-v2 3B 上均验证有效，但各模型的阶段边界因架构不同而有所偏移（如 MobileVLM 3B 浅层更宽）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
三阶段框架的使用方式：(1) 分析工具：通过 semantic projection（W_u @ h_last, Eq. 2）和 value-output matrix projection（W_u @ V_last @ O, Eq. 3/9）观察每层的功能语义；(2) 剪枝指导：每阶段采用不同的压缩策略——浅层 attention merging + attention skipping，中层 influence-based token selection，深层 vision exit；(3) 训练指导：VisiPruner Section 6 基于三阶段提出三条 MLLM 训练建议：(a) 截断浅层视觉层数并消除 cross/self-attention；(b) 训练模型在中层进行稀疏注意力直接识别关键 token；(c) 在深层引入视觉退出机制。框架推翻了此前"浅层是跨模态融合主要场所"的共识（Wu et al., 2024; Zhang et al., 2025a），指出该共识因过度依赖 attention scores 作为信息流代理而产生误导——attention scores 在浅层的分布主要由 attention sink 决定而非指令相关性。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs

## Episodic Memory（情节记忆/事件记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Episodic Memory（情节记忆）是受认知心理学启发的记忆概念，在视频理解中指存储具体事件/片段发生的"事实性"文本记忆。在WorldMM中，Episodic Memory以多时间尺度的知识图谱集合形式构建：将长视频按不同时间分辨率T={t₀,t₁,...,t_N}（如30s, 3min, 10min, 1h）分别切分，每段生成caption后提取(entity, action, entity)三元组，为每个时间尺度构建独立的知识图谱G_{t_i}，最终形成记忆集合M_e={G_{t₀},...,G_{t_N}}。与Semantic Memory不同，Episodic Memory存储具体事件（"某天18:34分Shure把空调设到26度"），而非长期关系或习惯。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Episodic Memory构建与检索流程：
```
# === 构建阶段 ===
输入: 视频V, 时间尺度集合T={30s, 3min, 10min, 1h}
M_e = {}
for t_i in T:
    segments = split_video(V, duration=t_i)
    G_ti = empty_KG()
    for seg in segments:
        frames = sample_frames(seg, fps=0.5)
        transcript = Whisper.transcribe(seg)
        caption = VideoLLM(frames, transcript)
        triplets = LLM.extract_triplets(caption)  # [(e1,action,e2)]
        for (subj, pred, obj) in triplets:
            G_ti.add_edge(subj, obj, relation=pred)
    M_e[t_i] = G_ti

# === 检索阶段 (Coarse-to-Fine) ===
candidates = []
for t_i in T:
    ppr = PersonalizedPageRank(G_ti, seed=extract_entities(q))
    top_k_nodes = argsort(ppr.scores)[:k]
    candidates += [(t_i, G_ti.get_captions(node)) for node in top_k_nodes]
top_m = LLM.cross_scale_rerank(q, candidates)  # prompt见Fig.13
return top_m
```
关键计算：PPR迭代 s=α·A^T·s+(1-α)·s₀，收敛后s[i]为节点i的PPR分数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖Video LLM生成caption + LLM提取三元组（prompt模板见WorldMM Fig.10-11）+ HippoRAG的PPR检索框架。多时间尺度根据视频总长设定：周级别{30s,3m,10m,1h}，小时级{10s,30s,3m,10m}。固定单尺度替代多尺度在WorldMM消融中导致6.1%精度下降。相关概念也出现在HERMES（同样区分episodic/semantic memory）和EgoRAG（层级事件记忆但仅单尺度）中。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
- HERMES__temporal-coHERent_long-forM_understanding_with_Episodes_and_Semantics

## Semantic Memory Consolidation（语义记忆整合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Memory Consolidation是WorldMM中维护持续演化的长期语义知识图谱的增量更新机制。捕获跨场景的抽象知识——人际关系、行为习惯、偏好等，区别于Episodic Memory的具体事件存储。Consolidation过程：新视频段到达时，先用embedding相似度(c >0.6)检测新三元组与已有图谱的重叠/冲突，再交LLM判断哪些旧三元组应删除(T_remove)、哪些应新建或修改(T_update)，执行G_new=(G_old\T_remove)∪T_update。避免纯append式记忆膨胀和冲突信息共存（如"dislikes sweet food" vs "likes sweet desserts"）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Consolidation流程：
```
输入: 当前语义图G_s, 新语义三元组T_new
for triplet_new in T_new:
    e_new = embed(triplet_new)
    matches = [(t_old, sim) for t_old in G_s 
               if cosine(embed(t_old), e_new) > 0.6]
    if matches:
        # LLM决策 (prompt Fig.15)
        result = LLM.consolidate(triplet_new, matches)
        G_s = (G_s - result["triples to remove"]) ∪ {result["updated triple"]}
    else:
        G_s.add(triplet_new)
```
实例：新"[I, uses WeChat for, money transfers]"与已有"[I, uses WeChat to send money]"合并；冲突"[Lucia, dislikes, overly sweet food]"替代"[Lucia, likes, sweet desserts]"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用LLM执行合并决策，embedding模型编码三元组文本。检索用PPR边级评分(edge_score=ppr(u)+ppr(v))取top-10三元组。去除Consolidation在HabitInsight类别上导致约7%精度下降。灵感来自认知科学中的记忆巩固理论。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning

## Multi-scale Temporal Knowledge Graph（多尺度时序知识图谱）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-scale Temporal Knowledge Graph是WorldMM中Episodic Memory的核心数据结构——对同一视频按不同时间粒度分别构建多个知识图谱。时间粒度T={t₀<t₁<...<t_N}，如周级视频{30s,3min,10min,1h}，小时级{10s,30s,3m,10m}。每个G_{t_i}是从粒度t_i的caption提取(entity,action,entity)三元组构成的图，实体为节点、动作为边。多尺度设计动机：不同query需不同时间跨度——"Where did I leave my glasses?"需秒级，"What happened in the match second half?"需十分钟级。单尺度要么冗余太多，要么信息不足。检索时从所有尺度并行召回后用LLM cross-scale reranker动态选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
T = {30s, 3min, 10min, 1h}  # EgoLifeQA配置
for t_i in T:
    for seg in partition(V, t_i):
        cap = VideoLLM.caption(seg)
        triplets = LLM.extract(cap)
    G_ti = build_graph(triplets)

# 检索: coarse-to-fine
all_candidates = []
for t_i in T:
    ppr = PPR(G_ti, seed=query_entities)
    all_candidates += [(t_i, cap) for cap in top_k(ppr)]
# LLM联合评估选择: 具体事件→偏好细粒度; 习惯/关系→偏好粗粒度
top_m = LLM.rerank(query, all_candidates)
```
粗粒度caption由LLM通过合并、摘要提示构造（prompt Fig.12）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
时间尺度选择根据视频总长设定。扰动实验({20s/2m/5m/50m}→65.2%, {30s/3m/10m/1h}→65.6%, {1m/5m/15m/1.5h}→64.8%)证明对精确值鲁棒，收益来自多尺度设计本身。固定单尺度导致6.1%精度下降。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning

## Personalized PageRank (PPR) Graph Retrieval（个性化PageRank图检索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Personalized PageRank (PPR) 是PageRank的扩展，指定个性化起始节点集合(seed nodes/teleport set)使随机游走偏向它们，计算图中每个节点相对于query的"个性化重要性"分数。WorldMM用于知识图谱检索：将query提取的实体节点设为seed(teleport概率偏高)，PPR迭代至收敛，节点分数s[i]为与seed的关联强度。Episodic Memory检索：PPR→节点→关联caption候选。Semantic Memory检索：边得分=两端节点PPR之和(edge_score=ppr(u)+ppr(v))，取top-k边对应三元组。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: KG邻接矩阵A[N,N], seed节点S, α(通常0.85)
s = [1/|S| if i in S else 0 for i in 0..N]
while not converged:
    s_new = α * A^T @ s + (1-α) * s_init
s = s_new

# Episodic Memory: 节点检索
caps = [G.get_caption(node) for node in argsort(s)[:k]]

# Semantic Memory: 边检索
edge_scores = {edge: s[u]+s[v] for edge=(u,v) in G_s}
top_triplets = argsort(edge_scores)[:10]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于HippoRAG框架(github.com/OSU-NLP-Group/HippoRAG)。PPR相对标准PageRank优势：teleport bias使结果与query相关而非仅全局重要性。Embedding检索替代PPR导致4.4%精度下降，验证了图结构检索优于纯相似度检索。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning

## KV Cache (Key-Value Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache（Key-Value Cache）是 Transformer 模型在自回归（autoregressive）推理中用于缓存历史 token 的 Key 和 Value 投影张量的显式记忆机制。其设计动机是避免每次生成新 token 时对历史所有 token 重复计算 Key/Value 投影。在标准自回归推理中，每个 decoding step 仅新生成一个 token，若缓存历史 K/V，则 attention 计算只需对当前 Query 与 [历史 K, 新 K] 和 [历史 V, 新 V] 执行一次矩阵乘法，将计算复杂度从 O(T²d) 降至 O(Td)（T 为序列长度，d 为隐维度）。

KV Cache 的生命周期：
1. **Prefill 阶段**：输入 prompt 的完整 token 序列并行通过 Transformer，所有层的 K、V 被计算并存入 cache。
2. **Decode 阶段**：每生成一个新 token，仅计算该 token 的 Q/K/V，将新 K/V 追加到 cache，Q 与完整 cache 执行 attention。
3. Cache 大小 = 2 × L × T × H × d_head × precision_bytes（L=层数，H=head数）。对于 1.2B 参数的 StreamVGGT，每帧产生 (1+R+N) 个 tokens，cache 随帧数线性增长至 OOM。

KV Cache 在 LLM（文本 token，逐个追加）和 Vision Transformer（视觉 token，逐帧批量追加）中的增长模式不同：LLM 每步追加 1 个 token，StreamVGGT 每帧追加数百个 patch tokens，cache 膨胀速度远快于 LLM 场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
StreamVGGT 中 KV Cache 在 temporal attention 中的计算流程：

```
# 第 t 帧推理时，第 ℓ 层的 temporal attention
# 输入: Q_t^(ℓ) (当前帧 Query), Cache = {K_{1:t-1}^(ℓ), V_{1:t-1}^(ℓ)}

# 1. 计算当前帧的 K, V
K_t^(ℓ) = H_t^(ℓ) · W_K^(ℓ)      # 新计算，shape: (1+R+N) × C
V_t^(ℓ) = H_t^(ℓ) · W_V^(ℓ)

# 2. 拼接历史 cache 和当前帧
K_all = concat(K_{1:t-1}^(ℓ), K_t^(ℓ))   # shape: (t·(1+R+N)) × C
V_all = concat(V_{1:t-1}^(ℓ), V_t^(ℓ))

# 3. Temporal causal attention
Out_t = FlashAttn(Q_t^(ℓ), K_all, V_all, causal_mask=True)

# 4. 追加当前帧 KV 到 cache
Cache.K_{1:t}^(ℓ) ← concat(Cache.K_{1:t-1}^(ℓ), K_t^(ℓ))
Cache.V_{1:t}^(ℓ) ← concat(Cache.V_{1:t-1}^(ℓ), V_t^(ℓ))
```

Cache 大小增长：t=1 时 ~(1+R+N)×2L 个张量元素，t=T 时 ~T×(1+R+N)×2L。无压缩时，T=300 帧的 StreamVGGT 在 80GB A100 上 OOM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中通过 Python list 或 pre-allocated tensor 管理，在每层 attention 计算前后：`past_key_values` (HuggingFace Transformers 的 `DynamicCache` 或 `StaticCache`)。vLLM 通过 PagedAttention 将 cache 分块管理（block size 16/32），TGI 通过 FlashAttention-compatible 的 block table 管理。对于多模态模型，视觉 encoder 输出的 patch tokens 在 LLM 的每一层也会产生对应的 KV cache 条目。XStreamVGGT 在每层 temporal attention 后对 cache 执行 pruning + quantization。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## KV Cache Pruning（KV Cache 剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Pruning 是一种通过评估历史 KV cache 中每个 token 的重要性，选择性丢弃低重要性 token 的 cache 来约束内存使用的技术。核心思路：并非所有历史 token 对当前/未来帧的 attention 计算同等重要——大量 token（尤其是视觉 patch tokens）携带冗余信息（spatial-temporal redundancy），可以在保持模型性能的同时被安全移除。

剪枝的关键要素：
- **重要性度量**：常用指标包括 attention scores（accumulated/current）、token saliency、QK 相似度、或基于 MLP 的 learned scorer。
- **剪枝粒度**：token-level（丢弃单个 token）、channel-level（丢弃特定 channel 的 KV 条目）、layer-level（不同层不同剪枝率）。
- **剪枝时机**：one-shot（prefill 后一次性剪枝，后续步骤用固定 cache）vs dynamic（每个 decoding step 重新评估和剪枝）。

XStreamVGGT 的剪枝机制特点：使用 query-guided 重要性评分（对 Query 分组池化后与 Key 计算内积），而非直接读取 attention scores，以保持与 FlashAttention 的兼容性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中的 query-guided KV cache pruning：

```
# 输入: Q_t^(ℓ) ∈ R^{(1+R+N)×C} (当前帧 Query)
#       Cache.K_{1:t}^(ℓ), Cache.V_{1:t}^(ℓ) (完整 KV cache)
#       L_max (缓存预算，如 2K tokens)
#       g = 16 (分组大小)

# 1. Query 分组池化
Q_special = Q_t^(ℓ)[:1+R, :]           # camera + register tokens
Q_normal  = Q_t^(ℓ)[1+R:, :]            # patch tokens
Q_pooled  = concat(Q_special, GroupAvg(Q_normal, g))
Q̄ = mean(Q_pooled, dim=heads)          # 跨 head 平均，shape: N_pooled × C

# 2. 提取中间帧 prunable keys（排除首帧和当前帧）
K̄_prunable = mean(K_{first+1 : t-1}, dim=heads)  # shape: T_prunable × C

# 3. 计算 token 重要性分数
S_matrix = Q̄ @ K̄_prunable^T              # QK 内积
S = mean(S_matrix, dim=query)             # 沿 query 维平均

# 4. Top-k 选择（保留首帧 + 当前帧 + 高分中间 token）
I_middle = TopK(S, k = L_max - T_first - T_current)
I_keep = {1..T_first} ∪ I_middle ∪ {T-T_current+1..T}

# 5. 同步剪枝 K 和 V
Cache.K = Cache.K[I_keep]
Cache.V = Cache.V[I_keep]
```

与文本 LLM 中的 KV pruning（如 H2O, SnapKV）的区别：后者通常基于 accumulated attention scores 选择 "heavy hitter" tokens，而 XStreamVGGT 使用 query-guided Q̄K̄^T 内积，适配 vision token 的 spatial-temporal 冗余特性，且 pooling 设计兼容 FlashAttention。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 原生实现：在每层 attention 计算后调用 pruning 函数，返回 mask indices 后通过 `torch.index_select` 或直接索引裁剪 cache tensor。与 FlashAttention 的兼容性要求 pruning 不能依赖 attention scores 的中间结果——因此使用独立的 Q̄K̄^T 计算（额外开销小）或基于 hidden state 变化（如 L2 distance）的方法。kvpress（HuggingFace）库提供统一的 KV cache 压缩接口。XStreamVGGT 码：https://github.com/ywh187/XStreamVGGT/。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## KV Cache Quantization（KV Cache 量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Quantization 是将高精度（通常 FP16/BF16）存储的 Key 和 Value 张量压缩到低比特整数表示（如 INT4、INT2）的技术，以减少 KV cache 的内存占用。与权重量化不同，KV cache 量化是在**推理时动态执行**的——每生成新 token 后，新的 K/V 需要即时量化存储，attention 计算前需反量化回浮点精度。

关键挑战：
1. **Outlier 问题**：K tensors 中存在显著的 channel-wise outliers（少数 channel 值远大于其他 channel），如果使用 per-tensor 或 per-token 量化，这些 outliers 会主导 scale factor，导致大量有效精度损失。
2. **分布不对称**：K 和 V 具有不同的分布特性——K 的 channel-wise outlier 更显著，V 分布相对均匀。
3. **动态范围变化**：随序列增长，KV cache 中的数值分布可能漂移。

XStreamVGGT 通过发现 StreamVGGT 中 K 的 channel-wise outlier 和 V 的相对均匀分布，提出了 per-channel K + per-token V 的维度自适应量化方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中的量化流程（基于 KIVI，INT4，group size 64）：

```
# 量化存储（pruning 后对保留的 cache 执行）
# 对 Key 使用 per-channel 量化
for c in range(C):  # 每个 channel 独立
    K_c = Cache.K[:, c]                          # shape: T_keep
    s_c = (max(K_c) - min(K_c)) / (2^4 - 1)      # scale (INT4)
    z_c = round(-min(K_c) / s_c)                  # zero-point
    K̂_c = clamp(round(K_c / s_c) + z_c, 0, 15)   # 量化值

# 对 Value 使用 per-token 量化
for i in range(T_keep):  # 每个 token 独立
    V_i = Cache.V[i, :]
    s_i = (max(V_i) - min(V_i)) / (2^4 - 1)
    z_i = round(-min(V_i) / s_i)
    V̂_i = clamp(round(V_i / s_i) + z_i, 0, 15)

# Attention 计算时反量化
K_deq = (K̂ - z_c) * s_c     # INT4 → FP16
V_deq = (V̂ - z_i) * s_i
Out = FlashAttn(Q_t, K_deq, V_deq)
```

内存节省（以 StreamVGGT 为例）：FP16 每元素 2 bytes → INT4 每元素 0.5 bytes（4× 压缩），加上 scale 和 zero-point metadata（per-channel K: C × 2 × 2 bytes ≈ 小开销），总计约 4× 内存减少。配合 pruning（cache 从无界到 2K tokens），总计 4.42× 内存减少。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
常用实现：
- **KIVI** (Liu et al., 2024)：asymmetric uniform quantization，支持 INT2/INT4，per-channel + group-wise 量化，tuning-free。本论文采用的方法。
- **KVQuant** (Hooper et al., 2024)：per-channel + per-group with dense-and-sparse，支持 3-bit/4-bit。
- **GEAR** (Kang et al., 2024)：codebook-based quantization with residual compensation。
- **Atom** (Zhao et al., 2025)：hardware-efficient INT4 with per-token group quantization。
- **RotateKV**：通过 Hadamard rotation 平滑 outlier 后执行 per-channel 量化。

PyTorch 中通过 `torch.quantize_per_channel` / `torch.quantize_per_tensor` 或自定义 kernel。vLLM 通过 `--kv-cache-dtype fp8` 或 `--quantization kv-cache` 自动启用。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Alternating-Attention（交替注意力机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Alternating-Attention 是 VGGT (Visual Geometry Grounded Transformer) 和 StreamVGGT 中使用的核心 Transformer 架构设计，由 Wang et al. (CVPR 2025) 提出。每层 Transformer block 中包含两种交替执行的 attention 操作：(1) **帧内空间自注意力（intra-frame spatial self-attention）** 对单帧内的所有 token（camera token + register tokens + patch tokens）执行标准 self-attention，捕获帧内空间结构；(2) **时序因果注意力（temporal causal attention）** 以当前帧 token 为 Query，跨所有历史帧的 token 为 Key/Value 执行 causal attention，聚合时序信息。

交替设计的动机：同时建模空间结构（同一帧内 patch 之间的关系）和时序动态（不同帧之间的运动/变化），而非将两种 attention 融合在一起（如 TimeSFormer 的 divided space-time attention），使每层可以独立优化空间和时序的 token 交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
StreamVGGT 中 Alternating-Attention 的计算流程（每层 ℓ）：

```
# 输入: Input_t^(ℓ-1) ∈ R^{(1+R+N)×C} (当前帧 token 序列)
#       Cache = {K_{1:t-1}^(ℓ), V_{1:t-1}^(ℓ)} (历史帧 KV cache)

# === Phase 1: 帧内空间自注意力 (Intra-frame Spatial Self-Attention) ===
Q_spatial = Input_t^(ℓ-1) · W_Q_spatial^(ℓ)
K_spatial = Input_t^(ℓ-1) · W_K_spatial^(ℓ)
V_spatial = Input_t^(ℓ-1) · W_V_spatial^(ℓ)
H_spatial = Softmax(Q_spatial · K_spatial^T / √d) · V_spatial
# 注：此处无 KV cache 参与，仅当前帧内 token 交互

# === Phase 2: 时序因果注意力 (Temporal Causal Attention) ===
Q_temporal = H_spatial · W_Q_temporal^(ℓ)     # 使用 spatial 输出
K_temporal = H_spatial · W_K_temporal^(ℓ)
V_temporal = H_spatial · W_V_temporal^(ℓ)

# 拼接历史 cache
K_all = concat(Cache.K_{1:t-1}^(ℓ), K_temporal)
V_all = concat(Cache.V_{1:t-1}^(ℓ), V_temporal)

# Causal attention（mask 防止访问未来帧）
Out_t^(ℓ) = FlashAttn(Q_temporal, K_all, V_all, causal_mask=True)

# 仅 temporal attention 维护 KV cache
Cache.K_{1:t}^(ℓ) ← K_all
Cache.V_{1:t}^(ℓ) ← V_all

# XStreamVGGT: 在此处对 temporal cache 执行 pruning + quantization
```

设计约束：仅 temporal global attention 模块维护 KV cache → 仅该部分受 cache growth 影响 → KV cache pruning 仅需处理 temporal attention 的 cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VGGT 基于 DINOv2 ViT-L backbone（24 layers），在其上构建 Alternating-Attention 架构。StreamVGGT 通过将 temporal attention 改为 frame-wise causal 版本适配流式推理。PyTorch 实现中 spatial attention 使用 `torch.nn.MultiheadAttention` 或 FlashAttention，temporal attention 额外传入 `attn_mask` (causal) 和 `past_key_value` (cache)。XStreamVGGT 在此基础上插入 pruning + quantization 逻辑。代码开源：VGGT (https://github.com/facebookresearch/vggt)，StreamVGGT (https://github.com/DongZhuo/StreamVGGT)。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Frame-wise Causal Attention（逐帧因果注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame-wise Causal Attention 是 StreamVGGT 将 VGGT 的全局 Alternating-Attention 改造为适合在线流式推理的注意力机制。核心改动：将 temporal attention 中的全局 attention（可访问所有帧的所有 token）替换为因果掩码的逐帧注意力——当前帧 t 的 Query 只能 attend 到帧 1..t 的 Key/Value，不能访问未来帧 t+1..T。

这与 LLM 的 autoregressive causal attention 哲学相似，但粒度不同：
- LLM：token-level causal（每个 token 只能看到前面的 tokens）
- StreamVGGT：frame-level causal（每帧可以看到当前及之前的帧，帧内 tokens 可以互见）

frame-wise 处理（而非 token-wise）的原因：视觉输入是逐帧到达的图像数据，每帧产生一批 patch tokens，帧内 spatial attention 无因果限制（帧内所有 patch 同时可得）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Frame-wise causal attention mask 结构
# 假设3帧，每帧2个token (R+N tokens 简化表示)
# Mask[t_i, t_j] = 0 if frame(i) ≤ frame(j) else -inf

# 序列: [f1_tok0, f1_tok1, f2_tok0, f2_tok1, f3_tok0, f3_tok1]
# Causal Mask (0=allow, -inf=block):
#   f1_tok0 f1_tok1 f2_tok0 f2_tok1 f3_tok0 f3_tok1
#   [0,      0,     -inf,   -inf,   -inf,   -inf  ]  # f1_tok0
#   [0,      0,     -inf,   -inf,   -inf,   -inf  ]  # f1_tok1
#   [0,      0,     0,      0,      -inf,   -inf  ]  # f2_tok0
#   [0,      0,     0,      0,      -inf,   -inf  ]  # f2_tok1
#   [0,      0,     0,      0,      0,      0     ]  # f3_tok0
#   [0,      0,     0,      0,      0,      0     ]  # f3_tok1

# 实现：按帧边界构建 block-wise causal mask
def frame_causal_mask(num_frames, tokens_per_frame):
    total = num_frames * tokens_per_frame
    mask = torch.zeros(total, total)
    for i in range(num_frames):
        for j in range(num_frames):
            if j > i:
                # 未来帧 → 该帧内所有 token 对后续帧不可见
                mask[i*tokens_per_frame:(i+1)*tokens_per_frame,
                     j*tokens_per_frame:(j+1)*tokens_per_frame] = float('-inf')
    return mask
```

与标准 causal attention 的关键区别：帧内 token 之间无因果限制（全互见），仅跨帧有因果限制。这使得 attention mask 呈 block-wise 结构，与 FlashAttention 的 block tiling 自然兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamVGGT 中，temporal causal attention 使用 FlashAttention 的 custom mask 参数：`flash_attn_func(q, k, v, causal=False)` 传入 custom attention mask。在 HuggingFace Transformers 框架下也可通过 `attention_mask` 参数实现。首帧 tokens 始终保留在 cache 中作为 "geometric reference"（XStreamVGGT 的设计选择），类似于 attention sink 但语义不同（编码场景的全局几何锚点而非注意力冗余接收者）。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Dimension-Adaptive KV Quantization（维度自适应KV量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dimension-Adaptive KV Quantization 是 XStreamVGGT 提出的针对视觉 Transformer 中 KV cache 的量化策略。核心思想：根据 K 和 V 张量的不同分布特性选择不同的量化粒度（quantization granularity），而非使用统一的量化方案。具体地：(1) Key tensors 使用 **per-channel 量化**（每个 channel 独立计算量化参数），以应对 Key 中显著的 channel-wise outliers；(2) Value tensors 使用 **per-token 量化**（每个 token 独立计算量化参数），因为 Value 分布更均匀且 per-channel 量化对 Value 的 MSE 改善不大。

决策依据来自对 StreamVGGT 的 KV 分布分析：
- K 的 per-channel 量化 INT4 MSE：$9.181 \times 10^{-3}$（per-token: $5.183 \times 10^{-2}$，改善 5.6×）
- V 的 per-channel vs per-token 量化 INT4 MSE：$4.704 \times 10^{-4}$ vs $5.035 \times 10^{-4}$（差异极小）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
维度自适应量化的执行流程：

```
# 输入: K ∈ R^{T×C} (Key, T=tokens, C=channels)
#        V ∈ R^{T×C} (Value)

# === Key: per-channel quantization ===
for c in range(C):
    K_c = K[:, c]                                 # shape: T (所有 token 的 channel c)
    s_K[c] = (max(K_c) - min(K_c)) / (2^b - 1)   # per-channel scale
    z_K[c] = round(-min(K_c) / s_K[c])            # per-channel zero-point
    K̂_c = clamp(round(K_c / s_K[c]) + z_K[c], 0, 2^b - 1)

# === Value: per-token quantization ===
for t in range(T):
    V_t = V[t, :]                                 # shape: C (token t 的所有 channel)
    s_V[t] = (max(V_t) - min(V_t)) / (2^b - 1)   # per-token scale
    z_V[t] = round(-min(V_t) / s_V[t])            # per-token zero-point
    V̂_t = clamp(round(V_t / s_V[t]) + z_V[t], 0, 2^b - 1)

# 存储: K̂ (4-bit), s_K (FP16 × C), z_K (FP16 × C)
#       V̂ (4-bit), s_V (FP16 × T), z_V (FP16 × T)

# Metadata overhead: per-channel K: 2C × 2 bytes; per-token V: 2T × 2 bytes
# 典型值 (C=1024, T=2000): K overhead = 4KB, V overhead = 8KB → negligible
```

与 LLM 量化策略的区别：流式视觉 Transformer 以帧为单位批量产生 token → per-channel 量化对 K 天然友好（大量 tokens 共享同一 channel 统计量，scale 稳定）；LLM decode 每步仅 1 个 token → per-channel 量化需要跨 step 累积足够 tokens 才稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 KIVI 框架实现：`kivi/quantization.py` 中的 `KIVIQuantizer` 支持 per-channel 或 per-token 配置。PyTorch 原生实现：`torch.quantize_per_channel(K, scales, zero_points, axis=1, dtype=torch.qint8)`（axis=1 表示 per-channel along token dim，即 channel-wise along feature dim）。量化紧耦合在 pruning 之后：先 pruning 减少 token 数（T → L_max），再对精简后的 cache 量化。XStreamVGGT 开源代码：https://github.com/ywh187/XStreamVGGT/。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Asymmetric Uniform Quantization（非对称均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Uniform Quantization 是一种将浮点张量映射到低比特整数空间的量化方案。与对称量化（symmetric quantization，zero-point=0）不同，非对称量化引入一个可调零值偏移（zero-point z），使量化区间不关于 0 对称。数学定义：

$$\hat{x} = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z,\ 0,\ 2^b - 1\right)$$

其中：
- s（scale）= $(x_{\text{max}} - x_{\text{min}}) / (2^b - 1)$，量化步长
- z（zero-point）= $\lfloor -x_{\text{min}} / s \rceil$，将浮点 0 映射到的整数值
- b = 量化位宽（如 4 for INT4）
- ⌊·⌉ = 四舍五入

反量化：$\tilde{x} = s \cdot (\hat{x} - z)$

非对称量化的优势：当数据分布不以 0 为中心时（如 ReLU 激活输出、KV cache 中的 V tensors），可以更充分利用量化区间，减少 clipping error。代价：需要额外存储 zero-point（与 scale 量级相当），且 dequantization 操作多一个减法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中非对称均匀量化的具体实现：

```
def asymmetric_quantize(x, b=4):
    """
    x: 浮点张量，shape 任意
    b: 量化位宽，默认 4 (INT4)
    返回: x̂ (量化值), s (scale), z (zero-point)
    """
    x_min, x_max = x.min(), x.max()

    # Scale: 量化区间均匀划分
    s = (x_max - x_min) / (2**b - 1)

    # Zero-point: 将浮点0映射到的整数位置
    z = torch.round(-x_min / s)
    z = torch.clamp(z, 0, 2**b - 1)   # zero-point也需在有效范围内

    # 量化: round(x/s) + z, clamp到[0, 2^b-1]
    x̂ = torch.round(x / s) + z
    x̂ = torch.clamp(x̂, 0, 2**b - 1)

    return x̂.to(torch.uint8), s, z

# 反量化
def asymmetric_dequantize(x̂, s, z):
    return s * (x̂.float() - z)
```

数值示例（INT4, b=4, 范围 0-15）：
- x = [-2.0, 0.0, 3.0, 5.5] → x_min=-2.0, x_max=5.5
- s = (5.5-(-2.0))/15 = 0.5
- z = round(2.0/0.5) = 4
- x̂ = [0, 4, 10, 15]
- 反量化: x̃ = [-2.0, 0.0, 3.0, 5.5] ✓ (无量化误差，此例恰好完美恢复)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 `torch.quantize_per_channel` / `torch.quantize_per_tensor` 配合 `torch.qint8` / `torch.quint8` 等 dtype 实现。对于自定义位宽（如 INT4），使用手动 round + clamp。KIVI 库提供完整的 INT2/INT4 asymmetric quantization pipeline（group-wise or channel-wise）。在 GPU kernel 层面，量化和反量化通常在 attention kernel 外完成（如 XStreamVGGT 的方案），但也有 fused kernel 方案将 dequantization 嵌入 attention 计算以减少 memory access（如 BitDecoding）。XStreamVGGT 中使用 group size=64，即每 64 个元素共享一组 (s, z) 参数。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Hyper Attention Transformer Block (HATB，超注意力Transformer块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hyper Attention Transformer Block (HATB) 是 mPLUG-Owl3 提出的轻量级多模态 Transformer 扩展模块。其核心创新在于：在语言模型的 Transformer block 中，将 cross-attention（文本查询→视觉特征）与 self-attention（文本内部）**并行执行**，而非 Flamingo 的串行插入方式。HATB 仅在 LLM 中稀疏替换少量层（如 Qwen2 28 层中的 4 层 [0, 9, 17, 25]），并通过四个关键设计实现高效多模态融合：(1) 共享 LayerNorm——复用 Transformer 原生 LN 同时对文本和视觉特征做归一化；(2) Modality-Specific KV Projection——视觉的 K/V 投影权重用 LLM 预训练 KV 权重初始化（W_img_KV ∈ R^{2D×D}）；(3) 共享 Query——cross-attention 的 Q 直接复用 self-attention 的 Q，使 LLM 的语言知识指导视觉特征选择；(4) Adaptive Gating——基于文本特征 Sigmoid 门控融合 self-attention 和 cross-attention 输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HATB 在 LLM 推理中的计算流程：
```
# 输入: H_text ∈ R^{L×D} (文本隐状态), H_img ∈ R^{V×D} (视觉特征)
# HATB_layers = [l1, l2, ..., lK]  # 稀疏选择的层索引

for layer_idx in range(N_layers):
    # === 标准 Transformer block ===
    H_norm = LayerNorm(H_text)
    H_self_attn = SelfAttention(H_norm, causal_mask)
    H_text = H_text + H_self_attn

    if layer_idx in HATB_layers:
        # === Hyper Attention (与 self-attention 并行) ===
        # 1. 共享 LayerNorm: 复用同一 LN 对视觉特征归一化
        H_img_norm = LayerNorm(H_img)  # 与文本使用同一个 LN

        # 2. 获取 Query (复用 self-attention 的 Q)
        Q_text = W_Q(H_norm)  # 标准 Q 投影

        # 3. 视觉 KV 投影 (modality-specific，权重初始化自 LLM KV)
        K_img, V_img = split(W_img_KV(H_img_norm), dim=-1)

        # 4. MI-Rope 位置编码
        Q_rope = apply_rotary_pos(Q_text, pos_text)       # 文本位置
        K_img_rope = apply_rotary_pos(K_img, pos_images)  # 图像占位符位置

        # 5. Causal Cross-Attention
        A_cross = softmax(Q_rope @ K_img_rope^T / sqrt(d_k) + causal_mask_img)
        H_cross = A_cross @ V_img

        # 6. Adaptive Gating
        g = Sigmoid(W_gate^T @ H_text)           # g ∈ R^{L×1}, 逐 token
        H_fused = H_self_attn * g + H_cross * (1 - g)
        H_text = H_fused

    # === FFN ===
    H_text = H_text + FFN(LayerNorm(H_text))
```
关键点：cross-attention 在 self-attention 之后、FFN 之前执行；视觉特征不进入 LLM context window，序列长度始终为文本长度 L，不随图像数量增长；W_img_KV 在 Stage 1 仅训练此参数；4 层 HATB 达最佳效果，8 层反而退化；causal mask 确保自回归特性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HATB 在 mPLUG-Owl3 中基于 Qwen2 实现。Vision Encoder: Siglip-400m（~400M 参数），Language Model: Qwen2（~7B），Linear Projection 对齐视觉隐空间到文本维度。HATB 额外参数量：W_img_KV ∈ R^{2D×D} 每层约 2D² 参数（D=3584），4 层共约 103M（占 LLM ~1.5%）；W_gate ∈ R^{D×1} 每层仅 D 参数。训练三阶段：Stage 1 仅训练新增模块 ~41M pairs；Stage 2 全参数训练多图数据；Stage 3 SFT。TP=4 单 GPU 显存 32-40GB。开源：https://github.com/X-PLUG/mPLUG-Owl。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

## MI-Rope (Multimodal-Interleaved Rotary Position Embedding，多模态交织旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MI-Rope 是 mPLUG-Owl3 提出的用于多图交织场景的位置编码方法。核心思想：在图像-文本交织输入中，每张图的所有 patch tokens 共享该图在文本序列中占位符 T_img 的 RoPE 位置编码。与 MRoPE（Qwen2.5-VL，使用 3D (T,H,W) 位置 ID）不同，MI-Rope 仅使用 1D 位置 ID——即文本序列中的 token 索引位置。这确保了：(1) 图像间的相对顺序由占位符的文本位置自然编码；(2) 图像与前后文本的上下文关系被保留；(3) 配合 causal attention mask，每个文本 token 仅能 cross-attend 到已出现的视觉特征，保持自回归生成的一致性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MI-Rope 的位置分配逻辑：
```
# 输入: S_text = [T1, T_img, T2, T_img, T3] (文本+占位符序列)
# 图像特征: I1 patches ∈ R^{P1×D}, I2 patches ∈ R^{P2×D}

# Step 1: 记录每张图的占位符位置
pos_images = []
for token_idx, token in enumerate(S_text):
    if token == "<|image|>":
        pos_images.append(token_idx)

# Step 2: 为所有 visual patches 分配位置编码
# I1 的所有 P1 个 patches 共享 S_text 中第一个 T_img 的位置
# I2 的所有 P2 个 patches 共享 S_text 中第二个 T_img 的位置
# 例: S_text = [0:T1, 1:T_img, 2:T2, 3:T_img, 4:T3]
#     I1 patches → pos=1, I2 patches → pos=3

# Step 3: 在 cross-attention 中应用 RoPE
Q_rope = rotary_embed(Q_text, pos_text)        # 文本 Q: 自身序列位置
K_img_rope = rotary_embed(K_img, pos_images)   # 视觉 K: 占位符位置
```

与 MRoPE 的关键区别：
- MRoPE (Qwen2.5-VL): pos = (temporal_id, height_id, width_id)，三维位置，三组频率分别旋转不同维度段
- MI-Rope (mPLUG-Owl3): pos = 占位符在文本序列中的 1D 索引，所有 patches 共享同一位置

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MI-Rope 在 PyTorch 中实现为：预计算每个 batch 的 pos_images 张量（shape: [total_img_patches]），在 HATB 的 cross-attention 中调用标准 RoPE 实现（如 transformers 库的 LlamaRotaryEmbedding），但传入的位置索引为占位符位置而非原始图像网格位置。消融实验验证：去掉 MI-Rope 后多图 benchmark（NLVR2, Mantis-Eval）性能显著下降，视频 benchmark 影响较小（视频帧有时间顺序可被隐式建模）。MI-Rope 与 Shared LayerNorm 和 Adaptive Gating 协同工作，共同构成 HATB 的完整设计。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

## Adaptive Gating for Multimodal Attention Fusion（自适应门控多模态注意力融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Gating 是 mPLUG-Owl3 提出的基于文本语义的门控机制，用于融合 self-attention（文本内信息）和 cross-attention（视觉补充信息）的输出。与 Flamingo 的固定 learnable scale 不同，Adaptive Gating 通过文本特征自身计算门控值 `g = Sigmoid(W_gate^T · H_text)`，使得每个 token 可以根据其语义需求动态决定从视觉模态摄取多少信息。例如：语义丰富的 token（名词、形容词）可能分配更低 g 值以获取更多视觉上下文，而功能词保持高 g 值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# H_self  ∈ R^{L×D}:  self-attention 输出
# H_cross ∈ R^{L×D}:  cross-attention 输出
# W_gate  ∈ R^{D×1}:  可学习门控投影（每 HATB 层一个）

g = Sigmoid(H_text @ W_gate)       # g ∈ R^{L×1}, 逐 token 门控值
H_fused = H_self * g + H_cross * (1 - g)

# g ≈ 1.0: 信任文本内部信息（视觉与此 token 无关）
# g ≈ 0.0: 依赖视觉补充信息
# g ≈ 0.5: 均等融合
```
消融实验（Table 10）：
- 无 Adaptive Gating + 无 Shared LN + 无 MI-Rope: GQA 53.3, NLVR2 52.7, Mantis 41.9
- +Adaptive Gating: GQA 55.7 (+2.4), Mantis 47.9 (+6.0)
- +Adaptive Gating + Shared LN: GQA 58.1 (+4.8), TextVQA 49.7 (+5.1)
- 全配置 (+MI-Rope): NLVR2 59.5 (+6.8), Mantis 51.6 (+9.7)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为一个线性层 + Sigmoid：`nn.Linear(hidden_dim, 1) + nn.Sigmoid()`。W_gate 在 Stage 1 作为可训练模块（仅 ~D 参数），Xavier uniform 初始化。与固定 learnable scale 的区别：Adaptive Gating 是 per-token 动态的（每个 token 独立决策），而 learnable scale 是 per-layer 静态的（所有 token 同一权重）。梯度回传使模型学习"哪些文本 token 需要更多视觉信息"的语义判断。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

## Cross-Attention based MLLM Architecture（基于交叉注意力的多模态大语言模型架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于交叉注意力的 MLLM 架构是多模态大语言模型的三大架构范式之一（另两种为 Concatenation-based 和 Q-Former/Token Compression-based）。核心特征是视觉特征不直接进入 LLM 的文本序列，而是通过 cross-attention 层以"外部记忆"形式注入到 LLM 的中间表示中。代表工作：Flamingo（每层插入 cross-attention）、IDEFICS、EVLM 和 mPLUG-Owl3（稀疏 HATB）。天然优势：视觉 tokens 不占用 LLM context window，序列长度不随图像数量增长，多图/长视频场景下显存和计算效率远优于 Concatenation-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种 MLLM 架构范式的对比：
```
=== Concatenation-based (LLaVA, InternVL, Mantis) ===
输入: [img_tokens_1, ..., img_tokens_N, text_tokens]
处理: 全序列进入 LLM，标准 causal self-attention
问题: 序列长度 = N×P + T, O(L²) attention
      N=100 张 384² 图 → ~57.6K visual tokens → 80GB GPU OOM

=== Token Compression-based (BLIP-2, Idefics2, MiniCPM) ===
输入: img_tokens → compressor (Q-Former/perceiver/pooling) → fixed-size tokens
处理: 压缩后 tokens 拼接进文本序列
问题: 信息压缩损失，固定 token 数不够灵活

=== Cross-Attention-based (Flamingo, mPLUG-Owl3) ===
输入: H_img ∈ R^{V×D} 作为外部 K/V 对
处理: LLM sparse 层 cross-attend 到视觉信息
H_img 不在 context window 中
优势: self-attention 复杂度 O(T²) 独立于 V
      mPLUG-Owl3 比 LLaVA 处理 ~6× 更多图像
```

mPLUG-Owl3 相对 Flamingo 的关键改进：
- 并行而非串行：cross-attention 与 self-attention 在同一 block 并行执行，共享 Q
- 稀疏而非稠密：仅 4/28 层含 cross-attention（Flamingo 每层）
- 复用 LLM 权重：W_img_KV 初始化自 LLM KV 权重，共享 LayerNorm
- 位置感知：MI-Rope 为图像赋予交织序列的位置信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过修改 Attention 模块实现。每个 HATB 层额外维护 W_img_KV 和 W_gate 参数。视觉特征经 Linear Projection 对齐到 LLM 隐空间维度后，在 HATB 层作为 cross-attention 的 K/V 输入。训练需 staged training（先对齐再微调），视觉编码器和 LLM 主体通常冻结。mPLUG-Owl3 开源：https://github.com/X-PLUG/mPLUG-Owl。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models
- Flamingo: a Visual Language Model for Few-Shot Learning

## DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 是 ByteDance Seed 与清华大学提出的大规模 LLM 强化学习算法（NeurIPS 2025, arXiv:2503.14476），专为激发长链推理能力而设计。DAPO 在 GRPO (Group Relative Policy Optimization) 基础上引入四项关键技术以解决其在大规模推理训练中的失效模式：

(1) **Clip-Higher（非对称裁剪）**：将 PPO 的对称裁剪边界分离为独立的低裁剪 ε_low 和高裁剪 ε_high（典型值 ε_low=0.2, ε_high=0.28）。高概率 token 的正常裁剪保持策略稳定性，低概率 token 的高裁剪上界允许"探索性"token 的概率增长，防止熵崩溃（所有 rollout 收敛到相同输出导致梯度为零）。

(2) **Dynamic Sampling（动态采样）**：过滤掉模型已完美掌握（100% 正确率）或完全无法处理（0% 正确率）的 prompt，仅对"有信息量"的 prompt（部分正确部分错误）进行过采样直至填满 batch。这避免了梯度消失问题——当所有 rollout 奖励相同时，advantage 为零，无学习信号。

(3) **Token-Level Policy Gradient Loss**：使用 `loss = (1/Σ|o_i|) Σ_i Σ_t min(r_i,t * A_i,t, ...)` 进行逐 token 损失求和，而非先按样本平均再按 batch 平均。这确保长推理链（通常质量更高）对梯度更新的贡献与序列长度成比例，而非被 token 平均稀释。

(4) **Overlong Reward Shaping（软长度惩罚）**：对被截断的超长响应施加长度感知的渐进惩罚（而非硬 -1），避免正确推理但因超出长度限制被截断的响应受到不合理的严重惩罚，减少奖励噪声。

DAPO 使用 Qwen2.5-32B 基座模型在 AIME 2024 上达到 50 分，超越 DeepSeek-R1-Zero-Qwen-32B（47 分），且训练步数减少 50%。完全开源（代码基于 veRL 框架、数据集 DAPO-Math-17K）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DAPO 训练迭代伪代码（结合 VTPerception-R1 的 perception-aware 扩展）：
```
# 超参数: G rollouts, ε_low=0.2, ε_high=0.28, overlong_threshold
for each training step:
    prompts = sample_batch(D_train)
    
    # 1. Dynamic Sampling: 过滤无信息量 prompt
    valid_prompts = []
    for prompt in prompts:
        # 快速评估当前策略在该 prompt 上的表现
        quick_rollouts = π_θ.sample(prompt, n=G)
        acc = mean([check_answer(r, gt) for r in quick_rollouts])
        if 0 < acc < 1:  # 部分正确部分错误 → 有信息量
            valid_prompts.append(prompt)
        # 若 valid_prompts 不足 batch_size，继续从 D_train 采样
    
    # 2. 采样 G 个 rollouts per valid prompt
    for prompt in valid_prompts:
        o[1..G] = π_θ_old.generate(prompt)
        # o_i 格式: <description> d_i <think> t_i <answer> a_i
        
        # 3. 计算奖励（VTPerception-R1 扩展了标准 DAPO 的奖励函数）
        for i in 1..G:
            R_acc[i] = exact_match(a_i, ground_truth)
            R_fmt[i] = check_format(o_i)
            R_vkey[i] = compute_visual_key_recall(d_i, K_v)  # perception reward
            R_tkey[i] = compute_textual_key_recall(t_i, K_t)
            R_cons[i] = compute_consistency(d_i, t_i, a_i)
            R_rep[i] = -count_repeated_ngrams(o_i)
            R[i] = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons
        
        # 4. Group-relative Advantage (与 GRPO 相同)
        R_mean = mean(R[1..G])
        for i in 1..G:
            A[i] = R[i] - R_mean  # 组内归一化
    
    # 5. Token-Level Clipped Policy Gradient（DAPO 核心）
    total_tokens = sum([len(o_i) for i in 1..G])
    loss = 0
    for i in 1..G:
        for t in 1..len(o_i):
            r_i_t = π_θ(o_i_t | x, o_i_<t) / π_θ_old(o_i_t | x, o_i_<t)
            # Clip-Higher: 非对称裁剪
            clipped = clip(r_i_t, 1 - ε_low, 1 + ε_high)
            loss += min(r_i_t * A[i], clipped * A[i])
    loss = loss / total_tokens  # token-level 归一化
    
    # 6. 梯度更新
    θ = θ - lr * ∇_θ loss
```

DAPO vs GRPO 关键差异：
| 特性 | GRPO | DAPO |
|------|------|------|
| 裁剪策略 | 对称 ε (如 0.2) | 非对称 ε_low ≠ ε_high |
| 采样策略 | 全量采样 | 动态过滤 + 过采样 |
| 损失归一化 | per-sample 平均 | token-level 求和后全局归一化 |
| 长度控制 | 无或硬截断 | Overlong Reward Shaping |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DAPO 通过 veRL (Volcano Engine RL) 框架实现，代码开源在 https://github.com/verl-project/verl（`docs/algo/dapo.md`）。使用方式：(1) 定义奖励函数为 Python callable（可包含多组件如 accuracy + format + perception）；(2) 配置 DAPO trainer（G, ε_low, ε_high, overlong_threshold, KL coefficient）；(3) rollout engine 使用 SGLang 或 vLLM 进行高效采样；(4) 训练 loop 中 auto-regressive 采样 → 计算多组件奖励 → 动态过滤无效 prompt → 计算 group-relative advantage → token-level clipped policy gradient update。在 VTPerception-R1 中，DAPO 作为 Stage II 的基础优化器，论文基于 EasyR1-perc 框架实现（EasyR1 对 DAPO 的封装），使用 Ray 分布式部署（1 主节点 + 1 ORM 节点），TP=4。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding

## Perceptual Grounding (感知接地)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perceptual Grounding（感知接地）在多模态大语言模型（MLLM）中指模型将推理过程锚定在可验证的视觉和文本感知证据上的能力。不同于隐式感知（模型自由决定关注图像的哪些部分且不对外暴露），显式感知接地要求模型明确输出其感知到的关键视觉元素（物体、属性、空间关系）、文本元素（OCR 文本、数值、约束条件），并将后续推理与这些感知证据绑定。

VTPerception-R1 通过系统实验定义了三种感知接地策略：(1) **Explicit Perception**：将预先标注的感知注释直接附加到输入中，模型利用这些外部提供的感知信息进行推理；(2) **Structured Grounding**：通过 prompt 要求模型在推理前先输出自身的感知分析，但感知能力取决于模型自身；(3) **Implicit Grounding**：仅通过轻量 prompt（如 "carefully observe the image"）隐含地引导模型注意视觉内容，不要求显式输出。实验证明：Explicit Perception 在 7B 和 32B 模型上均带来最大收益；Structured Grounding 在小模型上反而有害（模型自身感知能力不足时产生幻觉性观察）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTPerception-R1 中感知接地的实现流程：
```
# ===== 训练阶段：建立感知接地能力 =====
# Stage I SFT: 训练模型将感知显式化
Input: (image x, question q)
Target: "<description> 
         Key objects: △AOB, points A,O,B. 
         Property: ∠AOB = 90° (right angle, diameter subtends). 
         Relevant text: AB=BC=25, AC=30.
         </description>
         <think> Since BC is diameter, ∠BXC = 90° ... </think>
         <answer> XY = 10 </answer>"

# Stage II RL: 奖励感知接地质量
# R_vkey: description 覆盖了多少关键视觉线索
K_v = {"△AOB", "right angle", "diameter BC", "intersection X,Y"}  # 标注的关键视觉线索
D_desc = extract_facts(description)  # 从模型生成的 description 提取事实
cov_v = |K_v ∩ D_desc| / |K_v|  # 视觉线索覆盖率
R_vkey = discretize(cov_v, τ_hi, τ_lo)

# R_tkey: think 覆盖了多少关键文本线索
K_t = {"AB=BC=25", "AC=30", "diameter→right angle property"}
D_think = extract_facts(think)
cov_t = |K_t ∩ D_think| / |K_t|
R_tkey = discretize(cov_t, τ_hi, τ_lo)

# R_cons: 推理是否忠于感知证据
F_ans = extract_entities(think) ∪ extract_entities(answer)
E = extract_entities(description) ∪ extract_entities(question)
if has_conflict(F_ans, E):  # 推理引用了感知中不存在的实体
    R_cons = 0
else:
    R_cons = |F_ans ∩ E| / max(1, |F_ans|)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
感知接地通过两个阶段建立：(1) SFT 阶段的感知增强训练——使用经过自动化清洗（VLM dense caption → Grounding DINO 目标检测 → EasyOCR → 结构化描述 → LLM 重建 CoT → 多维度质量评分）的 12K 样本训练模型输出 `<description>` 字段；(2) RL 阶段的感知感知奖励——通过教师模型集成（72B 级模型）生成多样化推理路径，预算验证（top-B by log-probability → correctness + coherence filtering）筛选高质量轨迹，最后从轨迹中提取视觉和文本关键信息作为奖励计算依据。推理时，模型的 `<description>` 字段可以被外部检查者审计，验证感知是否正确。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding

## Perception-Aware Reinforcement Learning (感知感知强化学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception-Aware Reinforcement Learning（感知感知强化学习）是 VTPerception-R1 提出的 RL 训练范式，在标准 DAPO/GRPO 的答案正确性奖励（R_acc）和格式奖励（R_fmt）之上，引入三个专门衡量感知质量的奖励项，将"看到了什么"和"是否基于看到的信息推理"纳入强化学习目标。

三组感知专用奖励：
- **R_vkey (Visual Key-Info Reward)**：衡量 `<description>` 覆盖预标注关键视觉元素（物体属性、几何约束、空间关系）的比例。计算 recall = |K_v ∩ D_desc| / |K_v|，离散化为三档（≥τ_hi → 1.0, τ_lo~τ_hi → 0.5, <τ_lo → 0.0）。
- **R_tkey (Textual Key-Info Reward)**：衡量 `<think>` 覆盖预标注关键文本信息（OCR 文本、数值、单位约束、常识）的比例。计算方式同 R_vkey。
- **R_cons (Description-Reasoning Consistency Reward)**：检查 `<think> + <answer>` 中引用的实体/属性/数值是否在 `<description> + question` 中有据可查。cons = |F_ans ∩ E| / max(1, |F_ans|)；存在明确冲突（如推理引用感知中不存在的数值）时 R_cons = 0。

采用 **Perception-First 加权调度**：训练早期增大 R_vkey 和 R_tkey 的权重，优先建立稳健的感知基础；后期逐步切换到以 R_acc 为主，追求最终答案正确性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
总奖励函数 R = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons，计算流程：
```
# 采样阶段
for prompt x = (image, question) in batch:
    o[1..G] = π_θ_old.generate(x)  # G 个 rollout
    # 每个 o_i 解析为:
    d_i = extract_tag(o_i, "description")  # 感知描述
    t_i = extract_tag(o_i, "think")         # 推理链
    a_i = extract_tag(o_i, "answer")        # 最终答案

# 奖励计算（6 组件）
for i in 1..G:
    # 1. 答案正确性（序列级）
    R_acc[i] = 1.0 if a_i == ground_truth else 0.0
    
    # 2. 格式合规（结构检查）
    R_fmt[i] = 1.0 if has_all_tags(o_i, ["description","think","answer"]) 
                    and not has_duplicate_tags(o_i) else 0.0
    
    # 3. 视觉关键信息（基于 description）
    D_desc = extract_atomic_facts(d_i)
    cov_v = |K_v ∩ D_desc| / |K_v|  # K_v 来自 RL 数据构建流水线
    R_vkey[i] = 1.0 if cov_v ≥ 0.8 else (0.5 if cov_v ≥ 0.5 else 0.0)
    
    # 4. 文本关键信息（基于 think）
    D_think = extract_atomic_facts(t_i)
    cov_t = |K_t ∩ D_think| / |K_t|
    R_tkey[i] = 1.0 if cov_t ≥ 0.8 else (0.5 if cov_t ≥ 0.5 else 0.0)
    
    # 5. 重复惩罚
    R_rep[i] = -count_repeated_ngrams(o_i, n=3) / len(o_i)
    
    # 6. 描述-推理一致性
    F_ans = extract_entities(t_i + " " + a_i)
    E = extract_entities(d_i + " " + question)
    if has_explicit_conflict(F_ans, E):  # 如: 推理说 AB=30 但 question 说 AB=25
        R_cons[i] = 0.0
    else:
        R_cons[i] = |F_ans ∩ E| / max(1, |F_ans|)
    
    # 总奖励（带 perception-first 调度权重）
    w_acc = schedule_weight(step, "acc")     # 早期小, 后期大
    w_perc = schedule_weight(step, "perc")   # 早期大, 后期小
    R[i] = w_acc * (R_acc[i] + R_fmt[i] + R_rep[i]) 
         + w_perc * (R_vkey[i] + R_tkey[i] + R_cons[i])

# DAPO 策略更新（同 DAPO 条目）
```

消融实验结果（Table 3）验证了各组件的互补性：移除 R_cons → C-MMBench 下降 3.26，C-MMBench-TO 下降 1.70；移除 R_tkey → C-MMBench 下降 2.64，C-MMBench-TO 下降 3.31；移除 R_vkey → AI2D 下降 2.01，MMMU 下降 1.21。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
感知感知 RL 的实现需要配套的数据构建流水线（Algorithm 2 in 论文 Appendix A.3）：(1) 教师模型集成（多个 72B 级模型，随机解码生成多样化推理路径）；(2) 预算验证（按 log-probability 排序 → top-B 候选 → correctness scoring + coherence scoring → 阈值过滤）；(3) 关键信息提取（从验证通过的轨迹中提取视觉关键信息 V 和文本关键信息 Z，Z 包含事实到推理步骤的映射）。最终每个 RL 训练样本表示为 (x, q, verified_answer, verified_trajectory, {V, Z})。代码开源在 https://github.com/yizhuoDi/VTPerceprion-R1，基于 EasyR1-perc 框架实现。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding

## Perception-Augmented SFT (感知增强监督微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception-Augmented SFT（感知增强监督微调）是 VTPerception-R1 Stage I 的训练方法，将原始多模态 CoT 数据转换为结构化格式 `<description>...<think>...<answer>`，通过 token 级交叉熵损失训练模型在推理前先显式提取和表达与任务相关的视觉/文本证据。

核心设计原则：(1) `<description>` 不是通用图像描述，而是仅总结"与问题相关且对推理有用"的视觉/文本证据；(2) `<think>` 保留原始 CoT 推理链，但推理应基于 `<description>` 中的感知证据；(3) `<answer>` 为最终解答。这种 "先看、再想、后答" 的结构显式解耦了感知与推理，使得感知过程可审计、可干预。

数据准备：从 LLaVA-CoT (4K) 和 Vision-SR1 (8K) 采样 ~12K 样本，经过自动化清洗流水线处理——VLM dense caption（GPT-4o 级模型生成密集描述）→ Grounding DINO 目标检测 → EasyOCR 文本提取 → 合并为结构化规范描述 → LLM 基于规范描述重建 CoT → 多维度质量评分（formal_score: 描述准确性, cot_score: 推理逻辑清晰度, answer_score: 答案一致性, 幻觉检测分数）→ 阈值过滤（overall_score ≥ τ 保留）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SFT 训练流程：
```
# 数据转换: 原始 CoT → 结构化格式
# 输入: (image, question, original_cot, original_answer)
# 输出: "<description> relevant visual/textual facts </description>
#         <think> step-by-step reasoning </think>
#         <answer> final answer </answer>"

# 训练目标: token-level cross-entropy over full target sequence
L_SFT = -Σ_t log π_θ(y_t | x_image, x_question, y_<t)
# 梯度流经所有 token: <description> + <think> + <answer>

# 训练配置:
model = Qwen2.5-VL-7B-Instruct  # 全参数微调
optimizer = AdamW(lr=1e-5, weight_decay=0.1)
batch_size = 1  # per-device
gradient_accumulation = 8  # effective batch = 8
epochs = 3
precision = bf16
# DeepSpeed ZeRO-3 + gradient checkpointing
```

效果：SFT 后模型能够 (i) 高亮关键物体和属性，(ii) 捕获空间/语义关系，(iii) 将感知证据链接为推理步骤。SFT 为 Stage II RL 提供了稳定的感知-描述接口。Table 2 显示 Before RL (SFT-only) 的 VTPerception-R1-7B 在多个 benchmark 上已超过 Qwen2.5-VL-7B-Instruct baseline（如 AI2D: 80.4 vs 77.2, C-MMBench: 46.7 vs 43.1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖于自动化数据清洗流水线（论文 Appendix A.2）：(1) 图像分析阶段：VLM 生成 dense caption + Grounding DINO 目标检测 + EasyOCR → 合并为结构化规范描述（formal_description），这是后续所有 CoT 重建的唯一图像信息来源（single-source-of-truth 原则）；(2) CoT 重建阶段：LLM 基于问题 + formal_description 重新生成推理链，明确禁止参考原始 CoT 或外部知识；(3) 质量评估阶段：多维度评分（formal_score, cot_score, answer_score, 幻觉分数），加权求和 overall_score，阈值过滤。代码开源在 https://github.com/yizhuoDi/VTPerceprion-R1。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding

## Description-Think-Answer Pipeline (感知-推理-答案分离流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Description-Think-Answer Pipeline（DTA 流水线）是 VTPerception-R1 提出的多模态推理结构化输出格式，通过三个明确的 XML 标签将模型输出组织为 `<description> → <think> → <answer>` 的线性流水线。与传统的自由格式 CoT（感知和推理混杂在一起）相比，DTA 流水线强制显式分离：(1) 感知阶段（description）：提取任务相关的视觉和文本证据；(2) 推理阶段（think）：基于感知证据进行逻辑推导；(3) 答案阶段（answer）：给出最终答案。这种设计使感知过程可被外部审计（检查是否"看到了正确的信息"），推理可被验证是否忠于感知证据。

DTA 流水线在 Stage I SFT 中通过 token 级交叉熵损失训练建立，在 Stage II RL 中通过 R_fmt（格式合规）和 R_cons（一致性）奖励强化。与 Visionary-R1 的 "caption → reason → answer" 结构类似但更聚焦：description 只包含任务相关的感知证据而非通用描述，且额外引入了 R_cons 确保推理不偏离感知。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DTA 流水线的训练/推理流程

# === 训练阶段 ===
# 输入: 几何题图像 + "In triangle ABC, AB=BC=25, AC=30..."

# 模型自回归生成:
token_1..k = generate("<description>")
# 输出: "Isosceles triangle ABC with AB=BC=25, base AC=30. 
#        Circle with diameter BC intersects AB at X, AC at Y.
#        Key property: diameter → right angle at BXC and BYC."

token_{k+1}..m = generate("<think>")
# 输出: "Since BC is diameter, ∠BXC = ∠BYC = 90°.
#        CX is altitude from C to AB. 
#        Area = 1/2 * AC * BM = 1/2 * 30 * 20 = 300.
#        Also Area = 1/2 * AB * CX → CX = 24.
#        In right △BXC: BX = √(25²-24²) = 7 → AX = 25-7 = 18.
#        △AXY ~ △ABC → XY/25 = 18/30 → XY = 15."

token_{m+1}..n = generate("<answer>")
# 输出: "15"

# 损失: L_SFT = -Σ_{t=1..n} log π_θ(y_t | image, question, y_<t)

# === 推理阶段 ===
# 外部审计可以独立检查:
# 1. description 是否准确描述了图像中的关键几何元素?
# 2. think 中的推理步骤是否基于 description 中的证据?
# 3. answer 是否正确且一致?
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DTA 流水线通过两个阶段建立：(1) SFT 阶段：使用结构化数据训练模型生成带标签的输出格式，格式违规（缺少标签、标签重复）在 RL 阶段通过 R_fmt = 0 惩罚；(2) RL 阶段：R_fmt 强制格式合规，R_cons 确保推理内容忠于描述证据。在推理时，用户或外部系统可以解析 `<description>` 标签内容并独立验证感知准确性，这在需要可解释性和可审计性的应用场景（如医疗、自动驾驶安全分析）中尤为重要。代码实现基于 Qwen2.5-VL-7B-Instruct 的全参数微调，训练配置同 Perception-Augmented SFT。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
