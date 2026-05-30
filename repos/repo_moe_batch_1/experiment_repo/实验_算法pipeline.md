
## Continual Pre-training of MoEs How robust is your router

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MoE transformer 的大规模持续预训练（Continual Pre-training, CPT）策略。论文系统研究了两种路由算法（Penalty-Balanced Top-k, PBTk 和 Sinkhorn-Balanced Top-k, SBTk）和两种 MoE 架构（Granular MoE: 31 routed experts, K=3 active, 1 shared expert; Switch MoE: 8 routed experts, K=1 active, 无 shared expert）在分布偏移下的 CPT 行为。CPT 策略使用：(1) Infinite LR schedule (CosineInf)，从非衰减 checkpoint 恢复训练；(2) Replay 机制（30%/40% 旧数据回放）；(3) Learning rate re-warming + re-decaying（从衰减 checkpoint 开始时）。

  实验比较：(a) 4 种 MoE 架构（PB Granular, SB Granular, PB Switch, SB Switch） vs FLOP-matched Dense Baseline (570M) 在 FineWeb→Stack(Code) 和 FineWeb→German 两个分布偏移下的 CPT 表现；(b) CPT vs full re-training baseline（从头在 FineWeb∪Stack/German 联合数据上训练）；(c) 不同 replay 比例（0%/10%/40%）对遗忘和适应的影响；(d) 从衰减 vs 非衰减 checkpoint 开始 CPT 的对比；(e) 路由行为分析：Router Saturation (路由饱和率), Vocabulary Specialization (词汇专精), Expert Co-activation (专家共激活), Maximum Routing Imbalance (MRI, 最大路由不均衡度)。

- 硬件平台是什么，配置是什么。
  64 张 NVIDIA A100 GPU，使用 data parallelism + ZeRO-1（Rajbhandari et al., 2020）。显存和互联配置：论文未明确说明单卡显存（应为 A100-80GB 或 A100-40GB）。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - Dense Baseline: 24层 570M 参数 decoder-only transformer，Llama3 架构（但使用 GeLU 激活），Llama3 tokenizer，GEGLU FFN（中间维度 2816），hidden size 1024，16 attention heads，RoPE
  - Granular MoE: 570M active / 2B total，E=31 routed experts + 1 shared expert，K=3 active，FFN 中间维度 704（为 dense 的 1/4），GEGLU
  - Switch MoE: 570M active / 2B total，E=8 routed experts，K=1 active，无 shared expert，FFN 中间维度 2816（与 dense 相同），GEGLU
  所有 MoE 使用 Top-k routing（k=1 for Switch, k=3 for Granular），不 drop token。PBTk 使用 z-loss coefficient 0.001 + Aux-loss coefficient 0.01。SBTk 使用 tolerance 0.01。

  数据集：
  - Pre-training: FineWeb (English web crawl, 400B tokens)
  - CPT: The Stack (code, 200B tokens) 和 German Common Crawl (200B tokens)
  - Replay: FineWeb 数据按比例回放（30% for Stack, 40% for German）

  Benchmarks:
  - English: HellaSwag, Winogrande, PIQA, ARC-Easy, ARC-Challenge, SWAG, LAMBADA (OpenAI), SciQ, PubMedQA, MathQA
  - German (GPT-3.5 翻译): HellaSwag-DE, ARC-Challenge-DE, TruthfulQA-DE
  - Code: HumanEval (pass@k, k∈{1,10,50,100,150,200})
  - Validation loss: FineWeb, Stack, German 测试集上的 log perplexity
  - Routing metrics: MRI, Router Saturation, Vocabulary Specialization, Expert Co-activation

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供独立的代码仓库。实验基于 GPT-NeoX 库 (https://github.com/EleutherAI/gpt-neox) 和 Megablocks grouped GEMM kernel (https://github.com/tgale96/grouped_gemm)。训练使用 AdamW 优化器 (β₁=0.9, β₂=0.95)，weight decay 0.1，gradient clipping 1.0，batch size 1024，sequence length 2048。

  **算法 pipeline 伪代码（MoE CPT 训练循环）**：

  ```
  # Step 1: 初始化 MoE 模型 (以 Granular PBTk 为例)
  model = MoETransformer(
      num_layers=24, hidden_size=1024,
      num_routed_experts=31, num_active_experts=3,
      shared_expert=True, ffn_intermediate=704,
      router_type="PBTk"  # 或 "SBTk"
  )
  # 使用 Llama3 tokenizer, vocab_size=128000

  # Step 2: Pre-training Phase (FineWeb, 400B tokens)
  scheduler = CosineInf(  # Infinite LR schedule
      total_iters=192720, eta_max=3e-4, eta_min=3e-5,
      eta_const=1.65e-4, T_warmup=0.01, T_cooldown=0.70
  )
  for step in range(192720):
      batch = sample_fineweb(batch_size=1024, seq_len=2048)
      loss = model(batch).loss + aux_loss(model.router_logits, batch) * 0.01
                                  + z_loss(model.router_logits) * 0.001
      optimizer.step(loss)  # AdamW
      scheduler.step()

  # Step 3: CPT Phase (FineWeb→German or FineWeb→Stack)
  scheduler_cpt = CosineInf(  # 从非衰减 checkpoint 恢复
      total_iters=95370, eta_max=3e-4, eta_min=3e-5,
      eta_const=1.65e-4, T_warmup=0.01, T_constant=0.80
  )
  for step in range(95370):
      # Replay: 40% FineWeb + 60% German (或 30% FineWeb + 70% Stack)
      batch_replay = sample_fineweb(batch_size * 0.4, seq_len=2048)
      batch_new = sample_german(batch_size * 0.6, seq_len=2048)
      batch = concat([batch_replay, batch_new])
      loss = model(batch).loss + aux_loss * 0.01 + z_loss * 0.001
      optimizer.step(loss)
      scheduler_cpt.step()
  ```

  **MoE 层前向传播（以 token x 为例）**：
  ```
  # Router: W_r ∈ R^{H×E} (H=hidden_size, E=num_experts)
  logits = W_r @ x                    # [E]  线性投影
  probs = softmax(logits)             # [E]  PBTk: 恒等; SBTk: Sinkhorn re-weight
  topk_indices = topk(probs, k=3)     # 选择 top-3 experts
  topk_probs = probs[topk_indices]    # 对应的概率

  # Expert computation (GEGLU FFN)
  shared_out = SharedFFN(x)           # shared expert 输出
  expert_outs = []
  for idx in topk_indices:
      expert_outs.append(GEGLU_FFN_expert[idx](x))

  # Weighted combination
  combined = sum(topk_probs[i] * expert_outs[i] for i in range(k))
  combined = combined / sum(topk_probs)  # 归一化

  output = shared_out + combined      # MoE 层输出
  ```

  **Maximum Routing Imbalance (MRI) 计算**：
  ```
  # 对于 MoE 层 j，batch B 中的 tokens
  def compute_MRI(layer_j, batch_B):
      E = layer_j.num_experts
      k = layer_j.num_active_experts
      loads = zeros(E)
      for token in batch_B:
          topk_indices = layer_j.route(token)
          for idx in topk_indices:
              loads[idx] += 1
      loads = loads / len(batch_B)    # 归一化
      return max(loads)               # MRI
  ```

## Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一种双向 MoE 路由算法 ETR（Expert-Token Resonance），包含三项算法创新：(1) **GrAP（Grouped Average Pooling）路由层**：将传统 MLP Router 的稠密权重矩阵替换为对角稀疏亲和力矩阵 W_aff，参数降为原来的 1/D（D 为分组因子，按 expert 数 n 分组），计算复杂度从 O(d²) 降至 O(d²/D)，且正交性天然防止 expert 同质化；(2) **TCR+ECR 双向选择机制**：token 先通过余弦相似度亲和力分数选择 top-ℓ experts（TCR），然后每个 expert 再按其亲和力分数从已分配的 token 中选择 top-C tokens（ECR），实现"共振效应"；(3) **自适应容量策略**：基于训练进度动态调整 expert capacity 下界，理论证明可将容量下界降低最多 40%，消除 All-to-All 通信气泡。

  实验比较：(a) ETR（LocMoE+）vs Baseline（标准 Top-1 MoE routing + capacity factor 1.1）vs LocMoE（TCR only）vs LocMoE（ECR only）；(b) 训练效率对比：32N/64N/256N Ascend NPU 集群下的每步耗时、各计算阶段（computation/communication/overlap/idle）分布、operator 级别耗时（FFN MatMul、TopK、IndexPutV2）、显存占用；(c) 路由质量对比：Calinski-Harabasz (CH) Index 衡量 token 聚类质量、不同 loss 函数下的 token 分配分布（CDF/ECDF）；(d) 下游任务对比：GDAD（含 GDAD-1/2/3 三个子任务及 16+13+18 项子能力）、GPQA、HumanEval、MMLU、TeleQnA；(e) SFT 后 GDAD 16 项子能力对比。

- 硬件平台是什么，配置是什么。
  Huawei Ascend 910B3 NPU 集群，三组规模：(1) 32N：TP=4, PP=4, DP=2, EP=2；(2) 64N：TP=8, PP=4, DP=2, EP=2；(3) 256N：TP=8, PP=8, DP=4, EP=2；全局 batch size=128。单颗 910B3 NPU：20 AI Cores @ 1.8GHz，fp16 理论算力 313T，HBM 64GB @ 1.6GHz，带宽 1.6T。每 8 颗 NPU 安装在同一 Atlas 800T A2 服务器内，全 mesh 互联。

- 模型是什么。数据集和bench分别是什么。
  模型：Mixtral 8×7B（46.7B 参数），GQA 注意力，32 层 sparse MoE block，每层 8 experts，top-2 选择（实验中统一用 top-1），序列长度扩展至 32768 tokens。
  预训练数据集：自建 300B tokens（150B ICT 领域 + 150B 通用数据），含中英双语，领域数据来自 iCase、blogs、Wiki、feature documents 等华为内部技术文档。通用数据含网页、书籍、代码、问答等 (详见 Table 3)。
  SFT 数据集：762,321 通用 QA + 11,048 领域 QA（比例 68:1），两阶段训练（Stage1 ~2M 样本增强逻辑推理，Stage2 ~3M 样本增强指令遵循）。
  Benchmark：GDAD（自建，含 16 类领域任务能力 2657 题 + 13 类领域认证考试 13968 题 + 18 类通用能力 1435 题）、GPQA、HumanEval、MMLU、TeleQnA。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自华为，未提供公开 GitHub 仓库。代码基于华为内部 MindSpeed-LLM / MindSpeed / AscendSpeed 生态，运行于 Ascend NPU。以下给出 ETR 算法 pipeline 的伪代码：

```
# === ETR 路由算法 (每层 MoE block) ===
# 输入: x ∈ R^{s×d}  (s 个 token，每个 d 维)
# 超参: n (expert 数), D=n (分组因子), ℓ (top-k), C (自适应容量)

# Step 1: GrAP 构建亲和力矩阵 W_aff
# W_aff ∈ R^{d×n} 为对角稀疏矩阵
for i in range(n):
    start = i * d // n
    end = (i+1) * d // n
    w_i[j] = n/d  for j in [start, end)   # 公式(3)
    # W_aff 仅对角线有值，参数量 = d, 而传统 MLP Router 为 d×n
    # 计算复杂度 O(d²/n) vs 传统 O(d²)

# Step 2: 计算 token-expert 亲和力分数
# δ_{t,i} = cos(x_t, w_i) = x_t^T w_i / (||x_t|| * ||w_i||)  # 公式(4)
for t in range(s):
    for i in range(n):
        delta[t][i] = cosine_similarity(x[t], W_aff[:, i])

# Step 3: TCR — Token 选择 Top-ℓ experts
for t in range(s):
    top_experts[t] = TopK(delta[t, :], k=ℓ)  # 公式(5)
    # 每个 token 分配至其亲和力分数最高的 ℓ 个 expert

# Step 4: ECR — Expert 选择 Top-C tokens（双向选择）
# 动态计算容量 C = max(C_min, adaptive_capacity(delta, training_progress))
# C_min = (1/n) * exp(d * δ_max² / (2 - δ_max²))  # Remark 7

for i in range(n):
    # 获取第一步中被分配至 expert i 的所有 token
    assigned = [t for t in range(s) if i in top_experts[t]]
    # Expert i 按亲和力分数从 assigned 中选择 Bottom-C（最低分数）
    # 即保留最高亲和力的 C 个 token
    selected[i] = BottomC(delta[t, i] for t in assigned, c=C)  # 公式(6)

# Step 5: MoE 计算（仅对选中的 token-expert 对）
output = zeros(s, d)
for i in range(n):
    for t in selected[i]:
        output[t] += G_i(x[t]) * E_i(x[t])  # gating weight * expert FFN

# === Locality Loss（负载均衡）===
# L_loc = μ * KL(D_c || D_l)
# D_c: 当前 token 分布, D_l: 完全本地化分布
# 鼓励 token 发送至同节点 expert，减少跨节点通信
total_loss = task_loss + alpha * L_aux + beta * L_loc

# === 训练阶段自适应 ===
# Phase 1 (早期，q_i ≈ Θ(1)): 偏向 TCR, C = Θ(s)
# Phase 2 (后期，s·q_i ≤ C*): 偏向 ECR, C = Θ(1)（容量降低~40%）
```

关键理论依据：Theorem 5 证明早期训练中 TCR 成功率为 Θ(C·∑p_i/s)，ECR 呈指数衰减 e^{-s}；后期当 expert 获得判别能力后 (q_i << 1)，ECR 接近 100% 成功率，TCR 仍受限于 C/s。因此动态从 TCR 过渡到 ECR 能最大化全程训练成功率。

## EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MLLM 的 MoE-tuning 框架的两项算法创新：(1) **Expert Evolution**：通过动态平衡先验经验与梯度更新，从单个可训练 FFN 专家逐步演化出多个多样化专家，解决 MoE-tuning 中复制初始化导致的 expert uniformity 问题；(2) **Dynamic Token-aware Router (DTR)**：用两个 hypernetwork（分别处理视觉和文本 token）动态生成每个 router 的 up-sampling 和 down-sampling 层参数，实现基于模态和 token 内在值的自适应专家分配，解决线性路由器的 router rigidity 问题。

  实验比较：(a) EvoMoE vs MoE-LLaVA（MoE-tuning baseline）、Qwen-MoE 等 sparse MLLM，在 0.5B/1.6B/1.8B/2.7B/7B 多种 LLM 规模上对比；(b) EvoMoE 仅激活 top-1 expert vs MoE-LLaVA 激活 top-2 experts，激活参数更少但性能更优；(c) 消融实验：DTR vs Linear Router、Expert Evolution vs Replication Init、不同 DTR 架构设计（single router / modality-specific router / shared router / HyperNet）、不同 expert diversity 策略（Noise、Dropout、Contrastive Loss、Local Loss）；(d) MoE 策略探索：w/o first MoE layer、Shared Expert、不同阶段可训练参数、Alternating vs All-Layer MoE placement；(e) Evolution β 值的选择（固定 vs 随机采样）。

- 硬件平台是什么，配置是什么。
  8x NVIDIA A100-80G GPU，使用 DeepSpeed ZeRO-2（Stage I、II）和 ZeRO-2_offload（Stage III），精度 Bf16。Qwen2-0.5B 训练 25h，StableLM-1.6B 训练耗时未明确给出（Stage I 6h + Stage II 12h + Stage III 7h = 25h 为 Qwen-1.8B 配置），各模型 batch size 均为全局 64（Stage II/III，gradient accumulation 2），文本最大长度 2048。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 MoE-LLaVA 和 LLaVA 1.5 框架，LLM 包括 Qwen2-0.5B、StableLM-1.6B、Qwen-1.8B、Phi2-2.7B、OpenChat-7B，视觉编码器为 CLIP-L 和 SigLIP-L，图像投影层为 MLP with GeLU，输入分辨率 336×336（部分实验 384×384）。MoE 配置：4 个 FFN expert，top-1 路由，alternating MoE layer placement（非全部层）。Stage II 仅训练 FFN1（Expert 1），其他 experts 由 Expert 1 演化生成且冻结；Stage III 仅训练 DTR（hypernetworks + final linear layer），experts 冻结。

  数据集：Stage I（Warm-up）使用 MIMIC-IT、LRV、SViT、LVIS 混合数据；Stage II（Expert Evolution）和 Stage III（DTR）使用 LLaVA-mix-665k。
  
  Benchmarks：Image QA — VQA-v2、GQA、ScienceQA (SQA)、TextVQA；Multi-modal Understanding Toolkits — POPE、MME、MMBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供 EvoMoE 开源代码链接，arXiv 页面无 GitHub 仓库。底层框架 MoE-LLaVA（https://github.com/PKU-YuanGroup/MoE-LLaVA）和 LLaVA 1.5（https://github.com/haotian-liu/LLaVA）是开源的。核心算法流程：

  **Expert Evolution（Stage II）**：
  ```
  # 输入：Stage I 输出的 FFN 参数 θ_1（Expert 1），N=4 个 experts
  # 每步训练时，对 Expert n (n=2,3,4) 执行演化：
  for each training step:
      # 仅训练 θ_1（Expert 1），计算梯度 ∇θ_1
      ∇θ_1 = backward(L_regressive + α * L_aux)
      update θ_1 with optimizer

      # 对其他 experts 进行演化：
      for n in [2, 3, 4]:
          β_n = random_sample(range_n)  # range_2=[0.9,0.99], range_3=[0.8,0.89], range_4=[0.7,0.79]
          θ_n ← β_n · θ_1 + (1 - β_n) · ∇θ_1  # EMA 形式的参数演化

  # 关键：θ_2, θ_3, θ_4 不参与梯度计算，仅由 θ_1 通过不同 β 演化而来
  ```

  **DTR Forward Pass（Stage III）**：
  ```
  # 输入：经过 MSA 后的 visual tokens V ∈ R^{P×C} 和 text tokens T ∈ R^{M×C}
  # 仅 H_V, H_T（各含两个 MLP layer）和最终 linear layer φ 可训练

  # Hypernetwork 生成动态参数：
  for τ in {V, T}:
      z' = MSA(LN(z_prev)) + z_prev  # MSA block 输出
      Θ_up^τ, Θ_down^τ = H^τ(z')     # Hypernetwork 生成 up/down 投影矩阵
      
      # Token-aware routing：
      E^τ = Θ_up^τ · SwiGLU(Θ_down^τ · z')  # 动态投影 + SwiGLU 激活
      ρ^τ = φ(E^τ)                            # 最终 linear router 预测 top-k expert 概率

  # 对每个 token，选择 top-1 expert 执行 FFN：
  z_out = FFN_{selected_expert}(LN(z'))
  ```

  训练总损失：L_total = L_regressive + α · L_aux，其中 α=0.001，L_aux 为负载均衡损失（鼓励 experts 均匀处理 token）。

## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是两个 MoE 模型的完整预训练与后训练流水线：**Ling-Lite**（总参数 16.8B，激活 2.75B）和 **Ling-Plus**（总参数 290B，激活 28.8B）。算法层面的核心创新包括：(1) **Fine-Grained Experts + Shared Expert**：扩展专家数量并等比缩小每个专家的中间维度，同时引入一个无需路由的 Shared Expert 提供通用能力。(2) **NormHead**：对 LM-Head 权重进行 L2 归一化，缓解训练中输出 norm 不稳定导致的 loss spike。(3) **Stochastic Routing Warmup**：在训练早期以线性衰减权重混合随机路由 logits 和学习到的路由 logits，防止训练初期路由崩溃和不均衡。(4) **Scaling Laws for MoE**：系统分析 MoE 架构的 batch size 和 learning rate 随 compute budget 的幂律关系，以及 MoE vs Dense 的 efficiency lever（~3x）。(5) **Skip Loss Spikes & Sample Retry**：检测到 loss spike 时跳过当前更新并将数据随机重注入后续 batch，持续 spike 则自动降低学习率。(6) **SFT→DPO 后训练流程**：含 quality assurance（rule-based filtering + LLM judge）、semantic deduplication、Vanilla DPO + Robustness Optimization + format-focused DPO。

  实验比较：(a) 与 Dense 架构对比 scaling law（FLOPs-to-Loss 曲线）；(b) Ling-Lite-Base vs Qwen2.5-7B、LLaMA-3.1-8B、Mistral-7B-v0.3；(c) Ling-Plus-Base vs DeepSeek-V2-Base、Qwen2.5-72B-Base、LLaMA-3.1-70B-Base；(d) Ling-Lite instruct vs Qwen2.5-7B-Instruct、LLaMA-3.1-8B-Instruct、Mistral-7B-v0.3-Instruct；(e) Ling-Plus instruct vs DeepSeek-V2.5-Chat、Qwen2.5-72B-Instruct、LLaMA-3.1-70B-Instruct、GPT4o-0806；(f) 不同加速器 (Device A vs Device D) 上的训练一致性；(g) Safety vs False Refusal trade-off；(h) Needle-in-A-Haystack 测试（最长 64K）。

- 硬件平台是什么，配置是什么。
  五种异构 AI 加速器（按可用性降序）：Device A (370 TFLOPS, 64GB, 无 FP8)、Device B (120 TFLOPS, 96GB, 无 FP8)、Device C (312 TFLOPS, 80GB, 无 FP8)、Device D (989 TFLOPS, 80GB, 支持 FP8)、Device E (147 TFLOPS, 96GB, 支持 FP8)。训练共使用 9T tokens 跨五种硬件配置混合训练。Scaling law 实验 compute budget 从 1e18 到 6e20 FLOPs。Ling-Plus 在高性能硬件 (Device D) 上训练 1T tokens 成本约 635 万 RMB，低规格硬件降至约 508 万 RMB（节省 ~20%）。

- 模型是什么。数据集和bench分别是什么。
  模型：Ling-Lite (16.8B/2.75B active) 和 Ling-Plus (290B/28.8B active)，均为 MoE 架构，使用 fine-grained experts + shared expert，dropless 路由，支持 4K→16K context（RoPE θ 从 10K→600K）。预训练数据：9T tokens（1T 中文 + 5.5T 英文 + 2.5T 代码），来源包括 Common Crawl、书籍、学术论文、社交媒体、百科、数学、编程代码。Benchmarks：英文（MMLU、MMLU-Pro、MMLU-Redux、BBH、HellaSwag、PIQA、ARC-Challenge、WinoGrande、RACE-Middle/High）、中文（C-Eval、CMMLU）、数学（GSM8K、MATH）、代码（HumanEval、MBPP、CRUXEval-I/O）。Instruct 额外评估：IFEval、GPQA-Diamond、SimpleQA、C-SimpleQA、MultiPL-E、LiveCodeBench、AIME-2024、BFCL-v2、Nexus、T-eval、Arena-Hard、Arena Safety、Cvalues、Xstest、Orbench-Hard-1k。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  模型权重：[https://huggingface.co/inclusionAI](https://huggingface.co/inclusionAI)。核心算法流程：

  **MoE Forward with Fine-Grained Experts + Shared Expert**:
  ```
  # 输入: h_t ∈ R^d, N experts, top-k routing
  p_t = Softmax(R(h_t))              # 路由概率
  o_t = Σ_i p_{t,i} * E_i(h_t)      # top-k 专家输出加权和
  o_t' = o_t + E_share(h_t)         # 加上 Shared Expert
  ```

  **Stochastic Routing Warmup** (step i ≤ W):
  ```
  s_t = Linear(h_t)                  # 原始路由 logits
  μ_s, σ_s = running_stats(s_t)      # 运行时均值/标准差
  ϵ ~ N(0, I)                        # 标准正态噪声
  ŝ_t = α · s_t + (1-α) · (μ_s + σ_s · ϵ)
  α = min(i/W, 1.0)                  # 从 0 线性增长到 1
  ```

  **NormHead**:
  ```
  h_o = (W_lm_head / ||W_lm_head||_2) · h
  ```

  **Skip Loss Spikes & Sample Retry**:
  ```
  if detect_loss_spike(current_loss):
      skip current_update()
      save affected_data()
      randomly_reinject_data_to_future_batches()
      if spike_persists:
          lr *= decay_factor  # 自动降学习率
  ```

  **Efficiency Lever (Scaling Law)**：给定相同 training loss，MoE 所需 compute budget 约为 Dense 的 1/3，且随 compute budget 增大 efficiency lever 从 ~3× (@ 1e21 FLOPs) 增长到 >3.5× (@ 1e24 FLOPs)。

## EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 EAC-MoE，结合两部分：(1) **QESC (Quantization with Expert-Selection Calibration)**：静态量化方法，对 MoE 模型进行逐层 GPTQ 量化后，使用 TopK-MSE 损失校准路由器，缓解低比特量化引起的 expert-shift 问题。MHSA 保持 4-bit，Router 保持全精度，Expert 量化到 2/2.5/3-bit（最终平均位宽 2.06/2.54/3.03-bit）。校准用 WikiText2 训练集的 128 条 2048 长度序列。(2) **PESF (Pruning based on Expert-Selection Frequency)**：动态专家剪枝，在推理时根据当前序列中各 expert 被选中的频率，剪枝低于阈值 α × (l×K/N) 的 expert。α=0.3 保守（~10% 加速无精度损失），α=0.7 激进（~30%+ 加速）。PESF 仅适用于 prefill 阶段。(3) **EAC-MoE = QESC + PESF** 组合，在量化基础上进一步剪枝。
  实验比较：(a) 量化对比：vs GPTQ、BSP、PMQ 在三种位宽下的 PPL 和 8 个 zero-shot 任务准确率；(b) 剪枝对比：vs EES、ODP 在准确率和加速比上；(c) 量化+剪枝组合 vs MC-MoE；(d) 消融实验：TopK-MSE vs MSE 损失；(e) 挑战任务：GSM8K 和 HumanEval；(f) 过拟合分析：混合精度方法用不同校准集在跨任务数据上的性能。

- 硬件平台是什么，配置是什么。
  量化过程在单张 A100 40G GPU 上执行。推理加速测试和部署演示在 RTX 3090 GPU 上进行。量化耗时：Mixtral-8x7B 总耗时约 1.32h（GPTQ 1.30h + Router Calibration 0.02h），其余模型类似。

- 模型是什么。数据集和bench分别是什么。
  模型：Mixtral-8x7B（8 expert, top-2）、Phi3.5-moe（16 expert, top-2）、Deepseek-moe-16b-base（64 expert, top-6, 含 shared expert）、Qwen1.5-MoE-A2.7B（60+4 shared expert, top-4）。数据集/benc"hmarks：(1) WikiText2 测试集 PPL；(2) 8 个 zero-shot 任务（EleutherAI LM Harness）：Winogrande、PIQA、ARC-Easy、ARC-Challenge、BoolQ、MathQA、HellaSwag、MMLU；(3) GSM8K（数学）、HumanEval（代码，pass@10，Bigcode-Evaluation-Harness 框架）。专家选择分析额外使用 19 个数据集涵盖 QA/CR、Math、Code、French 四类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 CAS 自动化所，发表在 ACL 2025。**论文未提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。算法核心流程如下：

## Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MoE 教师模型的知识蒸馏方法，包含两个核心创新：(1) **Knowledge Augmentation (KA)**：对每个输入进行 M 次前向传播，每次以概率 λ 随机采样 N-1 个 expert、以概率 1-λ 取 Top N-1 个 expert，从而增广来自不同 expert 组合的知识。使用 student 生成的 pseudo-target 和 reverse KL divergence 进行蒸馏。(2) **Student-Aware Router (SAR)**：先以 student 反馈（reverse KL divergence + auxiliary load balancing loss）训练 MoE 教师的路由器，再使用更新后的路由器聚合所有 expert 输出进行蒸馏。路由器训练中所有 expert 均被激活，输出通过加权求和聚合。

  实验比较：(a) KD baseline (forward KL, Sanh 2019) vs 论文方法；(b) GKD baseline (reverse KL + on-policy) vs 论文方法；(c) ALL（直接激活所有 expert 不训练 router）作为 SAR 的消融；(d) KA 中 M（增广样本数）和 λ（采样概率）的消融实验；(e) 评估 Sheared-Llama 2.7B 密集教师 vs Llama-MoE 教师的效果对比。

- 硬件平台是什么，配置是什么。
  4 张 Intel Gaudi v2 加速器，使用 SynapseAI 1.18.0 框架。

- 模型是什么。数据集和bench分别是什么。
  教师模型：Llama-MoE-3.5B (4/16, 2/8 variants) 和 Llama-MoE-3.0B (2/16)；密集教师：Sheared-Llama-2.7B。学生模型：Sheared-Llama-1.3B（密集模型）。数据集：Dolly (databricks-dolly-15k, 14k train / 500 val / 500 test)、SelfInst (252条)、Vicuna (80条)、S-NI (SUPER-NATURALINSTRUCTIONS test set 9k条)、UnNI (UNNATURALINSTRUCTIONS core set 10k条)。评估指标：ROUGE-L（5个随机种子取平均）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 KAIST，发表于 arXiv 2502.12947。**论文未提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。算法核心流程如下：

  **=== Knowledge Augmentation (KA) ===**
  输入: student qθ, 数据分布 p_x, 教师前向次数 M, 训练步数 K, 学习率 η
  输出: 训练后的 student θ

  MoE 教师前向传播使用 Noise Top-k Gating (Shazeer et al., 2017):
  ```
  H(x)_i = (x·W_g)_i + StandardNormal() · Softplus((x·W_noise)_i)   // gate logits
  G(x) = Softmax(KeepTopK(H(x), k))                                     // gate probs
  y = Σ_i G(x)_i · E_i(x)                                              // expert output aggregation
  ```

  KA 将 expert 选择扩展为 N-1 个:
  ```
  E = { Sampled N-1 experts (按 gate prob 采样)  w.p. λ,
        Top N-1 experts                        w.p. 1-λ }
  KA(v, E)_i = { v_i  if expert i ∈ E,
                 -∞   otherwise }
  G^KA(x) = Softmax(KA(H(x), E))
  ```

  KA 训练流程（Algorithm 1）:
  ```
  For each step k = 1..K:
      Sample request x from p_x
      Sample response y from qθ(·|x)          // student 生成 pseudo-target
      For m = 1..M:                            // 增广 M 次
          y_teacher = MoE_forward_with_KA(x)   // 用 KA 策略选 N-1 experts
          θ ← θ - η·∇ D_KL(qθ(·|x) || p_KA(·|x))  // reverse KL + student on-policy
  ```

  **=== Student-Aware Router (SAR) ===**
  SAR 在每次迭代包含两个阶段：路由器更新 + 知识蒸馏。

  SAR 路由器训练（Algorithm 2）:
  ```
  For each step k = 1..K:
      Sample request x from p_x
      Sample response y from qθ(·|x)          // student 生成 pseudo-target
      // 阶段1: 路由器更新（仅更新 W_g 和 W_noise）
      W_g ← W_g - η·∇ L_SAR
      W_noise ← W_noise - η·∇ L_SAR
      // L_SAR = D_KL(qθ(·|x) || p_all(·|x)) + β·L_b
      // 其中 p_all 使用所有 experts（全激活）
      // L_b = CV(m)² + CV(P)²  是 load balancing loss
      // 阶段2: 知识蒸馏
      y_teacher = MoE_with_updated_router(x)  // 用更新后 router，激活所有 experts
      θ ← θ - η·∇ D_KL(qθ(·|x) || p_SAR(·|x))
  ```

  超参数：λ=0.05, M=2, β=0.01, batch size=16, LR=1e-5 (student & router), epochs=10, max_seq_len=512, AdamW optimizer。temperature=1.0, top-k=0, top-p=1.0 for generation。

  算法pipeline分为两部分：QESC（量化）和 PESF（剪枝）。

  **=== QESC: Quantization with Expert-Selection Calibration ===**
  输入: MoE模型M（FP16），校准数据集C（WikiText2 128条序列），目标位宽B（expert 2/2.5/3-bit）
  输出: 量化后的MoE模型M_q
  ```
  For each layer l in [0..L-1]:
      1. 量化 MHSA 模块到 4-bit（使用 group-wise GPTQ, group size=128）
      2. 获取该层输入 x_l（从校准集前向传播）
      3. 路由器校准:
         对 MoE layer l 的每个 router:
           保存原始 router 输出标签: y_full = W_r * x_l  （W_r是全精度router权重）
           获取量化后输入: x_hat_l（经过量化MHSA和已量化expert的激活）
           计算 TopK-MSE Loss:
             L = (1/K) * Σ_{i∈top-K(W_r*x_l)} ((W_r*x_l)_i - (W_r*x_hat_l)_i)²
           优化 router 权重 W_r 以最小化 L（仅微调router，补偿量化干扰）
      4. 量化该层所有 experts 到 B-bit（GPTQ, group-wise asymmetric, group size=128）
      5. 保持 router 为原始精度（FP16）
  ```

  **=== PESF: Pruning based on Expert-Selection Frequency ===**
  输入: MoE模型M（已量化或全精度），输入序列seq（长度l），阈值α
  超参数: 每层N个expert，每token选K个expert
  ```
  For each MoE layer with N experts:
      计算阈值: T = (l * K / N) * α
      统计序列中该层每个expert被选中的次数 c_i
      For each expert i:
          if c_i < T:
              剪枝 expert i（不计算其输出）
          else:
              正常计算 expert i 输出
      对未剪枝的expert输出加权求和得到最终输出
  ```

  **=== 组合使用 EAC-MoE = QESC + PESF ===**
  ```
  1. 离线阶段: 使用QESC对模型进行3.03-bit量化（MHSA 4-bit, experts 3-bit, router FP16）
  2. 在线推理（prefill阶段）:
     对每个输入序列:
       For each MoE layer:
           使用量化expert权重 + BitBLAS混合精度计算
           执行PESF动态剪枝（α=0.3）
           仅计算被保留的expert
  ```

  **关键数学公式：**
  
  量化重建问题: argmin_{W_q} ||WX - W_qX||_2²  (GPTQ使用Hessian近似 H=2XX^T)
  
  专家选择概率（token x）: r = {r_0,...,r_{N-1}}, s = Softmax(r), 选top-K
  
  MoE输出: z = Σ_{j=0}^{K-1} (s_{e_j} / Σ_{i=0}^{K-1} s_{e_i}) · E_{e_j}(x)
  
  TopK-MSE Loss: L = (1/K) Σ_{i∈top-K(W_r*x)} ((W_r*x)_i - (W_r*x̂)_i)²
  
  剪枝条件: c_i < (l*K/N) * α  → 剪枝expert i

  **核心数据流**：
  全精度输入 x → 量化MHSA(4-bit) → Router(top-K选择+动态剪枝决策) → 量化Expert(B-bit, GPTQ group-wise) → 加权求和输出
  校准过程使用BitBLAS处理量化权重的混合精度BLAS操作。K值设置：Phi3.5-moe=8, Deepseek-moe-16b-base=20, Qwen1.5-MoE-A2.7B=20（通过MMLU上的网格搜索确定最优K值）。

## Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一个研究 LLM 自适应计算的框架，分为三个阶段：(1) **Stage 1: 训练 Duo FFN 模块**：在每层 FFN 中并排放置一个 big FFN（inner dim=10240）和一个 small FFN（inner dim=640，小 16 倍），共享 attention；训练时以 0.5 概率随机路由 token 到 big 或 small 模块，使两个模块可互换使用。(2) **Stage 2: Oracle 引导的最优路由**：对每条输入序列，穷举所有可能的路由路径（仅 big/small 二选一时为 2^n 条，加入 skip 时为 3^n 条），在固定计算预算约束下选择使 perplexity 最低的路由，作为理论最优路由上界。(3) **Stage 3: 学习路由近似 Oracle**：训练一个类似 MoE 的 learnable router（每层一个线性层 W_{r,l}），通过 soft routing（带温度 τ 的 softmax）结合 budget loss 约束全局 big 模块使用比例（而非 per-layer load balancing），学习近似 oracle 的最优路由。实验比较：(a) Oracle vs 最优随机模式 vs trained router 在不同预算下的 perplexity；(b) Oracle 在不同 big layer 预算下的路由模式（C4 + Code holdout）；(c) Trained router 的路由模式与 oracle 的对比；(d) Token difficulty 分析（small model loss vs loss gap）；(e) Duo-LLM 与同等 FLOPs dense 模型的 accuracy 对比（arc_easy, hellaswag）；(f) 从 scratch 训练 vs freeze big + fine-tune small 的对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练硬件平台和 GPU 配置。论文提到未来可以考虑基于 Megablocks 的 block-sparse matrix multiplication 在单 GPU 上高效执行 Duo-LLM，但论文未提供具体硬件规格。

- 模型是什么。数据集和bench分别是什么。
  模型：12 层 Llama 2 风格架构，hidden dim=2560，共 1.399B 参数（big FFN: 944M，small FFN: 59M，attention: 314M，其余为 embeddings）。训练数据：300B tokens，来源包括 FineWeb、Wiki、Flan（来自 Dolma）、Python code（Stack-v2）。评估 holdout sets：(1) C4 validation set 中随机采样 1024 条；(2) GitHub MIT license 的 Python code 中随机采样 1024 条。Benchmarks：arc_easy、hellaswag（用于与 dense 模型对比）。附录中在 OPT-1.3B 上进行了额外路由实验。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 Apple，论文未提供开源代码仓库。算法核心流程如下：

  ```
  === Stage 1: Duo FFN Training (Random Routing) ===
  For each layer l in [1..L]:
      h = Attention(x)              # shared attention
      # Random routing with p=0.5
      if random() < 0.5:
          h_out = BigFFN_l(h)       # inner_dim = 10240
      else:
          h_out = SmallFFN_l(h)     # inner_dim = 640 (16x smaller)
      x = x + h_out                 # residual connection

  === Stage 2: Oracle Optimal Routing (Exhaustive Search) ===
  Given: input sequence x, budget B (e.g., 4 big layers out of 12)
  For each possible route r in {0,1}^L (0=small, 1=big):
      if sum(r) == B:               # meet budget constraint
          forward(x, route=r)       # execute with chosen modules
          loss_r = CrossEntropy(output, labels)
  optimal_route = argmin_r(loss_r)

  Oracle routing with skip: {0=small, 1=big, 2=skip}^L
  (3^L total routes, exhaustively searched)

  === Stage 3: Learned Router Training ===
  For each layer l:
      h_l = x @ W_{r,l}                       # router logits
      P_{l,big} = softmax(h_l / τ)            # routing probabilities
      P_{l,small} = 1 - P_{l,big}
      H_big = BigFFN_l(x)
      H_small = SmallFFN_l(x)
      output_l = P_{l,big} * H_big + P_{l,small} * H_small  # soft combination

  Budget loss (across all layers, not per-layer):
      L_budget = (mean(P_{:,big}) - target_budget)^2
      L_total = L_CE + α * L_budget

  Temperature τ gradually increased for hard assignment.
  ```

## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包含三个层面的算法pipeline创新：(1) **Expert Partition（专家划分）**：在 post-training 阶段将预训练 MoE 模型中每个 expert 的神经元划分为更细粒度的 experts，包括两种变换——**Complete Transformation**（完整变换）：重复 gating network 权重 P 倍、均匀划分 expert neurons 为 P 个 finer-grained experts、将 down-projection 权重 W₂ 缩放 P 倍以保持数学一致性，`y_i^P = y_i / P`（需将 W₂ 乘以 P 补偿），最终实现 `f_e(x) = Σ_{p=1..P} f_{e,p}(x)`；**Partial Transformation**（部分变换）：保持原始 gating network 不变，仅重复 gating scores 并将 expert indices 重映射，`I^P = [i₁P, i₂P, ..., i_KP, i₁P+1, ..., i_KP+P-1]`，不缩放 W₂ 权重。(2) **Token-Expert Computation Dropping**：包括 **1T-Drop**（单一阈值）：对每个 token 在每层 MoE 中计算 Top-K activated experts 的归一化 gating scores，丢弃低于阈值 T¹_drop 的 token-expert 计算，选择性地保留高于阈值的 experts；**2T-Drop**（双阈值）：在 expert partition + neuron reconstruction 之后，对 major sub-expert 使用较低阈值 T²_major、对 minor sub-expert 使用较高阈值 T²_minor，实现对高重要度神经元保留更多计算、低重要度神经元更激进丢弃的策略。(3) **Static Neuron-Level Reconstruction**：基于 calibration samples 对每个 expert 内神经元进行 importance profiling（四种方法：accumulated gate value、accumulated absolute gate value、accumulated gate-up value、accumulated absolute gate-up value），将神经元按重要性重排序并重构为一个 major sub-expert（高重要性神经元）和一个 minor sub-expert（低重要性神经元）。实验比较：(a) 不同 expert partition 配置（P=1/2/4）的 downstream accuracy 和 fine-tuning loss；(b) 1T-Drop vs 2T-Drop (partition only) vs 2T-Drop (partition + reconstruct) 在不同 drop rate 下的 accuracy；(c) 四种 neuron importance profiling 方法的 accuracy 比较；(d) 与 EES、EEP、Wanda 等 sparsity 方法的 accuracy vs speedup 对比。

- 硬件平台是什么，配置是什么。
  8×NVIDIA H20 GPU 服务器节点。使用 PyTorch Distributed framework + NCCL backend。不同部署策略：(a) Mixtral-8×7B 使用 TP=8 在单节点 8×H20 上；(b) OLMoE-Instruct 使用单张 H20 GPU；(c) DeepSeek-V2-Lite-Chat 使用 EP=8 在 8×H20 节点上。

- 模型是什么。数据集和bench分别是什么。
  模型：三个 MoE 模型——(1) Mixtral-8×7B（8 experts, Top-2 selection, SwiGLU FFN experts）；(2) OLMoE-Instruct（64 experts, Top-8 selection）；(3) DeepSeek-V2-Lite-Chat（含 shared expert 架构的 routed experts）。Fine-tuning 数据集：Tulu-3-sft-mixture。Calibration 数据集：MMLU。Benchmarks（zero-shot）：ARC-C、BoolQ、HellaSwag、MMLU、OBQA、PIQA、RTE、WinoGrande；5-shot: GSM8K。Speedup 评估：2,000 条随机 prompts（input length=500, output length=100）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供独立开源代码仓库，但基于 SGLang 框架实现。算法核心流程如下：

  ```
  === Complete Transformation ===
  Input: pre-trained MoE model with E experts, intermediate_size=d_ffn
  Parameter: partition factor P (e.g., P=2, 4)

  Step 1: Expand Gating Network
    Original: W_g ∈ R^{d_model × E} = [h_1, h_2, ..., h_E]
    Transformed: W_g^P ∈ R^{d_model × (E×P)} = [h_1, h_1, ..., h_1, h_2, ..., h_E]
    (each h_e repeated P times)
    Top-K → Top-(K×P)

  Step 2: Partition Expert Weights (per expert e)
    Original: W_1 ∈ R^{d_model × d_ffn}, W_2 ∈ R^{d_ffn × d_model}, W_3 ∈ R^{d_model × d_ffn}
    For p = 0 to P-1:
      W_{1,p} = W_1[:, p*d_ffn/P : (p+1)*d_ffn/P]
      W_{3,p} = W_3[:, p*d_ffn/P : (p+1)*d_ffn/P]
      W_{2,p} = W_2[p*d_ffn/P : (p+1)*d_ffn/P, :] * P  # scale down-proj

  Gating Score for each finer-grained expert:
    s_{e,p} = exp(l_e) / (P * Σ_{i=1..E} exp(l_i)) = s_e / P
    Sum of s_{e,1..P} = s_e (all partitions from same original expert)

  Output consistency: y_i^P = y_i (after W_2 scaling compensation)

  === 1T-Drop (Single Threshold) ===
  For each token x at each MoE layer:
    s = Softmax(x · W_g)                     # original gating scores
    {e_1, ..., e_K} = TopK(s, K)             # select top-K experts
    s_norm = [s_{e_j} / Σ_{k=1..K} s_{e_k}]  # normalize selected scores
    For each e_j:
      if s_norm[j] < T_drop^1: skip f_{e_j}(x)  # drop computation
    y = Σ_{j: s_norm[j] >= T_drop^1} s_{e_j} · f_{e_j}(x)

  === 2T-Drop with Neuron Reconstruction ===
  Pre-processing (Static, done once per model):
    For each expert e:
      For each neuron n in expert:
        Importance_e[n] = Σ_{samples} |Swish(x · W_1^n)|  # accumulated abs gate
        # OR: Importance_e[n] = Σ |Swish(x · W_1^n) ⊙ (x · W_3^n)|  # abs gate-up
      Sort neurons by Importance descending
      major_expert = neurons[0 : d_ffn/2]    # top 50% high-importance
      minor_expert = neurons[d_ffn/2 : d_ffn] # bottom 50% low-importance

  Inference (Dynamic):
    For each token x at each MoE layer:
      s_norm = normalized gating scores of Top-K experts
      For each activated expert e_j:
        if s_norm[j] < T_major^2: skip entirely
        elif s_norm[j] < T_minor^2: compute only major_expert_{e_j}(x)
        else: compute full expert_{e_j}(x)
      y = weighted sum of computed expert outputs
  ```


- 属于算法pipeline的实现是什么？实验比较什么？
  实现是将 MoE 训练中的 Load-balancing Loss (LBL) 从 **micro-batch 级别**改为 **global-batch 级别**计算，包含两个关键技术：(1) **跨并行组同步专家选择频率**：在各 Data Parallel 组之间同步专家选择频率 f_i（形状仅 N_E），用全局 f̄_i 替换本地 f_i 计算 LBL，从而将负载均衡约束从序列级放松到语料库级；(2) **Buffer 机制近似 Global-Batch**：当计算节点有限、微批总和小于全局批大小时，在 Gradient Accumulation 的各步中缓冲同步后的专家选择计数 c_i，逐步累积逼近 global f̄_i。实验比较：在不同规模 MoE 模型（3.4A0.6B、15A2.54B、43A6.6B）上对比 micro-batch LBL (LBL_micro) vs global-batch LBL (LBL_global) 的 pretraining PPL 和下游 benchmark 表现，同时对比 Auxiliary-Loss-Free 方法在 micro/global 条件下的差异，通过 Shuffle LBL_micro 消融实验区分 token 数量与 token 多样性两个影响因素。

- 硬件平台是什么，配置是什么。
  GPU 集群，支持 Data Parallelism 和 Expert Parallelism。3.4A0.6B 模型使用纯数据并行，最大 micro-batch size 为 4，可在单节点 8 GPU 内或跨 16 节点同步 f_i，Balance BSZ 可达 512。15A2.54B 和 43A6.6B 模型使用 Expert Parallelism，每 GPU micro-batch size 分别为 2 和 1。

- 模型是什么。数据集和bench分别是什么。
  模型：MoE-based LLMs，使用 fine-grained experts + shared experts 架构。三个规模：(1) 3.4A0.6B（64 total experts, top4 activated, 4 shared experts）；(2) 15A2.54B（160 total experts, top4 activated, 4 shared experts）；(3) 43A6.6B（160 total experts, top4 activated, 4 shared experts）。均使用 softmax gating、z-loss、dropless routing（dMoE 风格）。训练数据：120B 和 400B 高质量多语言 tokens（含 multilinugal、math、general knowledge）。Sequence length 4096。Benchmark：Hellaswag、MMLU、GSM8k、C-eval，以及 held-out PPL 测试集（SFT-EN、EN-Literature、SFT-Code、SFT-Math、SFT-ZH、ZH-Law、ZH-Literature、SFT-Other）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供明确的开源代码链接。方法基于开源 MoE 训练框架（Deepspeed-MoE、Tutel、Megablocks、Megatron-Core）的修改。核心算法流程如下：
  ```
  # 标准 micro-batch LBL (现有框架)
  for each parallel group j in [1..N_P]:
    f_i^j = expert_selection_frequency_in_group_j  # shape: [N_E]
    P_i^j = average_gating_score_in_group_j         # shape: [N_E]
  LBL_micro = (1/N_P) * sum_j( N_E * sum_i(f_i^j * P_i^j) )

  # Global-batch LBL (本文方法)
  for each parallel group j:
    f_i^j = expert_selection_frequency_in_group_j
  f_bar_i = all_reduce(f_i^j) / N_P     # 同步得到全局频率
  for each parallel group j:
    LBL = N_E * sum_i(f_bar_i * P_i^j)  # 用全局频分布替换本地
  LBL_global = (1/N_P) * sum_j(LBL)
  ```
  当节点有限时使用 Buffer 近似（Algorithm 1）：
  ```
  c_i = 0  # 各专家缓冲区，shape: [N_E]
  for each GA step:
    c_i += sync_count_i        # 累加同步后的选择计数
    f_i = c_i / total_tokens   # 用缓冲区估算全局频
    LBL = N_E * sum(f_i * P_i) # 用近似 f_i 计算 LBL
  optimizer.step()
  c_i = 0  # optimizer step 后清空
  ```
  通信仅涉及 f_i ∈ R^{N_E}（专家数量维向量），远小于 token-expert 选择矩阵。同步+LBL 计算可与其他网络计算 overlap，额外开销 <3% latency。Global-batch LBL 将约束从"每个 micro-batch 内均匀分布"放松为"全局均匀分布"，使得 expert specialization 按语料级数据域自然涌现。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括四部分：**(1) Multi-head Latent Attention (MLA)**：通过低秩键值联合压缩将 KV cache 压缩为 latent vector c_t^{KV}（d_c=512），同时解耦 RoPE 规避低秩压缩与位置编码的不兼容，KV cache 量级远小于 MHA。**(2) DeepSeekMoE with Auxiliary-Loss-Free Load Balancing**：使用 1 shared expert + 256 routed experts（每 token 激活 8 个），通过动态 bias 项 b_i 实现无需 auxiliary loss 的负载均衡，仅保留极小互补 sequence-wise balance loss（α=0.0001），使用 sigmoid gating + top-K affinity normalization。**(3) Multi-Token Prediction (MTP)**：1-depth 的 MTP 模块，每个 position 额外预测下下个 token，保持完整 causal chain，MTP 模块在推理时可丢弃或用于 speculative decoding。**(4) FP8 Mixed Precision Training**：fine-grained quantization（activations: 1×128 tile-wise, weights: 128×128 block-wise），promotion to CUDA Cores for high-precision FP32 accumulation，EMA stored in CPU。实验比较：(a) DeepSeek-V3-Base vs DeepSeek-V2-Base, Qwen2.5 72B Base, LLaMA-3.1 405B Base 的综合 benchmark 性能；(b) DeepSeek-V3-Chat vs DeepSeek-V2-0506, DeepSeek-V2.5-0905, Qwen2.5 72B Instruct, LLaMA-3.1 405B Instruct, Claude-3.5-Sonnet-1022, GPT-4o-0513；(c) MTP 消融实验；(d) auxiliary-loss-free balancing 消融实验。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU 集群，共 2048 张 GPU。每个节点 8 张 H800 GPU，节点内通过 NVLink (160 GB/s) + NVSwitch 互联，节点间通过 InfiniBand (50 GB/s) 互联。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-V3 (61 层 Transformer, hidden dim=7168, 128 heads, d_h=128, d_c=512, d_c'=1536, d_h^R=64, 1 shared + 256 routed experts, K_r=8, M=4 node limit, 671B total / 37B activated)。数据集：14.8T tokens 高质量多语料（增强数学和编程比例，扩展多语言覆盖），BBPE tokenizer，vocab=128K，FIM rate=0.1。Base Benchmarks：MMLU, MMLU-Redux, MMLU-Pro, MMMLU, C-Eval, CMMLU, HellaSwag, PIQA, ARC (Easy/Challenge), BBH, TriviaQA, NaturalQuestions, RACE (Middle/High), DROP, C3, CMRC, CLUEWSC, WinoGrande, Pile-test, CCPM, GSM8K, MATH, MGSM, CMath, HumanEval, LiveCodeBench-Base, MBPP, CRUXEval, AGIEval。Chat Benchmarks 额外：IFEval, FRAMES, LongBench v2, GPQA, SimpleQA, C-SimpleQA, SWE-Bench Verified, Aider, LiveCodeBench (08-11/2024), Codeforces, CNMO 2024, AIME 2024, Arena-Hard, AlpacaEval 2.0, RewardBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，checkpoints 发布于 https://github.com/deepseek-ai/DeepSeek-V3。

  **MLA + DeepSeekMoE 算法流程（以推理阶段单 token 为例）**：
  ```
  Input: h_t ∈ R^d (d=7168)
  
  // Attention: MLA
  c_t^{KV} = W^{DKV} @ h_t          // [512×7168] @ [7168×1] → d_c=512
  k_t^C  = W^{UK} @ c_t^{KV}         // W^{UK} absorbed into W^{UQ} at inference
  v_t^C  = W^{UV} @ c_t^{KV}         // W^{UV} absorbed into W^O at inference  
  k_t^R  = RoPE(W^{KR} @ h_t)        // decoupled RoPE key, d_h^R=64 per head
  // Only c_t^{KV} and k_t^R cached → minimal KV cache
  
  c_t^Q  = W^{DQ} @ h_t             // query compression, d_c'=1536
  q_t^C  = W^{UQ} @ c_t^Q
  q_t^R  = RoPE(W^{QR} @ c_t^Q)
  q_{t,i} = [q_{t,i}^C; q_{t,i}^R]  // per-head: 128 + 64 = 192
  
  o_{t,i} = sum_j Softmax(q_{t,i}^T k_{j,i} / sqrt(192)) v_{j,i}^C
  u_t = W^O @ concat(o_{t,1},...,o_{t,128})
  
  // FFN: DeepSeekMoE with Aux-Loss-Free Routing
  h_t' = u_t + sum_{i=1}^{1} FFN_i^{(s)}(u_t)                    // shared expert
             + sum_{i∈TopK} g_{i,t} * FFN_i^{(r)}(u_t)           // 8 routed experts
  
  // Routing (no auxiliary loss):
  s_{i,t} = Sigmoid(u_t^T e_i)                                   // affinity score
  selected = TopK({s_{j,t} + b_j | j=1..256}, K_r=8)             // bias b_j for balance
  g_{i,t}' = Normalize({s_{i,t} for i in selected})              // gating values
  // After each step: b_i = b_i + γ*sign(target_load - actual_load)
  ```
  
  **MTP 训练流程**：
  ```
  // Main model output: h_i^0 (i-th token representation)
  // MTP module depth D=1 (predict 2nd next token)
  h_i'^{1} = M_1 @ [RMSNorm(h_i^0); RMSNorm(Emb(t_{i+1}))]  // [d×2d] concat
  h_{1:T-1}^{1} = TRM_1(h_{1:T-1}'^{1})                      // Transformer block
  P_{i+2}^{1} = Softmax(OutHead(h_i^{1}))                     // Shared output head
  L_MTP = λ/D * sum_k CrossEntropy(P^{k}_{2+k:T+1}, t_{2+k:T+1}), λ=0.3
  ```

## DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **(1) Multi-head Latent Attention (MLA)**：通过低秩键值联合压缩将 KV cache 压缩为 latent vector，同时使用解耦 RoPE 策略规避 RoPE 与低秩压缩的不兼容问题，使得 KV cache 仅为 MHA 的约 4%，但性能优于 MHA。**(2) DeepSeekMoE**：采用细粒度专家分割（fine-grained expert segmentation）和共享专家隔离（shared expert isolation），配合 Device-Limited Routing、三层辅助损失（Expert-Level / Device-Level / Communication Balance Loss）和 Token-Dropping Strategy，在保证负载均衡的同时实现经济的训练。实验比较：(a) MLA vs MHA/GQA/MQA 的 KV cache 开销和 benchmark 性能；(b) DeepSeek-V2 (236B total/21B activated) vs DeepSeek 67B (dense)、Qwen1.5 72B、LLaMA3 70B、Mixtral 8x22B 的综合 benchmark 性能；(c) 训练成本（GPU hours/trillion tokens）和推理吞吐（tokens/sec on 8xH800）。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU 集群。每个节点 8 张 H800 GPU，节点内通过 NVLink 和 NVSwitch 互联，节点间通过 InfiniBand 互联。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-V2 (60层 Transformer, hidden dim=5120, 128 heads, d_h=128, d_c=512, d_c'=1536, d_h^R=64, 2 shared + 160 routed experts, K_r=6, 236B total/21B activated)。数据集：8.1T tokens 双语（中英文约 12% 中文更多）预训练语料，BBPE tokenizer，vocab size=100K。Benchmarks：MMLU, C-Eval, CMMLU, HellaSwag, PIQA, ARC (Easy/Challenge), BBH, TriviaQA, NaturalQuestions, RACE (Middle/High), DROP, C3, CMRC, WinoGrande, CLUEWSC, Pile-test, CHID, CCPM, GSM8K, MATH, CMath, HumanEval, MBPP, CRUXEval, AGIEval。Chat 版本额外评估：MT-Bench, AlpacaEval 2.0, AlignBench, IFEval, LiveCodeBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，checkpoints 发布于 https://github.com/deepseek-ai/DeepSeek-V2。

  **MLA 算法流程（以推理阶段单 token 为例）**：
  ```
  Input: h_t ∈ R^d (第 t 个 token 的 hidden state, d=5120)
  
  // 1. Query 低秩压缩（训练时减少激活内存）
  c_t^Q = W^{DQ} @ h_t           // [d_c' × d] @ [d × 1], d_c'=1536
  q_t^C = W^{UQ} @ c_t^Q         // [d_h*n_h × d_c'] @ [d_c' × 1], d_h*n_h=128*128=16384
  q_t^R = RoPE(W^{QR} @ c_t^Q)   // [d_h^R*n_h × d_c'] @ [d_c' × 1], d_h^R*n_h=64*128=8192
  q_{t,i} = concat(q_{t,i}^C, q_{t,i}^R)  // per-head: [128+64] = [192]
  
  // 2. KV 低秩联合压缩（核心：大幅减少 KV cache）
  c_t^{KV} = W^{DKV} @ h_t       // [d_c × d] @ [d × 1], d_c=512（仅此需缓存！）
  k_t^C = W^{UK} @ c_t^{KV}      // 推理时 W^{UK} 被吸收进 W^{UQ}
  v_t^C = W^{UV} @ c_t^{KV}      // 推理时 W^{UV} 被吸收进 W^O
  k_t^R = RoPE(W^{KR} @ h_t)     // 解耦的 RoPE key，也需缓存
  k_{t,i} = concat(k_{t,i}^C, k_t^R)  // per-head: for attention
  
  // 3. Attention 计算
  o_{t,i} = sum_{j=1..t} Softmax(q_{t,i}^T @ k_{j,i} / sqrt(d_h+d_h^R)) * v_{j,i}^C
  u_t = W^O @ concat(o_{t,1}, ..., o_{t,128})
  
  // KV cache per token: (d_c + d_h^R) * l = (512 + 64) * 60 = 34560 elements
  // 相比 MHA: 2 * n_h * d_h * l = 2 * 128 * 128 * 60 = 1,966,080 elements
  // MLA KV cache 仅为 MHA 的 1.76%
  ```

  **DeepSeekMoE 算法流程**：
  ```
  Input: u_t ∈ R^d (FFN 输入)
  
  // 1. 共享专家（所有 token 都经过）
  h_shared = sum_{i=1}^{N_s} FFN_i^{(s)}(u_t)   // N_s=2
  
  // 2. 路由专家（Top-K 选择）
  s_{i,t} = Softmax_i(u_t^T @ e_i)              // token-expert affinity, N_r=160
  g_{i,t} = s_{i,t} if s_{i,t} ∈ TopK({s_{j,t}}, K_r=6) else 0
  
  // 3. Device-Limited Routing（约束到最多 M=3 个设备）
  // 先选 M 个 affinity 最高的设备，再在设备内做 Top-K
  
  // 4. 输出
  h_t' = u_t + h_shared + sum_{i=1}^{N_r} g_{i,t} * FFN_i^{(r)}(u_t)
  
  // 辅助损失:
  L_ExpBal = α1 * sum(f_i * P_i)      // expert 级负载均衡, α1=0.003
  L_DevBal = α2 * sum(f_i' * P_i')    // device 级负载均衡, α2=0.05
  L_CommBal = α3 * sum(f_i'' * P_i'') // 通信负载均衡, α3=0.02
  ```

## DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三部分：**(1) Standard MoE for NLG**：将 MoE 架构应用于自回归 NLG 模型（GPT-like Transformer），每两层的前馈层之一替换为 MoE 层（128 experts, top-1 gating），在 300B tokens 训练下实现与 4-5x 参数量 dense 模型相同的质量。**(2) PR-MoE (Pyramid-Residual MoE)**：结合两种创新 —— Pyramid-MoE（深层 MoE 层使用更多 experts，如最后两层 2x experts）+ Residual-MoE（每 token 同时经过固定 MLP 和选定 expert，等价于 Top-2 gating 的精度但仅需 Top-1 通信量），参数效率提升 3x。**(3) MoS (Mixture-of-Students)**：Staged Knowledge Distillation，在预训练早期使用 KD（teacher PR-MoE → student PR-MoE，student 深度减少 12.5%），400K steps 后停用 KD 仅用标准 LM loss 继续训练，解决学生模型容量不足造成的 underfitting。MoS 进一步减少模型大小至 3.7x。实验比较：(a) 350M+MoE-128 vs 350M dense, 1.3B+MoE-128 vs 1.3B/6.7B dense 的 validation loss 和 zero-shot；(b) PR-MoE vs Standard MoE 的参数量和精度对比（Table 4）；(c) 5 种 MoE 架构的 ablation study（Standard-MoE-32/128, Pyramid-MoE, Residual-MoE, PR-MoE）；(d) PR-MoE+MoS vs PR-MoE 直接减层的 zero-shot 对比（Table 5）；(e) Staged KD vs Full KD vs No KD 的 validation loss 对比。

- 硬件平台是什么，配置是什么。
  128 张 NVIDIA A100 GPU（Azure ND A100 instances），通过 NCCL, Mellanox OFED, Sharp, CUDA 优化。使用 DeepSpeed 进行数据和 expert parallel 训练。

- 模型是什么。数据集和bench分别是什么。
  模型：350M (24 layers, 1024 hidden, 16 heads), 1.3B (24 layers, 2048 hidden, 16 heads), 6.7B (32 layers, 4096 hidden, 32 heads) 的 dense baseline；350M+MoE-128 (13B params), 1.3B+MoE-128 (52B params) 的 standard MoE；350M+PR-MoE-32/64 (4B params), 1.3B+PR-MoE-64/128 (31B params) 的 PR-MoE；350M+PR-MoE+L21+MoS (3.5B), 1.3B+PR-MoE+L21+MoS (27B) 的 MoS。训练数据：与 MT-NLG 相同的 300B tokens，sequence length 2K。Benchmarks：6 个 zero-shot 任务 —— LAMBADA（补全预测）、PIQA（常识推理）、BoolQ 和 RACE-h（阅读理解）、TriviaQA 和 WebQs（问答）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，代码和文档在 DeepSpeed GitHub (https://github.com/microsoft/DeepSpeed) 和 DeepSpeed 官网 (https://www.deepspeed.ai/)。论文 ICML 2022。

  **PR-MoE 算法流程（以 350M+PR-MoE-32/64, 24 layers 为例）**：
  ```
  Input: token embeddings x_1,...,x_S ∈ R^{M}, M=1024

  // 前 10 个 MoE 层使用 32 experts (Pyramid-MoE shallow)
  For layer l ∈ MoE layers 1..10:
    // 每层 MoE 位于两个 Attention 层之间
    h = Attention(x)
    // Residual-MoE: 固定 MLP + 选定 expert 各自处理
    h_mlp = W2_fixed @ GeLU(W1_fixed @ h)        // 固定 dense MLP（所有 token 共享）
    gate_logits = W_gate @ h                        // [32] per token
    expert_id = argmax(gate_logits)                 // Top-1 gating
    h_expert = W2_expert[expert_id] @ GeLU(W1_expert[expert_id] @ h)
    x = x + h_mlp + h_expert                       // 残差连接相加

  // 后 2 个 MoE 层使用 64 experts (Pyramid-MoE deep)
  For layer l ∈ MoE layers 11..12:
    h = Attention(x)
    h_mlp = W2_fixed @ GeLU(W1_fixed @ h)
    gate_logits = W_gate @ h                        // [64] per token
    expert_id = argmax(gate_logits)                 // Top-1 gating
    h_expert = W2_expert[expert_id] @ GeLU(W1_expert[expert_id] @ h)
    x = x + h_mlp + h_expert
  ```

  **MoS Staged KD 训练流程**：
  ```
  // Teacher: 24-layer PR-MoE (1.3B+PR-MoE-64/128, 31B params)
  // Student: 21-layer PR-MoE (1.3B+PR-MoE-64/128, 27B params)

  For training step t = 1..T:
    x = next_batch()
    teacher_logits = TeacherPRMoE(x)         // 教师 soft label
    student_logits = StudentPRMoE(x)         // 学生预测

    L_CE = CrossEntropy(student_logits, y)   // 标准语言模型 loss
    L_KD = KLDiv(student_logits, teacher_logits)  // 蒸馏 loss

    if t <= 400K:
      L = L_CE + α * L_KD                   // 使用 KD loss + CE loss
    else:
      L = L_CE                               // 停用 KD，仅优化标准 LM loss
    optimizer.step(L)
  ```
  关键创新：发现全程 KD 在训练后期（>400K steps）开始伤害精度，原因在于学生容量不足导致 underfitting —— 学生无法同时最小化 L_CE 和 L_KD。Staged KD 通过早期 KD 提供引导、后期放开自主优化来解决此问题。

## DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DeRS (Decompose, Replace, Synthesis)** 范式，通过分解 upcycled MoE 专家为专家共享基础权重 + 专家专属 delta 权重，并对 delta 权重使用稀疏化或量化等轻量表示来提升参数效率。包含两种应用：
  
  **DeRS Compression**（推理阶段压缩）：对已训练的 vanilla upcycled MoE 模型，将 N 个专家权重 {W_1,...,W_N} 分解为共享基础权重 W_base 和 N 个 delta 权重 {Δ_1,...,Δ_N}（W_i = W_base + Δ_i），然后对 delta 权重应用后处理轻量技术：
  - DeRS-Sparsification：随机丢弃 delta 权重中比例为 p 的元素，以紧凑向量存储稀疏矩阵。MoE 层参数从 N·d·d_h 降至 (1+N·(1-p))·d·d_h
  - DeRS-Quantization：将 delta 权重从 K 比特量化为 k 比特。存储成本从 N·K 降至 K+N·k
  推理时按需合成专家权重：Ŵ_i = W_base + F_post(Δ_i)
  
  **DeRS Upcycling**（训练阶段高效 upcycling）：不复制原始 FFN N 次构建专家，而是将 N 个专家分解为一个可训练共享权重 W_shared 和一个专家专属增量权重 F_pre(Δ_i)，训练和推理时通过 W_i = W_shared + F_pre(Δ_i) 合成专家权重：
  - DeRS-SM（Sparse Matrix）：使用索引向量 I 和值向量 V 两个紧凑行向量表示稀疏矩阵，通过 torch.scatter 映射回去。训练参数从 N·d·d_h 降至 (1+N·(1-p))·d·d_h
  - DeRS-LM（Low-rank Matrix）：使用两个低秩矩阵 A∈R^{d×r} 和 B∈R^{r×d_h} 表示增量权重，F_pre(Δ_i) = A_i·B_i。训练参数从 N·d·d_h 降至 d·d_h + N·r·(d+d_h)
  
  实验比较（三大任务六种 MoE 架构）：
  
  **通用多模态任务（MoE-LLaVA）**：
  - DeRS Compression：在 MoE-LLaVA-StableLM/Qwen/Phi 上对比不同 drop rate (0.2~0.99) 和量化位宽 (8/4/2/1 bit) 下的性能。drop rate 0.9 时 MoE 层参数减少 65% 无性能损失；2-bit 量化可达 16→2 位宽降低。极端设置 (0.99 drop rate / 1-bit) 仍无明显性能退化
  - DeRS Upcycling：对比 Vanilla Upcycling vs DeRS-SM vs DeRS-LM 在三个 MoE-LLaVA 架构上的 Added Params 和 Overall 性能
  
  **医学多模态任务（Med-MoE）**：
  - DeRS Compression：在 Med-MoE-StableLM/Phi 上对比不同压缩率。极端压缩 (remove 99% 元素或 1-bit 量化) 对性能影响可忽略
  - DeRS Upcycling + Extended DeRS Upcycling（扩展至 universal FFN）
  
  **代码生成任务（Coder-MoE）**：
  - DeRS Compression：delta 权重冗余度比医学任务低（因 dense model 未进行先验微调），drop rate 0.6 或 2-bit 量化无性能损失
  - DeRS Upcycling + Extended DeRS Upcycling
  
  **消融实验**：
  - 冻结共享基础 FFN vs 不冻结：冻结导致 DeRS-SM 下降 1.3%，DeRS-LM 下降 1.6%
  - 稀疏率/秩超参数扫描：DeRS-SM 低稀疏率更好；DeRS-LM rank 1/4/16 效果相近，rank 64 性能退化
  - 成本分析：DeRS-LM (4rank) 减少模型大小 52.7%、训练内存 21.2%、推理内存 43.8%，性能提升 0.7%

- 硬件平台是什么，配置是什么。
  - 通用多模态任务：8× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=4，Gradient Accumulation Steps=4
  - 医学多模态任务：4× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=8，Gradient Accumulation Steps=2
  - 代码生成任务：8× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=4，Gradient Accumulation Steps=2
  - 使用 CFFF platform of Fudan University

- 模型是什么。数据集和bench分别是什么。
  **模型**：
  - MoE-LLaVA 框架：CLIP-Large 视觉编码器 + 语言骨干（StableLM-2-1.6B / Qwen-1.8B / Phi-2-2.7B），每间隔一个 block 的 FFN 层 upcycled 为 4 专家的 MoE 层，top-2 激活
  - Med-MoE 框架：CLIP-Large 视觉编码器 + 语言骨干（StableLM-2-1.6B / Phi-2-2.7B），每间隔一个 block 的 FFN 替换为 universal FFN + 4 专家 MoE 层的并行结构，top-1 激活
  - Coder-MoE 框架：DeepSeek-Coder-Base-1.3B，每个 block 的 FFN 替换为 universal FFN + 4 专家 MoE 层的并行结构，top-1 激活
  
  **数据集与 Benchmark**：
  - 通用多模态：微调 LLaVA-mix-665k；评估 VQA-v2, GQA, VisWiz, ScienceQA-IMG, TextVQA + POPE, MMBench, MM-Vet
  - 医学多模态：微调/评估 VQA-RAD, SLAKE, PathVQA（含开放/封闭式问答）
  - 代码生成：微调 evol-codealpaca-v1 (110K instruction-output pairs)；评估 HumanEval, HumanEval+, MBPP, MBPP+

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码链接，arXiv 页面亦无代码可用声明。以下基于论文描述给出算法 pipeline 伪代码：
  
  **DeRS Compression 流程**：
  ```
  # 输入：已训练的 vanilla upcycled MoE 模型，包含 N 个专家 {W_1,...,W_N}
  # 输出：压缩后的 MoE 模型
  
  # Step 1: Decompose
  W_base = 原始 FFN 权重  # shape: [d, d_h]
  for i in 1..N:
      Δ_i = W_i - W_base  # shape: [d, d_h]
  
  # Step 2: Replace (Sparsification 示例)
  for i in 1..N:
      M_i ~ Bernoulli(p)  # shape: [d, d_h]，每个元素独立 Bernoulli
      F_post(Δ_i) = (1 - M_i) ⊙ Δ_i / (1-p)  # 元素级 drop + rescale
      # 以紧凑格式存储：仅保存非零元素的值和索引
  
  # Step 3: Synthesis（推理时按需合成）
  # 当 Router 选择 expert E_k 时：
  Ŵ_k = W_base + F_post(Δ_k)  # shape: [d, d_h]
  y = x @ Ŵ_k  # 使用合成权重进行 FFN 计算
  ```
  
  **DeRS Upcycling (DeRS-LM) 流程**：
  ```
  # 输入：预训练 Dense 模型的 FFN 权重 W_orig，shape [d, d_h]
  # 输出：参数高效的 MoE 模型
  
  # 初始化
  W_shared = W_orig.clone()  # 专家共享基础权重，可训练
  for i in 1..N:
      A_i = random_init([d, r])  # 低秩矩阵 A，可训练
      B_i = zeros([r, d_h])      # 低秩矩阵 B，可训练（零初始化确保初始 Δ=0）
  
  # 训练/推理时合成专家权重
  for each input x:
      Router_score = TopK(softmax(x @ W_R), k)  # 路由计算
      for i in selected_experts:
          F_pre(Δ_i) = A_i @ B_i          # shape: [d, d_h]，低秩分解
          W_i = W_shared + F_pre(Δ_i)      # 合成专家权重
          y_i = FFN(x, W_i)                # 使用合成权重计算
      y = Σ Router_score_i * y_i
  
  # 可训练参数对比：
  # Vanilla: N * d * d_h 个参数
  # DeRS-LM: d * d_h + N * r * (d + d_h) 个参数
  # DeRS-SM: d * d_h + N * d * d_h * (1-p) 个参数（紧凑存储）
  ```
  
  以 MoE-LLaVA-Phi (d=2560, d_h=10240, N=4) 为例：
  - Vanilla Upcycling：4 × 2560 × 10240 ≈ 2.52B 新增参数
  - DeRS-SM (p=0.9999)：2560×10240 + 4×(2560×10240×0.0001) ≈ 26.2M + 0.26M ≈ 1.11M 有效新增参数（2270× 减少）
  - DeRS-LM (r=1)：2560×10240 + 4×1×(2560+10240) ≈ 26.2M + 2.42M = 2.42M 新增参数（1041× 减少）

## DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DSMoE (Dynamic Sparse Mixture-of-Experts)**，一种将预训练 Dense 模型的 FFN 层矩阵沿 intermediate 维度划分为多个 expert 块，并通过 sigmoid 门控 + straight-through estimator + 稀疏损失实现动态、输入自适应的稀疏激活的方法。三个核心模块：
  
  **FFN Partitioning**：将 SwiGLU FFN 中的上投影矩阵 U、门控矩阵 W、下投影矩阵 V 沿 intermediate 维度均等划分为 n 组（如 LLaMA-7B 中 D=11008 划分为 8×1376），每组构成一个 expert。划分后所有 expert 输出之和在数学上等价于原始 FFN 输出。
  
  **Straight-Through Estimator**：前向传播时通过阈值 τ=0.5 的阶跃函数 G(x) 控制稀疏激活；反向传播时通过 S(x)=sg(G(x))+x-sg(x) 允许梯度穿过未激活 expert 的门控参数 Y_i（公式16），使非激活 expert 也能根据输出 o_i 是否有益于降低损失来更新路由参数，解决"死 expert"问题。
  
  **Sparse Loss**：L = L_LM + (1/LN) Σ G(σ(ĥY_n))，L1 范数惩罚门控激活值，与门控梯度形成对抗效应，鼓励模型主动抑制不重要 expert 的输出。不引入传统 MoE 的 load balancing loss。
  
  实验比较：
  - **Perplexity (Table 1)**：DSMoE vs LLM-Pruner (channel-wise/block-wise)、SparseGPT (非结构化剪枝)、LLaMA-MoE (传统 MoE top-k)，在 LLaMA-1B (激活参数 735M) 和 LLaMA-7B (激活参数 3.93B) 两档
  - **Downstream Benchmarks (Table 2)**：10 个下游任务（HellaSwag/LAMBADA/PIQA/SIQA/StoryCloze/Winogrande + GSM8K/NaturalQs/TriviaQA/WebQs），zero-shot 和 5-shot
  - **Ablation: Straight-Through Estimator (Table 3)**：有 S(x) vs 无 S(x)（仅用 G(x)），PPL 从 7.41 退化至 12.75
  - **Ablation: Piecewise Function G(x) (Fig. 2)**：训练时不使用 G(x)（用连续 sigmoid），推理时加阈值，PPL 随 τ 增大急剧上升
  - **Layer-wise Activation Patterns (Fig. 3)**：热力图分析各层专家激活数分布，发现 W 形激活模式
  - **Threshold Sweep (Table 4)**：τ=0.2~0.8 下 PPL 与激活参数比例的关系

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号和具体硬件配置。论文提及使用 LLaMA-1B 和 LLaMA-7B 模型进行继续预训练，训练 10B tokens 数据，batch size=32，sequence length=1024，learning rate=2e-5，但未披露使用的 GPU 类型、数量和显存配置。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **LLaMA-1B**（基于 LLaMA-3.2-1B）：d=2048, D=8192, 总参数 1.24B, 激活参数 735M (8 experts, D=1024×8)
  - **LLaMA-7B**（基于 LLaMA-2-7B）：d=4096, D=11008, 总参数 6.74B, 激活参数 3.93B (8 experts, D=1376×8)
  
  数据集：
  - 继续预训练数据：Fineweb-edu (通用)、OpenWebMath (数学)、StarCoder (代码)、Cosmopedia (合成数据)，混合后总计 10B tokens，tokenizer 限制最大长度 1024
  
  Benchmarks：
  - 验证集 PPL（从各数据集随机采样 5000 条非重叠样本）
  - Zero-shot: HellaSwag, LAMBADA, PIQA, SIQA, StoryCloze, Winogrande
  - 5-shot: GSM8K (exact match), NaturalQs (exact match), TriviaQA (exact match), WebQs (exact match)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：**论文未提供公开开源代码仓库**。PapersWithCode 页面标注 "No code implementations yet"。论文对应的 arXiv ID 为 2502.12455。

  **DSMoE 算法 Pipeline（基于 PyTorch 伪代码）**：

  ```
  # === 符号 ===
  # x: [B, d] 输入 hidden states
  # U_list, W_list, V_list: 各 n 个切片的 FFN 矩阵
  #   U_i: [d, D/n], W_i: [d, D/n], V_i: [D/n, d]
  # Y: [d, n] 门控网络参数矩阵
  # tau: 激活阈值 (默认 0.5)
  
  def dsmo_e_ffn_forward(x, U_list, W_list, V_list, Y, tau=0.5, training=True):
      """
      DSMoE FFN layer forward pass
      x: [B, d] - input hidden states after attention
      """
      n = len(U_list)  # number of experts
      B, d = x.shape
      
      # Step 1: Compute gate logits and sigmoid
      gate_logits = x @ Y  # [B, n]
      gate_probs = sigmoid(gate_logits)  # [B, n], values in (0, 1)
      
      # Step 2: Piecewise gating with Straight-Through Estimator
      if training:
          # STE: forward uses hard threshold, backward passes gradient
          gate_hard = gate_probs.clone()
          gate_hard[gate_hard <= tau] = 0.0  # G(x) in forward
          gate_values = gate_hard + gate_probs - gate_probs.detach()  # S(x) = sg(G) + x - sg(x)
      else:
          # Inference: only hard threshold
          gate_values = gate_probs.clone()
          gate_values[gate_values <= tau] = 0.0
      
      # Step 3: Compute expert outputs
      outputs = []
      for i in range(n):
          # SwiGLU FFN for expert i
          # o_i = (act(x @ W_i) ⊙ (x @ U_i)) @ V_i
          gate_part = silu(x @ W_list[i])  # [B, D/n]
          up_part = x @ U_list[i]          # [B, D/n]
          expert_out = (gate_part * up_part) @ V_list[i]  # [B, d]
          outputs.append(expert_out)
      
      # Step 4: Weighted sum with gating
      # h = Σ o_i * S(σ(x @ Y_i))
      h = sum(
          outputs[i] * gate_values[:, i:i+1]  # [B, d] * [B, 1]
          for i in range(n)
      )  # [B, d]
      
      # Step 5: Activation count normalization
      # Scale by n / num_active to maintain output norm
      active_mask = (gate_probs > tau).float()  # [B, n]
      num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
      h = h * (n / num_active)  # [B, d]
      
      return h, gate_values, num_active.mean()
  
  # === Loss Computation ===
  def compute_loss(lm_loss, gate_values_list, L, N):
      """
      Total loss = Language Modeling Loss + Sparse Loss
      gate_values_list: list of gate_values from each layer
      L: number of Transformer layers
      N: number of experts per layer
      """
      sparse_loss = 0.0
      for gate_vals in gate_values_list:
          # L1 norm on gated activations (gate_vals already thresholded)
          sparse_loss += gate_vals.sum()
      sparse_loss = sparse_loss / (L * N)
      return lm_loss + sparse_loss
  ```

  **张量计算流程（以 LLaMA-7B, d=4096, n=8, D/n=1376 为例）**：
  
  1. 输入 x: [B, 4096]
  2. 门控计算：x @ Y: [B, 4096] × [4096, 8] → [B, 8]，sigmoid 后得到每个 expert 的激活概率
  3. 硬阈值：STE 前向将 ≤0.5 的值置零，反向保持梯度流
  4. Expert 计算：每个 expert i 执行 x @ W_i [B, 1376] ⊙ x @ U_i [B, 1376] → intermediate [B, 1376]，再 @ V_i [1376, 4096] → [B, 4096]
  5. 加权求和：Σ o_i · gate_i → [B, 4096]
  6. 归一化：× n / num_active → 最终输出 [B, 4096]

## Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **LUFFY**，一个通信高效的分布式 MoE 训练系统，包含两个核心算法级技术：
  
  **Token Condensation（令牌凝聚）**：利用 MoE 训练中被路由到同一 expert 的 token 之间存在高相似度（例如 MoE-TransformerXL 中约 62% 的 token 对被路由到同一 expert 且高度相似）的观察，提出令牌凝聚算法来消除冗余 token 传输。
  - **快速相似度测量（Fast Similarity Measurement）**：将 token 和相似度关系建模为全连接图。三步策略：(1) 被路由到不同 expert 的 token 直接标记为不相似（边权重=0），因为不同 expert 设计为处理不同输入类型；(2) 极端相似/不相似的 token 对根据历史相似度信息（前一 block 的相似度值）直接分配边权重为 1 或 0，跳过计算；(3) 仅对剩余高度不确定的 token 对进行真实 cosine 相似度计算。
  - **自适应凝聚策略（Adaptive Token Condensation）**：动态调整相似度阈值 $h_t = 1/(1+\exp(l_{norm}))$，其中 $l_{norm} = (l_{ini} - l_{t-1})/l_{ini}$。训练早期 $l_{norm}$ 小 → 阈值大 → 保留更多 token 保证收敛；训练后期 $l_{norm}$ 大 → 阈值降低 → 凝聚更多 token 减少通信。

  实验比较：
  - **End-to-End Performance (Fig. 8)**：LUFFY vs Vanilla (DeepSpeed expert parallelism)、EXT (Janus expert transfer)、HYT (FasterMoE hybrid)，在 MoE-TransformerXL/MoE-BERT-Large/MoE-GPT2 三种模型上，expert 数量 2/4/8/16 下的 batch training time speedup
  - **Performance Breakdown (Table III)**：Computation time vs Communication time 详细分解
  - **Ablation Study (Fig. 9)**：Token Condensation only vs Sequence Migration only vs Both，分析各组件对不同模型的贡献
  - **Convergence Evaluation (Table IV)**：MoE-TransformerXL on WikiText-103 (PPL)、MoE-BERT-Large on SQuAD (F1)、MoE-GPT2 on SAMSum (ROUGE-1)，对比 static threshold (h=0.3, h=0.8) vs adaptive LUFFY vs Vanilla
  - **Sensitivity Analysis (Fig. 10)**：候选 GPU 数 q、cost model 精度、fast similarity measurement 参数 S₁/S₂ 的敏感性

- 硬件平台是什么，配置是什么。
  16× NVIDIA V100 GPU (16GB HBM)，PCIe 互联。Ubuntu 20.04 (kernel 5.15)，NVIDIA driver 525.85，CUDA 11.7，cuDNN 8.6.0。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **MoE-TransformerXL**：18-block decoder，d_model=1024，d_hidden=4096/4090，参数 0.44B-2.55B
  - **MoE-BERT-Large**：24-block encoder，d_model=768/708，d_hidden=3072，参数 0.54B-3.36B
  - **MoE-GPT2**：12-block decoder，d_model=768，d_hidden=3072，参数 0.18B-0.97B
  - Expert 数量配置：2/4/8/16 per MoE layer，top-2 gating，batch size=64
  
  数据集与 Benchmarks：
  - MoE-TransformerXL: WikiText-103 → Perplexity (PPL↓)
  - MoE-BERT-Large: SQuAD → F1 (F1↑)
  - MoE-GPT2: SAMSum → ROUGE-1 (ROUGE-1↑)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未提供公开开源代码仓库。LUFFY 基于 PyTorch 实现，约 4.5K 行 Python 代码，以 plug-and-play 插件形式提供。未搜索到 GitHub 仓库链接。

  **LUFFY Token Condensation 算法 Pipeline**：

  ```
  # === 符号说明 ===
  # T: tokens, E: experts, d: hidden dimension
  # N_gpu: number of GPUs, B: number of sequences

  # === Token Condensation (Dispatch & Combine Phase) ===
  def token_condensation(tokens, gate_output, block_idx, prev_similarity, loss_prev):
      """
      tokens: [num_tokens, d] - token embeddings after attention
      gate_output: {token_idx -> expert_idx} - gate routing result
      block_idx: current MoE block index
      prev_similarity: historical similarity matrix from block (b-1)
      loss_prev: loss value from previous training iteration
      """
      
      # Step 1: Build token similarity graph
      graph = build_token_graph(tokens, gate_output)
      
      # Step 2: Fast similarity measurement
      for (token_i, token_j) in graph.edges:
          # 2a: Different experts → dissimilar (weight = 0)
          if gate_output[token_i] != gate_output[token_j]:
              graph[token_i][token_j].weight = 0
              continue
          
          # 2b: Historical similarity check
          s_prev = prev_similarity.get((token_i, token_j), None)
          if s_prev is not None:
              if s_prev > S1:  # extremely similar
                  graph[token_i][token_j].weight = 1
                  continue
              if s_prev < S2:  # extremely dissimilar
                  graph[token_i][token_j].weight = 0
                  continue
          
          # 2c: Compute real cosine similarity for uncertain pairs
          sim = cosine_similarity(tokens[token_i], tokens[token_j])
          graph[token_i][token_j].weight = sim
      
      # Step 3: Adaptive condensation threshold
      l_norm = (loss_ini - loss_prev) / loss_ini
      h_t = 1.0 / (1 + exp(l_norm))
      
      # Step 4: Condense similar tokens
      # Remove edges with weight < h_t → sparse graph with subgraphs
      for subgraph in connected_components(graph, threshold=h_t):
          # Keep token with highest degree, condense others
          representative = argmax(degree(subgraph))
          for token in subgraph \ {representative}:
              token_to_token[token] = representative  # mapping table
      
      # Step 5: Dispatch — only send representative tokens
      for expert_idx in unique_experts:
          tokens_to_send = [t for t in tokens 
                           if gate_output[t] == expert_idx 
                           and t in representatives]
          all_to_all_send(tokens_to_send, target_gpu=expert_owner[expert_idx])
      
      # Step 6: Expert computation (fewer tokens → less computation)
      for expert_idx, received_tokens in received_tokens_by_expert:
          expert_output = expert_ffn(received_tokens)  # Fused MoE kernel
      
      # Step 7: Combine — expand condensed tokens using representative output
      for token in all_tokens:
          if token in token_to_token:
              # Use representative's expert output
              token_output[token] = expert_output[token_to_token[token]]
          else:
              token_output[token] = expert_output[token]
      
      return token_output
  ```

  **关键张量计算与通信量变化**：
  
  以 MoE-TransformerXL，4 experts，batch=8 为例：
  - Baseline (Vanilla Expert Parallelism): All-to-All 通信量 = 3.19 GB/batch，通信时间 327ms (18.1%)
  - LUFFY Token Condensation: 凝聚约 62% 相似 token 后，通信量显著减少
  - LUFFY 总体: Communication speedup 1.76×-3.72× vs Vanilla

  **Token Condensation 的核心计算复杂度**：
  - Naive pairwise: O(T²·d) 对所有 token 对 → 不可行
  - Fast measurement: 大部分 token 对通过 expert activation (O(T)) 和历史相似度 (O(1) lookup) 直接判断 → 仅剩余少量不确定对需 O(d) 余弦计算
  - 图凝聚: O(T log T) 通过连通分量分析

## BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **BigMac**，一种通信高效的 fine-grained MoE 模型结构，核心设计包括：

  **DCCA（descend-communicate-communicate-ascend）低维通信策略**：将 fine-grained MoE 原有的 CDAC（communicate-descend-ascend-communicate）方式改为 DCCA——在每个 MoE 层的入口增加 descending projection（$W'_{\downarrow}$）将 token hidden dimension 从 h 压缩至 r·h，再进行 All-to-All 通信分发 token 到各 expert，expert 计算完成后再通过 ascending projection（$W'_{\uparrow}$）恢复到原始维度 h。通信量从 $C = 2 \times top\_k \times \frac{ep-1}{ep} bsh$ 降至 $C' = 2 \times top\_k \times \frac{ep-1}{ep} bsr h$（减少至原来的 r 倍，r 为 downscaling factor，论文设 r=0.25）。

  **BigMac Expert 设计**：为适配 DCCA 策略，重新设计 expert 结构——将 fine-grained MoE 的 expert（$E_i(x) = \sigma(xW_{i,\downarrow})W_{i,\uparrow}$，先降维再升维）改为 BigMac expert（$E_i(x) = \sigma(xW_{i,\uparrow})W_{i,\downarrow}$，先升维再降维）。由于 DCCA 已将输入 token 维度缩减至 r·h，expert 内部先升维可保证总参数量与 fine-grained MoE 对齐，避免模型质量下降。

  实验比较：
  - **Pre-training convergence**：GPT-Vanilla（conventional MoE）、GPT-Fine-Grained（DeepSeekMoE 式 fine-grained）、GPT-BigMac，在 Wikipedia 3.6B tokens 上预训练，比较 validation perplexity 收敛曲线和 wall-clock time
  - **Downstream tasks（同时长训练后）**：BigMac vs Fine-Grained on PTB, WikiText103, WikiText2, LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H（基于 GPT3-XL）
  - **Downstream tasks（同 token 数训练后）**：BigMac vs Fine-Grained vs Vanilla on PTB, WikiText103, WikiText2, LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H（基于 GPT3-Medium）
  - **Long-context evaluation**：GovReport（summarization）、NeedleInAHaystack（retrieval），BigMac vs Fine-Grained
  - **Training latency（Megatron）**：GPT-BigMac vs GPT-Fine-Grained on four base models (GPT3-Medium/XL/2.7B/6.7B)，top-4/top-8 routing，不同 EP/TP 配置下的 step time breakdown
  - **Inference throughput（Megatron）**：GPT-BigMac vs GPT-Fine-Grained on 16/32 GPUs，不同 prompt length (128-1024)，top-4/top-8 routing
  - **Training on Tutel**：with 2DH All-to-All + overlap degree=4，fixed capacity factor f=1.2 vs dynamic capacity factor f=∞
  - **Inference on Tutel & DeepSpeed-Inference**：GPT-BigMac vs GPT-Fine-Grained on Tutel（不同 prompt length）+ DeepSpeed-Inference（不同 generation length 1/2/5/10）

- 硬件平台是什么，配置是什么。
  集群：4 machines connected with 100 Gbps InfiniBand。每 machine 含 8 GPUs，每 GPU 通过 PCIe 4.0 x 16 连接，48 GB HBM，149.7 TFLOPS（FP16），96 cores。训练时并行度：Tensor Parallelism = 4, Expert Parallelism = 4, Data Parallelism = 2（pre-training 阶段）；训练延迟/推理吞吐评估时配置 EP = 1~32, TP = 1~8（ep × tp = 32）。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **GPT3-Medium**（hidden dim = 1,024, 用于下游任务同 token 数对比和 Tutel/DeepSpeed 评估）
  - **GPT3-XL**（hidden dim = 2,048, 用于 pre-training convergence 和下游任务同时长对比）
  - **GPT3-2.7B** 和 **GPT3-6.7B**（用于 Megatron 训练延迟 scaling 实验）
  - MoE 配置：64 experts/层，top-4/top-8 routing，expert capacity factor = 1.2，load balance type = aux_loss（系数 α=0.001），downscaling factor r = 0.25
  
  数据集：
  - Wikipedia dataset（3.6B tokens，用于 pre-training convergence）
  - OpenWebText2 dataset（14.8B tokens，用于 downstream task 评估）
  
  Benchmarks：
  - Perplexity: PTB, WikiText103, WikiText2 — PPL↓
  - Accuracy: LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H — ACC↑
  - Long-context: GovReport（summarization score）, NeedleInAHaystack（recall score across depths 10-90%）
  - Efficiency: Training step latency (ms), All-to-All latency (ms), Inference throughput (tokens/s)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未提供开源代码仓库。BigMac 基于 Megatron-LM、Tutel、DeepSpeed-Inference 等开源框架评估，但模型结构和路由策略的修改代码未公开。

  **BigMac 算法 Pipeline 核心计算流程**：

  ```
  # === 符号说明 ===
  # b: global batch size, s: sequence length, h: hidden dimension
  # e: number of experts, top_k: activated experts per token
  # r: downscaling factor (论文 r=0.25, 如 DeepSeek-V2 从 5120→1536)
  # ep: expert parallelism degree

  # === BigMac MoE Layer Forward Pass (DCCA Strategy) ===
  def bigmac_moe_forward(x):                    # x ∈ R^{batch×seq×h}
      # Step 1: Gating (在降维前的 full dimension 做路由)
      gate_logits = x @ W_gate                  # W_gate ∈ R^{h × e}
      gate_probs = SoftMax(gate_logits)         # [batch, seq, e]
      topk_weights, topk_indices = TopK(gate_probs, k=top_k)

      # Step 2: Descend — 降维投影（DCCA 的第一步 D）
      x_low = x @ W'_down                       # W'_down ∈ R^{h × (r·h)}
                                                 # x_low ∈ R^{batch×seq×(r·h)}
      
      # Step 3: All-to-All Dispatch — 低维通信（DCCA 的 C·C）
      # Token dispatch 到各 expert 所在 GPU
      # 通信量: 2 × top_k × (ep-1)/ep × b × s × (r·h)（比 CDAC 减少 r 倍）
      dispatched_tokens = all_to_all_dispatch(x_low, topk_indices)

      # Step 4: Expert Computation（BigMac Expert, 先升后降）
      for each expert i in assigned_experts:
          # BigMac Expert: E_i(x) = σ(x @ W_{i,up}) @ W_{i,down}
          # W_{i,up}: [(r·h) → h_ff], W_{i,down}: [h_ff → (r·h)]
          h_up = tokens @ W_{i,up}              # 先升维：r·h → h_ff
          h_act = σ(h_up)                        # activation (e.g. GeLU/SwiGLU)
          h_out = h_act @ W_{i,down}            # 再降维：h_ff → r·h
          expert_outputs[i] = topk_weights[i] * h_out

      # Step 5: All-to-All Combine — 低维收集
      combined = all_to_all_combine(expert_outputs)

      # Step 6: Ascend — 升维投影（DCCA 的最后一步 A）
      y = combined @ W'_up                      # W'_up ∈ R^{(r·h) × h}
                                                 # y ∈ R^{batch×seq×h}
      return y

  # === Fine-Grained MoE (CDAC) 对比 ===
  def finegrained_moe_forward(x):
      # Step 1: Gating
      gate_logits = x @ W_gate
      gate_probs = SoftMax(gate_logits)
      topk_weights, topk_indices = TopK(gate_probs, k=top_k)

      # Step 2: All-to-All Dispatch — 高维通信（CDAC 的 C）
      # 通信量: 2 × top_k × (ep-1)/ep × b × s × h（在全维度 h 上进行）
      dispatched_tokens = all_to_all_dispatch(x, topk_indices)

      # Step 3: Descend — Expert 内降维（CDAC 的 D）
      for each expert i:
          h_down = dispatched_tokens @ W_{i,down}  # h → h_ff
          h_act = σ(h_down)
          # Step 4: Ascend — Expert 内升维（CDAC 的 A）
          h_out = h_act @ W_{i,up}                   # h_ff → h
          expert_outputs[i] = topk_weights[i] * h_out

      # Step 5: All-to-All Combine — 高维通信（CDAC 的 C）
      combined = all_to_all_combine(expert_outputs)
      return combined
  ```

  **关键张量计算对比**：
  
  | 指标 | GPT-Fine-Grained | GPT-BigMac |
  |------|-----------------|------------|
  | #Param | $4h^2 + 8h + (2rh^2 + 2rh)e$ | $4h^2 + 8h + (2rh^2 + 2rh)e + 2rlh^2$ |
  | #FLOPs | $12bslh^2(2+s/h+v/2lh+rtop\_k)$ | $12bslh^2(2+s/h+v/2lh+rtop\_k) + 12rbslh^2$ |
  | #A2A | $8bslhtop\_k\frac{ep-1}{ep}$ | $8bslhtop\_k\frac{ep-1}{ep}r$ |

  以 GPT3-XL + 64 experts, top_k=8, r=0.25, ep=32 为例：
  - #Param: Fine-Grained 3.73B → BigMac 3.78B (+1.35%)
  - #FLOPs: Fine-Grained 3,490.67 TFLOPs → BigMac 3,649.00 TFLOPs (+4.54%)
  - #A2A: Fine-Grained 1,488.00 GB → BigMac 372.00 GB (-75.00%)

  **额外优势**：
  - **Dropless Token Routing**：通信量大幅减少后，可移除 expert capacity 限制（不再丢 token），进一步提升模型质量
  - **Flexible top_k**：通信高效的 BigMac 可使用更大的 top_k 值以增强模型性能（如 Top8 BigMac 仍快于 Top4 Fine-Grained）


## Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 MoE 推理中不同专家（expert）进行压缩误差敏感性分析，提出使用 error-bounded lossy compression（如 SZ3、CuSZp）来压缩非激活专家以减少 PCIe offloading 开销。核心实验方法：随机生成服从正态分布 $N \sim (0, \hat{e})$ 的误差注入 expert 参数，模拟 SZ3/CuSZp 压缩引入的有界误差。从七个维度全面分析压缩误差对 MoE 推理精度的影响：
  - ① 单层单个 expert 注入误差（含低激活频率 expert 和最高频 expert）
  - ② 单个层最高频 expert 注入不同大小误差
  - ③ 不同层（L1/L13/L20/L26）最高频 expert 注入误差
  - ④ 单个层 Top-K（6个）最高频 expert 同时注入误差
  - ⑤ 单个层全部 64 个 expert 注入误差
  - ⑥ 一组层（每10层一组）最高频 expert 注入误差
  - ⑦ 不同 benchmark（GSM8K、Math dataset）上的泛化评估

  实验比较：Baseline（无误差）vs 不同 error bound（30%/50%/80% × 平均 L1 范数）下的 ICA（Instruction Compliance Accuracy）和 PIA（Pure Inference Accuracy），分析各层 expert 对误差的敏感性差异。

- 硬件平台是什么，配置是什么。
  论文为分析性研究，使用 error injection 模拟压缩误差，论文未明确说明具体 GPU/CPU 型号等硬件配置。

- 模型是什么。数据集和bench分别是什么。
  模型：**Moonlight**（16B 参数 MoE 模型），26 个 expert layer，每层 64 个 expert 子模块，top-6 routing。每次推理每层激活 6 个 expert。
  数据集和 Benchmarks：
  - **GSM8K**：数学文字推理题，主要 benchmark
  - **Math dataset**（Hendrycks et al. 2021）：更难的数学数据集，用于泛化验证
  评估指标：ICA（Instruction Compliance Accuracy，指令合规精度）和 PIA（Pure Inference Accuracy，纯推理精度，忽略格式要求）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供代码开源链接，SC'25 workshop 论文。以下基于论文描述的方法伪代码：

  ```
  输入: MoE模型权重 θ = {θ_{l,e} | l=1..L, e=1..E}, 数据集 D, error_bound_ratio r (e.g., 30%/50%/80%)
  输出: ICA, PIA for each perturbation scenario

  For each expert (l, e) in target set:
      # 计算该 expert 的平均 L1 范数作为误差基准
      avg_L1 = ||θ_{l,e}||_1 / n_{l,e}

      # 设置 error bound
      ê = r * avg_L1

      # 生成服从 N(0, ê) 的正态分布误差
      noise = Normal(mean=0, std=ê).sample(n_{l,e})

      # 将误差注入 expert 参数（模拟有损压缩-解压后的参数）
      θ'_{l,e} = θ_{l,e} + noise

  # 在注入误差的模型上推理
  For each sample x in D:
      For each token t in x:
          For each MoE layer l:
              # Router 选择 top-k experts
              weights, expert_ids = Router(token_embedding, k=6)
              # 计算加权输出（使用已注入误差的 expert 参数）
              output = Σ w_i * Expert_i(token_embedding; θ'_{l,expert_i})
          # 生成下一个 token
      # 计算 ICA: 检查输出格式（如 boxed{}）和内容正确性
      # 计算 PIA: 仅检查内容正确性
  ```

  ## Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 MoE 模型的统一压缩框架，包含两大视角：(1) **Expert Trimming**：通过结构化移除专家/层/块来压缩 MoE——Expert Drop（按重要性评分移除不重要专家，含 layer-wise 和 global 两种策略）、Layer Drop（基于 cosine similarity 移除整个 MoE 层和对应的 Norm 模块）、Block Drop（进一步移除包括 Attention 在内的整个 Transformer block）；(2) **Expert Slimming**：压缩单个专家内部权重——Pruning（Wanda / SparseGPT, unstructured 50% 和 2:4 semi-structured sparsity）和 Quantization（GPTQ / AWQ, 4-bit）；(3) **集成策略**：先 Expert Slimming 后 Expert Trimming（"S+T" order），结合 AWQ 量化 + Block Drop 在 Mixtral-8×7B 上实现 6.05× speedup + 77.1% 内存节省（20GB），维持 >92% 性能；(4) **Post-Finetuning**：在 Alpaca-GPT4 数据集上 full-finetune 3 epochs (lr=8e-6, cosine schedule, warmup ratio=0.03, global batch=32)，Block Drop 后模型性能差距从 5.5% 缩小至 0.6%。实验比较：(a) Expert Drop vs Layer Drop vs Block Drop 在不同压缩率下的 benchmark 性能和 speedup/memory；(b) Pruning (Wanda/SparseGPT, 50%/2:4) vs Quantization (GPTQ/AWQ, 4-bit) 的性能与效率对比；(c) Expert Trimming + Expert Slimming 集成在不同组合下的综合对比；(d) 压缩后 Post-Finetuning 的性能恢复能力；(e) MoE vs Dense 模型的冗余度对比（同深度 Mixtral-8×7B vs Mistral-7B 在 Layer/Block Drop 下的性能衰减差异）。

- 硬件平台是什么，配置是什么。
  NVIDIA GPU（论文提及部署目标为 NVIDIA RTX 3090 GPU）。AWQ 量化 speedup 在 5.08× (Mixtral-8×7B) 和 3.16× (DeepSeek-MoE-16B)。FLOPs、Memory 和 Speedup 通过 forward pass on input sequence of length 2,048 测量。论文未明确说明 eval 使用的 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  模型：**Mixtral-8×7B**（32 层，8 experts/layer, top-2 routing, 47B total/13B activated, 87.7GB FP16 memory）和 **DeepSeek-MoE-16B**（28 层 / 27 MoE layers, 2 shared experts + 64 routed experts, top-6 routing, 30.8GB FP16 memory, 首个 block 使用 dense FFN）。压缩校准数据：C4 数据集 128 samples × seq_len=2048（用于相似度计算和 pruning 校准）；量化校准：128 samples from Alpaca (GPTQ) 和 Pile (AWQ)；GPTQ group_size=128 (Mixtral) / 64 (DeepSeek)；Post-finetuning：Alpaca-GPT4 3 epochs。Benchmarks（EleutherAI LM Harness）：ARC-C, BoolQ, HellaSwag, MMLU, OBQA, PIQA, RTE, WinoGrande（全部 zero-shot normalized accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，代码发布于 https://github.com/CASE-Lab-UMD/Unified-MoE-Compression。以下为核心算法流程：

  **Layer Drop 和 Block Drop 的相似度度量（用于选择要移除的层/块）**：
  ```
  # S^{(M)}: MoE input-output cosine similarity
  # S^{(NM)}: Norm+MoE+residual 的整体相似度（论文采用此度量）
  # x: MoE input, y = MoE(x): MoE output
  # x': 残差连接前的输入 (block input)
  
  S^{(M)} = cos_sim(x, y)  # = (x·y) / (||x|| * ||y||)
  
  y' = x' + MoE(Norm(x'))  # 含 Norm + MoE + residual
  S^{(NM)} = cos_sim(x', y')
  
  # 对每个 layer/block，用 128 个 C4 样本计算平均 S^{(NM)}
  # 按 S^{(NM)} 从高到低排序（相似度越高 → 冗余越大）
  # 依次移除高冗余度的层/块
  ```

  **Expert Drop 重要性评分与移除**：
  ```
  # 专家重要性评分 S(E_i) = 批数据上的平均路由分
  S(E_i) = (1/|X|) * Σ_{x∈X} G_i(x)  # G_i(x) 为 router 对 expert i 的输出
  
  # Layer-wise dropping: 每层保留相同数量专家
  T'(l) = {E_t^(l)} where S(E_t^(l)) ∈ TopK({S(E_i^(l))}_{i=1..n}, n')
  
  # Global dropping: 全局跨层保留 Top experts
  T'(l) = {E_t^(l)} where S(E_t^(l)) ∈ TopK(∪_{j=1..L}{S(E_i^(j))}_{i=1..n}, n'*L)
  ```

  **统一压缩框架（Expert Trimming + Expert Slimming）**：
  ```
  # 通用形式: y = Σ_{i∈T'} G_i · E_i(x | f(W_i))
  # T': 保留的专家子集 (Expert Trimming)
  # f(W_i): 压缩后的专家权重 (Expert Slimming)
  
  # Expert Slimming → Expert Trimming 顺序 ("S+T"):
  # Step 1: 对所有 expert 应用 AWQ 4-bit 量化
  for each expert i in all experts:
      W_i_quant = AWQ_quantize(W_i, bits=4, calib=Pile_128samples)
  
  # Step 2: 基于量化后模型计算相似度，执行 Layer/Block Drop
  for each layer/block l:
      S_l = mean(cos_sim(x', x' + MoE_quant(Norm(x'))))  # 量化后计算
  sort layers by S_l descending
  remove top K layers/blocks  # Layer Drop / Block Drop
  
  # Step 3 (可选): Post-finetuning on Alpaca-GPT4
  for epoch in 1..3:
      for batch in Alpaca-GPT4 (global_bsz=32):
          loss = CrossEntropy(model_compressed(x), y)
          loss.backward()
          optimizer.step()  # Adam, lr=8e-6, cosine schedule
  ```

  **关键实验结果（Mixtral-8×7B, 综合最佳配置）**：
  ```
  # AWQ only:            5.08× speedup, 24.4GB, Avg=70.8 (vs 71.5 baseline)
  # AWQ + L8/32:         6.05× speedup, 20.0GB, Avg=66.1 (92.4% of baseline)
  # AWQ + B5/32:         5.94× speedup, 21.9GB, Avg=68.0 (95.1% of baseline)
  ```

## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **Context-Aware Mixed-Precision Quantization for NDP**，包含两个核心算法：
  
  **Expert Bitwidth Selector**：基于 prefill 阶段收集的 expert 激活频率 $P_{l,e}$ 和路由评分 $W_{l,e}$ 计算重要性分数 $S_{l,e} = \alpha \widetilde{P}_{l,e} + (1-\alpha) \widetilde{W}_{l,e}$。对 NDP-resident experts，离线预计算 1/2/3/4-bit GPTQ 量化版本和 per-bitwidth 量化损失表 $L_{l,e}(b)$（MSE vs FP16 reference）。运行时使用 prefix-structured mixed-precision allocation：按重要性降序排列 experts，枚举 $(n_4, n_3, n_2, n_1)$ 分配方案满足平均 bitwidth budget $\bar{b}$，最大化累积增益 $G(n_4,n_3,n_2) = C_4(n_4) + [C_3(n_4+n_3)-C_3(n_4)] + [C_2(n_4+n_3+n_2)-C_2(n_4+n_3)]$。时间复杂度 $O(LE_{\text{NDP}}^2)$。
  
  **Expert Placement Module**：同样基于 $S_{l,e}$ 选择 top-K experts 以 FP16 驻留 GPU HBM，其余 experts 分配至 CXL-NDP 执行。Placement 仅在 prefill 后执行一次，decoding 阶段不变。

  实验比较：
  - **Accuracy**：Ours-3bit/Ours-2bit vs Original (MoNDE, FP16 lossless) vs w/o Expert Bitwidth Selector variant，在 MMLU/MathQA/HellaSwag/ARC-E/ARC-C/BoolQ/WinoGrande/PIQA 八个 benchmark 上。Ours-3bit 仅 0.13% 平均精度下降，Ours-2bit 仅 3.4% 下降。
  - **Ablation**：w/ vs w/o Expert Bitwidth Selector → Ours-2bit 带 selector 比不带 selector 高 3.2% 平均精度，验证 context-aware bitwidth 选择的有效性。
  - **Performance**：Ours-3bit vs Ours-2bit vs MoNDE (same GPU-NDP) vs Hobbit (GPU-only mixed-precision offloading)，end-to-end latency 和 decoding throughput。

- 硬件平台是什么，配置是什么。
  系统：1× NVIDIA H100 GPU (132 SM, 989.4 TFLOP/s, 80GB HBM3) + 1× DDR-based NDP device (512 GB, 512 GB/s bandwidth, 64×(4×4) systolic arrays, 1 GHz clock)。PCIe Gen4 ×16 互联。NDP 模拟器：基于 Ramulator [19] 构建。

- 模型是什么。数据集和bench分别是什么。
  模型：**Mixtral-8×7B**（32 layers, 8 experts/layer, top-2, 46.7B params）和 **Mixtral-8×22B**（56 layers, 8 experts/layer, top-2, 140.6B params）。GPU-side：Mixtral-8×7B 每层 4 experts GPU + 4 NDP；Mixtral-8×22B 每层 2 experts GPU + 6 NDP。
  数据集：C4 (1024 samples for calibration), WikiText-2, TruthfulQA (activation analysis)。
  Benchmarks：MMLU (5-shot), MathQA, HellaSwag, ARC-Easy, ARC-Challenge, BoolQ, WinoGrande, PIQA (zero-shot)，使用 EleutherAI LM Evaluation Harness [10] 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确说明代码开源。基础量化方法使用 GPTQ [9]，NDP 模拟器基于 Ramulator [19] (https://github.com/CMU-SAFARI/ramulator)。

  **Context-Aware Expert Bitwidth Selector 算法 Pipeline**：

  ```
  # === 离线阶段 (Offline Calibration) ===
  # 对每层 l 和每个 expert e，预计算 1/2/3/4-bit 量化损失
  for l in 1..L:
      for e in 1..E:
          for b in {1, 2, 3, 4}:
              W_q = GPTQ_quantize(W_{l,e}, bits=b, calib_data=D_cal)
              L_{l,e}(b) = MSE(W_q(x_calib), W_fp16(x_calib))
  
  # === 在线阶段 (Online Inference) ===
  # Step 1: Prefill 统计收集
  def prefill_with_stats(x_seq):
      for each MoE layer l:
          for each expert e:
              P_{l,e} = count(tokens routed to expert e)  # 激活频率
              W_{l,e} = sum(routing_scores for expert e)   # 累计路由分
      return {P, W}
  
  # Step 2: 重要性评分计算
  def compute_importance(P, W, alpha=0.5):
      for l in 1..L:
          P_tilde = P_{l,:} / sum(P_{l,:})     # 归一化
          W_tilde = W_{l,:} / sum(W_{l,:})
          S_{l,e} = alpha * P_tilde[e] + (1-alpha) * W_tilde[e]
      return S  # [L, E] importance scores
  
  # Step 3: Expert Placement
  def place_experts(S, K):
      for l in 1..L:
          sorted_experts = argsort(S[l,:], descending=True)
          H_l = sorted_experts[:K]    # GPU: FP16, hot experts
          C_l = sorted_experts[K:]    # NDP: quantized, cold experts
      return H, C
  
  # Step 4: Prefix-Structured Bitwidth Allocation
  def prefix_split_bitwidth(S_ndp, loss_table, b_bar):
      for l in 1..L:
          E_ndp = len(C_l)
          R = E_ndp * (b_bar - 1)  # bitwidth increment budget
          
          # 按重要性降序排列 NDP experts
          idx = argsort(S_ndp[l,:], descending=True)  # i=1..E_ndp
          
          # 预计算 prefix sums
          delta_2[i] = L_i(1) - L_i(2)
          delta_3[i] = L_i(1) - L_i(3)  
          delta_4[i] = L_i(1) - L_i(4)
          
          C_2(k) = sum_{i=1..k} delta_2[i]  # prefix sum
          C_3(k) = sum_{i=1..k} delta_3[i]
          C_4(k) = sum_{i=1..k} delta_4[i]
          
          # 枚举最优 (n4, n3, n2) 满足预算约束
          best_gain = -inf
          for n4 in 0..E_ndp where 3*n4 <= R:
              for n3 in 0..(E_ndp-n4) where 3*n4+2*n3 <= R:
                  n2 = R - 3*n4 - 2*n3
                  if n2 < 0 or n4+n3+n2 > E_ndp: continue
                  n1 = E_ndp - n4 - n3 - n2
                  
                  # 前缀结构: 最重要的 n4→4bit, 其次 n3→3bit, n2→2bit, n1→1bit
                  gain = C_4(n4) + (C_3(n4+n3)-C_3(n4)) + (C_2(n4+n3+n2)-C_2(n4+n3))
                  if gain > best_gain:
                      best_gain = gain
                      best_assignment = (n4, n3, n2, n1)
          
          # 分配 bitwidth: top-n4→4bit, next-n3→3bit, next-n2→2bit, rest→1bit
          b_{l, idx[0:n4]} = 4
          b_{l, idx[n4:n4+n3]} = 3
          b_{l, idx[n4+n3:n4+n3+n2]} = 2
          b_{l, idx[n4+n3+n2:]} = 1
      return b  # per-expert bitwidth for decoding
  
  # Step 5: Decoding with fixed placement + quantization
  def decoding_step(token, H, C, b):
      for l in 1..L:
          experts = router(token)
          for e in experts:
              if e in H_l:
                  out += expert_fp16_gpu(e, token)    # GPU FP16
              else:
                  out += expert_quant_ndp(e, token, b_{l,e})  # NDP with assigned bits
      return out
  ```

  **关键张量计算流程**（以 Mixtral-8×7B, K=4 GPU experts/layer, b_bar=3 为例）：
  - Prefill: [batch, seq_len, 4096] tokens → 32 MoE layers → 每层收集 8 experts 的 (P_{l,e}, W_{l,e}) → 计算 S_{l,1..8}
  - Placement: Top-4 by S → GPU FP16; Bottom-4 → NDP
  - Bitwidth: 4 NDP experts, b_bar=3 → R=4×(3-1)=8 increments → enumerate (n4,n3,n2) → e.g. (2,2,0,0): 最重要2个4bit, 其次2个3bit
  - Decoding: 每个 token 激活 top-2 experts → 若两者均在 GPU → 全 FP16；若一 GPU 一 NDP → GPU FP16 + NDP 3/4-bit

Error-bounded lossy compression pipeline（SZ3/CuSZp风格）：
  数据 → 预测（Lorenzo predictor/linear regression）→ 量化（基于 error bound ê 控制量化步长）→ 编码（Huffman/变长编码）→ 压缩数据。解压时逆过程：解码 → 反量化 → 反预测。关键特性：所有重建值与原值的绝对误差 ≤ ê（有界保证），压缩比由 ê 和数据分布决定。在 MoE offloading 场景中，expert 参数在传输前压缩（减少 PCIe 数据量），GPU 端解压后用于推理，参数中含 bounded error。本论文的 error injection 实验模拟了这一流程中解压后的参数状态。

## Continual Pre-training of MoEs: How robust is your router?

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 **MoE 持续预训练（Continual Pre-training of MoEs）** 的系统性实证研究。研究了两种主流路由算法（Penalty-Balanced Top-k / Sinkhorn-Balanced Top-k）和两种 MoE 架构（Switch MoE / Granular MoE）在经历从英文到代码/德语的分布偏移时的持续预训练行为。核心技术包括：Infinite LR Schedule（CosineInf）、Replay（回放旧数据）、LR Re-warming + Re-decaying。提出新指标 **MRI (Maximum Routing Imbalance)** 衡量最坏情况延迟下的路由不平衡。
  
  实验比较：
  - **CPT MoE vs FLOP-matched Dense Baseline**：验证 MoE 在持续预训练中是否保持样本效率优势
  - **CPT MoE vs Full Re-training MoE**：验证 CPT 是否能以更低成本匹配完全重训练的性能
  - **PBTk vs SBTk 路由算法**：比较两种路由在分布偏移下的鲁棒性（性能、MRI、路由行为变化）
  - **Switch MoE vs Granular MoE 架构**：比较两种架构的 CPT 表现
  - **Replay 百分比消融（0%, 10%, 30%, 40%）**：分析 replay 对遗忘和适应的 trade-off
  - **Decayed vs Non-decayed checkpoint CPT**：比较从衰减后 checkpoint 和不衰减 checkpoint 开始 CPT
  - **路由行为变化分析**：Router Saturation、Vocabulary Specialization、Expert Co-activation 三指标分析分布式偏移前后的路由决策变化

- 硬件平台是什么，配置是什么。
  64× NVIDIA A100 GPU，使用数据并行（Data Parallelism）和 ZeRO-1（Rajbhandari et al., 2020）。为加速 dropless MoE 前向传播使用了 Megablocks kernel（Gale et al., 2023）。代码基于 GPT-NeoX 库（Andonian et al., 2023）实现。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型（共 5 个，均为 decoder-only，Llama3 架构骨架，GeLU 激活，Llama3 tokenizer，序列长度 2048）：
  - **Dense Baseline**：24 层，570M 参数，hidden size=1024，FFN intermediate=2816，GEGLU FFN
  - **PB Switch MoE**：8 个 routed experts，K=1 active，无 shared expert，full-sized FFN (2816)，Penalty-Balanced（Z-loss coeff=0.001 + Aux-loss coeff=0.01）路由，~2B total / 570M active
  - **SB Switch MoE**：同上但使用 Sinkhorn-Balanced 路由（tolerance=0.01），~2B total / 570M active
  - **PB Granular MoE**：31 个 routed experts，K=3 active，1 个 shared expert，fine-grained FFN (intermediate=704，dense 的 1/4)，Penalty-Balanced 路由，~2B total / 570M active
  - **SB Granular MoE**：同上但使用 Sinkhorn-Balanced 路由，~2B total / 570M active
  所有模型使用 AdamW optimizer（β1=0.9, β2=0.95），weight decay=0.1，gradient clipping=1.0，batch size=1024，Rotary positional embedding（PCT=0.25），vocab size=128000。

  数据集：
  - **Pre-training (Task 1)**：FineWeb（英文 Web），400B tokens，采样自 2916.65B 子集
  - **CPT (Task 2)**：Stack（Code，251.819B 子集）200B tokens，German Common Crawl（169.291B）200B tokens

  Benchmarks：
  - **英文（0-shot）**：HellaSwag, Winogrande, PIQA, ARC-Easy, ARC-Challenge, SWAG, LAMBADA, SciQ, PubMedQA, MathQA
  - **德文（0-shot，GPT-3.5 翻译版）**：HellaSwag-DE, ARC-Challenge-DE, TruthfulQA-DE
  - **代码**：HumanEval（pass@1/10/50/100/150/200）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供独立开源代码仓库。训练基于开源库 GPT-NeoX (https://github.com/eleutherai/gpt-neox) 和 Megablocks grouped GEMM kernel (https://github.com/tgale96/grouped_gemm)。

  **MoE CPT 算法 Pipeline**：

  ```
  # === MoE Layer 前向传播（每层） ===
  # x: [S, H] token hidden states, S=seq_len, H=hidden_dim
  # W_r: [H, E] router weight, E=num_experts
  
  def moe_layer_forward(x, experts, router, shared_expert=None, routing="PBTk"):
      logits = x @ W_r  # [S, E]
      if routing == "SBTk":
          probs = sinkhorn_balance(logits)  # Sinkhorn-Knopp 迭代
      else:  # PBTk
          probs = softmax(logits)
      
      # Top-k 专家选择
      topk_vals, topk_idx = topk(probs, k)  # [S, k]
      
      # MoE 输出计算
      moe_out = zeros_like(x)
      for each token s:
          norm = sum(probs[s, topk_idx[s]])
          for i in topk_idx[s]:
              moe_out[s] += probs[s,i] * experts[i](x[s]) / norm
      
      # Shared Expert (Granular MoE only)
      if shared_expert is not None:
          moe_out += shared_expert(x)
      
      return moe_out
  ```

  ```
  # === CPT Training Loop ===
  # Phase 1: Pre-training on FineWeb (400B tokens)
  for step in range(192720):
      batch = sample(FineWeb, batch_size=1024)
      # Loss = LM loss + α * Aux Loss + β * Z-Loss (仅 PBTk)
      loss = lm_loss + 0.01 * aux_loss + 0.001 * z_loss  
      optimizer.step()  # CosineInf schedule, lr_const=1.65e-4
  
  # Phase 2: CPT (200B tokens, 30-40% replay)
  for step in range(95370):
      batch_fw = sample(FineWeb, batch_size=1024 * replay_pct)
      batch_new = sample(target, batch_size=1024 * (1-replay_pct))
      batch = concat(batch_fw, batch_new)
      loss = model(batch)
      optimizer.step()  # CosineInf: 从 const LR 继续，无 cooldown
      
      # MRI 监控
      for layer in moe_layers:
          mri = max(token_load_per_expert / total_tokens)  # Eq. (1)
  ```

  **MRI 定义**（Eq. 1）：
  $$MRI(t,j) := \max_{i \in [1,\dots,E]} \left[ \frac{\sum_{x \in B} \mathbb{1}\{i \in I_k(x)\}}{|B|} \right]$$
  其中 $B$ 为一个 batch 中的所有 tokens，$I_k(x)$ 为 token $x$ 的 top-k 专家索引集合。MRI 越大表示最繁忙 expert 承载越多 tokens → 最坏情况延迟越高。

  **核心 CPT 策略**：
  - **CosineInf Schedule**：预训练阶段用 CosineInf（constant 80% + cooldown 70%），CPT 阶段从 $\eta_{const}=1.65\times10^{-4}$ 继续，cooldown=0%（LR 始终保持 constant），$\eta_{max}=3\times10^{-4}$，$\eta_{min}=3\times10^{-5}$
  - **Replay**：每 batch 中 X% 的样本来自旧分布，(100-X)% 来自新分布。Compute Equivalent Replay：增加 replay 不增总 token budget 而减少新数据量
  - **PBTk Routing**：$L_{total} = L_{LM} + \alpha \cdot L_{aux} + \beta \cdot L_z$，$\alpha=0.01, \beta=0.001$
  - **SBTk Routing**：softmax 前应用 Sinkhorn-Knopp 迭代近似求解线性分配问题，推理时去掉 balancing step（不兼容自回归生成）

## DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三部分核心创新：**(1) Dynamic Tiling Vision Encoding**：将高分辨率图像按候选分辨率集 C={(m·384, n·384) | 1≤m,n≤9} 动态切分为 m×n 个 384×384 local tiles + 1 个 global thumbnail tile，通过 SigLIP-SO400M-384 共享视觉编码器处理所有 tile，经 2×2 pixel shuffle 压缩（27×27→14×14=196 tokens/tile），再通过 special tokens (<tile_newline>, <view_separator>) 组织 visual sequence 送入 LLM。**(2) DeepSeekMoE LLM with Multi-head Latent Attention (MLA)**：MLA 将 KV cache 压缩为低秩 latent vector，大幅减少推理时 KV cache 内存占用，提升吞吐；MoE 使用 shared experts + routed experts 架构，Tiny/Small 使用 Softmax routing (64 experts, Top-6)，DeepSeek-VL2 使用 Sigmoid routing + expert correction bias (72 experts, Top-6)，实现稀疏激活的高效推理。**(3) 三阶段训练 + 精细化数据管线**：Stage 1 VL Alignment（冻结 LLM，训练 vision encoder + MLP，ShareGPT4V 1.2M），Stage 2 VL Pretraining（全参数训练，~800B tokens，70% VL + 30% text-only），Stage 3 SFT（全参数 fine-tuning，~20B tokens）。实验比较：(a) DeepSeek-VL2-Tiny/Small/DeepSeek-VL2 vs 同参数/同激活参数量的开源密集和 MoE 模型（LLaVA-OV, InternVL2, Qwen2-VL, Molmo, MM1.5, Aria-MoE, Phi-3.5, Pixtral）；(b) vs 闭源模型 GPT-4V/GPT-4o/Claude 3.5 Sonnet/Gemini-1.5-Pro。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100 GPU 集群。DeepSeek-VL2-Tiny: 16 节点 × 8 A100（128 GPUs），训练 7 天。DeepSeek-VL2-Small: 33 节点 × 8 A100（264 GPUs），训练 10 天。DeepSeek-VL2: 42 节点 × 8 A100（336 GPUs），训练 14 天。训练使用 HAI-LLM 框架，pipeline parallelism + tensor parallelism + expert parallelism。推理部署：Tiny 模型可部署在 10GB 单 GPU，Small 模型 40GB 单 GPU，DeepSeek-VL2 模型 80GB 单 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-style 架构，Vision Encoder (SigLIP-SO400M-384, ~0.4B) + VL Adaptor (2×2 pixel shuffle + 2-layer MLP) + DeepSeekMoE LLM。三个变体：**(1) Tiny**：LLM 3B total/0.57B activated, d=1280, 10 heads, 12 layers, MHA, 64 routed+2 shared experts, Top-6 Softmax。**(2) Small**：LLM 16B total/2.4B activated, d=2048, 16 heads, 27 layers, MLA(rank=512), 64 routed+2 shared, Top-6 Softmax。**(3) DeepSeek-VL2**：LLM 27B total/4.1B activated, d=2560, 32 heads, 30 layers, MLA(rank=512), 72 routed+2 shared, Top-6 Sigmoid + expert bias correction。数据集：**Alignment**: ShareGPT4V 1.2M；**Pretraining**: 交错图文(WIT, WikiHow, OBELICS, Wanjuan)、重新标注图像描述(内部 captioner+DeepSeek Chat 质量评分)、OCR(LaTeX OCR, RenderedText)、VQA、视觉定位(Objects365, KOSMOS-2)、Grounded conversation 等，~800B tokens；**SFT**: 通用 VQA, OCR/文档, 表格/图表, 推理/数学, 视觉定位, Grounded conversation, 纯文本, ~20B tokens。Benchmarks: DocVQA, ChartQA, InfoVQA, TextVQA, OCRBench, AI2D, MMMU(Val), MMStar, MathVista(TestMini), MME, MMBench, MMBench-V1.1, MMT-Bench, RealWorldQA, RefCOCO/RefCOCO+/RefCOCOg。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  MIT License 开源，代码和预训练模型发布于 https://github.com/deepseek-ai/DeepSeek-VL2。

  **Dynamic Tiling 算法流程（单张高分辨率图像推理）**：
  ```
  Input: image I of size (H, W)
  Params: base_res=384, max_grid=9

  // Step 1: Select best resolution minimizing padding
  C = {(m*384, n*384) | 1<=m,n<=9}
  For (h_c,w_c) in C:
      scale = min(h_c/H, w_c/W)
      pad = h_c*w_c - (H*scale)*(W*scale)
  Select (m*,n*) = argmin pad

  // Step 2: Dynamic Tiling
  Resize I to (m*·384, n*·384), pad to maintain aspect ratio
  Split -> m*×n* local tiles (384×384) + 1 global thumbnail (384×384)

  // Step 3: Vision Encoding
  For each tile:
      v = SigLIP-SO400M-384(tile)  // 27×27×1152
      v = PixelShuffle(v)          // 2×2 -> 14×14=196 tokens, dim=4608

  // Step 4: Visual sequence construction
  Global: 14×(14+<tile_newline>) = 210 tokens
  Local grid: (m*·14)×(n*·14) + n*·14 <tile_newline>
  Full: [210 global] + <view_separator> + [local grid]
  Total visual tokens: 210+1+n*·14·(m*·14+1)

  // Step 5: VL Adaptor projection
  v_proj = 2-layer MLP(v_token)  // 4608->d_LLM

  // Step 6: DeepSeekMoE LLM with MLA (for Small/VL2)
  For each layer l:
      // MLA: KV compression into latent
      c_KV = W_DKV · h_t              // -> rank 512
      k_C = W_UK · c_KV               // compressed key
      v_C = W_UV · c_KV               // compressed value
      k_R = RoPE(W_KR · h_t)          // decoupled RoPE
      q = W_Q · h_t,  q_R = RoPE(q)
      o = MHA([q;q_R], [k_C;k_R], v_C)

      // MoE FFN: sparse activation
      s = Sigmoid(W_gate·h_t)         // or Softmax for Tiny/Small
      s += bias (DeepSeek-VL2 only)
      TopK = TopK(s, K=6)
      FFN_out = Σ g_i · FFN_i(h_t)    // 2 shared + 6 routed experts
  ```

  **三阶段训练**：
  - Stage 1 (VL Alignment): frozen LLM, train vision encoder+MLP, ~2B tokens, batch=256, seq=4096, Cosine LR=~4.5e-4
  - Stage 2 (VL Pretraining): unfreeze all, ~800B tokens, batch=2304~3360, seq=4096, Step LR (÷√10 at 50%/75%), pipeline+ tensor+ expert parallelism
  - Stage 3 (SFT): unfreeze all, ~20B tokens, batch=64, seq=4096, Constant LR=1.4e-5~3e-5
  - 所有阶段：AdamW (β1=0.9, β2=0.95), weight_decay=0.1, grad_clip=1.0, aux_loss_weight=0.001~0.0001

## Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DefaultMoE**：一种轻量级 MoE 训练方法，在保持前向稀疏激活（仅 Top-K experts 计算前向）的同时，通过为未被激活的 expert 提供 **default vector**（EMA 历史输出的指数移动平均），使得 Router 在反向传播时收到来自所有 N 个 experts 的 **dense gradient**，而非仅有被选中的 K 个 experts 的梯度。具体机制包括：(1) 为每个 expert i 维护 EMA buffer Ê_i = β·Ê_i^{(t-1)} + (1-β)·E_i(x)，记录 expert 历史输出的期望值；(2) 前向传播时，对非激活 expert 用 Ê_i 替代真实 E_i(x)，计算 y = Σ π_i · (E_i(x) if i∈TopK else Ê_i)；(3) 反向传播时，路由器收到所有 N 个 experts 的梯度信号 ∂y/∂π = [Ê_1, ..., E_i(x) for i∈TopK, ..., Ê_N]^T。实验比较：DefaultMoE vs Standard TopK MoE、SparseMixer、ReMoE、Loss-Free Balancing，在 1.96B 总参数 MoE（8c1/8c2/32c1/32c2/32c4 多种配置）上训练 160B tokens，对比 pretraining PPL、收敛速度（token-to-target-PPL）、下游 12 个 benchmark（LogiQA, MathQA, MMLU, OpenBookQA, Lambada, SocialIQA, HellaSwag, ARC, Winogrande, PubMedQA, BoolQ, PIQA, SciQ）、以及 Router gradient 与 dense gradient 的相似度。

- 硬件平台是什么，配置是什么。
  AWS 集群 64 GPUs（论文具体 GPU 型号未明确说明，训练代码和配置推断为 A100/H100 级别 GPU）。单 GPU throughput 测试使用 1024 和 2048 hidden dim 的模型分别在 1 GPU 上测试。7.33B 参数模型 per-node throughput 约 1393 tokens/sec（TopK）vs 1391 tokens/sec（DefaultMoE）。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Llama 架构的 MoE Transformer，使用 SwiGLU FFN、16 attention heads (dim 64)、LayerNorm、RoPE、DeepNet 初始化。总参数量 1.96B（其中 366M 非 MoE 参数，1.6B MoE 参数），active params 根据配置变化（8c1: 565M, 8c2: 764M, 32c1: 416M, 32c2: 466M, 32c4: 565M）。也测试了 hidden dim=512 (557M) 到 hidden dim=2048 (7.33B) 的模型。使用 Llama3 tokenizer。数据集：FineWeb-Edu 和 FineWeb。训练 160B tokens（≈283 tokens/param 的 overtraining 比例）。Benchmarks：LogiQA, MathQA, MMLU, OpenBookQA, LAMBADA, SocialIQA, HellaSwag, ARC (Easy+Challenge), Winogrande, PubMedQA, BoolQ, PIQA, SciQ，使用 lm-eval-harness 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码已开源：https://anonymous.4open.science/r/default-moe-6C74/，配置文件：https://anonymous.4open.science/r/default-moe-6C74/configs/default-moe-2B.yml。训练基于 gpt-neox + Megablocks + liger kernel (Triton)。核心算法流程：

  伪代码（DefaultMoE 单层前向+反向）：
  ```
  # 前向传播
  Input: token x, router params W (d_token -> N), experts E_0..E_{N-1}
  1. logits = W @ x                          # [N], router logits
  2. pi = Softmax(logits)                    # [N], expert weights
  3. A = TopK(pi, K)                         # indices of K selected experts
  4. for i in A:
       y_i = E_i(x)                          # compute only K expert outputs
  5. for i not in A:
       y_i = EMA_buffer[i]                   # use stored EMA default vector
  6. y = sum(pi_i * y_i for i in 0..N-1)    # weighted sum of all N outputs
  7. for i in A:
       EMA_buffer[i] = beta * EMA_buffer[i] + (1-beta) * mean(E_i(x) over batch)
     # EMA update only for activated experts in this batch

  # 反向传播
  8. dL/d(pi_i) = dL/dy * y_i               # dense gradient: ALL experts contribute
     - Activated experts: y_i = E_i(x)      [true output]
     - Non-activated experts: y_i = EMA_buffer[i]  [default vector]
  9. dL/dW = dL/d(pi) @ d(pi)/dW             # router gets gradient from all N experts
  ```

  张量计算示意（batch_size=B, N=8 experts, K=1 active, hidden=H）：
  ```
  x: [B, H]
  W: [N, H]
  pi: [B, N] = Softmax(x @ W^T)
  TopK mask: [B, N] with exactly K ones per row
  
  # Standard TopK MoE forward
  y_topk: [B, H] = sum_i( mask[b,i] * pi[b,i] * E_i(x[b]) )
  
  # Default MoE forward (dense sum with EMA substitution)
  y_default: [B, H] = sum_i( pi[b,i] * (mask[b,i]*E_i(x[b]) + (1-mask[b,i])*EMA[i]) )
  # Note: EMA[i] is [H], broadcast across batch
  
  # EMA update (only for activated experts)
  for i where mask[:,i] has any True:
    activated_outputs = E_i(x[mask[:,i]])  # [num_activated_i, H]
    EMA[i] = beta * EMA[i] + (1-beta) * mean(activated_outputs, dim=0)
  
  # Router gradient comparison
  # TopK gradient:
  dL/dW[i,:] = dL/dy * (1/B) * sum_b( mask[b,i] * pi[b,i] * x[b] * E_i(x[b]) )
  # DefaultMoE gradient (dense):
  dL/dW[i,:] = dL/dy * (1/B) * sum_b( pi[b,i] * x[b] * (mask[b,i]*E_i(x[b]) + (1-mask[b,i])*EMA[i]) )
  ```
  
  关键超参数：β=0.9（8c1/8c2）、β=0.65（32c1）、β=0.95（32c2）、β=0.999（32c4）。使用 weighted update（按 router probability 加权更新 EMA）后 β 不再敏感。EMA 初始化为零。

## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 DS-MoE（Dense Training, Sparse Inference）框架，属于 MoE 模型训练范式的创新——训练阶段所有 expert 全部参与前向/反向计算（dense training），推理阶段仅激活 top-K 个 expert（sparse inference）。实验比较：(1) 与同参数量 Dense 模型比性能和计算效率；(2) 与同性能 Sparse MoE (SMoE, 传统 sparse training) 比参数效率；(3) 在 vLLM 上与 Mistral-7B、DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B 比吞吐量；(4) ablate MI loss weight α 和 expert sampling strategy (Threshold / TopK / Threshold-TopK)。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB × 8（1B-scale 训练，24h）、H100 80GB × 32（3B/6B-scale 训练，64h/124h）。推理评估使用 NVIDIA A100-80GB 和 H100-80GB。

- 模型是什么。数据集和bench分别是什么。
  模型规模：DS-MoE-1B (1067M)、DS-MoE-3B (2846M)、DS-MoE-6B (6343M)。每个 MLP 层有 32 个 expert（D_ffd 分别为 256/384/512），每个 Attention 层有 8-16 个 MoA expert。训练数据：Pile 子集，1B-scale 用 30B tokens，3B/6B-scale 用 100B tokens，tokenizer 使用 CodeGen tokenizer。Benchmarks：PiQA、HellaSwag、WinoGrande、SciQ、Arc-e、Arc-c（zero-shot），WikiText perplexity（language modeling）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文代码未在 GitHub 公开。使用 SimpleMoE (Tan et al. 2024) 的 ParallelLinear 实现稀疏推理，使用 dMoE (Gale et al. 2023, MegaBlocks) 实现 SMoE baseline。vLLM 部署部分使用开源 vLLM (Kwon et al. 2023)。

  **算法 Pipeline 核心流程**：
  
  1. **Dense Training（前向）**：
  ```
  # Router 计算所有 N 个 expert 的分数
  S = Softmax(h(X))           # S ∈ R^N
  # 计算所有 N 个 expert 的输出
  for i in 1..N:
    O_i = e_i(X)              # Expert_i 的前向
  # 加权求和所有 expert（而非仅 top-K）
  O = sum_i(S_i * O_i)        # dense weighted sum
  ```
  
  2. **Dense Training（反向）**：
  ```
  # 传统稀疏训练的 Router 梯度（有 mask M ∈ {0,1}^N）:
  ∇S = [e_1(X), ..., e_N(X)]^T ∇O ⊙ M   # 仅激活 expert 有梯度
  # DS-MoE 的 Dense 梯度：
  ∇S = [e_1(X), ..., e_N(X)]^T ∇O        # 所有 expert 梯度保留
  ∇e_i(X) = S_i · ∇O                      # 每个 expert 获得完整梯度
  ```
  
  3. **MI Loss 负载均衡**：
  ```
  # 最大化 expert 分布熵 H(e) 促进负载均衡
  H(e) = -Σ_{i=1..N} p(e_i) log p(e_i)
  # 最小化条件熵 H(e|X) 促进专家集中
  H(e|X) = -Σ p(e_i|x) log p(e_i|x)
  # 总 MI Loss
  L_MI = -H(e) + (1/|X|) Σ_{x∈X} H(e|x)
  # 总 loss
  L = L_LM + α · L_MI
  ```
  
  4. **Sparse Inference**：
  ```
  # 方法一：固定 TopK
  A = argtopK(S, K)           # 取分数最高的 K 个 expert
  O = Σ_{i∈A} S_i · e_i(X)   # 仅计算选中的 expert
  # 方法二：动态阈值
  p_norm_i = S_i · N           # 归一化概率
  A = {i | p_norm_i > ε}       # 分数超过阈值 ε 的 expert
  ```
  
  5. **Mixture of Attention Head (MoA)**：
  ```
  # 每个 MoA expert i 计算 N_head 个 query vectors
  Q_i = W_q_i @ X              # Q_i ∈ R^{N_head × d_head}
  # 共享的 KV cache
  K, V shared among all experts
  O_ij = Softmax(Q_ij @ K^T) @ V @ W_o_j
  # 最终输出：top-K experts 的加权和
  O = Σ_{k=1..K} S_{A_k} · Σ_{j=1..N_head} O_{A_k,j}
  ```
  
  张量计算细节（以 DS-MoE-3B, D_emb=3072, N_ffd=32, D_ffd=384 为例）：
  ```
  X: [B, 3072]                  # 输入 hidden states
  Router: W_r [3072, 32]        # Router 权重
  S: [B, 32] = Softmax(X @ W_r) # Router scores
  # Dense Training Forward
  E_i(X) = GeLU(X @ W_up_i + b_up_i) @ W_down_i + b_down_i  # Expert FFN
  O = Σ_i S_i · E_i(X)          # [B, 3072]
  # Sparse Inference (TopK=6)
  top_idx = argtopK(S, 6)       # [B, 6]
  O = ParallelLinear(X, top_idx, all_expert_weights)  # SimpleMoE
  # 活跃参数: 6 × (3072×384 + 384×3072) ≈ 14M per layer
  # 活跃hidden比例: 14M/40M ≈ 34%
  ```
  
  关键超参数：α_MoA: 3.5e-4 (1B) / 2e-4 (3B/6B), α_MoE: 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)。Sparse inference threshold ε=0.48 用于主实验。Optimizer: AdamW, lr=3e-4, cosine schedule, warmup 1B/2B tokens, weight decay 0.01, gradient clip 1.0, batch size 0.5M/2M tokens, seq_len 2048, FSDP + activation checkpointing.

## DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是将 MoE 专家剪枝重新定义为连续优化问题，提出 Differentiable Expert Pruning (DiEP)，包含三个核心组件：(1) **Intra-layer/Inter-layer 双层次可微分数**：定义 intra-layer importance scores α（每层内各专家的相对重要性）和 inter-layer importance scores β（每层对整体模型的贡献权重），通过 softmax 归一化得到 continuous relaxation；(2) **交替梯度优化**：以 α:β = 3:1 的比例交替更新两个参数组，目标函数为 L = L_ce(y, F'(x; α, β)) + λ·∥F'(x; α, β) − F(x)∥_F（λ=0.01），其中正则项为 reconstruction regularization term，鼓励剪枝后模型与原模型输出一致；(3) **全局排序剪枝**：最终 s_i^(l) = α_i^(l) · β^(l)，全局排序所有专家重要性，按照 sparsity ratio r 统一删除 bottom-K（K = N·L·r）最不重要的专家，实现跨层非均匀剪枝。(4) **Adaptive Expert Skipping 在线推理加速**：在推理时为每个 token 跳过冗余专家计算，γ = γ1 × γ2，其中 γ1 是 calibration data 中 routing weight ratio w_e1/w_e0 的中位数，γ2 是基于 CKA similarity 的专家输出相似度比。当 w_e1 < γ·w_e0 时跳过专家 e1。实验比较：在 Mixtral 8×7B、Mixtral 8×7B-Instruct、Mixtral 8×22B、Deepseek-MoE-16B、Qwen2-57B-14A 五个 MoE 模型上，在 25% 和 50% expert sparsity 下对比 M-SMoE (merge)、Expert Trimming (activation frequency)、NAEE (exhaustive search)、S-SMoE (similarity-based merge) 的 MMLU/OpenBookQA/BoolQ/RTE 等 zero-shot benchmark 性能。消融实验验证 α、β 组件重要性、交替更新比例、λ 超参数、epoch 数、calibration data size 的影响。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A800 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mixtral 8×7B（32 MoE layers, 8 experts/layer, top-2 activation）；(2) Mixtral 8×7B-Instruct（同架构指令微调版）；(3) Mixtral 8×22B（56 MoE layers, 8 experts/layer, top-2, 141B total/39B activated）；(4) Deepseek-MoE-16B（28 layers, 64 experts/layer, 2 shared + 6 routed per token）；(5) Qwen2-57B-14A（28 MoE layers, 64 experts/layer, top-8 activation）。Calibration 数据：C4 dataset 随机采样 128 条序列用于 differentiable search。Evaluation benchmarks：MMLU（57 subtasks, 4 domains）、OpenBookQA、BoolQ、RTE；附录中额外包含 ARC-c、ARC-e、HellaSwag、WinoGrande。Domain-specific 验证：GSM8K 数学推理数据集（使用 C4 和 MATH 两种 calibration data）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码链接。方法基于 HuggingFace Transformers + LM Evaluation Harness (lm-eval-harness) 实现。核心算法流程如下：
  ```
  # Algorithm: DiEP Differentiable Expert Pruning
  Input: 校准数据集 D_cal (128 samples from C4), 完整 MoE 模型 F,
         初始化 α_i^(l) = 1, β^(l) = 1, λ = 0.01, epochs = 10
  
  for epoch in 1..E:
    for batch in D_cal:
      # Forward with continuous relaxation
      ᾱ_i^(l) = softmax(α_i^(l))                     # [N_experts] per layer
      y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · FFN_i(x^(l)) # Eq.5, 加权专家输出
      L = L_ce(y, F'(x; α, β)) + λ · ∥F'(x; α, β) − F(x)∥_F  # Eq.7
  
    # Alternating update (3:1 ratio)
    for step in 1..3:                                # α updates (3 steps)
      α ← α − η_α · ∇_α L(α, β)                     # fix β, update α
    for step in 1..1:                                # β updates (1 step)
      β ← β − η_β · ∇_β L(α, β)                     # fix α, update β

  # Global pruning after optimization
  for each expert i in layer l:
    s_i^(l) = α_i^(l) · β^(l)                        # Eq.10, global importance
  K = N_layers · N_experts · r                        # total experts to prune
  P = bottom-K indices sorted by s_i^(l) globally      # select least important
  m_i^(l) = 0 if i ∈ P else 1                        # Eq.11, pruning mask
  
  # Adaptive Inference Skipping
  γ1 = median(w_e1 / w_e0) over calibration data     # per-layer routing ratio
  γ2 = ρ(y_e0, y_e1) / mean(ρ(y_ei, y_ej))          # CKA similarity ratio
  γ = γ1 · γ2                                         # per-layer skip threshold
  During inference: if w_e1 < γ · w_e0, skip expert e1
  ```
  关键张量维度：α ∈ R^(L×N)（L 层，N experts/layer），β ∈ R^L。剪枝后 α 仅需约 0.01% 额外参数量。Pruning time: Mixtral 8×7B 仅 0.23h（vs NAEE 1.31h 的 exhaustive search）。Deepseek-MoE-16B pruning 仅 0.28h（vs NAEE ≈94000 days 因 exhaustive search 不可行）。50% sparsity 下 Mixtral 8×7B 推理获得 1.28× speedup、48% GPU memory reduction，保留约 92% 原模型性能。

## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  DES (Dynamic Expert Sharing) 将 MoE 优化从 token-centric pruning 转变为 sequence-level coreset selection，最大化并行解码块内的 expert 复用。两种策略：
  1. **DES-Seq**（Intra-Sequence Sharing）：对每个 token 取 Top-k experts，取所有 token 的并集作为共享 coreset —— C_DES-Seq = ∪_{n=1}^{N} TopK(I_n, k)。
  2. **DES-Vote**（Saliency-Aware Voting）：所有 token 按加权 router saliency 投票选举 coreset —— 先 mask 每 token 的 local Top-K 之外权重，跨序列聚合加权投票 V_i = Σ_{n=1}^{N} Masked(I_{n,i})，再取 Top-M_core experts。
  实验比较 DES-Seq (k=2, k=3) 和 DES-Vote (β=0.10, β=0.15 for LLaDA2.0; β=0.4, β=0.6 for LLaDA-MoE-7B) vs Vanilla、Top-K、NAEE、MC-MoE 在生成 benchmark 上的 accuracy 和 expert load、latency 表现。

- 硬件平台是什么，配置是什么。
  NVIDIA B200 GPU，CUDA 13.1，Intel Xeon 6960P CPU。使用 NVIDIA Nsight Systems 进行 kernel profiling。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaDA2.0-Mini (16B) 和 LLaDA-MoE-7B-A1B-Instruct (7B)，均为 MoE dLLM 架构。推理框架：dInfer + Fast-dLLM (KV cache 方法，0.9 confidence-based sampling)。
  数据集/benchmark：HumanEval、MBPP、GSM8K、MATH500，评估 long-form generative decoding 和多样化推理能力。Block length 32 (16 prefix + 16 suffix cache tokens)。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码（arXiv 2602.00879，2026年1月）。Catalyzex 标记为 "Paper and Code" 但无公开 GitHub 链接。

  **DES 算法伪代码（Algorithm 1）**：
  ```
  输入: 序列信息 I, Coreset 选择函数 Φ, 激活函数 σ, 目标 K
  输出: 层输出 Y
  
  Stage 1: Sequence-level Consensus
    C ← Φ(I)                        // 识别高效用 expert coreset
  
  Stage 2: Constrained Local Routing
    for each token n ∈ {1, ..., N}:
      S_n ← TopK(I_n|_{i∈C}, K)     // 在 coreset 内路由
      g_n ← σ(I_n|_{i∈S_n})          // 重新归一化 gate weights
      y_n ← Σ_{i∈S_n} g_{n,i} · E_i(x_n)
    return Y = {y_1, ..., y_N}
  ```

  **DES-Vote 具体过程（Algorithm 3）**：
  ```
  输入: Router logits I (shape: N×M), Coreset size M_core, Top-K
  1: I_m ← Mask(I, K)               // 保留 local Top-K，其余置零
  2: V ← Σ_{n=1}^{N} I_{m,n}        // 跨 token 聚合加权投票 (shape: M)
  3: C ← TopK(V, M_core)            // 排序选 top-M_core experts
  ```

  **延迟模型**：L_MoE(Φ) ≤ b·|Φ(I)| + a·(N·K)，其中 b 为 HBM→SRAM weight fetching cost，a 为 marginal compute cost。优化目标：min |Φ(I)| s.t. A(Φ(I)) ≥ A_base - ε。

  **关键结果**：DES-Vote (β=0.15) 在 LLaDA2.0-Mini 上减少 unique expert activations 55%（T=84→38），保留 99.5% relative accuracy，MoE 层延迟降低 38.0%。DES-Vote 在相同 coreset size 下始终优于 DES-Seq（Top-K recall 更高，reconstruction loss 更低）。

## Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **EEP (Efficient Expert Pruning)**，一种无梯度进化策略（gradient-free evolutionary strategy），用于 SMoE 模型的 expert 剪枝和合并。EEP 分为两阶段：(1) **Expert Pruning Phase**：通过进化搜索找到最优的 expert 剪枝模式，引入 Router Mapping 矩阵 WRM ∈ ℝ^{E'×E} 和 Expert Merging 矩阵 WEM ∈ ℝ^{E'×E}，两矩阵初始化为 one-hot 行向量，仅保留选中的 expert 权重及对应路由权重。此阶段不更新任何网络参数(no gradient computation)。(2) **Expert Merging Phase**：WRM 和 WEM 解耦，从离散 0/1 值过渡到连续值，将剪枝掉的 expert 知识融合到保留的 expert 中，通过 block-wise weighted sum 合并 expert 权重: θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}。进化搜索采用 Crossover + Mutation + Selection 迭代优化。
  实验比较包含两大使用场景：(a) **减少 total expert 数量**（节省 GPU 显存）：EEP vs Random Selection / Frequency-based pruning / Soft Activation pruning / NAEE（对 Mixtral 8×7B 从 8 expert 剪枝到 4/2；Mixtral 8×22B 剪枝到 4/2；Qwen1.5-MoE 从 60→30/15；Qwen2-MoE 从 64→32/16/8/4/2/1）；(b) **减少 active expert 数量**（加速推理）：Top-2→Top-1 with EEP merging vs Full Model / NAEE Dynamic Skipping；(c) 组合场景：total=4, active=1；(d) 泛化测试：MMLU 50+7 split, IID + OOD；(e) 消融实验：group number (4 vs 32), search iterations；(f) 微调场景：EEP 用于不剪枝情况下的 fine-tuning；(g) 性能画像：显存占用和推理加速比。

- 硬件平台是什么，配置是什么。
  主要实验在 NVIDIA GPU 上完成（具体型号论文未明确说明计算卡型）。性能画像实验使用 2× NVIDIA A100 GPU，batch size=256 测试 SQuAD 数据集。EEP search 过程仅需推理（无梯度计算），可在仅支持推理的设备上运行。搜索过程：Pruning Phase 40 iterations + Expert Merging Phase 160 iterations。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mixtral 8×7B-Instruct (8 experts, top-2, 47B total/13B active)；(2) Mixtral 8×22B-Instruct (8 experts, top-2, 141B total/39B active)；(3) Qwen1.5-MoE-A2.7B-Chat (60 experts, top-4, 14.3B total/2.7B active)；(4) Qwen2-MoE-A14B-Chat (64 experts, top-4, 57B total/14B active)。
  数据集/benchmarks：(1) SuperGLUE tasks: COPA, MultiRC, WIC, WSC, RTE, BoolQ, CB, ReCoRD；(2) SQuAD（阅读理解和问答）、DROP（离散推理阅读理解）；(3) MMLU（57 数据集，用于 IID/OOD 泛化测试）。每个数据集随机抽取训练集子集做进化搜索，测试集做评估。所有数据集使用统一生成式评估方法，基于 OpenCompass 框架实现 prompt 设计和模板匹配。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自清华大学、Infinigence AI、KU Leuven、Microsoft Research、上海交通大学。代码开源在 https://github.com/imagination-research/EEP。

  **EEP 算法核心流程（对应论文 Algorithm 1）**：

  ```
  Algorithm: Evolutionary Search of EEP
  Input: 全量 expert 权重 Θ={θ^l_1,...,θ^l_E}_{l=1..L}, 评估器 F
  Hyperparams: Epochs (迭代轮数), M_CP (candidate parents 数量), Iter (每轮 mutation 数)
  Output: 最优 W* = {W^l_EM, W^l_RM}_{l=1..L}

  1: P ← ∅
  2: 随机初始化 W_init，确保每行是 one-hot vector
  3: P ← P ∪ {(W_init, F(W_init))}
  4: for phase in {Pruning Phase, Merging Phase}:
  5:   for t = 1..Iters:
  6:     NG ← ∅
  7:     for i = 1..Epochs:
  8:       CP ← {W_i | F(W_i·Θ) ranks within top min(M_CP,|P|) in P}
  9:       W_f, W_m ← RandomSample(CP)  // 从候选父代中随机采样两个
  10:      W_new ← Mutate(Crossover(W_f, W_m))
  11:      NG ← NG ∪ {(W_new, F(W_new))}
  12:     P ← P ∪ NG
  13: return W* ← argmin F(W)
  ```

  **Expert Pruning Phase 张量计算**：
  - WRM, WEM 初始化为 one-hot rows (每行仅一个元素为1，其余为0)
  - 且约束 WRM = WEM
  - Router 变换: G' = WRM · softmax(Z · W_G)，将 E 维路由权重降为 E' 维
  - Expert 剪枝: θ'_j = WEM 的 one-hot 行选择对应的原始 expert 权重

  **Expert Merging Phase 张量计算**：
  - WRM 和 WEM 解耦，元素从离散 0/1 过渡到连续值
  - 对第 j 个新 expert: θ'_j = {Σ_{i=1}^{E} ω_ji · W₁i, Σ_{i=1}^{E} ω_ji · W₂i, Σ_{i=1}^{E} ω_ji · W₃i}
  - 其中 ω_ji 来自 WEM 的第 j 行第 i 列
  - 可在不更新任何网络参数的情况下完成，仅需模型推理

  **Crossover 操作**：沿 retained expert 维度组合父代的 merging coefficients。
  **Mutation 操作**：Pruning Phase 随机替换 pruned experts 为其他 experts 并相应设置路由权重；Merging Phase 对 merging coefficients 逐元素加入 Gaussian noise。

  **关键结果**：Mixtral 8×7B 剪枝 75% experts（8→2），参数减少 72%，性能与 full model 可比。剪枝 50% experts 在 SQuAD 上准确率从 53.4% 提升至 75.4%。Active experts 从 2→1 实现 prefill 加速 1.63×。4 total + 1 active expert 组合节省 47% GPU 显存，1.41× 推理加速。

## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 MELD（Mixture of Experts on Large Language Models for Data Preprocessing），一个基于 MoE 架构的通用低资源数据预处理（DP）求解器。核心贡献包括：(1) 增强型 RAG 系统用于跨域检索与自标注；(2) 启发式 meta-path 搜索用于数据增强；(3) 基于信息瓶颈理论的 expert 精炼；(4) 独立 router network 基于对比学习训练实现 top-k expert 调度。实验在 19 个数据集、10 个 DP 任务上与 non-LLM baseline、LLM baseline 和 Mixtral 8×7B 进行 few-shot 性能比较。

- 硬件平台是什么，配置是什么。
  单机：256GB RAM、32 处理器 Intel Xeon Gold 5320 CPU @2.20GHz、4× NVIDIA GeForce RTX 3090 GPU（24GB VRAM）。训练与推理均在 consumer-level GPU 上完成。

- 模型是什么。数据集和bench分别是什么。
  **Backbone 模型**：RAG 系统使用 bge-large-en（Sentence-BERT），Expert 模型使用 Mistral-7B。LoRA（Low-Rank Adaptation）进行参数高效微调。对比 baseline 包括：
  (1) Non-LLM 方法：Raha(ED)、IPM(DI)、DeepBlocker(Blocking)、Ditto/PromptEM(EM)、Baran/Garf(DC)、RECA(CTA)、TURL(RE/EL)、CONSchema/SMAT(SM)、MAVE(AVE)；
  (2) LLM 方法：JellyFish(13B)、TableLLaMa(7B)、ExtractGPT；
  (3) MoE 模型：Mixtral 8×7B。
  **19 个数据集**覆盖 10 个 DP 任务：EM(Amazon-Google, Walmart-Amazon, WDC-All, Ant-Buy)、Blocking(Semi-Text-Watch, Semi-Text-Computer)、DC(Hospital, Rayyan, Beer)、ED(Hospital, Rayyan, Beer)、CTA(SemTab19, WebTables)、RE(WikiGS-RE)、EL(WikiGS-EL)、SM(CMS, Synthea)、DI(Walmart, Amazon, Restaurant)、AVE(OA-mine)。
  **指标**：F1 score(EM/ED/DC/SM)、accuracy(DI/AVE)、top-1 accuracy(EL)、top-1 recall(Blocking)、micro-F1(CTA/RE)。Few-shot 设置为 ≤10% 标注数据。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/authurlord/MELD。使用 LLaMA-Factory 进行 expert 训练，Punica + vLLM 进行多 LoRA 推理。

  **算法 Pipeline 伪代码**（MELD 训练流程）：
  ```
  Input:  Tasks T = {T_1,...,T_n}, few-shot labeled data X = {X_1,...,X_n}
  Output: Expert set E^{aug}, Router network N

  // Step 1: Enhanced RAG for Cross-domain Retrieval
  for each task T_i:
    初始化 sentence-bert 模型 M_RAG
    for each query q in X_i:
      搜索正例集 P_q（对齐的 entries），负例集 N_q（未对齐的 entries）
    Fine-tune M_RAG with contrastive loss:
      min Σ_{p∈P_q} -log( exp(<emb_q, emb_p>/τ) / Σ_{p'∈P_q∪N_q} exp(<emb_q, emb_{p'}>/τ) )
    用 fine-tuned M_RAG 自标注未标注数据 X̃_i
    通过 query 变换跨任务扩充标注数据 X_i → X_i (enlarged)
    初始化 expert e_i: 用 X_i 对 Mistral-7B 做 LoRA fine-tune

  // Step 2: Heuristic Meta-path Search
  for each task T_i:
    贪心搜索 meta-path E_i = {e_{j1}, ..., e_{jm}}
      目标: argmax_{E_i} Eval(e_i, X_i^{E_i})
      约束: 用户定义的 sub-optimal paths 缩减搜索空间
    X_i^{aug} ← 沿 meta-path E_i 依次查询 experts 进行数据增强

  // Step 3: Expert Refinement (Information Bottleneck)
  for each expert e_i:
    迭代 σ 次:
      (a) min_{θ_M_RAG} I(M_G(X_i); M_G(RAG(X_i)))
          // 通过控制 RAG 采样参数和 meta-path 添加多样化训练数据 ΔX_i
      (b) max_{θ_M_G} I(M_G(X_i); Y_i)
          // 用 X_i ∪ ΔX_i 继续 LoRA fine-tune M_G
  Output: E^{aug} = {e_1^{aug}, ..., e_n^{aug}}

  // Step 4: Router Network Training
  初始化 transformer-based router N (共享 M_RAG 编码层)
  for each labeled query q_u:
    N(q_u) → top-k experts from E^{aug}
    优化目标:
      max Σ_{e_i∈N(q_u)} I(e_i(q_u^i); l_u^i)   // 专家相关
      min Σ_{e_i≠e_j∈N(q_u)} I(e_i(q_u^i); e_j(q_u^j))  // 专家多样
    近似为对比学习损失训练 N
  ```

  **张量计算示例**（以 EM 任务 meta-path 增强为例）：
  - 给定 EM query q = (t_1, t_2)，meta-path E_EM = {e_blocking, e_DI, e_AVE, e_EM}
  - Step 1: e_blocking(q) → 判断 t_1, t_2 是否可能匹配（候选对筛选）
  - Step 2: e_DI(t_1) → 对 t_1 缺失属性做数据填补，输出增强 t_1'
  - Step 3: e_AVE(t_1') → 提取 t_1' 的关键属性值，附加为新特征
  - Step 4: e_EM(t_1', t_2) → 最终匹配判断

  **Router 推理张量计算**：
  - Input: query embedding emb_q ∈ R^{d}（从 M_RAG 编码器获取）
  - N(emb_q) → softmax(W_N · emb_q) ∈ R^{n}（n 个 expert 的权重分布）
  - Top-k 选择: indices = argsort(weights)[-k:]
  - Final output: y = Σ_{i∈top-k} g_i · e_i^{aug}(q)，其中 g_i = softmax(weights_i)
  - 实际部署时每个 query 只激活 k=3 个 experts，其余 expert 的 LoRA 权重不加载

## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **Faster-MoA** 的两个算法级创新：(1) **层次化树状 Agent 拓扑**：将 all-to-all 全连接 MoA 替换为三层树结构（9-3-1），每层 agent 被分组为 clusters，下一层 agent 仅连接其对应 cluster 的前驱 agent，形成局部信息聚合→全局聚合的层级结构。(2) **语义引导的运行时动态 Early-Exit**：在每层中，通过 FrobCosSim（Frobenius Cosine Similarity）+ 置信度几何平均计算早退概率 Q，在小 agent 输出足够高质量时以概率 Q 提前终止大 agent 运行。核心计算：先用 Qwen3-Embedding-4B 将各 agent 输出文本编码为 last-layer hidden states T_i ∈ R^{n×h}，计算 feature-wise correlation matrix U = T_i^T × T_i ∈ R^{h×h}，在两矩阵之间计算 FrobCosSim；再结合 token-level log-probability 的几何平均置信度 C_ℓ，计算合成质量分数 Q = √(C̄ · B)，B 为校准后的相似度。
  实验比较：(a) 模型激活分布（4B/8B/32B 各被调用的比例），更难任务（IFBench）更大模型被更多调用; (b) EE 开销（~5% 额外延迟换来 10-50% E2E 减少）; (c) Tree-only vs Tree+Incremental Prefill vs Fully-integrated Faster-MoA vs All-to-all Baseline 的 E2E 延迟和准确率; (d) 准确率在五个 benchmark 上对比（GSM8K/MATH-500/AIME2025/MMLU-ProX-Lite/IFBench）。

- 硬件平台是什么，配置是什么。
  6× NVIDIA H200 GPU（单台 H200 HGX Server），每个模型用两张 GPU（1 PE + 1 DE），总计三组模型（Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instruct、Qwen3-VL-32B-Instruct + 额外 Qwen3-Embedding-4B 用于动态 EE）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instruct、Qwen3-VL-32B-Instruct（来自 Qwen model family），外加 Qwen3-Embedding-4B 做 embedding 用于动态 EE routing。三个模型共享相同 tokenizer 避免异构 tokenizer 编排问题。采样参数按 model card 推荐设置。
  数据集/benc"hmarks：五个——GSM8K（小学数学推理）、MATH-500（中等数学）、AIME2025（竞赛数学）、MMLU-ProX-Lite（STEM 综合科学）、IFBench（指令遵循测试）。覆盖从易到难的数学推理和通用科学 QA 任务。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文代码未公开（Georgia Tech + Peking University + Samsung，提交 DAC 2026）。以下基于论文 (Sec. 4.1-4.2) 给出算法 pipeline 伪代码：

  **=== 层次化树状拓扑 (Tree Topology) ===**
  ```
  输入: user query
  输出: final answer from root agent

  Layer 1 (9 leaf agents, 3 clusters):
    Cluster 1: agents {a_{1,1}, a_{1,2}, a_{1,3}} (各自用 4B/8B/32B Qwen3-VL)
    Cluster 2: agents {a_{1,4}, a_{1,5}, a_{1,6}}
    Cluster 3: agents {a_{1,7}, a_{1,8}, a_{1,9}}
    每个 cluster 独立并行执行

  Layer 2 (3 aggregation agents):
    a_{2,1} ← 仅依赖 Cluster 1 的输出 (C(a_{2,1}) = Cluster 1)
    a_{2,2} ← 仅依赖 Cluster 2 的出 (C(a_{2,2}) = Cluster 2)
    a_{2,3} ← 仅依赖 Cluster 3 的输出 (C(a_{2,3}) = Cluster 3)
    只需自己的 local precursors 完成即可启动，无需等待其他 cluster

  Layer 3 (1 root aggregator agent):
    a_{3,1} ← 依赖所有 Layer 2 agents 输出 → 生成最终答案

  Latency 优势:
    T_ℓ^{tree} ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c
    vs T_ℓ^{all} = max_i t_{ℓ,i} (all-to-all)
    每个 successor 仅等待其连接的 precursors，无关子树可并发
  ```

  **=== 动态 Early-Exit (Algorithm 1) ===**
  参数：偏好相似度阈值 τ = 0.7（经验最优）
  ```
  输入: 当前层 ℓ 的已完成 LLM 输出 {O_1,...,O_ℓ} 和 token-level log-prob {log p_ℓ^i}_{i=1..n_a}
  输出: Early-exit probability Q

  1. 置信度计算:
     for i = 1..n_a:  // n_a = 当前已完成 LLM 数
       C_ℓ ← exp( (1/n_a) * Σ_{i=1}^{n_a} log p_ℓ^i )  // 几何平均置信度
     C̄ ← √( (1/ℓ) * Σ_{i=1}^{ℓ} C_i^2 )  // RMS 历史置信度

  2. 语义相似度计算:
     for i = 1..ℓ:
       T_i ← Embed(O_i) via Qwen3-Embedding-4B  // [n_i × h] last-layer hidden states
       T_ℓ ← Embed(O_ℓ)
       U ← T_i^T × T_i  ∈ R^{h×h}  // feature-wise correlation matrix
       V ← T_ℓ^T × T_ℓ  ∈ R^{h×h}
       Sim[i,ℓ] ← FrobCosSim(U, V)
         = trace(Corr(U)^T · Corr(V)) / (||Corr(U)||_F · ||Corr(V)||_F)
       Sim[ℓ,i] ← Sim[i,ℓ]

  3. 置信度加权相似度:
     W ← Σ_{i=1}^{ℓ} Σ_{j=1}^{i} C_i · C_j
     P ← (1/W) * Σ_{i=1}^{ℓ} Σ_{j=1}^{i} C_i · C_j · Sim[i,j] ∈ [0,1]

  4. 校准 (防止过度一致):
     B ← 1 - |P - τ| / τ  ∈ [0,1]  // τ=0.7 经验最优

  5. 合成质量分数:
     Q ← √(C̄ · B)^(1/τ)

  6. 以概率 Q 执行早退:
     终止当前层剩余未完成的 LLM
  ```

  关键数学公式：
  - Frobenius Cosine Similarity: FrobCosSim(U,V) = ⟨Corr(U), Corr(V)⟩_F / (||Corr(U)||_F · ||Corr(V)||_F)
  - 其中 Corr(U)_ij = U_ij / √(U_ii · U_jj)（将矩阵转为 correlation matrix 消除尺度/单位影响）
  - ⟨U,V⟩_F = trace(U^T V), ||U||_F = √⟨U,U⟩_F
  - 概率 Q 高 → 当前已完成 agent 输出置信度高且语义适度一致 → 可以早退
  - 概率 Q 低 → 需要等待更多（尤其是更大的）agent 完成

## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 METRO（Minimum Expert Token ROuting），一种面向 memory-bound 状态下 Expert Parallelism (EP) MoE serving 的 token-routing 算法。核心思想：在 memory-bound 的 decode 阶段，GPU 的 MoE 层延迟由 "activated expert replicas 数量" 而非 "处理 token 数量" 决定；而现有 token-balancing 路由算法（如 EPLB）会 inflate activated experts，导致 decode 性能退化。METRO 将 token routing 建模为 MIN-EXP-ROUTING ILP 问题——给定 N 个 expert、G 个 GPU、expert–GPU placement matrix A、每个 expert 在当前 batch 中的 token 数 T[1..N]，目标是最小化各 GPU 上 activated expert 数量的最大值 λ。Lemma 1 证明任何可行解可约化为"每个 expert 的所有 token 仅路由到一个 replica"。由于 ILP 最优解（二分搜索 + bipartite matching / Dinic max-flow）在 CPU 上的计算开销达 FFN 层时间的 31.4%–41.3%，GPU 上达 86.4%–103.8%，METRO 提出 GPU-native 贪心近似算法：并行遍历每个有 token 的 expert i，获取候选 GPU 集合 G_i，按 GPU ID 全序加锁避免死锁，选择当前 activated expert 计数器 L[g] 最小的 GPU g* 进行分配，复杂度 O(|A|)。同时 METRO 将 EP 的 all-to-all dispatch 替换为 all-gather dispatch，使每个 GPU 获得全局 top-k 信息作为算法输入。

  实验比较：(a) METRO vs EPLB token routing 在真实系统（vLLM, 8×A100）上的 decode latency (TPOT) 和 total token throughput（prefill+decode co-deployed）；(b) METRO vs EPLB 在专有模拟器（8-16×B200）上的相同指标；(c) METRO vs Optimal（二分搜索+max-flow）的 routing quality（max activated experts per GPU per decode batch）；(d) METRO 的 latency breakdown（greedy algorithm overhead, top-k overhead, communication overhead vs FFN reduction）；(e) decode throughput-latency Pareto-optimality 分析：变 batch size 和 TP/EP 配置下的 Pareto 前沿比较，METRO 在固定 SLO 下实现 1.98×–4.11× decode throughput 提升。

- 硬件平台是什么，配置是什么。
  真实系统：Google Cloud a2-highgpu-8g VM，8×NVIDIA A100 40GB GPU，600 GB/s NVLink（全部 GPU 在同一 NVLink domain）。模拟器：专有工业级 multi-GPU 性能模拟器（proprietary analytical roofline model），建模 8×B200 192GB（Qwen3-235B 实验）和 16×B200 192GB（DeepSeek-V3 实验），900 GB/s NVLink。模拟器支持 register、shared memory、compute、L2、HBM、network 多级硬件建模，含 TP/EP 并行策略映射，account for workload imbalance by estimating runtime based on the most bottlenecked GPU。

- 模型是什么。数据集和bench分别是什么。
  真实系统模型：Qwen3-30B-A3B（128 experts, fine-grained MoE）。模拟器模型：Qwen3-235B-A22B（128 experts）、DeepSeek-V3-671B（256 experts）。数据集：(a) 真实系统——InstructCoder（~114K code-editing instruction triplets, decode-heavy）、NuminaMath-1.5（~900K competition-level math problems with chain-of-thought, decode-heavy）；(b) 模拟器——Humaneval（164 Python programming problems, decode-heavy）、GSM8K（8,500 grade-school math word problems, prefill-heavy）。Context length：8K（真实系统），1K input + 2K output（模拟器）。Metrics：Total Token Throughput（prefill+decode co-deployed）、Decode Latency (TPOT)、Max activated experts per GPU per decode batch。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文作者包含 NVIDIA（N. Oswald, Q. Huang, H. Linsenmaier, C. Mei, R. Zhao, R. Borkar, B. D. Rouhani, D. Nellans, R. Krashinsky）和 Yale/Princeton/CMU 学术机构。**论文未明确提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。已集成 vLLM 的实现细节在 §V 中描述。算法核心流程如下：

  **=== MIN-EXP-ROUTING ILP 问题形式化 ===**
  ```
  输入: N experts, G GPUs, placement matrix A in {0,1}^{N×G}, token counts T[1..N]
  决策变量:
    x_{i,g} >= 0  : expert i 在 GPU g 上处理的 token 数
    y_{i,g} in {0,1}: expert i 是否在 GPU g 上被激活
    lambda >= 0     : 所有 GPU 中最大 activated experts 数
  目标: min lambda
  约束:
    (1) Sum_{i=1..N} y_{i,g} <= lambda, for all g   # 每 GPU activated experts <= lambda
    (2) Sum_{g=1..G} x_{i,g} = T[i], for all i      # 所有 token 必须被路由
    (3) x_{i,g} = y_{i,g} = 0 if A_{i,g}=0          # 路由遵守 placement matrix
    (4) x_{i,g} <= T[i] * y_{i,g}                   # token 仅路由到 activated expert
  ```

  **Lemma 1**: 任何可行解要么已将所有 token 路由到各 expert 的单个 replica，要么可映射到满足此性质的解而不增加目标值。

  **=== METRO 贪心近似算法（CUDA kernel, 单 SM）===**
  ```
  输入: N, G, A in {0,1}^{N×G}, T[1..N]
  输出: y_{i,g}
  初始化: L[g] <- 0, lock l_g for each g=1..G; y_{i,g} <- 0
  For each expert i = 1 to N in parallel:   // 并行度 = N (128-256)
      if T[i] > 0:
          G_i <- {g | A_{i,g} = 1}          // 查 placement matrix 获取候选 GPU
          acquire all locks {l_g | g in G_i} in GPU ID total order  // 全序加锁防死锁
          g* <- argmin_{g in G_i} L[g]      // 选 activated experts 最少的 GPU
          y_{i,g*} <- 1; L[g*] <- L[g*] + 1
          release all locks {l_g | g in G_i}
  lambda = max_g Sum_i y_{i,g}
  x_{i,g} = T[i] if y_{i,g}=1 else 0        // Lemma 1: 所有 token -> 单个 replica
  ```
  复杂度: O(|A|) vs 最优解 O((N+G)^2 * (|A|/G+N+G) * log(|A|/G))

  **=== METRO all-gather dispatch 流程 ===**
  ```
  For each MoE layer:
    1. All-gather: 每个 GPU 将本地 tokens 广播到所有 GPU
       (替换传统 all-to-all dispatch，获得全局 token 集合)
    2. Top-K: 每个 GPU 在全局 token 集合上独立计算 top-k
       -> 构建全局 T[1..N]（每个 expert 的总 token 数）
    3. METRO Routing: 每个 GPU 独立执行 Algorithm 1
       -> 确定每个 expert 在哪个 GPU 上激活
    4. FFN Compute: 每个 GPU 仅计算分配给自己的 expert FFN
    5. All-to-all Combine: 将 expert 输出 embedding 返回原 GPU
  ```

  关键性能数据：(a) Algorithm 1 开销最多 26us (1.5x replication)，但 FFN 时间减少最多 81us；(b) all-gather redundant top-k 额外开销 <3us (<1% 层时间)；(c) all-gather 通信在 memory-bound 小 batch 下（2MB/GPU on 600 GB/s NVLink ~ 3us）远低于 NCCL launch 固定开销（~100us）；(d) routing quality 在最优解的 10.9% 以内，比 EPLB 降低 up to 42.3% activated experts；(e) decode latency 降低 11%-22%, total token throughput 提升 3%-21%（co-deployed prefill+decode）。

## Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于 Inter-Layer Expert Affinity 的 ILP 建模算法，通过离线 profiling 捕捉 pre-trained MoE 模型中 cross-layer 的 expert routing conditional probability，将最大化 expert affinity 的 placement 问题转化为整数线性规划（ILP）并利用 Lagrange 对偶转化为最小化 token re-routing cost 问题。算法消除了对 expert replicas 的需求，不改变模型参数、无需 fine-tuning 或 topology-aware training loss。此外提出 Context-Coherent Expert Parallelism 将每 MoE 层的 2 次 Alltoall 降为 1 次 + 1 次轻量 AllGather。
  实验比较：(1) ExFlow vs Deepspeed-MoE baseline 在 Alltoall 通信量、cross-GPU token routing 比例、throughput 的对比；(2) expert affinity 在训练过程中的演化（iteration 0-18000）；(3) profiling 所需 token 数量敏感性（100-10k tokens）；(4) OOD 数据集 expert affinity 一致性（Pile vs C4/Dolma/Yelp）。

- 硬件平台是什么，配置是什么。
  Wilkes3 Ampere GPU 集群：每节点 2× AMD EPYC 7763 64-Core，4× NVIDIA A100-SXM4-80GB，NVLINK intra-node，dual-rail Mellanox HDR200 InfiniBand inter-node。实验规模 1-16 节点（4-64 GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT MoE 系列，基于 DeepSpeed-Megatron 从 scratch pre-train：
  - MoE GPT-M 350M: 24 layers, hidden dim 1024, experts 8/16/32/64
  - MoE GPT-M 470M: 32 layers, hidden dim 1024, 32 experts
  - MoE GPT-M 590M: 40 layers, hidden dim 1024, 32 experts
  - MoE GPT-XL 1.3B: 24 layers, hidden dim 2048, 16 experts
  所有模型使用 Top-1 gating + variable token capacity + GShard load balancing loss。
  数据集：Pile（800GB，train/eval split）。OOD 验证用 C4、Dolma、Yelp Reviews。
  Benchmark：端到端推理 throughput（tokens/sec）、Alltoall 通信时间占比、cross-GPU/cross-node token routing 比例。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/YJHMITWEB/ExFlow

  算法 pipeline 伪代码：
  ```
  # ===== 离线 Profiling =====
  N = 3000  # sampled tokens from Pile
  E = 64    # experts per layer
  L = 24    # MoE layers
  route_log = zeros(N, L)  # route_log[k][j] = expert index for token k at layer j

  for token k in sampled_tokens:
      for layer j in 0..L-1:
          expert_idx = model.gating[j](token_embedding[k])
          route_log[k][j] = expert_idx

  # ===== Build Conditional Probability Matrix =====
  # P[j][i][p] = P(E_{p,j+1} | E_{i,j})
  for layer j in 0..L-2:
      for token k in 0..N-1:
          i = route_log[k][j]
          p = route_log[k][j+1]
          count[j][i][p] += 1
      P[j][i][p] = count[j][i][p] / sum(count[j][i][:])

  # ===== ILP Formulation (Lagrange Dual) =====
  # Minimize:  Σ_{k=1..N} Σ_{j=1..L-1} R_{k,j}
  # Subject to:
  #   Σ_{i=1..E} x_{i,j}^p = E/P          (load balance: P GPUs/nodes)
  #   Σ_{p=1..P} x_{i,j}^p = 1            (each expert exclusive to one GPU)
  #   R_{k,j} >= x_{i,j}^p - x_{i,j+1}^p   (cross-GPU routing indicator)
  #   R_{k,j} >= x_{i,j+1}^p - x_{i,j}^p
  #   x_{i,j}^p in {0,1}, R_{k,j} in {0,1}

  # Stage 1: minimize inter-node routing
  # Stage 2: minimize intra-node routing (given stage 1 result)

  solution = solve_ilp(objective, constraints)
  expert_placement[j][i] = p  # expert i at layer j goes to GPU p

  # ===== Online Inference with Affinity Placement =====
  # Load model: GPU p loads experts {i: placement[j][i]=p} at each layer j
  AllGather(all_contexts)  # ensure context coherence
  for iteration in 0..max_tokens:
      for layer j in 0..L-1:
          # In-place attention (context already on local GPU)
          h = attention(h, local_context)
          # Gating: top-1 expert selection
          e_idx = gating[j](h)  # shared gating on all GPUs
          target_gpu = placement[j][e_idx]
          # Single Alltoall: dispatch tokens only
          h = Alltoall_dispatch(h, target_gpu)
          # Expert FFN on local GPU
          h = expert[e_idx](h)
      # End of iteration: sync new tokens
      AllGather(new_tokens)
  ```

  关键 insight：通过 Lagrange 对偶将最大化 affinity 问题转化为最小化 token re-routing 问题，避免了显式 replica 和 training-time topology loss。Expert affinity 是模型固有属性（论文证明在 OOD 数据集上归一化 affinity 值 0.989-1.005），仅需 1000-3000 样本 token 即可精确捕捉。

## Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于 Inter-Layer Expert Affinity 的 ILP 建模算法，通过离线 profiling 捕捉 pre-trained MoE 模型中 cross-layer 的 expert routing conditional probability，将最大化 expert affinity 的 placement 问题转化为整数线性规划（ILP）并利用 Lagrange 对偶转化为最小化 token re-routing cost 问题。算法消除了对 expert replicas 的需求，不改变模型参数、无需 fine-tuning 或 topology-aware training loss。此外提出 Context-Coherent Expert Parallelism 将每 MoE 层的 2 次 Alltoall 降为 1 次 + 1 次轻量 AllGather。
  实验比较：(1) ExFlow vs Deepspeed-MoE baseline 在 Alltoall 通信量、cross-GPU token routing 比例、throughput 的对比；(2) expert affinity 在训练过程中的演化（iteration 0-18000）；(3) profiling 所需 token 数量敏感性（100-10k tokens）；(4) OOD 数据集 expert affinity 一致性（Pile vs C4/Dolma/Yelp）。

- 硬件平台是什么，配置是什么。
  Wilkes3 Ampere GPU 集群：每节点 2× AMD EPYC 7763 64-Core，4× NVIDIA A100-SXM4-80GB，NVLINK intra-node，dual-rail Mellanox HDR200 InfiniBand inter-node。实验规模 1-16 节点（4-64 GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT MoE 系列，基于 DeepSpeed-Megatron 从 scratch pre-train：
  - MoE GPT-M 350M: 24 layers, hidden dim 1024, experts 8/16/32/64
  - MoE GPT-M 470M: 32 layers, hidden dim 1024, 32 experts
  - MoE GPT-M 590M: 40 layers, hidden dim 1024, 32 experts
  - MoE GPT-XL 1.3B: 24 layers, hidden dim 2048, 16 experts
  所有模型使用 Top-1 gating + variable token capacity + GShard load balancing loss。
  数据集：Pile（800GB，train/eval split）。OOD 验证用 C4、Dolma、Yelp Reviews。
  Benchmark：端到端推理 throughput（tokens/sec）、Alltoall 通信时间占比、cross-GPU/cross-node token routing 比例。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/YJHMITWEB/ExFlow

  算法 pipeline 伪代码：
  ```
  # ===== 离线 Profiling =====
  N = 3000  # sampled tokens from Pile
  E = 64    # experts per layer
  L = 24    # MoE layers
  route_log = zeros(N, L)

  for token k in sampled_tokens:
      for layer j in 0..L-1:
          expert_idx = model.gating[j](token_embedding[k])
          route_log[k][j] = expert_idx

  # ===== Build Conditional Probability Matrix =====
  for layer j in 0..L-2:
      for token k in 0..N-1:
          i = route_log[k][j]
          p = route_log[k][j+1]
          count[j][i][p] += 1
      P[j][i][p] = count[j][i][p] / sum(count[j][i][:])

  # ===== ILP Formulation =====
  # Minimize:  sum_{k,j} R_{k,j}
  # s.t.:
  #   sum_i x_{i,j}^p = E/P          (load balance)
  #   sum_p x_{i,j}^p = 1            (expert exclusive)
  #   R_{k,j} ge x_{i,j}^p - x_{i,j+1}^p
  #   R_{k,j} ge x_{i,j+1}^p - x_{i,j}^p
  #   x_{i,j}^p in {0,1}, R_{k,j} in {0,1}

  # Stage 1: minimize inter-node routing
  # Stage 2: minimize intra-node routing

  solution = solve_ilp(objective, constraints)
  expert_placement[j][i] = p

  # ===== Online Inference =====
  AllGather(all_contexts)
  for iteration in 0..max_tokens:
      for layer j in 0..L-1:
          h = attention(h, local_context)
          e_idx = gating[j](h)
          target_gpu = placement[j][e_idx]
          h = Alltoall_dispatch(h, target_gpu)
          h = expert[e_idx](h)
      AllGather(new_tokens)
  ```

  关键 insight：通过 Lagrange 对偶将最大化 affinity 问题转化为最小化 token re-routing 问题，避免了显式 replica 和 training-time topology loss。Expert affinity 是模型固有属性（OOD 数据集归一化 affinity 0.989-1.005），仅需 1000-3000 样本 token 即可精确捕捉。

## CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：将 Top-K 稀疏门控 MoE 块通过 Co-Upcycling 策略集成到多模态 LLM 的 **CLIP 视觉编码器**（ViT-L）和 **MLP 连接器**（两层线性 MLP）中。具体为：(1) 将 MLP 连接器的单个 MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(2) 将 CLIP ViT 每个 transformer encoder 层的 MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(3) 每个 MoE 专家的权重从预训练/预微调后的同位置 MLP 块初始化（Co-Upcycling）；(4) 三阶段训练：MLP 连接器预训练 → 全参数预微调（ALLaVA 标注数据）→ 含 MoE 块的视觉指令微调；(5) 辅助损失：负载均衡损失 L_b（α_b=0.1）+ Router z-loss L_z（α_z=0.01）。
  - **实验比较**：在 VQA（VQAv2, GQA, ScienceQA-IMG, TextVQA）和指令跟随（POPE, MME, MMBench, SEED-Bench, LLaVA-Wild, MM-Vet, MMMU, MathVista）benchmark 上与各尺寸组的 SOTA 多模态 LLM 对比（7B/13B/7B-MoE），并消融 MLP-MoE、CLIP-MoE、LLM-MoE（upcycled vs pre-trained Mixtral）、多分辨率输入、预微调阶段。

- 硬件平台是什么，配置是什么。
  - **预训练阶段**：8×A100 GPU，每 GPU batch size 32，ZeRO-2
  - **预微调阶段**：16×A100 GPU，每 GPU batch size 8，ZeRO-3
  - **视觉指令微调阶段**：32×A100 GPU，每 GPU batch size 8，总 batch size 256，ZeRO-3-offload
  - **学习率**：预训练 1e-3 → 预微调 2e-6 → 指令微调 4e-6（最终模型），Cosine 调度
  - **优化器**：AdamW

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Baseline = CLIP ViT-L（视觉编码器）+ 两层 MLP（连接器）+ Mistral-7B / Mixtral-8×7B（LLM）。CuMo 变体：
    - CuMo Mistral-7B：激活参数 7.80B，总参数 8.26B
    - CuMo Mixtral-8×7B：激活参数 13.45B，总参数 47.71B
    - CLIP-MoE 激活参数 0.50B（总 0.91B），MLP-MoE 激活参数 0.05B（总 0.10B）
  - **预训练数据**：LLaVA-558K
  - **预微调数据**：ALLaVA-Caption 708K（高质量图像标注）
  - **指令微调数据**：LLaVA-665K + ShareGPT4V 102K + LAION-GPT-V 11K + DocVQA 10K + SynDog-EN 50K + ChartQA 4K + DVQA 50K + AI2D 2K + InfoVQA 4K + ALLaVA 708K + LIMA 1K + ALLaVA-Text 143K，总计约 1.65M（全部开源）
  - **Benchmark**：ScienceQA-IMG, TextVQA, GQA, POPE, MME, MMBench(EN/CN), MM-Vet, VQAv2, LLaVA-Wild, SEED-IMG, MMMU(val), MathVista

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：代码 Apache 2.0，模型权重 CC BY-NC 4.0。GitHub: https://github.com/SHI-Labs/CuMo
  - **算法 Pipeline 伪代码**（以 CLIP 中一层 MoE 为例）：

    ```
    # 输入: X ∈ R^{N×C_in}（N 个 visual tokens, C_in 通道数）
    # 稀疏 MoE 块替换标准 MLP 的计算流程:

    # === Step 1: Router 计算专家权重 ===
    W = Softmax(Linear(X))           # W ∈ R^{N×S}, S 个 experts

    # === Step 2: Top-K 选择 ===
    W_K_indices, W_K_values = TopK(W, K)   # 选 K 个最高分 expert
    W_K = Softmax(W_K_values)              # ∈ R^{N×K}

    # === Step 3: 每个 token 仅通过选中的 K 个 expert ===
    X_out = zeros(N, C_out)
    for i in 1..K:
        expert_idx = W_K_indices[:, i]          # 每个 token 的第 i 个选中 expert
        tokens_i = X[expert_idx]                 # 路由到该 expert 的 tokens
        expert_out = MLP_expert[t](tokens_i)     # 通过对应 expert MLP
        X_out += W_K[:, i:i+1] * expert_out      # 加权累加

    # === Step 4: Co-Upcycling 初始化 ===
    # 每个 expert MLP 权重 = 同位置预训练/预微调阶段 MLP 权重
    for t in 1..S:
        MLP_expert[t].weight = MLP_pretrained[t].weight

    # === Step 5: 辅助损失 ===
    L_total = L_ce + 0.1 * L_balance + 0.01 * L_z_loss
    ```
  - **使用方式**：`pip install -e ".[train]"` 安装后，训练用 `python -m cumo.serve`；推理 CLI: `python -m cumo.serve.cli --model-path checkpoints/CuMo-mistral-7b --image-file <img>`，支持 4-bit/8-bit 量化。框架基于 PyTorch + LLaVA + flash-attn。

## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三项算法创新：(1) **Token-Adaptive Bit-Width Selection**：在每层 MoE 的每个 expert 前放置一个轻量级可训练的 bit-width router，动态选择每个 token 激活的 expert 的最优 bit-width。通过 Quantized Expert Capacity（{c_k}_{k=1}^K，预定义各 bit-width 的 token 容量上限，超限 token 随机丢弃）和 Dynamic Bit-Width Selection Loss（CE loss + α/L·Σp_k·b_k 正则化项，平衡 accuracy 和 bit-width 选择频率）实现动态 bit-width 分配。(2) **Matryoshka Weight Quantization (MWQ)**：首先用非对称量化 (asymmetric quantization) 将 expert 权重量化到最低 bit-width b_1（如 INT2），保留 block-level compensation（类似 GPTQ）；再用 binary residual quantization 对残差权重 R_{b_{k-1}} 逐次递增加 1-bit（±1 权重），使高 bit-width 嵌套低 bit-width，实现 INT2⊂INT3⊂INT4 的套娃结构，避免存储多份量化版本。(3) **Token-Expert-Weight Three-Level Co-Design**：将专家选择从"仅选 expert ID"扩展为"同时选 expert ID 和 bit-width"的 dual routing 范式（Figure 1），在算法层面实现精度-内存-延迟三目标联合优化。

  实验比较：(a) D2MoE-V1 (b_1=2, b_K=4) vs Hold-in-Memory (INT8, 全量 GPU 内存)、Matryoshka-Free (GPTQ INT2/3/4 独立存储+按需加载)、Hold-in-Memory-AWQ (INT4, 全量 GPU)、EdgeMoE (离线 profiling 固定 bit-width)、MoQE-DynaIO (统一 bit-width + 按需加载)，在 LLaMA-MoE-3.5B 和 Mixtral 8×7B 上的 PPL 和 5 个 zero-shot benchmarks (PIQA/ARC.e/BoolQ/HellaSwag/Winogrande)；(b) D2MoE 扩展到 dense LLM (LLaMA2-13B) 对比固定 INT4 GPTQ 的吞吐和峰值内存。

- 硬件平台是什么，配置是什么。
  离线预处理阶段（bit-width router 微调 + MWQ）：GPU server 配备 NVIDIA RTX 2×A6000。在线推理阶段：Environment 1 — NVIDIA RTX 3060 (6GB GPU memory) + Intel Core i7-11800H (32GB CPU) + Samsung 970 EVO (1TB, 3.5GB/s disk read)；Environment 2 — NVIDIA Jetson AGX Orin 64GB (SoC GPU) + ARM Cortex-A78AE + Samsung 970 EVO (1TB)。单卡推理，非分布式。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-MoE-3.5B（8 experts/layer, Top-2 routing）和 Mixtral 8×7B（8 experts/layer, Top-2 routing），均为 decoder-only MoE-based sparse LLMs。Dense 实验使用 LLaMA2-13B。训练数据：C4 数据集，2048 random 2048-token segments 用于训练 bit-width routers，128 random 2048-token segments 用于 MWQ calibration。Benchmarks：WikiText2 (PPL)、PIQA、ARC.e (ARC-Easy/Challenge)、BoolQ、HellaSwag、Winogrande（使用 lm-evaluation-harness）。吞吐评估：input/output length 均为 128。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供公开开源代码仓库。论文提及实现约 2,500+ LOC Python + CUDA，基于 PyTorch、Triton（I/O-compute 并行编程）和 CUDA（NVIDIA Ampere/Ada Lovelace 架构）。MWQ 算法流程：

  ```
  === Token-Adaptive Bit-Width Selection (Online Inference) ===
  Input: token x, layer l

  Step 1: Original MoE routing (select top-K experts)
    gate_logits = W_gate @ x                   // [E] expert selection logits
    topk_indices = TopK(Softmax(gate_logits), K=2)  // select 2 experts
  
  Step 2: Bit-width routing (per selected expert)
    For each selected expert e_i:
      bw_logits = W_bw_router[e_i] @ x        // [K_bw] bit-width logits
      bw_probs = Softmax(bw_logits)            // e.g., [INT2_prob, INT3_prob, INT4_prob]
      b_k = argmax(bw_probs)                   // selected bit-width
      // Loss: L = CE(p(x), q(x)) + α/L * Σ_l Σ_k p_k^l(x) * b_k
      // Second term pushes towards lower bit-width to save memory
  
  Output: [(expert_id_1, bit_width_1), (expert_id_2, bit_width_2)]

  === MWQ Quantization (Offline, per expert weight matrix W) ===
  Input: W ∈ R^{s×h} (FP16), candidate bit-widths {b_1, b_2, ..., b_K} e.g., {2, 3, 4}

  # Step 1: Asymmetric quantization to b_1 (e.g., INT2)
  Q_W_b1 = round(W / s_b1 + z_b1)              // [s×h] quantized to b_1 bits
  W_hat_b1 = (Q_W_b1 - z_b1) * s_b1            // dequantized approximation
  # Optimize s_b1, z_b1 via: argmin ||WX - W_hat_b1 X||_2^2

  # Step 2: Binary residual quantization for b_2...b_K
  R_b1 = W - W_hat_b1                           // residual
  For k = 2 to K:
    Q_W_bk = round(R_{b_{k-1}} / s_bk)         // binary residual (±1 values) 
    W_hat_bk = W_hat_b1 + Σ_{i=2}^{k} s_bi * Q_W_bi  // nested reconstruction
    R_bk = R_{b_{k-1}} - s_bk * Q_W_bk
    # Optimize s_bk via: argmin ||WX - W_hat_bk X||_2^2

  # Key property: W_hat_b2 ⊂ W_hat_b3 ⊂ ... ⊂ W_hat_bK (nested)
  # Storage: W_hat_b4 = W_hat_b2 + s_b3*Q_W_b3 + s_b4*Q_W_b4
  # vs traditional: need separate INT2, INT3, INT4 weights (3× storage)

  Output: {Q_W_{b_i}}_{i=1}^K (nested quantized weights)

  === MWQ Dequantization (Online, to get b_k-bit weight) ===
  Input: {Q_W_{b_1}, ..., Q_W_{b_k}}, {s_{b_1}, ..., s_{b_k}}, {z_{b_1}}
  
  W_fp16 = (Q_W_b1 - z_b1) * s_b1              // base b_1 weight to FP16
  For i = 2 to k:
    W_fp16 += s_{b_i} * Q_W_{b_i}               // accumulate binary residuals
  # W_fp16 is now equivalent to a b_k-bit quantized weight

  === Bit-width router training loss ===
  For batch S with T tokens, L layers, K bit-widths:
    L_CE = (1/T) * Σ_x CE(p(x), q(x))          // cross-entropy with FP16 teacher
    L_reg = (α/L) * Σ_l Σ_k p_k^l(x) * b_k     // regularization favoring low bit-width
    L_total = L_CE + L_reg
  # p_k^l(x): probability fraction for k-th bit-width at layer l
  # α controls accuracy-efficiency trade-off
  ```

  Quantized Expert Capacity：每个 bit-width expert 的 token 容量上限为 c_k·T（Σc_k=1），如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4。超限 token 随机丢弃，防止 fine-tuning 时 bit-width router 过拟合特定 token 序列。
