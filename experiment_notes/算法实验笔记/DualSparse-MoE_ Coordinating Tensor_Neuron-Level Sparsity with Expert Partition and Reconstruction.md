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
