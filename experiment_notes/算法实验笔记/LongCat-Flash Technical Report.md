## LongCat-Flash Technical Report

- 属于算法pipeline的实现是什么？实验比较什么？
  - LongCat-Flash 提出四项算法 pipeline 创新：
    1. **Zero-Computation Experts（零计算专家）**：在 MoE 的 FFN expert pool 中引入 Z 个 zero-computation experts，其输出直接等于输入（identity function），不引入额外计算量。Router 从 N+Z 个 experts 中选 top-K，实际激活的 FFN experts 数量随 token 的上下文重要性动态变化（18.6B-31.3B 参数，平均 27B）。通过 expert bias + PID 控制器调节零计算专家选择比例，确保平均计算负载收敛到目标值。公式：`MoE(x_t) = Σ g_i · E_i(x_t)`，其中 `E_i(x_t) = FFN_i(x_t)` if `1 ≤ i ≤ N` else `E_i(x_t) = x_t`。
    2. **Shortcut-Connected MoE (ScMoE)**：引入跨层 shortcut 连接，从同一层第一个 MLA block 的输出直连到 MoE block，允许前一层的 Dense FFN 计算与当前层 MoE 的 dispatch/combine 通信并行执行。将 token 维度切分为两个 chunk，实现 chunk 间互相重叠以及与 dense FFN 的重叠。
    3. **Variance Alignment for MLA**：在 MLA 的低秩分解路径中引入 scale-correction 因子 α_q 和 α_kv。因 query 压缩维度 d_q 和 KV 压缩维度 d_kv 产生的 query 分量 q_t^C 和 key 分量 k_t^C 方差与 d_model 不同，通过 `α_q = √(d_model/d_q)` 和 `α_kv = √(d_model/d_kv)` 将低秩路径分量的方差对齐到 d_model 参考尺度，解决缩放过程中的注意力分数不稳定问题。
    4. **Variance Compensation for Experts Init**：fine-grained expert segmentation 将每个 expert 细分为 m 个小 expert 后，gating dilution 和 dimensional reduction 各使输出方差减少约 m 倍。通过聚合输出乘以缩放因子 `γ = m` 补偿方差，保持 MoE 层输出方差与分割前一致。
    5. **Multi-Token Prediction (MTP)**：单一 dense layer 作为 MTP head，在训练中期引入，接受率超 90%。
  - 实验比较：
    - with/without zero-computation experts 在匹配计算预算下的 validation loss（Figure 3a）：zero-expert 变体激活 4.2B-7.0B 参数但保持 8 FFN experts 期望，loss 持续低于固定 top-k=8 的 baseline。
    - with/without ScMoE 在四种模型配置（2.4B-16B MLA, 3B-20B MHA, 15B-193B GQA）下的 training loss 曲线（Figure 4）：loss 几乎完全相同，证明 ScMoE 是 quality-neutral。
    - with/without scale-correction MLA 在 1B activated MoE 上的 validation loss 收敛曲线（Figure 5a）：scale-correction 带来更低 loss。
    - Model growth vs random init 在 6B activated MoE 上的 validation loss（Figure 5b）：model growth 初期 loss 上升但最终收敛到更优值。
    - MTP head 结构对比（Table 5）：Dense layer (1.41% params, 92.1% accept rate) vs ScMoE layer (4.17% params, 92.9% accept rate)，Dense layer 以更少参数取得接近的接受率。
    - Base model vs DeepSeek-V3.1, Llama-4-Maverick, Kimi-K2 在 MMLU/MMLU-Pro/CEval 等全面 benchmark 上评估。
    - Chat model vs DeepSeek-V3.1, Qwen3-235B, Kimi-K2, GPT-4.1, Claude4-Sonnet, Gemini2.5-Flash 在 ArenaHard, IFEval, MATH500, AIME, SWE-Bench, τ²-Bench 等全面 benchmark 上评估。

- 硬件平台是什么，配置是什么。
  - **训练**：NVIDIA H800-80GB GPU × 数万张（tens of thousands），200Gb/s per accelerator RDMA 网络，NVLink intra-node 互联。以 Expert Parallelism Group (EP=32) 为基本单元，CP=8，V-ZB pipeline。训练持续 30 天，98.48% 可用率。
  - **推理**：NVIDIA H800-80GB GPU，128 GPUs 作为典型部署单元（2 nodes × 16 GPUs 为最小 PD-disaggregation 部署单元），NVLink intra-node + RDMA inter-node (GPUDirect RDMA)。EP 部署可根据需要伸缩到上千 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LongCat-Flash，560B total params MoE，28 layers（不含 MTP layer），hidden dim 6144，64 attention heads per MLA，per-head dim 128。KV compression dim 512，query compression dim 1536。Dense FFN intermediate dim 12288，每个 FFN expert dim 2048。每层 512 FFN experts + 256 zero-computation experts，top-K=12。平均激活 ~27B params（18.6B-31.3B 范围）。Tokenizer: BPE，vocab size 131,072。
  - **Pre-training 数据**：~20T tokens（第一阶段 8k seqlen），多阶段包括通用预训练（two-stage data mixture）+ 推理代码增强（hundreds of billions of high-quality tokens）+ 长上下文扩展（80B → 32k + 20B → 128k tokens）。
  - **评估 Benchmark**：
    - Base model: MMLU, MMLU-Pro, CEval, CMMLU, GPQA, SuperGPQA, BBH, DROP, PIQA, WinoGrande, CLUEWSC, GSM8K, MATH, MBPP+, HumanEval+, MultiPL-E, CRUXEval
    - Chat model: MMLU, MMLU-Pro, ArenaHard-V2, CEval, CMMLU, IFEval, COLLIE, Meeseeks-zh, MATH500, AIME24, AIME25, BeyondAIME, GPQA-diamond, DROP, ZebraLogic, GraphWalks-128k, LiveCodeBench, Humaneval+, MBPP+, SWE-Bench-Verified, TerminalBench, τ²-Bench, AceBench, VitaBench
    - 内部 benchmarks: Meeseeks (multi-turn instruction-following), VitaBench (real-world agentic tasks from Meituan business scenarios)

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：模型权重开源在 Hugging Face (https://huggingface.co/meituan-longcat)，GitHub (https://github.com/meituan-longcat)。
  - **Zero-Computation Experts 例子（张量计算）**：
    假设 N=4 个 FFN experts，Z=2 个 zero-computation experts，K=3，K_e=2。
    ```
    # input x_t: [batch, seq, d_model]
    # router: softmax over N+Z=6 dimensions
    router_logits = router(x_t)  # [batch, seq, 6]
    router_probs = softmax(router_logits + expert_bias)  # expert_bias 由 PID controller 动态更新
    topk_indices = topk(router_probs, k=3)  # 从 6 个中选 3 个

    # Expert computation
    output = zeros_like(x_t)
    for idx in topk_indices:
        g_i = router_probs[idx]
        if idx < 4:  # FFN experts
            output += g_i * FFN[idx](x_t)
        else:        # Zero-computation experts
            output += g_i * x_t  # identity, no FLOPs

    # PID bias update (对 FFN experts only, zero-comp experts 不更新):
    # Δb_i = μ * (K_e/K * 1/N - T_i/(K*T_all))  for 1 <= i <= N
    ```
  - **ScMoE 例子（调度时序）**：
    Token 维度分为 chunk_a 和 chunk_b。单层执行顺序：
    ```
    Stage 1: MLA_0(x) → output → 分叉
    Stage 2 (并行):
      - Dense FFN(chunk_a_input) + Attn_0_QKV(chunk_a_input)
      - All-to-All Dispatch(chunk_b tokens)
    Stage 3: MoE GEMM(chunk_b) → All-to-All Combine(chunk_b)
    Stage 4 (并行):
      - All-to-All Combine(chunk_a) + Dense FFN(chunk_b_input)
      - Attn_1_Core(chunk_a): Core Attention + Output Projection
    ```
    通过 ScMoE shortcut，Dense FFN 计算可与 MoE 的 dispatch/combine 通信充分重叠，TPOT 理论值降低近 50%（vs DeepSeek-V3 的 TBO）。
