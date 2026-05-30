# 实验_Serving调度

## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Flood**，一个高效的离线推理框架。核心设计：(1) **全流水线并行 (PP) 架构**：放弃主流框架（如 vLLM）广泛使用的 Tensor Parallelism (TP)，因为它依赖高带宽互联（如 NVLINK），在没有 NVLINK 的低规格设备上通信开销可能占总执行时间一半以上。Flood 纯用 PP 在节点间和节点内实现，无需张量切分。(2) **多对一进程映射**：在单加速器上部署多个进程，每个进程绑定独立 CUDA stream，实现多进程共享单加速器，零 CPU 开销。(3) **Segment Cache**：替代 vLLM 的 block-based KV cache（PageAttention），在连续内存空间中分配 KV cache 为 `[max_token_num, num_head, head_dim]` 形状的大 block，避免小 block 导致的计算资源低效利用。支持超长输出时的 extend/append/wait 策略，以及原生 prefix caching。(4) 单机 8 加速器部署 9 个进程，确保始终有一个进程在等待 pipeline 第一阶段加速器可用，消除空闲时间。

  实验比较：Flood vs vLLM (v0.6.6.post2) 在 shareGPT benchmark 上的吞吐量（tokens/s）对比——Ling-Lite 在 1×Device E 上 5869 vs 4355 (1.35×)；Ling-Lite 在 1×Device C 上 5451 vs 3576 (1.52×)；Ling-Plus 在 16×Device B 上 4857 vs 2331 (2.08×)；Ling-Plus(FP8) 在 8×Device E 上 6569 vs 2742 (2.40×)。

- 硬件平台是什么，配置是什么。
  五种异构 AI 加速器：Device A (370 TFLOPS, 64GB)、Device B (120 TFLOPS, 96GB)、Device C (312 TFLOPS, 80GB)、Device D (989 TFLOPS, 80GB)、Device E (147 TFLOPS, 96GB)。推理性能测试使用 1×、8×、16× 配置，缺乏 NVLINK 等高速互连。

- 开源Serving框架是什么。修改了什么。
  开源框架：**Flood**（[https://github.com/alipay/PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration)），是全新设计的离线推理框架。对比如 vLLM，Flood 的修改/创新：(a) 完全放弃 TP 转而采用全 PP 架构；(b) 用 Segment Cache 替代 PageAttention/block table；(c) 多对一进程-加速器映射替代一对一映射；(d) 多 stream 隔离替代单 stream。Flood 未修改 vLLM，而是独立设计。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：[https://github.com/alipay/PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration)。Flood 全 PP 离线推理全流程：

  ```
  === 系统初始化 ===
  单机 8 加速器场景:
    启动 9 个进程（>加速器数量）
    Pipeline Stage 分配: DP_0→Acc0, DP_1→Acc1, ..., DP_7→Acc7
    每个进程绑定独立的 CUDA stream
    第 9 个进程为备选，等待任意 stage 空闲

  === Segment Cache 初始化 ===
  对每个请求预分配连续 KV cache:
    cache_tensor = [max_token_num, num_head, head_dim]   # 连续内存
    # vs vLLM: [num_blocks, block_size, num_head, head_dim]  # 分散小 block

  === 单请求推理流程 ===
  Input: prompt_tokens (请求文本)

  Step 1 - Prompt 处理:
    # 分配连续 segment 给 prompt
    segment = allocate_contiguous_memory(prompt_len + max_output_len)
    # TP 模式需切分 tensor 到多卡; PP 模式无此开销
    for stage in pipeline_stages:  # 各加速器顺序执行
      hidden = model_layer(hidden)
      # PP 天然支持 batch 内 prefix: 共享前缀仅存一份

  Step 2 - Token 生成循环:
    while not done:
      hidden = model_layer(hidden)
      logits = lm_head(hidden)
      next_token = sample(logits)

      # Segment Cache 写 KV: 连续地址直接索引
      kv_cache[seq_pos] = (key, value)

      # 若实际生成超出预分配 segment:
      if seq_pos > segment.max_len:
        if adjacent_free():         extend segment   # 策略1
        elif other_free():          append segment   # 策略2
        else:                       wait_list()      # 策略3

      if eos_token: break

  === 并发调度（多对一映射） ===
  加速器 0 上进程 A: 执行 pipeline stage 0 of batch_1 (stream A)
  加速器 0 上进程 B: 执行 pipeline stage 0 of batch_2 (stream B)
  # 两进程通过不同 stream 并发，GPU 资源持续利用
  # 无需跨进程 tensor 通信（vs TP 每层需 AllReduce）
  ```

  关键优势：PP 无需 NVLINK，在低规格/异构设备上 TP 通信开销 >50%，PP 零通信开销换取更高吞吐。Segment Cache 避免 small-block overhead，同时天然支持 prefix caching（共享前缀的多请求复用一个 segment）。

## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- 属于Serving调度的实现是什么？实验比较什么？
  实现包括两个层面的 Serving 调度创新：(1) **Load-Aware Thresholding in Expert Parallelism**：在 MoE 分布式推理中，EP 设备间的负载不均是限制效率的主要因素。本文提出基于设备负载动态调整 token-expert 计算丢弃率的机制，采用 step-down thresholding 策略——高负载设备使用更高的丢弃阈值（激进丢弃更多 token-expert 计算），低负载设备使用更低的丢弃阈值（保守保留更多计算）。具体实现：计算每个设备的 actual load / ideal balanced load 比值，若比值 > 1 则阈值设为预定义的最大值，若比值 < 1 则按比例降低阈值，确保所有设备以最小精度损失实现负载均衡。(2) **Soft Expert-Tensor Parallelism (S-ETP)**：从算法层面实现 TP-like 效果而非法仅系统层面修改框架。S-ETP 通过 partial transformation（expert partition + EP）替代传统的 Expert-Tensor Parallelism (ETP)，仅需 AlltoAll 通信（vs ETP 的 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll"），减少 kernel launch 和同步开销，提高 interconnect link utilization。实验比较：(a) ETP vs S-ETP 在不同 EP/TP 配置下的通信带宽（real-world 8×H20 测试 + ASTRA-SIM 模拟 NVL72 和 CloudMatrix384）；(b) load-aware thresholding 下的 speedup vs accuracy trade-off（1T-Drop、2T-Drop、2T-Drop+load-aware 三者对比）。

- 硬件平台是什么，配置是什么。
  8×NVIDIA H20 GPU 服务器节点（单机 8 H20），使用 PyTorch Distributed framework + NCCL backend。模拟环境：NVL72 (NVIDIA GB200, EP=9, TP=8)、CloudMatrix384 (CM384, EP=48, TP=8)，使用 ASTRA-SIM 模拟器进行大规模通信仿真。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**SGLang**（https://github.com/sgl-project/sglang），支持 Mixtral、OLMoE、DeepSeek 等 MoE 模型的高效分布式推理。论文修改/新增：(a) 在 SGLang 框架中实现 DualSparse-MoE inference system，包括 token-expert computation dropping（1T-Drop/2T-Drop）、load-aware thresholding 和 expert partition processing；(b) 实现 S-ETP 通信模式，将 ETP 的 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll" 通信模式简化为单 AlltoAll；(c) 实现 expert partition 的 preprocessing 和 inference 阶段的 neuron reconstruction 及 dual-threshold 控制逻辑；(d) 实现基于设备负载实时调整 drop threshold 的 load-aware 调度策略。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立的开源代码仓库，但基于开源 SGLang 框架实现。DualSparse-MoE 的 Load-Aware EP 推理全流程如下：

  ```
  Input: batch of tokens, model partitioned across N EP devices

  === Preprocessing (one-time, static) ===
  For each expert on each device:
    Importance profiling on calibration samples (MMLU)
    Sort neurons by importance, split into major/minor sub-experts
    Store the reconstructed expert weights

  === Inference Per-Layer Flow (with Load-Aware 2T-Drop) ===

  Step 1: Gating (each device)
    Input token hidden states x on device d
    Compute gating logits: l = x · W_g
    Top-K selection: {e_1, ..., e_K} = TopK(l, K)
    Normalize gating scores of selected experts

  Step 2: Load-Aware Threshold Adjustment (each device)
    // Communication: gather load info across EP devices
    load_d = count of tokens routed to experts on device d
    load_ideal = total_tokens / N_EP
    ratio_d = load_d / load_ideal

    For each token-expert pair on device d:
      if ratio_d > 1:  // overloaded device
        T_eff = T_max (预定义固定值)
      else:
        T_eff = T_base * ratio_d  // proportionally reduced

  Step 3: Dual-Threshold Token-Expert Dropping (each device)
    For each token-expert pair (t, e_j):
      s_norm = normalized gating score
      if s_norm < T_eff_major: skip computation  // T_major from 2T-Drop
      elif s_norm < T_eff_minor: compute only major sub-expert
      else: compute full expert (major + minor)

  Step 4: Expert Computation + AlltoAll
    Compute selected (sub-)expert FFN outputs locally
    AlltoAll: exchange token results back to original devices
    (S-ETP variant: single AlltoAll vs ETP's multi-stage communication)

  Step 5: Combine Results
    y = weighted sum of computed expert outputs by original gating scores
    Continue to next layer

  === Communication Patterns: ETP vs S-ETP ===
  ETP (Expert-Tensor Parallelism):
    AlltoAll → TP AllGather | ReduceScatter → AlltoAll
    Multiple kernel launches + synchronization barriers

  S-ETP (Soft Expert-Tensor Parallelism, 本文):
    AlltoAll (single operation)
    Expert partition via partial transformation handles TP splitting algorithmically
    Result: 3.0%-29.9% bandwidth improvement on real H20; 10.2%-80.4% on NVL72 simulation
  ```

  **S-ETP bandwidth improvement 实测结果**：EP=4, TP=2 on 8×H20：3.0%-29.9%；EP=2, TP=4 on 8×H20：9.2%-15.2%。模拟环境：NVL72 (EP=9, TP=8)：10.2%-80.4%；CloudMatrix384 (EP=48, TP=8)：9.9%-28.3%。

  **Load-Aware 最终结果**：DeepSeek-V2-Lite-Chat on 8×H20 with EP=8：2T-Drop + load-aware thresholding → 1.41× MoE module speedup、1.13× end-to-end speedup，仅 0.5% average accuracy loss。

## DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 DeepSpeed-MoE inference system，提供针对 MoE 模型的多 GPU 推理优化：**(1) Flexible Multi-Dimensional Parallelism**：对 expert 参数使用 expert parallelism + expert-slicing（tensor-slicing of experts），对 non-expert 参数使用 tensor-slicing + data parallelism，协同组合实现 trillions 参数模型扩展到数十/数百 GPU。**(2) Hierarchical All-to-All**：将全连接 all-to-all 拆分为两阶段（intra-node + inter-node）加数据布局变换，通信 hops 从 O(p) 降至 O(G+p/G)，对小 batch size 延迟敏感场景优化显著。**(3) Parallelism-Coordinated Communication**：当 tensor-slicing 与 expert parallelism 组合时，利用 tensor-slicing all-reduce 导致的数据复制，将 all-to-all 操作限制在同 tensor-slicing rank 的子集内，延迟从 O(p) 降至 O(p/L)，L 为 tensor-slicing degree。**(4) Optimized MoE Kernels**：gating 函数 kernel fusion + dense token-to-expert mapping 替代 sparse einsum，数据布局变换实现 token 排序/反排序。实验比较：(a) DeepSpeed-MoE vs PyTorch baseline（full-featured distributed PyTorch）在 52B MoE 模型上的延迟和吞吐（8→64 GPUs）；(b) 不同模型规模（107B→2T params）下的延迟和吞吐对比（up to 256 A100 GPUs）；(c) PR-MoE+MoS 进一步压缩后的延迟改进和最小 GPU 数量需求对比；(d) MoE vs quality-equivalent dense model 的推理延迟和成本对比（52B MoE vs 6.7B dense; 1.5T MoE vs 175B dense）。

- 硬件平台是什么，配置是什么。
  Azure ND A100 instances，最多 256 张 NVIDIA A100 GPU。节点内 8 张 GPU 通过 NVLink 互联，节点间网络使用 Mellanox InfiniBand。支持 Microsoft SCCL 优化的通信后端替代 NCCL。

- 开源Serving框架是什么。修改了什么。
  开源框架：DeepSpeed-MoE，作为 DeepSpeed 库的一部分（https://github.com/microsoft/DeepSpeed）。修改/新增：(a) 实现 expert parallelism + expert-slicing 协同调度；(b) 实现 Hierarchical All-to-All 通信原语，使用底层 NCCL P2P 操作 + CUDA kernel 进行数据布局变换；(c) 实现 Parallelism-Coordinated Communication 调度器，将 all-to-all 与 tensor-slicing all-reduce 联动优化；(d) 实现 multi-expert、multi-data parallelism 的灵活训练/推理并行策略。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源，代码位于 https://github.com/microsoft/DeepSpeed，提供 tutorials 和文档。

  **DeepSpeed-MoE Inference 全流程（以 1.3B+MoE-128, 52B params, 128 GPUs EP=128, TP=1 为例）**：
  ```
  Input: 一组 token 序列（batch of tokens）S 个 tokens

  === 初始化: 模型分区 ===
  Model Partitioning (128 GPUs):
    Non-expert params (Attention): Data-parallel replicas or Tensor-slicing across GPUs
    Expert params: 128 experts distributed, 1 expert per GPU (Expert Parallelism=128)
    
  === 逐 Token 推理流程 ===
  For each Transformer layer:
  
  Step 1: Attention (non-expert)
    For each GPU (tensor-slicing group or data-parallel):
      Q, K, V = Linear projections
      Attention computation (DeepSpeed inference optimized kernels)
      Output from attention block
    Communication: All-reduce IF tensor-slicing; none IF data-parallel only
    
  Step 2: MoE Gating + Token Routing (on GPU holding the current token)
    For each token t in batch:
      gate_logits = W_gate @ h_t                    // [E] per token, E=128
      expert_id[t] = argmax(Softmax(gate_logits))   // Top-1 expert selection
      // Build dense token-to-expert mapping table:
      // mapping[i] = token_id assigned to expert i
    
  Step 3: All-to-All Dispatch (Hierarchical)
    // Parallelism-Coordinated Optimization:
    // If TP=8 (each tensor-slicing group has 8 GPUs), 
    // all-to-all happens only within GPUs sharing same TP rank
    Intra-node All-to-All within each node (8 GPUs):
      For each GPU in node:
        tokens_for_local_experts = []
        tokens_for_remote_nodes = []
        // Route tokens: local expert → keep; remote → send
        NCCL P2P send/recv tokens to correct GPU within node
    
    Inter-node All-to-All:
      Data-layout transformation (CUDA kernel) → regroup tokens by target node
      NCCL P2P send/recv across nodes
      Data-layout transformation → regroup tokens by target GPU
    
  Step 4: Expert Computation (per GPU)
    For each expert e on this GPU (only 1 expert when EP=128):
      tokens_e = received_tokens_for_expert_e
      output_e = W2_e @ GeLU(W1_e @ tokens_e)       // Standard FFN
    // Expert-slicing alternative: split expert FFN across GPUs
    
  Step 5: All-to-All Combine (reverse dispatch)
    Intra-node all-to-all → tokens return to original GPU
    Inter-node all-to-all
    CUDA kernel: re-order tokens to original sequence order
    
    // Parallelism-Coordinated: if TP was used, 
    // add AllGather between TP ranks after combine
  
  Step 6: Residual connection + LayerNorm
    h = h + MoE_output
  ```

  关键性能特征：
  - Expert parallelism=128 时，每个 GPU 仅需加载 1/128 的 expert 参数，critical data path = 1.3B（仅 base dense model 大小）
  - 与 PyTorch baseline 相比：7.3x 延迟降低，7.3x 吞吐提升
  - Trillion-parameter MoE 模型推理延迟 <25ms
  - 比 quality-equivalent dense model（6.7B）：up to 2.4x faster（PR-MoE+MoS）
  - 比 quality-equivalent dense model（175B）：up to 4.5x faster, 9x cheaper（PR-MoE+MoS, 1.5T MoE）

## DeepSeek-V3 Technical Report

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 DeepSeek-V3 的生产级推理部署策略：**(1) Prefill-Decode 分离部署**：Prefill 阶段最小部署单元 4 nodes (32 GPUs)，TP4+SP+DP8 for attention, EP32 for MoE；Decode 阶段最小部署单元 40 nodes (320 GPUs)，TP4+SP+DP80, EP320。**(2) 冗余专家部署 (Redundant Experts)**：复制高负载 expert 并在节点内重新排列，prefill 阶段设 32 个冗余 expert，decode 阶段 64 个 GPU 负责冗余 + shared expert。**(3) Micro-batch 双流水线重叠**：prefill 阶段重叠 attention+MoE of micro-batch A 与 dispatch+combine of micro-batch B；decode 阶段重叠 attention of micro-batch A 与 dispatch+MoE+combine of micro-batch B。实验比较：DeepSeek-V3 vs DeepSeek-V2 的端到端生成速度（超过 2 倍），以及通过 MTP speculative decoding 实现 1.8x TPS 解码加速。

- 硬件平台是什么，配置是什么。
  H800 集群部署。节点内 NVLink 互联，节点间 InfiniBand 全互联。Prefill 部署 32 GPUs (4 nodes)，Decoding 部署 320 GPUs (40 nodes)。使用 IBGDA (GPUDirect Async) 降低延迟。

- 开源Serving框架是什么。修改了什么。
  论文未使用开源 Serving 框架，而是基于自研 HAI-LLM 框架进行推理部署。修改包括：(a) 实现 prefill-decode 分离部署架构；(b) 动态冗余专家部署策略，每 10 分钟更新一次高负载专家检测；(c) 自研 cross-node all-to-all communication kernels 用于 MoE dispatch/combine；(d) 正在探索 "dynamic redundancy" 策略（每个 GPU 部署 16 个 expert 但仅激活动态选择的 9 个）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  模型 checkpoint 开源在 https://github.com/deepseek-ai/DeepSeek-V3。Serving 部署框架细节未开源。

  **DeepSeek-V3 Prefill-Decoding 分离部署全流程**：
  ```
  Input: 用户请求（prompt + 生成参数 max_tokens）
  
  === Prefill 阶段 (4 nodes × 8 H800 = 32 GPUs) ===
  
  Step 1: 请求路由到 Prefill 集群
    Tokenizer (BBPE, vocab=128K) → token_ids
  
  Step 2: Attention 部分 (TP4 + SP, DP8)
    for each layer with MLA:
      h → c^{KV}=W^{DKV}@h (d_c=512, FP8, 需缓存)
      h → k^R=RoPE(W^{KR}@h) (d_h^R=64 per head, 需缓存)
      // Absorb W^{UK} into W^{UQ}, W^{UV} into W^O
      Compute attention with FlashAttention-style kernel
    Communication: TP all-reduce within 4 GPUs, DP all-reduce across 8 replicas
  
  Step 3: MoE 部分 (EP32, 32-way expert parallelism)
    for each MoE layer (layers 4-61):
      Gate: s_{i,t}=Sigmoid(h_t^T e_i), TopK(s_{i,t}+b_i, K_r=8)
      // Node-limited routing: max 4 nodes, avg 3.2 experts/node
      Dispatch: IB transfer → target node GPUs → NVLink forward to expert GPU
        (20 SMs, 10 channels, warp specialization)
      Expert FFN: FFN_i^{(s/r)}(h_t) for each selected expert
      Combine: NVLink reduction → IB transfer back
    Redundant expert deployment: 32 extra replicas of high-load experts
  
  Step 4: Micro-batch双流水线
    Micro-batch A: Attention + MoE computation
    Micro-batch B: All-to-all dispatch + combine (overlapped)
    → 通信完全隐藏在计算中
  
  Step 5: KV cache 传输到 Decode 集群
    c^{KV} (FP8) + k^R → IB → Decode 集群 GPU 内存
  
  === Decoding 阶段 (40 nodes × 8 H800 = 320 GPUs) ===
  
  Step 6: 自回归生成循环
    for each new token:
      // Shared expert treated as always-selected routed expert (9 experts activated)
      MoE: EP320, each GPU hosts 1 expert
        64 GPUs dedicated to redundant/shared experts
      Attention: TP4+SP, DP80
      All-to-all via direct point-to-point IB (IBGDA for low latency)
  
  Step 7: Micro-batch overlap (decode)
    Micro-batch A: Attention (attention dominates in decode)
    Micro-batch B: Dispatch + MoE + Combine (fewer SMs, memory-bound)
  
  Step 8: MTP speculative decoding (optional)
    MTP module predicts 2nd next token
    Acceptance rate: 85-90% → 1.8x TPS speedup
  
  Output: 生成 tokens → detokenizer → 返回用户
  ```

## DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 DeepSeek-V2 的推理部署优化：**(1) FP8 参数精度转换**，将模型参数转为 FP8 精度部署；**(2) KV cache 6-bit 量化**，将 MLA 的 latent KV cache 进一步压缩到平均每个元素 6 bits；**(3) 基于 vLLM 推理后端**进行高效 serving。实验比较：单节点 8×H800 上 DeepSeek-V2 的生成吞吐量 vs DeepSeek 67B（dense, 67B），以及 prompt input throughput。

- 硬件平台是什么，配置是什么。
  单节点配备 8 张 NVIDIA H800 GPU，节点内 NVLink + NVSwitch 互联，节点间 InfiniBand。推理部署使用 FP8 精度。

- 开源Serving框架是什么。修改了什么。
  论文未明确说明 Serving 框架的具体修改细节。论文提到使用 vLLM (Kwon et al., 2023) 作为 RL 训练阶段的推理后端，部署阶段的 serving 框架未详细说明。论文说明其训练框架为 HAI-LLM（High-flyer 内部开发），推理部署进行了 FP8 量化和 KV cache 6-bit 量化。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  模型 checkpoint 开源在 https://github.com/deepseek-ai/DeepSeek-V2，但 serving 部署框架细节未开源说明。

  **DeepSeek-V2 推理部署全流程（以单节点 8×H800 推理为例）**：
  ```
  输入：用户 prompt tokens [t1, t2, ..., tn]
  
  Step 1: Tokenization
    BBPE tokenizer (vocab=100K) → token_ids [id1, id2, ..., idn]
  
  Step 2: Prefill 阶段（8×H800, 专家并行 D=8, TP 不需要因为激活参数仅 21B）
    for each Transformer layer (l=1..60):
      // MLA Attention（使用改进版 FlashAttention-2）
      h → W^{DQ} → c^Q → W^{UQ} → q^C
      h → W^{DKV} → c^{KV}（压缩到 512 维，FP8 存储 → 512 bytes）
      h → W^{KR} → k^R（RoPE，64 维）
      // 吸收优化：W^{UK} ⊂ W^{UQ}, W^{UV} ⊂ W^O，无需显式计算 k^C, v^C
      FlashAttention-2(q, k, v) → attention output
      
      // DeepSeekMoE FFN
      2 共享专家（计算在本地设备）
      160 路由专家分布在 8 设备 → Top-6 选择，最多 3 设备
      all-to-all 通信传输 token hidden states
      Token-Dropping（推理阶段可选，不丢 token 则全算）
  
  Step 3: Decode 阶段（逐 token 生成）
    每个 decode step：
      - MLA 只需计算新 token 的 c^{KV} 并追加到 KV cache
      - KV cache per layer: (d_c + d_h^R) = 512+64 = 576 元素
      - 6-bit 量化后每层 KV cache ≈ 576*6/8 = 432 bytes/token
      - 60 层 × 432 bytes = ~25.9 KB per token in KV cache
      - vs DeepSeek 67B MHA: ~1.9M elements × 2 bytes = ~3.8 MB per token per layer × 60 ≈ 很多
  
  Step 4: 输出
    生成 token → 经 BBPE decoder → 文本输出
  ```

  **吞吐数据**：单节点 8×H800，DeepSeek-V2 生成吞吐 >50K tokens/s（5.76× DeepSeek 67B），prompt 输入吞吐 >100K tokens/s。
  **训练成本**：每 1T tokens，DeepSeek 67B 需要 300.6K GPU hours，DeepSeek-V2 仅需 172.8K GPU hours（节省 42.5%）。

## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **BrownoutServe** —— 一个面向 MoE LLM 的端到端 inference serving 框架，核心包含两大机制：(1) **United Experts**：通过知识蒸馏将多个 MoE expert 的知识合并到单个同参数规模的 united expert，减少推理时的 expert 访问次数；(2) **Dynamic Brownout Mechanism**（含 Brownout Approach 和 SLO-Aware Latency Control/SALC 算法）：在资源受限或突发流量时，动态将部分 token 路由到 united experts 处理，减少 expert 访问开销，同时通过 SALC 算法自适应调整 brownout threshold 以平衡延迟和精度。

  实验比较：
  - 吞吐量：BrownoutServe vs vLLM (non-fused) 和 vLLM (native/fused MoE)，在 ShareGPT 和 Alpaca 数据集上，不同 request rate 下持续 10 分钟
  - 精度：在不同 (way, threshold) 配置下的 accuracy loss，使用 PIQA、COPA、CEVAL、OBQA 四个 5-shot 任务
  - SLO 违规率：在突发流量场景下（250s trace，t=75s 时 RPS 翻倍），BrownoutServe vs vLLM 的 prefill/decoding 阶段 token-level SLO 违规率
  - 延迟 trace 分析：250s 内的 P90 prefill/decoding latency 变化轨迹
  - Threshold 自适应变化分析

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100-PCIE-40GB GPU（每卡 40GB），Intel Xeon Gold 6238 CPU。

- 开源Serving框架是什么。修改了什么。
  BrownoutServe 是**自研的定制 Serving 框架**，并非直接修改 vLLM 源码，而是用约 5.5k 行 Python 从 PyTorch 构建，同时**集成了 vLLM 中的多项优化技术**：PagedAttention（并进行了优化——将 block table 移到 GPU，block table 操作实现为 GPU kernel）、FlashAttention、ContinuousBatching/iteration-level scheduling。MoE 模块引入 brownout approach，MoE 算子使用 Triton 重写。

  **控制平面修改**：
  - **Scheduler**: 使用 FCFS 调度，当 engine 达到最大 batch 容量时多余请求进入等待队列；支持 streaming I/O，允许动态插入新请求和提前移除已完成请求
  - **SLO Analyzer**: 持续监控 TTFT 和 TPOT，运行 SALC 算法动态调整 brownout threshold
  - **Experts Loader**: 负责加载/卸载 united experts，更新 united experts 的 way 配置

  **数据平面修改**：
  - 集成 BrowoutMoE 模块（含 fused MoE 和 brownout routing）
  - 优化 PagedAttention：block table 移至 GPU，block table 操作实现为 CUDA kernel
  - MoE 算子全部使用 Triton 重写

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **开源**: https://github.com/beyondHJM/BrownoutServe（Apache-2.0 协议，约 32 commits），但预训练的 United Expert 权重需联系作者获取。

  **框架输入到硬件执行的全过程（以 Qwen1.5-MoE-A2.7B-Chat, partial-brownout, way=8, threshold=0.4 为例）**：

  1. **请求到达**: 用户请求通过 HTTP/gRPC 到达 Scheduler（FCFS），请求包含 prompt text 和 SLO 要求
  2. **Batch 组装**: Scheduler 从等待队列中取出请求组装 batch（max batch size=64），支持 ContinuousBatching——每 iteration 完成后从 batch 中移除已完成请求，加入新请求
  3. **Prefill 阶段**: 所有 prompt tokens → Embedding → Attention (FlashAttention + PagedAttention with GPU-side block table) → FFN → **BrownoutMoE**:
     - Gate 单元计算每个 token 对所有 60 个 experts 的 affinity score s_{i,t} = x_t^T · e_i
     - Top-K routing 选出每个 token 的 top experts
     - 统计每个 expert 的 token 数量，按降序排列
     - 根据 threshold=0.4，累计 token 数达到 40% 的 experts 进入 S1（由原 experts 处理），其余进入 S2（由 united experts 处理）
     - S1 tokens → 原 experts FFN（fused MoE kernel on GPU）→ 输出
     - S2 tokens → 按 way=8 分组（60 experts → 8 groups）→ 每组 tokens concat → 对应的 united expert FFN → 输出
     - 因部分 tokens 使用 united expert（参数固定在 GPU 显存中），减少 expert 访问次数 → 降低 latency
  4. **Decoding 阶段**: 逐 token 自回归生成，每个新 token 经过相同的 BrownoutMoE 流程
  5. **SLO Analyzer 反馈**: 每 iteration 后 SLO Analyzer 收集 P90 TTFT/TPOT latency。若 P90 latency > SLO → threshold × shrink_ratio (如 0.8) 降低更多 token 走 brownout；若 P90 latency < warning_line (SLO × warning_factor) → threshold + increment (如 +0.1) 提升精度
  6. **输出返回**: 生成的 token 流式返回给客户端


## Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Task-level MoE Sub-Network Extraction for Serving**，核心思想：通过 task-level routing 将 MoE 模型的 decoder 从 "每 token 动态选 expert" 改为 "每 task 静态选 expert"，使得推理时只需加载 task-specific sub-network（K 个 experts），而非全部 E 个 experts。这避免了 token-level MoE 因 decoder 参数过大（远超单加速器内存）而需要的模型并行和跨设备通信。同时避免了蒸馏（distillation only preserves 32% of BLEU gains）。在 WMT 30 language pairs 上 peak throughput 提升 1.87x，decoder 参数从 221M 降至 25M（6.3%）。在 200 language pairs 上 peak throughput 提升 2.6x，decoder 参数从 6.5B 降至 201M（1.6%），communication overhead 从 36% 降至 0.2%。

  实验比较：
  - Task-MoE vs Token-MoE 在不同 batch size 下的推理吞吐量（Figure 2, Figure 4）
  - Task-MoE vs Distillation (Token MoE → Transformer-Base student) 在 8 个语言对上的 BLEU（Table 2）
  - 推理通信开销对比：Task-MoE (0.0%-0.2%) vs Token-MoE (26.9%-36% of step time)
  - Hybrid 策略：Task-MoE 在 decoder only (encoder 用 Token routing) 效果最好

- 硬件平台是什么，配置是什么。
  Cloud TPU V3：WMT 实验吞吐量测量使用 32 cores，大规模实验使用 128 cores。解码 WMT14 En-De test set 测量吞吐量。

- 开源Serving框架是什么。修改了什么。
  论文基于 Google 内部 GShard 框架（Lepikhin et al. 2020, TensorFlow/Lingvo），未开源。论文未修改 Serving 框架本身——其创新在于**路由算法层面**使得 MoE 模型天然适合高效 serving：task-level routing 使 decoder 的 expert 选择与 token 无关，因此每个 task 仅需加载 task-specific experts，无需模型并行或 all-to-all 通信。

  **Token-MoE Serving 的问题**：
  - 每个 token 独立选 expert → 不同 token 可能路由到不同加速器上的不同 expert → 需要动态加载 experts（host↔device 通信）或模型并行（inter-device all-to-all 通信）
  - 自回归解码的每一步都需要跨设备通信 → 通信开销被乘以 decoding steps
  - 小 batch 时只有部分 expert 被激活 → 设备利用率低

  **Task-MoE Serving 的解决方案**：
  - 相同 task 的所有 token 路由到相同 experts → 每个 task 只需预加载 K=2 个 experts
  - 不同 task 可独立、并行地在不同加速器上解码
  - 无跨设备通信（Task-MoE: 0.0%-0.2% vs Token-MoE: 26.9%-36%）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源情况**: 论文未公开代码。Google Research 内部基于 GShard/TensorFlow 实现。

  **Task-MoE Serving 全过程（以 WMT En→Fr 翻译为例，32 TPU V3 cores）**：

  1. **输入/请求**: 源语言句子 (En) → 目标语言 (Fr) 的翻译请求到达。目标语言 Fr 作为 task 标识。

  2. **Task-Specific Sub-Network 选择**: 根据 task_id (Fr) 查询 task embedding table → Router(task_emb) → Top-2 experts（如 expert 5 和 expert 17）→ 仅将这 2 个 experts 的权重加载到 TPU 内存。对于 32 expert 模型，decoder 仅用 2/32 experts → 25M params vs 221M full decoder。

  3. **Encoder 前向（Token-level routing）**: 源句子 tokens 经 encoder 处理。Encoder 使用 token-level MoE → 每个 token 独立 router → 动态选择 top-2 experts → encoder 输出源语言表示。（Encoder 推理成本可忽略：decoder 每步时间是 encoder 的 200x）。

  4. **Decoder 前向（Task-level routing）**: 自回归解码每步：
     - 所有 decoder tokens 因同属 task "Fr" → router 返回相同 top-2 experts（expert 5, 17）
     - Decoder MoE layer: x_s → FFN_5(x_s) + FFN_17(x_s)（加权）
     - 无需 all-to-all 通信（expert 5 和 17 已在同一设备）
     - 无 expert 动态加载（task 开始时预加载，持续复用）

  5. **多 Task 并行**: 不同 task（如 En→Fr, En→De）可分配到不同 TPU cores 或 core groups，各自加载自己的 task-specific experts 子网络，完全独立解码，无跨 task 通信。

  6. **输出**: 解码完成 → 输出 Fr 翻译结果。

  **与蒸馏对比（Table 2）**: Distillation 将 Token-MoE (533M) 蒸馏到 Dense Transformer-Base (142M) → BLEU 26.9（仅保留 32% MoE 增益）。Task-MoE (Token encoder + Target decoder) → BLEU 29.0（保留 100% MoE 增益 + 额外 +2.1 BLEU），同时 decoder 参数量 25M << 142M。

  **通信开销对比**: Token-MoE 解码时 26.9%-36% step time 用于跨设备通信（WMT/Large-scale）。Task-MoE 解码时 0.0%-0.2% step time 用于通信（可忽略），因为所有 token 路由到相同设备上的相同 experts。

## Accelerating MoE Model Inference with Expert Sharding

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 MoEShard，一个基于 PyTorch 的 MoE 推理系统，通过 **expert tensor sharding** 替代传统 expert parallelism。核心思想：不再将完整的 expert 分配给不同 GPU，而是将每个 expert 的矩阵 W_i（列切分）和 W_o（行切分）切分到所有 GPU 上，每个 GPU 持有所有 expert 的 shard。所有 GPU 处理所有 token 的 partial computation，最后 pointwise 求和得到等价完整输出，实现 perfect load balancing，无论路由分布如何倾斜都不产生 GPU 空闲或 token dropping。

  实验比较：
  - **Per-layer latency**: MoEShard vs DeepSpeed-MoE (expert parallelism)，batch=250, seq=120, 128 experts → MoEShard 41.5~43.5ms vs DeepSpeed 177~180ms，达 4.25× 加速
  - **TTFT varying experts (8→256)**: MoEShard vs DeepSpeed，fixed batch=250, seq=120, skew (k_r=10%, α_r=0.6) → 峰值加速 6.45× (64 experts), 最低 2.39× (256 experts，因 DeepSpeed CF 丢 token)
  - **TTFT varying batch size (10→450)**: 128 experts → 小 batch(10)时 MoEShard 慢于 DeepSpeed, batch=100 起超越，batch=450 时达 6.24× 加速，接近线性增长
  - **Ablation (with/without MegaBlocks sparse MM)**: expert≥64 时 MegaBlocks 版更优；batch 变化时 MegaBlocks 版始终优于无 MegaBlocks 版

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 GPU（每卡 80GB HBM），NVLink 互联（双向 600 GiB/s），同一计算节点。CPU: AMD EPYC 7543 32-core @ 3.7GHz，PCIe 连接。CUDA 12.6。

- 开源Serving框架是什么。修改了什么。
  **Baseline 框架**: DeepSpeed-MoE (https://github.com/microsoft/DeepSpeed)，使用 expert parallelism (EP)，每 GPU 持有若干完整 expert，router 分配 token 后通过 all-to-all scatter/gather 通信路由 token 到对应 GPU 执行 expert 计算。DeepSpeed 默认使用 capacity factor (CF) 限制每 expert token 数（实验固定为 min(|E|, 50)），超限 token 被丢弃。

  **MoEShard 开源**: https://github.com/sacs-epfl/moe-inference，Python 3 + PyTorch 实现。

  **MoEShard 修改的核心逻辑**（相对于传统 EP）:
  1. **Expert Sharding 替代 Expert Parallelism**: 将每个 expert 的 W_i ∈ R^(h_i×h_o) 列切分为 |G| 份、W_o ∈ R^(h_o×h_i) 行切分为 |G| 份，每 GPU 持有所有 expert 的 shard。Forward pass 每 GPU 计算所有 token 对所有 expert 的 partial output (x · W_i^g · W_o^g)，最后 all-reduce 求和得到完整结果。
  2. **Token 全复制而非路由**: Step 2-3 中所有 GPU 互相发送 metadata（每 expert token 数）和全部 token（每 GPU 发送 ≈88 MiB for batch=250,seq=120,h=768），由 NVLink 高速带宽吸收（~0.15ms）。
  3. **Step 5 Gather + Pointwise Aggregation**: 各 GPU 将计算的 partial output 发回源 GPU，按元素求和恢复完整 token 输出。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **MoEShard 推理全过程（以 4 GPU, 128 expert Switch-Base encoder, batch=250, seq=120, h=768 为例）**:

  1. **输入/Token Routing**: batch tokens x ∈ R^(250×120×768) 进入 MoE block。Self-attention 层已在此前完成（复制在所有 GPU 上）。每 GPU 独立执行 ROUTER(x) → m_expert（token→expert 映射）。Router 为 Switch Transformer 的 top-1 gating（或自定义 skew 控制 router）。

  2. **Metadata Exchange**: 每 GPU 按 expert 分组 token → 统计 per-expert token count m_sizes (size=|E|=128) → all-to-all broadcast metadata。每 GPU 现在知道每张卡对每个 expert 有多少 token。

  3. **Token Scatter**: 每 GPU 将其所有 input tokens（concatenated）发送给所有其他 GPU（all-to-all scatter）。接收后组织为 W[g][e]：来自 GPU g 且目标 expert e 的 token 集合。以 batch=250,seq=120,h=768, 4B/element 计，每 GPU 发送 ≈88 MiB，NVLink 3.0 600 GiB/s 下耗时 ~0.15ms。

  4. **Expert Computation (Sharded)**: 每 GPU 遍历 expert e∈E，加载该 expert 在 *本 GPU rank* 的 shard: W_i^g (列 shard, h_i × h_o/|G|) 和 W_o^g (行 shard, h_o/|G| × h_i)。对分配给 expert e 的 token 执行 x · W_i^g · W_o^g → partial output y_g。优化：(a) 将同一 expert 的来自所有 GPU 的 token concatenate，减少 kernel launch 从 |E|×|G| 到 |E|；(b) 使用 MegaBlocks block-sparse MM 将所有 expert shard 计算融合为单次稀疏矩阵乘法。

  5. **Token Gather + Aggregation**: 每 GPU 将其计算的 W[g] (对 GPU g 原始 token 的 partial output) 发回 GPU g。GPU g 收到所有 partial outputs y_g 后 pointwise sum → x_final。等价于未经 sharding 的完整 expert 输出。

  6. **输出**: 聚合后的 token 作为当前 MoE block 输出，传递给下一个 transformer block。

  关键特性：所有 GPU 的计算量完全相等（均处理全部 token × 全部 expert shard），无论路由分布多倾斜，无 token dropping，无 GPU idle time。

## Accelerating Distributed MoE Training and Inference with Lina

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 Lina 系统，基于 DeepSpeed MoE (Microsoft) 构建，**在 Training 端**通过 tensor partitioning + micro-op priority scheduling 优先 all-to-all 避免与 allreduce 争抢网络带宽，并引入 expert packing 提升流水线效率；**在 Inference 端**通过 token-level expert selection pattern 估算 expert popularity，动态调度 expert-device 映射（两阶段调度：phase 1 基于估算预分配、phase 2 偏差过大时微调），以均衡设备负载和 all-to-all 带宽。
  
  实验比较：
  - **Training**: Lina vs DeepSpeed Baseline vs Tutel，比较 training step time、MoE layer time（前向/反向）、all-to-all time、GPU utilization、pipelining efficiency
  - **Inference**: Lina（完整版） vs Lina w/o estimation vs Lina w/o fine-tuning vs Baseline vs Ideal（perfectly balanced gating），比较 median/95%ile inference time、MoE layer time、tail all-to-all time
  
  Benchmark: Transformer-XL (24L, text generation on Enwik8), BERT-Large (12L, translation on WMT En-De), 以及 IMDB Reviews/Twitter sentiment analysis, WMT French/Russian 等泛化任务。

- 硬件平台是什么，配置是什么。
  4 个 worker 节点，每节点 4 块 NVIDIA Ampere A100 GPU（40GB HBM），节点间 100Gbps InfiniBand 互联。Training 使用与 expert 数量相等的 GPU（最多 16 GPUs，即 4 节点×4 GPU），Inference 同理。

- 开源Serving框架是什么。修改了什么。
  **开源框架**: DeepSpeed MoE (https://github.com/microsoft/DeepSpeed)，PyTorch 1.10 + CUDA 11 + NCCL 2.10。
  **论文代码开源情况**: 论文未明确提供独立开源仓库，约 7500 LoC 修改（C++/Python）基于 DeepSpeed MoE 和 PyTorch。
  
  **修改内容**:
  1. **Training 端——Communication Scheduler**：在 backward pass 中将 all-to-all 和 allreduce 分解为 micro-ops（tensor partitioning 到固定大小 chunk 如 30MB），使用 priority queue 调度，保证 all-to-all 优先获满带宽，allreduce micro-ops 仅在无 all-to-all 待处理时发射。修改 PyTorch DistributedDataParallel 的 bucketing 机制（不再 fuse gradients，而是分区每个梯度）。
  2. **Training 端——Expert Packing Coordinator**：动态调整每设备 expert 数量（powers of two，从 1→2→4），当 FFN micro-op 时间短于 all-to-all micro-op 时触发 packing；不足时使用 DRAM-offloading swap expert param。
  3. **Inference 端——Resource Scheduler**：在 device 0 上运行独立线程，管理 expert-device 映射。两阶段调度：Phase 1 基于 profiling 阶段采集的 token-level expert selection path patterns 估算 expert popularity（样本路径长度 l=3），按 `n_e = N × Σ P(e) / N_t` 分配设备，使用 first-fit-decreasing heuristic 打包；Phase 2 比较估计与实际 routing 结果，top-2k 不一致时重新计算分配。
  4. **Inference 端——All-to-All Coordination**：使用 unequal split all-to-all 避免多 process group 开销；无 token 导向某 device 时传 placeholder pointer。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **Lina Training 端执行全过程（以 16-expert Transformer-XL 一个 training step 为例）**:
  1. **输入**: batch tokens 分布在 16 GPU 上（data parallelism），每个 GPU 持 1 个 expert（expert parallelism degree=1，或 packing 后每 GPU 2-4 experts）
  2. **Forward Pass**: 非 MoE 层（Attention/LayerNorm）本地计算 → Gate 路由选择 top-2 experts → all-to-all dispatch tokens → Expert FFN 计算（如 packing=2，两 expert 串行执行）→ all-to-all combine → 输出 combine
  3. **Backward Pass**: 反向传播开始
     - **Commission Scheduler** 将 gradient tensor 按 30MB chunk 分为 micro-ops
     - **Priority Queue**: 当 all-to-all micro-op 到达时置顶；allreduce micro-op 仅在队列无 all-to-all 时发射
     - 避免了 all-to-all 与 allreduce 同时使用网络带宽（baseline 中两 CUDA stream 各自发射导致带宽均分）
  4. **Expert Packing**: 每 4 steps 检查 FFN vs all-to-all micro-op 时间比 → 若 FFN < all-to-all → 增倍 packing → 通过一次 synchronous all-to-all 交换 expert 参数 → 下次 iteration 生效
  5. **优化器步**: 所有 gradient allreduce 完成后 optimizer.step()

  **Lina Inference 端执行全过程（以 16-expert BERT-Large 推理一个 batch 为例）**:
  1. **输入**: 推理请求 batch tokens 分布在 16 GPU 上
  2. **Layer 1-3（warm-up）**: 标准 MoE 推理（Gate → All-to-All → Expert → All-to-All），不作调度（累积 expert selection path）
  3. **Layer 4+ —— Phase 1（预调度）**:
     - Device 0 Scheduler: 根据 profiled expert selection patterns `{Ψ}` 和当前 token sample paths（长度 l=3），估算下一层各 expert 的 popularity `n_e`
     - 估算信息 piggyback 在第一个 all-to-all 中发送到 device 0
     - Scheduler 计算新 expert-device 映射（first-fit-decreasing），popular expert 复制到多 device → unpopular 打包到少 device
     - 映射结果通过第二个 all-to-all 下发 → 各 device 从 host DRAM swap in 对应 expert 权重
  4. **Phase 2（微调）**:
     - Gate 执行后各 device 对比实际 routing vs 估算 → 通过 NCCL send 报告
     - 若 top-2k expert 一致（~77% cases）→ scheduler 广播 resume 信号 → 模型继续
     - 若不一致 → scheduler 重算 expert-device mapping → 广播新映射 → 模型 blocked 直到收到命令
  5. **Expert Computation**: 多 expert 的 device 串行执行各 expert FFN（每次 load 一个 expert 权重）
  6. **Unequal Split All-to-All**: combine 阶段按 device 实际 token 量发送，非均匀拆分
  7. **重复 3-6** 直至所有 MoE layer 完成

  **Baseline (DeepSpeed MoE) 对比**:
  - Training: backward pass 中 all-to-all (stream b) 与 allreduce (stream c) 独立发射无协调，网络带宽公平共享 → all-to-all 被延长 1.83x~4.14x
  - Inference: 所有 device 均匀持有 1 个 expert → popular expert 处理 token 量远多于 unpopular（最大 5.56x）→ 延迟尾部拖长



## A Survey on Mixture of Experts in Large Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  本论文为综述，不提供原始实验。它在系统层面（Section 5）对 MoE 系统的 Computation、Communication、Storage 三方面进行了系统性分类：
  - **Expert Parallelism 流程**（Fig 8a）：Gate Routing → Input Encode → All-to-All Dispatch → Expert Computation → All-to-All Combine → Output Decode
  - **混合并行策略**（Fig 8b-d）：Data+Expert+Tensor Parallelism、Data+Expert+Pipeline Parallelism、Expert+Tensor Parallelism
  - **计算优化（Section 5.1）**：动态专家放置（SE-MoE, Tutel, FlexMoE, SmartMoE）、动态影子专家策略（FasterMoE）、定制 GPU kernel（DeepSpeed-MoE, FastMoE, HetuMoE, Tutel）
  - **通信优化（Section 5.2）**：分层 All-to-All（DeepSpeed-MoE, HetuMoE, ScheMoE）、拓扑感知路由（FasterMoE, TA-MoE, SE-MoE）、专家亲和性预分配（ExFlow）、计算-通信流水线重叠（Tutel, FasterMoE, PipeMoE, MPipeMoE, Lancet）、架构解耦打破通信依赖（ScMoE, Arctic Dense-MoE hybrid）
  - **存储优化（Section 5.3）**：层级存储专家 offloading（SE-MoE, Pre-gated MoE, EdgeMoE: GPU HBM → CPU → SSD）、缓存/预取（expert selection forecasting + prefetching）、激活值内存优化（MPipeMoE: buffer sharing + recomputation/CPU offload）

- 硬件平台是什么，配置是什么。
  综述覆盖的硬件平台为多 GPU 分布式系统（NVIDIA GPU），通信通道涵盖 intra-node（PCIe, pre-4th-gen NVLink）和 inter-node（Ethernet, InfiniBand, 4th-gen NVLink）。

- 开源Serving框架是什么。修改了什么。
  **开源框架**（Table 4，截至2024年6月 Star 数）：
  - DeepSpeed-MoE（Microsoft, 33K stars）：在 DeepSpeed 上增加 MoE 支持（expert parallelism + 分层 All-to-All + 定制 GPU kernel）
  - ColossalAI/OpenMoE（38K stars）：MoE 训练与推理支持
  - Tutel（Microsoft, 672 stars）: 自适应并行策略切换 + 通信计算重叠
  - FastMoE（Tsinghua, 1.4K stars）: 定制GPU kernel + 通信优化
  - Fairseq-moe（Meta, 29K stars）：MoE 多语言训练
  - Megablocks（Stanford, 1.1K stars）：块稀疏 GPU kernel
  - ScatterMoE（Mila, 140 stars）：ParallelLinear 散操作
  - SE-MoE（Baidu, 21K stars, 基于 PaddlePaddle）：动态放置 + 通信 + 存储
  - HetuMoE（PKU, 236 stars）：定制 kernel + 分层 All-to-All
  - Mesh-TensorFlow（Google, 1.6K stars）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文关联资源仓库：https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs。
  各框架开源链接见 Table 4。

  **Expert Parallelism 端到端执行过程（以 DeepSpeed-MoE 为例）**：
  1. **输入**：batch tokens 经由数据并行分布在 N 个 GPU 上，每 GPU 持有部分 expert + 全部非 expert 参数（Attention, LayerNorm, Router 等）
  2. **Attention + Router 计算**（本地 GPU 执行，无通信）
  3. **Input Encode**：每个 GPU 将需要发送到同一 expert 的 tokens 聚合为连续内存块
  4. **All-to-All Dispatch**：将编码后的 token 数据发送到持有对应 expert 的 GPU（跨 GPU 通信）
  5. **Expert Computation**：各 GPU 对接收到的 tokens 执行本地 FFN 计算
  6. **All-to-All Combine**：将 expert 输出传回原始 token 所在的 GPU
  7. **Output Decode**：恢复原始 token 排序，加权合并 expert 输出
  8. **继续下一层**：重复 2-7

  **关键优化技术示例**：
  - **分层 All-to-All（DeepSpeed-MoE）**：优先利用高带宽 intra-node 通道（NVLink），减少低带宽 inter-node 数据交换
  - **拓扑感知路由（TA-MoE）**：将 token 优先路由到同节点的 expert，minimize cross-node 通信
  - **Lancet 重叠**：将非 MoE 计算（Attention, LayerNorm）插入 All-to-All 通信缝隙，延长重叠窗口

## EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  EPS-MoE 提出了一种用于 MoE 推理的 Expert Pipeline Scheduler，包含三个核心模块：(1) 并行策略选择——对 Attention 块使用 DP 或 TP、对 MoE 块使用 EP，支持 TP+TP、DP+EP、TP+EP 三种模式的理论分析；(2) Expert Pipeline Scheduler——基于水平切分输入张量和按专家切分权重的方法，实现 load-aware 动态切换 GroupGemm 和 DenseGemm；(3) 计算与通信重叠——在 kernel 级别将 GEMM 计算与 all2all 通信 pipeline 并行，并通过控制 SM 数量优化重叠效率。
  
  实验比较：
  - Baseline: vLLM 的 TP+TP 实现（已 kernel tuning 至最佳性能）
  - EPS-MoE 的 DP+EP（DeepSeekV2）或 TP+EP（Mixtral/DBRX）模式
  - 测试 PN=1（仅 GEMM 切换，无 pipeline）、PN=2/5/8/20（pipeline 调度）
  - 消融实验：GEMM 切换 vs 重叠的贡献分解、FP8 通信影响、SM 数量控制必要性、不同输出维度影响、最优 PN 选择

- 硬件平台是什么，配置是什么。
  - DeepSeekV2: 8xH800-80GB SXM
  - Mixtral 8x7B: 4xH800-80GB SXM  
  - DBRX: 8xH800-80GB SXM
  - Snowflake Arctic: 8xH800-80GB SXM
  - 消融实验: 4xH800-80GB SXM（单层 Gate/Up/Down GEMM 测试）
  - 分析用: NVIDIA A100-80GB SXM（NVLink，用于通信和计算 profiling）
  - 矩阵计算库：cutlass、cublas

- 开源Serving框架是什么。修改了什么。
  基于 vLLM 框架集成 EPS-MoE。具体修改包括：
  1. **并行策略替换**：将 vLLM 原生的 TP+TP 替换为 TP+EP（对 MHA/GQA/MQA Attention）或 DP+EP（对 MLA Attention）。将 ncclAllReduce 分解为 ncclReduceScatter + all2all（dispatch 阶段）和 all2all + ncclAllGather（combine 阶段），减少 MoE 模型通信量。
  2. **Expert Pipeline Scheduler**：实现水平切分（行切分）输入张量、按专家切分 MoE 权重的调度策略，每次只传输当前专家组所需的 token 到对应设备。动态根据负载切换 GroupGemm（小 token 数时更优）和 DenseGemm（大 token 数时，如 m≥4096 时更优）。
  3. **SM 数量控制**：通过限制 GEMM kernel 占用的 SM 数量，为通信 kernel 留出 SM 资源，实现计算与通信的完美重叠。最优配置为 GEMM 116 SM + 通信 16 SM（H800 共 132 SM）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供开源代码链接（作者来自美团，arXiv:2410.12247）。论文说明了与 vLLM 的集成方式。
  
  **Serving 框架使用流程（以 Mixtral 8x7B 推理为例）**：
  
  1. **输入阶段**：用户请求 batch 到达，tokenized 为输入序列。batch 中的 token 被分发到各 GPU 设备。
  
  2. **Attention 块（TP 模式）**：Attention 权重按 TP 切分在各设备上。每设备持有 1/D 的 KVCache 和权重。输入 activation 经 TP 通信（ncclAllReduce，在 EPS-MoE 中分解为 ReduceScatter+AllGather+Flux 融合）完成 Attention 计算。
  
  3. **MoE 块（EP 模式 + Expert Pipeline）**：
     - **Router/Gating**：每个 token 经 gating 网络计算 top-k 专家选择。
     - **Dispatch 通信**：通过 all2all 将每个 token 发送到其选中的专家所在设备。EPS-MoE 使用 ReduceScatter+all2all 替代 ncclAllReduce，通信量更小。
     - **Expert Pipeline Scheduler**：
       - 输入张量按行水平切分（split rows），权重按专家切分。
       - 将专家分组为 N 个 pipeline stage，每个 stage 顺序提交一组专家的 GEMM 计算。
       - 若 m ≥ 4096（compute-bound prefill 阶段），切换到 DenseGemm（cublas）；若 m < 2048（memory-bound decode 阶段），使用 GroupGemm（cutlass）。
       - GEMM 计算与下一轮 all2all 通信重叠：通过限制 GEMM 占用 116 SM，通信占用 16 SM，两者在不同 SM 上并行执行。
     - **Combine 通信**：all2all + ncclAllGather 聚合各 token 的专家输出。
  
  4. **输出阶段**：MoE FFN 输出与 Attention 输出经 residual 连接，经 LayerNorm 后进入下一层。最终经 LM Head 输出 token 概率分布。
  
  **全流程数据路径**：
  `Input tokens → Tokenization → [Attention(TP) → Dispatch(all2all) → Expert FFN(GroupGemm/DenseGemm pipeline) → Combine(all2all+AllGather)] × L layers → LM Head → Output tokens`
  
  **关键加速器**：H800 SXM GPU (NVLink 互联)，每 GPU 132 SM，通过 SM 分区实现 GEMM 和通信的并行调度。

## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  本论文是综述，不提供原始实验。它系统性地分类和比较了 MoE 推理优化中的系统级技术（Section 4）：
  - **Expert Parallelism（4.1）**：
    - 并行策略设计：Tutel（动态切换并行策略）、Alpa（intra/inter-operator并行分类）、DeepSpeed-TED（data+tensor+expert混合并行）、BaGuaLu（MoDa策略）、SmartMoE（异构感知混合并行）、MPMoE（pipeline并行优化）
    - 负载均衡：Prophet（性能建模+greedy搜索）、MoE-Prediction（预测专家负载）、Lazarus（专家副本分配）、FlexMoE（细粒度复制）、Brainstorm（历史分配数据）、Lynx（减少batch中的激活专家）、BaseLayers（线性分配问题）、MoE-ECR（expert选择token）
    - All-to-All通信优化：Tutel/HetuMoE/DeepSpeed-MoE（分层all-to-all）、TA-MoE/DeepSpeed-TED（数据压缩）、Janus（以数据为中心移动expert）、ExFlow（减少all-to-all操作数）、Aurora（有序token传输避免带宽竞争）、LocMoE/Parm（inter-node转intra-node）
    - 任务调度：ScMoE（shortcut架构解耦通信）、HiDup（microbatch重叠通信与计算）、MoESys（2D预取+融合通信）、ScheMoE（模块化+自适应调度）、PipeMoE（性能建模+pipeline调度）、EPS-MoE（动态kernel选择重叠FFN与通信）
  - **Expert Offloading（4.2）**：
    - 预取：Mixtral-Offloading/AdapMoE/HOBBIT（基于当前门控输入预测下层expert）、Pre-gated MoE（预门控结构）、EdgeMoE（预测表）、DyNN-Offload（pilot模型预测）、MoE-Infinity（请求级频率追踪）、ProMoE（学习型预测器滑窗预取）、ExpertFlow/SiDA（一次性预测所有expert）
    - 缓存：LRU策略（Mixtral-Offloading等）、LFU策略（MoE-Infinity）、静态重要性配置（Fiddler）、动态缓存更新（SwapMoE）、动态缓存大小（AdapMoE）、多维策略+混合精度（HOBBIT的LRU+LFU+LHU）、cache-aware routing（CacheMoE）
    - 加载优化：低精度expert加载（EdgeMoE、HOBBIT）、自适应跳过不重要expert（AdapMoE）
    - CPU辅助：Fiddler（CPU执行expert计算）、HOBBIT（CPU处理低精度expert）、MoE-Lightning（CPU-GPU-I/O流水线）
  
  表5-10汇总了各系统的加速比、内存节省、GPU利用率等性能指标。

- 硬件平台是什么，配置是什么。
  - 云集群场景：多GPU服务器（如NVIDIA A100/H100），支持分布式训练和推理
  - 边缘设备场景：单GPU内存受限设备，部分使用Jetson Orin等嵌入平台
  - 论文未统一规定硬件配置，各被综述系统使用各自的实验配置

- 开源Serving框架是什么。修改了什么。
  **开源框架基础**（Table 10统计）：
  - PyTorch（12个并行系统+4个offloading系统基于此）
  - DeepSpeed（9个并行系统+1个offloading系统）
  - Transformers（5个并行系统+7个offloading系统）
  - 其他：Fairseq、Llama.cpp、vLLM、FasterTransformer各1-2个系统
  
  **主要修改方向**：
  - DeepSpeed-MoE：在DeepSpeed上增加MoE支持，包括分层all-to-all、expert parallelism
  - Tutel：在Fairseq/PyTorch上实现自适应并行策略切换
  - vLLM-based：利用PagedAttention管理expert参数的KV cache

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源情况**：综述关联仓库 https://github.com/MoE-Inf/awesome-moe-inference/ 汇总各系统的开源链接。主要开源系统包括：
  - DeepSpeed-MoE: https://github.com/microsoft/DeepSpeed
  - Tutel: https://github.com/microsoft/tutel
  - FasterMoE: https://github.com/laekov/fastermoe
  - vLLM: https://github.com/vllm-project/vllm

  **Expert Offloading框架执行全过程（以Mixtral-8x7B + HOBBIT为例）**：
  1. 输入：用户请求tokens到达GPU
  2. 非expert参数（Attention、Router、LayerNorm）常驻GPU显存
  3. Expert Cache：GPU显存中缓存高频expert（FP16高精度）
  4. Router计算：对每个token计算θ = Softmax(R(x))，选出top-K expert
  5. Cache查询：检查所需expert是否在GPU expert cache中
  6. Cache Miss处理：
     a. 计算该expert的importance score（基于gate输出）
     b. 若score低于阈值 → 从CPU/SSD加载低精度版本（INT4）
     c. 若score高于阈值 → 加载高精度版本（FP16）
     d. 同时触发下层expert预取（基于当前gate输出预测下层）
  7. Expert计算：GPU对已加载expert执行FFN计算
  8. 加权聚合：合并expert输出
  9. 输出：生成下一个token

  **Expert Parallelism框架执行全过程（以DeepSpeed-MoE为例）**：
  1. 输入：batch tokens分布在各GPU上
  2. 每GPU持有部分expert + 全部非expert参数
  3. Attention + Router计算（本地）
  4. All-to-All通信：根据Router输出将token分发到持有对应expert的GPU
  5. Expert计算（本地GPU对分配到的token执行FFN）
  6. All-to-All通信：将expert输出传回原始GPU
  7. 继续下一层


## BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **BuddyMoE Runtime System**——在 llama.cpp serving 框架中集成 buddy expert substitution 的完整推理运行时。核心修改是在 router 和 expert execution 之间插入 buddy replacement 中间层，拦截 prefetch miss 场景并用功能相似的 GPU-resident buddy expert 替代 CPU-resident expert。

  实验比较：
  - **Baseline 1: Original (On-demand)**：缺失 expert 从 CPU 同步加载 → 保持最大精度但吞吐受限
  - **Baseline 2: Random Replacement**：缺失 expert 随机替换为 GPU-resident expert → 吞吐高但精度极低
  - **BuddyMoE 配置消融**：三种 cache rate (c=0.375/0.50/0.75) × 不同 τ/|B|/ρ 组合
  - **PCIe 带宽对比**：Base vs BuddyMoE 的 PCIe read 带宽使用
  - **评估指标**：ARC-Easy/ARC-Challenge accuracy, tokens/s, PCIe bandwidth

- 硬件平台是什么，配置是什么。
  单节点系统：1× NVIDIA A100 GPU（PCIe 接口），Intel Xeon Platinum 8457C CPU。使用单 GPU + CPU offloading 配置，模拟资源受限部署场景。GPU memory 仅保留部分 expert（cache rate c=0.375/0.50/0.75），其余 expert offload 到 CPU memory。

- 开源Serving框架是什么。修改了什么。
  **开源框架**: llama.cpp（https://github.com/ggerganov/llama.cpp），使用其 CUDA backend 进行 GPU kernel 执行和原生 CPU offloading 机制管理 expert 的 GPU↔CPU 传输。

  **BuddyMoE 修改内容**:
  1. **Buddy Replacement Runtime 中间层**：在 router（gating network）和 expert execution 之间插入 buddy substitution logic。当 router 选择的 expert 不在 GPU 内存时，不触发同步 CPU→GPU 传输，而是查找 buddy list 中的 GPU-resident 替代 expert。
  2. **三阶段决策 Pipeline**：TAE gate（token 级敏感度评估）→ Distribution gate（batch 级 CPU residency 检查）→ Buddy selection（优先级排序 + 拓扑感知），每个阶段都有独立的安全阀。
  3. **Custom CUDA Kernel**：实现并行 buddy substitution——每个 CUDA thread block 处理一个 token，block 内 thread 并行检查 k 个 expert 的 GPU residency 状态，使用 atomic CAS 操作保证 uniqueness constraint。
  4. **Buddy Profile 集成**：预计算的 B_ℓ(i;α) lookup table 序列化后随模型 checkpoint 加载，O(K_max · E_ℓ) 存储开销可忽略。
  5. **与现有 Prefetching 互补**：BuddyMoE 不替代 prefetching，而是补充——当 prefetch 命中时正常工作，当 prefetch 失败时用 buddy replacement 避免同步 stall。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **开源情况**：论文未提供 BuddyMoE 公开代码仓库，web 搜索未发现 GitHub 链接。BuddyMoE 基于 llama.cpp 实现，修改其推理 pipeline 加入 buddy replacement 机制。

  **BuddyMoE Serving 框架输入到硬件执行全过程（以 DeepSeek-V2-Lite, 64 experts/layer, cache rate=0.75, τ=0.95, |B|=16, ρ=3 为例）**：

  1. **请求到达**：用户推理请求（prompt tokens）进入 llama.cpp 推理引擎。所有非-expert 参数（Attention weights, LayerNorm, Embedding, Router）常驻 GPU 显存。

  2. **Expert Cache 管理**：GPU expert cache 保留 c=75% experts（按历史激活频率选择），剩余 25% experts offload 到 CPU memory。Expert 的 GPU/CPU residency 由布尔 mask M ∈ {true,false}^64 维护。

  3. **Router 前向（GPU 执行）**：token embedding x → Router: logits = W_g · x → TopK(top-6) → Softmax → 选出 6 个 experts → 检查 M 中每个 expert 的 GPU residency。

  4. **Prefetching 尝试**：Prefetching 机制（如 MoE-Infinity 式基于历史激活频率的预测）尝试提前将下一层需要的 experts 从 CPU 预取到 GPU。同时在上层计算时异步传输。

  5. **Buddy Replacement 决策（GPU CUDA kernel）**：
     - TAE gate：计算 token 的 routing entropy → 若 TAE ≤ 0.95（peaked routing，对替换敏感）→ 禁止替换 → 若 expert 在 CPU → 触发同步加载（penalty）
     - Distribution gate：检查 batch 级 CPU residency ratio δ → 若 δ ≥ β（过多 expert 在 CPU）→ 禁止替换
     - Buddy selection：对每个 CPU-resident expert，查询 B_ℓ 中 GPU-resident buddy → 按 Ψ 优先级排序 → 最多 ρ=3 次替换 → 原子更新 S'
     - 若无合适 buddy → fallback: 触发 prefetch 原始 expert 或 skip expert

  6. **Expert FFN 计算（GPU kernel）**：对替换后的 expert set S' 中所有 GPU-resident experts → 执行标准 FFN GEMM (x · W_gate → GeLU → W_up → multiply → W_down) → 加权求和出输出

  7. **PCIe 带宽节省**：BuddyMoE 的 buddy replacement 完全在 GPU memory 内完成（不触发 CPU→GPU 传输），PCIe read 带宽使用比 base method 减少约 20%（Figure 8）。

  8. **输出**：每层 MoE 输出传递至下一层 → 最终 token 输出 → 流式返回给客户端。

  **关键对比——Expert Miss 处理路径差异**：

  | 场景 | Baseline (On-demand) | Baseline (Prefetch Miss) | BuddyMoE |
  |------|---------------------|-------------------------|----------|
  | Expert 在 GPU | 直接计算 (~0ms) | 直接计算 (~0ms) | 直接计算 (~0ms) |
  | Expert 在 CPU, 有 GPU buddy | — | CPU→GPU 传输 (~9-10ms) | Buddy 替代 (~0ms + minimal accuracy loss) |
  | Expert 在 CPU, 无 GPU buddy | CPU→GPU 传输 (~9-10ms) | CPU→GPU 传输 (~9-10ms) | Fallback: CPU→GPU 传输 或 skip |
  | Expert 不在 cache, prefetch 命中 | — | 异步预取已隐藏 (~0ms) | 异步预取已隐藏 (~0ms) |

  BuddyMoE 的核心价值在于：当 cache miss 发生时（prefetch 失败 + expert 在 CPU），不等待完整的同步传输，而是用 GPU-resident buddy 即时替代，将 ~10ms 的传输延迟降低为 ~0ms 的查找延迟。

## CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **CoServe**——一个面向 CoE（Collaboration-of-Experts）模型推理的 heterogeneous CPU+GPU serving 系统，专为内存受限的边缘设备设计。核心包含三大技术：(1) **Dependency-aware Request Scheduling**：利用专家依赖关系，将依赖相同专家的请求在队列中分组排列，减少专家切换频率；同时动态分配请求到不同 CPU/GPU executor 队列中平衡负载；(2) **Dependency-aware Expert Management**：两阶段专家淘汰策略——优先淘汰无前置依赖的后续专家，不足时按预评估的使用概率淘汰低概率专家；(3) **Offline Profiler**：通过 microbenchmarks 自动确定最优显存分配和 executor 数量，生成专家性能矩阵和路由规则供在线调度使用。

  实验比较：
  - 吞吐量：CoServe Best/CoServe Casual vs Samba-CoE (FCFS+LRU)、Samba-CoE FIFO、Samba-CoE Parallel，在 NUMA (RTX3080Ti) 和 UMA (Apple M2) 设备上，4 个任务 (A1/A2/B1/B2) 对比
  - 专家切换次数：各方法在 4 个任务上的 expert switching 次数
  - 消融实验：CoServe None → CoServe EM (Expert Management) → CoServe EM+RA (Request Arranging) → CoServe (完整)，吞吐量和专家切换次数分解
  - Executor 数量消融：不同 GPU/CPU executor 组合 (G3C1, G4C1 等)
  - 内存分配搜索：sliding decay window 方法从 CDF 中选择最优加载 expert 数，对比吞吐量变化
  - 调度开销分析：request scheduling latency vs inference latency vs pre-scheduled inference，expert management 时间占比

- 硬件平台是什么，配置是什么。
  NUMA 设备：NVIDIA RTX3080Ti (12GB GPU Memory) + Intel Xeon Silver 4214R CPU (16GB Memory) + MICRON MTFDDAK480TDS SSD (530 MB/s)。
  UMA 设备：Apple M2 (24GB 统一内存) + APPLE SSD AP0512Z (~3000 MB/s)。

- 开源Serving框架是什么。修改了什么。
  CoServe 为**自研 PyTorch Serving 系统**，并非基于现有开源 Serving 框架（如 vLLM、SGLang）修改。其 baseline Samba-CoE 未公开代码。CoServe 在 PyTorch 基础上构建了完整的 CoE serving 运行时。

  **CoServe 架构三阶段**：
  
  **Offline Phase 修改**：
  - **Performance Profiler**：运行 microbenchmarks 为每个专家（同架构专家仅 profile 一次）测量最大 batch size、执行延迟 (K, B)、加载延迟、显存占用，生成 performance matrix
  - **Routing Rules & Usage Probabilities**：从用户提供的路由规则或小样本数据集计算每个专家的使用概率，生成 CDF 曲线
  - **Memory Allocation Optimizer**：在 GPU 上通过 sliding decay window 方法搜索最优加载专家数量，在 CPU 上使用最大 batch size 策略

  **System Initialization Phase**：
  - **Executor Creator**：根据 offline profile 结果创建指定数量的 GPU/CPU inference executors
  - **Expert Initializer**：按使用概率降序，round-robin 将专家加载到各 executor 的 model pool 中，直到内存用尽

  **Online Phase 修改**：
  - **Dependency-aware Request Scheduler**：
    - Prediction: 估算添加新请求到各 executor 队列后的额外推理延迟 = 执行延迟 (K × requests_in_batch + B) + 专家切换延迟 (0 或 expert loading time)
    - Assigning: 选择使当前各队列最大总推理时间最小的队列；平局时选择额外延迟最小的队列
    - Arranging: 将新请求排在同专家请求之后，实现请求分组
    - Batch Splitter: 根据当前可用内存和最大 batch size 将同专家请求组拆分为多个 batch
  - **Dependency-aware Expert Manager**：
    - Stage 1: 优先淘汰无前置依赖的后续专家（按显存降序淘汰直至足够）
    - Stage 2: 按使用概率升序淘汰（预评估概率，非 LRU 历史统计）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  
  **开源情况**: 论文未公开代码仓库，web 搜索未发现 GitHub 链接。CoServe 使用 PyTorch 实现，评估使用自建的电路板缺陷检测 CoE 模型（论文明确指出"no publicly available CoE model exists for testing"）。

  **CoServe Serving 框架输入到硬件执行全过程（以 NUMA RTX3080Ti, 3 GPU executors + 1 CPU executor, Task A1 为例）**：

  1. **Offline Profiling（部署前执行一次）**：
     - 运行 ResNet101/YOLOv5 专家的 microbenchmarks → 获取最大 batch size (如 GPU batch=6)、执行延迟 K/B 参数、加载延迟、显存占用
     - 计算各专家使用概率（从电路板组件分布可知各组件被检测的概率 → 对应专家使用概率）
     - Sliding decay window 搜索 → 确定 GPU 加载 35 个专家、CPU 加载若干专家

  2. **System Initialization**：
     - Executor Creator 创建 3 个 GPU executors + 1 个 CPU executor
     - Expert Initializer 按使用概率降序 round-robin 分配专家：GPU Executor 1 加载专家 1,4,7...；GPU Executor 2 加载专家 2,5,8...；GPU Executor 3 加载专家 3,6,9...
     - 每个 executor 维护独立 model pool（GPU 显存 ∩ CPU 内存）

  3. **请求到达**：电路板组件图像连续输入（每 4ms 一个图像），每个图像携带目标组件类型信息，路由规则确定需要的分类专家（可能还有目标检测专家）。

  4. **Dependency-aware Request Scheduling**：
     - Scheduler 预测每个 executor 队列的额外推理延迟：
       - 若目标 expert 已在 model pool 中 → 切换延迟 = 0
       - 若队列中已有同 expert 请求 → 切换延迟 = 0（专家在前序请求处理期间加载）
       - 否则 → 切换延迟 = expert loading time
       - 执行延迟 = K × (batch requests) + B
     - 选择使最大队列总时间最小化的 executor → 将请求排在同专家请求之后
     - Batch Splitter 按当前可用内存和最大 batch size 拆分请求组

  5. **Expert Switching（如需要）**：
     - 若所需 expert 不在 model pool → Expert Manager 执行两阶段淘汰：
       - Stage 1: 找无前置依赖的后续专家 → 按显存降序逐一淘汰
       - Stage 2: 仍不足 → 按使用概率升序淘汰
     - 从 SSD/CPU memory 加载新 expert 到 GPU model pool

  6. **Inference Execution**：
     - GPU executor: batch 图像 → ResNet101/YOLOv5 expert FFN → 分类/检测结果
     - CPU executor: 并行处理低优先级 batch（使用 CPU 上的 expert）
     - 多 executor 并行执行，专家切换与推理重叠

  7. **输出返回**：缺陷检测结果（组件类型、缺陷类型、对齐点、焊接方向）返回给产线控制系统。

  **关键对比——CoServe vs Samba-CoE 请求处理差异**：

  | 阶段 | Samba-CoE | CoServe |
  |------|-----------|---------|
  | 请求调度 | FCFS，无重排序 | Dependency-aware: 同 expert 请求分组 + 负载均衡分配 |
  | 专家管理 | LRU（仅历史统计） | 两阶段: 依赖感知 + 使用概率 |
  | 内存分配 | 静态/手动 | Offline profiler + sliding decay window 自动搜索 |
  | 并行度 | 单 executor 或 round-robin 多 executor | 多 executor + 请求级动态分配 |

  **性能提升来源**：Dependency-aware scheduling 将同专家请求集中处理 → 一次加载服务多个请求 → 减少 expert switching；Dependency-aware eviction 更准确预测未来使用 → 减少不必要切换。


## Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  在Megatron-LM分布式推理框架中修改MoE层的token路由和调度逻辑，通过在All-to-All通信前施加expert容量约束（Token Drop）和扩展本地候选集（Expanded Drop）来缓解expert parallelism下的"Straggler Effect"——即高负载expert完成计算慢导致低负载expert和GPU空闲等待同步barrier的问题。
  实验比较：(a) 不同capacity factor γ下各模型单MoE层的加速比（Figure 4）及端到端加速比（Figure 5）；(b) 不同expert-per-GPU配置对加速效果的影响——Mixtral 1-2E/GPU时效果最优（1.85-1.87×），而OLMoE 8E/GPU时加速减弱；(c) 推理延迟分解分析（Figure 6）：expert computation、permutation、communication各阶段时间随γ变化；(d) Device-Level vs Expert-Level容量约束的端到端speedup对比（Qwen3-MoE：1.31× vs 1.23×, γ=1.0/1.5）；(e) 不同workload（batch size 1K-8K, prompt length 0.1K-4K）下的speedup（Table 10）。

- 硬件平台是什么，配置是什么。
  8× NVIDIA H20 GPU。分布式推理策略：8-way Data Parallelism (DP) + 8-way Expert Parallelism (EP)，通过Megatron-LM框架编排。输入batch配置为batch size 8K、sequence length 512，模拟高吞吐实时serving场景。

- 开源Serving框架是什么。修改了什么。
  开源Serving框架：**Megatron-LM**（Shoeybi et al., 2019），用于实现expert parallelism + data parallelism的分布式MoE推理。
  修改内容：在MoE层的forward流程中，在Gate/Router计算之后、All-to-All token dispatch之前插入容量感知逻辑：
  1. **Token Drop**：Router计算gating scores → 根据expert capacity C=γN̄和gating scores对每expert做top-cap token选择 → 超载expert的剩余token被丢弃（score不参与后续dispatch和FFN计算）
  2. **Expanded Drop**：Router计算gating scores → 在top-k基础上扩展候选集为top-k+m（m=本地设备expert数）→ 逐expert应用capacity constraint → 保留的token经All-to-All dispatch到持有对应expert的GPU → expert FFN计算 → All-to-All combine
  3. **Device-Level变体**：将约束粒度从per-expert放宽到per-device aggregat

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE。修改基于Megatron-LM的MoE分布式推理流程。

  **全链路执行过程（以Mixtral-8×7B-Instruct, 8×H20 GPU, 8-way EP + 8-way DP, γ=1.5, batch 8K × seq 512 为例）**：

  1. **输入阶段**：8K个token组成的batch（batch_size=8K, seq_len=512）→ 均匀分配到8个DP group → 每个GPU持有1000 token × 512 seq 的输入tensor [1000, 512, d_model]

  2. **Self-Attention**（各GPU独立）：标准Multi-Head Attention → 输出hidden states [1000, 512, d_model]。Attention层非MoE，所有GPU均持有完整参数副本（DP复制）。

  3. **MoE Gate/Router**（各GPU独立）：hidden states → Gate Linear: W_g ∈ R^{d×8} → softmax → TopK(k=2, dim=-1) → gating_scores [1000×512, 8], topk_indices [1000×512, 2]

  4. **Capacity-Aware Token Drop/Expanded Drop**（本地GPU执行，论文插入点）：
     - Token Drop: scores × topk_mask → 每expert取top-cap=γ×(N×k)/E个token（dim=0 topk）→ 超载expert的低分token被mask
     - Expanded Drop: topk_idx ∪ local_expert_ids → 扩展候选掩码 → 逐expert top-cap筛选
     - 输出final_map [N, 8]标注每个token最终路由到哪些expert

  5. **All-to-All Token Dispatch**（跨GPU通信）：根据final_map中expert→GPU映射，通过NCCL All-to-All将token发送到持有对应expert的GPU。因capacity constraint减少了超载expert的token数，此次通信数据量减小。

  6. **Expert FFN Computation**（各GPU本地）：收到token的各GPU执行expert FFN：x → W_gate [d, d_ff] → GeLU → W_up [d_ff, d_ff] → × W_down [d_ff, d] → 输出。Mixtral每GPU 1-2个expert，各expert处理的token数因capacity constraint更均衡→ 减少GPU空闲等待。

  7. **All-to-All Combine**（跨GPU通信）：FFN输出通过All-to-All返回原token所在GPU。

  8. **输出Merge**：gate score加权求和各expert输出 → residual add → 输出至下一Transformer层。

  9. **端到端**：重复步骤2-8共32层（Mixtral-8×7B有32层，MoE替换alternate layers的FFN）→ final LM head → token预测。

  **关键性能影响（Figure 6 latency breakdown）**：
  - 无capacity constraint时：expert computation + permutation + communication占主导，gate processing耗时可忽略
  - Token Drop/Expanded Drop后：expert computation显著减少（因丢弃超载expert token），permutation和communication时间也相应减少
  - Expanded Drop扩展到跨设备global experts时communication增加（需传输扩展token）→ 论文因此限制扩展仅在本地设备内

  **加速比受expert-per-GPU配置影响的核心原因**：EP下每GPU托管n_l个expert，总load为n_l个expert的token数之和。若n_l大（如OLMoE 8E/GPU），单个straggler expert load占总load比例小，capacity constraint减少的load比例也小→加速效果削弱。若n_l小（如Mixtral 1-2E/GPU），straggler expert load占比大→容量约束效果显著→1.85-1.87×加速。因此论文建议分配更多GPU做expert分布以增强效果。

## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **SpecMoEOff**，在 SGLang 上构建的 MoE offloading serving 系统，通过 speculative decoding 增大每次前向 workload 来隐藏 offloading 延迟。核心 Serving 调度设计：(1) **Target Model Execution Engine**——基于 MoE-Lightning 的 CPU-GPU 流水线架构，batch 拆分为两个 micro-batch 交替执行 GPU Other1 → CPU Attention → GPU Other2 → GPU MoE，同时异步预取下一层 expert weights；(2) **Draft Model Execution Engine**——draft model KV cache 按 batch 维度切分为 GPU Part 和 CPU Part，两部分 attention 并行执行后统一在 GPU 做 FFN，动态调整 GPU/CPU 分离比例；(3) **Hyperparameter Optimizer**——自动搜索最优 batch size、micro-batch size、draft token 数量 k、内存管理策略和 execution strategy；(4) **Memory Manager**——管理 GPU HBM 和 CPU DRAM 的 KV cache 和 expert cache 分配。

  实验比较：
  - SpecMoEOff vs DeepSpeed-ZeRO-Inference vs MoE-Lightning 的端到端吞吐量和解码吞吐量
  - A30 vs 4090D 硬件环境下的性能 (不同 CPU memory 限制)
  - APPS vs CNN/DailyMail 数据集 (不同输入长度和 acceptance rate)
  - 不同输出长度 (128/256/512/1024) 下的 decode throughput
  - Micro-benchmark: varying draft length, input/output len, CPU/GPU memory

- 硬件平台是什么，配置是什么。
  A30: NVIDIA A30 GPU (165 TFLOPS) + Intel Xeon Gold 6426Y CPU, 250 GB CPU memory, CPU-GPU 25 GB/s。
  4090D: NVIDIA 4090D GPU (83 TFLOPS) + Intel Xeon Gold 5418Y CPU, 190 GB CPU memory, CPU-GPU 23 GB/s。

- 开源Serving框架是什么。修改了什么。
  基于 **SGLang** [33]，采纳 MoE-Lightning [6] 的 FFN/expert cache 设计，增加 20,000+ 行 Python/C++/CUDA。

  **修改内容**：
  1. **Speculative Decoding Pipeline 集成**：在 SGLang 的 decoding 循环中插入 draft model 生成 + target model 验证的双阶段执行流程
  2. **CPU-GPU Pipeline 编排**：target model 端将 batch 拆分为两个 micro-batch，交替执行 GPU Other1/CPU Attention/GPU Other2/GPU MoE，同时异步预取下一层 expert weights；使用分离的 CUDA Streams 管理 GPU 计算、expert 加载、activation 加载和 offloading
  3. **Draft Model Execution 分离**：draft model KV cache 按 batch 维度分片，GPU Part 全 GPU 执行，CPU Part 的 attention 在 CPU 计算后 hidden states 传回 GPU 做 FFN
  4. **Pin Memory + Dynamic Allocation**：使用 pin memory 减少 CPU→GPU 传输开销，动态内存分配避免内存碎片
  5. **Hyperparameter Optimizer**：自动搜索最优 (b, m, k, S_memory, S_execution)，凸优化预决定 b/m/S 参数，profiling estimator + DAG 模拟估计不同 k 的吞吐量，选择最优 k

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未公开独立开源仓库。基于 SGLang 构建。

  **SpecMoEOff Serving 框架输入到硬件执行全过程（以 Mixtral-8x7B + EAGLE draft, batch b 拆分为 2 micro-batches 为例）**：

  1. **请求到达与 Batch 组装**：b 个请求到达 → Scheduler 组装 batch → Hyperparameter Optimizer 决定 m (micro-batch size= b/2)、k (draft tokens)、expert cache 策略

  2. **Prefill 阶段**：chunk global batch 为 micro-batches → 每层加载 expert 参数到 GPU → 迭代 micro-batches → offload KV cache 和 hidden states 到 CPU DRAM。Draft model 所有参数放在 GPU HBM，执行 GPU-based prefill。

  3. **Decoding 阶段（Target Model Execution）**：
     - microbatch 1: GPU Other1 (LayerNorm, residual) → CPU Attention (Intel MKL chunked attention, KV cache from CPU DRAM) → GPU Other2 (router, etc.) → GPU MoE (expert weights 从 CPU DRAM 加载)
     - microbatch 2: 与 microbatch 1 交错执行（CPU Attention of microbatch 2 与 GPU MoE of microbatch 1 重叠）
     - 下一层 expert weights 在当前层计算期间通过独立 CUDA Stream 异步传输

  4. **Draft Model Execution**：
     - GPU Part requests: attention + FFN 全在 GPU（KV cache 在 GPU HBM）
     - CPU Part requests: attention 在 CPU 计算 → hidden states 传回 GPU → FFN 在 GPU 执行
     - 两部分并行执行 → 迭代 k 次生成 k 个 draft tokens
     - 动态调整：初始阶段更多在 GPU，序列变长后部分迁移至 CPU；请求完成后动态回迁

  5. **验证与同步**：draft tokens 与 original tokens 拼接 → target model 一次性前向验证 → 确定接受的 token 数量 a(k) → 更新 KV cache → 下一 iteration

  6. **Hyperparameter 动态调整**：随 sequence length 增长动态调整 k——生成初期 k 较大，后期减小；请求完成后增大 k

  7. **输出返回**：流式返回生成的 tokens


## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 Comet 在 **Megatron-LM** 中集成 fine-grained communication-computation fused kernel 替代默认 MoE layer 实现。Comet 提供 Python API 无缝接入 Megatron-LM 的 forward pass 流程——用户仅需将原有 MoE layer 的 expert 计算和 all-to-all 通信替换为 Comet 的 fused kernel 调用。Comet 支持所有 Megatron-LM 的混合并行策略（EP + TP + DP），并与 Megatron-LM 的 pipeline 并行兼容。

  实验比较（Figure 9，端到端 MoE 模型延迟）：
  - **Models**: Mixtral 8x7B (E=8, topk=2, N=4096, K=14336), Qwen2-MoE-2.7B (E=64, topk=4, N=2048, K=1408), Phi-3.5-MoE (E=16, topk=2, N=4096, K=6400)
  - **Baselines**: Megatron-Cutlass, Megatron-TE (Transformer Engine), FasterMoE, Tutel
  - **各种 parallelism (EP×TP)**: Mixtral: 8×1/4×2/2×4; Qwen2: 8×1/4×2/8×2/8×4 (不同 M); Phi: 8×1/8×2/4×4/8×4 (不同 M)
  - **End-to-end latency reduction**: Comet vs Megatron-Cutlass -34.1%, vs Megatron-TE -42.6%, vs FasterMoE -44.4%, vs Tutel -31.8%
  - **生产部署**: 已部署到超过万卡 GPU 的生产集群，累计节省数百万 GPU 小时

- 硬件平台是什么，配置是什么。
  **H800 集群**: 8× NVIDIA H800 GPU (80GB HBM)，NVLink 互联。CUDA 12.3, NVSHMEM 2.11, PyTorch 2.4.0。
  **L20 集群**: 8× NVIDIA L20 GPU (46GB)，PCIe 桥互联，GPU-to-GPU 带宽约 25 GB/s。

- 开源Serving框架是什么。修改了什么。
  **开源框架**: **Megatron-LM** (git-hash 6dbe4c)，用于实现 expert parallelism + tensor parallelism + data parallelism 的分布式 MoE 训练/推理。

  **Comet 修改内容（~12k lines C++/CUDA + 2k lines Python）**:
  
  1. **MoE Layer 替换**: 将 Megatron-LM 中默认的 MoE forward 流程（Router → All-to-All dispatch → Expert FFN → All-to-All combine）替换为 Comet 的 fused kernel。Comet 提供 Python API:
     ```python
     # Megatron-LM original MoE layer:
     #   token_permutation → alltoall_dispatch → expert_gemm → alltoall_combine → token_unpermutation
     
     # Comet replacement:
     #   Layer0: NVSHMEM receive (fine-grained) + GroupGEMM (tile-rescheduled) in fused kernel
     #   Layer1: GroupGEMM (column-wise) + topk-reduce + NVSHMEM send in fused kernel
     ```
  
  2. **Shared Tensor Buffer 管理**: 在每个 GPU 上通过 NVSHMEM 分配 shared memory buffer（大小 = 2×M×N bytes for BF16/FP16），作为 layer0 和 layer1 的共享缓冲区。该 buffer 跨所有 MoE layers 和 experts 全局复用，内存开销可忽略（M=4096 时 Mixtral 仅 32MB, Qwen2 仅 16MB）。
  
  3. **Parallelism 兼容**: 
     - Expert Parallelism (EP): expert 分布在不同 GPU → NVSHMEM 跨 GPU fine-grained 读写 token
     - Tensor Parallelism (TP): expert 权重沿 hidden 维度分片 → GroupGEMM tile reschedule 消除 weight switching overhead
     - TP < W 时 Megatron-LM 对非 MoE 层启用 Data Parallelism → Comet 仅修改 MoE 层，与 attention 层的 DP 兼容

  4. **Adaptive Kernel Selection**: 预编译内核库含多个 (n^c, n^p) 变体 → 部署前 profile 最优配置 → 运行时根据 M 和 parallelism 查表选择 kernel。

  5. **Production Integration**: 在生产环境 Megatron-LM 中，Comet 替换了 MoE 层的 forward/backward 实现，支持 training 和 inference。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源**: https://github.com/bytedance/flux（Project Page）。Comet 将开源。

  **Comet + Megatron-LM 端到端 MoE 执行全过程（以 Mixtral 8x7B, EP=8, TP=1, DP=1, M=4096, 一个 MoE layer 完整 forward 为例）**:

  **=== 框架层面执行流程 ===**
  
  1. **输入阶段**: batch tokens [M, N] = [4096, 4096] → 8 GPU DP=1 复制（各 GPU 持有 M/W = 512 tokens）→ Embedding → 32 Transformer layers，其中 MoE layers（alternate layers）使用 Comet

  2. **Self-Attention**（各 GPU 独立，标准 Megatron-LM）: FlashAttention on all 512 tokens → 输出 hidden states [512, 4096]

  3. **MoE Router**（各 GPU 独立）: hidden states → Gate Linear W_g[4096, 8] → Softmax → TopK(k=2) → routing_map (每个 token → top-2 experts)

  4. **MoE Layer0 - Comet Fused Kernel (NVSHMEM Receive + GroupGEMM)**:
     - Comet Python API 调用: `comet.moe_layer0_forward(hidden_states, routing_map, expert_weights)`
     - **Shared tensor 分配**: NVSHMEM buffer [M×topk, N] = [8192, 4096]，全局复用
     - **Dependency resolving**: shared tensor 沿 M 维度分解，tokens 按 source rank 排序
     - **Fused kernel launch**: 同时包含通信 TB（NVSHMEM get 拉取 remote tokens）和计算 TB（CUTLASS GroupGEMM 处理已就绪的 tiles）
     - **Tile 调度**: local token tiles 优先计算 → remote token tiles 延后（等待 NVSHMEM 完成）
     - 输出: expert FFN layer0 结果 [M×topk, K] = [8192, 14336]

  5. **MoE Layer1 - Comet Fused Kernel (Column-wise GroupGEMM + Reduce + NVSHMEM Send)**:
     - Comet Python API 调用: `comet.moe_layer1_forward(layer0_output, routing_map, expert_weights)`
     - **Shared tensor 分配**: 复用 layer0 的 NVSHMEM buffer
     - **Column-wise GEMM**: 所有 expert 并行计算第 col_block 列 → T^N 列完成后立即 top-K reduce → NVSHMEM write 回 source rank
     - 后续 col_blocks 计算与 reduce/通信重叠
     - 输出: 返回各 token source rank 的 MoE 输出 [M, N] = [4096, 4096]

  6. **Residual Add + Next Layer**: MoE 输出 + attention 输出 → LayerNorm → 下一 transformer layer

  **=== 与 Megatron-LM baseline 的关键差异 ===**
  | 阶段 | Megatron-LM Baseline | Comet |
  |------|---------------------|-------|
  | Token dispatch | NCCL all-to-all (coarse, 完整大 tensor) | NVSHMEM get (fine, token-level, fused in kernel) |
  | Expert FFN | Sequential per-expert GEMM kernel launches | Fused GroupGEMM + tile-rescheduled |
  | Token combine | NCCL all-to-all | NVSHMEM write fused with column-wise GEMM |
  | 通信-计算重叠 | 无（顺序执行） | Fine-grained overlap (hide 86.5% comm) |
  | Host scheduling | 每步多次 kernel launch from CPU | 单 fused kernel launch, kernel 内调度 |


## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Context-Aware Expert Placement Module**，在 GPU-NDP 异构系统上动态决定每层哪些 experts 驻留 GPU HBM（FP16 全精度），哪些 experts 驻留 CXL-NDP 设备（量化低精度）。核心机制：(1) 在 prefill 阶段收集每 expert 的激活频率 $P_{l,e}$ 和累计路由评分 $W_{l,e}$；(2) 计算归一化重要性分数 $S_{l,e} = \alpha\widetilde{P}_{l,e} + (1-\alpha)\widetilde{W}_{l,e}$；(3) 按 $S_{l,e}$ 降序选择每层 top-K experts 迁移至 GPU，其余保留在 NDP；(4) placement 仅执行一次（prefill 后），decoding 阶段不再迁移。将传统 GPU-NDP 系统以"Parameter Movement"为代价的 expert offloading 转化为以 "Activation Movement" 为核心的 NDP 近数据执行。

  实验比较：
  - vs **MoNDE** [18] (context-agnostic GPU-NDP expert offloading)：Ours-3bit 端到端 6.6-8.3× speedup，Ours-2bit 7.9-10.6×
  - vs **HOBBIT** [31] (GPU-only mixed-precision offloading)：Ours-2bit 达 18-19× speedup
  - Decoding throughput：Ours-3bit 8.7×, Ours-2bit 11.2× (Mixtral-8×7B)
  - NDP 侧 latency reduction：Ours-3bit ~5×, Ours-2bit ~8×

- 硬件平台是什么，配置是什么。
  系统：1× H100 GPU (80GB HBM3, 132 SM, 989.4 TFLOP/s) + 1× DDR-based CXL-NDP device (512 GB DDR, 512 GB/s bandwidth, 64×(4×4) systolic arrays @ 1 GHz)。PCIe Gen4 ×16 互联。

- 开源Serving框架是什么。修改了什么。
  论文未基于现有开源 Serving 框架（如 vLLM/SGLang）修改。系统评价使用 Ramulator [19] 模拟 NDP 设备，并自行实现 MoE 推理 pipeline。Baseline MoNDE 为 GPU-NDP MoE 系统，论文与其在同一模拟环境下对比。

  **相对于 MoNDE（context-agnostic）的核心修改**：
  1. **Prefill 统计注入**：在 MoE forward pass 的 prefill 阶段，每层 Gate/Router 计算后额外收集 $(P_{l,e}, W_{l,e})$，通过轻量级累加器实现（metadata 开销可忽略）。
  2. **单次 Expert Placement 调度**：MoNDE 使用 on-demand swapping 或 static placement——experts 在 GPU↔NDP 间动态迁移或固定分配，导致频繁迁移开销和带宽争用。本论文改为 prefill-guided once-per-sequence placement，消除 decoding 期间的 expert migration。
  3. **Hot/Cold 动态识别**：MoNDE 的 hot/cold 分类基于全局历史频率统计，忽略 context dependence。本论文使用 per-sequence prefill 统计做动态识别，捕捉不同输入序列的 expert 激活变化。
  4. **GPU-NDP 计算重叠**：GPU 执行其 hot experts 的 FFN 计算时，NDP 并行执行 cold experts 的量化计算，两者 overlap 最大化 pipeline 效率。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未公开独立代码仓库。NDP 模拟基于 Ramulator [19] (https://github.com/CMU-SAFARI/ramulator)。量化使用 GPTQ [9]。

  **Context-Aware GPU-NDP MoE Serving 全过程（以 Mixtral-8×7B, K=4 GPU/4 NDP experts/layer, Ours-3bit, 一个推理请求为例）**：

  1. **请求到达**：用户 prompt tokens 到达系统 → 非-expert 参数（Attention, Router, LayerNorm, shared params）常驻 GPU HBM。
  
  2. **Prefill 阶段（含统计收集）**：
     - 所有 prompt tokens 经 GPU 执行 Attention (FlashAttention on H100) → Router: logits = Softmax(W_g · x) → TopK(k=2) → 每层 8 experts 中选出 top-2
     - **统计收集**（论文创新点）：每层 l，维护 counter array [E=8]：
       - $P_{l,e}$ += 1（若 expert e 被任何 token 选中）
       - $W_{l,e}$ += routing_score（累计门控输出的 softmax 权重）
     - Expert FFN 计算：prefill 期间所有 experts 仍在 GPU 执行（因 prefill tokens 多，expert 激活均匀）
     - 每层输出传递至下一层，统计累加器一同传递

  3. **Expert Importance 计算**（prefill 结束后，解码前）：
     - 每层归一化：$\widetilde{P}_{l,e} = P_{l,e} / \sum_e P_{l,e}$, $\widetilde{W}_{l,e} = W_{l,e} / \sum_e W_{l,e}$
     - 重要性：$S_{l,e} = 0.5 \times \widetilde{P}_{l,e} + 0.5 \times \widetilde{W}_{l,e}$
     - 每层按 $S_{l,e}$ 降序 → top-4 experts → $\mathcal{H}_l$ (GPU, FP16)
     - 其余 4 experts → $\mathcal{C}_l$ (NDP, 由 Bitwidth Selector 分配 1-4 bit)

  4. **一次性 Expert 迁移**（仅在 prefill 后执行一次）：
     - GPU→NDP：$\mathcal{C}_l$ 中原本在 GPU 的 experts 的量化权重（pre-cached 1/2/3/4-bit GPTQ replicas）通过 PCIe → NDP memory。仅传输量化权重（如 3-bit：~45.1B × 3/8 ≈ 16.9 GB / 32 layers × 4 experts/layer ≈ 2.1 GB per layer），远小于全精度传输。
     - NDP→GPU：若 $\mathcal{H}_l$ 中有 expert 原在 NDP，其 FP16 权重 → GPU HBM。
     - 此后 decoding 阶段 zero migration。

  5. **Decoding 阶段（GPU-NDP 重叠执行）**：
     - 每个 decoding step，token x_t 经 GPU Router → top-2 experts 选择
     - **Case 1: 两个均在 GPU** → GPU FFN 直接计算 (FP16 GEMM on H100 tensor cores) → 输出
     - **Case 2: 一个 GPU + 一个 NDP** → GPU 计算 hot expert FFN 同时，activation x_t 通过 PCIe → NDP device → NDP 用指定的 b_{l,e} bitwidth 量化权重执行 FFN（64×(4×4) systolic arrays 并行）→ NDP 输出 activation 通过 PCIe → GPU → 加权求和
     - **Case 3: 两个均在 NDP** → x_t 经 PCIe → NDP → 两个 cold experts 依次执行 → 两个 output activations 经 PCIe → GPU → 求和
     - GPU 和 NDP 的计算在 per-layer 粒度实现 pipeline overlap：GPU 计算本层 hot experts 时，NDP 已开始下层 cold experts 的量化计算

  6. **关键优势——Activation Movement vs Parameter Movement**：
     - Baseline (MoNDE on-demand): 每次需要 cold expert → expert weight (FP16, ~45.1B/8/32×K MB per expert) 从 NDP → GPU → 大参数传输
     - Ours (prefill-guided): 仅传输 activation x_t (4096-dim FP16 = 8KB per expert per token) 从 GPU → NDP 或 NDP → GPU → 小激活传输
     - 单 decoding step 数据移动量对比：~数百 MB (parameter) vs ~数 KB (activation)，减少约 10^4-10^5×

  7. **输出返回**：autoregressive 生成完成的 tokens 流式返回客户端。

## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  论文将 DS-MoE-6B 模型部署到 vLLM 开源 serving 框架中进行吞吐量和延迟 benchmark，与 Mistral-7B、DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B 进行比较。实验比较在相同 GPU 硬件（A100-80GB / H100-80GB）和相同场景（1000 input tokens + 1000 output tokens）下的 requests/sec（Throughput）和 tokens/sec（TPS）。论文未修改 vLLM 框架本身，而是利用 vLLM 的现有能力进行标准 serving 部署和 benchmark。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80GB GPU 和 H100-80GB GPU，GPU 内存利用率设置为 0.9。

- 开源Serving框架是什么。修改了什么。
  使用 vLLM (Kwon et al. 2023) 作为部署框架，未修改框架本身。另外使用 HuggingFace Transformers (Wolf et al. 2020) 进行 latency 和 input token throughput 测量。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文代码未开源。vLLM 部署流程（基于论文 Table 7 描述）：
  
  1. **模型加载**：将 DS-MoE-6B（6.5B total params, FP16, ~12.6 GiB GPU memory）加载到 vLLM 的 LLMEngine。vLLM 使用 PagedAttention 管理 KV cache 内存，0.9 GPU memory utilization。
  
  2. **请求输入**：continuous batching 接收请求流，每个请求 1000 input + 1000 output tokens。
  
  3. **Prefill 阶段**：vLLM 将 1000 input tokens 作为 batch 送入模型 forward pass。Self-attention 层使用 dense inference（torch.nn；因 DS-MoE 的 attention 层 sparsity < 40%，密集计算更快）。MLP 层使用 sparse inference，通过 SimpleMoE 的 ParallelLinear 实现 top-K expert 选择。针对 batch 中不同 token 可能选择不同 expert 的情况，使用 Threshold-TopK 策略（先统计平均激活 expert 数，再统一选 K 个）。
  
  4. **Decode 阶段**：autoregressive 生成 1000 output tokens，每步仅处理新 token + KV cache。KV cache 用 vLLM PagedAttention 管理（block-based 分页），保证 memory fragmentation 最低。
  
  5. **性能指标**：Throughput = requests completed per second。TPS = total tokens processed (input + output) per second。测量结果：A100-80GB 上 DS-MoE-6B 吞吐 2.00 req/s, TPS 3992.8；H100-80GB 上 2.30 req/s, TPS 4603.9。分别相比 Mistral-7B 加速 1.86× (A100) / 1.64× (H100)，相比 Qwen1.5-MoE-A2.7B 加速 1.50× (A100) / 1.27× (H100)。

  另外在 HuggingFace Transformers 上测量 latency 和 input token throughput：
  - Latency：batch=64 sentences, 2000 tokens each, generate 20 tokens。DS-MoE-3B: 3.68s (vs Dense-3B 4.28s, 1.16× speedup); DS-MoE-6B: 5.75s (vs Dense-6B 8.58s, 1.49× speedup)。
  - Input TPS：seq_len=256, max batch size fit in GPU memory。DS-MoE-3B: 61515.9 (vs Dense-3B 40854.5, 1.51×); DS-MoE-6B: 35046.7 (vs Dense-6B 18354.2, 1.91×)。

  6. **扩展模型 Serving 分析**：为模拟更大 scale 下的性能，扩展模型到 10B/14B/19B 级别（Dense-10B/14B/19B, SMoE-17B/25B/34B (2× MLP params), DS-MoE-10B/14B/19B）。在 computation-bounded 场景（prefill, input throughput）和 I/O-bounded 场景（decode, output throughput）分别测量。DS-MoE 在两个场景均优于 SMoE（因 DS-MoE total params 更少，GPU memory 占用更低，batch size 可以更大），在 computation-bounded 场景显著优于 Dense（因 active params 少，计算量少）。

## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- 属于Serving调度的实现是什么？实验比较什么？
  MELD 在推理阶段基于 Punica + vLLM 构建多 LoRA query 系统，支持在单 GPU 上同时 serving 一个 base LLM model 和最多 200 个 LoRA weights（即 experts），动态为 incoming queries 生成和切换 expert 而无显著计算效率损失。实验比较了 MELD、JellyFish(13B) 和 Mixtral(8×7B) 在 4×3090 和 1×3090 配置下的推理吞吐量和模型处理时间。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 3090（24GB VRAM）。两种配置：(1) 4×3090 GPU (vLLM)；(2) 1×3090 GPU (vLLM)。单机 256GB RAM，Intel Xeon Gold 5320 CPU @2.20GHz。

- 开源Serving框架是什么。修改了什么。
  使用 vLLM (Kwon et al. 2023) 作为 serving 框架，结合 Punica (Chen et al. 2023) 的多 LoRA serving 能力。论文未修改框架本身，而是提出了 **动态 LoRA 切换（Dynamic LoRA Switch）** 技术：传统 MoE serving 需要将多个 LoRA merge 到 base model 中（耗时），MELD 避免 merge 操作，仅加载和 concatenate 多个 LoRA 权重，显著降低 I/O 开销。MELD 的 model process time 比 JellyFish 快 10×，比 Mixtral 快 30×。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  代码开源：https://github.com/authurlord/MELD。使用 Punica + vLLM 架构部署。

  **多 LoRA Serving 流程**（从 query 输入到 GPU 输出的全过程）：

  1. **Query 到达与序列化**：Incoming DP query q 经过 serializer 转换为统一 dict 格式（包含 tuple 内容、table title、column header 等元数据），并添加 task-specific prompt instruction。

  2. **Router 调度（CPU 端）**：Query embedding 通过 M_RAG（fine-tuned sentence-bert）编码为 emb_q。Router network N 计算 softmax(W_N · emb_q)，选择 top-k（默认 k=3）experts。

  3. **LoRA 加载（GPU 端 - Punica）**：vLLM 的 LLMEngine 已加载 base model（Mistral-7B, FP16）到 GPU 显存。Punica 的 multi-LoRA 机制为每个 query 动态加载对应 k 个 expert 的 LoRA 权重（每个 LoRA 约数 MB）。单 3090 GPU 可同时持有 base model + 最多 200 个 LoRA weights。MELD 的动态 LoRA switch 避免 traditional merge：不将 LoRA 权重写入 base model，而是 concatenate 后在 forward pass 中按需应用。每个 query 仅激活 k 个 LoRA adapter。

  4. **vLLM 推理执行（GPU 端）**：
     - **Prefill 阶段**：query tokens 批量送入 Mistral-7B backbone。在 MoE Router 层，Punica 根据 N(q_u) 选择的 expert ID 加载对应的 k 个 LoRA adapter。每个 expert 输出经 gating weight g_i 加权求和。
     - **Load Balancing 优化**：MELD + vLLM 将相似 query 聚集到同一 GPU，在 4×3090 配置下实现 data parallelism（而非 tensor parallelism）。因为每个 expert（7B + LoRA）足够小，单 3090 可容纳 16 个 experts。
     - **Decode 阶段**：autoregressive 生成 output tokens，PagedAttention 管理 KV cache。

  5. **性能结果**：
     - 4×3090：MELD 吞吐量为 JellyFish(13B) 的 3.7×（JellyFish 需 tensor parallelism 跨 GPU 通信），为 Mixtral(8×7B, 56B total) 的 5.6×。
     - 1×3090：MELD 可 full precision 运行；JellyFish 需 4-bit quantization 才能部署（导致 1.3× 吞吐优势但性能下降显著）；Mixtral 即使 4-bit 量化也 OOM 无法部署。
     - Model process time（LoRA 合并与模型准备）：MELD 比 JellyFish 快 10×，比 Mixtral 快 30×。

## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Faster-MoA**，一个统一的算法-系统协同设计，针对 MoA (Mixture-of-Agents) serving 的三个Serving调度创新：(1) **Shell Router + Agent Prompt Cache (APC)**：在 SGLang 标准 PD 引擎之外实现独立的 shell router，负责任务分发和编排。APC 存储每个依赖 agent 的部分解码文本/token，使后继 agent 可增量构建 prompt。(2) **依赖感知增量 Prefilling**：shell router 将依赖 agent 的输入 prompt 按前驱 agent 输出槽分割为独立前缀+依赖段。前缀段无数据依赖可立即 prefill，依赖段随着前驱 agent 解码逐 chunk 流式到来进行增量 prefilling（基于 KV cache 复用，仅计算新增 token 的 KV），实现 decode 和 prefill 的重叠。(3) **两个 API entrypoint**：/generate（标准 PD pipeline）和 /prefill_only（仅执行 prefill 并缓存 KV blocks，不触发 PE→DE 传输）。
  实验比较：(a) 动态 Early-Exit 消融：Tree+EE vs Tree-only 的模型激活分布（4B/8B/32B 各被调用的比例）、EE 开销（仅 ~5% 额外延迟但不带来 10-50% E2E 延迟减少）; (b) 增量 Prefilling vs 三个 baseline（Naive PD disaggregation only、Data Parallelism only、DP+chunked prefill）的第二层 E2E 延迟（最大 27.4% 减少 vs baseline 仅 ~10%）; (c) 最终对比：All-to-all Baseline vs Tree-only vs Tree+Incremental Prefill vs fully-integrated Faster-MoA，E2E 延迟分别减少 ~62%、~76%、~90%，同时准确率 ≤±1%。

- 硬件平台是什么，配置是什么。
  6 张 NVIDIA H200 GPU（单台 H200 HGX Server 内），每模型配置为一台 Prefill Engine (PE) + 一台 Decode Engine (DE)，跑在两个独立 GPU 上。PE 与 DE 之间通过 NVLink 传输预填的 KV blocks。最大输出 token capped at 65535，scheduling conservativeness=0（SGLang 激进调度最大化显存利用率）。

- 开源Serving框架是什么。修改了什么。
  两个开源框架：(1) **SGLang v0.5.3**——用于精确延迟测量，修改包括：添加 /prefill_only API entrypoint、集成 Shell Router 编排逻辑、Agent Prompt Cache 机制、增量 prefilling 流程（fetch→append→incremental-prefill loop），设 concurrency=1 获取精确 per-sample 延迟。(2) **vLLM v0.11.0**——用于大规模 batch dataset-wise 验证，修改包括集成增量 prefilling 和 early-exit 逻辑，设 concurrency=32 questions/batch 加速验证。
  核心修改架构：在 SGLang 的 native PD router + PE/DE 引擎之上添加外层 **Shell Router**。Shell Router 处理四步：(1) Dependency identification——独立请求直接转发 native PD router；(2) Dependent requests handling——按前驱 agent 输出槽分割 prompt，发送前缀到 PE 开始 prefill，监控第一个依赖 agent 的 APC；(3) Incremental prefilling loop——周期性从 APC fetch text/token chunk，append 到已 prefilled 前缀后，发出轻量 /prefill_only update，利用已驻留 HBM 的 prefix KV 达到近 100% KV cache hit；(4) Forward prefill-done requests——所有槽填满后转发 /generate 请求到 native PD router。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文代码未公开（截至查询时无 GitHub 仓库）。来自 Georgia Tech + Peking University + Samsung，提交 DAC 2026。下面基于论文 (Sec. 4.3) 描述给出 Faster-MoA SGLang serving 全流程：

  **以 3-agent 依赖（agent 3 依赖 agent 1、agent 2 输出）在 9-3-1 三层树结构第二层的执行为例**：

  1. **Agent 1 和 Agent 2 先独立执行**：二者的 prompt 无数据依赖 → Shell Router 识别为独立请求 → 直接转发 native SGLang PD router → 标准 prefill+decode → decode 出的 text/token 流式写入各自 APC。

  2. **Agent 3 依赖识别**：Shell Router 接收 agent 3 请求，解析其 prompt 发现依赖 agent 1、agent 2 的输出槽 → 将 prompt 分割为三段：Segment 0 (agent 3 自身前缀)、Slot 1 (agent 1 的输出槽)、Slot 2 (agent 2 的输出槽)。

  3. **前缀 Prefilling（立即启动）**：Segment 0 无数据依赖 → Shell Router 立即发出 /prefill_only 请求到 PE → PE 计算 Segment 0 全部 token 的 KV → KV blocks 驻留 PE HBM（不传输到 DE，因 /prefill_only 跳过了 KV block 传输）。

  4. **增量 Prefilling Loop - Slot 1**：Shell Router 周期性监控 agent 1 的 APC → APC 收到 agent 1 decode 的第一个 text/token chunk → Shell Router 将 chunk 追加到 Segment 0 之后 → 发出 /prefill_only update（仅新 chunk 的 KV 需计算，prefix KV 从 HBM 复用，近 100% cache hit rate）→ 继续 fetch APC → append → incremental prefill 直至 Slot 1 填满（agent 1 decode 完成）。

  5. **增量 Prefilling Loop - Slot 2**：Slot 1 依赖 agent 1 全部输出完成后，Slot 2 依赖的 agent 2 输出段在 prompt 中紧随 Slot 1 → Shell Router 类似地 fetch agent 2 的 APC chunk → append → incremental prefill 直至 Slot 2 填满。

  6. **Prefill 完成 + Decode 启动**：所有 slot 填满 → agent 3 输入 prompt 完整且 prefilling 已在 overlap 中完成（计算被前驱 decode 时间隐藏）→ Shell Router 转发 /generate 请求到 native PD router → DE 执行标准自回归解码。

  **关键执行对比（Fig 3 底部 bubble diagram）**：
  - Vanilla MoA: Agent 1 decode → Agent 2 decode → Agent 3 prefill（等待两者完成）→ Agent 3 decode → 总时间为串行累加
  - Faster-MoA: Agent 1 decode / Agent 2 decode（并行） | Agent 3 prefix prefill（立即） | Agent 3 incremental prefill（与 Agent 1/2 decode 重叠） → Agent 3 decode → 总时间仅略长于最慢前驱

  数据流：`User Prompt → Shell Router → Dependency ID → Segment split → PE(prefill-only, store KV) → APC(poll decoded chunks) → Incremental /prefill_only(append, reuse KV) → /generate to DE → Output tokens`

## Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 FinDEP —— 一个针对 Disaggregated Expert Parallelism (DEP) 的细粒度任务调度框架，用于优化 MoE 模型推理吞吐。包含三个关键创新：(1) 将 AG 和 EG 中的计算和通信任务沿 batch 维度和 token 维度分别切分为更小的子任务（AG 端沿 batch 维度切分为 r1 个 micro-batch pipeline，EG 端沿 token 维度切分为 r2 个 fine-grained pipeline），以实现细粒度任务流水线和最大程度的重叠；(2) 建立包含计算和通信开销的端到端性能模型，形式化一个优化问题来表征 DEP 推理时间，涵盖任务顺序、tensor 切分粒度 r1/r2 和 micro-batch size ma/me；(3) 开发一个多项式时间复杂度的算法（Algorithm 1）在巨大解空间中搜索近似最优调度配置。实验比较 FinDEP 与 PPPipe (MegaScale-Infer 中的 Ping-Pong Pipeline)，评估在四种 GPU 平台（8×A6000、8×A10、8×H20、32×H20）上使用 DeepSeek-V2（有 shared experts）和 Qwen3-MoE（无 shared experts）两种 backbone 的推理吞吐（tokens/s），以及非重叠通信时间的减少效果。同时评估在线场景下 FinDEP 快速 solver（<1s）的自适应能力。

- 硬件平台是什么，配置是什么。
  四个硬件 Testbed：
  - Testbed A: 单节点 8×NVIDIA RTX A6000 (48GB, Ampere, Boost 1.46GHz, NVLink Yes, PCIe 4.0×16)
  - Testbed B: 单节点 8×NVIDIA A10 (24GB, Ampere, Boost 1.41GHz, NVLink No, PCIe 4.0×16)
  - Testbed C: 单节点 8×NVIDIA H20 (96GB, Hopper, Boost 1.98GHz, NVLink Yes, PCIe 4.0×16)
  - Testbed D: 四节点 32×NVIDIA H20 (每节点 8×H20, 96GB, Hopper, Boost 1.98GHz, NVLink Yes, PCIe 4.0×16)
  软件环境：Ubuntu 22.04, Python 3.10, CUDA 11.3, PyTorch 2.4, NCCL 2.27.5, FlashInfer 0.3.0

- 开源Serving框架是什么。修改了什么。
  论文基于 MegaScale-Infer [36] 中的 PPPipe 算法进行复现和对比，但并未修改特定开源 Serving 框架来部署 FinDEP。论文在白盒实现中直接实现了 DEP 的基础设施：(1) 实现了 AG/EG 分组，AG 负责 Attention 层和 Shared Expert（如有）计算，EG 负责所有 sparse experts 的计算；(2) 使用 NCCL 实现 A2E 和 E2A 通信原语；(3) 使用 FlashInfer 0.3.0 实现 Attention 计算；(4) 实现了 PPPipe 的 micro-batch 流水线调度和 FinDEP 的细粒度任务调度（ASAS 和 AASS 两种执行顺序）；(5) 实现了 offline 性能模型参数采集（α_gm, β_gm, α_attn, β_attn, α_a2e, β_a2e）和 online 快速 solver（Algorithm 1）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供独立的开源代码仓库链接。FinDEP 的 DEP 推理全流程如下：

  ```
  === Offline Phase (one-time) ===
  1. 选定 serving model (DeepSeek-V2 或 Qwen3-MoE)
  2. 确定 AG 和 EG 大小 (ag, eg)，满足 ag+eg=P
  3. 运行 micro-benchmark 采集性能模型系数：
     - GEMM: t_gm(x) = α_gm + β_gm * x，测试 MLA 中所有矩阵配置
     - Attention: t_attn(y) = α_attn + β_attn * y
     - A2E/E2A Communication: t_c(z) = α_c + β_c * z，测试不同 (ag, eg) 组合
     全过程 < 2 分钟

  === Online Phase (per-request adaptive) ===
  4. 接收用户请求，获取 sequence length S 和 batch size B
  5. 执行 Algorithm 1 快速求解:
     for ma = M downto 1:                           // 按内存上限递减
         r1 = getMaxR1(ag, eg, ma, ...)             // 内存约束下的最大 r1
         if r1 == 0 or r1 == previous: continue     // 跳过非Pareto最优
         for order in {ASAS, AASS}:
             r2* = solve convex min(1/r2) Eq.17     // 凸优化求最优 r2
             me = ma * ag * top_k * S / (r2* * E)   // 反推 me
             if tps > best_tps: update best_config
     返回 best_config = (ma, r1, me, r2, order)
     耗时 < 1 秒

  === Per-Layer DEP Execution (with FinDEP schedule) ===
  For each MoE layer t=1..T:
    AG (per GPU):
      For i = 0..r1-1:
        τ_a^(t,i): Attention 计算 ma 个样本             // t_a(ma) 时间
        τ_s^(t,i): Shared Expert 计算 (ASAS顺序下与下一 Attention 交替)
        τ_a2e^(t,i,j): j=0..r2-1, A2E 通信发送 me 个 token
    EG (per GPU, E/eg experts per device):
      For i = 0..r1-1, j = 0..r2-1:
        τ_e^(t,i,j): Expert FFN 计算 me 个 token         // t_e(me) 时间
        τ_e2a^(t,i,j): E2A 通信返回 expert 输出
  ```

  FinDEP 的核心效果体现在：(1) r1 micro-batch pipeline 使 AG 和 EG 可并行执行，A2E 与 Shared Expert 可并行；(2) r2 fine-grained pipeline 使 A2E/E2A 通信与 EG 计算进一步重叠；(3) ASAS/AASS 两种执行顺序选择使系统能根据 shared expert 开销自适应选择最优策略。在 8×A6000 DeepSeek-V2 S=4096 条件下，非重叠通信时间从 Naive-DEP 的 905.49ms → PPPipe 的 528.94ms → FinDEP 的 309.81ms。

## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 METRO 在 vLLM Serving 框架中的集成，核心修改是 EP (Expert Parallelism) dispatch 通信模式的替换：(1) 将传统 all-to-all dispatch 替换为 all-gather dispatch——每个 GPU 在 MoE expert 层计算前，先将本地 tokens all-gather 到所有 GPU，使每个 GPU 获得全局 token 集合；(2) 每个 GPU 基于全局 token 集合独立执行 top-k 计算，构建全局 T[1..N]（每个 expert 的总 token 数）；(3) 执行 METRO greedy routing（Algorithm 1）决定每个 expert 在哪个 GPU 上激活；(4) 仅计算分配给本 GPU 的 expert FFN；(5) all-to-all combine 将输出返回原 GPU。该修改的动机是：传统 all-to-all dispatch 只让各 GPU 知道本地 top-k 结果，无法获得全局 expert token 分布信息，而 METRO 的 MIN-EXP-ROUTING 算法需要全局 top-k knowledge (T[1..N]) 才能做 informed routing 决策。此外 METRO 集成 vLLM 的 CUDA Graph compilation framework，将 decode phase 的路由逻辑编译进 power-of-two batch sizes（up to 32 tokens per GPU）的 CUDA Graphs，消除额外 kernel launch overhead。

  实验比较：(a) METRO vs EPLB token routing 在 vLLM (8×A100) 上的 decode latency (TPOT) 和 total token throughput；(b) METRO vs EPLB 在 decode-heavy（InstructCoder, NuminaMath, Humaneval）和 prefill-heavy（GSM8K）workloads 下的吞吐影响差异；(c) METRO all-gather vs EPLB all-to-all 的通信时间对比；(d) 不同 replication ratio（1.0x, 1.125x, 1.25x, 1.5x）下 METRO 的性能增益变化；(e) decode throughput-latency Pareto 分析，变 batch size (64–1024) 和 parallelism (TP1-16 × EP1-16) 组合。

- 硬件平台是什么，配置是什么。
  真实系统：Google Cloud a2-highgpu-8g VM，8×NVIDIA A100 40GB GPU，600 GB/s NVLink（全部 GPU 在同一 NVLink domain）。batch size 限制：decode phase 最多 32 tokens/GPU，prefill phase 最多 32 prompts/GPU，context length 8K。模拟器：专有工业级 multi-GPU performance simulator，8×B200 192GB (Qwen3-235B) 和 16×B200 192GB (DeepSeek-V3)，900 GB/s NVLink。模拟器 configs: global decode batch size 1K, chunked prefill limited to 8K, sequence length 1K input + 2K output。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**vLLM**（https://github.com/vllm-project/vllm）。论文在 vLLM 中做了以下修改/新增：
  (a) **EP dispatch 通信模式替换**：将 MoE expert layer 的 dispatch 阶段从 all-to-all 改为 all-gather，使得每个 GPU 在 top-k 前获得全局 token 集合；
  (b) **METRO routing kernel 集成**：实现 Algorithm 1 的 CUDA kernel，运行在单个 SM 上，使用 test-and-set lock 和 SM-local shared memory；
  (c) **CUDA Graph 集成**：利用 vLLM compilation framework 将 METRO routing 编译进 decode phase CUDA Graphs，预编译 power-of-two batch sizes 的图（up to 32 tokens/GPU），非 power-of-two batch 通过 padding 复用；
  (d) **EPLB placement/replication 保留**：METRO 仅替换 token routing 部分，不修改 EPLB 的 expert placement 和 replication 策略，避免干扰 prefill phase 性能；
  (e) **METRO 仅应用于 decode phase**：prefill phase 继续使用 EPLB token routing，因为 prefill 是 compute-bound 的。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文作者包含 NVIDIA 团队，**论文未明确提供 METRO 的独立开源代码仓库**。实现基于开源 vLLM 框架。以下是 METRO 在 vLLM 中的完整推理流程：

  ```
  === METRO + vLLM EP MoE Serving 全流程 ===

  Input: batch of requests with prompts, model distributed across G GPUs via EP

  === Prefill Phase (使用 EPLB token routing, compute-bound) ===
  For each transformer layer:
    Step 1 Attention: 数据并行 (DP)，每个 GPU 计算本地 tokens 的 attention
    Step 2 MoE Gating: 每个 GPU 独立对本地 tokens 计算 router top-k
    Step 3 EPLB Token Routing: 将每个 expert 的 token 均匀分配到其 replicas 上
    Step 4 All-to-all Dispatch: tokens 根据路由决策发送到目标 GPU
    Step 5 Expert FFN: 每个 GPU 计算分配给自己的 expert FFN
    Step 6 All-to-all Combine: expert 输出 embedding 返回原 GPU

  === Decode Phase (使用 METRO routing, memory-bound) ===
  For each transformer layer:
    Step 1 Attention: compute attention on local tokens (DP)
    Step 2 All-gather Tokens: 每个 GPU 将本地 tokens all-gather 到所有 GPU
        替换传统 all-to-all！
        通信量: 2MB/GPU (32 tokens * hidden_dim on 8 GPUs, fp16)
        NVLink 带宽开销: ~3us (on 600 GB/s)
        NCCL launch fixed cost: ~100us -> 带宽开销远低于固定开销
    Step 3 Global Top-K: 每个 GPU 在全局 (~256 tokens) 上计算 top-k
        冗余计算开销: <3us (<1% 层时间)
        -> 构建 T[1..N]: 每个 expert 在全局 batch 中的 token 数
    Step 4 METRO Routing (CUDA kernel, 单 SM):
        执行 Algorithm 1——greedy assign each expert to GPU with fewest activated experts
        开销: 最多 26us (1.5x replication)
    Step 5 Expert FFN: 每个 GPU 仅计算分配给自己的 expert FFN
        仅在激活的 expert replicas 上计算 -> 减少内存流量
        FFN 时间减少: 最多 81us (1.5x replication)
    Step 6 All-to-all Combine: expert 输出 embedding 返回原 GPU
    Step 7 Layer output: attention output + MoE output combined
  ```

  关键性能收益：(a) METRO 将 activated experts 数量减少 up to 42.3% vs EPLB routing；(b) decode latency 降低 11%-22%；(c) total token throughput 提升 3%-21%（co-deployed prefill+decode）；(d) 在 decode-heavy workloads 上增益更显著（up to 21%），prefill-heavy 上仍有 4.2% 提升；(e) 在固定 SLO 下 decode throughput 可达 EPLB 的 1.98x-4.11x。

## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **ElasticMoE 弹性自动伸缩框架**，在 vLLM 之上新增三个模块实现 MoE 模型的细粒度、低延迟、零停机垂直缩放：(1) **Coordinator**：用户请求入口，维护活跃请求队列，监控 SLO 达标率，触发 scale-up/scale-down 指令，并在新旧配置间无缝切换流量；(2) **HBM Management Module (HMM)**：全局控制面（Python/Ray）+ 每设备 worker（C++/CANN API），管理 HBM 上的模型权重和 KV cache，与推理执行解耦。权重只加载一次，跨推理实例通过 zero-copy IPC 共享，缩放时计算最小代价的权重再分配计划（最大化 zero-copy 复用、最小化 P2P 传输）；(3) **Inference Management Module (IMM)**：管理多个推理实例（基于 vLLM），同一时刻仅一个活跃。通过 LRU 缓存 pre-initialized standby 实例，缩放时从 HMM 获取 zero-copy 引用句柄附加权重和 KV cache，快速激活新配置。核心 primitives：`zero-copy`（Ascend IPC 跨进程共享张量）、`p2p-copy`（HCCL P2P 传输绕过 host memory）、`vpage-remap`（虚拟内存管理 expert 权重，避免大缓冲区重分配）、`disk-copy`（选择性磁盘加载避免重复读取）。缩放策略：固定 TP 度，仅调整 DP 和 EP 度。实验比较：(a) Scaling Latency：Horizontal (Replica)、Vertical (Cold Restart)、Vertical (Extravagant)、Vertical (Colocated) vs ElasticMoE，在 DeepSeekV2 Lite、Qwen3-30B-A3B、DeepSeek V3 三个模型上，ElasticMoE 缩放延迟仅为最佳 baseline 的 ≈0.11×（提升约 80.9%）；(b) SLO Recovery：scale-up 4→6 NPU 下 SLO 恢复速度和 scale-down 6→4 NPU 下 SLO/NPU 成本效率；(c) SLO Compliance：RPS 递增负载下各方法维持 SLO≥90% 的能力；(d) Throughput During Scaling：缩放窗口前后和期间吞吐量对比；(e) Ablation：逐步禁用 IPCAlloc、HCCL P2P、PreInit、ZeroCopy 各组件的缩放延迟和 downtime。

- 硬件平台是什么，配置是什么。
  **Huawei CloudMatrix384 supernode**：集成 384×Ascend 910C NPU（每颗 64 GB HBM），192×Kunpeng 920 CPU，分布在 24 个节点。每节点 16×Ascend 910C + 4×Kunpeng 920（1.5 TB 系统 RAM）。所有 CPU 和 NPU 通过 Unified Bus (UB) 互联，提供 non-blocking all-to-all 连接，近均匀的节点内/节点间通信延迟。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**vLLM**（https://github.com/vllm-project/vllm）。基于 ascend-vLLM（华为 Ascend NPU 适配版）实现。修改/新增：(a) **HMM 模块**：新增 HBM 管理守护进程，负责模型权重持久化加载、KV cache 管理、缩放计划计算和权重再分配。控制面用 Python/Ray，数据面用 C++/CANN API + PyBind11；(b) **IMM 模块**：新增 `ZeroCopyLoader` 替代传统 `DiskLoader`，新增 `Instance Manager` 管理推理实例生命周期、LRU cache 和流量切换；(c) **Coordinator 模块**：新增 `SLO-aware Load Estimator` 监控 SLO 并触发缩放，新增流量无缝切换逻辑；(d) **低层 primitives**：`IpcSafeAllocator` 覆盖 PyTorch 默认内存分配器（torch.ones/empty/full），`p2p-copy`（HCCL isend/irecv/broadcast），`zero-copy`（rtIpcSetMemoryName/rtIpcOpenMemory），`vpage-remap`（aclrtMallocPhysical/aclrtReserveMemAddress/aclrtMapMem），`disk-copy`（按名称/partition/layer 选择性加载），`add-nodes`（运行时动态扩展 HMM 管理的节点和 NPU）；(e) **整体架构**：Coordinator→HMM→IMM 三级模块通过 ZMQ/UNIX domain socket IPC 通信，Coordinator 对外暴露 TCP API（OpenAI-style inference API）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立开源代码仓库，但基于开源 vLLM + 华为 ascend-vLLM 和 CANN API 实现，HMM 支持 vLLM model loader backend。ElasticMoE 的弹性 Serving 全流程如下：

  ```
  === 初始化阶段 ===
  Step 1: HMM 从磁盘加载初始配置（如 DP2-TP2-EP4 on NPU 0-3）的模型权重和 KV cache
  Step 2: HMM 通过 IpcSafeAllocator 分配 IPC 兼容内存，export_handle 导出张量引用
  Step 3: IMM 创建活跃推理实例（vLLM），通过 ZeroCopyLoader.open_tensor 获取 HMM 引用句柄附加权重
  Step 4: IMM 可选 pre-initialize standby 实例（仅 CPU 内存），存入 LRU cache
  Step 5: Coordinator 开始路由请求到活跃实例

  === Scale-Up 操作 (NPU 0-3 → NPU 0-5, DP2→DP3, EP4→EP6) ===
  Step 1: Coordinator 的 SLO-aware Load Estimator 检测到 SLO 持续低于 90%，触发 scale-up 命令
  Step 2: HMM 分析旧配置 DP2-TP2-EP4 和新配置 DP3-TP2-EP6，生成最小代价再分配计划
  Step 3: Attention 权重：NPU 0-3 上保持不变（TP 固定），通过 zero-copy 复用；NPU 4-5 通过 p2p-copy (HCCL) 从 NPU 0-1 异步传输
  Step 4: KV Cache：NPU 0-3 已存在的 KV cache 通过 zero-copy 直接复用（无重复分配），旧实例继续使用同一份 cache 服务 in-flight 请求；NPU 4-5 初始化新 KV cache
  Step 5: Expert 权重：全局 remap 专家→NPU 映射以平衡负载，通过 p2p-copy 迁移专家到新 NPU，通过 vpage-remap 更新虚拟→物理映射（旧映射保持活跃直到新实例接管）
  Step 6: IMM 从 LRU cache 取/创建 6-NPU 推理实例，通过 zero-copy 附加权重和 KV cache，标记为 ready
  Step 7: Coordinator 停止向旧实例路由新请求，等待 in-flight 请求完成，旧实例标记 inactive，流量切到新实例
  Step 8: 旧实例终止，释放不再使用的物理内存页

  === 推理运行 ===
  用户请求 → Coordinator TCP API → OpenAI-style forward → 活跃 IMM 实例 (vLLM on Ascend) → HCCL 集合通信 (all-to-all/TP) → Ascend 910C NPU 执行 attention/expert GEMM → 返回 token
  ```

  关键性能收益：(a) Scale-up latency 约 0.11× 最佳 baseline（≈9× 改善），scale-up 2.43s (DP3→DP4, DeepSeek V2 Lite)；(b) Peak memory 仅比 Cold Restart 高 2-3%，比 Extravagant 低 35-40%；(c) 缩放到 4→6 NPU 后几乎立即恢复 SLO 合规（≥90%），Cold Restart 需要额外数十秒恢复；(d) 缩放期间 throughput 达 Cold Restart 的 ≈2×；(e) 零 downtime。

## ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **ExpertFlow**，一个面向单 GPU 内存受限场景的 MoE 推理系统，包含三个协同组件：
  (1) **Routing Path Predictor (RPP)**：T5-style encoder-decoder 架构，在单次前向传播中预测所有 token 在所有 MoE 层的 expert 激活路径，输出形状为 (B, S, L, E) 的激活概率矩阵。训练使用 binary cross-entropy 的多标签分类任务，预测器大小 7.21 MB（FFN dim=2048, hidden size=32）。在多数 in-domain 场景下达到 >90% 预测准确率，跨域仅下降 5-10%。
  (2) **Token Scheduler (TS)**：基于 K-means 聚类将具有相似路由路径的 token 重新分组到同一 batch 中。以两个相邻 batch 的 2T 个 tokens 为输入，构造 routing path 相似度矩阵，通过最小化 batch 级 expert 激活数（公式: min Σ(R1 + R2)）将 token 重新分配到两个等大小 batch 中，减少 active expert 数并提高 per-expert token 负载。包含自适应 KV-Cache 管理（Merge + Reindex）和 Dual-Batch Inference Pipeline 以隐藏 overhead。
  (3) **Expert Cache Engine (ECE)**：包含 Predictive Locality-aware Expert Caching (PLEC) 和 Real-time Correction。PLEC 基于 RPP 预测自适应分配各层 cache slot，预取预测需要的 expert；运行时检测误预测并执行优先交换，与 compute 重叠以减少 I/O 等待。

  实验比较：(a) in-domain throughput vs Cache-MoE/SE-MoE/Pregated-MoE（Switch 系列在 WMT16, Mixtral-8 在 XSUM, Qwen1.5 在 Alpaca, Deepseek-MoE 在 AIME2024）；(b) cross-domain throughput（Qwen1.5 在 WMT16/XSUM，RPP 用 Alpaca 训练）；(c) 峰值 GPU memory vs All-in-GPU；(d) RPP 预测准确率 vs TLP/SLP baselines；(e) Cache hit ratio PLEC vs LRU；(f) TS 对 throughput 的 ablation。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A40 GPU (48 GB memory)，CPU 为 Intel(R) Xeon(R) Gold 6338 @ 2.00GHz。

- 开源Serving框架是什么。修改了什么。
  ExpertFlow 是独立设计的 MoE 推理系统，从零构建，而非基于现有开源 Serving 框架（如 vLLM、SGLang）修改。其核心组件（RPP、TS、ECE）均为全新实现。RPP 使用 T5-style encoder-decoder 架构；TS 使用 K-means 聚类进行 token rebatching；ECE 实现 PLEC + real-time correction 的预测驱动缓存策略。论文未明确说明基于哪个 Serving 框架集成。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **未开源**（论文未提供代码链接，GitHub 搜索无公开仓库）。

  以下基于论文描述给出 ExpertFlow 从输入到硬件执行的全过程：

  ```
  === 离线阶段：RPP 训练 ===
  对每个 (task, MoE model) 组合:
    采样 10,000 个输入序列
    每个序列运行 MoE 模型 3 次，收集 30,000 个 (input, output, routing_path) triple
    每个 routing path 编码为 r ∈ {0,1}^{L×E} 的 binary matrix
    训练 RPP (T5 encoder-decoder, FFN=2048, hidden=32, 7.21MB):
      Loss = BCE(r, p)  where p = RPP(input)
  
  === 在线推理：Dual-Batch Pipeline ===
  Input: batch_0, batch_1 (各 B 个 sequence, 每个 S 个 tokens)
  
  Step 1 - RPP 预测 (与上一 scheduling unit 的 MoE 执行并行):
    # T5 encoder 编码全输入序列
    encoder_output = T5_encoder.encode(concat(batch_0_inputs, batch_1_inputs))
    # T5 decoder + L 个 light-weight heads (每层一个) 一次性输出
    p = RPP_decoder(encoder_output)  # shape: (2B, S, L, E)
    # p[l][e] = 预测 expert e 在 layer l 被激活的概率
    activation_matrix = (p > threshold)  # 二值化

  Step 2 - TS Token 重新分组 (CPU, <10ms):
    # 2T 个 tokens 的 routing path: r_i ∈ {0,1}^{L×E}
    # 计算 Hamming distance 相似度矩阵 S ∈ R^{2T×2T}
    S[i][j] = 1 - Hamming(r_i, r_j) / (L*E)
    # K-means 聚类为两个等大小 batch
    while not converged and iter < max_iter:
      分配每个 token 到最近的 cluster centroid
      更新 centroid 为 intra-cluster 平均相似度最高的 token
    yield (T1, T2)
    # Merge + Reindex KV cache 以保持 attention 语义

  Step 3 - ECE Expert 预取与缓存:
    # PLEC: 基于预测分配各层 cache slot
    预测需求: layer_1 需要 3 experts, layer_2 需要 2 experts
    GPU cache capacity: 4 experts
    allocation = [3 slots for layer_1, 1 slot for layer_2]
    # 预取最可能需要的 4 个 experts: CPU→GPU copy
    prefetch([e_12, e_13, e_14, e_22])

  Step 4 - MoE 模型执行:
    for layer in layers:
      # Gating (GPU 上执行)
      gate_scores = softmax(x @ W_gate)
      top_k_experts = topk(gate_scores)
      
      # 检查 expert 是否已在 GPU cache 中
      for expert in top_k_experts:
        if expert not in gpu_cache:
          # Real-time Correction: 异步 CPU→GPU 加载缺失 expert
          # 与当前 running expert 的 compute 并行 (overlap)
          async_load(expert)
        # 执行 expert FFN
        out += gate_score * expert_ffn(x)
      
      # 释放已完成 early-layer expert，加载下一层 expert
      free_completed_experts()
      prefetch_next_layer_experts()

  Step 5 - Token 输出:
    # 与 baseline offloading 相同
    logits = lm_head(hidden) → sample → next_token
    
  === GPU Memory 管理 (NVIDIA A40 48GB) ===
  GPU 常驻: attention weights + gate weights + RPP (7.21MB)
  GPU 动态: expert cache (大小由 cache_size 参数控制, 如 4/8/16 experts)
  CPU 常驻: 全部 expert 参数 (Mixtral-8×7B: 45.1B params in experts)
  ```

  关键性能收益：(a) Switch-128 在 CS=4, BS=32 下达 9.99× throughput vs SE-MoE；(b) GPU memory 最大降低 93.72%（Switch-128: 15.26GB → 1.03GB）；(c) Mixtral-8×7B 在 AIG 下 OOM 但 ExpertFlow 下仅需 15.99GB；(d) PLEC cache hit ratio 91.90%（CS=16, BS=4），比 LRU 高 15-36%；(e) TS 在 Switch-128 上额外提升 1.17× throughput；(f) RPP 跨域 accuracy 仅下降 5-10%，Qwen1.5 上达 >95%。

## Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

- 属于Serving调度的实现是什么？实验比较什么？
  提出 ExFlow，通过两层优化加速 GPT MoE 分布式推理：(1) **Context-Coherent Expert Parallelism**：在推理开始和每轮迭代结束时使用 AllGather 使所有 GPU 持有全部 token 的 context，从而每个 MoE 层只需 1 次 Alltoall（token dispatch）而非传统 expert parallelism 的 2 次 Alltoall（dispatch + gather），因为 token 可在任意 GPU 上原地执行 attention 计算；(2) **ILP-based Expert Affinity Placement**：通过整数线性规划（ILP）建模 cross-layer expert affinity，离线求解最优 expert 放置方案，使 affiliated experts 尽可能放在同一 GPU（intra-GPU affinity）或同一节点（intra-node affinity），最大化 token 留在本地 GPU/节点的概率。
  实验比较 ExFlow vs Deepspeed-MoE baseline，评估 Alltoall 通信量、cross-GPU/cross-node token routing 比例、端到端推理 throughput（tokens/sec），以及 expert affinity 在训练过程中的演化。baseline（Deepspeed-MoE）使用传统 expert parallelism，每 MoE 层严格 2 次 Alltoall，expert 按 rank 随机放置无 affinity 优化。

- 硬件平台是什么，配置是什么。
  Wilkes3 Ampere GPU 集群：每节点 2× AMD EPYC 7763 64-Core，4× NVIDIA A100-SXM4-80GB（NVLINK intra-node），inter-node 为 dual-rail Mellanox HDR200 InfiniBand。实验使用 1-16 节点（4-64 GPU）。

- 开源Serving框架是什么。修改了什么。
  框架：基于 **DeepSpeed-MoE / Megatron-DeepSpeed** 进行 pre-training，推理时在 DeepSpeed 的 expert parallelism 基础之上修改。修改内容：
  (1) **Alltoall 次数减半**：每 MoE 层从 dispatch+collect（2× Alltoall）变为仅 dispatch（1× Alltoall），省去 token gather；新增推理开始和每轮迭代结束时的 AllGather 以同步 context。
  (2) **Expert 放置策略**：模型加载时不再按 GPU rank 均匀随机分配 expert，而是根据 ILP 求解结果放置 expert，使 high-affinity expert 对在后续层中处于同一 GPU。
  (3) **Staged Expert Affinity**：先最小化 inter-node routing（stage 1），再最小化 intra-node routing（stage 2），适应 NVLINK > InfiniBand 的带宽层次。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/YJHMITWEB/ExFlow（论文声明 available）
  全栈执行流程：
  ```
  === 离线阶段（Profiling） ===
  1. 从 Pile 数据集随机采样 1000-3000 token
  2. 将 token 输入 pre-trained GPT MoE 模型，记录每个 token 在每层被 route 到的 expert 编号
  3. 构建 conditional probability matrix P(E_{p,j+1} | E_{i,j})
  4. 将 max-affinity expert placement 建模为 ILP：
     Minimize Σ R_{k,j}（R_{k,j}=1 表示 token k 在 layer j 需跨 GPU/node 路由）
     Subject to: 每 GPU/node 每层 expert 数 = E/P（负载均衡）
                 每个 expert 仅由 1 个 GPU/node 持有
                 R_{k,j} >= x_{i,j}^p - x_{i,j+1}^p（路由判定约束）
  5. 分两阶段求解：stage 1 最小化 inter-node routing，stage 2 在 stage 1 结果上最小化 intra-node routing
  6. 求解得到 x_{i,j}^p（每个 expert 在每层应放在哪个 GPU/node）

  === 在线推理阶段 ===
  [推理开始] AllGather: 所有 GPU 广播各自的 context → 所有 GPU 持有全部 context
  For each token (in parallel on each GPU):
    For each MoE layer j:
      [Attention] token 在本地 GPU 用本地 context 执行 self-attention（无通信）
      [Gating] 共享 gating function 决定 token 应 route 到哪个 expert
      [Alltoall Dispatch] token 被发送到持有目标 expert 的 GPU（仅 1 次 Alltoall）
      [Expert FFN] 在目标 GPU 上执行 FFN 计算
      （无需 Alltoall Gather：token 留在当前 GPU）
  [迭代结束] AllGather: 每个 GPU 广播本轮新生成的 token → 所有 GPU 更新 context
  ```
  与 baseline 的关键区别：baseline 每层需要 dispatch Alltoall + gather Alltoall（token 必须回到原 GPU 做 attention），ExFlow 通过 context coherence 消除 gather Alltoall，通过 affinity placement 减少 dispatch 的跨 GPU 比例。

  性能结果：
  - MoE-8: 4 GPU 每 GPU 2 experts，Alltoall ~15%；8 GPU 跨节点时 ~10% throughput 提升
  - MoE-16: 8 GPU（每 GPU 2 experts）达 **2.2x throughput** vs DeepSpeed-MoE
  - MoE-32: 8-16 GPU 达 **1.6x speedup**
  - MoE-64: 4 GPU 每 GPU 16 experts，cross-GPU token routing 减少 40%；32 GPU 仍减少 25%
  - 跨节点 token routing：tokens 平均 **2x** 更可能留在同节点内
  - 最多减少 **67%** cross-GPU routing latency，最多 **120%** throughput 提升

## eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

- 属于Serving调度的实现是什么？实验比较什么？
  eMoE 是一个面向 MoE-based LLM 的记忆体高效推理系统，由四个协同组件构成：(1) **Expert Prediction**：使用 BERT-XLNet（0.108B 参数）基于历史 expert 路由分布预测未来的 expert 序列，支持逐层预测（eMoE-L，用 prev-layer expert 预测 next-layer expert）和全层预测（eMoE-A，用上一条 prompt 的 expert 分布预测当前全部层 expert）；(2) **Periodic Expert Invocation**：每 p 条 prompt（实验确定 p=40）调用一次预测器，reuse expert 减少加载开销而基本不影响 perplexity；(3) **Task-aware Expert Loading**：利用不同任务对 token-to-expert routing accuracy 的敏感度差异，仅对敏感任务加载预测 expert，对分类/对比任务跳过预测以降低加载延迟；(4) **Task-aware Request Scheduling**：联合考虑用户 SLO、profiled 任务特定生成长度、expert 加载延迟进行贪心调度，最小化端到端推理延迟。

  实验比较对象：vLLM、DeepSpeedFastGen（端到端延迟）、Pre-gatedMoE、MoEInfinity、Random（记忆体消耗与准确率）、量化模型 4-bit/8-bit（记忆体 vs 准确率 trade-off）。

- 硬件平台是什么，配置是什么。
  Intel Xeon 处理器 + 128GB host memory，4× Nvidia A100 Tensor Core GPU（40GB 设备内存）。推理延迟实验生成 synthetic request trace（Poisson 分布 + multinomial task 分布），最大生成 token 数设为 1000 以避免 OOM。

- 开源Serving框架是什么。修改了什么。
  基于 **DeepSpeed-FastGen**（github.com/microsoft/DeepSpeed-MII）作为推理引擎，包装 HuggingFace Transformers 模型。对 HuggingFace 模型代码做了以下扩展：
  1. 为每个 MoE 层维护 Python multiprocessing lock，将 MoE layer 计算包装在 lock 内部以同步 expert 加载与计算；
  2. 为每个 MoE 层的 expert 加载包装 CUDA event，MoE 层同步该 event 以防使用 stale model weights；
  3. 异步 expert 加载：通过 `torch.Tensor.copy_(non_blocking=True)` 从 host 到 device 传输 expert，与 non-expert layer 计算重叠；
  4. 条件加载：当前 MoE 层的 expert 加载以前一层加载完成为条件，防止 PCIe 带宽饱和。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供 eMoE 的独立开源代码仓库。实现基于 DeepSpeed-FastGen（开源）和 HuggingFace Transformers（开源）。专家预测器使用 BERT-XLNet（HuggingFace pretrained），PyTorch 实现，运行在独立进程中。

  **eMoE 推理全流程（Full Pipeline）**：
  ```
  === eMoE Serving 全流程 ===

  Input: inference requests 流 (prompt + task type)，4× A100 40GB GPU + 128GB CPU host memory

  Step 1 ─ Task Type Extraction (CPU)
    For each incoming request:
      Parse input token → match profiled keywords → assign task type
      (SUM/CLSFY/QA/COMP/CONV)

  Step 2 ─ Task-aware Request Scheduling (CPU, 见 Algorithm 1)
    Input: waiting queue Qw, scheduled queue Qs, max tokens Tmax
    1. Sort Qw by SLO stringiness (first-token latency deadline, 升序)
    2. For each request R in Qw:
       if R.inputTokens + Qs.totalTokens < Tmax:
         for each scheduled request S:
           compute t_i = ΔE + (W + n_i·G_i)·c + r_i  (Eq. 3)
           if S.expectedLatency < S.SLO: schedule R
    其中 ΔE = profiled expert loading latency
         G_i = task-specific profiled token生成数（运行时递减）
         c = average expert computation+communication latency per input token

  Step 3 ─ Expert Prediction (GPU, 每 p=40 prompts 调用一次)
    eMoE-A (all-layer):
      Input: expert sequence of previous prompt → [e1^r1, e2^r1, ..., em^r1]
        where e_i = [k expert indices] for top-k routing
      XLNet predicts: [e1^r2, e2^r2, ..., em^r2] for current prompt
    eMoE-L (layer-by-layer):
      Input: expert sequence of layer i-1 → e_{i-1}^r1
      XLNet predicts: e_i^r1 for layer i
    Memory: 0.24%-1.3% of MoE model memory

  Step 4 ─ Task-aware Expert Loading (GPU)
    For each MoE layer:
      1. Compute N_i = (Σ W_j + T·W_o) · s · f_i  (Eq. 2)
         where s ∈ {0,1}: task sensitivity to routing accuracy at this layer
               f_i: predicted routing frequency for expert i
               T: # running requests of this task type
      2. Sum N_i across all task types → expected tokens per expert
      3. Sort experts by expected tokens (descending) → pick top L
         (L set by memory budget)
      4. Load new experts: compare predicted vs already on GPU
         → torch.Tensor.copy_(non_blocking=True) for new experts
         → move unpredicted experts to CPU
      5. Condition: wait for previous MoE layer's loading CUDA event
         → prevents PCIe saturation

  Step 5 ─ Inference Engine (DeepSpeed-FastGen, GPU)
    For each transformer layer:
      a. Self-Attention: dense inference (standard HuggingFace forward)
      b. MoE Layer:
         - Acquire multiprocessing lock on this MoE layer
         - Wait for CUDA event (expert loading complete, prevent stale weights)
         - Router gate: compute gating scores → top-k expert selection
         - Expert FFN: execute only loaded experts
         - Token routing: if expert not on GPU, route to next top-k on GPU
         - Release lock
      c. Continue with loaded experts for subsequent prompts (until next p-th)

  Step 6 ─ Periodic Expert Re-invocation
    Maintain request index counter: 0, 1, 2, ...
    When index % p == 0:
      → goto Step 3 (Expert Prediction)
      → goto Step 4 (Expert Loading)
    Else:
      → reuse currently loaded experts on GPU

  Output: generated token sequences → return to client

  === Key Trade-offs ===
  - Memory vs Accuracy: 60% experts loaded → 98.2%-98.8% accuracy
                     80% experts loaded → 99.6%-99.7% accuracy
  - eMoE-A time overhead: ~0.381s (OpenMoE) / ~0.334s (Mixtral) per predictor call
    eMoE-L time overhead: ~1.387s (OpenMoE) / ~4.211s (Mixtral) per predictor call
    Amortized over 40 prompts: 0.47%-3.11% of avg inference time per request
  ```

  性能结果：
  - 记忆体：减少 **up to 80%** vs Baseline（所有 expert 在 GPU）
  - 延迟：降低 **up to 17%** vs DeepSpeedFastGen（Mixtral-8x22B 最大）
  - Prompt 长度：支持 **40× longer** prompts（Mixtral-8x7B, 512→20480）
  - Batch size：支持 **4.5× larger** batches（Mixtral, 4→18）
  - Throughput：**1.5× higher** tokens/second vs Baseline
  - Accuracy：60% experts loaded → 98.2%-98.8%，80% experts → 99.6%-99.7%
  - vs 量化：eMoE-A accuracy 98.2% vs 量化8-bit 95.2% vs 量化4-bit 91.5%（Mixtral-8x7B）
  - eMoE-A vs eMoE-L: 精度相近，eMoE-A 开销更小（见 §5.7）

## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 D2MoE 的在线执行引擎，包含三个核心 Serving 调度组件：(1) **Bit-Width-Aware I/O-Compute Pipeline**：将传统"按 expert ID 顺序执行 I/O+Compute"改为按 fine-grained bit-width 级重新排序（Figure 9d），利用 MWQ 嵌套结构使低 bit-width 权重可被多个请求复用，减少 I/O 等待时间。(2) **Hottest-Expert-Bit-First (HEBF) 调度算法**：为每个 expert 构建按 bit-width 升序排列的队列 Q_i，每轮从各队列头部弹出元素并选择激活频率最高的（hottest）优先进入 I/O 队列；加载完立即开始计算，使高激活频率 expert 的长计算时间与后续 expert 加载重叠，最小化 I/O-compute bubble。满足三项约束：(6a) 计算在加载完成后开始；(6b) 同 expert 按 bit-width 升序加载以最大化低 bit-width 复用；(6c) 等待时间 = 当前计算完成时间 - 上一计算完成时间 - 当前计算量。(3) **Memory Budget Scheduler (Algorithm 2)**：引入可配置参数 M（GPU 分配给 expert 的内存上限），每层检查当前所需内存是否超出预算 — 若超出则逐层释放高 bit-width 权重（lines 4-6）；若仍不足则释放低 bit-width 权重（lines 7-8）；满足预算后执行 bit-width-aware pipeline 并更新预算。

  实验比较：(a) D2MoE vs EdgeMoE (基于重要性的固定 bit-width + 预加载预测)、Hold-in-Memory-AWQ (INT4 全量 GPU)、MoQE-DynaIO (统一 bit-width + on-demand 加载) 在不同 memory budget M (200MB~2500MB) 下的吞吐量 (tokens/s)，覆盖 LLaMA-MoE-3.5B 和 Mixtral 8×7B 在 Environment 1 和 2 上的结果；(b) D2MoE 扩展到 dense LLM (LLaMA2-13B) 对比固定 INT4 loading 方法的吞吐和峰值内存；(c) 消融实验（Figure 14）：+Router → +MWQ → +HEBF → +Budget 四步累积的吞吐提升分解；(d) 系统开销分析：bit-width router 开销（<0.5% 计算/内存，~1.5% 延迟）、dequantization 开销（4 requests 时 ~20.77% / 32 requests 时 ~5.3%）、parallelism planning 开销。

- 硬件平台是什么，配置是什么。
  Environment 1: NVIDIA RTX 3060 (6GB GPU) + Intel Core i7-11800H (32GB CPU) + Samsung 970 EVO (1TB NVMe, 3.5 GB/s read)。Environment 2: NVIDIA Jetson AGX Orin 64GB (SoC, shared memory) + ARM Cortex-A78AE + Samsung 970 EVO (1TB)。离线预处理: GPU server with 2× NVIDIA RTX A6000。单卡推理，非分布式，无 NVLink/InfiniBand。

- 开源Serving框架是什么。修改了什么。
  D2MoE 是**自研的端侧 MoE Serving 引擎**（~2,500 LOC Python + CUDA），并非基于现有开源 Serving 框架修改。基于 PyTorch，使用 Triton 进行 I/O-compute 并行编程，CUDA kernel 基于 NVIDIA Ampere/Ada Lovelace 架构。论文提到其设计可适配 TensorRT 和 vLLM。

  D2MoE 的 Serving 架构分为两阶段：
  
  **Offline 预处理阶段**（部署前执行一次）：
  - Token-adaptive bit-width router 微调：使用 C4 通用数据集微调每层每个 expert 的 bit-width router
  - MWQ 量化：使用 calibration 数据集（128 random 2048-token segments）对所有 expert 权重执行 MWQ 量化
  - Offline Profiling：测量各 bit-width 的 I/O 延迟 T_io(b_k) 和计算延迟 T_comp(b_k)（data-independent，一次测量可复用）

  **Online 执行阶段**修改：
  - **Bit-Width-Aware I/O-Compute Pipeline**：将传统的"按 expert ID 顺序 I/O → 按 expert ID 顺序 Compute"替换为 fine-grained bit-width 级重排序 + HEBF 调度
  - **Memory Budget Scheduler**：替换无内存限制的专家加载策略，增加动态内存预算检查与释放逻辑
  - **Dequantization Kernel 集成**：在每个 expert 计算前插入 MWQ 反量化步骤

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供公开开源代码仓库。基于论文描述的 D2MoE 引擎架构。

  **D2MoE Serving 框架输入到硬件执行全过程（以 LLaMA-MoE-3.5B, Environment 1, M=1600MB, 32 requests 为例）**：

  ```
  === 系统初始化 ===
  1. 离线预处理完成: bit-width routers 已微调，expert 权重已 MWQ 量化 (b_1=2, b_K=4)
  2. 加载非 expert 参数到 GPU（Attention, LayerNorm, Embedding, Router 等常驻）
  3. Offline profiling 完成: 记录 T_io(INT2), T_io(INT3), T_io(INT4), T_comp(INT2/3/4)
  4. Memory budget M=1600MB 配置完成

  === 逐 Token 推理流程（per MoE layer） ===

  Step 1: Attention (常驻 GPU，无加载开销)
    h = FlashAttention(x)

  Step 2: Dual Routing (GPU compute)
    # Router 1: Original MoE expert routing
    gate_logits = W_gate @ h                    // [E=8] per token
    selected_experts = TopK(Softmax(gate_logits), K=2)  // 选 top-2 experts

    # Router 2: Bit-width routing (per selected expert)
    For each selected expert e_i:
      bw_logits = W_bw_router[e_i] @ h         // [K=3] bit-width logits (INT2/3/4)
      b_k = argmax(Softmax(bw_logits))          // 选择最优 bit-width
    # 输出: [(expert_id, bit_width), ...] for all tokens in batch

  Step 3: Memory Budget Check (CPU scheduler)
    For current layer j:
      total_expert_mem = Σ memory(expert_id, bit_width)  // 本层所有被选 experts 总内存
      If total_expert_mem > M:
        # 释放高 bit-width 权重 (Algorithm 2, lines 4-6)
        For k = K-1 down to 0:
          Free(layer[j-1][k])                  // 释放上一层的高 bit-width experts
          Update M
        # 若仍不足，释放低 bit-width 权重 (lines 7-8)
        If layers[j] > M:
          Free(layer[j-1][1])                  // 释放上层的低 bit-width experts
      # 加载本层所需 experts
      Load and Store(layer[j])

  Step 4: HEBF Scheduling (CPU scheduler)
    # 为每个 expert 构建按 bit-width 升序的队列
    For each expert e:
      Q_e = sorted([(bit_width, count) for activations of expert e], key=bit_width asc)

    # HEBF 算法主循环
    While any Q_e is non-empty:
      candidates = []
      For each expert e with non-empty Q_e:
        (bw, count) = Q_e.front()               // 当前最低 bit-width
        candidates.append((e, bw, count))
      
      # 选激活频率最高的 (hottest expert-bit)
      (e*, bw*, count*) = argmax(count) in candidates
      Enqueue (e*, bw*) to I/O Queue             // 优先 I/O 加载
      Pop from Q_{e*}
      
      # I/O 完成后立即提交到 Compute Queue
      # 高频率 expert 的计算时间长 → 可与后续 expert 的 I/O 重叠

  Step 5: Bit-Width-Aware I/O-Compute Pipeline (GPU + I/O)
    磁盘 → GPU (I/O Stream):     | Load E2(INT2) | Load E5(INT2) | Load E2(INT3) | ...
                                  |<- 重叠 ——>|
    GPU Compute Stream:           | idle | Dequant+FFN(E2,INT2) | Dequant+FFN(E5,INT2) | ...
                                            |<- E5 I/O 与 E2 Compute 重叠 ->|
    
    # INT2 权重被所有选 INT2/3/4 的请求共用（MWQ 嵌套）
    # INT3 权重仅被选 INT3/4 的请求额外加载
    # INT4 权重仅被选 INT4 的请求额外加载
    # → 低 bit-width 权重加载次数远多于高 bit-width → HEBF 优先加载

  Step 6: Expert FFN Computation (GPU)
    For each completed I/O load:
      W_fp16 = MWQ_dequant(Q_W_b1, Q_W_b2, ..., s_b1, s_b2, ..., z_b1)
      # Dequantization: CUDA Core 执行 binary shift 操作
      output = W_fp16 @ h                       // Tensor Core GEMM
      output = output * gate_score              // Gating 权重加权

  Step 7: Combine & Continue
    final_output = Σ routed_expert_outputs
    h = h + final_output                        // Residual
    Continue to next layer

  === 关键约束满足 ===
  - (6a) Compute 仅在 I/O 完成后启动: HEBF 保证 L(s+1,j,k) ≤ C(s,j,k)
  - (6b) 同 expert 按 bit-width 升序加载: Q_e 始终升序排列
  - (6c) 等待时间公式: T_wait = C(s,j,k) - C(s-1,j,k) - B_{j,k}·T_comp(k)
    → HEBF 最小化 T_wait，使高频率 expert 计算时间覆盖后续 I/O
  ```

  **关键性能特征**：
  - Mixtral 8×7B 在 Environment 1 (6GB GPU) 上：EdgeMoE 和 Hold-in-Memory-AWQ 失败（显存不足），D2MoE 达到 38.07 tokens/s
  - LLaMA-MoE-3.5B 在 Environment 1: D2MoE 吞吐比 EdgeMoE 高 1.06×-1.16×，内存减少 33%-53%
  - 随 memory budget 增大，D2MoE 吞吐接近 Hold-in-Memory-AWQ（无 I/O 加载开销的理想上界）
  - 32 requests, M=200MB → 66.45 tokens/s; M=1600MB → 83.14 tokens/s (near-linear scaling)
  - Parallelism planning 开销：虽然随 request 数增长，但占总推理时间比例递减
