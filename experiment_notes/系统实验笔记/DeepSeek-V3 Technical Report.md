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
