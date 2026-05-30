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
