## KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction (Softmax-Free Kernel Variant)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  KVzip 在附录 C.3 中提出 softmax-free 重要性评分变体。标准 KVzip 在 Algorithm 1 中使用 Softmax 归一化后的注意力分数作为重要性得分，这需要在前向传播之外额外读取注意力矩阵。Softmax-free 变体通过实现自定义 Triton-based FlashAttention CUDA kernel，直接在 fused attention kernel 内部使用未归一化的 QK product（logits）作为重要性得分，省略 Softmax 归一化步骤，从而将评分步骤嵌入前向传播，消除冗余计算。实验比较 softmax-free 变体与标准 KVzip 在 Retr.KV (SCBench) 上的压缩性能（LLaMA3.1-8B）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB GPU，Bfloat16 精度。使用 Triton DSL 编写 CUDA kernel。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：基于 FlashAttention-2 的 LLM 推理 pipeline。修改点：在 FlashAttention fused kernel 内部，在 Softmax 之前截取 QK product 矩阵中 KV_c 对应部分，直接沿 query 维度取 max 作为重要性得分，绕过 Softmax 归一化和后续的注意力矩阵物化。评分过程原本约占 forward 总时间的 10%，该 kernel 将此 10% 开销消除。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/snu-mllab/KVzip（论文未明确说明 softmax-free kernel 是否在开源仓库中独立提供）
  
  评估原理：标准 FlashAttention 的分块算法在 on-chip SRAM 中计算 QK^T → Softmax → ×V，中间注意力矩阵不写回 HBM。KVzip 需要在 Softmax 之后沿 query 维度取 max，这与 FlashAttention 的逐块计算模式不兼容（需要跨 query 维度的全局 max）。Softmax-free 变体直接跳过 Softmax，将 QK^T logits 作为得分，可在分块计算时直接累积 per-chunk 的最大值，与 FlashAttention 的 online softmax rescaling 逻辑兼容。
  
  Kernel 输入：Query tensor Q ∈ R^{G×H×n_in×d}，Key tensor K ∈ R^{H×(n_c+n_in)×d}（从 KV_c + 当前 input 拼接）。
  Kernel 输出：attention output + importance score S_{l,h,t}（每个分块的 QK^T logits 沿 query 轴 max）。
  
  性能 trade-off：消除 ~10% 评分开销，但压缩比下降约 10%（Figure 15），因未归一化的 logits 不能准确反映注意力权重分布。 by Passing Compressed Context Blocks across GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一个定制化的 FLASHATTN kernel 以及优化的分布式通信调度策略，用于支持 APB 的 approximate attention 机制。具体包括：(1) 修改 FLASHATTN kernel 的 attention mask，以支持 [anchor block, passing block, local context block] 三部分联合的注意力计算，在 tiling 层面正确实现 M' 遮罩；(2) 在每层 Transformer 中内嵌两次 AllGather 通信（分别对压缩后的 K^C 和 V^C），与 QKV 投影和 attention 计算协同调度；(3) Decoding 阶段实现 Gather + MergeScore（online softmax lse 合并）的分布式精确注意力。实验比较了 APB 与 baselines（FLASHATTN、ULYSSES、RINGATTN、MINFERENCE、STARATTN）在多种 context length（32K-512K）下的 wall-time 分解和推理速度。

- 后端平台是什么，配置是什么。
  8× NVIDIA A800-80GB GPU（NVLink 3.0 互联），104 核 Intel Xeon Platinum 8470 CPU，跨机 HDR InfiniBand，CentOS Linux 7 (Core)。GPU 间通信利用 NVLink 进行 AllGather（intra-node），跨节点使用 InfiniBand。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 HuggingFace Transformers 框架（https://github.com/huggingface/transformers）进行推理实验。核心修改：
  1. **FLASHATTN kernel 修改**：修改 attention mask 为 M'，在 tiling 计算中正确实现 [A, P_h, B_h] 的因果/跨块注意力遮罩
  2. **通信调度**：在每层 attention 计算前后插入 AllGather（K^C, V^C），实现通信-计算流水线
  3. **Decoding 阶段**：实现 STARATTN stage-2 的 Gather+MergeScore 分布式解码，各 host 独立计算 partial attention 后合并

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/thunlp/APB

  **评估原理与 Kernel 执行全流程（以单层 Transformer Prefill 为例）**：

  ```
  输入：每 host 上的 H_a（anchor hidden states）和 H_h（local context hidden states）
  输出：H_a^out, H_h^out, K_a, K_h, V_a, V_h

  Step 1: QKV Projection
    [Q_a, Q_h], [K_a, K_h], [V_a, V_h] = layer.qkv_proj([H_a, H_h])
    // 在 A800 GPU 上利用 cuBLAS 执行矩阵乘法

  Step 2: KV Cache Compression (retaining heads)
    s_1, ..., s_{l_b} = layer.R([Q_h, K_h, V_h])  // MLP 推理
    indices = ArgTop-l_p(s_1, ..., s_{l_b})       // GPU top-k
    K_h^C, V_h^C = K_h[indices], V_h[indices]      // gather 操作

  Step 3: AllGather Communication
    K_{1:H}^C = AllGather(K_h^C)   // NCCL AllGather on NVLink/IB
    V_{1:H}^C = AllGather(V_h^C)

  Step 4: Construct Passing Block
    K_p = concat(K_1^C, ..., K_{h-1}^C)   // host h 只取前序 host 的压缩块
    V_p = concat(V_1^C, ..., V_{h-1}^C)

  Step 5: Modified FLASHATTN Attention
    // 在 SRAM 中分 tile 计算，带 M' attention mask
    // K,V layout: [K_a, K_p, K_h], [V_a, V_p, V_h]
    // Q layout: [Q_a, Q_h]
    A_a, A_h = flash_attn_with_mask([Q_a, Q_h], [K_a, K_p, K_h], [V_a, V_p, V_h], M')

  Step 6: FFN（仅 anchor 和 local context，不含 passing block）
    H_a^out, H_h^out = FFN(A_a, A_h)

  Step 7: Discard passing blocks
    // P_h 在 attention 计算后丢弃，不缓存
  ```

  **Wall-time 分解（128K 输入，Llama-3.1-8B，8 hosts，每 Transformer Block）**：
  - QKV Projection: 4.01 ms
  - Retaining Head: 1.72 ms
  - Communication (AllGather): 0.62 ms
  - Attention (Modified FLASHATTN): 34.07 ms
  - O Projection: 2.67 ms
  - FFN: 30.76 ms
  - Others: 6.33 ms
  - **Total/Block: 80.18 ms**

  **速度评估指标**：
  $$speed = \frac{\#input\_tokens + \#output\_tokens}{prefill\_time + decoding\_time}$$

  **关键性能数据（128K，Llama-3.1-8B，RULER avg）**：
  - FLASHATTN: 4,086 tok/s（单 GPU，OOM at >128K）
  - ULYSSES: 26,200 tok/s
  - RINGATTN: 17,822 tok/s
  - MINFERENCE: 4,545 tok/s（单 GPU）
  - STARATTN: 29,600 tok/s
  - APB: **37,575 tok/s**（最高）

  **通信开销分析**：APB 的 AllGather 通信仅占每 block 总时间 0.62 ms（<1%），远小于 RINGATTN 的 P2P ring 通信（18.40 ms，占 ~9%），这是因为 APB 只传输 top-l_p 个压缩 KV pair（l_p=2K vs 原始 l_b=16K）。
