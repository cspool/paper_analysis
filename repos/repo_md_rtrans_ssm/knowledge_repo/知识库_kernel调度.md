## Tensor Parallelism (in SSM/Transformer Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tensor Parallelism (TP) 由Shoeybi et al.(2019)在Megatron-LM中提出，是一种模型并行策略：将单层内的权重矩阵按列或行切分到多个GPU上，每层计算后通过all-reduce同步部分结果。对于Transformer的self-attention层，TP沿attention heads维度切分Q/K/V权重；对于MLP层，沿列切分第一个线性层，沿行切分第二个线性层。每层仅需1次all-reduce（在MLP output或attention output之后）。论文中所有模型使用TP size=4进行训练。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba与Mamba-2的TP通信差异：
```
// Transformer TP (1 all-reduce per layer):
// Attention: Q,K,V按head切分 → 各GPU独立计算attention → all-reduce(output)
// MLP: column-parallel Linear1 → GELU → row-parallel Linear2 → all-reduce

// Mamba TP (2 all-reduces per layer):
// - 第1次all-reduce: 在input projection (Linear_proj) 之后
// - 第2次all-reduce: 在SSM scan & gating之后, output projection之前
// 原因: Mamba的SSM scan需要完整的hidden dim进行计算

// Mamba-2 TP (1 all-reduce per layer):
// - 利用SSD的多头结构, 类似attention沿head维度切分
// - 仅需在output projection后1次all-reduce
// 约束: 必须使用GroupNorm(而非LayerNorm)作为内部归一化
// GroupNorm: 沿hidden dim分组归一化, 各组独立计算
// 要求group_size > 256以保证统计量精度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Megatron-LM中通过--tensor-model-parallel-size参数设置。论文训练使用TP=4, DP=256, 共1024 H100 GPUs。TP的核心trade-off：减少单GPU显存需求，但引入通信开销（all-reduce延迟）。Mamba-2通过SSD的多头结构将TP通信降至1次all-reduce，与Transformer持平，而Mamba需要2次，增加了通信开销。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models

---

## Selective Scan (Parallel Scan in Mamba/Mamba-2)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Selective Scan是Mamba/Mamba-2中实现SSM状态递归更新的核心计算。在标准RNN中，隐状态更新h_t = A_t * h_{t-1} + B_t * x_t是严格串行的。Mamba利用该递归的线性性质，将其视为binary associative operator作用于有序序列上的prefix sum问题，通过parallel scan（parallel prefix sum的推广）算法将O(L)的串行时间复杂度降低为O(log L)并行步骤。Mamba的selective scan因A_t依赖输入（非时不变），需使用更复杂的associative scan。Mamba-2则利用SSD对偶性将scan重构为分块矩阵乘法（chunked parallelism），chunk内使用高效MatMul，chunk间使用recurrent传递状态，充分利用Tensor Core。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Chunked Parallel Scan（Mamba-2 SSD kernel）伪代码：
```
// Input: A_bar ∈ R^{B×L×H×P}, B_bar ∈ R^{B×L×H×P}, C ∈ R^{B×L×H×P}, V ∈ R^{B×L×H×P}
// H=head数, P=head_dim, L=seq_len
// chunk_size = min(64, L)

num_chunks = ceil(L / chunk_size)

// Phase 1: Intra-chunk (parallel across all chunks via MatMul)
for each chunk i in parallel:
    // Chunk内SSM scan, 等价于:
    // M_chunk = L_chunk ◦ (C_chunk @ B_chunk^T)  // 半可分离矩阵
    // Y_chunk = M_chunk @ V_chunk  // MatMul, 利用Tensor Core
    Y_local[i], h_final[i] = chunk_ssd(Q_chunk, K_chunk, V_chunk, h_init=0)

// Phase 2: Inter-chunk (recurrent, sequential)
h_running = 0
for i in 0..num_chunks-1:
    // 用前一chunk的最终状态调整当前chunk输出
    Y[i] = Y_local[i] + correction_term(h_running, C_chunk, V_chunk)
    h_running = h_final[i] * exp(-Δ_sum) + correction_state(h_running, ...)
```

Mamba-1的selective scan无chunked策略，纯parallel scan比Mamba-2慢约8x。

术语一般如何实现？如何使用？
Mamba开源实现提供CUDA kernel（https://github.com/state-spaces/mamba）。Mamba-2实现使用Triton kernel（chunked scan）。论文使用Megatron-LM中的实现。核心优化原则：对长序列使用chunked策略并行化；保证memory coalescing访问；避免thread divergence。在GPU上，Mamba-2的SSD scan比Mamba-1 scan快8倍。

SAMBA 论文使用 Mamba 的硬件感知并行扫描算法实现高效训练，同时在混合架构中与 FlashAttention 2 配合：SWA 层（窗口=2048）使用 FA2——训练速度与 Mamba 的 selective parallel scan 在 seqlen=2048 时相当（基于 Gu & Dao 2023 测量）。推理时 Mamba 层 O(1) 状态更新 + SWA 层 O(window) 计算，总体解码仍为线性时间复杂度。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## Model FLOPs Utilization (MFU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Model FLOPs Utilization (MFU) 是衡量大规模模型训练中GPU计算效率的指标，由PaLM论文(Chowdhery et al., 2022)引入，Korthikanti et al.(2022)在Megatron-LM中使用。定义：MFU = (单次forward+backward的理论FLOPs / 每次迭代时间) / (GPU理论峰值FLOPs/s × GPU数量)。分子是实际达到的FLOPs/s（模型理论FLOPs除以实测迭代时间），分母是硬件理论峰值FLOPs/s总和。MFU考量了模型计算（而非全部操作如attention的带宽瓶颈部分）对硬件的利用率。相比于传统Hardware FLOPs Utilization(HFU)，MFU通常更高，因为模型FLOPs仅计入了矩阵乘法等主力计算。

从kernel调度角度拆解术语：
```
MFU计算流程：
1. 计算模型理论FLOPs = f(params, seq_len, hidden_dim, num_layers)
   - Transformer: 主要来自QKV投影、attention MatMul(忽略softmax)、MLP MatMul
   - Mamba-2: 主要来自input projection、SSD scan的MatMul、output projection
2. 测量迭代时间t_iter (forward+backward)
3. 查找GPU理论峰值: H100 SXM BF16 = 989.8 TFLOPS/GPU
4. MFU = (model_FLOPs / t_iter) / (989.8e12 * num_GPUs)
```

论文结果：8B Mamba-2-Hybrid在1024 H100 GPUs上MFU=29.9%，接近同规模Transformer的30.7%。这表明Hybrid模型在Megatron-LM中的实现效率与成熟Transformer实现相当。

术语一般如何实现？如何使用？
Megatron-LM和PaLM论文提供MFU计算脚本。通常TP/DP配置、micro batch size、activation checkpointing等因素影响MFU。论文中TP=4, DP=256, micro_batch=4, global_batch=1024。达到30% MFU对8B参数规模的大模型训练是合理的。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models

---

## cu_seqlens (Cumulative Sequence Lengths / 累积序列长度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cu_seqlens是Mamba库（https://github.com/state-spaces/mamba）中用于处理变长序列的关键机制，类似Flash Attention的`varlen_fwd`/`varlen_bwd` API。它将多个不同长度的序列pack到单个flattened tensor中（batch_size=1），用cumulative sequence lengths数组（shape=[num_seqs+1]）标记每个序列的起始和结束位置。例如batch中有3个序列长度分别为5, 10, 3，则cu_seqlens=[0, 5, 15, 18]。Mamba的selective scan CUDA kernel在cu_seqlens定义的每个序列边界处自动重置hidden state，确保不同序列间的SSM state不交叉污染。相比传统的padding方式，cu_seqlens避免了浪费在padding token上的计算，在真实数据集上可获2-4x端到端加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
cu_seqlens在Mamba selective scan kernel中的使用：
```
// Mamba selective scan kernel with cu_seqlens support
Input: x ∈ R^{total_tokens×D}    // flattened batch
       cu_seqlens ∈ Z^{B+1}       // cumulative lengths

for batch_idx = 0 to B-1:
  seq_start = cu_seqlens[batch_idx]
  seq_end = cu_seqlens[batch_idx + 1]

  // Reset hidden state at each sequence boundary
  h = zeros(D, N_state)

  // Process only within this sequence
  for t = seq_start to seq_end-1:
    h = A[t] * h + B[t] * x[t]^T
    y[t] = C[t] @ h
```

在Attamba中的用法：Attamba利用cu_seqlens处理变长chunk——不同chunk可能大小不同（如Random chunking），cu_seqlens指定每个chunk的起止位置，SSM在每个chunk边界重置hidden state。Cyclic chunking中不同层的不同chunk边界也通过修改cu_seqlens偏移量实现。整个过程中不需要reshape或padding输入序列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba官方实现内建cu_seqlens支持（GitHub PR #244）。使用时传入cu_seqlens参数到Mamba block的forward函数。注意事项：(1) 仅支持训练，推理/生成场景尚未支持；(2) Bidirectional Mamba需要额外的reverse_cu_seqlens；(3) Mamba-2同样支持cu_seqlens用于序列并行。Attamba利用cu_seqlens实现了无需reshape的灵活chunking，是降低实现复杂度的关键设计选择。

涉及论文标题：
- Attamba__Attending_To_Multi-Token_States

---

## SRAM-Resident WKV State Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
**原始 RWKV 论文（EMNLP 2023）**首次引入 custom CUDA kernel 用于 WKV 计算，以解决串行扫描在标准深度学习框架中的低效问题。Eagle/Finch 训练时沿用的 custom CUDA kernel，核心设计选择：**不沿时间维度并行**（尽管 WKV 可通过 associative scan 做 time-parallel），而是**沿非时间维度并行**，将 recurrent state 操作保持在 GPU SRAM 中。原始 RWKV 的 WKV kernel 面向向量 state（head=1），计算量较小（仅 d 维逐元素操作）；Eagle/Finch 的矩阵 state（head=64）大幅增加了 SRAM 驻留需求。原理是：time-parallel 的 associative scan 虽并行度高，但每次迭代需要从 HBM 读取中间结果→SRAM 计算→写回 HBM，memory bandwidth 成为瓶颈；非时间维度并行将 state s∈R^{(D/h)×(D/h)} 驻留在 SRAM 中，每时间步仅读写 token 输入/输出（远小于 state 矩阵），memory 开销显著降低。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Finch WKV kernel 执行流程：
```
// 配置: batch=8, D=4096, head=64, h=64
// State: s ∈ R^{B×h×64×64} = 8×64×64×64×4B ≈ 8MB (SRAM-resident)

For t = 0 to seq_len-1:
  // 沿非时间维并行: batch和head维度
  parallel_for (b, head):
    // 1. 从HBM加载当前token (小数据量): x_t ∈ R^D
    //    → ddlerp Token Shift (SRAM内计算)
    //    使用LoRA: tanh(x@A)@B, A∈R^{D×32}, B∈R^{32×D}
    
    // 2. WKV计算 (SRAM内, state在SRAM中驻留):
    k_t, v_t ← Token Shift + Linear
    wkv = (u⊙k_t^T)⊗v_t + s[b,head]    // 矩阵乘: k_t^T·v_t
    
    // 3. 更新state (仍在SRAM):
    w_t = exp(-exp(lora_d(ddlerp_d(·))))
    s[b,head] = diag(w_t)·s[b,head] + k_t^T@v_t
    
    // 4. 输出 (receptance + SiLU gate):
    o_t = LayerNorm(r_t @ wkv)
    o_t = SiLU(g_t)⊙o_t
    
  // 5. 写入HBM: o_t ∈ R^{B×h×D/h}
```
对比 time-parallel (associative scan) 方案:
```
// 需要反复HBM↔SRAM传输state中间结果
For each pair of adjacent sequence elements (parallel scan tree):
  Load s_left from HBM → SRAM → merge s_left+s_right → write s_merged to HBM
// 每层scan tree depth=log(T), 每层都需全量state读写
// 总HBM传输量: O(T×log(T)×D²/h)  vs SRAM方案: O(T×D)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源在 RWKV-LM (https://github.com/RWKV/RWKV-LM)。纯 PyTorch 也有 time-parallel 实现（基于 GLA 方法，https://github.com/RWKV/RWKV-infctx-trainer）。性能：16k 序列 Finch kernel 比 Flash Attention v2 快 4.2×，比 Mamba 省 17% 内存、比 Flash Attention 省 40% 内存（A100 80GB, batch=8, D=4096, head=64）。论文指出该 kernel 还有进一步优化空间（algorithmic improvements），留待未来工作。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

---

## IO-aware Prefill CUDA Kernel for Linear Attention (Based/ThunderKittens)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IO-aware prefill CUDA kernel是ThunderKittens框架中为Based线性注意力架构设计的自定义CUDA kernel，通过warp-register分区存储矩阵值recurrent state（KV-state ∈ R^{d×d̃}，d̃≈273 for Taylor 2nd-order feature map）避免HBM↔SRAM反复传输，实现IO最优的prefill计算。核心策略：将KV-state矩阵分片存储在各warp的register file中（而非global memory），在prefill阶段沿序列维度扫描时仅在register中累加更新state，最终一次性写回HBM。JRT论文扩展此kernel支持Prefix Linear Attention (PLA)：第一次调用fnbased(k_e,v_e)用非因果sum计算encoder KV-state存于寄存器A0/A1/A2（对应Taylor 0/1/2阶项），第二次调用fnbased(q_d,k_d,v_d)从该register状态续算decoder输出写SRAM→HBM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ThunderKittens Based kernel寄存器管理 (per warp):
Register A0: stores 0th-order KV-state contribution  // Σ 1·v
Register A1: stores 1st-order KV-state contribution  // Σ k·v (linear term)
Register A2: stores 2nd-order KV-state contribution  // Σ vec(k⊗k)·v (quadratic term)

// JRT-RNN扩展 (Algorithm 2 in JRT paper):
1. 初始化SRAM buffer和register fragments
2. fnbased(k_e, v_e):  // Encoder prefill
   - 使用非因果sum (而非causal cumsum)
   - 不乘queries (与原Based kernel不同)
   - KV-state = Σ_{j=1}^{M} (k_e[j]^T v_e[j])
   - 结果存于寄存器A0/A1/A2
3. fnbased(q_d, k_d, v_d):  // Decoder prefill  
   - 从encoder初始化的register state续算
   - KV-state_dec = encoder_state + Σ_{j=1}^{i} (k_d[j]^T v_d[j])
   - K-state_dec = encoder_k_sum + Σ_{j=1}^{i} k_d[j]
   - y_i = (q_d[i]·KV-state_dec) / (q_d[i]·K-state_dec) 写入SRAM
4. Store y from SRAM → HBM

// FLOPS (per layer, B=batch, N=seqlen, H=heads, D=head_dim, d=model_dim):
// Causal LA: 2BNHD(feature map) + 4BNHdD(KV dot+cumsum+Q dot+D sum)
// PLA add:   BMHD(k_e feature map) + 3BMHdD(k_e·v_e + D sum + state merge)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/HazyResearch/ThunderKittens（Based kernel），PLA扩展：https://github.com/HazyResearch/prefix-linear-attention。性能（H100）：N=32768/B=16时JRT-RNN CUDA 5.6ms vs FA2 107.8ms (19.2× faster)，vs FLA Triton 123.7ms (22.0× faster)。JRT-Prompt CUDA (2N prefill): 9.0ms → 11.9× > FA2, 13.7× > FLA。PLA decode每token O(1)无修改。ThunderKittens framework仅~282-316 lines per kernel (vs Triton 89-104)，实现14×于FLA Triton的线性注意力加速。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## Lightning Attention 2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Lightning Attention 2 由 Qin et al. (2024) 提出，是第一个在实际中实现线性注意力理论计算优势的 Triton kernel 实现。核心采用 tiling 策略将注意力计算分为两部分：(1) Intra-block：使用传统 softmax attention（利用局部性）；(2) Inter-block：应用线性注意力的右乘（right product）技巧——先计算 KV 累积值再与 Q 相乘。通过 IO-aware 设计将 KV state 保持在 SRAM 中以最小化 HBM↔SRAM 数据传输。在因果（causal/autoregressive）设置下，解决了之前线性注意力实现中 cumsum 操作无法发挥理论优势的问题。性能：8K context 比 FlashAttention-2 快 1.5×，32K context 快 3×；训练速度与序列长度无关（恒定），推理 per-token 速度也与 context 长度无关。

在 SUPRA 中的用法：SUPRA 使用 Lightning Attention 2 的 Triton kernel（`lightning_attn_ops`）进行训练时的并行线性注意力计算。输入为 RoPE 后的 MLP kernel 特征 q/k、scale 后的 k 和 v，以及 decay slope tensor。Kernel 内部处理带衰减的线性注意力：O_i = Σ_{j=1}^{i} γ^{i-j} (φ(q_i)·φ(k_j)) v_j，通过 tiling 分解为 intra-block 和 inter-block 计算。这使 SUPRA 的训练 throughput 达到约 4300 tokens/s/GPU（7B 模型，H100）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Lightning Attention 2 tiling 策略伪代码：
```
# 将序列分为 chunks of size B
# Intra-block (块内): 使用左乘 (QK^T)·V (softmax attention style)
# Inter-block (块间): 使用右乘 K^T·V 累积 (linear attention style)

Input: Q, K, V ∈ R^{N×d}, decay slope s ∈ R^h
KV_state = 0  # 存于 SRAM

For i in 0, B, 2B, ..., N-B:
    Q_block = Q[i:i+B], K_block = K[i:i+B], V_block = V[i:i+B]
    
    # Intra-block: 标准 causal attention (左乘, block 内)
    Attn_block = causal_softmax(Q_block @ K_block^T / √d) @ V_block
    
    # Inter-block: 线性注意力累积 (右乘, 跨 block)
    KV_state_decayed = KV_state * exp(-s * B)
    Linear_block = Q_block @ KV_state_decayed  # O(B·d²)
    
    # 更新 KV state (存回 SRAM)
    KV_state = KV_state_decayed + K_block^T @ V_block  # O(d²)
    
    # 合并输出
    O[i:i+B] = Attn_block + Linear_block

Return O
```

关键设计：KV_state 在 SRAM 中持续累加（不再写回 HBM），仅 Q/K/V blocks 从 HBM 读取和 O blocks 写回 HBM。Triton 实现利用 `tl.dot` 进行 block 内 MatMul，利用 Triton 的 automatic memory coalescing 优化 HBM 访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/OpenNLPLab/lightning-attention (Triton kernel)。SUPRA 在其 OpenLM fork 中集成了该 kernel，调用方式为 `lightning_attn_ops(q, k * qk_scale, v, slope_tensor)`。Lightning Attention 2 的局限性：(1) 仅支持固定 decay 的线性注意力（无法直接处理 data-dependent decay 如 Finch 的 w_t）；(2) block size 需 tuned 以获得最优性能。适用于使用固定或 learnable decay 向量的线性注意力模型（如 RetNet、SUPRA、TransNormer）。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## Fused Generation Kernels for Recurrent LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Generation Kernels for Recurrent LLMs 是将递归神经网络的多个独立 GPU kernel 调用融合为单个 kernel 的优化技术。以 xLSTM 7B 的 mLSTM cell 为例，其自回归生成时的 recurrent 公式（Eq. 2-9）涉及 outer product（v_t ⊗ k_t）、多个 dot product、max 操作、exp 操作和 pointwise 乘法——在标准实现中每个操作都是一个独立 GPU kernel 调用。每个 kernel 需要从 HBM 加载输入并将输出写回 HBM，大量慢速内存操作成为瓶颈。Fused kernel 将所有操作在单个 kernel 中完成，中间结果（gate values、outer product 部分和、normalizer update 等）保持在 GPU SM 的 SRAM/Register File 上，仅最终 hidden state h_t 和更新后的 recurrent state (C_t, n_t, m_t) 写回 HBM。xLSTM 7B 的 Triton-based fused generation kernel 开源在 https://github.com/NX-AI/mlstm_kernels。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Fused mLSTM Recurrent Kernel 伪代码 (单次 timestep)
// GPU Grid: 1 block per head (共 N_head 个 block 并行)
// 每个 block 在 SM SRAM 上执行:

输入: x_t ∈ R^{d_model}, state_{t-1} = (C_{t-1}, n_{t-1}, m_{t-1}) (均在 HBM)
输出: h_t ∈ R^{d_model}, state_t = (C_t, n_t, m_t) (写回 HBM)

// 1. 从 HBM 加载 state 到 SRAM
C_prev = load_from_hbm(C_{t-1})   // d_qk × d_hv floats
n_prev = load_from_hbm(n_{t-1})   // d_qk floats
m_prev = load_from_hbm(m_{t-1})   // 1 float

// 2. Gate computation (在 SRAM)
q = W_q @ x_t    // d_qk
k = W_k @ x_t    // d_qk
v = W_v @ x_t    // d_hv
i_tilde = softcap_15(w_i^T @ x_t + b_i)    // scalar
f_tilde = softcap_15(w_f^T @ x_t + b_f)    // scalar

// 3. State update (全部在 SRAM, 不写 HBM)
m_cur = max(log_sigmoid(f_tilde) + m_prev, i_tilde)
f = exp(log_sigmoid(f_tilde) + m_prev - m_cur)
i = exp(i_tilde - m_cur)

// 4. Memory update (outer product in SRAM)
C_cur = f * C_prev + i * (v ⊗ k^T)  // rank-1 update
n_cur = f * n_prev + i * k

// 5. Hidden state retrieval
q_norm = q / sqrt(d_qk)
h_tilde = C_cur^T @ q_norm / max(|n_cur^T @ q_norm|, exp(-m_cur))
o = sigmoid(W_o @ x_t + b_o)
h = o ⊙ LayerNorm(h_tilde)

// 6. 写回最终结果到 HBM
store_to_hbm(h)      // d_model floats
store_to_hbm(C_cur)  // d_qk × d_hv floats (next step state)
store_to_hbm(n_cur)  // d_qk floats
store_to_hbm(m_cur)  // 1 float

// 关键优化: C_prev, C_cur 的加载/存储在 SRAM 内完成
// outer product v⊗k^T 直接用 register tile 在片上计算
// 相比 unfused: 中间 C_prev 和 C_cur 的多次 HBM 读写被消除
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源 Triton kernel 实现：https://github.com/NX-AI/mlstm_kernels
  - `recurrent` mode: 用于自回归推理的 fused step kernel
  - `chunkwise` mode: 用于训练的 chunkwise-parallel kernel (TFLA)
  - `parallel` mode: 二次复杂度 attention-like kernel（用于短序列或验证）
- 在 HuggingFace transformers 中使用：加载 xLSTM 7B 模型后，`model.generate()` 自动调用 fused recurrent kernel
- PyTorch 集成：`torch.compile` + CUDA Graphs 进一步减少 kernel launch overhead
- 适用范围：任何具有固定大小 recurrence state 的线性 RNN（mLSTM、Mamba、RWKV、GLA 等）均可受益于此类 fused kernel

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference
