# 知识库_算法pipeline

## DLRM (Deep Learning Recommendation Model)

术语是什么？
DLRM（Deep Learning Recommendation Model）是Meta提出的推荐系统基础架构，将推荐任务分解为sparse feature处理（通过embedding tables将categorical inputs映射到dense vectors）和dense feature处理（通过MLP处理continuous features），两者在interaction layer中通过feature crossing（如dot product/Factorization Machine）结合后送入final MLP输出prediction。DLRM是Meta ads ranking的生产基础架构，服务每天超过数百trillion次推理。

从算法pipeline角度拆解术语：
传统DLRM pipeline：
```
Input Features:
  - Sparse features: categorical IDs (post ID, page ID) --[embedding lookup]--> dense vectors
  - Dense features: continuous values (age, CTR, engagement) --[MLP]--> dense vectors
  
Interaction Layer:
  - Pairwise dot product between all feature pairs: XX^T
  - Factorization Machine (FM): low-rank approximation (X · (X^T Y)) reducing O(N²D) → O(NKD)
  
Output: Click-through rate / engagement prediction
```

近年来DLRM architecture evolution带来了新的computational patterns：
- **HSTU (Hierarchical Sequential Transduction Unit)**: 将user history作为jagged tensor序列处理，引入Transformer-like attention机制
- **InterFormer**: bidirectional information flow between non-sequential (user demographics) and sequential (browsing history) features，通过Personalized FeedForward Network (PFFN)
- **WuKong**: 引入Optimized FM——通过learnable projection matrix Y降低interaction complexity
- **Generative Recommendation (OneRec)**: 将推荐建模为sequence generation，使用RQ-VAE/RQ-Kmean将continuous embeddings量化为discrete semantic IDs供LLM处理

这些新架构引入10-100× per-request complexity increase vs传统DLRM，并需要specialized attention kernels、jagged tensor operations和quantization primitives。

术语一般如何实现？如何使用？
Meta生产环境使用多阶段ranking pipeline：Retrieval (millions→10K-100K candidates, low-complexity models) → Early-stage ranking (thousands→hundreds candidates, moderate complexity) → Late-stage ranking (hundreds→final ranking, heavyweight models up to 2 GFLOPS/sample)。每个阶段有不同的kernel优化需求：retrieval优先throughput at large batch sizes；late-stage ranking需要compute-intensive fused interactions under sub-100ms latency；sequence models需要jagged tensor operations。KernelEvolve通过在异构硬件上自动生成优化kernel来服务这些diverse requirements。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## Data Preprocessing Operators in Recommendation Systems

术语是什么？
Data Preprocessing Operators是推荐系统推理pipeline中的前置数据转换算子族，负责将raw features转换为model-ready inputs，在model inference的latency-critical path上同步执行。KernelEvolve论文识别的三类核心preprocessing transformation：(1) Dense normalization——BoxCox/Logit统计变换、one-hot encoding、linear scaling (shift + multiplication)；(2) Sparse processing——top-k selection with truncation、cryptographic hashing mapping IDs to embedding table indices、type downcasting (int64 → int32)；(3) Feature derivation——bucketizing continuous values to categorical bins、set operations across multiple sparse lists、n-gram hashing for text features。

从算法pipeline角度拆解术语：
论文给出了三个具体preprocessing kernel的算法流程：

**MapIdTransform**：将sparse high-cardinality categorical IDs映射为dense consecutive integers for embedding lookup。
```
Algorithm:
Input: values V, sorted mapping M
For each v in V:
  1. Binary search: idx = bucketize(v, M)  # find insertion index
  2. Clamp: idx = min(idx, |M| - 1)
  3. Validate: if M[idx] == v → output idx + 1; else → output 0 (unknown)
Example: V=[100,300,500,200,999], M=[100,200,300,400,500] → output=[1,3,5,2,0]
```

**MBDT (MergeBucketizedDense Transform)**：将continuous features映射到discrete bin indices for embedding lookup。
```
Algorithm:
Input: X ∈ R^{F×B}, border lists {B_f} for each feature f
For each feature f, batch element i:
  1. For each border value b_k in B_f:
      if X_{f,i} < b_k → bin = k (first match)
  Output: Y_{f,i} = bin_index + feature_offset
Example: feature 0 borders [0.3, 0.6], values [0.1, 0.4, 0.8] → bins [0, 1, 2]
```

**Batch Event Truncate**：截断sequence learning中的event-based features (EBF)。
```
Input: nested jagged tensors — outer_lengths, inner_lengths, values — for multiple features
Operation: per-user, per-feature, truncate to top-N events (coordinate across all features simultaneously)
Output: truncated nested tensors preserving multi-feature consistency
```

术语一般如何实现？如何使用？
Preprocessing operators在训练中由distributed workers执行，在推理中嵌入model module内同步执行（preprocessing latency直接影响end-to-end inference response time）。KernelEvolve通过自动生成融合Triton kernel（将多个PyTorch operator融合为单个accelerator launch，消除intermediate tensor materialization和host-device synchronization）优化这些operators。MapIdTransform在MTIA v2i上实现3.28-4.07× speedup；MBDT实现2.94-9.25× speedup；Batch Event Truncate batched kernel实现1.4-14.5× speedup（vs per-feature sequential baseline）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## Factorization Machine (FM) and Optimized FM

术语是什么？
Factorization Machine (FM) 是推荐系统中的特征交互建模方法，通过分解的隐向量（latent vectors）捕获sparse categorical features之间的pairwise interactions。标准FM计算XX^T（O(N²D) complexity）。Wukong的Optimized FM引入learnable projection matrix Y ∈ R^{N×K} (K << N)，利用associativity将计算重组为out = X · (X^T Y)，将复杂度从O(N²D)降低至O(NKD)。这是通过low-rank approximation实现的——用N×K投影替代N×N全pairwise matrix。

从算法pipeline角度拆解术语：
```
Standard FM (pairwise dot product):
  out_{ij} = <x_i, x_j> for all feature pairs (i,j)
  Complexity: O(N²D) — prohibitive for thousands of features

Optimized FM (Wukong):
  Step 1: X^T Y = torch.bmm(x.permute(0,2,1), y)  # (B,D,N) @ (B,N,K) = (B,D,K)
  Step 2: out = torch.bmm(x, xty)                   # (B,N,D) @ (B,D,K) = (B,N,K)
  Complexity: O(NKD) — reduced from O(N²D) since K << N
  
  Production shapes (WuKong variant): (B, N, D, K) ∈ {
    (1024, 24, 224, 2198), (1024, 40, 224, 448), (1024, 48, 224, 448)
  }
```

KernelEvolve将两步bmm融合为单个Triton kernel：X^T Y intermediate result保持在SRAM中，消除HBM round-trip（从2次load+2次write减少到1次load+1次write），在production shapes上实现2-4× speedup。

术语一般如何实现？如何使用？
Optimal FM是WuKong recommendation model的核心primitive。KernelEvolve生成的fused kernel通过shape-specific tiling（tile尺寸适配SRAM容量以保证full computation chain on-chip）和cross-operation tile reuse（同一tile的load完成两次matmul后才写回HBM）实现优化。当feature count N增大到tiling overhead超过fusion benefit时（N > 64），系统自动fallback到PyTorch unfused baseline。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## PFFN (Personalized FeedForward Network) in InterFormer

术语是什么？
PFFN（Personalized FeedForward Network）是InterFormer推荐模型架构中的个性化前馈网络组件，用于实现sequential features（浏览历史）和non-sequential features（用户人口统计）之间的bidirectional information flow。PFFN由五个顺序操作组成：(1) batched matrix multiplication with bias (FFN layer 1)；(2) GELU activation；(3) RMSNorm（root-mean-square normalization）；(4) FFN layer 2 (batched matrix multiplication with bias)；(5) 最终RMSNorm。处理tensor X ∈ R^{B×N×D}，权重矩阵W1 ∈ R^{B×D×K}和W2 ∈ R^{B×K×D}。

从算法pipeline角度拆解术语：
```
PFFN module forward pass:
  Input: X ∈ R^{B×N×D}, W1 ∈ R^{B×D×K}, W2 ∈ R^{B×K×D}
  
  Step 1: H1 = X @ W1 + b1              # FFN layer 1: batch matmul
  Step 2: H2 = GELU(H1)                  # Activation
  Step 3: H3 = RMSNorm(H2)               # Normalization
  Step 4: H4 = H3 @ W2 + b2              # FFN layer 2: batch matmul
  Step 5: Output = RMSNorm(H4)           # Final normalization
  
  Production shapes: (B, N, D, K) ∈ {
    (1024, 200, 256, 160), (1024, 200, 192, 96),
    (1024, 400, 256, 160), (1024, 150, 96, 192)
  }
```

PyTorch baseline使用torch.compile生成两个独立kernel：(1) extern_kernels.bmm（单pass：load→compute→write），(2) triton_per_fused_rms_norm_add_gelu（两pass：pass1 load+bias+RMSNorm statistics accumulation，pass2 reload+normalization应用）。总计3次memory round-trips。

KernelEvolve生成single-pass fused kernel：tile加载一次→完成全部5个operations（matmul+bias+GELU+RMSNorm+matmul+bias+RMSNorm）→写回HBM，仅需1次load+1次write per tile。Production shapes上实现1.2-2.6× speedup。

术语一般如何实现？如何使用？
PFFN fused kernel通过shape-specific tiling和cross-operation tile reuse实现优化。对于D∈[96,256]、K∈[96,256]的production shapes，tile尺寸适配SRAM容量以保证full operator chain on-chip execution。Batch size增大时speedup从2.0-2.6× (B≤256) 收敛到1.2-1.4× (B>512)，因为更大的batch amortizes了kernel launch overhead。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## W4A8 Quantization (Weight 4-bit, Activation 8-bit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A8量化是一种混合精度量化方案：模型权重量化为4-bit整数（UINT4），激活值保持8-bit整数（INT8），推理时执行非对称精度GEMM（Asymmetric GEMM）。相比W8A8，W4A8将权重内存占用减半，降低内存带宽需求，在memory-bound small-batch场景下有优势，并将memory-to-compute转折点batch size减半（H100上从300降至150）；相比W4A16（FP16 activation），W4A8使用INT8 Tensor Core MMA提供更高计算吞吐；相比W4A4，W4A8保持8-bit激活避免激进量化带来的显著精度损失。但W4A8的dequantization（UINT4→INT8）需在GEMM main-loop中通过CUDA Cores完成，若实现不高效会成为瓶颈，使实际性能远低于roofline预测。

从算法pipeline角度拆解术语，给出具体例子。
W4A8 GEMM量化pipeline（两级量化架构）：
```
离线:
  FP16 weight → per-channel quant → INT8 ([-119,119]) → per-group quant → UINT4

在线推理 (main loop per K-tile):
  1. Load UINT4 weight tile from GMEM → SMEM → RF
  2. Unpack 4-bit → 8-bit (register-level)
  3. Dequantize UINT4 → INT8 on CUDA Cores (核心瓶颈)
  4. WGMMA INT8 MMA on Tensor Cores: C += A_int8 × W_int8
  5. Epilogue: INT32 → FP16 (per-channel scale)
```

dequantization开销由三个因素决定：(a) 每元素指令数α，(b) CUDA Cores吞吐Φ_CUDA（远低于Tensor Cores），(c) 权重矩阵大小N×K。在H100上，为与weight loading重叠需α≤5.07，为与MMA重叠需α≤5.05。QServe的QoQ算法（α≥10）无法满足此阈值；LiquidQuant的α≈0.875（含unpack）满足要求。

术语一般如何实现？如何使用？
实现代表：QServe（QoQ algorithm, group=128）、LiquidGEMM（LiquidQuant, group=64）。适用于需平衡精度和内存的LLM serving场景——4-bit权重使大模型可在单GPU运行，8-bit激活保持推理精度。激活量化用SmoothQuant per-token动态量化（FP16→INT8），fuse到前序kernel epilogue消除overhead。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---

## LiquidQuant (LQQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LiquidQuant (LQQ) 是LiquidGEMM论文提出的硬件高效W4A8 dequantization算法。核心机制：通过rotation-based transformation将INT8先shift到UINT8域再量化到UINT4，利用two's complement同余性质（i ≡ j mod 2^8 → 相同二进制表示）设计无溢出dequantization。Dequantization公式：Q_i8 = (Q_u4 × s_u8 + a) XOR 0x80，其中a = 128 + min(Q_i8)预计算offline。关键洞察：(1) Q_u4∈[0,15], s_u8≤16 → Q_u4×s_u8≤240（UINT8安全）；(2) a∈[9,247]（UINT8安全）；(3) XOR 0x80等价于翻转MSB，实现条件性±128，使结果落在INT8的二进制表示内。仅需两条32-bit指令（IMAD + XOR）处理四个元素 vs QServe的QoQ算法10+指令。8个元素（含unpack）仅需7条指令（0.875指令/元素）。

从算法pipeline角度拆解术语：
```
量化 (离线):
  Q_i8 ∈ [-119, 119] (per-channel, protective range)
  Q_u8 = Q_i8 - min(Q_i8)        // shift到UINT8: [-119,119] → [0,238]
  s_u8 = round(max(Q_u8) / 15)   // ≤16
  Q_u4 = round(Q_u8 / s_u8)      // [0,15] UINT4
  a = 128 + min(Q_i8)            // [9, 247], precomputed per-group

Dequantization (在线, CUDA Cores):
  // Unpack (QServe method): 8×UINT4 → 2×32-bit regs
  // Dequantization (2 instructions per reg, 4 elements):
  reg = IMAD(reg, s_u8_bcast, a_bcast)  // multiply-add (1 instr)
  reg = XOR(reg, 0x80808080)            // flip MSB each byte (1 instr)
  // Result: INT8 binary in UINT8 registers → directly usable by WGMMA
```

术语一般如何实现？如何使用？
LQQ用CUDA实现（非PTX），利用IMAD和XOR原生GPU指令。s_u8和a预计算per-group（共K/64组/channel）。集成到CUTLASS/Cute warp-specialized kernel中，fuse dequantization到MMA mainloop。LQQ保持模型精度——WikiText2 perplexity和zero-shot准确率与QServe相当。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---

## Hybrid LLMs (Attention-SSM混合架构)

术语是什么？
Hybrid LLMs是将Attention层和State Space Model (SSM)层混合组成的语言模型架构。典型设计：少量Attention层（如4层）捕捉token间显式交互和检索能力，大量SSM层（如24层）通过recurrent state高效处理长序列。这种架构平衡了Attention的语言建模能力强（O(L²)计算复杂度、O(L) KV cache内存）和SSM的推理效率高（O(L)计算复杂度、O(1) fixed-size state内存）之间的tradeoff。

代表模型：NVIDIA Mamba2-Hybrid系列（4 Attention + 24 SSM + 28 MLP layers, 7B参数）、Jamba系列（1 Attention per 6-10 SSM layers, 最高398B参数）、Phi-4等。Marconi论文评估的默认模型为Mamba2-Hybrid-7B。

从算法pipeline角度拆解术语：
```
Hybrid LLM Forward Pass (per layer):

Attention layer (位置: 特定层索引):
  Input: X ∈ R^{L×d}
  1. Q, K, V = W_Q·X, W_K·X, W_V·X    // Linear projections
  2. A = softmax(Q·K^T/√d_k)·V         // Multi-head attention, O(L²·d)
  3. O = W_O·A                          // Output projection
  4. Store KV cache: (K, V) per token   // O(L·d) 内存, 可切片复用

SSM layer (位置: 大多数层):
  Input: X ∈ R^{L×d}
  1. B_t, C_t, Δ_t = Project(X_t)      // 输入依赖的选择性参数 (Mamba2)
  2. h_t = Ā_t·h_{t-1} + B̄_t·X_t        // Recurrent state update (in-place!)
  3. Y_t = C_t·h_t                      // Output
  4. Store SSM state: h ∈ R^{d_state×d} // O(1) 固定内存, 无法回滚!

MLP layer (每层后):
  gate = SiLU(W_g·X); up = W_u·X; down = W_d·(gate ⊙ up)
```

关键差异: Attention的KV cache是per-token的（可任意切片→prefix caching直接），SSM的recurrent state是per-sequence的（in-place更新→无法回滚→prefix caching需额外checkpoint机制）。
```

术语一般如何实现？如何使用？
实现：基于Mamba2 selective SSM kernel（CUDA官方实现），与FlashAttention交替排列构建Hybrid architecture。Marconi通过radix_cache_hybrid.py统一管理两者的prefix caching。评估workloads：LMSys/ShareGPT (conversational)和SWEBench (agentic)。趋势：Hybrid LLMs中SSM比例不断增加（更高效率），Marconi在higher SSM ratio下性能增益更大。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---

## SSM Recurrent States (for LLM Inference)

术语是什么？
SSM Recurrent State是State Space Model层在LLM推理时维护的固定大小hidden state h_t ∈ R^{d_state × d_model}（或Mamba2的变体形式）。与Attention的KV cache（per-token存储、大小随序列长度线性增长）不同，SSM state通过in-place递推更新：h_t = f(h_{t-1}, x_t)，每个新token更新整个state而非追加新条目。这使得SSM state具有两个关键特性：(1) 固定内存占用——无论序列多长，state大小始终为d_state × d_model × sizeof(fp32)；(2) 不可回滚——state是sequence-level的累积表示，无法像KV cache那样通过切片回退到前缀的任意中间位置。

从算法pipeline角度拆解术语：
```
// Mamba2 SSM state更新 (简化):
Input: x_t ∈ R^d_model, h_{t-1} ∈ R^{d_state × d_model}
Parameters: A, B, C, Δ (输入依赖)

Step 1: Δ_t = softplus(W_Δ·x_t + b_Δ)         // 选择性时间步长
Step 2: Ā_t = exp(Δ_t · A)                     // 离散化状态转移矩阵
Step 3: B̄_t = (Δ_t·A)^{-1}·(exp(Δ_t·A)-I)·Δ_t·B  // 离散化输入投影
Step 4: h_t = Ā_t·h_{t-1} + B̄_t·x_t             // In-place递推更新!
Step 5: y_t = C_t·h_t                           // 输出

// 关键: h_t 直接覆盖 h_{t-1}，不保留历史版本
// h_5 可以表示序列[1..5]，但无法回退表示[1..3]
```

Implication for prefix caching:
- Attention: K_{1..5}, V_{1..5} → 可直接取子集 K_{1..3}, V_{1..3} 表示前缀
- SSM: h_5 → 无法从h_5推导h_3 → 必须单独checkpoint h_3 才能复用前缀[1..3]
```

术语一般如何实现？如何使用？
Mamba/Mamba2 CUDA kernel实现recurrent state更新。Training时使用parallel scan（所有时间步并行计算），Inference时使用recurrent mode（逐token递推）。SSM state在GPU memory中的大小：Mamba2-Hybrid-7B d_state=128, d_model=4096 → 约128×4096×4=2MB per layer。24 SSM layers → 约48MB per sequence（远大于单token KV但远小于完整序列KV）。Marconi通过每序列至多2个checkpoint控制总缓存中SSM state数量。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---

## Fine-grained SSM State Checkpointing

术语是什么？
Fine-grained SSM State Checkpointing是将纯Attention prefix caching扩展至Hybrid LLMs的naive baseline方案。由于SSM state的in-place递推特性（无法回滚），该方法每隔固定k个token保存一次SSM layer的完整recurrent state作为checkpoint。当后续请求匹配到某个checkpoint对应的前缀时，从该checkpoint恢复SSM state并继续forward计算。

该方案存在两个致命缺陷（Marconi论文核心motivation）：(1) Cache entries are sparsely-hit——大部分checkpoint位于无人复用的token位置（如对话中间而非开始或结尾）；(2) Cache entries are huge——单checkpoint覆盖所有SSM layers，NVIDIA Mamba2-Hybrid-7B约48MB/checkpoint（24 SSM layers × 2MB/layer），大量低命中率checkpoint迅速填满缓存。

从算法pipeline角度拆解术语：
```
Naive Algorithm:
  Prompt: "NYC is a busy city" (5 tokens), k=2
  Checkpoints:
    h_2 (position 2: "NYC is")      → 缓存
    h_4 (position 4: "NYC is a busy") → 缓存
    h_5 (position 5: full sequence)  → 缓存 (最后一个)

  新请求: "NYC is" + "new query" (shared prefix: "NYC is" = position 2)
    → 命中h_2 checkpoint → 恢复 → 从position 2继续prefill

  Problems:
  - h_4被缓存但几乎不会被复用（"NYC is a busy"不是自然的对话开始点）
  - 每k token一个checkpoint → k越小checkpoint越多 → 缓存越满
  - 长对话: 1000 token sequence, k=5 → 200 checkpoints × 48MB = 9.6GB仅一个序列
```

术语一般如何实现？如何使用？
vLLM+/SGLang+ baseline采用此方案（扩展原始框架的prefix caching以支持SSM states）。使用标准LRU eviction管理所有检查点。Marconi替代方案：每序列至多2个checkpoint（purely-input分支点 × 1 + leaf末尾 × 1），通过radix tree的复用模式识别精准定位值得缓存的token位置。vs fine-grained checkpointing token hit rate提升4.5×–34.4×。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---

## Block Low-Rank (BLR) Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Low-Rank (BLR) 压缩是一种结构化权重矩阵压缩技术：将神经网络中一个大的dense权重矩阵W∈R^{i×o}划分为b₁×b₂个块（block），每个块W_{l,k}∈R^{p×q}（其中p=i/b₁, q=o/b₂）分别用低秩分解表示。与全局低秩分解（W=VU，对所有元素统一做rank-r分解）不同，BLR在各block内部独立rank分配，允许不同block捕捉不同程度的局部相关性。这使得BLR比标准low-rank具有更高的表达能力——在相同参数预算（压缩比）下保持更好准确率。

典型BLR方法的参数和计算复杂度：参数b₁b₂r'(p+q)，FLOP nb₁b₂r'(p+q)，n是序列长度。当b₁=b₂=b且r=r'b时渐近复杂度与标准low-rank相同：r(i+o)参数、nr(i+o) FLOP。与dense（i×o参数, n×i×o FLOP）相比，压缩比CF≈1.8-3×。

从算法pipeline角度拆解术语，给出具体例子。
以b₁=16, b₂=16, p=256, q=256, r'=64为例：

```
# 离线阶段：将dense权重W∈R^{4096×4096}分解为BLR参数
for l in range(16):
  for k in range(16):
    W_{l,k} ∈ R^{256×256} ≈ V_{l,k}·U_{l,k}
    # V_{l,k} ∈ R^{256×64}, U_{l,k} ∈ R^{64×256}

# 在线推理 (n=1024 tokens):
Input: X ∈ R^{1024×4096}
X_blocks = X.view(1024, 16, 256)

for k in range(16):  # 每个输出block
    Y_k = zeros(1024, 256)
    for l in range(16):  # accumulate所有输入block的贡献
        Y_k += (X_blocks[:,l,:] @ V_{l,k}) @ U_{l,k}
        # [1024,256]@[256,64]=[1024,64]; [1024,64]@[64,256]=[1024,256]

Y = concat([Y_0,...,Y_15], dim=-1)  # [1024, 4096]
```

术语一般如何实现？如何使用？
BLR训练模式：从头训练（BLR权重参数化初始化）或压缩预训练模型（block-wise SVD for Monarch、preconditioned gradient descent 300步 for BLAST）+fine-tune。BLR的核心tradeoff：更多block→更细粒度的表达能力→更好精度，但也产生更多b×n×r中间张量数据移动（如b=16, n=1024, r=1024时128MB BF16）。在多token推理中，这额外的数据移动可能将compute-bound线性层推入memory-bound→需kernel级优化恢复性能。标准低秩（全局单rank）数据移动最优但高压缩比下精度急剧下降——BLR通过块结构在精度和效率间寻找更优平衡点。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Monarch Matrix Decomposition

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Monarch是Dao et al. (2022, ICML)提出的结构化矩阵分解方法，属于BLR的一种。原始定义：M = P₁ L P₂^T R，L和R是block-diagonal矩阵，P₁和P₂是固定permutation矩阵。Monarch将dense权重划分为b₁×b₂个块，每块独立低秩分解W_{l,k}=V_{l,k}U_{l,k}。关键特征是两次permutation（r'↔b₂→b₂↔b₁）实现跨block信息混合。常用配置b₁=b₂=b=4-16。参数b₁b₂r'(p+q)，FLOP nb₁b₂r'(p+q)。

从算法pipeline角度拆解术语：
```
# Monarch权重: V∈R^{b₁×(r'b₂)×p}, U∈R^{b₂×q×(b₁r')}
# b₁=b₂=16, p=q=256, r'=64

X_blocks = X.view(n, 16, 256)             # [n, b₁, p]
Z = batched_bmm(X_blocks, V^T)             # [b₁, n, r'b₂]

# ↑ 基线性能瓶颈: 两次permutation ↓
Z = Z.reshape(16, n, 16, 64)              # b₁→b₁, n→n, r'b₂→(b₂, r')
Z = Z.transpose(0,2).transpose(1,2)        # → [b₂, n, b₁·r']
# ↑ 需要clone tensor, uncoalesced access

for k in range(16):
    Y_k = Z[k] @ U[k]                     # [n, 1024]@[1024, 256]
Y = final_permute(stack(Y_k))              # [b₂,n,q]→[n,q,b₂]
```

术语一般如何实现？如何使用？
开源：https://github.com/HazyResearch/monarch。训练方式：从头训练Monarch参数化模型或压缩预训练权重。Monarch在ViT-B CF=3×下ImageNet=79.2% vs low-rank 78.9%；GPT2-S CF=1.85×下WikiText-103 PPL=21.1 vs low-rank 21.7。主要问题：多token推理(n=1024)时两次permutation kernel+4bnr bytes中间数据使实际速度比dense慢1.14-1.68×（A40），需通过Triton kernel的permutation fusion和V重排布优化恢复性能。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## BLAST (Block-Level Adaptive Structured Matrices)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BLAST是Lee et al. (2024, NeurIPS)提出的通用化BLR结构。与Monarch每block独立分解不同，BLAST引入三种因子共享：每个输入block l共享左因子V_l∈R^{p×r}，每个输出block k共享右因子U_k∈R^{r×q}，跨block交互由per-block对角矩阵S_{l,k}∈R^{r×r}建模。完整分解：W_{l,k}=V_l·S_{l,k}·U_k。统一表达性：通过设置S_{l,k}可恢复标准低秩（S全部相同）、Monarch（S为特定结构化pattern）、block-diagonal（S_{l,k}=0 for l≠k）等。参数r(p+q+b²)，FLOP nr(p+q+b²)，b²项通常可忽略（b≤16）。

从算法pipeline角度拆解术语：
```
# BLAST权重: V∈R^{16×256×1024}, S∈R^{16×16×1024}, U∈R^{16×1024×256}

X_blocks = X.view(n, 16, 256)
Z_l = batched_bmm(X_blocks, V)             # [16, n, 1024]

for k in range(16):
    Y_k = zeros(n, 256)
    for l in range(16):
        Y_k += (Z_l[l] * S[l,k]) @ U[k]   # Hadamard ⊙ + bmm
    Y.append(Y_k)
return concat(Y, dim=-1)                   # [n, 4096]
```

术语一般如何实现？如何使用？
开源：https://github.com/changwoolee/BLAST；HuggingFace：https://huggingface.co/cwoolee/blast-llama-4B。压缩通过preconditioned gradient descent（300步）分解dense权重→fine-tune。BLAST精度最优：Llama-7B CF=2× WikiText-2 PPL=14.21 vs Monarch 19.54 vs LR 26.33；ViT-B CF=3× ImageNet=79.3%略高于Dense 78.7%。代价是多token推理PyTorch基线性能最差——比dense慢2.63-4.31×（A40），因为两组中间张量(8bnr bytes)+两组permutation。论文Triton kernel优化⑤（permutation-only fusion with tensor core）是实现BLAST实用化的关键。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Multi-Token vs Single-Token Inference (for BLR Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Token Inference（n>1, prefill阶段）和Single-Token Inference（n=1, decode阶段）是LLM推理的两种模式。在BLR压缩模型上下文中，两种模式的瓶颈性质根本不同：单token推理的memory流量由权重读取主导（o·i），激活(n·i)可忽略，压缩权重直接加速；多token推理的激活流量随n线性增长，BLR的block结构产生dense不存在的b×n×r中间张量（Monarch: 4bnr bytes, BLAST: 8bnr bytes），将compute-bound线性层推入memory-bound区域。

从算法pipeline角度拆解术语：
A40 BF16 Q/K/V/Oproj layer (i=o=4096, r=1024, b=16) 的roofline分析：

| 场景 | 方法 | FLOP | Memory Traffic | α=FLOP/Bytes | 瓶颈 |
|------|------|------|----------------|-------------|------|
| n=1 | Dense | 34M | 32KB | ~1074 | Compute |
| n=1 | BLAST | 17M | 40KB | ~425 | Compute |
| n=1024 | Dense | 34G | 34MB | ~994 | Compute |
| n=1024 | Low-Rank | 17G | 17MB | ~978 | Compute |
| n=1024 | Monarch | 17G | 138MB | ~123 | **Memory!** |
| n=1024 | BLAST | 17G | 266MB | ~64 | **Memory!** |

α阈值 ≈ 155。α_Monarch=123 < 155 → memory-bound（比dense慢1.14-1.68×）。α_BLAST=64 << 155 → strongly memory-bound（比dense慢2.63-4.31×）。尽管FLOP减半，额外数据移动反而使推理变慢。

术语一般如何实现？如何使用？
实际LLM服务中prefill和decode交替发生：prefill（n=prompt_len, 多token）→ decode（n=1, 逐token生成）。BLR模型在decode阶段天然受益于权重压缩（memory-bound→压缩直接加速），在prefill阶段需kernel级优化（如论文Triton kernel的partial fusion、memory layout optimization）将FLOP减少转化为实际加速。论文最终实现BLAST ⑤在end-to-end推理中1.13-1.48× over dense，证明多token瓶颈可以被kernel优化克服。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Expert Skipping (in MoE LLMs/MLLMs)

术语是什么？
Expert Skipping是一种训练无关的MoE模型推理加速技术：在推理时动态跳过对当前token贡献不足的冗余expert，仅激活真正重要的expert子集执行计算。与expert pruning（永久移除）不同，expert skipping是per-token的动态决策。核心机制：(1) 计算top-k候选expert对当前token的重要性分数；(2) 将分数低于阈值的expert标记为跳过；(3) 仅对保留的active expert执行FFN计算和加权聚合。MoDES识别出MLLMs场景下两个关键因素：层间贡献不均（shallow layer experts更重要，因error在后续层被放大）和模态行为差异（vision token的expert冗余度更高）。

从算法pipeline角度拆解术语：
以Qwen3-VL-MoE-30B-A3B-Instruct的l-th MoE层为例（128 experts, k=8）：
```
r = Router(x)                                    // [128] routing logits
π = softmax(r)                                   // routing probabilities
S = topk_indices(π, k=8)                         // 8 candidate experts

for each i in S:
    s_i = α̃^{(l)} · π_i                          // GMLG: global × local importance
τ = is_text(x) ? τ_t : τ_v                      // DMT: modality-specific threshold
active = {i ∈ S : s_i ≥ τ}                       // keep only important experts

y = Σ_{m ∈ active} π_m · Expert_m(x)             // weighted aggregation
```
跳过比例skip_ratio = 1 - |active|/k。MoDES在跳过88% expert时仍保持97.33%原始性能。

术语一般如何实现？如何使用？
现有方法：NAEE（routing probability阈值判定）、MC-MoE（attention-aware protection）、DiEP（differentiable pruning with expert similarity）。MoDES在此基础上引入GMLG（全局层重要性×局部routing概率）和DMT（text/vision分别设阈值），配合Frontier Search找最优阈值。Custom CUDA kernel内嵌branch-free masked comparison实现，跳过expert路由为sentinel ID并在dispatch/gather阶段过滤。适用所有MoE架构的LLMs/MLLMs推理加速，尤其在高跳过率（>80%）下优势显著。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## GMLG (Globally-Modulated Local Gating)

术语是什么？
GMLG是MoDES提出的expert重要性评估机制。将离线校准的全局逐层重要性α^{(l)}与推理时的局部routing概率π_i^{(l)}相乘：s_i^{(l)} = α^{(l)} · π_i^{(l)}。α^{(l)}通过KL divergence量化跳过第l层所有expert对final output的影响——浅层α^{(l)}大（对最终输出影响大，应少跳过），深层α^{(l)}小（可多跳过）。Inference时α^{(l)}已预计算，s_i^{(l)}仅需一次乘法——零额外开销。

从算法pipeline角度拆解术语：
```
// 离线校准 (one-time):
C = 1024 randomly sampled examples from GQA
for each MoE layer l:
    for each example c_j in C:
        prob_j = full_model(c_j)
        prob_j^{(l)} = model_with_layer_l_skipped(c_j)
    α^{(l)} = (1/N) · Σ_j D_KL(prob_j || prob_j^{(l)})
α̃^{(l)} = α^{(l)} / Σ α^{(l')}                     // normalize across layers

// 在线推理 (per token, zero overhead):
s_i^{(l)} = α̃^{(l)} · π_i^{(l)}
// α̃^{(l)}大 → 浅层 → 整体s_i偏高 → 跳过少
// α̃^{(l)}小 → 深层 → 整体s_i偏低 → 跳过多
```

术语一般如何实现？如何使用？
校准在8×H200上执行，每层需2次forward pass。校准数据鲁棒——GQA/COCO/VMMMU上α^{(l)}趋势一致，性能差异<1%。α^{(l)}在浅层大、深层小的趋势在所有模型和数据集上一致，且与论文motivation中降低浅层vs深层k值的实验结论一致（浅层降低k性能下降更严重）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## DMT (Dual-Modality Thresholding)

术语是什么？
DMT是MoDES的模态感知expert skipping策略。基于发现：(1) vision token在FFN前后的余弦相似度更高（FFN对其更新幅度更小）；(2) 降低vision token的top-k对performance影响更小（vision expert冗余度更高）。DMT为text和vision token分别设置阈值τ_t和τ_v（τ_v < τ_t），对vision token更激进跳过expert。这是text-only LLM expert skipping工作未曾考虑的因素。

从算法pipeline角度拆解术语：
```
// Per token, per MoE layer:
τ = (token is text) ? τ_t : τ_v   // τ_v < τ_t
for each expert i in topk:
    if s_i < τ: skip Expert_i
```
效果：vision token的跳过率显著高于text token；深层跳过率显著高于浅层（对应α^{(l)}小的深层）。最佳阈值(τ_t*, τ_v*)通过Frontier Search在O(ND)时间找到，约束target skipping ratio ρ。

术语一般如何实现？如何使用？
τ_t和τ_v离线搜索确定。推理时在custom CUDA router kernel内通过branch-free masked comparison实现：`mask = (s_i < τ); topk[i] = mask ? M+1 : topk[i]`。DMT与GMLG叠加使用效果最强——单一组件在低跳过率时差异不大，但在高跳过率（>80%）下两者叠加的非线性增益显著（DMT+GMLG比仅Thresholding高~10%）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## Frontier Search (Monotonicity-Based Threshold Optimization)

术语是什么？
Frontier Search是MoDES提出的基于单调性的二维阈值搜索算法。在B×B空间中（100个grid points × 100个grid points）找到满足target skipping ratio ρ约束下最小化KL divergence f的最优(τ_t, τ_v)对。利用f和g（skipping ratio）关于参数的单调非递减性，将O(ND²) exhaustive search降至O(ND)，搜索时间~45×降低（>2天→<2小时），最优解性能差异<0.01%。

从算法pipeline角度拆解术语：
```
FrontierSearch(B={τ^{(1)},...,τ^{(D)}}, ρ):
    frontier = ∅; p = D
    for q = 1 to D:                      // increasing τ_t
        while p ≥ 1 and g(τ^{(q)}, τ^{(p)}) ≥ ρ:
            p = p - 1                    // monotonicity: shrink τ_v
        p_{(q)} = p + 1
        if p_{(q)} ≤ D:
            compute f(τ^{(q)}, τ^{(p_{(q)})})
            frontier ∪= {(q, p_{(q)})}
    return argmin_{(q,p)∈frontier} f     // optimal (τ_t*, τ_v*)
```
关键性质：p_{(q)}关于q非递增（更大的τ_t→需更小τ_v满足ρ）；最优解必在frontier上（非frontier解被dominated）。

术语一般如何实现？如何使用？
在calibration set C（1024 GQA samples）上evaluate，每对(τ_t, τ_v)需1次forward pass。Frontier search总forward pass≤200（vs naive 10,000）。对20-30B MLLM，calibration+search在8×H200上20分钟至<4小时。D=100经ablation验证足够。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## MoE in Multimodal LLMs

术语是什么？
MoE in Multimodal LLMs是将Mixture-of-Experts应用于MLLM的LLM backbone FFN层的设计范式。MLLM由Visual Encoder（提取visual token）→ Projector（对齐到text embedding space）→ LLM Backbone（transformer with MoE FFNs）组成。典型MLLM MoE配置：Kimi-VL-A3B（64 experts/layer, k=6）、Qwen3-VL-MoE-30B-A3B（128 experts/layer, k=8）、InternVL-3.5-30B-A3B（128 experts/layer, k=8）。MLLM中MoE的独特挑战：(1) vision token数量大→MoE计算开销显著；(2) 不同模态token在FFN中行为不同（modality gap）；(3) 浅层expert贡献远大于深层。

从算法pipeline角度拆解术语：
```
MLLM Forward:
    V = VisualEncoder(image/video)             // [N_v, d_v]
    V' = Projector(V)                           // [N_v, d_model]
    X = concat([V', T])                         // vision + text tokens
    for each transformer layer l:
        X = Attention(LN(X)) + X
        π = softmax(Router(LN(X)))              // [M] routing probs
        y = Σ_{m∈topk(π,k)} π_m · Expert_m(LN(X))
        X = y + X
```

术语一般如何实现？如何使用？
开源MLLMs：Kimi-VL, Qwen3-VL-MoE, InternVL-3.5。router为`nn.Linear(d_model, M)` + Softmax。Expert为独立FFN。评估使用lmm-eval框架 + multimodal benchmarks。MoDES通过GMLG+DMT+Frontier Search实现MLLM特定的expert skipping。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## Modality Gap in MLLM FFNs

术语是什么？
Modality Gap是MoDES发现的MLLMs中不同模态token在FFN层的行为差异：(1) t-SNE可视化显示text/vision token的FFN输入表示跨所有层存在一致分布差异；(2) vision token在FFN前后的余弦相似度高于text token——FFN对vision token更新幅度更小；(3) vision token与FFN权重的夹角更接近90°（正交），减弱更新量；(4) 降低vision token的top-k对性能影响更小——vision expert冗余度更高。该发现直接motivate DMT的设计。

从算法pipeline角度拆解术语：
```
度量1: cos_sim(x_pre, x_post)  → text较低(大更新) vs vision较高(小更新)
度量2: angle(x, W_FFN)         → text较小 vs vision接近90°(弱交互)
度量3: sensitivity to k_reduction:
    Δacc(text, k↓) > Δacc(vision, k↓)  → vision冗余度更高
```
这些度量在GQA数据集上使用Kimi-VL-A3B-Instruct验证。

术语一般如何实现？如何使用？
通过t-SNE降维可视化和余弦相似度分析在GQA数据集上验证。这是MoDES区别于text-only LLM expert skipping方法的根本motivation。解释了MoDES在高跳过率（>80%）下仍保持性能的原因——vision token的expert可被大幅跳过而不影响输出质量。该insight也预测了DMT中τ_v < τ_t的设计合理性。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Decoder-Only (DecOnly) LMM Architecture

术语是什么？

Decoder-Only LMM 是多模态大模型的一种主流架构，核心特征是复用未经修改的纯文本 decoder-only LLM（如 Qwen2, LLaMA）作为 backbone，将 image encoder 输出的 image tokens 与 text tokens 在 LLM 的 self-attention 中统一处理。Image tokens 通过 connector/MLP 映射到 LLM token embedding space 后，与 text tokens 拼接为一个统一序列输入 LLM。所有 transformer layer 中的 self-attention 对 text 和 image tokens 进行同等计算。代表模型包括 LLaVA-OneVision（Qwen2 backbone + SigLIP encoder）、InternVL（InternLM backbone + InternViT encoder）、NVLM-D（Qwen2-Instruct backbone + InternViT encoder）、DeepSeek Janus。

从算法pipeline角度拆解术语：

DecOnly LMM 推理 token 级计算过程：
```
输入: text_prompt (N_t tokens) + image

Step 1 - Image Preprocessing (CPU):
  raw image → resize/rescale/pad/normalize → tile segmentation

Step 2 - Image Encoding (GPU, ViT):
  tiles → ViT forward → image_tokens [N_img, d_enc]

Step 3 - Connector Projection:
  image_tokens → MLP → [N_img, d_llm]

Step 4 - LLM Prefill (all layers self-attention on all tokens):
  input_seq = [image_tokens | text_tokens]  // N = N_img + N_t
  for l in 1..L:
    Q, K, V = W_Q×h, W_K×h, W_V×h  // text+image 同等处理
    A = softmax(Q@K^T / sqrt(d))    // N×N full attention
    h = FFN(h)
```

关键特征：prefill 中 image tokens 参与每一层 self-attention，计算复杂度 O((N_img+N_t)²·L)。高分辨率图像产生大量 tokens（LLaVA-OV 每张 896×896 图像产生 7290 tokens），严重增加 prefill 延迟。ModServe Insight 3：DecOnly 模型 LLM prefill 延迟可达同规模 CroAttn 的 10×。

术语一般如何实现？如何使用？

两阶段训练：(1) pre-training——冻结 encoder 和 LLM，仅训练 connector；(2) instruction fine-tuning——解冻 LLM+connector。推理时 connector 极轻量（<0.1% 参数，<0.4% TTFT）。DecOnly 模型在 image-heavy workload 下因 prefill 高延迟而严重资源争用——ModServe 发现 stage decoupling 对 DecOnly 收益更大（InternVL 5.5× vs Llama3.2 3.3× throughput improvement）。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

## Cross-Attention (CroAttn) LMM Architecture

术语是什么？

Cross-Attention LMM 是多模态大模型的第二种主流架构，在 LLM backbone 中插入专用 cross-attention layers 处理 image tokens。大部分层保持原有 self-attention（仅处理 text tokens），少数层（如 Llama3.2-11B 中 4/40 层）被替换为 cross-attention layers——text tokens 通过 cross-attention attend to image tokens，image tokens 不参与 self-attention。代表模型：Llama 3.2 Vision（4/40 CroAttn layers）、NVLM-X、Flamingo。

从算法pipeline角度拆解术语：

CroAttn LMM 推理（Llama3.2-11B）：
```
for l in 1..40:
  if l is self-attention layer (36 layers):
    Q,K,V from text tokens only
    A = softmax(Q_t@K_t^T/sqrt(d))
  if l is cross-attention layer (4 layers):
    Q_t from text tokens, K_i,V_i from image tokens
    A_cross = softmax(Q_t@K_i^T/sqrt(d))
    O = A_cross@V_i
  h = FFN(O)
```

关键特征：(1) prefill 复杂度大幅降低——self-attention 仅 O(N_t²)，cross-attention O(N_t·N_img)；(2) Image encoding 成为 TTFT 主要瓶颈——Llama3.2-11B 中 79% TTFT 来自 encoding（Insight 1）；(3) Image token 比例增加时 LLM prefill 时间反而减少（Insight 7），因此 CroAttn 对 image burst 更具弹性——autoscaling 仅需扩容 Image Instances。

术语一般如何实现？如何使用？

训练：先预训练 vision encoder 和 cross-attention adapter，再端到端 fine-tune。推理时 cross-attention K/V projection 需要额外权重和 KV cache 管理。ModServe 关键发现：CroAttn 模型 image burst 时 Text Instances 无需扩容——这是 41.3% cost saving（vs DecOnly 25%）的原因。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

## Image Encoding in Multimodal Models (Vision Transformer Encoder)

术语是什么？

Image encoding 是 LMM 推理中将预处理图像 tiles 转换为 image token embeddings 的阶段。当前主流 LMM 使用 Vision Transformer (ViT) 作为 encoder，将每个 image tile 编码为固定数量的 visual feature tokens。不同 LMM 使用不同 encoder：ViT-H/14 (630M, Llama3.2)、SigLIP (400M, LLaVA-OV)、InternViT (6B, InternVL/NVLM-D)。

从算法pipeline角度拆解术语：

ViT encoding（处理 560×560 tile）：
```
conv2d (kernel=14, stride=14): [1,3,560,560] → [1,d,40,40]
flatten + [CLS]: [1, 1601, d]
for l in 1..L_enc:
  h = h + MHA(LayerNorm(h))
  h = h + FFN(LayerNorm(h))
→ image_tokens [1, 1601, d_enc]
```

ModServe Insight 2: Image encoding 是 compute-bound（SM activity ~100%, DRAM util <30%）。Encoder 最佳 TP 度通常为 TP-1——因 630M 模型分到 8 GPU 时 inter-GPU communication > compute savings，与 LLM backend（需 TP-4/8）形成鲜明对比。这一差异是 ModServe stage decoupling 的物理基础——允许 encoder 和 LLM 使用不同 TP 度。

术语一般如何实现？如何使用？

通过 HuggingFace Transformers 加载。不同 LMM tile 配置不同（Table 1）：Llama3.2 560×560 tiles → 4 tiles, LLaVA-OV 384×384 → up to 10 tiles, InternVL 448×448 → 5 tiles。Tokens/tile: ViT-H/14 1601, SigLIP 729, InternViT 256。ModServe 的 Image Instance 通过多个 TP-1 encoder 并行化同请求多 tiles 的 encoding（tiles 间无依赖，Insight 2）。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

## Multimodal Connector / Modality-Alignment Module

术语是什么？

Multimodal Connector（又称 projector / modality-alignment module）是将 image encoder 输出映射到 LLM token embedding space 的轻量级模块。因 encoder 和 LLM 独立预训练，embedding spaces 维度和语义对齐方式不同，connector 桥接这一 gap。最常见形式为 2-layer MLP（LLaVA 系列）。参数量极小——<0.1% 总参数，<0.4% TTFT（ModServe 论文）。

从算法pipeline角度拆解术语：

```
encoder_output: [N_tokens, d_enc]
  → W_1 @ x + b_1 → GELU → W_2 @ h + b_2
  → [N_tokens, d_llm]
```

维度示例: ViT-H/14 d_enc=1280, Llama 3.1 8B d_llm=4096。计算量远小于 LLM prefill。

ModServe 部署选择：connector 共置于 Text Instance（与 LLM 共享 GPU），Image Instance 仅输出原始 encoder output，connector forward 在 RDMA 传输后在 Text Instance 侧执行——避免为极轻量模块分配独立 GPU。

术语一般如何实现？如何使用？

实现方式因模型而异：LLaVA 使用 2-layer MLP+GELU，InternVL 使用 pixel shuffle+MLP，BLIP-2 使用 Q-Former（learnable query-based）。ModServe 兼容各种 connector。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

---

## CKA (Centered Kernel Alignment)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CKA (Centered Kernel Alignment) 是 Kornblith et al. (ICML 2019) 提出的神经网络表示相似度度量方法。通过比较两个模型在相同输入上的**样本间相似结构**（而非直接比较特征向量）来衡量表示相似度。公式：CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))，HSIC(K, L) = Tr(K H L H)，K/L 是激活值的核矩阵（如线性核 K=XX^T），H = I - (1/n)1·1^T 是中心化矩阵。值∈[0,1]。关键特性：(1) 对正交变换不变；(2) 对同向缩放不变；(3) **不对任意可逆线性变换不变**——使其能捕捉有意义的结构差异，而 CCA 等方法在低样本/高维度下失效；(4) 可比较不同 shape 的表示矩阵（p1≠p2），传统 cosine similarity 无法做到。

从算法pipeline角度拆解术语，给出具体例子。
Mordal 中 CKA 的两步聚类流程：
```
// Step 1: Vision Encoder CKA
for each pair (VE_A, VE_B):
    act_A = VE_A(images)  // [N, d_ve_A]
    act_B = VE_B(images)  // [N, d_ve_B]
    K = act_A @ act_A.T; L = act_B @ act_B.T
    H = I - 1/N * ones(N,N)
    cka = Tr(K@H@L@H) / sqrt(Tr(K@H@K@H) * Tr(L@H@L@H))
    dist = 1 - cka
C_ve = HierarchicalClustering(dist, t_ve=0.7)

// Step 2: LLM CKA (per VE cluster)
medoid_ve = PickMostCentral(VE_cluster)
fixed_output = WarmupProjector(medoid_ve(images))  // 统一shape
for each pair (LLM_A, LLM_B):
    rep_A = LLM_A.last_hidden_state(fixed_output)
    rep_B = LLM_B.last_hidden_state(fixed_output)
    cka = CKA(rep_A, rep_B)
// → LLM clusters → Cartesian product → VLM candidate clusters
```
Mordal 选择 CKA 的两个关键原因：(1) 不同 VE 输出维度不同（d_CLIP≠d_SigLIP），CKA 通过核矩阵投影回避维度对齐；(2) MLP Feature Projector 的变换不影响 CKA 的鲁棒性。

术语一般如何实现？如何使用？
Mordal 使用 MinibatchCKA (Nguyen et al. 2020, Raghu et al. 2021) 支持大数据集，`scipy.cluster.hierarchy` 进行层次聚类。PyPI 包：`cka`。论文验证：ScienceQA 和 VizWiz 上相似 CKA 表示的 VE 产生相似性能。适用场景：模型调试（检测冗余层）、迁移学习（识别 freeze 层）、架构比较。局限：对异常值敏感，不满足三角不等式。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---

## VLM Pretrained Model Selection (Automated)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLM Pretrained Model Selection 是 Mordal 首次定义的问题：给定下游任务、alignment 数据集和 pretrained 模型库（VE 集合 + LLM 集合），在资源约束下找到最优 VE×LLM 组合，使得经过 vision-text alignment 训练后在目标任务性能最优。核心难点：未 alignment 的 VLM 无法评估 zero-shot 性能（feature projector 未训练时 LLM 不理解 image embeddings，会产生随机输出），因此传统 model selection 方法（LogME, LEEP 等，设计用于 vision-only 或 text-only 任务）在此场景失效——必须实际训练 projector 后才能评估。Mordal 将其建模为资源约束的组合优化问题，minimize search cost while maximizing selection accuracy。

从算法pipeline角度拆解术语，给出具体例子。
问题形式化：
```
Given: M_ve × M_llm (m×n candidates), D_align, D_task, Budget B
Goal: (VE*, LLM*) = argmax Perf(align(VE, LLM, D_align), D_task)
      s.t. total search cost ≤ B

Naive grid search: m=7, n=7 → 49 candidates
  Each: ~111 GPU hours (projector training + LoRA fine-tuning on A40)
  Total: 5439 GPU hours on 16×A40
  All found best VLMs surpass LLaVA-1.5-7B equivalent (CLIP-Vicuna)

Challenge: chicken-and-egg — must train to evaluate, cannot pre-filter
```

Mordal 三阶段 solution：(1) CKA clustering 将相似候选分组减少搜索空间；(2) SHA early stopping 快速淘汰差 clusters；(3) Scaling prediction 从部分数据训练预测完整性能。

术语一般如何实现？如何使用？
开源：https://github.com/SymbioticLab/Mordal。接口：`mordal.query_for_model(data, task, pretrained_ve_zoo, pretrained_llm_zoo, ...)`。用户提供 alignment data + target task data + model zoo list。支持配置 projector 架构（MLP/Linear）、freeze 策略（freeze VE, LoRA fine-tune LLM）、clustering/exploration/prediction 超参。在 7VE×7LLM×6 tasks 评估中 8.9×–11.6× speedup，5/6 任务选出 Top-1。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---

## Observational Scaling Law for VLM Alignment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Observational scaling law for VLM alignment 是 Mordal 发现的经验规律：固定 VLM pretrained 模型参数，alignment 性能（下游任务 error）与对齐训练数据量存在 log-linear 关系——log(Error) ∝ log(data_ratio)。与标准 scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022) 的区别：标准 law 研究 pretraining 时 compute/model/data scale 与 loss 的关系；observational scaling law (Ruan et al. 2024 NeurIPS Spotlight, Lin et al. 2024) 关注固定模型参数下从 observational data 预测不同 training scale 的最终性能。Mordal 将此概念首次应用于 VLM alignment 场景。

从算法pipeline角度拆解术语，给出具体例子。
Scaling Prediction (Algorithm 1)：
```
for each candidate c in remaining:
    P = []; r = 0.125  // start from 1/8 data
    while True:
        train_from_checkpoint(c, data_ratio=r)
        Err = evaluate(c, D_task)
        P.append((log(r), log(Err)))
        if len(P) > p (e.g., 3):
            f_c = LinearRegression(P)  // log(Err) = α·log(r) + β
            if fitting_loss(f_c) < δ (e.g., 5e-5): break
        r = r / u  // reduce data (u=2)
    predicted_err = exp(f_c(log(1)))  // predict at r=1 (full data)
select argmin(predicted_err)
```
关键发现（Figure 9）：log-linear 仅在**一定训练样本量后**出现（consistent with Ruan et al.）；不同 VLM 候选有**不同斜率**——解释不同组合的收敛速度差异。从大→小 ratio 递减以利用已有 checkpoint 节省计算。

术语一般如何实现？如何使用？
Mordal 中仅用于 intra-cluster evaluation（inter-cluster 阶段 speculative prediction 不可靠）。默认 p=3, δ=5e-5。其他相关工作：Ruan et al. (2024) 通过 PCA 从 80+ LLM 提取 latent capability 度量（PC-1 解释 ~80% 方差），证明 capability 与 compute 呈 log-linear (R²>0.9)，可预测 GPT-4 等未公开模型的性能。Mordal 的差异：关注单 VLM 在 alignment 过程中的 data scaling，而非跨模型 compute scaling。限制：仅在 7B 级模型验证；log-linear 关系需要一定数据阈值后才可观测。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---

## SHA (Successive Halving Algorithm) for Model Evaluation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Successive Halving Algorithm (SHA) 是 Jamieson & Talwalkar (AISTATS 2016) 的超参数优化 bandit 算法。将有限计算预算均分给所有候选，通过多轮淘汰赛（每轮保留 top 1/η）快速筛选最优候选。每轮预算递增（×η），survivor 获得更充分评估。在 Mordal 中，SHA 用作 inter-cluster evaluation 的 early stopping——快速淘汰差 cluster，资源集中于有潜力的 cluster。

从算法pipeline角度拆解术语，给出具体例子。
Mordal 中 SHA 流程：
```
representatives = [cluster_medoids]; budget = b = 0.03
while len(representatives) > top_k_inter:
    for each rep: train(rep, data_ratio=budget); score = evaluate(rep)
    representatives = top_k(score, k = ceil(len/eta))
    budget *= eta
// Example: 7 reps, eta=2
// Rung0: budget=3%, 7→4; Rung1: budget=6%, 4→3 (≤top_k_inter=3, done)
// Total: 7×3%+4×6% ≈ 0.81× one full training
```

术语一般如何实现？如何使用？
Mordal 限制 SHA 于 inter-cluster evaluation（rough filtering）。η=2, R=0.125, b=0.03。消融实验：SHA 显著减少搜索时间但 aggressive elimination 可能误淘汰有潜力候选（如 AI2D 的 SigLIP-Qwen 在早期表现不佳）。因此 Mordal 在 intra-cluster 阶段使用 scaling prediction 替代 SHA 做精细评估。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---

## Two-Step VLM Candidate Clustering

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Step VLM Candidate Clustering 是 Mordal 避免 O(m²n²) pair-wise CKA 的聚类策略。直接对所有 m×n VLM 候选两两计算 CKA 需 O(m²n²) 次评估。两步策略：(1) 先在 VE 空间做 CKA 聚类（O(m²) 次）；(2) 再在每个 VE cluster 内基于 medoid VE 输出对 LLM 做 CKA 聚类（O(C_ve·n²) 次，C_ve≪m²）；(3) Cartesian product 合成 VLM candidate clusters。避免了计算 dissimilar VE 与不同 LLM 组合间的 CKA——不同 VE cluster 的候选性能差异大，不需要 cluster 间细粒度比较。

从算法pipeline角度拆解术语，给出具体例子。
```
Algorithm 2:
Step 1 - VE Clustering:
  for each pair in 7 VEs: dist = 1 - CKA(VE_A, VE_B)
  C_ve = HierarchicalClustering(dist, t_ve=0.7)
  // e.g., 3 clusters: {CLIP,SigLIP,DFN}, {InternViT,DINOv2}, {EVA-CLIP,ConvNeXt}

Step 2 - LLM Clustering (per VE cluster):
  for each VE_cluster:
      medoid = PickMostCentral(VE_cluster)
      fixed_out = WarmupProjector(medoid(images))  // 10 rounds
      for each pair in 7 LLMs: dist = 1 - CKA(LLM_A.last_hidden, LLM_B.last_hidden)
      C_llm = HierarchicalClustering(dist, t_llm=0.8)
      // e.g., {Vicuna,Llama-2}, {Llama-3,Mistral,Qwen}, {Phi-3,Gemma}

Step 3 - Cartesian: C_vlm = {VE_c × LLM_c for each combination}
// Total: ~9-15 candidate clusters (vs 49 individuals)
```

术语一般如何实现？如何使用？
使用 MinibatchCKA + `scipy.cluster.hierarchy`。关键超参：t_ve=0.7, t_llm=0.8。消融（Table 5）：t_ve=0.5→τ=0.52（太粗），t_ve=0.9→τ=0.86 但 1041h（太细）。LLM 聚类使用 last hidden state（最佳聚类性能）。Warmup round=10 确保 medoid projector 充分训练产生有意义的 LLM 输入表示。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---

## LoRA (Low-Rank Adaptation) for VLM Fine-Tuning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
LoRA (Low-Rank Adaptation) 是 Hu et al. (ICLR 2022) 提出的参数高效微调（PEFT）方法。冻结预训练模型权重 W₀ ∈ R^{d×k}，注入可训练低秩分解 B·A（B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)），前向计算 h = W₀x + BAx。推理时 BA 融入 W₀ 无额外延迟。相比 full fine-tuning，LoRA 将可训练参数减少 10000×，GPU memory 减少 3×。

从算法pipeline角度拆解术语：
Mordal 中使用 LoRA 对 LLM 进行 fine-tuning（当 `freeze_llm=False` 时）：
```
# VLM alignment with LoRA:
for each training step:
    img_emb = VE(images)               # frozen
    aligned_emb = Projector(img_emb)   # trainable (from scratch)
    text_emb = TokenEmbed(text)
    all_emb = concat([aligned_emb, text_emb])
    for layer in LLM:
        h = Attention(QKV_proj(x))     # QKV: W₀x + BAx (LoRA)
        h = FFN(gate_up_down(x))       # FFN: W₀x + BAx (LoRA)
    loss = CrossEntropy(output, labels)
# 仅更新 Projector 参数 + 所有 LoRA 的 B,A 矩阵
```
Mordal 在 LLM 的 Q/K/V/O projection 和 FFN 层注入 LoRA（通过 PEFT 库配置），仅训练 LoRA 参数 + 从头训练 Feature Projector。

术语一般如何实现？如何使用？
HuggingFace PEFT 库（https://github.com/huggingface/peft）实现。典型配置：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])`。Mordal 通过 `vlm_kwargs={'freeze_llm': False}` 启用 LoRA fine-tuning。对于 VLM alignment，LoRA 在 7B LLM 上仅增加 ~10M 可训练参数（vs 7B full），大幅降低每候选计算成本。局限：某些任务 full fine-tuning 仍优于 LoRA；LoRA rank 和 target modules 需经验性选择。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

## OmniAlignNet

术语是什么？
OmniAlignNet 是 NVIDIA OmniVinci 论文提出的跨模态对齐网络模块，用于在共享的 omni-modal 潜在空间中强化视觉嵌入和音频嵌入的对齐。其核心思想是：对于同一视频的视觉帧和同步音频轨，两者存在内在的语义互补关系（视觉提供空间/物体信息，音频提供语音/环境声信息），通过 CLIP-style 双向对比学习，使同一视频的视觉-音频嵌入对在共享空间中相互拉近，不同视频的拉远。

具体流程：给定视频的视觉嵌入序列 $\mathbf{E}_v \in \mathbb{R}^{N_v \times C}$ 和音频嵌入序列 $\mathbf{E}_a \in \mathbb{R}^{N_a \times C}$（$C$ 为潜在维度），初始化可学习 query $\mathbf{Q}_v, \mathbf{Q}_a \in \mathbb{R}^{1 \times C}$，通过 cross-attention 将变长序列投影为固定大小 $(1 \times C)$ 的表示；再经 3 层 self-attention + L2 归一化得到 $\mathbf{V}, \mathbf{A} \in \mathbb{R}^{K \times C}$（$K$ 为 batch 中视频数）。对比损失为对称交叉熵：$\mathcal{L}_{\text{o-align}} = \frac{1}{2}(\mathcal{L}_{v \to a} + \mathcal{L}_{a \to v})$，其中 $\mathcal{L}_{v \to a} = -\frac{1}{K}\sum_i \log\frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}$，$s_{ij} = \mathbf{V}_i^T \mathbf{A}_j$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
OmniAlignNet 位于视觉 projector 和音频 projector 之后、LLM backbone 之前，作为独立的对齐模块运行。在训练阶段，OmniAlignNet 的对比损失 $\mathcal{L}_{\text{o-align}}$ 作为辅助损失加入总训练目标，与 LM loss 联合优化。在推理阶段，OmniAlignNet 的对齐权重已固化在模型参数中，视觉和音频嵌入直接通过训练好的对齐空间进入 LLM，无需额外计算对比损失。核心计算流程：
```
# 训练阶段
E_v = VisualProjector(ViT(video_frames))    # [N_v, C]
E_a = AudioProjector(AF_Whisper(audio))     # [N_a, C]
V = L2Norm(SelfAttn3(CrossAttn(Q_v, E_v)))  # [K, C]
A = L2Norm(SelfAttn3(CrossAttn(Q_a, E_a)))  # [K, C]
L_align = CLIPContrastiveLoss(V, A)         # 辅助损失

# 推理阶段
# OmniAlignNet 对齐参数已固化，视觉/音频嵌入直接送入 LLM
omni_seq = [V_embeds, A_embeds]  # 按 TEG 重排后输入 LLM
output = LLM(omni_seq, text_prompt)
```

消融实验：+TEG+CRTE baseline 平均得分 50.25，加入 OmniAlignNet 后提升至 52.59 (+2.34)，其中 Omnibench 提升最显著 (+6.1)，证明跨模态对比对齐对 image-audio 联合理解尤为关键。

术语一般如何实现？如何使用？
受 ImageBind [Girdhar et al., CVPR 2023] 启发，使用共享嵌入空间绑定多模态。实现上基于 PyTorch，OmniAlignNet 模块包含：可学习 query 向量、cross-attention 层（将变长序列压缩为固定维度）、3 层 self-attention（增强模态内和跨模态交互）、L2 归一化、以及对称 CLIP 对比损失。训练时 batch 内需要正样本对（同一视频的视觉+音频），batch size 越大对比效果越好。开源实现：GitHub (NVlabs/OmniVinci)，社区 PyTorch 实现 (kyegomez/OmniAlignNet)。使用场景：任何需要对齐视觉和音频模态的多模态模型，尤其适用于视频理解任务。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Temporal Embedding Grouping (TEG)

术语是什么？
Temporal Embedding Grouping (TEG) 是 OmniVinci 提出的时序嵌入分组机制，用于在 omni-modal 嵌入序列中编码视觉和音频信号的**相对时序关系**。核心思想：按固定时间窗口 $T_G$ 将时间轴划分为多个 chunk，根据每个视觉帧和音频采样点的时间戳，将其对应的嵌入分配到相应的时序组中，然后按时间顺序交叠排列各组，形成 $[G_v^1, G_a^1, G_v^2, G_a^2, ...]$ 的 omni-modal 嵌入序列。

具体例子：假设 $T_G$ 为某时长，4 个视觉帧时间戳为 $t_v^1 < t_v^2 < T_G < t_v^3 < t_v^4 < 2T_G$，4 个音频采样时间戳为 $t_a^1 < t_a^2 < T_G < t_a^3 < t_a^4 < 2T_G$。TEG 将嵌入分组为 $G_v^1 = \{\mathbf{e}_v^{t_v^1}, \mathbf{e}_v^{t_v^2}\}, G_v^2 = \{\mathbf{e}_v^{t_v^3}, \mathbf{e}_v^{t_v^4}\}, G_a^1 = \{\mathbf{e}_a^{t_a^1}, \mathbf{e}_a^{t_a^2}\}, G_a^2 = \{\mathbf{e}_a^{t_a^3}, \mathbf{e}_a^{t_a^4}\}$，最终序列为 $[\mathbf{e}_v^{t_v^1}, \mathbf{e}_v^{t_v^2}, \mathbf{e}_a^{t_a^1}, \mathbf{e}_a^{t_a^2}, \mathbf{e}_v^{t_v^3}, \mathbf{e}_v^{t_v^4}, \mathbf{e}_a^{t_a^3}, \mathbf{e}_a^{t_a^4}]$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
TEG 在视觉/音频嵌入完成 projection 之后、送入 LLM 之前执行，是对 token 序列顺序的确定性重排操作，不涉及可学习参数。计算流程：
```
输入: visual_embeds {e_v_i, t_v_i}, audio_embeds {e_a_j, t_a_j}, group_duration T_G

groups = {}
for each embed in visual_embeds + audio_embeds:
    g_idx = floor(t / T_G)  # 根据时间戳确定所属组
    groups[g_idx].append(embed)

# 按组索引升序排列
sorted_seq = []
for g_idx in sorted(groups.keys()):
    sorted_seq.extend(groups[g_idx])  # 组内视觉嵌入在前，音频在后

输出: omni_modal_sequence = sorted_seq
```
TEG 的核心假设是：LLM 的 position embedding 将序列位置隐式编码为相对时序信息，同一时间窗口内的视觉和音频嵌入在序列中相邻，LLM 的 self-attention 能更好地捕获跨模态时间对应关系。

术语一般如何实现？如何使用？
TEG 是确定性算法，无需训练，在数据预处理/批处理阶段执行。实现要点：(1) $T_G$ 的选择需平衡时序粒度——太小导致组过多、序列碎片化，太大导致时序区分度不足，OmniVinci 论文中 $T_G$ 为超参数通过消融确定；(2) 组内排列顺序（视觉→音频或交替）影响 LLM attention pattern，OmniVinci 采用视觉嵌入在前、音频嵌入在后的固定顺序；(3) 与 CRTE 互补——TEG 提供相对顺序，CRTE 提供绝对时间戳。消融实验中 TEG 使平均得分从 45.51 (Token Concatenation Baseline) 提升至 47.72 (+2.21)，Dailyomni 增益最显著 (+6.44)。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Constrained Rotary Time Embedding (CRTE)

术语是什么？
Constrained Rotary Time Embedding (CRTE) 是 OmniVinci 提出的绝对时间戳编码方法，用于将时间信息直接注入 omni-modal 嵌入向量中。CRTE 继承 RoPE (Rotary Position Embedding) 的旋转编码思想，但将其从编码序列位置扩展为编码绝对时间戳。核心创新是引入最大时间范围 $T_{\max}$ 约束和几何级数频率设计，实现多尺度时间编码——高频维度捕获细粒度时间差（如毫秒级声音事件），低频维度编码粗粒度长时间关系（如场景切换）。

三个计算阶段：
1. **基础频率生成**: $\omega_i = 2\pi / (T_{\max} \theta^{i/C}), i=0,...,C-1$，其中 $\theta \geq 1$ 控制频率缩放，$C$ 为嵌入维度，$T_{\max}$ 定义最粗时间分辨率。小 $i$（前几个维度对）分母小 → $\omega_i$ 大 → 高频 → 对细粒度时间差敏感；大 $i$（后几个维度对）分母大 → $\omega_i$ 小 → 低频 → 编码长程时间关系。

2. **频率调制**: $\Omega_{i,j} = \omega_i \cdot t_j$，将基础频率与实际时间戳 $t_j$ 相乘。

3. **旋转变换**: $\text{CRTE}(\mathbf{x}, \Omega) = \mathbf{x} \odot \cos(\Omega) + \text{RotateHalf}(\mathbf{x}) \odot \sin(\Omega)$，其中 $\text{RotateHalf}(\mathbf{x}) = [-x_2, x_1, -x_4, x_3, ..., -x_C, x_{C-1}]$，将 $C$ 维嵌入分成 $C/2$ 个独立的 2D 旋转平面。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
CRTE 在 TEG 之后、LLM 输入之前应用，作为一个无参数的确定性变换层。伪代码：
```
输入: embedding x [C], timestamp t_j, T_max, theta

# 阶段1：基础频率 (可预计算)
omega = [2*pi / (T_max * theta^(i/C)) for i in range(C)]

# 阶段2：频率调制
Omega = [omega[i] * t_j for i in range(C)]

# 阶段3：旋转变换
x_rotated = zeros(C)
for i in range(0, C, 2):
    cos_val, sin_val = cos(Omega[i]), sin(Omega[i])
    x_rotated[i]   = x[i] * cos_val - x[i+1] * sin_val  # 2D 旋转
    x_rotated[i+1] = x[i+1] * cos_val + x[i] * sin_val

输出: x_rotated  # 时间编码后的嵌入
```

与 RoPE 的关键区别：(1) RoPE 编码序列位置 $pos$，CRTE 编码绝对时间戳 $t_j$；(2) CRTE 引入 $T_{\max}$ 约束，使频率范围可控；(3) CRTE 的频率呈几何级数分布，天然实现多尺度。消融实验中 CRTE (50.25) 优于 Learned Time Embedding (47.30) 和 RoTE (47.80)，证明约束频率+旋转编码对绝对时间的编码效果最好。

术语一般如何实现？如何使用？
基于 PyTorch 实现，作为无参数模块嵌入到模型的前向传播中。$T_{\max}$ 和 $\theta$ 为超参数，OmniVinci 论文中未明确给出具体值，通常 $T_{\max}$ 设为视频最大时长（如 120s），$\theta$ 通过消融确定。对于视觉嵌入，$t_j$ 为帧采样时间戳；对于音频嵌入，$t_j$ 为音频帧的采样时间点。CRTE 与 TEG 组合使用：TEG 提供相对时序嵌入序列的 token 排列，CRTE 在此基础上为每个 token 注入绝对时间信息。CRTE 在视频理解、audio-visual synchronization、temporal reasoning 等任务中表现突出。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Omni-Modal Data Engine

术语是什么？
Omni-Modal Data Engine 是 OmniVinci 提出的全模态数据合成流水线，用于从视频中自动生成高质量的 omni-modal（视觉+音频）对话数据。流水线分三步：(1) **独立模态 Captioning**：使用预训练视觉 captioning 模型（如 InternVL3）和音频 captioning 模型（如 Qwen2.5-Omni 的音频模块）分别对视频的视觉轨和音频轨生成独立标注；(2) **跨模态纠错与总结**：使用 LLM（如 Qwen3）接收视觉和音频两个独立 caption，基于双方信息进行纠错和综合，生成准确的 omni-modal joint caption；(3) **QA 合成**：使用 reasoning LLM（如 DeepSeek-R1）从 omni-modal caption 中合成带推理链的 QA 对。

核心动机是解决 **Modality-Specific Hallucination**（模态特定幻觉）：纯视觉 captioning 模型看不到音频信息，可能将深海探索视频误判为"人类科技"；纯音频 captioning 模型看不到视觉信息，可能仅凭语音内容误判为"地球内部"。跨模态纠错 LLM 综合两者信息后可生成正确的综合描述。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Data Engine 是离线数据合成流水线，在模型训练前执行：
```
输入: 带音频轨的视频集合

for each 2-min video segment:
    # Step 1: 独立 captioning（可并行）
    vis_caption = VisualCaptioningModel(video_frames)
    aud_caption = AudioCaptioningModel(audio_track)

    # Step 2: 跨模态纠错
    prompt = f"Visual caption: {vis_caption}\nAudio caption: {aud_caption}\n\
              Please correct and summarize into a joint caption."
    joint_caption = LLM_Corrector(prompt)

    # Step 3: QA 合成
    qa_pairs = ReasoningLLM_Synthesize(joint_caption)
    # 生成 MCQ 或开放式 QA，含 reasoning trace

输出: omni-modal QA dataset (3.6M conversations)
```
最终生成的 omni-modal 数据占训练数据总量的 15%（omni QA 12% + omni captioning 3%），配合 modality-specific 数据（image 36%, sound 21%, speech 17%, video 11%）共 24M 样本。

术语一般如何实现？如何使用？
Data Engine 是离线 pipeline，各组件可独立替换：(1) Captioning 模型可根据场景替换为更强的模型（如 GPT-4o 替代 InternVL3）；(2) 跨模态纠错 LLM 可使用任何 instruction-tuned LLM；(3) QA 合成可用 reasoning LLM（DeepSeek-R1、Qwen3 等）生成带 CoT 的复杂问题。关键在于跨模态纠错步骤——直接拼接两个 caption 给 LLM 不够，需明确 prompt 指示 LLM 识别并解决两个 modal caption 的矛盾。Data Engine 生成的数据用于 Explicit Omni-Modal Learning，与 Implicit Learning（利用视频自带 audio track 的隐式监督）互补。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Modality-Specific Hallucination

术语是什么？
Modality-Specific Hallucination（模态特定幻觉）是 OmniVinci 提出的概念，指单模态感知模型由于缺乏其他模态的互补信息而产生的系统性理解错误。具体表现为：纯视觉模型只能看到画面但听不到语音，可能将"深海探索视频"误解为仅关于"人类科技"（因为画面上有潜艇和设备，但语音讨论的是海洋生物）；纯音频模型只能听到语音但看不到画面，可能仅凭讨论内容将同样的视频误解为关于"地球内部"。

这一概念的学术价值在于：它从理论上论证了 omni-modal 联合理解的必要性——单模态感知不仅是不完整的，而且是**系统性错误的**，因为它缺乏跨模态纠错的机制。这与传统的 multi-modal fusion 有本质区别：传统 fusion 追求"更多信息→更好理解"，而 modality-specific hallucination 揭示了"信息不完整→**错误**理解"的定性差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Modality-Specific Hallucination 是数据层面的概念，在 Data Engine 的 Step 2（跨模态纠错）中被显式解决。该概念指导了数据合成策略的设计：Data Engine 中的跨模态纠错 LLM 的核心任务就是识别和修正 modality-specific hallucination。在训练策略中，Explicit Omni-Modal Learning 通过提供"正确"的 omni-modal 标注，直接训练模型抵抗 modality-specific hallucination。

术语一般如何实现？如何使用？
该概念用于：(1) 指导 omni-modal 数据合成 pipeline 设计——必须包含跨模态纠错步骤；(2) 评估 omni-modal 模型——设计测试用例检验模型是否会在单模态信息不足时产生幻觉；(3) 论证 omni-modal 模型的必要性——纯 vision-language 或 audio-language 模型存在系统性的幻觉风险。OmniVinci 在 Omnibench (Image-Audio QA) 上 45.74（+OmniAlignNet）的表现验证了解决 modality-specific hallucination 的有效性。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Implicit and Explicit Omni-Modal Learning

术语是什么？
OmniVinci 提出的两种互补的 omni-modal 学习策略：

**Implicit Omni-Modal Learning（隐式全模态学习）**：利用现有 video QA 数据集中自然存在的同步音频轨进行隐式监督。大多数先前 video LLM 仅使用视频的视觉帧，丢弃了同步音频轨中的信息。OmniVinci 将音频轨作为额外输入，让模型在 video QA 任务中隐式学习视觉-音频的联合理解，无需额外的 omni-modal 标注。关键洞察："Videos are naturally omni-modal when visual and audio streams are present simultaneously but remains under explored."

**Explicit Omni-Modal Learning（显式全模态学习）**：通过 Omni-Modal Data Engine 合成带有显式 omni-modal 标签的对话数据，直接监督模型的视觉-音频联合理解能力。与 Implicit Learning 的"间接"监督不同，Explicit Learning 的 QA 对明确要求模型同时利用视觉和音频信息回答问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Implicit Learning 数据流
Video QA dataset (如 Video-MME) 
  → 使用 video frames + audio track 作为输入（而非仅 frames）
  → 在 video QA 任务上 fine-tune
  → 模型隐式学习利用 audio 辅助视觉理解

# Explicit Learning 数据流
Video corpus with audio
  → Omni-Modal Data Engine 合成 omni-modal QA
  → 明确要求模型结合视觉+音频信息回答
  → 模型显式学习 omni-modal 联合推理

# 训练混合策略（Omni-Modal Joint Training）
每个 batch: 随机采样
  - modality-specific data (image-only, audio-only, video-only)
  - omni-modal implicit data (video QA with audio track)
  - omni-modal explicit data (synthetic omni-modal QA)
```

消融实验：Visual Alone baseline (Video-MME w/o sub. 61.67) → +Implicit Learning (63.76, +2.09) → +Explicit Learning (67.37, +5.70)，证明两种学习策略均有显著增益且互补。Implicit Learning 即使在有 subtitle 的情况下也带来提升 (66.37→66.96)，说明直接从音频学习与从文本 subtitle 学习是不同的信息通道。

术语一般如何实现？如何使用？
Implicit Learning 实现简单——在 video QA 训练时附加音频 encoder 的输出作为额外 token，无需额外数据标注，可以充分利用任何已有 video QA 数据集。Explicit Learning 需要先运行 Data Engine 合成数据，成本较高但效果更好。最佳实践是两者结合使用，如 OmniVinci 的 Omni-Modal Joint Training 阶段。训练细节：200B tokens, cosine LR schedule (warmup 3%), base LR=2e-5, vision/audio encoders frozen。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

## Omni-Modal Joint Training

术语是什么？
Omni-Modal Joint Training 是 OmniVinci 提出的两阶段渐进式训练策略中的第二阶段（第一阶段为 Modality-Specific Training），目标是使 LLM 获得统一的 omni-modal 理解能力。该阶段的核心设计是**多模态数据混合采样**——在每个 batch 中同时包含：(1) modality-specific 数据（纯视觉、纯音频、纯视频），防止单一模态能力退化；(2) omni-modal implicit learning 数据（带音频轨的 video QA），提供隐式跨模态监督；(3) omni-modal explicit learning 数据（Data Engine 合成的 omni-modal QA），提供显式跨模态监督。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Omni-Modal Joint Training 是连接各模态训练阶段的枢纽。全训练流程共 7 个阶段：
```
Stage 1-5: Vision Training (follow NVILA recipe)
  1. Vision Projector Alignment → 2. Vision Encoder Alignment
  → 3. Vision Pre-Training → 4. Image Instruction Tuning
  → 5. Video Instruction Tuning
  → 产出 "Vision Preliminary Checkpoint"

Stage 6: Audio Training
  1. Audio Projector & Encoder Alignment (50K audio-language pairs)
  2. Audio Instruction Tuning (9.6M audio-SFT samples, full model)
  → 注意：此时 visual understanding 能力会退化

Stage 7: Omni-Modal Joint Training ← 本术语
  数据: 24M samples (image 36%, sound 21%, speech 17%, omni 15%, video 11%)
  配置: cosine LR + linear warmup (3%), base LR=2e-5
  冻结: vision encoder + audio encoder (仅训练 projector + LLM)
  token 总量: ~200B (0.2T)
  → 恢复并提升 visual + audio 联合理解能力
```

训练配置的关键约束：(1) vision/audio encoder 冻结——防止大规模多模态训练破坏预训练好的编码器表示；(2) 低学习率 (2e-5)——在前阶段 checkpoint 附近微调；(3) modality-specific 数据占比 > omni-modal 数据——防止遗忘单模态能力；(4) 200B tokens 总量远小于 Qwen2.5-Omni 的 1.2T，证明良好的架构+数据设计可大幅降低训练成本。

术语一般如何实现？如何使用？
基于 PyTorch 分布式训练（NVIDIA DGX H100 集群），使用标准 LM loss + 可选的 OmniAlignNet contrastive loss 作为联合优化目标。多模态 batch 采样策略：按数据集原始大小加权采样（weighted sampling），确保小数据集不被忽略。Omni-Modal Joint Training 可视为"多模态 SFT + 能力恢复"阶段，后续可选 GRPO post-training（18K omni-modal MCQ, rollout=8, temperature=1.0, top-p=0.99）进一步提升 omni-modal reasoning 能力。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

---

## Ring Attention (Sequence-Parallel Distributed Attention)

术语是什么？
Ring Attention是一种序列并行(Sequence Parallelism)方法，将长序列的KV tensors按块分片到多个GPU上。每个GPU对其本地KV块执行blockwise attention计算，同时将KV块沿GPU环(ring)传输到下一个GPU。计算和通信重叠：GPU计算当前KV块的attention时，同时将前一块的KV异步发送给peer。这样每个token最终能attend到序列中的所有token，而不需要将整个KV cache复制到每个GPU。

从算法pipeline角度拆解术语：
Ring Attention核心算法流程（以8 GPU, 序列长度S为例，每GPU持有S/8个token的Q分片）：
```
for step in range(num_gpus):
    kv_rank = (local_rank + step) % num_gpus
    // 1. 计算当前KV块的blockwise attention
    attn_score = Q_local @ K[kv_rank].T  // scaled dot-product, 本地softmax累加
    attn_output += p @ V[kv_rank]        // softmax_weighted V累加
    // 2. 异步发送/接收KV块 (与step+1重叠)
    if step < num_gpus - 1:
        async_send(K_local, V_local, next_gpu)
        async_recv(K_next, V_next, prev_gpu)
```
通信复杂度O(S)而非O(S²)。ParallelKittens通过fused单kernel实现inter-SM overlapping：专用communication SM将下一块KV批量传输到local HBM（避免remote L2 cache miss导致的重复传输），compute SM执行FlashAttention计算，SM分配自动调优。

术语一般如何实现？如何使用？
xDiT baseline使用NCCL P2P send + FlashAttention-3在独立CUDA stream上做coarse-grained overlap。ParallelKittens将其融合为单个kernel（LCSC模板），通过num_comm_sms控制SM分配，实现1.07x-4.08x speedup（B=16, H=16, D=128）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## DeepSpeed-Ulysses (All-to-All Sequence Parallelism)

术语是什么？
DeepSpeed-Ulysses在self-attention前后执行all-to-all通信：进入attention前从sequence-sharded切换到head-sharded（每GPU拥有所有token但仅部分head），使self-attention可直接head-independently执行而无需通信；attention后all-to-all恢复sequence-sharded。与Ring Attention的渐进式块传输不同，Ulysses通过两次全量4D all-to-all完成并行，通信开销集中在all-to-all阶段。

从算法pipeline角度拆解术语：
```
// 8 GPU, 输入X: (B, S/8, H, D) per GPU — sequence-sharded
// Step 1: S→H shard转换 (all-to-all)
X_head = all_to_all(X, scatter_dim=S, gather_dim=H)  // (B, S, H/8, D)
// Step 2: head-independent self-attention (无通信)
Y = attention(X_head)  // per-GPU Q/K/V shapes: (B, S, H/8, D)
// Step 3: H→S shard恢复 (all-to-all)
Y_seq = all_to_all(Y, scatter_dim=H, gather_dim=S)     // (B, S/8, H, D)
```
瓶颈：all-to-all沿inner dimension（H/S dim），NCCL不支持非连续layout的直接collective，baseline需先reshape再通信。PK实现细粒度4D tile级P2P all-to-all绕过此overhead，kernel代码<50行。

术语一般如何实现？如何使用？
YunChang baseline: NCCL+reshape。ParallelKittens: Tile级P2P直接传输，4D (B,S,H,D) all-to-all。达成1.01x-1.39x speedup（B=16, H=128, D=128）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## Expert Parallelism (MoE) — Token Dispatch and Gather

术语是什么？
Expert Parallelism将MoE层的多个expert FFN权重均匀分布到不同GPU，输入tokens通过router选出top-k expert后，经all-to-all dispatch发送到对应GPU，执行expert MLP计算，再all-to-all gather回原GPU。通信开销源于两次all-to-all，消息粒度细（per-token KB级），对通信库的连续大块传输设计不友好。

从算法pipeline角度拆解术语：
```
// 输入tokens: (T, H), 每GPU T/G个token, E experts, k=top-k
// Step 1: Router (local per GPU)
logits = tokens @ W_router            // (T/G, E)
topk_vals, topk_idx = topk(logits, k) // per-token top-k expert assignment

// Step 2: All-to-All Dispatch (token → expert GPU)
permuted_tokens = all_to_all(tokens, indices=topk_idx)  // 每个token发送到持有其topk expert的GPU
// 通信量: T × d_model bytes

// Step 3: Expert MLP Compute (per GPU, independent)
for each received_token:
    expert_output = FFN_expert(received_token, W_expert)

// Step 4: All-to-All Gather (result → original GPU)
output = all_to_all(expert_output, reverse_indices)
```
瓶颈：fine-grained all-to-all通信，每token需发送到up to k个expert GPU。PK通过TMA tile级P2P与Grouped GEMM做intra-SM overlap，fused kernel <40行device code，vs Comet手写kernel达成0.92-1.22x性能。

术语一般如何实现？如何使用？
通信库：DeepEP (DeepSeek)、Comet、FlashDMoE。PK: TMA store_async + intra-SM overlap。与Data/Tensor Parallelism混合使用（3D parallelism）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## Modulated Quantization (MoDiff)

术语是什么？
Modulated Quantization（调制量化）是MoDiff论文提出的核心算法技术，通过利用扩散采样过程中相邻时间步之间激活的相似性，将每层线性算子的计算从直接量化激活 $Q(\mathbf{a}_t)$ 重构为量化时序差分 $Q(\mathbf{a}_t - \mathbf{a}_{t+1})$，然后将差分计算结果累加到前一时间步的输出上：$\hat{\mathbf{o}}_t = \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + \hat{\mathbf{o}}_{t+1}$。其数学正确性源于线性算子 $\mathcal{A}$ 的加法分解性质：$\mathcal{A}(\mathbf{a}_t) = \mathcal{A}(\mathbf{a}_t - \mathbf{a}_{t+1}) + \mathcal{A}(\mathbf{a}_{t+1})$。

核心洞察：时序差分 $\mathbf{a}_t - \mathbf{a}_{t+1}$ 的分布范围比原始激活小10×以上，且更集中、异常值更少（Figure 1b中橙色vs蓝色violin plot），因此相同位宽的量化误差大幅降低。Theorem 4.3证明了量化误差界 $\|\mathbf{x} - Q(\mathbf{x})\|_2^2 \le (\max(\mathbf{x}) - \min(\mathbf{x}))^2 d / (2^b - 1)^2$，即误差正比于输入范围平方——范围缩小10×意味着误差降低100×。

从算法pipeline角度拆解术语：
```
// 标准PTQ方法（baseline）：逐层独立计算
for t = T, T-1, ..., 1:  // T个diffusion steps
    for layer l in 1..L:
        â_t^{(l)} = Q(a_t^{(l)})                      // 量化原始激活
        ô_t^{(l)} = A^{(l)}(â_t^{(l)})                 // 矩阵乘法

// MoDiff调制量化：逐层增量计算
for t = T, T-1, ..., 1:
    for layer l in 1..L:
        // Step 1: 量化时序差分（而非原始激活）
        diff = a_t^{(l)} - â_{t+1}^{(l)}               // 计算差分
        diff_q = Q(diff)                               // 量化差分（范围小→误差小）
        // Step 2: 增量计算输出
        ô_t^{(l)} = A^{(l)}(diff_q) + ô_{t+1}^{(l)}    // 累加到前一步输出
```

时序差分性质的量化误差分析：
```
激活分布:          范围变化大, 异常值多, 长尾分布    → 低bit量化困难
时序差分分布:      范围一致(小), 集中, 几乎无异常值  → 低bit量化容易

Err(原始激活) ≈ (range_a)² / (2^b - 1)²
Err(时序差分) ≈ (range_diff)² / (2^b - 1)²
若 range_diff < range_a / 10 → Err(差分) < Err(原始) / 100
等效：可用低3-4 bits达到相同误差界
```

术语一般如何实现？如何使用？
论文实现于PyTorch框架：对每个线性层（Conv2d、Linear），在forward pass中用MoDiff替换标准量化计算。需从原始权重中去除bias项（保证纯线性以允许加法分解）。第一时间步T使用全精度激活作为warm-up（4-5步收敛到可忽略误差）。MoDiff与具体量化方法（Q-Diffusion、动态per-channel/tensor量化）正交，可直接叠加于现有PTQ方法。在DDIM CIFAR-10上W8A3时FID=4.14（vs Q-Diff FID=143.39，10× Bops节省vs FP32）。支持0-bit skipping：当时序差分幅度低于阈值时跳过计算（此时等价于caching方法的特例）。开源：https://github.com/WeizhiGao/MoDiff。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---

## Error-Compensated Modulation

术语是什么？
Error-Compensated Modulation（误差补偿调制）是MoDiff框架的第二个核心组件，解决标准调制方法中量化误差在时间步间累积的问题。通过引入中间变量 $\hat{\mathbf{a}}_t$ 存储经过量化后的近似激活，使当前时间步的量化误差 $\mathbf{e}_t = \mathbf{a}_t - \hat{\mathbf{a}}_t$ 被显式追踪，并在下一时间步通过输入反馈被算子补偿。

具体机制：$\hat{\mathbf{a}}_t = Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1}) + \hat{\mathbf{a}}_{t+1}$。当前步的量化误差被保留在 $\hat{\mathbf{a}}_t$ 中（因为 $\hat{\mathbf{a}}_t = \mathbf{a}_t - \mathbf{e}_t'$），下一步计算的差分基是 $\hat{\mathbf{a}}_{t+1}$ 而非 $\mathbf{a}_{t+1}$，从而将上一步遗漏的 $A(\mathbf{e}_{t+1})$ 重新纳入计算。

从算法pipeline角度拆解术语：
```
// 标准调制（无误差补偿）——误差指数增长
// õ_t = A(Q(a_t - a_{t+1})) + õ_{t+1}
// Theorem 4.4: 误差界 ∝ Σ 2^{T-k-1} × c ∥A∥² ∥d_k∥²  (指数增长)

// MoDiff误差补偿调制——误差指数衰减
// â_t = Q(a_t - â_{t+1}) + â_{t+1}
// ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
// Theorem 4.4: 误差界 ∝ Σ (2c)^{T-k-1} × ∥A∥² ∥d_k∥²  (指数衰减, 当c<1/2)

// 误差追踪 (Eq.18):
e_t = (a_t - â_{t+1}) - Q(a_t - â_{t+1})    // 当前步量化误差
    = (a_t - â_{t+1}) - (â_t - â_{t+1})     // 代入Eq.13: â_t = Q(a_t - â_{t+1}) + â_{t+1}
    = a_t - â_t                              // 简化

// 误差补偿在下一步生效：
// 下一步输入 = a_{t-1} - â_t = a_{t-1} - (a_t - e_t) = (a_{t-1} - a_t) + e_t
// 算子输出 = A((a_{t-1} - a_t) + e_t) ≈ A(a_{t-1} - a_t) + A(e_t)
// 其中 A(e_t) 补偿了上一步遗漏的误差分量
```

关键数学保证：Theorem 4.4证明当量化误差系数 $c < 1/2$ 时（Corollary A.3表明通过选择足够高的位宽 $\hat{b} \ge \log_2(\sqrt{4d/c} + 1)$ 可达），标准调制的误差以 $2^{T-k-1}$ 速率指数增长（误差被逐步放大），而误差补偿调制的误差以 $(2c)^{T-k-1}$ 速率指数衰减——c<1/2保证每一步误差均被缩小而非放大。

术语一般如何实现？如何使用？
论文在PyTorch中实现：为每个被MoDiff改造的线性层维护两个额外中间变量 $\hat{\mathbf{a}}_t$ 和 $\hat{\mathbf{o}}_t$ 在时间步间传递。Abalation验证（Table 4）：W8A4时标准调制（w/o EC）FID=25.42，误差补偿调制FID=4.38，差距6×。Figure 3显示误差补偿使relative ℓ₂ distance在3-bit时保持接近0（vs w/o EC持续增长到40%）。额外内存开销：单张CIFAR-10上W8A4时仅~4MB（Table 6）。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---

## Post-Training Quantization (PTQ) for Diffusion Models

术语是什么？
Post-Training Quantization (PTQ) 是一种训练无关的模型量化技术，在预训练完成后直接对网络参数和/或激活值估计量化参数（scaling factor s 和 zero-point z），无需任何fine-tuning或retraining。对于扩散模型，PTQ面临的特有挑战是：(1) 激活在不同时间步（t=T→1）之间的分布范围大幅变化，静态scaling factor难以覆盖全范围；(2) 激活在每个时间步内存在显著异常值（outliers with long-tailed distributions），min-max scaling受极端值主导导致大多数正常值被过粗粒度量化；(3) 低bit（<6-bit）时clipping error和rounding error无法同时被控制。

标准PTQ量化公式：$\mathbf{x}_{\text{int}} = \text{clamp}(\lfloor \mathbf{x}/s \rceil + z, 0, 2^b - 1)$，$Q(\mathbf{x}) = s(\mathbf{x}_{\text{int}} - z)$。Q-Diffusion通过time-step-aware calibration data sampling和MSE reconstruction loss优化scaling factor；BRECQ/LCQ通过block reconstruction和per-channel量化提高精度。但现有方法在扩散模型中仅能将activation量化到8-bit，更低精度（<6-bit）时质量急剧塌陷。

从算法pipeline角度拆解术语：
```
// 扩散模型PTQ的逐层流程（以Q-Diffusion为例）
// 离线校准阶段：
calibration_data = sample_from_diffusion_steps(images, T_steps)
for layer l in 1..L:
    for time_step t in calibration_steps:
        a_t^{(l)} = forward_layer(l, x_t, t)  // 收集每层的激活值
    // MSE重建学习scaling factor
    s^{(l)} = argmin_s ||A^{(l)}(a^{(l)}) - A^{(l)}(Q_s(a^{(l)}))||²

// 在线量化推理：
for t = T..1:
    for layer l in 1..L:
        â_t^{(l)} = Q_s(a_t^{(l)})     // 使用校准的s量化激活
        ô_t^{(l)} = A^{(l)}(â_t^{(l)}) // 低精度整数矩阵乘法
```

MoDiff论文在PTQ评估中使用的方法：Q-Diffusion（time-step-aware MSE reconstruction）、LCQ（dynamic per-channel min-max quantization from BRECQ框架）、LTQ（dynamic per-tensor min-max quantization）。Weight quantization: per-channel MSE reconstruction to 4/8 bit。Activation quantization: per-channel or per-tensor dynamic scaling to 2-8 bit。效率评估使用DeepSpeed计算GBops而非实际硬件加速。

术语一般如何实现？如何使用？
Q-Diffusion开源：https://github.com/true-grub/Q-Diffusion。评估指标：IS/FID/sFID（标准生成模型指标）。主要应用于扩散模型的线性层（Conv2d、Linear, 计算瓶颈）。MoDiff证明可与任意PTQ方法正交叠加，将activation量化从8-bit推进到3-bit而无质量损失。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---

## Temporal Difference in Diffusion Sampling

术语是什么？
Temporal Difference（时序差分）在扩散模型上下文中指相邻扩散时间步（t 和 t+1）之间激活的差值 $\mathbf{a}_t^{(l)} - \mathbf{a}_{t+1}^{(l)}$。MoDiff论文首次系统研究了扩散采样过程中时序差分的统计性质——其分布范围比原始激活小10×以上（Figure 1b橙色vs蓝色violin plot高度对比），分布更集中在零附近，且在不同时间步之间范围更一致，几乎无异常值。这一统计性质是MoDiff调制量化有效性的理论基石。

时序差分之所以具有这些有利属性，是因为扩散采样过程的连续性质：相邻时间步的去噪输出高度相似（这也是DeepCache等方法能够复用缓存的基础），因此它们的差分自然地落在更小的数值范围内。

从算法pipeline角度拆解术语：
```
// 时序差分的统计性质（基于CIFAR-10 DDIM, 100 steps）

// 原始激活分布：
a_t^{(l)} ~ LongTail(μ_t, σ_t²)     // 不同t的μ_t、σ_t变化大
range(a) ≈ [−R, R]                   // R ≈ O(10)
outliers: P(|a| > 3σ) 显著           // 长尾分布导致scaling factor被极端值主导

// 时序差分分布：
d_t^{(l)} = a_t^{(l)} - a_{t+1}^{(l)}
d_t^{(l)} ~ Concentrated(≈0, σ_d²)   // 集中在0附近
range(d) ≈ [−r, r], r < R/10         // 范围小10×以上
outliers: P(|d| > 3σ_d) 几乎为0      // 无长尾

// Theorem 4.3量化误差分析：
Err_Q(a) = (2R)² × d / (2^b - 1)²
Err_Q(d) = (2r)² × d / (2^b - 1)²
Err_Q(d) / Err_Q(a) = (r/R)² < 1/100  // 100×+ reduction
// 等效：用相同误差界可将位宽降低 log₂(R/r) ≈ 3-4 bits
```

术语一般如何实现？如何使用？
时序差分的统计性质不需要额外计算——它是扩散采样过程的固有属性。MoDiff利用它：将量化对象从原始激活切换为时序差分（无额外数据依赖或预处理）。使用时仅需在每层线性算子的forward中做一次矩阵减法（a_t - â_{t+1}）。论文在多个数据集/模型上验证了时序差分性质的普适性（CIFAR-10 DDIM、LSUN LDM、Stable Diffusion、DiT），证明不限于特定架构或采样器。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---

## Caching Methods for Diffusion Models (DeepCache)

术语是什么？
Caching methods for diffusion models（扩散模型缓存方法）是一类利用扩散采样过程中相邻时间步特征高度相似性来跳过高计算代价特征重算的加速技术。代表性方法是DeepCache (Ma et al., CVPR 2024)，它每隔N个去噪步骤在U-Net的高层缓存一次high-level features，在中间的N-1步直接复用cached features，跳过这些层在该步的重计算。其他方法包括Cache Me If You Can (Wimbauer et al., CVPR 2024)、Δ-DiT (Chen et al., 2024) 和 Learning-to-Cache (Ma et al., NeurIPS 2024)。

核心缺陷（如MoDiff论文§3.1/Figure 1a所示）：缓存复用引入approximation error——cached features是过去时间步的近似值而非当前步的精确值——该误差在迭代中累积：每步的偏差传到下一步后被放大。Relative ℓ₂ distance在最终step可达40%（即使每隔3步更新cache），导致生成质量下降。此外，最优的cache更新频率N需要通过heuristic search或retraining确定，泛化性差。

从算法pipeline角度拆解术语：
```
// DeepCache pipeline（以U-Net为例）：
schedule = [1, 0, 0, 1, 0, 0, ...]  // 1=完整计算, 0=复用缓存

for t = T..1:
    if schedule[t] == 1:
        h_low = encoder_layers(x_t, t)      // 低层：新鲜计算
        h_mid = middle_block(h_low, t)       // 中层：计算并缓存
        h_high = decoder_layers(h_mid, t)    // 高层
        cache = h_mid                        // 保存缓存
    else:  // schedule[t] == 0
        h_low = encoder_layers(x_t, t)       // 低层：新鲜计算
        h_mid = cache                        // 直接复用缓存（近似！）
        h_high = decoder_layers(h_mid, t)    // 使用近似中间特征

// 误差分析：
// 复用步：h_mid(实际值) ≠ cache(上次计算)
// 偏差通过decoder传播 → 迭代累积 → 最终step L2 distance达40%
```

术语一般如何实现？如何使用？
DeepCache开源：https://github.com/horseee/DeepCache。集成方式：对预训练扩散模型的U-Net包装DeepCacheModule，指定cache更新间隔N。适用于所有使用U-Net架构的扩散模型（DDPM、DDIM、LDM等）。MoDiff证明了caching方法是MoDiff在0-bit差分时的特例（Remark 4.1）：当时序差分幅度低于可容忍阈值时，MoDiff可分配0-bit跳过计算——此时等价于cache-and-reuse。MoDiff的优势在于通过误差补偿消除了cache的误差累积问题。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

## Diffusion Transformer (DiT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diffusion Transformer (DiT) 是 Peebles & Xie (2022, ICCV 2023 Best Paper) 提出的扩散模型架构，将传统U-Net骨干替换为Vision Transformer (ViT)。核心流程：(1) VAE编码器将输入图像/视频压缩到latent space；(2) 将latent切分为不重叠的patches并投影到token embedding；(3) 通过标准Transformer blocks（self-attention + MLP，每层注入时间步t和条件c的adaLN-Zero调制参数）处理token序列；(4) 输出head预测噪声或速度。DiT-XL/2（675M参数）在ImageNet 256×256上FID=2.27，比ADM-U U-Net高效约6×（118.6 vs 742 Gflops）。在视频生成中（Wan2.1、Sora、CogVideoX），视频作为3D volume (T×H×W)处理，patches变为spacetime cubes，序列长度达10K-100K tokens，self-attention的O(N²)成为主要瓶颈。DiT的成功基于将视觉生成建模为序列建模——与LLMs同构，使FlashAttention、tensor parallelism等Transformer基础设施可无缝迁移。

从算法pipeline角度拆解术语：
```
DiT Video Generation per Denoising Step t:
Input: Noisy latent z_t ∈ R^{T×H×W×C}, timestep t, text condition c
→ Patch Embedding: z_t → tokens X ∈ R^{N×d}
→ For each DiT Block:
    (t,c) → MLP → (γ₁,β₁,γ₂,β₂,α₁,α₂)  // adaLN-Zero modulation
    X = X + α₁ ⊙ Attention(LN(X)×(1+γ₁)+β₁)   // O(N²) bottleneck
    X = X + α₂ ⊙ MLP(LN(X)×(1+γ₂)+β₂)
→ Output Head: noise/velocity prediction
```

在Wan2.1-1.3B (N≈30K)中，注意力占单步52.75T FLOPs，SLA降至2.73T（95%稀疏度），实现2.2×端到端加速。

术语一般如何实现？如何使用？
主流开源实现：DiT (https://github.com/facebookresearch/DiT)、Wan2.1 (https://github.com/Wan-Video/Wan2.1)、CogVideoX。SLA使用Wan2.1-1.3B（视频，30K tokens）和LightningDiT-1p0B/1（图像，ImageNet 512×512）为实验模型。FlashAttention、tensor parallelism等LLM基础设施可直接用于DiT推理加速。视频DiT通常使用spatiotemporal patches（如16×16×4 tubelets）和causal temporal attention（每帧仅关注前序帧）降低有效N。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Linear Attention (for Diffusion Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Attention是通过解耦softmax将标准注意力从O(N²d)降至O(Nd²)的方法族。引入feature map φ(·)（激活函数），用φ(Q)φ(K)^T替代softmax(QK^T)，利用矩阵结合律重排：先算φ(K)^T V ∈ R^{d×d}，再乘φ(Q)得O ∈ R^{N×d}。关键技术公式：

$$H = \phi(K)^T V,\; Z = \operatorname{rowsum}(\phi(K)^T),\; O = \frac{\phi(Q)H}{\phi(Q)Z}$$

线性注意力的表达能力上限为rank d（映射维度），而full softmax注意力的stable rank可远大于d——这是线性注意力在许多场景失效的根本原因（Fan et al., 2025）。SLA实证：在Wan2.1视频生成中，Linear Only的VA=0.042（vs Full Attention 76.78），完全塌陷。但SLA的关键洞察是：去除top 8%大值后的注意力矩阵stable rank骤降至~20，远小于d，因此线性注意力可准确近似这92%的低秩部分。

从算法pipeline角度拆解术语：
```
Standard Attention:  S = QK^T/√d → P = softmax(S) → O = PV  // O(N²d)
Linear Attention:   H = φ(K)^T V | Z = rowsum(φ(K)^T)          // O(Nd²)
                    O = φ(Q)H / φ(Q)Z                           // no N×N matrices
```

SLA在marginal块（~85%）上使用线性注意力：预计算h_j = φ(K_j)^T V_j（d×d矩阵）后，每个marginal块仅需单次H_i += h_j加法，cost <0.5% full attention。φ函数的消融：softmax > elu+1 > hedgehog（表2）。

术语一般如何实现？如何使用？
主要φ选择：ELU(x)+1（Performer, Choromanski 2020）、ReLU(x)、softmax(x)（SLA推荐）。代表性线性注意力模型：Performer、Linear Transformer (Katharopoulos 2020)、cosFormer (Qin 2022)、Lightning Attention-2 (Qin 2024)、RetNet (Sun 2023)、Mamba2（SSM形式，数学等价线性注意力）。SLA中线性注意力不作为full attention的直接替代，而是通过Proj(O^l)投影和fine-tuning作为稀疏注意力的learnable compensation。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Sparse-Linear Attention (SLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLA是Zhang et al. (2025, Tsinghua & UC Berkeley) 提出的可训练混合注意力机制，融合稀疏和线性注意力来加速DiT模型。核心洞察：注意力权重可分解为P = (P⊙M) + (P⊙(1-M))，前者（~8%）高rank需O(N²)，后者（~92%）极低rank可用线性注意力。SLA通过压缩mask预测将注意力块分三级：Critical (top k_h%, 默认5%) → O(N²) FlashAttention；Marginal (中间~85%) → O(N) 线性注意力（预计算后仅矩阵加法）；Negligible (bottom k_l%, 默认10%) → 跳过。关键设计：可学习投影Proj(O^l)减少分布不匹配，fine-tuning仅需2000步使模型自适应。

从算法pipeline角度拆解术语：
```
SLA Pipeline per attention forward:
1. P_c = Softmax(pool(Q)pool(K)^T/√d)            // compressed N/b_q × N/b_kv
2. M_c = classify(P_c, k_h=5%, k_l=10%)           // +1/0/-1 per block
3. Precompute: h_j=φ(K_j)^T V_j, z_j=rowsum(φ(K_j)^T)
4. Fused loop: for i,j blocks:
     M_c[i,j]==+1 → FlashAttention(O(N²))         // ~5% blocks
     M_c[i,j]==0  → H_i+=h_j; Z_i+=z_j            // ~85% blocks (O(N))
     M_c[i,j]==-1 → skip                          // ~10% blocks
5. O = O_s + Proj(O_l)                            // fused output
```
SLA在95% sparsity下FLOPs=2.73T (19.3× vs full 52.75T)，VA=76.96≈Full 76.78，远优于Sparse Only 85% (VA=64.00, 7.91T) 和Linear Only (VA=0.042, 0.10T)。

术语一般如何实现？如何使用？
代码：https://github.com/thu-ml/SLA。单fused CUDA kernel实现前向+反向。使用流程：加载DiT → 替换注意力层 → fine-tune 2000步 → SLA kernel推理。推荐超参数：k_h=5%, b_q=b_{kv}=64, φ=softmax。RTX 5090上：13.7× kernel加速 vs FlashAttention2，2.2×端到端加速（Wan2.1-1.3B视频生成）。额外效率优化：Lookup table（sparsity>90%）、Pre-aggregation（减法替代加法）、Method of Four Russians。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Three-Level Attention Classification (Critical/Marginal/Negligible)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLA的三级注意力权重分类策略，将attention block分为critical（top k_h%, 完整O(N²)计算）、marginal（中间，O(N)线性注意力）和negligible（bottom k_l%, 跳过）。相较于传统稀疏注意力的二级分类（保留/跳过），引入marginal层打破了稀疏度天花板：传统方法跳过中间值会引入显著误差（L1 error从3%跃至33%），保留中间值又严重降低稀疏度（<90%）。SLA对marginal块用几乎免费的线性注意力（cost <0.5% full attention），实现95%有效稀疏度且不损失精度。

从算法pipeline角度拆解：
基于压缩注意力矩阵P_c[i,j]（Q/K mean-pooled后计算dot product），每Q block行独立执行：TopK选前k_h% → critical (M_c=+1)；BottomK选底k_l% → negligible (M_c=-1)；其余 → marginal (M_c=0)。分类参数：k_h=5%源自约8.1%权重大于平均值1/N（保留top 5%已足够），k_l=10%源自约45%权重<1/(100N)（跳过bottom 10%误差<3%）。

术语一般如何实现？如何使用？
在SLA fused GPU kernel内执行。T_m = T_n ≈ 469 for Wan2.1 (N=30K, b=64)，分类开销可忽略。k_h=5%, k_l=10%为推荐默认值；可调整以适应精度-效率tradeoff。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## FP4 Microscaling Quantization (NVFP4/MXFP4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FP4 Microscaling Quantization 是一种基于 GPU Tensor Core 原生支持的 4-bit 浮点量化方案。每个元素使用 FP4 格式（1-bit sign + 2-bit exponent + 1-bit mantissa, E2M1），仅可表示 16 个值（含符号），但通过将数据分组为固定大小的 block，每组共享一个 FP8 格式的 scale factor，实现"微缩放"（microscaling）——scale factor 在计算时动态恢复值域。NVFP4 是 NVIDIA Blackwell 架构的原生实现，量化 block 大小为 1×16，scale factor 为 E4M3 FP8 格式。MXFP4 是 OCP 开放标准的对应格式，block 大小为 1×32，scale factor 为 E8M0 格式。两者均通过 Blackwell 的 FP4MMA 指令直接执行硬件加速的矩阵乘法，无需软件反量化。SageAttention3 对比后选择 NVFP4，因为其在 attention 量化中精度显著优于 MXFP4（更小的 block 粒度 + E4M3 scale 提供更多有效量化级别）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FP4 microscaling attention 的 pipeline（SageAttention3 Algorithm 1 简化）：
```
输入: Q, K, V ∈ FP16, shape N×d
分块: Q → {Q_i} (B_q×d), K → {K_j} (B_kv×d), V → {V_j} (B_kv×d)

// Smoothing Q (SageAttention2 technique)
q̄_i = mean(Q_i)
s_Q, Q̂_i = φ(Q_i - q̄_i)  // φ: NVFP4 microscaling quant, 1×16 block

for j in range(T_n):
    s_K, K̂_j = φ(K_j^T)    // NVFP4 quant, K needs transpose
    s_V, V̂_j = φ(V_j)      // NVFP4 quant
    
    // QK^T in FP4
    S_ij = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)
    
    // Online softmax
    m_ij = max(m_{i,j-1}, rowmax(S_ij))
    P̃_ij = exp(S_ij - m_ij)
    l_ij = e^{m_{i,j-1}-m_ij} * l_{i,j-1} + rowsum(P̃_ij)
    
    // Two-level quantization for P
    s_P1 = rowmax(P̃_ij) / (448×6)
    P̃_ij = P̃_ij / s_P1
    s_P2, P̂_ij = φ(P̃_ij)
    
    // PV in FP4
    O_ij = diag(e^{m_{i,j-1}-m_ij}) * O_{i,j-1} 
           + FP4MM(P̂_ij, s_P2, V̂_j, s_V) × s_P1

O_i = diag(l_{i,T_n})^{-1} * O_{i,T_n}
```
量化函数 φ(X)：将 X ∈ R^{N×d} 分为 1×16 块 X_ij，s_ij = max(|X_ij|)/6，X̂_ij = ⌈X_ij/s_ij⌋（FP4 rounding），scale s_ij ∈ E4M3 FP8。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：使用 NVIDIA CUTLASS 3.x + CUDA，调用 Blackwell FP4MMA PTX 指令（mmasm4或等价指令）。数据布局要求：(1) FP4 数据 packed 为 INT32（每 8 个 FP4 元素占 1 个 INT32）；(2) scale factor 按 FP8 (E4M3) 格式排列，每个 1×16 块对应 1 个 scale；(3) accumulator 为 FP32 布局，但与 operand A 寄存器布局不匹配时需 permutation 优化（SageAttention3 通过重排 accumulator 布局 + fuse K 列重排到量化 kernel 解决）。使用场景：所有需要 attention 计算的 Transformer 模型推理，Blackwell 架构 GPU（RTX5090/B200/B300），plug-and-play 替换现有 attention 实现。开源参考：https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

## Two-Level Quantization for Attention Map

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Two-Level Quantization 是 SageAttention3 提出的针对 attention map P（softmax 输出，值域 [0, 1]）的专用量化策略。由于 P 的值域极窄（[0, 1]），直接做 FP4 microscaling quantization 时，scale factor 的范围仅为 [0, 0.167]，导致 scale factor 在 E4M3 FP8 格式下仅能使用 280 个（35×8）有效量化输出值。Two-Level Quantization 通过两级量化将有效输出扩大至 1016 个（127×8）：(1) Level 1 — per-token 将 P 归一化到 [0, 448×6]，在 FP32 中无损；(2) Level 2 — 对归一化后的 P 做标准 FP4 microscaling quantization，此时 scale factor 充分利用 E4M3 的 127 个有效值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Level 1: Per-token normalization (FP32, lossless)
s_P1 = rowmax(P̃) / (448 × 6)    // shape: [B_q], FP32
P̃_2 = P̃ / s_P1                   // P̃_2 ∈ [0, 448×6], FP32

// Level 2: Standard FP4 microscaling quantization
s_P2, P̂ = φ(P̃_2)                 // s_P2 ∈ E4M3, range [0, 448]
                                  // P̂ ∈ E2M1 (NVFP4)

// Dequantization for PV MatMul
P̃ ≈ P̂ × s_P2 × s_P1              // Three-factor dequant
O = FP4MM(P̂, s_P2, V̂, s_V) × s_P1 // s_P1 applied as post-scaling
```
选择 448×6 是因为 448 是 E4M3 的最大 representable 正值，6 是 NVFP4 (E2M1) 的最大正数值（max(|FP4|) = 6）。rowmax(P̃)/(448×6) 保证 P̃/s_P1 的最大值恰好为 448×6，从而 s_P2 的最大值为 448。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在 FlashAttention 的 inner loop 中，online softmax 计算 P̃ 后立即执行两级量化。Level 1 的 rowmax 可利用 online softmax 已计算的 rowmax 值（softmax 的 S 矩阵 rowmax），通过 max reduction over 16 个连续元素匹配 NVFP4 的 1×16 block 粒度。Level 2 复用 SageAttention3 已有的 FP4 microscaling 量化 kernel，融合到 softmax epilogue 中。仅增加一次 per-token element-wise division 和一次 scalar-vector multiplication，几乎无额外开销。使用场景：所有需要对 softmax 输出做极低比特量化的 attention 实现，特别是 FP4 等极低比特格式。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

## INT8 Trainable Attention (SageBwd)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SageBwd 是首个支持训练的 8-bit 量化 attention 方案。不同于已有工作（FlashAttention3 FP8、SageAttention INT8）仅支持推理，SageBwd 同时实现 attention 的前向和反向 INT8 量化计算。前向对 QK^T（per-block INT8）和 PV（P 做 per-token INT8 + V 做 per-block INT8）两个 MatMul 量化。反向涉及 5 个 MatMul（S=QK^T, dV=P^T dO, dP=dO V^T, dQ=dS K, dK=dS^T Q），其中 dP=dO V^T 被识别为精度最关键的操作——其误差通过 FlashAttention 循环沿序列长度累积到 dQ/dK——因此保持 dOV^T 在 FP16，其余 4 个 MatMul 做 INT8 per-block 量化。选择 INT8 而非 FP8 作为训练量化精度，因 INT8 反向梯度精度更高（CosSim 0.9987 vs FP8 0.9880）且硬件支持更广泛（A100/AMD/Ascend 均支持 INT8 Tensor Core）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

前向（Algorithm 2 简化）：
```
输入: Q, K, V ∈ FP16
s_Q, Q̂_i = ψ(Q_i)   // per-block INT8: s_X = max(|X|)/127
s_K, K̂_j = ψ(K_j^T)
s_V, V̂_j = ψ(V_j)
K_m = mean(K); K ← K - K_m  // Smoothing K

// QK^T matmul in INT8
S_ij = MM(Q̂_i, K̂_j) × s_Q × s_K + rowsum(Q_i)K_m^T  (注:论文Algorithm 2实际未用FP4MM而是量化MatMul)

// Online softmax + per-token quant for P
m_ij = max(m_{i,j-1}, rowmax(S_ij))
P̃_ij = exp(S_ij - m_ij)
s_P = exp(rowmax(S_ij) - m_ij) / 127   // per-token scale, reuse softmax max
P̂_ij = P̃_ij / s_P                        // per-token INT8 quantization

// PV matmul
O_ij = diag(e^{m_{i,j-1}-m_ij}) * O_{i,j-1} + MM(P̂_ij, V̂_j) × s_P × s_V
```

反向（Algorithm 3 简化，关键操作）：
```
// 仅 dP = dO V^T 保持 FP16
dP_ij = MM(dO, V_j^T)   // FP16, 不量化

// 其余 4 个 MatMul 做 INT8 per-block
dS_ij = P_ij ∘ (dP_ij - D_i)             // element-wise
s_dS, dŜ_ij = ψ(dS_ij)                    // INT8 per-block
dQ_i += MM(dŜ_ij, K̂_j) × s_dS × s_K       // INT8
dK_j += MM(dŜ_ij^T, Q̂_i) × s_dS × s_Q     // INT8
s_dO, dÔ_i = ψ(dO_i)                      // INT8 per-block
s_P, P̂_ij = ψ(P_ij)                        // INT8 per-block (重算P)
dV_j += MM(P̂_ij^T, dÔ_i) × s_P × s_dO     // INT8
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SageBwd 使用 OpenAI Triton 实现 forward+backward kernel。相比 FP16 FlashAttention，前向实现约 2× 加速，反向 1.2~1.6× 加速，端到端 forward+backward 最高 1.67× 加速（RTX4090）。适用场景：fine-tuning 任务（Qwen2.5、Llama3.2 fine-tune on GSM8K/MMLU/DROP/HELLASWAG 达到 BF16 同等精度），但不适用于 pretraining（收敛速度较慢）。INT8 选择比 FP8 更优：梯度 L1 error 更低（dQ: 0.029 vs 0.070）、硬件支持更广。开源参考：https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

## Smoothing Q / Smoothing K (SageAttention Outlier Smoothing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Smoothing Q 和 Smoothing K 是 SageAttention 系列（SageAttention → SageAttention2 → SageAttention3）中用于提升低比特量化 attention 精度的离群值平滑技术。问题来源于 Q 和 K 矩阵中存在统计离群值（outliers），这些大值在 per-block 或 per-tensor 量化时主导 scale factor 的计算（scale = max(|X|)/N），导致大量小值被量化到极少数 level 或完全归零。Smoothing K（SageAttention 提出）对 K 做 per-head 均值减法：K ← K - mean(K)，消除 K 的 DC 分量，因为 mean(K) 主导了 QK^T 结果但本身不含位置信息。Smoothing Q（SageAttention2 提出）对 Q 做 per-block 均值减法：Q_i ← Q_i - mean(Q_i)，补偿为 Q_i - q̄_i 参与 FP4 QK^T + 额外的 GEMV(q̄_i, K_j^T) 修正项。SageAttention3 继承了两者。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Smoothing K (applied once per head, pre-loop)
K_m = mean(K)           // per-head mean, shape [1×d]
K ← K - K_m             // K centered

// Smoothing Q (per-block, inside loop)
for each Q_i block:
    q̄_i = mean(Q_i)     // per-block mean, shape [B_q]
    Q_i' = Q_i - q̄_i    // centered Q block
    
    // FP4 quantize centered Q
    s_Q, Q̂_i = φ(Q_i')
    
    // QK^T with correction
    S_ij = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)
    // GEMV term recovers the mean contribution exactly in FP16
```

Ablation 结果（CogVideoX-2B, CosSim）：No smoothing 0.916 → SmoothQuant 0.930 → Hadamard 0.941 → Smoothing_Q 0.983 → Smoothing_K 0.991。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：Smoothing K 作为预处理 kernel，对每个 head 的 K 做 row-wise mean subtraction，与 K 的 FP4 量化 kernel 融合（加载 K → 计算 mean → 减去 mean → 用量化值填充 packed FP4）。Smoothing Q 在 attention inner loop 中，对每个 Q block 在线计算 mean 并减去，用 GEMV 补回（FP16 精度，计算量很小 O(B_q × d) vs O(B_q × B_kv × d)）。两项平滑技术在所有 SageAttention 系列的 INT8/INT4/FP4 attention 中均使用。SmoothQuant 和 Hadamard 变换虽然也是离群值抑制方法，但在 attention 量化中效果不如 Smoothing Q/K（因为前者针对 weight-activation 量化设计，后者针对 attention 特有分布）。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

## Arbitrary Bit-Width Low-Precision Data Types (任意位宽低精度数据类型，1-8 bit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
任意位宽低精度数据类型是指bit width在1到8之间（非powers-of-two的任意值）的数值表示格式，用于LLM推理中的模型权重量化。传统量化方法（如INT8、INT4）使用powers-of-two位宽以对齐GPU的byte边界处理，但4-bit可能过于激进（精度损失大）、8-bit相对浪费（带宽节省少）。5-bit、6-bit、7-bit等中间位宽可在精度与效率间取得更优trade-off，但因GPU架构和软件栈以byte为最小处理单元而缺乏高效支持。

Tilus支持三类family共21种低精度类型：(1) 有符号整数int2-int8；(2) 无符号整数uint1-uint8；(3) 浮点数float3-float8（含任意exponent/mantissa分布如e4m3, e3m3, e3m2, e2m2, e2m1, e1m1）。所有类型可在同一参数化程序模板中支持，无需为每种位宽编写单独kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
低精度推理pipeline（以FP16×INT6矩阵乘法为例）：
```
# 预处理：权重layout变换
B_transformed = rearrange(B[i6][K,N] → u8[K/BK, N/BN, ceil(BK*BN*6/8)])

# Kernel内：低精度计算流程
for k in 0..K step BK:
    a_tile = LoadGlobal(A_f16, layout=m16n8k16_compat, offset)     # FP16 activation
    b_tile = LoadGlobal(B_transformed, dtype=u8, layout=local(3).spatial(32), offset)  # u8紧凑加载
    b_tile = View(b_tile, dtype=i6, layout=tensor_core_compat)     # 零开销reinterpret
    b_tile = Cast(b_tile, dtype=f16)                                 # PRMT/LOP3向量化casting
    C_accum = Dot(a_tile, b_tile, C_accum)                          # Tensor Core mma
C_out = Cast(C_accum, f16)
StoreGlobal(C_out)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于两点关键洞察：(1) 紧凑存储（bit packing）将低精度元素连续打包进u8字节，可能跨字节边界；(2) layout变换使紧凑存储的字节序列可通过标准u8加载指令高效读取，再在registers内通过View reinterpret恢复到低精度表示。Tilus的Cast操作使用CUDA的PRMT（permute bytes）、LOP3（三输入逻辑操作）和bitwise指令在registers内完成向量化类型转换，无需shared memory往返。通过参数化的程序模板（约200配置per operator, auto-tuning tile大小），所有1-8 bit类型共享同一kernel模板。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Compact Storage of Low-Precision Data (低精度数据紧凑存储)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
低精度数据紧凑存储（Compact Storage）是将bit width小于8的元素连续打包进字节序列的存储方式，消除字节内的bit gaps。例如4个int6元素（共24 bits）紧凑存储在3个uint8字节中，而非各浪费2 bits独立存储。由于单个低精度值可能跨越两个连续字节边界（如Figure 8中的b[1]），访问需要bitwise操作：提取用AND+SHIFT+OR组合，写入用MASK+OR。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
紧凑存储下的元素访问伪代码：
```
# 读取第i个bit_width位宽的紧凑存储元素
def read_compact(u8* data, int i, int bit_width):
    bit_offset = i * bit_width
    byte_offset = bit_offset // 8
    bit_pos = bit_offset % 8
    val = data[byte_offset] >> bit_pos
    if bit_pos + bit_width > 8:  # 跨字节边界
        val |= data[byte_offset+1] << (8 - bit_pos)
    return val & ((1 << bit_width) - 1)
```

在Tilus中，紧凑存储是低精度weight loading的基础：权重tensor被变换为连续u8字节序列后，通过LoadGlobal以标准类型加载到registers，再通过View指令零开销reinterpret回原始低精度类型（含正确的layout），无需在加载时逐元素做bitwise提取。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tilus在kernel启动前运行预处理kernel（transform_b, Figure 9），将权重从原始类型（如i6[K,N]）变换为紧凑u8格式（u8[K/BK, N/BN, BK*BN*6/8]）。变换后的tensor中每BK×BN tile的所有bits连续排列，实现coalesced memory access。compact storage方法generalize到任意bit width：给定per-thread bytes数n和T线程，使用u8 dtype和layout local(n/gcd(n,16)).spatial(T).local(gcd(n,16))进行加载。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Weight Layout Transformation for Low-Precision Loading (低精度权重layout变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Layout Transformation是Tilus用来高效加载低精度权重的预处理技术。核心思想：将原始低精度权重tensor的global memory layout变换为标准类型（uint8）兼容的紧凑格式，从而利用硬件友好的coalesced memory access和pipelined async copy，避免低精度bitwise extraction在加载时的开销。例如，将i6[K,N]权重变换为u8[K/BK, N/BN, ceil(BK*BN*6/8)]，每个tile内的bits连续排列为u8字节序列。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
变换过程（Figure 9）：
```
# 输入: i6[BK, BN] weight tile, 32 threads
# 输出: u8[ceil(BK*BN*6/8)] compact bytes
transform_kernel:
    b_in = ViewGlobal(w_ptr, dtype=i6, shape=[K, N])
    b_out = ViewGlobal(t_ptr, dtype=u8, shape=[K/BK, N/BN, ceil(BK*BN*6/8)])
    for each tile [bk, bj]:
        b_reg = LoadGlobal(b_in, dtype=i6, layout=原layout, offset=[bk*BK:, bj*BN:])
        b_reg = View(b_reg, dtype=u8, layout=local(n_bytes_per_thread).spatial(32))
        StoreGlobal(b_reg, b_out, offset=[bk, bj, 0:])
```
变换后的权重使kernel内LoadGlobal可以使用标准u8类型、连续内存访问和pipelined async copy（CopyAsync），然后通过View零开销reinterpret恢复低精度类型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
变换的关键参数计算：给定每线程n bytes和T线程，使用u8 dtype和layout local(n2).spatial(T).local(n1)，其中n1=gcd(n,16), n2=n/gcd(n,16)。该变换在kernel启动前作为预处理执行一次，变换后的权重在各次推理中复用。Tilus的artifact中，此变换作为模型加载的一部分自动完成。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Top-p Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-p Sparse Attention（top-p 稀疏注意力）是将 LLM text generation 中的 nucleus sampling (top-p sampling) 引入 attention sparsity 的技术。核心思想：用累积概率阈值 p 替代固定 token 数量 k 来决定 sparse attention 中保留多少 KV cache token。具体而言，给定估计的 attention weights W ∈ R^N，选择最小的 token 子集 I 使得 Σ_{i∈I} W[i] ≥ p（而非选择固定数量 B 个 token）。这使得稀疏 attention 的 budget（被选 token 数）可以自适应不同 attention head、不同 layer、不同 query 下 attention weight 分布的动态性——对 focused attention（权重集中）自动选少量 token，对 diffuse attention（权重平坦）自动选更多 token。理论误差界：||o - ô|| ≤ (1-p) · ||V||_F（来自 Frobenius norm 的 sub-multiplicative 性质）。

从算法pipeline角度拆解术语，给出具体例子。
```
// Top-k Sparse Attention (baseline):
I = argmax_I Σ_{i∈I} W[i]  s.t. |I| = B  // 固定budget B
// 问题: B无法适应不同分布——focused分布下B过大(浪费), diffuse分布下B过小(精度不足)

// Top-p Sparse Attention:
I = argmin_I |I|  s.t. Σ_{i∈I} W[i] ≥ p  // 固定累积概率p
// 优势: budget自适应——分布决定B, 而非预设B
```

Twilight中的实现：Token Selector用保守大budget B0≈N/4预选token → INT4 SpGEMV估计attention weights → Top-p binary search精筛到B1（累积概率≥p的最小token集）。p值选择比k更鲁棒——p代表累积概率，对不同分布head/layer/query的敏感度远低于k。p=0.85-0.95 typically。

术语一般如何实现？如何使用？
基于FlashInfer，使用4-bit quantized K cache做SpGEMV估计attention weights，GPU上tensorized binary search找满足ΣW[i]≥p的最小token子集。p值通过PG-19等小数据集calibration确定。适用场景：任何使用top-k sparse attention的LLM推理系统，可作为drop-in optimizer叠加到现有算法上。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

## Hierarchical Select-then-Prune Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Select-then-Prune Architecture是Twilight提出的统一sparse attention优化框架。采用两阶段设计：(1) Token Selector——将现有top-k sparse attention算法作为黑盒，使用保守的大budget（如1/4 sparsity）预选token子集，保证高recall；(2) Twilight Pruner——在预选子集上用INT4 K cache估计精确attention weights，然后通过top-p thresholding进一步剪枝到最小token子集。最终sparse attention kernel仅对top-p token执行精确计算。

从算法pipeline角度拆解术语，给出具体例子。
以Quest作为base algorithm为例：
```
// Stage 1: Token Selector (Quest with conservative budget B0=N/4)
page_scores = q @ max_pool(K, page_size=16)^T
top_pages = TopK(page_scores, k=B0/16)
I0 = expand_pages_to_tokens(top_pages)    // |I0| = B0

// Stage 2: Twilight Pruner
W_approx = q @ K_int4[I0]^T               // INT4 SpGEMV
W_norm = softmax(W_approx)                // normalize
I1 = top_p_binary_search(W_norm, p=0.85)  // |I1| = B1 << B0

// Stage 3: Sparse Attention
O = FlashAttention(q, K[I1], V[I1])       // only B1 tokens
```
关键优势：任何top-k sparse attention算法都可被"升级"——只需将其结果作为Token Selector的输出。

术语一般如何实现？如何使用？
基于FlashInfer实现。Token Selector复用原算法kernel，Pruner使用自研INT4 SpGEMV + top-p binary search kernel，Sparse Attention使用FlashInfer的varlen attention kernel。额外开销：INT4 K cache（1/8 FP16 KV cache）。适用于LLM serving。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

## KV Cache Budget Dynamism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Budget Dynamism 是指 sparse attention 中 KV cache budget（被选中参与计算的 token 数量 B）在运行时动态变化的现象。Twilight 识别了四个维度的 dynamism：(1) Prompt-wise：不同 task 的 attention distribution 不同；(2) Query-wise：同一 prompt 内不同 query 的 attention 分布不同；(3) Layer-wise：浅层和深层的最优 budget 不同；(4) Head-wise：retrieval heads（关注全局信息，diffuse → 需大 B）vs streaming heads（关注局部信息，focused → 需小 B），budget 需求完全不同。

从算法pipeline角度拆解术语：
根本原因：attention weight 分布在运行时呈现两种极端——focused attention（权重集中在少数 token）和 diffuse attention（权重接近均匀分布）。Top-k 用固定 B 无法同时覆盖两者：对 focused 造成 over-selection（浪费带宽），对 diffuse 造成 under-selection（丢失 context）。Top-p 用累积概率阈值 p 自适应 B 的大小——focused 分布自动选少量 token，diffuse 分布自动选更多 token——天然支持四维 dynamism。

术语一般如何实现？如何使用？
Twilight 通过 head-wise varlen attention（不同 head 不同 B）+ GQA group union（同一 query group 取各 head 选 token 的并集）+ flatten head dim + load balancing 处理不平衡。Longchat-7B 平均 budget 126-146 tokens（32k context, 99.6% pruning），LLaMA-3.1 平均 427-478 tokens（128k context, 99.6% pruning）。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

---

## Multi-Step Visual Decision-Making

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Step Visual Decision-Making（多步视觉决策）是VLM agent在视觉交互环境中通过多轮观察-动作-反馈循环完成复杂任务的问题范式。与静态VQA（单次图像→文本回答）不同，多步视觉决策要求模型：(1) 在每一步t根据完整历史 H_t = (I, {(o_τ, a_τ, f_τ)}_{τ<t}) 选择动作a_t（I为任务指令，o_τ为视觉观察，a_τ为历史动作，f_τ为环境文本反馈）；(2) 维护跨时间步的内部状态和记忆；(3) 从环境反馈（visual transitions + textual feedback）中推断动作后果和任务进展；(4) 在有限步数内（如20-30步）完成目标。

VisGym将多步视觉决策形式化为RL-style gym paradigm，每个任务的episode由一系列(observation, action, feedback)元组构成，环境在agent发出stop动作时终止并返回二进制成功奖励。核心挑战：VLMs需要将视觉感知（理解图像中的物体位置/状态变化）、长时记忆（跟踪历史交互）和动作规划（选择最优动作序列）三个能力紧密耦合。

从算法pipeline角度拆解术语，给出具体例子。
多步视觉决策的交互循环（以VisGym Jigsaw任务为例）：
```
# Episode execution loop:
Step 0: env.reset() → observation o_0 (scrambled 2x2 jigsaw image)
         prompt = instruction(I) + available_actions + o_0
Step 1: VLM(prompt) → a_1 = ('swap', ((0,0), (0,1)))
         env.step(a_1) → o_1 (new image), f_1 ("Action executed successfully")
         H_1 = {I, (o_0, a_1, f_1)}
Step 2: VLM(H_1 + o_1) → a_2 = ('swap', ((0,1), (1,0)))
         env.step(a_2) → o_2, f_2 ("Action executed successfully")
...
Step k: VLM(H_{k-1} + o_{k-1}) → a_k = ('stop', 'stop')
         env.step(a_k) → reward = 1 if correct else 0
```

关键特性：(a) 每步输入包含完整交互历史；(b) 动作表示为function calls格式；(c) 环境同时提供视觉和文本反馈。

术语一般如何实现？如何使用？
VisGym基于Gymnasium框架实现。评估协议：每环境×setting 70 episodes，easy最大20步/hard最大30步。模型通过OpenRouter API（proprietary）或本地GPU推理（open-weight）。训练协议：使用solver-generated demonstrations进行SFT，仅用easy难度数据训练，hard衡量泛化。关键诊断开关：可控制history length（1/2/4/∞）、observation modality（image/ASCII text）、textual feedback（on/off）、goal observation（with/without）进行受控实验。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---

## Function-Conditioned Action Space

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Function-Conditioned Action Space（函数条件化动作空间）是VisGym为VLM agent设计的一种动作表示方法，将agent的动作建模为带参数的函数调用而非传统离散/连续动作向量。例如，('swap', ((0,0), (0,1)))表示交换两个坐标的拼图块，('rotate', (30.5, 20.4, 15.1))表示绕三个轴旋转。设计动机：VLMs天然具备function-calling能力（经过instruction tuning），使用函数调用格式可充分利用这一能力实现跨任务策略组合。

从算法pipeline角度拆解术语，给出具体例子。
VisGym统一step函数处理function-conditioned action的流程：
```
function Step(a):  # a = "('swap', ((0,0), (0,1)))"
    ρ ← 0; (τ, υ) ← (false, false)
    Parse a → (α, π)  # action name, payload
    if invalid format: return (obs, 0, τ, υ, "invalid format")
    if α in A and π in A[α]:
        (φ, τ, υ) ← Apply(α, π)  # execute, return feedback + flags
    else: return (obs, 0, τ, υ, "invalid action")
    if τ: ρ ← ComputeReward()
    return (obs, ρ, τ, υ, φ)
```

17个环境各自定义可用action function集合（Table 2）：
- Colorization: rotate(θ), saturate(δ), stop()
- Maze 3D: move(0), turn(d), stop()
- Matchstick Equation: move([si, ss, di, ds]), undo(), stop()
- MuJoCo Fetch: move([x,y,z]), gripper(g), stop()
- Jigsaw: swap((r1,c1),(r2,c2)), reorder([...]), stop()

术语一般如何实现？如何使用？
Function-conditioned action space通过Gymnasium扩展接口实现。每个任务的`action_space()`返回可用函数列表和参数约束，`step(action_string)`通过解析器提取函数名和参数。使用方式：(1) 初始prompt包含所有函数的自然语言描述（Function Instructions）；(2) VLM每步输出格式化的函数调用字符串；(3) 环境解析执行后返回新的视觉+文本反馈。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---

## Multi-Step Oracle Solver for VLM Agent Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Step Oracle Solver（多步神谕求解器）是VisGym中为每个视觉交互环境设计的启发式算法，能使用环境提供的action functions完整完成任务。Solver扮演双重角色：(1) 验证任务实例可解性；(2) 生成demonstration轨迹用于VLM的监督微调（SFT）。每个solver支持多策略（如Jigsaw的swap/reorder策略）和可选随机性（通过插入可逆padding动作对生成多样化轨迹）。

Solver使用经典算法：BFS（Sliding Block、Matchstick Equation）、DFS（Matchstick Equation）、图搜索（Maze 2D/3D）、状态机oracle（MuJoCo Fetch）、贪心重排（Jigsaw/Video Unshuffle）。

从算法pipeline角度拆解术语，给出具体例子。
Maze 2D solver生成demonstration的流程：
```
1. GraphSearch(maze_grid, start, target):
     shortest_path = BFS from start to target
     action_seq = convert to move(0/1/2/3 for right/up/left/down)
2. Optional padding:
     for i in range(target_steps - len(action_seq)):
         insert reversible pair: move(d), move(opposite(d))
3. Return trajectory: [(o_0,a_1,f_1), (o_1,a_2,f_2), ..., stop]
```

其他solver设计（Appendix A）：
- **Sliding Block**: BFS最短序列 → pad with back-and-forth move pairs
- **Matchstick Equation**: BFS/DFS → SOS策略（最短路径 + 随机可逆detours）
- **Jigsaw**: reorder（单次排列整个puzzle）或swap（贪心逐个纠正）
- **Fetch Pick-and-Place**: 状态机oracle（open→descend→close→per-axis move to goal）
- **Mental Rotation 3D**: 分解为yaw/pitch/roll → 每轴4×90° padding旋转后再纠正

术语一般如何实现？如何使用？
Solver作为环境类内置方法实现。Demonstration生成流程：(1) solver生成最优动作序列；(2) replay生成完整(observation, action, feedback)轨迹；(3) 预处理过滤失败轨迹和test-set重叠的初始状态；(4) 仅easy难度demonstration用于SFT训练（hard衡量泛化）。多策略和随机性使同一任务实例可生成多条不同demonstration。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---

## Information-Revealing Demonstrations for SFT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Information-Revealing Demonstrations（信息揭示型示范轨迹）是VisGym提出的SFT数据筛选策略，针对部分可观察（POMDP）或未知动态（unknown dynamics）环境的VLM训练。核心洞察：标准demonstration仅展示到达目标的动作序列，但在未知状态转换规则和隐藏信息时，VLM无法从中学习状态表示。信息揭示型demonstration在到达目标前先执行结构化探索步骤暴露隐藏状态或环境动态，使VLM学习更准确的状态表示。

从算法pipeline角度拆解术语，给出具体例子。
两个关键案例（Sec 5.4）：

**Matchstick Rotation（未知动态/unknown scale）**：
- 标准：直接三次stochastic moves到目标 → 成功率32.9%
- 信息揭示型：先两次unit-scale探索步骤（'move', [1,0,0]和'move', [0,1,0]）暴露scale对应关系 → 最后对齐 → 成功率70.0%（+37.1%）

**Mental Rotation 3D Objaverse（部分可观察）**：
- 标准（Solve-Only）：沿每轴旋转一次直接对齐
- 信息揭示型（Rotate-Then-Solve）：先完整旋转每轴暴露3D几何，再对齐
- 验证：在Rotate-Then-Solve模型上继续训练Solve-Only → 性能恶化，确认改善来自demonstration的信息结构而非长度/数量

```
# 信息揭示型demonstration的通用结构：
# Phase 1: Exploration
for each unknown/hidden dimension:
    execute exploratory action (unit move, full rotation)
    observe effect → learn dynamics / expose geometry
# Phase 2: Exploitation
compute and execute goal-directed actions
stop()
```

术语一般如何实现？如何使用？
通过修改solver策略实现（配置为先exploration再solution）。仅用于partial observability或unknown dynamics任务（fully observable + known dynamics任务的标准demonstration已足够）。信息揭示型demonstration帮助VLM"学会学习"环境动态，而非仅模仿动作序列。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

## Any-to-Any Multimodal Models

术语是什么？

Any-to-Any Multimodal Models 是能够接受和生成多种模态（text、image、video、audio）的统一模型架构。与传统的 text-to-text LLM 或 multimodal-input text-output 模型不同，any-to-any 模型支持交叉模态的端到端训练和理解-生成的统一。代表性模型包括：Qwen-Omni 系列（text+image+video+audio 输入，text+audio 输出）、GLM-Image（text+image 输入，text+image 输出）、BAGEL（Mixture-of-Transformers 设计，分离 multimodal understanding 和 visual generation experts）、LongCat-Flash-Omni（560B MoE LLM backbone + LSTM/CNN audio decoder）等。

从算法pipeline角度拆解术语：

Any-to-Any 模型的典型 pipeline 组成（以 Qwen3-Omni 为例）：
```
Input: Text + Audio + Image + Video
  │
  ├─ Text → tokenizer → text token embeddings
  ├─ Audio → Whisper audio encoder → audio embeddings
  ├─ Image → ViT/SigLIP vision encoder → image embeddings
  └─ Video → Vision encoder + temporal aggregation → video embeddings
  │
  ▼
Multimodal Embedding Concatenation
  (所有模态 embeddings concat 后输入 LLM backbone)
  │
  ▼
LLM Backbone (AR Decoder):
  ├─ Thinker: 自回归生成 text tokens + 输出 hidden states
  │   每 step: self-attention over (text + multimodal) tokens
  │   → 产生 text output tokens + per-step hidden states
  │
  └─ Modality-specific Generator:
      ├─ Audio: Talker LLM → codec tokens → Vocoder → waveform
      └─ Visual: DiT → iterative denoising → image/video pixel output
  │
  ▼
Modality-specific Decoder Output
```

关键算法特征：
1. **共享 embedding space**：所有模态通过专用 encoders 映射到统一 embedding space，LLM backbone 在此空间做 cross-modal reasoning
2. **AR Semantic + Modality-Specific Synthesis**：LLM backbone 负责高层语义理解/生成 → modality-specific decoders 负责低层信号合成
3. **Multi-AR pipeline**：Thinker-Talker 等双 AR 设计意味着 pipeline 中有多个需要 KV cache management 的 autoregressive stages
4. **AR+DiT hybrid**：BAGEL 等模型将 AR understanding 和 DiT generation 耦合成单个 inference pipeline

术语一般如何实现？如何使用？

现有 any-to-any 模型多通过 HuggingFace Transformers 实现，开发者手动编排 multi-stage pipeline——每个 stage 的 generate loop 独立实现、cross-stage transfer 手动进行、无 framework-level batching/scheduling 优化。vLLM-Omni 是首个原生支持 any-to-any model serving 的框架——通过 stage graph abstraction 将 multi-stage pipeline 分解为独立 stages，每个 stage 由 vLLM engine 或 diffusion engine 服务，stage 间通过 Unified Connector 传输数据。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

## Thinker-Talker Architecture

术语是什么？

Thinker-Talker Architecture 是 Qwen-Omni 系列（Qwen2.5-Omni、Qwen3-Omni）和 Ming-Omni 系列采用的**双 AR LLM decoder**架构，专为同时生成 text 和 audio 输出设计。它包含两个 sequential AR LLM stages 加一个 Vocoder：

- **Thinker (LLM)**：较大的 AR LLM（Qwen3-Omni: 30B，Qwen2.5-Omni: 7B），接收 multimodal input 并生成 text tokens + per-step hidden states
- **Talker (LLM)**：较小的 AR LLM，每 decoding step concat Thinker hidden states + Talker input embeddings + original multimodal embeddings，自回归生成 audio codec tokens
- **Vocoder**：将 discrete codec tokens 转换为 continuous audio waveform（Qwen2.5-Omni 使用 DiT Vocoder，Qwen3-Omni 使用 lightweight CNN Vocoder）

从算法pipeline角度拆解术语：

Thinker-Talker 的伪代码计算流程：
```
Algorithm: Thinker-Talker Inference

Input: multimodal_input
Output: text_response, audio_waveform

// Phase 1: Multimodal Encoding
mm_input_emb = Concat(
    TokenEmbed(text_prompt),
    AudioEncoder(audio),
    VisionEncoder(image),
    VisionEncoder(video_frames))

// Phase 2: Thinker AR Decode
hidden_states_list = []
text_tokens = []
for step in 1..max_text:
    logits, hidden = Thinker(mm_input_emb)
    token = Argmax(logits)
    text_tokens.append(token)
    hidden_states_list.append(hidden)
    if token == EOS: break
    mm_input_emb = Embed(token)

// Phase 3: Talker AR Decode (cross-stage dependency)
talker_emb = Embed(BOS_audio)
codec_tokens = []
for step in 1..max_audio:
    // KEY: concatenate Thinker hidden states every step
    talker_input = Concat(hidden_states_list, mm_embeddings, talker_emb)
    codec, _ = Talker(talker_input)
    codec_tokens.append(codec)
    if codec == EOS: break
    talker_emb = Embed(codec)

// Phase 4: Vocoder
waveform = Vocoder(codec_tokens)
Return (text_tokens, waveform)
```

关键计算特征：
- Talker 每 step 需 access 完整 Thinker hidden states（跨 stage 数据依赖，非仅单 token）
- Thinker output text tokens (~150.9 avg) ≪ Talker audio codec tokens (~545.4 avg) → Talker 占总延迟大部分
- Thinker model 更大但 text token 输出少，Talker 较小但 audio token 多 → 两者 compute profile 截然不同
- 两 AR stage 都需要独立的 KV cache management（各自的生成序列不同）

术语一般如何实现？如何使用？

Baseline 实现中，开发者需在 HuggingFace Transformers 上手写 custom generate loop for each stage，手动管理 Thinker→Talker hidden state transfer（无 framework-level batching）。vLLM-Omni 将 Thinker 和 Talker 定义为独立 stages，各由 vLLM engine 服务（含 continuous batching + PagedAttention + chunked prefill），Thinker hidden states 通过 preprocess 函数每 iteration 注入 Talker，cross-stage transfer 由 Unified Connector 处理。这种解耦使 Thinker 获 12.97× TPS speedup、Talker 获 7.98× TPS speedup（vs Transformers baseline）。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

## Transfusion (Multimodal Training Framework)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transfusion（Zhou et al., 2024）是一种统一的多模态生成模型训练框架，通过单个Transformer模型同时学习文本自回归生成和图像扩散去噪。其核心思想是：将文本token和图像latent patch交替排列为一个长序列，使用标准Transformer（与Llama等主流LLM架构一致）通过end-to-end训练同时优化language modeling loss（cross-entropy on discrete tokens）和DDPM loss（MSE on predicted noise）。与分别训练语言模型和扩散模型再拼接的多阶段方法不同，Transfusion在同一个Transformer backbone内联合训练两种模态，使文本和图像在attention层中进行双向信息交互（文本可条件化图像生成，图像可辅助文本理解）。Transfusion引入U-Net downsampler/upsampler在Transformer前后压缩/还原图像latent的维度，降低attention计算的开销。该框架是从头训练的——不使用预训练LLM的权重，需要同时使用大量language-only data和image-caption data训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Transfusion的训练pipeline（以caption→image数据为例）：
```
# 输入: text tokens x_txt (discrete), image x_img (raw pixels)
# Step 1: Image encoding
z_img = VAE_encoder(x_img)              # [H, W, 3] → [h, w, c] continuous latent

# Step 2: Image diffusion forward process
t ~ Uniform(1, T)                        # 随机采样扩散时间步
noise ~ N(0, I)                          # 高斯噪声
z_t = sqrt(ᾱ_t) * z_img + sqrt(1-ᾱ_t) * noise  # 加噪latent

# Step 3: U-Net downsampling
h_img = UNet_Down(z_t, t)               # [h*w, c] → [N_patches, d]

# Step 4: Input embedding
h_txt = Embedding(x_txt)                 # [M, d] text token embeddings

# Step 5: Unified Transformer (single shared QKV/FFN/O)
h_all = concat(h_txt, h_img)            # [M+N_patches, d]
for layer in 1..L:
    Q, K, V = QKV_proj(h_all)           # 共享QKV，同时处理text和image
    # 混合attention mask: text用因果mask（autoregressive），image用双向mask（diffusion）
    A = softmax(Q@K^T/sqrt(d) + M)      # M为混合mask
    h_all = FFN(A @ V)                  # 共享FFN

# Step 6: Separate output heads
h_txt_out, h_img_out = split(h_all)
p_logits = LM_Head(h_txt_out)            # [M, vocab_size] 文本logits
ε_pred = UNet_Up(h_img_out, t, z_t)     # [N_patches, c] 预测噪声

# Step 7: Combined loss
L = CrossEntropy(p_logits, x_txt_labels) + λ * MSE(ε_pred, noise)
```
关键特性：所有Transformer参数（QKV, O, FFN）跨模态共享；混合attention mask使文本token只能attend之前的文本（因果），图像patch可双向attend；不使用预训练LLM权重，全部从头训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Transfusion由Meta FAIR提出（arXiv 2408.11039, 2024年8月）。实现基于PyTorch，使用标准Transformer + U-Net模块。训练数据含language-only text data（0.25T tokens）和image-caption pairs（0.25T image tokens）。Transfusion 7B与Llama-3 8B有相同Transformer尺寸（差异仅来自vocabulary大小影响embedding层）。LMFusion将其作为核心baseline对比：Transfusion从头训练虽架构统一，但(1) 需大量language-only data维持语言能力，(2) 语言benchmarks仍低于专用text-only LLM，(3) 总FLOPs是LMFusion（冻结文本模块）的2倍。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## Modality-Specific Transformer Modules (模态特异性Transformer模块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality-Specific Transformer Modules是LMFusion提出的多模态生成模型架构设计模式，核心思想：为每种模态（文本、图像）创建独立的Transformer计算模块（QKV投影、O投影、FFN和LayerNorm），各模态数据仅路由到其专用模块处理，而自注意力层的Q/K/V在拼接后跨模态共享。LMFusion使用两套并行Transformer参数：(1) 文本模块 Proj_text/QKV_text/O_text/FFN_text/LM_Head_text 从Llama-3 8B初始化；(2) 图像模块 UNet_Down_img/QKV_img/O_img/FFN_img/UNet_Up_img 也从Llama-3 8B初始化（除U-Net从头训练）。每个token仅激活其所在模态的模块（一半参数），因此虽总参数量是dense模型的2倍，每次前向FLOPs与dense模型相同。与Mixture of Experts中modality-aware expert routing有概念联系——每个模态有专属"expert"参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMFusion单层前向（文本token数M，图像patch数N）：
```
# Step 1: 模态特异性输入投影
h_txt = Proj_text(x_txt)                # [M, d], text专用embedding
h_img = UNet_Down_img(x_img_t, t)       # [N, d], image专用下采样

# Step 2: 模态特异性QKV投影
Q_txt, K_txt, V_txt = QKV_text(h_txt)   # text专用QKV
Q_img, K_img, V_img = QKV_img(h_img)    # image专用QKV

# Step 3: 跨模态自注意力
# Text queries attend到所有keys: [K_img, K_txt]
A_txt = softmax(Q_txt @ [K_img, K_txt]^T / sqrt(d) + M)
h_O_txt = O_text(A_txt @ [V_img, V_txt])  # text专用O投影

# Image queries attend到所有keys: [K_txt, K_img]
A_img = softmax(Q_img @ [K_txt, K_img]^T / sqrt(d) + M)
h_O_img = O_img(A_img @ [V_txt, V_img])  # image专用O投影

# Step 4: 模态特异性FFN
h_FFN_txt = FFN_text(h_O_txt)
h_FFN_img = FFN_img(h_O_img)

# Step 5: 模态特异性输出
p_logits = LM_Head_text(h_FFN_txt)
ε_pred = UNet_Up_img(h_FFN_img, t, h_img)
```
文本和图像在attention层有双向cross-modal交互，但由于QKV分离，两者attention计算独立——text的attention不改变image的QKV参数，反之亦然。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
文本模块从Llama-3 8B加载后冻结（η_text=0），图像模块从Llama-3 8B初始化后可训练（η_img=1e-4）。实现要点：(1) 计算复用——预训练LLM语言知识通过冻结文本模块完整保留；(2) 知识迁移——图像模块从Llama-3权重初始化获得文本预训练的权重先验；(3) 梯度隔离——图像扩散的梯度不反向传播到文本模块，避免灾难性遗忘；(4) FLOPs效率——每token仅激活对应模态的模块（一半参数）。LLaVAFusion验证了相同范式可应用于已有VLM（LLaVA-NeXT）。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## Modality Separation（模态分离：Deep vs Shallow）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality Separation是多模态模型中将不同模态（文本、图像）的计算路径分割到独立参数集合中的架构设计策略。LMFusion系统化探索了三种分离程度：(1) No Separation（无分离/Dense）——所有模态共享单一QKV/O/FFN参数，仅在U-Net有模态差异；(2) Shallow Separation（浅层/仅FFN分离）——FFN为模态特异性（FFN_text ≠ FFN_img），QKV和O共享，类似Mixture of Modality Experts (MoMa)；(3) Deep Separation（深层/FFN+Attention分离，LMFusion最终设计）——QKV、O和FFN均为模态特异性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
消融实验关键结论（训练250K步，0.03T text + 0.03T image data）：
- No Separation + r=1：语言能力严重退化（HellaSwag -15% initially, -7% persistent gap），即使降低文本lr（r=0.1）也只能缩小gap到-2%而image性能受损——存在trade-off
- Shallow Separation + r=0（仅FFN分离、文本冻结）：明显优于No Separation，但image generation性能受限——FFN仅处理attention后的特征变换，attention pattern的模态特异性更重要
- Deep Separation + r=0（FFN+Attention分离、文本冻结）：所有benchmarks最佳，且image性能甚至超越r=1的dense模型

核心洞察：attention层的模态分离比FFN层更重要——不同模态的attention pattern有本质差异（文本需causal/语义关联，图像需bidirectional/空间关联），共享attention参数时两种模式相互干扰（gradient conflict）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
设计可推广到将预训练text-only LLM适配到任何多模态任务的场景。关键实现要点：图像模块参数从预训练LLM权重初始化（获language knowledge transfer）；文本模块冻结；学习率解耦实现对不同模态的差异化训练速度控制。LLaVAFusion验证了相同范式可直接应用于已有VLM。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## Learning Rate Decoupling in Multimodal Training（多模态训练中的学习率解耦）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learning Rate Decoupling是LMFusion提出的多模态模型训练策略。核心思想：为不同模态的参数组分配独立学习率（lr），各模态参数以不同速度更新或完全冻结。LMFusion将参数划分为文本参数组θ_text（配以η_text）和图像参数组θ_img（配以η_img），学习率比r = η_text/η_img控制文本模块相对更新速度。主实验使用r=0（文本完全冻结），消融探索r ∈ {0, 0.1, 1}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 优化器配置
optimizer = AdamW([
    {'params': θ_text, 'lr': η_text},    # η_text = 0 (冻结) or η_img/10 or η_img
    {'params': θ_img,  'lr': η_img}      # η_img = 1e-4, cosine decay
])

# 三种配置的实验结论：
# r=1 (等速/standard continual pretraining):
#   No Separation: HellaSwag -15%, persistent -7% gap
# r=0.1 (文本慢更新):
#   No Separation: gap缩小到-2%, 但image性能也下降 — 存在trade-off
# r=0 (文本冻结):
#   Deep Separation: 语言能力完全保持, image性能最佳 — Pareto最优
```

关键发现：r=0在dense模型中严重损害image learning（共享参数被冻结后image数据无法学习attention pattern），但在Deep Separation中是帕累托最优——image模块有独立可训练参数，不受文本冻结影响。此技术可推广到任何需保护预训练知识的迁移学习场景（domain adaptation、continual learning、multi-task learning）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch实现：通过`torch.optim.AdamW`的不同param_groups设置差异化lr，或通过`requires_grad=False`+分组实现。关键设计选择是"哪些参数应被冻结/慢更新"——LMFusion证明仅当模态计算路径被充分解耦（Deep Separation）后，冻结才能同时实现知识保留和新能力学习。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## DDPM in Multimodal LLMs（多模态LLM中的扩散图像生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Denoising Diffusion Probabilistic Models (DDPM, Ho et al., 2020) 是一种通过迭代去噪生成图像的生成模型框架。在LMFusion/Transfusion等多模态LLM中，DDPM被集成到Transformer backbone内部——不同于传统扩散模型使用独立U-Net backbone（如Stable Diffusion），多模态LLM在Transformer hidden state空间中运作。流程：(1) VAE encoder将图像压缩到连续latent space，(2) 前向扩散逐步加噪（x_0→x_T），(3) 反向去噪时基于文本条件预测每步噪声，(4) VAE decoder解码去噪latent为像素图像。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练阶段
z_0 = VAE_encoder(x_img)                   # 编码到latent
t ~ Uniform(1, T); ε ~ N(0, I)             # 时间步+噪声
z_t = sqrt(ᾱ_t) * z_0 + sqrt(1-ᾱ_t) * ε    # 前向加噪
h_txt = text_forward(x_txt)                 # 文本表征
ε_pred = image_forward(z_t, t, h_txt)      # 文本条件化噪声预测
L_DDPM = MSE(ε_pred, ε)

# 推理阶段（T步去噪）
z_T ~ N(0, I)
for t = T...1:
    ε_pred = image_forward(z_t, t, text_context)
    z_{t-1} = denoise_step(z_t, ε_pred, t)  # DDPM/DDIM sampler
generated_image = VAE_decoder(z_0)
```
关键设计：噪声预测以文本为条件——text tokens通过cross-modal attention注入去噪过程；图像token使用双向attention mask（去噪需全局上下文），文本token使用因果mask；cosine noise schedule（Nichol and Dhariwal, 2021）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion使用256×256图像压缩为32×32×8 latent（VAE f=8下采样），经U-Net downsampler降至256 patches。Loss权重λ平衡LM loss和DDPM loss。此设计使单一模型同时具备文本生成和图像生成能力。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## Classifier-Free Guidance (CFG) for Multimodal Diffusion（多模态扩散中的无分类器引导）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Classifier-Free Guidance（CFG, Ho and Salimans, 2021）是提升扩散模型图像生成质量的技术。核心思想：推理时结合条件预测和无条件预测引导生成方向，ε̂ = ε_uncond + w*(ε_cond - ε_uncond)，w为guidance scale。w>1时模型放大条件信息影响，使生成图像更贴合文本（更高CLIP score）；w=1等价标准条件生成。需每扩散步两次前向pass（conditional + unconditional），推理延迟翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for t = T...1:
    ε_cond = model(z_t, t, text_prompt)     # 条件前向
    ε_uncond = model(z_t, t, null_text)     # 无条件前向
    ε_guided = ε_uncond + w*(ε_cond - ε_uncond)  # CFG组合, w=1.55
    z_{t-1} = denoise_step(z_t, ε_guided, t)
```
LMFusion在MS-COCO上评估：无CFG (w=1.0)的FID和带CFG (w=1.55)的结果，通常w在1.5-3.0间选择以平衡质量和多样性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion采用w=1.55。CFG代价是每步两次前向，但显著提升文本-图像对齐度（CLIP score）。实际使用中训练时可随机drop文本条件（如10%概率）使模型学习无条件生成能力，推理时即可使用CFG。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation

## VAE Tokenizer for Multimodal Models（多模态模型中的VAE图像分词器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VAE (Variational Autoencoder) Tokenizer是扩散模型和多模态LLM中将图像编码为连续latent representation的模块。LMFusion/Transfusion使用预训练VAE encoder（stabilityai/sd-vae-ft-mse）将256×256 RGB图像压缩为32×32×8连续latent tensor（f=8下采样，8通道），在latent space中进行扩散去噪。与离散tokenizer（VQ-VAE, VQGAN）不同，VAE tokenizer产生连续latent配合DDPM做连续空间扩散。VAE encoder冻结（不参与训练），latent进一步经可训练U-Net downsampler压缩为256 patches送入Transformer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 编码: x_img [3,256,256] → z [8,32,32] → h_img [256,d]
z = VAE_encoder(x_img)
h_img = UNet_Down(reshape(z, [1024,8]), t)

# 解码: z_0 [8,32,32] → generated_image [3,256,256]
generated_image = VAE_decoder(z_0)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion使用sd-vae-ft-mse (https://huggingface.co/stabilityai/sd-vae-ft-mse)，在OpenImages上fine-tuned，MSE loss优化重建质量。选择VAE over VQ-VAE的优势：连续latent更适合diffusion连续加噪/去噪；无codebook collapse问题；与主流扩散模型兼容。代价是latent无离散token语义，不能用自回归next-token方式解码，必须用diffusion框架。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation


## Dequantized GEMM / Weight-Only Quantization Matmul (反量化矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dequantized GEMM (反量化矩阵乘法) 是 Weight-Only Quantization 场景下的核心计算操作：权重（weights）以低精度格式存储（如 INT4、NF4、INT2、FP4），激活值（activations）保持较高精度（如 FP16），在 kernel 执行时动态将低精度权重反量化（dequantize）到计算精度，再执行矩阵乘法。TileLang 支持多种 dequantized GEMM 方案：W_INT2 × A_INT8（最高 7.65× over cuBLAS FP16），W_INT4 × A_FP16（1.04× over Marlin），W_NF4 × A_FP16（1.62× over BitsandBytes），FP4_E2M1 × FP16。TileLang 的关键优势在于通过寄存器内反量化（in-register dequantization）消除 Triton 的 shared memory layout conversion 瓶颈。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TileLang Dequantized GEMM (FP4_E2M1 × FP16) 算法 pipeline（图 17）：
```
输入: A[M, K] f16 (activation), B[N, K] u8 packed (4-bit weight, 2 elems per byte)
输出: C[N, M] f16

// === Step 1: 内存分配 ===
A_shared  = T.alloc_shared([block_M, block_K], f16)         // activation tile (shared mem)
B_shared  = T.alloc_shared([block_K, block_N//2], u8)       // packed weight tile (shared mem)
B_local   = T.alloc_fragment([block_N, block_K], u8)        // 解包前 (registers)
B_deq     = T.alloc_fragment([block_N, block_K], f16)       // 解包后 (registers)
Ct_local  = T.alloc_fragment([block_N, block_M], f32)       // accumulator (registers)

// === Step 2: Pipelined 主循环 ===
T.clear(Ct_local)
for k in T.Pipelined(K // block_K, num_stages=2):
  // Step 2a: 异步加载 activation 和 packed weight 到 shared memory
  T.copy(A[by*block_M, k*block_K], A_shared)               // cp.async global→shared
  T.copy(B[bx*block_N, k*block_K // 2], B_shared)          // B 每 byte 含 2 个 FP4 元素

  // Step 2b: Shared → Register (加载 packed weight)
  T.copy(B_shared, B_local)                                 // u8 连续字节 → registers

  // Step 2c: 寄存器内解量化 (FP4 → FP16)
  for i, j in T.Parallel(block_N, block_K):
    // _tir_packed_to_unsigned_convert: 从 u8 字节提取指定半字节 → unsigned int → cast to f16
    B_deq[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
      num_bits=4,                     // 每个 weight element 4 bits
      B_local[i, j // 2],             // 源 u8 字节 (含 2 个 FP4)
      j % 2,                          // 选择高/低 4-bit
      dtype=f16)

  // Step 2d: Tensor Core GEMM (transpose B)
  T.gemm(B_deq, A_shared, Ct_local, transpose_B=True)
  // B_deq^T [block_K, block_N] × A_shared^T [block_K, block_M] → Ct_local [block_N, block_M]

// === Step 3: 写出结果 ===
T.copy(Ct_local, Ct[bx*block_N, by*block_M])
```

关键优化对比：
- Triton 方式: unpack in registers → store to shared memory for layout conversion → ldmatrix reload → MMA（额外 shared memory 往返）
- TileLang 方式: load u8 in registers → View 零开销 reinterpret (u8→i4) → Cast vectorize to f16 → MMA（全程寄存器内完成，消除 shared memory 往返）
- TileLang 还支持 PTX 级 fast precision conversion 指令和 Ladder 的平滑内存访问优化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源代码：TileLang weights-only quantization GEMM kernel 通过 BitBLAS-TileLang 后端实现（BitBLAS 原是 TensorIR 后端，论文中替换为 TileLang 后端做对比）。同一程序模板通过参数化支持 INT2/INT4/NF4/FP4 等多种格式，仅需改变 num_bits 和 dtype 参数。在 A100 上，W_INT2A_INT8 达 7.65× cuBLAS FP16 speedup。对于不支持的量化格式，用户可自定义 _tir_packed_to_unsigned_convert 等 utility 函数。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

## Multi-head Latent Attention (MLA / 多头潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的 attention 机制创新。核心思想是将传统 Multi-Head Attention (MHA) 的 Key-Value (KV) cache 压缩为低维 latent vector，通过低秩分解在两个阶段工作：(1) Latent Space Encoding：将 K/V 投影到低维 latent 空间（如 d_model=5120 压缩到 latent_dim=512，约 10× 压缩），仅存储压缩后的 latent vector 而非每个 head 的完整 K/V；(2) Dynamic Decoding：注意力计算时，从 latent vector 动态上投影恢复各 head 的 K/V 表示。MLA 通过矩阵乘法结合律将解压矩阵与 Q 投影权重融合（"matrix fusion trick"），避免推理时额外计算开销。对于需要 Rotary Position Embedding (RoPE) 的部分维度，MLA 采用混合设计——部分维度带 RoPE（跨 head 共享）、部分不带 RoPE（允许 fusion trick）。MLA 在保持接近 MHA 表达能力的同时，将 KV cache 减少 87-92%。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA forward pass 伪代码（TileLang 约 70 行 Python 实现）：
```
输入: hidden_states [batch, seq_len, d_model]
输出: attention_output [batch, seq_len, d_model]

// Stage 1: Latent Compression (Down-Projection)
c_KV = W_down_KV × hidden_states        // [batch, seq_len, latent_dim]
c_Q  = W_down_Q  × hidden_states        // [batch, seq_len, latent_dim_Q]

// Stage 2: Up-Projection for K and V
K = W_up_K × c_KV                        // [batch, head, seq_len, dimqk]
V = W_up_V × c_KV                        // [batch, head, seq_len, dimv]

// Stage 3: RoPE Handling (hybrid design)
K_rope = RoPE(K[:, :, :d_rope])          // 部分维度带 RoPE，跨 head 共享
K_nope = K[:, :, d_rope:]                // 其余维度不带 RoPE
K_final = concat(K_rope, K_nope)

// Stage 4: Q with Matrix Fusion Trick
Q = W_up_Q × c_Q                         // 融合后的等效 Q

// Stage 5: Standard Attention (Parallel Pattern)
scores = Q × K_final^T                   // relevance scoring
scores = softmax(scores / sqrt(dimqk))   // RowNorm
output = scores × V                      // aggregation
// dimqk ≠ dimv (如 DeepSeek-V3: dimqk=576, dimv=512)
```

MLA 的关键特征：(1) KV cache 仅存 c_KV（latent vector，如 512 维），每个 head 不再存独立 K/V；(2) dimqk 和 dimv 通常不相等；(3) head 数远大于 head_kv（如 head=128, head_kv=1）；(4) query seqlen=1 的解码场景下，Q 只有一个 token。

TileLang 实现的 MLA kernel（图 18，~70 行 Python）使用 T.Pipelined loop over KV tiles，在 H100 上达 FlashMLA（手写 CUDA ~1.7k 行）的 98% 性能，在 MI300X 上达 AITER 手写 kernel 的 95% 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA 有多个优化实现：(1) FlashMLA（DeepSeek 官方）——约 1.7k 行 CUDA，专门针对 H100/H800，使用 TMA + wgmma.mma_async + warp specialization；(2) TileLang 实现——约 70 行 Python，自动利用 TMA + WGMMA + warp specialization on H100，利用 HIP async copy on MI300X。开源：FlashMLA https://github.com/deepseek-ai/FlashMLA；TileLang MLA kernel 在 TileLang 仓库的 examples 中。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---

## Linear Attention (Mamba-2 Chunk-Scan/Chunk-State)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Linear Attention 是一类通过解耦 softmax 将标准注意力从 O(N²d) 降为 O(Nd²) 的方法族。引入 feature map φ(·)，用 φ(Q)φ(K)^T 替代 softmax(QK^T)，利用矩阵结合律重排：先算 φ(K)^T V ∈ R^{d×d}，再乘 φ(Q) 得 O ∈ R^{N×d}。TileLang 论文中的 Linear Attention 特指 Mamba-2 模型的 chunk-scan 和 chunk-state 函数——这些是 State Space Model (SSM) 中的 recurrent computation kernel，数学上等价于线性注意力形式。TileLang 将这两个函数作为算子级 benchmark，在 H100 上对比 Triton 实现：chunk-scan 平均 1.77× speedup，chunk-state 平均 2.10× speedup。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mamba-2 Linear Attention 的 chunk 分解：
```
// Mamba-2 的核心递归:
h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t       // state update
y_t = C_t^T h_t                         // output

// 序列切为 chunks 实现并行化:
for each chunk_i in range(seq_len / chunk_size):
  // Step 1: Intra-chunk (chunk 内部并行)
  for j in chunk_i:
    h_j = A_j ⊙ h_{j-1} + B_j ⊙ x_j
    y_j = C_j^T h_j
  // Step 2: Inter-chunk (chunk 间顺序传递 compressed state)
  h_chunk = A_chunk ⊙ h_prev + B_chunk  // chunk-state 函数

// chunk-scan: 对 chunks 做 parallel scan (类似 prefix sum)
// chunk-state: 计算单 chunk 的初始→最终 state 映射矩阵
```

TileLang 的 benchmark 使用 Table 4 的 12 种 shape 配置（chunk-scan CC0-CC5 和 chunk-state CT0-CT5），覆盖 batch=1/64, nheads=64, seq_len=1024/2048/8192, head_dim=64, d_state=128。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Mamba-2 的 Linear Attention 用 Triton 实现（参考代码在 mamba 仓库）。TileLang 用约 50 行 Python 实现等价功能（使用 T.Pipelined + T.gemm + T.reduce），在 H100 上获得 1.77-2.10× speedup。加速原因：TileLang 自动利用 TMA + wgmma.mma_async + warp specialization，而 Triton 在 H100 上未充分利用这些 Hopper 特性。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

