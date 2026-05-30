## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Twilight的多项GPU kernel优化，基于FlashInfer构建：(1) **Efficient SpGEMV with INT4 K cache** —— 使用per-head asymmetric INT4 quantization存储额外K cache（在shared memory中解包+dequantize），cp.async异步加载实现2-stage pipeline（加载+计算overlap），将SpGEMV的global memory access降至1/4。Dequantization参考QServe采用per-head动态量化（FP16 scale+zero），利用FasterTransformer的PTX汇编实现INT4→FP16快速类型转换；(2) **Top-p via Binary Search kernel** —— 修改FlashInfer的top-p sampling kernel用于attention weights。采用parallel-friendly binary search（Algorithm 1），element-wise操作(max/where/sum)融合为单次tensorized GPU循环，不物化中间变量W0，避免O(N log N)排序；(3) **Head-wise varlen attention** —— 支持MHA的head级动态budget和GQA的group级动态budget。GQA下每query group取各head选择token的union，使用flattened paged KV cache layout，复用FlashInfer的load balancing算法（flatten head dim）处理head间不平衡；(4) **SpGEMV kernel优化** —— INT4 K元素bit-packed到uint8_t buffer（2×4-bit per byte），地址计算remap到4-bit granularity（halving effective byte offset），interleaved packing简化dequantization。

  实验比较：(a) Self-attention operator —— FlashInfer-Twi vs FlashInfer, Quest-Twi vs Quest, vs FlashAttention2, vs PyTorch SDPA (Memory-Efficient Attention backend)，batch=32-256, seq_len=10k-30k, 测量latency和speedup；(b) End-to-end decoding —— Quest-Twi vs Quest vs FlashInfer, batch=32-256, 测量TPOT (Time-Per-Output-Token)；(c) Ablation —— time breakdown (TokenSel+SpGEMV+Top-p+Attention), quantization bits vs compute time (Figure 12); (d) Offloading scenarios —— Quest vs Quest-Twi, tokens loaded from CPU memory。

- 后端平台是什么，配置是什么。
  单张NVIDIA A100 GPU。Software: PyTorch, CUDA, OpenAI Triton, FlashInfer (https://github.com/flashinfer-ai/flashinfer)。SpGEMV kernel基于FlashInfer的attention decoding kernel修改，top-p kernel修改自FlashInfer的top-p sampling kernel。Per-head动态量化参数（FP16 scale+zero）使用paged memory layout存储。

- 评估性能的软件/脚本是什么。修改了什么。
  使用FlashInfer作为基础kernel库。修改：(1) 新增SpGEMV kernel——将FlashInfer的decode attention kernel修改为sparse GEMV（q_fp16 @ K_int4），加入INT4 dequantization逻辑，使用cp.async + 2-stage software pipeline，FP16 dequantized K cache（非FP32以优化计算）；(2) 新增top-p binary search kernel——修改FlashInfer的top-p sampling kernel（原用于LLM token sampling）用于attention weight累积概率计算；(3) 修改attention kernel——支持head-wise/group-wise varlen attention with flattened paged KV cache；(4) Quest kernels修改——支持batch inference（原始Quest仅支持batch=1）。

  对比baselines：FlashInfer (原始attention kernel), Quest (SOTA sparse attention runtime), FlashAttention2 (xformers Memory-Efficient Attention backend), PyTorch SDPA。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/tsinghua-ideal/Twilight

  评估原理：
  1. 使用Longbench中三种类型任务(Qasper/GovReport/LCC)的10k-30k prompts进行batch inference
  2. 自研kernel通过CUDA/Triton实现，集成至FlashInfer的attention pipeline
  3. 测量self-attention operator latency (CUDA Events) 和 end-to-end TPOT (wall-clock)
  4. 每配置warmup后多次测量取平均

  Kernel输入到性能输出全过程（以Quest-Twi, batch=64, seq_len=32k, LLaMA-3.1-8B-Instruct decode step为例）：
  ```
  Host: 启动Twilight attention pipeline (3个kernel launch)
  
  Kernel 1: Token Selector (Quest SpGEMV with FP16 K cache)
    Input: q ∈ R^{BS×H×d}, K_paged ∈ R^{N×d}
    ① Quest: max_pool K to page granularity (16 tokens/page)
    ② SpGEMV: q @ K_pooled^T → approximate scores
    ③ Top-k: select top B0/16 pages → expand to B0 tokens
    Output: I0 indices (B0 tokens), mask_0
  
  Kernel 2: Twilight Pruner (SpGEMV + Top-p)
    Input: q, K_int4 ∈ R^{N×d/2} (paged, per-head dynamic quantized)
    ① SpGEMV with INT4:
       - Load: cp.async from GMEM[K_int4] → SMEM (2-stage pipeline)
         Thread 0-31: async load next K_int4 tile while computing current tile
       - Dequantize in SMEM: unpack UINT4 → apply (K_int4 - zero) * scale → FP16
         Use FasterTransformer-style PTX asm for INT4→FP16 fast conversion
       - Dot product: q_fp16 @ K_fp16 in registers → W_approx[I0]
    ② Softmax: W_norm = softmax(W_approx[I0])
    ③ Top-p Binary Search (Algorithm 1, fully tensorized):
       - l=0, r=max(W_norm), B1=0
       - Loop (typically 8-12 iterations for ε=0.01):
         a. mask = (W_norm >= (l+r)/2)  // where op
         b. cumsum = sum(mask * W_norm)  // fused where+sum
         c. if cumsum >= p: l=(l+r)/2  // threshold too low → raise
            else: r=(l+r)/2              // threshold too high → lower
       - M = (W_norm >= l), B1 = count(M)
       - 所有element-wise/mask/sum操作在单次register循环中完成
    Output: I1 indices (B1 tokens, B1 << B0), M mask
  
  Kernel 3: Sparse Attention (varlen attention)
    Input: q, K[I1], V[I1] (仅加载B1个token的FP16 KV cache)
    ① GQA group union (for LLaMA-3.1):
       - For each query group (e.g., 4 Q heads → 1 KV head):
         I_group = union(I_head1, I_head2, I_head3, I_head4)
         B_group = |I_group|
    ② Flatten head dimension:
       - Concat all group token sets → [total_pruned_tokens]
       - Load balance: FlashInfer scatter-arrange by token count per group
    ③ Sparse FlashAttention:
       - Load Q tile [BM, d], K tile [B_group, d], V tile [B_group, d]
       - Online softmax: S = Q @ K^T / sqrt(d)
       - P = exp(S - rowmax), l = rowsum(P)
       - O += P @ V, update m, l
       - Finalize: O = diag(l)^{-1} @ O
    Output: O ∈ R^{BS×H×d}
  
  性能测量:
    - Self-attn latency (μs): CUDA Event record start/stop per kernel
    - Speedup = FlashAttention2 latency / Twilight latency
    - FlashInfer-Twi: 6.5× vs FA2, 2.4× vs FlashInfer (seq_len=32k, batch=64)
    - Quest-Twi: 15.8× vs FA2, 1.4× vs Quest (seq_len=32k)
    - End-to-end TPOT: Quest-Twi 3.9× vs FlashInfer, 1.35× vs Quest
    - Time breakdown (batch=64, 32k):
      TokenSel ~15%, Pruner(SpGEMV+Top-p) ~20%, SparseAttn ~65%
    - Offloading: Quest-Twi 7.2-16.1× vs Quest (10k-30k)
  ```

  关键kernel设计要点：
  - INT4 SpGEMV的memory access降至1/4 → memory-bound kernel直接受益于量化
  - Top-p binary search避免sorting：O(log(range/ε)) vs O(N log N)，且tensorized on GPU
  - Head-wise dynamism → load balancing：将不同head的不同budget展开为flat load，消除padding waste
  - GQA group union：trade-off accuracy vs repeated loading，实测group varlen优于head-wise varlen（重复加载）和padded（浪费计算）
  - 2-bit K cache精度不足（累积注意力权重显著下降），8-bit浪费带宽，4-bit是最优trade-off
