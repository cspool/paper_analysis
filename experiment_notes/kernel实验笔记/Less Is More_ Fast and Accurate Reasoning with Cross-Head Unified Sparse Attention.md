## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  在 FlashInfer（Ye et al., 2024）attention kernel 库之上实现针对 GQA 模型的定制化稀疏 attention kernel。核心优化：利用 LessIsMore 的跨 head 统一 token 选择（CUSA）特性，所有 query head 共享同一 token 索引集 ρ，避免了 TidalDecode/Quest 等 per-head 独立选择方案在 GQA 下的冗余 KV loading——在 TidalDecode 中，同一 KV group 的不同 query head 可能选择不同的 token 集合，导致 KV cache 需要加载更多 token。LessIsMore kernel 仅加载统一的 ρ token 集合（大小 K），单次加载即可服务整个 KV group 的所有 query heads，减少 global-to-shared memory 传输。实验比较：(a) kernel 级延迟：LessIsMore vs TidalDecode vs Quest/SeerAttn-R vs StreamingLLM vs Full Attention（FlashInfer），在 DeepSeek-R1-8B、2K budget、16K context 下的 FLOPs/global-to-shared memory/Mem/Latency（Table 4）；(b) 稀疏 attention kernel latency vs TidalDecode 在不同 token budget 下的 speedup（Figure 6b）；(c) 端到端 TBT speedup（Figure 6a）。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU。DeepSeek-R1-Distill-Llama-8B 模型。所有 kernel 基于 FlashInfer attention kernel 库实现。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashInfer（https://flashinfer.ai/）attention kernel 库。核心修改：
  1. **统一 token 索引的稀疏 attention kernel**：修改 FlashInfer 的 GQA attention kernel，将 per-head token 索引替换为单一统一索引 ρ，所有 query heads 共享同一个 KV 子集。在 GQA 架构下（如 DeepSeek-R1-8B: 32 query heads, 8 KV heads, group=4），避免同一 KV group 内不同 query head 的独立 KV loading。
  2. **KV cache 加载优化**：Sparse Attention Layers 从 KV cache 仅加载 K[ρ] 和 V[ρ]（仅 K 个 token），存储于 shared memory，随后所有 query head 共享此 KV tile。
  3. **GQA query grouping**：将同一 KV group 的多个 query head 在 kernel 内并行化，利用 Tensor Core 批量 GEMM（Q_g [r, d] @ K[ρ]^T [d, K] -> S [r, K]），饱满 Tensor Core 利用率。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/DerrickYLJ/LessIsMore（含 FlashInfer 定制 kernel）

  **Kernel 执行全流程（以 DeepSeek-R1-8B GQA 单 decode step 为例）**：

  ```
  输入：
    Q ∈ R^{32×1×128}（32 query heads, d=128）
    K_cache ∈ R^{L_kv×8×128}（8 KV heads, d=128）
    V_cache ∈ R^{L_kv×8×128}
    ρ ∈ N^{K}（统一 token 索引, K=2000）
  输出：O ∈ R^{32×1×128}

  Step 1: Token Selection Layer（在指定 layer 如 Layer 12 执行一次）
    P = Q @ K_cache^T            # [32, 1, L_kv], FlashInfer full attention
    # CUSA: 跨 head 统一选择
    for each KV group g (0..7):  # 4 query heads per KV group
        idx_group = []
        for h in [4g, ..., 4g+3]:
            idx_h = TopK(P[h,:,:], k=K·0.75)  # each query head proposes
            idx_group.append(idx_h)
        # Union across 4 query heads sharing same KV head
    ρ_unified = unique(flatten(all 32 heads' proposals))
    ρ = sort_by_score(ρ_unified)[:K·0.75] ∪ [L_kv-K·0.25, ..., L_kv-1]

  Step 2: Sparse Attention Kernel（后续层复用 ρ）
    # KV cache 加载（仅加载 ρ 中 K 个 token，而非全量 L_kv 个）
    for each KV head g (0..7):
        K_sparse = load_kv_tile(K_cache, ρ, head=g)    # [K, 128] from HBM → SMEM
        V_sparse = load_kv_tile(V_cache, ρ, head=g)    # [K, 128] from HBM → SMEM
        # 同一 KV group 的 4 个 query heads 共享 K_sparse, V_sparse
        Q_g = Q[4g:4g+4]                                # [4, 1, 128]
        S_g = Q_g @ K_sparse^T / √128                    # [4, 1, K], Tensor Core mma
        P_g = online_softmax(S_g)                        # [4, 1, K]
        O_g = P_g @ V_sparse                             # [4, 1, 128], Tensor Core mma
    O = concat([O_0, ..., O_7], dim=0)                  # [32, 1, 128]
  ```

  **Kernel 效率对比**（Table 4, DeepSeek-R1-8B, 2K budget, 16K context）：

  | Method | FLOPs | G2S Memory | On-device Mem | Latency |
  |--------|-------|------------|---------------|---------|
  | LessIsMore | 1.05M | 1.04MB | 8.38MB | 20.1µs |
  | TidalDecode | 1.05M | 2.34MB | 8.38MB | 32.1µs |
  | Quest/SeerAttn-R | 1.05M | 2.34MB | 8.38MB | 32.1µs |
  | StreamingLLM | 1.05M | 1.04MB | 1.04MB | 20.1µs |
  | Full Attention | 8.40M | 8.38MB | 8.38MB | 76.4µs |

  **关键差异**：LessIsMore 与 TidalDecode 计算量（FLOPs）相同，但 Global-to-Shared memory 传输仅 1.04MB vs 2.34MB（减少 55%），因为统一 token 选择避免了同一 KV group 内不同 query head 的冗余 KV loading。这使得 kernel latency 从 32.1µs 降至 20.1µs（1.6× speedup）。

  **Sparse Attention Kernel Speedup**（Figure 6b, vs TidalDecode）：
  - 各 token budget 下 LessIsMore kernel 比 TidalDecode kernel 快 1.3×-1.72×

  **端到端 Speedup**（Figure 6a, vs Full Attention, A100）：
  - 16K context: 1.09×-1.1×
  - 32K context: 1.22×-1.3×
  - 64K context: 1.48×-1.58×
