## LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LagKV，一种无需注意力权重的 KV Cache 压缩/驱逐方法。核心机制为递归分区压缩——将 KV cache 按 lag size L 分区，使用下一个相邻 chunk 的统计量（max/min）对当前 chunk 归一化，计算 channel-wise 标准差后 softmax 得到 token 重要性分数，再对 K 和 V 的分数求和，使用 top-K 策略选择保留 Token。同时保留 attention sink（前 S 个 token）和滑动窗口（最后一个分区）。
  - 实验比较：(1) RULER benchmark（16K context）对比 SnapKV、StreamingLLM 在 0.25/0.5/0.75/0.875 压缩比下表现；(2) LongBench 和 64-digit Passkey Retrieval 消融实验，测试不同 L（128/512/1024）和 r（2×/4×/6×/8×）组合；(3) chunk-by-chunk prefill 模式下 FGT 准确率和 needle score；(4) 不同 scoring 变体对比（LocalKV、L2 norm vs LagKV）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台和具体 GPU 型号/数量。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct（均使用 GQA 以减少 KV cache 大小）。
  - Benchmark：RULER（16K，含 13 个子任务：Single-Key/Single-Value/Multi-Key/Multi-Query/Variable Tracking/Common Word Extr/Freq. Word Extr/QA1/QA2）、LongBench（含 Single-doc QA、Multi-doc QA、Summarization、Few-shot、Synthetic、Code 子任务）、Needle-in-a-Haystack（64-digit Passkey Retrieval，背景为 Paul Graham Essays）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码仓库 https://github.com/AI-Lab-China-Merchants-Bank/LagKV，集成于 NVIDIA KVPress 框架 (https://github.com/NVIDIA/kvpress)。
  - 算法 pipeline（伪代码）：
    ```
    输入: K_i, V_i ∈ R^{n×d_h} for each head i
    超参: S=16 (sink size), L (lag size), r (retention ratio)

    def LagKV_compress(K_i, V_i, S, L, r):
        # 1. 保留 attention sink（前 S 个 token）
        compressed = K_i[:, :S], V_i[:, :S]

        # 2. 统计剩余长度，若 < 2L 则不压缩
        remaining_len = n - S
        if remaining_len < 2*L:
            return concat(compressed, K_i[:, S:], V_i[:, S:])

        # 3. 按 L 分区（最后一个分区作为滑动窗口）
        n_partitions = floor(remaining_len / L)
        partitions = split(K_i[:, S:], n_partitions, dim=seq)
        V_partitions = split(V_i[:, S:], n_partitions, dim=seq)

        # 4. 递归压缩每个分区（最后一个保留不压缩）
        for p in range(n_partitions - 1):
            K_cur = partitions[p]      # 当前 chunk: (h, L, d_h)
            K_ref = partitions[p+1]    # 参考 chunk（下一分区）

            # 4a. 使用参考 chunk 计算 token-wise min/max
            min_k = min(K_ref, dim=seq)     # (h, d_h) 每个 channel 在参考分区中的最小值
            max_k = max(K_ref, dim=seq)     # (h, d_h) 每个 channel 在参考分区中的最大值

            # 4b. 归一化当前 chunk
            K_norm = (K_cur - min_k) / (max_k - min_k + eps)  # (h, L, d_h)

            # 4c. 计算 channel-wise std + softmax
            K_std = std(K_norm, dim=channel)  # (h, L)
            score_K = softmax(K_std, dim=seq) # (h, L)

            # 对 V 做同样操作
            min_v = min(V_ref, dim=seq)
            max_v = max(V_ref, dim=seq)
            V_norm = (V_cur - min_v) / (max_v - min_v + eps)
            V_std = std(V_norm, dim=channel)
            score_V = softmax(V_std, dim=seq)

            # 4d. 求和得到最终 token score
            score = score_K + score_V  # (h, L)

            # 4e. Top-K 选择（每个 head 独立）
            topk_indices = topk(score, k=r*L, dim=seq)
            kept_K = gather(K_cur, topk_indices, dim=seq)
            kept_V = gather(V_cur, topk_indices, dim=seq)
            compressed_K.append(kept_K)
            compressed_V.append(kept_V)

        # 5. 加上滑动窗口（最后一个分区）和 Mod 余数
        compressed_K.append_all(partitions[-1])
        compressed_V.append_all(V_partitions[-1])

        return concat(compressed_K), concat(compressed_V)
    ```
  - 压缩比计算公式：L_R = S + rL*(⌊(L_s - S)/L⌋ - 1) + L + Mod(L_s - S, L)；C = 1 - L_R/L_s
  - 数学直觉：token-wise locality 使得相邻 token 的 K/V 值相似，用下一 chunk 归一化可消除 channel 偏移，保留 channel-wise variance 作为重要性度量。与 KIVI 量化思路类似但用于驱逐而非量化。完全不依赖 query 态或 attention weight → 与 FlashAttention 兼容且无指令依赖偏差。
