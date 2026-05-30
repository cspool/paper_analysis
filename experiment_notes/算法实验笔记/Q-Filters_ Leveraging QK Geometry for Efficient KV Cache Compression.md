## Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-Filters 是一种训练无关（training-free）的 KV Cache 压缩方法。核心算法分为两步：(1) **离线校准阶段**：从校准数据集（Pile 子集，约 3000 样本）中收集各层各头的 Query 表示 $Q^h$，对每个头的 $Q^h$ 矩阵进行 SVD 分解 $Q^h = U \Sigma V^\top$，取第一右奇异向量 $v_1$ 作为该头的 Q-Filter（并对符号做规范化 $v_1^+ = \operatorname{sgn}(\mathbf{1}u_1^\top)v_1$）；(2) **推理阶段**：对每个头计算所有 Key 向量在 Q-Filter 上的投影 $\langle K_t^h, v_1^+ \rangle$，保留投影值最大的 KV pairs，丢弃投影值最小的。该方法基于定理 3.3：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_i^h \rangle) \approx \kappa^h \langle K_i^h, u^h \rangle$，即 Key 在 Query 主方向上的投影可近似其期望注意力分数。对于 GQA，对每组 Query 的 Q-Filters 取平均。
  - 实验比较：在 Language Modelling（Pile 数据集上 perplexity）、Needle-in-a-Haystack（检索准确率）、Ruler 数据集（多子任务得分）上与 StreamingLLM、SnapKV、K-Norm（L2 范数方法）、Expected Attention 对比，压缩比从 2× 到 64×。同时测量 Time to First Token (TTFT) 以对比延迟。

- 硬件平台是什么，配置是什么。
  - 2 块 NVIDIA A100-80GB GPU（用于校准和推理实验）。Q-Filters 校准在 Llama-3.2-70B 上耗时不到 3 分钟（2×A100-80GB）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B、Llama-3.1-70B（Dubey et al., 2024）、Qwen-2.5-7B-Instruct（Qwen et al., 2025）、Llama-3.2-1B。
  - 数据集/benchmark：Pile（Gao et al., 2020）用于语言建模 perplexity 评估和 Q-Filters 校准；Needle-in-a-Haystack（合成检索任务，needle 深度 1k-64k tokens）；Ruler（Hsieh et al., 2024，含 CWE、FWE、Multi-Key、Multi-Query、Multi-Value、Single、QA、VT 子任务，序列长度 4096/8192/16384）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码在 https://github.com/NathanGodey/qfilters ；基于 KVPress 库（https://github.com/kvpress）和 HuggingFace Transformers。
  - 算法 pipeline 伪代码：
    ```
    # 离线校准阶段 (只执行一次)
    Q_bank = []  # 存储各层各头的 Query 激活
    for sample in calibration_dataset[:3000]:
        for layer in model.layers:
            for head in layer.heads:
                Q_bank[layer][head].append(Q_activations)
    
    q_filters = {}
    for layer in model.layers:
        for head in layer.heads:
            Q_matrix = stack(Q_bank[layer][head])  # [N*d_k]
            U, S, Vt = SVD(Q_matrix)
            v1 = Vt[0, :]  # 第一右奇异向量，d_k 维
            sign = sign(mean(ones @ U[:,0]))  # 保证正期望投影
            q_filters[layer][head] = sign * v1  # Q-Filter
    
    # GQA 处理：对每组共享 Query 的 head 取平均
    if model uses GQA:
        for group in kv_groups:
            q_filters[group] = mean(q_filters[heads_in_group])
    
    # 推理阶段
    def q_filters_compress(kv_cache, max_size):
        for layer, head in layers_and_heads:
            K = kv_cache[layer][head].keys  # [seq_len, d_k]
            scores = K @ q_filters[layer][head]  # [seq_len]
            keep_indices = topk(scores, max_size)
            kv_cache[layer][head] = kv_cache[layer][head][keep_indices]
    ```
    张量计算：给定 Key 矩阵 $K^h \in \mathbb{R}^{L \times d_H}$ 和 Q-Filter $v_1^+ \in \mathbb{R}^{d_H}$，重要性得分为 $s = K^h \cdot v_1^+ \in \mathbb{R}^L$。保留 $s$ 最大的 $k$ 个 KV pairs。该操作仅涉及一次矩阵-向量乘法和一次 top-k 选择，与 FlashAttention 兼容（无需物化注意力矩阵）。
