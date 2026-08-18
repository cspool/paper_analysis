## Per-token Magnitude Pruning（KV cache 每 token 幅度剪枝 / Mustafar）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对 KV cache 按 token 维度做非结构化幅度剪枝：每个 token 的 K/V 向量内仅保留幅度最大的 k 个元素（其余置 0 或存 bitmap 稀疏格式）。依据（Mustafar，NeurIPS 2025，arXiv:2505.22913）：Key cache 存在显著的 channel-wise outlier（特定头维通道持续大值），per-token 剪枝天然保留这些 outlier 通道；Value cache 分布均匀、无通道 outlier，但 attention 中同一 token 的所有 value 元素乘同一个 attention score，按幅度剪枝在功能上等价于 output-aware 剪枝。相比 ThinK 等结构化（整 token/通道）剪枝，非结构化 per-token 可在 70% 稀疏下无精度损失，且压缩后可达 45% 的稠密内存占用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for t in tokens:
    K_t' = K_t ⊙ topk(|K_t|, k)           # 每 token 保留幅度 top-k 元素
    V_t' = V_t ⊙ topk(|V_t|, k)
# 稀疏存储：1×64 列 tile + 64-bit bitmap + tile offset
# decode attention = 压缩 KV 上的 SpMV（共享内存反压缩、compute-as-dense）+ 最近 32 token 局部 dense MV + online softmax
```
本文用法：Cassandra 对 KV cache 用 per-token 幅度剪枝生成 speculation 组、被剪元素进 verification 组；再叠加 4-bit 尾数截断与指数压缩（unary/MX）进一步降低 KV 带宽；KV 侧优化使 Cassandra 在长序列 benchmark（LongBench-QMSum）上增益最大，接受率 0.78–0.91。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mustafar 开源：github.com/dhjoo98/mustafar（自定义 CUDA SpMV kernel：load-as-compressed、compute-as-dense、共享内存 tile 反压缩）。使用：长上下文 decode 加速（Llama-3-8B 上 batch 6→8、最高 2.23× tokens/sec）；与 KIVI 量化、H2O token 逐出正交叠加。局限：需要专用稀疏 attention kernel；不保留最近窗口会伤精度（保留最近 32 token 的 dense 窗口）。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
