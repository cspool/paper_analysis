## Skinny GEMM（瘦矩阵乘）与 PIM buffer 级复制实现

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Skinny GEMM 指一维（token 或 batch）维度远小于另一维的"细长"矩阵乘法，算术强度低、权重复用有限。RAG 推理中它是结构性的：文档 KV 预计算后 prefill 只编码短 query（平均 ~16 token），QKV 投影/FFN 退化为 16×d_model 的 skinny GEMM；个性化/隐私敏感部署 batch 很小，decode 的 FFN 也退化为 skinny GEMM 或 GEMV。roofline 分析（MERIDIAN 图 3）显示这类算子在 H100 上算术强度远低于计算饱和点，attention 与 FFN 全程 memory-bound——这是集中式 RAG 的"计算低效"瓶颈（Bottleneck 2）的根源。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# skinny GEMM（batch=16, d_model=3584）：M=16 ≪ K=N=3584
Y[16, 3584] = X[16, 3584] @ W[3584, 3584]   # 输出行数少 → 每次只复用同一 W 16 次
# 朴素 PIM 顺序执行（单 GEMV 单元）：
for i in 1..16: Y[i,:] = X[i,:] @ W          # 每次重新读 W（近存带宽/能量浪费）
# MERIDIAN buffer 级复制（只复制 buffer 不复制算术单元）：
# 4 个双缓冲 4KB buffer 存多份输入/中间结果，共享 16 乘法器+16 加法器
# 同一 W 从 DRAM 读一次、喂给多个 buffer 里的输入向量 → 权重复用 4 倍
```
对比：PAPI 用"完整复制 GEMV 数据通路"实现真并行，但 HBM 功率预算下可行、LPDDR 下不可行（通道窄、功率低）；MERIDIAN 只复制约占 GEMV 单元面积 14% 的 buffer 结构而共享算术单元，同时 DRAM 访问占能耗 >96%、更大 buffer 提升 row locality、减少 row-switch 开销——在带宽/功率受限的 LPDDR5X 上实现权重复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上 skinny GEMM 靠 batch 聚合/continuous batching 提高复用（或 FlashInfer 等 kernel 库的 ragged 处理）；PIM 上 MERIDIAN 用 buffer 复制 + All-Bank-Mode（命令广播到所有 bank 同地址）让各 bank 的 16-lane PU（16 FP16 乘法/加法器）并行执行多路 GEMV。使用场景：KV-precomputed RAG 的 prefill（Qwen-TB/Tulu3-Block-FT 的 QKV/FFN 投影）、小 batch decode 的 FFN；效果上 MERIDIAN 相对 CENT（GEMM 拆多次 GEMV）3.98×、相对 PAPI（GEMV 单元复制）3.32× 吞吐优势。极端长响应场景（response 远长于文档）MERIDIAN 可退回集中式执行并保留内存侧高效处理。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
