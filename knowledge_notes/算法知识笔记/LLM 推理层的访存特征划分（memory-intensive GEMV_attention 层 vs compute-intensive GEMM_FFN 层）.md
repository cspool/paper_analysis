## LLM 推理层的访存特征划分（memory-intensive GEMV/attention 层 vs compute-intensive GEMM/FFN 层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM（GPT-3、LLaMA-2 等）推理按层的算术强度可二分：compute-intensive 层是全连接 FFN 的 GEMM（batch×seqlen 大时权重复用高、受 GPU 算力主导）；memory-intensive 层是 attention 的 QKV 生成与 projection 的 GEMV 形态（QK^T、SV：权重=每请求新产生的 K/V 或激活，复用低、每元素约 2 FLOP，受外部带宽主导）。CPU/GPU 处理器为中心的系统跑 memory-intensive kernel 时被 off-chip 数据搬移瓶颈卡住——这正是 PIM offload 的机会窗口。DCC 的 kernel 选择：GEMV/RED/ATTN/VA/RELU（AttAcc 额外有 softmax/accumulator 单元支持整段 attention 在 PIM 侧）；GEMV/ATTN 输入尺寸 128 是 LLM 最常见的 per-head 维度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPT3-13B/LLaMA2-33B 一层的执行拆解（DCC + AttAcc 的划分）：
```
x = input_embeddings                       # [batch, seqlen, hidden]
Q, K, V = split(x @ W_qkv + b_qkv)         # QKV 生成：GEMV 形态 → PIM（DCC 使能后）
score = softmax(Q @ K^T / sqrt(d))         # attention：GEMV+softmax → PIM（AttAcc 原生支持）
ctx = score @ V                            # context projection：GEMV → PIM
h = x + W_o @ ctx                          # output projection：GEMV → PIM（DCC 使能后）
h = FFN(h)                                 # FFN：GEMM，主体留 GPU
```
AttAcc 默认实现只把 attention 放 PIM（QKV 生成与 projection 留 GPU，固定 tiling 下这两层放 PIM 反而落后 GPU 1.25×）；DCC 联合搜索数据分区与计算调度后，把 QKV 生成与 projection 也移上 PIM（分别 2.58×/2.91× 对 GPU），使 AttAcc_Full+DCC 端到端平均 4.52× 对 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：按"层访存强度 + 后端硬件（PIM 有无专用单元）"决定落点——有 softmax 单元的 AttAcc 可整段 attention 上 PIM，HBM-PIM 只有 16-way FP16 FPU 故跑 GEMV/RED/VA/RELU；DCC 在 LLM 中按张量尺寸动态选 draft（输入/输出 token 数、batch 变化触发在线生成）。对 batch 增大时 QKV/projection 变成 batched GEMM 形态，GPU 相对优势上升——DCC 在大 batch（MT-NLG-310B、batch≤64、8×A100）下仍较 AttAcc Base/Full 提速 1.59×/1.67×。参考划分原则（CompAir 等一致）：GeMM→compute 单元、GeMV→memory 单元。

EVA 补充视角（ISCA'26）：decode 阶段 GEMV 的低效有两层——(1) 计算侧：M=1 时 GEMM 单元窄条活跃、算术强度低（大多数 PE lane 空闲）；(2) 访存侧：权重矩阵每 token 全量重取、无复用。EVA 用向量量化（VQ）+ 码本驱动 GEMM 重构同时解两层：权重压缩为索引+码本（2-bit，memory 侧），并把解码从 GEMV 重写为"输入×码本"的 GEMM（计算量 K×N → K×2^n，约 16×，M 维扩到 V=K/d>512），使 decode 阶段成为可填满矩阵单元的 GEMM 形态。这与"GeMM→compute 单元、GeMV→memory 单元"的传统划分不同——EVA 在算法层就把 GeMV 变成 GeMM，计算形态不再固定受限于 batch=1。

- SMOOTH 视角（ISCA'26，移动 SoC）：移动 NPU（2–8MB SRAM、LPDDR5 13–34GB/s、batch=1）上 decode 期 OI 特征在单层内交替——线性投影（QKV、W0）是低 OI 的 GEMV、带宽饱和（I/O-bound），softmax/GELU 等非线性是高 OI、带宽空闲（compute-bound），导致突发性（bursty）访存与带宽浪费。三类平台上非线性运算占端到端时间 10–20%（Jetson/TinyLLaMA 20.4%、S24 17.0%、EdgeTPU 14.1%）。动机是：高 OI 运算的空闲带宽窗口可被预取利用，但静态编译器看不到运行期进度；静态 tile size 因序列长度与带宽波动失效（延迟最多恶化 2.9×）。
涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
