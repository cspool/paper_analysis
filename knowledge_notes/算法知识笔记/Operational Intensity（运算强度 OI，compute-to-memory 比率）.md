## Operational Intensity（运算强度 OI，compute-to-memory 比率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- OI 定义为"计算量/访存量"的比率（FLOPs per byte moved），roofline 模型中决定一个算子落在 compute-bound（高 OI，受算力限制）还是 memory-bound（低 OI，受带宽限制）区域。SMOOTH（ISCA'26）用它解释移动 NPU 上 LLM decode 的突发带宽：prompt 期全序列 self-attention 是 GEMM（高 OI、compute-bound）；token 生成期输入从 d×l 矩阵缩为 d×1 向量、attention 矩阵从 l×l 缩为 l×1，反复执行低 OI 的 GEMV（QKV 投影、W0 等线性运算需搬动整块 d×d 权重只做少量计算、极度 I/O-bound），而 softmax/GELU 等非线性运算主要靠向量吞吐、高 OI、带宽严重空闲。两类算子交替执行 → 带宽一会儿饱和一会儿空闲的 bursty 流量。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单层 decode 执行拆解（移动 NPU，batch=1）：QKV 投影 GEMV（OI=2·d·1/(4·d) ≈ 0.5 FLOP/byte 量级，I/O-bound，带宽饱和）→ FlashAttention 内 QK^T/softmax/SV（softmax 高 OI、带宽空闲）→ 输出投影 W0 GEMV（低 OI）→ FFN W1 GELU W2（W1/W2 低 OI、GELU 高 OI）。SMOOTH 的量化观测：非线性（高 OI）运算占端到端时间 10–20%（TinyLLaMA 在 Jetson 20.4%、S24 17.0%、EdgeTPU 14.1%；模拟器保守估计 9.4%/5.7%），这些高 OI 阶段正是可被预取利用的空闲带宽窗口。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：OI 是 roofline/架构分析的核心指标，SMOOTH 用它做动机分析（哪类算子带宽瓶颈、哪类有预取头寸），并用于判断静态编译器为何失效（tile size 的选择受运行期 OI 变化影响，序列长度与带宽波动使静态选择最多恶化延迟 2.9×）。也常见于 PIM/近存架构论文（如 Pimba、BAAP、InstAttention 用 OI 划分 GeMV→memory 单元、GeMM→compute 单元）。SMOOTH 不改变模型 OI 特性，而是用硬件内存管理把低 OI 阶段的带宽需求摊平到高 OI 阶段的空闲窗口。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
