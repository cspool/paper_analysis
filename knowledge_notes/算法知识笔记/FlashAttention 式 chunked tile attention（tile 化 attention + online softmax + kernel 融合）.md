## FlashAttention 式 chunked tile attention（tile 化 attention + online softmax + kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention 把 attention 沿序列维切成 chunk（tile）计算：每个 chunk 内算局部 score→局部 softmax，跨 chunk 用 online softmax 的 running 统计量（running max m、running sum l）增量修正合并，全程不物化完整的 S=QK^T 矩阵（O(n²) 显存降为 O(n)），并通过 kernel 融合减少中间读写。CHIME-PIM 借鉴该思想但动机不同：bank PU 的 result buffer 有限、rank PU 片上 SRAM 有限，chunk 化能约束中间 head 足迹并使 score（bank PU）与 softmax（rank PU）跨单元流水；其 chunk 定义 = 单 head 跨（可能多个）bank PU 并行产生的数据，per-chunk softmax 后做 streaming 跨 chunk 归一化得到全局正确 S，S 元素写回与 context（S×V）计算再流水。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
online softmax 的核心三量维护（chunk 粒度，等价 FlashAttention 算法）：
```
m = -inf; l = 0; O = 0              # running max / sum / 输出累加
for chunk in chunks:
    s_chunk = Q @ K_chunk^T         # 局部 score（CHIME：bank PU 算，写 result buffer）
    m_new = max(m, rowmax(s_chunk))
    l = l * exp(m - m_new) + rowsum(exp(s_chunk - m_new))   # 修正 running sum
    O = O * exp(m - m_new) + exp(s_chunk - m_new) @ V_chunk # 修正输出
    m = m_new
S = O / l                           # 最终归一化
```
CHIME 的流水映射：score（bank PU）→ rank PU 取回（外部总线，与下一 chunk 的 MAC 重叠）→ adder 累加 + per-chunk softmax（rank PU）→ 全部 token 后 streaming 归一化 → S 写回 DRAM 与 S×V 流水。该融合保证数学上精确（与逐 token softmax 等价），不损失精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上 FlashAttention-1/2（tiling + softmax recomputation + warp 协作 + 寄存器缓存，后续 FlashAttention-3/FlashDecoding 扩展）；PIM 上把"chunk 流水"映射到 bank PU/rank PU 的异步执行（CHIME），中间量经 result buffer/adder/softmax 单元流式传递。使用方式：任何长序列 attention（训练与推理）减少峰值显存/缓冲与中间往返；在 CHIME 中是 bubble-free pipelining 与跨 chip 传输隐藏的算法前提（chunk 越小流水越细、head 足迹越小，但传输次数越多——由 T_comm ≤ T_comp 约束 head 映射 N_hc 平衡）。

PLENA 补充视角（ISCA'26）：把 FlashAttention 从"GPU kernel 融合"下沉为"加速器 ISA 原生支持"——论文归纳现有 systolic 加速器无法原生支持 FlashAttention 的四个缺口：(1) 无 tile 级 off-chip 预取与计算重叠；(2) 无 transpose-on-read 与高效跨步/分块流式内存布局；(3) 只暴露 GEMM 原语，缺 online softmax 所需的行内归约与非线性（max/sum/exp/div）；(4) ISA 固定调度、粗粒度 kernel 边界，阻碍 tile-by-tile 融合。PLENA 对应机制：(1) H_LOAD_M/H_LOAD_V 指令控制的 SRAM 硬件预取引擎；(2) 转置可读 Matrix SRAM（见硬件架构库条目）；(3) vector/scalar 单元实现归约与元素操作，VLEN 可配对齐 tile，softmax 计算精度可配（常用高精度如 FP12）；(4) 47 条自定义 ISA 的 tile 级持久调度，把 QK^T→online softmax→PV 逐 tile 编排。收益：大中间激活留片内（Vector SRAM）不回写 off-chip，显著减少内存流量——这是长上下文内存墙下的关键收益；同时 aggressive 预取重叠隐藏 HBM 延迟。

MERIDIAN 补充视角（ISCA'26，跨设备 online-softmax 融合）：MERIDIAN 的文档注意力分解把 online softmax 的 running 统计量从"chunk/GPU SM 粒度"推广到"PIM 设备/分支粒度"——文档分支与上下文分支各自算局部 (o_d,m_d,l_d) 与 (o_c,m_c,l_c)，全局融合用共享基线 m=max(m_d,m_c) 后 l=e^{m_d-m}l_d+e^{m_c-m}l_c、o=(e^{m_d-m}o_d+e^{m_c-m}o_c)/l，与 FlashAttention 的 running max/sum 修正合并同构（等价 LogSumExp 结合律的两步归约，见 Tree Attention 跨 GPU LSE 归约）。关键差异：两个"chunk"分处不同物理设备，只有紧凑统计量跨设备交换（每设备只传 d_model/N 输出切片 + 2 个标量），无需像集中式那样聚拢全部 attention logits；softmax 本体用专用精度硬件（NMU 内 softmax 单元）保证数值稳定，而宽容算子（GeLU/Swish）用 LUT 分段线性。对比 M-non 消融（softmax 留在 GPU 需聚拢完整 logits、通信随文档长线性增长）仅 1.69×，MERIDIAN 全量 5.36×——证明跨设备 online-softmax 融合是去中心化 RAG 的通信关键。

QiMeng-Tensify 补充视角（ISCA'26）：FlashAttention 作为"手工专家优化"baseline（v2.7.3）与 benchmark 子图——(1) 子图级：SelfAtten 等 attention 子图上对比时使用 FlashAttention-V2（"FlashAttention-V2 used in FlashAttention, Triton, and QiMeng-Tensify"，Fig.9 注），QiMeng-Tensify 平均快 1.27×（FP16），但 NSA 等新型稀疏注意力上专家 FlashAttention 仍领先（论文承认 slightly behind expert-coded FlashAttention）；(2) 定位：FlashAttention 系列（FA1/FA2/FA3）是"手工专家优化"路线的代表（labor-intensive、难以随模型创新扩展），其 chunked tile + online softmax + kernel 融合是编译器/自动搜索（如 QiMeng-Tensify 的 MDP 图重写）试图自动复现的对象——对 QKNorm 子图 QiMeng-Tensify 甚至超过 FlashAttention 1.66×，说明自动图级优化可覆盖部分专家设计空间。

- SMOOTH 用法（ISCA'26）：FlashAttention 作为三种常见算子融合之一（QKV 投影融合、FlashAttention、FFN 融合）被建模为编译器的静态优化——融合虽提升数据复用与 locality，但强制 Q/K/V 激活同 kernel 同时存活、拉长中间 buffer lifetime，与 QKV/FFN 融合一起造成严重片上碎片；SMOOTH 的 block 级虚拟化 + early reclamation 正是为缓解这类融合带来的碎片与长 lifetime 而设计（融合版本下才能激进预取，明显降延迟；无融合时各策略收益都受限）。
涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
