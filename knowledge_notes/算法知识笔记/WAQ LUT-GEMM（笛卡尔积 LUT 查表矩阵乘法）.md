## WAQ LUT-GEMM（笛卡尔积 LUT 查表矩阵乘法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-GEMM 用查找表直接完成量化 GEMM 而非逐元素 MAC。现有 WOQ LUT-GEMM（FIGLUT [42]、LUT Tensor Core [37]、LUT-GEMM [43]）把"权重索引 → 组内 inner-product 结果"存入 LUT，μ 个权重 bit 作索引，运行时查表得部分和再跨组累加；缺陷：(1) LUT 依赖流式激活必须 on-the-fly 生成；(2) inner-product LUT 大小 2^μ·(K/μ)，K 大（LLaMA-7B q_proj K=4096）时爆炸，只能小 group（μ=4）抑制；(3) 跨 group partial-sum 增加 FLOPs、并行度受限。OASIS 的 WAQ LUT-GEMM（§III-B）利用 WAQ 三大机会：权重与激活质心均离线学习 → Cartesian Product LUT 可离线预计算（消除 on-the-fly 生成）；双操作数量化后可能乘积仅 2^(nW+nA) 项（W4A4 为 256）→ 存 Cartesian Product 而非 inner product，LUT 相对 4096×4096 层小 64×；Cartesian Product LUT 与归约长度 K 无关 → 归约粒度可达整层 K（group size 相对 μ=4 提高 1024×），归约 FLOPs 降 16×。与知识库现有"LUT 查找表量化（非解析量化 / Codebook-based Lookup）"条目的区别：后者（EVA 视角）是 weight-only 的查码本解码/输出码本 GEMM，本条目是双测量化下以拼接索引计数的 Cartesian Product 查表计算方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WAQ LUT-GEMM 计算过程（M=1, K=6, N=4, nW=nA=1 示例，论文 Fig.6）：
```
# 离线：LUT[j] = C_W[j>>nA] * C_A[j & (2^nA-1)]，共 2^(nW+nA) 项
# 在线：
# (1) 拼接：concat_idx[k,n] = (idx_A[k] << nA) | idx_W[k,n]
# (2) 计数：count[j] = Σ_k 1{concat_idx[k,n] == j}
# (3) 加权和：Y[n] = Σ_j count[j] * LUT[j]
```
FP16 加法次数从 K 降到 2^(nW+nA)（K=4096 → 256）；与 WOQ LUT-GEMM 的对比见论文 Table I（LUT 大小 2^(nW+nA) vs 2^μ·K/μ；归约 FLOPs 2^(nW+nA)·N vs K/μ·nW·N）。消融（论文 Fig.16，q_proj 层）：OASIS-A4 相对 FIGLUT/LUT Tensor Core 平均降 LUT 大小 62.1×、归约 FLOPs 497.1×；相对 LUT-GEMM 降 994.2×/248.6×；模型越大（K 从 4096 到 26728）WOQ 方案 LUT 爆炸而 OASIS 恒定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WOQ 侧 LUT-GEMM 开源：github.com/naver-aics/lut-gemm（BCQ 格式 GPU kernel）、FLUTE（MIT CUDA 库，LUT 向量化+跨 bank 复制消冲突）、T-MAC（CPU in-register LUT）、FIGLUT（HPCA 2025，RAC 单元硬件）。OASIS 无公开代码；其硬件实现见知识库硬件架构的 Concat Unit / Index Counter / MAC Tree 条目（OASIS 加速器 2KB Cartesian Product LUT、16 PE Line 流水执行）。使用场景：让 NU-WAQ 无需反量化直接高效 GEMM，尤其 decode（batch=1）等 memory-bound 与 prefill 等 compute-intensive 场景（Concat Unit 极简面积设计兼顾两者）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
