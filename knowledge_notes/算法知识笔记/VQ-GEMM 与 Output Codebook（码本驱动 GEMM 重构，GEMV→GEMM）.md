## VQ-GEMM 与 Output Codebook（码本驱动 GEMM 重构，GEMV→GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的核心算法贡献：把 VQ decode 的 GEMV 重构为 GEMM。由于每个权重向量都来自码本，无需现场重建权重——直接把输入向量与权重码本做点积：输入 x∈R^{1×K} reshape 为 X∈R^{(K/d)×d}，与码本 B∈R^{d×2^n} 相乘得到输出码本（Output Codebook, OC）O∈R^{(K/d)×2^n}（每元素 = 一个输入向量与一个 centroid 的点积，跨输出通道 N 复用）；再用索引矩阵 I 从 OC 查找并累加得最终输出 y=Lookup(O,I)。收益：①计算量从 K×N 降到 K×2^n（N=4096、2^n=256 时约 16× 少）；②M 维从 1 扩到 V=K/d>512，填满矩阵单元（GEMV→GEMM）；③访存规则化/合并化；④带宽每访问从 d 个 FP16 降到 1 个；⑤查 OC 无 bank 冲突（OC 行与 WI 行共享高度 V=K/d，每行独立 bank，同列不同索引自动落不同 bank）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 常规 VQ decode：重建权重后 GEMV  y = x @ W_hat（K×N MAC，memory-bound）
# EVA VQ-GEMM decode：
X = x.reshape(K/d, d)                # x∈R^{1×K}
O = X @ B                            # O∈R^{K/d × 2^n}，输出码本（GEMM，K×2^n MAC）
y = Lookup(O, I).sum(codebooks)      # I∈[0,2^n)^{K/d×N}，冲突无关查找 + 加法树归约
```
LLaMA-2-7B FC 例子（d=8,n=8,C=2）：x∈R^{1×4096}→X∈R^{512×8}；O_c=X·B_c∈R^{512×256}（512×256×8≈1.05M MAC vs 常规 4096×4096≈16.8M）；y=Lookup(O_1,I_1)+Lookup(O_2,I_2)∈R^{1×4096}。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：需要硬件支持（EVA 的 32×8 FP16 重配阵列跑 VQ-GEMM + Epilogue Unit 跑查找/归约）；算法上依赖 AQLM/GPTVQ 等 VQ 码本。使用方式：decode 阶段替代 GEMV；PE:EU 计算比 = 2^n:N（Table III）——2^n<N 时 GEMM 非瓶颈、EU（加法树）为关键路径，可加 EU 数扩展（4 EU 匹配 64GB/s 带宽饱和，再增仅增能耗）；2^n>N 时出现 spurious 乘法（centroid 无输出通道引用，利用率下降）——因此 n=8/256 条目是 EVA 的实用折中。EVA-A16W2 单 batch decode 对 SA/ANT/FIGNA/FIGLUT 分别 31.56×/32.53×/33.50×/11.17× 加速（Fig. 10）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
