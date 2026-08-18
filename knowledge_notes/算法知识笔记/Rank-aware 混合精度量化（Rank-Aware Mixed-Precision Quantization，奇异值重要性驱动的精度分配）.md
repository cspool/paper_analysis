## Rank-aware 混合精度量化（Rank-Aware Mixed-Precision Quantization，奇异值重要性驱动的精度分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Rank-aware 混合精度量化是 SingularBit 的核心算法：对 SVD 分解后的权重矩阵，按奇异值大小把 rank 分量划分成 K 个精度区域，奇异值大的区域（主导信息）分配高比特（4-bit）、奇异值小的长尾区域分配低比特（1–2-bit），实现"全 rank 保留 + 按重要性差异化位宽"的压缩，而不是均匀量化或截断。理论基础是 LLM 权重奇异值近似指数衰减（Fig.4），因此累计尾部占比 $C_i$ 可解析刻画每 rank 信息密度。SingularBit 用单一参数 p（rank ratio）闭式确定 K=4 区域的边界：$C_{r_k}=(1-p)^{K-k}$（Eq.3），并由目标平均精度 $B_{avg}=\frac{1}{R}\sum_k b_k(r_k-r_{k-1})$（Eq.4）反解出 p——全程解析、无需对每模型/每层做启发式超参搜索，泛化到不同架构。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Algorithm 1 精简）：①对 $W^T$ 做 SVD 得 U、S、V^T；②按累计尾部占比定边界 $\{r_1,r_2,r_3\}$（K=4，位宽 $\{4,3,2,1\}$）；③用 calibration 激活算 Hessian $H_U=x^Tx$，逆序逐块量化 U（每块按所在区域位宽做分层二进制量化），量化误差 $E_u=U-\hat{U}$ 乘 $H_U^{-1}$ 反馈回未量化参数（GPTQ 错误反馈）；④由已量化 $\hat{U}$ 推导有效 Hessian $H_{V^T}=S\hat{U}^T H_U \hat{U}S$（来自 $z=x\hat{U}S$ 的 $H=z^Tz$），正序逐块量化 V^T 并同样做误差反馈；⑤输出 $\hat{U},\hat{V}^T$。张量计算例子（LLaMA-7B 一个 FFN 线性层，W∈R^{11008×4096}）：SVD 后 r≈4096 个 rank 被分为 4 区（如 r1≈前 400 个 rank 用 4-bit、之后 3-bit、2-bit、1-bit），目标 B_avg=2 时 p 由 Eq.3/4 解出；推理时 U 计算走空间混合精度（不同 rank 分给不同 core 并行）、V^T 走时间混合精度（同 core 沿归约维顺序累加）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：离线一次性执行（权重静态，无推理开销）；精度区域表示采用 ARB-LLM 式分层二进制量化（每区域 $W=\sum_{i=1}^{b_k}\alpha_{r,i}\alpha_{c,i}B_i$），位宽 b_k 直接换算成硬件位串行延迟/能耗。设计要点：最大精度限 4-bit（更高精度需把更多 rank 压到低比特区以维持 B_avg，收益被抵消）；不分配 0-bit（不剪枝）——把 (n+1) 位降到 n 位只损失表征能力，而置 0 是完全丢失信息，这解释了 rank 截断方法（SVD-LLM/ASVD）为何在同等压缩率下掉点更多。论文结果：2-bit 下优于 OmniQuant、MagR+OPTQ 等专门低比特方法（LLaMA-7B Wiki 7.56 vs 9.72/9.89），并扩展到 KV 压缩（SingularBit-KV 的 rank 维策略直接复用本边界）。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
