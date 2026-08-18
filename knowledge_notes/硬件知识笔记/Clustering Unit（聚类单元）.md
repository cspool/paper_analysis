## Clustering Unit（聚类单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clustering Unit 是 OASIS 加速器中执行激活在线非均匀量化的硬件单元（§IV-C，图9b）：把每个 FP16 激活值映射到激活码本中最近的质心索引。实现方式：先计算相邻质心的边界值 b^i = (c^i + c^{i+1})/2，把值域划分为区间 [b^{i-1}, b^i)；对输入 x 用二分查找树结构做分层比较（log2(2^nA) 级），逐级确定所属聚类索引——例如 4 个质心（4-bit 码本 16 质心时为 4 级比较），每个 x 经 log2(4)=2 次层级比较定位簇。硬件配置：每芯片 4 个 Clustering Unit，面积 1.31×10⁻³ mm²/功耗 2.90×10⁻⁴ W（Table II，极小）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Clustering Unit 在 OASIS 主分支流水中的位置（图8 步骤①，图6 步骤①）：Clustering Unit 从 Output Buffer 读取整条 FP16 激活向量 → 对每个元素沿二分查找树比较边界值确定质心索引（4-bit 时 4 级、每级一次比较，类似二分搜索）→ 索引写入 16KB Activation Index Buffer → 广播到 PE Line 供 Concat Unit 拼接。示例：激活质心 C_A=[c0..c15]（排序后），输入 x 先与 (c7+c8)/2 比较决定上半/下半，再与四分之一边界比较……4 次比较得 4-bit 索引；因每 token 有独立质心，聚类按 token 粒度执行。设计动机：激活是动态生成的，无法离线预聚类，硬件必须在每 token 到达时在线完成"最近质心查找"——用二分查找树把朴素 16 次比较（线性扫描）降为 log2(16)=4 次。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：质心按序存储 + 边界值预计算表 + 二叉树 MUX/比较器网络（每级比较后位寄存器记录方向）；与 Orizuru 的树结构类似但比较的是数值与边界而非元素间大小。使用场景：学习码本激活量化的在线编码端（对应软件 PyTorch 的 `torch.cdist`/argmin 聚类 kernel）；配合 OASIS 的 Fisher 加权离线质心训练，在线只做分配。对比 GPU 量化 kernel（per-token 动态量化）的逐元素 argmin，Clustering Unit 是硬连线二分版本，面积 4 个仅 0.00131 mm²。无公开 RTL；面积/功耗来自 TSMC 28nm 综合（Table II）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
