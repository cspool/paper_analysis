## Orizuru（Top-k 异常值检测引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Orizuru 是 OASIS 加速器中的轻量 top-k 异常值检测引擎（§IV-D），在推理时动态识别每个激活 token 的 k 个最大与 k 个最小值（保留为 FP16 的 outlier）。名字取自双折叠完全二叉树的形状像纸鹤（Orizuru）。结构：两个完全二叉树——max 树 P 与 min 树 Q，共享 N 个叶子节点（存 FP16 激活值）；每个非叶节点是 2-to-1 MUX，由 bit buffer p_{l,i} 控制选择较大/较小子节点，根节点输出整个向量的最大值。初始化：自底向上逐层比较，N-1 次 FP16 比较；pop：沿根到叶的寄存器二进制位（log2(N) 步，无需比较、单 cycle）定位最大元素索引；maintenance：把被 pop 值视为负无穷更新祖先节点寄存器，每层 1 次比较、共 log2(N) 次，重复 k 次得 top-k。关键优化：min 树初始化复用 max 树最底层（log2 N 层）的比较结果（取反），省 50% 初始化比较。总比较数 1.5N + 2k·log2(N)，远小于 SpAtten 的 top-k 引擎（6N）。硬件配置：273 个 16-input Orizuru 单元/芯片，面积 0.739 mm²、功耗 0.273W（TSMC 28nm @500MHz，论文 Table II）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Orizuru 在 OASIS outlier 分支中的运转流程（N=8 示例，图10）：(1) 初始化——L=log2(8)=3 层，第 3 层 4 个非叶节点比较叶子 (x1,x2)…(x7,x8) 得 4 次比较，第 2 层 2 次、第 1 层 1 次，共 N-1=7 次；(2) pop max——从根沿 bit buffer 方向走 3 步得最大元素索引（如 "110" 前缀 1 成 "1110"=14），单 cycle 无比较；(3) maintenance——把节点 14 视为 −∞，自底向上只更新祖先（每层 1 比较），共 log2(8)=3 次比较；重复 k 次；(4) min 树复用 max 树第 3 层比较结果的反转，从第 2 层开始初始化。每 cycle 引擎顺序输出一个 (outlier 值, 通道索引) 对给 Error Calculation Unit 与 Dequant Unit，驱动误差补偿流水（见算法pipeline Look-ahead 条目）。tie 处理：约 2% token 存在相等激活值，确定性选左孩子保证每 token 恰好 k 个 max + k 个 min。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：完全二叉树 + MUX + bit buffer（FP16 值留在叶子 buffer，比较结果存寄存器），双树共享叶子以复用比较；mask 向量 m 标记已 pop 元素。论文用量化说明（§IV-D）：初始化 N-1 次比较、每次 pop log2(N) 次、min 树复用省 50%，273 个 16-input 单元并行覆盖 4096 维 token（每个单元处理 16 元素）。使用场景：任何"需在流式数据中实时找 top/bottom k"的硬件（激活 outlier 保留、稀疏选择）；对比 SpAtten（6N）与朴素排序，Orizuru 以接近 N 的比较数实现 k 次 pop，且比较结果跨树复用——是 tournament tree（MAERI/Flexagon 归约树谱系）在"极值选择"上的硬件落地。OASIS 无公开 RTL。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
