## Conv-aware Encoding（卷积感知 Fragment 编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Conv-aware Encoding 是 FEnc² 的第一个组件：一种卷积感知的 fragment（块）编码布局，把 4D 特征张量（batch、channel、H、W）按 S×S 子块分解，每个块编码进独立密文，从而同时解耦相邻像素（空间依赖）与跨通道（通道依赖）两类卷积数据依赖。通过解析旋转代价模型选择最优块大小 S*，使内/外旋转总代价最小，无需运行期 profiling（Algorithm 1 + Theorem 1）。
- 关键性质：一般性（generality）——S=1,BS=1 时退化为 row-major/Orion 编码，S=M 时退化为 CryptoNets 像素式编码，故 FEnc² 统一包含先验方案为非最优特例；最优性（optimality）——内旋转项随 1/S² 递减、外旋转项随 S²α 递增（式 (3)-(5)），两者相等时总旋转最小（Cauchy-Schwarz），得 S* = ⌈(K²N_in/(αN_out))^(1/4)⌉（式 (8)），S 还需满足上界 S ≤ √(BS·N_in/α) 防止槽浪费（式 (7)）。
- 大 batch 特例：batch 足够大时不同样本可单独装满密文，外旋转完全消除（式 (6) Rot_amortized 只剩内旋转），与实验（Fig.7 大 batch 用大块收益更明显）一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 1×16×4×4 输入、卷积 (16,16,3,1)、BS=1、16 slots 为例（本论文 Fig.4）：
```
M=max(4,4)=4；选 S=2 → m=2（2×2 个 2×2 块）
对每个块内坐标 (u,v)∈{(0,0),(0,1),(1,0),(1,1)}：
    收集 4 个块中 (u,v) 位置的元素 → X_{uv}∈R^{16×1×4}
    按 Algorithm 1 槽映射展平 → 16 槽满装 → 加密 ct_{uv}
卷积时：K=3>S=2 → 每密文内旋转 (⌈3/2⌉²−1)=3 次
外旋转 N_out/α×(αS²/BS−1) 由 α 与 S 共同控制
S=1：内旋转最大、无外旋转（Orion/CHET 式）；S=4：无内旋转、外旋转最大（CryptoNets 式）
S=2：内/外平衡 → 总旋转最小（最优）
```
- Annotations：S 决定每密文装多少同通道相关像素（1/S²）与多少通道（αS²/BS）；S* 解析可算、无需搜索或 profiling；stride≥2 卷积/平均池化直接丢弃多余密文实现（免去先验方案的后处理重排）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：客户端初始化时只发非敏感元数据（H,W,C,BS,模型 id），FEnc² 依据张量形状与模型结构自动算出 S* 与层间块大小调整方案（rot-mask-add 重打包，开销仅 0.42%-3.7%），返回索引-槽映射给客户端编码加密；服务端照布局执行。与硬件无关（任何 CPU/GPU/FPGA/ASIC HE 执行平台通用）。效果：旋转数相对 HELayers 降 67%-94%，keyswitch 降 80%-93%，NTT/iNTT 降 89%-94%，密文数与同态乘法降 78%-94%。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
