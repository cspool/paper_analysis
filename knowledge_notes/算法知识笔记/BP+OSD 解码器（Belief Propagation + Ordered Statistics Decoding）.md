## BP+OSD 解码器（Belief Propagation + Ordered Statistics Decoding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BP+OSD 是置信传播与有序统计解码的组合解码器，qLDPC 码事实标准（Panteleev & Kalachev 提出）：在 Tanner 图上运行 BP 消息传递，迭代估计与 syndrome 一致的最可能错误；当 BP 因量子简并/短环不收敛时，触发 OSD 后处理——按软信息可靠度（LLR）对错误位置排序、重排校验矩阵列、对可靠子集做矩阵求逆解线性方程确定错误。优点：对 qLDPC/表面码通用、精度接近最优；缺点：OSD 矩阵求逆昂贵，实时硬件化难（首款并行 BP+OSD FPGA/ASIC 2025 年才出现：EPJ Quantum Technology，d≤9 单 VCU129 @200 MHz 134 μs；bicycle 码 d≤12 @244 MHz 84 μs）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
1: 在 Tanner 图(校验节点=稳定子, 变量节点=qubit/错误)上迭代 BP
   消息 = 软信息(对数似然)传播，估计每边错误概率
2: if 硬判决满足全部 syndrome: return 估计错误     # BP 收敛
3: OSD: 按可靠度降序排列列 -> 最可靠列求逆(高斯消元)
   -> 解方程得错误集合; 必要时 OSD-0/CS/w 组合搜索
```
本论文用法：作为精度参照——product-sum BP + OSD-CS（order 15），同一 Tanner 图、repetition code、phenomenological noise、d∈{5,7}、p∈[0.04,0.08]：BP+OSD LER 距 MWPM 1.0–1.7×，本文陪集集成解码 1.0–1.4× 与之相当。作用是在"UF（快差）—MWPM（准慢）"谱系中给本文方法一个独立标定（本文不声称超越 MWPM/BP+OSD 精度，只求近似并保持低延迟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：开源 ldpc/BPOSD 实现（Roffe 的 ldpc 库）、qecsim、GPU 加速（NVIDIA CUDA-Q QEC 2026）。硬件：EPJ QT 2025 FPGA/ASIC 设计空间探索、进化式 EBP+OSD（arXiv:2512.18273，差分进化调权、更少 OSD 激活）。使用场景：qLDPC（BB 码）与表面码离线解码、其他解码器的精度上界参照；实时部署需控制 OSD 激活次数（Astra+OSD 可减少 >2000× OSD 调用）。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
