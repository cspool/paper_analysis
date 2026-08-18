## Redundantly-Encoded Fusion（冗余编码融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
冗余编码融合（redundantly-encoded fusion，Hilaire 等, Quantum 7, 992, 2023）是第一个面向自旋内存架构的融合失败容错方案：利用 caterpillar 态的特性，把逻辑线性图态的每个节点编码为 m 个叶量子比特组成的逻辑量子比特；两个逻辑量子比特融合时，对两方每对叶量子比特各施加一次融合操作——即对两个逻辑量子比特执行 m 次融合尝试，任意一次成功即逻辑融合成功。这压缩了逻辑失败率：P_fail=p_fail^m。然而代价是逻辑擦除率 P_eras=1-(1-p_eras)^(2m)：每个逻辑量子比特的 m 个物理量子比特各自独立暴露于擦除（2m 指数），因此 m 越大，失败率越低但擦除率越高——在真实光子丢失（p_eras≈10%）下，擦除很快成为主导错误，这正是该方案与 RUS 的共同致命缺陷。论文中 redundantly-encoded 作为树编码融合的 baseline：代码尺寸 m_Redun=5（按 [12][25] 最优容错性能）；对比结果：树编码执行时间为其 1.9×10^-3×，但光子源多 2.55×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
冗余编码融合的计算过程（伪代码）：
```
# 逻辑量子比特 L 编码：m 个叶 qubit {l_1,...,l_m}（来自 caterpillar 的叶分支）
# 逻辑融合 A-B：对每对叶 qubit 施加融合
for k in 1..m:
    outcome_k = type2_fusion(l_k(A), l_k(B))
    if outcome_k == SUCCESS:
        logical_success = True        # 任一成功即逻辑成功
        break
# 逻辑错误率：
# P_fail = p_fail^m
# P_eras = 1 - (1-p_eras)^(2m)      # 2m 个物理 qubit 独立暴露于擦除
```
例：p_fail=0.25、p_eras=0.1、m=5：P_fail≈9.8×10^-4，但 P_eras=1-(0.9)^10≈65%——擦除完全主导。对比树编码 b=4 时 S_tree≈96%（见树编码条目），冗余编码在 p_eras 升高时不可用（Fig.4(f)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——利用 caterpillar 态每个主路径顶点挂多个叶 qubit 的结构，天然提供 m 个叶量子比特；(2) 编译集成——论文按 [12][25] 的最新协议实现 redundantly-encoded 并集成进 MemTree 框架作对比（m_Redun=5）；(3) 模拟——论文 Fig.4(f) 以 10^3 次融合试验统计其成功率。使用场景：作为 boosted fusion 的早期方案，可用于融合失败为主、擦除可忽略的假设场景；在真实光子丢失条件下被树编码取代。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
