## Stream-Based Modeling for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Stream-Based Modeling 是 HybridEP 提出的用于在跨 DC 带宽受限环境下决定 MoE 训练中最优数据-专家混合传输比例的解析性能模型。它采用分治策略：(1) 将 MoE 训练解耦为计算流（GeMM modeling）和通信流（A2A + AG modeling）；(2) 分析两流之间的重叠关系（pre-expert 与 AG、expert 与 AG、expert 与 A2A）；(3) 联合计算、通信和重叠，构建以最小化训练延迟为目标的优化问题，求解最优的 A2A 数据比例 p。p=1 时退化为标准 EP（纯 A2A），p=0 时仅使用 AG。该建模的关键洞察是 A2A 与 AG 间存在 trade-off：减少 A2A 的 $\frac{D}{G}$ 流量换取 AG 的 $P_E$ 流量——当 expert 参数 $P_E$ 足够小或可充分压缩时（使 $2D \ge G \cdot P_E$），应完全使用 AG 替代 A2A。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Modeling 的四阶段流程：

```
Phase A: Computation Modeling
  Lat_comp = (m+1)*Lat_comp^Att + m*Lat_comp^FFN + n*Lat_comp^Ep
  Lat_comp^PE = (m+1)*Lat_comp^Att + m*Lat_comp^FFN  (pre-expert)
  其中 Lat_comp^GeMM = LMH / C  (C: GPU comp throughput)

Phase B: Communication Modeling
  V^A2A = D*(|G^A2A|-1) / |G^A2A|  → Lat_comm^A2A = V^A2A / B
  V^AG = P_E * (|G^AG|-1)          → Lat_comm^AG = V^AG / B
  Lat_comm = Lat_comm^AG + 2*Lat_comm^A2A

Phase C: Overlap Modeling
  Lat_ovlp = min(Lat_comp^PE, Lat_comm^AG) + n*Lat_comp^Ep
  (专家计算与 AG+A2A 完全重叠; pre-expert 仅与 AG 部分重叠)

Phase D: Problem Solution (Eq.10-12)
  min Lat_final(p) = Lat_comp + Lat_comm - Lat_ovlp
  3 cases based on 2D - G*P_E  sign:
    Case 1 (2D < G*P_E, Lat_comp^PE >= Lat_comm^AG):
      → 最小 p = 1 - B*Lat_comp^PE / (P_E*(G-1))  (混合 A2A+AG)
    Case 2.1 (2D < G*P_E, Lat_comp^PE < Lat_comm^AG):
      → 最大 p (接近 1)  (偏重 A2A)
    Case 2.2 (2D >= G*P_E):
      → p = 0  (纯 AG, 完全消除 A2A)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 训练前一次性计算：在训练开始时，Modeling 根据集群配置（GPU 数 G、带宽 B、expert 大小 P_E、数据大小 D、pre-expert 延迟 Lat_comp^PE）计算最优 p 值，指导 Domain-Based Partition 设定 Expert Domain 大小。
- 模型验证：HybridEP 实际测量了 computation、A2A 和 AG 的延迟并与模型估计对比（Figure 11），误差源于共享集群的网络波动但不影响模型决策有效性。
- 模型局限性：(1) 假设 gate network 均匀激活 expert；(2) 假设各 GPU 的 p 相同；(3) 初始假设每 DC 仅 1 GPU（用于对齐通信粒度）。论文声称这些简化不影响模型准确性，且模型可扩展到多 GPU per DC 场景。
- 类似的 modeling 工作：FasterMoE 使用线性模型建模 MoE 训练延迟进行 degree 优化；FSMoE 使用 α-β 通信模型 + SLSQP 求解器优化 pipeline degree；HybridEP 的独特之处是将 A2A→AG 转换作为新的优化维度引入。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
