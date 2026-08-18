## 地址 delta 序列特征（Delta-based Feature Representation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 地址 delta 特征是把连续访存地址的差（delta = addr_t − addr_{t-1}）作为建模对象，而不是原始 64-bit 地址。理由：原始地址缺乏平移不变性（同一模式在不同地址要重复学习）、特征空间巨大稀疏；delta 表示天然平移不变（stride +64 在不同基址都是 +64）且高度结构化。这在预取研究中历史悠久（Global History Buffer、BOP、MLOP、Berti 都用 delta）。Moirai 基于 SPEC 分析提出"delta 稀疏性"（delta sparsity）：按频率排序的 unique delta 中，top 5% 的"频繁 delta"覆盖 62.3% 的访存（cam4_s 达 82%，Figure 2），这让轻量网络只需把有限资源集中在少数关键模式上即可有效泛化；不规则 workload（pr、bc）分布平坦则触发控制单元暂停预取防污染。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- delta 序列 pipeline（Moirai）：
  ```
  # 原始流：addrs = [A0, A1, A2, ...]
  naive_deltas = [A1-A0, A2-A1, ...]        # 朴素 delta（受交替页访问/大跳变污染）
  # Window-based Extended Delta（Algorithm 1，ws=5，8KB 空间约束）：
  for i in 1..ws-1:
      if same_page_8KB(aw[0], aw[i]):       # (aw[0]>>13)⊙(aw[i]>>13)==1
          Δ = aw[i] - aw[0]; return Δ        # 只算低位的"真实局部 delta"
  return abnormal                            # 窗口内无同页 → 丢弃
  features = 滑动窗口取 10 个连续 Δ           # 输入 CaPNet
  ```
  该算法一次解决两个问题：交替页访问（多数据结构交错跳远页产生的振荡假 delta）与 delta 稀疏（异常控制流产生的大离群 delta 噪声）；spec 中交替页访问约占 60% 访存。8KB 空间约束是权衡：单页限制会截断跨页数据结构、放宽则膨胀 delta 词表并交叉污染独立流。
- 效果：干净的 delta 流让 TCN 聚焦频繁 delta 的转移学习；正则化交错形成可学习的新模式、不规则交错被卷积感受野平滑（表式预取器则单点噪声即断链）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件上只需位运算+减法（与运算判断同页、减法算 delta），无 PC 表存储（对比 PC-based stream splitting 需几百字节到 KB 的每-PC 状态表）。使用：作为 BNN/TCN 预取器的输入特征（Moirai 的 Input Processing Unit）；Berti 等表式预取器也按 IP 定位统计本地 delta 分布。局限：可能错过大的跨页 stride 模式（论文承认的权衡）；全局流（非 PC 分流）下的偶发"假 delta"靠 TCN 泛化能力兜底。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
