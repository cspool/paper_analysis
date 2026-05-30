## Domain-Based Partition

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Domain-Based Partition 是 HybridEP 中将由 Stream-Based Modeling 确定的最优混合通信比例（p 值）映射到 GPU 级通信拓扑的三步构建过程。它解决了从抽象模型参数到实际可执行通信模式的转换问题：即使已知应使用多少比例的 A2A 和 AG，通信的最小粒度是 GPU（而非 DC），需要在 GPU 级别确定每对 GPU 间的具体通信方式。三步构建流程为：(1) Multilevel Description——将集群的层级硬件架构（DC→Node→GPU）抽象为多层 worker 树，用 scaling factor $SF^i$ 描述层间扩展关系；(2) Location Renumbering——按 PyTorch 全局索引方式将 GPU 编号重映射为多层级位置坐标 $(x_0, x_1, ..., x_{L-1})$；(3) Topology Construction (Algorithm 1)——对每对 GPU，逐层根据 Expert Domain 规则判断通信类型（AG/A2A/None）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Algorithm 1 的 Topology Construction 伪代码（决定 GPU m 和 GPU n 间是否需要通信及通信类型）：

```
def topology_construction(m, n, l, SF, S_ED):
    # m, n: GPU global indices
    # l: current level
    # SF: scaling factor list
    # S_ED: expert domain sizes per level
    
    # Step 1: Multilevel location
    Loc_m = f(m)  # Eq.13: global index → (x_0, x_1, ..., x_{L-1})
    Loc_n = f(n)
    
    # Step 2: Get level-l worker and domain
    W_m, W_n = Loc_m[l], Loc_n[l]
    ED_m, off_m = W_m // S_ED[l], W_m % S_ED[l]
    ED_n, off_n = W_n // S_ED[l], W_n % S_ED[l]
    
    # Step 3: Communication scope check
    # Only communicate if higher-level locations match
    if Loc_m[l+1:] != Loc_n[l+1:]:
        return None  # 不在同一高层分组，不直接通信
    
    # Step 4: Domain-based decision
    if ED_m == ED_n and off_m != off_n:
        return AG  # 同域内：All-Gather expert parameters
    if ED_m != ED_n and off_m == off_n:
        return A2A  # 跨域：All-to-All token data
    
    return None  # 非同行/非同位，不需要直接通信
```

关键设计：(1) 限制通信范围——仅在高层位置一致的 GPU 间通信（`Loc_m[l+1:] == Loc_n[l+1:]`），避免跨层级通信混乱；(2) 同域同 offset 的 GPU 通过 AG 交换 expert，跨域同 offset 的 GPU 通过 A2A 交换 token；(3) 分层决策确保与物理拓扑的层级一致性——inter-DC 层决定跨 DC 通信模式，intra-DC 层决定节点内通信模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HybridEP 基于 Tutel + PyTorch v1.12.1 实现，代码未公开开源。
- Topology Construction 的输出是静态的通信拓扑（在训练开始前根据环境配置一次性计算），并非每 iteration 动态调整。拓扑固定后，每个 GPU 在每个 MoE layer 的前向和反向中使用相同的通信对（AG 或 A2A）。
- Domain-Based Partition 的关键优势是与现有层级硬件架构的兼容性——不假设扁平网络，而是显式建模和利用多层带宽差异（如 PCIe vs Ethernet），使高层（低带宽层）使用大 domain（少 A2A，多 AG），低层（高带宽层）使用小 domain。
- 与其他 EP 拓扑优化（如 Tutel-2DH、HierMoE 的 HierD-AlltoAll）的区别：Domain-Based Partition 通过引入 AG 改变通信本质（从 token data 转为 expert parameter），而不仅是优化 A2A 的层级分布。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
