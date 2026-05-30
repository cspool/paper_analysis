## Expert Domain (专家域)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Domain（专家域）是 HybridEP 提出的用于在跨数据中心 MoE 训练中分离和管理混合通信模式的核心抽象。一个 Expert Domain 定义为一组仅在其内部使用 All-Gather (AG) 通信的 DC（或 GPU）集合，遵循"域内 AG、域间 A2A"的通信规则。Expert Domain 的大小 $S_{ED}$ 定义为域内包含的 worker 数量（worker 可以是 DC、node 或 GPU）。Domain 的核心作用是：(1) 将混合通信（A2A 用于 token data，AG 用于 expert parameter）在空间上分离，避免通信模式冲突；(2) 通过扩展 Domain 大小（即减少 A2A 比例 p）来降低跨 DC 低带宽链路上的通信量；(3) 兼容多层级硬件架构——在 Multilevel Description 中，每个层级都有独立的 domain 大小 $S_{ED}^l$，通过 Location Renumbering 和 Topology Construction 映射到 GPU 级通信拓扑。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 HybridEP 的 Domain-Based Partition 中，Expert Domain 的系统运转流程：

```
# Expert Domain 系统架构流程（2 levels: inter-DC + intra-DC, 4 DCs × 8 GPUs）
# Level 0 (inter-DC): SF^0=4, S_ED^0=2  → 4 DCs 分 2 个 domain, 每 domain 2 DCs
# Level 1 (intra-DC): SF^1=8, S_ED^1=4  → 每 DC 内 8 GPUs 分 2 个 domain

Step 1: Multilevel Description
  抽象环境为 L=2 层: [Level 0: DC, Level 1: GPU]
  SF = [4, 8], S_ED = [2, 4]

Step 2: Location Renumbering (Eq.13)
  GPU 全局索引 m → (x_0, x_1)
  x_0 = m / 8  (DC编号)
  x_1 = m % 8  (DC内GPU编号)
  
Step 3: Domain Assignment
  GPU m 的 Level 0 domain = x_0 / 2   (DC属于哪个inter-DC domain)
  GPU m 的 Level 1 domain = x_1 / 4   (GPU属于哪个intra-DC domain)

Step 4: Topology Construction (Algorithm 1)
  对每对 GPU (m, n):
    For Level 0 (inter-DC):
        if domain_m == domain_n:
            → AG (域内, 跨DC通过AG交换expert)
        else:
            → A2A (域间, 通过A2A交换token)
    For Level 1 (intra-DC):
        if domain_m == domain_n:
            → AG (域内, 同DC内AG交换expert)
        else:
            → A2A (域间, 同DC内跨domain的A2A)

效果: 跨DC的A2A通信仅在不同domain的DC间发生
     同domain内的DC间通过AG（可压缩expert传输）替代A2A（不可压缩token传输）
```

Expert Domain 的扩展逻辑：$p = 1 - S_{ED}/G$，即域越大，A2A 比例越低（更多通信转为 AG）。极端情况 $S_{ED}=G$ 时 p=0（纯 AG，无 A2A），$S_{ED}=1$ 时 p=1（标准 EP，纯 A2A）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HybridEP 通过 Stream-Based Modeling 在训练前根据集群配置（G, B, P_E, D）自动确定最优 $S_{ED}$ 值（由最优 p 推导），无需手动调整。
- Domain 的粒度和层次与物理拓扑绑定——Level 0 对应 inter-DC 链路，Level 1 对应 intra-DC 链路，每层的 domain 大小独立设定以匹配该层的带宽特性（低带宽层用大 domain 减少 A2A，高带宽层可用小 domain）。
- Domain-Based Partition 的扩展性限制：固定 $S_{ED}$ 下，DC 数增加会导致 p 增大（$p \approx 1 - S_{ED}/G$），加速效果递减。HybridEP 在 1000 DC 仿真中，固定 $S_{ED}$ 的加速比从低 DC 数的 ~3.76× 降至 1000 DC 时的 1.45×。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
