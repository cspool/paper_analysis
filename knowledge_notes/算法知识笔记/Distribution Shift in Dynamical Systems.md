## Distribution Shift in Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Shift in Dynamical Systems（动态系统中的分布偏移）指训练和测试环境之间的数据分布不匹配，通常由系统参数 ξ（如弹簧系数 k、电荷量 q、分子类型）或初始状态分布的变化引起。形式化定义：设系统演化由 dX/dt = F(X, ξ) 决定，训练和测试的数据分布分别为 P_train(X⁰, ξ) 和 P_test(X⁰, ξ)。当 P_train ≠ P_test 时（环境参数从 ξ~P_train(ξ) 变为 ξ~P_test(ξ)），传统数据驱动方法（EGNN/EGNO 等）因仅从训练数据隐式学习分布而性能显著下降。LEGO 通过 LLM 显式理解环境参数（文本化 ξ）来选择合适的 model expert，缓解此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LEGO 论文中考虑的三种环境变化类型（以 Spring 为例）：
```
// 类型1: Hard environment（更强的物理系数）
训练: strength k = 1.0, start=30, end=40
测试: strength k = 1.10（弹簧更硬，移动更剧烈）
结果: EGNN MSE = 0.112 → EGNN+LEGO MSE = 0.078 (↓30.4%)

// 类型2: Soft environment（更弱的物理系数）
训练: strength k = 1.0, start=30, end=40
测试: strength k = 0.90（弹簧更软，移动更缓慢）
结果: EGNN MSE = 0.118 → EGNN+LEGO MSE = 0.114 (↓3.4%)

// 类型3: Temporal Shift（不同时间窗口）
训练: start=30, end=40
测试: start=20, end=30
结果: EGNN MSE = 0.115 → EGNN+LEGO MSE = 0.072 (↓37.4%)
```

跨分子迁移（OOD，MD17）：
```
训练分子: salicylic acid（9个重原子）
测试分子: naphthalene（10个碳原子，无氧原子）
// 分子拓扑和化学性质完全不同
EGNN MSE = 0.320 → Radial Field+LEGO MSE = 0.186 (↓41.9%)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 传统应对方法：(a) Domain Generalization（域泛化）：从多域数据中学习域不变特征；(b) Domain Adaptation（域适配）：利用目标域无标签数据做分布对齐；(c) Test-Time Adaptation：推理时在线调整模型参数
- LEGO 的创新：(a) 用 LLM 的常识推理替代数据驱动的域泛化——LLM 被告知"k=1.10"可推理出"弹簧更硬"并选择相应的 expert；(b) MoE 的 expert specialization 天然适合多域——不同 expert 可专门适配不同环境模式
- 评估 benchmark：Spring（Hard/Soft/Temporal Shift）、Charged（Hard/Soft/Temporal Shift + 多种 strength）、MD17（跨分子迁移）、Motion（跨受试者/运动类型迁移）
- 局限：(a) LLM 对环境的理解限于 prompt 中的信息；(b) 对于从未见过的全新物理系统类型（如训练集全是弹簧、测试集是电荷），LLM 的判断也可能不准确

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
