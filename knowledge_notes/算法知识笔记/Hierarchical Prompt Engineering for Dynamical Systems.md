## Hierarchical Prompt Engineering for Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Prompt Engineering for Dynamical Systems 是 LEGO 提出的将动态系统环境信息转化为 LLM 可理解文本的三层提示设计方法。三个层次分别捕获不同粒度的信息：(1) System Level（系统级）：系统的物理背景、参数（如弹簧系数 k、电荷量 q）及高层语义描述（如"The force on the balls are significant"）；(2) Object Level（物体级）：每个物体的初始状态（位置向量 (x,y,z) 和速度向量 (vx,vy,vz)），数值直接作为 digit token（遵循 Gruver et al. 2024 的做法）；(3) Edge Level（边级）：物体间的连接/交互关系，如"ball 2 connects ball 0, ball 1, ball 3"。三层信息构成对环境的完整文本化描述，使 LLM 能理解分布偏移的本质并据此选择合适的 expert。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 *Charged* 数据集为例的三层 prompt 结构（来自论文 Figure 6）：
```
// System Level（系统参数 + 物理背景）
System Description: There are 5 charged particles moving in a 3D space.
The particles interact via Coulomb's law: F = k * q1 * q2 / r².
The charge of each particle is: [1, -1, 1, -1, 1].
The interaction strength k = 1.01.
The system evolves from time step 30 to 40.

// Object Level（逐物体初始状态）
Object 0: initial position (0.12, -0.34, 0.56), initial velocity (0.01, 0.02, -0.01)
Object 1: initial position (-0.23, 0.45, -0.11), initial velocity (-0.02, 0.01, 0.03)
...

// Edge Level（连接/交互关系）
Edge Information: In this charged system, every particle interacts 
with every other particle (fully connected graph).
```

Ablation 实验（Table 4）验证了三层 prompt 的必要性：
- V1（仅 system level）：MSE = 0.761
- V2（system + edge）：MSE = 0.735
- V3（完整三层 prompt）：MSE = 0.728
Edge level 信息（连接关系文本化）的贡献最显著（V1→V2 降幅 > V2→V3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) 系统级 prompt 由环境参数模板填充（如 k=1.01 → "spring constant = 1.01"）；(b) 物体级 prompt 由初始状态矩阵 X⁽⁰⁾ 逐行转换为文本描述；(c) 边级 prompt 由邻接矩阵转换为自然语言连接描述
- 数值编码策略：数值以 digit 形式作为 token（如"0.12"作为单个 token），而非科学记数法或量化表示。遵循 Gruver et al. (2024) 证明 LLM 可直接处理数值序列
- 设计原则：(a) 环境变化相关信息优先（系统参数、边界条件）；(b) 空间结构显式文本化（连接关系）；(c) 数值直接作为 token（保持精度）
- 扩展性：可适配不同物理系统（将 Coulomb/F=ma 等物理规则替换为对应领域的专业描述）
- 局限：(a) 对大规模系统（数百个物体）prompt 可能过长（超过 LLM context window）；(b) 需要人工设计每类系统的 prompt 模板；(c) 科学领域需要领域知识辅助 prompt 设计

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
