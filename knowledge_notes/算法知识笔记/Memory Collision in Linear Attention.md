## Memory Collision in Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Collision 是线性 Transformer/RNN 的核心瓶颈（Schlag et al., 2021）。线性 Transformer 状态 S ∈ R^{d×d} 通过外积 v k^T 存储 key-value 关联，最大可存储的正交 key-value 对数受维度 d 限制。当序列长度 L > d 时，新 key-value 无法与已有对正交存储，信息在有限状态空间中叠加导致"碰撞"，使精确检索不可能。缓解策略：(a) Gating/Forgetting（Mamba2 α_t、RWKV w_t）—主动遗忘不相关信息；(b) Delta Rule—精确替换而非叠加；(c) State Expansion（Eagle/Finch head size > 1）—扩大实际存储容量；(d) Hybrid—混合 attention 提供精确检索。Gated DeltaNet 展示了 (a)+(b) 的组合是最优策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
S_t = Σ_{i=1}^t v_i k_i^T ∈ R^{d×d}
检索: o_t = S_t q_t = Σ v_i (k_i^T q_t)
若 k_i 正交，最多存 d 个独立 key-value 对；t > d 时必然 collision
```

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
