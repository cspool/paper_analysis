## Delta Rule (DeltaNet)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Delta Rule 由 Widrow et al. (1960) 提出，Schlag et al. (2021) 将其引入线性 Transformer 形成 DeltaNet。在 DeltaNet 中实现为：S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T，其中 S ∈ R^{d×d} 是记忆矩阵。计算分两步：（1）读旧值：v_t^old = S_{t-1} k_t；（2）写入增量：用 Householder 变换 (I - β_t k_t k_t^T) 擦除旧关联并写入 β_t v_t k_t^T。从快速权重编程视角，delta rule 等价于对在线回归目标 L(S) = 1/2 ||S k_t - v_t||² 执行一步 SGD：S_{t+1} = S_t - β_t ∇L(S_t)，β_t 为自适应学习率。DeltaNet 的优势在于精确 key-value 替换（优于 Mamba2 的简单叠加），局限在于缺乏全局遗忘机制（只能逐个修改 key-value 对）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
v_old = S_{t-1} @ k_t
v_new = β_t * v_t + (1 - β_t) * v_old
S_t = S_{t-1} - v_old @ k_t^T + v_new @ k_t^T
    = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Yang et al. (2024b) 提出基于 WY 表示的 chunkwise 并行训练算法，使 DeltaNet 训练从不可行变为接近 Mamba2 的速度。开源：https://github.com/NVlabs/GatedDeltaNet。适用于需要精确 key-value 联想记忆的序列建模场景。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
