## Instruction-Guided Attention (IGA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

IGA（Instruction-Guided Attention）是 TransPrune 中与 TTV 互补的 token 重要性评估准则。IGA 通过计算 instruction tokens 对 image tokens 的单向 attention 权重来评估每个 visual token 在给定指令下的语义相关性。与需要完整 N×N attention matrix 的传统方法不同，IGA 仅计算 instruction→image 的单向 attention（L×N 而非 N×N，L 为 instruction token 数通常仅几十个），因此计算开销极小。IGA = mean(softmax(Q_inst @ K_img^T / sqrt(d)), dim=instruction)，即对所有 instruction token 的 attention 权重取平均，得到每个 image token 的重要性分数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**IGA 的计算流程**（在 pruning layer p_i，使用下一层 p_{i+1} 的 attention weights）：

```
# 输入: Q_inst [L, d] (第 p_{i+1} 层 instruction tokens 的 query)
#        K_img [N_retained, d] (第 p_{i+1} 层保留 visual tokens 的 key)
# 输出: IGA [N_retained] (每个保留 visual token 的重要性分数)

# Step 1: 计算 instruction → image 的原始 attention scores
A_raw = Q_inst @ K_img^T           # [L, N_retained]

# Step 2: Scaled softmax (与标准 attention 一致)
A = softmax(A_raw / sqrt(d), dim=-1)  # [L, N_retained], 沿 image token 维度

# Step 3: 对所有 instruction tokens 取平均
IGA = mean(A, dim=0)               # [N_retained], Equation (5)
```

**与 TTV 的组合**（Equation 6）：
```
Score = α * TTV_acc + (1-α) * IGA  # α=0.5, 等权平衡
# 按 Score 升序排列，剪除得分最低的 tokens
```

IGA 引入任务相关的语义监督——TTV 仅依赖 token 自身的 transition 信号（与指令无关），IGA 补充了"该 token 是否与当前指令相关"的信息。消融实验（Table 12）显示：仅用 IGA 时 MME^P=1514；添加 TTV 的 magnitude 和 direction 组件后分别提升到 1532 和 1521；两者联合达到最优 1540。

术语一般如何实现？如何使用？

IGA 在 TransPrune 中的实现要点：(1) IGA 在每个 pruning layer 使用**下一层（p_i+1）**的 attention weights（因当前层的 attention 已在 forward 中计算，取下一层的 attention 在计算上更自然）；(2) IGA 不使用 accumulation 机制（仅 TTV 使用 accumulation），因为 IGA 直接反映当前指令的语义相关性，不需要"历史"信息；(3) IGA 的额外 FLOPs 为每个 pruning stage 的 L×n_i×d（L 为 instruction token 数，n_i 为当前保留的 visual token 数，d 为 hidden dim），在 VQA 场景中通常 L≈几十，与 N×N attention 相比开销极小；(4) IGA 不能完全消除 attention 的位置偏差问题——论文 Figure 4(a) 显示 IGA 仍呈现一定的首尾位置偏好（因为底层仍是 attention 计算），但通过与 TTV 结合可部分缓解（Figure 4 对比，TTV 保留 token 位置分布更均匀集中在图像中央语义区域）；(5) 参数 α 默认 0.5 使 TTV 和 IGA 等权贡献——Table 13 消融显示 α=0.4 时 MMB^en=65.5，α=0.5 时 MMB^en=66.0，α=0.6 时 MMB^en=65.9，α=0.5 最优。
