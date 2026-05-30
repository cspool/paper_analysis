## Expert Partition in MoE (MoE 专家划分)

术语解释
Expert Partition 是在 post-training 阶段将预训练 MoE 模型中每个 expert 划分为 P 个更细粒度 experts，通过增加 tensor-level sparsity 提升 fine-tuning quality 和 inference efficiency，而无需重新预训练。含 Complete Transformation（数学等价变换，适用 fine-tuning 提升）和 Partial Transformation（保持 gating network，适用系统效率优化）。

术语是什么？
MoE 的 expert granularity 在 pre-training 时确定无法后续改变，但 prior work 证明 finer-grained experts 在相同 per-token FLOPs 下可降低 pre-training loss。Expert Partition 在 post-training 阶段实现等效效果。Complete Transformation 三步：重复 W_g 中每个 expert-specific vector P 次 + Top-K → Top-(K×P) + 均分每个 expert 的 neurons 为 P 份 + 将每个新 expert 的 W₂ 乘以 P（补偿因 gating score 变为 1/P 倍导致的输出缩放）。数学验证：s_{e,p} = s_e/P (各子 expert gating score 相等)，输出 y_i^P = y_i（W₂ 补偿后等价）。Partial Transformation 两步：仅重复 gating scores + remap expert indices（contiguous mapping），不修改 W_g 参数和 W₂ 权重。

从算法pipeline角度拆解术语：
```
=== Complete Transformation (P=2, E=2→4) ===
W_g^new = [h_0|h_0|h_1|h_1]  (repeat P times)
Top-K → Top-2K (Top-4)
For each original expert e:
  W1_new^{e,0} = W1_e[:, :d_ffn/2], W1_new^{e,1} = W1_e[:, d_ffn/2:]
  W2_new^{e,0} = W2_e[:d_ffn/2, :] × P,  W2_new^{e,1} = W2_e[d_ffn/2:, :] × P
  W3_new^{e,0} = W3_e[:, :d_ffn/2], W3_new^{e,1} = W3_e[:, d_ffn/2:]

Output consistency:
  s_{e,p} = s_e / P  (repeated W_g vectors → equal logits)
  Σ_{p} s_{e,p} · f_{e,p} = (s_e/P) · Σ_{p} f_{e,p} = s_e · f_e / P
  After W₂×P: s_e · f_e ✓

=== Partial Transformation ===
Gating scores repeated P times, expert indices remapped contiguously.
W₂ NOT scaled. Output: s_e · Σ f_{e,p} = s_e · f_e ✓ (no compensation needed)
```

术语一般如何实现？如何使用？
- Complete Transformation：fine-tuning 前使用 → Mixtral 8→32: downstream accuracy +0.59%；与现有 MoE 框架原生兼容（输出即标准 MoE）
- Partial Transformation：(a) S-ETP 通信优化（AlltoAll 替代 AlltoAll+AllGather）；(b) DualSparse-MoE 推理加速（更细 dropping 粒度）；(c) EP scale-up（更多 experts 分布到更多 devices）
- 局限性：P 过大→marginal benefit 递减 + compute intensity 下降 + gating overhead 增加

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
