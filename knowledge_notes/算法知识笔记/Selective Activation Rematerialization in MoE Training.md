## Selective Activation Rematerialization in MoE Training

术语是什么？
Selective Activation Rematerialization (SAR) 是 MegaScale-MoE 提出的一种内存优化策略。在 MoE 训练中，由于 expert 参数数量庞大，GPU 内存压力显著。传统的 gradient checkpointing（activation recomputation）在反向传播时重计算前向的所有中间激活，以时间换空间；而 SAR 采取"选择性"策略——在前向传播中仅保留计算密集的激活（如 GroupedGEMM 的输入），丢弃可由轻量级计算或通信重新获得的激活（如 RMSNorm 的输出、all-gather 后的结果）。反向传播时，丢弃的激活通过与独立计算/通信 operator 重叠重新生成，不增加关键路径延迟。MegaScale-MoE 将单 MoE layer 的激活内存从 (2n+2k+3kf+12+5/m)bsh/n 降至 (2kf+4+2/m)bsh/n，节省约 50%。

从算法pipeline角度拆解术语：
以 Mixtral MoE layer 的 backward pass 为例：
```
// 前向保留的激活（存在GPU内存中）: hidden, ln1_out, attn_out, ln2_in, fc2_out
// 前向丢弃的激活（反向时重新生成）: fc2_in, ffn_in, fc1_out, fc3_out

// 反向传播时的激活恢复（与梯度通信重叠）:
// 1. 重新计算 ffn_in
recomputed_ln2_out = RMSNorm(ln2_in)        // 轻量级，隐藏在其他通信中
recomputed_ln2_out_ag = All-Gather(recomputed_ln2_out)  // 与上一个GEMM的反向计算重叠
recomputed_ffn_in = Scatter(recomputed_ln2_out_ag)

// 2. 重新计算 fc2_in (SwiGLU 的输入)
// fc1_out 和 fc3_out 通过重新执行 fc1 和 fc3 的 GroupedGEMM 获得
// 这些计算与 Δfc2_out 的 gradient 通信同时进行
recomputed_fc1_out = GroupedGEMM(recomputed_ffn_in, fc1_weight)
recomputed_fc3_out = GroupedGEMM(recomputed_ffn_in, fc3_weight)
recomputed_fc2_in = SiLU(recomputed_fc1_out) * recomputed_fc3_out

// 3. 使用恢复的激活完成反向计算
Δfc2_in = GroupedGEMM_backward(Δfc2_out, fc2_weight, recomputed_fc2_in)
```
关键设计：(1) 将 ffn_out 的加权求和立即放在 SwiGLU 激活函数后（而非单独存储 ffn_out），消除该激活的存储；(2) 不跨越非线性边界重排 operator，保证计算一致性。

术语一般如何实现？如何使用？
- 与 Holistic Scheduling 紧密配合：通过手动编排整个 MoE layer 的前向/反向 operator 执行顺序（而非依赖 torch.autograd），使重计算与通信 overlap。
- 实测在 Mixtral-8×7B 上节省 45.5% 激活内存（总内存节省 21.3%），在 Mixtral-8×22B 上节省 57.2% 激活内存（总内存节省 35%），训练 MFU 差异 <0.5%。
- 适用场景：MoE 模型训练（expert 数量多、激活内存压力大），配合 inter-operator overlap 使用效果最佳。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 采用粒度更细的 activation checkpoint 策略——针对特定激活函数（SwiGLU 中的 SiLU）而非整个通信/计算操作进行选择性重计算。策略基于两个观察：(1) SiLU 为 element-wise 操作（仅 point-wise multiply + sigmoid），在现代 GPU（H100）上 memory bandwidth bound；(2) tall-and-skinny 矩阵（L≫d）下 activation 的内存带宽瓶颈尤为显著。因此 forward 中不保存 SiLU(a)，仅保存 a 和 b 用于 GEMM 反向；backward 时从保存的 a recompute SiLU(a) = a·σ(a)，recompute 开销 ≈ 从 HBM 读取 SiLU(a) 的成本（memory bandwidth bound 条件下）。此策略与 kernel fusion 协同设计——fused kernel 中的 epilogue fusion 已消除了 a, b, σ(a), SiLU(a), y_swi 等中间结果的多次 HBM 往返，checkpoint 进一步消除 SiLU(a) 的存储。此策略在 SwiGLU 下贡献约 4× activation memory reduction（vs MegaBlocks baseline）。

---
