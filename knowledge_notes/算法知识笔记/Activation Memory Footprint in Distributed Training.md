## Activation Memory Footprint in Distributed Training

术语是什么？
Activation Memory Footprint Analysis 是对分布式训练中 activation tensors（前向中间结果，需保留至后向计算梯度）占用 GPU DRAM 的定量分析。MPMoE 将 MoE 训练的内存分解为三部分：(1) Model States M_ms（parameters + gradients + Adam momentum/variance）；(2) Activations M_act（T_I, T_DI, T_M, T_DO, T_O）；(3) Temporary Buffers M_buf（后向中间梯度峰值）。揭示了：随 batch size B 增大，M_act 和 M_buf 占比快速上升（Figure 2），成为限制大 batch size 训练的主要瓶颈。

从算法pipeline角度拆解术语：
内存公式（MPMoE Section 2.2.2）：
- M_ms = 4*(E*M + 2*H*M)：params + grads + momentum + variance（×4 for Adam）
- M_act = 4*B*M + B*H：5 个主要 tensors（T_I, T_DI, T_M, T_DO, T_O 各 (B,M) 或 (B,H)）
- M_buf = B*M + B*H：后向中相邻两个 tensor 的梯度峰值
- Pipeline 后 M_act^pipe = M_buf^pipe = 4*B*M + B*H（总量不变，时间分布改变）
- Memory Reuse 后 ΔM_buf = ΔM_act = B*(2M*(n-2)/n + H*(n-1)/n)
- 总节省率 φ = (ΔM_act + ΔM_buf)/(M_ms + M_act^pipe + M_buf^pipe)

实际案例：1.5B GPT-2 (seq_len=1K, batch_size=32) 约需 60GB GPU 内存，~80% 为 activations。

术语一般如何实现？如何使用？
- ZeRO（Rajbhandari et al. 2020）处理 model states 瓶颈（partitioning across devices），但未处理 activations。
- Gradient Checkpointing（Chen et al. 2016）丢弃中间 activations 后向重计算，以计算换内存。
- vDNN（Rhu et al. 2016）将 activations offload 到 CPU。
- MPMoE 的贡献：将三类方法（partitioning + recomputation + offloading）系统组合应用于 MoE pipeline 场景，并给出量化的内存分析公式指导策略选择。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---
