## Unified Pipeline Scheduling for MoE Training (FlowMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unified Pipeline Scheduling 是 FlowMoE (NeurIPS 2025) 提出的分布式 MoE 训练调度策略。其核心是将流水线调度边界从"仅 MoE 层内部"扩展到"整个 Transformer block"，统一编排 MHA 计算、gating、expert 计算、A2A 通信和 all-reduce 通信。传统方法（Tutel/ScheMoE/PipeMoE）仅对 MoE 层内的 A2A 通信和 expert 计算做 token-level 流水线重叠，MHA 和 gating 占单次迭代时间的 29.8%-36.1% 却完全串行，all-reduce 在反向传播结束后集中执行。FlowMoE 证明这些"被忽略"的任务可以通过统一流水线调度实现全重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 任务定义（每层 l，每子块 r ∈ [1, R]，R 通常=2）:
// AT_r(l): MHA + gating 计算子任务
// D_r(l):  Dispatch A2A 通信子任务
// E_r(l):  Expert 计算子任务
// C_r(l):  Combine A2A 通信子任务
// AR(l):   All-reduce 梯度（切成 S_p 大小的 chunk）

// 前向调度顺序（计算与 A2A 交错）:
// AT_1→AT_2→...→AT_R→E_1→...→E_R→AT_1(l+1)→...
// D_1→D_2→...→D_R→C_1→...→C_R→D_1(l+1)→...

// 反向调度顺序:
// E_R(l+1)→...→AT_1(l+1)→E_R(l)→...→E_1(l)→AT_R(l)→...→AT_1(l)
// C_R(l+1)→...→D_1(l+1)→C_R(l)→...→C_1(l)→D_R(l)→...→D_1(l)
// AR chunk 在 A2A 任务间隙插入（优先级: A2A > AR）

// 消融实验 (M=8192, H=8192, 16 GPU):
// Pipe-MoE only (Tutel):  1.46× vs vanillaEP
// + Pipe-AT (MHA+gating): 1.61× → MHA+gating 贡献 +10.3%
// + Pipe-AR (w/ BO):      1.82× → AR 贡献 +24.6%
// Full FlowMoE:           2.05×
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 PyTorch + Tutel 实现，三个队列（DataQueue/A2AQueue/ARQueue）+ 后台通信池管理器
- 类继承扩展 Tutel 的 MoE 层，修改 token 切分和 CUDA stream 调度
- R=2 保持与 Tutel/ScheMoE 相同的流水线度，通过扩展调度范围而非增大 R 来提升重叠
- 开源: https://github.com/ZJU-CNLAB/FlowMoE
- 在 675 个自定义 MoE 层配置和 4 个真实 MoE 模型上验证，所有有效配置下均快于 ScheMoE
- 硬件: 16× RTX 3090 (100Gb/s) + 8× RTX 2080Ti (10Gb/s)

涉及论文标题：
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training
