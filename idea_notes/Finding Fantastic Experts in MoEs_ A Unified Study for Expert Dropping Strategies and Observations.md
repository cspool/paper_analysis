## Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

- baseline方法是什么？
  - **One-shot Expert Pruning**（如 Lu et al., 2024; Muzio et al., 2024; He et al., 2024）：对 MoE 模型一次性估算所有 expert 的重要性，在每层丢弃特定比例的专家。流程为：使用单一准则（如 expert usage frequency 或 token reconstruction loss）一次估算所有 expert 重要性 → 按分数排序 → 每层丢弃最不重要的 r 个 expert → 从 router gating 矩阵 W_G^{d×n} 中删除对应列 → W_G^{d×(n-r)}。核心缺陷：(1) 一次性估算无法反映丢弃某些 expert 后其余 expert 重要性的变化，选出的子网络质量差；(2) 丢弃后 router 矩阵直接移除了对应 expert 入口，导致负载分布严重偏斜（某些 retained expert 被过度路由），子网络处于 sub-optimal 状态；(3) 随 sparsity 增加，性能急剧下降（≥25% sparsity 时 zero-shot MMLU 已降至 random guess 水平）。
  - **LLM Weight Pruning**（Wanda, Magnitude, Random）：对 FFN 权重矩阵做 2:4 structured sparsity（NVIDIA Ampere 硬件支持），移除不重要的权重连接，但不改变模型架构。缺陷是无法利用 MoE 架构特有的 expert-level redundancy 和 conditional computation 特性。
  - **全栈执行例子（One-shot Expert Pruning on Mixtral-8×7B Base）**：
    - **模型推理/训练算法层**：加载 HuggingFace Mixtral-8×7B → C4 calibration set 256 samples → forward pass 收集 expert usage frequency → 按层排序选取每层 top-r 最不常用 expert → 修改 model.config (num_experts 从 8→(8-r)) → 删除对应 router 列和 expert weights → 直接 zero-shot 评估 MMLU/ARC/WinoGrande。任务性能在 25% sparsity 时已崩溃。
    - **系统框架层**：HuggingFace Transformers + PyTorch，标准训练/推理脚本。论文未明确说明 Serving 框架修改。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM、PyTorch autograd。
    - **硬件架构层**：8×NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MC-Suite + MoE Lottery Subnetworks 方法**：
    1. **MC-Suite（16 种多维准则）**：从四个维度（Weight/Inference/Activation/Gradient）系统化评估 expert 重要性，涵盖 expert & router weight dynamics、expert inference behavior dynamics、intermediate activation properties、expert gradient properties。从 16 种准则中实验验证最优者：Min-EAN（最小激活范数）和 Min-EGE（最小梯度熵），因为它们同时考虑了 input tokens 和 weight parameters，比单纯基于 expert usage 的准则更精确。
    2. **Iterative Pruning（替代 One-shot）**：将 s% 的 expert 丢弃分成 k 轮，每轮丢 s/k%（如 50%=4×12.5%）。每轮丢弃后 re-estimate 剩余 expert 的 MC-Suite 准则值。这解决了 one-shot 无法反映 expert 间依赖关系变化的缺陷——第一轮丢弃后，剩余 expert 的重要性发生变化，re-estimation 捕捉到了这一点。
    3. **Task-Agnostic Budget Finetuning（MoE Lottery 核心）**：每轮 pruning 后使用 ~0.2M training tokens 做 next-token prediction finetuning（C4 calibration data），progressive schedule 逐轮翻倍（0.2M→0.4M→0.8M→1.6M），总 budget ~1M tokens 即饱和。Finetuning 的作用：(i) 重新调整 router weights 使负载分布重新均衡（Figure 6: 红色虚线→绿色实线大幅改善）；(ii) 恢复因 expert 丢弃造成的 abrupt performance drop。
    4. **Instruction-Following Recovery**：实验验证 expert dropping 主要损害的是 instruction-following 能力（非 pretraining knowledge/reasoning）。通过 k-shot examples 或 SFT（supervised fine-tuning with instruction dataset）可显著恢复下游性能。
  - **对应解决 Baseline 缺陷**：
    - One-shot 一次性估算 → Iterative 多轮 re-estimation 捕捉 expert 间依赖关系变化（Figures 5a/5b 直观展示了 one-shot 与 iterative 选出的 expert 高度不一致）。
    - One-shot 丢弃后 sub-optimal 状态 → Task-agnostic finetuning 重调 router weights + 负载均衡（Table 2: 75% sparsity 下 MoE Lottery pp=13.05 vs one-shot=30.59）。
    - 性能随 sparsity 急剧下降 → MoE Lottery @ 50% sparsity 仍保持 robust（MMLU 40.79 vs one-shot 18.91）；k-shot/SFT 可进一步恢复至接近 full-MoE 水平。
  - **全栈执行例子（MoE Lottery Subnetwork on Mixtral-8×7B Base, 50% sparsity, k=4 rounds, Min-EAN criterion）**：
    - **模型推理/训练算法层**：Round 1: C4 256 samples forward pass + forward hooks 收集每层 8 个 expert 的 output activations → 计算 Min-EAN = argmin ||A_Ep||₂ → 每层丢弃 1 个 expert → W_G^{4096×8}→W_G^{4096×7} → 0.2M tokens C4 next-token-prediction finetuning (AdamW, lr=1e-6, batch=8, cosine schedule) → Round 2: 重新 forward calib data 收集 7 个 expert 的 activations → 再丢 1 个 → 0.4M tokens finetuning → Round 3→Round 4 → 最终每层剩 4/8 experts, 50% expert sparsity → zero-shot 评估 5 个下游 benchmark → 对某些任务可补充 k-shot examples 或额外 SFT。关键张量流: router_score = softmax(H @ W_G^{4096×n}) → top-2 → expert_i(H) = SiLU(H @ W_gate) * (H @ W_up) @ W_down → ∑ G_i * expert_i。
    - **系统框架层**：HuggingFace Transformers (MixtralForCausalLM) + PyTorch DistributedDataParallel on 8×A100。论文未修改 Serving 框架，但 load balancing 的改善间接利于 GPU memory utilization。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM（FFN linear layers）+ PyTorch autograd（梯度计算）。
    - **硬件架构层**：8×NVIDIA A100 GPU。Memory 从 180GB→~99GB（50% sparsity），speedup 1.27×。论文未明确说明 GPU 架构级优化，speedup 来自 expert weight 减少后 kernel launch 次数和参数加载量减少。
