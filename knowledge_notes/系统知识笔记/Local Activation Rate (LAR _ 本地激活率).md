## Local Activation Rate (LAR / 本地激活率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Local Activation Rate (LAR) 是衡量 MoE Expert Parallelism 推理中 token-expert 通信效率的核心指标。定义为：LAR = (#tokens whose routed experts are on the local device) / (#total tokens processed)。在 EP 推理中，每个 GPU 执行其本地 experts 的 FFN 计算——如果 token 被路由到的 expert 恰好位于 token 当前所在的 GPU，则该计算为"本地激活"，无需跨设备 all-to-all 通信；否则需要 dispatch（发送 token 到 expert 所在 GPU）和 combine（将输出送回）。LAR 越高，all-to-all 通信量越小，推理效率越高。Baseline 轮询 expert placement 下 LAR≈1/E（E=EP degree），如 EP8 时约 12.5%-25%。Sem-MoE 通过 semantic-aware model-data co-scheduling 将 LAR 提升至 62%-68%。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

LAR 在 EP 推理系统中的作用：

```
通信模型：all-to-all 通信量 = α·k·B·S / G
其中：α = 1 - LAR (远程激活比例), k = top-k experts, 
      B = 全局 batch size, S = sequence length, G = device count

EP8 (8 GPUs), token 均匀分布在 8 devices, LAR=25% (baseline):
  α = 0.75 → 75% tokens 需要跨设备通信
  all-to-all volume per layer = 0.75 × kBS/8

Sem-MoE, LAR=62% (DeepSeek-V2-Lite):
  α = 0.38 → 38% tokens 需要跨设备通信
  all-to-all volume per layer = 0.38 × kBS/8
  → 通信量减少 49.3% vs baseline

Sem-MoE, LAR=68% (Qwen3-30B-A3B):
  α = 0.32 → 通信量减少 57.3% vs baseline
```

LAR 与 expert layer latency 的关系（论文 Figure 5a 的 mock routing 实验）：LAR 从 20% 逐步提升至 100%（理论上界，受 GPU memory 限制无法达到），all-to-all latency 线性下降。LAR 的提升来自两个因素：(1) offline model scheduling——expert placement 与 token affinity 对齐；(2) online data scheduling——token/request 路由到 expert 所在 device。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

测量方法：在 inference engine 中插桩 MoE layer 的 token routing 过程，统计 per-layer 的 (local_activations, total_activations) 对，取所有 layer 的 p50 作为系统的 LAR。LAR 被用于：(1) 评估 expert placement 策略质量（MoETuner, Sem-MoE 均以此为核心指标）；(2) 指导 offline co-clustering 的超参数选择（θ 控制 load balance vs LAR 的权衡）；(3) cross-dataset zero-shot transfer 评估——Sem-MoE 验证了跨数据集的 LAR 鲁棒性（ShareGPT→MMLU 仍保持 1.65× baseline LAR）。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
