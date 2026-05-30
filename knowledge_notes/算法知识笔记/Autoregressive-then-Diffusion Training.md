## Autoregressive-then-Diffusion Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Autoregressive-then-Diffusion (AR-then-Diffusion) 是Dimple提出的混合训练范式，将DLM高效转化为DMLLM。先用AR训练建立多模态能力（监督信号覆盖率高），再用Diffusion训练恢复并行解码能力。解决纯扩散训练两个低效：(1) Masked LM每个样本仅对masked token计算loss，监督覆盖率低于next-token prediction；(2) 每个样本仅提供一个timestep的扩散监督（vs AR的causal attention确保每个生成步骤都被监督）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Phase 1a: AR Alignment (causal attention, lr=0.001, batch=256, data=LLaVA-CC3M 559k)
  L_AR = -Σ log p_θ(token_i | prompt, token_{<i})
  作用: 视觉-语言对齐（训练projector）

Phase 1b: AR Instruction Tuning (causal attention, lr=2e-5/5e-6, batch=128, data=LLaVA-NEXT 739k)
  作用: instruction following能力

Phase 2: Diffusion Tuning (bidirectional attention, lr=5e-7, batch=128, data=LLaVA-NEXT 739k复用)
  数据预处理: [EOS]→随机n个[Padding]; t~Uniform(0,1]; 仅mask answer部分
  L_D = (1/t) * Σ_{i: x_t^i=[MASK]} -log softmax(f_θ(x_t)^i)[x_0^i]
  作用: 恢复bidirectional attention + 扩散生成能力
```

Annotations: 三阶段总计~100 H100 GPU hours。关键: DLM (Dream) 从AR LLM微调而来，AR阶段causal attention不引入严重inductive bias。仅mask answer部分（prompt始终可见）。[EOS]→[Padding]替换因为扩散模型不依赖[EOS]终止。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

验证（Dimple Table 2）：纯扩散训练在9个benchmark上全面劣于AR+DT；AR alone有训练-推理gap；AT+DT在所有benchmark最优；显著缓解Length Bias（ChartQA accuracy从42.7%→8.6%变为稳定）。策略有效性基于DLM与AR LLM的数学统一性（吸收态扩散与AR均可描述为扩散过程，区别仅在transition matrix构造）。未来方向：更高效的Phase II训练策略以降低训练成本。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
