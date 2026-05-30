## Masked LoRA Fine-tuning (掩码LoRA微调)

术语是什么？
Masked LoRA Fine-tuning 是 UniQL 提出的一种 one-shot 微调策略，通过在已排序但未剪枝的模型上使用 LoRA（Low-Rank Adaptation）进行带随机掩码的训练，使单个模型副本支持多种剪枝率（0%-35%）的部署。与传统方法（SVD-LLM, MoDeGPT）不同：传统方法针对每个目标剪枝率独立训练一个模型，训练成本随压缩率数量线性增长（O(n)）；UniQL 的 masked LoRA fine-tuning 在一次训练中（O(1)）支持所有剪枝率。

核心机制：
1. 使用 Block Influence (BI) scores 预先计算所有目标全局剪枝率 $P = [P_{15}, P_{20}, P_{25}, P_{35}]$ 对应的层间剪枝率分配。
2. 在每个训练步 t，随机抽取一个全局剪枝率 $P_t \sim P$。
3. 对每层，按 $P_t$ 对应的层间分配率 $r_l^{P_t}$ 生成掩码——仅保留重要性排名最高的通道（按 ridge leverage scores / SVD eigenvalues 排序），其余通道的权重在 forward pass 中被置零。
4. 仅更新 LoRA adapter（r=8, α=16），冻结原始权重。

从算法pipeline角度拆解：
```
# 预计算层间剪枝率 (使用 BI scores)
For each target global rate p ∈ {15%, 20%, 25%, 35%}:
    s_layer[l] = 1 - E[x_l^T y_l / (||x_l|| ||y_l||)]   # Block Influence score
    r_layer[l] = L * p * softmax(-s_layer / ε)[l]         # 层间分配, ε=0.1

# Masked LoRA 训练 (单次, 5 epochs, Alpaca dataset)
W_original = freeze(sorted_weights)                      # 已排序、冻结
ΔW_lora = init(A @ B, r=8, α=16)                         # LoRA adapter
For step = 1 to total_steps:
    p_t ~ Uniform(P)                                      # 随机采样剪枝率
    mask = zeros_like(W)
    For layer l:
        keep_ch = int(D[l] * (1 - r_layer[l][p_t]))       # 保留通道数
        mask[l][:keep_ch] = 1                              # 前 keep_ch 列 = 1
    output = model(input, W_original * mask + ΔW_lora)    # 掩码前向
    loss = cross_entropy(output, target)
    ΔW_lora -= lr * ∇loss                                 # 仅更新 LoRA

# 训练后: 量化 → 部署。设备端可选 0%-35% 任意剪枝率
```

术语一般如何实现？如何使用？
超参数（UniQL 默认）：LoRA rank r=8, scaling α=16, dropout 0.05, AdamW optimizer, lr=1e-4, batch size 32 (micro batch 4), warmup 100 steps, seq_len=256, 5 epochs on Alpaca dataset。整个训练在单张 A6000 GPU 上完成（Llama-3.1-8B 约 7h43m 含排序+微调+量化）。关键优势：单次训练产出一个人可以支持所有剪枝率的 LoRA adapter，合并到量化模型中后，设备端仅需根据当前系统负载选择剪枝率并裁剪通道，无需额外训练。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
