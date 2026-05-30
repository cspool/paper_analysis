## TEMP-LORA (Temporary Low-Rank Adaptation / 临时低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TEMP-LORA 是 SlowFast-VGen 提出的推理时快速学习方法，将情节记忆（episodic memory）存储在 LoRA 低秩参数中。原始 TEMP-LORA（Wang et al. 2024b）为长文本生成设计，逐步生成新文本 chunk 并将生成的 chunk 作为 ground-truth 训练模型。SlowFast-VGen 将其改进用于长视频生成：每轮推理迭代 i，生成新视频 chunk Y_i 后，将输入 latent X_i 和输出 latent Y_i 拼接成时序连续体 X_i' = X_i ⊕ Y_i，对全序列添加噪声得 z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε，然后用去噪 UNet 在全序列上训练 TEMP-LORA 参数 Θ_i（不含文本条件），使 LoRA 参数存储整个生成轨迹的记忆。核心改进：丢弃原始 TEMP-LORA 的 input→output 格式，对拼接全序列加噪去噪，强调记忆整个轨迹而非关注即时转换。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SlowFast-VGen 推理时 Fast Learning (TEMP-LORA)
# 输入: 冻结慢学习权重 Φ, TEMP-LORA 参数 Θ_0, 快速学习率 α
# 输出: 长视频序列 Y

X_0 = VAE_ENCODE(X_0)           # 首帧编码到 latent space
Y = X_0

for i in 0..I-1:
    if i != 0:
        X_i = Y_{i-1}            # 上一轮输出作为当前输入
    C_i = User_Input(i)           # 当前 action 文本条件
    Y_i = (Φ + Θ_i)(X_i, C_i)    # 生成当前 chunk（慢学习权重 + 快学习 LoRA）
    Y = Y ⊕ Y_i                   # 拼接到最终序列

    # 训练 TEMP-LORA 存储情节记忆
    X_i' = X_i ⊕ Y_i              # 拼接输入输出 latent
    z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε  # 全序列加噪
    loss_Θ = ||ε - ε_{Φ+Θ_i}(z_t^{i'}, t)||²    # 无文本条件，全序列去噪
    Θ_{i+1} = Θ_i - α·∇_Θ loss_Θ                # 更新 LoRA 参数

Y = VAE_DECODE(Y)                # latent 解码回像素空间
```

Annotations:
- Φ: 预训练 UNet 权重（slow learning weights），推理时冻结
- Θ_i: TEMP-LORA 低秩矩阵（fast learning weights），W' = Φ + Θ，rank=32
- X_i': 拼接的时序连续体，维度为 2·(fp+fg) 帧的 latent
- z_t^{i'}: 全序列加噪 latent（注意：不保留干净条件帧，与 slow learning 的 masked conditioning 不同）
- 训练不含文本条件 c，专注于轨迹记忆而非条件生成
- 推理 overhead: +6.8% 时延 (12.93s→13.81s)，+3.7% 显存 (9579MB→9931MB)
- 遵循 local learning rule: ΔW 仅依赖当前迭代的局部 input-output 对

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TEMP-LORA 在 SlowFast-VGen 中的实现：基于 ModelScopeT2V 的 3D UNet，在 UNet 的 target_modules 上应用 LoRA（rank=32），使用 Adam 优化器，fast learning rate=1e-4。每轮推理迭代在单张 V100 GPU 上执行 TEMP-LORA 训练（一次前向+反向传播，仅更新 LoRA 参数）。Slow-Fast Learning Loop 扩展用法：内层 fast learning 循环在每个 episode 上积累 TEMP-LORA 参数 Θ^e；外层 slow learning 循环固定 Θ^e，利用多 episode 数据更新核心权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。原始 TEMP-LORA（为文本设计）使用 input→output 格式，SlowFast-VGen 的消融实验（Table 4）显示原版 TEMP-LORA 的 SCuts=0.55 劣于改进版的 0.37，因为原版关注即时转换而非全轨迹一致性。消融还显示"无 local learning rule"变体（采样全序列训练）SCuts=0.36 表现也不错但会导致后期帧过平滑。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation
