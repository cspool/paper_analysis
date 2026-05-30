## Diffusion Language Models (DLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Diffusion Language Models (DLMs) 是基于扩散过程生成文本的语言模型，区别于传统的自回归语言模型（AR）。核心思想来自连续域扩散模型（DDPM）：通过前向噪声（逐步掩盖/破坏 token）和逆向去噪（逐步恢复 token）实现文本生成。对于长度为 L 的序列，DLM 生成从全 [MASK] 序列开始，经过 T 个去噪步逐步解码 token，每步可选择多个位置并行解码。与 AR 的关键区别：(1) 非自回归解码——每步可并行解码任意数量 token，而非逐个左到右生成；(2) 双向注意力——每步所有 token 互相 attend，而非因果掩码；(3) 灵活解码顺序——不限于左到右，可任意顺序填充位置。

文本域中的扩散可分为两类：(a) **连续扩散**（continuous diffusion）：在连续词嵌入空间进行扩散和去噪，如 Diffusion-LM、SSD-LM；(b) **离散扩散**（discrete diffusion）：在离散 token 空间通过转移矩阵定义前向/逆向过程，如 D3PM、SEDD、MDLM。当前主流的 scaling 方向是掩码离散扩散（masked discrete diffusion），其中前向过程以概率 β_t 将 token 替换为 [MASK]，逆向过程学习恢复原始 token。LLaDA（8B）和 Dream（7B）是首个 scaling 到数十亿参数的掩码扩散 LLM，性能已可比肩同等规模的 AR LLM。

从算法pipeline角度拆解术语。

**掩码离散扩散语言模型的前向和逆向过程**：

```
# 前向过程：逐步掩盖 token
# x_0: 原始序列的 one-hot 编码，V 为词表大小
# U_t: 转移矩阵，定义从 token i 到 token j 在步 t 的概率
# 掩码扩散中使用 absorbing state [MASK]
# \bar{α}_t = ∏_{i=1}^t (1 - β_i)，β_i 为掩码概率

前向过程：
  q(x_{c(t)} | x_0) = Cat(x_{c(t)}; p = x_0 \bar{U}_t)
  其中：
    [\bar{U}_t]_{ij} = 
      1                          if i = j = [MASK]
      \bar{α}_t                  if i = j ≠ [MASK]
      1 - \bar{α}_t              if j = [MASK], i ≠ [MASK]

逆向过程（去噪）：
  p_θ(x_{c(t-1)} | x_{c(t)}) ≈ q(x_{c(t-1)} | x_{c(t)}, x_0)
  模型 θ 预测 x_0，再与 x_{c(t)} 联合确定 x_{c(t-1)}

# 采样循环
x_T = [MASK, MASK, ..., MASK]  # 初始全掩码序列
for t = T down to 1:
    # 1. 调用模型预测干净 token
    p_θ(x_{c(t-1)} | x_{c(t)})  → 预测每个位置的 x_0
    # 2. Remasking: 根据置信度/随机策略选择保留哪些 token
    #    高置信度或随机选中的 token 成为 "decoded" (unmasked)
    #    其余 token 继续保持 [MASK]
    for position i in 1..L:
        if confidence_i > threshold OR position in selected_set:
            x_{c(t-1)}^i = predicted_token_i  # 解码
        else:
            x_{c(t-1)}^i = [MASK]              # 保持掩码
```

**DLM 与 AR 的推理复杂度对比**：

| 属性 | AR (with KV-Cache) | DLM (no cache) |
|------|-------------------|----------------|
| 每步计算 | O(n) 个 token（仅新 token） | O(L) 个 token（全序列） |
| 注意力类型 | Causal（单向） | Bidirectional（双向） |
| 总步数 | L（每步 1 token） | T（去噪步，通常≈L） |
| 总复杂度 | O(L³)（累积 O(L²)） | O(L² × T) ≈ O(L³) |
| 实际速度 | 更快（每步仅算 1 token） | 更慢（每步算全部 L token） |

术语一般如何实现？如何使用？

主流掩码扩散语言模型基于 Transformer Decoder 架构（如 LLaDA 基于 LLaMA 架构），主要修改：(1) 去掉 causal mask，使用 bidirectional mask；(2) 训练时使用掩码预测损失（masked prediction loss）而非 next-token prediction；(3) 推理时使用迭代采样+remasking 流程。LLaDA-8B 和 Dream-7B 均通过 HuggingFace Transformers 实现，使用标准 Transformer blocks + FlashAttention。采样策略影响生成质量：置信度 remasking（keep top-k confidence）常优于随机 remasking。DLM 的去噪步数 T 通常设置为序列长度 L 的 1-2×。可通过减少去噪步数（Few-Steps/Half-Steps）加速，但以生成质量为代价。最新加速方法如 dKV-Cache 通过引入 KV 缓存进一步缩小与 AR 的速度差距。

涉及论文标题：
- dKV-Cache: The Cache for Diffusion Language Models

---
