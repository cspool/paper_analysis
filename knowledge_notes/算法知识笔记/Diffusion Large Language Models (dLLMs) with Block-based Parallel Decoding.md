## Diffusion Large Language Models (dLLMs) with Block-based Parallel Decoding

术语解释
Diffusion Large Language Models (dLLMs) 是一类原生并行解码的 LLM，使用 block-based parallel decoding 替代传统自回归（AR）逐 token 生成。通过同时处理多个 masked tokens，在单次前向传播中生成整个 token block，平衡生成质量和吞吐。

术语是什么？
dLLM 的核心思想是 block diffusion：将输入文本分割为固定大小的 blocks（如 16/32/64 tokens），每个 block 内的 tokens 从 [MASK] 状态开始，经多步 denoising 逐步 unmask。与 AR decoding（token-by-token）和 speculative decoding（draft+verify）不同，dLLM 无需 verification step 即可直接并行生成。

关键架构组件：
- **Block-based generation**: 输入=N tokens，输出=N tokens，一次 forward pass 生成整个 block
- **Mask prediction**: 每个 token position 预测 masked token 的概率分布
- **Confidence-based sampling**: 通过 confidence threshold 决定 token 是否 finalize
- **KV Cache with bidirectional context**: 使用近似 KV cache 策略（如 Fast-dLLM）支持 block 内双向注意力

代表模型：LLaDA (Zhu et al., 2025), LLaDA2.0 (Bie et al., 2025), Dream (Ye et al., 2025), Block Diffusion (Arriola et al., 2025)

从算法pipeline角度拆解术语：
```
# Block-based Parallel Decoding Pipeline
for each block position p in [0, seq_len, block_size]:
    # Step 1: Initialize block
    block_tokens = [MASK] * block_size  # 全 mask 初始状态
    prefix_context = X[:p]               # 已解码的前缀
    
    # Step 2: Iterative denoising
    for step in range(max_steps):
        input_tokens = concat(prefix_context, block_tokens)
        logits = dLLM.forward(input_tokens)[-block_size:]  # 仅取 block 位置
        probs = softmax(logits)
        
        # Confidence-based finalization
        confidence = max(probs, dim=-1)
        for i where confidence[i] > threshold:
            block_tokens[i] = argmax(probs[i])  # finalize
        
        if all(finalized):
            break
    
    # Step 3: Append decoded block to context
    X = concat(X, block_tokens)
```
关键张量维度：输入 [batch, seq_len+block_size, d_model]，输出 [batch, block_size, vocab_size]。

与 MoE 结合的特殊问题：每 token 独立路由 → unique expert load 随 block_size N 线性增长（"expert explosion"）。

术语一般如何实现？如何使用？
- dInfer 框架（Ma et al., 2025）：专门为 dLLM 设计的推理框架
- Fast-dLLM (Wu et al., 2025)：训练无关的 KV cache + 并行解码加速
- SGLang 已规划 dLLM 支持（roadmap 2026 S1）
- 典型配置：block_size=32（16 prefix + 16 suffix cache tokens），confidence threshold=0.9
- 主要 trade-off：block size 增大提升吞吐 but 降低 generation quality；较小 block 保持 AR 级别质量但速度增益有限

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
