## Prefilling in Diffusion Models (DMLLM KV Cache Reuse)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Prefilling在扩散多模态大模型中是一种KV cache复用策略。在标准DMLLM推理中，由于使用bidirectional (full) attention mask，每步扩散迭代需对整个序列（prompt + answer）重新计算attention，复杂度为$O((L_{prompt} + L_{answer})^2)$。由于visual tokens和多轮对话使L_prompt可能非常长，这成为推理瓶颈。Prefilling策略：仅在首次扩散forward时完整计算prompt tokens的attention并缓存其Key/Value tensors；后续迭代直接复用缓存的prompt K/V，仅需计算answer部分的attention（复杂度降至$O(L_{answer}^2 + L_{answer} \cdot L_{prompt})$）。注意：在DMLLM中Prefilling理论上不是lossless——因为bidirectional attention下answer tokens会attend到prompt tokens，而缓存的K/V基于首次forward结果不变（不同于AR模型中prompt K/V本质上是静态的）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Prefilling在DMLLM推理中的运转流程：

```
Step 0 (首次forward — Prefill):
  输入: [Prompt tokens] + [[MASK] * L_answer]
  Attention计算:
    prompt × prompt: 完整计算 → 缓存 K_prompt, V_prompt
    answer × prompt: answer tokens作为query attend到K_prompt, V_prompt
    answer × answer: [MASK] tokens之间的attention（初始为空）
  输出: 预测分布 z_1 + 缓存的 K_prompt, V_prompt
  复杂度: O(L_prompt² + L_answer·L_prompt + L_answer²)

Step 1+ (后续forward — KV Cache Reuse):
  输入: [Prompt tokens] + [partially decoded answer]
  Attention计算:
    prompt × prompt: 跳过! 直接使用缓存的 K_prompt, V_prompt
    answer × prompt: answer作为query, K_prompt/V_prompt 从缓存读取
    answer × answer: 完整计算
  复杂度: O(L_answer·L_prompt + L_answer²)
  节省: O(L_prompt²) per step

性能影响（Dimple Table 3）:
  - 准确性: 平均性能下降仅0.8%（各benchmark差异小）
  - 加速比: batch=1 平均1.79×; batch=32 平均3.7-7.1×
  - response_length=4: batch=32 TPS加速3.4-3.8×
  - response_length=8: batch=32 TPS加速5.9-7.1×
```

Annotations: L_prompt在DMLLM中包括visual tokens（图像编码产生大量token）、system prompt、多轮对话历史，通常远大于L_answer；Prefilling的loss来源——首次forward时answer全为[MASK]，prompt K/V是基于[MASK] context计算的，但随着扩散进行answer tokens逐渐被填充，prompt tokens本应对新的answer context做出不同的attention响应（但缓存阻止了这种更新）；Dimple实验表明这种loss在实际中很小——说明visual perception和image token utilization在text generation过程中基本保持不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Prefilling实现：(1) 在首次forward pass中，通过attention module的KV cache接口保存prompt positions对应的K/V tensors；(2) 后续forward使用slicing/索引操作从缓存中取出K_prompt/V_prompt，与当前answer tokens的K/V拼接后计算attention；(3) 缓存管理——由于DMLLM的answer长度由response_length预定义，无需动态扩展缓存（与AR serving的动态KV cache不同）。适用于：(a) 长prompt场景（多图像、长文档）；(b) 多轮对话（prompt累积对话历史）；(c) high batch size场景（GPU利用率更高时加速效果更显著，因为attention计算占比更大）。限制：DMLLM特有——AR模型中的KV cache是精确无损的（causal attention保证已生成的token不需要attend到未来token），而DMLLM的bidirectional attention使Prefilling引入理论误差。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
