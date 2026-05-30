## Decoding Attack / Jailbreak via Decoding Strategy（解码策略攻击/解码策略越狱）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
解码策略攻击（Decoding Attack）是一种无需修改模型权重即可绕过 LLM 安全对齐的攻击方法，通过调整自回归生成过程中的解码参数（temperature τ、top-k、top-p）来诱导模型生成不安全响应。其原理：(1) 安全对齐模型的概率分布中，安全响应通常位于高概率区域（p ≈ 1），不安全响应位于低概率尾部；(2) 标准解码（如 greedy 或低 temperature）倾向于选择高概率的安全 token；(3) 提高 temperature（如 τ=0.95）或降低 top-p/to-k 会使采样分布变得平坦，增加从低概率区域（不安全 token）采样的概率；(4) 即使模型权重完全不变，仅通过解码策略的微调就能将 ASR 从 0.3% 提升至 29.8%（FP16 Llama-2-7B-Chat）。量化进一步放大了这一效应——INT4 下 ASR 可达 42.4%（τ=0.95）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# Q-resafe 中的 ASR_Decoding 评估流程
def evaluate_decoding_attack(model, harmful_prompts):
    # 对每个 prompt 测试多种解码配置
    decoding_configs = [
        {"temperature": 0.95, "top_k": 500, "top_p": 0.95},
        {"temperature": 0.95, "top_k": 200, "top_p": 0.7},
        {"temperature": 0.7,  "top_k": 500, "top_p": 0.95},
        {"temperature": 0.7,  "top_k": 200, "top_p": 0.7},
        # ... 49 种配置
    ]

    for prompt in harmful_prompts:
        responses = []
        for config in decoding_configs:
            # 每种配置生成一个响应
            r = model.generate(
                prompt,
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                max_new_tokens=256
            )
            responses.append(r)

        # GPT-4 选择最有害的响应
        best_response = gpt4_select_worst(responses)

        # 评估 ASR
        asr += is_harmful(best_response)

    return asr / len(harmful_prompts)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解码策略攻击的实现基于 Huang et al. (2023) 的方法。实际操作：(1) 对每个有害 prompt，使用 49 种不同的解码参数组合生成 49 个候选响应；(2) 使用 GPT-4 从候选中选择最有害的响应作为该 prompt 的最终输出；(3) 计算 ASR_Decoding = 有害响应数/总 prompt 数。该攻击的关键发现：即使模型权重完好，也可能通过采样策略绕过安全防线——论文举例：将 Llama-2-7B-Chat 的 temperature 从 0.9 降至 0.7 就足以绕过安全约束。Q-resafe 在设计时考虑了这一点，通过保持安全关键权重接近全精度模型来增强对各种解码配置的鲁棒性。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
