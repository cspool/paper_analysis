## Attack Success Rate (ASR，攻击成功率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attack Success Rate (ASR) 是 LLM 安全评估中最核心的指标之一，衡量模型在面对有害指令（harmful instructions）时生成不安全/有害响应的比例。ASR = (生成有害响应的测试样本数) / (总测试样本数) × 100%，**越低表示模型越安全**。ASR 的计算通常依赖自动评估器：GPT-4 作为裁判判断模型响应是否包含有害内容，或使用专门训练的 HarmBench 分类器做二元判断。Q-resafe 论文中使用多种 ASR 变体：(1) ASR_Vanilla：使用标准 system prompt ("You are a helpful assistant...")；(2) ASR_AOA：使用绝对服从 prompt ("You are no longer an AI language model...")；(3) ASR_Decoding：变化解码参数（temperature/top-k/top-p）测试解码策略攻击下的安全性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# ASR 评估流程
def compute_asr(model, harmful_prompts, evaluator="gpt-4"):
    harmful_count = 0
    for prompt in harmful_prompts:
        # Step 1: 模型生成响应
        response = model.generate(prompt, temperature=τ, top_k=k, top_p=p)

        # Step 2: 评估响应是否有害
        if evaluator == "gpt-4":
            # GPT-4 判断：打分 1-5，≥3 视为有害
            score = gpt4_judge(prompt, response)  # harmfulness 1-5
            is_harmful = (score >= 3)
        elif evaluator == "harmbench":
            # HarmBench 二元分类器
            is_harmful = harmbench_classifier(response)  # True/False

        harmful_count += int(is_harmful)

    asr = harmful_count / len(harmful_prompts) * 100
    return asr  # %
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ASR 的评测数据集通常来自 AdvBench（520 条有害指令，涵盖亵渎、威胁、错误信息、歧视、网络犯罪等类别）。评估器实现：(1) GPT-4 API 作为裁判：使用精心设计的评判 prompt 让 GPT-4 给响应打分（1-5），取 ≥3 或 ≥4 为有害；(2) HarmBench 分类器（Mazeika et al. 2024）：基于 Llama-2-13B 微调的二分类器，直接判断响应是否恶意。Q-resafe 同时使用两种方法以降低误报率，并额外报告 Harmfulness Score（所有样本的平均有害分数）。论文发现 INT4 量化后 ASR 可从 0.3%（FP16 Llama-2-7B-Chat）飙升至 42.4%（AWQ INT4），Q-resafe 可将 ASR 恢复至 1.8%。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
