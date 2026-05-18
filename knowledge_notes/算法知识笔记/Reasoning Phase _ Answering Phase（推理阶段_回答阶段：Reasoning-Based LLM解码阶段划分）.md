## Reasoning Phase / Answering Phase（推理阶段/回答阶段：Reasoning-Based LLM解码阶段划分）

术语是什么？

Reasoning phase和answering phase是reasoning-based LLMs（如DeepSeek-R1、OpenAI o3）解码阶段内部的两个功能上不同的子阶段。Reasoning phase生成hidden Chain-of-Thought (CoT) reasoning tokens（对用户不可见），answering phase生成user-visible answering tokens。这与conventional LLMs形成根本区别：conventional LLMs的decode阶段仅生成user-visible tokens，而reasoning-based LLMs将decode阶段内部划分出reasoning和answering两个语义不同的phase。

在PASCAL论文中，reasoning phase被定义为包含prefill stage和reasoning tokens的解码，因为两者都贡献于"第一个user-visible token出现前"的延迟。
具体的伪代码：
```
// Reasoning-based LLM decode phase
Input: prompt, model
// Phase 1: Prefill
KV_cache = Prefill(model, prompt)
first_token = Decode(KV_cache)  // could be reasoning or answering
while not end_of_sequence:
    if first_token == <\think> token:  // phase boundary detected
        // Reasoning phase ends, Answering phase begins
        break_from_reasoning = true
    if not break_from_reasoning:
        // Reasoning phase: generate hidden CoT tokens
        token = Decode(model, KV_cache)
        reasoning_tokens.append(token)
    else:
        // Answering phase: generate user-visible tokens
        token = Decode(model, KV_cache)
        answering_tokens.append(token)
TFAT = time from <\think> to first answering token  // TTFAT
TTFT = prefill_time + reasoning_time + TTFAT        // Redefined TTFT
```

从算法pipeline角度拆解术语：

Reasoning-based LLM推理pipeline（以DeepSeek-R1-Distill-Qwen-32B为例）：
1. **Prefill stage**：处理input prompt，生成初始KV cache。这步与conventional LLM一致。
2. **Reasoning phase（decoding）**：模型auto-regressively生成reasoning tokens（如"我们需要计算..."、"首先..."等中间推理步骤）。这些tokens对用户隐藏，但KV cache正常累积。Reasoning token count可从128到2048+不等（取决于问题复杂度）。PASCAL的motivation实验显示：reasoning phase latency受blocking/preemption严重影响——FCFS下128 reasoning tokens请求latency可达oracle的5.14×，RR下2048 reasoning tokens可达1.75×。
3. **Phase transition**：模型生成特殊token（如DeepSeek-R1的`<\think>`）表示reasoning结束。
4. **Answering phase（decoding）**：模型开始生成user-visible answering tokens。这些tokens stream到用户。Answering phase是threshold-sensitive：只需TTFAT ≤ 0.25s + TPOT ≤ 100ms即可满足QoE SLO。即使因preemption导致fragmented execution，SLO仍可保持（Figure 5b显示RR SLO attainment接近oracle）。

术语一般如何实现？如何使用？

Reasoning phase/answering phase的区分是模型训练时通过强化学习（如DeepSeek-R1的GRPO）内化到模型行为中的，不是serving系统手动划分的。模型自动在推理时决定何时进行reasoning、何时输出answer。Serving系统（如PASCAL）通过检测phase boundary token（如`<\think>`）来识别当前phase，并根据phase特性应用不同调度策略。

关键观察（来自PASCAL characterization）：
- Reasoning phase：interruption-sensitive → 需要最小化blocking/preemption。FCFS的HoL blocking显著延长reasoning latency，RR的frequent preemption对long reasoning造成1.75× overhead。
- Answering phase：threshold-sensitive → 可容忍moderate preemption。RR在answering phase保持高SLO attainment（接近oracle），因为只要TTFAT低+TPOT达标即可。
- 两个phase的asymmetric sensitivity是PASCAL phase-aware scheduling的核心motivation。

在conventional LLMs中不存在此区分——decode阶段的所有tokens对用户可见，TTFT=prefill latency，TPOT=decode token generation rate。PASCAL的工作表明这一区分对serving系统的设计有重大影响。

涉及论文标题：
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

