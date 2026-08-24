# Chain-of-Draft

Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.

Chain-of-Draft (CoD) [\(Xu et al.,](#page-10-0) [2025\)](#page-10-0) constrains each reasoning step to five words. However, this prompting strategy has limited adaptability, as the complexity of reasoning steps inherently depends on the task. Moreover, it may not effectively reduce token costs, as the number of steps could increase to compensate for brevity in each step.

### Concise Chain-of-Thought

Let's think step by step and limit the answer length to 45 words.

Concise Chain-of-Thought (CCoT) [\(Nayab](#page-9-5) [et al.,](#page-9-5) [2024\)](#page-9-5) constrains reasoning to 45 tokens while leaving answers unconstrained. Its limitations are similar to CoD, such as limited adaptability to varying task complexity. Moreover, it may fail to significantly reduce overall token usage when tasks require longer answers or additional reasoning steps to compensate for the strict constraint.

#### Token-Budget-Aware LLM Reasoning

Let's think step by step and use less than budget tokens.

Token-Budget-Aware Prompt (TALE-EP) [\(Han et al.,](#page-8-9) [2024\)](#page-8-9) predicts token budgets before generating answers, reducing costs through planned allocation. However, this two-stage method may cost more tokens as, the model's initial token prediction can sometimes be several times more than what is needed for the final answer.

The common limitation across these methods is their reliance on externalized reasoning—converting complex cognitive processes into sequences of tokens. Whether through full verbalization (CoT), word-limited steps (CoD), tokenconstrained reasoning (CCoT), or budget-aware generation (TALE-EP), all these approaches mandate that reasoning steps appear in the output.

# Chain-of-Draft

Answer:

- 1. Formula: CI SI = Rs.72
- 2. Principal = Rs.5000, Time = 2 years
- 3. CI formula: P(1 + r/100)<sup>n</sup>
- 4. SI formula: P × r × n/100
- 5. Calculate: (1 + r/100)<sup>2</sup> − 1 = 72/5000
- 6. Simplify: r <sup>2</sup>/10000 = 72/5000
- 7. Solve: r <sup>2</sup> = 144
- 8. Result: r = 12

Our method fundamentally changes how reasoning is performed with language models. Rather than forcing every computational step to be externalized in text, CoUT allows the model to leverage its internal neural mechanisms for straightforward calculations and reasoning steps. The model only outputs information when it's necessary to clarify its thought process or when explicitly requested.

#### Chain of Unconscious Thought

Answer:

Rate = 12%

# Chain-of-Draft

Answer:

Initial population: 8000. Annual decrease: 20%.

Population after 1 year: 8000 \* 0.8. Population after 2 years: 6400 \* 0.8. Population after 3 years: 5120 \* 0.8.

Final population: 4096.

#### B

#### Chain of Unconscious Thought

Answer:

Population after 3 years = 8000 ∗ (0.8)<sup>3</sup> = 4096.

#### B