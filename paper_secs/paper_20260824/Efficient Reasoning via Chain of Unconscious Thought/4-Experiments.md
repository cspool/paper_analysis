# 4 Experiments

This section evaluates the effectiveness of CoUT in reducing tokens while maintaining performance.

### 4.1 Experimental Setup

We compare CoUT with baseline methods, including CoT [\(Wei et al.,](#page-9-2) [2022\)](#page-9-2), CoD [\(Xu et al.,](#page-10-0) [2025\)](#page-10-0), CCoT [\(Renze and Guven,](#page-9-19) [2024\)](#page-9-19), and TALE-EP[\(Han et al.,](#page-8-9) [2024\)](#page-8-9), on several reasoning tasks. For TALE-EP, the recorded token count represents the sum of tokens from the model's two-round responses (first for estimating required tokens, then for generating the constrained answer). In this study, all experiments are conducted under a zeroshot learning setup, meaning that the models do not receive any training or fine-tuning on the specific datasets used for evaluation.

CoT is a prompting methodology that encourages language models to decompose complex reasoning tasks into a series of intermediate steps. By instructing the model to "think step by step,"

CoD maintains the step-by-step reasoning of CoT but constrains each step to a maximum of five words. This brevity reduces token usage and response time while preserving reasoning accuracy.

CCoT is a variant of CoT that limits the reasoning process to 45 tokens, promoting concise responses. The final answer remains unconstrained, balancing brevity with completeness.

TALE-EP is a two-step strategy designed to optimize token usage. First, the model estimates the number of tokens required to answer a question. Then, it generates a response within this predicted budget, effectively reducing token costs while maintaining accuracy.

CoUT boosts reasoning efficiency by internalizing logic in hidden layers, avoiding explicit steps. It employs token-efficient strategies to cut costs while retaining reasoning quality.

Our experiments evaluate the performance of four leading large language models on four math reasoning datasets: GPT-4o (gpt-4o-2024-0806) from OpenAI, Claude 3.5 Sonnet (claude3-5 sonnet-20240620) from Anthropic, O3-mini, and Qwen (QwQ-32B), representing a mix of top-tier

proprietary models and strong open-source alternatives. We evaluate CoUT on 4 math datasets:

- GSM8K [\(Cobbe et al.,](#page-8-11) [2021\)](#page-8-11): A dataset of grade-school-level word problems covering arithmetic, algebra, and logic.
- SVAMP [\(Patel et al.,](#page-9-20) [2021\)](#page-9-20): A dataset of multi-step word problems that require reasoning over multiple pieces of information.
- MathQA [\(Jie et al.,](#page-8-12) [2024\)](#page-8-12): This dataset tests the models' ability to solve math problems combining arithmetic and algebra.
- Aqua [\(Huang et al.,](#page-8-13) [2022\)](#page-8-13): A dataset containing multi-step arithmetic reasoning tasks designed to evaluate reasoning abilities.

