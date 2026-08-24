# **Your Output:**

A prefix of "Thinking Process", with Ground Truth at the end.

Figure 5: Our prompt for extraction of answer prefix.

- QwQ-32B. A medium-sized Qwen reasoning variant refined with SFT + RL; provides explicit <think> traces, 131 K context and DeepSeek-R1–level accuracy on hard evaluations.
- Llama-3.3-Nemotrom-Super-49B-V1. NVIDIA's NAS-pruned 49 B derivative of Llama-3.3-70B, post-trained for reasoning, RAG and tool calling; couples 128 K context with single-H100 deployment efficiency for cost-sensitive production.
- Deepseek-R1-Distill-Qwen-7B. A 7 B dense checkpoint distilled from DeepSeek-R1 onto the Qwen2.5 backbone, pushing small-model MATH-500 pass1 beyond 92 % and surpassing o1-mini on several reasoning suites while remaining laptop-friendly.
- Deepseek-R1-Distill-Qwen-1.5B. An ultra-compact 1.5 B model distilled from R1 that preserves chain-of-thought and achieves 83.9 % pass1 on MATH-500, bringing competitive analytical power to edge and mobile deployments.
- Qwen-2.5-3B-Instruct. A 3.09 B instruction-tuned model with 128 K context, strengthened coding/math skills and multilingual support, designed as a lightweight yet controllable chat foundation for downstream tasks.

### C.2. Dataset

We benchmark on the AIME25 [\(International Conference on Artificial Intelligence in Medicine\)](#page-9-18), MATH500 [\(Lightman](#page-9-19) [et al.,](#page-9-19) [2023\)](#page-9-19), GSM8K [\(Cobbe et al.,](#page-9-20) [2021\)](#page-9-20), OlympiadBench [\(Sun et al.,](#page-8-3) [2025\)](#page-8-3), AMC [\(Mathematical Association of](#page-9-21) [America\)](#page-9-21), GPQA Diamond [\(Rein et al.,](#page-9-22) [2024\)](#page-9-22) and LiveCodeBench [\(Jain et al.,](#page-9-23) [2024\)](#page-9-23) benchmarks in our paper. We

<span id="page-12-0"></span>> **[图片提取文字 (无描述)]:**
> Deploy Navigation **JSON Entry Review Interface** Displaying Entry: 1 of 2851 Previous. Next. Entry 1 Question: A line is parameterized by  $\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ -2 \end{pmatrix} + t \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ . A second line is parameterized by  $\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} -8 \\ 12 \end{pmatrix} + u \begin{pmatrix} 1 \\ 3 \end{pmatrix}$ . If  $\theta$  is the acute angle formed by the two lines, then find  $\cos \theta$ . Solution: Reasoning Prefix: Okay, so I have this problem where I need to find the cosine of the acute angle between two lines. Both lines are given in parametric form, which I remember is like starting at a point and then moving in a certain direction. The formula for the angle between two lines probably has something to do with their direction vectors, right? Let me think. First, let me write down the parameterizations to make it clear. The first line is:  $\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ -2 \end{pmatrix} + t \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ So, the direction vector for the first line is  $\binom{3}{4}$ . I'll call this vector v. The second line is:  $\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} -8 \\ 12 \end{pmatrix} + u \begin{pmatrix} 1 \\ 3 \end{pmatrix}$ So, the direction vector for the second line is  $\binom{1}{2}$ . I'll call this vector w. I remember that the angle 0 between two vectors can be found using the dot product formula: But since we're dealing with lines, not vectors from the origin, the direction vectors are what matter here. So, I can use the direction vectors v and w to compute this. First, I need to compute the dot product of v and w. The dot product is calculated by multiplying the corresponding components and then adding them up. So,  $v \cdot w = (3)(1) + (4)(3) = 3 + 12 = 15$ . Next, I need to find the magnitudes of v and w. Starting with v:  $||\mathbf{v}|| = \sqrt{22 + 42} = \sqrt{9 + 16} = \sqrt{95} = 5$
![](_page_12_Figure_1.jpeg)

Figure 6: The annotation tool to evaluate the LC-Extratcor.

### introduce them as follows:

- AIME25. A benchmark with 30 questions distilled from twenty-five years of *American Invitational Mathematics Examination* papers. Each item is a three-digit short-answer problem that probes upper-secondary algebra, geometry, combinatorics.
- MATH500. A 500-problem evaluation slice covering the full subject breadth of the original *MATH* competition corpus. Balanced across difficulty tiers and topics, it serves as a rigorous yardstick for advanced high-school and early undergraduate mathematical reasoning, without the runtime burden of the complete 12k-question set.
- GSM8K. The widely-adopted *Grade-School Math 8K* benchmark of 1,319 everyday word-problems. Requiring multistep arithmetic and commonsense, GSM8K remains the de-facto standard for assessing chain-of-thought quality on conversational math tasks.
- Olympiad. A curated collection of roughly 3 k national and international mathematics-olympiad problems. Predominantly proof-style or numeric-answer challenges, this benchmark gauges creative, non-routine reasoning at the highest preuniversity level.
- AMC. An aggregate of 83 from the *American Mathematics Competitions 10/12*. Spanning 2000–2024, it offers a longitudinal benchmark on foundational secondary-school math.
- GPQA Diamond. A benchmark with 198 graduate-level Google-proof multiple-choice questions requiring deep domain expertise and multi-step reasoning, curated by researchers from New York University, CohereAI, and Anthropic; evaluated in closed-book and open-book settings using accuracy as the metric.
- LiveCodeBench. A dynamic, contamination-free coding benchmark originally hosting 511 problems (release v2) collected from LeetCode, AtCoder, and CodeForces, designed by UC Berkeley, MIT, and Cornell researchers to holistically assess LLMs' code generation, execution, and test prediction capabilities using Pass@K.

### C.3. settings

We used a mixed-difficulty dataset, combining past AIME competition problems with the MATH dataset in an approximate 1:2 ratio to create 2500 training data. We use Trl[\(von Werra et al.,](#page-9-24) [2020\)](#page-9-24) framework to train models. Both models are trained with 4 \* A800-80G GPUs and the hyperparameters are presented in Table [4.](#page-13-0)

Table 4: Hyperparameters for LC-R1 training.

<span id="page-13-0"></span>

| Hyperparameter   | R1-Distill-Qwen-7B | R1-Distill-Qwen-1.5B |
|------------------|--------------------|----------------------|
| cutoff_len       | 8192               | 8192                 |
| batch_size       | 32                 | 32                   |
| learning_rate    | 3.0e-6             | 2.0e-6               |
| num_train_epochs | 1.0                | 1.0                  |
| α                | 1.0                | 1.0                  |
| β                | 0.04               | 0.04                 |
| γ                | 1.0                | 1.0                  |
| num_generations  | 6                  | 8                    |
| ϵ                | 0.2                | 0.2                  |

Baseline settings. We compare LC-R1 with 5 baseline—SFT, DPO, O1-Pruner, ThinkPrune, SFT+O1-Pruner. The last hybrid method shares same settings with each method, so we give out the settings of first four methods.

- SFT. We construct training dataset by extracting the valid thinking process to reconstruct a concise version of sequences sampled by themselves on MATH dataset. We set cutoff\_len=8192, epoch=1, learning\_rate = 3.0e-6, max\_samples = 5000.
- DPO. We construct preference training dataset by sampling 8 times on MATH dataset and choose the longest sample to be negative and shortest sample to be positive. We set cutoff\_len=8192, epoch=2, learning\_rate = 5e-6, max\_samples = 5000.
- O1-Pruner. We use the given python scripts to construct weight training dataset, with cutoff\_len=4096, epoch=2, learning\_rate = 2.0e-7, max\_samples = 10000.
- ThinkPrune-3K. We reproduce the training process on ThinkPrune-length3000 dataset, with size 2470. We set cutoff\_len=8192, epoch=2, learning\_rate = 2.0e-6, num\_generations=8, batch\_size=32.

