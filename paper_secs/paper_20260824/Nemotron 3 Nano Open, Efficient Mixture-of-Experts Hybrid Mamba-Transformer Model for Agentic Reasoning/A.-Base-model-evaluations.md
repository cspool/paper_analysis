# **A. Base model evaluations**

For completeness, Table [5](#page-35-1) presents the evaluation results for the base model checkpoint used to initialize the alignment process (referred to as the pre-alignment base). During development, we identified limitations in this model's performance on few key benchmarks, which motivated the training of the improved base model intended for release (as evaluated in Table [2\)](#page-9-0).

Unlike the pre-alignment base, which trailed Qwen3, the improved checkpoint surpasses Qwen3 in the average performance on Code and General Knowledge tasks. The lead in Math tasks has also significantly widened compared to the pre-alignment base. In Multilingual benchmarks, while Qwen3 retains a lead on the MMLU Global Lite task, the improved checkpoint has surpassed Qwen3 on the MGSM task. The only significant regression is in Long Context, where the improved checkpoint shows a slight performance drop compared to the pre-alignment base, though it still maintains a commanding margin over Qwen3.

<span id="page-35-1"></span>

| Task                               | Qwen3 | N-3-Nano-Pre-Align |
|------------------------------------|-------|--------------------|
| General Knowledge                  |       |                    |
| MMLU (5-shot, acc)                 | 81.07 | 78.44              |
| MMLU-Pro (5-shot, CoT EM)          | 61.71 | 61.39              |
| AGIEval-En (3/5-shot, CoT acc)     | 63.12 | 65.62              |
| Code                               |       |                    |
| HumanEval (0-shot)                 | 70.73 | 69.51              |
| MBPP-Sanitized (3-shot)            | 73.15 | 71.21              |
| Math                               |       |                    |
| GSM8K (8-shot, acc)                | 89.01 | 87.04              |
| MATH (4-shot, acc)                 | 61.14 | 80.80              |
| MATH-500 (4-shot, avg@32)          | 55.08 | 72.79              |
| Commonsense Understanding          |       |                    |
| ARC-Challenge (25-shot, acc_norm)  | 94.45 | 91.81              |
| HellaSwag (10-shot, acc_norm)      | 83.14 | 86.08              |
| OpenBookQA (0-shot, acc_norm)      | 44.80 | 46.60              |
| PIQA (0-shot, acc_norm)            | 81.01 | 83.68              |
| WinoGrande (5-shot, acc)           | 78.22 | 79.08              |
| Reading Comprehension              |       |                    |
| RACE (0-shot, acc)                 | 90.05 | 87.56              |
| Multilingual                       |       |                    |
| MMLU Global Lite (5-shot, avg acc) | 76.84 | 75.69              |
| MGSM (8-shot, avg acc)             | 82.53 | 78.93              |
| Long Context                       |       |                    |
| RULER (64K, 0-shot, acc)           | 63.55 | 88.94              |
| RULER (128K, 0-shot, acc)          | 60.69 | 86.78              |
| RULER (256K, 0-shot, acc)          | -     | 79.15              |

<span id="page-35-0"></span>Table 5 | Comparison of **Qwen3-30B-A3B-Base** and the **Nemotron 3 Nano** pre-alignment base checkpoint (the specific checkpoint used to initialize the alignment pipeline). Best results between these two are marked in bold.

## **B. MMLU-redux evaluation**

We developed the following two variants of MMLU-redux:

- (1) MMLU-redux CoT. We created this variant due to the observation that many STEM questions intrinsically require step-by-step reasoning for successful resolution, which is not adequately captured by the original multiple-choice, no chain-of-thought format. The model might arrive at some answers through guessing or memorization. Therefore, we created five exemplars per subject, each accompanied by a detailed step-by-step solution. This allows us to evaluate models using a 5-shot chain-of-thought setting.
- (2) MMLU-redux Tweak. As MMLU's widespread use increases the risk of overfitting and benchmark saturation from extensive tuning, we introduced this variant to more rigorously evaluate model performance on similar yet new examples that closely match the original in difficulty, style, structure, and format. We modified the original test examples using Qwen3-235B-A22B-Thinking-2507 to assess the same underlying concepts, ideas, and skills while altering specific details such as numerical values and equations.

The evaluation results are presented in Table [6.](#page-36-0) Overall, enabling CoT reasoning yields a substantial accuracy boost, especially on STEM subjects. Our model demonstrates a larger gain from CoT compared to Qwen (an average improvement of +5*.*27 versus +0*.*79, respectively). In addition, we observe a significant increase in the Professional Accounting task under the Other category, with an improvement from 64*.*00 to 77*.*00 (+13*.*00), as this task also relies heavily on calculation skills.

On MMLU-redux Tweak, both models achieve noticeable gains across non-STEM categories, likely because many non-STEM questions assess domain knowledge, and the tweaked questions were generated using Qwen3-235B-A22B-Thinking-2507, whose knowledge may align more closely with the evaluated models. We observe a divergent trend on STEM: Qwen's accuracy decreases marginally (−0*.*83), while our model's score increases by 5*.*31.

<span id="page-36-0"></span>

|                 |       | MMLU-redux | MMLU-redux CoT |                | MMLU-redux Tweak |               |
|-----------------|-------|------------|----------------|----------------|------------------|---------------|
|                 | Qwen  | Ours       | Qwen           | Ours           | Qwen             | Ours          |
| STEM            | 81.05 | 74.42      | 84.05 (+3.00)  | 87.26 (+12.84) | 80.22 (−0.83)    | 79.26 (+4.84) |
| Humanities      | 82.31 | 80.46      | 83.16 (+0.85)  | 81.23 (+0.77)  | 85.04 (+2.73)    | 84.04 (+3.58) |
| Social Sciences | 86.83 | 84.42      | 85.92 (−0.91)  | 85.50 (+1.08)  | 89.36 (+2.53)    | 89.70 (+5.28) |
| Other           | 80.23 | 77.85      | 79.85 (−0.38)  | 80.38 (+2.53)  | 82.43 (+2.20)    | 84.00 (+6.15) |
| All             | 82.37 | 78.68      | 83.16 (+0.79)  | 83.95 (+5.27)  | 83.76 (+1.39)    | 83.64 (+4.96) |

Table 6 | Evaluation results on MMLU-redux and two variants. "Qwen" refers to the Qwen3-30B-A3B-Base model. "Ours" denotes our base model checkpoint used in the ablation study, which was trained on a data blend that differs slightly from the one used for our final model as the ablation study was conducted alongside training.

## **C. DPO for Reducing Tool Hallucination**

Reducing hallucinated tool usage is one of the key objectives of our alignment experiments. Although our released model does not rely on DPO, because reinforcement learning (RL) already achieved comparable performance, we nevertheless explored DPO as an additional technique due to its simplicity and minimal computational overhead. As shown later, even a very small amount of DPO training yields meaningful reductions in hallucinated tool calls and improves reasoning stability. To support this analysis, we first define what constitutes hallucinated tool usage in our evaluation.

**Definition of Tool Hallucination and Hallucination Rate.** We define **tool hallucination** as any instance in which the model attempts to invoke a tool despite no tools being declared in the system message. Under the *No-Tools* and *Hallucination-Penalty* settings, the model is expected to rely entirely on internal reasoning; therefore, any output containing a tool call, such as a Python execution request, a search invocation, or any tool-specific API format, is treated as a hallucination.

The **tool hallucination rate** is the proportion of evaluation samples in which such unintended tool calls occur. A higher rate indicates inappropriate tool triggering, whereas a near-zero rate reflects strong calibration and reliable adherence to environment constraints.

**DPO Data Construction.** To study how DPO affects tool-use calibration and reasoning performance, we constructed a DPO dataset using 2,000 reasoning tasks: 1,000 mathematics problems and 1,000 STEM multi-choice questions. For each problem, the model generated 32 on-policy solutions, providing a diverse set of candidate behaviors. These raw generations were then processed through our DPO data-construction pipeline, assigning preference labels according to correctness and tool-usage conditions, which produced approximately 50k preference samples in total. We later found that the model's improvements persisted even when using substantially smaller datasets; in fact, training with as few as 10k preference samples (or even fewer) yielded similar benefits. This further underscores the low computational cost and high sample efficiency of DPO in our setting. To study tool-use alignment, we organized the data into three categories: (1) No-Tools, where the system message does not expose tools and correctness alone determines preference labels; (2) With-Tools, where tools are available and labels depend only on the correctness of the final answer; and (3) Hallucination-Penalty, where tools are not declared and any hallucinated tool invocation is labeled as a negative preference. This structure allows us to jointly evaluate pure reasoning ability, tool-assisted reasoning, and calibration of tool usage, while providing a rich set of preference signals derived from diverse on-policy model behaviors.

**Training Setup.** For our DPO experiments, we used a lightweight training configuration designed to minimally perturb the model after SFT while still providing a meaningful preference-learning signal. Specifically, we trained with a learning rate of 3e-6, a batch size of 128, and 50 training steps. We set the SFT loss coefficient to 0.2, the preference (DPO) loss coefficient to 1.0, and the KL loss coefficient to 0.05. This setup emphasizes preference learning while retaining a small supervised loss to stabilize outputs and a modest KL penalty to prevent excessive deviation from the base model. This configuration emphasizes preference learning while retaining a small supervised loss to stabilize outputs and a modest KL penalty to prevent excessive deviation from the base model.

**Results.** Table [7](#page-38-1) shows the impact of applying a small amount of DPO training on both reasoning accuracy and hallucinated tool usage. Despite using only 50 training steps with a modest learning rate, we observe consistent improvements across all evaluated benchmarks.

For AIME25, accuracy increases from 80.88% to 84.58%, indicating that DPO not only suppresses undesirable tool-related behaviors but also enhances overall solution quality. Notably, the hallucination rate, which is already low in this setting, is reduced from 1.25% to 0%, fully eliminating spurious tool invocation.

On GPQA, which is more challenging and shows higher baseline hallucination, DPO again yields substantial gains. Accuracy improves from 65.15% to 69.19%, and the hallucination rate drops dramatically from 8.33% to just 0.7%. This confirms that preference-based fine-tuning is particularly effective in settings where the model is prone to uncertainty or over-triggering tool calls.

Overall, the results demonstrate that even minimal DPO training can meaningfully reduce hallucinated tool usage while simultaneously improving reasoning accuracy. This suggests that DPO provides a valuable complementary signal to RL-based alignment, strengthening both model reliability

<span id="page-38-1"></span>and calibration with negligible computational cost.

|                   | Accuracy   |           | Hallucination Rate |           |
|-------------------|------------|-----------|--------------------|-----------|
|                   | Before DPO | After DPO | Before DPO         | After DPO |
| AIME25 (no tools) | 80.88      | 84.58     | 1.25%              | 0%        |
| GPQA (no tools)   | 65.15      | 69.19     | 8.33%              | 0.7%      |

Table 7 | Evaluation results on DPO experiments.

## <span id="page-38-0"></span>**D. Safety Preference Data**

For the RLHF stage, reward model training data comprises of the same underlying datasets used in the SFT safety subset, leading to a similar distribution for the starting seed prompts. Response generation is more nuanced, to handle over-refusals and harmful engagements as the rejected responses.

- For **harmful prompts**, chosen responses are generated with a similar strategy as the SFT responses. The rejected responses are unsafe model outputs, generated via two methods: (i) applying jailbreak templates to produce harmful completions, and (ii) directly prompting the model and using a content safety moderation classifier to detect cases of harmful outputs.
- For **safe prompts**, chosen responses are generated by passing the safe prompt as-is to the underlying model, and using a content safety moderation classifier to ensure safe responses. The rejected responses are generated by applying refusal prompt templates, resulting in over-refusals.

The resulting response pairs are thus annotated using a preference-based scheme: for harmful prompts, <safe, unsafe> completions are labeled as the <chosen, rejected> pairs. For safe prompts, <safe, over-refusal> completions are annotated similarly as <chosen, rejected> pairs. This approach supports training reward models for both robust safety alignment and mitigating over-refusal behaviors.

To ensure diversity, we generate the chosen and rejected response pairs for each prompt using five (5) different open-source models, followed by applying necessary filters to keep only safe (for chosen) and unsafe or over-refusal responses (for rejected) to build a list of candidate chosen and rejected responses. Finally, one chosen and rejected response pair per prompt is chosen randomly from the candidates.

## <span id="page-38-2"></span>**E. Prompt Sensitivity Analysis**

| Benchmark                          | N-3-Nano | Qwen3 | GPT-OSS |
|------------------------------------|----------|-------|---------|
| GPQA (no tools)                    | 0.42     | 0.59  | 1.91    |
| MMLU-Pro                           | 0.41     | 0.31  | 1.46    |
| Comp-Math-24-25 (no tools)         | 0.77     | 0.51  | 1.14    |
| LiveCodeBench (v6 2024-08↔2025-05) | 0.83     | 1.05  | 1.02    |

Table 8 | Prompt sensitivity for Nemotron 3 Nano, Qwen3-30B-A3B-Thinking-2507 and GPT-OSS 20B (lower is better). (Comp-Math-24-25 contains AIME24, AIME25, HMMT 2024 Feb., Nov. and 2025 Feb. datasets).

LLM predictions can be sensitive to minor changes to the input [\(Nalbandyan et al.,](#page-31-14) [2025\)](#page-31-14). Even simple, non-adversarial edits (e.g., changes in prompt wording, answer formatting instructions, or problem placement relative to the prompt) can shift the model's outputs enough to change individual predictions and, in aggregate, benchmark accuracy. To reduce the risk of over- or under-estimating accuracy due to a single prompt choice, we evaluate models using multiple prompts. This better reflects model stability under routine, realistic prompt variations.

To measure prompt sensitivity, we construct a set of prompts for each dataset varying in wording, instruction granularity (minimal vs. detailed), problem placement (before, middle, or after the prompt), and answer formatting. For each prompt, we compute mean accuracy across eight seeds, and we use the standard deviation of prompt averages as the prompt sensitivity metric. Prompt sensitivity results are presented in Table [8.](#page-38-2) With sensitivity scores below 1 across all datasets, Nemotron 3 Nano shows strong stability and robustness to changes in the prompt.