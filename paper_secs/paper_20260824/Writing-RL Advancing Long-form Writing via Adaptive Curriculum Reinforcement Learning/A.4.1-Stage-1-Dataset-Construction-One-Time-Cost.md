# A.4.1 Stage 1: Dataset Construction (One-Time Cost)

The data selection stage involves generating competitive reference responses and scoring candidates. We utilized a mix of closed-source APIs and opensource models. As detailed in Table [10,](#page-15-1) the total API cost for generating high-quality reference responses was \$608.71, utilizing Claude-3.7-Sonnet, GPT-4o, Qwen-Plus, and DeepSeek-R1.

We note that the majority of this expense (\$375.52) was attributed to Claude-3.7-Sonnet. Future reproductions can significantly reduce costs by substituting this with more cost-effective alternatives identified in our survey. Additionally, inference with the policy model and local scoring required 34.51 and 41.25 GPU hours, respectively.

Crucially, these are one-time offline costs used to produce a reusable, high-quality dataset. Once constructed, this dataset eliminates these expenses for all subsequent training runs and community adaptations.

## A.4.2 Stage 2: RL Training

For the RL training process, costs are recurring per run. We employed Qwen-Plus as the external judge model for reward assignment. As shown in Table [11,](#page-16-1) a complete training run incurs approximately \$53.50 in API costs and requires 24 hours on 8 GPUs.

To ensure cost-efficiency without compromising quality, we conducted a comparative survey of potential judge models (Table [9\)](#page-15-0). Qwen-Plus was selected because it achieves a high agreement rate with human judges (0.75), comparable to DeepSeek-R1 (0.76), while offering significantly lower latency (1.16s first token latency) and reduced API costs compared to GPT-4o and Claude-3.7-Sonnet.

In conclusion, the primary monetary cost is frontloaded in the dataset construction phase. The perrun training cost remains moderate (\$53.50 + 24 compute hours), which we believe is a reasonable expenditure given the performance gains and the reusability of the constructed resources.

## <span id="page-14-1"></span>B Benchmarks and Evaluation Methods

In this section, we introduce the benchmarks and evaluation prompt templates used in our experiments.

LongBench-Write LongBench-Write [\(Bai et al.,](#page-9-1) [2024b\)](#page-9-1) is designed to evaluate the LLM long-form generation abilities, which focuses on generating coherent outputs exceeding 10000 words, addressing challenges in maintaining consistency and quality over extended text. Key evaluation metrics include coherence, fluency and topic relevance. In this work, we use the Quality Score as the metric. The evaluation prompt template used is as follows:

You are an expert in evaluating text quality. Please evaluate the quality of an AI assistant's response to a user's writing request. Be as strict as possible.

You need to evaluate across the following six dimensions, with scores ranging from 1 to 5. The scoring criteria from 5 to 1 for each dimension are as follows:

<span id="page-15-0"></span>

| Model               | Agreement | Cost (Input / Output, \$/M tokens) | First Token Latency (s) |
|---------------------|-----------|------------------------------------|-------------------------|
| Claude-3.7-Sonnet   | 0.82      | 3.0 / 15.0                         | 5.35                    |
| R1                  | 0.76      | –                                  | –                       |
| GPT-4o (2024-11-20) | 0.70      | 2.5 / 10.0                         | 2.19                    |
| Qwen-Plus           | 0.75      | 0.4 / 1.2                          | 1.16                    |

Table 9: Performance and cost comparison of different LLM judges.

<span id="page-15-1"></span>

| Model                       | Input Price<br>(\$/1M) | Output Price<br>(\$/1M) | Total Cost<br>(\$) | GPU Hours<br>(GPU) |
|-----------------------------|------------------------|-------------------------|--------------------|--------------------|
| Claude-3.7-Sonnet           | 3.00                   | 15.00                   | 375.52             | –                  |
| GPT-4o                      | 2.50                   | 10.00                   | 164.33             | –                  |
| Qwen-Plus                   | 0.40                   | 1.20                    | 27.74              | –                  |
| DeepSeek-R1                 | 0.57                   | 2.29                    | 31.12              | –                  |
| Qwen2.5-7B-Instruct (SFT)   | –                      | –                       | –                  | 34.51              |
| Qwen2.5-7B-Instruct (Judge) | –                      | –                       | –                  | 41.25              |
| Total                       |                        |                         | 608.71             | 75.76              |

Table 10: Cost breakdown for Stage 1: Data Selection.

- 1. Relevance: From content highly relevant and fully applicable to the user's request to completely irrelevant or inapplicable.
- 2. Accuracy: From content completely accurate with no factual errors or misleading information to content with numerous errors and highly misleading.
- 3. Coherence: From clear structure with smooth logical connections to disorganized structure with no coherence.
- 4. Clarity: From clear language, rich in detail, and easy to understand to confusing expression with minimal details.
- 5. Breadth and Depth: From both broad and deep content with a lot of information to seriously lacking breadth and depth with minimal information.
- 6. Reading Experience: From excellent reading experience, engaging and easy to understand content to very poor reading experience, boring and hard to understand content.

Please evaluate the quality of the following response to a user's request according to the above requirements.

<User Request>

\$INST\$

</User Request>

<Response>

\$RESPONSE\$

</Response>

Please evaluate the quality of the response. You must first provide a brief analysis of its quality, then give a comprehensive analysis with scores for each dimension. The output must strictly follow the JSON format: "Analysis": ..., "Relevance": ..., "Accuracy": ..., "Coherence": ..., "Clarity": ..., "Breadth and Depth": ..., "Reading Experience": .... You do not need to consider whether the response meets the user's length requirements in your evaluation. Ensure that only one integer between 1 and 5 is output for each dimension score.

WritingBench WritingBench [\(Wu et al.,](#page-11-2) [2025c\)](#page-11-2) is designed to evaluate the LLM long-form generation capabilities across six domains: creative, persuasive, informative, technical, business, and legal writing. It includes over 1200 tasks, further divided into 100 subdomains, with each task evaluated based on five dynamic criteria, such as coherence and relevance. We use the fine-tuned critic model in the original paper to evaluate the responses. The evaluation prompt template used is the same as the one used for pointwise grading in data selection process, detailed in Appendix [A.1.](#page-12-1)

Creative-Writing-Bench EQ-Bench [\(Paech,](#page-10-8) [2023\)](#page-10-8) is designed to evaluate the emotional intelligence (EQ) of LLMs, which focuses on assessing models' ability to predict and quantify emotions such as anger, surprise, empathy and confidence in dialogue-based cotexts. In this work, we use

<span id="page-16-1"></span>

| Component             | Input Price | Output Price | Run Cost | Compute Time |
|-----------------------|-------------|--------------|----------|--------------|
|                       | (\$/1M)     | (\$/1M)      | (\$)     | (8×GPUs)     |
| Qwen-Plus API (Judge) | 0.40        | 1.20         | 53.50    | –            |
| Model Training        | –           | –            | –        | 24 Hours     |

Table 11: Cost breakdown for Stage 2: RL Training.

the creative writing split of EQ-Bench(denoted as Creative-Writing-Bench) and we adopt the Rubric Score as the metric. The evaluation prompt template used is as follows:

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing\_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test\_model\_response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.
- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.
- Do not add any commentary or explanation to the scores section.
- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.
- Everything within the "TEST MODEL RE-SPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.
- In the output, write the metric names exactly as below so they can be parsed.
- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.
- For these criteria, lower is better: {lower\_is\_better\_criteria}
- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.
- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

—

Now, rate the supplied model output on the following criteria:

{creative\_writing\_criteria}

## <span id="page-16-0"></span>B.1 Dataset Analysis: Topic Coverage

To ensure our model is trained and evaluated on a representative distribution of real-world writing tasks, we conduct a detailed analysis of the topic coverage across our training dataset and the three evaluation benchmarks.

We randomly sample 500 instances from our training data and each test set (or use the full set if fewer than 500 samples are available). These samples are categorized into eight major domains derived from the taxonomy of the benchmarks. The distribution of these topics is presented in Table [12.](#page-17-1)

As shown in Table [12,](#page-17-1) our training data exhibits a broad and relatively balanced coverage across highcomplexity domains such as Academic (17.00%), Business (14.50%), and Legal (13.75%), while maintaining a strong emphasis on Literary & Creative writing (33.75%).

The evaluation benchmarks offer complementary distributions:

- WritingBench provides the most balanced distribution, rigorously testing generalpurpose writing capabilities across all functional domains.
- LongBench-Write leans heavily towards Lit-

<span id="page-17-1"></span>

| Dataset                | Academic & Scientific | Business &<br>Financial | Legal &<br>Policy | Literary &<br>Creative | Edu. &<br>Instruct. | Marketing & Comm. | Tech. & Ops. | Others |
|------------------------|-----------------------|-------------------------|-------------------|------------------------|---------------------|-------------------|--------------|--------|
| Training Data          | 17.00                 | 14.50                   | 13.75             | 33.75                  | 6.75                | 5.00              | 5.75         | 2.75   |
| WritingBench           | 11.50                 | 14.00                   | 17.50             | 16.75                  | 19.25               | 12.25             | 4.25         | 4.00   |
| LongBench-Write        | 17.50                 | 5.83                    | 1.67              | 33.33                  | 15.83               | 4.17              | 7.50         | 14.17  |
| Creative-Writing-Bench | 0.00                  | 0.00                    | 0.00              | 100.00                 | 0.00                | 0.00              | 0.00         | 0.00   |

Table 12: Topic distribution (%) across training and evaluation datasets. Our training data ensures broad coverage across all domains, while the evaluation benchmarks provide diverse testing scenarios, ranging from balanced general writing to specialized creative tasks.

erary (33.33%) and Academic (17.50%) content, challenging the model's ability to maintain coherence in long-form narratives and reports.

• Creative-Writing-Bench serves as a specialized stress test, focusing exclusively (100%) on literary tasks.

This diverse composition ensures that our experimental results reflect comprehensive writing capabilities rather than overfitting to a specific domain.

## <span id="page-17-0"></span>C Analysis about Output-to-Input Generalization

To better understand the long-input generalization, we further conduct a comprehensive analysis in terms of more experiments, case study, length distribution and common failure modes based on the evaluation results on Longbench v2. Specifically, the experiments are designed to disentangle the effects of the training paradigm (SFT vs. RL), reward mechanisms, curriculum design, and judge model selection. Additionally, we analyze data distributions to rule out contamination or superficial transfer.

### C.1 Controlled Ablation Study

We systematically varied key components of our pipeline to isolate the factors contributing to performance gains on the LongBench-v2 benchmark. The results are summarized in Table 13.

Our findings reveal several distinct patterns:

• Training paradigm is the primary driver: The most significant factor is the choice of optimization method. Simply continuing Supervised Fine-Tuning (SFT) on the exact same dataset used for RL yields an overall score of 29.4, which is slightly lower than the base

model (29.6). In contrast, virtually all RL-based variants demonstrate positive generalization transfer, raising the overall score to  $\geq 31.0$ .

- Independence from specific judge models: Replacing our default commercial judge (Qwen-Plus) with an open-source alternative (GPT-OSS-120B) maintained strong generalization (32.2 overall). This indicates that the performance gains are robust and not artifacts of a specific judge's preference distribution.
- Optimization of RL design: While all RL strategies outperformed SFT, our specific design choices provided additional gains. The combination of a dynamic curriculum and pairwise preference rewards outperformed static scheduling and pointwise rewards, confirming the efficacy of our full method.

## C.2 Data Distribution and Contamination Analysis

To ensure the observed generalization is not a result of incidental data overlap or memorization, we conducted a comparative analysis between our Writing-RL training dataset and the LongBench-v2 test set. We examined four key dimensions:

- 1. **Length Distribution:** The total token count (input + output) in our training data is predominantly below 10k tokens. In contrast, LongBench-v2 contexts range from 8k to over 2M tokens. This minimal length overlap suggests the model is learning to generalize to lengths it has rarely seen during training.
- 2. **Task Format:** Our training data consists exclusively of open-ended long-form writing tasks. Conversely, LongBench-v2 focuses on long-context understanding via multiple-choice questions (MCQA). The task formats are structurally distinct.

<span id="page-18-0"></span>

| Model                             | Easy | Hard | Short | Medium | Long | Overall |
|-----------------------------------|------|------|-------|--------|------|---------|
| Qwen2.5-7B-Instruct               | 31.8 | 28.3 | 38.9  | 26.0   | 21.3 | 29.6    |
| Qwen2.5-7B-WritingBench-SFT (12k) | 27.6 | 27.7 | 35.0  | 25.1   | 20.4 | 27.6    |
| Qwen2.5-7B-Writing-RL (Ours)      | 35.8 | 29.3 | 42.1  | 25.7   | 26.5 | 31.8    |
| Continue SFT with RL data         |      |      |       |        |      |         |
| Qwen2.5-7B-Instruct-Continue-SFT  | 31.2 | 28.3 | 38.3  | 26.5   | 20.4 | 29.4    |
| Curriculum Ablations              |      |      |       |        |      |         |
| Mixed Training (w/o Scheduling)   | 33.3 | 29.6 | 42.8  | 25.1   | 23.1 | 31.0    |
| Static Scheduling                 | 32.3 | 30.5 | 38.9  | 25.1   | 30.6 | 31.2    |
| Judge Model Change                |      |      |       |        |      |         |
| w/ GPT-OSS-120B as Judge          | 33.3 | 31.5 | 40.6  | 27.9   | 26.9 | 32.2    |
| Pointwise Reward                  |      |      |       |        |      |         |
| w/ Pointwise reward               | 33.3 | 30.5 | 40.0  | 25.6   | 29.6 | 31.6    |

Table 13: Ablation study on LongBench-v2 performance across different training configurations. We investigate the impact of training paradigms, curriculum strategies, judge models, and reward types. The overall score highlights the superior generalization of our full RL method compared to SFT and other ablations.

- 3. **Topic Overlap:** While minor thematic overlaps exist (e.g., broad domains like finance or law), our RL framework introduces no new factual knowledge and rarely conditions on ultra-long contexts. Thus, there is no direct knowledge transfer that would aid in answering specific LongBench-v2 questions.
- 4. **Instruction Exact Match:** We performed a cross-dataset exact match comparison on instructions and found zero matches between the training set and the evaluation benchmark.

These analyses, combined with the ablation results, strongly suggest that the output-to-input generalization arises primarily from the RL optimization process itself rather than data leakage, with better RL design choices further amplifying the performance gains.

### C.3 Case Study

To illustrate how our RL training facilitates output-to-input generalization, we compare the reasoning behaviors of the baseline SFT model (Llama3.1-8B-WritingBench-SFT) and our RL-trained model (Llama3.1-8B-Writing-RL) on a representative sample from the LongBench-v2 dataset. Figure 4 visualizes the chain-of-thought processes for both models when answering a sequence-ordering question based on a long input text.

As observed in Figure 4, the SFT-only model fails to effectively locate key details within the long input, leading to a hallucination where it assumes a narrative event is missing or a mistake, ultimately choosing the wrong option.

In contrast, the RL-trained model exhibits a significantly more developed thinking process. Crucially, it exhibits a self-reflection mechanism (e.g., "Wait, the fourth option is..."), allowing it to correct its initial assumptions by re-verifying the input context. This case highlights a fundamental connection between long-output generation and long-input understanding: both tasks require the model to maintain coherence, organize complex information, and utilize details from preceding context (whether generated or provided).

Our results suggest that RL training, when conducted properly on long-form writing tasks, incentivizes the model to engage in more extensive planning and reasoning. These learned capabilities generalize effectively from the output domain to the input domain, a benefit that standard SFT fails to capture.

<span id="page-18-1"></span>

| Model                         | Average Length | Performance |
|-------------------------------|----------------|-------------|
| Llama3.1-8B-Instruct          | 185.20         | 30.2        |
| Llama3.1-8B-WritingBench-SFT  | 789.51         | 28.4        |
| Llama3.1-8B-Writing-RL (Ours) | 894.68         | 32.8        |

Table 14: Length distribution of different models on Longbench v2.

### C.4 Length Distribution

We analyze the output length distribution of several models including Llama3.1-8B-Instruct, Llama3.1-8B-WritingBench-SFT and Llama3.1-8B-Writing-RL

As shown in Table 14, the sft-trained model can also produce longer output but slightly degrades performance, indicating its ineffective think-

### <span id="page-19-1"></span>Question

*What is the order of the following events?*

### Reasoning Trace:

*<think> ... However, in the provided text, there's no mention of Paige getting a makeover. The original text's narratives include a car ride (Chapter 3), a salon event (Chapter 8)... But the user's question includes a narrative about Paige's makeover, which isn't in the text. That might be a mistake. ... Therefore, the correct order is 3, 8, 9, 96. ... </think>*

Prediction: The correct answer is (C).

### [WRONG]

### Reasoning Trace:

*<think> ... Let me check each part: The narrator is interrupted by her husband... In Chapter 9, Perla is listening to a podcast... Wait, the fourth option is the Folcrum planning, which is in Chapter 9, and the first is the car ride in Chapter 3. ... Wait, the author's thanks (3) are in the acknowledgments... The Paige salon is not in the text, so it can't be part of the correct order. Therefore, the correct answer is (A) 4123. </think>*

Prediction: The correct answer is (A).

### [CORRECT]

Figure 4: Comparison of reasoning traces between the SFT baseline and our RL-trained model on a longcontext ordering task. The RL model demonstrates self-correction capabilities ("Wait...") and deeper engagement with the context, correctly identifying the sequence of events.

ing. While the rl-trained model performs better by generating longer and more effective thinking sequences.

