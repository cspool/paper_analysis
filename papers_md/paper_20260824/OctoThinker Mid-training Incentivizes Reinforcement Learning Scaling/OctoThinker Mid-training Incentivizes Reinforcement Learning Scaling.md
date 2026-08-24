**GAIR Lab** 2025-6-26

# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling

Zengzhi Wang\*, Fan Zhou\*, Xuefeng Li\*, Pengfei Liu\*
Shanghai Jiao Tong University, SII, GAIR Lab
{zengzhi.wang, zhoufan98, xuefengli, pengfei}@sjtu.edu.cn

![](_page_0_Picture_4.jpeg)

GAIR-NLP/OctoThinker OctoThinker

Different base language model families—such as Llama and Qwen—exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy—Stable-then-Decay—in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max).

> **[图片提取文字 (无描述)]:**
> \* Zero RL refers to R1-Zero style RL training starting from a base language model without SFT. Zero RL Training Zero RL Training 66.4 65.2 60-Mid-training MATH 500 (Decay Stage) 40 -Mid-training 38.2 (Stable Stage) 25.8 Zero RL Training 20 22.4 10.0 7.4 Llama-3.2 Llama-3.2 OctoThinker-OctoThinker-OctoThinker-Qwen2.5-3B Qwen2.5-3B 3B-Base 3B-Zero Stable-3B-Base Long-3B-Base Long-3B-Zero Base Zero
![](_page_0_Figure_7.jpeg)

Figure 1 | Our strategic mid-training incentivizes Llama's RL scaling, matching Qwen2.5 performance.

<sup>\*</sup>Equal contribution. \*Corresponding author.

# **1. Introduction**

Incentivizing large language models (LLMs) to think deeply through the chain of thought (CoT [\(Wei](#page-23-0) [et al.,](#page-23-0) [2022\)](#page-23-0)) before giving the final answer with large-scale reinforcement learning (RL) is driving significant progress on the challenging reasoning tasks, i.e., solving competition-level mathematics problems, as demonstrated by OpenAI's o1 [\(OpenAI et al.,](#page-21-0) [2024\)](#page-21-0) and o3 [\(OpenAI,](#page-21-1) [2025\)](#page-21-1). This also underscores a growing attention centered on RL as a means of boosting LLMs' reasoning performance. Deepseek-R1-Zero [\(Guo et al.,](#page-19-0) [2025\)](#page-19-0) showcases a range of powerful and intriguing reasoning behaviors by directly applying large-scale RL to base language models, i.e., Deepseek-V3-Base [\(Liu et al.,](#page-21-2) [2024\)](#page-21-2). In line with this trend, several methods such as SimpleRL [\(Zeng et al.,](#page-23-1) [2025\)](#page-23-1) and Open-Reasoner-Zero [\(Hu et al.,](#page-20-0) [2025\)](#page-20-0) have explored RL training on smaller base models—particularly the Qwen series [\(Yang et al.,](#page-23-2) [2025,](#page-23-2) [2024b\)](#page-23-3)—achieving notable improvements in reasoning ability. However, despite these advances, replicating the success of R1-Zero-style training on other general-purpose base models, such as Llama series [\(Meta et al.,](#page-21-3) [2024\)](#page-21-3), has proven difficult, also evidenced by recent studies [\(Gandhi et al.,](#page-19-1) [2025;](#page-19-1) [Liu et al.,](#page-21-4) [2025\)](#page-21-4). This naturally raises a fundamental question: *What underlying factors cause the base models to exhibit divergent behaviors during RL training?* Understanding this could shed light on the scientific foundations that connect pre-training and the scalability of RL for reasoning, and may guide the design of future base models more amenable to reasoning-oriented RL.

In this work, we explore this question through the lens of mathematical reasoning and begin by observing a key difference in RL dynamics between two prominent model families: Qwen and Llama. Specifically, our preliminary studies reveal that Qwen models are much more amenable to RL scaling, while the Llama model tends to predict final answers prematurely and produce repetitive outputs during RL training. To better understand this discrepancy, we conduct a series of largescale and controlled mid-training interventions on Llama models, followed by RL training. Our findings highlight that the quality of mathematical pre-training corpora is critical for successful RL performance. For instance, we find that MegaMath-Web-Pro [\(Zhou et al.,](#page-23-4) [2025\)](#page-23-4) offers significantly greater benefits for RL scaling than corpora like FineMath-4plus [\(Allal et al.,](#page-18-0) [2025\)](#page-18-0). On top of a highquality mathematical pre-training corpus, incorporating QA-style data yields further improvements, and introducing a small amount of instruction-following data helps enhance RL effectiveness even more. We also observe that injecting long CoT data during mid-training introduces instability into the RL phase. To address this, we refine the RL prompt and adopt a progressive maximum response length scheduler to stabilize training and ensure consistent behavior. To support largescale mid-training, we also curate a reasoning-intensive mathematical corpus exceeding 70 billion tokens, namely MegaMath-Web-Pro-Max, with data quality on par with MegaMath-Web-Pro. In extended mid-training experiments on this dataset—scaling up to 100 billion tokens—we observe that increasing the mid-training budget can lead to noticeable improvements in downstream RL performance. Interestingly, these gains are often not immediately reflected in the standard evaluations of the mid-trained base model, highlighting a gap between base model evaluation metrics and RL-stage capabilities.

*Can we turn Llama into a foundation model well-suited for RL scaling by further scaling up its mid-training?* Building on the insights above, we explore this question by adopting a two-stage (*stable-then-decay*) mid-training strategy. In the first stable stage, we train Llama models on a highquality mixture of pre-training corpus for 200B tokens using a constant learning rate. In the second decay stage, we anneal the learning rate and introduce distinct data mixtures—short CoT, long CoT, and a hybrid of both—to mid-train three separate branches. These branches are later refined through RL training, equipping them with stronger reasoning capabilities. Inspired by the multi-armed nature of an octopus, we name this model family OctoThinker. Experiments across all model sizes and

14 mathematical reasoning benchmarks demonstrate the effectiveness of our approach: both stages of mid-training lead to substantial performance gains, especially the first stage, which consistently delivers 10–20% improvement. Building on these stronger base models, subsequent RL training further boosts performance, with each branch showing distinctive behavior patterns. Notably, our models post-RL achieve performance on par with Qwen2.5 of the same size, effectively narrowing the gap between Llama and other RL-friendly model families. These results confirm the power of scaled-up, reasoning-intensive mid-training in transforming Llama into a suitable base model for RL scaling. To foster open research, we will release our curated pre-training data, models, and training scripts. As we enter the era of RL scaling, we are prompted to ask: *What kind of foundation models do we need?* We believe this new phase brings unprecedented challenges for foundation models—and we hope OctoThinker offers a meaningful step toward the next generation of reasoning-capable AI systems.

# <span id="page-2-1"></span>**2. Preliminaries**

We begin by identifying a key difference in RL dynamics between two prominent model families—Qwen and Llama—through the lens of mathematical reasoning. This observation offers a concrete and measurable foundation that grounds our systematic investigation.

### **2.1. Experiment Setup**

**RL Setup** We perform our RL experiments based on the verl [\(Sheng et al.,](#page-22-0) [2024\)](#page-22-0) framework and utilize the GRPO [\(Shao et al.,](#page-22-1) [2024\)](#page-22-1) algorithm. For RL training prompts, we adopt the MATH8K [1](#page-2-0) dataset due to its moderate difficulty and concise composition. We configure the global training batch size to 128, set the number of rollout responses per query to 16, and use a PPO mini-batch size of 64. The sampling temperature is set to 1.0, with a maximum output length of 4096 tokens. We use a learning rate of 1 × 10−<sup>6</sup> and set the KL loss coefficient to 0 in the verl configuration. Empirically, we find that setting the ratio between sampling and gradient updates to 2 leads to more stable RL training. Unless otherwise specified, we employ a simple prompt template of "Question:{}\nAnswer:{}" to format training examples.

**Choices of Base Model** We employ Llama-3.2-3B-Base [\(Dubey et al.,](#page-19-2) [2024\)](#page-19-2) and Qwen2.5-3B-Base [\(Yang et al.,](#page-23-3) [2024b\)](#page-23-3) to perform R1-Zero styled RL training given the moderate model size.

**Evaluation** We adopt the few-shot prompting evaluation for base language models and employ zero-shot evaluation for RL-tuned models. Specifically, we employ GSM8K [\(Cobbe et al.,](#page-18-1) [2021\)](#page-18-1), MATH500 [\(Lightman et al.,](#page-20-1) [2023\)](#page-20-1), OlympiadBench [\(He et al.,](#page-19-3) [2024\)](#page-19-3), and AMC23 as indicator tasks to analyze RL dynamics. To assess base model performance, we further include MATH [\(Hendrycks et al.,](#page-19-4) [2021\)](#page-19-4), SAT-MATH [\(Azerbayev et al.,](#page-18-2) [2024\)](#page-18-2) , MathQA [\(Amini et al.,](#page-18-3) [2019\)](#page-18-3), MMLU-STEM [\(Hendrycks](#page-19-4) [et al.,](#page-19-4) [2021\)](#page-19-4), OCW Course [\(Lewkowycz et al.,](#page-20-2) [2022\)](#page-20-2), MAWPS [\(Koncel-Kedziorski et al.,](#page-20-3) [2016\)](#page-20-3), SVAMP [\(Patel et al.,](#page-22-2) [2021\)](#page-22-2), ASDiv [\(Miao et al.,](#page-21-5) [2020\)](#page-21-5), and TabMWP [\(Lu et al.,](#page-21-6) [2023\)](#page-21-6).

#### **2.2. Observations**

During RL training on Llama-3.2-3B-Base and Qwen2.5-3B-Base, we observe notably different and intriguing training dynamics regardless of their performance, as shown in Figure [2.](#page-3-0) Specifically, the length of correct responses from the Qwen model increases steadily and reasonably throughout training, whereas Llama exhibits abnormal behavior—its average response length escalated dramatically, reaching up to 4,096 tokens.

<span id="page-2-0"></span><sup>1</sup>[https://hf.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/tree/main/simplelr\\_qwen\\_level3to5](https://hf.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/tree/main/simplelr_qwen_level3to5)

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** OlympiadBench GSM8K AMC23 Accuracy (%) Response Length(K) Step Step Step Step Qwen2.5-3B Llama-3.2-3B
![](_page_3_Figure_1.jpeg)

**Figure 2** | Training dynamics comparison (downstream performance and the average length of correct responses) between Llama-3.2-3B and Qwen2.5-3B. The dashed line indicates the few-shot evaluation performance and average length of correct responses of the corresponding base models.

Upon closer inspection of the Llama model's output, we find that it typically begins with "\boxed:{}", followed by extremely obvious repetition until hitting the max response length, in stark contrast to Qwen's coherent and natural reasoning output. The evaluation results further highlight the divergence: The RL-tuned Qwen2.5-3B achieves substantial improvements over its base model across a wide spectrum of benchmarks, from simple to complex math reasoning tasks. Meanwhile, Llama-3.2-3B experiences only marginal gains—or even regressions, as seen on GSM8K—likely due to the distributional gap between the RL training set (e.g., MATH8K) and GSM8K. The above observations motivate us to attribute the reason to their potential divergence of pre-training despite their opaque details.

These observations also further prompt a more fundamental question: *Can we intervene in the Llama base language models via mid-training to make it more amenable to RL scaling?* Specifically, in this work, we aim to explore a range of mid-training intervention strategies—methods that adjust the pre-training trajectory of LLMs—to examine their downstream impact on RL dynamics.

### **What is Mid-training?**

Mid-training is a mid-stage whose computational and data (token) requirements are intermediate between pre-training and post-training. It aims to achieve specific objectives—such as domain and language expansion [\(Dou et al.,](#page-18-4) [2025,](#page-18-4) *inter alia*), long-context extension [\(Abdin](#page-17-0) [et al.,](#page-17-0) [2024a](#page-17-0)[,b,](#page-17-1) *inter alia*), improving data quality [\(Hu et al.,](#page-20-4) [2024a;](#page-20-4) [OLMo et al.,](#page-21-7) [2025,](#page-21-7) *inter alia*), leveraging large-scale synthetic data [\(Yang et al.,](#page-23-5) [2024a,](#page-23-5) [2025,](#page-23-2) [2024b,](#page-23-3) *inter alia*), and preparing for post-training, among others—by significantly altering data quality and distribution [\(Dubey et al.,](#page-19-2) [2024;](#page-19-2) [Wake et al.,](#page-22-3) [2024,](#page-22-3) *inter alia*) (and/or modifying model architecture to improve inference efficiency [\(Bercovich et al.,](#page-18-5) [2024,](#page-18-5) [2025,](#page-18-6) *inter alia*)).*[a](#page-3-1)*

<span id="page-3-1"></span>*a* In the absence of a precise or widely agreed-upon definition, here, we aim to introduce a concise and rigorous definition of *mid-training* within this context. The term was reportedly first mentioned in an OpenAI job description in mid-2024. A detailed blog for this term is available at [https://vintagedata.org/blog/posts/](https://vintagedata.org/blog/posts/what-is-mid-training) [what-is-mid-training](https://vintagedata.org/blog/posts/what-is-mid-training)

# **3. Digging Deeper: Exploring Key Factors through Controllable Mid-training**

We aim to investigate the impact of several factors during mid-training on RL performance through head-to-head experiments, as shown in Figure [3.](#page-4-0) Specifically, we examine the effects of data quality of math web corpora, the inclusion or exclusion of QA-format data, the nature of the QA data itself, the presence of general instruction-following data in mid-training, as well as the pre-training token budget. These systematic analyses help us gain a deeper understanding of the connection between pre-training and RL dynamics and figure out suitable recipes for scaled-up mid-training. Controlled Factors During Mid-training

<span id="page-4-0"></span>> **[图片提取文字 (无描述)]:**
> Math Web Mid-training QA Data Instruction Data Corpora Budget Finemath-Instruction-Long-CoT QA 20B tokens Following Data 4plus MegaMath-100B tokens Short-CoT QA None Web-Pro
![](_page_4_Figure_3.jpeg)

**Figure 3** | Potential factors in mid-training that could impact the post-training stage.

## **3.1. Experimental Setup**

**Mid-training setup** By default, we perform mid-training with Llama-3.2-3B-Base on diverse datasets and training configurations within a 20B-token training budget. We use a cosine learning rate scheduler without warmup, with a peak learning rate of 3e-5 and a minimum learning rate set to one-tenth of the peak. The default sequence length is 8,192, and the batch size is 4 million tokens. Training is conducted using the Nanotron framework. [2](#page-4-1)

**RL setup** We follow the exact same RL setup as described above in Section [2,](#page-2-1) unless stated otherwise.

<span id="page-4-2"></span>**Table 1** | Statistics and Types of different datasets used in our experiments. <sup>⋄</sup>We use the TULU3-sft-personna-instruction-following subset.

| Dataset                                                                                                       | Type                            | # Tokens (B)           |
|---------------------------------------------------------------------------------------------------------------|---------------------------------|------------------------|
| FineMath-4plus (Allal et al., 2025)<br>MegaMath-Web-Pro (Zhou et al., 2025)<br>MegaMath-Web-Pro-Max<br>(Ours) | Math Web Documents              | 9.57<br>13.00<br>73.80 |
| MegaMath-QA (Zhou et al., 2025)<br>OpenR1-Math-220K (HuggingFace, 2025)                                       | QA (Short-CoT)<br>QA (Long-CoT) | 5.94<br>1.05           |
| TULU3-sft⋄<br>(Lambert et al., 2024a)<br>WildChat (Zhao et al., 2024)<br>UltraChat-220K (Ding et al., 2023a)  | General Instruction Following   | 0.01<br>0.29<br>0.51   |

**Datasets** The datasets used to support our controllable experiments are summarized in Table [1.](#page-4-2) For the OpenR1 dataset, we concatenate the question and the thinking process enclosed within <think> and </think> using a line break. For the general instruction following datasets, we only retain

<span id="page-4-1"></span><sup>2</sup><https://github.com/huggingface/nanotron>

high-quality conversations, such as those derived from GPT-4, and formated the conversations as "User:{}\nAssistant:{}".

The Curation of MegaMath-Web-Pro-Max We curate MegaMath-Web-Pro-Max to support large-scale ablation studies and mid-training. The corpus is constructed using an efficient classifier to recall documents from MegaMath-Web (Zhou et al., 2025), followed by refinement using a powerful instruction-following LLM. Specifically, we uniformly and randomly sample millions of documents from the MegaMath-Web corpus, stratified by publication year, and annotate them using Llama-3.1-70B-instruct. Each document is graded for its usefulness in studying mathematics on a scale from 0 to 5 using a grading prompt (see Figure 15). We heuristically extract scores from the model's critiques: documents scoring below 3 were labeled as negative examples, while those scoring 3 or above were considered positive. We observe that existing classifiers, such as finemath-classifier, are highly sensitive to the choices of text extractors during data curation. This motivates us to train our own classifier, selecting fasttext for its efficiency. Consistent with the findings of Zhou et al. (2025), we find preprocessing to be critical for recall performance. Our preprocessing pipeline includes lowercasing text, filtering excessively long words, and removing line breaks and extraneous non-alphanumeric characters.

Following MegaMath-Web's yearly dump comparison setup, we evaluate the quality of our recalled corpus under different thresholds, as shown in Figure 4. The recall threshold controls the trade-off between data quantity and quality: a higher threshold (e.g., 0.9) yields better quality but retains fewer tokens. Finally, we select a threshold of 0.4. Given the noisy and poorly structured nature of many documents, we employ Llama-3.1-70B-instruct to refine the text using a prompt (see Figure 16) inspired by MegaMath-Web-Pro. The resulting dataset, MegaMath-Web-Pro-Max, contains approximately 5.5 times more tokens than MegaMath-Web-

<span id="page-5-0"></span>> **[图片提取文字 (无描述)]:**
> Avg Core CoT Performance Year MegaMath-Web's Best fasttext (0.9) Ours (0.4) Ours (0.9)
![](_page_5_Figure_4.jpeg)

**Figure 4** | Comparison between our fasttext-recalled corpus (w/o LLM refinement) and MegaMath-Web, following its yearly dump comparison setup under a 5B-token pre-training budget. Recall thresholds shown accordingly.

Pro. Empirical evaluations during pre-training indicate that MegaMath-Web-Pro-Max maintains comparable data quality, making it a strong candidate as a foundational corpus for large-scale midtraining. Besides, we also explored to supplement the positive seed set with (long)CoT examples from common math problem-solving datasets to improve the classifier's ability to recall reasoning-intensive content. However, this approach retained only around 20B tokens, which we deemed insufficient in scale and thus did not adopt.

#### 3.2. On the Inclusion and Data Quality of Math Web Corpora

Web corpora provide a solid foundation during pre-training. We believe that math-specific web corpora, along with their data quality, continue to play a crucial role during mid-training. We begin our systematic analysis by performing mid-training on different math web corpora and holding other factors being constant. As shown in the Figure 5, mid-training on math web data improves performance over the base model, with MegaMath-Web-Pro and MegaMath-Web-Pro-Max showing slightly better gains than Finemath-4plus. After RL training, we find that mid-training on math web corpora improves RL performance to varying degrees. MegaMath-Web-Pro and MegaMath-Web-Pro-Max bring significant

<span id="page-6-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K OlympiadBench AMC23 15.0 Accuracy (%) 12.5 10.0 7.5 5.0 2.5 0.0 Response Length(K) Step Step Step Step Finemath-4plus Megamath-Web-Pro Megamath-Web-Pro-Max Llama-3.2-3B
![](_page_6_Figure_1.jpeg)

**Figure 5** | The effect of different math web corpora during mid-training. We performed mid-training on each corpus with a 20B-token training budget.

gains for Llama in RL training, while Finemath-4plus yields only marginal improvements—highlighting the clear differences in data quality. Furthermore, we observe that models trained on FineMath-4plus exhibited abnormal behavior, with response lengths rapidly increasing until reaching the maximum limit of 4,096 tokens. The outputs typically begin with "\boxed{}" and devolve into repetitive "Solution" statements. Given these observations, we select MegaMath-Web-Pro as our default mathematical corpus and also MegaMath-Web-Pro-Max for scaled mid-training.

**Takeaway:** High-quality math pre-training corpora play a dominant role in RL scaling. We finally adopt MegaMath-Web-Pro and our curated MegaMath-Web-Pro-Max in this work.

#### **3.3. On the Inclusion and Nature of QA-Format Data**

Intuitively, introducing QA data into pre-training and mid-training improves model performance, as previously examplified in [Bi et al.](#page-18-8) [\(2024\)](#page-18-8) and [Hu et al.](#page-20-7) [\(2024b\)](#page-20-7). We further investigate this using a 9:1 web-to-QA data mix. We hypothesize that QA data's short Chain-of-Thought (short-CoT, from MegaMath-QA) and long-CoT (from OpenR1-Math-220K) reasoning, which may include self-reflection and backtracking, enhance base model performance and RL training. Maximum response lengths were 8,192 tokens for long-CoT models and 4,096 for others.

As shown in Figure [6,](#page-7-0) incorporating QA data into mid-training generally yields performance gains for the base model, though these gains are marginal, as indicated by dashed lines. After RL training, incorporating short-CoT data into mid-training shows no improvements compared to mid-training on web data alone, possibly due to the data distribution gap (see § [4.2](#page-11-0) for more ablation studies), while long-CoT data brings significant performance gains. However, incorporating long-CoT data introduces challenges with unstable RL training, evidenced by sudden performance drops and sharp increases in response length. We also explore methods for stabilizing RL training, which we discuss in the following sections.

<span id="page-7-0"></span>> **[图片提取文字 (无描述)]:**
> GSM8K **MATH500** OlympiadBench AMC23 17.5 70 40 15.0 60 20 Accuracy (%) 12.5 10.0 7.5 30 10 20 5.0 10 2.5 10 0.0 0 100 200 300 100 200 300 100 200 300 100 200 300 400 400 400 400 3.2 1.00 Response Length(K) 4.5 0.75 3.0 0.50 1.6 1.5 0.25 0.8 0.0 0.0 0.00 200 300 100 200 300 400 100 300 400 100 200 400 100 200 300 400 0 0 Step Step Step Step w/ short-CoT Web only Llama-3.2-3B w/ long-CoT
![](_page_7_Figure_1.jpeg)

**Figure 6** | Impact of incorporating CoT data with varying characteristics during mid-training (9:1 mixture ratio). The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base and its mid-trained variants for reference (in dashed line with different colors).

**Takeaway:** QA data could aid RL scaling, but gains depend on its distribution gap with downstream tasks. Long CoT patterns often induce excessive responses and sudden performance drops in RL-tuned models.

## **3.4. On the Inclusion of Instruction-following Data**

Incorporating instruction-following data into earlier-stage training has become an increasingly common practice. Works such as MiniCPM [\(Hu et al.,](#page-20-7) [2024b\)](#page-20-7) demonstrate that including high-quality unlabeled data and instruction-following data significantly improves downstream performance. We believe this inclusion is critically important for enhancing the base model's ability to follow instructions, which may be a potential key determining factor for successful RL training. We incorporate instruction-following data alongside web data and QA data in a 1:89:10 ratio. For this, we combine these high-quality datasets with appropriate filtering and formatting: TULU3-sft-personas-instructionfollowing [\(Lambert et al.,](#page-20-8) [2024b\)](#page-20-8), WildChat [\(Zhao et al.,](#page-23-6) [2024\)](#page-23-6), and UltraChat-200K [\(Ding et al.,](#page-18-9) [2023b\)](#page-18-9), totaling approximately 0.8B tokens.

**Incorporating instruction-following data into the short-CoT mid-training mixture.** As shown in Figure [7,](#page-8-0) after RL training, incorporating instruction-following data, unlocks the potential of short-CoT data, showing performance advantages over the exclusion case after 200 steps. Additionally, this inclusion helps stabilize response length, resulting in smoother increases compared to when instruction-following data is excluded.

**Incorporating instruction-following data into the long-CoT mid-training mixture.** Similar to the challenges encountered earlier in RL training with the long-CoT mid-trained base model, as shown in Figure [8,](#page-8-1) incorporating instruction-following data shows performance improvements after 150 steps. However, this addition still fails to prevent the overall decline in RL performance and the rapid increase in response length. Note that we set the maximum response length to 8,192 tokens for these experiments.

<span id="page-8-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K OlympiadBench AMC23 Accuracy (%) 10 -1.00 0.60 0.60 0.30 Response Length(K) 0.75 0.45 0.45 0.24 0.50 0.30 0.30 0.18 0.25 0.15 0.12 0.00 0.00 0.00 Step Step Step Step w/o Instruction Data w/ Instruction Data Llama-3.2-3B
![](_page_8_Figure_1.jpeg)

**Figure 7** | Impact of incorporating instruction-following data during mid-training with a mixture of web, short-CoT and instruction data in a ratio of 89: 10: 1 . The maximum response length is 4,096. The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base and its mid-trained variants for reference (in dashed line with different colors).

<span id="page-8-1"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K OlympiadBench AMC23 20.0 17.5 15.0 Accuracy (%) 12.5 10.0 7.5 5.0 2.5 0.0 6.0 1.2 Response Length(K) 4.5 0.8 3.0 0.4 1.5 0.0 0.0 0.0 Step Step Step Step w/ Instruction Data w/o Instruction Data Llama-3.2-3B
![](_page_8_Figure_3.jpeg)

**Figure 8** | Impact of incorporating instruction-following data during mid-training with a mixture of web, long-CoT and instruction data in a ratio of 89: 10: 1. The maximum response length is 8,192. The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base and its mid-trained variants for reference (in dashed line with different colors).

Given the challenges encountered during RL training on the base model mid-trained on long-CoT data, we explore strategies to stabilize RL training by modifying the RL prompt template and maximum length scheduler.

**Effect of RL prompt template** The default template is "Question:{}\nAnswer:{}", which we refer to as "*Simple Template*". Here, we introduce an alternative, the "*Complex Template*", adapted from the prompt design in Open-Reasoner-Zero [\(Hu et al.,](#page-20-0) [2025\)](#page-20-0):

<span id="page-9-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K OlympiadBench AMC23 20.0 17.5 15.0 Accuracy (%) 12.5 10.0 7.5 5.0 2.5 Response Length(K) 1.2 4.5 3.0 0.8 0.4 0.0 0.0 Step Step Step Step w/ Simple Template w/ Complex Template Llama-3.2-3B
![](_page_9_Figure_1.jpeg)

**Figure 9** | Impact of different RL prompt templates. The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base (in dashed line with different colors).

**Prompt Template:** A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: You must put your answer inside \\boxed{} and your final answer will be extracted automatically by the \\boxed{} tag. {{prompt}}

Assistant:

We also control the maximum response length as 8,192 tokens. As shown in Figure [9,](#page-9-0) we find this *complex template* could clearly stabilize RL training compared to the *simple template*, as evidenced by a smoother, more gradual increase in mean response length, as opposed to the sharp spikes observed with the simple template. Despite this stabilization, performance across evaluation benchmarks still deteriorates during the later stages of RL training, indicating need more exploring. Note that we adopt the *complex template* as the default for all subsequent RL experiments.

**Effect of the maximum response length** The default maximum context length is set to 8,192 tokens for long-CoT mid-trained models. Intuitively, we can delay the sharp rise in response length by gradually increasing the maximum response length in multiple stages. Specifically, we start with a limit of 2,048 tokens for the first 200 steps, increase it to 4,096 tokens from step 200 to step 320, and then further expand to the full 8,192-token context length from step 320 to step 400. As shown in Figure [10,](#page-10-0) this progressive scheduling strategy significantly stabilizes RL training up to 400 steps, while consistently improving performance across benchmarks. In addition, the response lengths grow steadily and appropriately, highlighting the effectiveness of the progressive length scheduler.

**Takeaway:** Introducing a small amount of instruction-following data can help unlock the potential of QA data and mitigate RL training collapse caused by long CoT. We address this issue by modifying the RL prompt template and applying a progressive maximum response length scheduler.

<span id="page-10-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K **Olympiad Bench** AMC23 22.5 50 30 20.0 70 17.5 25 Accuracy (%) 15.0 60 20 12.5 2K-4K 4K-8K 2K → 4K 4K → 8K 2K → 4K 4K → 8K 2K → 4K 4K → 8K 50 15 10.0 7.5 10 40 5.0 10 2.5 30 100 200 300 400 100 200 300 400 100 200 300 400 100 200 300 400 0 2.4 1.5 2.5 Response Length(K) 0.60 2.0 1.8 1.2 0.45 2K-4K 4K-8K 2K-4K 4K-8K/ 2K - 4K 4K - 8K 2K -4K 4K -8K 1.5 1.2 0.30 1.0 0.6 0.6 0.15 0.5 0.3 200 300 300 100 200 300 400 100 400 0 100 200 400 100 200 300 400 0 Step Step Step Step  $2K \rightarrow 4K \rightarrow 8K$ 8K Llama-3.2-3B
![](_page_10_Figure_1.jpeg)

**Figure 10** | Impact of the maximum length scheduler on the model response. The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base in a dashed line.

> **[图片提取文字 (无描述)]:**
> GSM8K OlympiadBench AMC23 MATH500 Accuracy (%) 0.8 2.0 0.60 Response Length(K) 1.2 1.5 0.6 0.45 0.8 1.0 0.30 0.4 0.4 0.15 0.00 0.0 0.0 Step Step Step Step 20B-Token Mid-training 70B-Token Mid-training 100B-Token Mid-training Llama-3.2-3B
![](_page_10_Figure_3.jpeg)

**Figure 11** | Impact of scaling up the mid-training budget. The figure also illustrates performance and average lengths of correct responses for Llama-3.2-3B-Base and its mid-trained variants for reference (in dashed lines with different colors).

### **3.5. On the Issue of Mid-training Budget**

Could further scaling up mid-training improve RL performance? To explore this, we conduct a 100Btoken mid-training run on MegaMath-Web-Pro-Maxusing a default cosine learning rate scheduler. We select three intermediate checkpoints—trained on 20B, 70B, and 100B tokens, respectively—and perform RL training. When evaluating the base models, we observe that the 70B and 100B checkpoints achieved comparable performance, both significantly outperforming the 20B model. After RL training, interestingly, we find that increasing the mid-training token count consistently leads to improvements on RL performance despite varying degrees, whether moving from 20B to 70B or from 70B to 100B tokens. These findings highlight the importance of further scaling up the mid-training budget to

unlock additional gains in downstream RL performance.

**Takeaway:** Increasing the mid-training budget can improve RL performance, even if such gains are not evident in base model evaluations.

# **4. OctoThinker-Base: Branching Reasoning Foundations via 2-Stage Mid-training**

Building upon the insights above, a natural question arises: *Can we turn Llama into a foundation model well-suited for RL scaling by scaled-up mid-training?* We ultimately adopt a two-stage (*stable-then-decay*) mid-training strategy to achieve both: (1) steady improvements in mathematical reasoning ability in the first stage; (2) diversified model behaviors via branching in the second decay stage. Multi-stage pre-training has been validated as effective in prior work [\(Hu et al.,](#page-20-7) [2024b;](#page-20-7) [OLMo](#page-21-7) [et al.,](#page-21-7) [2025\)](#page-21-7). The *stable-then-decay* setup offers flexibility: the decay phase can begin at any point, enabling checkpoint selection independent of a fixed schedule. This also supports fair comparisons across different mid-training configurations. Importantly, decaying the learning rate in the second stage amplifies the effect of injected data, helping shape model behaviors more efficiently. Since the decay stage used for shifting model behaviors (in other words data distribution) is typically shorter, this approach also reduces the overall training cost in general. We name this resulting model family OctoThinker[3](#page-11-1) , inspired by the octopus's multi-armed structure, reflecting its multiple branches.

## **4.1. Recipe for the First Stage: Building Strong Reasoning Foundations**

Although the previous analysis has revealed several factors that are critical to building strong reasoning models, our mid-training resource table (see Table [1\)](#page-4-2) clearly shows that truly high-quality tokens are still scarce at this moment. Therefore, in the first phase, we adopt a relatively conservative strategy—primarily relying on high-quality web corpora such as MegaMath-Web-Pro-Maxand DCLM-Baselines [\(Li et al.,](#page-20-9) [2024\)](#page-20-9), supplemented with a small portion of synthetic data—to enable the model to improve steadily at scale. Following the training settings used in MegaMath-Llama [\(Zhou et al.,](#page-23-4) [2025\)](#page-23-4), we reduce the proportion of synthetic data and adopt a WSD-style [\(Hu et al.,](#page-20-7) [2024b\)](#page-20-7) learning rate scheduler, replacing the cosine learning rate with a constant learning rate and training for 200B tokens. We provide specific training configurations, i.e., data mixture and training hyperparameters of the first-stage in Table [2](#page-11-2) and Table [3.](#page-11-2) We refer to the resulting mid-training models as OctoThinker-Base-Stable.

<span id="page-11-2"></span>**Table 2** | Dataset composition and weights in the first-stage.

| Dataset                  | Weight |  |
|--------------------------|--------|--|
| DCLM-Baseline            | 0.10   |  |
| MegaMath-Web-Pro-Max     | 0.725  |  |
| MegaMath-Code            | 0.0125 |  |
| MegaMath-QA              | 0.05   |  |
| MegaMath Trans. Code     | 0.0125 |  |
| MegaMath Text Code Block | 0.10   |  |

**Table 3** | hyper-parameters in stable stage.

| Hyper-parameter    | Llama-3.2-1B / 3B / 8B |
|--------------------|------------------------|
| Context Length     | 8,192                  |
| Batch Size         | 512                    |
| Max Steps          | 50,000                 |
| Warmup Steps       | 0                      |
| Weight Decay       | 0.1                    |
| Optimizer          | AdamW                  |
| LR Scheduler       | Constant               |
| Learning Rate (LR) | 5e-5/2e-5/1e-5         |

<span id="page-11-1"></span><span id="page-11-0"></span><sup>3</sup> "*Octo*" is derived from "octopus," symbolizing our base model family, which branches into variants trained with different strategies. "*Thinker*" reflects the model's final stage—reinforcement learning—where it is trained to think and reason, exhibiting frequent self-reflection and strong reasoning capabilities.

#### 4.2. Branching at the Second Stage: Seeking Perfect Blend for RL Scaling

#### 4.2.1. Pilot Studies

Building on prior experiments, we identify dataset quality and quantity as key drivers of effective mid-training and strong base model development. Before entering the decay stage, we conduct a series of controlled 10B-token mid-training experiments on the OctoThinker-3B-Base-Stable model—each followed by RL training—to investigate how different QA datasets affect downstream performance.

Data Composition and Its Impact on RL We experiment with three QA datasets—MegaMath-QA, OpenR1-Math-220K, and OpenMathInstruct-2 (OMI2)—in varying proportions (10%, 20%, 30%, and 40%) while holding constant 5% DCLM-Baselines data, 10% instruction data, and the remainder from MegaMath-Web-Pro. Ablation studies (see Figure 17 in the Appendix) reveal that the origin of QA data plays a critical role. Specifically, OpenR1-Math-220K and OMI2 are derived from structured downstream datasets (e.g., GSM8K, MATH), while MegaMath-QA is sourced from less curated web documents. These differences in data source and distribution substantially impact downstream RL performance, highlighting the importance of distributional alignment between mid-training data and downstream tasks. In light of this, we adopt OpenMathInstruct-2, OpenR1-Math-220K (and further adopt the a-m-team's distilled dataset <sup>4</sup>), and NuminaMath-1.5 <sup>5</sup> as our primary QA datasets for the decay stage, due to their closer resemblance to competition-style, reasoning-intensive benchmarks.

**Identifying the Optimal QA Ratio** Across our ablation studies (also see Figure 17), we observe a consistent trend: increasing the QA data ratio leads to improved RL performance, which aligns with expectations due to the format similarity with RL objectives. However, gains begin to plateau beyond a 30% QA mix, with 40% showing diminishing returns across most benchmarks. We attribute this to token redundancy and lack of diversity at higher QA proportions. As a result, we adopt 30% QA as the optimal ratio, balancing performance and data efficiency.

**Takeaway:** The distribution gap between QA data and downstream tasks notably affects RL performance. Therefore, we introduce distribution-aligned QA data during the decay stage and find that a 30% QA ratio offers the best trade-off between RL performance and QA data scale.

#### 4.2.2. Final Decay Recipe

For the decay stage, we explore two learning rate (LR) scheduler variants:

- 1. **Constant LR decay**, where the LR remains fixed at 10% of the final LR used in the stable stage.
- 2. Cosine decay to 10%, where the LR gradually decays to 10% of the stable-stage final LR.

Based on mid-training evaluation results, the cosine decay strategy demonstrates more consistent performance. We therefore adopt it as the default scheduler for the decay stage, with hyperparameters detailed in Table 4. During the decay stage, we branch the mid-training into three distinct variants based on data composition: OctoThinker-Long (long-reasoning data), OctoThinker-Short (short-reasoning data), OctoThinker-Hybrid (a mix of both) with decayed

Llama-3.2-1B / 3B / 8B Hyper-parameter Context Length 8,192 Batch Size 512 Max Steps 5,000 Warmup Steps 0 Weight Decay 0.1 AdamW Optimizer Cosine Decay LR Scheduler  $5e-5 \rightarrow 5e-6 / 2e-5 \rightarrow 2e-6 / 1e-5 \rightarrow 1e-6$ 

<span id="page-12-2"></span>**Table 4** | Hyper-parameters for decay stage.

learning rate. The corresponding data mixtures are shown in Table 5.

<span id="page-12-0"></span><sup>4</sup>https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M

<span id="page-12-1"></span><sup>&</sup>lt;sup>5</sup>https://huggingface.co/datasets/AI-MO/NuminaMath-1.5

Table 5 | Specific data mixture for each branch in the decay stage

### <span id="page-13-0"></span>(a) Long Branch Mixture

### (b) Short Branch Mixture

### (c) Hybrid Branch Mixture

| Dataset                   | Weight |
|---------------------------|--------|
| DCLM-Baseline             | 0.05   |
| Instruction Following     | 0.10   |
| MegaMath-Web-Pro          | 0.55   |
| Open R1                   | 0.15   |
| AM-DeepSeek-Distilled-40M | 0.15   |
|                           |        |

| Dataset               | Weight |
|-----------------------|--------|
| DCLM-Baseline         | 0.05   |
| Instruction Following | 0.10   |
| MegaMath-Web-Pro      | 0.55   |
| MegaMath-QA           | 0.025  |
| OpenMathInstruct2     | 0.175  |
| NuminaMath1.5         | 0.10   |

| Dataset               | Weight |
|-----------------------|--------|
| DCLM-Baseline         | 0.05   |
| Instruction Following | 0.10   |
| MegaMath-Web-Pro      | 0.55   |
| OpenMathInstruct2     | 0.10   |
| NuminaMath1.5         | 0.10   |
| Open R1               | 0.10   |

#### 4.3. Evaluation on OctoThinker-Base Series

We evaluate the performance of each branch on 13 mathematical benchmarks, alongside the original Llama base model and the model after stable-stage mid-training. As shown in Table 6,7,8, across all sizes, each OctoThinker branch demonstrates a noticeable 10%-20% improvement over the original base model and consistent gains over the stable-stage model. Notably, random and poor performance on challenging competition benchmarks highlights the necessity of post-training. Overall, these results reinforce our view that OctoThinker-Base series provide a strong foundation for studying RL scaling with solid reasoning capabilities.

**Table 6** | Evaluation results of Llama-3.2-1B and OctoThinker-1B series.

<span id="page-13-1"></span>

|            |                         | _            | O      | ctoThinker-1B-Base |        |       |
|------------|-------------------------|--------------|--------|--------------------|--------|-------|
| Benchmarks |                         | Llama-3.2-1B | Stable | Long               | Hybrid | Short |
|            | GSM8K (8-shot)          | 7.66         | 30.93  | 37.15              | 42.38  | 44.88 |
|            | MATH500 (4-shot)        | 4.60         | 17.40  | 16.40              | 26.40  | 27.80 |
| Core       | Olympiad Bench (4-shot) | 0.89         | 2.96   | 3.41               | 5.48   | 3.85  |
|            | AMC23 (0-shot)          | 0.00         | 10.00  | 7.50               | 10.00  | 10.00 |
|            | Average                 | 3.29         | 15.32  | 16.12              | 21.07  | 21.63 |
|            | MATH (4-shot)           | 4.34         | 18.26  | 21.74              | 28.50  | 29.98 |
|            | SAT MATH (4-shot)       | 12.50        | 46.88  | 31.25              | 56.25  | 46.88 |
|            | MathQA (8-shot)         | 12.20        | 24.80  | 33.20              | 36.90  | 36.70 |
|            | MMLU STEM (4-shot)      | 19.90        | 35.59  | 36.45              | 38.60  | 37.91 |
| Other      | OCW Course (4-shot)     | 4.41         | 6.25   | 4.04               | 6.25   | 6.62  |
| Other      | MAWPS (8-shot)          | 43.05        | 79.47  | 83.15              | 88.57  | 88.09 |
|            | SVAMP (8-shot)          | 20.90        | 47.10  | 55.80              | 63.20  | 61.20 |
|            | ASDiv (8-shot)          | 34.53        | 69.96  | 72.55              | 75.30  | 75.26 |
|            | TabMWP (8-shot)         | 24.40        | 45.10  | 50.10              | 51.60  | 51.20 |
|            | Average                 | 19.58        | 41.49  | 43.14              | 49.46  | 48.20 |

**Table 7** | Evaluation results of Llama-3.2-3B and OctoThinker-3B series.

<span id="page-14-0"></span>

|            |                         |              |        | OctoThinker-3B-Base |        |       |
|------------|-------------------------|--------------|--------|---------------------|--------|-------|
| Benchmarks |                         | Llama-3.2-3B | Stable | Long                | Hybrid | Short |
|            | GSM8K (8-shot)          | 30.48        | 55.95  | 56.10               | 64.37  | 63.31 |
|            | MATH500 (4-shot)        | 7.40         | 22.40  | 25.80               | 30.80  | 31.40 |
| Core       | Olympiad Bench (4-shot) | 2.07         | 3.41   | 4.74                | 4.00   | 4.74  |
|            | AMC23 (0-shot)          | 2.50         | 5.00   | 7.50                | 10.00  | 2.50  |
|            | Average                 | 10.61        | 21.69  | 23.54               | 27.29  | 25.49 |
|            | MATH (4-shot)           | 8.24         | 24.86  | 29.98               | 31.76  | 32.70 |
|            | SAT MATH (4-shot)       | 25.00        | 59.38  | 65.63               | 59.38  | 53.13 |
|            | MathQA (8-shot)         | 18.20        | 39.50  | 45.40               | 47.50  | 49.80 |
|            | MMLU STEM (4-shot)      | 38.63        | 46.32  | 48.11               | 49.73  | 48.87 |
|            | OCW Course (4-shot)     | 5.51         | 11.40  | 11.40               | 8.46   | 9.19  |
| Other      | MAWPS (8-shot)          | 79.90        | 89.69  | 91.67               | 94.24  | 93.51 |
|            | SVAMP (8-shot)          | 52.40        | 68.40  | 69.10               | 78.00  | 77.30 |
|            | ASDiv (8-shot)          | 60.09        | 79.59  | 79.91               | 82.80  | 82.26 |
|            | TabMWP (8-shot)         | 48.30        | 55.60  | 56.40               | 57.80  | 60.00 |
|            | Average                 | 37.36        | 52.75  | 55.29               | 56.63  | 56.31 |

**Table 8** | Evaluation results of Llama-3.1-8B and OctoThinker-8B series.

<span id="page-14-1"></span>

|       |                         |              | OctoThinker-8B-Base |       |        |       |
|-------|-------------------------|--------------|---------------------|-------|--------|-------|
|       | Benchmarks              | Llama-3.1-8B | Stable              | Long  | Hybrid | Short |
|       | GSM8K (8-shot)          | 55.11        | 71.27               | 72.48 | 77.41  | 77.10 |
|       | MATH500 (4-shot)        | 20.80        | 34.40               | 37.80 | 42.60  | 38.60 |
| Core  | Olympiad Bench (4-shot) | 3.56         | 9.78                | 11.85 | 4.74   | 10.07 |
|       | AMC23 (0-shot)          | 5.00         | 0.00                | 5.00  | 5.00   | 7.50  |
|       | Average                 | 21.12        | 28.86               | 31.78 | 32.44  | 33.32 |
|       | MATH (4-shot)           | 21.36        | 37.00               | 41.98 | 44.82  | 38.54 |
|       | SAT MATH (4-shot)       | 53.13        | 81.25               | 81.25 | 87.50  | 87.50 |
|       | MathQA (8-shot)         | 36.00        | 58.20               | 62.80 | 60.50  | 62.80 |
|       | MMLU STEM (4-shot)      | 54.44        | 62.03               | 63.75 | 64.08  | 64.38 |
|       | OCW Course (4-shot)     | 12.87        | 18.38               | 16.18 | 15.07  | 13.97 |
| Other | MAWPS (8-shot)          | 90.75        | 93.08               | 94.43 | 95.98  | 95.54 |
|       | SVAMP (8-shot)          | 70.50        | 79.50               | 82.40 | 86.10  | 86.40 |
|       | ASDiv (8-shot)          | 72.10        | 83.79               | 83.57 | 84.47  | 85.33 |
|       | TabMWP (8-shot)         | 63.10        | 67.90               | 70.10 | 68.90  | 71.60 |
|       | Average                 | 52.69        | 64.57               | 66.27 | 67.49  | 67.34 |

# **5. OctoThinker-Zero Families: RL Scaling with Diverse Thinking Behaviors**

We further train all OctoThinker base models—spanning different decay branches and model sizes (1B and 3B)—through a reinforcement learning stage, following our previously established setup. This yields a family of models optimized specifically for mathematical reasoning. As in the decay stage, the final RL-tuned models fall into three categories, each reflecting the data mixture used during decay and the distinct behaviors shaped during RL: OctoThinker-Short-Zero, OctoThinker-Hybrid-Zero, and OctoThinker-Long-Zero. The training dynamics of these models are shown in Figure [12](#page-15-0)[,13.](#page-15-1) The OctoThinker-Long branch tends to produce longer responses—within a controlled range—compared to other branches. While it slightly underperforms at the 1B scale, it demonstrates stronger performance as model size increases, such as at 3B.

<span id="page-15-0"></span>> **[图片提取文字 (无描述)]:**
> **MATH500** GSM8K OlympiadBench AMC23 Accuracy (%) 2.4 1.6 1.6 Response Length(K) 2.4 1.8 1.2 1.6 1.2 0.8 0.8 0.6 0.4 0.0 0.0 Step Step Step Step OctoThinker-Short-1B-Base OctoThinker-Hybrid-1B-Base OctoThinker-Long-1B-Base
![](_page_15_Figure_3.jpeg)

**Figure 12** | The RL training dynamics across different branches for OctoThinker-1B series

<span id="page-15-1"></span>> **[图片提取文字 (无描述)]:**
> **Olympiad Bench** MATH500 GSM8K AMC23 Accuracy (%) 1.6 0.60 Response Length(K) 1.5 1.00 1.2 0.45 0.75 1.0 8.0 0.30 0.50 0.5 0.4 0.15 0.25 0.0 Step Step Step Step OctoThinker-Short-3B-Base OctoThinker-Hybrid-3B-Base OctoThinker-Long-3B-Base
![](_page_15_Figure_5.jpeg)

**Figure 13** | The RL training dynamics across different branches for OctoThinker-3B series

<span id="page-16-0"></span>> **[图片提取文字 (无描述)]:**
> **Olympiad Bench** MATH500 GSM8K AMC23 Accuracy (%) Step Step Step Step OctoThinker-Long-3B-Base OctoThinker-Short-3B-Base OctoThinker-Hybrid-3B-Base Llama-3.2-3B-Base Qwen2.5-3B-Base
![](_page_16_Figure_1.jpeg)

**Figure 14** | RL training dynamics among Llama-3.2-3B-Base, OctoThinker series and Qwen2.5-Base.

**OctoThinker vs. Qwen2.5** An important question we investigate is: *To what extent can our OctoThinker models close the performance gap between the Llama-3.2 series and the stronger Qwen2.5 models in the RL setting?* To address this, we compare three 3B-scale base models: Llama-3.2-3B-Base, OctoThinker-Long-3B-Base, and Qwen2.5-3B-Base. As illustrated in Figure [14,](#page-16-0) during the reinforcement learning phase, OctoThinker-Long-3B consistently outperforms the original Llama-3.2-3B model. Remarkably, it reaches performance on par with Qwen2.5-3B, a model known for its strong reasoning capabilities and extensive pre-training, while the hybrid and short branches are marginally inferior, especially on challenging benchmarks. Overall, these results highlight the substantial gains introduced by our mid-training strategy and confirm that OctoThinker effectively narrows the performance gap, elevating Llama-3.2 models to a new level of competitiveness in mathematical reasoning tasks.

# **6. Related Works**

**Understanding RL along with Language Models** Large-scale RL has driven the major advances in language models on reasoning-intensive tasks, such as competition-level math (e.g., AIME), exemplified by OpenAI's o1 [\(OpenAI et al.,](#page-21-0) [2024\)](#page-21-0), o3 [\(OpenAI,](#page-21-1) [2025\)](#page-21-1) and DeepSeek's R1 [\(Guo et al.,](#page-19-0) [2025\)](#page-19-0). A wave of follow-up studies [\(Hu et al.,](#page-20-0) [2025;](#page-20-0) [Luo et al.,](#page-21-8) [2025;](#page-21-8) [Yu et al.,](#page-23-7) [2025;](#page-23-7) [Zeng et al.,](#page-23-1) [2025,](#page-23-1) *inter alia*) explored RL on smaller language models (less than 100B parameters), yet these successes are overwhelmingly limited to Qwen family. In contrast, replicating such results on other major model families—e.g., Llama-has proven difficult [\(Gandhi et al.,](#page-19-1) [2025;](#page-19-1) [Liu et al.,](#page-21-4) [2025\)](#page-21-4). The opacity of pre-training pipelines hinders our understanding of how pre-training interacts with RL scaling, prompting some unconventional investigations [\(Shao et al.,](#page-22-4) [2025;](#page-22-4) [Wang et al.,](#page-22-5) [2025\)](#page-22-5). For instance, [Wang et al.](#page-22-5) [\(2025\)](#page-22-5) showed that even one-shot prompting can enhance reasoning in Qwen, but yields minimal gains in Llama. The underlying science remains essential but largely unexplored. Our work takes a step toward filling this gap by performing controlled mid-training interventions on Llama models, revealing key factors that enable effective RL scaling. Building on these insights, we introduce OctoThinker via a two-stage mid-training strategy (over 200B tokens), followed by RL training, yielding models that match Qwen's performance at the same scale.

**Curation of Math Pre-training Corpora** Pre-training corpora are foundational to language models, especially for math reasoning tasks where large-scale mid-training is infeasible without high-quality, domain-specific data. Early open-source efforts—such as OpenWebMath [\(Paster et al.,](#page-22-6) [2024\)](#page-22-6), Math-Pile [\(Wang et al.,](#page-23-8) [2024\)](#page-23-8), InfiMM-Web-Math [\(Han et al.,](#page-19-5) [2024\)](#page-19-5), and FineMath [\(Allal et al.,](#page-18-0) [2025\)](#page-18-0)—have made meaningful progress but remain constrained in scale, typically under 100B tokens. The release of MegaMath [\(Zhou et al.,](#page-23-4) [2025\)](#page-23-4) marked a turning point, enabling scalable mid-training in this work. Building on its foundation, we curated a new reasoning-intensive and carefully refined math corpus, MegaMath-Web-Pro-Max, which exceeds 70B tokens and matches the quality of MegaMath-Web-Pro. This corpus powers the first stage of our mid-training of OctoThinker and will be released to support the broader open-source community.

# **7. Conclusion**

In this work, we investigate why base models like Llama and Qwen exhibit divergent behaviors during reinforcement learning for reasoning and demonstrated that mid-training can play a decisive role. Our findings show that high-quality, reasoning-intensive corpora—especially those like MegaMath-Web-Pro—can substantially improve RL stability and effectiveness. Building on these insights, we introduce a two-stage mid-training strategy that transforms Llama into a more RL-scalable foundation model. The resulting OctoThinker models achieve strong performance across mathematical reasoning tasks, closing the gap with RL-friendly model families. We hope this work provides a foundation for designing future base models better aligned with the demands of reasoning-centric RL.

# **Future Work**

We will actively explore more in the future, include: (1) curating higher-quality math corpora to further enhance mid-training; (2) designing RL-friendly base models using open recipes without distillation from those powerful long CoT reasoning models; (3) disentangling QA format and content to better understand their individual contributions; and (4) extending the OctoThinker families with additional branches, such as *tool-integrated reasoning*. We believe these efforts will provide deeper insights into the interplay between pre-training and reinforcement learning.

# **References**

<span id="page-17-0"></span>Marah I Abdin, Jyoti Aneja, Harkirat S. Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report. *CoRR*, abs/2412.08905, 2024a. doi: 10.48550/ARXIV.2412.08905. URL <https://doi.org/10.48550/arXiv.2412.08905>.

<span id="page-17-1"></span>Marah I Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat S. Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp A. Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone. *CoRR*, abs/2404.14219, 2024b. doi: 10.48550/ARXIV.2404.14219. URL <https://doi.org/10.48550/arXiv.2404.14219>.

- <span id="page-18-0"></span>Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big - data-centric training of a small language model. *CoRR*, abs/2502.02737, 2025. doi: 10.48550/ARXIV.2502.02737. URL [https://doi.org/10.48550/](https://doi.org/10.48550/arXiv.2502.02737) [arXiv.2502.02737](https://doi.org/10.48550/arXiv.2502.02737).
- <span id="page-18-3"></span>Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2357–2367, 2019.
- <span id="page-18-2"></span>Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=4WnqRR915j>.
- <span id="page-18-5"></span>Akhiad Bercovich, Tomer Ronen, Talor Abramovich, Nir Ailon, Nave Assaf, Mohammad Dabbah, Ido Galil, Amnon Geifman, Yonatan Geifman, Izhak Golan, Netanel Haber, Ehud Karpas, Roi Koren, Itay Levy, Pavlo Molchanov, Shahar Mor, Zach Moshe, Najeeb Nabwani, Omri Puny, Ran Rubin, Itamar Schen, Ido Shahaf, Oren Tropp, Omer Ullman Argov, Ran Zilberstein, and Ran El-Yaniv. Puzzle: Distillation-based NAS for inference-optimized llms. *CoRR*, abs/2411.19146, 2024. doi: 10.48550/ARXIV.2411.19146. URL <https://doi.org/10.48550/arXiv.2411.19146>.
- <span id="page-18-6"></span>Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. *arXiv preprint arXiv:2505.00949*, 2025.
- <span id="page-18-8"></span>Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024.
- <span id="page-18-1"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-18-7"></span>Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 3029– 3051. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.EMNLP-MAIN.183. URL <https://doi.org/10.18653/v1/2023.emnlp-main.183>.
- <span id="page-18-9"></span>Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. *arXiv preprint arXiv:2305.14233*, 2023b.
- <span id="page-18-4"></span>Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zi-Yan Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong Feng, Xin Mao, Man Tsung Yeung, Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Kydl'ivcek, Ze-Xuan Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Nirattisai Thongchim, Taechawat Konkaew,

Narong Borijindargoon, Anh Dao, Matichon Maneegard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphaibun, Hoang H. Tran, Mike Zhang, Shiqi Chen, Tianyu Pang, Chao Du, Xinyi Wan, Wei Lu, and Min Lin. Sailor2: Sailing in south-east asia with inclusive multilingual llms. *ArXiv*, abs/2502.12982, 2025. URL <https://arxiv.org/abs/2502.12982>.

<span id="page-19-2"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. *CoRR*, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL <https://doi.org/10.48550/arXiv.2407.21783>.

<span id="page-19-1"></span>Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. *arXiv preprint arXiv:2503.01307*, 2025.

<span id="page-19-0"></span>Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

<span id="page-19-5"></span>Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran He, Zhenheng Yang, and Quanzeng You. InfiMM-webmath-40b: Advancing multimodal pretraining for enhanced mathematical reasoning. In *The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24*, 2024. URL <https://openreview.net/forum?id=Twzrpa6V2o>.

<span id="page-19-3"></span>Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024*, pages 3828–3850. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.211. URL [https://doi.org/10.18653/](https://doi.org/10.18653/v1/2024.acl-long.211) [v1/2024.acl-long.211](https://doi.org/10.18653/v1/2024.acl-long.211).

<span id="page-19-4"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.

- <span id="page-20-0"></span>Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Openreasoner-zero: An open source approach to scaling up reinforcement learning on the base model. *arXiv preprint arXiv:2503.24290*, 2025.
- <span id="page-20-4"></span>Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies. *CoRR*, abs/2404.06395, 2024a. doi: 10.48550/ARXIV.2404.06395. URL <https://doi.org/10.48550/arXiv.2404.06395>.
- <span id="page-20-7"></span>Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint arXiv:2404.06395*, 2024b.
- <span id="page-20-5"></span>HuggingFace. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL [https:](https://github.com/huggingface/open-r1) [//github.com/huggingface/open-r1](https://github.com/huggingface/open-r1).
- <span id="page-20-3"></span>Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In *Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies*, pages 1152–1157, 2016.
- <span id="page-20-6"></span>Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. *CoRR*, abs/2411.15124, 2024a. doi: 10.48550/ARXIV.2411.15124. URL <https://doi.org/10.48550/arXiv.2411.15124>.
- <span id="page-20-8"></span>Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024b.
- <span id="page-20-2"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. URL [http://papers.nips.cc/paper\\_files/paper/2022/hash/](http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html) [18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html).
- <span id="page-20-9"></span>Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Guha, Sedrick Scott Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. *Advances in Neural Information Processing Systems*, 37: 14200–14282, 2024.
- <span id="page-20-1"></span>Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. *CoRR*, abs/2305.20050, 2023. doi: 10.48550/ARXIV.2305.20050. URL [https://doi.org/10.48550/arXiv.2305.](https://doi.org/10.48550/arXiv.2305.20050) [20050](https://doi.org/10.48550/arXiv.2305.20050).

- <span id="page-21-2"></span>Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-21-4"></span>Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. *arXiv preprint arXiv:2503.20783*, 2025.
- <span id="page-21-6"></span>Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In *International Conference on Learning Representations (ICLR)*, 2023.
- <span id="page-21-8"></span>Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. [Notion Blog,](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) 2025. Notion Blog.
- <span id="page-21-3"></span>Meta, Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-21-5"></span>Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 975–984, 2020.
- <span id="page-21-7"></span>Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James Validad Miranda, Jacob Daniel Morrison, Tyler C. Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Chris Wilhelm, Michael Wilson, Luke S. Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hanna Hajishirzi. 2 olmo 2 furious. *ArXiv*, abs/2501.00656, 2025. URL <https://arxiv.org/abs/2501.00656>.
- <span id="page-21-1"></span>OpenAI. Introducing openai o3 and o4-mini | openai, April 2025. URL [https://openai.com/](https://openai.com/index/introducing-o3-and-o4-mini/) [index/introducing-o3-and-o4-mini/](https://openai.com/index/introducing-o3-and-o4-mini/).
- <span id="page-21-0"></span>OpenAI, Ahmed El-Kishky, Daniel Selsam, Francis Song, Giambattista Parascandolo, Hongyu Ren, Hunter Lightman, Hyung Won, Ilge Akkaya, Ilya Sutskever, Jason Wei, Jonathan Gordon, Karl Cobbe, Kevin Yu, Lukasz Kondraciuk, Max Schwarzer, Mostafa Rohaninejad, Noam Brown, Shengjia Zhao, Trapit Bansal, Vineet Kosaraju, Wenda Zhou Leadership, Jakub W. Pachocki, Jerry Tworek, Liam Fedus, Lukasz Kaiser, Mark Chen, Szymon Sidor, Wojciech Zaremba, Alex Karpenko, Alexander Wei, Allison Tam, Ananya Kumar, Andre Saraiva, Andrew Kondrich, An drey Mishchenko, Ashvin Nair, B. Ghorbani, Brandon McKinzie, Chak Bry don Eastman, Ming Li, Chris Koch, Dan Roberts, David Dohan, David Mély, Dimitris Tsipras, Enoch Cheung, Eric Wallace, Hadi Salman, Haim ing Bao, Hessam Bagher-inezhad, Ilya Kostrikov, Jiacheng Feng, John Rizzo, Karina Nguyen, Kevin Lu, Kevin R. Stone, Lorenz Kuhn, Mason Meyer, Mikhail Pavlov, Nat McAleese, Oleg Boiko, Oleg Murk, Peter Zhokhov, Randall Lin, Raz Gaon, Rhythm Garg, Roshan James, Rui Shu, Scott McKinney, Shibani Santurkar, Suchir Balaji, Taylor Gordon, Thomas Dimson, Weiyi Zheng, Aaron Jaech, Adam Lerer, Aiden Low, Alex Carney, Alexander Neitz, Alexander Prokofiev, Benjamin Sokolowsky, Boaz Barak, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Camillo Lugaresi, Chelsea Voss, Chen Shen, Chris Orsinger, Daniel Kappler, Daniel Levy, Doug Li, Eben Freeman, Edmund Wong, Fan Wang, Felipe Petroski Such, Foivos Tsimpourlas, Geoff Salmon, Gildas Chabot, Guillaume Leclerc, Hart Andrin, Ian O'Connell, Ignasi Ian Osband, Clavera Gilaberte, Jean Harb, Jiahui Yu,

Jiayi Weng, Joe Palermo, John Hallman, Jonathan Ward, Julie Wang, Kai Chen, Katy Shi, Keren Gu-Lemberg, Kevin Liu, Leo Liu, Linden Li, Luke Metz, Maja Trebacz, Manas R. Joglekar, Marko Tintor, Melody Y. Guan, Mengyuan Yan, Mia Glaese, Michael Malek, Michelle Fradin, Mo Bavarian, Nikolas A. Tezak, Ofir Nachum, Paul Ashbourne, Pavel Izmailov, Raphael Gontijo Lopes, Reah Miyara, Reimar H. Leike, Robin Brown, Ryan Cheu, Ryan Greene, Saachi Jain, Scottie Yan, Shengli Hu, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Suvansh Sanjeev, Tao Wang, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Tianhao Zheng, T. Garipov, Valerie Qi, Vitchyr H. Pong, Vlad Fomenko, Yinghai Lu, Yining Chen, Yu Bai, Yuchen He, Yuchen Zhang, Zheng Shao, Zhuohan Li, Lauren Yang, Mianna Chen, Aidan Clark, Jieqi Yu, Kai Xiao, Sam Toizer, Sandhini Agarwal, Safety Research, Andrea Vallone, Chong Zhang, Ian Kivlichan, Meghan Shah, Sam Toyer, Shraman Ray Chaudhuri, Stephanie Lin, Adam Richardson, Andrew Duberstein, Charles de Bourcy, Dragos Oprica, Florencia Leoni, Made laine Boyd, Matt Jones, Matt Kaufer, Mehmet Ali Yatbaz, Mengyuan Xu, Mike McClay, Mingxuan Wang, Trevor Creech, Vinnie Monaco, Erik Ritter, Evan Mays, Joel Parish, Jonathan Uesato, Leon Maksin, Michele Wang, Miles Wang, Neil Chowdhury, Olivia Watkins, Patrick Chao, Rachel Dias, Samuel Miserendino, Red Teaming, Lama Ahmad, Michael Lampe, Troy Peterson, and Joost Huizinga. Openai o1 system card. *ArXiv*, abs/2412.16720, 2024. URL <https://arxiv.org/abs/2412.16720>.

<span id="page-22-6"></span>Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=jKHmjlpViu>.

<span id="page-22-2"></span>Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2080–2094, 2021.

<span id="page-22-4"></span>Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hanna Hajishirzi, Pang Wei Koh, and Luke S. Zettlemoyer. Spurious rewards: Rethinking training signals in rlvr. 2025. URL <https://arxiv.org/abs/2506.10947>.

<span id="page-22-1"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

<span id="page-22-0"></span>Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv:2409.19256*, 2024.

<span id="page-22-3"></span>Alan Wake, Bei Chen, C. X. Lv, Chao Li, Chengen Huang, Chenglin Cai, Chujie Zheng, Daniel Cooper, Fan Zhou, Feng Hu, Guoyin Wang, Heng Ji, Howard Qiu, Jiangcheng Zhu, Jun Tian, Katherine Su, Lihuan Zhang, Liying Li, Ming Song, Mou Li, Peng Liu, Qicheng Hu, Shawn Wang, Shijun Zhou, Shiming Yang, Shiyong Li, Tianhang Zhu, Wen Xie, Xiang He, Xiaobo Chen, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Yanpeng Li, Yongke Zhao, Yongzhen Luo, Yuchi Xu, Yuxuan Sha, Zhaodong Yan, Zhiyuan Liu, Zirui Zhang, and Zonghong Dai. Yi-lightning technical report. *CoRR*, abs/2412.01253, 2024. doi: 10.48550/ARXIV.2412.01253. URL <https://doi.org/10.48550/arXiv.2412.01253>.

<span id="page-22-5"></span>Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement learning for reasoning in large language models with one training example. *CoRR*, abs/2504.20571, 2025. doi: 10.48550/ARXIV.2504.20571. URL [https://doi.org/10.48550/](https://doi.org/10.48550/arXiv.2504.20571) [arXiv.2504.20571](https://doi.org/10.48550/arXiv.2504.20571).

- <span id="page-23-8"></span>Zengzhi Wang, Xuefeng Li, Rui Xia, and Pengfei Liu. Mathpile: A billion-token-scale pretraining corpus for math. In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024. URL <https://openreview.net/forum?id=RSvhU69sbG>.
- <span id="page-23-0"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. URL [http://papers.nips.cc/paper\\_files/paper/2022/hash/](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) [9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html).
- <span id="page-23-5"></span>An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via selfimprovement. *ArXiv*, abs/2409.12122, 2024a. URL <https://arxiv.org/abs/2409.12122>.
- <span id="page-23-2"></span>An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.
- <span id="page-23-3"></span>Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan, and Zekun Wang. Qwen2.5 technical report. *ArXiv*, abs/2412.15115, 2024b. URL <https://arxiv.org/abs/2412.15115>.
- <span id="page-23-7"></span>Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. *CoRR*, abs/2503.14476, 2025. doi: 10.48550/ARXIV.2503. 14476. URL <https://doi.org/10.48550/arXiv.2503.14476>.
- <span id="page-23-1"></span>Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. *arXiv preprint arXiv:2503.18892*, 2025.
- <span id="page-23-6"></span>Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=Bl8u7ZRlbM>.
- <span id="page-23-4"></span>Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P. Xing. Megamath: Pushing the limits of open math corpora. *ArXiv*, abs/2504.02807, 2025. URL <https://arxiv.org/abs/2504.02807>.

# **Appendix**

### <span id="page-24-0"></span>**Scoring Prompt of Usefulness for Studying Mathematics**

Evaluate the following text extract for its potential usefulness for studying mathematics up to high school and early undergraduate levels. Use the following 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract contains some mathematical content, even if it's not very useful for studying, or if it contains non-academic content such as advertisements and generated pages for converting weight and currencies.
- Add another point if the extract touches on mathematical topics, even if it's poorly written if it's too complex such as an academic paper that is too advanced.
- Award a third point if the extract demonstrates problem solving or logical reasoning in a mathematical context, even if it lacks step-by-step explanations.
- Grant a fourth point if the extract is at an appropriate level (up to high school and early undergraduate levels) and contains clear mathematical deductions and step-by-step solutions to mathematical problems. It should be similar to a chapter from a textbook or a tutorial.
- Give a fifth point if the extract is outstanding in its educational value for teaching and studying mathematics in middle school and high school. It should include very detailed and easy to follow explanations.

Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria.

The text extract: " <document> "

After examining the extract:

- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: Final score: <total points>.

**Figure 15** | Scoring prompt in FineMath [\(Allal et al.,](#page-18-0) [2025\)](#page-18-0) of usefulness for studying mathematics.

#### <span id="page-24-1"></span>**Web Text Refinement Prompt**

#### Task:

- Carefully analyze the provided text to extract key facts, concrete details, important numbers, and core concepts.
- Remove any irrelevant or noisy information, and reorganize the content into a logically structured, information-dense, and concise version that is easy to learn from. Output only the refined text.
- Strive to maintain the original length as much as possible (avoid excessive shortening).
- Refine multiple choice questions and answers if any.

Text:

<EXAMPLE>

Just output the refined text, no other text.

**Figure 16** | Web text refinement prompt used in MegaMath-Web-Pro [\(Zhou et al.,](#page-23-4) [2025\)](#page-23-4)

<span id="page-25-0"></span>> **[图片提取文字 (无描述)]:**
> **Olympiad Bench** MATH500 AMC23 GSM8K 25 40 60 20 30 Accuracy (%) 15 8 30 10 20 10 10 400 400 100 200 300 400 200 300 100 200 300 100 300 1.00 1.6 3.2 Response Length(K) 0.75 1.2 1.5 2.4 0.50 8.0 1.0 1.6 0.25 0.4 0.5 8.0 0.0 0.0 100 200 300 400 100 200 300 400 200 300 400 300 Step Step Step Step MegaMath-QA: 10% MegaMath-QA: 30% ---- OctoThinker-3B-Base-Stable MegaMath-QA: 20% MegaMath-QA: 40% Olympiad Bench AMC23 MATH500 GSM8K 60 80 20 50 30 Accuracy (%) 40 25 20 40 10 15 20 20 10 150 50 100 150 200 50 100 200 50 100 150 200 50 100 150 2.0 3.2 2.4 Response Length(K) 1.6 1.2 2.4 1.8 1.2 0.8 1.6 1.2 8.0 0.4 0.8 100 200 50 100 150 200 100 150 200 100 150 200 Step Step Step Step Open-R1-Math-220K: 10% — Open-R1-Math-220K: 30% ---- OctoThinker-3B-Base-Stable AMC23 OlympiadBench MATH500 GSM8K 55 30 80 17.5 50 25 70 Accuracy (%) 15.0 45 20 60 12.5 40 15 10.0 50 35 10 7.5 30 40 25 30 200 300 100 200 100 200 300 400 400 100 300 400 100 200 300 400 0.18 1.00 0.30 0.48 Response Length(K) 0.16 0.75 0.25 0.14 0.40 0.50 0.20 0.12 0.32 0.25 0.15 0.10 0.24 100 200 200 300 400 100 300 400 100 200 300 400 100 200 300 400 Step Step Step Step OMI2: 10% OMI2: 30% OctoThinker-3B-Base-Stable OMI2: 20% OMI2: 40%
![](_page_25_Figure_1.jpeg)

**Figure 17** | RL dynamics under different QA datasets and mixing ratios during the decay stage.