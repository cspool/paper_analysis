# TIMESEARCH-R: ADAPTIVE TEMPORAL SEARCH FOR LONG-FORM VIDEO UNDERSTANDING VIA SELF-VERIFICATION REINFORCEMENT LEARNING

Junwen Pan1<sup>∗</sup> Qizhe Zhang1,<sup>2</sup> <sup>∗</sup> Rui Zhang<sup>1</sup> Ming Lu<sup>2</sup> † Xin Wan<sup>1</sup> Yuan Zhang1,<sup>2</sup> Chang Liu<sup>1</sup> Qi She<sup>1</sup> <sup>B</sup> <sup>1</sup> ByteDance <sup>2</sup> School of Computer Science, Peking University {panjunwen,sheqi.roger}@bytedance.com

# ABSTRACT

Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose TimeSearch-R, which reformulates temporal search as interleaved text-video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning. Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves significant improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, as well as long-form video understanding benchmarks like VideoMME and MLVU. Notably, TimeSearch-R establishes a new state-of-the-art on LongVideoBench with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced video reasoning model Video-R1. *Our code is available at [https://github.com/Time-Search/TimeSearch-R.](https://github.com/Time-Search/TimeSearch-R)*

# 1 INTRODUCTION

Long-form video understanding requires models to navigate through tens of thousands of frames to identify the most relevant information for answering specific questions [\(Fu et al., 2024;](#page-9-0) [Zhou](#page-11-0) [et al., 2024;](#page-11-0) [Wu et al., 2024\)](#page-11-1). Temporal search lies at the heart of making long-video understanding both accurate and interpretable [\(Park et al., 2025;](#page-10-0) [Li et al., 2023;](#page-10-1) [Ye et al., 2025\)](#page-11-2). In contrast to the human visual system, which conducts adaptive temporal search [\(Yarbus, 1967;](#page-11-3) [Hayhoe & Ballard,](#page-9-1) [2005\)](#page-9-1), current large video-language models (LVLMs) primarily rely on hand-crafted search strategies with static frame sampling [\(Lin et al., 2023;](#page-10-2) [Bai et al., 2025a;](#page-9-2) [Feng et al., 2025\)](#page-9-3). Humans naturally alternate between broad scanning and targeted inspection, refining their focus iteratively based on intermediate findings [\(Castelhano & Henderson, 2007;](#page-9-4) [Henderson & Hayes, 2017\)](#page-9-5). In contrast, existing methods are limited to a fixed set of frames established before the reasoning process begins. This design presents a fundamental contradiction: video reasoning is a dynamic process where temporal search interleaves with video reasoning; however, the video frames accessible to the model remain fixed from the outset, ultimately hindering effective reasoning.

<sup>∗</sup>Equal contribution. † Project lead. <sup>B</sup>Corresponding author.

![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>Figure 1: (a) Different paradigms of temporal search. Previous works such as VideoAgent [\(Wang](#page-11-4) [et al., 2024\)](#page-11-4) and T\* [\(Ye et al., 2025\)](#page-11-2) predominantly rely on handcrafted workflows, resulting in suboptimal strategies. Our approach adopts end-to-end reinforcement learning, enabling the model to learn optimal search strategies directly from data. (b) Interleaved text-video thinking process. We reformulate the temporal search task as an interleaved text-video thinking process, where the temporal search is seamlessly interleaved into the reasoning process.

Inspired by the gap between human cognition and model reasoning, recent studies have explored interactive video agents that attempt to bridge this divide through multi-turn temporal search, as illustrated in Figure [1](#page-1-0) (a). VideoAgent [\(Wang et al., 2024\)](#page-11-4) first employs a large language model (LLM) as the central agent, which iteratively calls tools like vision-language models (VLMs) and CLIP [\(Radford et al., 2021\)](#page-10-3) for frame captioning and retrieval, and then aggregates information in the textual modality to perform reasoning and predict answers. T\* [\(Ye et al., 2025\)](#page-11-2) extends this paradigm by introducing an object-oriented spatial-temporal search. It first leverages a VLM to extract target objects from the question, then employs object detection models (e.g., YOLO [\(Cheng et al., 2024\)](#page-9-6)) to identify keyframes containing these objects, and finally uses the retrieved frame set to complete the task. Moreover, strategies that introduce tree-structured search to improve efficiency have also been explored [\(Wang et al., 2025;](#page-11-5) [Li et al., 2025;](#page-10-4) [Pan et al., 2025\)](#page-10-5). However, all of these approaches depend on manually designed workflows, which lead to suboptimal search strategies.

This motivates us to explore an end-to-end learning approach that discovers optimal temporal search strategies directly from data. In this work, we reformulate the temporal search task as an interleaved text-video thinking process, and propose TIMESEARCH-R, a model that learns to actively search for relevant temporal clips through reinforcement learning (RL). As shown in Figure [1](#page-1-0) (b), our model alternates between textual reasoning and temporal exploration, iteratively refining its understanding of the video. We refer to this dynamic process as Thinking with Videos—a paradigm where models gradually improves their comprehension by searching for relevant video content conditioned on intermediate reasoning states. This concept extends the recent advances in multimodal reasoning, *Thinking with Images* [\(Su et al., 2025;](#page-11-6) [Hu et al., 2024;](#page-9-7) [Zheng et al., 2025\)](#page-11-7), to the long-video domain.

Although recent works have successfully applied RL algorithms like Group Relative Policy Optimization (GRPO) [\(DeepSeek-AI, 2025\)](#page-9-8) to textual [\(Jin et al., 2025\)](#page-9-9) and spatial search [\(Zheng et al., 2025\)](#page-11-7), temporal search in videos poses unique challenges. The original GRPO rewards only the final output while ignoring intermediate search decisions, leading to several failure modes illustrated in Figure [2.](#page-2-0) The first mode, termed insufficient temporal exploration, arises because the final output reward provides no incentive for comprehensive exploration of video frames. LVLMs may arrive at correct answers through partial evidence or language bias without proper visual grounding [\(Niu et al., 2021\)](#page-10-6),

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>Figure 2: Two failure modes with the original GRPO reward. Left: Insufficient temporal exploration. The model misses critical frames required to correctly answer the question. Right: Inconsistent logical reasoning. The intermediate reasoning process contradicts the final answer.

missing critical frames required for reliable understanding. The second mode, termed inconsistent logical reasoning, emerges when models produce plausible thinking processes disconnected from the final answers, a phenomenon also observed in text-only reasoning [\(Lanham et al., 2023\)](#page-10-7). These two failure modes hinder proper temporal search and diminish the benefits of video reasoning.

To address these challenges, we propose Completeness Self-Verification (CSV) as a supplement to the original GRPO algorithm, providing supervision over the intermediate steps of temporal search. GRPO-CSV tackles insufficient temporal exploration by ensuring the model to acquire sufficient visual evidence through self-verification, and promotes consistency between intermediate reasoning and the final answer by re-answering the question using the searched frames. Besides, we construct a high-quality video reasoning dataset to support GRPO-CSV training. Existing datasets contain a large number of trivial samples solvable through prue linguistic bias, as well as noisy samples that remain unsolvable even with extensive search, severely hindering progress in long-video reasoning. We implement a two-stage data filtering pipeline to curate high-quality samples tailored to the demands of video reasoning, ensuring that the model learns the correct process of temporal search.

We evaluate our TimeSearch-R on both temporal search and long-form video understanding tasks, demonstrating its superiority in long video reasoning. On temporal search tasks, TimeSearch-R improving the temporal F1 score on Haystack-LVBench by 5.6% and the accuracy on Haystack-Ego4D by 8.5%, compared to the previous state-of-the-art (SOTA) method. On long-form video understanding tasks, TimeSearch-R establishes new SOTA results with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced reasoning model Video-R1 on LongVideoBench.

In summary, our main contributions are three-fold:

- 1. We propose the TimeSearch-R framework, which reformulates temporal search as interleaved text-video thinking and learns optimal search strategies directly from data.
- 2. We introduce GRPO-CSV, a novel RL algorithm, which ensures sufficient and accurate video exploration by supervising the intermediate steps of temporal search. To support GRPO-CSV training, we also construct a high-quality video reasoning dataset via a twostage filtering pipeline, enabling the model to learn correct temporal search processes.
- 3. Extensive experiments demonstrate the superiority of our approach on both temporal search and long-form video understanding. Notably, TimeSearch-R establishes a new SOTA on LongVideoBench, outperforming the latest reasoning model Video-R1 by 2.0%.

# 2 METHODS

In this section, we first reformulate the temporal search task as an interleaved text-video thinking process, enabling the model to learn optimal search strategies directly from data. To address the challenges of insufficient temporal exploration and inconsistent logical reasoning, we introduce GRPO-CSV as a novel RL algorithm for long videos, which ensures both sufficient and accurate video exploration by supervising intermediate steps of temporal search. Finally, we describe the model training process, including the construction of a high-quality long-video reasoning dataset.

### <span id="page-3-0"></span>2.1 TASK FORMULATION

Temporal Search within Thinking Process. To learn optimal search strategies directly from data, we reformulate temporal search as a multi-turn thinking process interleaved with video clip retrieval. Given a video V and a corresponding question Q, an initial preview  $\tilde{V}$  is uniformly sampled from V for subsequent reasoning. At each thinking step k, the policy model  $\pi_{\theta}$  generates a textual reasoning  $T_k$ . If  $T_k$  contains a search instruction, the video environment executes it according to frame timestamps, retrieving a clip  $V_k \subseteq V$  that is appended to the ongoing chain of thought (CoT) as input for later steps. The interleaved text-video CoT at reasoning step k is formalized as:

$$C_k \triangleq \{ (T_1, V_1), (T_2, V_2), \dots, (T_k, V_k) \}.$$
 (1)

This interaction process repeats until the model emits the final answer A or reaches the pre-defined reasoning budget. For further analysis, the entire reasoning chain can be decomposed into two components: temporal search and answer prediction, which can be formulated as:

$$P_{\theta}(A, C \mid \tilde{V}, Q) = \underbrace{P_{\theta}(C \mid \tilde{V}, Q)}_{Temporal \ Search} \cdot \underbrace{P_{\theta}(A \mid C, \tilde{V}, Q)}_{Answer \ Prediction}. \tag{2}$$

**Dynamic Video Frames.** During the interleaved thinking process, the model autonomously explores the video by searching for additional clips. At reasoning step k, if the model outputs a search instruction, it is also required to specify the temporal boundaries  $t_s^k$  and  $t_e^k$  to be explored, along with a corresponding textual query  $q^k$ . The video environment then executes a frame retrieval function to obtain additional F frames  $V_k = \text{search}(V; t_s^k, t_e^k, q^k, F) = \{f_k^1, f_k^2, \dots, f_k^F\}$ . This function serves as an interface to the policy model  $\pi_\theta$ , employing a small VLM (e.g., SigLIP (Zhai et al., 2023)) to calculate the similarity among frames within the specified temporal clip  $[t_s^k, t_e^k]$ , as well as the relevance with the textual query  $q^k$ . The most informative F frames are then sampled using determinantal point process (DPP) (Kulesza & Taskar, 2012), which has been widely used for information retrieval (Chen et al., 2018; Celis et al., 2018; Sun et al., 2025). This operation significantly improves the efficiency of temporal search, and more details can be found in Section A.

### <span id="page-3-1"></span>2.2 GRPO WITH COMPLETENESS SELF-VERIFICATION

Evaluating temporal search typically requires frame-level annotations (Ye et al., 2025), which are time-consuming and labor-intensive. To circumvent this challenge, previous works (Yu et al., 2025; Sun et al., 2025) have treated downstream video understanding task as a surrogate metric for assessing the searched frame set. However, these approaches are limited to selecting an optimal subset from a predefined pool of candidate frames, lacking interaction with and exploration of the video environment. Inspired by this, we design a **Completeness Self-Verification (CSV)** mechanism for GRPO, which is annotation-free and can be seamlessly integrated into RL training, serving as a complementary to the original outcome reward. The overall pipeline of GRPO-CSV is illustrated in Figure 3.

**GRPO-CSV.** We introduce CSV as a complement to GRPO with only outcome rewards. During the GRPO rollout phase, the policy model  $\pi_{\theta}$  generates a text-video interleaved CoT C and a final answer A. Applying rewards only to the final answer may reduce the effectiveness of intermediate search processes. To address this, we extract the video clips from C to form a dynamic frame set  $V_c$  as the input for the CSV phase. In the CSV rollout phase, the same policy model is required to re-answer the question Q using only  $V_c$ , yielding a CSV answer  $A_c$ . Critically, the model is prohibited from any further temporal searching and must rely solely on the currently searched frames to answer the question. The CSV answer  $A_c$  is expected to remain consistent with the original answer A:

$$P_{\theta}(A_c \mid V_c, Q) \approx P_{\theta}(A \mid C, \tilde{V}, Q). \tag{3}$$

**Completeness Reward.** We design a completeness reward for the CSV phase, which is computed using the original answer A, the CSV answer  $A_c$ , and the ground-truth answer  $A^*$  as follows:

$$R_c = \mathbb{1}[\text{Acc}(A, A^*) > 0.5] \cdot \text{Acc}(A_c, A^*).$$
 (4)

where  $Acc(A, A^*)$  and  $Acc(A_c, A^*)$  denote the correctness scores of the original answer and the CSV answer, respectively, and  $\mathbb{1}[\cdot]$  is an indicator function activated only when the original answer A is correct. This conditional design ensures that the CSV reward is applied only to promising reasoning trajectories, encouraging meaningful temporal search while verifying both the sufficiency of acquired visual evidence and the consistency between the reasoning process and the final answer.

![](_page_4_Figure_1.jpeg)

<span id="page-4-0"></span>Figure 3: **Overall pipeline of GRPO-CSV.** Building upon the original GRPO, CSV extracts a dynamic frame set from the multi-modal CoT and constructs a vision-only CoT for re-answering. This design verifies that the searched dynamic frames provide sufficient evidence for correct reasoning, ensuring completeness and consistency without requiring explicit frame-level supervision.

**Format Reward.** The format reward enforces adherence to a predefined schema throughout the multi-turn reasoning process, validating the structural integrity of the entire trajectory rather than individual steps. During reasoning, each step must follow either the <think>...</think><tool\_call>...</tool\_call> pattern for temporal search or the <think>...</think><answer>...</answer> pattern for the final response. We assign a binary score to the full trajectory: 1 if all steps are structurally valid, and 0 otherwise.

**Accuracy Reward.** We evaluate answer accuracy for two task types. For multiple-choice questions, we extract the option letter from the model's output and perform an exact match with the ground-truth option. For open-ended questions, we adopt an LLM-as-a-Judge approach (Zheng et al., 2023) to assess the semantic agreement between the model's final answer and the reference answer. The scores for both cases are given in binary form, with 1 indicating alignment with the standard answer.

**Overall Reward.** The total reward is the sum of completeness, format, and accuracy components:

$$R = R_c + R_{\text{fmt}} + R_{\text{acc}}.$$
 (5)

This composition encourages sufficient temporal exploration  $(R_c)$ , consistent reasoning structures  $(R_{\rm fmt})$ , and correct final answers  $(R_{\rm acc})$ , enhancing the model's ability to understand long-form videos.

### 2.3 MODEL TRAINING

**Dataset Construction.** A fundamental challenge in RL for long-video reasoning lies in the fact that a large number of samples in the existing datasets can be solved through pure linguistic bias, reducing the reliance on temporal search. Moreover, some noisy samples remain unsolvable even under ideal temporal search, preventing the model from effectively exploring the video. To address these challenges, we implement a two-stage data filtering pipeline to construct a high-quality dataset tailored to video reasoning. In the first stage, we remove samples that the policy model can solve correctly using only 4 uniformly sampled frames, thereby discouraging reliance on linguistic shortcuts. In the second stage, we further discard samples that remain unsolvable even with multiple temporal searches and numerous video frames, ensuring active video exploration. Additional details of this filtering pipeline are provided in Section B.1. And we enhance dataset diversity through the incorporation of samples sourced from Haystack-Ego4D (Ye et al., 2025), VideoMarathon (Lin et al., 2025), and CinePile (Rawal et al., 2024). A detailed analysis of the dataset is presented in Section B.2.

<span id="page-5-0"></span>Table 1: **Temporal search performance.** We report temporal similarity, visual similarity, and question-answering (QA) accuracy on Haystack-LVBench, as well as QA accuracy on Haystack-Ego4D test-tiny subset. Baseline results are directly cited from Ye et al. (2025). † indicates the average number of keyframes determined by the model adaptively.

| Method                            | Base Model # Fr | # Frame           | Temporal |      | Visual           |      | QA   |       |         |       |
|-----------------------------------|-----------------|-------------------|----------|------|------------------|------|------|-------|---------|-------|
| Method                            |                 | # Frame           | P        | R    | $\overline{F_1}$ | P    | R    | $F_1$ | LVBench | Ego4D |
|                                   |                 | Static Frame      | Samp     | ling |                  |      |      |       |         |       |
| Uniform                           | Qwen2.5VL-7B    | 8                 | 1.4      | 6.3  | 2.2              | 56.0 | 72.0 | 62.7  | 33.7    | 32.0  |
| Uniform                           | GPT-4o          | 8                 | 1.4      | 6.3  | 2.2              | 56.0 | 72.0 | 62.7  | 47.1    | 41.5  |
| Uniform                           | GPT-40          | 32                | 1.4      | 24.9 | 2.7              | 58.7 | 81.6 | 67.3  | 50.5    | 45.5  |
| Adaptive Temporal Search          |                 |                   |          |      |                  |      |      |       |         |       |
| VideoAgent (Wang et al., 2024)    | GPT-4           | 10.1 <sup>†</sup> | 1.2      | 8.5  | 2.1              | 58.8 | 73.2 | 64.7  | -       | -     |
| Retrieval-based (Ye et al., 2025) | GPT-4o          | 8                 | 1.5      | 6.3  | 2.3              | 63.1 | 65.5 | 64.1  | _       | -     |
| T* (Ye et al., 2025)              | GPT-4o          | 8                 | 1.6      | 7.1  | 2.5              | 58.4 | 72.7 | 64.3  | 51.9    | 45.0  |
| Retrieval-based (Ye et al., 2025) | GPT-40          | 32                | 1.3      | 21.8 | 2.4              | 59.9 | 80.8 | 67.8  |         | -     |
| T* (Ye et al., 2025)              | GPT-40          | 32                | 1.7      | 28.2 | 3.1              | 58.3 | 83.2 | 67.8  | 53.1    | 46.5  |
| Text-Video Interleaved Reasoning  |                 |                   |          |      |                  |      |      |       |         |       |
| TimeSearch-R (Ours)               | Qwen2.5VL-7B    | 8.8†              | 5.4      | 22.3 | 8.1              | 63.2 | 76.4 | 69.2  | 52.1    | 53.5  |

**Model Training.** We employ a two-stage training scheme for our TimeSearch-R. In the first stage, supervised fine-tuning (SFT) serves as a cold start, guiding the model to follow the correct reasoning format and enabling effective policy learning in the subsequent RL stage. SFT training adopts the above dataset construction pipeline, using GPT-40 (OpenAI, 2024) to generate the text-video interleaved reasoning processes and the corresponding final answers. Following practices in the text domain (Jin et al., 2025), we mask the temporal search results during training to force the model to learn meaningful temporal windows and textual queries. The objective in this stage is to minimize the standard cross-entropy loss over reasoning tokens, while excluding masked video tokens from gradient computation. Building on this cold-start, we further conduct RL post-training based on the proposed GRPO-CSV algorithm to stimulate the temporal reasoning capability of the model.

### 3 EXPERIMENTS

### 3.1 EXPERIMENTAL SETUP

**Baselines.** To comprehensively evaluate the effectiveness of TimeSearch-R, we compare it against three types of baselines: (1) Advanced foundation models with static frame sampling, including both API models (OpenAI, 2024; Team et al., 2024) and open-source models (Bai et al., 2025a). (2) State-of-the-art temporal search agents, such as VideoAgent (Wang et al., 2024), T\* (Ye et al., 2025), and VideoTree (Wang et al., 2025). (3) Video reasoning models like Video-R1 (Feng et al., 2025).

**Datasets.** We evaluate TimeSearch-R on two tasks: (1) Temporal search on Haystack-LVBench and Haystack-Ego4D (Ye et al., 2025), where the task is modeled as long video needle-in-a-haystack, measuring temporal and visual similarity as well as QA accuracy. (2) Long-form video understanding on VideoMME (Fu et al., 2024), MLVU (Zhou et al., 2024), and LongVideoBench (Wu et al., 2024).

**Evaluation Metrics.** Besides the original metrics used in the benchmarks, we additionally introduce two metrics to assess the quality of the text-video interleaved thinking process for ablation study. Among them, *completeness* measures whether the searched frame set is sufficient for the correct answer, while *consistency* measures the alignment between intermediate reasoning and the final answer. Further details on these two metrics are provided in Section D.

**Implementation Details.** We train TimeSearch-R based on Qwen2.5-VL-7B-Instruct (Bai et al., 2025a). In the RL training, we use the AdamW (Loshchilov & Hutter, 2017) optimizer with a learning rate of 1e-6, a KL penalty coefficient  $\beta = 0.005$ , and a batch size of 4 with 8 rollouts per prompt. We limit each search operation to retrieving at most 8 frames from a specified temporal clip, with up to 8 search steps in total. Training is conducted on 32 A100 GPUs. See more details in Section F.

<span id="page-6-0"></span>Table 2: **Video understanding performance.** E2E stands for end-to-end optimization. # Frame represents the number of input frames. † indicates the keyframes produced by temporal search.

| Model                                  | E2E # Frame           |                  | VideoMME (w/o sub) |        |      |         | MLVU  | LVB  |
|----------------------------------------|-----------------------|------------------|--------------------|--------|------|---------|-------|------|
| iviouei                                |                       | EZE # Frame      |                    | medium | long | overall | m-avg | val  |
|                                        | Static Frame Sampling |                  |                    |        |      |         |       |      |
| Qwen2.5VL-7B (Bai et al., 2025b)       | 1                     | 768              | 76.3               | 66.0   | 54.6 | 65.1    | 70.2  | 56.0 |
| GPT-4o (OpenAI, 2024)                  | 1                     | 384              | 80.0               | 70.3   | 65.3 | 71.9    | 64.6  | 66.7 |
| Gemini-1.5-Pro (Team et al., 2024)     | ✓                     | 1 fps            | 81.7               | 74.3   | 67.4 | 75.0    | -     | 64.0 |
|                                        | Adap                  | tive Tempor      | al Searc           | h      |      |         |       |      |
| VideoAgent (GPT-4) (Wang et al., 2024) | X                     | 87 <sup>†</sup>  | -                  | -      | 49.0 | 56.0    | -     | -    |
| VideoTree (GPT-4) (Wang et al., 2025)  | X                     | 128 <sup>†</sup> | 67.8               | 59.9   | 54.2 | _       | -     | _    |
| T* (GPT-40) (Ye et al., 2025)          | X                     | $32^{\dagger}$   | 69.5               | 63.5   | 59.3 | 64.1    | -     | _    |
|                                        | Te                    | xt-only Reas     | oning              |        |      |         |       |      |
| Video-R1-7B (Feng et al., 2025)        | 1                     | 32               | 71.1               | 59.0   | 49.4 | 59.9    | 61.6  | 56.4 |
| Video-R1-7B (Feng et al., 2025)        | ✓                     | 768              | 74.1               | 65.1   | 55.6 | 65.7    | 68.4  | 58.1 |
| T                                      | ext-Vide              | eo Interleave    | d Reaso            | ning   |      |         |       |      |
| Qwen2.5VL-7B + Search                  | X                     | 768              | 53.4               | 53.8   | 48.2 | 51.8    | 58.9  | 49.1 |
| TimeSearch-R-7B (Ours)                 | 1                     | 768              | 76.8               | 67.1   | 56.0 | 66.6    | 71.5  | 60.1 |
| $\Delta$ (v.s. Qwen2.5VL-7B)           | _                     | -                | +0.5               | +1.1   | +1.4 | +1.5    | +1.3  | +4.1 |

### 3.2 MAIN RESULTS

**Temporal Search.** On the temporal search task, TimeSearch-R establishes a new state-of-the-art on LV-Haystack, as shown in Table 1. Under a budget of 8 keyframes, our method achieves an  $F_1$  score of 8.1 in temporal similarity, more than three times the previous best result of 2.5 obtained by T\*. In visual similarity, TimeSearch-R reaches an  $F_1$  score of 69.2, surpassing the previous SOTA method VideoAgent by 5.5, and even outperforming the retrieval-based method and T\* with larger keyframe budgets. For the needle-in-a-haystack QA, our TimeSearch-R consistently outperforms the advanced API model GPT-40, achieving 52.1% accuracy on Haystack-LVBench and 53.5% on Haystack-Ego4D. These results demonstrate the superiority of end-to-end learned temporal search strategies over handcrafted workflows based on human heuristics.

**Long-Form Video Understanding.** Our TimeSearch-R also achieves strong performance on the long-form video understanding task, which is shown in Table 2. On VideoMME, our method reaches an overall accuracy of 66.6%, surpassing the base model Qwen2.5-VL by 1.5%. As the duration of the video increases, our method can achieve more gains, from 0.5% on short videos to 1.4% on long videos, demonstrating that temporal search becomes more valuable when the video length increases. On MLVU and LongVideoBench, TimeSearch-R achieves 71.5% and 60.1%, improving over the base model by 1.3% and 4.1%, respectively. Compared with video search agents, TimeSearch-R outperforms VideoAgent and T\* on VideoMME by 10.6% and 2.5%, highlighting the advantage of end-to-end optimization. Notably, our method consistently surpasses the latest video reasoning model Video-R1 across all benchmarks, validating that text-video interleaved reasoning is more effective than text-only reasoning for long-form video understanding. Moreover, directly applying temporal search to Qwen2.5-VL through CoT prompting without additional training actually degrades performance, underscoring the necessity of RL post-training with the proposed GRPO-CSV.

## 3.3 ABLATION STUDIES

**Training Scheme.** We explore the impact of different training stages in Table 4a, from zero-shot CoT to SFT and finally RL, yielding two key findings: (1) **SFT enables search capability:** The model cannot perform the search well only through zero-shot CoT prompts. SFT allows the model to rapidly acquire temporal search skills, dramatically improving temporal  $F_1$  from 0.0 to 7.8 and searched frame completeness from 44.2% to 60.5%. (2) **RL enhances video reasoning:** While RL provides modest improvements to temporal similarity and search completeness, its primary advantage lies in boosting overall understanding performance. The post-training stage improves reasoning consistency by 2.6%, which in turn raises QA accuracy from 59.2% to 66.6%.

**GRPO-CSV Component.** We further conduct an ablation study on the components of GRPO-CSV in Figure 4, and obtain three key findings: (1) **GRPO reduces search completeness.** Without CSV

| M.d. 1                                                                  | Hay                                                               | stack-LVBe                                   | ench                                   | VideoMME                                |                                       |                                                                      |  |
|-------------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------|----------------------------------------|-----------------------------------------|---------------------------------------|----------------------------------------------------------------------|--|
| Method                                                                  | P                                                                 | R                                            | $F_1$                                  | Comp.                                   | Cons.                                 | Acc.                                                                 |  |
| Qwen2.5-VL w/ search<br>SFT                                             | 0.0<br>7.4                                                        | 0.0<br>11.6                                  | 0.0<br>7.8                             | 44.2<br>60.5                            | 59.4<br>69.2                          | 51.8<br>59.2                                                         |  |
| GRPO (Before Collapse)<br>GRPO-CSV w/o Acc. Rwd<br>GRPO-CSV w/ Acc. Rwd | 5.2 <sub>-2.2</sub><br>6.1 <sub>-1.3</sub><br>5.4 <sub>-2.0</sub> | $18.8_{+7.2} \\ 19.8_{+8.2} \\ 22.3_{+10.7}$ | $7.4_{-0.4}$ $8.2_{+0.4}$ $8.1_{+0.3}$ | $57.2_{-3.3}  61.2_{+0.7}  60.2_{-0.3}$ | $69.3_{+0.1} 75.3_{+6.1} 71.8_{+2.6}$ | 65.1 <sub>+5.9</sub><br>64.8 <sub>+5.6</sub><br>66.6 <sub>+7.4</sub> |  |

![](_page_7_Figure_2.jpeg)

<span id="page-7-0"></span>(a) Ablation Results

<span id="page-7-2"></span>(b) Training Dynamics

<span id="page-7-1"></span>Figure 4: **Ablation study of GRPO-CSV.** (a) Comparison of different training schemes on temporal search and long-form video understanding. (b) When CSV is removed, training begins to collapse. The model gradually reduces the number of search calls and eventually stops searching altogether.

<span id="page-7-3"></span>Table 3: Ablation study of data composition. Line 1 shows the accuracy of original Qwen2.5-VL.

| Eas          | Ewo          | Eller        |       | Gene   | eral        |             | Reasoning   |             |        |             |
|--------------|--------------|--------------|-------|--------|-------------|-------------|-------------|-------------|--------|-------------|
| Ego          | Exo          | Filter       | short | medium | long        | overall     | temporal    | spatial     | action | object      |
| _            | -            | _            | 76.3  | 66.0   | 54.6        | <u>65.1</u> | 51.4        | 76.8        | 56.8   | <u>59.5</u> |
| <b>√</b>     | <b>√</b>     |              | 74.2  | 62.7   | 51.3        | 62.8        | 40.1        | 67.9        | 60.0   | 56.2        |
| $\checkmark$ |              | $\checkmark$ | 76.4  | 64.7   | <u>54.9</u> | 65.3        | <u>54.8</u> | 73.2        | 58.2   | 59.0        |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | 76.8  | 67.1   | 56.0        | 66.6        | 58.8        | <u>75.0</u> | 62.5   | 61.9        |

as a complement, GRPO drops completeness from 60.5% to 57.2% and temporal  $F_1$  from 7.8 to 7.4, demonstrating that outcome-only rewards lead to insufficient temporal exploration. (2) **GRPO-CSV improves training stability.** As illustrated in Figure4b, removing CSV causes training to collapse around step 300, after which the model ceases to make search calls and completeness drops to zero. (3) **GRPO-CSV with accuracy reward achieves the best QA performance.** While completeness reward alone achieves the highest completeness and consistency, it slightly reduces QA accuracy by 0.3%. Combining GRPO-CSV with accuracy reward leads to the best overall QA performance.

**Data Composition.** We also analyze the data composition in RL training, as shown in Table 3, revealing the contributions of data filtering and domain diversity. Without data filtering, RL training leads to a substantial performance drop compared to the original Qwen2.5-VL. This degradation arises because linguistic biases induce zero advantage in GRPO group computation: when questions can be trivially answered through linguistic shortcuts, all rollouts achieve perfect accuracy and completeness, yielding no learning signal and severely hindering RL efficiency and training stability. After applying data filtering, the model trained solely on egocentric data recovers baseline performance, but the lack of diversity weakens the benefits of RL. By incorporating exocentric data to enhance domain diversity, the model achieves its best general QA accuracy of 66.6%. Notably, although the training data only includes general long-video QA tasks, RL training significantly boosts the model's temporal and action reasoning capabilities, improving them by 7.4% and 5.7%, respectively. This remarkable performance demonstrates that TimeSearch-R learns fundamental cognitive patterns through end-to-end policy optimization, validating the strong generalization of our proposed GRPO-CSV algorithm.

### 3.4 CASE STUDIES

We analyze the search patterns that emerge during end-to-end RL training, demonstrating how the model executes temporal search within its reasoning process in a manner analogous to human cognition. These search patterns exhibit adaptability and flexibility across different task types:

**Hypothesis-driven search.** The model formulates hypotheses based on limited context and executes targeted searches to gather additional video frames as supporting evidence. (Figure 5)

**Confirmation or elimination.** When the initially sampled dynamic frame set provides insufficient support for an answer, the model employs multi-faceted search strategies or elimination methods to collect additional evidence and reduces uncertainties. (Figure 13 and 14)

**Sequential search.** The model performs segment-by-segment analysis to accomplish temporal reasoning tasks that require understanding sequential relationships. (Figure 15)

![](_page_8_Figure_1.jpeg)

<span id="page-8-0"></span>Figure 5: Hypothesis-driven search. Given the context that dogs are lying in a row across multiple scenes and remain still, the model hypothesizes that they are waiting to be photographed. It then searches for the person taking a photo to gather supporting evidence and provides the final answer.

# 4 RELATED WORK

Temporal Search for Long-Video Understanding. Traditional video understanding methods rely on static frame sampling, such as uniform sampling or heuristic-based strategies [\(Li et al., 2024;](#page-10-12) [Chen et al., 2024;](#page-9-13) [Bai et al., 2025a\)](#page-9-2), which fail to adapt to varying information density and evolving reasoning contexts. Recent work has explored more sophisticated mechanisms. Similarity-based methods like KeyVideoLLM [\(Liang et al., 2024\)](#page-10-13) achieve significant compression while maintaining performance , while learning-based approaches such as Frame-Voyager [\(Yu et al., 2025\)](#page-11-10) rank frame combinations based on prediction losses, emphasizing task-specific selection. Advanced semantic frameworks have emerged to address temporal dependencies. Logic-in-Frames [\(Guo et al., 2025\)](#page-9-14) defines logical relations including spatial co-occurrence and temporal proximity to guide dynamic frame sampling. T\* [\(Ye et al., 2025\)](#page-11-2) reframes temporal search as spatial search with adaptive zooming mechanisms. Interactive agents like VideoAgent [\(Wang et al., 2024\)](#page-11-4) and VideoTree [\(Wang et al.,](#page-11-5) [2025\)](#page-11-5) enable multi-turn temporal exploration through prompt-driven orchestration. However, none of the aforementioned methods adopt end-to-end optimization, resulting in suboptimal search strategies.

Reinforcement Learning for Multimodal Reasoning. Recent advances have explored RL to enhance reasoning capabilities in LLMs. GRPO [\(DeepSeek-AI, 2025\)](#page-9-8) demonstrates that outcomebased rewards can effectively elicit complex reasoning. Search-R1 [\(Jin et al., 2025\)](#page-9-9) extends this paradigm to text-based search tasks, showing that RL can facilitate adaptive information retrieval. Approaches like MM-Eureka [\(Meng et al., 2025\)](#page-10-14) and LMM-R1 [\(Peng et al., 2025\)](#page-10-15) have successfully applied RL to enhance multimodal reasoning, but focus primarily on static image understanding rather than dynamic video interaction. Video-R1 [\(Feng et al., 2025\)](#page-9-3) applies GRPO to video reasoning but limits the thinking process to prue text without visual interaction, while DeepEyes [\(Zheng et al., 2025\)](#page-11-7) uses RL for high-resolution image understanding through adaptive cropping operations but focuses on spatial rather than temporal exploration. Despite these advances, applying RL to interactive long video understanding remains largely unexplored and presents unique challenges.

# 5 CONCLUSION

In this work, we propose TimeSearch-R, a framework that reformulates temporal search as text-video interleaved thinking to learn optimal search strategies directly from data. To enhance temporal search through RL, we propose CSV as a complement to the outcome-only reward of GRPO, addressing the challenges of insufficient temporal exploration and inconsistent logical reasoning. TimeSearch-R achieves strong performance on both temporal search and long-form video understanding tasks, while exhibiting distinct search patterns across different task types. We hope this work contributes meaningful progress toward advancing long video understanding powered by reinforcement learning.

# REFERENCES

- <span id="page-9-2"></span>Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025a. URL <https://arxiv.org/abs/2502.13923>.
- <span id="page-9-12"></span>Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025b.
- <span id="page-9-4"></span>Monica S. Castelhano and John M. Henderson. Initial scene representations facilitate eye movement guidance in visual search. *Journal of Experimental Psychology: Human Perception and Performance*, 33(4):753–763, 2007. doi: 10.1037/0096-1523.33.4.753.
- <span id="page-9-11"></span>Elisa Celis, Vijay Keswani, Damian Straszak, Amit Deshpande, Tarun Kathuria, and Nisheeth Vishnoi. Fair and diverse dpp-based data summarization. In *International conference on machine learning*, pp. 716–725. PMLR, 2018.
- <span id="page-9-10"></span>Laming Chen, Guoxin Zhang, and Eric Zhou. Fast greedy map inference for determinantal point process to improve recommendation diversity. *Advances in Neural Information Processing Systems*, 31, 2018.
- <span id="page-9-13"></span>Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 24185–24198, 2024.
- <span id="page-9-6"></span>Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 16901–16911, 2024.
- <span id="page-9-8"></span>DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL <https://arxiv.org/abs/2501.12948>.
- <span id="page-9-3"></span>Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. *arXiv preprint arXiv:2503.21776*, 2025.
- <span id="page-9-0"></span>Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. *arXiv preprint arXiv:2405.21075*, 2024.
- <span id="page-9-14"></span>Yifan Guo, Liqiang Zou, Yang Li, Jia Chen, and Joey Tianyi Zhou. Logic-in-frames: Logical dependency modeling for frame selection in video question answering. *arXiv preprint arXiv:2501.00212*, 2025.
- <span id="page-9-1"></span>Mary M. Hayhoe and Dana H. Ballard. Eye movements in natural behavior. *Trends in Cognitive Sciences*, 9(4):188–194, 2005. doi: 10.1016/j.tics.2005.02.009.
- <span id="page-9-5"></span>John M. Henderson and Taylor R. Hayes. Meaning-based guidance of attention in scenes: Evidence from meaning maps. *Journal of Vision*, 17(6):23, 2017. doi: 10.1167/17.6.23.
- <span id="page-9-7"></span>Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL <https://openreview.net/forum?id=GNSMl1P5VR>.
- <span id="page-9-9"></span>Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. *arXiv preprint arXiv:2503.09516*, 2025.

- <span id="page-10-8"></span>Alex Kulesza and Ben Taskar. Determinantal point processes for machine learning. *Found. Trends Mach. Learn.*, 5(2-3):123–286, 2012. doi: 10.1561/2200000044. URL [https://doi.org/](https://doi.org/10.1561/2200000044) [10.1561/2200000044](https://doi.org/10.1561/2200000044).
- <span id="page-10-7"></span>Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning. *CoRR*, abs/2307.13702, 2023.
- <span id="page-10-12"></span>Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024.
- <span id="page-10-4"></span>Chenglin Li, Qianglong Chen, Yin Zhang, et al. Iterative zoom-in: Temporal interval exploration for long video understanding. *arXiv preprint arXiv:2507.02946*, 2025.
- <span id="page-10-1"></span>Yicong Li, Junbin Xiao, Chun Feng, Xiang Wang, and Tat-Seng Chua. Discovering spatio-temporal rationales for video question answering. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 13869–13878, 2023.
- <span id="page-10-13"></span>Dengxin Liang, Yixiao Shu, Ruobing Zhang, Songyang Chen, Xuanmo Li, and Minheng Wang. Keyvideollm: Towards large-scale video keyframe selection. *arXiv preprint arXiv:2407.03104*, 2024.
- <span id="page-10-2"></span>Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. *arXiv preprint arXiv:2311.10122*, 2023.
- <span id="page-10-9"></span>Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Hao Chen, Jiebo Luo, Zicheng Liu, and Emad Barsoum. Unleashing hour-scale video training for long video-language understanding. *arXiv preprint arXiv:2506.05332*, 2025.
- <span id="page-10-11"></span>Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.
- <span id="page-10-14"></span>Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. URL <https://arxiv.org/abs/2503.07365>.
- <span id="page-10-6"></span>Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. Counterfactual vqa: A cause-effect look at language bias. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 12700–12710, 2021.
- <span id="page-10-10"></span>OpenAI. Gpt-4o. <https://openai.com/index/hello-gpt-4o/>, May 2024.
- <span id="page-10-5"></span>Junwen Pan, Rui Zhang, Xin Wan, Yuan Zhang, Ming Lu, and Qi She. Timesearch: Hierarchical video search with spotlight and reflection for human-like long video understanding. *arXiv preprint arXiv:2504.01407*, 2025.
- <span id="page-10-0"></span>Jongwoo Park, Kanchana Ranasinghe, Kumara Kahatapitiya, Wonjeong Ryu, Donghyun Kim, and Michael S. Ryoo. Too many frames, not all useful: Efficient strategies for long-form video qa, 2025. URL <https://arxiv.org/abs/2406.09396>.
- <span id="page-10-15"></span>Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl, 2025. URL [https://arxiv.org/abs/2503.](https://arxiv.org/abs/2503.07536) [07536](https://arxiv.org/abs/2503.07536).
- <span id="page-10-3"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021.

- <span id="page-11-12"></span>Ruchit Rawal, Khalid Saifullah, Ronen Basri, David Jacobs, Gowthami Somepalli, and Tom Goldstein. Cinepile: A long video question answering dataset and benchmark. *CoRR*, abs/2405.08813, 2024.
- <span id="page-11-6"></span>Zhaochen Su, Peng Xiang, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, Linjie Li, Yu Cheng, Heng Ji, Junxian He, and Yi R. (May) Fung. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. *CoRR*, abs/2506.23918, 2025. doi: 10.48550/ARXIV.2506.23918. URL [https://doi.org/](https://doi.org/10.48550/arXiv.2506.23918) [10.48550/arXiv.2506.23918](https://doi.org/10.48550/arXiv.2506.23918).
- <span id="page-11-9"></span>Hui Sun, Shiyin Lu, Huanyu Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Ming Li. Mdp3: A training-free approach for list-wise frame selection in video-llms. *arXiv preprint arXiv:2501.02885*, 2025.
- <span id="page-11-13"></span>Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024.
- <span id="page-11-14"></span>Leandro von Werra, Younes Belkada, Lewis Tunstall, and Hugging Face. Trl: Transformer reinforcement learning. <https://github.com/huggingface/trl>, 2020. Accessed: 2025-09-16.
- <span id="page-11-4"></span>Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. *ECCV*, 2024.
- <span id="page-11-5"></span>Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos, 2025. URL <https://arxiv.org/abs/2405.19209>.
- <span id="page-11-1"></span>Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024.
- <span id="page-11-3"></span>Alfred L. Yarbus. *Eye Movements and Vision*. Plenum Press, New York, 1967.
- <span id="page-11-2"></span>Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, et al. Re-thinking temporal search for long-form video understanding. In *CVPR*, pp. 8579–8591, 2025.
- <span id="page-11-10"></span>Sicheng Yu, Chengkai Jin, Huanyu Wang, Zhenghao Chen, Sheng Jin, Zhongrong Zuo, Xiaolei Xu, Zhenbang Sun, Bingni Zhang, Jiawei Wu, Hao Zhang, and Qianru Sun. Frame-voyager: Learning to query frames for video large language models, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2410.03226) [2410.03226](https://arxiv.org/abs/2410.03226).
- <span id="page-11-8"></span>Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 11975–11986, 2023.
- <span id="page-11-11"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
- <span id="page-11-7"></span>Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing "thinking with images" via reinforcement learning. *arXiv preprint arXiv:2505.14362*, 2025.
- <span id="page-11-0"></span>Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. *arXiv preprint arXiv:2406.04264*, 2024.

# TIMESEARCH-R: ADAPTIVE TEMPORAL SEARCH FOR LONG-FORM VIDEO UNDERSTANDING VIA SELF-VERIFICATION REINFORCEMENT LEARNING

### **APPENDIX**

This appendix provides more details about our methods, dataset, training, more case studies, broader impacts, as well as the LLM usage, organized as follows:

- Section A: Search Function
- · Section B: Dataset Details
- Section C: Prompt Design
- Section D: Evaluation Metrics
- Section E: Efficiency Analysis
- Section F: Training Details
- Section G: More Case Studies
- Section H: Boarder Impacts
- Section I: LLM Usage

### <span id="page-12-0"></span>A SEARCH FUNCTION

### A.1 FRAME SELECTION

The video search function selects the most informative frames within predicted temporal clips. Specifically, we leverage determinantal point process (DPP) (Kulesza & Taskar, 2012) as the search optimization for its ability to naturally balance query relevance and diversity that penalizes redundancy, which has been widely applied in information retrieval (Celis et al., 2018; Sun et al., 2025).

Recall the definition of search in Sec. 2.1, it aims to select F optimal frames guided by a temporal clip  $[t_s,t_e]$  and a query q from the original video V. First, the function first subsamples N candidate frames  $\mathcal{F}_{[t_s,t_e]}=\{v_i\}_{i=1}^N$  within the temporal clip. Subsequently, we obtain a visual embedding  $\mathbf{h}_i \in \mathbb{R}^d$  for each candidate frame in  $\mathcal{F}_{[t_s,t_e]}$ , and a query embedding  $\mathbf{q} \in \mathbb{R}^d$  for q. Then we define the pairwise cosine similarity for candidate frames as  $S_{ij}=\mathbf{h}_i^{\top}\mathbf{h}_j$  and compute an unnormalized query relevance score for each frame as  $\tilde{r}_i=\mathbf{q}^{\top}\mathbf{h}_i$ , which is rescaled to [0,1] by min-max normalization  $r_i=\frac{\tilde{r}_i-\min \tilde{\mathbf{r}}}{\max \tilde{\mathbf{r}}-\min \tilde{\mathbf{r}}+\epsilon}$ , where  $\epsilon$  is a small constant to avoid division by zero. The kernel is constructed by diagonal conditioning with these relevance weights:

$$\tilde{\mathbf{L}} = \operatorname{diag}(\mathbf{r}) \mathbf{S} \operatorname{diag}(\mathbf{r}), \tag{6}$$

which is equivalent to  $\tilde{L}_{ij} = r_i r_j \mathbf{h}_i^{\top} \mathbf{h}_j$ . The optimal subset  $V^* \subset \mathcal{F}_{[t_s,t_e]}$  with  $|V^*| = F$  is then obtained through fast greedy MAP inference (Chen et al., 2018):

$$V^* = \arg \max_{S \subseteq \mathcal{F}_{[t_s, t_e], |S| = F}} \det(\tilde{\mathbf{L}}_S). \tag{7}$$

This formulation ensures that selected frames are both diverse and relevant to the query. When available frames are fewer than F, the search function degrades to uniform temporal sampling.

### A.2 FRAME REPRESENTATION

The selected clip frames are sparse and non-uniform. To maintain the temporal pace, we attach an explicit absolute timestamp to each frame by inserting a short text token with the time in seconds (e.g., "12.3s") immediately before the image. This simple interleaving of timestamp text and the corresponding image maintains absolute temporal grounding when inter-frame intervals vary and

![](_page_13_Figure_1.jpeg)

<span id="page-13-3"></span>Figure 6: Illustration of the proposed two-stage data filtering pipeline.

complements the native temporal ids. Explicit absolute timestamp augmented frame representation has also been observed to improve temporal capability in prior work on long-video temporal grounding [\(Pan et al., 2025\)](#page-10-5). For uniformly sampled preview frames, we employ the native dynamic-FPS and absolute time encoding following Qwen2.5-VL [\(Bai et al., 2025a\)](#page-9-2), which bind image token sequences to temporal ids aligned with real absolute timestamps.

# <span id="page-13-2"></span>B DATASET DETAILS

# <span id="page-13-0"></span>B.1 DATASET CONSTRUCTION

To ensure high-quality training data, we implement a two-stage filtering pipeline as shown in Fig. [6.](#page-13-3)

Stage 1: Visual Dependency Filtering. We uniformly sample 4 frames from each video and feed them along with the question to Qwen2.5-VL for inference. Questions that can be correctly answered with this limited visual information are considered to have low visual dependency and are subsequently filtered out. Only questions requiring richer visual context proceed to the next stage.

Stage 2: Search Usefulness Filtering. We increase the frame input to up to 64 frames and employ different LVLMs to perform dynamic temporal search for question-relevant video segments. Specifically, we use GPT-4o to generate SFT (Supervised Fine-Tuning) data and an early version of TimeSearch to obtain RL (Reinforcement Learning) training data. Although this stage produces CoT, only the CoT generated by GPT-4o is used for SFT training, while RL training utilizes only the question-answer pairs. To avoid search format errors, we implement format validation for LVLMs' responses, automatically retrying the model until obtaining properly formatted answers.

Human Selection for VideoMarathon (Panda-70M). Given that VideoMarathon's training set contains automatically generated question-answer pairs with potential unanswerable questions or incorrect ground-truth answers, we conduct manual annotation to ensure data quality. To minimize annotator bias in model answer evaluation, we establish a structured annotation protocol. First, annotators assess question reasonableness based on video content, filtering out unanswerable or ambiguous questions. Subsequently, annotators provide manual answers and compare them with synthetic ground-truth labels, removing data samples that are inconsistent with human responses.

# <span id="page-13-1"></span>B.2 DATASET ANALYSIS

The dataset exhibits a pronounced long-tail distribution in video duration with a mean length of 1,659 seconds. Most videos are shorter than 2,000 seconds, while a nontrivial tail extends beyond one hour, posing significant challenges for static frame sampling. This distribution motivates adaptive temporal search and multi-turn interaction to progressively retrieve evidence under tight keyframe budgets.

![](_page_14_Figure_1.jpeg)

![](_page_14_Figure_2.jpeg)

<span id="page-14-0"></span>Figure 7: **Dataset analysis.** (1) The training set is mainly composed of long videos. The average length is 1659 seconds, and the maximum length exceeds 10,000 seconds. (2) Egocentric QA pairs come from Haystack-Ego4D, and Exocentric QA data mainly from VideoMarathon and Cinepile, where VideoMarathon employs Panda-70M as the video source. (3) Question types include multiple-choice and open-ended questions. To obtain open-ended QA pairs, we convert some multiple-choice tasks into open-ended questions.

We curate data from four major sources to ensure coverage of diverse visual domains and camera styles. As shown in Fig. 7, Ego4D from Haystack-Ego4D (Ye et al., 2025) training set contributes 49.5% of samples, providing egocentric daily activities with frequent viewpoint changes. Panda-70M from VideoMarathon (Lin et al., 2025) accounts for 35.6%, expanding the variety of internet videos with heterogeneous motion patterns and scene dynamics. CinePile (Rawal et al., 2024) provides 9.5% of short videos with narrative structure and rapid scene transitions. The remaining 5.4% are from other sources and serve to reduce distributional bias.

Question types are intentionally imbalanced toward open-ended reasoning to better evaluate generative capabilities. Open-ended questions make up 60.3% of the data and emphasize step-by-step analysis, temporal grounding, and explanation quality. Multiple-choice questions comprise 39.7% and offer reliable automatic evaluation signals that complement outcome rewards in RL.

This composition yields wide coverage over motion intensity, scene diversity, and narrative structure while maintaining sufficient automatic evaluability. The mixture of long-tail durations and openended questions creates a setting where end-to-end RL and adaptive temporal search offer clear benefits over single-shot heuristics.

# <span id="page-15-0"></span>C PROMPT DESIGN

We design prompts to standardize interaction formats, minimize ambiguity, and provide explicit priors for temporal reasoning. Fig. [8](#page-15-1)[–11](#page-16-1) show the templates used during training and evaluation.

System Prompt. We follow the tool-use specification of the base Qwen2.5-VL family [\(Bai et al.,](#page-9-2) [2025a\)](#page-9-2) and adopt its tool\_call schema for invoking temporal search. This design ensures deterministic parsing by the environment and stable credit assignment for RL, as illustrated in Fig. [8.](#page-15-1)

```
System Prompt
You are a helpful video assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "seek_video_frames", "description": "Search and
select video frames according to textual query and temporal window. Time is in seconds.",
"parameters": {"type": "object", "properties": {"query": {"type": "string", "description":
 "The query is used to describe the object, scene, or event of interest in the video
thoroughly and clearly. "}, "start_time": {"type": "number", "description": "Start time of
 the segment of interest. "}, "end_time": {"type": "number", "description": "End time of
the segment of interest. "}, "num_frames": {"type": "integer", "description": "Number of
frames to sample (maximum 8). Default is 8."}}, "required": ["query"]}}}
</tools>
For each function call, return a json object with function name and arguments within <
tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

<span id="page-15-1"></span>Figure 8: The system prompt with tools.

Question Answering Prompt. The QA template enforces thorough reasoning inside <think> before any tool call or final answer. It restricts the output to exactly one of two formats and allows at most eight rounds of <tool\_call>. It explicitly provides the line *"The video duration: {duration} seconds."* to help the model produce absolute timestamps better. See Fig. [9.](#page-15-2)

```
Question Answering
You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling
 any tool or answering the question.
You must invoke tools to explore any video content you are interested in within <tool_call
> </tool_call> tags.
You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds.
When you have enough information to answer the question, provide your answer within <
answer> </answer> tags. Your answer should be supported by evidence from the video.
Your output must follow the format: <think>Your reasoning process</think><tool_call>
Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer
>Question: {question}
The video duration: {duration} seconds.
```

<span id="page-15-2"></span>Figure 9: The template for question answering.

Clip Frame Sampling and Search Response. After a search, the template returns the selected frames and their corresponding timestamps. If the frames are sufficient, the model must place the final answer in <answer>. Otherwise, the template asks the model to call the tool again with different parameters in JSON, thereby encouraging reflection and re-query. See Fig. [10.](#page-16-2)

### Temporal Search Response

```
Here are selected frames. They are located at {timestamps}.
If the frames provided above are sufficient to answer the user's question, please put your
 final answer within <answer></answer>.
Otherwise invoke the tool again with different parameters in JSON format.
```

<span id="page-16-2"></span>Figure 10: The response template of the temporal search.

Completeness Self-Verification Prompt. The CSV template asks the model to answer as briefly as possible and to say *"I don't know"* when the visual evidence is insufficient. No tools are available in this stage, which prevents new searches and ensures the answer is grounded only on the dynamic frame set gathered earlier. See Fig. [11.](#page-16-1)

### Completeness Self-Verification

You are a helpful assistant. Please answer visual questions as briefly as possible. When you don't have enough visual information, please say 'I don't know'.

<span id="page-16-1"></span>Figure 11: The template for CSV reasoning.

# <span id="page-16-0"></span>D EVALUATION METRICS

Completeness Rate. We measure the proportion of cases where the dynamic visual context alone suffices to produce the correct answer. Concretely, after the multi-turn search, we re-answer the question using only the gathered dynamic frame set and disallow further search, following the CSV procedure in Sec. [2.2](#page-3-1) and the prompt illustrated in Fig. [11.](#page-16-1) The resulting correctness is computed with the same task-specific accuracy used elsewhere, averaged over the whole dataset.

Consistency Rate. Consistency evaluates whether the intermediate reasoning coherently supports the final answer under the given question. We prompt a LLM model (GPT-4o) with the question, the reasoning text extracted from <think>...</think>, and the final answer from <answer>...</answer>, using the format in Fig. [12](#page-16-3) that requires a structured output: a short analysis in <think> followed by <answer> equal to "Yes" or "No". In implementation, we parse the LLM's output to obtain the binary decision; "Yes" is counted as 1 and "No" as 0, and any parsing failure is treated as 0. The Consistency Rate is the dataset average of these binary outcomes.

```
Consistency Score Evaluation
<system prompt>
You are a careful and logical reviewer. Your task is to verify whether the given reasoning
 process and the final answer are consistent in addressing the given question.
Please carefully read the following information:
Question: <Question>
Reasoning Process: <Reasoning>
Final Answer: <Answer>
Please follow this format strictly:
<think> Your analysis here </think> <answer> Yes/No </answer>
```

<span id="page-16-3"></span>Figure 12: The template for calculating consistency.

# <span id="page-17-1"></span>E EFFICIENCY ANALYSIS

Table 4: Efficiency evaluation on Haystack-Ego4D. Baseline results are directly cited from [Ye et al.](#page-11-2) [\(2025\)](#page-11-2). We report the overall latency of temporal search and answering. Evaluations are conducted on the Haystack-Ego4D using A100 GPUs. Temporal search metrics are reported in Tab. [1.](#page-5-0)

| Method                     | Question Grounding | Frame Retrieval | Latency (sec) ↓ |
|----------------------------|--------------------|-----------------|-----------------|
| VideoAgent                 | GPT4               | CLIP-1B         | 34.9            |
| Retrieval-based            | –                  | YOLO-world-110M | 32.2            |
| ∗<br>T<br>(Detector-based) | LLaVA-OV-7B        | YOLO-world-110M | 11.1            |
| TimeSearch-R               | –                  | SigLIP-400M     | 13.4            |

TimeSearch-R attains an end-to-end latency of 13.4 seconds on the Haystack-Ego4D test set, yielding a 61.6% speed-up over the 34.9-second latency of VideoAgent. Despite T<sup>∗</sup> operating with the lightweight YOLO-World-110M detector and completing inference in 11.1 seconds, our method maintains a comparable runtime while avoiding the complexity of hand-crafted scheduling. As shown in Tab. [1,](#page-5-0) TimeSearch-R markedly surpasses these baselines in temporal search metrics and QA accuracy, underscoring the effectiveness of reinforcement-driven temporal policies.

# <span id="page-17-2"></span><span id="page-17-0"></span>F TRAINING DETAILS

Table 5: Training hyperparameters of TimeSearch-R.

| Category             | Parameter                      | Value          |
|----------------------|--------------------------------|----------------|
|                      | Max FPS                        | 2              |
|                      | Max Frames per Video           | 768            |
| Video Processing     | Total Video Tokens             | 10,240         |
|                      | Min Tokens per Frame           | 12             |
|                      | Max Tokens per Frame           | 256            |
|                      | Max Search Turns               | 8              |
| Interaction Settings | Max Completion Length per Turn | 256            |
|                      | Number of Generations          | 8              |
|                      | KL Penalty Coefficient (β)     | 0.005          |
|                      | Scale Rewards                  | false          |
| GRPO Training        | Batch Size per GPU             | 1              |
|                      | Gradient Accumulation Steps    | 2              |
|                      | DeepSpeed Configuration        | ZeRO-3 Offload |
| Infrastructure       | VLLM Mode                      | colocate       |
|                      | Replay Buffer                  | true           |

We summarize the key hyperparameters in Table [5](#page-17-2) for reproducibility.

Training Configuration. TimeSearch-R employs a distributed training setup using PyTorch's native distributed data parallel framework with ZeRO-3 memory optimization through DeepSpeed. The training process leverages gradient accumulation to simulate larger batch sizes while maintaining memory efficiency on GPU clusters. We utilize mixed precision training with bfloat16 to accelerate computation while preserving numerical stability, coupled with Flash Attention 2.0 for efficient attention computation.

GRPO Training Setup. The reinforcement learning phase uses Group Relative Policy Optimization with 8 generations per prompt to provide sufficient policy gradient estimates. The KL divergence penalty coefficient β is set to 0.005 to balance between reward optimization and policy regularization. We employ VLLM in colocate mode for efficient inference during rollout generation, enabling faster

policy updates. This RL training stage is implemented on top of the TRL library [\(von Werra et al.,](#page-11-14) [2020\)](#page-11-14), following standard practice for outcome-driven policy optimization in large language models.

Video Processing Configuration. The model processes videos with a maximum of 768 frames and allocates up to 10,240 tokens for video content representation. Each interaction turn is limited to 8 search operations, with a maximum of 8 interaction turns per question to ensure comprehensive temporal exploration while maintaining computational efficiency. Frame tokens are dynamically allocated between 12 and 256 tokens per frame based on content complexity and relevance.

# <span id="page-18-0"></span>G MORE CASE STUDIES

This section provides more case studies of TimeSearch-R, including successful cases and failed cases.

Successful Cases. These representative success cases illustrate how TimeSearch-R conducts multiturn exploration to accumulate decisive visual evidence while maintaining alignment between the reasoning trace and the final answer. They encompass confirmation (Fig. [13\)](#page-19-0), elimination (Fig. [14\)](#page-19-1), and sequential exploration patterns (Fig. [15\)](#page-20-0), collectively demonstrating that the policy preserves the high completeness and consistency reported in Sec. [D.](#page-16-0)

Failed Cases. Figure [16](#page-21-0) illustrates a residual failure where the policy halts after reviewing only two of four candidate segments, leading to an incorrect answer. Figure [17](#page-21-1) illustrates a failure where the model hallucinates information related to riding.

# <span id="page-18-1"></span>H BROADER IMPACTS

TimeSearch-R contributes to several important areas beyond the immediate technical contributions:

Advancing Video Interpretability and Explainability. TimeSearch-R introduces interleaved text-video reasoning traces that provide transparent insights into the model's decision-making process. The completeness and consistency criteria we propose enable quantitative assessment of long-form video explanations, making temporal search decisions auditable and interpretable. This advancement represents a significant step toward more explainable AI systems in the video domain, where understanding the reasoning process is crucial for building trust and ensuring reliability.

Transforming Video Reasoning from Static to Dynamic Paradigms. Our approach fundamentally shifts the paradigm from static frame sampling to dynamic, interactive reasoning in video understanding. By operationalizing hypothesis-driven exploration through iterative temporal search, we promote a new methodology that emphasizes transparent, stepwise evidence gathering. This contrasts sharply with traditional one-shot inference over fixed visual contexts, encouraging researchers to develop more adaptive and interactive AI systems. The demonstrated effectiveness of our approach may inspire broader adoption of similar interactive paradigms across various multimodal tasks.

Exploring Scalable Weakly-Supervised Process Rewards. We introduce outcome-based process supervision that eliminates the need for costly process annotations. Through the integration of weak supervision and reinforcement learning via completeness self-verification, our method successfully aligns intermediate search decisions with correct outcomes. This approach offers a scalable solution for training complex interactive systems without requiring fine-grained procedural labels, potentially reducing annotation costs and enabling broader application across diverse domains.

# <span id="page-18-2"></span>I THE USE OF LARGE LANGUAGE MODELS

The authors declare that Large Language Models were used in this paper for polishing the writing. Specifically, the LLM assisted with tasks such as grammar checking, sentence simplification, and improving the overall fluency of the text. It is important to note that the LLM was not used for any literature review or research ideation. All research ideas and experimental analyses presented in the paper were solely conducted by the authors.

![](_page_19_Figure_1.jpeg)

<span id="page-19-0"></span>Figure 13: Search pattern: search confirmation.

![](_page_19_Figure_3.jpeg)

<span id="page-19-1"></span>Figure 14: Search pattern: elimination method.

![](_page_20_Figure_1.jpeg)

<span id="page-20-0"></span>Figure 15: Search pattern: sequential search.

![](_page_21_Figure_1.jpeg)

<span id="page-21-0"></span>Figure 16: Failure case: insufficient search. There were 4 options in total, but only 2 were reviewed before the search was terminated.

![](_page_21_Figure_3.jpeg)

<span id="page-21-1"></span>Figure 17: Failure case: visual hallucination. No information related to riding was found in the search results.