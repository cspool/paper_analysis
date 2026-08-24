## Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning

Qifan Yu <sup>1</sup> Zhenyu He <sup>1</sup> Sijie Li <sup>1</sup> Xun Zhou <sup>2</sup> Jun Zhang <sup>2</sup> Jingjing Xu <sup>2</sup> Di He <sup>1</sup>

## Abstract

Chain-of-Thought (CoT) prompting has emerged as a powerful technique for enhancing language model's reasoning capabilities. However, generating long and correct CoT trajectories is challenging. Recent studies have demonstrated that Looped Transformers possess remarkable length generalization capabilities, but their limited generality and adaptability prevent them from serving as an alternative to auto-regressive solutions. To better leverage the strengths of Looped Transformers, we propose RELAY (REasoning through Loop Alignment iterativelY). Specifically, we align the steps of Chain-of-Thought (CoT) reasoning with loop iterations and apply intermediate supervision during the training of Looped Transformers. This additional iteration-wise supervision not only preserves the Looped Transformer's ability for length generalization but also enables it to predict CoT reasoning steps for unseen data. Therefore, we leverage this Looped Transformer to generate accurate reasoning chains for complex problems that exceed the training length, which will then be used to fine-tune an auto-regressive model. We conduct extensive experiments, and the results demonstrate the effectiveness of our approach, with significant improvements in the performance of the auto-regressive model. Code will be released at <https://github.com/qifanyu/RELAY>.

## 1. Introduction

Reasoning plays a central role in shaping effective decisionmaking processes and guiding problem-solving strategies in artificial intelligence systems. For large language models (LLMs), the most effective way to achieve reasoning is through Chain-of-Thought [\(Wei et al.,](#page-9-0) [2022;](#page-9-0) [Khot et al.,](#page-9-1) [2022\)](#page-9-1), which generates all intermediate steps token by token until the final answer is reached. However, generating the correct reasoning process using LLMs is challenging. On

Working in Progress.

one hand, the Chain-of-Thought process can be very long, sometimes growing polynomially with respect to the prompt length [\(Feng et al.,](#page-8-0) [2024;](#page-8-0) [Merrill & Sabharwal,](#page-9-2) [2024\)](#page-9-2). When the reasoning length exceeds the training data length, it encounters the length generalization problem, where accuracy can drop significantly [\(Xiao & Liu,](#page-10-0) [2023;](#page-10-0) [Jin et al.,](#page-9-3) [2024\)](#page-9-3). On the other hand, web data is often noisy, and learning from incorrect trajectories can lead to incorrect answers. While synthetic data could mitigate this issue [\(Lightman et al.,](#page-9-4) [2024\)](#page-9-4), it requires significant human effort and knowledge to generate and curate.

Recently, an alternative framework has gained attention, known as the looped Transformer [\(Giannou et al.,](#page-8-1) [2023\)](#page-8-1). In general, the looped Transformer is a standard Transformer model with cross-block parameter sharing, like Al-BERT [\(Lan et al.,](#page-9-5) [2020\)](#page-9-5). In this framework, the input prompt (i.e., the problem) is processed through repeated iterations of the same block, with the number of iterations adaptively determined by the problem complexity. See Figure [1](#page-1-0) for an illustration. Several preliminary results [\(Fan](#page-8-2) [et al.,](#page-8-2) [2024\)](#page-8-2) show that the looped Transformer model has better length generalization capabilities, partially because the increase in problem complexity (e.g., problem length) is not as significant as in the Chain-of-Thought steps.

However, the success of this approach comes with some practical limitations. While determining appropriate loop iterations is feasible for reasoning tasks, it becomes problematic in general tasks, such as translation and summarization. Furthermore, although the looped Transformer can handle specific reasoning tasks, it remains unclear whether it possesses the capability to manage multiple reasoning tasks within a single model. Given these concerns, a natural question arises: if the looped Transformer is a general reasoner, can we explore ways to integrate its capabilities into the Chain-of-Thought framework of standard auto-regressive models? This integration would allow us to leverage the looped Transformer's strong performance on complex reasoning problems while preserving the versatility that allows auto-regressive models to excel in diverse language tasks.

In this paper, we introduce RELAY (REasoning through Loop Alignment iterativelY), a novel framework that leverages looped Transformer's superior capabilities to help autoregressive models handle longer reasoning chains. At its

<sup>1</sup> Peking University <sup>2</sup>ByteDance Inc..

> **[图片提取文字 (无描述)]:**
> linear linear Transformer block looped Transformer block iterations determined by the complexity of problem embedding embedding Auto-regressive CoT Model Looped Model reasoning token answer token problem token
![](_page_1_Picture_1.jpeg)

Figure 1. Visualization of Chain-of-Thought (CoT) and looping process. As the complexity of problem increases, in the autoregressive CoT model, the number of reasoning tokens escalates. In contrast, in the looped model, the number of iterations of the loop block increases.

<span id="page-1-0"></span>core, our approach centers on two key innovations. First, we demonstrate empirically that a single looped Transformer model can serve as a general reasoner across multiple tasks while maintaining strong length generalization abilities. Second, we propose an iteration-wise alignment between the looped Transformer and Chain-of-Thought reasoning steps, enabling the looped model to generate accurate reasoning chains for problems beyond training length. These generated reasoning chains can then serve as training data for auto-regressive models, establishing a bridge between the two architectural paradigms.

We conduct extensive experiments demonstrating that our approach significantly improves the reasoning abilities of auto-regressive Transformers through high-quality generated reasoning chains.

## 2. Related Work

### 2.1. Auto-regressive LLM with Chain-of-Thought

Chain-of-Thought (CoT) has emerged as a powerful technique for enhancing language models' reasoning capabilities both empirically [\(Wei et al.,](#page-9-0) [2022;](#page-9-0) [Khot et al.,](#page-9-1) [2022\)](#page-9-1) and theoretically [\(Feng et al.,](#page-8-0) [2024;](#page-8-0) [Merrill & Sabharwal,](#page-9-2) [2024\)](#page-9-2), especially in latest models such as OpenAI O1[1](#page-1-1) , DeepSeek r1 [\(DeepSeek-AI et al.,](#page-8-3) [2025\)](#page-8-3) and Qwen QwQ[2](#page-1-2) . By generating intermediate reasoning steps token by token, these models effectively decompose complex problems into sequential subprocesses. However, two critical challenges persist. First, obtaining high-quality CoT training data remains time-consuming and labor-intensive [\(Lightman et al.,](#page-9-4) [2024\)](#page-9-4), especially for problems requiring sophisticated reasoning chains. Second, the generation and understanding of extended reasoning sequences can be problematic [\(Xiao &](#page-10-0) [Liu,](#page-10-0) [2023;](#page-10-0) [Jin et al.,](#page-9-3) [2024;](#page-9-3) [Mao et al.,](#page-9-6) [2024\)](#page-9-6).

### 2.2. Looped Transformer

Research on looped Transformers has evolved significantly over recent years. The initial studies by [Dehghani et al.](#page-8-4) [\(2019\)](#page-8-4) and [Lan et al.](#page-9-5) [\(2020\)](#page-9-5) demonstrated the effectiveness of parameter sharing across layers in supervised learning and BERT pretraining. This line of research has since expanded in both theoretical and practical directions. On the theoretical front, [Giannou et al.](#page-8-1) [\(2023\)](#page-8-1) and [Xu & Sato](#page-10-1) [\(2024\)](#page-10-1) established fundamental properties of looped Transformers, proving their Turing completeness and characterizing their approximation capabilities. [Gatmiry et al.](#page-8-5) [\(2024\)](#page-8-5) further advanced this understanding by showing how to incorporate inductive biases for learning iterative algorithms, particularly in the context of multi-step gradient descent for in-context learning. Empirically, looped Transformers have shown promising results across various applications. [Yang et al.](#page-10-2) [\(2024\)](#page-10-2) demonstrated their parameter efficiency in data-fitting tasks, while [de Luca & Fountoulakis](#page-8-6) [\(2024\)](#page-8-6) and [Chen et al.](#page-8-7) [\(2024\)](#page-8-7) revealed their potential in graph algorithm simulation and in-context learning enhancement. Notably, [Fan et al.](#page-8-2) [\(2024\)](#page-8-2) established their superior length generalization capabilities in RASP-L tasks. In the domain of algorithm learning, [Gao et al.](#page-8-8) [\(2024\)](#page-8-8) introduced Algo-Former, a framework that leverages looped Transformers for algorithm representation and learning. While these works have extensively explored various aspects of looped Transformers, our work takes a distinct direction. We specifically focus on leveraging the better length generalization of looped Transformers for helping standard auto-regressive Transformers.

### 2.3. Approaches for Length Generalization

The capability of Transformers to generalize to longer sequence is influenced by their positional encodings [\(Press](#page-9-7) [et al.,](#page-9-7) [2022\)](#page-9-7). Recent research has pursued two primary directions to enhance length generalization capabilities of LLMs. The first focuses on developing advanced relative positional encoding schemes [\(Raffel et al.,](#page-9-8) [2020;](#page-9-8) [Press et al.,](#page-9-7) [2022;](#page-9-7) [Chi et al.,](#page-8-9) [2022;](#page-8-9) [Sun et al.,](#page-9-9) [2023;](#page-9-9) [Chi et al.,](#page-8-10) [2023;](#page-8-10) [Li et al.,](#page-9-10) [2024\)](#page-9-10), while the second explores modifications to positional representations through index granularity adjustments [\(Chen et al.,](#page-8-11) [2023;](#page-8-11) [Peng et al.,](#page-9-11) [2024\)](#page-9-11) and strategic index shifting [\(Ruoss et al.,](#page-9-12) [2023;](#page-9-12) [Zhu et al.,](#page-10-3) [2024\)](#page-10-3). These works are orthogonal to the central contributions of this paper. A parallel line of work focuses on improving the reasoning capabilities of LLMs through better training data. These methods typically leverage accessible labels or rewards to generate and filter reasoning steps, selecting those that yield correct solutions or high rewards [\(Zelikman et al.,](#page-10-4) [2022;](#page-10-4) [Yuan et al.,](#page-10-5) [2024;](#page-10-5) [Singh et al.,](#page-9-13) [2024;](#page-9-13) [Hosseini et al.,](#page-9-14) [2024\)](#page-9-14). However, a critical limitation emerges from LLMs' tendency to generate incorrect or superfluous intermediate reasoning steps while still arriving at correct solutions

<span id="page-1-1"></span><sup>1</sup> https://openai.com/o1

<span id="page-1-2"></span>https://qwenlm.github.io/blog/qwq-32b-preview

through chance [\(Paul et al.,](#page-9-15) [2024\)](#page-9-15). This phenomenon significantly constrains the effectiveness of LLM fine-tuning for complex reasoning tasks [\(Xia et al.,](#page-10-6) [2024;](#page-10-6) [Zhou et al.,](#page-10-7) [2023\)](#page-10-7).

## 3. Methodology

### 3.1. Notation

Any reasoning task can be decomposed into three components: the problem tokens, the reasoning tokens (i.e., chainof-thought steps), and the answer tokens. Let the problem token sequence be represented as x = [x1, x2, . . . , xn]. The Chain-of-Thought (CoT) process generates a sequence of intermediate reasoning tokens z = [z1, z2, . . . , zm], where n and m denote the number of problem tokens and reasoning tokens respectively. In this work, we focus on a simple setting where the problem's answer is represented by a single token, denoted as y.

CoT Auto-regressive Generation. For auto-regressive generation, the mapping from the problem sequence x to the answer y is performed through generating the intermediate tokens z token by token. Formally, this can be expressed as:

$$z_i \sim P(z_i|\mathbf{z}_{\leq i}, \mathbf{x}; \theta), \quad \text{for } i = 1, 2, \dots, m,$$
 (1)

where z<i = [z1, z2, . . . , zi−1] represents precedent reasoning tokens. and the final answer is obtained at the final step:

$$y \sim P(y|\boldsymbol{z}, \boldsymbol{x}; \theta).$$
 (2)

Looped Model. Different from the auto-regressive model that generates explicit tokens to obtain the answer, the looped model implicitly maps the input sequence x to the final answer y by executing the same function (e.g., a multilayer Transformer block) for T times in the representation space. The number of iterations T depends on the problem comlexity. The forward process consists of three steps: First, the token sequence x is mapped to embeddings through an embedding function h:

$$\boldsymbol{e}_0 = h(\boldsymbol{x}; \theta_{\text{emb}}), \tag{3}$$

where e<sup>0</sup> ∈ R <sup>d</sup>×<sup>n</sup> and d is the hidden dimension. Second, the embeddings are iteratively refined through transformation f:

$$e_t = f(e_{t-1}; \theta_{\text{model}}), \text{ for } t = 1, 2, \dots, T,$$
 (4)

where f is usually a Transformer model and the number of iterations T is adaptively determined based on the problem length. Finally, the answer is predicted through a finalanswer prediction head based on the representations in the last layer:

$$y \sim P(y|e_T; \theta_{\text{pred}}).$$
 (5)

This design enables the model to perform implicit reasoning through iterative refinement in the representation space, where each iteration can automatically capture different aspects of the reasoning process.

As a comparison, the CoT auto-regressive generation derives the final output y by first generating a sequence of intermediate reasoning tokens z = [z1, z2, . . . , zm] in an auto-regressive manner, where the length m can grow polynomially with input length n (i.e., m ∼ poly(n)). This variable and potentially large m poses challenges for positional encoding to correctly reflect attention relationships, leading to low accuracy for long sequence reasoning. In contrast, the looped model takes a fundamentally different approach. It directly processes the input sequence x and produces output y. The network only needs to handle x without z, mitigating the long-length problem in the reasoning chain.

### <span id="page-2-0"></span>3.2. Length Generalization on Single Reasoning Task

Before introducing our RELAY framework, we first empirically demonstrate the superior length generalization capability of looped Transformers compared to standard autoregressive models. This analysis serves as the foundation and motivation for our proposed framework.

Task Descriptions. To validate the capabilities of different methods, we use three representative tasks adapted from [Feng et al.](#page-8-0) [\(2024\)](#page-8-0), including Arithmetic, a mathematical task, and two dynamic programming (DP) problems: Edit Distance (ED) and Longest Increasing Subsequence (LIS). These tasks are selected for their diverse problemsolving patterns and varying levels of complexity, and the fact that they can be solved through a Chain-of-Thought reasoning process to arrive at the final answer. Performance is evaluated based on the accuracy of the final answer for both models. Detailed descriptions of these tasks are provided in Appendix [A.](#page-11-0)

Experimental Setup. For each task, we construct a dataset consisting of 1 million training samples and 100 k test samples, respectively. For the Arithmetic task, the problem complexity is defined as the number of operators. For the Edit Distance (ED) task, the problem complexity corresponds to the length of the shorter string in each pair. For the Longest Increasing Subsequence (LIS) task, we define the problem complexity as ⌈n/10⌉, where n is the length of the input sequence, as our dataset is structured with 10 numbers per reasoning step (see Appendix [A](#page-11-0) for details). The training datasets are constructed with the length of the problem token sequence x ≤ 15, 30, and 100 for Arithmetic, ED, and LIS, respectively. To evaluate the model's generalization capabilities, test datasets are created with problem lengths in the ranges [15, 25] for Arithmetic, [30, 40] for ED, and [100, 120] for LIS.

> **[图片提取文字 (无描述)]:**
> Task: Arithmetic Task: ED Task: LIS Models AR-CoT
> Vanilla Looped ≥ 0.8 0.8 8 0.6 0.0 30 31 32 33 34 35 36 37 38 39 40 15 16 17 18 19 20 21 22 23 24 25 120 104 116 Problem Length Problem Length Problem Length
![](_page_3_Figure_1.jpeg)

<span id="page-3-0"></span>Figure 2. Length generalization performance of looped Transformer versus auto-regressive CoT model on Arithmetic (train:  $\leq 15$ , test: [15, 25]), Edit Distance (train:  $\leq 30$ , test: [30, 40]), and Longest Increasing Subsequence (train:  $\leq 100$ , test: [100, 120]).

For the auto-regressive CoT model, we employ a standard decoder-only Transformer language model. For the looped model, we use an encoder-only Transformer with bi-directional attention. To address varying problem complexities, we implement dynamic iteration control in the looped model, setting the number of loop iterations equal to the problem complexity. The architectural configuration remains consistent across all models, comprising 3 layers, 256-dimensional hidden states, and 4 attention heads. For positional encoding, we adopt RoPE (Su et al., 2024) across all model variants to enhance sequence encoding for both training and test cases.

**Results.** Figure 2 illustrates the comparative performance of both models. Within the training distribution (e.g. Arithmetic:  $\leq 15$  operators), both the looped Transformer and the auto-regressive CoT Transformer achieve perfect accuracy. However, the models exhibit markedly different behaviors when tested on problems exceeding the training length: While the auto-regressive CoT Transformer's performance deteriorates significantly, the looped Transformer maintains superior performance across all length regimes. This demonstrates the length generalization capabilities of the looped Transformer.

While looped Transformers exhibit superior performance in final answer prediction, they lack interpretability in their intermediate computational processes. Moreover, their design philosophy may struggle with general language tasks, as determining the number of loop iterations becomes challenging beyond reasoning problems. This work seeks to harness the accurate reasoning predictions of the looped model to guide the training of auto-regressive Chain-of-Thought (CoT) models to better handle long-sequence reasoning.

### 3.3. Loop-Enhanced Chain-of-Thought Reasoning

A straightforward way to leverage a well-trained looped model to enhance the auto-regressive CoT model is by using it as a verifier. When a problem is presented, both models generate a final answer, and if both answers match, the CoT output is trusted. However, this approach often fails in practice, as CoT models can produce incorrect reasoning

trajectories even when reaching the correct final answer (see Section 4.3), making it unreliable to rely solely on the accuracy of the final answer as the guiding signal.

Our key insight is that an alignment can be established between the iterative structure of the looped Transformer and the stepwise nature of CoT reasoning. As shown in Figure 3, unlike the step-by-step token generation in CoT, looped models update their representations simultaneously in each iteration, and the number of such iterations naturally corresponds to the number of reasoning rounds. This structural similarity opens up the possibility of training the looped model to generate the corresponding CoT tokens for each round in parallel, while maintaining its ability to predict the final answer. With this insight, we propose **RELAY** (**RE**asoning through **L**oop **A**lignment iterativel**Y**), a two-stage framework that bridges looped and auto-regressive models.

Stage I: Training Looped Model with Explicit CoT Alignment. In the first stage, we train the looped model to generate intermediate reasoning processes that align with CoT steps. To formalize this alignment, assume we have a reasoning chain with T rounds. Given reasoning tokens  $z = [z_1, z_2, \dots, z_m]$ , denote  $k_t$  as the start token position of t-th reasoning round, where each round contains valid reasoning tokens  $\mathbf{z}_{[k_t:k_{t+1}-1]} = [z_{k_t}, z_{k_{t+1}}, \dots, z_{k_{t+1}-1}].$ Taking arithmetic reasoning as an example, consider a sequence of tokens representing the complete reasoning chain, " $3 \times 2 + 6 \div 3 = 6 + 6 \div 3 = 6 + 2 = 8$ ". This sequence can be naturally divided into T=3 rounds using the equal signs as delimiters: Given the input problem " $3 \times 2 + 6 \div 3 =$ ", the first round corresponds to " $6 + 6 \div 3 =$ ", the second round corresponds to "6 + 2 =", and the third (last) round presents the final answer "8".

Although the number of rounds aligns with the iteration count of the looped model, a key challenge arises from the mismatch in token lengths across different reasoning steps. For instance, earlier steps involving complex expressions (e.g., " $6+6\div 3=$ ") typically require more tokens than later steps (e.g., "6+2="). This variable length nature poses a challenge for the looped model, which requires fixed-length

Stage I: Training Looped Model with Explicit CoT Alignment Stage II: Enhancing Auto-regressive CoT Models

> **[图片提取文字 (无描述)]:**
> problem token reasoning token looped Transformer block looped Transformer block linear linear answer token linear O-positional encoding looped Transformer block looped Transformer block linear linear - causal mask Transformer block looped Transformer block linear looped Transformer block linear embedding embedding embedding complex problem simple problem ... (training data) (b) Generate Reasoning Tokens via the Looped Model (c) Long Reasoning Chain for (a) Train Looped Model with Alignment Auto-regressive Model Training
![](_page_4_Figure_3.jpeg)

<span id="page-4-0"></span>Figure 3. Overview of the RELAY framework. Stage I (left): Training looped model with explicit CoT alignment, where each iteration of the looped model learns to predict corresponding Chain-of-Thought (CoT) steps. Stage II (right): Using the trained looped model to generate CoT chains for enhancing auto-regressive CoT models. The looped model generates high-quality CoT chains for complex problems (beyond training length), which are then used to fine-tune the auto-regressive model to improve its reasoning capabilities.

representations of size n across iterations.

To address this length mismatch while preserving the parallel processing capability of the looped model, we employ a right-aligned padding strategy. For the t-th iteration, we construct a fixed-length sequence z˜<sup>t</sup> of length n by rightaligning the ground truth reasoning tokens z[kt:kt+1−1] and filling the remaining left positions with <pad> tokens. The fixed-length is determined based on the maximum length among all reasoning rounds and the original input problem (note that the length of a reasoning round usually does not exceed the length of the input problem; otherwise, each round can be further divided into shorter rounds). To track both valid reasoning tokens and the boundary of padding, we introduce a binary mask:

$$M_t[i] = \begin{cases} 1, & \text{if } i = p_t \text{ or } \tilde{z}_t[i] \neq \text{}, \\ 0, & \text{otherwise}, \end{cases}$$
 (6)

where M<sup>t</sup> indicates the positions of valid reasoning tokens and the position of the last <pad> token pt.

Using this alignment strategy, we train the looped model to predict the corresponding CoT tokens at each iteration, enabling it to generate CoT-aligned intermediate outputs. In detail, at each iteration t, we train the model to predict both the valid reasoning tokens and the last <pad> token through an intermediate prediction head:

$$P(\tilde{z}_t|e_t;\theta_{\text{pred-cot}}),$$
 (7)

For the intermediate reasoning steps, we ignore all preceding <pad> tokens except the last one, as they have no impact on the reasoning process. The loss of this part can

be formulated as :

$$\mathcal{L}_{\text{iter}} = \frac{1}{T} \sum_{t=1}^{T} \text{CrossEntropy}(P(\tilde{\boldsymbol{z}}_t | \boldsymbol{e}_t; \theta_{\text{pred-cot}}), \tilde{\boldsymbol{z}}_t) \odot M_t,$$
(8)

where the element-wise multiplication ⊙ ensures that the loss is computed only on valid reasoning tokens and the last <pad> token.

For the final answer, we have the answer prediction loss to ensure correct final predictions:

$$\mathcal{L}_{ans} = CrossEntropy(P(\boldsymbol{y}|\boldsymbol{e}_T; \theta_{pred}), \boldsymbol{y}), \qquad (9)$$

where y is the ground truth answer. The total training loss is then:

$$\mathcal{L} = \mathcal{L}_{ans} + \lambda \mathcal{L}_{iter}, \tag{10}$$

where λ is a hyperparameter balancing the two objectives.

This design enables the looped model to accurately predict the answer and provide interpretable intermediate reasoning steps that can be effectively utilized to guide the auto-regressive model in Stage II.

Stage II: Enhancing Auto-regressive CoT Models. In the second stage, we leverage the trained looped model to enhance auto-regressive CoT models through a systematic process:

First, we use the trained looped model in Stage I to generate reasoning demonstrations for problems of increasing complexity. For each problem x, we obtain:

$$(\boldsymbol{z}, y) \sim p(\cdot | \boldsymbol{x}; \theta_L),$$
 (11)

where θ<sup>L</sup> denotes the trained looped model from Stage I, z = [z1, z2, . . . , zm] represents the generated reasoning tokens across iterations, and y is the predicted answer.

We then utilize these demonstrations to fine-tune an autoregressive model. For problem lengths beyond the original training range, we generate a comprehensive dataset of reasoning demonstrations using the looped model. This newly generated data is then merged with the original training dataset, which contains problems within the initial training length. The combined dataset, spanning both the original and extended problem lengths, is then used to fine-tune the auto-regressive model in a single step. This approach allows the model to retain its original reasoning capabilities while acquiring the ability to effectively tackle more complex, longer problems, utilizing the structured insights provided by the demonstrations.

Comparison with Synthetic Data Generation Approaches. To effectively guide the LLMs to handle complex problems, prior works [\(Hendrycks et al.,](#page-8-12) [2021;](#page-8-12) [Light](#page-9-4)[man et al.,](#page-9-4) [2024\)](#page-9-4) have explored the synthetic data generation approach, where human labelers construct data generation pipelines based on their understanding of both the task and its solution process. This approach requires labelers to possess comprehensive knowledge in three aspects: (1) problem construction, (2) problem-solving strategies, and (3) pipeline development skills. While effective, this creates a high barrier for deployment across diverse domains, as finding experts who excel in all three areas can be challenging.

In contrast, RELAY reduces these requirements. Our approach follows a more automated pipeline: training data → looped model with strong generalization capability → longer problem construction → automated reasoning generation → auto-regressive CoT model training. The human involvement is primarily limited to longer problem construction, eliminating the need for expertise in solution strategies and pipeline development. This reduction in human expertise requirements makes our method more practical and scalable across different domains. Additionally, by leveraging the looped models' inherent generalization capabilities rather than manually designed rules, our approach can potentially capture more nuanced reasoning patterns that might be overlooked in hand-crafted pipelines.

## 4. Experiments

This section presents a comprehensive empirical evaluation of our RELAY framework through a series of experiments designed to address four key research questions:

- Q1: How effectively does the looped model with explicit CoT alignment serve as a general-purpose reasoner across diverse tasks? (Section [4.1\)](#page-5-0)
- Q2: What advantages does the looped model with explicit CoT alignment demonstrate in length generalization compared to auto-regressive CoT models? (Section [4.1\)](#page-5-0)

- Q3: How can the length generalization capabilities of the looped model with explicit CoT alignment be leveraged to enhance auto-regressive CoT models? (Section [4.2\)](#page-6-1)
- Q4: How reliable are the intermediate reasoning steps generated by the looped model with explicit CoT alignment? (Section [4.3\)](#page-6-0)

We address each question through carefully designed experiments, as detailed below.

### <span id="page-5-0"></span>4.1. Multitask Training

Following the single-task evaluation discussed in Section [3.2,](#page-2-0) we extend our analysis to a multitask learning setting to explore the general reasoning capabilities of three models: the looped model with explicit CoT alignment, the auto-regressive CoT model, and the vanilla looped model. In this setup, we jointly train the models on three representative reasoning tasks: Arithmetic, Edit Distance (ED), and Longest Increasing Subsequence (LIS), each requiring multi-step reasoning to arrive at accurate final answers. This setting enables a thorough comparison of the models' ability to generalize effectively across diverse tasks.

For a detailed description of the tasks, including example inputs, expected answers, and the corresponding Chain-of-Thought (CoT) reasoning steps, please refer to Appendix [A.](#page-11-0)

Experimental Setup. We conduct experiments on the three tasks: Arithmetic, Edit Distance (ED), and Longest Increasing Subsequence (LIS), to evaluate the generalization capabilities of the looped model with explicit CoT alignment in comparison with the auto-regressive CoT model and the vanilla looped model. Training datasets retain the same problem lengths as in Section [3.2:](#page-2-0) operator counts of ≤ 15 for Arithmetic, input string lengths of ≤ 30 for ED, and sequence lengths ≤ 100 for LIS. Similarly, test datasets are constructed with extended problem lengths of [15, 25], [30, 40], and [100, 120] for Arithmetic, ED, and LIS, respectively, to assess length generalization.

In this setup, each model—the looped model with explicit CoT alignment, the auto-regressive CoT model, and the vanilla looped model—is trained jointly on all three tasks by prepending a task-specific problem token ([ARI], [ED], [LIS]) to the input sequence, which distinguishes among tasks. All models are evaluated under the same metric, considering only the accuracy of final answer.

Results. Figure [4](#page-6-2) illustrates the comparative performance of the three models across the three tasks. All models achieve nearly 100 % accuracy on all tasks within the training distribution, demonstrating that the looped model with explicit CoT alignment can serve as a general-purpose reasoning engine capable of handling diverse tasks requiring

> **[图片提取文字 (无描述)]:**
> Task: Arithmetic Task: ED Task: LIS Models RELAY-enhanced CoT Looped with Explicit CoT Alignment ≥ 0.8 0.8 AR-CoT + Self Chains & Loop Answers 0.6 0.6 AR-CoT + Self Chains & GT Answers AR-CoT Baseline Vanilla Looped 0.0 0.0 15 16 17 18 19 20 21 22 23 24 25 35 36 100 104 120 Problem Length Problem Length Problem Length
![](_page_6_Figure_1.jpeg)

<span id="page-6-2"></span>Figure 4. Performance comparison of different models on long reasoning problems across three tasks: Arithmetic, Edit Distance (ED), and Longest Increasing Subsequence (LIS).

multi-step reasoning. (Q1)

However, for problems with lengths exceeding the training range, the looped model with explicit CoT alignment and the vanilla looped model significantly outperform the auto-regressive CoT model, showcasing the superiority of loop-based architectures in addressing tasks requiring generalization to longer inputs. Furthermore, the looped model with explicit CoT alignment not only maintains the strong length generalization capability of the vanilla looped model but even surpasses it notably, benefiting from the explicit alignment between CoT reasoning steps and loop iterations. This alignment provides structural guidance that enhances the model's reasoning capabilities over extended lengths as well as the ability to generate explicit intermediate CoT reasoning chains, making it both accurate and interpretable. These results establish the looped model with explicit CoT alignment as both a robust reasoning framework and a generally effective solution for length generalization challenges, outperforming standard auto-regressive CoT models across diverse tasks. (Q2)

