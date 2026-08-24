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

# <span id="page-6-1"></span>4.2. Enhancing Auto-regressive Model with RELAY-Generated CoT Data

In this section, we utilize the looped model with explicit CoT alignment trained in Section 4.1 to enhance the performance of the auto-regressive CoT model via effective data generation. Specifically, we leverage its ability to produce accurate reasoning chains for complex problems exceeding the training lengths. These reasoning chains serve as high-quality data, which are subsequently employed to fine-tune the auto-regressive CoT model.

**Experimental Setup.** First, we employ the looped model with explicit CoT alignment to generate CoT reasoning chains for problems of increased complexity, covering problem lengths of [15, 25], [30, 40], and [100, 120] for Arithmetic, ED, and LIS tasks, respectively. These newly generated data is then merged with the original training dataset, which contains problems within the initial training length. Details of the sample proportions for different

problem lengths when merging datasets are provided in Appendix D.2.

Next, we fine-tune the auto-regressive CoT model on this augmented dataset in a single phase. This fine-tuning process builds upon the well-trained auto-regressive CoT model from Section 4.1, retaining the same model structure while updating the weights with the augmented dataset. This process enables the model to incorporate longer CoT chains, thereby enhancing its reasoning capabilities on extended sequences.

**Results.** Figure 4 presents the accuracy curves of the RELAY-enhanced auto-regressive CoT model across problem lengths for the three tasks. Compared to the baseline auto-regressive CoT model, the auto-regressive CoT model fine-tuned with data generated by RELAY (i.e., by the looped model with explicit CoT alignment) exhibits significant improvements on problems exceeding the original training length. Notably, its performance approaches and even slightly surpasses that of the looped model with explicit CoT alignment in some cases, while consistently outperforming the baseline auto-regressive CoT model.

These results indicate that our RELAY framework effectively utilizes the length generalization capabilities of the looped model with explicit CoT alignment to improve the overall performance of auto-regressive model. By generating high-quality CoT reasoning data, RELAY enables the auto-regressive CoT model to better handle problems beyond its original training range, without altering its architecture. (Q3)

# <span id="page-6-0"></span>**4.3.** Evaluating the Reliability of RELAY-Generated Intermediate Reasoning Steps

This section aims to demonstrate the reliability of CoT chains generated by the looped model with explicit CoT alignment compared to the auto-regressive CoT model's self-generated data. Specifically, we highlight that the generated data from the auto-regressive CoT model, even when the final result is correct, often contains incorrect intermediate steps, which is why utilizing these data fails to improve

> **[图片提取文字 (无描述)]:**
> RELAY-Generated CoT Self-Generated 1.0 1.00 1.00 1.00 1.00 Step Ste 1.00 1.00 1.00 1.00 Token Token
![](_page_7_Figure_1.jpeg)

<span id="page-7-0"></span>Figure 5. Hit accuracy matrices for the LIS task with a problem length of 105 (T = 11 steps).

the model's performance in complex problems with longer lengths. In contrast, data generated by the looped model with explicit CoT alignment avoids these issues by ensuring both accurate intermediate reasoning steps and the final answer, enabling effective fine-tuning of the auto-regressive model and significantly enhancing its performance.

**Experimental Setup.** We evaluate the effectiveness of two types of generated data: by the looped model with explicit CoT alignment and the auto-regressive CoT model. This evaluation focuses on two metrics: (1) hit matrix and (2) bit accuracy, which provides a detailed perspective on reasoning steps reliability.

For the hit matrix, we select the LIS task as an example due to the structured nature of its reasoning steps. The intermediate reasoning steps of LIS tasks follows a  $T \times 11$  matrix format, where T corresponds to the number of CoT steps as well as the iteration number of the looped model, and 11 represents the number of tokens per step (10 numbers as prescribed in our dataset, along with one delimeter <sep>). This structured format makes the LIS task particularly suitable for evaluating and visualizing reasoning step reliability, offering an intuitive representation of the proportion of tokens at each position that match the ground truth reasoning steps.

Bit accuracy is provided across all three tasks of varying lengths, evaluating token-wise counted accuracy for the whole reasoning step. Comparisons are made between the auto-regressive CoT models fine-tuned by the two types of generated data respectively.

For the auto-regressive CoT model self-generated data, we conduct the following experiment under two parallel settings, using either a vanilla looped model or ground-truth answers as verifiers. The experiment consists of the following steps: (1) Use the auto-regressive CoT model to generate CoT chains for long problem lengths. (2) Filter these data by the looped model or ground-truth, retaining only those where the final answers match. (3) The filtered data are then used to fine-tune the auto-regressive CoT model. The fine-tuning process aims to improve the model's ability to

generate reasoning trajectories and reach the correct final answer for longer problems.

Meanwhile, data generated by the looped model with explicit CoT alignment is employed as the fine-tuning dataset for the same initial auto-regressive CoT model checkpoint, under the same fine-tuning parameters and controlled ratio of samples with different lengths. Both approaches are evaluated across three tasks with varying lengths to assess performance improvements.

Results. We evaluate the hit accuracy matrix for the LIS task with a problem length of 105, which corresponds to  $T = \lceil 105/10 \rceil = 11$  steps, resulting in an  $11 \times 11$  matrix (We also provide results for problem length of 101 in Appendix B). As shown in Figure 5, data generated by the looped model with explicit CoT alignment achieves consistently high token accuracy across all positions, with most values approaching 100 %, demonstrating its ability to produce high-quality and reliable data. (Q4) In contrast, the data generated by the auto-regressive CoT model exhibits high token accuracy only in the first few positions, while the accuracy steadily decreases in later steps. Although the delimiter tokens <sep> at the end of each step achieve high accuracy, this simply implies that the auto-regressive CoT model has only captured the basic format of the reasoning process but fails to predict accurate tokens, which indicates its limited capability to maintain accurate prediction throughout the reasoning process for longer problems.

The bit accuracy results for the models fine-tuned with different datasets across the three tasks (Arithmetic, ED, LIS) and varying problem lengths are provided in Appendix C.

We additionally provide the accuracy of the final answer for the auto-regressive CoT model fine-tuned with self-generated data in Figure 4, noted as "AR-CoT + Self Chains & Loop/GT Answers", corresponding to data filtered by the looped model or ground-truth answers, respectively, which only shows a slight improvement over the baseline model.

#### 5. Conclusion

This paper introduces **RELAY** (**RE**asoning through **L**oop **A**lignment iterativel **Y**), a framework enhancing Chain-of-Thought reasoning by combining looped and auto-regressive Transformers. Our contributions show that (1) a looped Transformer can serve as a general-purpose reasoner with strong length generalization, (2) iteration-wise alignment enables accurate reasoning chain generation beyond training length, and (3) RELAY improves auto-regressive models through high-quality generated reasoning chains. Future work could explore the theoretical foundations of looped Transformers' length generalization and extend RELAY to broader language tasks.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

- <span id="page-8-7"></span>Chen, B., Li, X., Liang, Y., Shi, Z., and Song, Z. Bypassing the exponential dependency: Looped transformers efficiently learn in-context by multi-step gradient descent. *arXiv preprint arXiv:2410.11268*, 2024.
- <span id="page-8-11"></span>Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*, 2023.
- <span id="page-8-9"></span>Chi, T.-C., Fan, T.-H., Ramadge, P. J., and Rudnicky, A. Kerple: Kernelized relative positional embedding for length extrapolation. *Advances in Neural Information Processing Systems*, 35:8386–8399, 2022.
- <span id="page-8-10"></span>Chi, T.-C., Fan, T.-H., Rudnicky, A., and Ramadge, P. Dissecting transformer length extrapolation via the lens of receptive field analysis. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 13522–13537, 2023.
- <span id="page-8-6"></span>de Luca, A. B. and Fountoulakis, K. Simulation of graph algorithms with looped transformers. In *Fortyfirst International Conference on Machine Learning*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=aA2326y3hf) [id=aA2326y3hf](https://openreview.net/forum?id=aA2326y3hf).
- <span id="page-8-3"></span>DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang,

- X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL <https://arxiv.org/abs/2501.12948>.
- <span id="page-8-4"></span>Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, L. Universal transformers. In *International Conference on Learning Representations*, 2019. URL [https://openreview.net/forum?](https://openreview.net/forum?id=HyzdRiR9Y7) [id=HyzdRiR9Y7](https://openreview.net/forum?id=HyzdRiR9Y7).
- <span id="page-8-2"></span>Fan, Y., Du, Y., Ramchandran, K., and Lee, K. Looped transformers for length generalization. *arXiv preprint arXiv:2409.15647*, 2024.
- <span id="page-8-0"></span>Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., and Wang, L. Towards revealing the mystery behind chain of thought: a theoretical perspective. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-8-8"></span>Gao, Y., Zheng, C., Xie, E., Shi, H., Hu, T., Li, Y., Ng, M. K., Li, Z., and Liu, Z. On the expressive power of a variant of the looped transformer. *arXiv preprint arXiv:2402.13572*, 2024.
- <span id="page-8-5"></span>Gatmiry, K., Saunshi, N., Reddi, S. J., Jegelka, S., and Kumar, S. Can looped transformers learn to implement multi-step gradient descent for in-context learning? *arXiv preprint arXiv:2410.08292*, 2024.
- <span id="page-8-1"></span>Giannou, A., Rajput, S., Sohn, J.-Y., Lee, K., Lee, J. D., and Papailiopoulos, D. Looped transformers as programmable computers. In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 11398–11442. PMLR, 23–29 Jul 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/giannou23a.html) [v202/giannou23a.html](https://proceedings.mlr.press/v202/giannou23a.html).
- <span id="page-8-12"></span>Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=7Bywt2mQsCe) [id=7Bywt2mQsCe](https://openreview.net/forum?id=7Bywt2mQsCe).

- <span id="page-9-14"></span>Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., and Agarwal, R. V-STar: Training verifiers for self-taught reasoners. In *First Conference on Language Modeling*, 2024. URL [https://openreview.net/](https://openreview.net/forum?id=stmqBSW2dV) [forum?id=stmqBSW2dV](https://openreview.net/forum?id=stmqBSW2dV).
- <span id="page-9-3"></span>Jin, M., Yu, Q., Shu, D., Zhao, H., Hua, W., Meng, Y., Zhang, Y., and Du, M. The impact of reasoning step length on large language models. In *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 1830–1842. Association for Computational Linguistics, August 2024. URL [https://aclanthology.org/](https://aclanthology.org/2024.findings-acl.108/) [2024.findings-acl.108/](https://aclanthology.org/2024.findings-acl.108/).
- <span id="page-9-1"></span>Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., and Sabharwal, A. Decomposed prompting: A modular approach for solving complex tasks. *arXiv preprint arXiv:2210.02406*, 2022.
- <span id="page-9-5"></span>Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., and Soricut, R. Albert: A lite bert for selfsupervised learning of language representations. In *International Conference on Learning Representations*, 2020. URL [https://openreview.net/forum?](https://openreview.net/forum?id=H1eA7AEtvS) [id=H1eA7AEtvS](https://openreview.net/forum?id=H1eA7AEtvS).
- <span id="page-9-10"></span>Li, S., You, C., Guruganesh, G., Ainslie, J., Ontanon, S., Zaheer, M., Sanghai, S., Yang, Y., Kumar, S., and Bhojanapalli, S. Functional interpolation for relative positions improves long context transformers. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=rR03qFesqk) [id=rR03qFesqk](https://openreview.net/forum?id=rR03qFesqk).
- <span id="page-9-4"></span>Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let's verify step by step. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=v8L0pN6EOi) [id=v8L0pN6EOi](https://openreview.net/forum?id=v8L0pN6EOi).
- <span id="page-9-6"></span>Mao, Y., Li, J., Meng, F., Xiong, J., Zheng, Z., and Zhang, M. Lift: Improving long context understanding through long input fine-tuning. *arXiv preprint arXiv:2412.13626*, 2024.
- <span id="page-9-2"></span>Merrill, W. and Sabharwal, A. The expressive power of transformers with chain of thought. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=NjNGlPh8Wh) [id=NjNGlPh8Wh](https://openreview.net/forum?id=NjNGlPh8Wh).
- <span id="page-9-15"></span>Paul, D., West, R., Bosselut, A., and Faltings, B. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pp. 15012–15032, Miami, Florida, USA,

- November 2024. Association for Computational Linguistics. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.findings-emnlp.882/) [findings-emnlp.882/](https://aclanthology.org/2024.findings-emnlp.882/).
- <span id="page-9-11"></span>Peng, B., Quesnelle, J., Fan, H., and Shippole, E. YaRN: Efficient context window extension of large language models. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.](https://openreview.net/forum?id=wHBfxhZu1u) [net/forum?id=wHBfxhZu1u](https://openreview.net/forum?id=wHBfxhZu1u).
- <span id="page-9-7"></span>Press, O., Smith, N., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. 2022.
- <span id="page-9-8"></span>Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 21(1):5485–5551, 2020.
- <span id="page-9-12"></span>Ruoss, A., Deletang, G., Genewein, T., Grau-Moya, J., ´ Csordas, R., Bennani, M., Legg, S., and Veness, J. Ran- ´ domized positional encodings boost length generalization of transformers. pp. 1889–1903, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-short.161. URL [https://](https://aclanthology.org/2023.acl-short.161/) [aclanthology.org/2023.acl-short.161/](https://aclanthology.org/2023.acl-short.161/).
- <span id="page-9-13"></span>Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Garcia, X., Liu, P. J., Harrison, J., Lee, J., Xu, K., Parisi, A. T., Kumar, A., Alemi, A. A., Rizkowsky, A., Nova, A., Adlam, B., Bohnet, B., Elsayed, G. F., Sedghi, H., Mordatch, I., Simpson, I., Gur, I., Snoek, J., Pennington, J., Hron, J., Kenealy, K., Swersky, K., Mahajan, K., Culp, L. A., Xiao, L., Bileschi, M., Constant, N., Novak, R., Liu, R., Warkentin, T., Bansal, Y., Dyer, E., Neyshabur, B., Sohl-Dickstein, J., and Fiedel, N. Beyond human data: Scaling self-training for problem-solving with language models. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL [https://](https://openreview.net/forum?id=lNAyUngGFK) [openreview.net/forum?id=lNAyUngGFK](https://openreview.net/forum?id=lNAyUngGFK). Expert Certification.
- <span id="page-9-16"></span>Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. *Neurocomput.*, 568(C), March 2024.
- <span id="page-9-9"></span>Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A., Chaudhary, V., Song, X., and Wei, F. A lengthextrapolatable transformer. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, July 2023.
- <span id="page-9-0"></span>Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in*

- *neural information processing systems*, 35:24824–24837, 2022.
- <span id="page-10-6"></span>Xia, M., Malladi, S., Gururangan, S., Arora, S., and Chen, D. LESS: Selecting influential data for targeted instruction tuning. In *International Conference on Machine Learning (ICML)*, 2024.
- <span id="page-10-0"></span>Xiao, C. and Liu, B. Conditions for length generalization in learning reasoning skills. *arXiv preprint arXiv:2311.16173*, 2023.
- <span id="page-10-1"></span>Xu, K. and Sato, I. On expressive power of looped transformers: Theoretical analysis and enhancement via timestep encoding. *arXiv preprint arXiv:2410.01405*, 2024.
- <span id="page-10-2"></span>Yang, L., Lee, K., Nowak, R. D., and Papailiopoulos, D. Looped transformers are better at learning learning algorithms. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https:](https://openreview.net/forum?id=HHbRxoDTxE) [//openreview.net/forum?id=HHbRxoDTxE](https://openreview.net/forum?id=HHbRxoDTxE).
- <span id="page-10-5"></span>Yuan, Z., Yuan, H., Li, C., Dong, G., Lu, K., Tan, C., Zhou, C., and Zhou, J. Scaling relationship on learning mathematical reasoning with large language models, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=cijO0f8u35) [id=cijO0f8u35](https://openreview.net/forum?id=cijO0f8u35).
- <span id="page-10-4"></span>Zelikman, E., Wu, Y., Mu, J., and Goodman, N. STar: Bootstrapping reasoning with reasoning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=_3ELRdg2sgI) [id=\\_3ELRdg2sgI](https://openreview.net/forum?id=_3ELRdg2sgI).
- <span id="page-10-7"></span>Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., YU, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., and Levy, O. LIMA: Less is more for alignment. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL [https:](https://openreview.net/forum?id=KBMOKmX2he) [//openreview.net/forum?id=KBMOKmX2he](https://openreview.net/forum?id=KBMOKmX2he).
- <span id="page-10-3"></span>Zhu, D., Yang, N., Wang, L., Song, Y., Wu, W., Wei, F., and Li, S. PoSE: Efficient context window extension of LLMs via positional skip-wise training. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=3Z1gxuAQrA) [id=3Z1gxuAQrA](https://openreview.net/forum?id=3Z1gxuAQrA).

## <span id="page-11-0"></span>A. Task Descriptions

Below, we present the detailed descriptions of each task from [Feng et al.](#page-8-0) [\(2024\)](#page-8-0), including examples of inputs, expected answers, and the corresponding Chain-of-Thought (CoT) reasoning steps used to derive the final answers.

- 1. Arithmetic. This task involves computing the answer of arithmetic expressions containing numbers, basic operations (+, −, ×, ÷, =), and brackets. For example:
  - Input: (6 + 9) ÷ (7 + 2 × 5 − 4 × 3) =
  - CoT Steps:

$$15 \div (7 + 2 \times 5 - 4 \times 3) =$$

$$15 \div (7 + 10 - 4 \times 3) =$$

$$15 \div (17 - 4 \times 3) =$$

$$15 \div (17 - 12) =$$

$$15 \div 5 =$$

- Answer: 3
- 2. Edit Distance (ED). This task requires computing the minimum number of operations (insert, delete, or replace) needed to transform one sequence into another. The input consists of two sequences separated by a delimiter |:
  - Input: o t m l | o t t m l <sep> • CoT Steps: 0 2 4 6 7 , 2 0 2 4 6 , 4 2 3 2 4 ,

6 4 5 4 2 ,

• Answer: 2

Each row corresponds to the edit distance matrix, and the final answer is the edit distance.

- 3. Longest Increasing Subsequence (LIS). This task identifies the length of longest strictly increasing subsequence in a numerical sequence. The input is a sequence of integers followed by a delimiter <sep>:
  - Input: 103 110 145 217 233 18 30 82 141 150 159 161 167 239 <sep>
  - CoT Steps:

1 2 3 4 5 1 2 3 4 5 <sep> 6 7 8 9 9 9 9 9 9 9 <sep>

• Answer: 9

Here, each CoT step represents an intermediate computation in the dynamic programming process, folded into fixed-size groups (10 numbers per step in our setting) to align with the model structure. If the last group has fewer than 10 numbers, the last number is repeated until the group size reaches 10.

## <span id="page-11-1"></span>B. Hit Matrix for LIS Task with Length 101

We also analyze the hit accuracy matrix for the LIS task with a problem length of 101, corresponding to T = ⌈101/10⌉ = 11 reasoning steps, as shown in Figure [6.](#page-12-1) The results exhibit a similar trend to those observed for length 105, with a notable decline in accuracy in the later reasoning steps for data generated by the auto-regressive CoT model, while RELAY-generated data consistently maintains high accuracy. Specifically, while the initial steps maintain relatively high token accuracy, the accuracy deteriorates significantly in later steps, failing to achieve accurate final answers. This highlights a key limitation of using CoT self-generated data for supervision: even when the problem length is only slightly beyond the training length, the CoT model struggles to generate accurate reasoning steps towards the end, making it infeasible to fine-tune the model using only the final answer as supervision, as the lack of intermediate reasoning accuracy prevents meaningful improvements in model's performance of handling longer problems.

> **[图片提取文字 (无描述)]:**
> RELAY-Generated CoT Self-Generated 1.0 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 - 0.8 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 Step 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 1.00 0.99 0.99 1.00 1.00 1.00 1.00 1.00 - 0.2 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.98 0.98 0.99 0.99 0.99 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 1.00 1.00 1.00 0.00 0.00 0.00 8 Token Token
![](_page_12_Figure_1.jpeg)

<span id="page-12-1"></span>Figure 6. Hit accuracy matrices for the LIS task with a problem length of 101 (T = 11 steps).

## <span id="page-12-0"></span>C. Bit Accuracy

The bit accuracy results for the models fine-tuned with different datasets across the three tasks (Arithmetic, ED, LIS) and varying problem lengths are shown in Figure 7. Each subfigure corresponds to one task and includes curves showing the bit accuracy of five models over varying problem lengths: (1) **RELAY-enhanced CoT:** the auto-regressive CoT model fine-tuned with data generated by RELAY, (2) **Looped with Explicit CoT Alignment**, (3) **AR-CoT + Self Chains & Loop Answers:** the auto-regressive CoT model fine-tuned with its self-generated data using looped model answers as labels, (4) **AR-CoT + Self Chains & GT Answers:** the auto-regressive CoT model fine-tuned with its self-generated data using ground-truth answers as labels, and (5) **AR-CoT Baseline:** the baseline auto-regressive CoT model as a reference, also as the initial auto-regressive CoT model before fine-tuned.

As illustrated in Figure 7, both the looped model with CoT alignment and the auto-regressive CoT model fine-tuned with data generated by it consistently achieve high bit accuracy across all tasks and problem lengths. Notably, they maintain over 90% bit accuracy even at lengths extending beyond the training data (up to +10 for Arithmetic and ED tasks, and +20 for the LIS task in our setting). This highlights not only the robustness and reliability of the looped model with CoT alignment in length extrapolation scenarios but also the effectiveness of its generated data in significantly enhancing the performance of the auto-regressive CoT model.

In contrast, the auto-regressive CoT model fine-tuned with its own self-generated data shows limited improvement over the baseline model. This is consistent across all tasks and highlights a critical limitation: the self-generated data often contain incorrect intermediate steps, even when the final results are correct. These inaccuracies hinder the model's ability to generalize and perform well on longer problem lengths, reinforcing the importance of reliable intermediate reasoning steps for effective fine-tuning.

> **[图片提取文字 (无描述)]:**
> Task: Arithmetic Task: ED Task: LIS Models --- RELAY-enhanced CoT Looped with Explicit
> CoT Alignment AR-CoT + Self Chains & Loop Answers AR-CoT + Self Chains & GT Answers AR-CoT Baseline 15 16 17 18 19 20 21 22 23 24 25 30 31 32 33 34 35 36 37 38 116 120 Problem Length Problem Length Problem Length
![](_page_12_Figure_7.jpeg)

<span id="page-12-2"></span>Figure 7. Bit accuracy over varying problem lengths for three tasks: Arithmetic, ED, and LIS.

## D. Details for training and fine-tuning

## D.1. Hyper-parameters

In our experiments, we trained three different models: (1) the looped model with CoT alignment, (2) the auto-regressive CoT model, and (3) the vanilla looped model. All models were trained from scratch on the same dataset, which consists of 1 million samples for each of the three tasks. The task-specific training weights and training hyper-parameters are provided in Table [1.](#page-13-1)

| Training Hyper-parameters | Looped Model with CoT Alignment | CoT Model | Vanilla Looped Model |
|---------------------------|---------------------------------|-----------|----------------------|
| Epoch                     | 500                             | 500       | 500                  |
| Batch Size                | 512                             | 512       | 512                  |
| Learning Rate             | 5e-4                            | 5e-4      | 1e-3                 |
| Learning Rate Schedule    | linear                          | linear    | linear               |
| Warmup Ratio              | 0.01                            | 0.01      | 0.01                 |
| Optimizer                 | AdamW                           | AdamW     | AdamW                |
| Weight Decay              | 0.01                            | 0.01      | 0.01                 |
| Drop out                  | 0.1                             | 0.1       | 0.1                  |
| Weight of ARI             | 1                               | 1         | 1                    |
| Weight of ED              | 1                               | 10        | 10                   |
| Weight of LIS             | 1                               | 5         | 5                    |

<span id="page-13-1"></span>Table 1. Training Hyper-parameters of Different Models

For fine-tuning, we used data generated by RELAY and CoT model self-generated data to fine-tune the auto-regressive CoT model. The fine-tuning process followed a similar setup, as detailed in Table [2.](#page-13-2)

| Fine-tuning Hyper-parameters | RELAY-Generated Data | Self-Generated Data |  |
|------------------------------|----------------------|---------------------|--|
| Epoch                        | 500                  | 100 (per phase)     |  |
| Batch Size                   | 512                  | 512                 |  |
| Learning Rate                | 1e-4                 | 5e-5                |  |
| Learning Rate Schedule       | linear               | linear              |  |
| Warmup Ratio                 | 0.01<br>0.01         |                     |  |
| Optimizer                    | AdamW                | AdamW               |  |
| Weight Decay                 | 0.01                 | 0.01                |  |
| Drop out                     | 0.1                  | 0.1                 |  |
| Weight of ARI                | 1                    | 1                   |  |
| Weight of ED                 | 10                   | 1                   |  |
| Weight of LIS                | 5                    | 1                   |  |

<span id="page-13-2"></span>Table 2. Fine-tuning Hyper-parameters

### <span id="page-13-0"></span>D.2. Sample Length Distribution of Datasets

The original training dataset for training the three models consists of 1 million samples for each task, following a distribution where the number of samples is proportional to problem length.

The merged dataset used for fine-tuning consists of 100 k samples for each of the three tasks, incorporating both the original training data and newly generated samples from extended problem lengths.

For the looped model with explicit CoT alignment, we introduce additional data covering problem lengths of [16, 25] for Arithmetic, [31, 40] for ED, and [101, 120] for LIS. These newly generated samples are merged with the original dataset while maintaining a balanced proportion across different length ranges to ensure effective training. The specific numbers of

#### Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning

samples for different problem lengths in the final merged dataset are provided in Table [3.](#page-15-0)

For the self-generated dataset, we adopt an incremental approach, since the accuracy of original CoT model diminishes rapidly as the problem length increases. Specifically, we maintain a total dataset size of 100 k samples for each task. The initial dataset consists of problems with lengths ≤ 15, 30, and 100 for Arithmetic, ED, and LIS, respectively. The CoT model is progressively fine-tuned over five phases, each including self-generation on slightly longer problems and followed by 100 epochs of fine-tuning. After each phase, a subset of the current dataset is randomly combined with the newly generated reasoning steps to form an updated synthetic dataset. The maximum number of samples selected for each problem length is detailed in Table [4.](#page-16-0)

Table 3. Number of Samples for Different Problem Lengths in Merged Dataset

<span id="page-15-0"></span>

| Task                        | Arithmetic | ED         | LIS         |
|-----------------------------|------------|------------|-------------|
| Length                      | ≤ 15       | ≤ 30       | ≤ 100       |
| Number of Samples           | 42515      | 60844      | 73235       |
| Length                      | 16         | 31         | 101         |
| Number of Samples           | 6477       | 4195       | 1479        |
| Length                      | 17         | 32         | 102         |
| Number of Samples           | 6882       | 4195       | 1464        |
| Length                      | 18         | 33         | 103         |
| Number of Samples           | 7287       | 4055       | 1449        |
| Length                      | 19         | 34         | 104         |
| Number of Samples           | 6477       | 4055       | 1434        |
| Length                      | 20         | 35         | 105         |
| Number of Samples           | 6072       | 3916       | 1420        |
| Length                      | 21         | 36         | 106         |
| Number of Samples           | 5668       | 3916       | 1405        |
| Length                      | 22         | 37         | 107         |
| Number of Samples           | 5263       | 3776       | 1390        |
| Length                      | 23         | 38         | 108         |
| Number of Samples           | 4858       | 3776       | 1375        |
| Length                      | 24         | 39         | 109         |
| Number of Samples           | 4453       | 3636       | 1360        |
| Length<br>Number of Samples | 25<br>4048 | 40<br>3636 | 110<br>1346 |
|                             |            |            |             |
| Length<br>Number of Samples |            |            | 111<br>1331 |
| Length                      |            |            | 112         |
| Number of Samples           |            |            | 1316        |
| Length                      |            |            | 113         |
| Number of Samples           |            |            | 1301        |
| Length                      |            |            | 114         |
| Number of Samples           |            |            | 1286        |
| Length                      |            |            | 115         |
| Number of Samples           |            |            | 1272        |
| Length                      |            |            | 116         |
| Number of Samples           |            |            | 1257        |
| Length<br>Number of Samples |            |            | 117<br>1242 |
| Length                      |            |            | 118         |
| Number of Samples           |            |            | 1227        |
| Length                      |            |            | 119         |
| Number of Samples           |            |            | 1213        |
| Length                      |            |            | 120         |
| Number of Samples           |            |            | 1198        |

<span id="page-16-0"></span>Table 4. Number of Samples Generated for Different Lengths

| Properties |                | ARI   | ED    | LIS   |
|------------|----------------|-------|-------|-------|
| Phase I    | Length         | 16    | 31    | 101   |
|            | Data Generated | 15000 | 15000 | 15000 |
| Phase II   | Length         | 17    | 32    | 102   |
|            | Data Generated | 10000 | 10000 | 10000 |
| Phase III  | Length         | 18    | 33    | 103   |
|            | Data Generated | 7500  | 7500  | 7500  |
| Phase IV   | Length         | 19    | 34    | 104   |
|            | Data Generated | 6000  | 3000  | 3000  |
| Phase V    | Length         | 20    | 35    | 105   |
|            | Data Generated | 5000  | 3000  | 3000  |