# PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning

## Zongqian Li, Yixuan Su, Nigel Collier

University of Cambridge {zl510, ys484, nhc30}@cam.ac.uk

## Abstract

Parameter-efficient fine-tuning (PEFT) methods have shown promise in adapting large language models, yet existing approaches exhibit counter-intuitive phenomena: integrating router into prompt tuning (PT) increases training efficiency yet does not improve performance universally; parameter reduction through matrix decomposition can improve performance in specific domains. Motivated by these observations and the modular nature of PT, we propose PT-MoE, a novel framework that integrates matrix decomposition with mixtureof-experts (MoE) routing for efficient PT. Results across 17 datasets demonstrate that PT-MoE achieves state-of-the-art performance in both question answering (QA) and mathematical problem solving tasks, improving F1 score by 1.49 points over PT and 2.13 points over LoRA in QA tasks, while enhancing mathematical accuracy by 10.75 points over PT and 0.44 points over LoRA, all while using 25% fewer parameters than LoRA. Our analysis reveals that while PT methods generally excel in QA tasks and LoRA-based methods in math datasets, the integration of matrix decomposition and MoE in PT-MoE yields complementary benefits: decomposition enables efficient parameter sharing across experts while MoE provides dynamic adaptation, collectively enabling PT-MoE to demonstrate cross-task consistency and generalization abilities. These findings, along with ablation studies on routing mechanisms and architectural components, provide insights for future PEFT methods. [1](#page-0-0)

## 1 Introduction

Large language models (LLMs) have shown remarkable capabilities but require improvements in efficiency across data [\(Li and Cole](#page-9-0) , [2025\)](#page-9-0), training, and inference [\(Li et al.,](#page-9-1) [2024,](#page-9-1) [2025a,](#page-9-2)[b\)](#page-9-3). PEFT methods address training efficiency challenge by

<span id="page-0-1"></span>![](_page_0_Figure_9.jpeg)

Figure 1: Performance comparison of PEFT methods on 12 QA datasets in the MRQA benchmark (upper) and 5 math datasets (lower). ↑ indicates higher is better; ↓ indicates lower is better.

updating only a small subset of parameters [\(Han](#page-8-0) [et al.](#page-8-0) , [2024\)](#page-8-0). Prompt tuning (PT) stands out among PEFT approaches with its unique advantages: minimizing trainable parameters through soft prompt optimization, enabling modular utilization through task-specific prompts without model modifications, and supporting flexible knowledge composition [\(Lester et al.](#page-8-1) , [2021\)](#page-8-1). These properties make it effective for low-resource and multi-task applications where efficient adaptation is essential.

Despite these advantages, we observe two counter-intuitive phenomena in prompt tuning. First, integrating router into prompt tuning does

<span id="page-0-0"></span><sup>1</sup> <https://github.com/ZongqianLi/PT-MoE>

not decrease training efficiency yet improves performance in specific domains rather than universally (SMoP vs PT in Table [2\)](#page-4-0), suggesting domaindependent optimization dynamics. Second, decomposing soft prompts into low-rank matrices, while reducing parameters, can surprisingly improve model performance in specific areas (DPT vs PT in Table [4\)](#page-5-0). These phenomena indicate that the relationship between parameter efficiency and model effectiveness in prompt tuning is more nuanced than previously understood, motivating the need for a more sophisticated approach to prompt optimization.

Based on these observations, we propose a novel framework, Prompt Tuning with Efficient Mixture-of-Experts (PT-MoE), that combines matrix decomposition with MoE routing. As shown in Figure [1,](#page-0-1) our approach not only achieves stateof-the-art performance, but also uses minimal trainable parameters and moderate training steps.

Our work makes three key contributions:

- Novel finetuning framework: We propose PT-MoE, integrating matrix decomposition with MoE for prompt tuning. Our framework achieves state-of-the-art performance with fewer parameters while outperforming either method alone, demonstrating their complementary benefits.
- Design dynamics: We thoroughly analyze key variables influencing the performance of PT-MoE, including prompt length, expert count, trainable parameters, routing mechanisms, and model size. Findings provide design guidelines for future parameter-efficient tuning approaches.
- Comprehensive analysis: We provide detailed empirical studies across diverse tasks, including QA and mathematical problem solving, establishing a basis for future work in efficient finetuning methods.

The remainder of this paper is organized as follows: Section [2](#page-1-0) reviews related work in prompt tuning, covering both direct tuning approaches and transfer learning methods. Section [3](#page-2-0) presents our PT-MoE framework, detailing the matrix decomposition strategy, dynamic router design, and training methodology. Section [4](#page-3-0) describes our experimental design across QA and mathematical problemsolving tasks. Section [5](#page-4-1) presents comprehensive results, including detailed ablation studies analyzing the influence of prompt length, parameter count, expert number, routing mechanisms, and model size, followed by efficiency analysis. Section [6](#page-7-0) concludes with key findings and future directions.

## <span id="page-1-0"></span>2 Related Work

To contextualize our approach, we review existing prompt tuning methods, which fall into two categories: direct prompt tuning approaches focusing on architectural innovations, and transfer learning methods enabling cross-task knowledge sharing.

Direct prompt tuning methods have developed into four main branches: (1) General approaches that directly optimize prompt parameters, including Prompt Tuning that prepends trainable vectors to input while freezing the language model [\(Lester](#page-8-1) [et al.,](#page-8-1) [2021\)](#page-8-1), XPrompt that employs hierarchical structured pruning to identify and retain important prompt tokens [\(Ma et al.,](#page-9-4) [2022\)](#page-9-4), and P-Tuning v2 that introduces deep prompts across all transformer layers [\(Liu et al.,](#page-9-5) [2022\)](#page-9-5); (2) Encoder-based methods that leverage additional modules, such as P-Tuning that incorporates an encoder to learn dependencies between continuous embeddings [\(Liu](#page-9-6) [et al.,](#page-9-6) [2023\)](#page-9-6), Residual Prompt Tuning (RPT) that employs a residual part with down/up-projection layers for stable optimization [\(Razdaibiedina et al.,](#page-9-7) [2023\)](#page-9-7), and Prefix Tuning that prepends trainable key-value pairs at each layer through a reparameterization section [\(Li and Liang,](#page-9-8) [2021\)](#page-9-8); (3) Decomposition methods that decompose prompt embeddings, including Decomposed Prompt Tuning (DPT) that applies low-rank matrix decomposition to reduce parameter count [\(Xiao et al.,](#page-10-0) [2023\)](#page-10-0), and DePT that combines shorter soft prompts with lowrank updates to word embeddings [\(Shi and Lipani,](#page-9-9) [2024\)](#page-9-9); and (4) MoE approaches such as Sparse Mixture-of-Prompts (SMoP) that employs multiple shorter prompts with a dynamic router to route inputs to the most suitable soft prompt [\(Choi et al.,](#page-8-2) [2023\)](#page-8-2).

Transfer learning approaches in prompt tuning have developed into three categories: (1) General approaches that directly transfer prompt knowledge, including SPoT that introduces both generic transfer through multi-task pre-training and targeted transfer via task similarity matching [\(Vu](#page-9-10) [et al.,](#page-9-10) [2022\)](#page-9-10), and ATTEMPT that dynamically combines multiple source prompts through an attention-based mixing mechanism with instancelevel adaptation [\(Asai et al.,](#page-8-3) [2022\)](#page-8-3); (2) Encoderbased methods that facilitate knowledge transfer through additional modules, such as TransPrompt that employs parallel task-specific and universal encoders with balancing mechanisms for obtaining both task-dependent and task-agnostic knowledge

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 2: **Framework** of PT-MoE. Each soft prompt is decomposed into an input-specific matrix  $A_i$  and a shared matrix B, with a router adaptively selecting and combining prompt components based on input. The resulting soft prompt is prepended to the input for the frozen LLM.

(Wang et al., 2021), and Cross-Task Prompt Tuning (CTPT) that leverages multi-head attention for cross-task knowledge transfer with dimension reduction and derivative-free optimization (Xu et al., 2023); and (3) Decomposition methods exemplified by Multitask Prompt Tuning (MPT) that decomposes prompts into shared and task-specific components through knowledge distillation, enabling efficient transfer while preserving task-specific adaptability through a rank-one decomposition strategy (Wang et al., 2023).

#### <span id="page-2-0"></span>3 Methods

Building upon the insights from prior work, we propose a new parameter-efficient prompt tuning framework, PT-MoE, shown in Figure 2 and Algorithm 1

Framework Overview. PT-MoE integrates matrix decomposition and dynamic routing. Given an input sequence  $\mathbf{x}$ , our framework first generates routing weights  $\mathbf{w}$  through a router  $R: \mathbf{w} = R(\mathbf{x})$ . These weights determine the selection among N decomposed prompts, where each prompt  $\mathbf{P}_i$  is decomposed as  $\mathbf{P}_i = \mathbf{A}_i \mathbf{B}$ , with  $\mathbf{A}_i$  being promptspecific and  $\mathbf{B}$  being shared across all prompts. The final soft prompt  $\mathbf{P}$  is computed as  $\mathbf{P} = \sum_{i=1}^N w_i \mathbf{A}_i \mathbf{B}$ , which is then prepended to the input sequence for the frozen language model.

**Matrix Decomposition.** To achieve parameter efficiency, we decompose each prompt matrix  $\mathbf{P}_i \in \mathbb{R}^{T \times H}$  into a prompt-specific matrix  $\mathbf{A}_i \in \mathbb{R}^{T \times R}$  and a shared matrix  $\mathbf{B} \in \mathbb{R}^{R \times H}$ , where T, H, and R denote the prompt length, hid-

## Algorithm 1 Pseudo code of PT-MoE

```
Require: Base model \mathcal{M}; input batch X = x_1, ..., x_b; pa-
     rameters \theta
Notation: b - batch size; s - sequence length; n - number of
      prompts; k - tokens per prompt; d - low-rank dimension;
      h - hidden dimension
 1: for batch x \in X do
         Get input embeddings E = \mathcal{M}_{embed}(x)
         where E \in \mathbb{R}^{b \times s \times h}
 3:
         Calculate mean embeddings
         \mu = \operatorname{mean}(E, \dim = 1) \ \text{where} \ \mu \in \mathbb{R}^{b \times h}
         Compute router logits l = W\mu + b
         where W \in \mathbb{R}^{n \times h}, b \in \mathbb{R}^n, l \in \mathbb{R}^{b \times n}
 5:
         Get router weights
         w = \operatorname{softmax}(\overline{l}) \text{ where } w \in \mathbb{R}^{b \times n}
 6:
         for each sample j in batch do
 7:
              Find indices of top-k weights:
              i_{topk} = \operatorname{argsort}(w_j)[-k:]
 8:
              Zero all weights except top-k:
              w_j[i] = 0 for all i \notin i_{topk}
 9.
         end for
         Initialize prompt embeddings P = 0, P \in \mathbb{R}^{b \times k \times d}
10:
11:
         for each weight w_i in w do
12:
              Compute weighted prompts
              P = P + w_i \tilde{A}_i \text{ where } \tilde{A}_i \in \mathbb{R}^{k \times d}
          end for
13:
          Project to model dimension
          P = P \times B \text{ where } B \in \mathbb{R}^{d \times h}
          Combine with input: C = \operatorname{concat}(P, E)
         where C \in \mathbb{R}^{b \times (\hat{k}+s) \times h}
16:
          Generate through base model: y = \mathcal{M}(C)
17: end for
Ensure: Model predictions y
```

<span id="page-2-2"></span>den dimension, and low-rank dimension respectively. This reduces parameters from O(NTH) to O(NTR+RH) for N prompts. The low-rank dimension R is either manually determined or computed to maintain parameter efficiency. For initialization, we first transform task-relevant text into word embeddings  $\mathbf{E} \in \mathbb{R}^{T \times H}$ , then perform SVD:  $\mathbf{E} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\top}$ . Each  $\mathbf{A}_i$  is initialized as  $\mathbf{U} \colon R \mathbf{\Sigma} R^{1/2}$  and the shared  $\mathbf{B}$  as  $\mathbf{\Sigma} R^{1/2} \mathbf{V}_{R}^{\top}$ , where subscript R indicates truncation to the first R components. This approach ensures the initial prompts have task-relevant information while maintaining the parameter efficiency of decomposition.

**Dynamic Router.** The router adaptively selects prompts based on input context. Given an input sequence embedding  $\mathbf{x} \in \mathbb{R}^H$  (obtained by averaging token embeddings), the router computes logits through a linear projection:  $\mathbf{l} = \mathbf{W}\mathbf{x} + \mathbf{b}$ , where  $\mathbf{W} \in \mathbb{R}^{N \times H}$  and  $\mathbf{b} \in \mathbb{R}^N$ . During training, we apply multiplicative Gaussian noise to encourage exploration:  $\mathbf{l}' = \mathbf{l} \odot (1 + \epsilon)$ , where  $\epsilon \sim \mathcal{N}(0, \sigma^2)$ . The routing weights are computed as  $\mathbf{w} = \operatorname{softmax}(\mathbf{l}') \odot \mathbf{1}_{\operatorname{argmax}}$ , where  $\mathbf{1}_{\operatorname{argmax}}$  is a one-hot vector with 1 at the position of the maxi-

#### <span id="page-3-1"></span>MRQA (Extractive QA) In-domain SQuAD [\(Rajpurkar et al.,](#page-9-12) [2016\)](#page-9-12), TriviaQA [\(Joshi et al.,](#page-8-4) [2017\)](#page-8-4), SearchQA [\(Dunn et al.,](#page-8-5) [2017\)](#page-8-5), HotpotQA [\(Yang et al.,](#page-10-3) [2018\)](#page-10-3), NaturalQuestions [\(Kwiatkowski et al.,](#page-8-6) [2019\)](#page-8-6) Out-ofdomain BioASQ [\(Partalas et al.,](#page-9-13) [2013\)](#page-9-13), DROP [\(Dua et al.,](#page-8-7) [2019\)](#page-8-7), DuoRC [\(Saha et al.,](#page-9-14) [2018\)](#page-9-14), RACE [\(Lai et al.,](#page-8-8) [2017\)](#page-8-8), RelationExtraction [\(Levy et al.,](#page-9-15) [2017\)](#page-9-15), TextbookQA [\(Kembhavi et al.,](#page-8-9) [2017\)](#page-8-9)

Mathematics (Problem Solving)

In-domain GSM8K [\(Cobbe et al.,](#page-8-10) [2021\)](#page-8-10)

Out-ofdomain SVAMP: Subtraction, Addition, Common-Division, Multiplication [\(Patel et al.,](#page-9-16) [2021\)](#page-9-16); ASDIV [\(Miao](#page-9-17) [et al.,](#page-9-17) [2020\)](#page-9-17); MAWPS [\(Koncel-Kedziorski et al.,](#page-8-11) [2016\)](#page-8-11); MATH\_PROBLEMS [\(Nebrelbug,](#page-9-18) [2024\)](#page-9-18)

Table 1: Overview of training and evaluation datasets that span two task categories: extractive QA (MRQA benchmark with 12 QA datasets) and mathematical problem solving (GSM8K and specific mathematical datasets). For each category, datasets are divided into in-domain sets used for training, validation, and evaluation, and out-ofdomain sets used exclusively for testing generalization ability.

mum value. This hard selection strategy reduces overlap between prompts while maintaining endto-end differentiability through straight-through estimation.

Training and Prediction. During training, we optimize both the router parameters and decomposed prompt matrices while keeping the base model frozen. For language model training, we use negative log-likelihood loss computed only on non-prompt positions using a binary mask: L = − P <sup>t</sup>∈M log p(y<sup>t</sup> |x<t), where M denotes non-prompt positions. We employ AdamW optimizer with warmup followed by a constant learning rate schedule, and gradient accumulation for stable optimization. At inference, noise is not added in the router, ensuring deterministic prompt selection.

## <span id="page-3-0"></span>4 Experimental Design

### 4.1 Datasets

We complete evaluations across 17 diverse datasets, as shown in Table [1,](#page-3-1) where in-domain datasets are split into training, validation, and test sets, while out-of-domain datasets are used exclusively for testing. For QA, we utilize 12 MRQA datasets [\(Fisch](#page-8-12) [et al.,](#page-8-12) [2019\)](#page-8-12), with in-domain sets like SQuAD [\(Ra](#page-9-12)[jpurkar et al.,](#page-9-12) [2016\)](#page-9-12) testing information extraction abilities and out-of-domain sets like DROP [\(Dua](#page-8-7) [et al.,](#page-8-7) [2019\)](#page-8-7) evaluating domain adaptation. For mathematical problem solving, we use GSM8K [\(Cobbe et al.,](#page-8-10) [2021\)](#page-8-10) from MetaMath [\(Yu et al.,](#page-10-4) [2024\)](#page-10-4) as our in-domain dataset, complemented by specific out-of-domain datasets including all the subsets of SVAMP [\(Patel et al.,](#page-9-16) [2021\)](#page-9-16), ASDIV [\(Miao et al.,](#page-9-17) [2020\)](#page-9-17), MAWPS [\(Koncel-Kedziorski](#page-8-11) [et al.,](#page-8-11) [2016\)](#page-8-11), and MATHPROBLEMS [\(Nebrelbug,](#page-9-18) [2024\)](#page-9-18).

### 4.2 Gold Standard and Baselines

We employ full model fine-tuning as our gold standard, which updates all parameters but requires substantial computational resources. Our baselines[2](#page-3-2) include typical methods from prompt tuning categories: For direct prompt tuning, we select (1) PT from general approaches, (2) DPT from decomposition methods, and (3) SMoP from MoE approaches. While transfer learning methods like (4) ATTEMPT typically involve multi-turn training, we also evaluate its architecture under similar training for comprehensive comparison. We additionally compare other PEFT methods including (5) LoRA and (6) HydraLoRA, with HydraLoRA adopting a MoE-like architecture that uses a shared down-projection matrix and multiple routed up-projection matrices. These two LoRAbased methods require model architecture modifications unlike the modular nature of prompt tuning methods.

## 4.3 Evaluation Metrices

We employ task-specific evaluation metrics. For extractive QA tasks from MRQA, we adopt two metrics: F1 score, which evaluates the token-level overlap between predicted and ground truth answer spans, balancing precision and recall; and Exact Match (EM), which measures the percentage of predictions that exactly match the ground truth. For mathematical problem solving tasks, we use accuracy, defined as the percentage of correctly solved problems with exact answer matches.

### 4.4 Models

We get our main results using LLaMA-3.2-1B-Instruct as the base model for fine-tuning methods

<span id="page-3-2"></span><sup>2</sup>All methods are controlled to have similar parameter budgets, with detailed configurations shown in Table [9](#page-11-0) of the Appendix.

<span id="page-4-0"></span>

|           | #     |       | In-domain |       |       |       |       |       | Out-of-domain |       |       |       |       |       |
|-----------|-------|-------|-----------|-------|-------|-------|-------|-------|---------------|-------|-------|-------|-------|-------|
| Method    | para. | SQ    | News      | Tri   | Srch  | HP    | NQ    | BSQ   | DP            | DRC   | RC    | RE    | TB    | Avg.  |
| FT        | 1.2B  | 78.76 | 48.69     | 71.04 | 71.35 | 72.96 | 67.56 | 70.19 | 43.87         | 48.11 | 43.44 | 81.60 | 52.71 | 62.52 |
| LoRA      | 106k  | 69.82 | 39.91     | 70.61 | 55.56 | 63.29 | 65.92 | 65.38 | 35.25         | 43.69 | 38.04 | 74.09 | 52.00 | 56.13 |
| HydraLoRA | 278k  | 74.24 | 44.05     | 71.38 | 60.13 | 64.02 | 66.31 | 68.76 | 34.38         | 44.36 | 40.00 | 77.97 | 52.44 | 58.17 |
| PT        | 81k   | 72.31 | 48.18     | 65.93 | 49.74 | 58.69 | 62.18 | 68.59 | 40.39         | 43.30 | 42.10 | 82.43 | 47.34 | 56.77 |
| DPT       | 81k   | 70.99 | 48.42     | 65.41 | 46.94 | 58.49 | 61.65 | 65.56 | 38.80         | 43.64 | 41.89 | 80.85 | 46.62 | 55.77 |
| SMoP      | 86k   | 74.15 | 48.96     | 66.13 | 41.08 | 58.96 | 61.17 | 68.59 | 39.92         | 42.07 | 42.34 | 83.73 | 47.85 | 56.25 |
| ATTEMPT   | 90k   | 74.22 | 48.18     | 65.31 | 37.64 | 60.18 | 59.59 | 66.69 | 45.32         | 42.86 | 43.01 | 84.11 | 46.91 | 56.17 |
| PT-MoE    | 80k   | 73.85 | 48.24     | 67.34 | 51.33 | 62.16 | 62.95 | 69.33 | 48.02         | 43.96 | 42.51 | 83.70 | 45.71 | 58.26 |

Table 2: Evaluation results (F1 scores) for various PEFT methods on MRQA datasets. SQ: SQuAD; News: NewsQA; Tri: TriviaQA; Srch: SearchQA; HP: HotpotQA; NQ: NaturalQuestions; BSQ: BioASQ; DP: DROP; DRC: DuoRC; RC: RACE; RE: RelationExtraction; TB: TextbookQA. The bold values indicate the best performance among prompt tuning-based methods.

<span id="page-4-2"></span>

|           | #     | In-domain |       |       |       |       |       | Out-of-domain |       |       |       |       |       |       |
|-----------|-------|-----------|-------|-------|-------|-------|-------|---------------|-------|-------|-------|-------|-------|-------|
| Method    | para. | SQ        | News  | Tri   | Srch  | HP    | NQ    | BSQ           | DP    | DRC   | RC    | RE    | TB    | Avg.  |
| FT        | 1.2B  | 65.28     | 32.76 | 62.29 | 61.50 | 56.19 | 50.45 | 49.06         | 32.26 | 38.84 | 29.52 | 66.99 | 43.71 | 49.07 |
| LoRA      | 106k  | 56.26     | 25.26 | 64.11 | 46.10 | 47.48 | 49.54 | 42.02         | 25.48 | 33.24 | 24.92 | 58.58 | 44.17 | 43.09 |
| HydraLoRA | 278k  | 61.63     | 27.80 | 64.32 | 50.06 | 47.73 | 49.59 | 44.01         | 24.75 | 33.57 | 26.11 | 62.68 | 43.97 | 44.69 |
| PT        | 81k   | 61.25     | 32.62 | 59.49 | 42.40 | 44.45 | 47.28 | 51.79         | 30.60 | 34.64 | 29.82 | 72.45 | 39.52 | 45.52 |
| DPT       | 81k   | 58.49     | 32.88 | 58.56 | 39.65 | 44.33 | 46.54 | 49.46         | 28.74 | 35.64 | 30.26 | 70.48 | 38.72 | 44.48 |
| SMoP      | 86k   | 63.15     | 32.81 | 59.48 | 34.51 | 43.80 | 46.39 | 50.06         | 29.94 | 34.11 | 30.56 | 74.59 | 40.25 | 44.97 |
| ATTEMPT   | 90k   | 63.71     | 32.50 | 58.71 | 31.24 | 45.77 | 45.66 | 49.26         | 36.06 | 34.84 | 30.41 | 75.13 | 39.52 | 45.23 |
| PT-MoE    | 80k   | 63.34     | 32.85 | 60.87 | 43.98 | 47.29 | 48.18 | 52.06         | 37.12 | 35.64 | 31.75 | 74.18 | 38.25 | 47.13 |

Table 3: Evaluation results (Exact Match) for MRQA datasets.

[\(Grattafiori et al.,](#page-8-13) [2024\)](#page-8-13). For ablation studies on model size, we additionally employ LLaMA-3.2- 3B-Instruct.

## <span id="page-4-1"></span>5 Results

#### 5.1 Question Answering

The results on MRQA datasets shown in Table [2](#page-4-0) and [3](#page-4-2) demonstrate the effectiveness of PT-MoE across various QA tasks. We highlight seven key findings: (1) PT-MoE achieves superior overall performance with an average F1 score of 58.26%, outperforming SMoP (56.25%) by 2.01 points and the standard PT (56.77%) by 1.49 points, establishing a new state-of-the-art on the MRQA benchmark. (2) This improvement is further validated by Exact Match metrics, where PT-MoE demonstrates even more gains (47.13% for average, outperforming SMoP and PT by 2.16 and 1.61 points respectively). (3) PT-MoE exhibits strong generalization abilities across both in-domain and out-of-domain scenarios. It achieves the highest performance on four out of six in-domain datasets and three out of six out-of-domain datasets. (4) The stability of PT-MoE is evidenced by consistent improvements over PT across 11 out of 12 datasets, with only marginal decreases in the RACE dataset. In contrast, SMoP shows performance decrease on 5 datasets compared to PT. (5) Individual architectural components show limited gains: both matrix decomposition (DPT, 55.77% F1) and MoE (SMoP, 56.25% F1) underperform standard prompt tuning

(PT, 56.77% F1). (6) PT-MoE's integration of matrix decomposition and MoE yields complementary benefits, outperforming both DPT and SMoP by 2.49 and 2.01 points for F1 respectively. This improvement over individual approaches proves the mutually beneficial nature of these methods. (7) Notably, while PT-MoE achieves lower overall performance than FT, it reaches comparable or even higher scores than FT on specific datasets such as DROP (48.02% vs 43.87% F1) while using only 80K parameters compared to FT's 1.2B. These results collectively validate the effectiveness of the architectural design of PT-MoE and demonstrate its superior performance in accuracy and generalization across diverse QA scenarios.

#### 5.2 Mathematical Problem Solving

The results on mathematical tasks reveal several characteristics compared to QA tasks. We highlight six key findings: (1) PT-MoE achieves stateof-the-art performance with an average accuracy of 56.91%, improving upon PT (46.16%) by 10.75 points, demonstrating its effectiveness in mathematical reasoning. (2) The benefits of MoE integration shows method-dependent characteristics: in prompt tuning approaches, PT-MoE and SMoP show different changes over PT (by +10.75 and -5.11 points respectively); when applied to LoRA methods, HydraLoRA shows slightly performance decrease compared to LoRA. (3) LoRAbased methods demonstrate advantages in mathe-

<span id="page-5-0"></span>

|            | #     | In-domain | Out-of-domain |          |          |                |       |       |       |       |         |
|------------|-------|-----------|---------------|----------|----------|----------------|-------|-------|-------|-------|---------|
| Method     | para. | GSM8K     | Subtraction   | Addition | Division | Multiplication | SVAMP | ASDIV | MAWPS | MP500 | Average |
| FT         | 1.2B  | 58.15     | 68.75         | 64.40    | 62.50    | 48.48          | 61.03 | 86.04 | 82.53 | 30.60 | 63.67   |
| LoRA       | 106k  | 41.77     | 67.50         | 61.01    | 52.08    | 33.33          | 53.48 | 73.42 | 70.70 | 43.00 | 56.47   |
| HydraLoRA  | 278k  | 41.31     | 57.50         | 62.71    | 52.08    | 39.39          | 52.92 | 74.08 | 76.05 | 33.40 | 55.55   |
| PT         | 81k   | 34.11     | 41.87         | 50.84    | 66.66    | 33.33          | 48.18 | 60.13 | 57.18 | 31.20 | 46.16   |
| Decomp. PT | 81k   | 26.08     | 43.12         | 35.59    | 64.58    | 27.27          | 42.64 | 56.14 | 43.09 | 18.20 | 37.23   |
| SMoP       | 86k   | 27.97     | 38.12         | 35.59    | 33.33    | 33.33          | 35.09 | 49.50 | 65.91 | 26.80 | 41.05   |
| ATTEMPT    | 90k   | 27.36     | 40.00         | 35.59    | 37.50    | 27.27          | 35.09 | 24.91 | 49.01 | 14.60 | 30.19   |
| PT-MoE     | 80k   | 35.63     | 55.62         | 55.93    | 79.16    | 36.36          | 56.77 | 77.74 | 71.83 | 42.60 | 56.91   |

Table 4: Accuracy (%) on mathematical problem-solving tasks with the number of trainable parameters shown in the second column. The first four out-of-domain datasets are from the SVAMP dataset. MP500 denotes the first 500 questions from MATH\_PROBLEMS.

matical tasks compared to their performance in QA. While LoRA underperformed PT by 5.36 points in MRQA, it outperforms PT by 10.31 points in mathematical tasks, indicating task-specific strengths of different PEFT approaches. (4) PT-MoE demonstrates unique cross-task consistency: while prompt tuning methods excel in QA tasks and LoRA-based methods in mathematical tasks, PT-MoE achieves the highest average performance in both domains, indicating robust adaptability across different problem types. (5) While PEFT methods consistently underperform full fine-tuning, the performance gap is larger in mathematical tasks compared to QA tasks, with a wider performance range among different methods. Notably, PT-MoE achieves comparable or higher performance to full fine-tuning on specific datasets such as Division and MP500. (6) PT-MoE demonstrates superior parameter efficiency, achieving higher performance than LoRA while using only 75% of its parameters (80k vs 106k), and outperforming HydraLoRA which uses 3.5 times more parameters. These findings highlight both the unique challenges of mathematical tasks and the robust adaptability of PT-MoE across different problem domains.

#### 5.3 Case Study

To better understand the performance characteristics of PT-MoE, we present a detailed case study of polynomial addition in Table [5.](#page-6-0) In this example, the response of the base model exhibits information loss, specifically omitting the linear term during simplification steps, leading to an incorrect final result. The conventional prompt tuning approach exhibits hallucinations and conceptual errors, particularly in degree identification and term combination, resulting in wrong terms like 2y 4 and −6y 3 . PT-MoE maintains information completeness throughout the solution process and avoids hallucinations, ultimately producing the correct polynomial expression. Notably, PT-MoE achieves this with a more

concise solution process, demonstrating efficient problem-solving steps while maintaining accuracy.

### 5.4 Ablation Studies

To comprehensively evaluate the design choices in PT-MoE, we conduct ablation studies on five influencing variables: soft prompt length, trainable parameters, number of experts, routing mechanisms, and model size. For each variable, we keep other variables fixed at their default values (soft prompt length=40, trainable parameters≈80K, number of experts=2, probationary-selective routing, 1B base model) while varying the target component to identify its influence on model performance.

Soft prompt length. We evaluate prompt lengths ranging from 20 to 80 tokens (Figure [3](#page-6-1) Left). Three consistent observations appear: (1) Indomain performance exceeds out-of-domain across all lengths, maintaining a 5-6% F1 score margin; (2) Both domains achieve optimal performance at 40 tokens, with peak F1 scores of 60.66% and 55.28% respectively; and (3) Performance in both domains follows a similar trend, improving up to 40 tokens then decreasing. These findings indicate that the optimal prompt length is domain-agnostic, though the absolute performance levels remain domain-dependent.

Number of experts. We investigate the influence of expert count by varying it from 1 to 8 (Figure [3](#page-6-1) Center left). There are three key points: (1) Single-expert configuration yields the poorest performance (58.90% and 52.64% F1 for in-domain and out-of-domain), demonstrating the necessity of MoE; (2) Performance exhibits an initial increase followed by decrease, with in-domain peaking at N=2 (60.66% F1) and out-of-domain at N=4 (55.84% F1), suggesting different optimal routing abilities for each domain; (3) In-domain tasks consistently outperform out-of-domain scenarios by a 4-6% F1 margin across all expert counts. These observations demonstrate that the optimal number

```
Question: Let f(y) = y
                         4 − 3y
                                 3 + y − 3 and g(y) = y
                                                           3 + 7y
                                                                    2 − 2. Find f(y) + g(y). Write your answer as a polynomial with terms of decreasing degree.
Standard Answer: y
                    4 − 2y
                            3 + 7y
                                    2 + y − 5
Original:
## Step 1: Identify the polynomials f(y) and g(y)
The polynomial f(y) is given as f(y) = y
                                         4 − 3y
                                                 3 +
y − 3 and the polynomial g(y) is given as g(y) =
y
 3 + 7y
         2 − 2.
## Step 2: Add the polynomials f(y) and g(y)
To find the sum of f(y) and g(y), we need to add the
corresponding terms of each polynomial.
## Step 3: Perform the addition
f(y) + g(y) = (y
                     4 − 3y
                             3 + y − 3) + (y
                                                 3 +
7y
  2 − 2)
## Step 4: Combine like terms
Combine the terms with the same degree to simplify the
expression.
## Step 5: Simplify the expression
f(y) + g(y) = y
                   4 − 3y
                           3 + y
                                  3 + 7y
                                          2 − 2−3
## Step 6: Combine like terms
Combine the terms with the same degree.
## Step 7: Simplify the expression
f(y) + g(y) = y
                   4 − 2y
                           3 + 7y
                                   2−2
## Step 8: Write the final answer
The final answer is: y
                    4 − 2y
                            3 + 7y
                                    2−2
                                                         Prompt Tuning:
                                                         ## Step 1: To find the sum of the two polynomials f(y)
                                                         and g(y), we need to add their corresponding terms.
                                                         ## Step 2: The polynomial f(y) = y
                                                                                             4 −3y
                                                                                                    3 +y −3
                                                         has terms of degree 4, 3, 2, and 1, while the polynomial
                                                         g(y) = y
                                                                    3 + 7y
                                                                            2 − 2 has terms of degree 3, 2, and
                                                         1.
                                                         ## Step 3: When we add these two polynomials, we need
                                                         to combine like terms, which means adding the coeffi-
                                                         cients of terms with the same degree.
                                                         ## Step 4: The terms with degree 4 are y
                                                                                                  4 and y
                                                                                                          4, so
                                                         their sum is 2y
                                                                        4.
                                                         ## Step 5: The terms with degree 3 are −3y
                                                                                                  3 and −3y
                                                                                                             3,
                                                         so their sum is −6y
                                                                             3.
                                                         ## Step 6: The terms with degree 2 are y and 7y
                                                                                                          2, so
                                                         their sum is 7y
                                                                        2.
                                                         ## Step 7: The terms with degree 1 are 1 and −2, so their
                                                         sum is −1.
                                                         ## Step 8: Therefore, the sum of the two polynomials
                                                         f(y) and g(y) is 2y
                                                                              4 − 6y
                                                                                      3 + 7y
                                                                                               2−1.
                                                         ## Step 9: The answer is 2y
                                                                                    4 − 6y
                                                                                            3 + 7y
                                                                                                    2−1.
                                                                                                                   PT-MoE:
                                                                                                                   ## Step 1: Identify the polynomials f(y) and g(y)
                                                                                                                   The given polynomials are f(y) = y
                                                                                                                                                       4 −3y
                                                                                                                                                              3 +y −3
                                                                                                                   and g(y) = y
                                                                                                                                  3 + 7y
                                                                                                                                          2 − 2.
                                                                                                                   ## Step 2: Add the polynomials f(y) and g(y)
                                                                                                                   To find f(y) + g(y), we need to add the corresponding
                                                                                                                   terms of the two polynomials.
                                                                                                                   ## Step 3: Perform the addition
                                                                                                                   f(y) + g(y) = (y
                                                                                                                                         4 − 3y
                                                                                                                                                 3 + y − 3) + (y
                                                                                                                                                                    3 +
                                                                                                                   7y
                                                                                                                      2 − 2)
                                                                                                                   ## Step 4: Combine like terms
                                                                                                                   Combine the terms with the same degree of y: y
                                                                                                                                                                    4 +
                                                                                                                   (y
                                                                                                                      3 − 3y
                                                                                                                              3) + (y
                                                                                                                                      3 + 7y
                                                                                                                                              2) + (y − 3) − 2
                                                                                                                   ## Step 5: Simplify the expression
                                                                                                                   Simplify the expression by combining the coefficients of
                                                                                                                   like terms: y
                                                                                                                               4 − 2y
                                                                                                                                       3 + 7y
                                                                                                                                               2 + y − 5
                                                                                                                   ## Step 6: Write the final expression
                                                                                                                   The expression f(y) + g(y) is y
                                                                                                                                                    4 − 2y
                                                                                                                                                            3 + 7y
                                                                                                                                                                    2 +
                                                                                                                   y − 5.
                                                                                                                   ## Note: y is the variable, and the answer is y
                                                                                                                                                             4 −2y
                                                                                                                                                                     3 +
                                                                                                                   7y
                                                                                                                      2 + y − 5.
```

Table 5: A case study comparing three solution approaches (Original, Prompt Tuning, and PT-MoE) for a polynomial addition problem. Errors in the outputs are highlighted in red (incorrect terms), orange (missing terms), and blue (hallucinated terms).

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 3: Ablation studies on key components of PT-MoE, showing the influence of (Left) prompt length, (Center left) number of experts, (Center right) trainable parameters, and (Right) routing mechanisms ((N)S: (Non-)Selective, (N)P: (Non-)Probationary) on in-domain (ID) and out-of-domain (OOD) performance.

of experts varies by domain type and highlight the importance of balancing expert focus with routing difficulty.

Trainable parameters. We vary the parameter count from 18K to 163K to analyze its influence on model performance (Figure [3](#page-6-1) Center Right). Three key observations appear: (1) Performance consistently improves with increasing parameters, from 57.51% to 61.04% F1 for in-domain and 53.96% to 55.38% F1 for out-of-domain tasks, and notably maintains stability without decrease even at higher parameter counts, differing from conventional prompt tuning methods; (2) While both indomain and out-of-domain tasks show increasing trend, they exhibit different parameter dependence behaviours, in-domain tasks demonstrate rapid improvement before 80K parameters, while out-ofdomain tasks show accelerated growth in the 40K- 80K range; (3) In-domain performance maintains a consistent advantage over out-of-domain tasks across all parameter ranges, with F1 scores differing by approximately 4-6%. These findings show that PT-MoE effectively leverages additional parameters to achieve continuous performance gains.

Routing mechanisms. We examine two key routing design choices (Figure [3](#page-6-1) Right): selective routing, which uses only the highest-weighted expert versus non-selective routing that utilizes all experts with their respective weights, and probationary routing, which multiplies the output by the router's selection probability versus nonprobationary routing that uses original outputs. Our results show four key findings: (1) The combination of selective and probationary routing (S, P) consistently outperforms other configurations (NS, P and S, NP) across both in-domain (60.66% vs

<span id="page-7-1"></span>

|         | PT    | SMoP  | PT-MoE |
|---------|-------|-------|--------|
| GSM8K   | 56.70 | 61.78 | 59.74  |
| SVAMP   | 69.36 | 74.69 | 72.81  |
| ASDIV   | 76.41 | 80.06 | 81.39  |
| MAWPS   | 70.70 | 70.70 | 78.02  |
| MP500   | 59.00 | 60.80 | 63.60  |
| Average | 66.43 | 69.61 | 71.11  |

Table 6: Performance comparison (accuracy %) of standard and MoE-based prompt tuning methods on mathematical problem solving tasks using a 3B base model.

59.24% and 58.78% F1) and out-of-domain tasks (55.28% vs 53.41% and 52.64% F1), suggesting the complementary benefits of focused expert utilization and confidence-based output; (2) Probationary routing demonstrates superior performance over its non-probationary counterpart, indicating the value of incorporating router confidence in the final output; (3) Under probationary conditions, selective routing achieves 1.42% higher F1 score while reducing utilized parameters compared to non-selective routing, highlighting the effectiveness and efficiency of domain-specific knowledge; (4) All routing configurations maintain higher performance on in-domain tasks compared to out-ofdomain scenarios, though the relative performance rankings remain consistent across domains. These findings collectively demonstrate that the selective probationary routing mechanism achieves an optimal balance between model performance and computational efficiency.

Model size. We conduct additional studies using a 3B version of the base model, comparing PT-MoE with PT and the MoE-integrated method, SMoP (Table [6\)](#page-7-1). Three key findings are found: (1) PT-MoE maintains its advantage at larger sizes, achieving the highest average accuracy of 71.11%, outperforming standard PT (66.43%) and SMoP (69.61%). (2) SMoP shows size-dependent behaviour: while underperforming PT on the 1B model (56.77% vs 56.25%), it outperforms PT on the 3B model (69.61% vs 66.43%). (3) PT-MoE demonstrates robust performance by outperforming the baselines on three out of five mathematical datasets. These findings collectively validate the size-independence and stability of PT-MoE across different model sizes.

## 5.5 Efficiency Analysis

Results in Figure [4](#page-7-2) demonstrate PT-MoE's efficiency across both computational and parametric aspects. PT-MoE achieves the highest performance with only moderate training steps and

<span id="page-7-2"></span>![](_page_7_Figure_6.jpeg)

Figure 4: Parameter and training efficiency comparison across different methods. The x-axis shows training steps for the highest performance after training parameter search, while the y-axis shows the average accuracy on math datasets. Circle sizes indicate the number of trainable parameters, with larger circles indicating more parameters.

minimal parameters (80k). In contrast, LoRA and HydraLoRA require more parameters and training steps to achieve comparable performance. Other prompt tuning methods such as PT, SMoP, and DPT converge fast but achieve lower performance, suggesting a potential trade-off between training efficiency and model effectiveness. These results validate that PT-MoE balances the computational cost, parameter efficiency, and model performance.

## <span id="page-7-0"></span>6 Conclusions

This work introduces PT-MoE, a novel parameterefficient framework that integrates matrix decomposition with MoE routing for prompt tuning. Our results across 17 datasets demonstrate that PT-MoE achieves state-of-the-art performance while maintaining parameter efficiency, outperforming existing methods in both QA and mathematical tasks. Through ablation studies, we identify optimal configurations for prompt length, expert count, and routing mechanisms, providing insights for future parameter-efficient tuning approaches.

Future directions include exploring hierarchical routing mechanisms to better deal with diverse task distributions, and extending PT-MoE to continual learning scenarios for efficient adaptation and knowledge transfer across tasks.

## Limitations

While PT-MoE shows promising results, there are some key points that need to be noted. Like other fine-tuning methods, people should be careful of training data licensing and usage rights. Furthermore, while our results demonstrate strong performance across benchmark tasks, developing comprehensive insights for diverse applications would benefit from broader community contributions and open-source collaboration.

## Ethics Statement

No ethical approval was required for this study.

## Availability Statement

The codes and models related to this paper are uploaded to the open-source community at https://github.com/ZongqianLi/PT-MoE.

## References

- <span id="page-8-3"></span>Akari Asai, Mohammadreza Salehi, Matthew Peters, and Hannaneh Hajishirzi. 2022. [ATTEMPT:](https://doi.org/10.18653/v1/2022.emnlp-main.446) [Parameter-efficient multi-task tuning via attentional](https://doi.org/10.18653/v1/2022.emnlp-main.446) [mixtures of soft prompts.](https://doi.org/10.18653/v1/2022.emnlp-main.446) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 6655–6672, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
- <span id="page-8-2"></span>Joon-Young Choi, Junho Kim, Jun-Hyung Park, Wing-Lam Mok, and SangKeun Lee. 2023. [SMop: To](https://openreview.net/forum?id=5x5Vxclc1K)[wards efficient and effective prompt tuning with](https://openreview.net/forum?id=5x5Vxclc1K) [sparse mixture-of-prompts.](https://openreview.net/forum?id=5x5Vxclc1K) In *The 2023 Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-8-10"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training verifiers to solve math word prob](https://arxiv.org/abs/2110.14168)[lems.](https://arxiv.org/abs/2110.14168) *Preprint*, arXiv:2110.14168.
- <span id="page-8-7"></span>Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. [DROP: A reading comprehension benchmark requir](https://doi.org/10.18653/v1/N19-1246)[ing discrete reasoning over paragraphs.](https://doi.org/10.18653/v1/N19-1246) In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics.
- <span id="page-8-5"></span>Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. 2017. [Searchqa: A new q&a dataset augmented](https://arxiv.org/abs/1704.05179)

- [with context from a search engine.](https://arxiv.org/abs/1704.05179) *Preprint*, arXiv:1704.05179.
- <span id="page-8-12"></span>Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. 2019. MRQA 2019 shared task: Evaluating generalization in reading comprehension. In *Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP*.
- <span id="page-8-13"></span>Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, ..., and Zhiyu Ma. 2024. [The llama 3 herd of models.](https://arxiv.org/abs/2407.21783) *Preprint*, arXiv:2407.21783.
- <span id="page-8-0"></span>Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. 2024. [Parameter-efficient fine](https://openreview.net/forum?id=lIsCS8b6zj)[tuning for large models: A comprehensive survey.](https://openreview.net/forum?id=lIsCS8b6zj) *Transactions on Machine Learning Research*.
- <span id="page-8-4"></span>Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. [TriviaQA: A large scale distantly](https://doi.org/10.18653/v1/P17-1147) [supervised challenge dataset for reading comprehen](https://doi.org/10.18653/v1/P17-1147)[sion.](https://doi.org/10.18653/v1/P17-1147) In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.
- <span id="page-8-9"></span>Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. [Are you smarter than a sixth grader?](https://doi.org/10.1109/CVPR.2017.571) [textbook question answering for multimodal machine](https://doi.org/10.1109/CVPR.2017.571) [comprehension.](https://doi.org/10.1109/CVPR.2017.571) In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5376–5384.
- <span id="page-8-11"></span>Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. [MAWPS:](https://doi.org/10.18653/v1/N16-1136) [A math word problem repository.](https://doi.org/10.18653/v1/N16-1136) In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1152–1157, San Diego, California. Association for Computational Linguistics.
- <span id="page-8-6"></span>Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. [Natu](https://doi.org/10.1162/tacl_a_00276)[ral questions: A benchmark for question answering](https://doi.org/10.1162/tacl_a_00276) [research.](https://doi.org/10.1162/tacl_a_00276) *Transactions of the Association for Computational Linguistics*, 7:452–466.
- <span id="page-8-8"></span>Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. [RACE: Large-scale ReAd](https://doi.org/10.18653/v1/D17-1082)[ing comprehension dataset from examinations.](https://doi.org/10.18653/v1/D17-1082) In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 785– 794, Copenhagen, Denmark. Association for Computational Linguistics.
- <span id="page-8-1"></span>Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. [The power of scale for parameter-efficient prompt](https://doi.org/10.18653/v1/2021.emnlp-main.243) [tuning.](https://doi.org/10.18653/v1/2021.emnlp-main.243) In *Proceedings of the 2021 Conference on*

- *Empirical Methods in Natural Language Processing*, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- <span id="page-9-15"></span>Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017. [Zero-shot relation extraction via](https://doi.org/10.18653/v1/K17-1034) [reading comprehension.](https://doi.org/10.18653/v1/K17-1034) In *Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)*, pages 333–342, Vancouver, Canada. Association for Computational Linguistics.
- <span id="page-9-8"></span>Xiang Lisa Li and Percy Liang. 2021. [Prefix-tuning:](https://doi.org/10.18653/v1/2021.acl-long.353) [Optimizing continuous prompts for generation.](https://doi.org/10.18653/v1/2021.acl-long.353) In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 4582– 4597, Online. Association for Computational Linguistics.
- <span id="page-9-0"></span>Zongqian Li and Jacqueline M Cole. 2025. Autogenerating question-answering datasets with domainspecific knowledge for language models in scientific tasks. *Digital Discovery*, 4(4):998–1005.
- <span id="page-9-2"></span>Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Collier. 2025a. [Prompt compression for large language](https://aclanthology.org/2025.naacl-long.368/) [models: A survey.](https://aclanthology.org/2025.naacl-long.368/) In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 7182–7195, Albuquerque, New Mexico. Association for Computational Linguistics.
- <span id="page-9-3"></span>Zongqian Li, Ehsan Shareghi, and Nigel Collier. 2025b. [Reasongraph: Visualisation of reasoning paths.](https://arxiv.org/abs/2503.03979) *Preprint*, arXiv:2503.03979.
- <span id="page-9-1"></span>Zongqian Li, Yixuan Su, and Nigel Collier. 2024. [500xcompressor: Generalized prompt compres](https://arxiv.org/abs/2408.03094)[sion for large language models.](https://arxiv.org/abs/2408.03094) *Preprint*, arXiv:2408.03094.
- <span id="page-9-5"></span>Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022. [P-tuning:](https://doi.org/10.18653/v1/2022.acl-short.8) [Prompt tuning can be comparable to fine-tuning](https://doi.org/10.18653/v1/2022.acl-short.8) [across scales and tasks.](https://doi.org/10.18653/v1/2022.acl-short.8) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 61–68, Dublin, Ireland. Association for Computational Linguistics.
- <span id="page-9-6"></span>Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. 2023. [Gpt](https://arxiv.org/abs/2103.10385) [understands, too.](https://arxiv.org/abs/2103.10385) *Preprint*, arXiv:2103.10385.
- <span id="page-9-4"></span>Fang Ma, Chen Zhang, Lei Ren, Jingang Wang, Qifan Wang, Wei Wu, Xiaojun Quan, and Dawei Song. 2022. [XPrompt: Exploring the extreme of prompt](https://doi.org/10.18653/v1/2022.emnlp-main.758) [tuning.](https://doi.org/10.18653/v1/2022.emnlp-main.758) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 11033–11047, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

- <span id="page-9-17"></span>Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. [A diverse corpus for evaluating and developing](https://doi.org/10.18653/v1/2020.acl-main.92) [English math word problem solvers.](https://doi.org/10.18653/v1/2020.acl-main.92) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 975–984, Online. Association for Computational Linguistics.
- <span id="page-9-18"></span>Nebrelbug. 2024. [Math problems.](https://huggingface.co/datasets/nebrelbug/math-problems/tree/main) Hugging Face Hub.
- <span id="page-9-13"></span>Ioannis Partalas, Eric Gaussier, Axel-Cyrille Ngonga Ngomo, et al. 2013. Results of the first bioasq workshop. In *BioASQ@ CLEF*, pages 1–8.
- <span id="page-9-16"></span>Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. [Are NLP models really able to solve simple](https://doi.org/10.18653/v1/2021.naacl-main.168) [math word problems?](https://doi.org/10.18653/v1/2021.naacl-main.168) In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2080–2094, Online. Association for Computational Linguistics.
- <span id="page-9-12"></span>Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. [SQuAD: 100,000+ questions for](https://doi.org/10.18653/v1/D16-1264) [machine comprehension of text.](https://doi.org/10.18653/v1/D16-1264) In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.
- <span id="page-9-7"></span>Anastasiia Razdaibiedina, Yuning Mao, Madian Khabsa, Mike Lewis, Rui Hou, Jimmy Ba, and Amjad Almahairi. 2023. [Residual prompt tuning: improving](https://doi.org/10.18653/v1/2023.findings-acl.421) [prompt tuning with residual reparameterization.](https://doi.org/10.18653/v1/2023.findings-acl.421) In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 6740–6757, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-9-14"></span>Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra, and Karthik Sankaranarayanan. 2018. [DuoRC: Towards](https://doi.org/10.18653/v1/P18-1156) [complex language understanding with paraphrased](https://doi.org/10.18653/v1/P18-1156) [reading comprehension.](https://doi.org/10.18653/v1/P18-1156) In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1683– 1693, Melbourne, Australia. Association for Computational Linguistics.
- <span id="page-9-9"></span>Zhengxiang Shi and Aldo Lipani. 2024. [DePT: De](https://openreview.net/forum?id=KjegfPGRde)[composed prompt tuning for parameter-efficient fine](https://openreview.net/forum?id=KjegfPGRde)[tuning.](https://openreview.net/forum?id=KjegfPGRde) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-10"></span>Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou', and Daniel Cer. 2022. [SPoT: Better frozen model](https://doi.org/10.18653/v1/2022.acl-long.346) [adaptation through soft prompt transfer.](https://doi.org/10.18653/v1/2022.acl-long.346) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5039–5059, Dublin, Ireland. Association for Computational Linguistics.
- <span id="page-9-11"></span>Chengyu Wang, Jianing Wang, Minghui Qiu, Jun Huang, and Ming Gao. 2021. [TransPrompt: Towards](https://doi.org/10.18653/v1/2021.emnlp-main.221) [an automatic transferable prompting framework for](https://doi.org/10.18653/v1/2021.emnlp-main.221) [few-shot text classification.](https://doi.org/10.18653/v1/2021.emnlp-main.221) In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 2792–2802, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

<span id="page-10-2"></span>Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, and Yoon Kim. 2023. [Multitask](https://openreview.net/forum?id=Nk2pDtuhTq) [prompt tuning enables parameter-efficient transfer](https://openreview.net/forum?id=Nk2pDtuhTq) [learning.](https://openreview.net/forum?id=Nk2pDtuhTq) In *The Eleventh International Conference on Learning Representations*.

<span id="page-10-0"></span>Yao Xiao, Lu Xu, Jiaxi Li, Wei Lu, and Xiaoli Li. 2023. [Decomposed prompt tuning via low-rank reparame](https://doi.org/10.18653/v1/2023.findings-emnlp.890)[terization.](https://doi.org/10.18653/v1/2023.findings-emnlp.890) In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 13335– 13347, Singapore. Association for Computational Linguistics.

<span id="page-10-1"></span>Yige Xu, Zhiwei Zeng, and Zhiqi Shen. 2023. [Efficient](https://doi.org/10.18653/v1/2023.findings-emnlp.780) [cross-task prompt tuning for few-shot conversational](https://doi.org/10.18653/v1/2023.findings-emnlp.780) [emotion recognition.](https://doi.org/10.18653/v1/2023.findings-emnlp.780) In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 11654–11666, Singapore. Association for Computational Linguistics.

<span id="page-10-3"></span>Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. [HotpotQA: A dataset for](https://doi.org/10.18653/v1/D18-1259) [diverse, explainable multi-hop question answering.](https://doi.org/10.18653/v1/D18-1259) In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

<span id="page-10-4"></span>Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. [Metamath:](https://openreview.net/forum?id=N8N0hgNDRt) [Bootstrap your own mathematical questions for large](https://openreview.net/forum?id=N8N0hgNDRt) [language models.](https://openreview.net/forum?id=N8N0hgNDRt) In *The Twelfth International Conference on Learning Representations*.

## A Appendix

### A.1 Implementation Details

We provide implementation details, including training hyperparameters in Table [8,](#page-11-1) inference parameters in Table [7,](#page-10-5) and method-specific configurations in Tables [9](#page-11-0) to facilitate reproducibility. The models were finetuned on one node with four A100 80G.

<span id="page-10-5"></span>

|                | QA           | Math         |
|----------------|--------------|--------------|
| max_new_tokens | 100          | 768          |
| num_beams      | 1            | 1            |
| do_sample      | False        | False        |
| temperature    | 1.0          | 1.0          |
| top_p          | 1.0          | 1.0          |
| pad_token_id   | pad_token_id | pad_token_id |
| eos_token_id   | eos_token_id | eos_token_id |
| early_stopping | True         | True         |
| length_penalty | 1.0          | 1.0          |

Table 7: Inference parameters for QA and mathematical tasks.

Prompt for MRQA:

<| start\_header\_id | > user <| end\_header\_id | >\ n \ nExtract the exact text span from the given context that directly answers the question , without modifying or combining multiple parts of the text .\ n \ nContext : {}\ n \ nQuestion : {} <| eot\_id | > <| start\_header\_id | > assistant <| end\_header\_id | >\ n \ nAnswer :

#### Prompt for Math datasets:

```
<| start_header_id | > user <| end_header_id | >\
n \ nSolve the question and your response
should end with \" The answer is : [ answer
]\".\ n \ nQuestion : {} <| eot_id | > <|
start_header_id | > assistant <| end_header_id
| >\ n \ nAnswer :
```

Texts used to initialize soft prompt for finetuning on MRQA:

```
(
    " Read the following context and
    answer the question . "
    " Extract the answer from the context .
     "
    " The answer is a span of the context
    ."
    " Answer the question directly ."
    " Use the original words in the
    context ."
    " Do not introduce any words not
    present in the context ."
)
```

Texts used to initialize soft prompt for finetuning on Math datasets:

```
" Read the question carefully and make
 sure you understand it before
beginning . "
" Pay close attention to the details
and requirements of the question . "
" Answer the question , ensuring your
response is relevant to what is asked
. "
" Ensure your answer is both accurate
and correct ."
```

## A.2 Environment

(

)

```
python==3.11.5
torch==2.3.1+cu118
transformers==4.46.0
datasets==2.18.0
huggingface_hub==0.24.2
deepspeed==0.15.3
wandb==0.14.2
numpy==1.23.5
tqdm==4.66.4
```

<span id="page-11-1"></span>

|                             | QA                                      | Math                 |
|-----------------------------|-----------------------------------------|----------------------|
| Train steps                 | {500, 1000, 1500} for PT-based methods  | {500, 1000, 1500}    |
|                             | {200, 600, 1000} for LoRA-based methods |                      |
| Optimizer                   | AdamW                                   | AdamW                |
| Max length                  | 512                                     | 768                  |
| warmup_steps                | 500                                     | 500                  |
| learning_rate               | 2e-5                                    | 2e-5                 |
| per_device_train_batch_size | 32                                      | 16                   |
| lr_scheduler_type           | constant_with_warmup                    | constant_with_warmup |
| gradient_accumulation_steps | 2                                       | 2                    |

Table 8: Training hyperparameters for QA and mathematical tasks. {} means parameter search.

<span id="page-11-0"></span>

| Method    | Details                                                                                      |
|-----------|----------------------------------------------------------------------------------------------|
| LoRA      | r=1; lora_alpha=16; target_modules=["q_proj", "v_proj"]; lora_dropout=0; bias="none";        |
|           | task_type=TaskType.CAUSAL_LM                                                                 |
| HydraLoRA | r=1; alpha=16; target_modules=["q_proj", "v_proj"]; dropout=0.0; num_b_matrices=2;           |
|           | Router:<br>nn.Sequential(nn.Linear(input_dim,<br>num_b_matrices));<br>Initialization:<br>A:  |
|           | nn.init.kaiming_uniform_(, a=math.sqrt(5)), B: nn.init.zeros_()                              |
| PT        | Soft prompt length: 40; Initialization: Specific words                                       |
| DPT       | Soft prompt length: 40; low_rank_dim = 39; Initialization: Specific words; Decomposition     |
|           | method: SVD                                                                                  |
| SMoP      | Total soft prompt length: 40; Number of experts: 2; Initialization: Specific words; Noise:   |
|           | *(1+torch.randn_like()*0.01)                                                                 |
| ATTEMPT   | Total<br>soft<br>prompt<br>length:<br>40;<br>Number<br>of<br>experts:<br>2;<br>Encoder:      |
|           | nn.Linear(embedding_dim,<br>projection_dim=1),<br>nn.Linear(projection_dim=1,<br>em          |
|           | bedding_dim), nn.LayerNorm(embedding_dim); Initialization: Specific words                    |
| PT-MoE    | Soft<br>prompt<br>length:<br>40;<br>Number<br>of<br>expert:<br>2;<br>Rank:<br>36;<br>Router: |
|           | nn.Linear(embedding_dim, num_prompts); Noise: *(1+torch.randn_like()*0.01)                   |

Table 9: Method configurations for various PEFT methods.