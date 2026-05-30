# MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

Lin Ning\*, Harsh Lara, Meiqi Guo, Abhinav Rastogi\*†

Google DeepMind {linning, harshlara, mqguo, abhirast}@google.com

## Abstract

Parameter-efficient fine-tuning techniques like Low-Rank Adaptation (LoRA) have revolutionized the adaptation of large language models (LLMs) to diverse tasks. Recent efforts have explored mixtures of LoRA modules for multitask settings. However, our analysis reveals redundancy in the down-projection matrices of these architectures. This observation motivates our proposed method, Mixture of Dyadic Experts (MoDE), which introduces a novel design for efficient multi-task adaptation. This is done by sharing the down-projection matrix across tasks and employing atomic rank-one adapters, coupled with routers that allow more sophisticated task-level specialization. Our design allows for more fine-grained mixing, thereby increasing the model's ability to jointly handle multiple tasks. We evaluate MoDE on the Supernatural Instructions (SNI) benchmark consisting of a diverse set of 700+ tasks and demonstrate that it outperforms state-of-theart multi-task parameter-efficient fine-tuning (PEFT) methods, without introducing additional parameters. Our findings contribute to a deeper understanding of parameter efficiency in multi-task LLM adaptation and provide a practical solution for deploying high-performing, lightweight models.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, from text generation and translation to question-answering and summarization [\(Brown et al.,](#page-9-0) [2020;](#page-9-0) [Team et al.,](#page-10-0) [2023,](#page-10-0) [2024;](#page-10-1) [Ope](#page-10-2)[nAI and et al.,](#page-10-2) [2024\)](#page-10-2). The ability to perform well on a diverse set of tasks is essential for deploying LLMs in real-world applications, where they may need to handle a wide array of user requests and instructions. However, effectively adapting these

<span id="page-0-0"></span>![](_page_0_Picture_10.jpeg)

Figure 1: Mixture of Dyadic Experts with 3 experts and a rank of 4, with each slice corresponding to a rank dimension. Our architecture allows independent routing at each rank. When number of mixtures is 1, our architecture is equivalent to traditional LoRA.

large models to multiple tasks presents significant challenges. Fine-tuning a separate model for each task is computationally expensive and requires vast amount of storage due to the large model sizes. Moreover, independently trained models hinder knowledge transfer between tasks, potentially limiting the model's performance and its ability to generalize to unseen tasks.

Multi-task learning (MTL) [\(Caruana,](#page-9-1) [1997;](#page-9-1) [Ruder,](#page-10-3) [2017\)](#page-10-3) offers a promising solution to these challenges. By training a single model on multiple tasks simultaneously, MTL aims to improve parameter efficiency, enhance generalization, and potentially boost performance on individual tasks through knowledge transfer. Parameter-efficient fine-tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA) [\(Hu et al.,](#page-9-2) [2021\)](#page-9-2), have further enhanced efficiency by introducing only a small number of trainable parameters. LoRA efficiently represents weight changes during fine-tuning us-

<sup>\*</sup>Equal contribution.

<sup>†</sup>Corresponding author.

ing two low-rank projection matrices: one downprojects input features to a smaller size, and another up-projects the resulting low-dimensional representation back to the original output size.

Mixture-of-Experts (MoE) architectures [\(Sukhbaatar et al.,](#page-10-4) [2024;](#page-10-4) [Li et al.,](#page-9-3) [2022;](#page-9-3) [Jiang](#page-9-4) [et al.,](#page-9-4) [2024;](#page-9-4) [Fedus et al.,](#page-9-5) [2022\)](#page-9-5) have emerged as a powerful approach to scale model capacity and expertise, enabling LLMs to handle a wider range of tasks. Recent studies [\(Feng et al.,](#page-9-6) [2024;](#page-9-6) [Zhu et al.,](#page-10-5) [2023;](#page-10-5) [Zadouri et al.,](#page-10-6) [2023;](#page-10-6) [Liu et al.,](#page-9-7) [2023;](#page-9-7) [Li et al.,](#page-9-8) [2024\)](#page-9-8) have explored integrating LoRA with Mixture-of-Experts architectures (LoRA-MoE) to extend LLMs' capabilities to multi-task adaptation. However, our analysis reveals redundancy in down-projection matrices of these architectures. This redundancy leads to an inefficient utilization of parameters, potentially limiting the effectiveness in capturing the unique characteristics of each task.

In this work, we propose Mixture of Dyadic Experts (MoDE) (Figure [1\)](#page-0-0), a novel parameterefficient framework for multi-task adaptation. MoDE leverages a single, shared down-projection matrix across all experts to reduce parameter redundancy. Furthermore, MoDE introduces atomic rank-one adapters, enabling fine-grained task specialization and knowledge sharing. Crucially, MoDE incorporates a sophisticated routing mechanism that allows for more nuanced and flexible combinations of these rank-one adapters, further enhancing the model's expressive power while maintaining parameter efficiency.

We rigorously evaluate MoDE on the multi-task Supernatural Instructions benchmark [\(Wang et al.,](#page-10-7) [2022b\)](#page-10-7). Our results demonstrate that MoDE consistently outperforms state-of-the-art multi-task PEFT methods, including those based on LoRA-MoE, while utilizing comparable number of additional parameters. This underscores MoDE's efficacy in achieving both strong performance and parameter efficiency, making it a promising approach for deploying multi-task LLMs in real-world applications.

## Our key contributions are as follows:

- We identify and address the redundancy in down-projection matrices in existing LoRAbased MoE architectures.
- We propose MoDE, a novel architecture leveraging a shared down-projection matrix and atomic rank-one adapters, coupled with a so-

- phisticated routing mechanism for efficient and expressive performance.
- We demonstrate the superior performance of MoDE compared to state-of-the-art multi-task LoRA-based MoE methods on the Supernatural Instructions benchmark, while maintaining parameter efficiency.

## 2 Related Work

## 2.1 Parameter-efficient Fine-tuning (PEFT)

Parameter-efficient fine-tuning (PEFT) methods have emerged as a popular approach to adapt LLMs to downstream tasks without the high computational cost of full fine-tuning. LoRA (Low-Rank Adaptation) [\(Hu et al.,](#page-9-2) [2021\)](#page-9-2) is a particularly successful PEFT technique, achieving strong performance with a fraction of the trainable parameters.

## 2.2 Mixture-of-Experts and LoRA

Recent research has explored combining Mixtureof-Experts (MoE) architectures with LoRA to further improve efficiency and effectiveness, especially in multi-task scenarios. One notable approach is to propose frameworks that leverage domain-specific LoRA modules and explicit routing strategies to adapt to diverse tasks, such as Mixture-of-LoRAs (MoA) [\(Feng et al.,](#page-9-6) [2024\)](#page-9-6) and MOELoRA [\(Liu et al.,](#page-9-7) [2023\)](#page-9-7). Another group of work, represented by SiRA [\(Zhu et al.,](#page-10-5) [2023\)](#page-10-5) and MixLoRA [\(Li et al.,](#page-9-8) [2024\)](#page-9-8), introduce sparse MoE mechanisms with specialized routing and/or loadbalancing techniques to enhance efficiency while maintaining performance. MoLORA [\(Zadouri](#page-10-6) [et al.,](#page-10-6) [2023\)](#page-10-6) combines MoE with LoRA experts to achieve extreme parameter efficiency in instruction tuning. Their approach focuses on updating only a small fraction of the model's parameters, demonstrating comparable performance to full fine-tuning with significantly fewer resources. AdaMix [\(Wang](#page-10-8) [et al.,](#page-10-8) [2022a\)](#page-10-8) proposes a general PEFT method that tunes a mixture of adaptation modules within each Transformer layer to capture multiple views of a single task, using LoRA module sharing to enhance performance when labeled data is scarce. Their experiment show that sharing project-up has better performance. In contrast, our multi-task approach leverages a mixture of LoRA modules with shared project-down matrices, motivated by the observation of their similarity across different tasks.

#### 2.3 Multi-task PEFT

Other work has focused on extending PEFT to multi-task settings, where a single model needs to adapt to diverse tasks. LoraHub (Huang et al., 2023) investigates LoRA composability for crosstask generalization and introduces a framework for dynamically assembling LoRA modules trained on different tasks to adapt to unseen tasks. ZipLoRA (Shah et al., 2023) tackles the problem of combining independently trained style and subject LoRAs to achieve joint generation in a controllable manner. FLix (Lin et al., 2024) focuses on multi-task multilingual model adaptation, associating each dataset feature with its own low-rank weight update parameters for improved generalization across diverse datasets. MoLE (Wu et al., 2024) implements a hierarchical weight control approach with learnable gating functions to determine the optimal composition of trained LoRA layers, treating each layer as a distinct expert. Different from composing over all LoRA adaptors, Ostapenko et al. (2024) explores building a library of trained LoRA adapters and using a zero-shot routing mechanism (Arrow) to dynamically select relevant adapters for new tasks.

#### 3 Method

The Mixture of Dyadic Experts (MoDE) architecture presents a novel approach for multi-task learning, building upon and extending the traditional Low-Rank Adaptation (LoRA) and mixture-of-experts (MoE) design.

### <span id="page-2-1"></span>3.1 Background

**Low-Rank Adaption (LoRA)** LoRA (Hu et al., 2021) efficiently adapts LLMs to downstream tasks (Shah et al., 2023) by freezing pre-trained model weights and injecting trainable rank decomposition matrices into each layer. Given a feed-forward layer with input  $\mathbf{x} \in \mathbb{R}^{1 \times P}$  and weight matrix  $\mathbf{W_0} \in \mathbb{R}^{P \times Q}$ , LoRA introduces a down-projection matrix  $\mathbf{A} \in \mathbb{R}^{P \times r}$  and an up-projection matrix  $\mathbf{B} \in \mathbb{R}^{Q \times r}$  (Figure 2). The output of the layer is

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \mathbf{x}\mathbf{A}\mathbf{B}^T.$$

During training, only **A** and **B** are updated.

**Dyadic Product Representation** A dyadic product (or outer product) is a matrix multiplication between two vectors. Given vectors  $\mathbf{u} \in \mathbb{R}^{p \times 1}$  and  $\mathbf{v} \in \mathbb{R}^{q \times 1}$ , their dyadic product  $\mathbf{u} \otimes \mathbf{v}$  is a matrix of size  $p \times q$ . The LoRA update  $\Delta \mathbf{W} = \mathbf{A} \mathbf{B}^T$ 

<span id="page-2-0"></span>![](_page_2_Picture_9.jpeg)

Figure 2: Illustration of a basic LoRA module.

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Figure 3: Scatter plots showing the three most prominent principal components of all constituent vectors in the LoRA projection matrices for 15 independently trained single task model with shared initialization. Plots q\_6\_down and q\_12\_down (q\_6\_up and q\_12\_up) illustrate the down (up) projections of query matrices at layers 6 and 12, respectively. The clear clustering of down-projection vectors suggests that the down projection matrices are task-agnostic, motivating the design of the MoDE architecture.

can be expressed as a sum of dyadic products by decomposing **A** and **B** into their column vectors (Liu et al., 2024):

$$\Delta \mathbf{W} = [\mathbf{a}_1, \mathbf{a}_2, ..., \mathbf{a}_r] * [\mathbf{b}_1, \mathbf{b}_2, ..., \mathbf{b}_r]^T$$
$$= \sum_{i=1}^r (\mathbf{a}_i \otimes \mathbf{b}_i)$$

where  $\mathbf{a}_i$  and  $\mathbf{b}_i$  are column vectors of  $\mathbf{A}$  and  $\mathbf{B}$ , respectively. This can be plugged into the output equation to get:

$$\mathbf{y} = \mathbf{x} \mathbf{W_0} + \mathbf{x} \sum_{i=1}^r (\mathbf{a}_i \otimes \mathbf{b}_i)$$

**LoRA-MoE** Mixture-of-Experts (MoE) utilizes a combination of sub-models (experts), each spe-

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 4: Illustration of (a) traditional LoRA Mixture-of-Experts, (b) traditional LoRA Mixture-of-Experts with shared down-projection matrix.

cializing in different aspects of the underlying tasks, along with a gating mechanism to dynamically route inputs to the most suitable experts [\(Shazeer](#page-10-12) [et al.,](#page-10-12) [2017\)](#page-10-12). Figure [4\)](#page-3-0)(a) illustrates a traditional LoRA-MoE approach [\(Zadouri et al.,](#page-10-6) [2023\)](#page-10-6), where m LoRA experts (E<sup>i</sup> = AiBiT , i ∈ {1, ..., m}) are added to each layer. A router R, parameterized by W<sup>R</sup> ∈ R <sup>P</sup> <sup>×</sup>m, determines which expert to use, yielding the output:

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^{m} \mathcal{R}^i(\mathbf{x})(\mathbf{x}\mathbf{A}^i\mathbf{B}^{i^T})$$

where R<sup>i</sup> (x) is the routing probability for E<sup>i</sup> .

## <span id="page-3-1"></span>3.2 Motivating Observation

To motivate our proposed MoDE architecture, we first present an empirical analysis of projection matrices of LoRA modules that are independently trained on a set of tasks from the same initialization (B<sup>i</sup> to 0 and A<sup>i</sup> from a random normal distribution with a standard deviation of 0.01 and mean 0). We selected 15 diverse tasks from the Supernatural Instructions benchmark (see Section [4.1](#page-5-0) for details) and trained 15 LoRA modules for this study. We visualized the learned LoRA parameters using Principal Component Analysis (PCA), focusing on the distribution of vectors obtained by slicing the up-projection and down-projection matrices along their rank dimension, i.e., the vectors featuring in the dyadic product representation in Section [3.1.](#page-2-1)

Figure [3](#page-2-2) shows the resulting scatter plots. Notably, we observe that down-projection matrix vectors from different LoRA modules tend to cluster into distinct groups, with vectors corresponding to

the same position along the rank dimension forming tight clusters. In contrast, up-projection matrices exhibit no such clustering. This suggests down-projection matrices are task-agnostic, while up-projection matrices are more task-specific.

This empirical finding motivates the MoDE architecture, which leverages a shared downprojection matrix to reduce parameter redundancy. We further improve this design by leveraging the dyadic formulation to introduce a more sophisticated routing strategy which enables a more finegrained task-specific adaptation. Subsequent sections will detail MoDE's architecture and demonstrate its effectiveness in achieving both parameter efficiency and strong multi-task performance.

### 3.3 Mixture of Dyadic Experts (MoDE)

Inspired by the observations in Section [3.2,](#page-3-1) we introduce Mixture of Dyadic Experts (MoDE), a novel framework for efficient multi-task adaptation that incorporates two key innovations: (i) shared down-projection matrices for more efficient parameter utilization, and (ii) a sophisticated routing strategy to promote better task-level specialization.

### 3.3.1 Shared Down-Projection Matrix

Before introducing MoDE, we share a simple modification of the traditional LoRA-MoE by having all experts share a single down-projection matrix A (Figure [4\(](#page-3-0)b)), which we refer to as LoRA-MoE-SD. The output of a layer with LoRA-MoE-SD is:

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^{m} \mathcal{R}^i(\mathbf{x})(\mathbf{x}\mathbf{A}\mathbf{B}^{iT})$$

where **A** is the shared down-projection matrix, and  $\mathbf{B}^{i}$  is the up-projection matrix for expert  $E_{i}$ .

**Parameter Efficiency** By sharing a single down-projection matrix **A** across all experts, LoRA-MoE-SD reduces the number of trainable parameters for these matrices from  $m \cdot P \cdot r$  to  $P \cdot r$ .

### 3.3.2 Fine-Grained Routing

LoRA-MoE-SD, albeit with smaller parameters, offers only m choices for up projection. This is because the router inherently introduces a constraint that all the r dimensions of each expert must route together. In other words, the dyadic representation of LoRA-MoE update contains  $m \times r$  terms, but the router only provides m weights. MoDE addresses this by introducing atomic rank-one adapters. This design choice allows MoDE to leverage the dyadic product representation of LoRA, where each rank-one update captures a specific direction of change in the original weight matrix.

MoDE employs m rank-one experts for each column vector  $\mathbf{b_j}$  in  $\mathbf{B}$  (Figure 1), resulting in  $m \times r$  experts. Each expert  $E^i_j$  (where  $i \in \{1,...,m\}$  and  $j \in \{1,...,m\}$ ) specializes in a specific component of the up-projection, represented as a dyadic product  $\mathbf{a}_j \otimes \mathbf{b}_j^{iT}$ , where  $\mathbf{a}_j$  is the j-th column vector of the shared down-projection matrix  $\mathbf{A}$  and  $\mathbf{b}_j^i$  is the vector representing the i-th rank-one expert for the j-th component of  $\mathbf{B}$ .

Let  $\mathcal{R}_{j}^{i}(\mathbf{x})$  is the routing probability for expert  $E_{j}^{i}$  given input  $\mathbf{x}$ , the output with the MoDE module is:

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^m \sum_{j=1}^r \mathcal{R}_j^i(\mathbf{x})(\mathbf{x}(\mathbf{a}_j \otimes \mathbf{b}_j^{i^T})).$$

**Model Expressivity** The router in MoDE independently selects the expert used for each vector  $\mathbf{b}_j$ . This fine-grained control allows for flexible combination of these dyadic product experts, enabling MoDE to dynamically compose a specialized upprojection matrix tailored to the input and task. For example, if  $\mathbf{B}$  has rank 4, the router might select  $E_1^1$  for  $\mathbf{b}_1$ ,  $E_2^3$  for  $\mathbf{b}_2$ ,  $E_3^2$  for  $\mathbf{b}_3$ , and  $E_4^1$  for  $\mathbf{b}_4$ .

With m rank-one experts per vector  $\mathbf{b}_j$  of a rank-r up-projection matrix, MoDE can model  $m^r$  different expert compositions, allowing for a wide range of task-specific adaptations compared to the m experts in a traditional LoRA-MoE, given a similar number of parameters. This increased expressivity, derived from the flexible combination of dyadic products, allows MoDE to better capture

the nuances of individual tasks while maintaining scalability for a large number of tasks.

**MoDE Routing** MoDE utilizes a token-level soft routing strategy, where the router  $\mathcal{R}$  assign a weight to each rank-one expert for a given input token. The weighted sum of experts outputs determines the final output. This approach enables dynamic utilization of the most relevant experts for each input, facilitating nuanced and context-aware adaptation.

The router network is denoted as  $\mathbf{W}_{\mathcal{R}} \in \mathbb{R}^{r \times P \times m}$ , where  $\mathbf{W}_{\mathcal{R};j} \in \mathbb{R}^{P \times m}$  represents the network for vector  $\mathbf{b}_j$  in the up-projection matrix. For an input  $\mathbf{x}$ , the routing weights  $\mathcal{R}_j \in \mathbb{R}^{1 \times m}$  for the experts corresponding to  $\mathbf{b}_j$  are calculated as

$$\mathcal{R}_j(\mathbf{x}) = softmax(\mathbf{x} \cdot \mathbf{W}_{\mathcal{R};j}).$$

This mechanism allows MoDE to adaptively combine the expertise of multiple rank-one adapters, leading to improved multi-task performance.

#### <span id="page-4-0"></span>3.4 Generalization

The rank-1 adapters in MoDE can be generalized to rank-p, where the router selects a composition of rank-p adapters for each input. This requires the LoRA rank r to be divisible by adapter rank p. The generalized output calculation becomes:

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^{m} \sum_{k=1}^{r/p} \mathcal{R}_k^i(\mathbf{x}) \cdot \mathbf{x}\mathbf{A}_k \mathbf{B}_k^{iT}$$

where

$$\mathbf{A}_k \mathbf{B}_k^{iT} = \sum_{j=1}^p (\mathbf{a}_{j+p(k-1)} \otimes \mathbf{b}_{j+p(k-1)}^i).$$

We denoted this generalized module as MoDE  $m \times r \times p$ . Note that MoDE  $1 \times r \times r$  is computationally equivalent to a LoRA module of rank r, and MoDE  $m \times r \times r$  is computationally equivalent to LoRA-MoE-SD with rank r and m experts.

### 4 Experiments

We comprehensively evaluate MoDE's performance and analyze its design choices through three sets of experiments on the Supernatural Instructions (SNI) benchmark (Wang et al., 2022b): (1) multi-task evaluation on the full dataset, (2) an ablation study on generalized MoDE architecture (Section 3.4) on the full dataset, and (3) a case study with a fixed number of tasks and parameter budgets. This section details the experimental setup and presents the results.

<span id="page-5-1"></span>

| Category (Task ID)             | Instruct | Input     | Output  |
|--------------------------------|----------|-----------|---------|
| QuestionAnswering(24)          | 104      | 87/90.9   | 8/8.8   |
| WrongCandidateGeneration(25)   | 127      | 100/104.1 | 8/8.8   |
| QuestionGeneration(74)         | 97       | 143/155   | 12/12.5 |
| GrammarErrorDetection(89)      | 89       | 10/10.6   | 6/6     |
| LinguisticProbing(114)         | 66       | 25/25.7   | 1/1     |
| PosTagging(155)                | 19       | 23/23.7   | 1/1     |
| Explanation(192)               | 43       | 123/131.4 | 30/33.7 |
| StoryComposition(269)          | 184      | 82/82.2   | 34/4.7  |
| StereotypeDetection(279)       | 89       | 15/15.5   | 2/2     |
| CommonsenseClassification(291) | 54       | 17/17.8   | 1/1     |
| ProgramExecution(622)          | 62       | 94/93.6   | 99/98.5 |
| FillInTheBlank(672)            | 24       | 13/3.2    | 1/1     |
| PoemGeneration(1711)           | 83       | 3/3.6     | 44/59   |
| DialogueGeneration(1729)       | 58       | 155/156.9 | 13/12.7 |

Table 1: Sequence lengths of instruction, input (median/mean) and output (median/mean) in the selected SNI datasets for the fixed parameter budget case study.

### <span id="page-5-0"></span>4.1 Datasets and Metrics

We leverage the Supernatural Instructions (SNI) dataset (with 1,616 diverse instruction-following tasks covering 76 distinct task types) for our experiments, focusing on the 756 English-only tasks from the default train split for both training and evaluation. For each task, we split the examples into a 90% training set and 10% evaluation set. For multi-task experiments, we create mixed training and evaluation datasets by combining examples from all 756 tasks.

We also curate a diverse subset of 15 individual tasks from the 756 tasks for the case study given a fixed parameter budgets. These tasks are selected from 15 different categories to ensure a comprehensive evaluation across different domains. Statistics about the sequence length of each task is shown in Tab. [1.](#page-5-1) Each of the selected tasks contains more than 5k instances for training and from 500 to 650 instances for evaluation.

Evaluation Metric We report ROUGE-L, the default metric for SNI dataset, for all experiments.

### 4.2 Implementation Details

Model The Gemma 2B language model [\(Team](#page-10-1) [et al.,](#page-10-1) [2024\)](#page-10-1) serves as the foundational LLM for all experiments due to its state-of-the-art performance on a variety of natural language processing tasks and its efficient size.

Fine-tuning Setup For all experiments, we finetune the parameter-efficient adaptors using Adafactor optimizer[\(Shazeer and Stern,](#page-10-13) [2018\)](#page-10-13) with a learning rate of 1e-3, a total sequence length of 1024, and a batch size of 128 over 20,000 steps.

<span id="page-5-2"></span>

|                | Eval  | Add. Params. |
|----------------|-------|--------------|
| LoRA 64        | 56.11 | 6.31%        |
| MoLORA 16×4    | 57.77 | 7.62%        |
| MoLORA-SD 16×4 | 58.28 | 2.71%        |
| MoDE 16×4      | 60.00 | 6.64%        |
| MoDE 8×4       | 59.00 | 3.48%        |
| MoDE 6×4       | 60.91 | 2.69%        |
| MoDE 4×4       | 60.18 | 1.90%        |
| MoDE 4×6       | 60.53 | 2.86%        |
| MoDE 4×8       | 58.92 | 3.81%        |
| MoDE 4×16      | 60.04 | 7.62%        |

Table 2: Row 2-5: multitask performance comparison between LoRA, MoLORA, MoLORA-SD and MoDE. Row 6-11: ablation study on MoDE with varying number of ranks and experts. The evaluation metric used is ROUGE-L. The last column represents the total number of adapter parameters as a percentage of the total number of non-embedding parameters in Gemma-2B.

### 4.3 Multi-task performance

We assess model performance on the the full dataset comprising 756 tasks. We compare MoDE with vanilla LoRA and MoLORA [\(Zadouri et al.,](#page-10-6) [2023\)](#page-10-6), a strong baseline approach of LoRA-MoE for multi-task adaptation. To better understand the benefit of removing redundancy in down-projection matrices, we also apply down-projection sharing to MoLORA, referred to as MoLORA-SD. In addition to evaluating overall performance, we conduct an ablation study to investigate the impact of varying the number of experts (m) and the rank (r) of the LoRA matrices on MoDE's performances.

The specific models and configurations we evaluate are as follows:

- *LoRA 64*: A base LoRA module with rank 64, having roughly the same number of parameters as other MoE models.
- *MoLORA 16*×*4*: A LoRA-MoE model with 4 experts, each using a rank-4 LoRA module.
- *MoLORA-SD 16*×*4*: A LoRA-MoE model with a shared down-projection matrix and 4 experts with rank-4 up-projection matrices.
- *MoDE 16*×*4*: A MoDE model with 4 experts per vector of a rank-4 up-projection matrix.
- *MoDE m*×*r*: MoDE models with different number of experts (m) and ranks (r) for ablation study. We explore the combinations 8×4, 6×4, 4×4, 4×6, 4×8, and 4×16.

Model Comparison Raw 2 - 5 in Table [2](#page-5-2) reveal several key findings. First, all LoRA-MoE methods outperforms a single LoRA, with improvements ranging from 2.96% (MoLORA 16x4) to 6.93% (MoDE 16x4) in ROUGE-L scores. This demon-

<span id="page-6-0"></span>

| Model     | LoRA | MoLORA | MoLORA-SD |
|-----------|------|--------|-----------|
| LoRA      | \    | \      | \         |
| MoLORA    | 69%  | \      | \         |
| MoLORA-SD | 69%  | 60%    | \         |
| MoDE      | 78%  | 73%    | 68%       |

Table 3: Win rate against baseline models at the tasklevel evaluation.

strates the effectiveness of MoE-based architectures in multi-task settings, allowing the model to leverage specialized experts for different tasks.

The advantage of parameter sharing is evident in the comparison between MoLoRA 16x4 and MoLoRA-SD 16x4. By sharing the downprojection matrix, MoLoRA-SD achieves an 0.88% improvement over MoLoRA while using only 36% of the additional parameters, highlighting the benefit of reducing parameter redundancy.

Notably, all four models with a 16x4 configuration utilize the same number of effective parameters during inference. Among these, MoDE 16x4 achieves the highest overall performance by leveraging both shared down-projection and rankone adapters. This showcases the effectiveness of MoDE's design in balancing parameter efficiency with the need for expressive and adaptable models in multi-task scenarios.

We further conduct a task-level analysis of all the 756 evaluation tasks and report the win rate between each pair of models in Table [3.](#page-6-0) Each raw of the table shows the win rate of target model against models in each column. We find that all LoRA-MoE methods outperform the single LoRA baseline in around 70%-80% of tasks. Importantly, MoDE significantly outperforms all three baselines, passing the significance test over 50% win rate at 0.99 confidence, demonstrating its consistent superiority across a wide range of tasks.

Ablation Studies The ablation study results (rows 6-11 in Table [2\)](#page-5-2) delve into the impact of the number of experts (m) and the rank of the downprojection matrix (r).

*Number of Experts (m)*: Increasing the number of experts initially improves performance (MoDE 4×4 v.s. MoDE 6×4), suggesting that having more experts allows for better specialization. However, further increasing the number of experts to 8 or 16 (MoDE 8×4 or MoDE 16×4) does not lead to any improvement in performance, suggesting diminishing returns beyond a certain point.

*Rank r*: For a fixed number of experts (4), increasing the rank of the LoRA matrices from 4 to 6

(MoDE 4×4 vs. MoDE 4×6) results in a slight performance improvement (0.6018 vs. 0.6053 ROUGE-L). This suggests that higher rank matrices can capture more nuanced information, leading to better adaptation to different tasks. Further increasing the rank of both down-projection and up-projection matrices to 8 or 16 (MoDE 4×8 or MoDE 4×16) leads to a decrease in performance.

## 4.4 Generalized MoDE Architecture

To gain a deeper understanding of the impact expert rank (p) on model performance, we conduct two sets of experiments with the generalized MoDE architecture (Section [3.4\)](#page-4-0). Following the notations m and r, the number of experts becomes m × r/p, where each expert is a rank p adaptor.

Varying Expert Rank (p) In the first set of experiments, we vary the expert rank p while keeping m and r fixed on two m and r combinations (4 × 16 and 16 × 4). The results are presented in Table [4.](#page-6-1) We observe that, with fixed m and r, increasing the expert rank generally leads to improved performance, as indicated by higher ROUGE-L scores. This suggests that increasing the expressiveness of individual experts contributes to better overall multi-task performance.

<span id="page-6-1"></span>

| m  | Model Config.<br>r | p  | Eval  | Add. Params. |  |  |
|----|--------------------|----|-------|--------------|--|--|
| 4  | 16                 | 16 | 58.51 | 2.71%        |  |  |
| 4  | 16                 | 8  | 59.15 | 3.04%        |  |  |
| 4  | 16                 | 4  | 59.56 | 3.69%        |  |  |
| 4  | 16                 | 2  | 59.76 | 5.00%        |  |  |
| 4  | 16                 | 1  | 59.93 | 7.62%        |  |  |
| 16 | 4                  | 4  | 58.97 | 2.71%        |  |  |
| 16 | 4                  | 2  | 59.30 | 4.02%        |  |  |
| 16 | 4                  | 1  | 59.91 | 6.64%        |  |  |

Table 4: Experiments on generalized MoDE architecture. r: the rank of the LoRA matrices. p: the rank of each experts. Number of experts: m×r/p. The last column represents the total number of adapter parameters as a percentage of the total number of non-embedding parameters in Gemma-2B.

Iso-parametric Configurations In the second set of experiments, we explore iso-parametric configurations of MoDE, where the total number of added parameters remains approximately constant across different model configurations. We vary the LoRA rank r (4, 8, or 16) and expert rank p (from 1 to r), adjusting m to maintain a consistent parameter budget. Table [5](#page-7-0) presents the results of these experiments, providing insights into the trade-offs

<span id="page-7-0"></span>between different hyperparameter choices under a fixed resource constraint.

| m  | Model Config.<br>r | p  | Eval  | Add. Params. |
|----|--------------------|----|-------|--------------|
| 42 | 4                  | 4  | 59.89 | 6.58%        |
| 27 | 4                  | 2  | 60.06 | 6.55%        |
| 16 | 4                  | 1  | 59.91 | 6.64%        |
| 27 | 8                  | 8  | 60.52 | 6.48%        |
| 20 | 8                  | 4  | 60.74 | 6.61%        |
| 12 | 8                  | 2  | 60.23 | 6.19%        |
| 7  | 8                  | 1  | 59.73 | 6.18%        |
| 15 | 16                 | 16 | 60.77 | 6.55%        |
| 12 | 16                 | 8  | 60.94 | 6.49%        |
| 8  | 16                 | 4  | 60.77 | 6.07%        |
| 5  | 16                 | 2  | 60.28 | 5.92%        |
| 3  | 16                 | 1  | 59.42 | 6.04%        |

Table 5: Experiments on generalized MoDE architecture with iso-parametric constraint.

Impact of Expert Rank (p): For a fixed LoRA rank r, increasing the expert rank p generally improve performance, as seen when comparing configurations with the same LoRA rank but different expert ranks (e.g., 16×1 v.s. 16×2 v.s. 16×4 v.s. 16×8). This indicates that enhancing the expressiveness of individual experts with higher expert ranks contributes to better multi-task performance under the iso-parametric setting. However, gains diminish as the expert rank p approaches to the LoRA rank r, suggesting that using p < r is beneficial.

The best overall performance is achieved by the 12×16×8 configuration, which balances a moderate number of experts with a reasonably high LoRA rank and expert rank. This emphasize the importance of finding the optimal balance between these hyperparameters for strong multi-task performance.

### 4.5 Case Study with Fixed Parameter Budget

In real-world scenarios, there are often constraints on the number of additional parameters that can be introduced during model adaptation. To assess the effectiveness of MoDE under such constraints, we conduct a case study with a fixed parameter budget determined by the baseline models.

We leverage the diverse subset of 15 individual tasks from the SNI dataset, each belonging to a distinct category, as described in Section [4.1.](#page-5-0) Our baseline model consists of 15 individual rank-4 LoRA adapters (denoted as *LoRA 15*×*4*), one for each task, resulting in approximately 6 million additional trainable parameters compared to the frozen LLM backbone. This baseline establishes our parameter budget for further experimentation.

To ensure a fair comparison, we identify configurations for LoRA (trained on mixture of tasks), MoDE, MoLORA, and MoLORA-SD that introduce a similar number of parameters (approximately 6 million). We systematically explore combinations of experts (m) and ranks (r) for each method, aiming to keep the total number of additional parameters as close as possible to the baseline budget. Specifically, we experiment with ranks of 4, 8, 16, and 32, adjusting the number of experts accordingly to maintain the desired parameter count. Our notation "m×r" indicates a model with m experts, each using adapters associated with a LoRA with rank r. This results in the following configurations:

- *LoRA 1*×*60*
- *MoRA 14*×*4, 6*×*8, 3*×*16*
- *MoLORA 12*×*4, 6*×*8, 3*×*16*
- *MoLORA-SD 36*×*4, 24*×*8, 12*×*16, 5*×*32*

Note that MoLORA-SD, due to its parameter efficiency from sharing the down-projection matrix, can accommodate a configuration with a higher rank (5x32) while still adhering to the budget.

Overall Performance As shown in Figure [5,](#page-8-0) MoDE consistently achieves comparable or superior performance to the baseline of 15 individual LoRA adapters and alternative MoE approaches (MoLORA and MoLORA-SD). This demonstrates MoDE's ability to effectively leverage its shared down-projection matrix and dyadic experts to achieve strong multi-task performance while maintaining parameter efficiency.

Notably, MoDE achieves a substantial improvement in the overall average ROUGE-L score compared to the baseline and other MoE models. This result highlights MoDE's effectiveness in balancing parameter efficiency with the flexibility to adapt to diverse tasks.

Impact of Experts and Rank While individual task performance varies across different MoDE configurations, we observe that the overall average performance across all tasks and examples remains relatively stable despite changes in the number of experts (m) and the rank (r). This suggests that MoDE's performance is robust to these hyperparameter choices, and there may not be a single "best" configuration for all scenarios. The optimal choice of experts and rank might depend on specific task characteristics, resource constraints, or desired

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 5: Performance comparison among various model configurations on 15 tasks with a fixed parameter budget.

trade-offs between model size and performance.

Benefit of down-projection sharing MoLORA-SD, which shares the down-projection matrix like MoDE, generally outperforms the standard MoLORA with independent down-projection matrices. This highlights the importance of reducing parameter redundancy and promoting knowledge sharing across tasks through a shared down-projection matrix.

Benefit of Mixture of Dyadic Experts Comparing MoDE and MoLORA-SD configurations, we observe that MoDE often achieves better performance. This suggests that the dyadic experts in MoDE contribute to its superior expressivity and adaptability compared to simply sharing the down-projection matrix.

These findings demonstrates that MoDE can effectively leverage a fixed parameter budget to achieve strong multi-task performance. Its shared down-projection matrix and mixtures of dyadic experts enable a balance between parameter efficiency and expressive power, making it a promising approach for deploying multi-task LLMs in resource-constrained environments.

### 5 Conclusion and Future Work

In this paper, we introduce MoDE (Mixture of Dyadic Experts), a novel parameter-efficient fine-tuning method for multi-task adaptation of large

language models. MoDE addresses the limitations of existing LoRA-based MoE architectures by sharing the down-projection matrix across experts to remove parameter redundancy. The key innovation of MoDE lies in its use of rank-one adapters, combined with a sophisticated routing mechanism that allows for nuanced and flexible combinations of these adapters. This design fosters knowledge sharing, reduces redundancy, and increases expressive power, enabling the model to effectively capture the unique characteristics of each task.

Our experiments on the Supernatural Instructions benchmark demonstrate that MoDE consistently outperforms state-of-the-art multi-task PEFT methods, achieving superior performance with comparable parameter efficiency. This highlight MoDE's potential as an efficient and effective solution for multi-task LLM adaptation, particularly in resource-constrained environments.

Future work will explore top-k routing strategies to further enhance MoDE's efficiency and adaptability. We also plan to analyze the routing behavior to identify task-specific patterns, and assess MoDE's generalization capabilities to unseen tasks. Additionally, we aim to evaluate MoDE on larger models and alternative PEFT techniques.

# 6 Limitations

While our proposed MoDE architecture demonstrates promising results in multi-task LLM adaptation, there are several limitations that warrant further investigation.

Routing Strategy The current MoDE implementation utilizes a relatively simple routing mechanism based on a softmax function. While effective in our experiments, exploring more sophisticated routing strategies that incorporate task relationships or input-specific features could potentially further improve performance.

Hyperparameter Sensitivity The optimal number of experts and rank of the LoRA matrices can vary depending on the specific task distribution and available resources. While our ablation study provides some insights, a more comprehensive exploration of hyperparameter sensitivity could help identify optimal configurations for different scenarios.

Computational Overhead While MoDE significantly reduces parameter count compared to traditional LoRA-MoE, the routing mechanism introduces additional computational overhead during inference. This overhead could become a bottleneck in real-time applications with strict latency requirements. Investigating ways to optimize the routing process or reduce its computational cost would be beneficial.

Evaluation Benchmark Our evaluation primarily focuses on the Supernatural Instructions benchmark. While this dataset covers a wide range of tasks, it may not fully represent the diversity of real-world applications. Evaluating MoDE on other multi-task benchmarks or in specific domains could further validate its effectiveness and generalizability. Addressing these limitations could lead to even more efficient and adaptable multi-task LLM architectures, further expanding the potential of parameter-efficient fine-tuning for a wider range of applications.

# References

<span id="page-9-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901.

- <span id="page-9-1"></span>Rich Caruana. 1997. Multitask learning. *Machine learning*, 28:41–75.
- <span id="page-9-5"></span>William Fedus, Barret Zoph, and Noam Shazeer. 2022. [Switch transformers: Scaling to trillion parameter](https://arxiv.org/abs/2101.03961) [models with simple and efficient sparsity.](https://arxiv.org/abs/2101.03961) *Preprint*, arXiv:2101.03961.
- <span id="page-9-6"></span>Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. 2024. Mixture-of-loras: An efficient multitask tuning for large language models. *arXiv preprint arXiv:2403.03432*.
- <span id="page-9-2"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.
- <span id="page-9-9"></span>Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2023. Lorahub: Efficient cross-task generalization via dynamic lora composition. *arXiv preprint arXiv:2307.13269*.
- <span id="page-9-4"></span>Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. [Mix](https://arxiv.org/abs/2401.04088)[tral of experts.](https://arxiv.org/abs/2401.04088) *Preprint*, arXiv:2401.04088.
- <span id="page-9-8"></span>Dengchun Li, Yingzi Ma, Naizheng Wang, Zhiyuan Cheng, Lei Duan, Jie Zuo, Cal Yang, and Mingjie Tang. 2024. Mixlora: Enhancing large language models fine-tuning with lora based mixture of experts. *arXiv preprint arXiv:2404.15159*.
- <span id="page-9-3"></span>Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. 2022. [Branch-train-merge: Embarrassingly](https://arxiv.org/abs/2208.03306) [parallel training of expert language models.](https://arxiv.org/abs/2208.03306) *Preprint*, arXiv:2208.03306.
- <span id="page-9-10"></span>Chu-Cheng Lin, Xinyi Wang, Jonathan H Clark, Han Lu, Yun Zhu, Chenxi Whitehouse, and Hongkun Yu. 2024. Multitask multilingual model adaptation with featurized low-rank mixtures. *arXiv preprint arXiv:2402.17934*.
- <span id="page-9-7"></span>Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, and Yefeng Zheng. 2023. Moelora: An moe-based parameter efficient finetuning method for multi-task medical applications. *arXiv preprint arXiv:2310.18339*.
- <span id="page-9-11"></span>Yijiang Liu, Rongyu Zhang, Huanrui Yang, Kurt Keutzer, Yuan Du, Li Du, and Shanghang Zhang. 2024. [Intuition-aware mixture-of-rank-1-experts](https://arxiv.org/abs/2404.08985) [for parameter efficient finetuning.](https://arxiv.org/abs/2404.08985) *Preprint*, arXiv:2404.08985.

- <span id="page-10-2"></span>OpenAI and Josh Achiam et al. 2024. [Gpt-4 technical](https://arxiv.org/abs/2303.08774) [report.](https://arxiv.org/abs/2303.08774) *Preprint*, arXiv:2303.08774.
- <span id="page-10-11"></span>Oleksiy Ostapenko, Zhan Su, Edoardo Maria Ponti, Laurent Charlin, Nicolas Le Roux, Matheus Pereira, Lucas Caccia, and Alessandro Sordoni. 2024. Towards modular llms by building and reusing a library of loras. *arXiv preprint arXiv:2405.11157*.
- <span id="page-10-3"></span>Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.
- <span id="page-10-9"></span>Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. 2023. Ziplora: Any subject in any style by effectively merging loras. *arXiv preprint arXiv:2311.13600*.
- <span id="page-10-12"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.
- <span id="page-10-13"></span>Noam Shazeer and Mitchell Stern. 2018. Adafactor: Adaptive learning rates with sublinear memory cost. In *International Conference on Machine Learning*, pages 4596–4604. PMLR.
- <span id="page-10-4"></span>Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Rozière, Jacob Kahn, Daniel Li, Wen tau Yih, Jason Weston, and Xian Li. 2024. [Branch-train-mix: Mixing ex](https://arxiv.org/abs/2403.07816)[pert llms into a mixture-of-experts llm.](https://arxiv.org/abs/2403.07816) *Preprint*, arXiv:2403.07816.
- <span id="page-10-0"></span>Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*.
- <span id="page-10-1"></span>Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. *arXiv preprint arXiv:2403.08295*.
- <span id="page-10-8"></span>Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. 2022a. Adamix: Mixtureof-adaptations for parameter-efficient model tuning. *arXiv preprint arXiv:2205.12410*.
- <span id="page-10-7"></span>Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. 2022b. Super-naturalinstructions:generalization via declarative instructions on 1600+ tasks. In *EMNLP*.
- <span id="page-10-10"></span>Xun Wu, Shaohan Huang, and Furu Wei. 2024. [Mixture](https://openreview.net/forum?id=uWvKBCYh4S) [of lora experts.](https://openreview.net/forum?id=uWvKBCYh4S)

- <span id="page-10-6"></span>Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermi¸s, Acyr Locatelli, and Sara Hooker. 2023. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. *arXiv preprint arXiv:2309.05444*.
- <span id="page-10-5"></span>Yun Zhu, Nevan Wichers, Chu-Cheng Lin, Xinyi Wang, Tianlong Chen, Lei Shu, Han Lu, Canoee Liu, Liangchen Luo, Jindong Chen, et al. 2023. Sira: Sparse mixture of low rank adaptation. *arXiv preprint arXiv:2311.09179*.

## A Detailed Results for Case Study with Fixed Parameter Budge

Table [6](#page-11-0) presents the detailed model performance with 12 model configures on the 15 tasks.

<span id="page-11-0"></span>

| LoRA<br>Task ID |       |       | MoLORA |       |       | MoLORA-SD |       |       | MoDE  |       |       |       |
|-----------------|-------|-------|--------|-------|-------|-----------|-------|-------|-------|-------|-------|-------|
|                 | 15×4  | 1×60  | 12×4   | 6×8   | 3×16  | 36×4      | 24×8  | 12×6  | 5×32  | 14×4  | 6×8   | 3×16  |
| task24          | 25.91 | 25.30 | 26.67  | 26.16 | 26.29 | 26.81     | 27.26 | 27.12 | 27.71 | 27.40 | 27.31 | 27.55 |
| task25          | 41.27 | 41.35 | 41.7   | 41.43 | 41.99 | 41.91     | 41.73 | 41.92 | 42.36 | 42.29 | 41.92 | 42.97 |
| task74          | 42.73 | 42.53 | 43.12  | 42.86 | 42.64 | 42.67     | 42.88 | 43.53 | 43.30 | 42.34 | 43.35 | 43.03 |
| task89          | 33.90 | 38.21 | 64.41  | 63.64 | 57.24 | 71.42     | 71.19 | 64.71 | 61.48 | 77.73 | 76.43 | 78.20 |
| task114         | 84.15 | 87.85 | 91.54  | 89.85 | 90.15 | 91.08     | 92.00 | 90.92 | 91.38 | 90.77 | 90.62 | 90.62 |
| task141         | 75.79 | 82.87 | 94.00  | 91.08 | 91.85 | 94.15     | 94.31 | 93.95 | 93.49 | 96.00 | 95.33 | 94.92 |
| task155         | 51.09 | 53.88 | 62.42  | 59.16 | 56.99 | 61.18     | 61.80 | 54.35 | 60.71 | 63.82 | 65.37 | 63.66 |
| task192         | 72.47 | 77.21 | 83.35  | 80.41 | 80.86 | 82.99     | 82.68 | 84.47 | 83.61 | 84.78 | 83.42 | 85.69 |
| task269         | 75.40 | 75.51 | 75.46  | 75.56 | 75.50 | 75.61     | 75.52 | 75.33 | 75.72 | 75.59 | 75.71 | 75.52 |
| task279         | 69.90 | 82.59 | 89.52  | 88.39 | 88.44 | 90.60     | 89.68 | 90.45 | 91.42 | 91.53 | 91.73 | 91.58 |
| task291         | 79.97 | 85.14 | 86.64  | 86.31 | 85.31 | 86.31     | 86.31 | 85.81 | 85.81 | 87.48 | 87.65 | 86.81 |
| task622         | 99.95 | 99.70 | 99.97  | 99.93 | 99.96 | 99.96     | 99.93 | 99.93 | 99.94 | 99.96 | 99.94 | 99.93 |
| task672         | 39.69 | 45.08 | 48.92  | 46.46 | 45.85 | 48.31     | 49.69 | 51.69 | 48.62 | 50.46 | 51.85 | 50.31 |
| task1711        | 4.19  | 8.33  | 10.40  | 7.66  | 8.43  | 9.81      | 10.16 | 9.85  | 10.19 | 9.97  | 8.37  | 9.16  |
| task1729        | 17.21 | 17.11 | 17.43  | 16.77 | 17.34 | 17.25     | 17.35 | 17.60 | 17.68 | 17.31 | 17.75 | 17.61 |
| Overall         | 53.83 | 57.07 | 61.93  | 60.63 | 60.16 | 62.25     | 62.42 | 61.66 | 61.79 | 63.39 | 63.37 | 63.39 |

Table 6: Performance comparison among various model configurations on 15 tasks with a fixed parameter budget. The scores in blue and green correspond to the highest and second-highest scores for the corresponding task.

## B PCA Clustering of LoRA Matrices

Scatter plots after applying Principal Component Analysis (PCA) on all of the LoRA projection matrices at layer 6 and 12 are shown in Figure [6.](#page-12-0) The distinct grouping of down-projection vectors indicates common representations across tasks, providing the inspiration for the MoDE architecture.

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 6: Scatter plots after applying Principal Component Analysis (PCA) on all LoRA projection matrices, i.e. query, key, value, and output, sliced along the rank dimension.The clear clustering of down-projection vectors suggests the presence of shared representations across tasks, motivating the design of the MoDE architecture.