# Abstract

As large language models continue to scale, computational costs and resource consumption have emerged as significant challenges. While existing sparsification methods like pruning reduce computational overhead, they risk losing model knowledge through parameter removal. This paper proposes DSMoE (Dynamic Sparse Mixture-of-Experts), a novel approach that achieves sparsification by partitioning pretrained FFN layers into computational blocks. We implement adaptive expert routing using sigmoid activation and straight-through estimators, enabling tokens to flexibly access different aspects of model knowledge based on input complexity. Additionally, we introduce a sparsity loss term to balance performance and computational efficiency. Extensive experiments on LLaMA models demonstrate that under equivalent computational constraints, DSMoE achieves superior performance compared to existing pruning and MoE approaches across language modeling and downstream tasks, particularly excelling in generation tasks. Analysis reveals that DSMoE learns distinctive layerwise activation patterns, providing new insights for future MoE architecture design.

## 1 Introduction

Large Language Models(LLM) have demonstrated remarkable performance across various downstream tasks[\(Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Dai et al.,](#page-9-0) [2022;](#page-9-0) [Anil et al.,](#page-8-0) [2023;](#page-8-0) [Biderman et al.,](#page-8-1) [2023\)](#page-8-1). However, as model sizes continue to expand, computational costs and resource consumption grow exponentially. How to improve computational efficiency while maintaining model performance has become a pressing challenge[\(Cheng et al.,](#page-8-2) [2024\)](#page-8-2).

At the algorithmic level, approaches to model efficiency optimization generally follow two paradigms: post-training compression and acceleration of dense models, or training of Mixture of Experts (MoE) architectures. While compression methods like pruning achieve efficiency through permanent parameter removal[\(Ashkboos et al.,](#page-8-3) [2024;](#page-8-3) [Ma et al.,](#page-9-1) [2023;](#page-9-1) [Frantar and Alistarh,](#page-9-2) [2023\)](#page-9-2), they may discard valuable knowledge and lack flexibility in handling inputs of varying complexity. Conversely, though MoE approaches effectively expand model capacity[\(Fedus et al.,](#page-9-3) [2022;](#page-9-3) [Dai et al.,](#page-8-4) [2024;](#page-8-4) [Liu et al.,](#page-9-4) [2024\)](#page-9-4), traditional MoE typically employs fixed activation patterns where each token can only access a predetermined number of experts, lacking the ability to dynamically adjust computation based on input complexity. Given that the most widely used and effective foundation models still maintain dense architectures (such as LLaMA[\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0), Qwen[\(Bai et al.,](#page-8-5) [2023\)](#page-8-5)), we face a critical challenge: how to achieve truly input-adaptive computation while preserving pre-trained knowledge, allowing models to dynamically adjust activated parameters according to varying input complexity, thereby reaching an optimal balance between computational efficiency and model performance.

To address this challenge, we propose DSMoE, a novel approach that partitions pre-trained FFN layers into computational blocks and introduces dynamic routing mechanisms. DSMoE fundamentally differs from existing methods by preserving the original model parameters and reorganizing them into expert networks, while incorporating adaptive routing mechanisms that enable dynamic expert activation based on input complexity, rather than fixed activation strategies. Through straightthrough estimator and sparsity loss design, DSMoE enables the model to autonomously learn sparse expert activation patterns, achieving computational resource allocation for inputs of varying complex-

<sup>\*</sup>These authors contributed equally to this work.

<sup>†</sup>Corresponding authors.

This work was supported by National Natural Science Foundation of China (Nos. 62271281, 62441235, 62525103). It was also sponsored by CCF-Kuaishou Large Model Explorer Fund (No. CCF-Kuaishou 2024001).

![](_page_1_Figure_0.jpeg)

Figure 1: The Overview of DSMoE versus Traditional MoE Framework Architectures. The structure shown in the figure is a simplified representation of the transformer backbone. We have simplified the FFN layer structure here; the FFN layer also includes a gating matrix with dimensions matching the upper matrix, which performs Hadamard multiplication with the upper matrix without affecting our partitioning scheme. In the FFN layer, we partition matrices along the intermediate dimension, where portions corresponding to the original matrix multiplication form new expert FFN layers.

ity.

Extensive experiments conducted on LLaMA-1B and LLaMA-7B models demonstrate encouraging results. Under equivalent computational constraints, our method achieves significant improvements in language modeling perplexity and downstream task performance compared to existing pruning and MoE approaches. Notably superior performance is observed in reasoning and question-answering tasks, particularly in generation tasks.

The main contributions of this work include:

- proposing a novel approach that enables transition from dense to dynamically sparse models by preserving and partitioning pre-trained knowledge, enabling different tokens to adaptively access varying portions of model knowledge.
- validating the method's effectiveness across multiple benchmarks through extensive experimentation, providing new insights for MoE large model optimization.

#### 2 Related Work

Model pruning is an effective approach to achieving sparse LLMs while maintaining model functionality. Pruning methods can be categorized into two main types: unstructured and structured pruning. Unstructured pruning operates at the weight level, allowing for arbitrary weight removal (Lee et al., 2018). In large language models, pruned

weights are set to zero (Frantar and Alistarh, 2023; Sun et al., 2023). However, this method requires specialized hardware and software support for acceleration(Han et al., 2015; Wen et al., 2016; Filters'Importance, 2016; Tang et al., 2021). Structured pruning takes a coarser-grained approach by removing complete structural units such as convolution kernels, channels, attention heads, or entire layers (You et al., 2019; Ashkboos et al., 2024; Liu et al., 2021; Ma et al., 2023; Men et al.). Its main advantage is the ability to directly produce regular, narrow model architectures that can achieve acceleration without specialized sparse computation libraries (Luo et al., 2017; Liu et al., 2021; Filters'Importance, 2016; Nonnenmacher et al., 2021). However, both approaches face a fundamental limitation: achieving efficiency through permanent parameter removal may discard valuable knowledge and lose the ability to adapt computation based on input complexity.

The Mixture of Experts architecture is recognized as a promising approach for model sparsification. Recently, it has garnered significant research attention, with several studies investigating methodologies for converting pre-trained models into MoE architectures.

While various MoE approaches exist with different objectives, methods like Llama-MoE v2 focus on post-training optimization of instruction-tuned models, and approaches like DTSI target parameter efficiency during training from scratch. How-

ever, these methods either address specialized posttraining scenarios or require training models from initialization, whereas our approach specifically targets sparsification of pre-trained dense models during the pre-training stage.

MoEfication(Zhang et al., 2022) trains routers to predict the activation patterns of experts that are partitioned from FFNs while keeping model parameters frozen, thereby activating a fixed number of experts. However, this method was primarily designed for ReLU activation functions and requires additional transformation steps for SiLU/GeLU activation functions that are widely utilized in contemporary Transformer architectures. FactorLLM(Zhao et al., 2024) employs a multistage training strategy, initially utilizing the original dense model to guide router training, followed by fixing the router and subsequently training the experts. This sequential training methodology constrains collaborative optimization between routers and experts, and its dependence on a teacher-student framework introduces additional training complexity. LLaMA-MoE(Zhu et al., 2024) explores the decomposition of FFNs and organizes training according to the Switch Transformer(Fedus et al., 2022) paradigm; however, it merely provides improved expert initialization while lacking flexible input-adaptive computation mechanisms. Given that MoEfication and FactorLLM differ significantly from mainstream MoE methods in architecture design and training paradigms, we choose to use LLaMA-MoE as a comparative approach.

Recent dynamic pruning methods such as DejaVu(Liu et al., 2023) and PowerInfer(Song et al., 2024) can adaptively select activated weights based on input patterns. However, these approaches primarily focus on system-level acceleration through specialized hardware configurations: DejaVu requires integration with asynchronous hardware-aware implementations including kernel fusion and memory coalescing, while PowerInfer employs GPU-CPU hybrid inference engines to exploit locality patterns and minimize communication overhead. In contrast, our method employs algorithm-level sparsification, which reduces the model's floating-point operations.

#### 3 Background

For simplicity, we focus on the prevalent architecture of generative large language models while

maintaining a concise mathematical formulation. In autoregressive generation tasks, given a sequence  $X=(x_1,x_2,...,x_T)$  of length T, the model iteratively produces a probability distribution over the vocabulary for each position conditioned on preceding tokens. This process can be formulated as:

$$\begin{split} P_{\cdot,t} &= \operatorname{softmax}(EH_{\cdot,t}^L) \\ H^L &= \operatorname{Transformer}(x_1, x_2, ..., x_{T-1}) \end{split} \tag{1}$$

Here, L denotes the number of layers in the Transformer architecture. For any position t,  $P_{\cdot,t}$  represents the probability distribution over the vocabulary, derived from the t-th column of the hidden state matrix  $h^L$ . Specifically,  $H^L = [h_1^L, h_2^L, ..., h_{T-1}^L]$  contains the hidden representations from the final layer, where  $h_t^L$  is the contextual embedding at position t. The probability of the ground-truth token  $x_{t+1}$  is denoted as  $P_{x_{t+1},t}$  in the distribution  $P \cdot t$ . The transformation from hidden states to probability distributions is achieved through a linear projection matrix E, followed by a softmax operation.

In typical scenarios, we employ cross-entropy loss for autoregressive learning, which can be expressed as:

$$\mathcal{L}_{LM} = -\sum_{t=1}^{T-1} \log P(x_{t+1}|x_{\leq t})$$
 (2)

The Transformer architecture consists of multiple layer-wise submodules, where each layer comprises a self-attention module and a Feed-Forward Network (FFN) module. The simplified mathematical formulation can be expressed as:

$$\hat{h_t^l} = \text{Attn}([h_1^{l-1}, h_2^{l-1}, ..., h_t^{l-1}]) \tag{3}$$

$$h_t^l = \text{FFN}(\hat{h_t^l}) \tag{4}$$

FFN modules typically consist of two matrix transformations with a non-linear activation function. In modern language models, the most prevalent FFN implementation uses SwiGLU activation, which involves three essential matrices: the upprojection matrix  $\mathbf{U}_{up}$ , the down-projection matrix  $\mathbf{V}_{down}$ , and the gate matrix  $\mathbf{W}_{gate}$ . The upprojection matrix transforms the input to a higher dimensional space for richer feature representation, the down-projection matrix compresses the information back to the original dimension, and the gate matrix controls information flow through adaptive

feature weighting. The FFN output is computed through the following operation:

<span id="page-3-0"></span>
$$h_t^l = (act(\hat{h_t^l}W_{gate}) \odot (\hat{h_t^l}U_{up}))V_{down} \quad (5)$$

In this formulation,  $act(\cdot)$  represents the activation function and  $\odot$  denotes Hadamard product.

### 4 Method

Although our method is termed DSMoE, its training approach differs from traditional MoE methods such as Switch Transformer and DeepSeeKMoE (Dai et al., 2024). Our objective is to achieve sparsity through partitioning pre-trained models, where each expert inherits a distinct portion of the original model's knowledge. Our approach is based on the principle that the model should learn to selectively utilize different aspects of pre-trained knowledge based on input complexity, rather than routing tokens among independently trained experts. To implement this insight, we present our method in three modules.

#### 4.1 FFN Partitioning

The widespread adoption of MoE architectures inspires our exploration of sparsity in FFN layers, suggesting that different parts of computation can be dynamically activated based on input patterns. Previous work has further revealed that FFN lavers essentially operate as key-value memories, where different portions of the layer specialize in detecting and processing distinct input patterns(Geva et al., 2020). Building on these insights, we propose to directly partition pre-trained FFN layers. As shown in Equation 5, we partition the matrices U, V, and W into n groups along the intermediate dimension, where each group can be viewed as an "expert" that inherits a portion of the original transformation capabilities. When summing all expert outputs, this partitioned form is mathematically equivalent to the original FFN computation:

$$h_t^l = (act(\hat{h}_t^l [W_1 \cdots W_n]) \odot$$

$$(\hat{h}_t^l [U_1 \cdots U_n])) \begin{bmatrix} V_1 \\ \vdots \\ V_n \end{bmatrix}$$

$$= (act(\hat{h}_t^l W_1) \odot \hat{h}_t^l U_1) V_1 + \cdots$$

$$+ (act(\hat{h}_t^l W_n) \odot \hat{h}_t^l U_n) V_n$$

$$(6)$$

To enable dynamic expert activation based on input, we employ a gating network that determines which experts should be activated. The expert's output is propagated to the subsequent layer only when the corresponding gating activation value exceeds a certain threshold  $\tau$ . This can be formulated as:

<span id="page-3-1"></span>
$$o_{i} = (act(\hat{h}_{t}^{l}W_{i}) \odot \hat{h}_{t}^{l}U_{i})V_{i}$$

$$h_{t}^{l} = \sum_{i=1}^{n} o_{i} * G(\sigma(\hat{h}_{t}^{l}\mathbf{Y}_{i}))$$

$$G(x) = \begin{cases} x & \text{if } x > \tau \\ 0 & \text{others} \end{cases}$$

$$(7)$$

where  $\mathbf{Y} = [\mathbf{Y}_1, \dots, \mathbf{Y}_n] \in \mathbb{R}^{d \times n}$  represents the parameters of the gating network, and  $\sigma(\cdot)$  denotes the sigmoid activation function.

To maintain consistent output norm regardless of the number of active experts, similar to dropout, we scale  $h_t^l$  by the ratio of total expert count n to the number of activated experts. This normalization can be expressed as:

$$h_t^l = \frac{n \cdot h_t^l}{\sum_{i=1}^n \mathbb{I}[\sigma(\hat{h}_t^l \mathbf{Y}_k) > \tau]}$$
(8)

#### 4.2 Straight-Through Estimator

A key challenge in converting dense models to sparse ones is maintaining the learning capability of all experts. During the forward pass, experts with activation values below the threshold  $\tau$  do not participate in computation, as defined by the gating function G(x) in Equation 7. However, this thresholding operation creates a critical problem during backpropagation - experts that are not activated receive zero gradients:

$$\frac{\partial h_t^l}{\partial \mathbf{V}_i} = \frac{\partial h_t^l}{\partial \mathbf{W}_i} = \frac{\partial h_t^l}{\partial \mathbf{U}_i} = \frac{\partial h_t^l}{\partial \mathbf{Y}_i} = \mathbf{0}, \text{if } \sigma(\hat{h}_t^l \mathbf{Y}_i) \le \tau$$
(9)

This gradient blocking prevents non-activated experts from receiving training signals, leading to a "dead expert" problem where these experts become permanently inactive. Due to random initialization of the sigmoid gating parameters, experts with initially low activation probabilities below the threshold receive zero gradients and cannot improve through training, creating a Matthew effect where inactive experts remain progressively underutilized. Unlike traditional MoE models that train experts from scratch, our experts inherit pre-trained

<span id="page-4-0"></span>

| Model              | Configuration                         | Params | Activated Params | PPL (↓) |
|--------------------|---------------------------------------|--------|------------------|---------|
| LLaMA-1B           | d=2048, D=8192                        | 1.24B  | 1.24B            | 5.67    |
| LLaMA-7B           | d=4096, D=11008                       | 6.74B  | 6.74B            | 3.40    |
| LLaMA-1B           |                                       |        |                  |         |
| LLM-Pruner-channel | d=1215, D=8192                        | 889M   | 889M             | 7.51    |
| LLM-Pruner-block   | d=2048, D=3896.4                      | 735M   | 735M             | 7.46    |
| SparseGPT          | d=2048, D=8192                        | 1.24B  | 735M             | 9.82    |
| LLaMA-MoE          | $d=2048$ , $D=1024 \times 8$ , topK=3 | 1.24B  | 736M             | 7.45    |
| DSMoE(ours)        | d=2048, D=1024 ×8                     | 1.24B  | 735M             | 7.41    |
| LLaMA-7B           |                                       |        |                  |         |
| LLM-Pruner-channel | d=2401, D=11008                       | 3.95B  | 3.95B            | 4.01    |
| LLM-Pruner-block   | d=4096, D=6256.5                      | 3.94B  | 3.94B            | 4.01    |
| SparseGPT          | d=4096, D=11008                       | 6.74B  | 3.93B            | 3.96    |
| LLaMA-MoE          | d=4096, D=1376 $\times 8$ , topK=3    | 6.74B  | 3.98B            | 4.12    |
| DSMoE(ours)        | d=4096, D=1376 ×8                     | 6.74B  | 3.93B            | 3.91    |

Table 1: Results of perplexity (PPL) across different language models. The **bold** values indicate the best-performing method among various acceleration approaches. The Configuration column describes the specific model architecture, where d represents the hidden dimension, D denotes the expansion dimension in FFN layers (for LLM-Pruner-block method, this represents the average value),  $\times$  n indicates the use of n parallel FFN layers, and topK specifies the number of activated experts per layer in the MoE architecture. The Params column shows the total number of model parameters, while Activated Params indicates the average number of parameters activated during inference.

knowledge that we wish to preserve and adapt. To address this issue, we employ the straight-through estimator technique, which allows gradient flow through non-activated experts while maintaining thresholded activation during the forward pass:

$$S(x) = sq(G(x)) + x - sq(x)$$
 (10)

<span id="page-4-1"></span>
$$h_t^l = \sum_{i=1}^n o_i \cdot S(\sigma(\hat{h}_t^l \mathbf{Y}_k))$$
 (11)

where the operator " $sg(\cdot)$ " is the "stop gradient" operator to prevent gradient back propagation. The partial derivatives for experts and their gates below the threshold are as follows. Let:

$$a_{i} = \operatorname{act}(\hat{h}_{t}^{l} \mathbf{W}_{i}), \quad a'_{i} = \operatorname{act}'(\hat{h}_{t}^{l} \mathbf{W}_{i})$$

$$q_{i} = \sigma(\hat{h}_{t}^{l} \mathbf{Y}_{i}), \quad u_{i} = \hat{h}_{t}^{l} \mathbf{U}_{i}$$
(12)

The gradients for expert parameters and their gates can be derived as:

$$\frac{\partial h_t^l}{\partial \mathbf{V}_i} = \begin{cases} (a_i \odot u_i)^\top \cdot g_i & \text{if } g_i > \tau \\ \mathbf{0} & \text{if } g_i \le \tau \end{cases}$$
(13)

$$\frac{\partial h_t^l}{\partial \mathbf{W}_i} = \begin{cases} (\hat{h}_t^l)^\top \odot a_i' \cdot ((u_i \odot \mathbf{V}_i) \cdot g_i) & \text{if } g_i > \tau \\ \mathbf{0} & \text{if } g_i \le \tau \end{cases}$$
(14)

$$\frac{\partial h_t^l}{\partial \mathbf{U}_i} = \begin{cases} (\hat{h}_t^l)^\top \cdot (a_i \odot \mathbf{V}_i \cdot g_i) & \text{if } g_i > \tau \\ \mathbf{0} & \text{if } g_i \le \tau \end{cases}$$
(15)

$$\frac{\partial h_t^l}{\partial \mathbf{Y}_i} = (\hat{h}_t^l)^\top \cdot (o_i \cdot \sigma'(\hat{h}_t^l \mathbf{Y}_i))$$
(16)

The gradient dynamics show a key property: with the straight-through estimator, experts receive gradients for their gating parameters regardless of activation status. The gradient direction for  $\mathbf{Y}_i$  depends on whether the expert's output  $o_i$  would reduce the overall loss. This allows experts to adaptively learn when to activate based on their usefulness for specific input patterns.

#### 4.3 Sparse Loss

Since our experts inherit from a dense model, the model naturally tends to activate all experts to access complete knowledge. However, this conflicts with our goal of sparse computation. We introduce a sparsity loss term that creates an adversarial effect with expert gate gradients, encouraging the model to learn which knowledge is truly necessary for different inputs.

$$\mathcal{L} = \mathcal{L}_{LM} + \mathcal{L}_{sparse} \tag{17}$$

where  $\mathcal{L}_{sparse}$  denotes the sparsity loss term, which we abbreviate as  $\mathcal{L}s$  in subsequent equations.

$$\mathcal{L} = \mathcal{L}_{LM} + \frac{1}{LN} \sum_{l=1}^{L} \sum_{n=1}^{N} \mathcal{L}_s(G(\sigma(\hat{h}_t^l \mathbf{Y}_n)))$$
 (18)

We employ L1 norm as the sparsity function  $\mathcal{L}_s$ . Given that our activation function  $\sigma(x) > 0$ , our final loss function becomes:

$$\mathcal{L} = \mathcal{L}_{LM} + \frac{1}{LN} \sum_{l=1}^{L} \sum_{n=1}^{N} G(\sigma(\hat{h}_t^l \mathbf{Y}_n))$$
 (19)

The gradients introduced by this sparse loss term create an adversarial effect with the gate gradients, encouraging the model to actively suppress the output of less important experts across different layers.

It is worth noting that our approach differs fundamentally from the MoE framework and therefore does not require auxiliary load balancing losses. While load balancing losses in MoE aim to ensure uniform training across experts, our objective is solely focused on learning sparse activation patterns. Furthermore, unlike MoE which typically enforces a fixed number of active experts, our method allows for flexible activation patterns determined by the learned gating mechanism.

