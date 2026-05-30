# Mixture of Diverse Size Experts

### Manxi Sun, Wei Liu, Jian Luan, Pengzhi Gao, and Bin Wang

Xiaomi AI Lab, Beijing, China

{sunmanxi, liuwei40, luanjian, gaopengzhi, wangbin11}@xiaomi.com

### Abstract

The Sparsely-Activated Mixture-of-Experts (MoE) has gained increasing popularity for scaling up large language models (LLMs) without exploding computational costs. Despite its success, the current design faces a challenge where all experts have the same size, limiting the ability of tokens to choose the experts with the most appropriate size for generating the next token. In this paper, we propose the Mixture of Diverse Size Experts (MoDSE), a new MoE architecture with layers designed to have experts of different sizes. Our analysis of difficult token generation tasks shows that experts of various sizes achieve better predictions, and the routing path of the experts tends to be stable after a training period. However, having experts of diverse sizes can lead to uneven workload distribution. To tackle this limitation, we introduce an expert-pair allocation strategy to evenly distribute the workload across multiple GPUs. Comprehensive evaluations across multiple benchmarks demonstrate the effectiveness of MoDSE, as it outperforms existing MoEs by allocating the parameter budget to experts adaptively while maintaining the same total parameter size and the number of experts.

### 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable performance in a variety of NLP tasks and have become valuable assistants through a wide range of applications. The scaling law [\(Ka](#page-7-0)[plan et al.,](#page-7-0) [2020\)](#page-7-0) demonstrates that larger models exhibit superior performance. However, training larger models requires increased computational resources, posing a critical challenge. Mixture-of-Experts (MoE) [\(Fedus et al.,](#page-7-1) [2022;](#page-7-1) [Lepikhin et al.,](#page-7-2) [2021\)](#page-7-2) address this challenge by using sparse activation to scale up the trainable parameters while maintaining high training and inference efficiency. Recent MoE-based architectures, such as Mixtral of Experts [\(Jiang et al.,](#page-7-3) [2024\)](#page-7-3), DeepSeekMoE [\(Dai](#page-7-4)

[et al.,](#page-7-4) [2024\)](#page-7-4), and OpenMoE [\(Xue et al.,](#page-8-0) [2024\)](#page-8-0) have shown superior performance in various tasks.

Specifically, [Dai et al.](#page-7-4) [\(2024\)](#page-7-4) discuss two main issues in the design of the MoE Feed-Forward Networks (FFNs) architecture: Knowledge Hybridity, where each expert covers diverse knowledge due to the limited number of experts, and Knowledge Redundancy, where multiple experts share common knowledge. To address these issues, they propose Fine-Grained Expert Segmentation by splitting the FFN intermediate hidden dimension and Shared Expert Isolation by isolating certain experts to be always activated as shared experts. Additionally, [Zhao et al.](#page-8-1) [\(2024\)](#page-8-1) introduce Hypernetworks and HyperExperts modules to capture the cross-expert and cross-layer knowledge.

However, almost all existing MoE architectures consist of experts with identical structures and sizes. This homogeneous architecture becomes a significant bottleneck when generating tokens with varying difficulty; some tokens are easier to predict, while others are more challenging. To deal with the varied difficulty, we propose the Diverse Size Experts structure for each FFN layer, where each expert has a different parameter size to handle generating tasks of varying difficulty. Note that we find a similar recent work called Heterogeneous the Mixture of Experts [\(Wang et al.,](#page-8-2) [2024\)](#page-8-2), which shares a similar motivation and utilizes parameter penalty loss and router entropy loss to control the size and number of activated experts.

Our contributions are summarized as follows:

• Diverse Size Experts We introduce the Mixture of Diverse Size Experts (MoDSE) in Section [3,](#page-1-0) a new type of FFN layer designed for the MoE framework. Unlike conventional MoEs, which consist of experts of the same size, MoDSE has experts of different sizes. It assigns each token to the expert that best matches its prediction needs in terms of capability, thereby enhancing the model's ability.

- Load Balance GPU nodes containing larger experts in MoDSE will have a heavier workload. To address this issue, we propose the expert-pair allocation method in Section 3.2, which ensures that each GPU node carries an even distribution of parameters, thus maintaining load balance.
- Empirical Validation MoDSE outperforms conventional MoE with a lower loss value across a diverse set of benchmarks in the  $700M \times 8$  model setting, confirming the effectiveness of our approach. We present the evaluation results in Section 4.2.
- Token Routing Analysis We collect the routing distribution of tokens in both the MoE baseline model and MoDSE, and conduct a thorough analysis in Section 4.3. MoDSE exhibits an equally even distribution as the baseline. Additionally, we analyze tokens that are more difficult to predict and find that they are better predicted when routed to an expert which is better suited to handle them.

#### 2 Preliminaries: Mixture of Experts

MoE models are usually constructed by replacing dense FFNs layers in the Transformer (Vaswani et al., 2017) with MoE layers. An MoE layer typically consists of multiple experts  $E_1(\cdot) \cdots E_N(\cdot)$ and the corresponding gate model  $G_1(\cdot) \cdots G_N(\cdot)$ , N indicates the number of the experts. The gate model (Shazeer et al., 2017) with trainable weight matrices  $W_q \in \mathcal{R}^{h_{input} \times h}$  and  $W_n \in \mathcal{R}^{h_{input} \times h}$ selects the top k experts and combines the outputs of experts to produce the output  $y \in \mathcal{R}^h$ , where  $h_{input}$  is the dimension of input x and h is the dimension of the hidden layer. Fedus et al. (2022) set k as one, while Lepikhin et al. (2021); Jiang et al. (2024) set as two. The outputs of experts are added with the noise to help with load balance. The noise generated from the input hidden vector x is multiplied by  $W_n$  and processed by Softplus and the Root Mean Square Layer Normalization function (RMSNorm), where  $\gamma$  is a learnable coefficient.

$$y = \sum_{i=1}^{N} G_i(x)E_i(x) \tag{1}$$

$$G(x) = Softmax(KeepTopK(H(x), k))$$
 (2)

$$H(x)_i = (x \cdot W_a)_i + \text{RMSNorm}(f((x \cdot W_n)_i))$$

$$KeepTopK(v, k)_i = \begin{cases} v_i & v_i \in topk(v) \\ -\infty & otherwise \end{cases}$$
 (3)

RMSNorm
$$(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}$$
 (4)

$$f(\cdot) = Softplus(\cdot) = \log(1 + \exp(\cdot))$$
 (5)

#### <span id="page-1-0"></span>3 MoDSE Architecture

Predicting the next token is easier within frequently appearing token pairs in the corpus. Tokens within the same word or phrase are easier to generate than those between two phrases or words. Analogous to the human brain, the amount of thought required to generate the next word varies among different words. Inspired by the fact that the difficulty of generating each next token varies, we propose MoDSE as shown in Figure 1. In our work, the size of the expert parameters is used to quantify the amount of thinking involved. We assign experts a range of parameter sizes by setting the dimensions of the hidden layers to various lengths. However, the imbalance in expert size leads to an uneven workload. To address this issue, we propose a meticulously designed expert-pair allocation method to ensure each GPU node's workload is evenly distributed.

#### 3.1 Diverse Size Experts

In a traditional MoE structure (Fedus et al., 2022; Lepikhin et al., 2021), the gating network combines a set of experts with the same size. We here adjust the scale of experts to ensure that different experts can handle tasks of varying difficulty. Note that we denote the designed Diverse Size Experts as  $\{\hat{E}_1(\cdot), \cdots, \hat{E}_N(\cdot)\}$ , and the dimension of the hidden layer for  $\hat{E}_i(\cdot)$  is  $\hat{h}_i$ .

$$\hat{y} = \sum_{i=1}^{N} \hat{G}_{i}(x)\hat{E}_{i}(x)$$
 (6)

$$(i_1^1, i_1^2), \cdots, (i_n^1, i_n^2), \text{ with } n = \frac{N}{2}$$
 (7)

$$\hat{h}_{i_k^1} + \hat{h}_{i_k^2} = 2 \times h$$
 , , with  $k \in 1 \cdots n$  (8)

To maintain the overall parameter size, the experts are grouped into pairs  $(i_k^1, i_k^2)$ , where  $k \in 1 \cdots n$  indicates the pair of the experts. The average value of  $\hat{h}_i$  within each pair equals h, with one expert being larger than the average size and the other smaller. Typically, the number of experts is even, ensuring the experts can be grouped into pairs, thus the total parameter size of the MoDSE model matches that of the vanilla MoE model.

<span id="page-2-1"></span>![](_page_2_Picture_0.jpeg)

Figure 1: Overview of a MoDSE layer with different sizes of experts. In this case, expert1\_0 and expert2\_0 are selected. With the output of the gating network, the outputs of two experts are integrated.

#### <span id="page-2-0"></span>3.2 Load Balance Consideration

In MoDSE, experts with hidden layer sizes larger than the average have a higher workload due to the increased number of parameters, both during training and inference phrases. To address this load imbalance problem, we propose the expertpair allocation strategy, which places each pair of experts on the same GPU and ensures that each GPU contains an equal number of parameters. For instance, in Figure [1,](#page-2-1) expert pairs are enclosed by dotted line frames, with expert 0 and expert 1 on the same GPU, and so forth.

Besides the standard cross entropy (CE) loss, we use the auxiliary load balance loss L<sup>a</sup> from Switch Transformers [\(Fedus et al.,](#page-7-1) [2022\)](#page-7-1) to penalize the unbalanced routing distribution among experts. Consequently, each expert has the same frequency of being routed. In Section [4.3,](#page-4-0) we will demonstrate that after the entire training process, all tokens in the pre-training dataset are evenly spread across all experts. Along with the expert-pair allocation method, this ensures that the final workload of each GPU is balanced.

$$L_a = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i, \tag{9}$$

where α is a scalar hyperparameter. f<sup>i</sup> is the fraction of tokens routed to expert i, i ∈

$$\{1,2,\cdots,N\}$$
:

$$f_i = \frac{1}{T} \sum_{x \in \text{Batch}} \mathbf{1} \{ \operatorname{argmax} \ p(x) = i \}, \qquad (10)$$

$$p(x) = [p_1(x), p_2(x), \cdots, p_N(x)],$$
 (11)

where T is the number of tokens and P<sup>i</sup> is the fraction of the router probability for expert i:

$$P_i = \frac{1}{T} \sum_{x \in \text{Batch}} p_i(x) \tag{12}$$

### 4 Experiments

