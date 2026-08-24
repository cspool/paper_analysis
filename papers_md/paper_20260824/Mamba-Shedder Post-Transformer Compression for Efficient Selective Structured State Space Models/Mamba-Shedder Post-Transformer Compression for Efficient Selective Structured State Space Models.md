# Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models

J. Pablo Muñoz <sup>1</sup>\*, Jinjie Yuan<sup>2</sup>\*, Nilesh Jain<sup>1</sup>

1 Intel Labs, <sup>2</sup> Intel Corporation

{pablo.munoz, jinjie.yuan, nilesh.jain}@intel.com

# Abstract

Large pre-trained models have achieved outstanding results in sequence modeling. The Transformer block and its attention mechanism have been the main drivers of the success of these models. Recently, alternative architectures, such as Selective Structured State Space Models (SSMs), have been proposed to address the inefficiencies of Transformers. This paper explores the compression of SSM-based models, particularly Mamba and its hybrids. We study the sensitivity of these models to the removal of selected components at different granularities to reduce the model size and computational overhead, thus improving their efficiency while maintaining accuracy. The proposed solutions, collectively referred to as Mamba-Shedder, achieve a speedup of up to 1.4x during inference, demonstrating that model efficiency can be improved by eliminating several redundancies with minimal impact on the overall model performance. The code is available at [https://github.com/IntelLabs/Hardware-](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)[Aware-Automated-Machine-Learning.](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)

# 1 Introduction

We have seen an outstanding increase in the number of Transformer-based models [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0) developed to tackle tasks from Natural Language Processing (NLP) and other domains [\(Par](#page-11-1)[mar et al.,](#page-11-1) [2018;](#page-11-1) [Dosovitskiy et al.,](#page-10-0) [2021;](#page-10-0) [Arnab](#page-9-0) [et al.,](#page-9-0) [2021;](#page-9-0) [Gong et al.,](#page-10-1) [2021\)](#page-10-1) due to their effectiveness at modeling sequences. However, these models also present critical efficiency challenges. For example, the cost of training these models scales quadratically in the sequence length. In the generation stage, Transformers, in their original form, require large caches to store the previously seen tokens. Several variants of Transformers have been proposed to address these efficiency challenges, but researchers have also explored alternative postTransformer architectures to address these limitations. *Structured state space models (SSMs)*, e.g., S4 [\(Gu et al.,](#page-10-2) [2022\)](#page-10-2), followed by *Selective state space models*, e.g., Mamba [\(Gu and Dao,](#page-10-3) [2023;](#page-10-3) [Dao and Gu,](#page-10-4) [2024\)](#page-10-4) have been proposed as efficient alternatives that achieve training time with linear scaling in sequence length, and during generation, maintain constant state size.

Model compression methods, e.g., pruning and quantization, have been broadly explored and applied to Transformer-based models. However, more must be done to explore compression in their structured state space counterparts. This paper explores the pruning of these alternative architectures, presenting results that provide insights into potential opportunities to increase their efficiency without sacrificing accuracy. The rest of the paper discusses the following contributions:

- A pruning solution, Mamba-Shedder, which targets structures in selective structured state space models, improving their computational and memory efficiency.
- Comprehensive experiments to determine the tolerance of SSM-based models to the removal of their structures.
- Insights on how the differences in the SSM building blocks and their interaction with Transformer blocks in hybrid models affect the trade-off between efficiency and accuracy.

The following content is organized as follows: Section [2](#page-1-0) provides the reader with details of the alternative architectures utilized in our study and popular strategies for element removal in large models. Section [3](#page-2-0) describes methods to study network pruning in Mamba and hybrid architectures. Section [4](#page-3-0) presents the results of our experiments and ablation studies, and we offer concluding remarks in Section [5.](#page-8-0) A Related Work section is included in the Appendix.

<sup>\*</sup>Co-first authors.

# <span id="page-1-0"></span>2 Preliminaries

#### 2.1 State Space Models

State space models (SSMs) have a long history of modeling sequences and dynamic systems. Recently, *structured* SSMs, e.g., S4 [\(Gu et al.,](#page-10-2) [2022\)](#page-10-2), have been proposed as an alternative to Transformers because of their efficient capabilities for mapping input to output signals. When dealing with discrete sequences as in Natural Language Processing (NLP), the parameters A, B and C of these models are discretized to transform an input sequence, x<sup>t</sup> , and hidden state, h<sup>t</sup> , to obtain the output sequence, yt . It can be formalized as:

<span id="page-1-1"></span>
$$h_t = \mathbf{A}h_{t-1} + \mathbf{B}x_t, y_t = \mathbf{C}^{\top}h_t.$$
 (1)

Mamba: Selective State Space Models S4 and other structured SSMs are linear time-invariant (LTI), i.e., their parameters are fixed, limiting their effectiveness for sequence modeling. For instance, structured state space models fail in many contentand context-based reasoning tasks. These limitations have motivated the development of timevarying alternatives, e.g., Mamba [\(Gu and Dao,](#page-10-3) [2023\)](#page-10-3), which incorporate selection mechanisms and are suitable for solving tasks previously SSM generations failed. Specifically, Mamba's SSM module, S6, allows its parameters to depend on the input, thereby modifying the formulation from time-invariant to time-varying. A second improvement proposed in Mamba compared to previous SSMs is a hardware-aware algorithm that speeds up execution while reducing memory IOs.

Furthermore, Mamba-2 [\(Dao and Gu,](#page-10-4) [2024\)](#page-10-4) improves the original Mamba architecture by proposing *state space duality (SSD)*, which improves its efficiency on hardware accelerators compared to S6. This improvement is achieved by changing the *state matrix*, A, which directly controls the latent state, h. A is modified from being structured as a diagonal matrix to a formulation that utilizes a scalar-times-identity structure.

Additionally, Mamba-2 introduces the concept of heads in SSMs inspired by how multi-head attention (MHA) works and implementing a groupedvalue attention (GVA) head structure. Overall, the Mamba-2 architecture, with its SSD core component, allows for improved parallelism of the block's projections.

Mamba block Mamba models comprise several blocks stacked after each other. Figure [1](#page-2-1) on the left illustrates a single Mamba block. Each block has the selective SSM mechanism (S6 for Mamba-1 and SSD for Mamba-2) at its core, placed within a larger structure that combines a gated multilayer perceptron (MLP), a convolution, and SILU activation functions [\(Elfwing et al.,](#page-10-5) [2018\)](#page-10-5).

For more details about selective structured state space models, we refer the reader to [Gu and Dao](#page-10-3) [\(2023\)](#page-10-3) and [Dao and Gu](#page-10-4) [\(2024\)](#page-10-4).

#### 2.2 Hybrid Models

Lately, new models have been proposed that achieve the best of both worlds (Transformers and Selective SSMs) by proposing architectures with both classes of blocks. Zamba [\(Glorioso et al.,](#page-10-6) [2024\)](#page-10-6) is one example of such a hybrid model. It combines the strengths of Mamba's backbone and the efficiency of selective SSMs with a shared Transformer block that incorporates Transformers' powerful in-context learning capabilities. The *shared attention* mechanism, in which two attention blocks are reused and interleaved in an ABAB pattern throughout the network, is a characteristic innovation of Zamba. This model also applies LoRA adapters [\(Hu et al.,](#page-10-7) [2022\)](#page-10-7) to the shared MLP blocks, achieving specialization when interacting with the affected layers, memory efficiency, and faster inference with reduced computational overhead.

Another example of a hybrid model is Hymba [\(Dong et al.,](#page-10-8) [2024\)](#page-10-8). This model takes a different approach than Zamba, proposing an entirely new hybrid-head module, illustrated in Figure [1](#page-2-1) on the right, in which the SSM and Attention mechanisms contribute in parallel to the sequence modeling. Additionally, Hymba benefits from group query attention, cross-layer KV cache sharing, and learnable meta-tokens, resulting in higher throughput, reduced memory requirements, and competitive performance compared to models of similar size.

# 2.3 Model Pruning

A popular model compression technique, *pruning* [\(LeCun et al.,](#page-11-2) [1989\)](#page-11-2), has been effectively used to reduce the size of deep learning models and improve their efficiency. Network pruning operates at two levels: (1) *Unstructured pruning* identifies the importance of individual weights that can be masked to minimize their impact on overall model behavior. At a different level, (2) *structured pruning* focuses on removing more significant structural components of the model, such as whole Trans-

<span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> → Mamba Block Linear > Mamba Block → Channel A → MLP Block Linear Remove A -> MLP Block !>% SSM MLP SSM > SSM MLP SSM → SSM ->
> 
> Hymba Block -> Attn Block SSM Conv Hybrid-head Conv Attn Linear Linear module Attn Linear Linear -> € Attn Block Mamba Mamba + Transformers Hymba
![](_page_2_Figure_0.jpeg)

Figure 1: Overview of Mamba-Shedder. This figure illustrates the pruning strategy for three types of Mamba-based models. The first type includes Mamba models such as Mamba-1 [\(Gu and Dao,](#page-10-3) [2023\)](#page-10-3), Mamba-2 [\(Dao and Gu,](#page-10-4) [2024\)](#page-10-4), and Falcon-Mamba [\(Zuo et al.,](#page-12-0) [2024\)](#page-12-0). The second type comprises Mamba + Transformers architectures, including Zamba [\(Glorioso et al.,](#page-10-6) [2024\)](#page-10-6). The third type is Hymba [\(Dong et al.,](#page-10-8) [2024\)](#page-10-8), a novel architecture with hybrid heads. Red dashed lines indicate potential removal. In Transformers, channel pruning can also be applied to MLP block (width pruning).

former blocks [\(Men et al.,](#page-11-3) [2024\)](#page-11-3), or reducing the granularity to target subcomponents of these layers [\(Zhong et al.,](#page-11-4) [2024;](#page-11-4) [Muñoz et al.,](#page-11-5) [2025\)](#page-11-5). Other dimensions for pruning include groups of channels in the Transformer's MLPs or heads from the MHA layer. In this paper, the focus is solely on structured pruning applied to Mamba-based models.

Next, we discuss Mamba-Shedder's methodology to study redundancies in Mamba and hybrid models.

# <span id="page-2-0"></span>3 Methodology

Due to the large sizes of current state-of-the-art sequence models, Mamba-Shedder requires an efficient strategy to identify structures that can be removed without significantly affecting the model's accuracy. We approach this problem using a training-free approach, in which the least essential elements are considered for removal. Similar strategies have been explored in Transformer-based large language models [\(Ashkboos et al.,](#page-9-1) [2024;](#page-9-1) [Men](#page-11-3) [et al.,](#page-11-3) [2024;](#page-11-3) [Zhong et al.,](#page-11-4) [2024\)](#page-11-4). However, to our knowledge, no study explores the removal of structures in Selective Structured State Space models. Mamba-Shedder conducts structure removal of Mamba models and their hybrid variants at different granularities. As illustrated in the left of Figure [1,](#page-2-1) in the case of models with only Mamba blocks, we explore the iterative removal of entire Mamba blocks ([§2.1\)](#page-1-1), or their SSM subcomponents, either S6 or SSD modules depending on the version of Mamba (Figure [1\)](#page-2-1).

The proponents of the Mamba architecture do not provide a rationale for the number of Mamba blocks required to build robust models, opening an opportunity for Mamba-Shedder to investigate whether some components might be redundant and hence removed from the model with a minor impact in accuracy.

In addition to these components, in the case of hybrid models that also contain Transformer blocks (middle of Figure [1\)](#page-2-1), we also explore the removal of entire Transformer blocks or their subblocks: multilayer perceptrons (MLP) modules and multihead attention (MHA) modules. In hybrid models, Mamba-Shedder also explores the removal of structures at a finer granularity by targeting groups of channels in the MLP's linear layers, i.e., based on a channel group size, g, Mamba-Shedder explores the removal of ng channels, where n is the number of groups that could be removed based on their impact of the overall model performance.

#### <span id="page-2-2"></span>Algorithm 1 Block / Module Pruning

Input: Set of blocks/modulesMfrom a model m, Calibration dataset C, Metric ϕ, Target pruning steps t. Output: Pruned model m<sup>∗</sup>

```
1: for k ← 1 to t do
2: for all Mi ∈ M do
3: Si ← Importance(Mi, m, C, ϕ)
4: end for
5: Mmin ← arg minMi∈M Si
6: M ← M \ {Mmin} ▷ Block/Module Pruning
7: end for
8: return m∗ with the remaining blocks/modules in M
```

Algorithm [1](#page-2-2) details the procedure to remove en-

<span id="page-3-2"></span>> **[图片提取文字 (无描述)]:**
> Mamba-2.8B Lambada PPL Avg. Accuracy Lambada PPL AV9.85 Number of Pruned Mamba Blocks
![](_page_3_Figure_0.jpeg)

> **[图片提取文字 (无描述)]:**
> Mamba2-2.7B Lambada PPL Avg. Accuracy Lambada PPL 58 BA Number of Pruned Mamba Blocks
![](_page_3_Figure_1.jpeg)

> **[图片提取文字 (无描述)]:**
> Zamba2-2.7B Lambada PPL Lambada PPL 58 BA Avg. Accuracy Number of Pruned Mamba Blocks
![](_page_3_Figure_2.jpeg)

Figure 2: Pruning Mamba blocks. *Avg. Accuracy* indicates the average accuracy for seven tasks. The model composed of Mamba 1 blocks (left) can tolerate the removal of entire blocks without significantly increasing its perplexity or decreasing accuracy compared to Mamba-2 and Zamba-2. In all three models, removing each Mamba block reduces 0.04B parameters from the model. These are *training-free* results, and drops in accuracy can be reduced by a subsequent fine-tuning stage (§4.5).

<span id="page-3-3"></span>

| Model       | Method                  | Num. of Pruned<br>Mamba Blocks | Ratio            | Lambada<br>PPL (↓)                              | Lambada      | HellaS       | PIQA         | ARC-e        | ARC-c        | WinoG        | OBQA         | Average                                       |
|-------------|-------------------------|--------------------------------|------------------|-------------------------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------------------------------------------|
|             | Dense                   | 0 / 64                         | 0%               | 4.23                                            | 69.2         | 66.1         | 75.2         | 69.7         | 36.3         | 63.5         | 39.6         | 59.9                                          |
| Mamba-2.8B  | Mamba Block Pruning     | 7 / 64                         | 10.43%           | $4.94_{+0.71}$                                  | 65.8         | 63.7         | 73.8         | 68.0         | 33.5         | 62.5         | 36.8         | 57.7.2.2                                      |
|             | Mailloa Block Fluilling | 14 / 64                        | 20.86%           | $7.51_{+3.28}$                                  | 58.9         | 57.6         | 71.0         | 62.7         | 32.0         | 61.1         | 33.2         | 53.8-6.1                                      |
|             | Dense                   | 0 / 64                         | 0%               | 4.10                                            | 69.7         | 66.6         | 76.4         | 69.6         | 36.4         | 64.0         | 38.8         | 60.2                                          |
| Mamba2-2.7B | Mamba Block Pruning     | 7 / 64                         | 10.42%           | 8.43+4.33                                       | 53.0         | 63.8         | 73.9         | 66.6         | 36.4         | 64.5         | 35.0         | 56.2.4.0                                      |
|             | Mailloa Block Fluilling | 14 / 64                        | 20.83%           | 11.53+7.43                                      | 47.0         | 59.4         | 71.1         | 60.6         | 35.6         | 60.8         | 35.0         | 52.8-7.4                                      |
|             | Dense                   | 0 / 54                         | 0%               | 4.01                                            | 69.7         | 77.0         | 79.8         | 77.5         | 48.5         | 72.1         | 45.8         | 67.2                                          |
| Zamba2-2.7B | Mamba Block Pruning     | 7 / 54<br>14 / 54              | 10.38%<br>20.77% | 6.80 <sub>+2.79</sub><br>15.8 <sub>+11.79</sub> | 58.9<br>44.3 | 69.7<br>62.8 | 77.0<br>72.7 | 69.8<br>54.3 | 39.6<br>34.5 | 67.0<br>64.3 | 41.8<br>37.2 | 60.5 <sub>-6.7</sub><br>52.9 <sub>-14.3</sub> |

Table 1: Detailed results of Mamba-Shedder with *training-free* Mamba block pruning. Lambada, HellaS, PIQA, ARC-e, ARC-c, WinoG, and OBQA represent their respective accuracies. <u>Underlined</u> numbers indicate the smallest average accuracy gap with the dense model under the same level of pruning.

tire structures, e.g., Mamba or Transformer blocks, MLPs, MHA, or SSM modules. Given a set  $\mathcal{M}$  of structures selected for potential removal, a proxy data set C and a metric  $\phi$  are used to measure the importance of an individual structure and the impact of removing it from the model (Zhong et al., 2024). In addition to entire structures, Mamba-Shedder follows the same logic to remove channel groups as detailed in Algorithm 2.

#### <span id="page-3-1"></span>**Algorithm 2** MLP Channel Pruning

**Input:** Set of MLP blocks  $\mathcal{M}_{\text{MLP}}$  from a model m, Calibration dataset  $\mathcal{C}$ , Metric  $\phi$ , Target pruning steps t, MLP channel group size q.

Output: Pruned model  $m^*$ 

```
1: for k \leftarrow 1 to t do

2: for all M_i \in \mathcal{M}_{\text{MLP}} do

3: S_i \leftarrow \text{Importance}(M_i[:,:-g], m, \mathcal{C}, \phi)

4: end for

5: M_{\min} = \arg\min_{M_i \in \mathcal{M}_{\text{MLP}}} S_i

6: M_{\min} = M_{\min}[:,:-g] \triangleright Channel Pruning

7: end for

8: return m^* with the altered MLP blocks in \mathcal{M}
```

Depending on the pruning objective, Mamba-Shedder might treat these pruning targets in isolation, but Section 4 also presents the results of configurations in which Mamba-Shedder sequentially prunes larger structures (e.g., Mamba blocks) and, at a later stage, smaller components, e.g., SSM modules in the remaining Mamba blocks. Future work will explore larger search spaces with more complex configurations of candidate structures for removal. For example, the importance of Mamba blocks and their SSM modules can be assessed in the same pruning iteration.

#### <span id="page-3-0"></span>4 Experiments

We evaluate Mamba-Shedder and study the removal of structures from SSM-based models utilizing several open-source models and datasets. We analyze their absolute and relative drop in accuracy and quantify the inference speedup obtained by the pruned models. Next, we discuss the resources utilized for our experiments and details of our setup and results.

#### 4.1 Models

Our experiments employed the following pretrained Mamba and hybrid models: Mamba-

<span id="page-4-7"></span>> **[图片提取文字 (无描述)]:**
> Mamba-2.8B Lambada PPL Avg. Accuracy Tampada PPL 10 Number of Pruned SSMs
![](_page_4_Figure_0.jpeg)

> **[图片提取文字 (无描述)]:**
> Mamba2-2.7B Lambada PPL Avg. Accuracy eo 5 Lambada PPL Number of Pruned SSMs
![](_page_4_Figure_1.jpeg)

> **[图片提取文字 (无描述)]:**
> Zamba2-2.7B Lambada PPL Lambada PPL Avg. Accuracy Number of Pruned SSMs
![](_page_4_Figure_2.jpeg)

Figure 3: Pruning SSM (S6 and SSD modules). Mamba-2.8B and Mamba2-2.7B have 64 SSM modules, while Zamba2-2.7B has 54 SSM (SSD) modules. *Avg. Accuracy* is for the seven tasks evaluated.

<span id="page-4-6"></span>

| Model         | Method      | Num. of<br>Pruned SSMs | Lambada<br>PPL (↓) | Lambada | HellaS | PIQA | ARC-e | ARC-c | WinoG | OBQA | Average              |
|---------------|-------------|------------------------|--------------------|---------|--------|------|-------|-------|-------|------|----------------------|
|               | Dense       | 0 / 64                 | 4.23               | 69.2    | 66.1   | 75.2 | 69.7  | 36.3  | 63.5  | 39.6 | 59.9                 |
| Mamba-2.8B    |             | 16 / 64                | 9.23+5.00          | 55.2    | 52.1   | 68.1 | 57.8  | 28.4  | 55.6  | 31.6 | 49.8.10.1            |
| Mamba-2.6D    | SSM Pruning | 20 / 64                | $10.10_{+5.87}$    | 57.1    | 48.2   | 65.5 | 50.9  | 25.9  | 56.0  | 29.4 | 47.6-12.3            |
|               |             | 24 / 64                | 22.55+18.32        | 44.4    | 43.2   | 64.4 | 47.4  | 25.8  | 53.6  | 29.8 | 44.1.15.8            |
|               | Dense       | 0 / 64                 | 4.10               | 69.7    | 66.6   | 76.4 | 69.6  | 36.4  | 64.0  | 38.8 | 60.2                 |
| Mamba2-2.7B   | SSM Pruning | 16 / 64                | $4.26_{+0.16}$     | 66.9    | 66.1   | 76.4 | 68.6  | 37.2  | 64.0  | 39.2 | 59.8-0.4             |
| Maiiiba2-2.7B |             | 20 / 64                | $5.89_{+1.79}$     | 59.8    | 66.0   | 76.1 | 68.9  | 36.7  | 63.6  | 39.2 | 58.6-1.6             |
|               |             | 24 / 64                | $14.95_{+10.85}$   | 43.4    | 65.8   | 74.8 | 67.1  | 36.6  | 62.9  | 38.0 | 55.5 <sub>-4.7</sub> |
|               | Dense       | 0 / 54                 | 4.01               | 69.7    | 77.0   | 79.8 | 77.5  | 48.5  | 72.1  | 45.8 | 67.2                 |
| Zamba2-2.7B   |             | 16 / 54                | $4.14_{+0.13}$     | 69.2    | 75.8   | 79.2 | 75.8  | 46.5  | 72.2  | 45.8 | 66.4 <sub>-0.8</sub> |
|               | SSM Pruning | 20 / 54                | 5.07+1.06          | 64.2    | 75.8   | 79.3 | 75.5  | 46.2  | 73.2  | 46.0 | 65.7-1.5             |
|               |             | 24 / 54                | 5.46+1.45          | 62.3    | 74.7   | 79.0 | 75.4  | 44.3  | 70.9  | 46.4 | 64.7 <sub>-2.5</sub> |

Table 2: Detailed results of Mamba-Shedder with *training-free* SSM pruning. The remaining tasks represent their respective accuracy. Here, we do not consider the pruning ratio, as the number of SSM's parameter weights is small. Its benefit is the reduction of computational overhead. <u>Underlined</u> numbers indicate the smallest gap with Dense under the same level of pruning.

**2.8b** (Gu and Dao, 2023), consists of 64 S6 blocks<sup>1</sup>. Mamba2-2.7b (Dao and Gu, 2024), consists of 64 SSD blocks <sup>2</sup>. Both Mamba models were trained on 300B tokens on the Pile dataset (Gao et al., 2020). For our choice of a hybrid model, we explored **Zamba2-2.7B** (Glorioso et al., 2024)<sup>3</sup>. It has 54 layers, including 45 single Mamba-2 Blocks and 9 hybrid layers composed of both Mamba-2 Blocks and Transformer Blocks. Zamba-2 was trained on 3T tokens from open web datasets, including Zyda (Tokpanov et al., 2024), and subsequently annealed with 100B additional tokens. The aforementioned models are all of the same size and can be compared directly. For Mamba models of different sizes, we also explored Falcon-Mamba-7B (Zuo et al., 2024)<sup>4</sup>, which is based on the Mamba-1 architecture and is the best-performing Mamba model at this scale in the literature, as well as Hymba-**1.5B-Base** (Dong et al., 2024)<sup>5</sup>, which features a

hybrid architecture incorporating both Mamba and Attention heads.

#### 4.2 Datasets

Following the language modeling evaluation of Mamba (Gu and Dao, 2023; Dao and Gu, 2024), we utilize *lm-eval-harness* (Gao et al., 2023) to assess the zero-shot performance, which includes measuring perplexity on Lambada (Paperno et al., 2016), and accuracy on the following downstream tasks: HellaSwag (Zellers et al., 2019), Physical Interaction Question Answering (PIQA) (Bisk et al., 2020), AI2 Reasoning Challenges (Arc-e, Arc-c) (Clark et al., 2018), Large-scale Winograd Schema Challenge (WinoGrande) (Sakaguchi et al., 2021), and the Open Book Question Answering (Mihaylov et al., 2018) dataset.

Regarding the calibration dataset, we follow BlockPruner (Zhong et al., 2024) in using the Alpaca dataset <sup>6</sup> as the calibration dataset and employ perplexity as the metric for calculating importance scores. All the hyperparameters used in our experi-

<span id="page-4-0"></span><sup>&</sup>lt;sup>1</sup>https://huggingface.co/state-spaces/mamba-2.8b

<span id="page-4-1"></span><sup>&</sup>lt;sup>2</sup>https://huggingface.co/state-spaces/mamba2-2.7b

<span id="page-4-2"></span><sup>&</sup>lt;sup>3</sup>https://huggingface.co/Zyphra/Zamba2-2.7B

<span id="page-4-3"></span><sup>&</sup>lt;sup>4</sup>https://huggingface.co/tiiuae/falcon-mamba-7b

<span id="page-4-4"></span><sup>&</sup>lt;sup>5</sup>https://huggingface.co/nvidia/Hymba-1.5B-Base

<span id="page-4-5"></span><sup>&</sup>lt;sup>6</sup>https://github.com/tatsu-lab/stanford\_alpaca

ments are detailed in the Appendix.

#### 4.3 Results

#### 4.3.1 Pruning Target: Mamba Block

This section explores the impact of pruning Mamba blocks on model performance. Figure [2](#page-3-2) and Table [1](#page-3-3) present the results of applying Mamba-Shedder to Mamba-2.8B, Mamba2-2.7B, and Zamba2-2.7B models with a focus on removing redundant entire Mamba blocks. The model that utilizes the first version of Mamba blocks (S6) appears to tolerate a higher number of removed blocks without significantly affecting its performance. Specifically, the Mamba-2.8B model demonstrates robustness, with its perplexity (PPL) increasing from 4.23 to 7.51 and average accuracy dropping from 59.9 to 53.8 when the pruning ratio reaches 20.86%. In contrast, the Mamba2-2.7B and Zamba2-2.7B models exhibit more significant performance degradation, although they performed better before pruning (Dense). The poorer pruning performance of Zamba2-2.7B may be attributed to the pruning of Mamba blocks disrupting a certain balance within the hybrid layers. Overall, the effects of Mamba block pruning vary across different models, depending on the model architecture and the characteristics of the pre-training stage. In this round, Mamba-1 comes out on top.

#### <span id="page-5-0"></span>4.3.2 Pruning Target: SSM Module

In this section, we delve into assessing the impact of pruning only the SSM modules within Mamba blocks on the performance of various models, as illustrated in Table [2](#page-4-6) and Figure [3.](#page-4-7) When using the same target in Mamba-2.8B, we observe that further pruning SSMs results in a noticeable increase in perplexity, soaring to 22.55 and decreasing average accuracy to 44.1. This result indicates a significant sensitivity to SSM pruning for Mamba-1, where performance degradation is pronounced even at moderate pruning levels. Conversely, Mamba2-2.7B and Zamba2-2.7B exhibit remarkable resilience to SSM pruning. Even with 24 SSMs pruned, the model maintains a relatively stable performance. This robustness suggests that Mamba-2 blocks can tolerate higher SSM module pruning, potentially due to Mamba-2's optimizations or different training strategies with Mamba-1. The Zamba2-2.7B model, with the hybrid architecture, outperforms both Mamba-1 and Mamba-2. Pruning 12 out of its 54 SSMs results in a negligible PPL increase from 4.01 to 4.02, while the average

accuracy slightly decreases from 67.2% to 67.0%. The hybrid nature of Zamba2-2.7B may contribute to its ability to maintain performance despite SSM pruning. Overall, these findings underscore the importance of model architecture and training strategies in determining the impact of SSM pruning. They offer valuable insights for optimizing model efficiency without compromising performance. In this round, the model with Mamba-2 blocks comes out on top.

# 4.3.3 Pruning Target: Finer-grained removal of Mamba and Transformer blocks, and their subcomponents

Table [3](#page-6-0) presents the results of pruning various components of the Zamba2-2.7B model, including combinations of Mamba-2 blocks, entire Transformer blocks, and their subcomponents, i.e., MHA blocks, MLP blocks, MLP channels, and SSM modules. We design four search spaces to study the effectiveness of different granularities and their combinations. "&" indicates that the pruning targets are considered together in the same pruning step, while "+" signifies the distinction between pruning stages, with pruning occurring sequentially:

Mamba Block & Transformer Block Pruning This experiment involves pruning the entire Mamba-2 blocks and Transformer blocks.

Mamba Block & MLP & MHA Pruning This experiment decomposes the transformer block into sub-blocks, pruning Mamba-2 blocks as well as MHA and MLP.

Mamba Block & MLP & MHA + MLP Channel Pruning This experiment prunes the Mamba-2 blocks, MHA, and MLP at the first stage and further prunes the MLP channels at the next stage.

Mamba Block & MLP & MHA + MLP Channel Pruning + SSM Add additional SSM pruning following the previous solution.

The results indicate that pruning Mamba blocks and Transformer blocks alone leads to significant performance degradation. However, more granular pruning strategies show a more favorable trade-off between pruning ratio and performance. Specifically, pruning Mamba blocks, MLP, MHA (single stage), and MLP channels subsequently performs the best. Inspired by the SSM pruning of Mamba-2 in Section [4.3.2,](#page-5-0) we further add SSM pruning to the third strategy, and the results show that removing around 18 SSMs

<span id="page-6-0"></span>

| Pruning Target                              | Ratio  | Additional<br>(Block, Width) Pruned SSMs PPL (↓) | Lambada Lambada HellaS PIQA ARC-e ARC-c WinoG OBQA |      |      |      |      |      |      |      | Avg.           |
|---------------------------------------------|--------|--------------------------------------------------|----------------------------------------------------|------|------|------|------|------|------|------|----------------|
| /                                           | 0%     | 0 / 54                                           | 4.01                                               | 69.7 | 77.0 | 79.8 | 77.5 | 48.5 | 72.1 | 45.8 | 67.2           |
| Mamba Block & Transformer Block             | 10.40% | 0 / 54                                           | 9.18+5.17                                          | 53.5 | 67.3 | 76.3 | 63.5 | 37.8 | 64.3 | 40.6 | 57.6-9.6       |
| Mamba Block & MLP & MHA                     | 10.33% | 0 / 54                                           | 5.01+1.00                                          | 65.6 | 73.6 | 78.5 | 75.3 | 43.8 | 69.3 | 45.2 | 64.5-2.7       |
| Mamba Block & MLP & MHA + MLP Channel       | 10.27% | 0 / 54                                           | 5.45+1.44                                          | 63.4 | 74.9 | 80.1 | 79.0 | 49.7 | 70.9 | 46.0 | 66.3-0.9       |
| Mamba Block & MLP & MHA + MLP Channel + SSM | 10.27% | 18 / 54                                          | 5.18+.1.17                                         | 63.4 | 73.9 | 80.0 | 79.0 | 48.7 | 69.5 | 46.6 | 65.9-1.3       |
| Mamba Block & Transformer Block             | 15.89% | 0 / 54                                           | 10.38+.6.37                                        | 51.4 | 65.6 | 74.0 | 61.7 | 37.7 | 63.5 |      | 39.6 56.2-11.0 |
| Mamba Block & MLP & MHA                     | 15.54% | 0 / 54                                           | 10.64+.6.63                                        | 49.3 | 69.2 | 76.9 | 66.1 | 38.1 | 66.0 | 41.8 | 58.2-9.0       |
| Mamba Block & MLP & MHA + MLP Channel       | 15.48% | 0 / 54                                           | 7.39+.3.38                                         | 57.6 | 70.0 | 78.5 | 74.5 | 43.9 | 67.5 | 43.8 | 62.3-4.9       |
| Mamba Block & MLP & MHA + MLP Channel + SSM | 15.48% | 18 / 54                                          | 7.43+.3.42                                         | 56.5 | 68.9 | 77.9 | 73.4 | 41.8 | 67.7 | 42.8 | 61.3-5.9       |

Table 3: Results of Zamba2-2.7B were achieved by pruning its Mamba-2 and Transformers blocks at multiple granularities, including entire Mamba-2 block, MHA block, MLP block, MLP channel, and SSM module. The remaining tasks represent their respective accuracies. "&" indicates that the pruning targets are considered together in the same pruning step, while "+" signifies the distinction between pruning stages, with pruning occurring sequentially. Bold numbers indicate the best performance under the same level of pruning (excluding Dense).

can maintain accuracy performance while reducing computational overhead. An interesting finding is that pruning SSMs can even lower PPL; for instance, at a 10% pruning ratio, PPL decreases from 5.45 to 5.18, suggesting that some SSM modules are redundant after the second pruning stage. Overall, these findings indicate that multi-granularity pruning methods, particularly those including MLP channels and SSM modules, can effectively reduce the complexity of hybrid Mamba models while maintaining a higher level of performance.

#### 4.3.4 Pruning Mamba Models of Other Sizes

Hymba Table [4](#page-7-0) shows the results of Mamba-Shedder with training-free Hymba Block pruning for Hymba-1.5B-Base. The dense configuration achieves an average accuracy of 63.8, which decreases as more blocks are pruned, dropping to 60.5 when 8 blocks are pruned, indicating a general decline in performance across benchmarks. Further analysis of inference acceleration and recovery tuning experiments for Hymba-1.5B-Base will be discussed in the subsequent sections.

Falcon-Mamba While the previous sections focused on exploring the pruning of Mamba models with sizes around 2.7B or 2.8B, we also investigated the impact of Mamba-Shedder on a largerscale Mamba model, specifically Falcon-Mamba-7B (Table [5\)](#page-7-1). Pruning SSM modules in the Falcon-Mamba-7B model shows better tolerance in terms of perplexity, suggesting that SSM pruning is more effective in maintaining lower perplexity. Regarding average accuracy, pruning entire Mamba blocks is more beneficial.

Additionally, it is important to note that prun-

ing entire Mamba blocks yields more significant computational benefits than SSM pruning, suggesting that while SSM pruning is advantageous for maintaining perplexity, pruning Mamba blocks offers a better trade-off between computational efficiency and accuracy. The choice of pruning strategy should be guided by the specific performance metric of interest and the desired balance between computational efficiency and model accuracy.

None of the above results have undergone finetuning to improve the performance of the pruned models. As in many other works, the drop in the accuracy performance of pruned models can be recovered by fine-tuning, which will be incorporated in Section [4.5.](#page-8-1)

# 4.4 Inference Acceleration

Through the above analysis, we have gained a good understanding and insight into the impact of Mamba-Shedder's structured pruning on model accuracy and perplexity performance. In addition, through structured pruning, Mamba-Shedder achieves an additional speedup to these already highly efficient models. Next, we discuss the impact of inference acceleration. All the following tests were conducted on a single Tesla V100 32GB GPU.

Mamba-1 When removing entire Mamba blocks, as shown in Table [6,](#page-7-2) Mamba-Shedder speeds up the decoding stage up to 1.29x when removing 14 blocks, and 1.13x when removing only 7 blocks, which highlights the potential of Mamba-Shedder to optimize computational efficiency in Mamba models. The user's decision on how aggressively to prune will impact the average accuracy or the

<span id="page-7-0"></span>

| Model              | Method              | Num. of Pruned<br>Hymba Blocks             | HellaS | PIQA | ARC-e | ARC-c | WinoG | Average |
|--------------------|---------------------|--------------------------------------------|--------|------|-------|-------|-------|---------|
|                    | Dense               | 0/32                                       | 53.5   | 77.1 | 76.6  | 45.4  | 66.1  | 63.8    |
| Hymba-1.5B-Base    |                     | $-\frac{1}{6} \frac{1}{32} - \frac{1}{32}$ | 50.5   | 75.8 | 76.0  | 44.9  | 64.1  | 62.3    |
| 11yiiiba-1.5D-Dase | Hymba Block Pruning | 7 / 32                                     | 49.9   | 74.9 | 74.8  | 43.9  | 64.9  | 61.7    |
|                    |                     | 8/32                                       | 49.2   | 74.3 | 74.2  | 43.2  | 61.5  | 60.5    |

Table 4: Results of Mamba-Shedder with *training-free* Hymba block pruning for Hymba-1.5B-Base (Dong et al., 2024). Five commonsense reasoning tasks are used for evaluation: HellaSwag, PIQA, ARC-e, ARC-c, and WinoGrande.

<span id="page-7-1"></span>

| Model           | Method              | Num. of Pruned<br>Mamba Blocks / SSMs | Lambada<br>PPL (↓)  | Lambada | HellaS | PIQA | ARC-e | ARC-c | WinoG | OBQA | Average |
|-----------------|---------------------|---------------------------------------|---------------------|---------|--------|------|-------|-------|-------|------|---------|
|                 | Dense               | 0 / 64                                | 3.15                | 74.3    | 80.3   | 82.0 | 84.4  | 58.9  | 75.1  | 49.0 | 72.0    |
|                 |                     | 5/64                                  | $-\frac{1}{4.01}$   | 69.2    | 78.6   | 81.9 | 82.2  | 54.6  | 72.5  | 47.6 | 69.5    |
|                 | Mamba Block Pruning | 10 / 64                               | 4.97                | 65.1    | 75.0   | 79.5 | 79.7  | 51.5  | 70.2  | 43.8 | 66.4    |
|                 |                     | 15 / 64                               | 5.63                | 62.4    | 71.2   | 77.8 | 76.1  | 49.1  | 70.2  | 41.8 | 64.1    |
| Falcon-Mamba-7B |                     | 20 / 64                               | 39.31               | 31.5    | 65.9   | 74.3 | 72.2  | 42.3  | 65.2  | 38.4 | 55.7    |
|                 |                     | 5/64                                  | - <del>3.47</del> - | 71.6    | 77.3   | 81.2 | 77.8  | 49.2  | 73.2  | 47.2 | 68.2    |
|                 | CCM Danning         | 10 / 64                               | 4.24                | 67.2    | 73.6   | 79.8 | 75.3  | 48.3  | 70.2  | 43.0 | 65.4    |
|                 | SSM Pruning         | 15 / 64                               | 5.37                | 63.3    | 69.6   | 78.2 | 72.4  | 43.4  | 68.8  | 41.8 | 62.5    |
|                 |                     | 20 / 64                               | 14.14               | 46.3    | 63.4   | 74.9 | 60.7  | 36.7  | 65.7  | 37.8 | 55.1    |

Table 5: Results of Mamba-Shedder with *training-free* Mamba block pruning and SSM pruning for Falcon-Mamba-7B.

<span id="page-7-2"></span>Table 6: Inference benchmark results for Mamba-2.8B. The batch size is 1. Number of batches is 10. The prompt length is 512. Number of new tokens is 16.

| Model      | Method           | Num. of Pruned | Inference Speedup |        |  |
|------------|------------------|----------------|-------------------|--------|--|
| Model      | Wiethod          | Mamba Blocks   | Prefill           | Decode |  |
|            | Dense            | 0 / 64         | 1.00×             | 1.00×  |  |
| Mamba-2.8B | Mamba-Shedder    | 7 / 64         | 1.12×             | 1.13×  |  |
|            | ivianioa-Snedder | 14 / 64        | 1.31×             | 1.29×  |  |

<span id="page-7-3"></span>Table 7: Inference benchmark results for Mamba2-2.7B, with test-related hyperparameters consistent with Table 6.

| Model       | Method        | Num. of     | Inference Speedup |        |  |
|-------------|---------------|-------------|-------------------|--------|--|
| Model       | Method        | Pruned SSMs | Prefill           | Decode |  |
|             | Dense         | 0 / 64      | 1.00×             | 1.00×  |  |
| Mamba2-2.7B |               | 16 / 64     | 1.13×             | 1.11×  |  |
| Mamba2-2.7B | Mamba-Shedder | 20 / 64     | $1.16 \times$     | 1.14×  |  |
|             |               | 24 / 64     | $1.20\times$      | 1.18×  |  |

<span id="page-7-4"></span>Table 8: Inference benchmark results for Zamba2-2.7B, with test-related hyperparameters consistent with Table 6. The calculation of *Ratio* includes block pruning (Mamba Block, MHA, and MLP) and width pruning (MLP Channel). Refer to Table 3 for more information.

| Model | Method          | Ratio          | Additional | Inferen                | ce Speedup |
|-------|-----------------|----------------|------------|------------------------|------------|
| Model | Wiethod         | (Block, Width) | Pruned SSM | Prefill                | Decode     |
| Zamba | Dense           | 0%             | 0 / 54     | 1.00×                  | 1.00×      |
| -2.7B | Mamba-Shedder   | 15.48%         | 0 / 54     | 1.16×                  | 1.34×      |
| -2.7B | Maiiba-Sileddei | 15.48%         | 18 / 64    | $\textbf{1.25} \times$ | 1.39×      |

perplexity as observed in Table 1.

Mamba-2 As detailed in Table 7, removing 24 SSM modules (44% of the total number of modules) results in up to a 1.20x speedup in the prefill stage and a 1.18x speedup in the decoding stage during of inference. A more conservative pruning ratio achieves 1.11x speedup when removing 16 SSM modules. Based on previous observations, the impact on performance metrics is minimal (0.4% for accuracy and 0.16 for PPL). These results underscore the effectiveness of SSM pruning in enhancing computational efficiency while barely affecting model performance, making it a viable strategy for optimizing Mamba models.

**Zamba-2** As detailed in Table 8, we observe significant acceleration on inference after multiple granularities pruning of Zamba-2. Specifically, pruning Mamba blocks, MLP, and MHA blocks along with MLP channels results in a 1.34x speedup in the decoding stage. When SSM pruning is included, the speedup increases to 1.39x, indicating that a comprehensive pruning strategy that includes multiple components can significantly enhance inference speed while maximizing the preservation of model performance.

**Hymba** As shown in Table 9, the hymba block pruning of Hymba-1.5B-Base demonstrates notable improvements in inference speed. By removing 7

<span id="page-8-2"></span>Table 9: Inference benchmark results for Hymba-1.5B-Base, where the test-related hyperparameters consistent with Table 6, except that number of new tokens is 256.

| Model           | Method                 | Num. of Pruned<br>Hymba Blocks      |                | ce Speedup<br>Decode |
|-----------------|------------------------|-------------------------------------|----------------|----------------------|
| Hymba-1.5B-Base | Dense<br>Mamba-Shedder | <del>0 / 64</del> <del>7 / 64</del> | 1.00×<br>1.15× | - 1.00×<br>1.24×     |

<span id="page-8-3"></span>

| Model       | Method                | Num. of<br>Pruned SSMs | Lambada<br>PPL (↓) | U        |
|-------------|-----------------------|------------------------|--------------------|----------|
|             | Dense                 | 0 / 64                 | 4.10               | 60.2     |
| Mamba2-2.7B | Mamba-Shedder         | 20 / 64                | 5.89               | 58.6     |
|             | Mamba-Shedder w/ tune | 20 / 64                | 4.44-1.45          | 59.6+1.0 |

Table 10: Results of the compressed Mamba2-2.7B model with recovery tuning (post-training).

out of 64 Hymba blocks, Mamba-Shedder achieves a 1.15x speedup in the prefill stage and a 1.24x speedup in the decoding stage, suggesting that significant computational efficiency gains can be realized even with a relatively modest pruning ratio. The results highlight the potential of Mamba-Shedder to optimize the performance of Hymba models, making them more efficient for real-time applications without substantial sacrifices in model accuracy.

#### <span id="page-8-1"></span>4.5 Recovery Tuning of the Pruned Model

Following most of the work (Ma et al., 2023; Zhong et al., 2024), we performed post-training on the Mamba-Shedder compressed model using the cleaned version of Alpaca. The results summarized in Tables 10, 11, and 12 demonstrate substantial performance gains after just two epochs of recovery tuning (see Appendix for more hyperparameters). For instance, the Mamba-Shedder model obtained by removing Mamba Blocks & MLPs & MHAs + MLP Channels + SSM in Zamba-2 (Table 3), initially exhibits a perplexity of 5.18 and an average accuracy of 65.9 when 18 out of 54 SSMs are pruned. However, after recovery tuning, it achieves a significantly reduced PPL of 4.58 and an improved average accuracy of 67.0, which is almost on par with the Dense model. Similarly, as shown in Table 12, the recovery tuning of the Hymba-1.5B-Base model also yields significant improvements. Initially, the pruned model with 7 out of 32 Hymba blocks removed shows an average accuracy of 61.7. After recovery tuning, the average accuracy increases to 63.7, which is nearly equivalent to the dense model's accuracy of 63.8.

<span id="page-8-4"></span>> **[图片提取文字 (无描述)]:**
> Mamba-1 Mamba-1 Mamba-2 Mamba-2 Calibration 10 3.8 ם 30 **Number of Pruned Mamba Blocks** Number of Pruned SSMs
![](_page_8_Figure_7.jpeg)

Figure 4: Close examination of the impact of removing Mamba blocks or SSMs from the two versions of Mamba models reveals distinct differences in their tolerance levels. Mamba-1 exhibits a higher tolerance for removing its blocks, while Mamba-2 exhibits greater tolerance for removing the SSM subcomponent.

These results indicate that the recovery fine-tuning phase effectively enhances the performance of the pruned model, bringing it closer to the original dense model's performance while maintaining computational efficiency. In summary, recovery tuning is crucial to optimize pruned models, making them more viable for practical applications.

# 4.6 Insights on the Compression Sensitivity of the Variants of Mamba

A research question during our investigation considered, will the improvements in Mamba-2 make it more sensitive to removing its inner structures?

The proponents of Mamba modified the original architecture to restrict the expressivity in Mamba-2 and increase the training efficiency. As illustrated on the left side of Figure 4, our experiments suggest that these changes make Mamba-2 models less robust to removing entire blocks than the previous version of the Mamba block. As soon as we remove blocks with the least importance, Mamba-1 exhibits a more robust behavior. However, Mamba-2 demonstrates a significantly higher tolerance to removing SSMs, maintaining a stable perplexity even as more SSMs are pruned, suggesting that while Mamba-2's architectural improvements have made it more sensitive to the removal of Mamba blocks, they have also enhanced its robustness to SSM pruning.

# <span id="page-8-0"></span>5 Conclusion

Selective structure state space models have become an efficient alternative to Transformer-based models. In this paper, we propose Mamba-Shedder and investigate structured pruning strategies to remove elements from Mamba and hybrid models and reduce model size, accelerating inference. The results demonstrate that selective structured state space architectures have several redundancies that

<span id="page-9-3"></span>

| Model       | Method                 | Ratio<br>(Block, Width) | Additional<br>Pruned SSMs | Lambada<br>PPL (↓) | Average<br>Accuracy |
|-------------|------------------------|-------------------------|---------------------------|--------------------|---------------------|
|             | Dense<br>Mamba-Shedder | -<br>10.27%             | 0 / 54<br>18 / 54         | 4.01<br>5.18       | 67.2<br>65.9        |
| Zamba2-2.7B | Mamba-Shedder w/ tune  | 10.27%                  | 18 / 54                   | 4.58-0.60          | 67.0+1.1            |
|             | Mamba-Shedder          | 15.48%                  | 18 / 54                   | 7.43               | 61.3                |
|             | Mamba-Shedder w/ tune  | 15.48%                  | 18 / 54                   | 5.88-1.55          | 64.4+3.1            |

Table 11: Results of the compressed Mamba2-2.7B and Zamba2-2.7B models with recovery tuning.

<span id="page-9-4"></span>

| Model           | Method                | Num. of Pruned<br>Hymba Blocks | Average<br>Accuracy |
|-----------------|-----------------------|--------------------------------|---------------------|
|                 | Dense                 | 0 / 32                         | 63.8                |
| Hymba-1.5B-Base | Mamba-Shedder         | 7 / 32                         | 61.7                |
|                 | Mamba-Shedder w/ tune | 7 / 32                         | 63.7+2.0            |

Table 12: Results of the compressed Hymba-1.5B-Base model with recovery tuning. *Average Accuracy* is calculated over HellaSwag, PIQA, ARC-e, ARC-c, and WinoGrande tasks (Table [4\)](#page-7-0).

can be removed without significantly affecting the model's performance.

# Limitations

Despite their outstanding results, large sequence models are still under investigation to better understand their capabilities and limitations. Mamba-Shedder is, to the best of our knowledge, the first work to investigate the removal of structures in Mamba-based models, including hybrids with Transformer blocks. Our goal is to motivate the research community to better understand this class of models to identify opportunities for future improvements in the model architecture and applicable compression techniques. The results indicate that these models contain redundant elements that might be removed to improve their efficiency. However, future work must explore and attempt to better understand the trade-offs between efficiency and accuracy when removing these models' components. Even more research questions can be entertained when considering Transformer blocks and hybrid models, as in the case of Zamba. For instance, there is much to understand about the right mix of the SSM- and Transformer-based elements.

# Ethics Statement

Due to the well-known flaws in modern sequence models, e.g., hallucinations, many guard rails must be in place when considering deploying them in production. Our research focuses on improving the

efficiency of these models in existing downstream tasks and datasets. However, further experimentation and analysis are needed when considering deploying these compressed models in environments where their output might affect people's well-being.

# Acknowledgments

We are grateful to Michael Beale from Intel Labs, who helped us set up the infrastructure for sharing our models during the review stage and the final release and guided us through open-sourcing our compressed models. We also thank the anonymous reviewers for their insightful suggestions, which helped us improve the paper.

# References

<span id="page-9-0"></span>A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lucic, and C. Schmid. 2021. [Vivit: A video vision trans](https://doi.org/10.1109/ICCV48922.2021.00676)[former.](https://doi.org/10.1109/ICCV48922.2021.00676) In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 6816–6826, Los Alamitos, CA, USA. IEEE Computer Society.

<span id="page-9-1"></span>Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. [SliceGPT: Compress large language models](https://openreview.net/forum?id=vXxardq6db) [by deleting rows and columns.](https://openreview.net/forum?id=vXxardq6db) In *The Twelfth International Conference on Learning Representations*.

<span id="page-9-6"></span>Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. *arXiv:2004.05150*.

<span id="page-9-2"></span>Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial Intelligence*.

<span id="page-9-5"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.

- [Language models are few-shot learners.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc.
- <span id="page-10-15"></span>Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. 2021. [Rethinking attention with performers.](https://openreview.net/forum?id=Ua6zuk0WRH) In *International Conference on Learning Representations*.
- <span id="page-10-11"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. [Think you have solved question an](https://api.semanticscholar.org/CorpusID:3922816)[swering? try arc, the ai2 reasoning challenge.](https://api.semanticscholar.org/CorpusID:3922816) *ArXiv*, abs/1803.05457.
- <span id="page-10-13"></span>Gonçalo M. Correia, Vlad Niculae, and André F. T. Martins. 2019. [Adaptively sparse transformers.](https://doi.org/10.18653/v1/D19-1223) In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2174– 2184, Hong Kong, China. Association for Computational Linguistics.
- <span id="page-10-14"></span>Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V. Le. 2020. Funnel-transformer: filtering out sequential redundancy for efficient language processing. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NIPS '20, Red Hook, NY, USA. Curran Associates Inc.
- <span id="page-10-4"></span>Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In *International Conference on Machine Learning (ICML)*.
- <span id="page-10-12"></span>Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Association for Computational Linguistics (ACL)*.
- <span id="page-10-8"></span>Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, et al. 2024. Hymba: A hybrid-head architecture for small language models. *arXiv preprint arXiv:2411.13676*.
- <span id="page-10-0"></span>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. [An image](https://arxiv.org/abs/2010.11929) [is worth 16x16 words: Transformers for image](https://arxiv.org/abs/2010.11929) [recognition at scale.](https://arxiv.org/abs/2010.11929) *Preprint*, arXiv:2010.11929.
- <span id="page-10-5"></span>Stefan Elfwing, Eiji Uchibe, and Kenji Doya. 2018. [Sigmoid-weighted linear units for neural network](https://doi.org/10.1016/j.neunet.2017.12.012) [function approximation in reinforcement learning.](https://doi.org/10.1016/j.neunet.2017.12.012) *Neural Networks*, 107:3–11. Special issue on deep reinforcement learning.

- <span id="page-10-19"></span>Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: Accurate post-training compression for generative pretrained transformers. *arXiv preprint arXiv:2210.17323*.
- <span id="page-10-17"></span>Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. 2023. Hungry Hungry Hippos: Towards language modeling with state space models. In *International Conference on Learning Representations*.
- <span id="page-10-9"></span>Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. [The pile: An](https://arxiv.org/abs/2101.00027) [800gb dataset of diverse text for language modeling.](https://arxiv.org/abs/2101.00027) *Preprint*, arXiv:2101.00027.
- <span id="page-10-10"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. [A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836) [evaluation.](https://doi.org/10.5281/zenodo.10256836)
- <span id="page-10-6"></span>Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. 2024. [Zamba: A compact 7b](https://arxiv.org/abs/2405.16712) [ssm hybrid model.](https://arxiv.org/abs/2405.16712) *Preprint*, arXiv:2405.16712.
- <span id="page-10-1"></span>Yuan Gong, Yu-An Chung, and James Glass. 2021. [AST: Audio Spectrogram Transformer.](https://doi.org/10.21437/Interspeech.2021-698) In *Proc. Interspeech 2021*, pages 571–575.
- <span id="page-10-3"></span>Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.
- <span id="page-10-2"></span>Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In *The International Conference on Learning Representations (ICLR)*.
- <span id="page-10-16"></span>Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. 2024. Combining recurrent, convolutional, and continuous-time models with linear state-space layers. In *Proceedings of the 35th International Conference on Neural Information Processing Systems*, NIPS '21, Red Hook, NY, USA. Curran Associates Inc.
- <span id="page-10-18"></span>Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. 2021. Sparsity in deep learning: pruning and growth for efficient inference and training in neural networks. *J. Mach. Learn. Res.*, 22(1).
- <span id="page-10-7"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.

- <span id="page-11-15"></span>Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: fast autoregressive transformers with linear attention. In *Proceedings of the 37th International Conference on Machine Learning*, ICML'20. JMLR.org.
- <span id="page-11-20"></span>François Lagunas, Ella Charlaix, Victor Sanh, and Alexander Rush. 2021. [Block pruning for faster trans](https://doi.org/10.18653/v1/2021.emnlp-main.829)[formers.](https://doi.org/10.18653/v1/2021.emnlp-main.829) In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 10619–10629, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- <span id="page-11-2"></span>Yann LeCun, John Denker, and Sara Solla. 1989. [Op](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)[timal brain damage.](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) In *Advances in Neural Information Processing Systems*, volume 2. Morgan-Kaufmann.
- <span id="page-11-11"></span>Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. [LLM-pruner: On the structural pruning of large lan](https://openreview.net/forum?id=J8Ajf9WfXP)[guage models.](https://openreview.net/forum?id=J8Ajf9WfXP) In *Thirty-seventh Conference on Neural Information Processing Systems*.
- <span id="page-11-3"></span>Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. [Shortgpt: Layers in large language mod](https://arxiv.org/abs/2403.03853)[els are more redundant than you expect.](https://arxiv.org/abs/2403.03853) *Preprint*, arXiv:2403.03853.
- <span id="page-11-10"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. [Can a suit of armor conduct elec](https://api.semanticscholar.org/CorpusID:52183757)[tricity? a new dataset for open book question answer](https://api.semanticscholar.org/CorpusID:52183757)[ing.](https://api.semanticscholar.org/CorpusID:52183757) In *Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-11-19"></span>J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain. 2024. [Shears: Unstructured sparsity with neural low-rank](https://doi.org/10.18653/v1/2024.naacl-industry.34) [adapter search.](https://doi.org/10.18653/v1/2024.naacl-industry.34) In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track)*, pages 395–405, Mexico City, Mexico. Association for Computational Linguistics.
- <span id="page-11-5"></span>J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain. 2025. [Multipruner: Balanced structure removal in founda](https://arxiv.org/abs/2501.09949)[tion models.](https://arxiv.org/abs/2501.09949) *Preprint*, arXiv:2501.09949.
- <span id="page-11-7"></span>Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. [The LAMBADA dataset: Word](https://doi.org/10.18653/v1/P16-1144) [prediction requiring a broad discourse context.](https://doi.org/10.18653/v1/P16-1144) In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1525–1534, Berlin, Germany. Association for Computational Linguistics.
- <span id="page-11-1"></span>Niki J. Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. 2018. [Image transformer.](http://proceedings.mlr.press/v80/parmar18a.html) In *International Conference on Machine Learning (ICML)*.

- <span id="page-11-13"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning (ICML)*.
- <span id="page-11-9"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. [Winogrande: An adver](https://doi.org/10.1145/3474381)[sarial winograd schema challenge at scale.](https://doi.org/10.1145/3474381) *Commun. ACM*, 64(9):99–106.
- <span id="page-11-17"></span>Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2023. A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*.
- <span id="page-11-6"></span>Yury Tokpanov, Beren Millidge, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, and Quentin Anthony. 2024. Zyda: A 1.3 t dataset for open language modeling. *arXiv preprint arXiv:2406.01981*.
- <span id="page-11-12"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-11-0"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [you need.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) In *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc.
- <span id="page-11-18"></span>Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, and Ping Luo. 2024. [Besa: Pruning large language](https://arxiv.org/abs/2402.16880) [models with blockwise parameter-efficient sparsity](https://arxiv.org/abs/2402.16880) [allocation.](https://arxiv.org/abs/2402.16880) *Preprint*, arXiv:2402.16880.
- <span id="page-11-8"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*.
- <span id="page-11-14"></span>Li Zhang, Jiachen Lu, Sixia Zheng, Xinxuan Zhao, Xiatian Zhu, Yanwei Fu, Xiang Tao, and Jianfeng Feng. 2023. Vision transformers: From semantic segmentation to dense prediction. *arXiv*.
- <span id="page-11-16"></span>Lin Zheng, Chong Wang, and Lingpeng Kong. 2022. Linear complexity randomized self-attention mechanism. In *International Conference on Machine Learning*, pages 27011–27041. PMLR.
- <span id="page-11-4"></span>Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, and Liangzhi Li. 2024. [Blockpruner: Fine](https://arxiv.org/abs/2406.10594)[grained pruning for large language models.](https://arxiv.org/abs/2406.10594) *Preprint*, arXiv:2406.10594.

<span id="page-12-0"></span>Jingwei Zuo, Maksim Velikanov, Dhia Eddine Rhaiem, Ilyas Chahed, Younes Belkada, Guillaume Kunsch, and Hakim Hacid. 2024. Falcon mamba: The first competitive attention-free 7b language model. *arXiv preprint arXiv:2410.05355*.

# Supplementary Material

# A Related Work

Transformers [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0) and its variants are the primary building block of successful deep learning architectures, e.g., Llama [\(Touvron](#page-11-12) [et al.,](#page-11-12) [2023\)](#page-11-12) and GPT [\(Brown et al.,](#page-9-5) [2020\)](#page-9-5), that have revolutionized Natural Language Processing (NLP) [\(Devlin et al.,](#page-10-12) [2019;](#page-10-12) [Gao et al.,](#page-10-10) [2023\)](#page-10-10), Computer Vision (CV) [\(Parmar et al.,](#page-11-1) [2018;](#page-11-1) [Radford](#page-11-13) [et al.,](#page-11-13) [2021;](#page-11-13) [Zhang et al.,](#page-11-14) [2023\)](#page-11-14), and many other domains. Due to the Transformer's popularity, researchers have proposed variants to improve their computational and memory efficiency further and tackle issues like their quadratic complexity in sequence length during training [\(Correia et al.,](#page-10-13) [2019;](#page-10-13) [Beltagy et al.,](#page-9-6) [2020;](#page-9-6) [Dai et al.,](#page-10-14) [2020;](#page-10-14) [Choroman](#page-10-15)[ski et al.,](#page-10-15) [2021;](#page-10-15) [Katharopoulos et al.,](#page-11-15) [2020;](#page-11-15) [Zheng](#page-11-16) [et al.,](#page-11-16) [2022\)](#page-11-16).

A parallel research effort investigates alternatives to Transformers in the form of *structured state space models* (SSMs) that can power the next generation of sequence models. The initial proposals of structured SSMs were linear time-invariant, e.g., LSSL [\(Gu et al.,](#page-10-16) [2024\)](#page-10-16), S4 [\(Gu et al.,](#page-10-2) [2022\)](#page-10-2), H3 [\(Fu et al.,](#page-10-17) [2023\)](#page-10-17). Recent improvements to the state space model formulation have resulted in the proposal of time-varying selective SSMs, e.g., Mamba [\(Gu and Dao,](#page-10-3) [2023;](#page-10-3) [Dao and Gu,](#page-10-4) [2024\)](#page-10-4).

To our knowledge, Mamba-Shedder is the first study on pruning selective structured state space models (Mamba) and their hybrids. On the other hand, many works have proposed pruning techniques for Transformer-based models [\(Hoefler](#page-10-18) [et al.,](#page-10-18) [2021\)](#page-10-18). Several of these works focus on *unstructured* pruning [\(Sun et al.,](#page-11-17) [2023;](#page-11-17) [Xu et al.,](#page-11-18) [2024;](#page-11-18) [Frantar et al.,](#page-10-19) [2022\)](#page-10-19), which can achieve higher sparsity levels. However, it requires highly optimized runtimes to realize the benefits of sparsity. Sophisticated solutions have been proposed to fine-tune sparse models and recover any accuracy drop from the pruning stage [\(Muñoz et al.,](#page-11-19) [2024\)](#page-11-19). Recently, *training-free* approaches have been proposed for *structured* pruning of Transformers. These approaches cannot achieve high sparsity levels as the *unstructured* pruning approaches. However, they are very convenient because their compressed models do not require specialized runtimes and exhibit beneficial inference acceleration. In this line of research, LLMPruner [\(Ma et al.,](#page-11-11) [2023\)](#page-11-11), ShortGPT [\(Men et al.,](#page-11-3) [2024\)](#page-11-3), BlockPruner [\(Lagunas et al.,](#page-11-20) [2021\)](#page-11-20), SliceGPT [\(Ashkboos et al.,](#page-9-1)

<span id="page-13-0"></span>

| Hyper-parameter                       | Value            |
|---------------------------------------|------------------|
| Pruning Stage:                        |                  |
| Calibration Dataset                   | tatsu-lab/alpaca |
| Importance Metric                     | Perplexity (PPL) |
| Number of Calibration Samples         | 256              |
| MLP Channel Group Size (Zamba2)       | 1024             |
| Steps of MLP Channel Pruning (Zamba2) | 20               |

Table 13: Hyper-parameters used in the experiments.

[2024\)](#page-9-1), and MultiPruner [\(Muñoz et al.,](#page-11-5) [2025\)](#page-11-5) have demonstrated efficient methods for Transformer pruning. BlockPruner improved over many previous approaches by proposing a global metric that can be used to determine the importance of a selected network structure. MultiPruner extended this approach to pruning the width dimension, as well. Mamba-Shedder builds on these works and the rest of the extensive literature on *structured* block pruning to explore opportunities for removing redundancies in models with Mamba blocks.

# B Hyperparameters

Table [13](#page-13-0) offers a detailed summary of the hyperparameters employed in our experiments, promoting both reproducibility and clarity.