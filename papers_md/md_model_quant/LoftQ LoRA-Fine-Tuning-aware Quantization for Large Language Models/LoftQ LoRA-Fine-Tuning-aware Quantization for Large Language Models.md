# LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

Yixiao Li∗∗, Yifan Yu∗∗, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao <sup>∗</sup>

November 29, 2023

#### Abstract

Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning [\(Dettmers et al.,](#page-14-0) [2023\)](#page-14-0). In this work we focus on the scenario where quantization and LoRA fine-tuning are applied together on a pretrained model. In such cases it is common to observe a consistent gap in the performance on downstream tasks between full fine-tuning and quantization plus LoRA fine-tuning approach. In response, we propose LoftQ (LoRA-Fine-Tuning-aware Quantization), a novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning. Such an initialization alleviates the discrepancy between the quantized and full-precision model and significantly improves generalization in downstream tasks. We evaluate our method on natural language understanding, question answering, summarization, and natural language generation tasks. Experiments show that our method is highly effective and outperforms existing quantization methods, especially in the challenging 2-bit and 2/4-bit mixed precision regimes. The code is available on <https://github.com/yxli2123/LoftQ>.

# 1 Introduction

The advent of Pre-trained Language Models (PLMs) has marked a transformative shift in the field of Natural Language Processing (NLP), offering versatile solutions across various applications [\(He et al.,](#page-14-1) [2021b;](#page-14-1) [Lewis et al.,](#page-14-2) [2019;](#page-14-2) [Touvron et al.,](#page-16-0) [2023\)](#page-16-0). They have showcased unparalleled proficiency in executing a variety of language tasks, including Natural Language Understanding (NLU) and Natural Language Generation (NLG). These models typically have millions or even billions of parameters, necessitating substantial computational and memory requirements. However, the extensive computational and memory demands of these models pose significant challenges,

<sup>∗</sup>Li, Yu, Liang and Zhao are affiliated with Georgia Tech. He, Karampatziakisand and Chen are affiliated with Microsoft Azure. Correspondence to <yixiaoli@gatech.edu>, <yyu429@gatech.edu> and <tourzhao@gatech.edu>.

<sup>\*\*</sup>Equal contributions

especially in real-world deployments where resources are often constrained and need to be shared among many users.

To mitigate the extensive storage requirements of pre-trained models, quantization serves as a pivotal compression technique [\(Zafrir et al.,](#page-16-1) [2019;](#page-16-1) [Shen et al.,](#page-15-0) [2020;](#page-15-0) [Bai et al.,](#page-13-0) [2022;](#page-13-0) [Dettmers et al.,](#page-14-3) [2022\)](#page-14-3), converting high-precision numerical values into a discrete set of values. Typically, model parameters, originally stored in a 16-bit float format, are transformed into a 4-bit integer format through quantization, resulting in a substantial 75% reduction in storage overhead. Additionally, to facilitate the adaptation of quantized pre-trained models to downstream tasks efficiently, Low-Rank Adaptation (LoRA) is a viable approach [\(Hu et al.,](#page-14-4) [2021\)](#page-14-4). This technique is a parameter-efficient fine-tuning method traditionally applied to high-precision pre-trained models. It is based on the hypothesis that the differences between fully fine-tuned weights and pre-trained weights exhibit low-rank properties. This allows these differences to be represented using low-rank matrices. As a result, the original pre-trained weights remain unaltered, with adaptations confined solely to these low-rank matrices, enabling effective task adaptation.

When quantizing pre-trained models, practitioners often concentrate primarily on the quantization technique, inadvertently neglecting the importance of subsequent LoRA fine-tuning [\(Dettmers](#page-14-0) [et al.,](#page-14-0) [2023;](#page-14-0) [Diao et al.,](#page-14-5) [2023\)](#page-14-5). For example, QLoRA inherits the fixup initialization [\(Zhang et al.,](#page-16-2) [2019\)](#page-16-2) used in LoRA, which [\(Dettmers et al.,](#page-14-0) [2023\)](#page-14-0) attaches zero initialized low-rank adapters (see Section [2.3\)](#page-4-0) to the quantized pre-trained model. The inevitable discrepancy introduced by quantization during the approximation of the original high-precision numbers, a scenario particularly pronounced in low-bit situations such as the 2-bit regime, can adversely impact the initialization of LoRA fine-tuning. As illustrated in Figure [1a,](#page-2-0) the quantized pre-trained model obtained by QLoRA exhibits severe degradation below the 3-bit level. This deviation in initialization often results in an inferior fine-tuning performance. As illustrated in Figure [1b,](#page-2-0) the fine-tuning performance drops as the quantization bit decreases when applying QLoRA. Moreover, it is noteworthy that QLoRA fails below the 3-bit level.

In this paper, we introduce a novel quantization framework, called LoRA-Fine-Tuning-aware Quantization (LoftQ). It is designed specifically for pre-trained models that require quantization and LoRA fine-tuning. This framework actively integrates low-rank approximation, working in tandem with quantization to jointly approximate the original high-precision pre-trained weights. This synergy significantly enhances alignment with the original pre-trained weights as illustrated in Figure [2.](#page-2-1) Consequently, our method provides an advantageous initialization point for subsequent LoRA fine-tuning, leading to improvements in downstream tasks.

We evaluate our quantization framework by conducting extensive experiments on downstream tasks, such as NLU, question answering, summarization, and NLG. Experiments show that LoftQ consistently outperforms QLoRA across all precision levels. For instance, with 4-bit quantization, we achieve a 1.1 and 0.8 gain in Rouge-1 for XSum [\(Narayan et al.,](#page-15-1) [2018\)](#page-15-1) and CNN/DailyMail [\(Hermann et al.,](#page-14-6) [2015\)](#page-14-6), respectively. LoftQ excels particularly in low-bit scenarios and works

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

- (a) Pre-trained LLAMA-2-13b on WikiText-2
- (b) Fine-tuned LLAMA-2-13b on WikiText-2

Figure 1: QLoRA performance with different bits. **Left:** QLoRA initialization of LLAMA-2-13b on WikiText-2. **Right:** Apply QLoRA to LLAMA-2-13b on WikiText-2 language modeling task. Smaller perplexity indicates better performance.

effectively with different quantization methods. For example, we achieve over an 8% gain on MNLI (Wang et al., 2019) and more than 10% on SQuADv1.1 (Rajpurkar et al., 2016) with both 2-bit NormalFloat and the 2-bit uniform quantization. We have not seen our approach performs worse than QLoRA.

<span id="page-2-1"></span>![](_page_2_Figure_5.jpeg)

- (a) Spectral norm of the initialization difference
- (b) Frobenius norm of the initialization difference

Figure 2: Initialization discrepancy between the LoRA initialization and the original pre-trained weight matrix, described by the spectral norm and Frobenius norm of the difference. The weight matrix in the above figures is randomly selected in BART-large. The initialization is obtained by QLoRA and LoftQ, with Uniform and NormalFloat quantization methods applied at both 2-bit and 4-bit levels. LoftQ successfully mitigates the discrepancy, especially at the 2-bit level.

# 2 Background

### 2.1 Transformer Models

A transformer model contains a sequence of layers, where each layer consists of two sub-layers: a multi-head self-attention (MHA) and a fully connected feed forward network (FFN) [\(Vaswani et al.,](#page-16-4) [2017\)](#page-16-4). Given the input *X* ∈ R*n*×*<sup>d</sup>* , where *n* is the sequence length and *d* is the hidden dimension of the model, MHA computes the *h* attention heads in parallel:

$$\label{eq:MHA} \mathbf{MHA}(X) = \mathbf{Concat}(\mathbf{head}_1,...,\mathbf{head}_h)W_o,$$
 where  $\mathbf{head}_i = \mathbf{Softmax}(XW_{q_i}(XW_{k_i})^\top/\sqrt{d_h})XW_{v_i}$  for  $i=1,...,h,$ 

where *Wq<sup>i</sup> ,Wk<sup>i</sup> ,Wv<sup>i</sup>* <sup>∈</sup> <sup>R</sup>*d*×*d<sup>h</sup>* are query, key, and value matrices, *<sup>W</sup><sup>o</sup>* <sup>∈</sup> <sup>R</sup>*d*×*<sup>d</sup>* is the output matrix, and *d<sup>h</sup>* = *d/h*. FFN comprises two linear transformations and an activation function, and is defined as FFN(*X*) = *σ*(*XWf*<sup>1</sup> + *b*1)*Wf*<sup>2</sup> + *b*2*,* where *Wf*<sup>1</sup> <sup>∈</sup> <sup>R</sup>*d*×*d<sup>m</sup>* , *<sup>W</sup>f*<sup>2</sup> ∈ R*dm*×*<sup>d</sup>* , and *σ*(·) is the activation function. A residual connection is used and followed by layer normalization.

### 2.2 Quantization

Quantization. Given a high-precision number, e.g., such as 32-bit floating point number, *X* HP ∈ R, *N*-bit quantization encodes it to an integer *X* INT ∈ {0*,*1*,...,*2 *<sup>N</sup>* − 1}. This process can be expressed as

<span id="page-3-0"></span>
$$X^{\text{INT}} = \text{round}\left((2^N - 1)F\left(X^{\text{HP}}\right)\right),\tag{1}$$

where *F*(·): R 7→ [0*,*1] is a normalization function. Uniform quantization assumes *F*(*X*) = (*X* − *X*min)*/*(*X*max − *X*min). [Dettmers et al.](#page-14-0) [\(2023\)](#page-14-0) proposes 4-bit NormalFloat Quantization (NF4). It assumes *X* ∼ N (0*, σ*<sup>2</sup> ) and hence *F*(*X*) = Φ(*X/σ*), where Φ(·) is the cumulative distribution function of the standard normal distribution.

Dequantization. A lookup table T , where

$$\mathcal{T}[i] = F^{-1}\left(\frac{i}{2^N - 1}\right), i = 0, 1, ..., 2^N - 1,$$
(2)

is used to decode the integer *X* INT to its simulated high-precision counterpart *X* <sup>D</sup> ∈ R. Therefore, the dequantization can be expressed as

<span id="page-3-2"></span><span id="page-3-1"></span>
$$X^{D} = \mathcal{T}[X^{\text{INT}}]. \tag{3}$$

Simulated Quantization for Matrices. While it is possible to perform multiplication directly between quantized representations, it is common to apply simulated quantization for matrices [\(Bai](#page-13-1) [et al.,](#page-13-1) [2020;](#page-13-1) [Shen et al.,](#page-15-0) [2020\)](#page-15-0). There, quantized weight matrices are stored as encoded integers in memory, and are temporarily dequantized to simulated high-precision matrices by the lookup table when engaged in multiplication operations. In simulated quantization, it is only necessary to analyze the map from a high-precision matrix to a simulated high-precision matrix. We denote this end-to-end process by *<sup>q</sup><sup>N</sup>* (·): <sup>R</sup>*m*×*<sup>n</sup>* 7→ R *m*×*n N* , where R*<sup>N</sup>* : {T [*i*] ∈ R|0 ≤ *i <* 2 *<sup>N</sup>* }.

### <span id="page-4-0"></span>2.3 Low-Rank Adaptation

LoRA (Hu et al., 2021) updates two small weight matrices A and B that are attached to a frozen pre-trained weight matrix W. Hence, a linear transformation, Y = XW, is reformulated as

<span id="page-4-1"></span>
$$Y = XW + XAB^{\top},\tag{4}$$

where  $X \in \mathbb{R}^{n \times d_1}$ ,  $W \in \mathbb{R}^{d_1 \times d_2}$ ,  $A \in \mathbb{R}^{d_1 \times r}$ ,  $B \in \mathbb{R}^{d_2 \times r}$ , and  $r \ll \min\{d_1, d_2\}$ . Initially,

$$A \sim \mathcal{N}(0, \sigma^2), B = 0, \tag{5}$$

so as to align to the pre-trained weights. During the fine-tuning, W is fixed while A and B are updated by some SGD-type optimization method.

It is worth noting that if low-rank adapters A and B are attached to a quantized backbone  $Q = q_N(W)$  and are initialized by (5), the starting weight  $Q + AB^{\top}$  is no longer equal to the pretrained weight W due to the discrepancy introduced by the quantization.

### 3 Method

We propose **Lo**RA-Fine-Tuning-aware **Q**uantization (LoftQ), a quantization framework for LLMs. It alternatively applies quantization and low-rank approximation to approximate original pretrained weights. This quantization framework provides a promising initialization for LoRA fine-tuning, which alleviates the quantization discrepancy in QLoRA and improves generalization in downstream tasks significantly.

#### 3.1 LoRA-Aware Quantization

We use an N-bit quantized weight  $Q \in \mathbb{R}_N^{d_1 \times d_2}$  and low-rank approximations  $A \in \mathbb{R}^{d_1 \times r}$ ,  $B \in \mathbb{R}^{d_2 \times r}$  to approximate the original high-precision pre-trained weight  $W \in \mathbb{R}^{d_1 \times d_2}$  as the initialization of LoRA fine-tuning. Specifically, before fine-tuning, we initialize the network by minimizing the following objective:

<span id="page-4-2"></span>
$$\min_{Q,A,B} \left\| W - Q - AB^{\top} \right\|_{F},\tag{6}$$

where  $\|\cdot\|_F$  denotes the Frobenious norm. This objective in (6) takes LoRA fine-tuning into consideration by jointly optimizing the initial values of the quantized backbone Q and low-rank adapters A, B. Contrarily, practitioners typically convert the pre-trained weight W into a quantized weight Q outright, neglecting the subsequent LoRA fine-tuning process. This oversight leads to notable performance degradation in downstream tasks arising from the quantization discrepancy.

### 3.2 Alternating Optimization

We solve the minimization problem in (6) by alternating between quantization and singular value decomposition (SVD). To begin with, we set  $A_0$ , and  $B_0$  equal to 0.

**Quantization**. At the *t*-th step, we quantize the difference between the original pre-trained weight W and the low-rank approximation  $A_{t-1}B_{t-1}^{\top}$  from the last step to obtain the quantized weight  $Q_t$  by

$$Q_t = q_N(W - A_{t-1}B_{t-1}^{\mathsf{T}}), \tag{7}$$

where  $q_N(\cdot)$  maps a high-precision weight matrix to a quantized matrix.

We remark that our algorithm is compatible with different quantization functions  $q_N(\cdot)$ . We apply NF4 and the uniform quantization in Section 4 as examples. We also remark that  $Q_t$  is not an exact solution of the minimization in (6), given the fixed  $A_{t-1}B_{t-1}^{\top}$ , but it is an efficient approximation.

**SVD**. After obtaining the *t*-th quantized weight  $Q_t$ , SVD is applied to the residual of the quantization denoted by  $R_t = W - Q_t$  by

$$R_{t} = \sum_{i=1}^{d} \sigma_{t,i} u_{t,i} v_{t,i}^{\top}, \tag{8}$$

where  $d = \min\{d_1, d_2\}$ ,  $\sigma_{t,1} \ge \sigma_{t,2} \ge ... \ge \sigma_{t,d}$  are the singular values of  $R_t$ ,  $u_{t,i}$ 's and  $v_{t,i}$ 's are the associated left and right singular vectors of  $R_t$ . We then obtain a rank-r approximation of  $R_t$  by  $A_t B_t^{\mathsf{T}}$ , where

<span id="page-5-0"></span>
$$A_{t} = [\sqrt{\sigma_{t,1}} u_{t,1}, ..., \sqrt{\sigma_{t,r}} u_{t,r}],$$

$$B_{t} = [\sqrt{\sigma_{t,1}} v_{t,1}, ..., \sqrt{\sigma_{t,r}} v_{t,r}].$$
(9)

We summarize our method in Algorithm 1. It is worth noting that T = 1 is a special case where  $Q_1$  is the exact quantized weight obtained by QLoRA, and low-rank approximations  $A_1$ ,  $B_1$  are obtained by the SVD of the quantization residual  $W - Q_1$ . T = 1 is sufficient to mitigate the quantization discrepancy, and alternating optimization helps to find a closer initialization to the pre-trained weight W, which further improves the performance (see Section 3).

We remark that the computational cost of LoftQ is negligible because it is applied to individual weight matrices and therefore can be executed in parallel. We also remark one can apply LoftQ only once to a pre-trained model and reuse the initialization obtained by LoftQ for different downstream tasks.

#### 3.3 Applying to LoRA Fine-tuning

We store the  $Q_T \in \mathbb{R}_N^{d_1 \times d_2}$  obtained by LoftQ using an integer matrix M by (1) and a lookup table T by (2). We initialize the backbone with the integer matrix M and initialize the low-rank adapters with  $A_T, B_T$  obtained by LoftQ.

### <span id="page-6-1"></span>Algorithm 1 LoftQ

```
input Pre-trained weight W , target rank r, N-bit quantization function qN (·), alternating step T
```

- 1: Initialize *A*<sup>0</sup> ← 0*,B*<sup>0</sup> ← 0
- 2: for t = 1 to *T* do
- 3: Obtain quantized weight *Q<sup>t</sup>* ← *q<sup>N</sup>* (*W* − *At*−1*B* ⊤ *t*−1 )
- 4: Obtain low-rank approximation *A<sup>t</sup> ,B<sup>t</sup>* ← SVD(*W* − *Q<sup>t</sup>* ) by [\(9\)](#page-5-0)
- 5: end for

output *Q<sup>T</sup> ,A<sup>T</sup> ,B<sup>T</sup>*

During LoRA fine-tuning, we freeze the integer weight *M* and optimize the low-rank adapters with an efficient optimization algorithm, e.g., AdamW [\(Loshchilov and Hutter,](#page-15-3) [2017\)](#page-15-3). In forward propagation, the integer weight *M* is temporarily dequantized to the simulated high-precision weight *Q<sup>T</sup>* by its lookup table, as described in [\(3\)](#page-3-2). In back propagation, gradients and optimizer state are only related to low-rank adapters *A,B*, which reduces considerable training cost.

# <span id="page-6-0"></span>4 Experiments

We evaluate our method on NLU and NLG tasks. We apply LoftQ for quantizing DeBERTaV3-base [\(He et al.,](#page-14-1) [2021b\)](#page-14-1), BART-large [\(Lewis et al.,](#page-14-2) [2019\)](#page-14-2), and LLAMA-2 series [\(Touvron et al.,](#page-16-0) [2023\)](#page-16-0). Implementation Details. Following the prior works of LoRA variants [\(Zhang et al.,](#page-16-5) [2023;](#page-16-5) [He](#page-14-7) [et al.,](#page-14-7) [2021a\)](#page-14-7), we freeze all the backbone weight matrices and add low-rank adapters to weight matrices in MHA and FFN of all layers. We quantize the weight matrices that are attached by low-rank adapters. All the quantized models and adapters used in this paper are available on <https://huggingface.co/LoftQ>. Our implementation is based on publicly available *Huggingface Transformers* code-base [\(Paszke et al.,](#page-15-4) [2019\)](#page-15-4). All the experiments are conducted on NVIDIA A100 GPUs.

Quantization Methods. We apply two quantization methods to demonstrate LoftQ is compatible with different quantization functions:

- *Uniform quantization* is a classic quantization method. It uniformly divides a continuous interval into 2*<sup>N</sup>* categories and stores a local maximum absolute value for dequantization.
- *NF4* and its 2-bit variant *NF2* are quantization methods used in QLoRA [\(Dettmers et al.,](#page-14-0) [2023\)](#page-14-0). They assume that the high-precision values are drawn from a Gaussian distribution and map these values to discrete slots that have equal probability.

We perform 2-bit and 4-bit quantization on all models, achieving compression ratios of 25-30% and 15-20% at the 4-bit and 2-bit levels, respectively. The compression ratios and trainable parameter ratios for all models are detailed in the Appendix [A.](#page-17-0)

Baselines. We compare LoftQ with the following baseline methods:

- *Full fine-tuning* is the most common approach for adapting a pre-trained model to downstream tasks. The model is initialized with pre-trained weights and all parameters are updated through an SGD-type optimization method.
- *Full precision LoRA (LoRA)* is a lightweight method for task adaptation, where it stores the backbone using 16-bit numbers and optimizes the low-rank adaptors only. The adaptors are applied to the same matrices as in LoftQ.
- *QLoRA* is similar to *LoRA* except the backbone is quantized into low-bit regime. The low-rank adapters are initialized using [\(5\)](#page-4-1) and are applied to the same matrices as in LoftQ.

### 4.1 Encoder-only Model: DeBERTaV3

Models and Datasets. We quantize the DeBERTaV3-base [\(He et al.,](#page-14-1) [2021b\)](#page-14-1) with LoftQ, then finetune and evaluate the model on the General Language Understanding Evaluation (GLUE) benchmark [\(Wang et al.,](#page-16-3) [2019\)](#page-16-3), SQuADv1.1 [\(Rajpurkar et al.,](#page-15-2) [2016\)](#page-15-2), and ANLI [\(Nie et al.,](#page-15-5) [2019\)](#page-15-5). The specific tasks of GLUE are given in Appendix [C.](#page-18-0) Following previous works [\(Zhang et al.,](#page-16-5) [2023\)](#page-16-5), we exclude WNLI in the experiments.

Implementation Details. We select the learning rates from {1 × 10−<sup>5</sup> *,*5 × 10−<sup>5</sup> *,*1 × 10−<sup>4</sup> 5 × 10−<sup>4</sup> }. We quantize the entire backbone. Given that GLUE, SQuADv1.1, and ANLI are relatively easy NLU tasks, we also quantize the embedding layer for higher compression efficiency. We apply the NormalFloat and the uniform quantization for LoftQ and QLoRA at both 2-bit and 4-bit levels. We use rank 16 and 32 for low-rank adapters. More implementation details, such as the training epochs and batch sizes, are presented in Appendix [D.2.](#page-19-0)

Main Results. Table [1](#page-8-0) and Table [2](#page-8-1) summarize the results for 2-bit quantization on the GLUE, SQuADv1.1, and ANLI datasets, by NF2 and the uniform quantization, respectively. Our method consistently outperforms QLoRA on all settings with respect to different ranks, quantization methods, and datasets. When using the uniform quantization (Table [2\)](#page-8-1), our method achieves 88.0% accuracy on MNLI-m, surpassing the QLoRA baseline by 8%. For tasks like SST and SQuADv1.1, our method even approaches the full fine-tuning performance at 2-bit level. The 4-bit quantization experiment results are presented in Appendix [D.1](#page-19-1) as both LoftQ and QLoRA achieve performance close to full fine-tuning.

Our method is also more stable compared to QLoRA in the low-bit regime. For instance, while QLoRA fails to converge on CoLA for both quantization methods and ranks, LoftQ converges in all cases and achieves a score of 60.5 using uniform quantization at rank 32. LoftQ stands out in its ability to consistently attain robust and improved performance by effectively preserving the starting point of pre-trained weights.

<span id="page-8-0"></span>Table 1: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set, SQuADv1.1 development set, ANLI test set using NF2 quantization. We report the median over four seeds. *N.A.* indicates the model does not converge. The best results on each dataset are shown in bold.

| Rank | Method         | MNLI<br>m / mm         | QNLI<br>Acc  | RTE<br>Acc   | SST<br>Acc   | MRPC<br>Acc            | CoLA<br>Matt | QQP<br>Acc             | STSB<br>P/S Corr       | SQuAD<br>EM/F1           | ANLI<br>Acc  |
|------|----------------|------------------------|--------------|--------------|--------------|------------------------|--------------|------------------------|------------------------|--------------------------|--------------|
| -    | Full FT        | 90.5/90.6              | 94.0         | 82.0         | 95.3         | 89.5/93.3              | 69.2         | 92.4/89.8              | 91.6/91.1              | 88.5/92.8                | 59.8         |
| 16   | LoRA           | 90.4/90.5              | 94.6         | 85.1         | 95.1         | 89.9/93.6              | 69.9         | 92.0/89.4              | 91.7/91.1              | 87.3/93.1                | 60.2         |
| 16   | QLoRA<br>LoftQ | 75.4/75.6<br>84.7/85.1 | 82.4<br>86.6 | 55.9<br>61.4 | 86.5<br>90.2 | 73.8/82.8<br>83.8/88.6 | N.A.<br>37.4 | 86.8/82.3<br>90.3/86.9 | 83.0/82.8<br>87.1/86.9 | 61.5 / 71.2<br>81.5/88.6 | N.A.<br>47.1 |
| 32   | QLoRA<br>LoftQ | 78.5/78.7<br>86.0/86.1 | 80.4<br>89.9 | 56.7<br>61.7 | 86.9<br>92.0 | 73.8/82.7<br>83.6/87.2 | N.A.<br>47.5 | 87.1/82.7<br>91.0/87.9 | 83.6/83.3<br>87.5/87.0 | 64.6/73.8<br>82.9/89.8   | N.A.<br>49.0 |

<span id="page-8-1"></span>Table 2: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set, SQuADv1.1 development set using Uniform quantization . We report the median over four seeds. *N.A.* indicates the model does not converge. The best results on each task are shown in bold.

| Rank | Method         | MNLI<br>m / mm         | QNLI<br>Acc  | RTE<br>Acc   | SST<br>Acc   | MRPC<br>Acc            | CoLA<br>Matt | QQP<br>Acc             | STSB<br>P/S Corr       | SQuAD<br>Em/F1         |
|------|----------------|------------------------|--------------|--------------|--------------|------------------------|--------------|------------------------|------------------------|------------------------|
| -    | Full FT        | 90.5/90.6              | 94.0         | 82.0         | 95.3         | 89.5/93.3              | 69.2         | 92.4/89.8              | 91.6/91.1              | 88.5/92.8              |
| 16   | LoRA           | 90.4/90.5              | 94.6         | 85.1         | 95.1         | 89.9/93.6              | 69.9         | 92.0/89.4              | 91.7/91.1              | 87.3/93.1              |
| 16   | QLoRA<br>LoftQ | 76.5/76.3<br>87.3/87.1 | 83.8<br>90.6 | 56.7<br>61.1 | 86.6<br>94.0 | 75.7/84.7<br>87.0/90.6 | N.A.<br>59.1 | 87.1/82.6<br>90.9/88.0 | 83.5/83.4<br>87.9/87.6 | 69.5/77.6<br>84.4/91.2 |
| 32   | QLoRA<br>LoftQ | 79.9/79.5<br>88.0/88.1 | 83.7<br>92.2 | 57.8<br>63.2 | 86.9<br>94.7 | 76.5/84.5<br>87.5/91.2 | N.A.<br>60.5 | 88.6/84.7<br>91.3/88.3 | 84.1/84.0<br>89.5/89.2 | 71.6/80.2<br>85.2/91.6 |

### 4.2 Encoder-Decoder Model: BART

Models and Datasets. We quantize BART-large model [\(Lewis et al.,](#page-14-8) [2020\)](#page-14-8) with LoftQ, then finetune and evaluate the model on two commonly used summarization datasets: XSum [\(Narayan et al.,](#page-15-1) [2018\)](#page-15-1) and CNN/DailyMail[\(Hermann et al.,](#page-14-6) [2015\)](#page-14-6).

Implementation Details. We apply LoftQ to weight matrices in MHA and FFN of both encoder and decoder layers. We report ROUGE 1/2/L scores, which are the metrics for summarization tasks [\(Lin,](#page-15-6) [2004\)](#page-15-6). We conduct quantization experiments in both 2-bit and 4-bit scenarios. We experiment with both NormalFloat and the uniform quantization in both 2-bit and 4-bit scenarios. In each precision, we choose rank equal to 8 and 16 for a fair comparison with the full precision LoRA baseline [\(Zhang et al.,](#page-16-5) [2023\)](#page-16-5). Please see Appendix [E](#page-20-0) for detailed configurations.

Main Results. Table [3](#page-9-0) summarizes our 4-bit quantization experiment results on the XSum and CNN/DailyMail test sets. Our method consistently outperforms QLoRA at both ranks on both datasets. It even surpasses full precision LoRA at both ranks on Xsum. We will discuss this unexpected results in Section [5.](#page-11-0) The 2-bit quantization results are shown in Table [4.](#page-9-1) Our observation is consistent with the NLU experiments, that LoftQ demonstrates the convergence to reasonable results, while QLoRA does not converge. This indicates our method is robuster by narrowing the initialization gap.

<span id="page-9-0"></span>Table 3: Results with 4-bit LoftQ of BART-large on XSum and CNN/DailyMail. We report ROUGE-1/2/L, the higher the better. *Lead-3* means choosing the first 3 sentences as the summary. *N.A.* indicates the model does not converge. *Full FT* refers to the full fine-tuning where all parameters are tuned. We report the median over five seeds.

| Quantization   | Rank | Method  | XSum              | CNN/DailyMail     |
|----------------|------|---------|-------------------|-------------------|
|                |      | Lead-3  | 16.30/1.60/11.95  | 40.42/17.62/36.67 |
| Full Precision | -    | Full FT | 45.14/22.27/37.25 | 44.16/21.28/40.90 |
|                | 8    | LoRA    | 43.40/20.20/35.20 | 44.72/21.58/41.84 |
|                | 16   | LoRA    | 43.95/20.72/35.68 | 45.03/21.84/42.15 |
|                |      | QLoRA   | 42.91/19.72/34.82 | 43.10/20.22/40.06 |
| NF4            | 8    | LoftQ   | 44.08/20.72/35.89 | 43.81/20.95/40.84 |
|                |      | QLoRA   | 43.29/20.05/35.15 | 43.42/20.62/40.44 |
|                | 16   | LoftQ   | 44.51/21.14/36.18 | 43.96/21.06/40.96 |
|                |      | QLoRA   | 41.84/18.71/33.74 | N.A.              |
| Uniform        | 8    | LoftQ   | 43.86/20.51/35.69 | 43.73/20.91/40.77 |
|                |      | QLoRA   | 42.45/19.36/34.38 | 43.00/20.19/40.02 |
|                | 16   | LoftQ   | 44.29/20.90/36.00 | 43.87/20.99/40.92 |

<span id="page-9-1"></span>Table 4: Results with 2-bit LoftQ of BART-large on XSum and CNN/DailyMail using NF2 quantization. *N.A.* indicates the model does not converge. We report ROUGE-1/2/L, the higher the better. We report the median over five seeds.

| Rank | Method | XSum              | CNN/DailyMail     |  |  |
|------|--------|-------------------|-------------------|--|--|
| 8    | QLoRA  | N.A.              | N.A.              |  |  |
|      | LoftQ  | 39.63/16.65/31.62 | 42.24/19.44/29.04 |  |  |
| 16   | QLoRA  | N.A.              | N.A.              |  |  |
|      | LoftQ  | 40.81/17.85/32.80 | 42.52/19.81/39.51 |  |  |

### 4.3 Decoder-only Model: LLAMA-2

Models and Datasets. We quantize LLAMA-2-7b and LLAMA-2-13b [\(Touvron et al.,](#page-16-0) [2023\)](#page-16-0) with LoftQ. We then fine-tune and evaluate the models on two NLG datasets: GSM8K [\(Cobbe et al.,](#page-13-2) [2021\)](#page-13-2) and WikiText-2 [\(Merity et al.,](#page-15-7) [2016\)](#page-15-7). Please see Appendix [F](#page-21-0) for more details about the datasets.

Implementation Details. Similarly, we apply LoftQ to weight matrices in MHA and FFN of all layers. In WikiText-2 evaluation, we report perplexity. In GSM8K evaluation, we extract numerical answers in the generated solutions and then calculate the accuracy using those numerical answers. We conduct experiments with both NF2 and NF4. Please see Appendix [F](#page-21-0) for detailed configurations. Main Results. Table [5](#page-11-1) presents a summary of our experiments on LLAMA-2-7b and LLAMA-2-13b using 2-bit, 4-bit, and mixed-precision NormalFloat quantization methods on WikiText-2 and GSM8K datasets. In WikiText-2, our method consistently outperforms QLoRA across all quantization precision settings on both models. When dealing with the challenging 2-bit precision, where QLoRA fails to converge, LoftQ manages to achieve a perplexity of 7.85. In GSM8K, our method achieves better or on par performance compared to QLoRA across different model sizes and quantization precision levels. For example, our method achieves 20.9% accuracy using 2-bit precision, where QLoRA doesn't converge.

We find LoftQ outperforms full precision LoRA in GSM8K with LLAMA-2-13b. One possible explanation is that the lack of regularization causes overfitting on full precision LoRA fine-tuning. Therefore, we conduct full precision LoRA with weight decay on GSM8K. From Table [5,](#page-11-1) regularization helps LLAMA-2-13b full precision LoRA fine-tuning, but fails in LLAMA-2-7b. This indicates LLAMA-2-13b is prone to overfitting and quantization has implicit regularization to overcome such overfitting.

To provide a customized trade-off between the performance and precision, we also explore mixed-precision quantization where matrices in the first 4 layers are quantized using 4 bits, and the rest matrices remain 2 bits. We witness a remarkable 5.9% accuracy boost on the GSM8K dataset using LLAMA-2-7b and a 12.7% boost using LLAMA-2-13b. This result underscores the potential of LoftQ for complex mixed-precision quantization scenarios.

### 4.4 Analysis

Effectiveness of Alternating Optimization. We conduct experiments with different alternating step *T* to verify the effectiveness of the alternating optimization and to find the best value *T* as a hyperparameter for different models. Across all tasks and models, we observed that alternating optimization yields substantial improvements even with a minimal alternating step. This suggests that it rapidly narrows the discrepancy between quantized weights and pre-trained weights, making our method easy to apply. For example, our method achieves 88.0% accuracy on MNLI-m dataset using only 5 alternating steps and 21.14 Rouge-2 score using only 1 step. Interestingly, we

<span id="page-11-1"></span>Table 5: Results of LoftQ using NormalFloat for LLAMA-2 series on WikiText-2 and GSM8K. 3/2.5/2.25-bit indicates mixed-precision quantization: 4-bit precision for the first 16/8/4 layers and 2-bit precision for the rest of layers. We report the perplexity (the smaller the better) for WikiText-2 and accuracy for GSM8K. The rank of low-rank adapters is 64. *N.A.* indicates the model does not converge. We report the median over five random seeds.

| Method   | Bit  | LLAMA-2-7b  |        | LLAMA-2-13b |        |
|----------|------|-------------|--------|-------------|--------|
|          |      | WikiText-2↓ | GSM8K↑ | WikiText-2↓ | GSM8K↑ |
| LoRA     | 16   | 5.08        | 36.9   | 5.12        | 43.1   |
| LoRA+Reg | 16   | –           | 34.4   | –           | 45.3   |
| QLoRA    | 4    | 5.70        | 35.1   | 5.22        | 39.9   |
| LoftQ    | 4    | 5.24        | 35.0   | 5.16        | 45.0   |
| QLoRA    | 3    | 5.73        | 32.1   | 5.22        | 40.7   |
| LoftQ    | 3    | 5.63        | 32.9   | 5.13        | 44.4   |
| QLoRA    | 2.5  | N.A.        | N.A.   | 19.39       | N.A.   |
| LoftQ    | 2.5  | 5.78        | 31.1   | 5.22        | 41.1   |
| QLoRA    | 2.25 | N.A.        | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2.25 | 6.13        | 26.5   | 5.45        | 38.1   |
| QLoRA    | 2    | N.A         | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2    | 7.85        | 20.9   | 7.69        | 25.4   |

noticed that increasing the alternating step beyond a certain point tends to result in diminishing returns. We suspect this phenomenon occurs because, as the gap becomes smaller, it becomes more challenging for alternating optimization to consistently minimize the gap at each step. This challenge emerges because of the inherent errors introduced by the quantization method. Nevertheless, results from Figure [3](#page-12-0) indicate our method is not sensitive to the alternating step *T* and is able to consistently enhance downstream fine-tuning performance.

# <span id="page-11-0"></span>5 Discussion

Start with quantization or SVD in the alternating optimization? An alternative algorithm to the alternating optimization is that we first obtain the low-rank approximation *A<sup>t</sup> ,B<sup>t</sup>* and then obtain the quantized weight *Q<sup>t</sup>* by switching Line 3 and Line 4 in Algorithm [1.](#page-6-1) We note this is a valid alternative method as both still jointly minimize the objective in [\(6\)](#page-4-2). Table [6](#page-12-1) summarizes the performance of this alternative method. It is noteworthy that the alternative method still outperforms QLoRA significantly, even though it is worse than the primary version. This observation underscores the potential for performance improvement by achieving a closer

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 3: Comparison of different alternating step *T* used in LoftQ. *T* = 0 indicates we use QLoRA method that initializes low-rank adapters by [\(5\)](#page-4-1). *T* = 1*,*5*,*10 indicates we use different *T* for LoftQ described in Algorithm [1.](#page-6-1) Left: Uniform 2-bit DeBERTaV3-base. Middle: NF4 2-bit LLAMA-2-13b. Right: NF4 BART-large.

approximation of pre-trained weights within the low-precision regime.

<span id="page-12-1"></span>Table 6: Results of 2-bit uniformly quantized DeBERTaV3-base on part of GLUE. LoftQ(SVD First) indicates the alternative LoftQ that swiches Line 3 and Line 4 in Algorithm [1.](#page-6-1) We report the median over four random seeds. The best results on each task are shown in bold.

| Method                   | Rank | MNLI<br>m / mm | QNLI<br>Acc | SST2<br>Acc |
|--------------------------|------|----------------|-------------|-------------|
| Full FT                  | -    | 90.5/90.6      | 94.0        | 95.3        |
| QLoRA                    | 32   | 79.9/79.5      | 83.8        | 86.6        |
| LoftQ(SVD First)         | 32   | 87.8/87.7      | 84.9        | 89.7        |
| LoftQ(Quantiztion First) | 32   | 88.0/88.1      | 92.2        | 94.7        |

# 6 Related Work

Quantization-Aware Training (QAT) is often used to obtain quantized models that are adapted in downstream tasks [\(Peri et al.,](#page-15-8) [2020;](#page-15-8) [Liu et al.,](#page-15-9) [2023\)](#page-15-9). It involves quantization and full model fine-tuning at the same time. However, QAT requires massive training cost, such as the gradient and optimization state. Moreover, it is difficult to compute the gradient of quantized weights. Our method, with the help of LoRA, sidesteps the aforementioned issues, providing a light approach for downstream task adaptation.

Post-Training Quantization (PTQ) is a category of popular quantization frameworks [\(Frantar et al.,](#page-14-9) [2022;](#page-14-9) [Xiao et al.,](#page-16-6) [2023\)](#page-16-6), which can also be used for task adaptation. It calibrates the high-precision

model with a small subset of the training dataset. Therefore, the subsequent quantization is guided by the training dataset, providing task-specific quantized models. Besides, it does not involve any gradient backpropagation, so it is cost-efficient. However, it usually results in lower accuracy compared to QAT.

# 7 Conclusion

We propose LoftQ, a quantization framework for LLMs, which alternatively applies quantization and low-rank approximation to the original high-precision pre-trained weights, to obtain an initialization for the subsequent LoRA fine-tuning. Experiments on natural language understanding, question answering, summarization, and natural language generation show that our framework remarkably surpasses existing methods, e.g., QLoRA, for quantizing encoder-only, encoder-decoder, and decoder-only models. We have not observed our method exhibiting worse performance over QLoRA. Moreover, our quantization framework demonstrates effectiveness and robustness particularly in low-bit quantization regimes, e.g., the 2-bit level.

# References

- <span id="page-13-0"></span>Bai, H., Hou, L., Shang, L., Jiang, X., King, I. and Lyu, M. R. (2022). Towards efficient post-training quantization of pre-trained language models. *Advances in Neural Information Processing Systems*, 35 1405–1418.
- <span id="page-13-1"></span>Bai, H., Zhang, W., Hou, L., Shang, L., Jin, J., Jiang, X., Liu, Q., Lyu, M. and King, I. (2020). Binarybert: Pushing the limit of bert quantization. *arXiv preprint arXiv:2012.15701*.
- <span id="page-13-5"></span>Bar-Haim, R., Dagan, I., Dolan, B., Ferro, L., Giampiccolo, D., Magnini, B. and Szpektor, I. (2006). The second pascal recognising textual entailment challenge.
- <span id="page-13-6"></span>Bentivogli, L., Clark, P., Dagan, I. and Giampiccolo, D. (2009). The fifth pascal recognizing textual entailment challenge. In *TAC*.
- <span id="page-13-3"></span>Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I. and Specia, L. (2017). SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*. Association for Computational Linguistics, Vancouver, Canada.
- <span id="page-13-2"></span>Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. et al. (2021). Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.
- <span id="page-13-4"></span>Dagan, I., Glickman, O. and Magnini, B. (2007). The pascal recognising textual entailment challenge. In *Machine Learning Challenges Workshop*.

- <span id="page-14-3"></span>Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022). Llm. int8 (): 8-bit matrix multiplication for transformers at scale. *arXiv preprint arXiv:2208.07339*.
- <span id="page-14-0"></span>Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. *arXiv preprint arXiv:2305.14314*.
- <span id="page-14-5"></span>Diao, S., Pan, R., Dong, H., Shum, K. S., Zhang, J., Xiong, W. and Zhang, T. (2023). Lmflow: An extensible toolkit for finetuning and inference of large foundation models. *arXiv preprint arXiv:2306.12420*.
- <span id="page-14-10"></span>Dolan, W. B. and Brockett, C. (2005). Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*.
- <span id="page-14-9"></span>Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*.
- <span id="page-14-11"></span>Giampiccolo, D., Magnini, B., Dagan, I. and Dolan, B. (2007). The third PASCAL recognizing textual entailment challenge. In *Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing*. Association for Computational Linguistics, Prague.
- <span id="page-14-7"></span>He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T. and Neubig, G. (2021a). Towards a unified view of parameter-efficient transfer learning. *arXiv preprint arXiv:2110.04366*.
- <span id="page-14-1"></span>He, P., Gao, J. and Chen, W. (2021b). Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing. *arXiv preprint arXiv:2111.09543*.
- <span id="page-14-6"></span>Hermann, K. M., Kocisky, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M. and Blunsom, P. (2015). Teaching machines to read and comprehend. *Advances in neural information processing systems*, 28.
- <span id="page-14-4"></span>Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2021). Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.
- <span id="page-14-12"></span>Levesque, H., Davis, E. and Morgenstern, L. (2012). The winograd schema challenge. In *Thirteenth international conference on the principles of knowledge representation and reasoning*.
- <span id="page-14-2"></span>Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and Zettlemoyer, L. (2019). Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *arXiv preprint arXiv:1910.13461*.
- <span id="page-14-8"></span>Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, Online.

- <span id="page-15-11"></span>Li, Y., Yu, Y., Zhang, Q., Liang, C., He, P., Chen, W. and Zhao, T. (2023). Losparse: Structured compression of large language models based on low-rank and sparse approximation. *arXiv preprint arXiv:2306.11222*.
- <span id="page-15-6"></span>Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*. Association for Computational Linguistics, Barcelona, Spain.
- <span id="page-15-9"></span>Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y., Shi, Y., Krishnamoorthi, R. and Chandra, V. (2023). Llm-qat: Data-free quantization aware training for large language models. *arXiv preprint arXiv:2305.17888*.
- <span id="page-15-3"></span>Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
- <span id="page-15-7"></span>Merity, S., Xiong, C., Bradbury, J. and Socher, R. (2016). Pointer sentinel mixture models.
- <span id="page-15-1"></span>Narayan, S., Cohen, S. B. and Lapata, M. (2018). Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. *ArXiv*, abs/1808.08745.
- <span id="page-15-5"></span>Nie, Y., Williams, A., Dinan, E., Bansal, M., Weston, J. and Kiela, D. (2019). Adversarial nli: A new benchmark for natural language understanding. *ArXiv*, abs/1910.14599. <https://api.semanticscholar.org/CorpusID:207756753>
- <span id="page-15-4"></span>Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32*. Curran Associates, Inc., 8024–8035.
- <span id="page-15-8"></span>Peri, D., Patel, J. and Park, J. (2020). Deploying quantization-aware trained networks using tensorrt. In *GPU Technology Conference*.
- <span id="page-15-2"></span>Rajpurkar, P., Zhang, J., Lopyrev, K. and Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, Austin, Texas.
- <span id="page-15-0"></span>Shen, S., Dong, Z., Ye, J., Ma, L., Yao, Z., Gholami, A., Mahoney, M. W. and Keutzer, K. (2020). Q-bert: Hessian based ultra low precision quantization of bert. In *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34.
- <span id="page-15-10"></span>Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. and Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, Seattle, Washington, USA.

- <span id="page-16-0"></span>Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S. et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-16-4"></span>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
- <span id="page-16-3"></span>Wang, A., Singh, A., Michael, J., Hill, F., Levy, O. and Bowman, S. R. (2019). GLUE: A multi-task benchmark and analysis platform for natural language understanding. In *International Conference on Learning Representations*.
- <span id="page-16-7"></span>Warstadt, A., Singh, A. and Bowman, S. R. (2019). Neural network acceptability judgments. *Transactions of the Association for Computational Linguistics*, 7 625–641.
- <span id="page-16-8"></span>Williams, A., Nangia, N. and Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*. Association for Computational Linguistics, New Orleans, Louisiana.
- <span id="page-16-6"></span>Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J. and Han, S. (2023). Smoothquant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*. PMLR.
- <span id="page-16-1"></span>Zafrir, O., Boudoukh, G., Izsak, P. and Wasserblat, M. (2019). Q8bert: Quantized 8bit bert. In *2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS)*. IEEE.
- <span id="page-16-2"></span>Zhang, H., Dauphin, Y. N. and Ma, T. (2019). Fixup initialization: Residual learning without normalization. *arXiv preprint arXiv:1901.09321*.
- <span id="page-16-5"></span>Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W. and Zhao, T. (2023). Adaptive budget allocation for parameter-efficient fine-tuning. *arXiv preprint arXiv:2303.10512*.

# <span id="page-17-0"></span>A Model Compression Ratio and Memory Footprint

We report the compression ratio after applying LoftQ in Table [7.](#page-17-1) It is defined as

$$compression \ ration = \frac{backbone \ size + LoRA \ adapter \ size}{pre\text{-trained size}}.$$

We also measure the GPU memory cost during training. Given that GPU memory varies by models, tasks, sequence lengths, batch sizes, etc. We report LLAMA-2 on GSM8K as an example in Table [8.](#page-17-2)

Table 7: Compression ratios of backbones.

<span id="page-17-1"></span>

| Model          | Compression<br>ratio (%) | Trainable<br>ratio (%) | Rank | Bits | Quantization<br>method |
|----------------|--------------------------|------------------------|------|------|------------------------|
| DeBERTaV3-base | 15.6                     | 3.1                    | 16   | 2    | Uniform                |
| DeBERTaV3-base | 18.8                     | 6.3                    | 32   | 2    | Uniform                |
| DeBERTaV3-base | 17.2                     | 3.1                    | 16   | 2    | NF2                    |
| DeBERTaV3-base | 20.4                     | 6.3                    | 32   | 2    | NF2                    |
| BART-large     | 15.3                     | 1.2                    | 8    | 4    | NF2                    |
| BART-large     | 16.7                     | 2.5                    | 16   | 4    | NF2                    |
| BART-large     | 27.8                     | 1.2                    | 8    | 4    | NF4                    |
| BART-large     | 29.0                     | 2.5                    | 16   | 4    | NF4                    |
| BART-large     | 26.2                     | 1.2                    | 8    | 4    | Uniform                |
| BART-large     | 27.5                     | 2.5                    | 16   | 4    | Uniform                |
| LLAMA-2-7b     | 16.6                     | 2.4                    | 64   | 2    | Nf2                    |
| LLAMA-2-7b     | 29.0                     | 2.4                    | 64   | 4    | Nf4                    |
| LLAMA-2-13b    | 16.0                     | 1.9                    | 64   | 2    | Nf2                    |
| LLAMA-2-13b    | 28.5                     | 1.9                    | 64   | 4    | Nf4                    |

Table 8: GPU memory footprint

<span id="page-17-2"></span>

| Model       | Dataset | Seq length | Batch size | GPU Mem |
|-------------|---------|------------|------------|---------|
| LLAMA-2-7b  | GSM8K   | 384        | 1          | 15GB    |
| LLAMA-2-13b | GSM8K   | 384        | 1          | 24GB    |

# B Quantization Time

We report the execution time of LoftQ applying to a single weight matrix in Table [9.](#page-18-1) The time is tested on Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz.

Table 9: Execution time of LoftQ applying to different weight matrices.

<span id="page-18-1"></span>

| Model          | Size              | Step<br>T | Quantization method | Time |
|----------------|-------------------|-----------|---------------------|------|
| DeBERTaV3-base | ×<br>768<br>768   | 5         | Uniform             | 1s   |
| BART-large     | ×<br>1024<br>1024 | 5         | NF4                 | 1s   |
| LLAMA-2-7b     | ×<br>4096<br>4096 | 5         | NF4                 | 21s  |
| LLAMA-2-13b    | ×<br>5120<br>5120 | 5         | NF4                 | 43s  |

# <span id="page-18-0"></span>C GLUE Dataset Statistics

We present the dataset statistics of GLUE [Wang et al.](#page-16-3) [\(2019\)](#page-16-3) in the following table.

| Corpus                              | Task                                  | #Train | #Dev | #Test | #Label | Metrics               |  |  |  |  |
|-------------------------------------|---------------------------------------|--------|------|-------|--------|-----------------------|--|--|--|--|
|                                     | Single-Sentence Classification (GLUE) |        |      |       |        |                       |  |  |  |  |
| CoLA                                | Acceptability                         | 8.5k   | 1k   | 1k    | 2      | Matthews corr         |  |  |  |  |
| SST                                 | Sentiment                             | 67k    | 872  | 1.8k  | 2      | Accuracy              |  |  |  |  |
| Pairwise Text Classification (GLUE) |                                       |        |      |       |        |                       |  |  |  |  |
| MNLI                                | NLI                                   | 393k   | 20k  | 20k   | 3      | Accuracy              |  |  |  |  |
| RTE                                 | NLI                                   | 2.5k   | 276  | 3k    | 2      | Accuracy              |  |  |  |  |
| QQP                                 | Paraphrase                            | 364k   | 40k  | 391k  | 2      | Accuracy/F1           |  |  |  |  |
| MRPC                                | Paraphrase                            | 3.7k   | 408  | 1.7k  | 2      | Accuracy/F1           |  |  |  |  |
| QNLI                                | QA/NLI                                | 108k   | 5.7k | 5.7k  | 2      | Accuracy              |  |  |  |  |
|                                     | Text Similarity (GLUE)                |        |      |       |        |                       |  |  |  |  |
| STS-B                               | Similarity                            | 7k     | 1.5k | 1.4k  | 1      | Pearson/Spearman corr |  |  |  |  |

Table 10: Summary of the GLUE benchmark.

GLUE includes two single-sentence classification tasks: SST-2 [\(Socher et al.,](#page-15-10) [2013\)](#page-15-10) and CoLA [\(Warstadt et al.,](#page-16-7) [2019\)](#page-16-7), and three similarity and paraphrase tasks: MRPC [\(Dolan and Brockett,](#page-14-10) [2005\)](#page-14-10), STS-B [\(Cer et al.,](#page-13-3) [2017\)](#page-13-3), and QQP. GLUE also includes four natural language inference tasks in GLUE: MNLI [\(Williams et al.,](#page-16-8) [2018\)](#page-16-8), QNLI [\(Rajpurkar et al.,](#page-15-2) [2016\)](#page-15-2), RTE [\(Dagan et al.,](#page-13-4) [2007;](#page-13-4) [Bar-Haim et al.,](#page-13-5) [2006;](#page-13-5) [Giampiccolo et al.,](#page-14-11) [2007;](#page-14-11) [Bentivogli et al.,](#page-13-6) [2009\)](#page-13-6), and WNLI [\(Levesque et al.,](#page-14-12) [2012\)](#page-14-12).

# D Natural Language Understanding

### <span id="page-19-1"></span>D.1 GLUE with 4-bit

We show the 4-bits results in the Table [11.](#page-19-2) Both methods can achieve performance close to full-finetuning.

<span id="page-19-2"></span>Table 11: Results with 4-bit LoftQ of DeBERTaV3-base models on GLUE development set using NF4 quantization. We report the median over four seeds. Results with N.A. indicate the model does not converge. The best results on each dataset are shown in bold

| Method  | Rank | MNLI      | SST-2 | QNLI | ANLI |
|---------|------|-----------|-------|------|------|
|         |      | m / mm    | Acc   | Acc  | Acc  |
| Full FT | -    | 90.5/90.6 | 95.3  | 94.0 | 59.8 |
| QLoRA   | 32   | 89.9/89.9 | 95.3  | 94.2 | 59.4 |
| LoftQ   | 32   | 89.9/90.0 | 95.3  | 94.1 | 59.9 |

### <span id="page-19-0"></span>D.2 Training Details

Implementation Details. The implementation of LoftQ is based on publicly available Huggingface [\(Paszke et al.,](#page-15-4) [2019\)](#page-15-4) code-base [\\*\\*](#page-19-3) .

Hyper-parameter Details. We select the learning rate of {1 × 10−<sup>5</sup> *,*5 × 10−<sup>5</sup> *,*1 × 10−<sup>4</sup> *,*5 × 10−<sup>4</sup> }, and use the selected learning rate for both uniform quantization experiments and nf2 quantization experiments. We use batch size of 32 for all GLUE tasks and ANLI. We use batch size of 16 for SQuADv1.1. We use LoftQ of 5 iterations for all GLUE tasks.

Table [12](#page-19-4) summarizes the detailed hyperparameters for each task used in training DeBERTaV3 base using uniform quantization. Table [13](#page-20-1) summarizes the detailed hyperparameters for each task used in training DeBERTaV3-base using nf2 quantization.

<span id="page-19-4"></span>Table 12: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base using Uniform quantization.

| Hyper-parameter | MNLI     | RTE      | QNLI     | MRPC     | QQP      | SST-2    | CoLA     | STS-B    | SQuADv1.1 | ANLI     |
|-----------------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|----------|
| # epochs        | 5        | 20       | 10       | 60       | 10       | 10       | 60       | 60       | 10        | 12       |
| Learning rate   | 1 × 10−4 | 5 × 10−4 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5  | 5 × 10−5 |

<span id="page-19-3"></span><sup>\*\*</sup><https://github.com/huggingface/transformers/tree/main/examples/pytorch>

<span id="page-20-1"></span>Table 13: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base using NF2 quantization.

| Hyper-parameter | MNLI     | RTE      | QNLI     | MRPC     | QQP      | SST-2    | CoLA     | STS-B    | SQuADv1.1 | ANLI     |
|-----------------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|----------|
| # epochs        | 5        | 20       | 10       | 60       | 10       | 10       | 60       | 60       | 10        | 12       |
| Learning rate   | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5  | 5 × 10−5 |

# <span id="page-20-0"></span>E Summarization

### E.1 Training Details

We choose Adam as the optimizer and try learning rate from{1×10−<sup>5</sup> *,*5×10−<sup>5</sup> *,*7×10−<sup>5</sup> *,*2×10−<sup>4</sup> *,*3× 10−<sup>4</sup> *,*4 × 10−<sup>4</sup> }. We show the optimal learning rate for different settings in Table [14.](#page-20-2) We use LoftQ of 1 iteration for all BART-large experiments. Table [14](#page-20-2) and Table [15](#page-20-3) summarize the learning rate and other hyper-parameters for CNN/DailyMail and XSum.

<span id="page-20-2"></span>Table 14: Hyper-parameter setup of LoftQ BART-large on CNN/DailyMail

|                | NF4   |        | 4-bit Uniform |        | NF2   |        |
|----------------|-------|--------|---------------|--------|-------|--------|
| Hyperparameter | rank8 | rank16 | rank8         | rank16 | rank8 | rank16 |
| Learning rate  | 2e-4  | 2e-4   | 2e-4          | 3e-4   | 2e-4  | 2e-4   |
| Epoch          | 15    | 15     | 15            | 15     | 15    | 15     |
| Batch size     | 64    | 64     | 64            | 64     | 64    | 64     |

Table 15: Hyper-parameter setup of LoftQ BART-large on XSum

<span id="page-20-3"></span>

|                | NF4   |        |       | 4-bit Uniform | NF2   |        |
|----------------|-------|--------|-------|---------------|-------|--------|
| Hyperparameter | rank8 | rank16 | rank8 | rank16        | rank8 | rank16 |
| Learning rate  | 2e-4  | 2e-4   | 2e-4  | 2e-4          | 2e-4  | 2e-4   |
| Epoch          | 25    | 25     | 25    | 25            | 25    | 25     |
| Batch size     | 32    | 32     | 32    | 32            | 32    | 32     |

# <span id="page-21-0"></span>F Natural Language Generation

<span id="page-21-1"></span>We set the batch size as 32 for WikiText-2 and 16 for GSM8K. We train 2 epochs on WikiText-2 and 6 epochs on GSM8K. We select learning rate from{1×10−<sup>5</sup> *,*5×10−<sup>5</sup> *,*7×10−<sup>5</sup> *,*1×10−<sup>4</sup> *,,*3×10−<sup>4</sup> *,*4×10−<sup>4</sup> }. Specific settings are summarized in Table [16](#page-21-1) and Table [17.](#page-21-2)

Table 16: Hyper-parameter setup of LoftQ LLAMA-2-series on GSM8K

| Model       | Hyperparameter | NF4            | NF2            | Mixed-precision |
|-------------|----------------|----------------|----------------|-----------------|
| LLAMA-2-7b  | learning rate  | 10−4<br>×<br>3 | 10−4<br>×<br>3 | 10−4<br>×<br>3  |
| LLAMA-2-13b | learning rate  | 10−4<br>×<br>1 | 10−4<br>×<br>1 | 10−4<br>×<br>3  |

<span id="page-21-2"></span>Table 17: Hyper-parameter setup of LoftQ LLAMA-2-series on WikiText-2

| Model       | Hyperparameter | NF4            | NF2            | Mixed-precision |
|-------------|----------------|----------------|----------------|-----------------|
| LLAMA-2-7b  | learning rate  | 10−4<br>×<br>3 | 10−4<br>×<br>3 | 10−4<br>×<br>3  |
| LLAMA-2-13b | learning rate  | 10−4<br>×<br>1 | 10−4<br>×<br>1 | 10−4<br>×<br>3  |

# G Comparison to Pruning

Pruning is also a widely used compression method. Here we compare LoftQ with the state-of-theart pruning method [Li et al.](#page-15-11) [\(2023\)](#page-15-11). We show the comparison in Table [18.](#page-22-0) We can see our method significantly outperforms the pruning methods on DeBERTaV3-base model. We also remark that LoftQ can consistently reduce the memory of both training and storage. In contrast, pruning requires training the entire full-precision matrix, which implies that it can not achieve any memory savings during the training stage.

# H Extension to Convolutional Layers

Low-rank adapters can also be applied to convolutional layers. Given an input feature map *X* ∈ <sup>R</sup>*h*×*w*×*c*<sup>1</sup> and *<sup>c</sup>*<sup>2</sup> 2D convolutional kernels *<sup>K</sup><sup>i</sup>* <sup>∈</sup> <sup>R</sup>*c*1×*d*×*<sup>d</sup> , i* = 1*,*2*,..., c*2, the output of the convolutional layer is

<span id="page-21-3"></span>
$$Y = \operatorname{stack}(X \otimes K_1, ..., X \otimes K_{c_2}), \tag{10}$$

where *Y* ∈ R*h*×*w*×*c*<sup>2</sup> and ⊗ denotes the 2D convolution operation.

<span id="page-22-0"></span>Table 18: Results of LoftQ using 2-bits uniform quantization compared with LoSparse with DeBERTaV3-base models on some of GLUE development sets. Here *Ratio* is the proportion of total remaining weights. Results with *N.A.* indicate the model does not converge.

| Method   | Ratio | MNLI<br>m / mm | SST-2<br>Acc | QNLI<br>Acc |  |
|----------|-------|----------------|--------------|-------------|--|
| Full FT  | 100%  | 90.5 / 90.6    | 95.3         | 94.0        |  |
| LoSparse | 15%   | 83.3/82.9      | 87.6         | 90.4        |  |
|          | 20%   | 84.5/83.8      | 91.7         | 88.6        |  |
| LoftQ    | 15.6% | 87.3/87.1      | 94.0         | 90.6        |  |
|          | 18.8% | 88.0/88.1      | 94.7         | 92.4        |  |

We can reformulate Equation [\(10\)](#page-21-3) into matrix multiplication as

$$Y = Z \times H^{\top},$$

where *Z* ∈ R*hw*×*c*1*<sup>d</sup>* 2 *,H* ∈ R*c*2×*c*1*<sup>d</sup>* 2 , by extending and flattening the input *X* together with concatenating and flattening kernels. We first extend a vector *xi,j* ∈ R*c*<sup>1</sup> by its neighbor vectors within the kernel window:

$$x_{i,j}^{'} = \text{Concat}(x_{i-\frac{d}{2},j-\frac{d}{2}},...,x_{i+\frac{d}{2},j+\frac{d}{2}}).$$

Now, *X* becomes *X* ′ ∈ R*h*×*w*×*c*1*<sup>d</sup>* . We then flatten *X* ′ into *Z* ∈ R*hw*×*c*1*<sup>d</sup>* 2 . For kernels, we first concatenate {*K*1*,...,Kc*<sup>2</sup> } into *H*′ ∈ R*c*2×*c*1×*d*×*<sup>d</sup>* . We then flatten *H*′ into *H*.

Note that *H* can be approximated by a low-rank matrix

$$R = UV^{\top},$$

where *U* ∈ R*c*2×*<sup>r</sup> ,V* ∈ R*c*1*<sup>d</sup>* <sup>2</sup>×*r , r* ≪ min{*c*2*, c*1*d* 2 } by SVD. Therefore, the original convolution layer can be approximated as

$$\widehat{Y} = Z \times (UV^{\top})^{\top} \tag{11}$$

$$= (Z \times V) \times U^{\top} \tag{12}$$

$$= M \times U^{\top}. \tag{13}$$

Note that *Z* × *V* can be restored into a convolution operation where we have *r* kernels *D<sup>i</sup>* ∈ R*c*1×*d*×*<sup>d</sup> , i* = 1*,*2*,,..., r* and *M* × *U*<sup>⊤</sup> can also be restored into a convolution operation where we have *<sup>c</sup>*<sup>2</sup> kernels *<sup>U</sup><sup>i</sup>* <sup>∈</sup> <sup>R</sup>*r*×1×<sup>1</sup> *, i* = 1*,*2*,,..., c*2.