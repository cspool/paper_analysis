# DiJiang: Efficient Large Language Models through Compact Kernelization

Hanting Chen \* 1 Zhicheng Liu \* 1 Xutao Wang <sup>1</sup> Yuchuan Tian <sup>2</sup> Yunhe Wang <sup>1</sup>

{chenhanting,yunhe.wang}@huawei.com;

### Abstract

In an effort to reduce the computational load of Transformers, research on linear attention has gained significant momentum. However, the improvement strategies for attention mechanisms typically necessitate extensive retraining, which is impractical for large language models with a vast array of parameters. In this paper, we present DiJiang, a novel Frequency Domain Kernelization approach that enables the transformation of a pre-trained vanilla Transformer into a linear complexity model with little training costs. By employing a weighted Quasi-Monte Carlo method for sampling, the proposed approach theoretically offers superior approximation efficiency. To further reduce the training computational complexity, our kernelization is based on Discrete Cosine Transform (DCT) operations. Extensive experiments demonstrate that the proposed method achieves comparable performance to the original Transformer, but with significantly reduced training costs and much faster inference speeds. Our DiJiang-7B achieves comparable performance with LLaMA2-7B on various benchmark while requires only about 1/50 training cost. Code is available at [https:](https://github.com/YuchuanTian/DiJiang) [//github.com/YuchuanTian/DiJiang](https://github.com/YuchuanTian/DiJiang).

1. Introduction

The Transformer architecture [\(Vaswani et al.,](#page-9-0) [2017\)](#page-9-0) has revolutionized the field of Natural Language Processing (NLP), achieving outstanding results in various tasks such as speech recognition [\(Dong et al.,](#page-8-0) [2018\)](#page-8-0), machine translation [\(Wang](#page-9-1) [et al.,](#page-9-1) [2019\)](#page-9-1), and document generation/summarization [\(Kim](#page-9-2) [et al.,](#page-9-2) [2022\)](#page-9-2). This success has led to an era dominated by large language models (LLMs), where the Transformer

Preprint.

structure is scaled up to handle increasingly complex tasks. However, this scaling brings with it substantial computational demands, especially due to the attention mechanism which requires cross-correlation calculations between each token. These computational requirements, coupled with the significant inference costs and energy consumption, present considerable obstacles to deploying these models in resource-constrained environments like mobile devices and robotics.

In response to the pressing need for more efficient Transformer models, the research community has directed its efforts towards optimizing the Transformer architecture. A myriad of strategies has been put forward, encompassing methods such as model pruning, quantization, and the development of more efficient attention mechanisms. Among these initiatives, simplifying the attention mechanism has emerged as a particularly promising avenue. This approach focuses on transforming the traditionally quadratic complexity of attention mechanisms into a more manageable linear scale. [\(Katharopoulos et al.,](#page-9-3) [2020\)](#page-9-3) introduces Linear Transformers, which leverage kernel feature maps to transform self-attention, reducing complexity from quadratic to linear while maintaining comparable results to traditional Transformers. [\(Kitaev et al.,](#page-9-4) [2020\)](#page-9-4) proposes replacies dotproduct attention with locality-sensitive hashing and using reversible residual layers to minimize memory usage in training. Performer [\(Choromanski et al.,](#page-8-1) [2020\)](#page-8-1) utilize positive orthogonal random features to approximate softmax-based self-attention in Transformers, achieving a transformative leap to linear complexity.

However, the majority of existing methods for optimizing Transformers, particularly in relation to their attention mechanisms, necessitate comprehensive retraining. This retraining process presents a formidable challenge, especially for models with an immense array of parameters. It requires a significant investment in terms of computational resources and time. For instance, the training of a large model like LLaMA-7B [\(Touvron et al.,](#page-9-5) [2023\)](#page-9-5) demands approximately 82,432 GPU-hours and incurs a total power consumption of around 36 MWh. Undertaking such extensive retraining for models of this magnitude is not only economically taxing but also raises environmental concerns due to the substantial

<sup>\*</sup>Equal contribution <sup>1</sup>Huawei Noah's Ark Lab Peking University. Correspondence to: Yunhe Wang <yunhe.wang@huawei.com>.

energy expenditure involved. This underscores the need for more efficient approaches to adapt and optimize these largescale models. Undertaking such extensive retraining for models of this magnitude is not only economically taxing but also raises environmental concerns due to the substantial energy expenditure involved. Despite few research [\(Zheng](#page-10-0) [et al.,](#page-10-0) [2023;](#page-10-0) [Choromanski et al.,](#page-8-1) [2020\)](#page-8-1) efforts focusing on finding fast approximations for attention mechanisms, these methods have not been thoroughly validated in large-scale language models.

To address the issue of fast attention approximations in large language models, we conducted a thorough analysis of existing linear attention schemes. We discovered that the main source of approximation error in these methods is due to sampling based on the Monte Carlo method. Consequently, we propose the use of weighted Quasi-Monte Carlo sampling for mapping, specifically introducing Frequency Domain Kernelization. This approach efficiently and accurately maps the queries and keys of a Transformer to the frequency domain using Discrete Cosine Transform (DCT). This mapping allows us to effectively eliminate the softmax operation in the attention mechanism, rendering the attention computation linear in complexity, which is shown in Figure [1.](#page-2-0) We theoretically demonstrate that this frequency domain mapping is an approximate equivalent to the original attention mechanism. Our experiments show that our method achieves performance comparable to the original Transformer with a significantly smaller training cost (< 1/10), while also benefiting from faster inference speeds (up to about 10x).

## 2. Related Works

#### 2.1. Linear Transformers

Reducing the computational load of attention in Transformers remains a hot topic in research. [\(Child et al.,](#page-8-2) [2019\)](#page-8-2) achieved this by sparsifying attention, thereby reducing its computational cost. Similarly, [\(Kitaev et al.,](#page-9-4) [2020\)](#page-9-4) used locality-sensitive hashing to expedite the computation of attention. However, these methods are hard to apply in auto-regressive Transformer models. As a result, there has been a series of works focusing on removing or substituting the softmax in attention. Notably, the Linear Transformer, first introduced by [\(Katharopoulos et al.,](#page-9-3) [2020\)](#page-9-3), represents a significant stride in this direction. [\(Qin et al.,](#page-9-6) [2022\)](#page-9-6) approximated attention calculations using a linear operator and a cosine-based distance reweighting. [\(Zhai et al.,](#page-9-7) [2021\)](#page-9-7) achieved linear complexity in Transformers by preprocessing keys and values. [\(Lu et al.,](#page-9-8) [2021\)](#page-9-8) used Gaussian kernel functions in place of dot-product similarity, allowing for the approximation of the full self-attention matrix through low-rank matrix decomposition. [\(Bello,](#page-8-3) [2021\)](#page-8-3) bypassed the need for attention calculations by capturing interactions

through transforming available contexts into linear functions and applying them to each input, showcasing the variety of methods explored to optimize attention mechanisms in Transformer models.

Additionally, recent proposals like RWKV [\(Peng et al.,](#page-9-9) [2023\)](#page-9-9), RetNet [\(Sun et al.,](#page-9-10) [2023\)](#page-9-10), and Mamba [\(Gu & Dao,](#page-8-4) [2023\)](#page-8-4) have introduced potential alternatives to the Transformer with linear complexity. However, these existing improvements typically require significant modifications to the model's architecture and often necessitate training a new model from scratch to achieve optimal performance. Given the substantial training costs associated with large language models, such retraining is not always feasible. While methods like StreamingLLM [\(Xiao et al.,](#page-9-11) [2023\)](#page-9-11) or Longformer [\(Beltagy et al.,](#page-8-5) [2020\)](#page-8-5) can be implemented through fine-tuning, their reliance on window attention compromises their ability to truly model long sequences, leading to a decrease in accuracy. This highlights the challenge of balancing model training efficiency with the ability to maintain high performance in handling long sequences.

#### 2.2. Frequency-based Transformers

A various of research has focused on applying the Transformer architecture in the frequency domain. For instance, FNet [\(Lee-Thorp et al.,](#page-9-12) [2021\)](#page-9-12) replaces the self-attention in BERT with Fourier Transform, significantly speeding up Transformer computations. A similar concept [\(Buch](#page-8-6)[holz & Jug,](#page-8-6) [2022\)](#page-8-6) has been adapted for image processing tasks. DCFormer [\(Li et al.,](#page-9-13) [2023\)](#page-9-13) proposes a Transformerbased network that learns semantic representations directly from frequency domain representations using Discrete Cosine Transform (DCT). In the realm of video prediction, ideas like the local frequency domain transformer [\(Farazi](#page-8-7) [et al.,](#page-8-7) [2021\)](#page-8-7) have been introduced. However, applying these concepts to existing decoder-only large language models presents challenges. The auto-regressive inference style of these models makes token-level frequency domain transformations cumbersome. Each new token requires frequency domain transformation in conjunction with all previous tokens, which fails to reduce complexity and undermines the potential efficiency gains of frequency domain approaches in large-scale language models.

### 3. Kernelized Attention in Frequency Domain

In our study, we begin by revisiting the general form of selfattention [\(Vaswani et al.,](#page-9-0) [2017\)](#page-9-0). To simplify the notation and focus on the core aspects, we consider the single head form of self-attention and omit normalization factors. The self-attention mechanism is fundamentally composed of

![](_page_2_Figure_1.jpeg)

Figure 1. Illustration of the proposed method, where the computation of queries and keys in the attention mechanism of a Transformer is efficiently mapped to the frequency domain using a fast Discrete Cosine Transform (DCT). This mapping effectively eliminates the softmax operation, thereby substantially reducing the computational complexity of the Transformer.

queries Q, keys K, and values V, expressed in the formula:

<span id="page-2-1"></span>Attention
$$(Q, K, V) = \operatorname{softmax}(QK^{\mathsf{T}})V,$$
  
where  $Q, K, V \in \mathbb{R}^{n \times d},$  (1)

where n denotes the number of tokens and d denotes the hidden dimension of the attention. Specifically, when we denote Q as  $(q_1, q_2, ..., q_n)$ , K as  $(k_1, k_2, ..., k_n)$ , V as  $(v_1, v_2, ..., v_n)$ , and output O as  $(o_1, o_2, ..., o_n)$ , Equation 1 can be reformulated as:

$$o_{i} = \sum_{j=1}^{n} \frac{e^{q_{i}k_{j}^{\mathsf{T}}}}{\sum_{j'=1}^{n} e^{q_{i}k_{j'}^{\mathsf{T}}}} v_{j},$$
where  $q_{i}, k_{i}, v_{i} \in \mathbb{R}^{1 \times d}, i = \{1, 2, ..., n\}.$  (2)

It can be observed that the computational and memory complexity for calculating each output in a Transformer model is  $\mathcal{O}(nd)$ , where n is the sequence length and d is the dimensionality of the representation. Consequently, the time and memory complexity for processing a sentence of length n scales quadratically, becoming  $\mathcal{O}(n^2d)$ . This quadratic scaling poses a significant computational burden, particularly for longer sequences where n is large, making processing resource-intensive and challenging.

To mitigate this complexity, the concept of a kernel mechanism has been introduced as a means to reduce the computational demands of attention mechanisms, which has been introduced in (Tsai et al., 2019; Katharopoulos et al., 2020; Choromanski et al., 2020). Specifically, this involves the introduction of a kernel function  $K(\cdot, \cdot)$ , which acts as a positive-definite kernel capable of measuring similarity. By utilizing this kernel, the attention mechanism can be

<span id="page-2-0"></span>reformulated as:

$$o_i = \sum_{j=1}^{n} \frac{K(q_i, k_j)}{\sum_{j'=1}^{n} \mathcal{K}(q_i, k_{j'})} v_j,$$
 (3)

By applying the kernel trick, it's possible to linearly decompose the attention mechanism:

<span id="page-2-2"></span>
$$o_{i} = \sum_{j=1}^{n} \frac{\phi(q_{i})\phi(k_{j})^{\mathsf{T}}}{\sum_{j'=1}^{n} \phi(q_{i})\phi(k_{j'})^{\mathsf{T}}} v_{j}, \tag{4}$$

where  $\phi(\cdot):\mathbb{R}^d\to\mathbb{R}^m$  is a projection to map the inputs into m dimension features. This decomposition benefits from the fact that the computational dimensions of the keys and values can be merged, effectively reducing the computational complexity from  $\mathcal{O}(n^2d)$  to  $\mathcal{O}(nmd)$ . Given that the dimensionality d and m is typically much smaller than the sequence length n, this linearization of the attention mechanism results in a substantial decrease in computational intensity.

In the context of large language models, the cost of retraining is prohibitively high. In such scenarios, it becomes imperative to find a kernel that can equivalently replace the vanilla attention mechanism without necessitating extensive retraining. Positive Random Features (PRF) (Choromanski et al., 2020) emerge as a viable candidate in this regard:

<span id="page-2-3"></span>
$$\phi_{\text{PRF}}(x) = e^{\omega x^{\mathsf{T}} - \frac{\|x\|^2}{2}},\tag{5}$$

where  $\omega \in \mathbb{R}^{m \times d}$ . Theoretical demonstrations have established that  $e^{qk^\intercal} = \mathbb{E}_{\omega \sim \mathcal{N}(0.I)}[e^{\omega q^\intercal - \frac{\|q\|^2}{2}}e^{\omega k^\intercal - \frac{\|k\|^2}{2}}]$ . It means that when m, the dimension of the feature space, is sufficiently large, Positive Random Features (PRF) mapping becomes an equivalent of the original attention mechanism.

This equivalence suggests that, in theory, it is feasible to directly transform existing vanilla attention into linear attention using PRF mapping, thereby achieving an acceleration without loss of functionality. However, a notable challenge arises due to the need for m to be set to a significantly large value to maintain the performance by reducing the approximation error. This requirement leads to a non-negligible increase in computational demand. For instance, in the case of the Performer (Choromanski et al., 2020), to achieve a lossless linear attention, m often needs to be set to larger than d, diminishing the benefits of reduced computational load brought by linear attention.

To address this issue, we first conduct a theoretical analysis of the kernel-based approach for approximating attention mechanisms. We begin with the application of Bochner's Theorem. This theorem allows us to equate the original attention computation involving queries (Q) and keys (K) – specifically the Gaussian kernel – to an integral computation akin to Equation 4.

**Theorem 3.1.** (Bochner's Theorem) (Feller, 1966). A continuous shift invariant scaled kernel function K(x,z):  $\mathbb{R}^d \to R$  is positive definite if and only if it is the Fourier Transform of a unique finite probability measure p on  $\mathbb{R}^d$ .

<span id="page-3-1"></span>
$$K(x,z) = \int_{\mathbb{R}^d} e^{i(x-z)^{\mathsf{T}} w} p(w) dw = E_{w \sim p(\cdot)} [e^{iw^{\mathsf{T}} x} (e^{iw^{\mathsf{T}} z})^*],$$
(6)

where the symbol  $z^*$  denotes the complex conjugate of z.

According to Bochner's theorem, there is a one-to-one correspondence between the kernel function K(x,z) and the probability density p(w) defined on  $\mathbb{R}^d$ . Monte Carlo is equal weight approximation to kernel integrals. Taking  $\varphi_p(x) := \frac{1}{\sqrt{m}} [e^{-iw_1^\intercal x},...,e^{-iw_m^\intercal x}]^\intercal$ , the feature maps can be constructed as:

$$K(x,z) = E_{w \sim p(\cdot)}[e^{iw^\intercal x}(e^{iw^\intercal z})^*] \approx \varphi_p(x)^\intercal \varphi_p^*(z), \tag{7}$$

where  $w_i \sim p(\cdot)$  are samples constructed by Monte Carlo methods.  $\varphi_p(\cdot)$  is the explicit finite dimensional feature map, which depends on the kernel K. Moving forward, instead of employing the Monte Carlo method as suggested in (Choromanski et al., 2020), we utilize the Quasi-Monte Carlo method (Le et al., 2013). This shift enables the estimation of the integral using a specific uniform distribution as opposed to a randomly sampled distribution.

Utilizing Bochner's theorem allows for a transformative interpretation of the attention mechanism in Transformer models. For the Gaussian Kernel:

$$K_G(x,y) := e^{-\frac{\|x-y\|^2}{2}} = e^{-\frac{\|x\|^2 + \|y\|^2}{2}} e^{x^{\mathsf{T}}y},$$
 (8)

since the x and y in attention mechanism is usually normalized, the Gaussian Kernel can be regarded as  $e^{x^{\mathsf{T}}y}$ , which is the same as the calculation between the queries and keys.

**Theorem 3.2.** The Positive Fixed Features (PFF) is formulated as:

<span id="page-3-0"></span>
$$\varphi_{PFF}(x) := \frac{e^{-\|x\|^2}}{\sqrt{m}} [e^{\Phi^{-1}(t_1)x^{\mathsf{T}}v_1}, ..., e^{\Phi^{-1}(t_m)x^{\mathsf{T}}v_m}]^{\mathsf{T}},$$
(9)

where  $V = [v_1, ..., v_m] \in \mathbb{S}^{d \times m}$  is asymptotically uniformly distributed and  $t_i \sim U(0, 1)$ . Then,  $\varphi_{PFF}(x)^{\mathsf{T}} \varphi_{PFF}(z)$  is an unbiased estimate of Gaussian kernel  $K_G(x, y)$ .

The proof of this theorem involves a transformation to spherical coordinates, which can be found in the supplementary material. Through this transformation, we demonstrate that an approximation based on any asymptotically uniformly distribution can closely approximate the original Gaussian kernel. Furthermore, according to (Asmussen & Glynn, 2007), when utilizing uniform sequences, the Quasi-Monte Carlo method can offer superior approximation efficiency compared to the traditional Monte Carlo method. The approximation efficiency of Quasi-Monte Carlo is  $\mathcal{O}(1/m)$ , which is more favorable than the  $\mathcal{O}(1/m^{-0.5})$  efficiency of Monte Carlo. Consequently, this implies that using the PFF 9 kernel for approximating the Gaussian kernel is more advantageous than the PRF kernel in Equation 5.

**Theorem 3.3.** The Weighted Positive Fixed Features (WPFF) is formulated as:

$$\varphi_{WPFF}(x) := \frac{De^{-\|x\|^2}}{\sqrt{m}} [e^{\Phi^{-1}(t_1)x^{\mathsf{T}}v_1}, ..., e^{\Phi^{-1}(t_m)x^{\mathsf{T}}v_m}]^{\mathsf{T}},$$
(10)

where D is a learnable parameter which can be optimized by the input x. Then the upper bound of the integral estimation error of the objective function by WPFF (Weighted Positive Fixed Features) method is not greater than the upper bound of the integral estimation error of the objective function by PFF (Positive Fixed Features) method.

Building upon the Quasi-Monte Carlo foundation, we further introduce the concept of weighted Quasi-Monte Carlo to enhance the efficiency of approximation. This advancement aims to leverage the strengths of the Quasi-Monte Carlo method, augmenting it with strategically weighted sampling to improve the precision and convergence rates of our approximations. The detailed proof is provided in the supplementary materials.

To further accelerate the training speed, we propose the use of frequency domain transformations to reduce the required computational resources. Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT) are commonly used methods for such transformations. Compared to ordinary orthogonal transformations, frequency domain transformations have algorithms for rapid computation, significantly reducing the computational cost of our proposed mapping. Specifically, the complexity of  $\mathcal{O}(m)$  can be reduced to  $\mathcal{O}(\log(m))$ . Additionally, since DCT operates in the real

**Algorithm 1** Frequency domain kernelization for efficient language models.

<span id="page-4-1"></span>**input** A small amount of data  $x_i$ , a pre-trained Transformer model M.

- **1. Initialization:** the DCT coefficient C, the weight D, the diagonal matrix T in Equation 12 for each layer in M.
- **2. Transformation:** transform the vanilla attention calculation  $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\intercal)V$  to  $\operatorname{FKA}(Q,K,V)=\phi_{\operatorname{WDCF}}(Q)\phi_{\operatorname{WDCF}}(K)^\intercal V$  using the Weighted Discrete Cosine Features for each layer in M.
- **3.** Get the transformed model  $M_{\rm FKA}$ .

#### repeat

- **4.** Randomly select a batch of data from  $x_i$ .
- 5. Employ the transformed model  $M_{\rm FKA}$  on the minibatch.
- **6.** Update weights in  $M_{\rm FKA}$  according to the loss and gradient;

until convergence.

**output** An efficient language model  $M_{\text{FKA}}$ .

number domain, it demands even less computational resources and is more hardware-friendly. Therefore, we opt for the DCT to carry out our kernel mapping.

Specifically, a DCT coefficient  $C \in \mathbb{R}^{d \times d}$  in the frequency domain is defined as:

$$\mathcal{C}_{j_1j_2} = s_{j_1}s_{j_2} \sum_{i_1=0}^{n-1} \sum_{i_2=0}^{d-1} \cos\left(\frac{\pi(2i_1+1)j_1}{2d}\right) \cos\left(\frac{\pi(2i_2+1)j_2}{2d}\right), \tag{11}$$

where  $s_j = \sqrt{1/d}$  if j = 0 and  $s_j = \sqrt{2/d}$  otherwise. The weighted mapping using DCT (which is called Weighted Discrete Cosine Features) can be reformulated as:

<span id="page-4-0"></span>
$$\phi_{\text{WDCF}}(x) = De^{T\mathcal{C}x^{\mathsf{T}}},\tag{12}$$

where  $\mathcal{C} \in \mathbb{R}^{m \times d}$  is the DCT coefficient,  $D \in \mathbb{R}^m$  is a learnable weight, and  $T = \operatorname{diag}(t_1, \dots, t_m)$  is a random diagonal matrix following the inverse cumulative distribution. Note that since the x in attention mechanism is usually normalized, we ignore the term of  $\|x\|^2$  in Equation 9 for efficiency. Therefore, using DCT as a kernel can closely approximate the original attention mechanism while have low computation complexity. For scenarios where m > d, more DCT transformations can be derived using different boundary conditions. Details can be referred to (Ahmed et al., 1974). It is noted that we set m = d to avoid increasing computational complexity in the subsequent experiments.

Therefore, the kernelized attention in frequency domain (FKA) is then reformulated as:

$$\begin{aligned} \text{FKA}(Q, K, V) &= \phi_{\text{WDCF}}(Q) \phi_{\text{WDCF}}(K)^\intercal V, \\ \text{where } Q, K, V &\in \mathbb{R}^{n \times d}, \end{aligned} \tag{13}$$

This approach achieves a notable reduction in computational complexity by employing the Discrete Cosine Transform (DCT) to map the queries and keys within the Transformer's attention mechanism to a domain where operations are inherently more efficient.

In summary, our method leverages frequency domain kernelization for Transformer attention mechanisms, significantly cutting computational costs while either preserving or enhancing model performance. The details are shown in Algorithm 1. Through the strategic use of the weighted Quasi-Monte Carlo method, which outperforms traditional Monte Carlo sampling in efficiency and accuracy, combined with DCT for efficient frequency domain transformations, we attain linear complexity in attention computation. This reformulation not only improves the scalability of Transformers, enabling them to handle larger datasets and extended sequences with ease, but also markedly accelerates the training and inference phases.

#### 4. Experiments

In this section, we conduct extensive experimental validation of the proposed architecture, encompassing results across language models of varying scales. Additionally, we provide detailed analyses to substantiate the effectiveness of our approach.

#### 4.1. Evaluation on Different Scales

Given the challenge of replicating the training processes of most language models, as only their checkpoints are openly available, we opted to validate our method using Pythia (Biderman et al., 2023), a model with a fully public dataset and training procedure, enabling fair comparisons.

We adhered to the exact training settings employed by Pythia, including learning rates, optimizers, and other hyperparameters, and utilized the Pile dataset. The Pile (Gao et al., 2020) is an 825 GiB corpus of English text, specifically designed for training large-scale language models. This project is composed of 22 distinct, high-quality subsets, both pre-existing and newly constructed, many of which originate from academic or professional sources. This comprehensive and diverse dataset serves as a robust foundation for developing and fine-tuning language models Our DiJiang model was fine-tuned from the pre-trained Pythia model. We evaluated our approach on six public datasets used by Pythia: PIQA (Bisk et al., 2020), WinoGrande, WSC (Sakaguchi et al., 2021), ARC-E, ARC-C (Clark et al., 2018), and LogiQA (Liu et al., 2020). The Pythia model's checkpoint was obtained from HuggingFace<sup>1</sup>. We adapt the learned gating mechanism (Peng et al., 2021) similar with the RetNet (Sun et al., 2023) to augment our DiJiang.

<span id="page-4-2"></span>https://huggingface.co/EleutherAI

<span id="page-5-0"></span>Table 1. The experimental results of the proposed method. Training time is measured using A800. Inference throughput is evaluated with token length of 2048. \* denotes results from [\(He et al.,](#page-9-19) [2024\)](#page-9-19).

| Model          | PIQA  | WinoGrande | WSC   | ARC-E | ARC-C | LogiQA | Avg   | Training<br>(day) | Inference<br>(tokens/s) |
|----------------|-------|------------|-------|-------|-------|--------|-------|-------------------|-------------------------|
| Pythia-70M     | 0.498 | 0.484      | 0.596 | 0.25  | 0.221 | 0.202  | 0.375 | 21.3              | 2037                    |
| DiJiang-70M    | 0.587 | 0.511      | 0.365 | 0.403 | 0.213 | 0.253  | 0.389 | 1.3               | 2605                    |
| Pythia-160M    | 0.532 | 0.484      | 0.634 | 0.265 | 0.227 | 0.202  | 0.391 | 42.9              | 622                     |
| DiJiang-160M   | 0.618 | 0.490      | 0.384 | 0.439 | 0.217 | 0.239  | 0.398 | 2.7               | 1315                    |
| Pythia-410M    | 0.668 | 0.537      | 0.567 | 0.521 | 0.213 | 0.22   | 0.454 | 105.8             | 203                     |
| DiJiang-410M   | 0.663 | 0.524      | 0.567 | 0.492 | 0.244 | 0.247  | 0.456 | 6.6               | 787                     |
| Pythia-1B      | 0.706 | 0.533      | 0.365 | 0.569 | 0.269 | 0.296  | 0.456 | 201.2             | 105                     |
| Mamba-1.3B*    | 0.663 | 0.530      | 0.365 | 0.508 | 0.251 | 0.263  | 0.430 | -                 | -                       |
| DiJiang-1B     | 0.677 | 0.521      | 0.365 | 0.537 | 0.253 | 0.284  | 0.440 | 12.6              | 611                     |
| Pythia-2.8B    | 0.737 | 0.596      | 0.384 | 0.640 | 0.295 | 0.215  | 0.478 | 593.3             | 34                      |
| DiJiang-2.8B   | 0.713 | 0.545      | 0.413 | 0.597 | 0.289 | 0.279  | 0.473 | 37.1              | 284                     |
| OPT-350M       | 0.645 | 0.524      | 0.365 | 0.441 | 0.208 | 0.210  | 0.399 | -                 | 201                     |
| DiJiang-350M   | 0.550 | 0.507      | 0.635 | 0.286 | 0.227 | 0.223  | 0.404 | 5.6               | 820                     |
| TinyLLaMA-1.1B | 0.666 | 0.541      | 0.413 | 0.487 | 0.211 | 0.228  | 0.424 | -                 | 74                      |
| DiJiang-1.1B   | 0.535 | 0.508      | 0.635 | 0.286 | 0.243 | 0.212  | 0.403 | 13.9              | 613                     |

The experimental results, as shown in Table [1,](#page-5-0) indicate that our method achieved remarkable outcomes across different model sizes, ranging from 70M to 2.8B parameters. On average, the performance on the six datasets was nearly identical to that of the original Pythia, but with only ∼ 1/16 of the training cost. Furthermore, the inference speed of our DiJiang model was significantly faster than that of the original Pythia. These results substantiate the effectiveness of our approach, demonstrating its potential to enhance the efficiency of large language models without compromising performance.

#### 4.2. Evaluation on Different Models

To evaluate the effectiveness of our method across different models, as shown in Table [1,](#page-5-0) we further applied our approach to the OPT-350M [\(Zhang et al.,](#page-9-20) [2022\)](#page-9-20) and TinyLLaMA-1.1B[3](#page-5-2) models. It's important to note that since their training data are not fully accessible, we continued to use the Pile dataset for fine-tuning them.

Finally, we conducted further experiments on the wellknown publicly available large language model, LLaMA2- 7B, fine-tuning it into the DiJiang-7B model. Table [3](#page-6-0) reveal that the DiJiang-7B model achieves results that are virtually identical to the original LLaMA2-7B across various benchmarks. Remarkably, our model required only 40B

training data, significantly less than the 2T tokens used by LLaMA2-7B. This demonstrates the successful application of our method to large-scale models at the 7B parameter level, highlighting the efficiency and effectiveness of our fine-tuning approach even when scaling to vast model sizes.

Interestingly, we found that despite using a limited dataset, our method achieved results similar to the original models with a significantly lower training cost and faster speed. This outcome further demonstrates the strong generalizability and flexibility of our approach, underscoring its potential applicability across a broad spectrum of language models, even in scenarios where the original training datasets are not available.

#### 4.3. Comparison with Linear Transformers

To compare the superiority of our approach against other linear-complexity self-attention Transformer models, we validated the fine-tuning results on Pythia-400M for different models including Linformer, Performer, RetNet, and Cosformer. For a fair comparison, we employed the same training settings and data. Table [2](#page-6-1) displays the comparative results, revealing that while existing methods can achieve good results through retraining, as evidenced by their original publications, most of them suffer from significant accuracy losses in scenarios where fine-tuning is done without retraining. This is largely because these methods struggle to accurately approximate the original attention mechanism, leading to an inability to restore the original accuracy with minimal training.

<span id="page-5-1"></span><sup>2</sup><https://huggingface.co/facebook/opt-350m>

<span id="page-5-2"></span><sup>3</sup>[https://huggingface.co/TinyLlama/](https://huggingface.co/TinyLlama/TinyLlama-1.1B-python-v0.1)

[TinyLlama-1.1B-python-v0.1](https://huggingface.co/TinyLlama/TinyLlama-1.1B-python-v0.1)

Table 2. Comparison of different linear attention models on fine-tuning Pythoia-410M [\(Biderman et al.,](#page-8-11) [2023\)](#page-8-11).

<span id="page-6-1"></span>

| Model                                | PIQA   | WinoGrande | WSC    | ARC-E  | ARC-C  | LogiQA<br>Avg    |
|--------------------------------------|--------|------------|--------|--------|--------|------------------|
| Pythia-410M (Biderman et al., 2023)  | 0.668  | 0.537      | 0.567  | 0.521  | 0.213  | 0.22<br>0.454    |
| Linformer (Wang et al., 2020)        | 0.5267 | 0.5114     | 0.6346 | 0.2656 | 0.244  | 0.2074<br>0.3982 |
| Cosformer (Qin et al., 2022)         | 0.5218 | 0.5059     | 0.6058 | 0.2673 | 0.2637 | 0.2642<br>0.4047 |
| Performer (Choromanski et al., 2020) | 0.6431 | 0.4964     | 0.4327 | 0.4701 | 0.2312 | 0.2366<br>0.4183 |
| RetNet (Sun et al., 2023)            | 0.4951 | 0.4957     | 0.6346 | 0.2508 | 0.227  | 0.2028<br>0.3843 |
| PFF (Equation 9)                     | 0.6453 | 0.4996     | 0.4712 | 0.4747 | 0.2295 | 0.2381<br>0.4264 |
| DiJiang (Ours)                       | 0.6638 | 0.5241     | 0.5673 | 0.4928 | 0.2449 | 0.2473<br>0.4567 |

Table 3. Comparison with LLaMA2-7B on various benchmarks.

<span id="page-6-0"></span>

| Model      | PIQA  | SIQA  | BoolQ | WSC   | HellaSwag | ARC-E | ARC-C | MMLU  | NQ    | COPA  | Race-Middle | Avg   | Tokens |
|------------|-------|-------|-------|-------|-----------|-------|-------|-------|-------|-------|-------------|-------|--------|
| LLaMA2-7B  | 0.782 | 0.485 | 0.749 | 0.663 | 0.740     | 0.561 | 0.403 | 0.468 | 0.192 | 0.670 | 0.402       | 0.565 | 2000B  |
| DiJiang-7B | 0.775 | 0.346 | 0.626 | 0.683 | 0.694     | 0.626 | 0.427 | 0.407 | 0.194 | 0.730 | 0.618       | 0.557 | 40B    |

Among these comparison methods, Performer achieved the best results by approximating the original attention with Positive Random Features (PRF). However, as previously discussed, this Monte Carlo-based approximation method cannot achieve satisfactory outcomes, resulting in accuracy loss. By switching from Monte Carlo to the Quasi-Monte Carlo scheme using Positive Fixed Features (PFF) as described in Equation [9,](#page-3-0) we surpassed the accuracy of Performer but still fell short of the original vanilla Transformer's performance. Furthermore, by incorporating the Discrete Cosine Transform (DCT), our method achieves higher efficiency than approaches using PFF kernels. The DCT transformation enables a more compact and efficient representation of the frequency components of the attention mechanism. This efficiency stems from the DCT's ability to concentrate energy, allowing for a sparse representation that captures the most significant features of the data with fewer coefficients. Consequently, our approach not only closely approximates the original attention but also does so with improved computational performance compared to PFF-based methods. This advantage highlights the effectiveness of using DCT in optimizing the approximation of attention mechanisms, further underscoring the potential of our method in enhancing the efficiency of Transformer models. Further incorporating weighted Quasi-Monte Carlo, our DiJiang architecture ultimately achieved accuracy nearly identical to the original Pythia-400M, validating the efficacy of our approximation method. This demonstrates not only the potential of our approach for fine-tuning large-scale language models but also underscores the importance of choosing an efficient approximation strategy to maintain model performance.

We further visualized the training curves to showcase the approximation efficiency of different linear Transformer models, as depicted in Figure [2.](#page-6-2) RetNet, as an emerging language model architecture, has shown its potential by achieving significantly low loss values, underscoring its

![](_page_6_Figure_7.jpeg)

<span id="page-6-2"></span>Figure 2. Training Curve of different methods. The proposed method achieves the lowest PPL and the fastest converge speed.

capability for language tasks. Despite its low loss, RetNet does not necessarily outperform on benchmark metrics and, in some cases, even falls short of the results achieved by the Performer. This discrepancy highlights the importance and advantages of employing kernel methods to approximate the original attention computation, particularly in fine-tuning scenarios.

Our method demonstrates the fastest rate of loss reduction and ultimately achieves the lowest loss value. This rapid convergence indicates that our approach can quickly reach a performance level similar to that of the original Transformer. The visualization clearly underscores the superiority of our method in terms of both convergence speed and final model accuracy, validating our approach's effectiveness in efficiently approximating the attention mechanism

![](_page_7_Figure_1.jpeg)

Figure 3. Visualization of attention map of different architectures. The results are averaged by multiple heads.

![](_page_7_Figure_3.jpeg)

Figure 4. Comparison of inference memory and throughput between the proposed DiJIang and vanilla Transformer architecture.

while maintaining high performance standards. This visual evidence further solidifies our claim that our method stands out among linear Transformer alternatives, offering a compelling solution for optimizing Transformer models without compromising on quality.

#### <span id="page-7-1"></span>4.4. Comparison of Inference Cost

Furthermore, we also evaluated the memory usage and throughput of our method in comparison to the original Transformer model under various conditions. We selected the Pythia-410M model as our primary subject for analysis. We follow the implementation of RetNet [\(Sun et al.,](#page-9-10) [2023\)](#page-9-10) to efficient inference. The specific results, as depicted in Figure [4,](#page-7-0) demonstrate that as the token length increases, the memory footprint and inference speed of our model do not escalate. This observation is attributed to the linear complexity characteristic of our approach, indicating that our method is more conducive to long-sequence inference. In contrast, due to the quadratic complexity of attention computations, the original Transformer model experiences a continuous increase in both inference time and required memory as the token length grows. This comparison highlights the efficiency and practicality of our solution, particularly in scenarios involving extensive sequences where computational resources are a critical concern.

### 4.5. Visualization

<span id="page-7-0"></span>To further demonstrate the effectiveness of our model's approximation of the attention mechanism, we present attention maps generated by different methods in Figure [3.](#page-7-1) It is evident that the original Transformer's attention map (Figure [3](#page-7-1) (a)) is rich in information, laying the foundation for its robust capabilities. In contrast, attention maps produced by other linear attention methods such as Performer (Figure [3](#page-7-1) (b)) fail to adequately capture the relationships between tokens, resulting in maps that are dissimilar to those of the original Transformer and ultimately leading to decreased model accuracy, despite fine-tuning efforts. In contrast, our method (Figure [3](#page-7-1) (c)), by employing the weighted Quasi-Monte Carlo scheme, closely approximates the original attention mechanism. This allows it to effectively model the relationships between different tokens, achieving results nearly identical to those of the original Transformer but

with significantly faster inference efficiency. This comparison not only highlights the inadequacies of other linear attention methods in capturing token interdependencies but also showcases the superiority of our approach in accurately approximating attention while enhancing computational efficiency.

### 5. Conclusion

This paper introduces DiJiang, a groundbreaking Frequency Domain Kernelization method designed to address the computational inefficiencies inherent in traditional Transformer models. By leveraging linear attention mechanisms and a novel application of the weighted Quasi-Monte Carlo method for efficient sampling, our approach significantly reduces the necessity for extensive retraining. This is particularly beneficial for large language models, where the cost and time associated with training are substantial barriers to progress. The kernelization process, underpinned by Discrete Cosine Transform (DCT), not only diminishes the computational complexity but also ensures that the adaptation from a vanilla Transformer to a linear attention model incurs minimal training costs. Our extensive experiments validate that DiJiang achieves performance on par with conventional Transformers while reducing training costs by about 10x and enhancing inference speeds. This method represents a significant advancement in the development of efficient and scalable Transformer models, promising wider applicability and facilitating advancements in various tasks within the realm of natural language processing and beyond.

### Broader Impact

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

