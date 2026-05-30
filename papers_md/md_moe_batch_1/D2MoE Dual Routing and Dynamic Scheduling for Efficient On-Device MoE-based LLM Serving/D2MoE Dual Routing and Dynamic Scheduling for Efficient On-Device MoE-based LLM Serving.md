# <span id="page-0-0"></span>D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

[Haodong Wang](https://orcid.org/0009-0001-8977-850X) hwanghb@connect.ust.hk Hong Kong University of Science and Technology Hong Kong, China

Zicong Hong<sup>∗</sup> congcong@ust.hk Hong Kong University of Science and Technology Hong Kong, China

# Abstract

The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefits, MoE is still too expensive to deploy on resource-constrained edge devices, especially with the demands of on-device inference services. Recent research efforts often apply model compression techniques, such as quantization, pruning and merging, to restrict MoE complexity. Unfortunately, due to their predefined static model optimization strategies, they cannot always achieve the desired quality-overhead trade-off when handling multiple requests, finally degrading the on-device quality of service. These limitations motivate us to propose the D2MoE, an algorithm-system co-design framework that matches diverse task requirements by dynamically allocating the most proper bit-width to each expert. Specifically, inspired by the nested structure of matryoshka dolls, we propose the matryoshka weight quantization (MWQ) to progressively compress expert weights in a bit-nested manner and reduce the required

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. ACM MOBICOM '25, Hong Kong, China

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1129-9/2025/11 <https://doi.org/10.1145/3680207.3723493>

# Qihua Zhou<sup>∗</sup>

qihuazhou@szu.edu.cn College of Computer Science and Software Engineering, Shenzhen University Shenzhen, China

# Song Guo

songguo@cse.ust.hk Hong Kong University of Science and Technology Hong Kong, China

runtime memory. On top of it, we further optimize the I/Ocomputation pipeline and design a heuristic scheduling algorithm following our hottest-expert-bit-first (HEBF) principle, which maximizes the expert parallelism between I/O and computation queue under constrained memory budgets, thus significantly reducing the idle temporal bubbles waiting for the experts to load. Evaluations on real edge devices show that D2MoE improves the overall inference throughput by up to 1.39× and reduces the peak memory footprint by up to 53% over the latest on-device inference frameworks, while still preserving comparable serving accuracy as its INT8 counterparts.

# CCS Concepts

• Human-centered computing → Ubiquitous and mobile computing; • Computing methodologies → Artificial intelligence.

# Keywords

On-Device Inference, Mixture of Experts, Large Language Models Serving

#### ACM Reference Format:

Haodong Wang, Qihua Zhou, Zicong Hong, and Song Guo. 2025. D <sup>2</sup>MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving. In The 31st Annual International Conference on Mobile Computing and Networking (ACM MOBICOM '25), November 4–8, 2025, Hong Kong, China. ACM, New York, NY, USA, [15](#page-14-0) pages.<https://doi.org/10.1145/3680207.3723493>

# 1 INTRODUCTION

The Large Language Models (LLMs) [\[17,](#page-13-0) [37,](#page-14-1) [38,](#page-14-2) [44\]](#page-14-3) are increasingly embedded in our work and everyday activities, supporting tasks like summarization, code completion, and decision making [\[2,](#page-13-1) [5,](#page-13-2) [12,](#page-13-3) [28\]](#page-14-4). To further scale these models without incurring prohibitive training and inference costs,

<sup>∗</sup>Corresponding author.

![](_page_1_Figure_2.jpeg)

Figure 1: Traditional MoE single routing (expert ID only) vs. our D2MoE dual routing (ID and bit-wdith).

recent research has introduced Mixture-of-Experts (MoE) models, where feedforward network (FFN) layers are often replaced with MoE layers. These MoE layers enable sparse activation for each token, dynamically routing computation to a subset of model components (i.e., experts). As model sizes continue to grow and task complexity increases, the sparse activation architecture of MoE models becomes increasingly critical. However, compared to other LLMs, MoEbased models face more severe memory bottlenecks during inference due to their large parameter sizes. This challenge is particularly pronounced in resource-constrained edge environments. For instance, when stored in a compact float16 format, the parameters of Mixtral 8×7B [\[17\]](#page-13-0) require over 90GB of memory, whereas consumer-grade GPUs such as the NVIDIA RTX 3060 laptop offer only 6GB of memory, highlighting the resource limitations of edge devices.

Quantization is one of the most prominent solutions for memory optimization during LLM inference on edge devices. Applying quantization to MoE models is particularly effective, as experts account for almost 90% of the parameters (e.g., in Mixtral-8×7B, experts contribute 89.9% of the parameters, while attention mechanisms account for less than 11.1%). Consequently, quantizing experts significantly reduce memory usage and the size of parameter transfers between storage hierarchies during inference. Recent efforts have proposed quantization strategies tailored to MoE-based models, where bit-width is assigned to different experts based on their importance on calibration dataset and fix bit-width assignments throughout inference. For instance, EdgeMoE [\[42\]](#page-14-5) assigns bit-width offline by quantifying the accuracy loss of experts on a calibration dataset, while MC-MoE [\[16\]](#page-13-4) uses expert activation frequency and confidence to design a fixed bit-width allocation strategy during inference.

However, due to the dynamic sparse activation characteristic of MoE-based models, the importance of an expert can vary with different input tokens. Inspired by related work on dense models [\[13,](#page-13-5) [31\]](#page-14-6), a straightforward approach involves

introducing a lightweight, trainable adapter before each transformer block to dynamically decide whether to skip the current block's computation. This approach significantly reduces overall FLOPs and accelerates inference. Specifically, the adapter dynamically allocates more computational blocks to important tokens while skipping less significant tokens, optimizing computational efficiency. Nevertheless, incorporating dynamic bit-width selection for activated experts inevitably increases memory overhead. This is because the weights for different bit-width in existing quantization methods are stored independently. For instance, in LLaMA-MoE [\[46\]](#page-14-7), when quantized using llama.cpp [\[11\]](#page-13-6), INT4 experts require 3.81 GB of memory, while storing INT2/3/4 experts simultaneously demands 9.62 GB. This significantly increases the memory footprint in resource-constrained edge devices.

Although parameters of inactive experts can be offloaded to lower-tier storage units (e.g., CPU or SSD), the I/O latency associated with quantized experts remains substantial (e.g., at Mixtral ×7B, INT4 quantization experts still account for more than 70% of the parameters), leading to significant computational bubbles caused by waiting for I/O. For example, on an NVIDIA RTX 3060, the average computation time for a single expert in LLaMA-MoE is 3.1 ms, whereas the average data transfer time is approximately 20 ms. Existing methods attempt to overlap I/O and computation by processing multiple batches simultaneously [\[3,](#page-13-7) [14\]](#page-13-8). However, due to the token variability inherent in gating mechanisms, the total number of activated experts and their associated bit-width can increase significantly. Furthermore, when the gating mechanism selects more than one expert per token (as in Mixtral-8×7B and LLaMA-MoE), the I/O overhead for transferring expert parameters is multiplied, causing GPU to experience more frequent idle periods while waiting for expert parameters.

To tackle these challenges, we propose a dually sparselygated Mixture-of-Experts paradigm. Specifically recognizing that expert importance varies dynamically with different tokens, we first introduce an end-to-end fine-tuning strategy for the gating network, called token-adaptive bit-width selection. This strategy enables dynamic bit-width decisions for each token, achieving a better trade-off among model accuracy and peak memory usage for MoE-based LLMs. Second, we present matryoshka weight quantization (MWQ), which compresses expert weights into a structure where bit-width can be shared hierarchically. In this approach, higher bitwidth always encapsulates lower bit-width, resembling the nested structure of matryoshka dolls.[1](#page-0-0) Finally, we employ a bit-width-aware I/O-compute pipeline that dynamically reorganizes the I/O and computation order for different bit-width.

<sup>1</sup>matryoshka dolls are a set of wooden dolls of decreasing size, nested within one another.

This pipeline processes multiple requests in batches, optimizing the parallel efficiency of expert I/O and computation. Together, these innovations improve the overall performance and efficiency of MoE-based LLMs, particularly in resourceconstrained environments.

In this paper, based on the above paradigm, we propose D2MoE, a novel MoE-based LLM dynamic quantization framework that can determine the bit-width of each expert online dynamically for expedited, resource-efficient, and highquality on-device inference. The main contributions are summarized as follows.

- We propose a dually sparsely-gated MoE paradigm that leverages the varying importance of experts across tokens to dynamically allocate expert bit-width. This approach aims to minimize peak memory usage and reduce the I/O overhead for experts.
- We design MWQ, a multi-step quantization method that nests high-bit-width expert weights into low-bit-width weights to reduce redundant memory usage.
- We propose a fine-grained expert I/O-compute pipeline paradigm to minimize bubbles between expert I/O and computation of different bit-width and design a hottest-expert-bitfirst algorithm to heuristically formulate execution plans for this paradigm.
- We implement D2MoE and conduct extensive experiments on various edge devices (e.g., NVIDIA RTX 3060, NVIDIA Jetson AGX Orin 64G) and MoE-based LLMs. Experimental results demonstrate that D2MoE achieves up to a 1.39× throughput under different memory budgets compared to state-of-the-art quantization frameworks [\[19,](#page-13-9) [42\]](#page-14-5).

# 2 BACKGROUND AND MOTIVATION

# 2.1 On-Device Memory Constraints

While many existing optimizations have focused on dynamically reducing the computational cost [\[32\]](#page-14-8), memory is the real bottleneck for on-device MoE-based LLM inference. Unlike computation, which may only slow down the inference, memory is often a hard constraint that directly determines whether it is feasible to run the model. This memory constraint manifests MoE-based LLMs in the following aspects:

Hardware. The growth of memory capacity in edge devices significantly lags behind that of high-performance data centers in the cloud. For instance, the memory capacity of NVIDIA's high-performance GPUs has increased nearly 90×, from the Tesla P100 in 2016 to the B200 in 2024. In contrast, the memory capacity of smartphones has only grown 6×, from the iPhone 6 in 2014 to the iPhone 15 in 2023.

Model. The memory requirements of MoE-based LLMs typically scale linearly with their model capacity [\[8\]](#page-13-10). Consequently, the memory demands of MoE-based LLMs have grown rapidly due to the expansion of model size and the

<span id="page-2-0"></span>Table 1: Trade-offs among different bit-width.

| Bit-width | mem. (GB) | lat. (token/s) | acc. (ppl ↓) |
|-----------|-----------|----------------|--------------|
| 2         | 3.04      | 50.47          | 20.95        |
| 3         | 3.80      | 45.91          | 15.10        |
| 4         | 4.48      | 43.82          | 14.72        |
| 5         | 5.10      | 40.15          | 14.63        |
| 6         | 5.60      | 37.72          | 14.62        |
| 8         | 7.24      | 35.34          | 14.55        |
| 16        | 13.60     | 23.45          | 14.55        |

need for higher performance. MoE-based LLMs, such as Switch Transformer and Mixtral, leverage multiple expert networks to enhance model capacity, which results in a substantial increase in memory consumption. For example, deploying one of the state-of-the-art MoE models, Mixtral-8×7B requires at least 90GB of memory, nearly 10× the memory capacity of the most advanced edge devices.

To bridge the gap between the limited memory capacity of edge devices and the high memory requirements of MoE-based LLMs, it is imperative to design an inference framework that prioritizes optimizing memory.

# 2.2 Observation

The above analysis highlights the necessity of optimizing memory usage by quantizing the model expert weights in edge environments. This section investigates the potential advantages of dynamically adjusting the expert bit-width to align with hardware characteristics, a pivotal consideration in the design of D2MoE.

Observation #1: Different bit-width in MoE-based LLM quantization bring different benefits in terms of accuracy-memory-latency. In most cases, the bit-width of quantization weights are predefined hyper-parameters that remain fixed during model inference. However, in MoE-based LLMs, varying the bit-width of different experts can provide distinct advantages in terms of accuracy, memory usage, and inference latency. As illustrated in Table [1,](#page-2-0) the LLaMA-MoE model [\[46\]](#page-14-7) was evaluated with various quantization bitwidth on an RTX 4090 GPU with 24GB of memory using the llama.cpp [\[11\]](#page-13-6) inference framework. The figure demonstrates the impact on memory footprint, latency, and model accuracy under different quantization settings. Specifically, quantizing the model to INT2 compared to INT8 reduces the memory footprint by 58% and improves latency by 30%, but the model accuracy drops drastically by 43%.

Summary: This highlights the importance of incorporating bit-width into the expert selection process for quantized MoEbased LLMs. This consideration is crucial for optimizing model performance in terms of accuracy, memory usage and latency.

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 2: Accuracy loss of expert quantization to INT1 across 10 samples from the Hellaswag dataset.

Observation #2: The importance of experts changes dynamically according to different input samples. Many LLM quantization methods have found that the model shards (e.g. layers and experts) show different importance to the accuracy of the model, and in order to ensure accuracy and reduce redundant I/O, they allocate higher bit-width to critical slices through offline profiling [14, 39, 42]. However, we observe that the importance of different experts changes dynamically with the input sample. As shown in Figure 2, we quantize expert 4 in layer 1 and expert 2 in layer 25 to INT1 while keeping other experts' bit-width unchanged. We evaluate the accuracy loss (expert importance) of LLaMA-MoE-3.5B and Mixtral 8×7B across 10 samples from the Hellaswag dataset. Our results reveal significant variability in precision loss across samples and even individual tokens. For instance, quantizing the  $4^{th}$  expert in the  $1^{st}$  layer to 1-bit results in a 0.5% accuracy drop on sample 1 for LLaMA-MoE-3.5B and a 0.2% drop for Mixtral 8×7B.

**Summary:** This highlights that the importance of each expert varies for different input samples, and thus, dynamically adjusting the expert bit-width to find the optimal setting for the activated experts is crucial for enhancing model performance.

Observation #3: Large bubble between I/O and computation of quantized experts led to substantial inference delays. Due to the limited memory capacity of edge devices, we adopt an on-demand method for loading activated quantized experts from disk during inference. However, as shown in Figure 3 the existing approach of loading experts in ascending order of expert IDs introduces significant bubbles between I/O and computation, leading to increased inference latency, particularly when request numbers exceeds 25. For instance, in LLaMA-MoE-3.5B with 32 requests, the expert I/O time is 2.6s, the computation time is 2.04s, and the total inference latency for the expert layer is 3.55s, which is 1.36× and 1.74× of the I/O and computation times.

**Summary:** This highlights that the I/O-compute pipeline paradigm for quantized experts create significant inefficiencies. There is a pressing need for designing scheduling plan aimed at minimizing bubbles during inference.

<span id="page-3-1"></span>![](_page_3_Figure_8.jpeg)

Figure 3: Comparison of expert I/O, computation, and inference latency with different request numbers.

### 2.3 Technical Challenges

Despite the insight that the dynamic expert bit-width routing policy is intuitive, there are still several challenges associated with implementing  $D^2MoE$  in complex edge environments. Challenge #1. Unbalanced and inefficient bit-width selection load. Edge devices, constrained by limited computational resources, require a trained lightweight gating network to dynamically select the appropriate bit-width for each expert without introducing significant computational overhead. The training of this gating network poses significant challenges, primarily due to two fundamental issues. Firstly, there is an imbalance in the selection of expert bit-width, as evidenced by the consistent selection of the same bit-width by numerous tokens for a specific expert [8]. Secondly, there is an irrational allocation of bit-width, where the selected expert bit-width is unable to ensure model accuracy [32].

Challenge #2. High memory overhead to store multiple versions of quantized experts. Limited memory on edge devices is also one of the main bottleneck constrain inference performance [14, 21]. If the basic LLM quantization [22, 41] is used, multiple quantization models with different bit-width versions have to be deployed, further exacerbating the high memory costs associated with LLM deployment. Therefore, to make D<sup>2</sup>MoE memory-efficient, a new quantization method should be devised to avoid storing multiple versions of different bit-width.

Challenge #3. Significant runtime overhead on weight dequantization operations. During model inference, the weight-only quantization approach requires the online transformation of quantized weights into the same data type as the activation for matrix computation. This dequantization operation introduces significant runtime overhead, typically accounting for 20%-70% of the entire inference process. To minimize the time spent on dequantization, it is crucial to design specific dequantization kernels tailored to the quantization method and aligned with the hardware characteristics. Challenge #4. Lightweight and efficient online scheduling strategies address the large parallelism bubble

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 4: The architecture overview of D<sup>2</sup>MoE.

between I/O and computation. Multi-requests are inherently heterogeneous and unpredictable in terms of resource and latency requirements, posing considerable challenges to LLM serving. Existing methods address these challenges by focusing on model placement policies and adaptive batch scheduling to achieve I/O-compute parallelism and reduce response latency [20, 35]. However, quantized LLMs serving must also consider the varying bit-width selections for different requests, complicating the design of a lightweight scheduling algorithm and parallel strategy.

#### 3 D<sup>2</sup>MoE SYSTEM DESIGN

#### 3.1 System Overview

D<sup>2</sup>MoE is the first execution engine designed to enable fast inference of quantized MoE-based LLMs on edge devices. As illustrated in Figure 4, D<sup>2</sup>MoE operates through two primary stages: the offline preprocessing phase and the online execution phase. The offline preprocessing phase, which is executed once prior to deployment, comprises two key modules: 1 token-adaptive bit-width selection and 2 matryoshka weight quantization. Initially, the token-adaptive bit-width selection module optimizes the bit-width allocation for different experts. Specifically, a lightweight plug-in network is trained for each expert using a generic dataset (e.g. C4 dataset) to dynamically select the bit-width for each token to achieve accuracy-memory-latency resources optimization. Following this, the D<sup>2</sup>MoE profiler applies the MWQ module to the MoE-based LLM using a small calibration dataset, effectively reducing the model's memory footprint.

In the online execution phase, the fine-tuned, quantized MoE-based LLMs from the offline preprocessing phase are deployed onto physical edge devices. The  $\rm D^2MoE$  engine then

<span id="page-4-1"></span>![](_page_4_Figure_9.jpeg)

Figure 5: Comparison between fixed and dynamic bitwidth allocation.

implements ③ the bit-width-aware I/O-compute pipeline to manage and schedule the execution of various requests in real-time which effectively minimizes the significant idle periods between I/O and computation, thereby enhancing overall efficiency.

#### 3.2 Token-Adaptive Bit-Width Selection

Inspired by MoD [31], the network can identify tokens critical to accuracy and assign higher bit-width activation experts accordingly. Moreover, dynamic bit-width selection aims to minimize peak memory footprint by making real-time decisions during inference. Our approach further reduces expert memory consumption while maintaining accuracy. As shown in Figure 5, traditional methods (left) quantize experts to a hybrid (e.g., INT2/3/4) bit-width offline and keep it static during inference. In contrast, the proposed token-adaptive bit-width selection (right) achieves the same output quality with significantly lower memory usage through dynamic bit-width allocation, enabling more efficient inference.

It involves 2 steps: (1) *quantized expert capacity* that balances the selection frequency of each bit-width during fine-tuning, and (2) *dynamic bit-width selection loss* that optimizes the router to dynamically allocate bit-width based on the quantized expert capacity.

**Quantized expert capacity** constrains the token capacity of each expert during fine-tuning to prevent overfitting to specific token sequences. Specifically, given that the total number of tokens processed by each transformer block is T, we define the quantized expert capacity as  $\{c_k\}_{k=1}^K$ , where  $\sum_{k=1}^K c_k = 1$ . This formulation indicates that the maximum number of tokens assigned to the k-th bit-width expert during each forward propagation in fine-tuning is  $c_k \cdot T$ . Any tokens exceeding this capacity will skip the computation of the corresponding expert. For example, if the total number of tokens is 60 and the capacity for a particular bit-width is 0.2, then this expert can process at most 12 tokens during fine-tuning. If 14 tokens are assigned to this expert in a forward pass, 2 tokens will be randomly dropped skipping computation for these tokens at this layer.

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Figure 6: The workflow of MWQ.

The values of  $\{c_k\}_{k=1}^K$  are predefined based on hardware constraints to optimize memory and computational efficiency and remain fixed during fine-tuning. In addition, a balanced allocation across bit-width is crucial, as higher bit-width increase memory consumption, while lower bit-width may compromise accuracy.

**Dynamic bit-width selection loss** is introduced to finetune the bit-width router for selecting the optimal bit-width for activated experts. A lightweight, trainable bit-width router is placed before each expert to dynamically allocate bit-width, ensuring that higher bit-width are used for computing the most critical tokens by appropriately adjusting the logits. However, unlike the expert router, the bit-width router primarily focuses on model accuracy, which may lead to consistently favoring high bit-width due to their typically lower accuracy loss. To address this, we propose a novel bit-width balancing loss that complements the model accuracy loss to balance the selection frequency of different bit-width. Specifically, given a list of candidate bit-width experts  $(\{b_k\}_{k=1}^K)$  and a batch  $\mathcal S$  containing T tokens in a forward propagation, the total loss function can be described as:

$$Loss = \frac{1}{T} \sum_{x \in \mathcal{S}} \left( CE(p(x), q(x)) + \frac{\alpha}{L} \sum_{l=1}^{L} \sum_{k=1}^{K} p_k^l(x) b_k \right). \quad (1)$$

where  $p_k^l(x)$  represents the probability fraction assigned by the bit-width router to the k-th bit-width expert at layer l, p(x) and q(x) denotes the logits of the  $D^2MoE$  model and the original precision model (e.g., FP16) for token x after the LM head layer, respectively.

In this loss function, the first term is the cross-entropy loss, which encourages the bit-width router to prioritize higher bit-width. The second term serves as a regularization term, promoting the selection of lower bit-width to achieve

<span id="page-5-1"></span>![](_page_5_Figure_9.jpeg)

Figure 7: An example of MWQ for the weight matrix W with  $b_1 = 2$  and K = 3.

a balance. Therefore, the bit-width router dynamically selects different bit-width during inference while maintaining overall model accuracy.

#### 3.3 Matryoshka Weight Quantization

Token-adaptive bit-width selection effectively reduces memory overhead during inference. However, traditional quantization methods typically require storing multiple quantized versions at different bit-width, resulting in significant storage overhead. To address this challenge, inspired by the nested structure of Russian matryoshka dolls, we propose a novel multi-step quantization technique, MWQ which restructures expert weights into a nested hierarchy, embedding low bit-width weights within high bit-width weights. In the following sections, we first introduce the MWQ quantization algorithm, followed by the design of its corresponding dequantization kernel tailored to this technique.

3.3.1 Quantization Method. Figure 6 illustrates the multistep process of the proposed MWQ technique leveraging a list of candidate bit-width  $(\{b_k\}_{k=1}^K)$ . The process begins by quantizing the weights to the minimum supported bit-width (e.g., INT2 or INT4, denoted as  $b_1$ ) using asymmetric quantization. Subsequently, the bit-width is iteratively increased by quantizing the residual weights through binary residual quantization until the final  $b_K$  bit-width weight is obtained. In each step, transitioning from  $b_k$  to  $b_{k+1}$ , one additional bit-width is added along with the related scale factors.

Asymmetric Quantization. Sparse expert weights have been shown to be robust to asymmetric quantization, particularly under low bit-width settings [19]. During inference, tensors are dequantized to FP16 to enable matrix multiplication with activations. To mitigate precision loss, we first

apply per-group asymmetric  $b_1$  bit-width quantization. Quantization and dequantization are computed as follows:

$$\mathbf{Q}_{\mathbf{W}_{b_1}} = round(\mathbf{W}/\mathbf{s}_{b_1} + \mathbf{z}_{b_1}), \hat{\mathbf{W}}_{b_1} = (\mathbf{Q}_{\mathbf{W}_{b_1}} - \mathbf{z}_{b_1}) \cdot \mathbf{s}_{b_1}, (2)$$

where  $\mathbf{W} \in \mathbb{R}^{s \times h}$  represents the floating-point weight tensor,  $\mathbf{Q}_{\mathbf{W}_{b_1}} \in \mathbb{R}^{s \times h}$  is the quantized weight tensor, and  $\mathbf{z}_{b_1}, \mathbf{s}_{b_1} \in \mathbb{R}^{s \times h/g}$  are the zero points and scale factors for group-wise quantization. These are optimized as follows:

$$\arg\min_{\mathbf{z}_{b_1},\mathbf{s}_{b_1}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}_{b_1}\mathbf{X}\|_2^2, \tag{3}$$

where  $\mathbf{X} \in \mathbb{R}^{h \times r}$  denotes the floating-point activation tensor, and g is the group size. To compute weights for higher bitwidth, we quantize the residuals  $\mathbf{R}_{b_1} = \mathbf{W} - \hat{\mathbf{W}}_{b_1}$ .

**Binary Residual Quantization.** To ensure that low bitwidth weights are subsets of higher bit-width weights while maintaining accuracy, we progressively apply per-group quantization with the binary residual approximation based on the  $b_1$  bit-width quantized residuals. The binary residual quantization and dequantization are computed as:

$$\mathbf{Q}_{\mathbf{W}_{b_{k}}} = round(\mathbf{R}_{b_{k-1}}/\mathbf{s}_{b_{k}}), \hat{\mathbf{Q}}_{\mathbf{W}_{b_{k}}} = \mathbf{s}_{b_{k}} \cdot \mathbf{Q}_{\mathbf{W}_{b_{k}}}, \tag{4}$$

where  $k=2,\cdots,K$ ,  $\mathbf{Q}_{\mathbf{W}_{b_k}} \in \{+1,-1\}^{s \times h}$  represents the accumulated one-bit weights from  $b_{k-1}$  to  $b_k$ , and  $\mathbf{s}_{b_k}$  is the per-group scale factor optimized as:

$$\arg\min_{\mathbf{s}_{b_k}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}_{b_k}\mathbf{X}\|_2^2, \tag{5}$$

where  $\hat{\mathbf{W}}_{b_k} = \hat{\mathbf{W}}_{b_1} + \sum_{i=b_2}^{b_k} \mathbf{s}_{b_i} \mathbf{Q}_{\mathbf{W}_{b_k}}$  is the floating-point approximation of  $b_k$  bit-width quantized weights. By iteratively adding low bit-width quantized weights, arbitrary bit-width quantized weights can be constructed.

For example, as shown in Figure 7, we first apply asymmetric quantization (group size = 6) to obtain INT2 weight  $\mathbf{Q}_{\mathbf{W}_2}$ . Next, the residual  $\mathbf{R}_2$  is quantized via binary residual quantization to obtain an additional 1-bit weight, forming the INT3 weight by combining  $\mathbf{Q}_{\mathbf{W}_2}$  and  $\mathbf{Q}_{\mathbf{W}_3}$ . This process is repeated to generate another quantized weight  $\mathbf{Q}_{\mathbf{W}_4}$ , resulting in INT4 weight as the sum of all previous quantized weights.

Inspired by GPTQ [9], we further enhance the efficiency of post-training quantization by retaining only block-level compensation while eliminating column-level error corrections, ensuring the effectiveness of the MWQ strategy. Algorithm 1 provides a detailed outline of the complete MWQ process.

3.3.2 Dequantization Kernel. In per-group quantization, balancing accuracy enhancement with dequantization overhead is critical, yet prior studies have not facilitated efficient GEMM parallelism on GPU for dynamic bit-width. The primary performance constraint in executing dequantization for MWQ on edge device is the limited parallelism between tensor loading from various storage levels in the GPU and

#### Algorithm 1: Main algorithm of MWQ

```
Input: Weight tensor \mathbf{W} \in \mathbb{R}^{s \times h}, input tensor
                        \mathbf{X} \in \mathbb{R}^{h \times r}, block size \gamma, Hessian regularizer \lambda.
  1: \mathbf{H}^c := Cholesky((2\mathbf{X}\mathbf{X}^T + \lambda \mathbf{I})^{-1})
 2: \mathbf{Q}_{\mathbf{W}_{b_i}} := \mathbf{0}_{s \times h}, i = 1, \cdots, K
 3: for i < K do
               for b = 0, \gamma, 2\gamma, \cdots do
 4:
                        \mathbf{W}^b := \mathbf{W}_{:b:b+v}
 5:
                        if i = 1 then
  6:
                          7:
  8:
  g.
                          \begin{vmatrix} \mathbf{Q}_{\mathbf{W}_{b_i}:,b:b+\gamma} := \text{res\_quant}(\mathbf{R}_i^b) \\ \mathbf{R}_{b_i}^b := \mathbf{R}_{b_{i-1}}^b - \hat{\mathbf{W}}_{b_i}^b \end{vmatrix}
10:
                       \mathbf{E} := (\mathbf{W}_{:,b:b+\gamma} - \mathbf{Q}_{\mathbf{W}_{b;:,b:b+\gamma}}) / \mathbf{H}_{b:b+\gamma,b+\gamma}^c
12:
                        \mathbf{W}_{:,b+\gamma:} := \mathbf{W}_{:,b+\gamma:} - \mathbf{E} \cdot \mathbf{H}_{b:b+\gamma,b+\gamma:}^{c}
       Output: \{\mathbf{Q}_{\mathbf{W}_{b_i}}\}_{i=1}^K
```

<span id="page-6-1"></span>![](_page_6_Figure_17.jpeg)

Figure 8: The dequantization overview of D<sup>2</sup>MoE.

computations within the CUDA cores and Tensor cores. To address this, we have developed a parallel loading dequantization kernel that optimizes all levels of GPU storage.

This approach leverages a key innovation: fully overlapping tensor loading with tensor computations to simultaneously maximize bandwidth usage and computation throughput. Our method achieves loading parallelism by dynamically transferring quantized data from disk directly to GPU's global memory, concurrently with activations moving from global memory to L2 cache. For computation parallelism, as illustrated in Figure 8, expert dequantization in the CUDA cores is synchronized with expert computation in the Tensor core. Notably, traditional bit-transpose methods from various integer formats to FP16 are inefficient; we instead employ an optimized binary operation from the Any-Precision LLM [29], significantly enhancing processing speed.

### 3.4 Bit-Width-Aware I/O-Compute Pipeline

The nested structure of MWO quantization reveals a limitation in the existing I/O-compute pipeline paradigm, which fails to account for scenarios where experts with different bit-width are invoked across multiple requests, leading to significant parallel bubbles. To be specific, Figure 9 compares four distinct scheduling paradigm. The traditional I/Ocompute execution paradigm, which does not employ MWQ, sequences the I/O and compute queue of the expert module in ascending order by expert IDs and bit-width (Figure 9a). MWQ reduces the I/O size of experts by nesting low bit-width weights within high bit-width weights, thereby increasing the utilization frequency of low bit-width weights. For instance, if three requests select Expert 2, with one selecting INT2 and two selecting INT3, MWO ensures that all three requests call the INT2 weight (light blue, Expert 2), while the two requests requiring INT3 further call the medium blue weight (Expert 2). This nesting improves parallel efficiency by reusing low bit-width weights. However, due to sequential execution, significant parallel bubbles still occur (Figure 9b). Furthermore, It has been demonstrated that MWO is capable of performing expert I/O and computation scheduling at a fine-grained bit-width level. (Figure 9c). Ultimately, the optimal schedule (Figure 9d) minimize parallel bubbles by determining the execution order of experts at a fine-grained bit-width level during inference. Therefore,  $D^2$ MoE employs the bit-width-aware I/O-compute pipeline paradigm to reorder the activated experts with different bit-width, thereby reducing I/O wait time. In the following, the memory budget scheduler is introduced with the aim of reducing the frequency of expert I/O. Secondly, the bit-width-aware pipeline problem will be formulated, and then the Hottest-Expert-Bit-First (HEBF) algorithm will be introduced as a solution to the pipeline problem.

3.4.1 Memory Budget. To support MoE-based LLM inference in edge environments with dynamic memory constraints and reduce frequently loading experts, we introduce a memory budget M during expert I/O-compute pipeline. This parameter defines the upper limit of GPU memory allocated to experts and is configurable based on the available memory resources of edge hardware. Increasing the parameter M enables low bit-width weights, which are activated with greater frequency, to remain in GPU memory. Therefore, the necessity for frequent reloading is reduced.

As shown in Algorithm 2, at each layer, we first check whether the memory required by the current expert exceeds the available budget M (line 3). If the memory is sufficient, the pipeline of bit-width-aware I/O and computation is executed directly (line 9), followed by an update of the memory budget (line 10). If the memory is insufficient, high bit-width expert weights are released to free memory (lines 4-6). If the budget

<span id="page-7-0"></span>![](_page_7_Figure_6.jpeg)

Figure 9: Comparison between different I/O-compute parallel strategies.

```
Algorithm 2: Memory-Budget Scheduler
   Input: Generate length n, number of layers L,
           number of bit-width K, available memory
           budget M.
1: for i < n do
       for j < L do
           if layers[j] > M then
 3:
               \mathbf{for}\ k = 0\ \mathbf{to}\ K - 1\ \mathbf{do}
 4:
                    Free(layer[j-1][k])
 5.
                   Update M
 6:
               if layers[j] > M then
 7:
                   Free(layer[j-1][1])
 8:
           Load and Store (layer[j])
 9:
           Update M
10:
```

remains inadequate, low bit-width weights are also released as needed (lines 7-8). Finally, the pipeline is executed, and the memory budget is updated accordingly (lines 9-10).

- 3.4.2 Offline Profiling.  $D^2MoE$  measures the following hardware capabilities of the edge device at installation time.
- $T_{io}(b_k)$ :  $D^2$ MoE measures the average disk access delay for loading one expert in  $b_k$  bit-width, where  $b_k \in \{b_k\}_{k=1}^K$ . It only has to measure one expert per bit-width because all others have the same amount of parameters.
- $T_{comp}(b_k)$ : D<sup>2</sup>MoE calculates the average computation delay by measuring the dequantization delay for an expert with bit-width  $b_k$  and the execution delay for processing

a token. As the sizes of expert weights are deterministic, measuring a single expert per bit-width suffices.

The delays can be recorded offline and subsequently replayed at runtime because they are data-independent [15] and consistently determined by the bit-width.

*3.4.3 Parallelism Planning.* : In order to minimize the inference latency while satisfying the constraints, the goal of parallelism planning is to find an optimal I/O-Compute execution queue.

Variables. For each transformer block l, the set  $\Omega_l$  represents the execution queue, encompassing all the experts' bit-width indices selected. The matrix  $B_{j,k} \in \mathbb{R}^{N \times K}$  indicates the number of times the k-th bit-width of the j-th expert was selected, where N and K respectively denote the total number of experts and bit-width. For every  $s \in \Omega_l$  L(s, j, k) and C(s, j, k) specify the start time of the s-th quantized expert in the execution queues and the index of this quantized expert is the kth bit-width of the jth expert. The terms  $T_{io}(b_k)$  and  $T_{comp}(b_k)$ , previously defined, apply here as well.

**Objective.** Given that inference latency is influenced by bubbles during the parallel execution of tasks in the I/O-compute queues, we define our latency target as the difference between the total time overhead required to complete the compute queue and the load queue:

$$\min \sum_{j=1}^{N} \sum_{k=1}^{K} \left( B_{j,k} T_{comp}(k) - \sigma(B_{j,k} > 0) T_{io}(k) + \sum_{s \in \Omega_{I}} T_{wait} \right)$$

s.t. 
$$L(s+1, j, k) \le C(s, j, k), \forall s \in \Omega_l$$
 (6a)

$$L(s, j, k) \le L(s, j, k+1), \forall k \in \{1, \dots, K\}$$
 (6b)

$$T_{wait} = C(s, j, k) - C(s - 1, j, k) - B_{j,k} T_{comp}(k),$$
 (6c)

where if  $B_{j,k} > 0$  holds,  $\sigma(B_{j,k} > 0) = 1$ , otherwise  $\sigma(B_{j,k} > 0) = 0$ . Constraint (6a) ensures that computation begins only after the loading of the *s*-th quantized expert is complete. Constraint (6b) stipulates that each quantized expert should be loaded sequentially by increasing bit-width, thereby maximizing the reuse of experts with lower bit-width. Constraint (6c) describes how the *s*-th quantized expert waits in the queue until the I/O queue has finished loading.

**Solution.** Although the above problem can be solved using integer linear programming or dynamic programming, doing so online for every token at each expert layer introduces substantial inference delays. To address this, we propose the HEBF algorithm, which prioritizes I/O and computation for experts with higher activation frequencies. Frequently activated experts typically have longer computation times, allowing their execution to overlap with subsequent expert loading, thereby minimizing idle periods. The algorithm proceeds as follows: 1. Construct a queue  $Q_i$  for each expert, sorted in ascending order of bit-width; 2. Pop the

bit-width from the "head" of all expert queues and enqueue the element with the highest frequency into the I/O queue; 3. Sequentially load bit-width experts from the I/O queue and begin computation upon completion of loading. The HEBF algorithm satisfies key constraints: it prioritizes low bit-width experts first (Constraint (6a)), minimizes waiting time by overlapping computation with loading (Constraint (6b)), and ensures that loading completes before computation begins (Constraint (6c)).

#### 4 IMPLEMENTATION

We have fully implemented a prototype system of D<sup>2</sup>MoE with over 2,500 LOC in Python and CUDA in total atop Py-Torch. We use PyTorch's *triton* library [36] for I/O-compute parallel programming, and our CUDA programming is based on NVIDIA Ampere and Ada Lovelace architecture. Our approach focuses on the general process of data loading and MoE-based LLM inference, making it easily adaptable to other frameworks, such as TensorRT[27] and vLLM [20].

#### **5 EVALUATION**

#### 5.1 Experimental Setup

**Models and Datasets.** We evaluate D<sup>2</sup>MoE through two popular decoder-only MoE-based sparse LLMs: LLaMA-MoE-3.5B [46] and Mixtral 8×7B [17] have 8 experts per layer and utilize Top-2 routing, meaning that 2 experts are activated per layer during inference. The pre-trained weights for these models were directly obtained from Hugging Face. Moreover, we use C4 dataset [30] as the training data consists of 2048 random 2048 token segments to train bit-width routers and the calibration data consists of 128 random 2048 token segments to implement MWQ.

Metrics. We focus on model accuracy and throughput under different memory budget for D<sup>2</sup>MoE and the baselines. To evaluate model accuracy, we assess language generation performance by reporting perplexity on WikiText2 [26], and measure zero-shot performance on several popular benchmarks, including PIQA [1], ARC [7], BoolQ [6], HellaSwag [43], and Winogrande [33], using the lm-evaluation-harness [10]. For throughput, both the input and output lengths are 128. Hardware. We evaluate D<sup>2</sup>MoE on two prominent kinds of edge devices, as shown in the Table 2. The offline preprocessing phase of D<sup>2</sup>MoE, including fine-tuning the bit-width routers and MWQ, is carried out on a GPU server equipped with NVIDIA RTX 2×A6000.

**Baselines.** We compare D<sup>2</sup>MoE with two baselines and three state-of-the-art on-device MoE-based LLM inference frameworks: (1) *Hold-in-Memory*: This method quantizes all experts to INT8 and assumes that all model weights hold in GPU memory. Since INT8 quantization is nearly lossless in

<span id="page-9-0"></span>Table 2: Hardware environments for evaluation.

| Hardware  | Environmer           | nt 1   | Environment 2    |             |  |
|-----------|----------------------|--------|------------------|-------------|--|
|           | Device               | Memory | Device           | Memory      |  |
| GPU       | NVIDIA RTX 3060      | 6GB    | Jetson AGX Orin  | 64GB        |  |
| CPU       | Intel Core i7-11800H | 32GB   | ARM Cortex-A78AE | (SoC share) |  |
| Disk      | Samsung 970 EVO      | 1T     | Samsung 970 EVO  | 1T          |  |
| Disk Read | 3.5 GB/s             |        | 3.5 GB/          | s           |  |

terms of accuracy, we use this method as a baseline for evaluating model accuracy. However, it is not memory-efficient. (2) Matryoshka-Free: This method uses GPTQ [9] to quantize all expert into multiple versions INT2/3/4 and employs ondemand loading of experts. This baseline demonstrates the performance enhancements that are attributable to MWQ. (3) Hold-in-Memory-AWQ [23]: This method quantizes all experts to INT4 and keeps all model weights in GPU memory. This baseline provides an efficient benchmark for assessing the overhead of dynamically loading experts in D<sup>2</sup>MoE. (4) EdgeMoE [42]: This approach quantizes experts to different bit-width based on their importance, utilizing a pre-loading mechanism to predict and dynamically load the required experts during inference. (5) MoQE-DynaIO: This method quantizes all expert weights to a uniform bit-width (e.g., INT4/INT8) using the MoQE quantization method [19] and dynamically loads them on demand during inference. Similar to D<sup>2</sup>MoE, the quantized expert weights in all the above methods must be dynamically converted back to the original FP16 format during inference.

**Configuration.** We set the group size as 128 in MWQ and utilize two quantized versions of  $D^2MoE$ :  $D^2MoE$ -V1, which is compared with INT4 MoE-based LLMs, and  $D^2MoE$ -V2, which is compared with INT8 MoE-based LLMs.  $D^2MoE$ -V1 equipped with  $b_1=2$  and  $b_K=4$ , while  $D^2MoE$ -V2 using  $b_1=5$  and  $b_K=8$ . Furthermore, in  $D^2MoE$ -V1, the quantized expert capacity is set to  $\{0.3,0.4,0.3\}$ , and in  $D^2MoE$ -V2, the capacity of each expert is 0.25.

#### 5.2 End-to-End Results

Model Accuracy. Table 3 presents the the model accuracy of D<sup>2</sup>MoE with the baselines in LLaMA-MoE-3.5B and Mixtral 8 ×7B. It has been demonstrated that D<sup>2</sup>MoE and Matryoshka-Free exhibit perplexity and zero-shot performance that closely approximates Hold-in-Memory. This suggests that dynamic bit-width selection can effectively guarantee model accuracy. While EdgeMoE demonstrates comparable perplexity to D<sup>2</sup>MoE, it exhibits reduced accuracy in specific zero-shot tasks. This disparity can be ascribed to the differing significance attributed to the various markers, with EdgeMoE utilizing a predetermined mixture of bit-width to ascertain the importance of the experts, resulting in diminished accuracy. In contrast, both MoQE-DynaIO-INT4 and MoQE-DynaIO-INT8 demonstrate weaker performance in perplexity and

zero-shot tasks. This is due to the quantization method, which results in accuracy loss.

Throughput. Figure 10 shows a combined comparison of the throughput of D<sup>2</sup>MoE with the baseline with different memory budgets in environments 1 and 2. On Mixtral 8×7B,  $D^2MoE$  achieves a throughput improvement of  $1.14\times-1.39\times$ compared to EdgeMoE, and on LLaMA-MoE-3.5B, it improves throughput by 1.06×-1.16× while reducing memory usage by 33%-53%. Against MoQE-DynaIO, D<sup>2</sup>MoE delivers a 1.42×-3.37× throughput gain, particularly in Environment 1, underscoring its suitability for memory-constrained edge devices. The throughput of D<sup>2</sup>MoE approaches that of Holdin-Memory-AWQ as the memory budget increases, demonstrating near-complete overlap between expert loading and computation. For example, as shown in Figure 10(a) in Environment 1, Hold-in-Memory-AWQ achieves 94.3 tokens/s with 2500MB memory, while D<sup>2</sup>MoE reaches 89.14 tokens/s with a reduced budget of 1600MB.

D<sup>2</sup>MoE can adapt to diverse memory budgets on edge devices. For instance, when running Mixtral 8×7B in Environment 1, EdgeMoE and Hold-in-Memory-AWQ fail due to limited GPU memory, while D<sup>2</sup>MoE achieves a throughput of 38.07 tokens/s. Moreover, D<sup>2</sup>MoE scales throughput with memory budgets. As shown in Figure 10(c), with 32 requests, throughput rises from 66.45 tokens/s at M = 200MB to 83.14 tokens/s at M = 1600MB.

**Dense LLM Architecture.** We extend D<sup>2</sup>MoE to dense LLM architecture to compare throughput and peak memory footprint with the traditional approach using a fixed bit-width INT4 on environment 1 with  $M=1600 \mathrm{MB}$ . As illustrated in Figure 11, D<sup>2</sup>MoE consistently outperforms the method that dynamically loads a fixed bit-width FFN layer with varying request numbers, achieving up to a 1.22× increase in throughput and up to a 12% reduction in peak memory consumption. This advantage arises because conventional loading-based quantization methods only load INT4 bit-width, whereas D<sup>2</sup>MoE can dynamically load bit-width lower than INT4, thereby reducing data transfer overhead more effectively.

Moreover, as the number of requests increases, the throughput generally exhibits a near-linear growth. However, once the request number reaches 25, the increase slows down due to hardware computational constraints. This phenomenon also exists in MoE-based models. Nevertheless, the performance gains offered by  $\rm D^2MoE$  are not as pronounced as those observed in MoE-based LLMs, since the FFN layer typically constitutes only  $\rm 50\%-60\%$  of the total parameters in dense models. After quantization, this proportion becomes even smaller, thereby shifting the memory bottleneck to the attention layer.

| Model          | Method             | Perplexity ↓ | PiQA  | Arc.e | BoolQ | HellaSwag | Winogrande |
|----------------|--------------------|--------------|-------|-------|-------|-----------|------------|
| LLaMA-MoE-3.5B | Hold-in-Memory     | 14.55        | 72.32 | 48.76 | 65.56 | 66.34     | 61.77      |
|                | Matryoshke-Free    | 14.58        | 72.29 | 48.71 | 65.48 | 66.28     | 61.76      |
|                | Hold-in-Memory-AWQ | 15.89        | 69.32 | 46.52 | 62.54 | 61.55     | 57.44      |
|                | EdgeMoE            | 14.78        | 71.46 | 47.36 | 64.43 | 64.32     | 59.52      |
|                | MoQE-DynaIO-INT4   | 15.66        | 69.34 | 46.52 | 62.58 | 61.53     | 57.40      |
|                | MoQE-DynaIO-INT8   | 14.55        | 72.32 | 48.74 | 65.58 | 66.28     | 61.71      |
|                | 2MoE-V1<br>D       | 15.68        | 69.32 | 46.52 | 62.50 | 64.28     | 59.52      |
|                | 2MoE-V2<br>D       | 14.58        | 72.29 | 48.72 | 65.51 | 66.23     | 61.71      |
| Mixtral 8×7B   | Hold-in-Memory     | 4.04         | 82.4  | 82.6  | 80.56 | 84.1      | 76.5       |
|                | Matryoshke-Free    | 4.28         | 80.29 | 80.71 | 78.48 | 82.28     | 74.76      |
|                | Hold-in-Memory-AWQ | 4.25         | 80.32 | 81.52 | 78.54 | 83.55     | 75.44      |
|                | EdgeMoE            | 4.38         | 78.46 | 80.36 | 77.43 | 82.32     | 75.52      |
|                | MoQE-DynaIO-INT4   | 4.25         | 80.34 | 81.52 | 78.58 | 83.53     | 75.40      |
|                | MoQE-DynaIO-INT8   | 4.08         | 82.32 | 81.74 | 79.58 | 83.28     | 74.71      |
|                | 2MoE-V1<br>D       | 4.28         | 81.32 | 81.52 | 78.05 | 82.88     | 75.52      |
|                | 2MoE-V2<br>D       | 4.09         | 82.29 | 81.72 | 79.51 | 83.23     | 74.71      |
|                |                    |              |       |       |       |           |            |

<span id="page-10-0"></span>Table 3: Perplexity accuracy (lower is better) and zero-shot accuracy (higher is better) of D2MoE and the baselines in LLaMA-MoE-3.5B and Mixtral 8×7B.

# 5.3 System Overhead

D2MoE Setup Overhead. The overhead of setting up D2MoE before inference, including fine-tuning bit-width routers and applying MWQ to quantize experts, is as follows. For LLaMA-MoE-3.5B, fine-tuning the bit-width routers with a batch size of 64 required approximately 2 hours, while MWQ with a batch size of 16 was completed in 10 minutes. For Mixtral 8×7B, fine-tuning with a batch size of 16 took over 4 hours, and MWQ with a batch size of 4 was completed in 20 minutes. Bit-Width Routing Overhead. As shown in Table [4,](#page-10-1) we evaluate the end-to-end overhead of the D2MoE bit-width router in terms of computation, memory usage, and latency on the LLaMA-MoE-3.5B and Mixtral 8 × 7B models under Environment 1. The additional computation and memory overhead compared to the original MoE-based model is under 0.5%, with an extra latency of approximately 1.5%, primarily attributed to the softmax operation in the router. Despite this minor overhead, the bit-width router significantly reduces data transfer by dynamically selecting lower-bit-width experts while maintaining accuracy.

<span id="page-10-1"></span>Table 4: Overhead of D2MoE bit-width router on LLaMA-MoE-3.5B and Mixtral 8×7B.

| Model          | Computation | Memory | Latency |
|----------------|-------------|--------|---------|
| LLaMA-MoE-3.5B | 0.28%       | 0.53%  | 1.67%   |
| Mixtral 8×7B   | 0.22%       | 0.12%  | 1.04%   |

MWQ Dequantization Overhead. As shown in Figure [12,](#page-11-1) we evaluated the dequantization overhead of D2MoE in terms of computation, peak memory usage, and latency during inference on the LLaMA-MoE-3.5B and Mixtral 8×7B models in Environment 1. Dequantization introduces overhead by converting integer weights to floating-point representations via shift operations. This overhead is more significant with fewer requests, as fewer experts reuse the same bit-width. However, as the number of requests increases, the overhead decreases due to improved weight utilization. For instance, on the Mixtral 8×7B, when the number of requests increases from 4 to 32, the computational and latency overhead of D <sup>2</sup>MoE-V1 decreases from 20.77% and 18.56% to 16.77% and 5.3%, respectively. While FP16 weights temporarily increase peak memory during dequantization, this memory is released immediately after use, resulting in minimal impact on overall inference efficiency.

Parallelism Planning Overhead. Parallelism planning for various quantized experts constitutes a significant overhead in the D2MoE framework. We conducted profiling on LLaMA-MoE-3.5B and Mixtral 8×7B with request numbers ranging from 4 to 32. The total execution times and the proportion of planning overhead for these models in environment 1 are depicted in Figure [13.](#page-11-2) It is evident that while the execution time for parallelism planning increases with the number of requests, its relative share of the overall inference process decreases. This reduction occurs because the number of loaded quantized experts rises with the increase in requests. Consequently, the additional overhead from parallelism planning

<span id="page-11-0"></span>![](_page_11_Figure_2.jpeg)

(a) Throughput of LLaMA-MoE in Environment 1.

![](_page_11_Figure_4.jpeg)

(b) Throughput of Mixtral 8×7B in Environment 1.

![](_page_11_Figure_6.jpeg)

(c) Throughput of LLaMA-MoE in Environment 2.

![](_page_11_Figure_8.jpeg)

(d) Throughput of Mixtral 8×7B in Environment 2.

Figure 10: Throughput of D<sup>2</sup>MoE and baselines with different memory budgets.

![](_page_11_Figure_11.jpeg)

Figure 11: Comparison of GPTQ and D<sup>2</sup>MoE throughput and peak memory in LLaMA2-13B.

remains manageable under on-device inference conditions with multiple requests.

<span id="page-11-1"></span>![](_page_11_Figure_14.jpeg)

(a) LLaMA-MoE-3.5B in Environment 1.

![](_page_11_Figure_16.jpeg)

(b) Mixtral 8×7B in Environment 1

Figure 12: MWQ dequantization overhead of D<sup>2</sup>MoE.

<span id="page-11-2"></span>![](_page_11_Figure_19.jpeg)

Figure 13: The overall execution time (ms) and the overhead proportion (%) under different request numbers.

#### 5.4 Ablation Study

Figure 14 illustrates the contribution of each D<sup>2</sup>MoE component to throughput improvement through step-by-step integration. First, token-adaptive bit-width selection ("+Router") is added, enabling dynamic bit-width selection for each expert but still relying on conventional quantization, which stores multiple bit-width versions. Building on +Router, MWQ ("+MWQ") is introduced, nesting lower bit-width within higher ones to reduce expert loading overhead, improving throughput by 1.91×-4.95×. Next, the hottest-expert-bit-first criterion ("+HEBF") is added, parallelizing expert I/O and computation, further reducing I/O-compute bubbles and increasing throughput by 1.11×-1.21×. Finally, integrating expert memory budget ("+Budget") retains frequently activated low bit-width experts in GPU memory, reducing repeated loading and achieving an additional 1.06×-1.21× improvement. These enhancements due to reduced weight loading and a fine-grained balance of I/O and computation overhead.

<span id="page-12-0"></span>![](_page_12_Figure_2.jpeg)

Figure 14: Ablation study for each component of D<sup>2</sup>MoE on LLaMA-MoE and Mixtral 8×7B.

#### 6 DISCUSSION

The transition of LLMs from a dense to a MoE structure effectively sparsifies the model, reducing computational overhead. Building on this, we further slice the experts at the bit-width level, making the model even more sparse. In the future, we could explore finer granularities, such as sparsification at the neuron level within experts, which remains an open challenge. Currently, the D<sup>2</sup>MoE system has several limitations:

Asynchronous Execution on Edge Devices. D<sup>2</sup>MoE does not accommodate a parallel execution strategy for multiple, asynchronously received requests on edge devices. In scenarios where multiple requests initiate reasoning asynchronously, different requests activate distinct layers of experts, substantially increasing bandwidth pressure on edge devices. Developing a more sophisticated quantized expert scheduling algorithm could enhance inference speed by improving the I/O efficiency of quantized experts.

**Expert Loading Strategy.** D<sup>2</sup>MoE relies on the on-demand loading of experts, without considering preloading. This approach, while responsive, may introduce additional I/O wait times. Anticipating and preloading the experts likely to be activated later could foster a more efficient I/O/Compute parallel strategy. However, preloading complicates the I/O and Compute strategy, making it more dynamic. We aim to further investigate the viability of this intricate strategy in future research.

**Suitability for Edge Devices.** D<sup>2</sup>MoE may not perform optimally on edge devices with limited computational resources, such as smartphones that depend on mobile GPU capabilities. For mobile devices utilizing NPUs, the efficiency of the dequantization process in D<sup>2</sup>MoE cannot be assured. Nonetheless, these challenges can be solved through tailored NPU arithmetic and system optimizations, areas we plan to explore in our forthcoming efforts.

#### 7 RELATED WORK

**On-device MoE-based LLM Inference.** PowerInfer [34] leverages activation sparsity to facilitate inference acceleration on consumer-grade GPUs. EdgeMoE [42] enhances both

memory and computation efficiency by structuring the MoE models within a hierarchical storage framework. AdapMoE [45] features adaptive expert gating and management by a cohesive algorithm-system co-design, which boosts the inference speeds on edge devices. Fiddler [18] utilizes hybrid GPU-CPU computation on the experts of MoE models, thus minimizing the data transfer between CPUs and GPUs.

LLM Quantization. Post-training quantization currently stands as a prevalent technique for compressing LLMs, significantly reducing storage and I/O overhead by transforming high-precision floating-point numbers into low-precision integers. On the one hand, GPTQ [9], AWQ [22], and Quant-LLM [40] exemplify weight-only quantization methods, where weights are dynamically converted to the same data type as activation during inference. On the other hand, weight-activation quantization methods such as SmoothQuant [41], QServe [24], and AffineQuant [25] enhance inference speed by directly leveraging integer matrix multiplication operators embedded within hardware.

Pipeline Parallelism for LLM. Pipelining techniques are extensively employed to accelerate LLM reasoning by minimizing the gap between I/O and computation. STI [14] enhances I/O and compute resource utilization by applying mixed bit-width quantization to BERT model weights, thereby accommodating various target latencies and accuracy levels. OTAS [4] guarantees resilient transformer model services and handles fluctuations in both user requests and query loads through efficient token management.

#### 8 CONCLUSION

Aiming at enabling efficient on-device MoE-based LLM serving, we conduct an algorithm-system co-design and propose the  $\mathrm{D}^2\mathrm{MoE}$  framework, which introduces adaptive nested quantization based on token properties during multi-request inference.  $\mathrm{D}^2\mathrm{MoE}$  achieves optimal expert bit-width selection through dual routing and dynamic quantized expert scheduling with minimal I/O-compute parallelism bubbles. Realistic evaluation shows that  $\mathrm{D}^2\mathrm{MoE}$  significantly improves the latency-memory trade-off with an affordable inference overhead and guaranteed accuracy on edge devices.

#### 9 ACKNOWLEDGMENTS

This research was supported by fundings from the Hong Kong RGC General Research Fund (152169/22E, 152228/23E, 162161/24E), Research Impact Fund (No. R5060-19, No. R5011-23), NSFC/RGC Collaborative Research Scheme (Grant No. 62461160332 & CRS\_HKUST602/24), Collaborative Research Fund (No. C1042-23GF), Areas of Excellence Scheme (AoE/E-601/22-R), and the InnoHK (HKGAI). We also thank MetaX Technology for supporting this research.

# References

- <span id="page-13-16"></span>[1] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI conference on artificial intelligence (AAAI '22, Vol. 34). 7432–7439.
- <span id="page-13-1"></span>[2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20). Curran Associates Inc., Article 159, 25 pages.
- <span id="page-13-7"></span>[3] Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph E. Gonzalez, Matei Zaharia, and Ion Stoica. 2025. MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs.
- <span id="page-13-24"></span>[4] Jinyu Chen, Wenchao Xu, Zicong Hong, Song Guo, Haozhao Wang, Jie Zhang, and Deze Zeng. 2024. OTAS: An Elastic Transformer Serving System via Token Adaptation. In IEEE Conference on Computer Communications (INFOCOM '24). 1021–1030.
- <span id="page-13-2"></span>[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde, Jared Kaplan, Harrison Edwards, Yura Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, David W. Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William H. Guss, Alex Nichol, Igor Babuschkin, Suchir Balaji, Shantanu Jain, Andrew Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew M. Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. ArXiv abs/2107.03374 (2021).
- <span id="page-13-18"></span>[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Vol. 1. 2924–2936.
- <span id="page-13-17"></span>[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv[:1803.05457](https://arxiv.org/abs/1803.05457)
- <span id="page-13-10"></span>[8] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research 23, 1 (jan 2022), 39 pages.
- <span id="page-13-14"></span>[9] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers. The Eleventh International Conference on Learning Representations (2023).
- <span id="page-13-19"></span>[10] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin

- Wang, and Andy Zou. 2024. A framework for few-shot language model evaluation.
- <span id="page-13-6"></span>[11] Georgi Gerganov. 2023. llama.cpp. [https://github.com/ggerganov/](https://github.com/ ggerganov/llama.cpp) [llama.cpp](https://github.com/ ggerganov/llama.cpp)
- <span id="page-13-3"></span>[12] Github. 2022. Copilot. [https://github.com/features/copilot](https://github.com/features/ copilot)
- <span id="page-13-5"></span>[13] Zhuocheng Gong, Ang Lv, Jian Guan, Wei Wu, Huishuai Zhang, Minlie Huang, Dongyan Zhao, and Rui Yan. 2024. Mixture-of-Modules: Reinventing Transformers as Dynamic Assemblies of Modules. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 20924–20938.
- <span id="page-13-8"></span>[14] Liwei Guo, Wonkyo Choe, and Felix Xiaozhu Lin. 2023. STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS 2023). 791–803.
- <span id="page-13-15"></span>[15] Liwei Guo and Felix Xiaozhu Lin. 2022. Minimum viable device drivers for ARM trustzone. In Proceedings of the Seventeenth European Conference on Computer Systems (EuroSys '22). 300–316.
- <span id="page-13-4"></span>[16] Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and Xiaojuan Qi. 2025. Mc-moe: Mixture compressor for mixture-of-experts llms gains more. The Eleventh International Conference on Learning Representations (2025).
- <span id="page-13-0"></span>[17] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts. arXiv[:2401.04088](https://arxiv.org/abs/2401.04088)
- <span id="page-13-21"></span>[18] Keisuke Kamahori, Yile Gu, Kan Zhu, and Baris Kasikci. 2024. Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models. In 5th Workshop on practical ML for limited/low resource settings.
- <span id="page-13-9"></span>[19] Young Jin Kim, Raffy Fahim, and Hany Hassan Awadalla. 2023. Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness. arXiv[:2310.02410](https://arxiv.org/abs/2310.02410)
- <span id="page-13-13"></span>[20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23). 611–626.
- <span id="page-13-11"></span>[21] Xiangyu Li, Yuanchun Li, Yuanzhe Li, Ting Cao, and Yunxin Liu. 2024. FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking (MobiCom '24). 709–723.
- <span id="page-13-12"></span>[22] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. In Proceedings of Machine Learning and Systems (MLSys '24, Vol. 6). 87–100.
- <span id="page-13-20"></span>[23] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. Proceedings of Machine Learning and Systems 6 (2024), 87–100.
- <span id="page-13-22"></span>[24] Yujun Lin\*, Haotian Tang\*, Shang Yang\*, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. 2024. QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving. arXiv preprint arXiv:2405.04532 (2024).
- <span id="page-13-23"></span>[25] Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. 2024. AffineQuant:

- <span id="page-14-0"></span>Affine Transformation Quantization for Large Language Models. In The Twelfth International Conference on Learning Representations (ICLR '24).
- <span id="page-14-16"></span>[26] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models. arXiv[:1609.07843](https://arxiv.org/abs/1609.07843)
- <span id="page-14-14"></span>[27] NVIDIA. 2023b. NVIDIA. Tensorrt-llm. [https://github.com/NVIDIA/](https://github.com/ NVIDIA/TensorRT-LLM) [TensorRT-LLM](https://github.com/ NVIDIA/TensorRT-LLM)
- <span id="page-14-4"></span>[28] OpenAI. 2022. Chatgpt. [https://openai.com/blog/chatgpt](https://openai.com/blog/ chatgpt)
- <span id="page-14-12"></span>[29] Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, and Jae W. Lee. 2024. Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs. In Proceedings of the 41st International Conference on Machine Learning (ICML '24).
- <span id="page-14-15"></span>[30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research 21, 1, Article 140 (2020), 67 pages.
- <span id="page-14-6"></span>[31] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. 2024. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. arXiv[:2404.02258](https://arxiv.org/abs/2404.02258)
- <span id="page-14-8"></span>[32] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. 2024. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. arXiv[:2404.02258](https://arxiv.org/abs/2404.02258)
- <span id="page-14-18"></span>[33] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. WinoGrande: an adversarial winograd schema challenge at scale. Commun. ACM 64, 9 (aug 2021), 99–106.
- <span id="page-14-19"></span>[34] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2023. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. arXiv[:2312.12456](https://arxiv.org/abs/2312.12456)
- <span id="page-14-11"></span>[35] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. 2024. Llumnix: Dynamic Scheduling for Large Language Model Serving. 18th USENIX Symposium on Operating Systems Design and Implementation (2024).
- <span id="page-14-13"></span>[36] Philippe Tillet, H. T. Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL '19). 10–19.
- <span id="page-14-1"></span>[37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv[:2307.09288](https://arxiv.org/abs/2307.09288)
- <span id="page-14-2"></span>[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). 6000–6010.
- <span id="page-14-9"></span>[39] Manni Wang, Shaohua Ding, Ting Cao, Yunxin Liu, and Fengyuan Xu. 2021. AsyMo: scalable and efficient deep-learning inference on

- asymmetric mobile CPUs. In Proceedings of the 27th Annual International Conference on Mobile Computing and Networking (MobiCom '21). 215–228.
- <span id="page-14-21"></span>[40] Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, and Shuaiwen Leon Song. 2024. Quant-LLM: Accelerating the Serving of Large Language Models via FP6-Centric Algorithm-System Co-Design on Modern GPUs. In 2024 USENIX Annual Technical Conference (USENIX ATC '24). 699–713.
- <span id="page-14-10"></span>[41] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In Proceedings of the 40th International Conference on Machine Learning (ICML '23).
- <span id="page-14-5"></span>[42] Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, and Mengwei Xu. 2023. EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models. ArXiv abs/2308.14352 (2023).
- <span id="page-14-17"></span>[43] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a Machine Really Finish Your Sentence?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL '19). 4791–4800.
- <span id="page-14-3"></span>[44] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: Open Pre-trained Transformer Language Models. arXiv[:2205.01068](https://arxiv.org/abs/2205.01068)
- <span id="page-14-20"></span>[45] Shuzhang Zhong, Ling Liang, Yuan Wang, Runsheng Wang, and Meng Li Ru Huang. 2024. AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference.. In IEEE International Conference on Computer-Aided Design (ICCAD '24).
- <span id="page-14-7"></span>[46] Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng. 2024. LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training. arXiv preprint arXiv:2406.16554 (2024).