# 1 Introduction

Large Language Models (LLMs) are increasingly utilized across various applications, leading to a trend toward larger model sizes. This expansion in model size significantly escalates training overheads, making the process more costly and resource-intensive. To mitigate the time-consuming

<sup>∗</sup>Equal Contribution.

nature of training LLMs, it is common to employ multiple GPUs in a data-parallel configuration. However, naive Data Parallelism (DP) necessitates that each GPU replicates the entire optimizer states, a strategy often impractical due to the limited memory capacity of individual GPUs. This limitation becomes particularly critical with the substantial size of modern LLMs.

Sharded Data Parallelism (ShardedDP) evolves from naive DP to reduce the memory footprint by sharding optimizer states among GPUs. However, the sharding mechanism significantly changes the communication pattern of DP, which brings up new challenges in system optimization. As a result, ShardedDP suffers from heavy communication overheads of both weights and gradients, particularly when internode bandwidth is limited. This can significantly increase the end-to-end (E2E) training time, especially when using a small gradient accumulation step.

Quantization is a widely used strategy to reduce the communication overhead of naive DP, albeit with some accuracy loss. Unfortunately, few prior studies have specifically addressed the is-

<span id="page-1-0"></span>![](_page_1_Figure_3.jpeg)

Figure 1: Training validation loss for GPT-6.7B; SDP4Bit is closely aligned with full precision training.

sue of communication reduction in ShardedDP. Recently, QSDP [18] and ZeRO++ [32] attempted to quantize the communication of ShardedDP to Int4. However, when pushing the communication ratio to its limits, both QSDP and ZeRO++ fail to maintain comparable training loss to the baseline. Furthermore, ZeRO++ lacks theoretical convergence guarantees, and QSDP is limited to one specific quantizer called "random shift" and strong assumptions. Thus, there is no effective solution to reduce ShardedDP's communication to nearly 4 bits without compromising the training loss.

To address these issues, this paper proposes a novel communication reduction strategy, **SDP4Bit**. SDP4Bit comprises two main techniques: (1) **Quantization on Weight Differences**: Instead of directly quantizing weights, we apply 4-bit quantization to compress the weight differences between current and previous iterations; (2) **Two-Level Gradient Smooth Quantization**: We apply 8-bit quantization to intra-node gradients and 4-bit quantization to inter-node gradients, with Hadamard Transform for smoothing the outliers. To the best of our knowledge, SDP4Bit is the first work to successfully reduce both gradients and weights to nearly 4 bits without compromising training accuracy. As shown in Figure 1, the training validation loss for GPT-6.7B using SDP4Bit is closely aligned with full precision training. Our main contributions are summarized as follows:

- We propose a low-bit (i.e., nearly 4-bit) communication reduction strategy for ShardedDP that preserves E2E training accuracy.
- We establish a convergence guarantee for the proposed strategy, showing the same convergence rate as the ordinary Stochastic Gradient Descent (SGD), with extended choices of biased compressors and weaker assumptions compared to the previous theoretical results.
- We implement our method within the Megatron-LM framework and enhance it with runtime optimizations such as buffer reuse, operation pruning, and kernel fusion.
- Our results validate that SDP4Bit successfully compresses the communication of weights and gradients to nearly 4 bits, with a negligible impact on final loss. Notably, compared to non-quantized baseline, it achieves 4.08× speedup for a GPT-18B model trained on 128 H800 GPUs.

#### 2 Preliminaries

#### <span id="page-1-1"></span>2.1 Sharded Data Parallelism

Sharded Data Parallelism (ShardedDP) modifies traditional Data Parallelism (DP) to reduce the memory footprint per GPU. Unlike traditional DP, which duplicates high-precision optimizer states (typically including model weights and momentum variables in Float32) on each GPU, ShardedDP partitions them across all GPUs. Each GPU manages  $\frac{1}{P}$  of the optimizer states, hence reducing the corresponding memory footprint by a factor of  $\frac{1}{P}$ , where P represents the number of GPUs involved.

```
Algorithm 1 QSDP / ZeRO++
```

#### Algorithm 2 Megatron-LM with SDP4Bit

```
Require: worker: p, weight in shard p: w[p], local gradi-Require: worker: p, weight in shard p: w[p], local
      ent on worker p: g_{model}^p, global gradient in shard p:
                                                                                       gradient on worker p: g_{model}^p, global gradient in
                                                                                       shard p: g_{main}[p], weight difference: d
      g_{main}|p|
 1: function CompressedForwardPass
                                                                                   1: function CompressedForwardPass
         \tilde{w}_{main}[p] \leftarrow \text{QuantizeWeights}(w_{main}[p])
                                                                                          d[p] = w_{main}[p] - w_{model}[p]
         w_{model} \leftarrow \text{AllGather}(\tilde{w}_{main}[p])
                                                                                   3:
                                                                                          d[p] \leftarrow \text{QuantizeWeightsDiff}(d[p])
         output^p \leftarrow ForwardPass(w_{model}, input^p)
                                                                                   4:
                                                                                          d \leftarrow AllGather(d[p])
 5:
         free(w_{model})
                                                                                   5:
                                                                                          w_{model} \leftarrow w_{model} + d
 6: function CompressedBackwardPass
                                                                                          output^p \leftarrow ForwardPass(w_{model}, input^p)
         \tilde{w}_{main}[p] \leftarrow \text{QuantizeWeights}(w_{main}[p])
                                                                                   7: function CompressedBackwardPass
         w_{model} \leftarrow \text{AllGather}(\tilde{w}_{main}[p])
                                                                                          g_{model}^p \leftarrow \text{Gradient}(w_{model}, output^p)
 9:
         g_{model}^p \leftarrow \text{Gradient}(w_{model}, output^p)
                                                                                   9:
                                                                                           g_{main}[p] \leftarrow \text{ReduceScatter TLq-HS}(g_{model}^p)
10:
         free(w_{model})
                                                                                 10:
                                                                                           w_{main}[p] \leftarrow \text{Optimizer}(g_{main}[p], w_{main}[p])
         \begin{split} \tilde{g}^p_{model} &\leftarrow \text{QuantizeGradients}(g^p_{model}) \\ g_{main}[p] &\leftarrow \text{ReduceScatter/TwoAlltoAll}(\tilde{g}^p_{model}) \\ w_{main}[p] &\leftarrow \text{Optimizer}(g_{main}[p], w_{main}[p]) \end{split}
11:
12:
13:
```

With high-precision model weights sharded across GPUs (referred to as "main weights"), an all-gather operation is required to collect the weights for the forward-backward steps (typically in relatively low precision, such as Float16, referred to as "model weights"). For gradient synchronization, a reduce-scatter operation is performed before the optimization steps to ensure that each GPU has the corresponding shard of averaged gradients. In summary, each iteration necessitates an all-gather for weights and a reduce-scatter for gradients.

Driven by the need to train larger models within the constraints of GPU memory, ShardedDP is incorporated into several popular training frameworks, including Megatron-LM (Distributed Optimizer), DeepSpeed (ZeRO), and PyTorch (FSDP), each with slightly different implementation strategies. Notably, ZeRO-3 and FSDP release the collected weights after each computation to enhance memory efficiency, necessitating additional weight collective communication during the backward pass. Conversely, ZeRO-2 and Megatron-LM retain the collected weights throughout, thus eliminating the need for weight collection during the backward pass. Our weight reduction strategy is particularly well-suited for Megatron-LM, as it maintains a full model's weights at all times (see Algorithm 2). Additionally, Megatron-LM provides flexible parallelism support, such as tensor parallelism, which partitions models vertically to alleviate memory limitations. This approach enables the training of larger models compared to DeepSpeed.

#### 2.2 Quantization

Quantization is a commonly used strategy in data compression. In this paper, we explore symmetric linear (integer) quantization due to its low overhead and latency. It is defined as follows:

$$x_{\text{int}} = \text{round}\left(\frac{x}{s} \cdot (2^{k-1} - 1)\right), \quad s = \max(x),$$

where k represents the bit-width of the quantized values, and s is referred to as "scales".

Additionally, group-wise quantization [25] is employed to minimize quantization error by dividing the data into multiple groups and quantizing each group individually. This approach results in a lower compression ratio due to the need to store additional scales.

#### <span id="page-2-1"></span>2.3 Collective Reduction Communication with Quantization

State-of-the-art (SOTA) collective communication libraries (e.g., NCCL, Gloo) employ a ring-based algorithm for its optimal bandwidth [21]. This algorithm executes reduce-scatter operations across P-1 rounds, during which each GPU sends local data and aggregates the received data. When quantization is applied, this necessitates P-1 rounds of quantization and dequantization, potentially leading to error propagation and increased latency [11]. Some strategies replace reduce-scatter with all-to-all communication, but this increases inter-node communication, typically with lower bandwidth.

ZeRO++ [32] modifies this approach by substituting the conventional reduce-scatter (used by QSDP) with two all-to-all operations (shown in Algorithm 1, with different colors to distinguish ZeRO++

from QSDP). The first operation is confined within each node, and post-reduction, the data size is diminished to  $\frac{1}{N}$ , where N is the number of GPUs per node. The subsequent all-to-all operation occurs between GPUs across different nodes that share the same local rank. In ZeRO++, each all-to-all operation follows a 4-bit quantization step to minimize communication data size.

As shown in Figure 5, while this method efficiently integrates quantization into reduce-scatter without augmenting inter-node communication, the repeated 4-bit quantization steps can accumulate quantization errors, potentially leading to suboptimal training outcomes.

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 2: Communication of quantized weight differences.

![](_page_3_Figure_4.jpeg)

Figure 3: Two-level gradient quantization: 8-bit intra-node and 4-bit internode quantization.

## 3 Methodology

#### 3.1 Quantization on Weight Differences (qWD)

As discussed in Section 2.1, ShardedDP requires each GPU to send/receive updated weights (main weights) to/from other GPUs in each iteration. However, weights generally exhibit a wide range and directly applying 4-bit quantization leads to significant quantization errors. Even with groupwise quantization, a gap in E2E training loss compared to full precision remains despite using small group sizes.

**qWD:** To address this issue, we quantize weight differences instead of the original weights during communication. As illustrated in Figure 2 and Algorithm 2, after the optimizer step, each GPU calculates the

<span id="page-3-1"></span>![](_page_3_Figure_10.jpeg)

Figure 4: Histogram of (a) weights and (b) weight differences. Each vertical dashed line represents a quantization level corresponding to a 4-bit quantization lattice.

differences between the main weights and the model weights. These differences are then quantized and all-gathered across all GPUs. After all-gathering, each GPU dequantizes the received data to obtain the weight differences. These differences are then added to the model weights to obtain the updated weights.

There are two main benefits to applying quantization to weight differences.

1) In practice, weight differences are generally easier to quantize. As shown in Figure 4, weight differences are more uniformly distributed in a smaller range compared to the weights themselves, resulting in smaller errors for INT4 quantization. Furthermore, since intuitively the magnitudes of weight differences are smaller than those of weights themselves (informally supposed that  $\|\delta w_t\| = \|w_t - w_{t-1}\| < \|w_t\|$ ) and the relative quantization errors are similar between weights and weight differences (informally supposed that  $\frac{\|q(\delta w_t) - \delta w_t\|}{\|\delta w_t\|} \approx \frac{\|q(w_t) - w_t\|}{\|w_t\|}$ ), the

weight differences compression potentially has a smaller error relative to the weights themselves:  $\frac{\|q(\delta w_t) - \delta w_t\|}{\|w_t\|} \lessapprox \frac{\|q(w_t) - w_t\|}{\|w_t\|}, \text{ where } q(\cdot) \text{ is the quantization function.}$ 

2) In theory, weight differences compression improves convergence compared to naive weight compression. When extended to biased compressors, we present theoretical guarantees for convergence at the same rate as ordinary SGD, as detailed in Section 4.2. In contrast, we demonstrate that using biased compressors directly on weights can lead to convergence failure, as illustrated in an example in Section 4.1. This proves that biased compressors are not compatible with QSDP or ZeRO++.

#### 3.2 Two-Level Gradient Smooth Quantization

#### 3.2.1 Two-level Gradient Quantization (TLq)

As discussed in Section 2.3, the 2-step all-to-all communication strategy benefits gradient communication when quantization is applied. However, it also introduces error accumulation due to consecutive 4-bit quantization steps, necessitating additional rounds of training and communication that diminish the per-iteration communication savings. We observe that applying ULq, which quantizes gradeints to extremely low precision, such as 4-bit, leads to noticeable deviations in training loss compared to full-precision training, as illustrated in Figure 5.

**TLq:** Instead of employing a global 4-bit quantization strategy for both inter-node and intra-node all-to-all communications, we propose a two-level precision quantization strategy. This approach balances performance and accuracy by enhancing accuracy without introducing additional overhead. For intra-node all-to-all communication, gradients are quantized to INT8 before sending. After receiving the data, each GPU dequantizes the received data back to the full precision (i.e., FP32) for local reduction. This reduced data is then quantized to INT4 to minimize inter-node communication overhead. The detailed methodology is depicted in Figure 3. Since the two all-to-all operations utilize different network bandwidths, their communications can be effectively overlapped (see Table 4).

#### 3.2.2 *TLq* with Hadamard Smoother (*TLq-HS*)

While *TLq* brings the training loss closer to the baseline, it does not achieve perfect alignment. In gradient quantization, *outliers can significantly amplify quantization errors*. Although group-wise quantization isolates outliers to minimize their impact on the precision of values in other groups, the values within the same group remain affected.

**TLq-HS:** To mitigate the outlier issue, we apply Hadamard transform [9] to the gradients before quantization. The Hadamard transform, a specific type of generalized Fourier transform, exhibits properties such as  $H = H^T$  and  $H \cdot H^T = I$ , distributing outlier information across nearby elements and effectively smoothing them out. For a detailed description of the methodology, see Algorithm 3.

## 3.3 Performance Optimizations in Implementation

**Optimizing Memory Efficiency for** Weight Differences Computation: After the all-gather communication, each GPU receives weight differences from the others, which are then added to the model weights to update them. As discussed in Section 2.1, with ShardedDP enabled in Megatron-LM, each GPU maintains a complete copy of the model weights for forward and backward computation. This contrasts with ZeRO, where model weights are released after computation. By reusing these locally stored model weights in Megatron-LM, our implementation eliminates the need for additional buffers to retain model weights for calculating the differences, thus enhancing memory efficiency.

## <span id="page-4-0"></span>Algorithm 3 TLq with Hadamard Smoother

```
Require: gradient grad
  1: function TLq-HS
 2:
            \hat{g} \leftarrow \text{Hadamard}(grad)
             \hat{qg}_{8bit} \leftarrow \text{Quantize8Bit}(\hat{g})
  3:
             list(\hat{qg}_{8bit\_intra}) \leftarrow \textbf{IntraAlltoAll}(\hat{qg}_{8bit})
  4:
            \begin{array}{l} list(\hat{g}_{intra}) \leftarrow \text{Dequantize}(list(\hat{q}\hat{g}_{8bit\_intra})) \\ list(\hat{g}_{intra}) \leftarrow \text{Hadamard}(list(\hat{g}_{intra})) \end{array}
  5:
  6:
             \hat{g}_{reduced} \leftarrow \text{Reduction}(list(\hat{g}_{intra}))
  7:
             \hat{g}_{reduced} \leftarrow \mathsf{Hadamard}(\hat{g}_{reduced})
  8:
             \hat{qg}_{4bit} \leftarrow \text{Quantize4Bit}(\hat{g}_{reduced})
 9:
            list(\hat{qg}_{4bit\_inter}) \leftarrow \textbf{InterAlltoAll}(\hat{qg}_{4bit})
10:
             list(\hat{g}_{inter}) \leftarrow \text{Dequantize}(list(\hat{q}g_{4bit\ inter}))
11:
             \hat{g}_{final\_reduced} \leftarrow \text{Reduction}(list(\hat{g}_{inter}))
            g_{final\_reduced} \leftarrow \text{Hadamard}(\hat{g}_{final\_reduced})
```

Simplifying Hadamard Transforms: In a naive implementation, the Hadamard transform would be applied at each step before quantization and after dequantization. However, by leveraging the orthogonality of the Hadamard transform, i.e.  $H \cdot H = I$ , we omit the transform after the intra-node all-to-all dequantization (Algorithm 3, Line 6) and before the inter-node quantization (Algorithm 3, Line 8). Furthermore, by utilizing the distributive property, i.e.,  $\sum_i Hg_i = H\sum_i g_i$ , we move the second Hadamard transform from after the inter-node dequantization (Algorithm 3, Line 11) to after the final reduction (Algorithm 3, Line 12). These simplifications reduce the unnecessary computational overhead associated with repeated Hadamard transforms.

Fusing GPU Kernels with Group Size Alignment: To mitigate additional data movement from global memory—which typically exhibits the slowest memory bandwidth—, we fuse the Hadamard transform with the (de)quantization operations into a single CUDA kernel. This fusion allows the operations to run nearly as fast as the quantization operation alone. It is worth noting that, for this fusion to be efficient, there must be an alignment between the two. Specifically, the size of the quantization group must be divisible by the size of the Hadamard matrix, ensuring that memory traffic remains within the kernel block. We choose H to be small (e.g.,  $32 \times 32$ ) because, at this size, the transform operation on the GPU is typically memory-bound and incurs minimal overhead. While larger H sizes offer better smoothing capabilities, we find that a  $32 \times 32$  matrix is sufficient to effectively smooth outliers in gradients.

## **Theoretical Analysis**

#### Counterexample of Biased Weight Compression

One of the advantages of our proposed weight difference compression is the compatibility to both biased and unbiased compressors. Note that using biased compressors directly on weight compression incurs issues in convergence under standard assumptions, as avoided in QSDP [18] or ZeRO++ [32]. We illustrate such issues in the following toy example.

<span id="page-5-1"></span>**Counterexample 4.1.** Consider a least square problem with  $w^* = (0,0)^{\top}$ :  $\min_{w \in \mathbb{R}^2} \left[ f(w) = \|w\|^2 \right]$ , and stochastic gradient  $g(w) = (4w_1, 0)^{\top}$  with probability 0.5 and  $g(w) = (0, 4w_2)^{\top}$  with probability 0.5, thus  $\mathbb{E}[g(w)] = \nabla_w f(w)$ . We use the initial value  $w_{init} = (1, -1)^{\mathsf{T}}$ , the learning rate  $\eta < 0.125$ , and the following nearest ternary quantizer:  $s = \max(|w|), q(w) = round(w/s) * s$ , where  $|\cdot|$  is element-wise absolute value, and  $round(\cdot)$ quantizes each element to the nearest value in  $\{-1, 0, 1\}$ . It is easy to check that for SGD under such settings, the weights before quantization will be either  $(1-4\eta,-1)^{\top}$  or  $(1,-1+4\eta)^{\top}$ , resulting in  $q(w) = (1, -1)^{\mathsf{T}}$ , which means that SGD with ternary weight quantization gets stuck at the initial value in this case, while SGD without weight quantization and SGD with weight difference quantization both converge to the optimal.

## <span id="page-5-0"></span>4.2 Convergence Analysis

To theoretically analyze the convergence of our distributed training algorithm with communication compression, we focus on the following SGD variant with gradient compression and weight difference compression. We use SGD to solve the following optimization problem:  $f^* = \min_w f(w)$ , where f(w) is the objective function,  $w \in \mathbb{R}^d$  is the model parameter.

#### Algorithm 4 SGD with SDP4Bit

- 1: Initialize main parameter weights  $w_0$
- 2: Initialize compressed parameter weights  $\tilde{w}_0 \leftarrow w_0$
- 3: **for all** iteration  $t \in [T]$  **do**
- Compute gradient:  $g_{t-1} = \nabla f(\tilde{w}_{t-1}; \zeta_{t-1})$ Compress gradient:  $\tilde{g}_{t-1} = \mathcal{U}_g(g_{t-1})$
- 5:
- Update main weights:  $w_t \leftarrow w_{t-1} \eta \tilde{g}_{t-1}$ 6:
- Compress weight difference:  $\tilde{\Delta}_t = C_w(w_t \tilde{w}_{t-1})$
- Update compressed weights:  $\tilde{w}_t \leftarrow \tilde{w}_{t-1} + \tilde{\Delta}_t$
- <span id="page-5-2"></span>9: end for

Note that we use unbiased compressors for gradient reduction, and arbitrary (potentially biased) compressors for weight collection. We formally define these two classes of compressors as follows.

**Definition 4.1** (Unbiased  $\kappa$ -approximate compressor [1]). An operator  $\mathcal{U}: \mathbb{R}^d \to \mathbb{R}^d$  is a  $\kappa$ -approximate compressor for  $\kappa \geq 0$  if  $\mathbb{E}[\mathcal{U}(v)] = v$  and  $\mathbb{E}||\mathcal{U}(v) - v||^2 \leq \kappa ||v||^2$ ,  $\forall v \in \mathbb{R}^d$ .

**Definition 4.2** ( $\delta$ -approximate compressor [13]). An operator  $\mathcal{C}: \mathbb{R}^d \to \mathbb{R}^d$  is a  $\delta$ -approximate compressor for  $\delta \in [0,1]$  if  $\mathbb{E}\|\mathcal{C}(v)-v\|^2 \leq (1-\delta)\|v\|^2, \forall v \in \mathbb{R}^d$ .

Remark 4.1. Note that, in a certain sense, the class of  $\delta$ -approximate compressors contains the class of unbiased compressors. It is easy to check that any  $\kappa$ -approximate unbiased compressor  $\mathcal{U}$  can be converted to a  $\frac{1}{1+\kappa}$ -approximate biased compressor  $\mathcal{C}(v) = \frac{1}{1+\kappa}\mathcal{U}(v)$ . Furthermore, the class of  $\delta$ -approximate compressors typically provides more options such as top-k sparsifiers, and top-k low-rank compressors. Thus, we consider arbitrary (biased or unbiased)  $\delta$ -approximate compressors for weight compression in our theoretical analysis.

Remark 4.2. For distributed training with P workers, we define the compressed gradient as  $\tilde{g}_t = \mathcal{U}_g(g_t) = \frac{1}{P} \sum_{i \in [P]} \mathcal{U}_g'(g_{t,i})$ , where  $g_t = \frac{1}{P} \sum_{i \in [P]} g_{t,i}$ , and  $g_{t,i}$  is the stochastic gradient from the ith worker in t iteration. We assume that  $\mathcal{U}_g$  is an unbiased  $\kappa$ -approximate compressor of the average gradient  $g_t$ .

<span id="page-6-0"></span>**Assumption 4.1.** (Smoothness) We assume that f(x) is L-smooth:  $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|, \forall x, y \in \mathbb{R}^d$ , which implies  $f(y) - f(x) \le \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|y - x\|^2$ .

<span id="page-6-1"></span>**Assumption 4.2.** For any stochastic gradient  $\nabla f(w;\zeta)$ , where  $\zeta$  is an independent random sample, we assume unbiasedness  $\mathbb{E}[\nabla f(w;\zeta)|w] = \nabla f(w)$ , and bounded variance  $\mathbb{E}[\nabla f(w;\zeta) - \nabla f(w)||^2|w] \leq \rho \|\nabla f(w)\|^2 + \sigma^2$  ([27], Assumption 3).

We derive the following error bounds on the convergence of SDP4Bit under the above assumptions. All proofs can be found in Appendix A.

<span id="page-6-2"></span>**Theorem 4.1** (Convergence error bound). For arbitrary non-convex function under Assumption 4.1 and Assumption 4.2, taking learning rate  $\eta \leq \frac{1}{10L(\frac{2}{\delta} + \rho\kappa + \rho + \kappa)}$ , Algorithm 4 converges to a critical point with the following error bound:

$$\frac{\sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2]}{T+1} \le \frac{80L\left(\frac{2}{\delta} + \rho\kappa + \rho + \kappa\right)(f(w_0) - f^*)}{T+1} + 4\sigma\sqrt{\frac{(11-\delta)(\kappa+1)L(f(w_0) - f^*)}{T+1}}.$$

Remark 4.3. Note that compared to QSDP [18], our convergence analysis does not require Polyak-Łojasiewicz condition or the specific choice of weight quantization (random shift). In other words, Theorem 4.1 shows that our proposed algorithm has the same  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  convergence rate as ordinary SGD for general non-convex functions, but under much weaker assumptions compared to QSDP.

## 5 Evaluation

#### 5.1 Experimental Setup

**Hardware:** The experiments are conducted on two different clusters to evaluate SDP4Bit across varying network environments: **1)** 16 nodes, each node equipped with 4 Nvidia A100-SXM4-40GB GPUs. All nodes are interconnected with a 100 Gbps Slingshot10 network, providing slower internode bandwidth. **2)** 16 nodes, each node equipped with 8 Nvidia H800-SXM5-80GB GPUs. Each node is connected using 8 InfiniBand links, achieving a total bandwidth of 3.2 Tbps, providing higher inter-node bandwidth.

**Baselines:** We use BFloat16/Float32 (weights/gradients) mixed-precision in Megatron-LM [26] as our basic *Baseline* for both accuracy and E2E throughput analysis. Within each set of experiments, we ensure consistent hyper-parameters to ensure fairness. Detailed parameters are provided in Appendix D. Additionally, we implement another baseline for comparison in Megatron-LM, using the same quantization strategy in ZeRO++, employing 4-bit quantization for both weights (groupwise weight quantization, refered to as qW) and gradients (twice all-to-all with uniform level 4-bit quantization, refer to as ULq).

**Dataset and Models:** To demonstrate that SDP4Bit does not adversely affect end-to-end training loss, we conduct pre-training on GPT-series [23] models ranging from 125M to 6.7B parameters

Table 1: Final validation loss↓ of pre-training with different quantization strategies.

<span id="page-7-1"></span>

| GPT Model | Baseline | We<br>qW | ight<br>qWD | Gra<br>TLq | dient<br>TLq-HS | SDP4Bit |
|-----------|----------|----------|-------------|------------|-----------------|---------|
| 125M      | 2.29392  | 2.27405  | 2.29274     | 2.30479    | 2.29528         | 2.29590 |
| 350M      | 2.08719  |          | 2.08730     | 2.09551    | 2.08912         | 2.08964 |
| 1.3B      | 1.92774  |          | 1.92881     | 1.95075    | 1.93134         | 1.93238 |

on the Pile dataset [8], using validation loss as the accuracy measure. Each test runs for 80,000 iterations, processing over 40 billion tokens. For throughput evaluation, we select models ranging from 1.3B to 18B parameters, with end-to-end training throughput as the metric. In these tests, the accumulation step is set to 1. Note that model parallel is required for models larger than 6.7B, and different tensor parallel sizes are used on A100 and H800 clusters for models larger than 13B. Please refer to Appendix B Table 8 for detailed model parallel configuration.

#### 5.2 Accuracy Evaluation

First, we analyze the impact of SDP4Bit on the accuracy of E2E training. As shown in Table 1, the training results for three different model sizes indicate that the final loss of SDP4Bit is comparable to the baseline, with a maximum increase of only 0.24%. Additionally, Figure 1 details the training curve of GPT-6.7B, demonstrating that the training curve of SDP4Bit perfectly aligns with the baseline. This indicates that *the impact of SDP4Bit on accuracy is negligible*. In contrast, the 4-bit quantization strategy in ZeRO++ (which directly applies quantization to weights and uniformly uses 4-bit quantization with all-to-all for gradients) results in significant accuracy degradation.

Next, we break down and analyze each strategy within SDP4Bit. 1) For weight communication reduction, as shown in Table 1, directly quantizing weights (qW) to 4 bits results in a validation loss that increase of up to 12% compared to the baseline. In contrast, our weight difference quantization (qWD) method achieves a validation loss nearly identical to the baseline. Notably, we use a consistent quantization group size of 2048 for both tests. 2) For gradient communication reduction, as shown in Figure 5, applying Uniform Level quantization (ULq), similar to the method used in ZeRO++, results in a significant gap in the loss compared to the baseline. In comparison, our Two Level quantization (TLq) significantly mitigate the loss gap between with baseline. Additionally, Figure 6 illustrates the effectiveness of the Hadamard transformation in smoothing outliers. Table 1 and Figure 5 further demonstrate the contribution of the Hadamard smoother to accuracy. Notably, compared to TLq, TLq-HS further narrows the validation loss gap, making it almost identical to the baseline.

<span id="page-7-0"></span>![](_page_7_Figure_6.jpeg)

Figure 5: Validation loss comparison for the Baseline, ULq, TLq, and TLq-HS on the GPT-125M model. Uniformly applying 4-bit gradient quantization twice results in a noticeable gap compared to the baseline. In contrast, two-level quantization (8-bit for intranode and 4-bit for inter-node) mitigates this gap. The Hadamard smoother further reduces the gap, making the loss nearly identical to the baseline.

![](_page_7_Figure_8.jpeg)

Figure 6: Comparison of gradient histograms before and after the Hadamard transformation. The transformation reduces the impact of outliers, resulting in a smoother gradient distribution.

Table 2: E2E throughput↑ on different model sizes with std.

<span id="page-8-1"></span>

|               | 4xA100, 10         | 6 nodes (Slings   | 8xH800,       | 16 nodes (Infini   | iBand)            |               |
|---------------|--------------------|-------------------|---------------|--------------------|-------------------|---------------|
| Model<br>Size | Baseline<br>TFLOPs | SDP4Bit<br>TFLOPs | Speedup       | Baseline<br>TFLOPs | SDP4Bit<br>TFLOPs | Speedup       |
| 1.3B          | 24.1 ±0.03         | $57.6 \pm 0.03$   | 2.39×         | 69.1 ±0.96         | $106.0 \pm 2.66$  | 1.53×         |
| 2.7B          | $24.0 \pm 0.00$    | $58.4 \pm 0.07$   | $2.43 \times$ | $71.9 \pm 0.56$    | $116.9 \pm 0.98$  | $1.63 \times$ |
| 6.7B          | $10.8 \pm 0.00$    | $37.1 \pm 0.00$   | $3.44 \times$ | $26.2 \pm 0.33$    | $77.9 \pm 2.43$   | $2.97 \times$ |
| 13B           | $9.7 \pm 0.04$     | $26.0 \pm 0.03$   | $2.68 \times$ | $13.9 \pm 0.17$    | $53.5 \pm 1.36$   | $3.85 \times$ |
| 18B           | $10.2 \pm 0.00$    | $29.8 \pm 0.04$   | $2.92 \times$ | $14.5 \pm 0.07$    | $59.2 \pm 1.37$   | $4.08 \times$ |

<span id="page-8-0"></span>Table 3: Final validation loss↓ of GPT-125M with different group sizes.

|        | C 1                    |               |                |                 |
|--------|------------------------|---------------|----------------|-----------------|
| Baseli | Baseline Val Loss      |               | 2.29392        |                 |
| TLq-HS | Group Size<br>Val Loss | 64<br>2.29537 | 128<br>2.29528 | 512<br>2.29670  |
| qWD    | Group Size<br>Val Loss |               |                | )48<br>9274     |
| qW     | Group Size<br>Val Loss | 32<br>2.39580 | 128<br>2.44712 | 2048<br>2.57312 |

Table 4: Performance of Different Quantization Strategies on GPT-1.3B over 32 A100 with standard deviation.

| Quantization<br>Strategy  | Grad. Comm.<br>Time (ms) | TFLOPs          |
|---------------------------|--------------------------|-----------------|
| Baseline                  | $379.3 \pm 0.34$         | $24.4 \pm 0.00$ |
| TLq-HS                    | $45.9 \pm 0.03$          | $43.3 \pm 0.05$ |
| ULq                       | $45.0 \pm 0.03$          | $43.3 \pm 0.04$ |
| SDP4Bit                   | $45.8 \pm 0.03$          | $58.5 \pm 0.07$ |
| SDP4Bit<br>(HS w/o fused) | 64.6 ±0.05               | 55.2 ±0.07      |

#### 5.3 Throughput Evaluation

Next, we evaluate the improved E2E throughput, measured in FLOPS per second, of SDP4Bit on both hardware platforms. For all tests, the results are averaged over 10 iterations after 20 warm-up iterations. As shown in Table 2, SDP4Bit achieves an E2E training speedup of up to  $4.08\times$ . For models with the same model parallel configuration (e.g., 1.3B and 2.7B; 13B and 18B), both the E2E throughput and speedup from SDP4Bit increase as the model size grows due to larger models having higher computational efficiency but also encountering increased communication overhead.

The throughput of the 1.3B, 2.7B, and 6.7B models across the two platforms indicates that SDP4Bit provides a more significant speedup when network bandwidth is lower. This is because lower bandwidth results in higher communication overhead, which SDP4Bit effectively reduces through efficient quantization techniques.

<span id="page-8-2"></span>![](_page_8_Figure_9.jpeg)

Figure 7: Scalability of SDP4Bit.

![](_page_8_Figure_11.jpeg)

Figure 8: Throughput breakdown of SDP4Bit on GPT-2.7B.

In addition, we demonstrate the scalability of SDP4Bit using GPT models of 6.7B and 13B parameters, with tests conducted on up to 128 GPUs, as shown in Figure 7. Under low bandwidth conditions, SDP4Bit achieves an average speedup of  $3.40\times$  for the 6.7B model and  $2.49\times$  for the 13B model. In high-bandwidth InfiniBand environments, the speedup averages  $3.08\times$  for the 6.7B model and  $3.73\times$  for the 13B model. The comparatively lower speedup for the 13B model under low bandwidth conditions can be attributed to the introduction of pipeline parallelism, which diminishes the proportion of communication handled by ShardedDP. Overall, SDP4Bit consistently maintains stable speedup performance across various GPU numbers and network environments.

#### 5.4 Ablation Study

Components Breakdown. Figure [8](#page-8-2) demonstrates the throughput improvement of qWD, TLq-HS, and their combination (SDP4Bit) on two different platforms. qWD alone provides a speedup ranging from 1.1× to 1.2×, while TLq-HS alone results in an E2E speedup of 1.4× to 1.8×. The notable benefit from gradient quantization stems from the high communication overhead associated with Float32 gradients in baseline training, which is higher compared to BFloat16 weights. When they are applied together, SDP4Bit achieves a more substantial speedup, ranging from 1.6× to 2.4×.

TLq-HS vs. ULq. Table [4](#page-8-0) compares gradient quantization between TLq-HS and ULq. The results show that although TLq-HS employs 8-bit quantization for intra-node gradient communication, it introduces negligible overhead compared to 4-bit communication. This is due to 1) the high bandwidth of intra-node communication and 2) the fact that most intra-node communication is overlapped with the slower inter-node communication.

Hadamard Kernel Fusion. Table [4](#page-8-0) shows that, compared to the SDP4Bit without fusing Hadamard Transform kernel, our optimized SDP4Bit reduces gradient communication overhead by 29%. Additionally, we provide a throughput comparison in Table [5](#page-17-0) to further illustrate the impact of the Hadamard transformation. The results confirm that our Hadamard kernel fusion effectively reduces the overhead, making the transformation nearly zero-overhead and even matching the performance of quantization without the Hadamard transformation.

Convergence with Different Group Sizes. Table [3](#page-8-0) examines the impact of various quantization granularities on the end-to-end validation loss during the pre-training of the GPT-125M model. For TLq-HS, a gradient quantization group size of 128 presents sufficient, with smaller sizes yielding no significant accuracy improvements. For qWD, a quantization group size of 2048 achieves training accuracy comparable to the baseline. Table [3](#page-8-0) also presents the 4-bit weight quantization (*qW*) while using small group size. It is evident that even with very small group size (e.g., 32), direct 4-bit quantization leads to a significant gap in accuracy compared to the baseline, making 4-bit quantization for weights suboptimal.

