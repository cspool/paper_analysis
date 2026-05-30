# <span id="page-0-1"></span>VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization

Sihan Yang<sup>1</sup> Runsen Xu<sup>1</sup>,<sup>2</sup> Chenhang Cui<sup>3</sup> Tai Wang<sup>1</sup>† Dahua Lin<sup>1</sup>,<sup>2</sup> Jiangmiao Pang<sup>1</sup>† 1 Shanghai AI Laboratory <sup>2</sup> The Chinese University of Hong Kong <sup>3</sup> National University of Singapore † Corresponding Author

taiwang.me@gmail.com pangjiangmiao@gmail.com

## Abstract

*Large Multimodal Models (LMMs) excel in visual-language tasks by leveraging numerous visual tokens for fine-grained visual information, but this token redundancy results in significant computational costs. Previous research aimed at reducing visual tokens during inference typically leverages importance maps derived from attention scores among vision-only tokens or vision-language tokens to prune tokens across one or multiple pruning stages. Despite this progress, pruning frameworks and strategies remain simplistic and insufficiently explored, often resulting in substantial performance degradation. In this paper, we propose VFlowOpt, a token pruning framework that introduces an importance map derivation process and a progressive pruning module with a recycling mechanism. The hyperparameters of its pruning strategy are further optimized by a visual information flow-guided method. Specifically, we compute an importance map for image tokens based on their attention-derived context relevance and patch-level information entropy. We then decide which tokens to retain or prune and aggregate the pruned ones as recycled tokens to avoid potential information loss. Finally, we apply a visual information flow-guided method that regards the last token in the LMM as the most representative signal of text-visual interactions. This method minimizes the discrepancy between token representations in LMMs with and without pruning, thereby enabling superior pruning strategies tailored to different LMMs. Experiments demonstrate that VFlowOpt can prune 90% of visual tokens while maintaining comparable performance, leading to an 89% reduction in KV-Cache memory and 3.8*× *faster inference. Code is at [https://github.com/sihany077/VFlowOpt.](https://github.com/sihany077/VFlowOpt)*

## 1. Introduction

Large Multimodal Models (LMMs) [\[3,](#page-8-0) [9,](#page-8-1) [24,](#page-9-0) [25\]](#page-9-1) have attained remarkable performance in tasks such as visual

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Figure 1. VFlowOpt Performance and Efficiency. Our VFlowOpt significantly outperforms other LMM token reduction methods across three general benchmarks (MME, MMStar, and MMBench) on LLaVA-OneVision-7B. It achieves zero performance degradation when reducing 50% of visual tokens. Furthermore, when retaining only 10% of the tokens, it achieves 90% of the original performance while reducing KV-Cache memory usage by 89% and accelerating inference by 3.8×.

question answering [\[2,](#page-8-2) [14\]](#page-8-3) and multimodal reasoning [\[47,](#page-10-0) [48\]](#page-10-1), making them indispensable for applications like autonomous driving [\[10,](#page-8-4) [34,](#page-9-2) [38\]](#page-9-3) and robotics [\[20,](#page-9-4) [22,](#page-9-5) [27\]](#page-9-6). To capture fine-grained visual details, modern LMMs treat images as token sequences, with models such as LLaVA-1.5 [\[24\]](#page-9-0) processing hundreds of tokens and LLaVA-OneVision [\[19\]](#page-8-5) handling up to several thousand tokens. Nonetheless, the increasing quantity of visual tokens significantly magnifies computational cost, memory usage, and inference time, creating a critical bottleneck for LMM deployment, particularly in resource-constrained environments or latency-sensitive scenarios.

To alleviate these constraints, recent work explores various methods for reducing the number of visual tokens during inference. Most existing approaches evaluate the importance of visual tokens in LMMs using attention-based importance maps derived either from vision-only tokens [\[43,](#page-10-2) [52\]](#page-10-3) or from text-visual interactions [\[39,](#page-9-7) [55\]](#page-10-4). Following a predefined pruning ratio, visual tokens are pruned in a sin<span id="page-1-0"></span>gle stage [\[36\]](#page-9-8) or progressively across multiple stages [\[45\]](#page-10-5), often with heuristic strategies applied uniformly across different LMMs. Although promising, these methods are simplistic and underexplored, with coarse-grained pruning often causing significant performance drops due to lacking fitness for different LMMs models' characteristics.

In response to these limitations, we propose a novel token pruning framework, VFlowOpt, that more accurately estimates token importance by integrating attention calibration and entropy, and reduces information loss through token recycling and progressive pruning strategy with finegrained pruning ratios. Furthermore, by customizing pruning strategies for different models, our framework better preserves model performance. Specifically, the pruning framework consists of importance map computation and a progressive pruning module with a recycling mechanism. We decompose the importance map of image tokens into two aspects: the importance of tokens in the visual context, reflected by their attention maps, and the richness of visual information in each image patch, captured by its information entropy. Based on the weighted summation of these two factors, we determine which tokens to retain or prune across multiple pruning stages and introduce a recycling mechanism to aggregate pruned tokens to avoid potential information loss. Built with this pruning strategy with several hyperparameters, a key ingredient is the visual information flow-guided method to optimize the pruning strategy, which treats the last token in each pruning stage as the most representative signal of text-visual interactions during inference. By minimizing the difference between token representations in LMMs with and without pruning, VFlowOpt ultimately delivers superior pruning strategies specifically tailored to different LMMs.

Comprehensive experiments on multiple visionlanguage benchmarks validate the efficacy of VFlowOpt. In particular, it substantially lowers computational cost while preserving competitive performance. As shown in Fig. [1,](#page-0-0) VFlowOpt can prune 50% of visual tokens with negligible performance loss. Moreover, it can prune 90% of visual tokens while maintaining 90% of the model's performance, resulting in an 89% reduction in KV-Cache memory usage and a 3.8× speedup in inference. These experimental results highlight its effectiveness as a practical and efficient solution for deploying LMMs in resource-constrained environments.

## 2. Related Work

## 2.1. Large Multimodal Models

Large Multimodal Models (LMMs) [\[3,](#page-8-0) [9,](#page-8-1) [24,](#page-9-0) [25\]](#page-9-1) extend the reasoning capabilities of Large Language Models (LLMs) [\[4,](#page-8-6) [35,](#page-9-9) [41,](#page-9-10) [42\]](#page-10-6) to vision-language tasks by integrating a pre-trained vision encoder [\[32,](#page-9-11) [49\]](#page-10-7) with a language model, linked by an alignment module such as an MLP, or a query-based network. This design transforms visual inputs into token sequences that the LLM can process, facilitating multimodal prompts for tasks like visual question answering [\[2,](#page-8-2) [7,](#page-8-7) [14,](#page-8-3) [47\]](#page-10-0). To enhance performance, advanced LMMs, such as LLaVA-OneVision [\[19\]](#page-8-5) and Qwen2-VL [\[37\]](#page-9-12), can encode higher-resolution images into more image tokens, thereby capturing more granular visual details. However, as image resolution grows, the number of visual tokens rises exponentially, leading to substantially higher computational costs. For example, LLaVA-1.5 [\[24\]](#page-9-0) processes 336×336 images into 576 tokens, whereas LLaVA-OneVision can handle 1152×1152 images, producing 7,290 tokens. This challenge becomes even more pronounced in video-based models like LongVA [\[51\]](#page-10-8), which must process tokens across numerous frames. Finegrained visual tokenization boosts LMM performance but poses an inference bottleneck, driving efforts to balance performance and cost through token reduction.

## 2.2. Token Reduction for LMMs

Token reduction [\[5,](#page-8-8) [23\]](#page-9-13) has become a key strategy for improving the efficiency of LMMs by mitigating the computational cost associated with extensive visual token sequences. Existing methods can be broadly classified into training-based and training-free approaches. Trainingbased methods, such as LLaVA-Mini [\[53\]](#page-10-9) and LLaVolta [\[6\]](#page-8-9), introduce additional modules during model training to compress visual tokens and preserve critical information, while approaches like ATP-LLaVA [\[46\]](#page-10-10) and p-MoD [\[50\]](#page-10-11) train pruning modules to dynamically retain important tokens across LLM layers. However, these methods demand substantial computational resources to retrain the models, limiting their real-world applicability. In contrast, trainingfree methods prune tokens without additional training, often leveraging attention mechanisms to identify and discard redundant tokens. For instance, FastV[\[8\]](#page-8-10) and SparseVLM exploit text–visual attention to rank token importance, whereas FasterVLM [\[52\]](#page-10-3), VisionZip [\[43\]](#page-10-2), and VTC-CLS [\[36\]](#page-9-8) rely on [CLS] token attention in the vision encoder to evaluate token importance and prune redundant tokens. FitPrune [\[45\]](#page-10-5) and PDrop [\[39\]](#page-9-7) propose a progressive multi-stage pruning strategy in LMMs to fully utilize visual information. However, the pruning strategies in these methods often lack adaptability to different LMMs, which frequently leads to significant performance drops.

## 3. Method

In this section, we introduce VFlowOpt, a framework that employs a pruning strategy to reduce redundant visual tokens during LMM inference while preserving essential visual information. This strategy involves multiple hyperparameters that our VFlowOpt framework automatically op-

<span id="page-2-3"></span><span id="page-2-2"></span>![](_page_2_Figure_0.jpeg)

Figure 2. **Overview of VFlowOpt.** (1) During inference, VFlowOpt first assesses the importance of visual tokens, based on which progressive token pruning is performed. After the initial pruning stage, the pruned tokens are merged and recycled. The pruning strategy used in this process is defined by the (2) Optimization Stage. (a) The importance map is computed by combining the attention of relatively important tokens with the entropy of image patches. (b) The pruned tokens are grouped into grid cells, where each cell has a side length of a. Within each grid cell, the pruned tokens are fused using a weighted average, with their importance values as weights, and then recycled. (2) VFlowOpt optimizes the pruning strategy by minimizing the discrepancy of the last token in the final layer of the LMM with and without applying the pruning strategy.

<span id="page-2-1"></span>![](_page_2_Figure_2.jpeg)

Figure 3. The attention of redundant tokens (marked in red) fails to reflect the importance of other tokens and instead focuses on similar tokens, such as background elements.

timizes for proper configuration—an essential factor for maintaining model performance. Specifically, Section 3.1 explains how our method evaluates the importance of visual tokens, Section 3.2 presents progressive pruning and token recycling strategies, and finally, Section 3.3 details how

VFlowOpt customizes the superior pruning strategy (i.e., selects the pruning hyperparameters) for different LMMs.

#### <span id="page-2-0"></span>3.1. Visual Token Importance Estimation

For a general approach to evaluate visual token importance, the previous works [16, 44] propose that the importance of visual tokens can be estimated using the average attention from all other tokens in ViTs. Although it has achieved some success, we find that redundant tokens (e.g., tokens corresponding to background regions) often assign disproportionately high attention to other similar redundant tokens (shown in Fig. 3), reducing the reliability of the importance estimation. To address this, we first identify relatively important tokens based on the attention they receive from all tokens (tokens receiving higher attention are considered relatively important), as shown in Fig. 2 (a). We then exclude redundant tokens and use the attention from these relatively important tokens as a more robust metric. Additionally, we incorporate the information entropy of image patches into

<span id="page-3-2"></span>the importance score to prioritize tokens corresponding to visually informative regions.

To identify relatively important visual tokens, we define a threshold:

$$\tau = t \cdot \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N} A_{ij}, \tag{1}$$

where  $\tau$  is the threshold,  $A_{ij}$  is the attention weight from token i to token j within a ViT layer, N is the total number of tokens, and t is a sensitivity hyperparameter. We then treat tokens with a total attention exceeding  $\tau$  as relatively important:

$$\mathcal{K} = \{ j \mid \sum_{i=1}^{N} A_{ij} > \tau \}.$$
 (2)

Here, K denotes the set of indices of these relatively important tokens.

The information entropy of an image patch corresponding to token i is defined as:

$$H(V_i) = -\sum_{k=0}^{L-1} p_k \log p_k,$$
 (3)

where  $V_i$  is the image patch corresponding to token  $i, p_k$  is the proportion of pixels in the patch with gray level k, and L is the total number of possible gray levels (256 for 8-bit images). Here, the gray level is defined as the average of the RGB channel values, providing a scalar representation of pixel brightness. Higher entropy reflects greater diversity in pixel intensities and richer visual information.

The importance score of token i is computed by combining the attention it receives from tokens in  $\mathcal{K}$  with the entropy of its corresponding image patch, normalized via the softmax function:

$$I_i = \sum_{k \in \mathcal{K}} A_{ki} + \alpha \cdot \frac{\exp(H(V_i))}{\sum_{j=1}^{N} \exp(H(V_j))}, \tag{4}$$

where  $I_i$  denotes the importance score of token i,  $A_{ki}$  is the attention weight from token k to token i,  $H(V_i)$  is the entropy of the image patch corresponding to token i,  $\alpha$  is a hyperparameter that determines the entropy term's contribution, and N is the total number of tokens. This approach mitigates biases from redundant tokens and leverages the visual information of image patches, thereby facilitating effective estimation of visual token importance.

#### <span id="page-3-0"></span>3.2. Progressive Pruning and Token Recycling

Previous studies [40, 54] indicate that visual tokens play a more critical role in the shallower layers of LMMs, whereas redundancy tends to increase in deeper layers. Based on this, we adopt a progressive pruning strategy. To balance the pruning process's simplicity with its fine-grained configuration, we evenly divide the LMM into three stages.

At the beginning of each stage, a predetermined fraction of visual tokens with higher importance scores (as defined in Sec. 3.1) is retained according to the stage-specific retention ratios  $R = [R_1, R_2, R_3]$ , while the rest are pruned. Notably, the position IDs of visual tokens remain unchanged following pruning, preserving the original spatial structure of the visual input. Following the previous work [40], we compute the average visual token retention rate across the entire LMM as follows:

$$\overline{R} = \frac{R_1 \cdot L_1 + R_1 \cdot R_2 \cdot L_2 + R_1 \cdot R_2 \cdot R_3 \cdot L_3}{L}, \quad (5)$$

where  $L_1$ ,  $L_2$ , and  $L_3$  represent the number of layers in the three stages, respectively, and L denotes the total number of layers in the LMM.

To prevent the loss of any small but potentially significant information during the initial pruning process (before visual tokens are fed into the LLM), we propose a token merging and recycling strategy to compactly represent redundant visual information. Specifically, We use a square grid (with each cell having side length *a*) to group pruned tokens that fall into the same grid cell, as shown in Fig. 2 (b). Within each cell, the pruned tokens are fused into a single token by computing a weighted average of their representations, using their importance scores as weights. This fused token then replaces the pruned token with the highest importance score in that cell and is incorporated into the set of retained tokens.

Formally, a token  $t_i$  with spatial coordinates  $(x_i, y_i)$  belongs to a grid cell  $\mathcal{G}_{p,q}$  if

$$|x_i/a| = p \quad \text{and} \quad |y_i/a| = q, \tag{6}$$

where p and q are the row and column indices of the grid cell, respectively. Suppose there are k pruned tokens in  $\mathcal{G}_{p,q}$  with corresponding importance scores  $I_1, I_2, \ldots, I_k$  and token representations  $\mathbf{t}_1, \mathbf{t}_2, \ldots, \mathbf{t}_k$ . The fused token  $\mathbf{t}_{\text{merged}}^{p,q}$  is computed as:

$$\mathbf{t}_{\text{merged}}^{p,q} = \frac{\sum_{i=1}^{k} I_i \cdot \mathbf{t}_i}{\sum_{i=1}^{k} I_i}.$$
 (7)

We then assign  $\mathbf{t}_{\text{merged}}^{p,q}$  to the position of the pruned token with the highest importance score  $I_{\text{max}}$  in  $\mathcal{G}_{p,q}$ , counting it among the retained tokens. By combining token pruning with this recycling strategy, we reduce token redundancy while avoiding potential information loss.

#### <span id="page-3-1"></span>3.3. Pruning Strategy Optimization

The pruning strategies described in Sec. 3.1 and Sec. 3.2 involve several key hyperparameters that directly affect the performance of LMMs after token pruning. Properly defining these hyperparameters is crucial for preserving model performance. However, existing approaches often rely on

<span id="page-4-1"></span>manually designed pruning strategies and apply the same strategy across different LMMs, without considering the unique characteristics of each model. This coarse-grained approach can lead to significant performance degradation due to its lack of fitness for different LMMs, leaving the task of designing pruning strategies tailored to specific LMMs as a formidable challenge.

Previous interpretability studies of LMMs [\[17,](#page-8-12) [56\]](#page-10-14) offer crucial insights into their internal mechanisms. Specifically, these works reveal that in the lower and middle layers of LMMs, visual information from visual tokens is aggregated into the corresponding query text tokens. In the higher layers, the multimodal representation encoded in the query text tokens is further progressively propagated to the final position of the input sequence, ultimately influencing the subsequent inference process.

Inspired by this insight, we propose a framework that requires only a small amount of unlabeled data and leverages the internal flow of visual information within LMMs to search for the superior pruning strategy tailored to the characteristics of different LMMs, as shown in Fig. [2](#page-2-2) (2). Specifically, we recast the task of designing the superior pruning strategy as an optimization problem, aiming to minimize the discrepancy in visual information flow with and without applying a pruning strategy to perform visual token pruning. In this framework, we treat the final token in the last layer as the representative outcome of the visual information flow. We use cosine similarity to measure the similarity of the final token representation with and without visual token pruning. A higher similarity indicates that the discrepancy between the visual information flow with and without pruning is smaller. Accordingly, we define the following optimization objective:

$$\max_{s \in \mathcal{S}} f(s) = \operatorname{CosineSim}(h_f, g_s(h_f)). \tag{8}$$

Here, h<sup>f</sup> represents the representation of the final token in the last layer before pruning, and gs(h<sup>f</sup> ) represents the final token representation after applying the pruning strategy s (with g<sup>s</sup> modeling its effect on the token feature), and S is the solution space.

To efficiently search for the superior pruning strategy s, we employ Bayesian optimization, which systematically explores the hyperparameter space—including the threshold sensitivity t, the entropy weight α, the grid size a, and the pruning ratios R1, R2, and R3—to maximize the target function f(s). In short, Bayesian optimization constructs a surrogate model to approximate the objective function and employs an acquisition function to balance exploration and exploitation, thereby efficiently guiding the search for promising hyperparameter settings. The details of this optimization process are presented in Algorithm [1.](#page-4-0)

#### <span id="page-4-0"></span>Algorithm 1 VFlowOpt with Bayesian Optimization

Input: LMM θ, data samples D, number of Bayesian optimization iterations T, computation budget R, target function f(·): the sum of cosine similarities between the last token in the last layer of the LMM θ with and without visual token pruning, computed over all data samples D.

Output: retention rates R1, R2, R3; threshold sensitivity t; entropy weight in importance score α; grid size a.

- 1: Initialize a Gaussian Process model GP
- 2: Define the acquisition function A(·) (Expected Improvement is adopted)
- 3: Uniformly sample initial points:

```
X0 = {(R1, R2, t, α, a) | Rv, R1, t, α, a ∈ valid ranges}
```

```
4: for all x ∈ X0 do
5: Calculate R3 using the constraint: R = (R1 · L1 +
   R1 · R2 · L2 + R1 · R2 · R3 · L3)/L
6: Form pruning strategy: s = (R1, R2, R3, t, α, a)
7: Evaluate the target function f(s | θ, D)
8: end for
9: for n = 0 to T − 1 do
10: Fit GP to the observed data (Sn, f(Sn | θ, D))
11: Select the next point: xn+1 ← arg maxx A(x; GP)
12: Calculate R3 using the constraint: R = (R1 · L1 +
   R1 · R2 · L2 + R1 · R2 · R3 · L3)/L
13: Form pruning strategy: sn+1 = (R1, R2, R3, t, α, a)n+1
14: Update Sn+1 ← Sn ∪ {sn+1}
15: end for
16: return (R1, R2, R3, t, α, a)
                              ∗
                               that maximize f
```

## 4. Experiments

In this section, we evaluate our approach on various LMMs across diverse image and video benchmarks, followed by an efficiency analysis and an ablation study of each component. Finally, we illustrate how our method affects discrepancies in the visual information flow with and without pruning, offering deeper insights into our approach.

## 4.1. Experiment Setting

Datasets. We evaluate our method on ten imagebased multimodal benchmarks: GQA [\[1\]](#page-8-13), VizWiz [\[15\]](#page-8-14), ScienceQA-IMG [\[29\]](#page-9-15), TextVQA [\[33\]](#page-9-16), ChartQA [\[30\]](#page-9-17), POPE [\[21\]](#page-9-18), MME [\[11\]](#page-8-15), MMBench [\[28\]](#page-9-19), MMStar [\[7\]](#page-8-7), and DocVQA [\[31\]](#page-9-20). For video understanding, we adopt two datasets—SeedBench (video) [\[18\]](#page-8-16) and VideoMME [\[12\]](#page-8-17), where VideoMME is partitioned by video length into short, medium, and long subsets. Further details are presented in the Appendix.

Model Architectures. We integrate VFlowOpt into multiple LMMs, including LLaVA-OneVision-7B, LLaVA-

<span id="page-5-1"></span><span id="page-5-0"></span>

| Method             | MMStar MME |      | MMB  | SQA  | POPE                      | GQA  | VizWiz |      | VQAText ChartQA DocVQA |      | Avg.  |
|--------------------|------------|------|------|------|---------------------------|------|--------|------|------------------------|------|-------|
|                    |            |      |      |      | Upper Bound, 100% Tokens  |      |        |      |                        |      |       |
| Vanilla            | 61.7       | 1581 | 80.8 | 95.8 | 89.1                      | 62.2 | 60.4   | 76.0 | 80.0                   | 87.5 | 100%  |
|                    |            |      |      |      | Retain 50% Tokens (↓ 50%) |      |        |      |                        |      |       |
| FastV(ECCV24)      | 58.9       | 1549 | 79.4 | 92.8 | 87.9                      | 61.5 | 61.1   | 72.5 | 68.6                   | 84.0 | 96.5% |
| SparseVLM(2024.10) | 59.8       | 1577 | 80.5 | 94.1 | 88.1                      | 61.9 | 60.4   | 73.9 | 70.5                   | 80.8 | 97.1% |
| VisionZip(CVPR25)  | 60.4       | 1587 | 80.3 | 94.6 | 89.3                      | 62.7 | 59.8   | 74.2 | 75.4                   | 88.4 | 98.9% |
| VFlowOpt           | 61.3       | 1591 | 81.1 | 95.4 | 89.4                      | 62.4 | 60.0   | 75.1 | 77.8                   | 90.0 | 99.9% |
|                    |            |      |      |      | Retain 25% Tokens (↓ 75%) |      |        |      |                        |      |       |
| FastV(ECCV24)      | 54.0       | 1539 | 77.0 | 88.6 | 83.8                      | 58.2 | 61.0   | 58.3 | 42.7                   | 62.9 | 86.4% |
| SparseVLM(2024.10) | 56.6       | 1520 | 78.7 | 90.3 | 87.2                      | 59.7 | 60.8   | 66.3 | 54.0                   | 66.6 | 90.5% |
| VisionZip(CVPR25)  | 54.6       | 1562 | 78.9 | 90.4 | 88.8                      | 61.0 | 60.4   | 70.0 | 66.3                   | 79.6 | 94.3% |
| VFlowOpt           | 57.8       | 1570 | 79.9 | 92.3 | 89.1                      | 61.2 | 60.4   | 72.5 | 69.1                   | 82.3 | 96.3% |
|                    |            |      |      |      | Retain 10% Tokens (↓ 90%) |      |        |      |                        |      |       |
| FastV(ECCV24)      | 46.0       | 1209 | 70.1 | 81.7 | 77.0                      | 51.5 | 56.4   | 35.6 | 21.3                   | 33.2 | 69.7% |
| SparseVLM(2024.10) | 45.1       | 1191 | 71.8 | 83.7 | 80.0                      | 54.6 | 56.4   | 39.8 | 37.6                   | 39.6 | 74.0% |
| VisionZip(CVPR25)  | 49.5       | 1389 | 74.8 | 86.2 | 86.1                      | 57.2 | 56.8   | 56.4 | 46.1                   | 49.0 | 82.1% |
| VFlowOpt           | 52.0       | 1464 | 75.1 | 88.4 | 85.2                      | 57.3 | 57.3   | 60.2 | 53.6                   | 56.1 | 85.5% |

Table 1. Performance comparison on LLaVA-OneVision-7B under different token retention conditions. "Avg." refers to average accuracy on 10 benchmarks. For each reduction ratio, the best average performance is shown in bold.

NeXT-7B [\[26\]](#page-9-21), and Qwen2-VL-7B [\[37\]](#page-9-12). These models employ various vision encoders (SigLIP, CLIP, and a ViT designed for Qwen2-VL) and LLM backbones (Qwen2- 7B and Vicuna-7B). To prevent out-of-memory issues in Qwen2-VL, we set *max pixels = 3000000*.

Comparison Methods. We compare VFlowOpt against three baseline methods: FastV [\[8\]](#page-8-10), SparseVLM [\[55\]](#page-10-4), and VisionZip [\[43\]](#page-10-2). FastV, and SparseVLM rely on text-visual attention in the LLM to prune visual tokens but differ as follows: FastV performs a single pruning step after the second LLM layer; and SparseVLM uses the attention weights of preselected text tokens to evaluate the importance of visual tokens. VisionZip, in contrast, determines token importance based on the [cls] token's attention; for models lacking a [cls] token, we follow VisionZip's original procedure by computing the average attention each token receives from every other token in the ViTs.

Implementation Details. For LLaVA-OneVision-7B and Qwen2-VL-7B, we perform token pruning at three distinct points: before the LLM, and after the 9th and 18th layers. For LLaVA-NeXT-7B, pruning is conducted before the LLM, and again after the 10th and 20th layers. During optimization, we sample 30 unlabeled instances from each model's training datasets; for models without publicly available training dataset, we instead use random samples from the LLaVA-OneVision training set. The optimization is performed for a total of 50 iterations. For LMMs that modify visual tokens output by the vision encoder (e.g., unpadding and interpolation in LLaVA-OneVision, and unpadding in LLaVA-NeXT), we apply corresponding transformations to the importance maps so they remain fully aligned with the final visual tokens.

#### 4.2. Image Understanding Tasks

We evaluate the proposed VFlowOpt on image understanding benchmarks with LLaVA-OneVision-7B using various pruning ratios, and present the results in Tab. [1.](#page-5-0) Compared to other baselines, VFlowOpt consistently maintains superior accuracy across different levels of token pruning. With 50% token retention, VFlowOpt achieves 99.9% of the original performance, exceeding the second-best approach by 1.0%. This negligible performance drop underscores the method's strong potential for practical deployment. Under more extreme pruning conditions with very few retained tokens, VFlowOpt's advantage becomes more pronounced. When only 10% of the visual tokens are retained, VFlowOpt preserves 85.5% of the original performance, surpassing the second-best approach by 3.3%. This finding suggests that VFlowOpt can effectively leverage limited visual information to maintain high performance under strict computational budgets. In contrast, VFlowOpt demonstrates a clear advantage by tailoring its pruning strategy to each model's unique characteristics.

To further validate VFlowOpt's generalization capability, we evaluate it on LLaVA-NeXT-7B and Qwen2-VL-7B. As shown in Tab. [2](#page-6-0) and Tab. [3,](#page-6-1) VFlowOpt once again delivers the top results on these LMMs. When retaining 25% of the tokens, VFlowOpt preserves 98.8% of the original performance on LLaVA-NeXT-7B and 97.8% on Qwen2- VL-7B. Even under the stringent condition of retaining only 10% of the tokens, VFlowOpt maintains 93.4% of the performance on LLaVA-NeXT-7B and 92.8% on Qwen2-VL-7B, demonstrating its strong generalizability.

<span id="page-6-0"></span>

| Method    | MMStar MME MMB SQA POPE GQA |                           |      |      |      |      | Avg.  |  |
|-----------|-----------------------------|---------------------------|------|------|------|------|-------|--|
|           | Upper Bound, 100% Tokens    |                           |      |      |      |      |       |  |
| Vanilla   | 37.6                        | 1519                      | 67.4 | 70.1 | 86.5 | 64.2 | 100%  |  |
|           | Retain 25% Tokens (↓ 75%)   |                           |      |      |      |      |       |  |
| FastV     | 35.1                        | 1477                      | 65.6 | 67.4 | 83.1 | 60.4 | 95.7% |  |
| VisionZip | 35.8                        | 1501                      | 65.4 | 67.9 | 86.7 | 61.5 | 97.3% |  |
| VFlowOpt  | 37.0                        | 1514                      | 67.0 | 67.7 | 87.6 | 62.6 | 98.8% |  |
|           |                             | Retain 10% Tokens (↓ 90%) |      |      |      |      |       |  |
| FastV     | 29.2                        | 1282                      | 61.6 | 63.8 | 71.7 | 55.9 | 85.7% |  |
| VisionZip | 32.6                        | 1378                      | 61.5 | 67.1 | 83.5 | 57.0 | 91.6% |  |
| VFlowOpt  | 35.1                        | 1393                      | 62.9 | 67.4 | 83.6 | 57.3 | 93.4% |  |

Table 2. Comparative experiments on LLaVA-NeXT-7B.

<span id="page-6-1"></span>

| Method    | MMStar MME MMB SQA POPE GQA |                           |      |      |      |      | Avg. |  |  |
|-----------|-----------------------------|---------------------------|------|------|------|------|------|--|--|
|           | Upper Bound, 100% Tokens    |                           |      |      |      |      |      |  |  |
| Vanilla   | 57.5                        | 1680                      | 80.3 | 84.7 | 88.4 | 62.2 | 100% |  |  |
|           | Retain 25% Tokens (↓ 75%)   |                           |      |      |      |      |      |  |  |
| FastV     | 54.5                        | 1597                      | 76.3 | 79.0 | 81.8 | 57.2 | 93.8 |  |  |
| VisionZip | 55.2                        | 1618                      | 78.9 | 81.3 | 86.9 | 60.0 | 96.8 |  |  |
| VFlowOpt  | 55.9                        | 1659                      | 79.8 | 81.5 | 87.1 | 60.4 | 97.8 |  |  |
|           |                             | Retain 10% Tokens (↓ 90%) |      |      |      |      |      |  |  |
| FastV     | 44.9                        | 1405                      | 70.0 | 75.5 | 75.8 | 51.6 | 84.4 |  |  |
| VisionZip | 49.3                        | 1518                      | 76.5 | 78.2 | 84.3 | 54.1 | 90.9 |  |  |
| VFlowOpt  | 51.0                        | 1591                      | 78.0 | 78.6 | 84.5 | 54.8 | 92.8 |  |  |

Table 3. Comparative experiments on Qwen2-VL-7B.

<span id="page-6-2"></span>

| Method    | SeedBench                 |        | VideoMME |      | Avg.  |
|-----------|---------------------------|--------|----------|------|-------|
|           | (video)                   | S<br>M |          | L    |       |
|           | Upper Bound, 100% Tokens  |        |          |      |       |
| Vanilla   | 56.9                      | 70.5   | 54.6     | 49.5 | 100%  |
|           | Retain 25% Tokens (↓ 75%) |        |          |      |       |
| FastV     | 54.7                      | 66.9   | 53.2     | 47.7 | 96.2% |
| VisionZip | 56.4                      | 68.3   | 55.3     | 49.0 | 99.0% |
| VFlowOpt  | 56.8                      | 68.9   | 55.8     | 49.2 | 100%  |
|           | Retain 10% Tokens (↓ 90%) |        |          |      |       |
| FastV     | 48.7                      | 53.7   | 47.6     | 42.3 | 83.6% |
| VisionZip | 55.0                      | 59.7   | 51.8     | 46.3 | 92.4% |
| VFlowOpt  | 55.5                      | 63.3   | 52.7     | 48.6 | 95.5% |

Table 4. Comparative experiments on video understanding tasks.

## 4.3. Generalization to Video Tasks

We further explore VFlowOpt's generalization across different modalities by evaluating it on video benchmarks using LLaVA-OneVision-7B. As shown in Tab. [4,](#page-6-2) VFlowOpt surpasses other baselines under various token retention ratios. Notably, when only 25% of the tokens are preserved, VFlowOpt incurs virtually no performance loss on LLaVA-OneVision-7B, maintaining 100% of its original performance. Even under the more stringent condition of retaining just 10% of the tokens, VFlowOpt still retains 95.5% of the original performance, underscoring its robust generalization in the video domain.

#### 4.4. Efficiency Analysis

We demonstrate the efficiency of VFlowOpt by conducting a comparative study on LLaVA-OneVision-7B running on a single NVIDIA A100-SXM4-80GB GPU, focusing on FLOPs, KV-Cache memory usage, and inference latency. The efficiency analysis of other baseline methods is provided in the Appendix.

We assess various token pruning ratios on the MME benchmark, measuring overall performance and average efficiency metrics across all samples. As shown in Tab. [5,](#page-7-0) VFlowOpt significantly enhances LLaVA-OneVision-7B's computational efficiency by reducing FLOPs, shrinking KV-Cache memory usage, and accelerating inference speed. Notably, at a 50% token pruning ratio, VFlowOpt achieves a 49.5% reduction in KV-Cache memory and a 1.8× speedup in inference, while boosting performance by 0.7%. At a more aggressive 75% pruning ratio, we observe a 74.2% reduction in KV-Cache memory and a 3.1× speedup, with only a 0.7% performance drop. Under extremely tight computational budgets, pruning 90% of visual tokens reduces the KV-Cache memory footprint by 89%, accelerates inference by 3.8×, and degrades performance by only 7.4%.

Such reductions in inference latency substantially benefit user-facing applications demanding real-time performance, such as autonomous driving and robotics. Moreover, by significantly shrinking the KV-Cache, large-batch inference on LMMs can accommodate more user requests simultaneously, thereby substantially reducing overall inference costs. In summary, VFlowOpt uses far less GPU memory and delivers faster inference while preserving model performance, offering a highly practical solution for efficiently deploying LMMs in real-world scenarios.

#### 4.5. Ablation Study

To confirm the contributions of each component in our token pruning strategy, we conduct ablation experiments on LLaVA-OneVision-7B under a computational budget of retaining 25% of tokens. As shown in Tab. [6,](#page-7-1) removing importance calibration (i.e., directly using the mean attention over all tokens received by each visual token as its importance score), omitting token recycling, or discarding progressive pruning (i.e., maintaining the same number of tokens at each layer) leads to noticeable performance degradation across MMStar, MMBench, and SQA. By contrast, the complete VFlowOpt method consistently exhibits minimal performance loss, highlighting the effectiveness of all its components in pruning redundant tokens while preserving essential visual information.

We further examine how the number of samples and the number of optimization iterations in the Bayesian optimization procedure influence both optimization time and final performance. As illustrated in Fig. [4,](#page-7-2) the optimization time increases linearly with the number of samples and steps.

<span id="page-7-0"></span>

| Methods            | Token<br>Reduction | FLOPs ↓<br>(T)      | ∆                          | Latency ↓<br>(ms)       | ∆                          | KV Cache ↓<br>(MB)      | ∆                          | Performance ↑        | ∆                       |
|--------------------|--------------------|---------------------|----------------------------|-------------------------|----------------------------|-------------------------|----------------------------|----------------------|-------------------------|
| LLaVA-OneVision-7B | -                  | 71.4                | -                          | 1040.1                  | -                          | 1786.4                  | -                          | 1581                 | -                       |
| + VFlowOpt         | 50%<br>75%<br>90%  | 37.2<br>19.1<br>7.7 | -48.0%<br>-73.2%<br>-89.2% | 584.2<br>328.5<br>272.1 | -43.8%<br>-68.4%<br>-73.8% | 902.8<br>460.6<br>197.1 | -49.5%<br>-74.2%<br>-89.0% | 1591<br>1570<br>1464 | +0.6%<br>-0.7%<br>-7.4% |

Table 5. Efficiency analysis of LLaVA-OneVision-7B with VFlowOpt. The detailed metric includes computation (FLOPs), latency, and KV-Cache memory. (∆) denotes the reduction ratio. .

<span id="page-7-1"></span>

| Pruning Strategy                                |              | MMStar MMBench SQA |              |
|-------------------------------------------------|--------------|--------------------|--------------|
| VFlowOpt                                        | 57.8         | 79.9               | 92.3         |
| w/o Importance Calibration<br>w/o Token Merging | 56.2<br>57.6 | 79.4<br>79.8       | 91.8<br>91.9 |
| w/o Progressive Pruning                         | 56.0         | 79.1               | 91.0         |

Table 6. Ablation studies of pruning strategy components.

<span id="page-7-2"></span>![](_page_7_Figure_4.jpeg)

Figure 4. Relationships of sample size (left) and optimization steps (right) with optimization time and final performance.

Using 30 samples and 50 steps strikes an effective balance between time efficiency and performance, requiring only about 30 minutes to reach an exceptional result. More ablation results are provided in the supplementary material.

#### 4.6. Visual Information Flow Analysis

We further investigate how Bayesian optimization formulates the superior pruning strategy by examining its impact on visual information flow. Specifically, we compare the randomly initialized pruning strategy derived from the Bayesian optimization process with its final optimized strategy, measuring their respective effects on the text tokens that follow the visual tokens at each LLM layer. Because these text tokens can receive information from the visual tokens via the LLM's unidirectional attention mechanism, examining differences in these tokens with and without pruning effectively reveals how visual information flow is altered. As shown in Fig. [5,](#page-7-3) under the same computational budget, the initial strategy causes a pronounced discrepancy between text token representations with and without pruning, indicating that it significantly disrupts the visual information flow. In contrast, when using the optimized strategy, the text tokens remain much more similar to those observed

<span id="page-7-3"></span>![](_page_7_Figure_9.jpeg)

Figure 5. Visualization of the discrepancies in visual information flow corresponding to the pruning strategies before and after optimization, along with an example of visual question answering.

without visual token pruning. This important finding suggests that the optimization process yields a pruning strategy that preserves crucial visual information by minimizing the difference in visual information flow with and without pruning, thereby retaining LMM performance to the greatest extent possible.

