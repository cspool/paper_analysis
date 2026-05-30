### MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

Songkai Ma Department of Computing Hong Kong Polytechnic University Hong Kong songkai.ma@connect.polyu.hk

Benben Liu LSCM R&D Center The University of Hong Kong Hong Kong benbenliu@hku.hk

Zhaorui Zhang<sup>∗</sup> Department of Computing Hong Kong Polytechnic University Hong Kong zhaorui.zhang@polyu.edu.hk

Xiaodong Yu Department of Computer Science Stevens Institute of Technology USA xyu38@stevens.edu

Dan Wang Department of Computing Hong Kong Polytechnic University Hong Kong dan.wang@polyu.edu.hk

Sheng Di Mathematics and Computer Science Division Argonne National Laboratory, USA sdi1@anl.gov

Xiaoyi Lu Department of Computer Science and Engineering University of California, Merced, USA xiaoyi.lu@ucmerced.edu

#### Abstract

With the widespread application of Mixture of Experts (MoE) reasoning models in the field of LLM learning, efficiently serving MoE models under limited GPU memory constraints has emerged as a significant challenge. Offloading the non-activated experts to main memory has been identified as an efficient approach to address such a problem, while it brings the challenges of transferring the expert between the GPU memory and main memory. We need to explore an efficient approach to compress the expert and analyze how the compression error affects the inference performance.

To bridge this gap, we propose employing error-bounded lossy compression algorithms (such as SZ3 and CuSZp) to compress non-activated experts, thereby reducing data transfer overhead during MoE inference. We conduct extensive experiments across various benchmarks and present a comprehensive analysis of how compression-induced errors in different experts affect overall inference accuracy. The results indicate that experts in the shallow layers, which are primarily responsible for the attention mechanism and the transformation of input tokens into vector representations, exhibit minimal degradation in inference accuracy when subjected to bounded errors. In contrast, errors in the middle-layer experts, which are central to model reasoning, significantly impair inference accuracy. Interestingly, introducing bounded errors in the deep-layer experts, which are mainly responsible for instruction following and output integration, can sometimes lead to improvements in inference accuracy.

Request permissions from permissions@acm.org. SC' 25,

#### Keywords

Model Compression, Mixture of Experts, Inference, Error Sensitivity

#### ACM Reference Format:

Songkai Ma, Zhaorui Zhang, Sheng Di, Benben Liu, Xiaodong Yu, Xiaoyi Lu, and Dan Wang. 2025. MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?. In Proceedings of SC' 25. ACM, St. Louis, MO, USA, [10](#page-9-0) pages.<https://doi.org/X>

#### 1 Introduction

In recent years, Mixture of Experts (MoE) foundation models have enabled large language models (LLMs) to transition from dense architectures to sparsely activated expert frameworks, as exemplified by models such as DeepSeek [\[8\]](#page-9-1), GPT-4 [\[2\]](#page-9-2), Phi-4 [\[1\]](#page-9-3), and Mixtral [\[19\]](#page-9-4). These sparse foundation models selectively activate only a subset of experts for each token, thereby substantially reducing computational overhead and inference costs while maintaining strong generative performance. However, the presence of numerous idle, non-activated experts during MoE inference poses a significant challenge to efficient GPU memory utilization, complicating the deployment of MoE models on GPUs with limited memory resources. For instance, serving the Mixtral-8x7B model requires approximately 94 GB of VRAM when using FP16 precision. In this scenario, only about 30% of the parameters—approximately 27.5 GB—are actively used during the decoding process, while the remaining 66.6 GB of memory is occupied by non-activated experts, resulting in considerable inefficiency [\[29\]](#page-9-5). Consequently, it is crucial to address the challenge of improving memory utilization efficiency, as this has a substantial impact on system performance in serving tasks for MoE foundation models.

Offloading techniques [\[7,](#page-9-6) [18,](#page-9-7) [28,](#page-9-8) [30,](#page-9-9) [31,](#page-9-10) [33,](#page-9-11) [37\]](#page-9-12), which transfer expert parameters from main memory to GPU memory on demand for each input, have been recognized as effective solutions to address GPU memory limitations. However, while offloading alleviates memory constraints, it shifts the bottleneck of MoE inference from

<sup>∗</sup>Zhaorui Zhang is the corresponding author. This paper has been accepted by the workshop of SC'25.

<sup>©</sup> 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 <https://doi.org/X>

being memory-bound to I/O-bound. This is primarily due to the need to transfer large volumes of parameters over the relatively lowbandwidth PCIe bus, resulting in substantial data transfer delays. For example, the DRAM-to-VRAM bandwidth provided by PCIe 4.0 (32 GB/s) is orders of magnitude lower than the bandwidth between GPU memory and on-chip computation units (300 GB/s). As a result, existing MoE inference systems employing expert offloading strategies for GPU memory-constrained scenarios [\[7,](#page-9-6) [18,](#page-9-7) [20,](#page-9-13) [31\]](#page-9-10) continue to experience sub-optimal performance, with loading delays that are perceptible to users and cannot be effectively masked by concurrent computation tasks. Although low-bit quantization strategies can reduce the size of transmitted parameters and mitigate the latency associated with loading activated experts [\[7,](#page-9-6) [28\]](#page-9-8), they often lead to significant degradation in generative performance. Therefore, it is essential to investigate efficient compression algorithms that can achieve high compression ratios while maintaining minimal compression error, thereby preserving the generative performance of MoE models. To achieve this goal, there are three critical steps. Our work mainly focuses on the first and second steps.

- Firstly, we need to investigate an efficient compression algorithm for expert parameters compression during inference that can achieve a high compression ratio while maintaining minimal compression error.
- Secondly, a comprehensive analysis of compression error sensitivity across different experts is necessary to understand its impact on the generative performance of MoE models. It is important to evaluate how compression errors associated with individual experts influence the overall generation quality. Such analysis will provide valuable insights for designing efficient compression algorithms that minimize offloading overhead while preserving model performance.
- Thirdly, further enhancements to system performance are required when integrating compression algorithms into the MoE inference framework. In particular, it is important to investigate the design of pipeline algorithms that can overlap compression and decompression operations with offloading tasks, thereby minimizing the associated latency and improving overall inference efficiency.

Currently, four primary expert compression strategies are employed in the context of MoE models: expert distillation [\[9,](#page-9-14) [24\]](#page-9-15), expert pruning [\[24,](#page-9-15) [27\]](#page-9-16), expert decomposition [\[12\]](#page-9-17), and expert quantization [\[6,](#page-9-18) [11,](#page-9-19) [34\]](#page-9-20). Expert distillation involves transferring or compressing the knowledge from a large MoE model with multiple experts into a smaller, more deployable model—typically a single model or a reduced set of experts—thereby preserving performance while reducing computational resource requirements. Expert pruning seeks to optimize resource utilization and reduce redundancy by identifying and removing experts that contribute minimally or perform suboptimally during training. The central principle of expert pruning is to retain experts that significantly enhance model performance and eliminate those whose contributions are limited or whose computational overhead is disproportionately high. Expert decomposition commonly utilizes low-rank decomposition techniques to reduce parameter count by factorizing the weight matrices of MoE models into products of lower-rank matrices. Expert quantization aims to reduce computational and storage costs by decreasing the bit-width of model parameters. By converting

floating-point parameters to low-precision integers, quantization can substantially reduce model size and accelerate inference, while striving to maintain model performance. In practice, expert quantization is often combined with other optimization techniques—such as distillation, pruning, and decomposition—to further enhance the efficiency and resource utilization of MoE models during deployment. However, low-bit quantization frequently leads to significant degradation in generative performance for MoE inference tasks, due to the uncontrollable and unpredictable errors introduced during the quantization of expert parameters.

Error-bounded lossy compression approaches [\[5,](#page-9-21) [13–](#page-9-22)[17,](#page-9-23) [21,](#page-9-24) [25,](#page-9-25) [26,](#page-9-26) [35,](#page-9-27) [38](#page-9-28)[–43\]](#page-9-29) offer a promising solution for compressing expert parameters with high compression ratios while maintaining minimal compression error, thereby preserving the generative performance of MoE models. These techniques have proven effective in substantially reducing data storage and transfer burdens, all while maintaining high fidelity in the reconstructed data. Numerous errorbounded lossy compressors have been developed to support a wide range of parallel and distributed computing scenarios, each employing distinct compression models and principles that confer specific advantages and limitations. The primary motivation for adopting error-bounded lossy compression for expert parameter reduction is its ability to guarantee a bounded error range, in contrast to traditional quantization methods, which often introduce uncontrollable and unpredictable errors. Furthermore, various error-bounded lossy compression algorithms have been optimized for different hardware platforms and accelerators, including CPU-based algorithms such as SZ3 [\[21\]](#page-9-24) and GPU-based algorithms such as CuSZp [\[16,](#page-9-30) [17\]](#page-9-23).

To the best of our knowledge, this work represents the first attempt to leverage error-bounded lossy compression techniques for compressing expert parameters, with the aim of reducing PCIe offloading overhead and enhancing GPU memory efficiency in MoE inference tasks. As outlined above, achieving these objectives involves three key steps. This study primarily focuses on the first two: ① investigating efficient compression algorithms that can achieve high compression ratios while maintaining minimal compression error, thereby preserving the generative performance of MoE models during inference; and ② conducting a comprehensive sensitivity analysis of compression errors across different experts to determine their impact on the overall generative performance of MoE models in inference scenarios. We introduce varying error bounds to add the errors in the experts during inference and evaluate the resulting generative performance using several of the most widely adopted MoE-based foundation models. The main contributions and key findings of this work are summarized as follows:

- ❶ To the best of our knowledge, this work is the first to propose the use of error-bounded lossy compression algorithms, such as SZ3 and CuSZp, for compressing non-activated experts in order to reduce the data transfer overhead between GPU memory and main memory during inference. In comparison to quantization-based approaches, error-bounded lossy compression algorithms offer higher compression ratios while maintaining minimal compression error, thereby better preserving model performance.
- ❷ We conduct comprehensive experiments using popular MoE models and benchmark datasets to provide an in-depth analysis of the impact of compression error on inference accuracy.

② The experimental results indicate that experts in the shallow layers primarily handle attention mechanisms and the transformation of input tokens into vector representations; introducing errors at this stage has only a minimal impact on inference accuracy. In contrast, experts in the middle layers are chiefly responsible for core model reasoning, and the presence of errors in these layers significantly degrades inference accuracy. Interestingly, experts in the deep layers are mainly involved in instruction following and output integration, where the introduction of bounded errors can, in some cases, lead to improvements in inference accuracy.

#### 2 Background and Motivations

#### 2.1 MoE Inference Process

Similar to LLMs, a typical layer in MoE models comprises a selfattention layer followed by a sparse MoE layer. The input tokens to each layer are initially processed by the self-attention mechanism, which can generally be divided into three stages: ① pre-attention stage, including QKV (query, key, value) projection; ② self-attention computation stage, involving the calculation of  $QK^T$ ; and ③ postattention stage, which includes output projection. Following the self-attention layer, tokens are passed to the sparse MoE layer, where a router assigns each token to a subset of experts, typically employing a top-k selection strategy [36]. Each token is processed by k selected experts, and the final output is obtained by computing a weighted average of the outputs from these experts. Certain model architectures, such as DeepSeek-V2 [22] and Qwen2MoE [32], incorporate a shared expert through which all tokens are processed. The resulting token representations are then forwarded to subsequent layers in the model. While layer normalization and residual connections are commonly present in MoE models, they are not central to the discussion here and are therefore omitted.

MoE batched inference closely mirrors the generative inference procedure of LLMs, operating in two distinct phases: ① Prefill: during which a batch of prompts is processed to generate the key-value (KV) cache at each attention layer; and ② Decoding: where new tokens are generated in an auto-regressive manner. In the decoding phase, the output tokens from the previous forward pass serve as the input for generating the next token. With each forward pass, the KV-cache corresponding to the new input token is generated and appended to the existing KV-cache, thereby constructing the complete context up to that point. Notably, the computational intensity during the decoding phase is typically orders of magnitude lower than in the prefill phase, as only a single token per sequence is processed by the model at each step.

#### 2.2 Expert Offloading in MoE Inference Process

The expert offloading strategy involves the management of two levels of memory: main memory, which stores excess model parameters (weights) and key-value (KV) states, and GPU memory, which is utilized for computation and rapid data access. When model parameters are needed for GPU computation, they can either be prefetched in advance—overlapping with other computations—or fetched on demand. To facilitate efficient data access, a resident store can be implemented in GPU memory to persistently hold model parameters and the KV-cache, while a staging buffer is employed to prefetch dynamic data. If the GPU attempts to access

data, including model parameters or KV states, that are not yet present in its resident store, computation must stall until the required data are transferred from main memory. In this context, the bandwidth of the PCIe interface between the host and GPU memory often constitutes a critical bottleneck. Recent research suggests that leveraging CPU computational resources to process a portion of the data locally represents a promising direction, with the potential to increase overall system throughput [3].

#### 2.3 Motivations from Quantization Approaches

We summarize various quantization methods in Tab. 1. Our findings indicate that quantization primarily reduces memory usage, with most methods achieving approximately a 4× decrease. While some quantization techniques also accelerate inference, others do not; for instance, QMOE incurs an additional 5% computational overhead. Moreover, our analysis of quantization as a form of lossy compression reveals that lower bit widths generally result in greater performance degradation. Consequently, it is essential to strike a balance among memory efficiency, inference accuracy, and computational speed when applying quantization. However, most existing quantization approaches are unable to simultaneously satisfy these requirements. In this work, we explore a novel direction by incorporating error-bounded lossy compression into the inference process to compress expert parameters, aiming to achieve an optimal balance among memory usage, accuracy, and acceleration.

<span id="page-2-0"></span>**Table 1: The Comparison for Quantization Approaches** 

| Method    | Mem_Save | Acc_Drop | Speedup        | Bits    |
|-----------|----------|----------|----------------|---------|
| MC-MoE    | 4.27×    | 3.8%     | 1.80×          | 1, 2, 3 |
| MoE-CSP   | 4.00×    | -        | $26.00 \times$ | 4, 8    |
| MoQE      | 4.90×    | 0.97%    | $-5 \times$    | 2, 3, 4 |
| QMoE      | 20×      | 6.7%     | 0.95x          | 1, 2    |
| CMoE      | 150×     | 23.81%   | -              | 1, 2, 4 |
| MoE-MPTQS | -        | 4.98%    | ↑ 20.63×       | 4, 8    |
| HOBBIT    | -        | 1%       | 1.35×          | 2, 4    |
| EdgeMoE   | ↑1.18×   | 5%       | ↑ 2.78×        | 2, 4, 8 |

### 3 Methodology for Error Sensitivity Analyze

#### 3.1 Benchmark and Experimental Design

**MoE Model and Datasets.** We first deployed the Moonlight [23] model, whose MoE architecture comprises 26 expert layers, each containing 64 expert submodules. During inference, each layer activates 6 experts, selected via a top–k routing mechanism, with each expert assigned distinct parameter values. For the reasoning tasks, we utilized the GSM8K dataset [4] as input.

**Errors.** To simulate the compression errors of most current state-of-the-art compressors, such as SZ3 [21], CuSZp [16, 17], etc., we randomly generated n errors which follows the normal distributions  $N \sim (0, \hat{e})$  and add these errors to the expert parameters during MoE inference, where n is the number of parameters that we try to analyze,  $\hat{e}$  indicate the error bound of the compression algorithms.

**Activated Frequency for the Experts.** After the model processes each input problem, we record the six experts selected in each layer along with their corresponding weights. This approach

| Table 2: Custom Variables Definition in Our Work |  |  |  |  |
|--------------------------------------------------|--|--|--|--|
|--------------------------------------------------|--|--|--|--|

<span id="page-3-3"></span>

| Custom variables     | Interpretation                                                           |  |  |
|----------------------|--------------------------------------------------------------------------|--|--|
| Imbalance Score      | Evaluation of whether the number of times experts are called is balanced |  |  |
| Expert Utilization   | The proportion of experts used                                           |  |  |
| Entropy (Normalized) | Metrics to measure the uncertainty of model decisions                    |  |  |
| Gini Coefficient     | Measuring the fairness or inequality of a distribution                   |  |  |

<span id="page-3-0"></span>![](_page_3_Figure_4.jpeg)

Figure 1: The activation frequency for the first layer.

enables detailed tracking of expert utilization across layers and facilitates analysis of how the model allocates tasks under different reasoning scenarios. We aggregate the selection frequency of each expert across all questions and analyze the usage patterns for each layer, as illustrated in Fig. [1](#page-3-0) and Fig. [2.](#page-3-1) This analysis provides insights into which experts play a more significant role during the reasoning process and serves as foundational data for subsequent error sensitivity experiments. Specifically, we present the usage frequencies of the experts in the first and 26th layers after processing all questions in Fig. [1](#page-3-0) and Fig. [2,](#page-3-1) respectively.

<span id="page-3-1"></span>![](_page_3_Figure_7.jpeg)

Figure 2: The activation frequency for layer-26.

Activation Frequency Heatmap for the Experts. To provide a more intuitive visualization, we present in Fig. [3](#page-3-2) a heat map depicting the activation frequency of all experts across all layers following the input of the questions. The heat map reveals a pronounced imbalance in expert utilization: approximately 10 experts are activated more than 35,000 times throughout the entire reasoning process, whereas the majority of other experts are selected far less frequently, with some being used fewer than 10,000 times. This observation indicates that, despite the model comprising a large number of experts, only a small subset plays a central role in most reasoning tasks (e.g., expert 12 in layer 1). These results suggest that, for mathematical reasoning problems, a limited number of experts

are responsible for the majority of computational tasks, leading to a marked "concentration" phenomenon in expert utilization.

<span id="page-3-2"></span>![](_page_3_Figure_11.jpeg)

Figure 3: Heat map of activation frequency of experts.

This uneven distribution of expert activation provides important insights for error sensitivity analysis for the experts. We can reasonably hypothesize that introducing errors into the frequently activated experts will have a more pronounced impact on the model's reasoning outcomes, as these experts predominantly drive the reasoning process. Experts that are activated less frequently, while still contributing to the overall model architecture, exert a relatively minor influence on overall performance when subjected to errors due to their limited participation. However, it is important to note that certain low-frequency experts (e.g., expert-0 in layer 1) also play critical roles in handling specific mathematical reasoning steps within particular tasks. Randomly setting the parameter values of expert-0 can lead to failures in managing these specialized scenarios, thereby disrupting the entire reasoning process. Although expert-0 is utilized far less than other experts, its involvement in key reasoning tasks renders it indispensable; errors in such experts can result in incorrect model outputs for those specific cases. Therefore, balancing the roles and resource allocation among experts—avoiding over-reliance on a small subset of high-frequency experts while ensuring that the potential of low-frequency experts is fully leveraged—emerges as a crucial consideration in our subsequent model optimization efforts. Finally, we define and track several customized variables to quantify and illustrate changes in expert activation across different layers of the original model on the GSM8K dataset, with detailed explanations provided in Tab. [2.](#page-3-3)

As illustrated in Fig. [4,](#page-4-0) expert utilization across the 26 layers is highly imbalanced, with certain layers containing experts that remain inactive (as indicated by blue bar graph values less than 1.0). This observation is further corroborated by the normalized entropy and Gini coefficient metrics, both of which quantitatively confirm the uneven distribution of expert activation.

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 4: Comparison of aggregated expert layers

## 3.2 Error Sensitivity Analysis for a Single Expert in Different Layers

To investigate the impact of parameter errors on model inference accuracy, we conducted a series of experiments to evaluate the error sensitivity of different experts and provide a comprehensive analysis of the experimental results.

3.2.1 Error Sensitivity Analysis for the Expert in the First Layer. The distribution of the compression error for the currently most popular error-bounded lossy compression approaches [16, 17, 21] follows the Normal distribution  $N \sim (0, \hat{e})$ , where the  $\hat{e}$  is the error bound of the compression approaches. Therefore, we randomly simulate n errors that follow a Normal distribution with different error bounds  $\hat{e}$  and add them to the experts in our experiments.

Firstly, we choose the first expert in the first layer (expert-0) and

add errors generated by different error bounds to it, including the error bound  $\hat{e}=(10\%*\frac{||\theta_{\ell_1,expert_0}||_1}{n_{\ell_1,expert_0}}),\,\hat{e}=(30\%*\frac{||\theta_{\ell_1,expert_0}||_1}{n_{\ell_1,expert_0}}),\,\hat{e}=(50\%*\frac{||\theta_{\ell_1,expert_0}||_1}{n_{\ell_1,expert_0}}),\,\hat{e}=(80\%*\frac{||\theta_{\ell_1,expert_0}||_1}{n_{\ell_1,expert_0}}),\,$  where the  $\frac{||\theta_{\ell_1,expert_0}||_1}{n_{\ell_1,expert_0}}$  indicates the average value of the L1 norm of the expert-0 in the first layer,  $n_{\ell_1,expert_0}$  is the number of parameters of the expert-0 in the first layer. The results indicate that introducing errors to expert 0 in the first layer does not affect the model's inference accuracy or the sequence of inference steps. This suggests that, for certain experts with minimal influence on overall model performance, adding small errors in their weights does not substantially impact the final output or the reasoning process. These findings demonstrate the model's robustness to minor weight fluctuations, particularly when such changes do not involve critical experts. Key observations and analyses from these experimental results are summarized as follows:

• This phenomenon highlights a characteristic of the model: certain experts exert minimal influence on task outputs and may even be redundant. Consequently, modifying the weights of these experts does not significantly alter inference results. The model thus demonstrates high stability and resilience to small-scale errors or noise affecting these less critical experts.

- However, when we randomized the weights of the first expert in the first layer, the results differed markedly. Although the model was still able to complete the inference task and generate an output, the inference result was entirely incorrect. This finding indicates that, even when the parameters of a single expert are severely corrupted, the model's structure remains operational; however, the accuracy of its reasoning is significantly compromised. These results underscore that the contribution of seemingly unimportant experts in the reasoning process should not be underestimated. In certain tasks, where such experts may play a more substantial role than initially anticipated, parameter errors can directly lead to a dramatic decline in overall model performance.
- Although the model exhibits a degree of fault tolerance to parameter errors in individual, less influential experts, the role of each expert remains critical. Errors in expert parameters can substantially impact reasoning outcomes and diminish overall accuracy. Therefore, when designing MoE models, it is essential to ensure the correctness and stability of each expert to preserve the integrity and effectiveness of the overall reasoning process.
- 3.2.2 Error Sensitivity Analysis for Highest-Frequently-Activated Expert. We selected the most frequently activated expert in the first layer—specifically, expert 12, whose activation rate was substantially higher than that of other experts within the same layer. Same as the above section, we simulate the error for the expert with different error bounds  $\hat{e}$  and add them to the expert, including the error bound  $(\hat{e})$   $\hat{e} = (30\% * \frac{||\theta_{\ell_1,expert_{26}}||_1}{n_{\ell_1,expert_{26}}})$ ,  $\hat{e} = (50\% * \frac{||\theta_{\ell_1,expert_{26}}||_1}{n_{\ell_1,expert_{26}}})$ , where the  $\frac{||\theta_{\ell_1,expert_{26}}||_1}{n_{\ell_1,expert_{26}}}$  indicates the average value of the L1 norm of the highest-frequently-activated expert (the expert-26 in our benchmark) in the first layer,  $n_{\ell_1,expert_{26}}$  is the number of parameters of the highest-frequently-activated expert (the expert-26 in our benchmark) in the first layer.

It is important to note that instruction compliance directly influences the overall effectiveness of the system. Even if the model's internal reasoning is correct, violations of output format or content constraints—referred to as non-instructional errors—can still lead to a reduction in system accuracy. Therefore, it is necessary to distinguish between two evaluation metrics as follows:

- Instruction Compliance Accuracy (ICA): the output results meet both content correctness and format specifications.
   We abbreviate it as ICA in our work.
- Pure Inference Accuracy (PIA): only evaluate content correctness (ignoring format requirements). We abbreviate it as PIA in our work.

Experimental results indicate that as the error amplitude increases, the accuracy of reasoning with instruction compliance (i.e., the accuracy of the system output) gradually declines, although it remains relatively high overall. Notably, the model's pure reasoning accuracy remains largely unaffected when errors are introduced exclusively to the most frequently activated expert in layer 1 (expert 26). The detailed results are presented in Tab. 3. A summary of the findings and corresponding analysis is provided below:

(1) Adaptive protection of routing mechanism: When the parameters of high-frequently-activated experts are distorted, the model dynamically adjusts its routing weights to reallocate tasks to other experts with intact functionality. This

<span id="page-5-0"></span>Table 3: The inference comparison for involving different errors in highest-frequently-activated expert (expert-26) in layer-1. ICA: Instruction Compliance Accuracy; PIA: Pure Inference Accuracy

| Error Bound                                                                                                                                                                             | ICA  | PIA  |  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|--|
| Baseline                                                                                                                                                                                | 0.86 | 0.96 |  |
| $\hat{e} = (30\% * \frac{  \theta_{\ell_1, expert_{26}}  _1}{n_{\ell_1, expert_{26}}  _1})$ $\hat{e} = (50\% * \frac{  \theta_{\ell_1, expert_{26}}  _1}{n_{\ell_1, expert_{26}}  _1})$ | 0.82 | 0.96 |  |
| $\hat{e} = (50\% * \frac{  \theta_{\ell_1,expert_{26}}  _1}{n_{\ell_1,expert_{26}}})$                                                                                                   | 0.80 | 0.96 |  |
| $\hat{e} = (80\% * \frac{n_{\ell_1, expert_{26}}}{n_{\ell_1, expert_{26}}  _1})$ $\hat{e} = (80\% * \frac{  \theta_{\ell_1, expert_{26}}  _1}{n_{\ell_1, expert_{26}}})$                | 0.79 | 0.95 |  |

adaptive mechanism enables the model to preserve its core reasoning capabilities when involving errors.

- (2) Decoupling characteristics of instruction compliance and reasoning capabilities: Parameter errors primarily disrupt output conventions—such as the box{} format required by instructions—rather than the underlying semantic generation capabilities, confirming the heterogeneity between the parameter space and the functional space in the MoE model.
- 3.2.3 Error Sensitivity Analysis for the Highest Frequently Activated Expert in Different Layers. To assess the generalizability of the aforementioned conclusions—specifically, whether the model's robustness to single expert parameter noise (error) is consistent across different layers—we conducted a series of controlled experiments targeting key layers throughout the model's depth. In the 13th, 20th, and final (26th, output decision) layers, we selected the experts with the highest activation rates within each layer as the subjects for parameter perturbation. Same as the above sections, we set error bound as  $\hat{e} = (80\% * \frac{||\theta_{\ell_x,expert_y}||_{1}}{n_{\ell_x,expert_y}})$ , where  $\ell_x$  indicate the x-th layer and  $expert_y$  indicate the expert—y. We examined the trends in both pure reasoning accuracy and instruction compliance accuracy of the model under cross-layer noise (error), and compared these results with those obtained from the first layer (Layer 1). The detailed experimental outcomes are presented in Tab. 4.

<span id="page-5-1"></span>Table 4: The inference accuracy comparison for involving error in the highest-frequently-activated expert in different layers. ICA: Instruction Compliance Accuracy; PIA: Pure Inference Accuracy

| Expert              | ICA  | PIA  |
|---------------------|------|------|
| Baseline            | 0.86 | 0.96 |
| Layer1 (Expert-26)  | 0.79 | 0.95 |
| Layer13 (Expert-25) | 0.75 | 0.94 |
| Layer20 (Expert-2)  | 0.89 | 0.96 |
| Layer26 (Expert-40) | 0.96 | 0.96 |

The experimental results demonstrate that the model's core reasoning ability for mathematical problems remains highly stable (with accuracy consistently  $\geq 94\%$ , even when errors are introduced to the most frequently activated experts. This suggests that the model's semantic generation mechanism possesses strong resilience to the noise. In contrast, the ability to follow instructions exhibits a pronounced non-hierarchical progression: introducing errors into shallow layers (e.g., layer 1 and layer 3) significantly

impairs this ability (by 10% - 20%), whereas introducing errors into deeper layers (e.g., layer 20 and layer 26) leads to performance gains (of 7% - 10%), likely due to implicit model integration effects. These findings reveal a functional decoupling between different expert layers (or experts) within the MoE architecture with respect to mathematical (semantic generation) logic and instruction parsing. Specifically, semantic reasoning is distributed and encoded throughout the entire network, while instruction compliance depends on specific layers. Moreover, adding noise to parameters in deeper layers can enhance task execution. This insight offers a novel perspective for the hierarchical design of robust MoE models.

