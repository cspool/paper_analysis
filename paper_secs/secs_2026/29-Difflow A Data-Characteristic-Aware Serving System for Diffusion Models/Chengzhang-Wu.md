# Chengzhang Wu\*

Tsinghua University Beijing, China wcz24@mails.tsinghua.edu.cn

### Kezhao Huang

Tsinghua University Beijing, China hkz20@tsinghua.org.cn

## Liyan Zheng\*

Tsinghua University Beijing, China zhengly20@mails.tsinghua.edu.cn

#### Zixuan Ma

Tsinghua University Beijing, China mzx22@mails.tsinghua.edu.cn

## Jidong Zhai

Tsinghua University Beijing, China zhaijidong@tsinghua.edu.cn

### Haojie Wang

Tsinghua University Beijing, China wanghaojie@tsinghua.edu.cn

## Dong Dong

Tsinghua University Beijing, China dongd@tsinghua.edu.cn

#### **Abstract**

Diffusion models have become the dominant approach for generative tasks in images, videos, and other domains. However, diverse data properties in generation requests, which are critical for efficient serving, remain underexploited.

To address this issue, we propose a diffusion model serving system CHITUDIFFUSION. CHITUDIFFUSION leverages the locality of data properties to recompose a diffusion pipeline into subgraphs with shared optimization opportunities, enabling thorough compile-time and runtime co-optimizations. During compilation, CHITUDIFFUSION compiles each subgraph into multiple execution engines optimized for specific data properties. At runtime, heterogeneous requests are elaborately reorganized into fine-grained batching tasks with similar properties and then efficiently executed by matched engines. Evaluation on five diffusion applications shows that CHITUDIFFUSION improves the throughput by up to 2.13×  $(1.58 \times \text{ on average})$  on A100 and  $2.19 \times (1.51 \times \text{ on average})$ on H100 compared with existing frameworks. The code for CHITUDIFFUSION and the production traces have been made open-source at https://github.com/thu-pacman/chitu/tree/ Diffusion.

CCS Concepts: • Computer systems organization  $\rightarrow$  Real-time systems; Parallel architectures; • Computing methodologies  $\rightarrow$  Machine learning; Computer vision.

\*Both authors contributed equally to this research.

![](_page_0_Picture_20.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

PPoPP '26, Sydney, NSW, Australia
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2310-0/2026/01
https://doi.org/10.1145/3774934.3786424

*Keywords:* Deep learning serving system, Diffusion models, Data-characteristic-aware optimization, Compiler

#### **ACM Reference Format:**

Chengzhang Wu, Liyan Zheng, Haojie Wang, Kezhao Huang, Zixuan Ma, Dong Dong, and Jidong Zhai. 2026. ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models. In *Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia.* ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3774934.3786424

#### 1 Introduction

Diffusion models have become a versatile class of generative algorithms across domains, including images [29, 51, 52, 67], videos [11, 16, 28, 30], 3D objects [32, 49], music [27, 43], and proteins [63]. Their flexibility supports applications from text-to-image [51] and text-to-video [11] generation to controllable editing [67], powering products like DALL·E 3 [20], Sora [6], and Firefly [2]. To meet application demands, diffusion pipelines integrate diverse input data and multiple DNNs, leading to high computational cost. Real-world requests further exhibit heterogeneous data properties—such as shared and partially duplicate inputs with varying generation shapes (Figure 1(a))—arising from both applications and user behaviors [62, 64]. Exploiting these properties offers acceleration opportunities beyond the scope of general ML frameworks [7, 48, 58].

Dedicated diffusion model frameworks, e.g., Diffusers [61], offer manually optimized pipelines designed for specific data properties, such as batched requests with identical prompts shown in Figure 1(b). Users can select the corresponding optimized pipeline for efficient executions. However, these manual optimizations address only a narrow subset of potential scenarios due to two challenges.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

**Figure 1.** (a) Image generation requests with different data properties. Yellow and blue circles represent identical computations and computations with the same generation shape, respectively. (b) Existing diffusion frameworks with manual optimizations for uniform properties but lack support for heterogeneous properties.

1) Diverse input data properties. The complexity of diffusion models means each generation request may involve tens of inputs with independent properties. Since these properties can interact, existing methods must enumerate all possible combinations for optimization. However, the combinatorial explosion across inputs makes exhaustive optimization intractable.

2) Heterogeneous requests. When batching for efficiency, requests often contain heterogeneous and dynamic data properties. As shown in Figure 1(b), different requests may share prompts or generation shapes, creating optimization opportunities. However, existing methods assume uniform properties within a batch, making it difficult to exploit such heterogeneity.

We present Chitudiffusion, a diffusion service system that leverages the data properties exhibiting *locality* across computations. Consecutive operators often share optimization opportunities induced by these properties—e.g., circles with the same color in Figure 1(a). Chitudiffusion decomposes the pipeline into *dGraphs*<sup>1</sup>, which explicitly capture data properties, enabling decoupled handling through codesigned compilation and runtime optimizations. As a result, Chitudiffusion enables a wide range of data-property-aware optimizations beyond existing frameworks, while maintaining a reasonable optimization cost.

In ChituDiffusion, we make the following design decisions.

During static compilation, Chitudiffusion decomposes the pipeline into dGraphs using symbolic variables to represent and propagate input data properties. Each dGraph is compiled into multiple dEngines, each specialized for a uniform property configuration. By reusing dEngines across requests with different properties, Chitudiffusion performs optimizations for diverse input properties while incurring only a small compilation overhead.

At runtime, heterogeneous requests are decomposed into fine-grained dTask (dGraph-level tasks). A dynamic programming based scheduler identifies and batches dTasks with the same properties. These dTasks batches are efficiently processed by corresponding dEngines that have been optimized for uniform data properties. Input and output properties are inferred asynchronously, enabling the overlap of scheduling and execution processes to mitigate scheduling overhead.

Specifically, ChituDiffusion implements two key data-property optimizations to accelerate diffusion pipelines (§2.2): redundancy and dynamic shapes. Redundancy propagation and elimination rules are designed to detect and remove dimension-level redundant computations and memory accesses in tensor algebra. To support batched requests with dynamic shapes (hereafter referred to as *raggedness*), ChituDiffusion applies *raggedness regularization*, transforming dynamic-shape operations into kernel-compatible forms without requiring new ragged kernels.

We evaluate ChituDiffusion on five diverse diffusion applications, covering image and video generation with several widely used model extensions. The evaluation shows that ChituDiffusion achieves a throughput up to 2.13× (1.58× on average) higher than widely-used DNN optimizers and diffusion-specific systems, showing the effectiveness of ChituDiffusion's data-property-aware optimizations on diffusion models.

This paper makes the following contributions:

- We observe the key data characteristics of diffusion model requests for optimizations, which exhibit locality in pipelines.
- We propose recomposed optimizations for pipelines and requests, orchestrating compile-time and runtime optimizations to exploit complex data properties.
- We implement CHITUDIFFUSION, a system for diffusion pipeline optimization, which outperforms the throughput of state-of-the-art frameworks by up to 2.13x.

