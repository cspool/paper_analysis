# **ExpertFlow:** Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling

Xin He<sup>1\*</sup>, Shunkang Zhang<sup>2</sup>, Kaijie Tang<sup>3</sup>, Shaohuai Shi<sup>3</sup>, Yuxin Wang<sup>4</sup>, Zihao Zeng<sup>5</sup>, Zhenheng Tang<sup>2</sup>, Xiaowen Chu<sup>6</sup>, Haiyan Yin<sup>1</sup>, Ivor W. Tsang<sup>1,5</sup>, Yew Soon Ong<sup>1,5\*</sup>

<sup>1</sup>CFAR, Agency for Science, Technology and Research (A\*STAR), Singapore

<sup>2</sup>The Hong Kong University of Science and Technology, Hong Kong

<sup>3</sup>Harbin Institute of Technology, Shenzhen, China

<sup>4</sup>Hong Kong Baptist University, Hong Kong

<sup>5</sup>Nanyang Technological University, Singapore

<sup>6</sup>The Hong Kong University of Science and Technology (Guangzhou), China

## **Abstract**

Sparse Mixture-of-Experts (MoE) models can outperform dense large language models at similar computation by activating only a small set of experts per token. However, stacking many expert modules introduces substantial parameter memory, which makes MoE models difficult to deploy in memory-constrained environments such as single-GPU devices. Offloading alleviates this issue by storing inactive experts in CPU memory and loading them on demand, but existing methods remain limited: static caches disregard input-dependent routing, and methods that train separate models to predict expert usage ahead of time are often inaccurate or require significant training cost. We propose ExpertFlow, a lightweight MoE inference system that addresses this routing dependency through three coordinated components: 1) a transformerbased routing path predictor that estimates expert usage across all MoE layers in a single forward pass, 2) a token scheduler that groups tokens with similar predicted routes to improve expert utilization, and 3) a predictive expert cache that loads only the required experts while correcting mispredictions at runtime. Together, these components enable efficient expert loading and execution, reducing GPU memory usage by up to 93.72% and improving inference throughput by up to 10× over strong offloading baselines on a single GPU.

## **CCS Concepts**

• Computer systems organization → Heterogeneous (hybrid) systems; • Computing methodologies → Natural language processing; Neural networks; Parallel computing methodologies.

#### Keywords

Large Language Model (LLM), Mixture-of-Experts (MoE), Hybrid System

#### **ACM Reference Format:**

Xin He<sup>1\*</sup>, Shunkang Zhang<sup>2</sup>, Kaijie Tang<sup>3</sup>, Shaohuai Shi<sup>3</sup>, Yuxin Wang<sup>4</sup>, Zihao Zeng<sup>5</sup>, Zhenheng Tang<sup>2</sup>, Xiaowen Chu<sup>6</sup>, Haiyan Yin<sup>1</sup>, Ivor W. Tsang<sup>1,5</sup>, Yew Soon Ong<sup>1,5\*</sup>. 2026. *ExpertFlow*: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling. In *63rd ACM/IEEE Design Automation Conference (DAC '26)*, July 26–29, 2026, Long Beach, CA, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3770743.3804292

 $<sup>^*</sup>Corresponding \ authors: \{he\_xin, ong\_yew\_soon\} @a\text{-star.edu.sg}$ 

![](_page_0_Picture_12.jpeg)

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

DAC '26, Long Beach, CA, USA
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2254-7/2026/07
https://doi.org/10.1145/3770743.3804292

