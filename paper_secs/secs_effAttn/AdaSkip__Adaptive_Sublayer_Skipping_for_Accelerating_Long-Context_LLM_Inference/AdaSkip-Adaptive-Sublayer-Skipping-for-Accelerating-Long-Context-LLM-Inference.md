# AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

Zhuomin He<sup>1\*†</sup>, Yizhen Yao<sup>1\*†</sup>, Pengfei Zuo<sup>2\*</sup>, Bin Gao<sup>3†</sup>, Qinya Li<sup>1‡</sup>, Zhenzhe Zheng<sup>1</sup>, Fan Wu<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>Huawei Cloud

<sup>3</sup>National University of Singapore
{dean\_hzm, 1975275148}@sjtu.edu.cn, pengfei.zuo@huawei.com, bingao@comp.nus.edu.sg
{qinyali, zhengzhenzhe}@sjtu.edu.cn, fwu@cs.sjtu.edu.cn

#### **Abstract**

Long-context large language models (LLMs) inference is increasingly critical, motivating a number of studies devoted to alleviating the substantial storage and computational costs in such scenarios. Layer-wise skipping methods are promising optimizations but rarely explored in long-context inference. We observe that existing layer-wise skipping strategies have several limitations when applied in long-context inference, including the inability to adapt to model and context variability, disregard for sublayer significance, and inapplicability for the prefilling phase. This paper proposes AdaSkip, an adaptive sublayer skipping method specifically designed for longcontext inference. AdaSkip adaptively identifies less important layers by leveraging on-the-fly similarity information, enables sublayer-wise skipping, and accelerates both the prefilling and decoding phases. The effectiveness of AdaSkip is demonstrated through extensive experiments on various long-context benchmarks and models, showcasing its superior inference performance over existing baselines.

### Introduction

Recently, large language models (LLMs) evolve to support long-context inference (Xiao et al. 2024; Srivatsa et al. 2024; DeepSeek-AI et al. 2024) up to 1M (Liu et al. 2024; AI et al. 2024), unlocking more complex real-world applications such as personal agent (Park et al. 2023; Wang et al. 2024), document summarization (Wu et al. 2023), and coding assistance (Liu, Xu, and McAuley 2023; Bairi et al. 2023; Jimenez et al. 2024). Long-context inference introduces more computational and storage demands. It is crucial to reduce the inference cost for long sequences.

Layer-wise skipping strategies, as an emerging technology, show great promise to reduce the LLM inference cost and latency by omitting the execution of transformer layers at specific positions, e.g., early skipping (Del Corro et al. 2023; Zhu et al. 2024), periodic skipping (Liu, Meng, and Zhou 2024), and early exit (Varshney et al. 2023; Fan et al. 2024; Chen et al. 2024).

However, we observe that these layer-wise skipping strategies all have their limitations in taking effect in long-context inference due to the following reasons. First, existing layerwise skipping strategies lead to a significant degradation in the generation quality due to predetermined fixed layers being skipped regardless of model and context variance. We observe that the importance distributions of transformer layers are different across models and contexts, and none of these strategies can perform consistently best across all models and contexts. Second, existing skipping strategies perform skipping at monolithic transformer layers which leads to suboptimal performance. We observe that the importance distributions of sublayers, i.e., attention and FFN modules, are independent. Moreover, in long-context inference, attention sublayers contribute significantly to inference latency (Tang et al. 2024; Jiang et al. 2024), highlighting the importance of prioritizing the skipping of more attention sublayers. Third, existing layer-wise skipping strategies are limited to the decoding phase, neglecting optimization of the prefilling phase in long-context inference, where the latency of the prefilling phase, i.e., time to first token (TTFT), imposes a significant burden on long-context inference latency.

To address the above limitations, we propose *AdaSkip*, an auto-adaptive, sublayer-wise skipping strategy tailored for long-context inference, which can benefit both the prefilling and decoding phases. Firstly, AdaSkip exploits on-the-fly similarity information during execution to adaptively identify the least important layers in different models, thereby improving the generation quality. Secondly, AdaSkip independently determines the importance distribution residing within sublayer modules like attention and FFN, enabling the sublayer-wise skipping. Finally, AdaSkip identifies the least important sublayers during both prefilling and decoding phases, significantly reducing the time and memory overhead of long-context scenarios. The code is released on Github<sup>1</sup>.

In summary, our contributions are as follows:

 We perform a comprehensive analysis of the importance distributions of various components including layer and sublayer modules across a range of different models. Based on the analysis, we present the limitations of the ex-

<sup>\*</sup>These authors contributed equally.

<sup>&</sup>lt;sup>†</sup>Work done during their internship at Huawei Cloud.

<sup>\*</sup>Corresponding author.

<sup>&</sup>lt;sup>1</sup>https://github.com/ASISys/AdaSkip

isting layer-wise skipping strategies in accelerating longcontext inference.

- 2. We propose an auto-adaptive, sublayer-wise skipping strategy that works for both the prefilling and decoding phases in long-context scenarios.
- 3. We conduct extensive experiments on various longcontext benchmarks and models, demonstrating that AdaSkip exhibits favorable inference performance over the baselines.

