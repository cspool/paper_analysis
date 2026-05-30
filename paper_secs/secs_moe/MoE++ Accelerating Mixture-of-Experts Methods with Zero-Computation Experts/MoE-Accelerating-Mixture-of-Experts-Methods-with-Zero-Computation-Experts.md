# MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

Peng Jin<sup>1,2,3\*</sup>, Bo Zhu<sup>4</sup>, Li Yuan<sup>1,2,3</sup>, Shuicheng Yan<sup>4</sup>

jp21@stu.pku.edu.cn, yuanli-ece@pku.edu.cn

Code: https://github.com/SkyworkAI/MoE-plus-plus

#### **ABSTRACT**

In this work, we aim to simultaneously enhance the effectiveness and efficiency of Mixture-of-Experts (MoE) methods. To achieve this, we propose MoE++, a general and heterogeneous MoE framework that integrates both Feed-Forward Network (FFN) and zero-computation experts. Specifically, we introduce three types of zero-computation experts: the zero expert, copy expert, and constant expert, which correspond to discard, skip, and replace operations, respectively. This design offers three key advantages: (i) Low Computing Overhead: Unlike the uniform mixing mechanism for all tokens within vanilla MoE, MoE++ allows each token to engage with a dynamic number of FFNs, be adjusted by constant vectors, or even skip the MoE layer entirely. (ii) High Performance: By enabling simple tokens to utilize fewer FFN experts, MoE++ allows more experts to focus on challenging tokens, thereby unlocking greater performance potential than vanilla MoE. (iii) **Deployment Friendly**: Given that zero-computation experts have negligible parameters, we can deploy all zero-computation experts on each GPU, eliminating the significant communication overhead and expert load imbalance associated with FFN experts distributed across different GPUs. Moreover, we leverage gating residuals, enabling each token to consider the pathway taken in the previous layer when selecting the appropriate experts. Extensive experimental results demonstrate that MoE++ achieves better performance while delivering  $1.1 \sim 2.1 \times$  expert forward throughput<sup>†</sup> compared to a vanilla MoE model of the same size, which lays a solid foundation for developing advanced and efficient MoE-related models.

### 1 Introduction

Large Language Models (LLMs) (Brown et al., 2020; OpenAI, 2022; Ouyang et al., 2022; Chowdhery et al., 2023; Achiam et al., 2023) have achieved substantial advancements, primarily attributed to the expansion of training data and a significant increase in model parameters. However, the pursuit of ever-larger model sizes incurs prohibitive computational costs. Therefore, the Mixture-of-Experts (MoE) architecture (Jacobs et al., 1991; Zhou et al., 2022; Roller et al., 2021), which allows for parameter scaling while keeping computational costs manageable, has become a preferred solution. The recent incorporation of MoE architectures into Transformers (Vaswani et al., 2017) has enabled the effective scaling of language models to impressive sizes, resulting in exceptional performance (Team, 2024; Dai et al., 2024; Jiang et al., 2024; Shen et al., 2024; Wei et al., 2024). These achievements underscore the significant potential and promise of MoE language models.

Most existing Mixture-of-Experts (MoE) methods (Du et al., 2022; Fedus et al., 2022; Lewis et al., 2021; Rajbhandari et al., 2022) typically activate a fixed number of Feed-Forward Networks (FFNs) for all tokens. In many works (Lepikhin et al., 2021; Xue et al., 2024), each token selects the top

<sup>&</sup>lt;sup>1</sup>School of Electronic and Computer Engineering, Peking University, Shenzhen, China

<sup>&</sup>lt;sup>2</sup>Peng Cheng Laboratory, Shenzhen, China

<sup>&</sup>lt;sup>3</sup>AI for Science (AI4S)-Preferred Program, Peking University Shenzhen Graduate School, China

<sup>&</sup>lt;sup>4</sup>Kunlun 2050 Research & Skywork AI, Singapore

<sup>\*</sup>This work was performed when Peng Jin was an Intern at Skywork AI. Corresponding author: Li Yuan, Shuicheng Yan.

<sup>&</sup>lt;sup>†</sup>We define expert throughput as the throughput of FFN experts and zero-computation experts (if present).

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1: A high-level comparison between the vanilla MoE and our proposed MoE++ architecture. Subfigure (a) illustrates a standard MoE layer utilizing a Top-2 routing strategy, while subfigure (b) demonstrates the integration of zero-computation experts in MoE++. It is worth noting that these zero-computation experts require an almost negligible number of parameters, ensuring that the total parameter count for MoE++ is preserved at the same level as the vanilla MoE.

two FFNs and aggregates their outputs as the input for the subsequent layer. However, it is evident that not all tokens hold equal prediction difficulty in language tasks. For example, simple tokens, such as punctuation marks like commas, may only require a single expert. Conversely, tokens that poorly align with the existing experts might potentially benefit from bypassing the MoE layer entirely. Drawing from this insight, we contend that the rigidly fixed mixing mechanism used in previous work leads to training and inference inefficiencies, ultimately resulting in sub-optimal model performance.

In this work, we propose a general and heterogeneous MoE framework, called MoE++. To achieve a flexible computation allocation, we introduce three types of zero-computation experts: the zero expert, which discards input; the copy expert, which replicates input; and the constant expert‡ , which substitutes input with a trainable vector. As shown in Fig. [1,](#page-1-0) unlike vanilla MoE methods that restrict each token to a fixed number of FFN experts, MoE++ allows each token to engage with a variable number of FFN experts, receive adjustments through constant vectors, or even bypass the MoE layer entirely. This heterogeneous structure has a higher fitting ability by broadening the range of sub-network combinations with less computing overhead than vanilla MoE. Furthermore, we incorporate gating scores from the previous layer into the expert selection of the current layer. These gating residuals enable each token to consider its previous pathway when selecting the experts.

Starting with a modest scale of 0.6B parameters and expanding to 7B, extensive experimental results show that our MoE++ method significantly outperforms the vanilla MoE method by a substantial margin. It is worth noting that when scaled to 7B parameters and trained from scratch with 1T tokens, the MoE++ model achieves better performance than OpenMoE-8B/32E [\(Xue et al.,](#page-13-4) [2024\)](#page-13-4), a larger MoE model trained from scratch with 1.1T tokens. Meanwhile, the MoE++ model requires only about 57% of the computational cost of OpenMoE-8B/32E. More encouragingly, MoE++ allows simple tokens to utilize fewer FFN experts, freeing up more FFN experts to focus on challenging tokens. This results in both Reduced Computation and Enhanced Performance. Moreover, since the memory overhead of zero-computation experts is negligible, we can deploy all zero-computation experts on each GPU, eliminating significant communication overhead and expert load imbalance. Therefore, MoE++ is highly Deployment-Friendly. Extensive experiments show that MoE++ achieves approximately a 15%∼111% increase in expert forward throughput compared to a vanilla MoE model of the same size. The main contributions of this work are summarized as follows:

- Zero-computation experts. To the best of our knowledge, we are the first to propose zerocomputation experts for the MoE architecture. By introducing zero-computation experts, MoE++ has a higher fitting ability with less computing overhead than vanilla MoE.
- Gating residuals. We introduce gating residuals, which empower each token to consider its previous pathway when selecting the appropriate experts in the current MoE++ layer.
- Flexible computation allocation. MoE++ optimizes computation allocation by assigning fewer FFN experts to simple tokens, allowing more FFN experts to be dedicated to challenging tokens. Extensive experiments demonstrate that MoE++ not only enhances overall

<sup>‡</sup>Constant experts involve negligible computation, so we also consider them as zero-computation experts.

performance but also delivers up to 2× expert forward throughput compared to vanilla MoE methods, laying a foundation for developing advanced and efficient language models.

