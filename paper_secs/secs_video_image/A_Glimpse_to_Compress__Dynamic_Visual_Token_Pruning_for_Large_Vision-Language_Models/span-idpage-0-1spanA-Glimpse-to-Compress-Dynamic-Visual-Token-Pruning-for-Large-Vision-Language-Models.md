# <span id="page-0-1"></span>A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models

Quan-Sheng Zeng1,<sup>2</sup> , Yunheng Li<sup>1</sup> , Qilong Wang<sup>3</sup> , Peng-Tao Jiang<sup>4</sup> , Zuxuan Wu<sup>2</sup> , Ming-Ming Cheng<sup>1</sup> , Qibin Hou†<sup>1</sup>

<sup>1</sup>VCIP, CS, Nankai University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Tianjin University <sup>4</sup>vivo Mobile Communication Co., Ltd †Corresponding author.

Abstract: Visual token compression is critical for Large Vision-Language Models (LVLMs) to efficiently process high-resolution inputs. Existing methods that typically adopt fixed compression ratios cannot adapt to scenes of varying complexity, often causing imprecise pruning that discards informative visual tokens and results in degraded model performance. To address this issue, we introduce a dynamic pruning framework, GlimpsePrune, inspired by human cognition. It takes a data-driven "glimpse" and prunes irrelevant visual tokens in a single forward pass before answer generation. This approach prunes 92.6% of visual tokens while on average fully retaining the baseline performance on free-form VQA tasks. The reduced computational cost also enables more effective fine-tuning: an enhanced GlimpsePrune+ achieves 110% of the baseline performance while maintaining a similarly high pruning rate. Our work paves a new way for building more powerful and efficient LVLMs.

Project Page: <https://github.com/HVision-NKU/GlimpsePrune>

HVision@Nankai

## 1 Introduction

The rise of Large Language Models (LLMs) has spurred the development of large vision language models (LVLMs). Standard LVLMs encode visual inputs into tokens, which are then projected into the textual embedding space and concatenated with text instructions for an LLM to process. A recent trend towards a fixed patch size, rather than a fixed number of patches, means that higher-resolution inputs often generate a large number of visual tokens.

This proliferation of visual tokens creates a significant computational burden, as the number of visual tokens often dwarfs the textual query tokens. Usually, only a small, querydependent subset of these visual tokens is relevant for answering a given question. This highlights the need for dynamic, instruction-guided visual token pruning. Existing methods either fix the compression ratio in the architectural design [\[1,](#page-6-0) [2\]](#page-6-1) or use hand-crafted compression metric design [\[3,](#page-6-2) [4,](#page-6-3) [5\]](#page-6-4). As shown in Fig. [1,](#page-0-0) when setting the same upper limit on the retention rate of visual tokens, existing methods still retain a high degree of redundancy among the selected tokens. This motivates our central research question: How can we learn a data-driven metric for dynamic and efficient visual token pruning?

Beyond their inflexible designs, some existing token pruning methods exhibit a critical limitation in real-world, freeform response scenarios. This is because they operate on the flawed assumption that the cross-attention from the first generated token to the visual tokens reliably indicates visual token importance [\[3,](#page-6-2) [6\]](#page-7-0). We find this assumption to be frag<span id="page-0-0"></span>*What player is number 21? Answer the question using a single word. Retention rate* ≤ *11%*

![](_page_0_Picture_12.jpeg)

![](_page_0_Picture_13.jpeg)

Figure 1 Compared to previous methods with fixed retention rates [\[5\]](#page-6-4), our method can dynamically select effective visual tokens while being more accurate.

ile. Tab. [1](#page-1-0) provides quantitative evidence, showing that the performance of prior methods drops sharply when they are not prompted for concise answers. The reason is visually evident in Fig. [2:](#page-1-1) when generating free-form responses, the cross-attention initially focuses on irrelevant regions, leading to imprecise pruning and a degraded final output.

While dynamic pruning at each decoding step appears to be a straightforward solution, its practical utility is limited. The critical information for pruning — the cross-attention maps that best indicate visual token importance — has been shown to emerge only at deeper layers K of the LLM decoder [\[3,](#page-6-2) [7,](#page-7-1) [5\]](#page-6-4). A per-step pruning strategy would thus have to operate

#### <span id="page-1-2"></span><span id="page-1-1"></span>*Q: How long has the drink on the right been aged?*

A: The drink on the right is Bowmore Islay Single Malt Scotch Whisky, and it is labeled as "Aged 10 Years." This indicates that the whisky has been aged for 10 years before bottling.

![](_page_1_Figure_2.jpeg)

Figure 2 Without a brief prompt, the cross-attention map may initially fail to focus on relevant visual tokens, and only begins to shift its focus once the generated text aligns with those visual elements.

<span id="page-1-0"></span>

| Method    | Qwen2.5-VL-7B | PDrop | VScan | GlimpsePrune |
|-----------|---------------|-------|-------|--------------|
| w/ brief  | 0.936         | 0.753 | 0.845 | 0.929        |
| w/o brief | 0.927         | 0.406 | 0.781 | 0.939        |

Table 1 On TextVQA, previous methods (PDrop, VScan) show high sensitivity to the brief prompt, while the base model (Qwen2.5-VL-7B) and our method remain robust.

after the forward pass through these initial K layers, which necessitates retaining the full KV cache for all visual tokens up to that point. Consequently, this strategy would offer limited savings on computation and memory.

To address these challenges, we introduce GlimpsePrune, a dynamic visual token pruning framework inspired by the human cognitive process of first glancing at relevant areas before responding. During the prefilling stage, we temporarily insert a learnable "glimpse token" after the instruction tokens. We leverage the forward pass through the initial K decoder layers to extract the cross-attention scores between this glimpse token and all visual tokens, and feed them into a lightweight predictor. Both the glimpse token and the predictor are trained on a small amount of Grounded Question Answering (GQA) data to learn a data-driven pruning metric. Based on this, we perform a one-time pruning of irrelevant visual tokens and their corresponding KV cache entries across all model layers. This strategy reduces the computational and memory footprint for the remaining L − K prefill layers and the entire subsequent decoding phase.

Extensive experiments demonstrate our predictor's strong generalization capabilities. Integrated with Qwen2.5-VL-7B, GlimpsePrune achieves an average visual token pruning rate of 92.6% while retaining 100% of the original model's perfor-

mance on free-form VQA tasks. Leveraging this efficiency, we show that under a reinforcement learning framework, our fine-tuned version, GlimpsePrune+, reaches 110% of the original performance while maintaining a high pruning rate, highlighting a powerful synergy between our pruning strategy and advanced fine-tuning.

## 2 Related Work

Large vision-language models. Pioneered by models like BLIP-2 [\[8\]](#page-7-2) and LLaVA [\[9\]](#page-7-3), LVLMs combine a vision encoder, a projector, and an LLM to achieve visual understanding. Recent models, including the InternVL [\[10,](#page-7-4) [11,](#page-7-5) [12,](#page-7-6) [13,](#page-7-7) [14\]](#page-7-8) and Qwen-VL [\[15,](#page-7-9) [16,](#page-7-10) [17\]](#page-8-0) series, support dynamic, highresolution inputs. While enabling fine-grained analysis, this capability dramatically increases visual tokens, creating a computational bottleneck that motivates token compression research.

Visual token compression. To mitigate the high computational cost of LVLMs, various token reduction strategies have been proposed. Some methods embed compression directly into the model architecture [\[1,](#page-6-0) [2\]](#page-6-1). However, these approaches are inflexible, suffering from fixed compression ratios tied to the architecture and requiring costly, full-scale retraining. Consequently, training-free, post-hoc methods that prune or merge tokens have gained prominence, which typically rely on two types of metrics: importance-based metrics and similarity-based metrics.

For importance-based metrics, prior works typically compute importance scores using the cross-attention from a global query representation to all visual tokens. This query representation can be derived from the instruction token [\[18,](#page-8-1) [19,](#page-8-2) [7,](#page-7-1) [20,](#page-8-3) [5\]](#page-6-4), or a dedicated global visual token [\[4,](#page-6-3) [21,](#page-8-4) [22\]](#page-8-5). However, such a hand-crafted importance metric can be unreliable for free-form generation tasks, where the user's core objective is not explicitly stated in the query, or the target information does not appear at the beginning of the generated response. Similarity-based metrics are also used by some prior works to prune and merge visual tokens, which typically use cosine distance or self-attention scores within the visual features [\[4,](#page-6-3) [23,](#page-8-6) [20,](#page-8-3) [24,](#page-8-7) [25\]](#page-8-8). However, these techniques incur substantial memory overhead from the computation of dense similarity matrices, potentially negating the efficiency gains from optimizations like FlashAttention [\[26,](#page-8-9) [27\]](#page-8-10).

Our work presents a data-driven pruning metric to enable a dynamic compression ratio and addresses the limitations of hand-crafted ones. Although some recent works also share the motivation for dynamic token compression [\[28,](#page-8-11) [29,](#page-8-12) [30,](#page-8-13) [31\]](#page-8-14), our contribution is distinct in several key aspects. (i) We identify and address the inadequacy of prior methods for free-form VQA tasks; (ii) Our proposed framework employs a glimpse token and a lightweight predictor, which are supported by a tailored training methodology. (iii) Our pruning strategy is fundamentally more efficient, performing a single full-depth KV cache prune per response. This is in stark contrast to the

<span id="page-2-2"></span><span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 3 Overview of GlimpsePrune. (a) At the prefilling stage, we prune irrelevant visual tokens after K layers of the LLM decoder. (b) The architecture of visual token importance predictor (VIP). (c) The details of conditional self-attention with 2D RoPE. (d) At the decoding stage, the answer generation relies on the pruned KV cache, saving memory and I/O bandwidth.

incremental pruning [3] or partial [7, 5] pruning in prior art, thus maximizing memory savings during decoding.

## 3 Preliminaries

Inference in LVLMs is a two-phase process: prefilling and decoding, facilitated by a KV cache. To mitigate the high memory cost of attention, techniques like FlashAttention [26, 27] are now standard. We denote the visual and text token sequences by lengths  $N_v$  and  $N_t$ , respectively (where typically  $N_v \gg N_t$ ), and consider an LLM with L decoder layers of hidden dimension D. The prefilling phase is compute-intensive. It processes the initial  $N_v + N_t$  tokens via causal self-attention, with a computational cost of  $\mathcal{O}((N_v + N_t)^2 D)$ , to populate a KV cache of size  $2L(N_v + N_t)D$ . Conversely, the decoding phase is memory-bandwidth-intensive. It generates tokens autoregressively, where the attention cost is reduced to  $\mathcal{O}((N_v + N_t)D)$  per token by leveraging the cache. The bottleneck thus shifts to the I/O cost of loading this large cache from memory.

While optimizations like FlashAttention reduce prefilling memory by avoiding the materialization of the full  $\mathcal{O}((N_v + N_t)^2)$  attention matrix, this efficiency gain is lost by pruning methods that require this exact matrix for token similarity assessment [20, 5]. In contrast, computing cross-attention from a single query to all visual tokens is a linear-cost operation compatible with FlashAttention's memory efficiency.

Our approach, therefore, intervenes mid-prefilling. After a designated layer K, we employ a lightweight selection module, informed by a linear-cost glimpse attention and precomputed visual features, to prune the visual token set from  $N_v$  down to  $N_v'$ . Critically, this action simultaneously prunes

the KV cache for all preceding layers (1 to K). This single intervention reduces the computational load for the remaining L-K prefill layers and, more importantly, slashes both memory and I/O costs for the entire subsequent decoding phase. More details are presented in Sec. 4.

#### <span id="page-2-0"></span>4 Method

#### Hierarchical visual features and glimpse attention extrac-

**tion.** When the visual encoder processes a patched image into  $N_v$  visual tokens, we extract C-dimensional visual features, denoted as  $V \in \mathbb{R}^{M \times N_v \times C}$ , from M of its intermediate layers. These hierarchical features provide multi-scale visual priors that are crucial for the subsequent visual token importance prediction.

For the prefilling stage, we introduce an  $L \times D$  matrix of learnable embeddings as glimpse token. The first embedding is concatenated to the end of the input sequence, and then at each decoder layer, its respective embedding is added to the hidden state at that newly appended position. The purpose of appending them at the final position is to leverage the LLM decoder's causal attention mechanism. This allows the glimpse token to fully interact with both visual and textual tokens across all layers, thereby capturing early signals about the importance of visual tokens relevant to the user's instruction, all without perturbing the original processing of the visual and textual tokens themselves.

We posit that by an intermediate layer K, the glimpse token has already aggregated sufficiently clear importance information. Our subsequent experiments demonstrate that setting  $K = \lceil 2/3 \times L \rceil$  strikes a favorable trade-off between performance and the computational overhead of the prefilling

<span id="page-3-1"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 4 Overview of the training process: (a) Training glimpse token and VIP for data-driven pruning metric. (b) RL fine-tuning LLM for GlimpsePrune+.

stage. We then extract the *glimpse attention* from the glimpse token to all visual tokens at layer K, denoted as  $A \in \mathbb{R}^{N_v \times H}$ , where H is the number of attention heads.

Visual importance prediction and one-shot pruning. Subsequently, the extracted glimpse attention A and the multilevel visual features V are fed into the selection module, named Visual Importance Predictor (VIP). As illustrated in Fig. 3(b), the VIP is composed of M Self-Attention Blocks, equipped with 2D Rotary Position Embeddings (RoPE) [32]. The glimpse attention A is first projected into a new embedding space  $A' \in \mathbb{R}^{N_v \times E}$ , where E is the VIP's hidden dimension. As illustrated in Fig. 3(c), within the m-th self-attention block, we project the corresponding m-th visual feature map  $V_m$  to  $V_m' \in \mathbb{R}^{N_v \times F}$  and concatenate it to the query and key vectors. This allows the relationship within the visual tokens of V to act as additional conditioning, influencing the final importance prediction.

By setting the dimensions  $E \ll D$ ,  $F \ll C$  (where D is the LLM's hidden dimension and C is the visual feature dimension), and implementing a memory-efficient attention mechanism, the computational overhead of the entire VIP remains negligible compared to a single LLM decoder layer. The VIP's output, an importance map P, dynamically and precisely identifies visual tokens relevant to the user's instruction, adapting to different inputs and visual targets. Furthermore, a maximum retention ratio can be configured to sparsify the visual tokens even when P focuses on large-scale visual objects.

Based on the importance map, we perform a *one-shot pruning* at layer K. Unimportant visual tokens are removed from the layer-K hidden state and their corresponding KV cache entries in all preceding layers, permanently reducing the sequence length to  $N_v' + N_t$  for all subsequent computations. The glimpse token is also discarded so that the answer generation will not be influenced by it.

Importance metric learning by glimpse token and VIP.

The glimpse tokens and the VIP are trained while keeping the parameters of the base VLM frozen, using only a small amount of Grounded Question Answering (GQA) data. Each data sample consists of a question, its corresponding answer, and the bounding boxes of the visual regions relevant to the answer. Our training objectives are two-fold.

On one hand, we apply a language modeling loss that encourages the model to generate the correct answer immediately after the position of the glimpse token. On the other hand, we employ a localization loss to optimize the VIP's output. This loss is a weighted sum of DiceLoss and Binary Cross-Entropy (BCE) Loss. Given that bounding box annotations are often coarse, we prioritize recalling foreground visual tokens over achieving precise boundary segmentation. Therefore, we set the weight ratio for DiceLoss to BCE Loss at 10:1. The language loss and the localization loss are balanced with a 1:1 weight ratio.

Since the LVLM's original attention mechanism already possesses an inherent localization capability, our training primarily serves to calibrate the newly introduced parameters and amplify this ability. Consequently, our method exhibits strong generalization, a claim we will substantiate in the experimental section.

**Reinforcement learning fine-tuning.** RL fine-tuning on dense visual inputs is computationally expensive and memory-intensive. Combining with our pruning mechanism mitigates this cost by first employing a pruning mechanism to create a sparse visual context. We then fine-tune the LLM decoder using the Group-wise Ranking Policy Optimization (GRPO) framework [33].

As illustrated in Fig. 4(b), the fine-tuning process begins after pruning at layer K. The policy model generates multiple candidate responses, which are evaluated by a reward model to compute a policy loss. The loss is regularized with a KL divergence term against a reference model. The new model, named GlimpsePrune+, is trained iteratively. We first update the LLM decoder via GRPO, and then train our glimpse token

<span id="page-4-0"></span>and predictor on the updated decoder. This process allows GlimpsePrune+ to achieve superior performance over the baseline model even at a high pruning rate.

## 5 Experiments

## 5.1 Experimental setup

Model architectures. Our primary experiments are conducted on Qwen2.5-VL-7B [\[17\]](#page-8-0), an open-source architecture that supports dynamically high-resolution inputs (from 4 to 16, 384 visual tokens). To demonstrate the generalizability of our approach, we also adapt and evaluate it on LLaVA-1.5- 7B [\[34\]](#page-8-17), a model designed for fixed-resolution inputs (576 visual tokens). In these architectures, we set the VIP's hidden dimension to E = 256 and the visual conditioning dimension to F = 512, with M layers of visual features. The pruning position is set to approximately K = ⌈2/3 × L⌉ of the original LLM decoder. For detailed settings on different model sizes, please refer to the Appendix [C.](#page-11-0)

Training setups. To demonstrate the generalization capability of GlimpsePrune, we train the VIP and glimpse token using only 20K randomly sampled data from the GQA dataset [\[35\]](#page-8-18). We only train for one epoch, which can be completed in about 0.5 hours on a single NVIDIA A100 GPU. For the RL finetuning in GlimpsePrune+, we use 240K samples randomly selected from the VisCoT dataset [\[36\]](#page-8-19), which provides a diverse set of visual question-answering scenarios. For the GRPO algorithm, we set the group size to 4, the number of policy update iteration to 1, and the KL divergence coefficient to 0.04, respectively. We apply LoRA [\[37\]](#page-8-20) with a rank of R = 64 to fine-tune the LLM decoder. More details can be found in the Appendix [B.](#page-10-0)

Evaluation setups. For free-form VQA, we utilize the comprehensive benchmark from VisCoT [\[36\]](#page-8-19), which comprises 12 diverse VQA datasets spanning multiple domains and featuring a wide range of target object scales. Crucially, evaluation is performed by a powerful LLM rather than rule-based string matching, providing a more holistic assessment of generative quality. We use Qwen2.5-32B-Instruct-GPTQ-Int8 [\[38\]](#page-8-21) as the evaluator LLM. For short-form VQA, we select 10 widely-used VQA benchmarks and evaluate them under the LMMs-Eval framework [\[39\]](#page-8-22). These benchmarks typically require concise answers, such as a short phrase or a single option from a list, and are evaluated using exact-match or rulebased string comparison. Further details on these datasets are provided in the Appendix [A.](#page-10-1)

Comparison methods. We compare our method against several state-of-the-art and representative visual token compression techniques: PDrop [\[7\]](#page-7-1), VisionZip [\[4\]](#page-6-3), DivPrune [\[23\]](#page-8-6), CDPruner [\[20\]](#page-8-3) and VScan [\[5\]](#page-6-4). Since these methods operate with a fixed compression ratio and apply compression at different stages, we set an upper bound on the average retention rate of visual tokens' KV cache during the decoding phase

for all methods. As our proposed method, GlimpsePrune, features a dynamic retention rate, we will report both its performance under this constraint and its average retention rate on each dataset.

## 5.2 Main results

GlimpsePrune for free-form VQA. As detailed in Tab. [2,](#page-5-0) we evaluate GlimpsePrune on 12 free-form VQA datasets under three distinct average retention rate constraints: 11.1%, 22.2%, and 33.3%. With an average retention rate as low as 11.1%, our method achieves a mean accuracy of 0.761, remarkably retaining 100% of the performance of the original Qwen2.5-VL-7B model.

The dynamic nature of GlimpsePrune is particularly evident in its adaptability to varying target object scales. On datasets with small target objects, such as DocVQA, GlimpsePrune dynamically adjusts to a much lower retention rate (3.6-3.8%) while exhibiting only a marginal performance drop (from 0.964 to 0.962). Conversely, on datasets featuring large-scale visual targets like VSR, even when constrained by a stringent upper bound, GlimpsePrune effectively sparsifies the visual context. It reduces the retention rate from 39.4% (unconstrained) to 10.3% with similar performance (0.618 vs. 0.620), These results strongly indicate that GlimpsePrune not only accurately identifies critical visual tokens but also intelligently sparsifies them when operating under tight computational budgets. We provide qualitative examples in the Appendix [D.](#page-11-1)

On high-resolution inputs, methods like VisionZip [\[4\]](#page-6-3) and VScan [\[5\]](#page-6-4) face practical limitations. Their reliance on computing dense similarity matrices leads to prohibitive memory usage that negates FlashAttention2 [\[27\]](#page-8-10) benefits, often causing Out-of-Memory (OOM) errors. In contrast, our method is designed to remain efficient under these conditions.

GlimpsePrune for short-form VQA. As shown in Tab. [3,](#page-6-5) GlimpsePrune also demonstrates strong performance on shortform VQA tasks. With an average retention rate capped at 11.1%, GlimpsePrune achieves an average accuracy of 70.0% across 10 benchmarks, which is 99.6% of the original Qwen2.5-VL-7B model's performance.

<span id="page-4-1"></span>GlimpsePrune for other architectures. As shown in Tab. [4,](#page-6-6) GlimpsePrune also demonstrates strong performance on the LLaVA-1.5-7B model. Due to the fixed number of 576 visual tokens used as input in LLaVA-1.5-7B, the dynamic range for compression is smaller than that of the Qwen2.5-VL-7B model. Nevertheless, GlimpsePrune still achieves an average accuracy of 0.490 with an average retention rate capped at 11.1%, which is 97.6% of the original LLaVA-1.5-7B model's performance. More results on other architectures can be found in the Appendix [D.](#page-11-1)

<span id="page-5-0"></span>

|               | General<br>VQA |      |             | Relation<br>Reasoning            |       | Fine<br>grained      |                                 |       | Doc/Text                                           |       |       | Chart |       |
|---------------|----------------|------|-------------|----------------------------------|-------|----------------------|---------------------------------|-------|----------------------------------------------------|-------|-------|-------|-------|
| Method        |                |      |             | Flickr30k V7W GQA OpenImages VSR |       | CUB                  |                                 |       | DocVQA TextCaps TextVQA DUDE SROIE InfoVQA Average |       |       |       |       |
| Qwen2.5-VL-7B | 0.761          |      | 0.615 0.567 | 0.425                            | 0.642 | 0.724                | 0.964                           | 0.868 | 0.927                                              | 0.801 | 0.977 | 0.860 | 0.761 |
|               |                |      |             |                                  |       |                      | Average retention rate ≤ 11.1 % |       |                                                    |       |       |       |       |
| PDrop         | 0.373          |      | 0.301 0.248 | 0.293                            | 0.177 | 0.315                | 0.263                           | 0.358 | 0.406                                              | 0.238 | 0.102 | 0.235 | 0.276 |
| VisionZip     | 0.657          |      | 0.520 0.505 | oom                              | 0.565 | 0.661                | oom                             | 0.660 | 0.674                                              | oom   | oom   | oom   | –     |
| VScan         | 0.635          |      | 0.518 0.475 | oom                              | 0.484 | 0.627                | oom                             | 0.678 | 0.781                                              | oom   | oom   | oom   | –     |
|               | 8.9%           | 8.8% | 8.7%        | 8.6%                             |       | 10.3% 10.5%          | 3.6%                            | 6.7%  | 7.1%                                               | 4.4%  | 6.7%  | 4.5%  | 7.4%  |
| GlimpsePrune  | 0.751          |      | 0.636 0.584 | 0.458                            | 0.620 | 0.700                | 0.962                           | 0.861 | 0.939                                              | 0.786 | 0.978 | 0.854 | 0.761 |
|               |                |      |             |                                  |       |                      | Average retention rate ≤ 22.2 % |       |                                                    |       |       |       |       |
| PDrop         | 0.465          |      | 0.380 0.318 | 0.354                            | 0.263 | 0.486                | 0.398                           | 0.473 | 0.545                                              | 0.352 | 0.176 | 0.288 | 0.375 |
| VisionZip     | 0.716          |      | 0.561 0.534 | oom                              | 0.598 | 0.713                | oom                             | 0.788 | 0.840                                              | oom   | oom   | oom   | –     |
| VScan         | 0.690          |      | 0.568 0.511 | oom                              | 0.582 | 0.645                | oom                             | 0.775 | 0.856                                              | oom   | oom   | oom   | –     |
|               | 18.6%          |      | 15.0% 14.5% | 13.8%                            |       | 18.4% 18.6%          | 3.8%                            | 8.5%  | 9.2%                                               | 5.3%  | 7.1%  | 5.2%  | 11.5% |
| GlimpsePrune  | 0.757          |      | 0.639 0.583 | 0.461                            | 0.621 | 0.717                | 0.962                           | 0.861 | 0.941                                              | 0.792 | 0.978 | 0.857 | 0.764 |
|               |                |      |             |                                  |       |                      | Average retention rate ≤ 33.3 % |       |                                                    |       |       |       |       |
| PDrop         | 0.537          |      | 0.427 0.366 | 0.392                            | 0.336 | 0.552                | 0.505                           | 0.562 | 0.637                                              | 0.420 | 0.248 | 0.392 | 0.448 |
| VisionZip     | 0.731          |      | 0.590 0.550 | oom                              | 0.619 | 0.718                | oom                             | 0.839 | 0.897                                              | oom   | oom   | oom   | –     |
| VScan         | 0.714          |      | 0.580 0.515 | oom                              | 0.590 | 0.662                | oom                             | 0.812 | 0.892                                              | oom   | oom   | oom   | –     |
|               | 18.3%          |      | 24.7% 18.4% | 17.2%                            |       | 24.7% 22.4%          | 3.8%                            | 9.3%  | 10.1%                                              | 5.6%  | 7.2%  | 5.5%  | 13.9% |
| GlimpsePrune  | 0.757          |      | 0.636 0.587 | 0.464                            | 0.620 | 0.723                | 0.963                           | 0.860 | 0.940                                              | 0.792 | 0.978 | 0.860 | 0.765 |
|               |                |      |             |                                  |       | Unlimited constraint |                                 |       |                                                    |       |       |       |       |
|               | 26.8%          |      | 31.2% 25.0% | 23.1%                            |       | 39.4% 25.5%          | 3.8%                            | 10.1% | 11.1%                                              | 6.0%  | 7.2%  | 5.5%  | 17.9% |
| GlimpsePrune  | 0.759          |      | 0.635 0.583 | 0.460                            | 0.618 | 0.722                | 0.963                           | 0.860 | 0.941                                              | 0.795 | 0.978 | 0.860 | 0.764 |
|               | 28.7%          |      | 31.9% 24.9% | 23.2%                            |       | 45.9% 23.7%          | 6.9%                            | 12.1% | 11.5%                                              | 9.9%  | 13.8% | 8.4%  | 20.1% |
| GlimpsePrune+ | 0.776          |      | 0.753 0.722 | 0.719                            | 0.800 | 0.830                | 0.956                           | 0.876 | 0.953                                              | 0.822 | 0.977 | 0.869 | 0.838 |

Table 2 Performance in free-form VQA tasks, "oom" indicates out-of-memory errors.

## 5.3 Ablation studies

We conduct ablation studies on the Qwen2.5-VL-3B model, which has an LLM decoder with L = 36 layers and achieves an average performance of 0.746 on free-form VQA tasks. We will analyze the impact of design choices and training components in GlimpsePrune under a fixed retention rate upper limit of 11.1%.

The importance of glimpse tokens. We first implement a baseline without training, as shown in the first row of Tab. [5:](#page-7-11) using the attention weights from the first generated token to the visual tokens at layer K = 24 as the pruning metric. This baseline achieves an average accuracy of only 0.546. When we add glimpse tokens and subsequently add language and location losses, the performance gradually improves to 0.684 and 0.702, respectively. This demonstrates that glimpse tokens can provide a good visual importance prediction before the model generates free-form answers.

The importance of visual conditions. The last two rows of Tab. [5](#page-7-11) highlight the necessity of conditioning our importance map optimization on visual features. Without them, merely adding a 4-layer self-attention VIP only improves the performance from 0.702 to 0.716. Conversely, incorporating visual features as conditions in VIP boosts performance to 0.736. Notably, this performance gain predominantly stems

from Doc/Text data, whose distribution significantly differs from the training set. This suggests that visual conditioning mitigates the model's tendency towards shortcut learning.

The selection of pruning layer. The choice of the pruning layer, K, presents a trade-off between performance and efficiency. While deeper layers provide more interpretable attention maps, they also increase prefill computation. Based on our analysis in Tab. [6,](#page-7-12) we found K = 24 (two-thirds of the total layers) to be an optimal balance. We therefore adopt this 2/3 ratio across all tested architectures. Notably, pruning at K = 36 (the last layer) leads to a performance drop to 0.556, as the glimpse attention of the final layer tends to focus less on visual information.

## 5.4 Efficiency analysis

Efficent inference. As shown in Tab. [7,](#page-7-13) our method significantly enhances inference efficiency. Benchmarked on a single A100 GPU with the Qwen2.5-VL-7B model on 100 DocVQA samples (with KV Cache and FlashAttention2), GlimpsePrune reduces the computation-intensive prefill cost to 69.1% of the baseline. More critically, for the memoryintensive decoding phase, it slashes the initial KV Cache length from 5,073.9 to 202.5 tokens. This leads to a reduction in peak memory usage across the entire generation process to

<span id="page-6-5"></span>

| Method               | $VQA_{vallite}^{v2}$                 | GQA  | VizWizval | SQA <sup>img</sup> | POPE | MME    | MMB <sup>en</sup> <sub>test</sub> | MMB <sup>cn</sup> <sub>test</sub> | SEED <sup>img</sup> | <b>V</b> * | Average |
|----------------------|--------------------------------------|------|-----------|--------------------|------|--------|-----------------------------------|-----------------------------------|---------------------|------------|---------|
| Qwen2.5-VL-7B        | 79.5                                 | 61.4 | 70.6      | 87.5               | 87.5 | 2327.9 | 83.2                              | 82.8                              | 77.5                | 71.7       | 70.3    |
|                      | Average retention rate $\leq 11.1\%$ |      |           |                    |      |        |                                   |                                   |                     |            |         |
| PDrop                | 53.3                                 | 41.8 | 53.3      | 77.0               | 68.4 | 1642.2 | 42.9                              | 38.7                              | 49.4                | 42.4       | 46.8    |
| VisionZip            | 68.4                                 | 54.9 | 66.4      | 82.7               | 82.9 | 2031.6 | 75.6                              | 72.6                              | _                   | 60.2       | _       |
| VScan                | 74.2                                 | 57.1 | 67.5      | 82.2               | 86.1 | 2129.8 | 74.1                              | 71.5                              | _                   | 59.7       | _       |
| GlimpsePrune         | 78.0                                 | 59.5 | 69.4      | 87.2               | 87.5 | 2348.0 | 82.6                              | 81.7                              | 77.4                | 72.8       | 70.0    |
| Unlimited constraint |                                      |      |           |                    |      |        |                                   |                                   |                     |            |         |
| GlimpsePrune         | 79.1                                 | 60.0 | 70.5      | 87.3               | 87.6 | 2323.1 | 82.9                              | 81.7                              | 77.4                | 72.8       | 70.0    |

Table 3 Performance in short-form VQA tasks, the maximum number of visual tokens is set to 2048 here.

<span id="page-6-6"></span>

|              | Gene<br>VQ           |       |       | Relation<br>Reasoning |         | Fine<br>grained | l                | Г        | Ooc/Text  |       |       | Chart   |         |
|--------------|----------------------|-------|-------|-----------------------|---------|-----------------|------------------|----------|-----------|-------|-------|---------|---------|
| Method       | Flickr30k            | V7W   | GQA   | OpenImages            | VSR     | CUB             | DocVQA           | TextCaps | s TextVQA | DUDE  | SROIE | InfoVQA | Average |
| LLaVA-1.5-7B | 0.694                | 0.658 | 0.635 | 0.519                 | 0.613   | 0.560           | 0.252            | 0.632    | 0.650     | 0.293 | 0.125 | 0.393   | 0.502   |
|              |                      |       |       | Av                    | erage 1 | retention       | $rate \leq 11$ . | .1 %     |           |       |       |         |         |
| PDrop        | 0.493                | 0.498 | 0.454 | 0.501                 | 0.462   | 0.457           | 0.134            | 0.335    | 0.314     | 0.148 | 0.022 | 0.356   | 0.348   |
| VisionZip    | 0.643                | 0.577 | 0.574 | 0.511                 | 0.577   | 0.502           | 0.207            | 0.595    | 0.613     | 0.242 | 0.107 | 0.349   | 0.458   |
| DivPrune     | 0.656                | 0.616 | 0.592 | 0.522                 | 0.562   | 0.483           | 0.210            | 0.533    | 0.551     | 0.230 | 0.077 | 0.382   | 0.451   |
| CDPruner     | 0.670                | 0.616 | 0.602 | 0.531                 | 0.560   | 0.506           | 0.206            | 0.551    | 0.575     | 0.217 | 0.064 | 0.368   | 0.455   |
| VScan        | 0.663                | 0.619 | 0.598 | 0.540                 | 0.584   | 0.516           | 0.224            | 0.610    | 0.624     | 0.244 | 0.101 | 0.370   | 0.474   |
| Cl:          | 9.2%                 | 9.5%  | 8.9%  | 9.8%                  | 10.7%   | 10.3%           | 8.4%             | 8.5%     | 8.7%      | 8.8%  | 9.3%  | 6.3%    | 9.0%    |
| GlimpsePrune | 0.690                | 0.666 | 0.654 | 0.526                 | 0.615   | 0.547           | 0.236            | 0.584    | 0.582     | 0.273 | 0.124 | 0.381   | 0.490   |
|              | Unlimited constraint |       |       |                       |         |                 |                  |          |           |       |       |         |         |
| Climana      | 22.9%                | 30.0% | 22.8% | 29.2%                 | 41.7%   | 18.1%           | 13.2%            | 14.8%    | 16.2%     | 16.1% | 10.9% | 8.8%    | 20.4%   |
| GlimpsePrune | 0.693                | 0.664 | 0.649 | 0.532                 | 0.606   | 0.551           | 0.247            | 0.605    | 0.621     | 0.272 | 0.119 | 0.384   | 0.495   |

Table 4 Performance in free-form VQA tasks using LLaVA-1.5-7B.

#### 72.8%.

Efficient RL fine-tuning. Our fine-tuning experiments on the Qwen2.5-VL-7B model (Tab. 8) demonstrate that our pruning method effectively addresses Out-of-Memory (OOM) errors encountered when training with token lengths over 3000. By applying this technique, we not only supported token length to 6000 but also reduced the average iteration time and GPU memory usage to 81% and 70% of the original, respectively. Crucially, this efficiency gain was achieved with negligible impact on performance (0.835 vs. 0.841 compared to the dense-token baseline).

## 6 Conclusions

This work demonstrates the failure of static token compression methods and introduces GlimpsePrune, a dynamic visual token pruning framework. By learning a data-driven pruning metric, GlimpsePrune prunes 92.6% of visual tokens while on average retaining 100% of the baseline performance. An enhanced version, GlimpsePrune+, further boosts performance to 110% using GRPO fine-tuning. Our method presents a practical and robust solution for building powerful and efficient LVLMs.

#### References

- <span id="page-6-0"></span>[1] Jieneng Chen, Luoxin Ye, Ju He, Zhao-Yang Wang, Daniel Khashabi, and Alan Yuille. Efficient large multi-modal models via visual context compression. *Advances in Neural Information Processing Systems*, 37:73986–74007, 2024. 1, 2
- <span id="page-6-1"></span>[2] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In *European Conference on Computer Vision*, pages 323–340. Springer, 2024. 1, 2
- <span id="page-6-2"></span>[3] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In *European Conference on Computer Vision*, pages 19–35. Springer, 2024. 1, 3
- <span id="page-6-3"></span>[4] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In *Proceedings* of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19792–19802, 2025. 1, 2, 5
- <span id="page-6-4"></span>[5] Ce Zhang, Kaixin Ma, Tianqing Fang, Wenhao Yu, Hong-ming Zhang, Zhisong Zhang, Yaqi Xie, Katia Sycara, Haitao Mi, and Dong Yu. Vscan: Rethinking visual token reduction for efficient large vision-language models. arXiv preprint arXiv:2505.22654, 2025. 1, 2, 3, 5

<span id="page-7-11"></span>

| Glimpse | $\mathcal{L}_{lang}$ | $\mathcal{L}_{loc}$ | VIP | Visual<br>Condition | General<br>VQA | Relation<br>Reasoning | Fine<br>Grained | Doc/Text | Chart | Average |
|---------|----------------------|---------------------|-----|---------------------|----------------|-----------------------|-----------------|----------|-------|---------|
|         | Qwe                  | n2.5-VL             | -3B |                     | 0.664          | 0.531                 | 0.748           | 0.892    | 0.816 | 0.746   |
| x       | Х                    | Х                   | Х   | Х                   | 0.585          | 0.515                 | 0.762           | 0.506    | 0.542 | 0.546   |
| ✓       | ✓                    | X                   | X   | ×                   | 0.647          | 0.537                 | 0.760           | 0.771    | 0.693 | 0.684   |
| ✓       | ✓                    | ✓                   | X   | ×                   | 0.641          | 0.521                 | 0.763           | 0.802    | 0.802 | 0.702   |
| ✓       | ✓                    | ✓                   | ✓   | ×                   | 0.639          | 0.528                 | 0.764           | 0.835    | 0.788 | 0.716   |
| ✓       | ✓                    | ✓                   | ✓   | ✓                   | 0.642          | 0.524                 | 0.760           | 0.882    | 0.804 | 0.736   |

**Table 5** Ablation study of training components and design choices in GlimpsePrune.

<span id="page-7-12"></span>

| Prune Layer (K) | Performance | Prefilling FLOPs(T) |
|-----------------|-------------|---------------------|
| Qwen2.5-VL-3B   | 0.746       | 33.8                |
| 18              | 0.690       | 17.4                |
| 24              | 0.736       | 23.1                |
| 30              | 0.739       | 28.5                |
| 36              | 0.556       | 34.1                |

**Table 6** The trade-off between performance and efficiency with different pruning layers.

<span id="page-7-13"></span>

| Qwen2.5<br>VL-7B | Glimpse<br>Prune                                         | Glimpse<br>Prune+                                                                                                                                                                                                              |
|------------------|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 77.8             | 53.8                                                     | 54.2                                                                                                                                                                                                                           |
| 423.3            | 306.9                                                    | 309.9                                                                                                                                                                                                                          |
| 5073.9           | 202.5                                                    | 291.1                                                                                                                                                                                                                          |
| 333.4            | 317.4                                                    | 256.6                                                                                                                                                                                                                          |
| 29.5             | 28.3                                                     | 28.4                                                                                                                                                                                                                           |
| 22.0             | 24.2                                                     | 19.5                                                                                                                                                                                                                           |
| 33.5             | 24.4                                                     | 23.7                                                                                                                                                                                                                           |
| 0.761            | 0.761                                                    | 0.835                                                                                                                                                                                                                          |
|                  | 77.8<br>423.3<br>5073.9<br>333.4<br>29.5<br>22.0<br>33.5 | VL-7B         Prune           77.8         53.8           423.3         306.9           5073.9         202.5           333.4         317.4           29.5         28.3           22.0         24.2           33.5         24.4 |

**Table 7** Efficiency analysis of GlimpsePrune on Qwen2.5-VL-7B model with 100 samples from DocVQA.

- <span id="page-7-0"></span>[6] Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In International Conference on Learning Representations, 2025.
- <span id="page-7-1"></span>[7] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024. 1, 2, 3, 5
- <span id="page-7-2"></span>[8] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International Conference on Machine Learning*, pages 19730–19742. PMLR, 2023.
- <span id="page-7-3"></span>[9] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36:34892–34916, 2023.

<span id="page-7-14"></span>

| Pruned | Time (s/iter) | Memory (GB)        | Performance |
|--------|---------------|--------------------|-------------|
| X      | 37.1          | $\geq 80 \times 4$ | 0.841       |
| ✓      | 29.9          | $\sim 56 \times 4$ | 0.835       |

**Table 8** GlimpsePrune reduces the cost of RL fine-tuning. Experiments are conducted on 4 NVIDIA A100 GPUs.

- <span id="page-7-4"></span>[10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of* the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2
- <span id="page-7-5"></span>[11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *Science China Information Sciences*, 67(12):220101, 2024. 2
- <span id="page-7-6"></span>[12] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. *arXiv preprint arXiv:2411.10442*, 2024. 2
- <span id="page-7-7"></span>[13] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. *arXiv preprint arXiv:2412.05271*, 2024. 2
- <span id="page-7-8"></span>[14] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. *arXiv* preprint arXiv:2504.10479, 2025. 2
- <span id="page-7-9"></span>[15] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 2
- <span id="page-7-10"></span>[16] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of

- the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024. [2](#page-1-2)
- <span id="page-8-0"></span>[17] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5 vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [2,](#page-1-2) [5,](#page-4-0) [12](#page-11-2)
- <span id="page-8-1"></span>[18] Kai Han, Jianyuan Guo, Yehui Tang, Wei He, Enhua Wu, and Yunhe Wang. Free video-llm: Prompt-guided visual perception for efficient training-free video llms. *arXiv preprint arXiv:2410.10441*, 2024. [2](#page-1-2)
- <span id="page-8-2"></span>[19] Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, and Liqiang Nie. Retake: Reducing temporal and knowledge redundancy for long video understanding. *arXiv preprint arXiv:2412.20504*, 2024. [2](#page-1-2)
- <span id="page-8-3"></span>[20] Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. *arXiv preprint arXiv:2506.10967*, 2025. [2,](#page-1-2) [3,](#page-2-2) [5](#page-4-0)
- <span id="page-8-4"></span>[21] Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. *arXiv preprint arXiv:2412.01818*, 2025. [2](#page-1-2)
- <span id="page-8-5"></span>[22] Mohamed Dhouib, Davide Buscaldi, Sonia Vanier, and Aymen Shabou. Pact: Pruning and clustering-based token reduction for faster visual language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14582–14592, 2025. [2](#page-1-2)
- <span id="page-8-6"></span>[23] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9392–9401, 2025. [2,](#page-1-2) [5](#page-4-0)
- <span id="page-8-7"></span>[24] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19803– 19813, 2025. [2](#page-1-2)
- <span id="page-8-8"></span>[25] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llavascissor: Token compression with semantic connected components for video llms. *arXiv preprint arXiv:2506.21862*, 2025. [2](#page-1-2)
- <span id="page-8-9"></span>[26] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and memory-efficient exact at- ´ tention with IO-awareness. In *Advances in Neural Information Processing Systems*, 2022. [2,](#page-1-2) [3](#page-2-2)
- <span id="page-8-10"></span>[27] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In *International Conference on Learning Representations*, 2024. [2,](#page-1-2) [3,](#page-2-2) [5](#page-4-0)

- <span id="page-8-11"></span>[28] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13700–13710, 2024. [2](#page-1-2)
- <span id="page-8-12"></span>[29] Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. Dynamic-LLaVA: Efficient multimodal large language models via dynamic vision-language context sparsification. In *International Conference on Learning Representations*, 2025. [2](#page-1-2)
- <span id="page-8-13"></span>[30] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18992– 19001, 2025. [2](#page-1-2)
- <span id="page-8-14"></span>[31] Xiaoyu Liang, Chaofeng Guan, Jiaying Lu, Huiyao Chen, Huan Wang, and Haoji Hu. Dynamic token reduction during generation for vision language models. *arXiv preprint arXiv:2501.14204*, 2025. [2](#page-1-2)
- <span id="page-8-15"></span>[32] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024. [4](#page-3-1)
- <span id="page-8-16"></span>[33] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024. [4,](#page-3-1) [11](#page-10-2)
- <span id="page-8-17"></span>[34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024. [5,](#page-4-0) [12](#page-11-2)
- <span id="page-8-18"></span>[35] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6700– 6709, 2019. [5,](#page-4-0) [11](#page-10-2)
- <span id="page-8-19"></span>[36] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. *Advances in Neural Information Processing Systems*, 37:8612– 8642, 2024. [5,](#page-4-0) [11](#page-10-2)
- <span id="page-8-20"></span>[37] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022. [5](#page-4-0)
- <span id="page-8-21"></span>[38] A Yang Qwen, Baosong Yang, B Zhang, B Hui, B Zheng, B Yu, Chengpeng Li, D Liu, F Huang, H Wei, et al. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*, 2024. [5](#page-4-0)
- <span id="page-8-22"></span>[39] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models. *arXiv preprint arXiv:2407.12772*, 2024. [5,](#page-4-0) [12](#page-11-2)
- <span id="page-8-23"></span>[40] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the

- role of image understanding in visual question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6904–6913, 2017. [11](#page-10-2)
- <span id="page-9-0"></span>[41] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3608–3617, 2018. [11](#page-10-2)
- <span id="page-9-1"></span>[42] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521, 2022. [11](#page-10-2)
- <span id="page-9-2"></span>[43] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. *arXiv preprint arXiv:2305.10355*, 2023. [11](#page-10-2)
- <span id="page-9-3"></span>[44] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv preprint arXiv:/2306.13394*, 2023. [11](#page-10-2)
- <span id="page-9-4"></span>[45] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In *European Conference on Computer Vision*, pages 216–233. Springer, 2024. [11](#page-10-2)
- <span id="page-9-5"></span>[46] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. *arXiv preprint arXiv:2307.16125*, 2023. [11](#page-10-2)
- <span id="page-9-6"></span>[47] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13084–13094, 2024. [11](#page-10-2)

#### <span id="page-10-2"></span><span id="page-10-1"></span>A Dataset Details

#### A.1 Free-form VQA datasets

The free-form VQA tasks are evaluated on the VisCoT benchmark [36], which consists of 12 datasets. Tab. 9 lists the number of samples and the average ratio of target objects in each dataset.

<span id="page-10-3"></span>

| Dataset    | #Samples | Ratio | Dataset   | #Samples | Ratio |
|------------|----------|-------|-----------|----------|-------|
| Flickr30k  | 1546     | 18.2% | DocVQA    | 888      | 6.8%  |
| V7W        | 1000     | 22.8% | TextCaps  | 853      | 4.9%  |
| GQA        | 978      | 20.2% | TextVQA   | 526      | 5.6%  |
| OpenImages | 945      | 24.3% | DUDE      | 603      | 1.7%  |
| VSR        | 404      | 41.6% | SROIE     | 686      | 10.1% |
| CUB        | 492      | 42.5% | In fo VQA | 360      | 12.3% |

**Table 9** Datasets used for free-form VQA evaluation. "Ratio" refers to the average ratio of target objects in the images.

#### A.2 Short-form VQA datasets

Tab. 10 lists the number of samples and answer types for the datasets used in short-form VQA evaluation, including VQAv2 [40], GQA [35], VizWiz [41], ScienceQA [42], POPE [43], MME [44], MMBench [45], SEEDBench [46], and V\* bench [47].

<span id="page-10-4"></span>

| Dataset                              | #Samples | Type      | Dataset             | #Samples | Type      |
|--------------------------------------|----------|-----------|---------------------|----------|-----------|
| VQA <sub>vallite</sub> <sup>v2</sup> | 500      | Phrase    | MME                 | 2374     | Phrase    |
| GQA                                  | 12578    | Phrase    | MMB <sub>test</sub> | 6666     | Selection |
| $VizWiz_{val}$                       | 4319     | Phrase    | $MMB_{test}^{en}$   | 6666     | Selection |
| $SQA^{img}$                          | 2017     | Selection | SEEDimg             | 17990    | Selection |
| POPE                                 | 9000     | Judgment  | V*                  | 191      | Selection |

**Table 10** Datasets used for short-form VQA evaluation. The "Type" column indicates the type of answers in the dataset. "Phrase" means the answer is a word or phrase, "Selection" means the answer is selected from a set of options, and "Judgment" means the answer is a judgment (e.g., yes/no).

### **A.3** Datasets for training GlimpsePrune

We train GlimpsePrune using 20,000 randomly sampled samples from the GQA dataset [35]. Each sample contains an image, a question, a short answer, and several bounding boxes of visual targets. We find that GlimpsePrune generalizes well to various VQA datasets after training on GQA, allowing our method to operate without the need for further expansion of the training dataset.

#### A.4 Datasets for RL fine-tuning

Since fine-tuning datasets for LLMs often require domain richness, we use the training sets of 12 datasets from Vis-CoT [36] for fine-tuning the LLM. Specifically, we randomly

sample 20,000 samples from each dataset, with 10,000 samples constructed as short-form VQA tasks and 10,000 samples as free-form VQA tasks. Due to the high image resolution or long answers in some data, which can easily lead to out-of-memory error during training, we limit the number of input tokens to 6000, and only compute the GRPO loss [33] for the first 128 generated tokens.

## <span id="page-10-0"></span>**B** Implementation Details

### **B.1** Training details.

Tab. 11 lists the hyperparameters used for training GlimpsePrune, and Tab. 12 lists the hyperparameters used for RL fine-tuning.

<span id="page-10-5"></span>

| Key            | Value  | Key                                       | Value    |
|----------------|--------|-------------------------------------------|----------|
| batch / device | 1      | dtype                                     | bfloat16 |
| grad acc steps | 8      | max grad norm                             | 1.0      |
| num gpus       | 2      | train data size                           | 20,000   |
| learning rate  | 1e-4   | num train epochs                          | 1        |
| warmup ratio   | 0.1    | weight of $\mathcal{L}_{\text{language}}$ | 1.0      |
| optimizer      | AdamW  | weight of $\mathcal{L}_{\text{dice}}$     | 1.0      |
| scheduler      | cosine | weight of $\mathcal{L}_{bce}$             | 0.1      |

**Table 11** Training arguments for GlimpsePrune.

<span id="page-10-6"></span>

| Key             | Value                          | Key                                   | Value    |  |  |  |
|-----------------|--------------------------------|---------------------------------------|----------|--|--|--|
| batch / device  | 1                              | optimizer                             | AdamW    |  |  |  |
| grad acc steps  | 8                              | scheduler                             | cosine   |  |  |  |
| num gpus        | 4                              | warmup ratio                          | 0.1      |  |  |  |
| num generations | 4                              | dtype                                 | bfloat16 |  |  |  |
| num iterations  | 1                              | max grad norm                         | 1.0      |  |  |  |
| lora rank       | 32                             | train data size                       | 240,000  |  |  |  |
| lora alpha      | 64                             | num train epochs                      | 1        |  |  |  |
| lora dropout    | 0.05                           | weight of $\mathcal{L}_{policy}$      | 1.0      |  |  |  |
| learning rate   | 1e-4                           | weight of $\mathcal{L}_{\mathrm{KL}}$ | 0.04     |  |  |  |
| reward model    | Qwen2.5-32B-Instruct-GPTQ-Int8 |                                       |          |  |  |  |

Table 12 Training arguments for RL fine-tuning.

<span id="page-10-7"></span>

| VQA            | LVLM            | Evaluator LLM      |      |  |  |  |
|----------------|-----------------|--------------------|------|--|--|--|
| temperature    | 0.0             | temperature        | 0.7  |  |  |  |
| dtype          | bfloat16        | top p              | 0.8  |  |  |  |
| use cache      | True            | repetition penalty | 1.05 |  |  |  |
| max new tokens | 1024            | max new tokens     | 512  |  |  |  |
| attn impl      | FlashAttention2 |                    |      |  |  |  |

Table 13 Evaluation details for free-form VQA tasks.

#### **B.2** Evaluation details.

For free-form VQA tasks, we use Qwen2.5-32B-Instruct-GPTQ-Int8 as the evaluator LLM. The evaluation template is

<span id="page-11-3"></span><span id="page-11-2"></span>You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.

```
question: {}
standard answer: {}
model's answer: {}
```

Figure 5 Input template for the evaluator LLM.

<span id="page-11-4"></span>

| Architecture                | Qwen2.5VL-3B    | Qwen2.5VL-7B    | LLaVA-1.5-7B    | LLaVA-1.5-13B   |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|
| Num visual tokens           | 4∼16384         | 4∼16384         | 576             | 576             |
| Num visual layers           | 32              | 32              | 24              | 24              |
| Visual feature size (C)     | 1280            | 1280            | 1024            | 1024            |
| Num LLM layers (L)          | 36              | 28              | 32              | 40              |
| Num LLM attention heads (H) | 16              | 28              | 32              | 40              |
| LLM hidden size (D)         | 2048            | 3584            | 4096            | 5120            |
| Selected visual layers      | [31, 23, 15, 7] | [31, 23, 15, 7] | [23, 17, 11, 5] | [23, 17, 11, 5] |
| Glimpse token shape         | 36 × 2048       | 28 × 3584       | 32 × 4096       | 40 × 5120       |
| Prune Layer (K)             | 24              | 19              | 22              | 27              |
| VIP hidden size (E)         | 256             | 256             | 256             | 256             |
| VIP condition size (F)      | 512             | 512             | 512             | 512             |
| VIP attention heads         | 4               | 4               | 4               | 4               |

Table 14 Details of the model architectures.

shown in Fig. [5.](#page-11-3) The evaluation settings for the LVLMs and the evaluator LLM are shown in Tab. [13.](#page-10-7)

<span id="page-11-0"></span>For short-form VQA tasks, we use the default configuration of the LMMs-Eval framework [\[39\]](#page-8-22).

## C Model Architecture Details

As demonstrated in the main paper, we validated the effectiveness of our method, GlimpsePrune, on several baseline models, including Qwen2.5-VL [\[17\]](#page-8-0) and LLaVA-1.5 [\[34\]](#page-8-17). The detailed architectural parameters for these models are presented in Tab. [14.](#page-11-4)

<span id="page-11-1"></span>Our approach introduces two primary new components: learnable glimpse tokens and a VIP module composed of M selfattention blocks. Specifically, the glimpse tokens are integrated into every layer of the LLM decoder. However, during inference, only the tokens in the first K layers are utilized. The position of pruning, K, is determined based on our ablation studies (Sec. [5.3\)](#page-4-1), following the rule K = ⌈2/3 × L⌉, where L is the total number of decoder layers. Furthermore, the VIP module is conditioned on M = 4 visual features that are uniformly sampled from the visual encoder.

## D Additional Results and Analysis

## D.1 Results on other architectures.

In addition to the results presented in Tab. [2](#page-5-0) and Tab. [4](#page-6-6) of the main paper, Tab. [15](#page-12-0) and Tab. [16](#page-12-1) present the performance of GlimpsePrune on the free-form VQA task, based on the Qwen2.5-VL-3B and LLaVA-1.5-13B models, respectively. Compared to previous methods, our approach demonstrates a superior ability to dynamically adjust the retention rate while achieving better performance. This highlights the adaptability and efficiency of GlimpsePrune across different model scales and architectures.

## D.2 Case studies.

Although Tab. [2](#page-5-0) shows that GlimpsePrune (based on Qwen2.5- VL-7B) matches the baseline's average score with only a 7.4% retention rate, its performance varies across different cases. Here, we present both successful and unsuccessful examples to facilitate future research. Fig. [6,](#page-12-2) Fig. [7](#page-13-0) and Fig. [8](#page-13-1) display successful cases where GlimpsePrune answers more accurately because it prunes question-irrelevant visual information. In contrast, Fig. [9](#page-14-0) and Fig. [10](#page-14-1) show two types of failures. In the first type (Fig. [9\)](#page-14-0), the retained visual information is insufficient. In the second type (Fig. [10\)](#page-14-1), the information is sufficient, but the model still answers incorrectly. We hope these examples provide valuable insights for future studies.

<span id="page-12-0"></span>

|                                |       | General<br>Relation<br>Fine<br>Doc/Text<br>VQA<br>Reasoning<br>grained |             |                                  |       | Chart       |       |       |                                                    |       |       |       |       |
|--------------------------------|-------|------------------------------------------------------------------------|-------------|----------------------------------|-------|-------------|-------|-------|----------------------------------------------------|-------|-------|-------|-------|
| Method                         |       |                                                                        |             | Flickr30k V7W GQA OpenImages VSR |       | CUB         |       |       | DocVQA TextCaps TextVQA DUDE SROIE InfoVQA Average |       |       |       |       |
| Qwen2.5VL-3B                   | 0.703 |                                                                        | 0.626 0.584 | 0.438                            | 0.573 | 0.748       | 0.944 | 0.840 | 0.927                                              | 0.793 | 0.957 | 0.816 | 0.746 |
| Average retention rate ≤ 11.1% |       |                                                                        |             |                                  |       |             |       |       |                                                    |       |       |       |       |
| PDrop                          | 0.401 |                                                                        | 0.298 0.267 | 0.378                            | 0.222 | 0.502       | 0.254 | 0.428 | 0.463                                              | 0.255 | 0.125 | 0.320 | 0.326 |
| VisionZip                      | 0.546 |                                                                        | 0.478 0.465 | oom                              | 0.426 | 0.713       | oom   | 0.600 | 0.603                                              | oom   | oom   | oom   | –     |
| VScan                          | 0.492 |                                                                        | 0.444 0.402 | oom                              | 0.367 | 0.717       | oom   | 0.564 | 0.618                                              | oom   | oom   | oom   | –     |
|                                | 8.9%  | 8.9%                                                                   | 8.8%        | 9.2%                             |       | 10.6% 10.4% | 5.2%  | 7.6%  | 7.7%                                               | 7.1%  | 7.3%  | 6.9%  | 8.2%  |
| GlimpsePrune                   | 0.675 |                                                                        | 0.609 0.567 | 0.444                            | 0.561 | 0.760       | 0.934 | 0.830 | 0.907                                              | 0.774 | 0.964 | 0.804 | 0.736 |
| Unlimited constraint           |       |                                                                        |             |                                  |       |             |       |       |                                                    |       |       |       |       |
| GlimpsePrune                   | 25.4% |                                                                        | 31.8% 25.9% | 27.0%                            |       | 44.1% 23.0% | 4.9%  | 11.7% | 13.3%                                              | 9.3%  | 6.5%  | 7.3%  | 19.2% |
|                                | 0.695 |                                                                        | 0.624 0.578 | 0.437                            | 0.560 | 0.767       | 0.931 | 0.828 | 0.916                                              | 0.776 | 0.965 | 0.813 | 0.741 |

Table 15 Performance in free-form VQA tasks using Qwen2.5VL-3B.

<span id="page-12-1"></span>

|                      | General<br>VQA |      | Relation<br>Reasoning |                                  | Fine<br>grained |             | Doc/Text                       |       |                                                    |       | Chart       |       |       |
|----------------------|----------------|------|-----------------------|----------------------------------|-----------------|-------------|--------------------------------|-------|----------------------------------------------------|-------|-------------|-------|-------|
| Method               |                |      |                       | Flickr30k V7W GQA OpenImages VSR |                 | CUB         |                                |       | DocVQA TextCaps TextVQA DUDE SROIE InfoVQA Average |       |             |       |       |
| LLaVA-1.5-13B        | 0.719          |      | 0.674 0.656           | 0.533                            | 0.643           | 0.608       | 0.273                          | 0.660 | 0.674                                              | 0.301 | 0.153       | 0.407 | 0.525 |
|                      |                |      |                       |                                  |                 |             | Average retention rate ≤ 11.1% |       |                                                    |       |             |       |       |
| PDrop                | 0.604          |      | 0.581 0.564           | 0.556                            | 0.551           | 0.539       | 0.163                          | 0.477 | 0.440                                              | 0.173 | 0.032       | 0.402 | 0.423 |
| VisionZip            | 0.662          |      | 0.593 0.581           | 0.517                            | 0.580           | 0.602       | 0.240                          | 0.622 | 0.633                                              | 0.243 | 0.109       | 0.387 | 0.481 |
| DivPrune             | 0.662          |      | 0.620 0.601           | 0.513                            | 0.582           | 0.562       | 0.219                          | 0.543 | 0.586                                              | 0.236 | 0.071       | 0.394 | 0.466 |
| CDPruner             | 0.682          |      | 0.639 0.609           | 0.514                            | 0.585           | 0.630       | 0.218                          | 0.583 | 0.602                                              | 0.244 | 0.069       | 0.381 | 0.480 |
| VScan                | 0.673          |      | 0.625 0.620           | 0.530                            | 0.600           | 0.604       | 0.247                          | 0.625 | 0.643                                              | 0.267 | 0.111       | 0.397 | 0.495 |
|                      | 9.5%           | 9.5% | 9.0%                  | 10.0%                            | 10.6%           | 9.4%        | 10.2%                          | 9.3%  | 9.4%                                               |       | 10.6% 10.0% | 9.7%  | 9.8%  |
| GlimpsePrune         | 0.677          |      | 0.691 0.673           | 0.523                            | 0.623           | 0.595       | 0.273                          | 0.631 | 0.655                                              | 0.283 | 0.148       | 0.418 | 0.516 |
| Unlimited constraint |                |      |                       |                                  |                 |             |                                |       |                                                    |       |             |       |       |
| GlimpsePrune         | 27.0%          |      | 32.8% 23.8%           | 31.9%                            |                 | 42.4% 14.3% | 21.8%                          | 17.2% | 17.4%                                              |       | 28.9% 17.4% | 26.0% | 25.1% |
|                      | 0.687          |      | 0.684 0.673           | 0.521                            | 0.625           | 0.605       | 0.273                          | 0.636 | 0.663                                              | 0.302 | 0.156       | 0.423 | 0.521 |

Table 16 Performance in free-form VQA tasks using LLaVA-1.5-13B.

<span id="page-12-2"></span>![](_page_12_Picture_4.jpeg)

Question: What is in the sky? Standard answer: Kites

Qwen2.5-VL-7B: In the sky, there are several kites. The kites appear to be of different shapes and colors, and they are flying at varying heights. The sky is clear, suggesting a good day for kite flying.

Score: 0.8

![](_page_12_Picture_8.jpeg)

Retention ratio: 4.5%

GlimpsePrune: In the sky, there are kites. There are at least three kites visible, each with a distinct shape and color.

Score: 1.0

Figure 6 Case 1: With only the relevant visual tokens retained, GlimpsePrune can focus more on the target object rather than other irrelevant information while answering the question correctly.

<span id="page-13-0"></span>![](_page_13_Picture_0.jpeg)

Question: What is on top of the building?

Standard answer: A clock.

Qwen2.5-VL-7B: The image shows a building with a statue on top of it. The statue appears to be of a person, possibly a historical figure, given its prominent placement and the formal setting. The building itself has a classical architectural style, with columns and a pediment, which is typical of neoclassical architecture.

Score: 0.0

![](_page_13_Picture_5.jpeg)

Retention ratio: 1.5%

GlimpsePrune: The image shows a building with a triangular pediment on top. On the pediment, there is a clock. The clock appears to be a decorative element, likely used to tell time for the building's occupants or visitors. The pediment itself is a classical architectural feature.

Score: 0.7

Figure 7 Case 2: By retaining the visual tokens at the correct position, GlimpsePrune can accurately answer that there is a clock on top of the building and provide relevant descriptions.

<span id="page-13-1"></span>![](_page_13_Figure_10.jpeg)

Question: In 1960, what is the population of females in the age group 0-4 (millions)?

Standard answer: 10

Qwen2.5-VL-7B: In 1960, the population of females in the age group 0-4 is approxi-

mately 5 million.

Score: 0.5

![](_page_13_Figure_16.jpeg)

Retention ratio: 3.6%

GlimpsePrune: In 1960, the population of females in the age group 0-4 is approximately 10 million.

Score: 1.0

Figure 8 Case 3: With the correct visual tokens retained, GlimpsePrune can answer the question related to the chart.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Question: What has been the greatest concern for non-smokers in 1990?

Standard answer: Getting enough exercise

Qwen2.5-VL-7B: The greatest concern for non-smokers in 1990, according to the table, is "Getting enough exercise," with 45% of non-smokers being concerned about it.

Score: 1.0

![](_page_14_Figure_5.jpeg)

Retention ratio: 3.7%

GlimpsePrune: The greatest concern for non-smokers in 1990, according to the chart, is "Maintaining right weight," with 43% of non-smokers indicating this as their primary concern.

Score: 0.0

Figure 9 Failure case 1: The retained visual tokens is insufficient to answer the question correctly.

<span id="page-14-1"></span>![](_page_14_Figure_10.jpeg)

Question: How many categories of income mentioned in this infographic?

Standard answer: 6

Qwen2.5-VL-7B: The infographic mentions five categories of income:

- 1. Under \$25k
- 2. \$25-\$35k
- 3. \$35-\$50k
- 4. \$50-\$75k
- 5. Over \$75-\$100k
- 6. Over \$100k

Score: 1.0

![](_page_14_Figure_21.jpeg)

Retention ratio: 4.4%

GlimpsePrune: The infographic mentions five categories of income:

- 1. Under \$25k
- 2. \$25-\$35k
- 3. \$35-\$50k
- 4. \$50-\$75k
- 5. Over \$100k

Score: 0.8

Figure 10 Failure case 2: The retained visual tokens is sufficient but the model fails to answer the question correctly.