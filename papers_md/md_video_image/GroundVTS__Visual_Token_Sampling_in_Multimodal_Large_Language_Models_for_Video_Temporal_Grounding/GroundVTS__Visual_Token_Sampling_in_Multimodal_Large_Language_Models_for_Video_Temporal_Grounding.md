# <span id="page-0-0"></span>GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding

Rong Fan<sup>1</sup>,2\* Kaiyan Xiao<sup>3</sup>\* Minghao Zhu<sup>3</sup> Liuyi Wang<sup>3</sup> Kai Dai<sup>1</sup> Zhao Yang<sup>1</sup>† <sup>1</sup>Newcapec AI Research <sup>2</sup>Fudan University <sup>3</sup>Tongji University

rfan24@m.fudan.edu.cn {xiaokaiyan,zmhh h,wly}@tongji.edu.cn {daikai,yangzhao}@newcapec.net

# Abstract

*Video temporal grounding (VTG) is a critical task in video understanding and a key capability for extending video large language models (Vid-LLMs) to broader applications. However, existing Vid-LLMs rely on uniform frame sampling to extract video information, resulting in a sparse distribution of key frames and the loss of crucial temporal cues. To address this limitation, we propose Grounded Visual Token Sampling (GroundVTS), a Vid-LLM architecture that focuses on the most informative temporal segments. Ground-VTS employs a fine-grained, query-guided mechanism to filter visual tokens before feeding them into the LLM, thereby preserving essential spatio-temporal information and maintaining temporal coherence. Futhermore, we introduce a progressive optimization strategy that enables the LLM to effectively adapt to the non-uniform distribution of visual features, enhancing its ability to model temporal dependencies and achieve precise video localization. We comprehensively evaluate GroundVTS on three standard VTG benchmarks, where it outperforms existing methods, achieving a 7.7-point improvement in mIoU for moment retrieval and 12.0-point improvement in mAP for highlight detection. Code is available at the [GroundVTS code repository.](https://github.com/Florence365/GroundVTS)*

# 1. Introduction

Understanding complex video content is fundamental to a wide range of applications, including video classification [\[4,](#page-8-0) [11,](#page-8-1) [13,](#page-8-2) [67\]](#page-10-0), video captioning [\[6,](#page-8-3) [26,](#page-9-0) [68\]](#page-10-1), video question-answering (VQA) [\[56,](#page-10-2) [60,](#page-10-3) [61\]](#page-10-4), and video temporal grounding (VTG) [\[9,](#page-8-4) [20,](#page-8-5) [21\]](#page-8-6). Among these applications, VTG is a representative task for fine-grained temporal understanding: it aims to precisely identify the video segment corresponding to a given natural-language query [\[9\]](#page-8-4).

Driven by the success of large-scale multimodal pretraining, numerous vision language models have achieved remarkable progress in multimodal reasoning [\[10,](#page-8-7) [42,](#page-9-1) [44,](#page-9-2) [47,](#page-9-3) [69\]](#page-10-5). Meanwhile, rapid advances in large language models (LLMs) have opened new opportunities to incorporate highlevel reasoning and instruction following into video understanding. When combined with visual encoders, LLMs provide a unified multimodal framework for interpreting complex temporal events and cross-modal relationships [\[46\]](#page-9-4). Building on this trend, many studies have developed video LLMs (Vid-LLMs) [\[24,](#page-9-5) [29,](#page-9-6) [35\]](#page-9-7), achieving notable progress in general video reasoning. However, current Vid-LLMs still struggle with granular temporal understanding.

Prior efforts to enhance Vid-LLMs for temporal grounding have introduced query-conditioned attention [\[62\]](#page-10-6), temporal boundary regressors [\[33\]](#page-9-8), and temporal modeling modules [\[3\]](#page-8-8). Yet these approaches often overlook the sampling and representation of visual tokens before entering the LLM, which can be important for precise temporal reasoning. As shown in Figure [1\(a\),](#page-1-0) most existing approaches adopt uniform frame sampling, a dominant strategy in video understanding tasks. While this strategy provides consistent temporal coverage, it allocates the input budget evenly across time; consequently, key moments can be diluted or even missed, especially when query-relevant events are sparse. Recently, several studies [\[28,](#page-9-9) [50,](#page-10-7) [53\]](#page-10-8) introduce query-guided frame sampling at the video input stage by attaching an external multimodal encoder (*e.g.*, CLIP [\[42\]](#page-9-1) or a captioning model) to compute cross-modal similarity for frame selection (Figure [1\(b\)\)](#page-1-0). However, they perform coarse-grained filtering and depend on auxiliary encoders, limiting localization precision and adaptability for VTG.

To address these limitations, we propose GroundVTS, a Vid-LLM architecture that performs query-guided visual token sampling at a finer granularity. As illustrated in Figure [1\(c\),](#page-1-0) GroundVTS introduces a Visual Token Sampling (VTS) module that operates after the visual encoder and multimodal projection layers. Unlike the coarse selection used in prior work, VTS selectively retains visual tokens based on token-level similarity to the textual query, yielding a non-uniform token distribution across both temporal and spatial dimensions. This helps the Vid-LLM focus on

<sup>\*</sup>Equal contribution.

<sup>†</sup>Corresponding author.

<span id="page-1-1"></span><span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1. Comparison of sampling strategies for Vid-LLMs. (a) Uniform sampling distributes attention evenly across frames, often missing query-relevant moments; (b) existing query-guided frame sampling relies on external encoders for coarse frame selection, limiting temporal understanding; and (c) our query-guided VTS adaptively selects query-relevant visual tokens within Vid-LLMs while maintaining temporal coherence, enabling efficient and precise video grounding.

crucial cues while suppressing irrelevant visual content, enabling more precise and robust temporal grounding.

Our main contributions are summarized as follows:

- We propose GroundVTS, a Vid-LLM framework with a query-guided VTS strategy that adaptively preserves critical spatio-temporal cues to sharpen temporal localization for VTG.
- We introduce a progressive optimization strategy that integrates the VTS module into existing Vid-LLM architectures, enabling effective adaptation to non-uniform visual token distributions while ensuring stable training.
- We conduct extensive experiments across multiple VTG benchmarks under varying visual token densities and model architectures, demonstrating the effectiveness and generality of GroundVTS.

# 2. Related Work

Video large language models (Vid-LLMs) are typically implemented by connecting visual encoders to LLMs through projection layers or multimodal adapters. Recent studies have focused on enhancing the global video understanding capabilities [\[24,](#page-9-5) [29,](#page-9-6) [35,](#page-9-7) [51,](#page-10-9) [53\]](#page-10-8). While these models achieve strong performance, they still struggle to capture granular temporal structures [\[32\]](#page-9-10). GroundVTS instead emphasizes fine-grained temporal perception, thereby improving the ability of Vid-LLMs to handle challenges of VTG. Video temporal grounding (VTG) is a fundamental video understanding task that aims to accurately localize the timestamps of events within a video [\[30\]](#page-9-11). Classical expert models formulate this as a cross-modal matching problem, employing proposal-based or regression-based architectures to improve grounding precision [\[15,](#page-8-9) [31,](#page-9-12) [58,](#page-10-10) [63\]](#page-10-11). These models are typically task-specific and require datasetlevel fine-tuning [\[49\]](#page-10-12). Recent studies [\[7,](#page-8-10) [12,](#page-8-11) [16,](#page-8-12) [17,](#page-8-13) [39,](#page-9-13) [55\]](#page-10-13) integrate VTG capabilities into Vid-LLMs, leveraging their strong reasoning and generalization strengths to unify diverse video understanding tasks within a single framework. To support VTG within this unified setting, many methods introduce explicit time representations (*e.g.*, time tokens/ position embeddings) or frame indexing to better model timestamps [\[12,](#page-8-11) [55\]](#page-10-13). However, these approaches primarily focus on temporal encoding or data adaptation; the sampling and representation of visual tokens before entering the LLM have received comparatively less attention. Ground-VTS addresses this gap by introducing a unified Vid-LLM framework with an effective token sampling mechanism that substantially improves temporal understanding.

Visual token compression and selection. In VTG and VQA, not all visual tokens contribute equally to answering a given query, and many contain redundant or irrelevant information that can distract the model from the relevant evidence [\[38\]](#page-9-14). This has motivated growing interest in token compression, pruning, and adaptive selection for Vid-LLMs [\[1,](#page-8-14) [18,](#page-8-15) [45\]](#page-9-15). However, most existing approaches rely on query-agnostic or saliency-based token reduction, often preserving visually prominent yet semantically irrelevant tokens, which can limit temporal reasoning and grounding precision. Recent token compression methods further deliver strong efficiency gains and improve scalability for long-video understanding [\[27,](#page-9-16) [34,](#page-9-17) [66\]](#page-10-14). Their impli-

<span id="page-2-4"></span><span id="page-2-3"></span>![](_page_2_Figure_0.jpeg)

Figure 2. Frame rate sensitivity of Qwen2.5VL-7B. Similar trends hold for InternVL3.5 (illustrated in the supplementary material).

cations for VTG—where fine-grained temporal localization is central—remain less explored. In contrast, our Ground-VTS dynamically samples query-relevant tokens within the Vid-LLM via token–query relevance, preserving spatio-temporal cues without external preprocessing.

#### 3. Method

We propose GroundVTS, a Vid-LLM architecture designed to enhance VTG performance through adaptive and efficient visual token utilization. Sec. 3.1 analyzes the sensitivity of VTG performance to frame density, which highlights the necessity of adaptive token sampling, motivating our design. Sec. 3.2 presents the overall architecture of Ground-VTS, followed by a detailed explanation of the Visual Token Sampling (VTS) module in Sec. 3.3. Finally, Sec. 3.4 outlines our progressive optimization strategy, which allows VTS to be seamlessly integrated into existing Vid-LLMs.

#### <span id="page-2-0"></span>3.1. Analysis of Frame Rate Sensitivity

Before introducing our sampling strategy, we first examine how the density of visual tokens inherently influences VTG performance. To this end, we evaluate the pretrained Qwen2.5VL-7B [3] model on the Charades-STA [9] dataset under varying frame rates from 0.2 to 3.0 frames per second (FPS), while keeping all other settings fixed.

As shown in Figure 2, VTG performance demonstrates a clear non-linear dependency on frame rate. When the frame rate is low (<1.0 FPS), the model lacks sufficient temporal cues, resulting in degraded mIoU. As the sampling rate increases, performance improves steadily and reaches its peak around 2.0–2.4 FPS, where mIoU attains 47.8%. However, further increasing the frame rate beyond this range leads to a sharp performance drop, indicating that redundant visual tokens dilute key temporal signals and hinder accuracy.

This observation supports our core hypothesis that the density and relevance of visual tokens critically influence VTG performance. Consequently, an adaptive token sampling mechanism is essential for achieving both accuracy and efficiency in temporal grounding, motivating the design of our GroundVTS framework.

#### <span id="page-2-1"></span>3.2. GroundVTS

The GroundVTS framework, illustrated in Figure 3(a), incorporates a query-guided visual token sampling module to enable efficient and fine-grained temporal grounding. Unlike prior methods that rely on additional preprocessing [53], GroundVTS integrates the sampling process directly into the Vid-LLM pipeline, allowing the model to dynamically focus on temporally and semantically relevant segments conditioned on the input query.

Specifically, given an input video after standard temporal downsampling,  $\mathcal{V} = \{F_t\}_{t=1}^T$ , and a text query, the query is first tokenized into a sequence of embeddings,  $Q = [q_1, q_2, \dots, q_{N_t}] \in \mathbb{R}^{N_t \times D}$ , using a pretrained language tokenizer. Here, T denotes the number of frames,  $N_t$  is the number of text tokens, and D is the text token embedding dimension. Each video frame  $F_t$  is divided into spatial patches and encoded by a pretrained vision encoder to produce dense spatio-temporal features,  $H_v = [h_1, h_2, \dots, h_{N_v}] \in \mathbb{R}^{N_v \times D_v}$ , where  $N_v$  denotes the total number of visual tokens and  $D_v$  denotes the feature dimension. Then, a multimodal projector (implemented as a multilayer perceptron) maps these features to a shared embedding space:  $V = [v_1, v_2, \dots, v_{N_n}] \in \mathbb{R}^{N_v \times D}$ , where each  $\boldsymbol{v_i} \in \mathbb{R}^D$  is the projected embedding of the i-th visual token, ensuring that visual embeddings are dimensionally aligned with the text token embeddings Q.

Conditioned on the query embeddings Q, the VTS module (which will be introduced in Sec. 3.3) evaluates the relevance of each visual token to the query and outputs a compact subset of visual token embeddings via a weighted differentiable top-K selection mechanism:

$$\widetilde{V} = VTS(V, Q) = [\widetilde{\boldsymbol{v}}_1, \widetilde{\boldsymbol{v}}_2, \dots, \widetilde{\boldsymbol{v}}_K],$$
 (1)

where K is the number of tokens selected by VTS. The resulting  $\widetilde{V}$  forms a non-uniform, query-guided distribution of visual tokens that allocates denser sampling around relevant moments and sparser coverage elsewhere.

To preserve temporal coherence under non-uniform sampling, we reuse the original positional encodings from dense sampling (masking out only those of unselected visual tokens), ensuring that the selected tokens remain temporally aligned with the input video.

Finally, the sampled visual tokens and query embeddings are concatenated to form a multimodal sequence and processed by the LLM for joint reasoning and generation.

#### <span id="page-2-2"></span>3.3. Visual Token Sampling

The VTS module is the core of GroundVTS, responsible for dynamically selecting the most informative visual tokens under the guidance of the textual query. As illustrated in Figure 3(b), VTS consists of the following two main steps.

**Query-Guided Token Scoring.** Given the projected visual embeddings,  $V = \{v_i\}_{i=1}^{N_v}$ , and the tokenized query

<span id="page-3-1"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3. **Overview of the proposed GroundVTS framework.** (a) GroundVTS integrates a query-guided VTS module into the Vid-LLM pipeline, enabling adaptive selection of query-relevant tokens; (b) The VTS module computes token-query similarity scores and performs weighted differentiable top-K sampling to retain the most informative tokens, supporting efficient and precise video temporal grounding.

embeddings,  $Q = \{q_j\}_{j=1}^{N_t}$ , to estimate the relevance between each visual token and the query, both sets of embeddings are first projected into lower-dimensional subspaces of dimension  $D_r$  through linear projections:

$$V' = W_v V, \ \boldsymbol{q'} = W_q \operatorname{Pool}(Q), \ W_v, W_q \in \mathbb{R}^{D \times D_r}.$$
 (2)

Here,  $W_v$  and  $W_q$  are trainable projection matrices, and  $Pool(\cdot)$  denotes mean pooling over text token embeddings.

Then, the relevance distribution is obtained through a softmax function applied to the temperature-scaled dot products between each pair of embeddings in V' and q':

$$\boldsymbol{w} = \operatorname{softmax} \left( V' \boldsymbol{q'}^{\top} / \tau \right),$$
 (3)

where  $\tau$  is a temperature hyperparameter controlling the sharpness of the distribution. This formulation can be interpreted as an attention mechanism, where the weight assigned to each visual token reflects both its alignment with the query and its relative importance within the sequence. The resulting weight vector,  $\boldsymbol{w} = [w_1, w_2, \dots, w_{N_v}]^{\top}$ , effectively emphasizes semantically relevant visual tokens while attenuating less informative ones.

**Differentiable Top-**K **Selection.** To efficiently retain the most relevant visual information, we select the top-K visual tokens based on w. The number of selected tokens is adaptively determined by the ratio  $\rho \in (0,1]$ , where  $K = \lceil \rho \cdot N_v \rceil$ . Let  $\mathcal{I}_K$  denote the indices of the top-K tokens.

Since hard top-K selection is non-differentiable, we employ a Straight-Through Estimator (STE) with Gumbel-Softmax relaxation [19] to enable end-to-end training. Specifically, we add Gumbel noise  $g_i \sim \text{Gumbel}(0,1)$  to the log-probabilities and compute a differentiable approximation of the discrete selection mask:

$$z_{i} = \frac{\exp\left((\log w_{i} + g_{i})/\tau_{g}\right)}{\sum_{j=1}^{N_{v}} \exp\left((\log w_{j} + g_{j})/\tau_{g}\right)},$$
(4)

where  $\tau_g$  is a Gumbel temperature controlling the smoothness of the relaxation. During the forward pass, a hard top-K operator is applied:

$$z_i^{\text{hard}} = \begin{cases} 1, & \text{if } i \in \mathcal{I}_K, \\ 0, & \text{otherwise,} \end{cases}$$
 (5)

while the backward propagation flows through the continuous relaxation  $z_i$ . To this end, the STE is implemented as the following, combining the hard and soft representations:

$$\tilde{z}_i = z_i^{\text{hard}} + z_i - \text{stopgrad}(z_i),$$
 (6)

where  $stopgrad(\cdot)$  denotes gradient detachment.

The final visual representations are then obtained via weighted differentiable top-K masking: the relevance weights of the selected top-K tokens are re-normalized to sum to one and serve as the non-zero elements of the final

<span id="page-4-2"></span>mask, while all non-top-K positions remain zero. Mathematically, this is formulated as:

$$\tilde{\boldsymbol{v}}_i = \hat{w}_i \cdot \text{MLP}(\boldsymbol{v}_i), \quad \hat{w}_i = \frac{\exp(w_i/\tau') \cdot \tilde{z}_i}{\sum_{j=1}^{N_v} \exp(w_j/\tau') \cdot \tilde{z}_j}, (7)$$

where MLP indicates a multilayer perceptron and τ ′ represents the temperature hyperparameter. This strategy ensures that token selection remains query-aware and trainable.

# <span id="page-4-0"></span>3.4. Progressive Optimization Strategy

To ensure stable convergence and effective cross-modal adaptation, GroundVTS is optimized through a three-stage progressive training strategy. Each stage focuses on a specific learning objective, allowing the model to gradually develop non-uniform visual token sampling and queryconditioned reasoning capabilities while maintaining the stability of the underlying Vid-LLM.

Stage 1: VTS Warm-up. The first stage aims to initialize the query-guided visual token sampling process. Since the VTS module dynamically samples tokens based on query relevance, jointly training it with the LLM from scratch can lead to unstable gradients and inconsistent selection behavior. To mitigate this, we train only the parameters of the VTS module while freezing all other components. This warm-up phase enables VTS to learn robust visual token-query relevance estimation and stable visual importance prediction before engaging in joint optimization.

Stage 2: Joint LoRA Adaptation. After the VTS has learned stable sampling behavior, the second stage focuses on aligning the non-uniform visual token distributions with textual semantics. We fine-tune the LLM using Low-Rank Adaptation (LoRA) [\[14\]](#page-8-17) while jointly updating the VTS and the multimodal projector. This stage leverages a large-scale multimodal dataset LLaVA-Video-178K [\[65\]](#page-10-15), exposing the model to diverse temporal structures and cross-modal reasoning scenarios. Through this joint optimization, the model learns to effectively interpret and reason over query-guided, non-uniform visual token sequences, enabling fine-grained and efficient temporal reasoning within the LLM.

Stage 3: Grounding Fine-tuning. The final stage aims to refine the model's temporal grounding and reasoning ability for real-world VTG tasks. We fine-tune GroundVTS on a newly curated dataset, Grounding-FT (see Sec. [4.1\)](#page-4-1), which aggregates training samples from multiple temporal grounding datasets. Each instance in Grounding-FT follows the same instruction-style QA format as in previous stages, and is used to guide the model to predict temporal boundaries or highlight relevant events. This unified instruction-based interface enables GroundVTS to perform temporal grounding and descriptive reasoning jointly within a single generative process. Throughout this stage, the module freezing configuration follows that of Stage 2, ensuring smooth and stable fine-tuning.

# 4. Experiments

# <span id="page-4-1"></span>4.1. Experimental Setup

Dataset Preparation. We employ two datasets in the training pipeline: (a) LLaVA-Video-178K [\[65\]](#page-10-15), a large-scale video dataset that provides diverse multimodal supervision across tasks such as video captioning and VQA for pretraining and alignment; and (b) Grounding-FT, a curated dataset we construct for VTG tasks. To adapt VTG supervision to the natural language input-output format of LLMs, we design a set of instruction templates with diverse linguistic expressions and combine them with temporal grounding queries to form QA-style training pairs. Grounding-FT is derived from multiple VTG training splits covering both moment retrieval and highlight detection tasks [\[9,](#page-8-4) [20,](#page-8-5) [21\]](#page-8-6), and contains a total of 70K annotated video-query pairs. Dataset details are provided in the supplementary material. LLaVA-Video-178K is used in the first two training stages, while Grounding-FT is employed in the third stage.

Evaluation. We evaluate our model on two representative

VTG tasks, namely moment retrieval (MR) [\[4,](#page-8-0) [9\]](#page-8-4) and highlight detection (HD) [\[21\]](#page-8-6). The MR task aims to identify the start and end timestamps of the video segment corresponding to a given natural language query. Following standard practice, we conduct evaluation on Charades-STA [\[9\]](#page-8-4), ActivityNet-Captions [\[4\]](#page-8-0), and QVHighlights [\[21\]](#page-8-6), using mean intersection-over-union (mIoU) and Recall@1 (R1@t) at thresholds t ∈ {0.3, 0.5, 0.7} [\[23,](#page-8-18) [41,](#page-9-18) [55\]](#page-10-13). The HD task requires the model to output all salient moments relevant to the query in the video together with their corresponding relevance scores. We use QVHighlights [\[21\]](#page-8-6) for evaluation and adopt mean average precision (mAP) and the hit ratio of the highest-scored clip (Hit@1) as metrics [\[12,](#page-8-11) [30,](#page-9-11) [43\]](#page-9-19). Implementation Details. We construct two model variants, GroundVTS-Q and GroundVTS-I, built upon Qwen2.5VL-7B [\[3\]](#page-8-8) and InternVL3.5-8B [\[48\]](#page-10-16), respectively. Both models are trained using the three-stage strategy described in Sec. [3.4,](#page-4-0) where stages 1–3 are trained for 1, 2, and 3 epochs, respectively, with learning rates of 1×10<sup>−</sup><sup>5</sup> , 2×10<sup>−</sup><sup>4</sup> , and 1×10<sup>−</sup><sup>4</sup> . The two base models differ in their intrinsic video sampling paradigms. QwenVL employs a fixed frame-rate strategy, uniformly sampling frames over time, whereas InternVL adopts a fixed frame-count strategy, representing each video with a constant number of frames regardless of duration. During training, GroundVTS-Q uses a frame rate of 2 FPS, while GroundVTS-I samples 16 frames per video. For the VTS module, the hidden dimension D<sup>r</sup> is set to 512 for GroundVTS-Q and 128 for GroundVTS-I. The visual token sampling ratio is fixed at ρ= 0.5. Additional training settings are detailed in the supplementary material.

Table 1. Comparison with state-of-the-art methods on Charades-STA and ActivityNet-Captions test splits.

<span id="page-5-3"></span><span id="page-5-0"></span>

| Method                 |                                | Charad                         | es-STA                              |                                |                          | ActivityNo              | et-Captions                    |                                     |
|------------------------|--------------------------------|--------------------------------|-------------------------------------|--------------------------------|--------------------------|-------------------------|--------------------------------|-------------------------------------|
| Method                 | R1@.3                          | R1@.5                          | R1@.7                               | mIoU                           | R1@.3                    | R1@.5                   | R1@.7                          | mIoU                                |
| LLaVA-OV[22] arXiv' 24 | 28.8                           | 16.6                           | 5.9                                 | 19.3                           | 20.2                     | 8.6                     | 2.2                            | 13.5                                |
| TimeChat[43] CVPR' 24  | 47.7                           | 22.9                           | 12.5                                | 30.6                           | 30.2                     | 16.9                    | 8.2                            | 21.8                                |
| VTimeLLM[16] CVPR' 24  | 51.0                           | 27.5                           | 11.4                                | 31.2                           | 44.0                     | 27.8                    | 14.3                           | 30.4                                |
| Momentor[39] ICML' 24  | 42.9                           | 23.0                           | 12.4                                | 29.3                           | 42.6                     | 26.6                    | 11.6                           | 28.5                                |
| HawkEye[52] arXiv' 24  | 50.6                           | 31.4                           | 14.5                                | 33.7                           | 49.1                     | 29.3                    | 10.7                           | 32.7                                |
| ChatVTG[41] CVPR' 24   | 52.7                           | 33.0                           | 15.9                                | 34.9                           | 40.7                     | 22.5                    | 9.4                            | 27.2                                |
| NumPro[55] CVPR' 25    | 63.8                           | 42.0                           | 20.6                                | 41.4                           | 55.6                     | 37.5                    | <u>20.6</u>                    | 38.8                                |
| LLaVA-ST[23] CVPR' 25  | 63.1                           | 44.8                           | 23.4                                | 42.4                           |                          |                         |                                |                                     |
| Qwen2.5VL-7B           | 34.2                           | 18.8                           | 8.6                                 | 22.1                           | 25.3                     | 11.5                    | 4.4                            | 17.1                                |
| Qwen2.5VL-7B-G         | 45.2                           | 32.7                           | 18.7                                | 31.7                           | 40.6                     | 23.9                    | 9.9                            | 26.7                                |
| GroundVTS-Q (ours)     | <b>71.5</b> <sub>(†26.3)</sub> | <b>57.5</b> <sub>(†24.8)</sub> | <b>34.2</b> <sub>(†15.5)</sub>      | <b>50.1</b> <sub>(†18.4)</sub> | $51.3_{(\uparrow 10.7)}$ | $33.6_{(\uparrow 9.7)}$ | <b>21.4</b> <sub>(†11.5)</sub> | $\underline{36.0}_{(\uparrow 9.3)}$ |
| InternVL3.5-8B         | 35.5                           | 25.7                           | 13.2                                | 24.6                           | 22.1                     | 12.0                    | 5.6                            | 15.8                                |
| InternVL3.5-8B-G       | 59.5                           | 42.0                           | 20.2                                | 39.4                           | 35.9                     | 20.6                    | 9.0                            | 24.5                                |
| GroundVTS-I (ours)     | $61.2_{(\uparrow 1.7)}$        | 44.2 <sub>(↑2.2)</sub>         | $\underline{23.7}_{(\uparrow 3.5)}$ | 41.6 <sub>(↑2.2)</sub>         | $37.9_{(\uparrow 2.0)}$  | 22.4 <sub>(↑1.8)</sub>  | 10.3 <sub>(↑1.3)</sub>         | $25.7_{(\uparrow 1.2)}$             |

Bold denotes the best, <u>underlined</u> denotes the second-best. "-G" denotes supervised fine-tuning on the Grounding-FT dataset. "-Q" and "-I" denote our proposed models based on Qwen2.5VL-7B [3] and InternVL3.5-8B [48], respectively. ↑ indicates improvement over the corresponding "-G" baseline.

<span id="page-5-1"></span>Table 2. Comparison with state-of-the-art methods on QVHigh-lights validation split.

| Method           | M                              | IR .                     | HD                      |                                |  |  |
|------------------|--------------------------------|--------------------------|-------------------------|--------------------------------|--|--|
| Method           | R1@.5                          | R1@.7                    | mAP                     | Hit@1                          |  |  |
| SeViLA°[59]      | 54.5                           | 36.5                     |                         |                                |  |  |
| UniVTG°[30]      | <u>58.9</u>                    | 40.9                     | 27.0                    | 55.3                           |  |  |
| VTG-LLM[12]      |                                |                          | 16.5                    | 33.5                           |  |  |
| TimeChat[43]     |                                |                          | 14.5                    | 23.9                           |  |  |
| NumPro[55]       |                                |                          | 40.5                    | <u>70.7</u>                    |  |  |
| Qwen2.5VL-7B     | 8.7                            | 2.4                      | 24.9                    | 0.6                            |  |  |
| Qwen2.5VL-7B-G   | 11.0                           | 4.3                      | 34.4                    | 44.5                           |  |  |
| GroundVTS-Q      | $23.6_{(\uparrow 12.6)}$       | $12.3_{(\uparrow 8.0)}$  | $35.7_{(\uparrow 1.3)}$ | $58.8_{(\uparrow 14.3)}$       |  |  |
| InternVL3.5-8B   | 8.7                            | 3.7                      | 24.8                    | 0.32                           |  |  |
| InternVL3.5-8B-G | 31.8                           | 15.0                     | 31.9                    | 39.8                           |  |  |
| GroundVTS-I      | <b>63.6</b> <sub>(†31.8)</sub> | $40.7_{(\uparrow 25.7)}$ | 52.5 <sub>(†20.6)</sub> | <b>88.4</b> <sub>(↑48.6)</sub> |  |  |

o indicates classical expert models; other notations follow Table 1.

#### 4.2. Main Results

Moment Retrieval. As summarized in Tables 1 and 2, our proposed GroundVTS consistently outperforms existing state-of-the-art methods on multiple VTG benchmarks. On Charades-STA, GroundVTS-Q substantially outperforms the fine-tuned Qwen2.5VL-7B baseline, achieving gains of 24.8 points in R1@0.5 and 18.4 points in mIoU, reaching 57.5 R1@0.5 and 50.1 mIoU. On ActivityNet-Captions, GroundVTS-Q improves R1@0.5 by 9.7 points and mIoU by 9.3 points, further confirming the effectiveness of our sampling approach for VTG. Building upon InternVL3.5-8B, GroundVTS-I also shows stable improvements (*e.g.*, +3.5 in R1@0.7 on Charades-STA), validating the generality of our approach across diverse Vid-LLM architectures.

<span id="page-5-2"></span>Table 3. Comparison with state-of-the-art methods on NExT-GQA test splits.

| Model               | mIoU | mIoP | IoU@.5 | IoP@.5 | Acc@GQA |
|---------------------|------|------|--------|--------|---------|
| TOGA° [37]          | 24.4 | 40.5 | 21.1   | 40.6   | 24.6    |
| VideoStreaming [40] | 19.3 | 32.2 | 13.3   | 31.0   | 17.8    |
| GroundVTS-Q         | 25.8 | 37.4 | 20.4   | 35.4   | 23.2    |
| GroundVTS-I         | 16.7 | 26.5 | 11.9   | 24.3   | 18.5    |

On QVHighlights, GroundVTS-I attains 63.6 in R1@0.5 and 40.7 in R1@0.7 for moment retrieval, comparable to specialized methods such as UniVTG [30].

**Highlight Detection.** As shown in Table 2, GroundVTS-I significantly outperforms InternVL3.5-8B-G, improving mAP and Hit@1 by 20.6 and 48.6 points, respectively, to 52.5 and 88.4. It also surpasses strong methods using frame indices as auxiliary inputs, such as NumPro [55], suggesting better sensitivity to key moments in highlight detection.

**Out-of-Distribution Evaluation.** To further assess the effectiveness and generality of our method under task shift, we evaluate our models *as-is* on grounded video question answering with NExT-GQA [57], without any further training; results are shown in Table 3. GroundVTS-Q achieves the highest mIoU and remains competitive on other metrics, despite not being specifically designed or trained for this task. Moreover, the supplementary material reports two additional *as-is* evaluations: DiDeMo [2] for out-of-distribution moment retrieval, and LongVideoBench [54] for transfer to a new long-video understanding task; on both benchmarks, our models either outperform or remain competitive with recent state-of-the-art methods.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

(b) Token efficiency under varying token densities.

Figure 4. Comparison between GroundVTS-Q and Qwen2.5VL-7B-G (denoted as OwenVL-G) under varying token densities.

#### 4.3. Effect of Visual Token Density

To verify the robustness of our GroundVTS with respect to visual token density, we conduct an analysis on the Charades-STA test split, comparing GroundVTS-Q with its fine-tuned base model, Qwen2.5VL-7B-G (abbreviated as QwenVL-G). We adjust the sampling ratio  $\rho$  from 0.1 to 1.0 in increments of 0.1, while keeping the dense sampling frame rate fixed at 2 FPS, to control the number of visual tokens involved in LLM inference. For a fair comparison, QwenVL-G continues to use uniform frame sampling, with the frame rate varied from 0.2 to 2 FPS in increments of 0.2.

As shown in Figure 4(a), the horizontal axis represents effective token density (FPS  $\times \rho$ ), and the vertical axis reports grounding accuracy in terms of R1@0.7. As the token density decreases, QwenVL-G degrades markedly, indicating a strong dependence on dense temporal sampling. In contrast, GroundVTS remains much more stable across the full density range, maintaining high accuracy even in sparse settings. With only half the token budget (FPS  $\times \rho$  = 1.0), GroundVTS achieves 34.2 R1@0.7, already surpassing QwenVL-G at full density (30.5 R1@0.7). Even under a more aggressive reduction (FPS  $\times \rho$  = 0.4), GroundVTS still attains 29.2 R1@0.7, exceeding QwenVL-G by 19.0 points. These results highlight the strong token efficiency and robustness of GroundVTS under sparse sampling.

Figure 4(b) illustrates token efficiency, defined as R1@0.7 divided by effective token density. When fewer visual tokens are available, GroundVTS-Q maintains higher efficiency than QwenVL-G, indicating more effective use of limited visual information. Exact values are provided in the supplementary material.

<span id="page-6-1"></span>Table 4. Ablation of different training-stage combinations for GroundVTS-Q on Charades-STA test split.

| Stage             | R1@0.3                                             | R1@0.5                         | R1@0.7                   | mIoU                           |
|-------------------|----------------------------------------------------|--------------------------------|--------------------------|--------------------------------|
| base <sup>‡</sup> | 34.2                                               | 18.8                           | 8.6                      | 22.1                           |
| None <sup>§</sup> | 8.6 <sub>(\psi25.6)</sub>                          | 5.0 <sub>(\psi13.8)</sub>      | $1.9_{(\downarrow 6.7)}$ | $5.6_{(\downarrow 16.5)}$      |
| 1                 | $31.2_{(\downarrow 3.0)}$                          | $20.5_{(\uparrow 1.7)}$        | $10.0_{(\uparrow 1.4)}$  | $20.9_{(\downarrow 1.2)}$      |
| 1, 2              | $45.8_{(\uparrow 11.6)}$                           | $28.8_{(\uparrow 10.0)}$       | $13.2_{(\uparrow 4.6)}$  | $30.1_{(\uparrow 8.0)}$        |
| 1, 3              | $49.1_{(\uparrow 14.9)}$                           | $32.5_{(\uparrow 13.7)}$       | $15.2_{(\uparrow 6.6)}$  | $32.4_{(\uparrow 10.3)}$       |
| 2, 3              | $69.4_{(\uparrow 35.2)}$                           | <u>53.0</u> (†34.2)            | $30.5_{(\uparrow 21.9)}$ | $47.4_{(\uparrow 25.3)}$       |
| 1, 2, 3           | <b>71.5</b> <sub>(<math>\uparrow</math>37.3)</sub> | <b>57.5</b> <sub>(↑38.7)</sub> | $34.2_{(\uparrow 25.6)}$ | <b>50.1</b> <sub>(↑28.0)</sub> |

<sup>&</sup>lt;sup>‡</sup> Arrowed values indicate absolute changes relative to the base model (Qwen2.5VL-7B). § "None" uses a randomly initialized VTS module.

## 4.4. Effect of the Progressive Optimization Strategy

Table 4 summarizes the effect of different training stages for GroundVTS-Q. Using an untrained VTS module (the "None" setting) causes a sharp drop across all metrics, showing that query-conditioned token sampling must be properly learned. Stage 1 (VTS Warm-up) largely recovers the base-model performance, indicating that VTS can be integrated without disrupting the original pipeline. Adding Stage 2 (Joint LoRA Adaptation) further improves performance, bringing gains of +11.6 in R1@0.3 and +8.0 in mIoU over the base model. Adding Stage 3 (Grounding Fine-tuning) yields the best results, reaching 71.5/57.5/34.2 R1@0.3/0.5/0.7 and 50.1 mIoU. The (1, 3) and (2, 3) variants remain below the full setting, confirming the importance of Stage 2 for large-scale adaptation to non-uniform token distributions and Stage 1 for stable initialization.

## 4.5. Ablation Study

We conduct ablation experiments on two key components of GroundVTS in Table 5: the visual token sampling strategy and the positional encoding used for temporal reasoning. All variants are evaluated under a matched token budget, equivalent to FPS = 2.0 and  $\rho = 0.5$ .

**Sampling Strategy.** We compare our query-guided token-level sampling with three alternatives: (a) *Uniform sampling*, implemented by evaluating Qwen2.5VL-7B-G at 1.0 FPS; (b) *Random sampling*, where 50% of visual tokens are randomly discarded; and (c) *Frame-level query selection*, where visual tokens within each frame are average-pooled to estimate frame-query relevance, and the top 50% frames are retained with all their tokens. Both the token-level and frame-level variants are trained with the same three-stage procedure, while the random variant is initialized from the token-level model after Stages 1 and 2 and trained only in Stage 3 with random dropping.

As shown in Table 5, our token-level VTS achieves the best performance on both datasets. On Charades-STA, it

Table 5. Ablation on sampling strategies and positional encoding (PE) in GroundVTS.

<span id="page-7-0"></span>

| VTS          | PE           | Sampling Methods |             | Charade     | s-STA       |             | ActivityNet-Captions |             |             |             |  |
|--------------|--------------|------------------|-------------|-------------|-------------|-------------|----------------------|-------------|-------------|-------------|--|
| V 1 3        | FE           | Sampling Methods | R1@0.3      | R1@0.5      | R1@0.7      | mIoU        | R1@0.3               | R1@0.5      | R1@0.7      | mIoU        |  |
| $\checkmark$ | ✓            | Token-Level      | 71.5        | 57.5        | 34.2        | 50.1        | 51.3                 | 33.6        | 21.4        | 36.0        |  |
| $\checkmark$ | $\checkmark$ | Frame-Level      | <u>61.7</u> | <u>44.9</u> | <u>23.3</u> | <u>41.6</u> | <u>43.7</u>          | <u>27.5</u> | <u>15.0</u> | <u>30.7</u> |  |
| _            | $\checkmark$ | Uniform          | 42.6        | 28.5        | 15.0        | 29.3        | 36.1                 | 19.5        | 7.5         | 23.4        |  |
| _            | $\checkmark$ | Random           | 54.9        | 35.0        | 16.3        | 35.7        | 40.3                 | 23.4        | 12.1        | 27.7        |  |
| $\checkmark$ | _            | Token-Level      | 15.1        | 7.0         | 2.7         | 9.5         | 22.2                 | 11.2        | 5.2         | 16.3        |  |

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 5. Qualitative comparison of temporal grounding predictions among GroundVTS-Q, Qwen2.5VL-7B-G, and Qwen2.5VL-7B.

improves mIoU over frame-level selection by 8.5 points (50.1 vs. 41.6) and reaches 57.5 R1@0.5. On ActivityNet-Captions, it also performs best, improving mIoU by 5.3 points over the frame-level variant (36.0 vs. 30.7). By contrast, both uniform and random sampling degrade performance, confirming the importance of query-guided fine-grained sampling for temporal grounding; random sampling nevertheless outperforms uniform sampling, possibly because it acts as data augmentation.

Effect of Positional Encoding. To assess the role of positional encoding, we remove position embeddings from GroundVTS while keeping training and inference settings fixed. As shown in Table 5, performance collapses on both datasets. On Charades-STA, mIoU drops from 50.1 to 9.5 and R1@0.5 from 57.5 to 7.0, with similarly severe degradation on ActivityNet-Captions. These results confirm the importance of temporal positional information in Ground-VTS, and validate our design choice of retaining the original relative positional embeddings for the selected tokens.

**Additional ablations** on training data and relevance estimation are provided in the supplementary material.

#### 4.6. Qualitative Study

Figure 5 shows a qualitative comparison on a Charades-STA example with the query "a person takes a book off a shelf." The red curve denotes the normalized token density produced by VTS, with higher values indicating stronger query relevance. GroundVTS-Q assigns most tokens to the early part of the video (roughly 0–13 s), which fully covers the ground-truth interval (6.2–12.0 s), while suppressing nearly all tokens in later frames. Based on these sampled tokens, GroundVTS-Q predicts 6.0–12.0 s, closely matching the ground truth. In contrast, Qwen2.5VL-7B-G predicts an earlier and less precise segment (4.5–10.3 s), while the base Qwen2.5VL-7B misses the target moment entirely. This shows that VTS focuses on relevant temporal regions for grounding. More results are provided in the supplement.

#### 5. Conclusion

In this paper, we present GroundVTS, a query-guided visual token sampling framework for video temporal grounding. Its core module, VTS, can be seamlessly integrated into mainstream Vid-LLMs via a progressive optimization strategy to better capture fine-grained temporal cues. Experiments show that GroundVTS consistently improves instruction-tuned base models and outperforms recent state-of-the-art methods. Further analyses confirm that Ground-VTS improves token utilization and maintains prediction stability across varying input densities.

**Acknowledgements**. The research of Liuyi Wang is supported in part by the National Natural Science Foundation

# References

- <span id="page-8-14"></span>[1] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 9392–9401, 2025. [2](#page-1-1)
- <span id="page-8-20"></span>[2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2017. [6,](#page-5-3) [1](#page-0-0)
- <span id="page-8-8"></span>[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [1,](#page-0-0) [3,](#page-2-4) [5,](#page-4-2) [6](#page-5-3)
- <span id="page-8-0"></span>[4] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In *Proceedings of the ieee conference on computer vision and pattern recognition*, pages 961–970, 2015. [1,](#page-0-0) [5](#page-4-2)
- <span id="page-8-22"></span>[5] Xinye Cao, Hongcan Guo, Jiawen Qian, Guoshun Nan, Chao Wang, Yuqi Pan, Tianhao Hou, Xiaojuan Wang, and Yutong Gao. Videominer: Iteratively grounding key frames of hourlong videos via tree-based group relative policy optimization. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 23773–23783, 2025. [2](#page-1-1)
- <span id="page-8-3"></span>[6] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. *Advances in Neural Information Processing Systems*, 36:72842–72866, 2023. [1](#page-0-0)
- <span id="page-8-10"></span>[7] Andong Deng, Zhongpai Gao, Anwesa Choudhuri, Benjamin Planche, Meng Zheng, Bin Wang, Terrence Chen, Chen Chen, and Ziyan Wu. Seq2time: Sequential knowledge transfer for video llm temporal grounding. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 13766–13775, 2025. [2](#page-1-1)
- <span id="page-8-21"></span>[8] Pengcheng Fang, Yuxia Chen, and Rui Guo. When and what: Diffusion-grounded videollm with entity aware segmentation for long video understanding. *arXiv preprint arXiv:2508.15641*, 2025. [2](#page-1-1)
- <span id="page-8-4"></span>[9] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In *Proceedings of the IEEE international conference on computer vision*, pages 5267–5275, 2017. [1,](#page-0-0) [3,](#page-2-4) [5](#page-4-2)
- <span id="page-8-7"></span>[10] Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Omnimae: Single model masked pretraining on images and videos. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10406–10417, 2023. [1](#page-0-0)
- <span id="page-8-1"></span>[11] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video

- database for learning and evaluating visual common sense. In *Proceedings of the IEEE international conference on computer vision*, pages 5842–5850, 2017. [1](#page-0-0)
- <span id="page-8-11"></span>[12] Yongxin Guo, Jingyu Liu, Mingda Li, Dingxin Cheng, Xiaoying Tang, Dianbo Sui, Qingbin Liu, Xi Chen, and Kevin Zhao. Vtg-llm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 3302–3310, 2025. [2,](#page-1-1) [5,](#page-4-2) [6](#page-5-3)
- <span id="page-8-2"></span>[13] Mingfei Han, Linjie Yang, Xiaojie Jin, Jiashi Feng, Xiaojun Chang, and Heng Wang. Video recognition in portrait mode. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 21831–21841, 2024. [1](#page-0-0)
- <span id="page-8-17"></span>[14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022. [5](#page-4-2)
- <span id="page-8-9"></span>[15] Jingjing Hu, Dan Guo, Kun Li, Zhan Si, Xun Yang, and Meng Wang. Maskable retentive network for video moment retrieval. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 1476–1485, 2024. [2](#page-1-1)
- <span id="page-8-12"></span>[16] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14271–14280, 2024. [2,](#page-1-1) [6](#page-5-3)
- <span id="page-8-13"></span>[17] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. In *European Conference on Computer Vision*, pages 202–218. Springer, 2024. [2](#page-1-1)
- <span id="page-8-15"></span>[18] Kai Huang, Hao Zou, Ye Xi, BoChen Wang, Zhen Xie, and Liang Yu. Ivtp: Instruction-guided visual token pruning for large vision-language models. In *European Conference on Computer Vision*, pages 214–230. Springer, 2024. [2](#page-1-1)
- <span id="page-8-16"></span>[19] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. *arXiv preprint arXiv:1611.01144*, 2016. [4](#page-3-1)
- <span id="page-8-5"></span>[20] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In *Proceedings of the IEEE international conference on computer vision*, pages 706–715, 2017. [1,](#page-0-0) [5,](#page-4-2) [3](#page-2-4)
- <span id="page-8-6"></span>[21] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. *Advances in Neural Information Processing Systems*, 34: 11846–11858, 2021. [1,](#page-0-0) [5,](#page-4-2) [3](#page-2-4)
- <span id="page-8-19"></span>[22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024. [6](#page-5-3)
- <span id="page-8-18"></span>[23] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. Llava-st: A multimodal large language model for fine-grained spatialtemporal understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8592–8603, 2025. [5,](#page-4-2) [6](#page-5-3)

- <span id="page-9-5"></span>[24] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. *arXiv preprint arXiv:2305.06355*, 2023. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-9-22"></span>[25] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22195– 22206, 2024. [1](#page-0-0)
- <span id="page-9-0"></span>[26] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 4641–4650, 2016. [1](#page-0-0)
- <span id="page-9-16"></span>[27] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In *European Conference on Computer Vision*, 2024. [2](#page-1-1)
- <span id="page-9-9"></span>[28] Hao Liang, Jiapeng Li, Tianyi Bai, Xijie Huang, Linzhuang Sun, Zhengren Wang, Conghui He, Bin Cui, Chong Chen, and Wentao Zhang. Keyvideollm: Towards large-scale video keyframe selection. *arXiv preprint arXiv:2407.03104*, 2024. [1](#page-0-0)
- <span id="page-9-6"></span>[29] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. *arXiv preprint arXiv:2311.10122*, 2023. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-9-11"></span>[30] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2794–2804, 2023. [2,](#page-1-1) [5,](#page-4-2) [6](#page-5-3)
- <span id="page-9-12"></span>[31] Daizong Liu, Xiaoye Qu, and Pan Zhou. Progressively guide to attend: An iterative alignment framework for temporal sentence grounding. *arXiv preprint arXiv:2109.06400*, 2021. [2](#page-1-1)
- <span id="page-9-10"></span>[32] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? *arXiv preprint arXiv:2403.00476*, 2024. [2](#page-1-1)
- <span id="page-9-8"></span>[33] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning. *arXiv preprint arXiv:2503.13444*, 2025. [1](#page-0-0)
- <span id="page-9-17"></span>[34] Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, and Hongtao Xie. Hybrid-level instruction injection for video token compression in multi-modal large language models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2025. [2](#page-1-1)
- <span id="page-9-7"></span>[35] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. *arXiv preprint arXiv:2306.05424*, 2023. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-9-23"></span>[36] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video un-

- derstanding via large vision and language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12585–12602, 2024. [2](#page-1-1)
- <span id="page-9-20"></span>[37] Your Name and Coauthor Name. Toga: Temporally grounded open-ended video qa with weak supervision. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2025. [6,](#page-5-3) [2](#page-1-1)
- <span id="page-9-14"></span>[38] Jongwoo Park, Kanchana Ranasinghe, Kumara Kahatapitiya, Wonjeong Ryu, Donghyun Kim, and Michael S Ryoo. Too many frames, not all useful: Efficient strategies for longform video qa. *arXiv preprint arXiv:2406.09396*, 2024. [2](#page-1-1)
- <span id="page-9-13"></span>[39] Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning. *arXiv preprint arXiv:2402.11435*, 2024. [2,](#page-1-1) [6](#page-5-3)
- <span id="page-9-21"></span>[40] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. *Advances in Neural Information Processing Systems*, 37:119336–119360, 2024. [6,](#page-5-3) [2](#page-1-1)
- <span id="page-9-18"></span>[41] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. Chatvtg: Video temporal grounding via chat with video dialogue large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1847–1856, 2024. [5,](#page-4-2) [6](#page-5-3)
- <span id="page-9-1"></span>[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PmLR, 2021. [1](#page-0-0)
- <span id="page-9-19"></span>[43] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14313–14323, 2024. [5,](#page-4-2) [6,](#page-5-3) [2](#page-1-1)
- <span id="page-9-2"></span>[44] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 7464–7473, 2019. [1](#page-0-0)
- <span id="page-9-15"></span>[45] Xudong Tan, Peng Ye, Chongjun Tu, Jianjian Cao, Yaoxin Yang, Lin Zhang, Dongzhan Zhou, and Tao Chen. Tokencarve: Information-preserving visual token compression in multimodal large language models. *arXiv preprint arXiv:2503.10501*, 2025. [2](#page-1-1)
- <span id="page-9-4"></span>[46] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. *IEEE Transactions on Circuits and Systems for Video Technology*, 2025. [1](#page-0-0)
- <span id="page-9-3"></span>[47] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. *Advances in neural information processing systems*, 35:10078–10093, 2022. [1](#page-0-0)

- <span id="page-10-16"></span>[48] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. *arXiv preprint arXiv:2508.18265*, 2025. [5,](#page-4-2) [6](#page-5-3)
- <span id="page-10-12"></span>[49] Xizi Wang, Feng Cheng, Ziyang Wang, Huiyu Wang, Md Mohaiminul Islam, Lorenzo Torresani, Mohit Bansal, Gedas Bertasius, and David Crandall. Timerefine: Temporal grounding with time refining video llm. *arXiv preprint arXiv:2412.09601*, 2024. [2](#page-1-1)
- <span id="page-10-7"></span>[50] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In *European Conference on Computer Vision*, pages 58–76. Springer, 2024. [1](#page-0-0)
- <span id="page-10-9"></span>[51] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In *European Conference on Computer Vision*, pages 396–416. Springer, 2024. [2](#page-1-1)
- <span id="page-10-17"></span>[52] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training videotext llms for grounding text in videos. *arXiv preprint arXiv:2403.10228*, 2024. [6](#page-5-3)
- <span id="page-10-8"></span>[53] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 3272– 3283, 2025. [1,](#page-0-0) [2,](#page-1-1) [3](#page-2-4)
- <span id="page-10-20"></span>[54] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. *Advances in Neural Information Processing Systems*, 37:28828–28857, 2024. [6,](#page-5-3) [1](#page-0-0)
- <span id="page-10-13"></span>[55] Yongliang Wu, Xinting Hu, Yuyang Sun, Yizhou Zhou, Wenbo Zhu, Fengyun Rao, Bernt Schiele, and Xu Yang. Number it: Temporal grounding videos like flipping manga. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 13754–13765, 2025. [2,](#page-1-1) [5,](#page-4-2) [6](#page-5-3)
- <span id="page-10-2"></span>[56] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9777–9786, 2021. [1](#page-0-0)
- <span id="page-10-19"></span>[57] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13204– 13214, 2024. [6,](#page-5-3) [1](#page-0-0)
- <span id="page-10-10"></span>[58] Jin Yang, Ping Wei, Huan Li, and Ziyang Ren. Task-driven exploration: Decoupling and inter-task feedback for joint moment retrieval and highlight detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18308–18318, 2024. [2](#page-1-1)
- <span id="page-10-18"></span>[59] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. *Advances in Neural Information Processing Systems*, 36:76749–76771, 2023. [6](#page-5-3)

- <span id="page-10-3"></span>[60] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 9127–9134, 2019. [1](#page-0-0)
- <span id="page-10-4"></span>[61] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6720–6731, 2019. [1](#page-0-0)
- <span id="page-10-6"></span>[62] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. *arXiv preprint arXiv:2410.19702*, 2024. [1](#page-0-0)
- <span id="page-10-11"></span>[63] Yingsen Zeng, Yujie Zhong, Chengjian Feng, and Lin Ma. Unimd: Towards unifying moment retrieval and temporal action detection. In *European Conference on Computer Vision*, pages 286–304. Springer, 2024. [2](#page-1-1)
- <span id="page-10-21"></span>[64] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. *arXiv preprint arXiv:2306.02858*, 2023. [2](#page-1-1)
- <span id="page-10-15"></span>[65] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. [5](#page-4-2)
- <span id="page-10-14"></span>[66] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In *International Conference on Machine Learning*, 2025. [2](#page-1-1)
- <span id="page-10-0"></span>[67] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 8668–8678, 2019. [1](#page-0-0)
- <span id="page-10-1"></span>[68] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In *Proceedings of the AAAI conference on artificial intelligence*, 2018. [1](#page-0-0)
- <span id="page-10-5"></span>[69] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8746–8755, 2020. [1](#page-0-0)

# GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding

# Supplementary Material

# 6. Influence on General VQA

While GroundVTS significantly improves temporal grounding accuracy, it is essential to evaluate whether its selective token sampling mechanism impacts the model's ability to handle general video understanding tasks. To investigate this, we assess GroundVTS on multiple subtasks of MVBench [\[25\]](#page-9-22), a benchmark that comprehensively measures various video question-answering (VQA) capabilities, including temporal reasoning, object interaction, and scenelevel comprehension.

As shown in Table [6,](#page-12-0) GroundVTS-Q achieves a slightly higher overall score (+0.8) compared to the base model Qwen2.5-VL-7B, indicating that the introduction of the VTS module does not compromise general VQA capability. The overall performance of GroundVTS-Q is competitive, with significant improvements in several tasks. The most notable gains appear in subtasks that focus on temporal reasoning and fine-grained action reasoning, such as Action Count (AC), Action Localization (AL) and Action Sequence (AS). In these tasks, GroundVTS-Q outperforms the base model by a substantial margin, with an improvement of 8.0 points in AC, 8.0 points in AL and 0.5 points in AS. These results align with the design objective of enhancing temporal sensitivity, confirming that VTS excels at capturing the temporal aspects of video data.

Meanwhile, in tasks that require a more general understanding of the scene or object interactions, such as Scene Transition (ST) and State Change (SC), GroundVTS-Q maintains competitive performance. For example, GroundVTS-Q achieves a slight drop in ST (-1.0) and SC (-2.0) compared to the base model, but still performs well overall. This suggests that the selective filtering mechanism of the VTS module does not undermine the model's ability to grasp global scene awareness or appearance-related information.

In summary, the results suggest that the introduction of VTS enhances the model's performance on tasks requiring fine-grained temporal reasoning, while maintaining strong performance in more general video understanding tasks. This balance between focused temporal sensitivity and general visual understanding demonstrates the versatility and effectiveness of the GroundVTS framework.

# 7. Out-of-distribution Data Experiment

To further evaluate the robustness and generalization ability of GroundVTS, we conduct out-of-distribution (OOD) experiments on three benchmarks: DiDeMo [\[2\]](#page-8-20), LongVideoBench [\[54\]](#page-10-20), and NExT-GQA [\[57\]](#page-10-19). None of these datasets are included in the fine-tuning data of either GroundVTS or its base models (Qwen2.5-VL and InternVL3.5). This evaluation examines whether our method can effectively transfer its temporal grounding capability to unseen domains and tasks.

Results on DiDeMo. As shown in Table [7,](#page-12-1) GroundVTS-Q achieves substantial gains over both the instructiontuned baseline QwenVL-G and the pretrained Qwen2.5- VL. Specifically, it improves by +13.7 R1@0.3, +10.1 R1@0.5, and +8.0 mIoU, establishing new SOTA performance on two of the three metrics. These results highlight the strong cross-domain adaptability brought by our queryguided VTS mechanism. On InternVL3.5-8B, GroundVTS-I also produces consistent improvements over InternVL-G, yielding gains of +0.9 R1@0.3, +1.3 R1@0.5, and +1.1 mIoU. While the improvements are smaller than those observed with Qwen2.5-VL, they confirm that GroundVTS remains effective even when the underlying base model already possesses strong temporal reasoning capability.

Results on LongVideoBench. LongVideoBench evaluates temporal reasoning on significantly longer videos, with durations ranging from tens of seconds to several minutes. As shown in Table [8,](#page-12-2) GroundVTS-I achieves the best accuracy on two of the three duration ranges, reaching 65.6% on (8,15]s and 68.0% on (15,60]s, and remains competitive on (180,600]s. These results suggest that the proposed query-guided temporal selection mechanism can effectively scale to long-video scenarios by filtering irrelevant temporal regions and concentrating computation on query-related segments.

Results on NExT-GQA. NExT-GQA evaluates both temporal grounding and question answering accuracy, requiring models to first localize relevant temporal segments before answering questions. As shown in Table [9,](#page-12-3) GroundVTS-Q achieves the best mIoU (25.8) among all compared methods and remains competitive on other grounding-related metrics. Notably, the strongest baseline, TOGA, is a classical expert model specifically designed for grounded video question answering, whereas Ground-VTS is built upon a general-purpose multimodal large language model without task-specific architecture. Despite this difference, GroundVTS-Q still attains comparable performance across most metrics, demonstrating that the proposed query-guided temporal selection mechanism can effectively transfer its temporal localization capability to reasoning-

Table 6. Comparison with base model on MVBench subtasks.

<span id="page-12-0"></span>

| Model         | AA   | AC | AL | AS   | EN   | ER   | FGA  | OI   | OS   | ST   | SC   | all  |
|---------------|------|----|----|------|------|------|------|------|------|------|------|------|
| Qwen2.5-VL-7B | 77   | 39 | 38 | 69.7 | 30   | 50   | 44.5 | 66.5 | 35.5 | 90.5 | 50.5 | 53.7 |
| GroundVTS-Q   | 75.5 | 47 | 46 | 70.2 | 28.5 | 46.5 | 41   | 63.5 | 43   | 89.5 | 48.5 | 54.5 |

<span id="page-12-1"></span>Table 7. Comparison with state-of-the-art methods on DiDeMo test splits.

| Model              | R1@0.3                         | R1@0.5                   | mIoU                          |
|--------------------|--------------------------------|--------------------------|-------------------------------|
| Video-LLaMA [64]   | 20.1                           | 8.2                      | 14.3                          |
| Video-ChatGPT [36] | 19.8                           | 6.5                      | 13.7                          |
| Valley             | 33.2                           | 13.4                     | 21.8                          |
| VideoChat [24]     | 34.5                           | 14.5                     | 22.4                          |
| Momenter           | 38.2                           | 21.8                     | 26.5                          |
| VTimeLLM [16]      | <u>45.0</u>                    | <u>28.8</u>              | 27.9                          |
| TimeChat [43]      | 42.8                           | 24.4                     | 28.2                          |
| HawkEye            | 44.8                           | 29.7                     | 29.5                          |
| Qwen2.5-VL-7B      | 28.7                           | 22.7                     | 22.2                          |
| QwenVL-G           | 32.6                           | 17.7                     | 22.0                          |
| GroundVTS-Q        | <b>46.3</b> <sub>(†13.7)</sub> | $27.8_{(\uparrow 10.1)}$ | <b>30.0</b> <sub>(↑8.0)</sub> |
| InternVL3.5-8B     | 29.5                           | 23.9                     | 23.0                          |
| InternVL-G         | 36.6                           | 21.0                     | 23.1                          |
| GroundVTS-I        | $37.5_{(\uparrow 0.9)}$        | $22.3_{(\uparrow 1.3)}$  | $24.2_{(\uparrow 1.1)}$       |

The baseline results are from reference [8].

<span id="page-12-2"></span>Table 8. Comparison with state-of-the-art methods on Long Video-Bench test splits (Acc).

| Model          | (8, 15]s    | (15, 60]s   | (180, 600]s |
|----------------|-------------|-------------|-------------|
| VideoTree [53] | 61.0        | 57.5        | 48.4        |
| VideoMiner [5] | <u>65.1</u> | <u>64.7</u> | 58.6        |
| GroundVTS-Q    | 52.9        | 60.5        | 44.2        |
| GroundVTS-I    | 65.6        | 68.0        | <u>52.4</u> |

intensive video QA tasks.

Overall, the OOD evaluation across three diverse benchmarks demonstrates that GroundVTS generalizes well to unseen datasets and tasks, including short video grounding (DiDeMo), long-video reasoning (LongVideoBench), and grounded video question answering (NExT-GQA). These results reinforce the robustness of the proposed framework and its ability to capture transferable fine-grained temporal cues beyond the training distribution.

#### 8. Parameter-free projection

We evaluate a parameter-free relevance estimation method (Table 10). Without additional training, this approach leads to a substantial performance drop due to the mismatch between the sampled-token distribution and the pretrained

<span id="page-12-3"></span>Table 9. Comparison with state-of-the-art methods on NExT-GQA test splits.

| Model             | mIoU             | mIoP | IoU@.5      | IoP@.5 | Acc@GQA |
|-------------------|------------------|------|-------------|--------|---------|
| TOGA [37]         | 24.4             | 40.5 | 21.1        | 40.6   | 24.6    |
| VidStreaming [40] | 19.3             | 32.2 | 13.3        | 31.0   | 17.8    |
| GroundVTS-Q       | <b>25.8</b> 16.7 | 37.4 | <u>20.4</u> | 35.4   | 23.2    |
| GroundVTS-I       |                  | 26.5 | 11.9        | 24.3   | 18.5    |

LLM (w/o training). To alleviate this issue, we further fine-tune the LLM following Stages 2&3 to adapt to the sampled-token distribution (w/ training), which partially recovers the performance. However, it still underperforms the full GroundVTS model with learned relevance projections.

# 9. Dataset Ablation

GroundVTS-Q is trained on the LLaVA-Video-178K and our constructed Grounding-FT datasets. For a fair comparison, the base Qwen2.5VL-7B is trained on the same datasets under three settings, as reported in Table 11: (i) *Qwen-G*, trained only on the Grounding-FT dataset; (ii) *Qwen-(L+G)*, trained on the concatenation of the two datasets; and (iii) *Qwen-(L→G)*, trained following the same Stage  $2\rightarrow$ Stage 3 curriculum. Note that Stage 1 (VTS warm-up) is not applicable to the base model.

As shown in Table 11, GroundVTS-Q consistently outperforms the Qwen baselines across all evaluation settings. In particular, GroundVTS-Q achieves 50.1 mIoU on Charades-STA, significantly surpassing Qwen-G with 31.7, Qwen-(L+G) with 28.5, and Qwen-(L→G) with 29.8. These results suggest that the proposed grounding-aware training strategy effectively improves performance under matched data and training configurations.

## 10. Frame Sampling Sensitivity of InternVL3.5

To examine whether the frame density sensitivity is specific to Qwen2.5-VL or reflects a more general phenomenon, we conduct an additional experiment using InternVL3.5. Unlike Qwen2.5-VL, InternVL3.5 adopts a fixed-number frame sampling strategy. Therefore, we vary the number of sampled frames to analyze how visual token density affects VTG performance.

Figure 6 presents the frame sensitivity results of InternVL3.5 on the QVHighlights dataset. Similar to the trend observed with Qwen2.5-VL, the performance again exhibits

Table 10. Parameter-free token sampling vs. GroundVTS.

<span id="page-13-0"></span>

| VTS Trainir  | Training     |        | Charade | s-STA  |      | ActivityNet-Captions |        |        |      |  |
|--------------|--------------|--------|---------|--------|------|----------------------|--------|--------|------|--|
| VTS Training |              | R1@0.3 | R1@0.5  | R1@0.7 | mIoU | R1@0.3               | R1@0.5 | R1@0.7 | mIoU |  |
| $\checkmark$ | <b>√</b>     | 71.5   | 57.5    | 34.2   | 50.1 | 51.3                 | 33.6   | 21.4   | 36.0 |  |
| _            | $\checkmark$ | 69.6   | 52.7    | 29.6   | 47.5 | 38.8                 | 25.0   | 13.7   | 27.8 |  |
| _            | _            | 21.2   | 13.6    | 6.8    | 14.5 | 9.1                  | 5.1    | 2.6    | 6.6  |  |

<span id="page-13-1"></span>Table 11. Dataset ablation on Charades-STA and ActivityNet-Captions test spilt.

| Variant                                                                            |                     | Charac              | des-ST              | A                   | Act                 | ActivityNet-Captions |                    |                     |  |
|------------------------------------------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|----------------------|--------------------|---------------------|--|
|                                                                                    | R1<br>@.3           | R1<br>@.5           | R1<br>@.7           | mIoU                | R1<br>@.3           | R1<br>@.5            | R1<br>@.7          | mIoU                |  |
| Qwen-G                                                                             | 45.2                | 32.7                | 18.7                | 31.7                | 40.6                | 23.9                 | 9.9                | 26.7                |  |
| Qwen-(L+G)                                                                         | 41.1                | 27.7                | 15.7                | 28.5                | 39.1                | 20.6                 | 7.8                | 24.9                |  |
| $\begin{array}{c} Qwen\text{-}(L{\rightarrow}G) \\ GroundVTS\text{-}Q \end{array}$ | 42.5<br><b>71.5</b> | 30.7<br><b>57.5</b> | 16.9<br><b>34.2</b> | 29.8<br><b>50.1</b> | 40.0<br><b>51.3</b> | 22.1<br><b>33.6</b>  | 8.6<br><b>21.4</b> | 25.9<br><b>36.0</b> |  |

<span id="page-13-2"></span>![](_page_13_Figure_4.jpeg)

Figure 6. Frame sensitivity of InternVL3.5 on QVHighlights.

a clear non-linear dependency on frame density. Increasing the number of sampled frames initially improves performance by providing richer temporal cues. However, beyond a certain point, further increasing the frame count leads to diminishing returns and eventually performance degradation, suggesting that excessive visual tokens introduce redundancy and interfere with effective temporal reasoning.

These results indicate that the sensitivity to visual token density is not limited to a specific model architecture, but appears to be a general characteristic of multimodal LLM-based VTG systems. This observation further supports our motivation for designing an adaptive token sampling mechanism in GroundVTS.

# 11. Additional Analysis of Visual Token Density

Table 12 provides the full quantitative results corresponding to the visual token density analysis discussed in the main paper. The results further substantiate the trends previously observed. For the pretrained Qwen2.5VL-7B, performance grows steadily as token density increases, but drops rapidly in sparse conditions. This confirms its heavy dependence on dense temporal evidence: when the effective density

falls below 1.0, all metrics decrease sharply (e.g., R1@0.5 drops from 47.1 to 18.8 as density reduces from 2.0 to 1.0). The fine-tuned QwenVL-G shows improved overall accuracy but remains highly sensitive to token density.

In contrast, GroundVTS-Q demonstrates remarkable stability across all density levels. At extremely sparse levels (e.g., density 0.2–0.6), its performance remains comparable to the best performance of its base model, avoiding the sharp degradation observed in QwenVL-G. As the density increases, the improvement of GroundVTS-Q is much more gradual, forming a plateau rather than a steep curve. This consistency appears across all evaluation metrics, including R1@0.3, R1@0.5, R1@0.7, and mIoU, illustrating that GroundVTS effectively mitigates the vulnerability of Vid-LLMs to insufficient visual tokens. These results further reinforces the conclusion that our query-guided sampling mechanism yields reliable grounding accuracy regardless of input density, while base Vid-LLMs suffer substantial degradation when token budgets are reduced.

# 12. Training Details

This section provides the detailed configurations and parameters used for training GroundVTS across its different stages, as well as the parameter values for the model variants. Table 13 outlines the settings for each stage of training, including the learning rate, optimizer, batch size, and other critical training details. Table 14 lists the total and trainable parameters for both GroundVTS-Q (Qwen2.5VL-based) and GroundVTS-I (InternVL-based) models.

#### 13. Grounding-FT Dataset

Grounding-FT is a curated dataset designed for instruction fine-tuning on Video Temporal Grounding (VTG) tasks. It aggregates the training splits of Charades-STA [9], QVHighlights [21], and ActivityNet-Captions [20], resulting in 70K annotated clips paired with instruction-style queries. The goal is to unify multiple VTG formulations under a consistent question-answering (QA) framework, facilitating language model training with natural conversational inputs rather than fixed task templates.

#### 13.1. Overview and Construction

Grounding-FT covers two main VTG task types:

Table 12. Quantitative analysis of visual token density on Charades-STA test split.

<span id="page-14-0"></span>

| EDC          |        | Qwen2.5 | VL-7B  |      |        | QwenV  | /L-G   |      | GroundVTS-Q |        |        |      |
|--------------|--------|---------|--------|------|--------|--------|--------|------|-------------|--------|--------|------|
| $FPS * \rho$ | R1@0.3 | R1@0.5  | R1@0.7 | mIoU | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | R1@0.3      | R1@0.5 | R1@0.7 | mIoU |
| 0.2          | 22.7   | 13.3    | 6.8    | 16.1 | 28.0   | 16.5   | 8.3    | 18.7 | 61.2        | 43.6   | 23.3   | 41.0 |
| 0.4          | 23.5   | 13.5    | 6.6    | 16.4 | 33.0   | 20.2   | 10.2   | 21.8 | 67.1        | 50.8   | 29.2   | 46.0 |
| 0.6          | 26.5   | 14.1    | 6.8    | 17.6 | 36.2   | 24.0   | 12.6   | 24.9 | 69.5        | 54.4   | 32.6   | 48.2 |
| 0.8          | 30.3   | 16.4    | 7.2    | 19.7 | 41.4   | 28.3   | 15.7   | 28.4 | 70.9        | 56.8   | 33.6   | 49.6 |
| 1.0          | 34.2   | 18.8    | 8.6    | 22.1 | 45.2   | 32.7   | 18.7   | 31.7 | 71.5        | 57.5   | 34.2   | 50.1 |
| 1.2          | 36.8   | 21.4    | 9.1    | 24.2 | 49.3   | 36.6   | 22.0   | 35.2 | 72.3        | 58.4   | 34.8   | 50.7 |
| 1.4          | 42.2   | 25.0    | 10.6   | 28.0 | 54.8   | 40.8   | 24.5   | 38.9 | 72.9        | 58.5   | 35.3   | 51.0 |
| 1.6          | 48.6   | 29.1    | 12.5   | 32.2 | 62.7   | 44.6   | 24.8   | 42.8 | 72.8        | 58.3   | 34.7   | 50.8 |
| 1.8          | 59.5   | 37.0    | 16.6   | 38.7 | 72.2   | 53.0   | 27.4   | 48.2 | 73.0        | 58.3   | 34.5   | 50.9 |
| 2.0          | 68.8   | 47.1    | 23.5   | 45.4 | 74.4   | 56.3   | 30.6   | 50.0 | 72.8        | 58.5   | 34.7   | 50.9 |

Table 13. Training configuration for each stage of the GroundVTS model.

<span id="page-14-1"></span>

| Stage                             | Trainable<br>Modules            | Learning<br>Rate | Optimizer                                  | Batch Size<br>(per GPU) | Grad. Acc.<br>Steps | Epochs | LoRA<br>Config                              | Dataset          |
|-----------------------------------|---------------------------------|------------------|--------------------------------------------|-------------------------|---------------------|--------|---------------------------------------------|------------------|
| Stage 1: VTS<br>Warm-up           | VTS                             | 1e-5             |                                            | 2                       | 4                   | 1      | -                                           | LLaVA-Video-178K |
| Stage 2: Joint<br>LoRA Adaptation | LLM (LoRA) +<br>VTS + Projector | 2e-4             | AdamW, $\beta_1 = 0.9$ , $\beta_2 = 0.999$ | 2                       | 4                   | 2      | $rank = 8,$ $\alpha = 16,$ $dropout = 0.05$ | LLaVA-Video-178K |
| Stage 3: Grounding<br>Fine-tuning | LLM (LoRA) +<br>VTS + Projector | 1e-4             | ,, 2                                       | 2                       | 4                   | 3      | $rank = 8,$ $\alpha = 16,$ $dropout = 0.05$ | Grounding-FT     |

<span id="page-14-2"></span>Table 14. Parameter statistics for GroundVTS-Q and GroundVTS-I models.

| M- 1-1      | T-4-1 D      | Trainable Params |           |       |        |  |
|-------------|--------------|------------------|-----------|-------|--------|--|
| Model       | Total Params | VTS              | Projector | LoRA  | All    |  |
| GroundVTS-Q | 8.32B        | 29.4M            | 44.6M     | 79.0M | 153.0M |  |
| GroundVTS-I | 8.56B        | 34.6M            | 33.6M     | 77.0M | 145.2M |  |

- (a) Moment Retrieval (MR)—identifying the temporal segment in a video that corresponds to a given natural language query.
- **(b) Highlight Detection (HD)**—output all salient moments relevant to the query in the video together with their corresponding relevance scores.

For MR, we aggregate annotations from the training splits of Charades-STA, QVHighlights, and ActivityNet-Captions. For HD, we use the training split of QVHighlights. All samples are reformulated into an instruction-response style and stored in the ShareGPT format, where each instance contains a conversational pair between a user (prompt) and an assistant (answer), along with the corresponding video path. To enhance linguistic diversity and improve generalization to natural language instructions, we construct a pool of prompt templates and randomly select one for each instance rather than relying on

a single fixed phrasing. This variation helps the fine-tuned model better adapt to free-form human queries. Note that timestamp information is not provided in the text prompt, and all models must rely on the positional encodings of visual tokens to infer temporal information.

#### 13.2. Moment Retrieval Task

Each MR training instance contains at least the video name <video>, a query phrase {query}, and the ground-truth {start} and {end} timestamps. We construct diverse instruction templates and randomly sample one for each example to enhance linguistic variability. The prompt templates and expected output format are summarized in Table 15. Examples before and after the conversion are as follows:

#### Example 1 (Charades-STA).

Original annotation:

```
Y6R7T 20.8 30.0##person start playing on their phone.
```

#### Reformatted instance:

```
"messages": [
    {"role":"user",
        "content":"<video>At what point in the
        video did the following events occur:
        person start playing on their phone.
        Output the start and end timestamps."},
```

Table 15. Prompt templates and output format for the MR task.

#### <span id="page-15-0"></span>Type Content Prompt Templates <video>At what point in the video did the following events occur: {query}? Output the start and end timestamps. <video>What is the location of the moment: {query}? <video>Find when the following event happens in the video: {query}. Give me the start and end times. <video>Please indicate the start and end timestamps for the event: {query}. <video>Please predict start and end time of the following moment: {query}. <video>During which time interval does this happen in the video: {query}? <video>Locate the moment in the video where this occurs: {query}. Provide start and end times. <video>For the video, when does this event take place: {query}? Answer with start and end timestamps. <video>I want to know the start and end times of the following event in the video: {query}. <video>Could you tell me from what time to what time this happens: {query}? <video>Can you tell me the time window of this event: {query}? <video>Please find the timestamps that mark the occurrence of this event: {query}. <video>Identify the start and end of the following event in the video: {query}. Expected Output from {start}s to {end}s

```
{"role":"assistant",
     "content":"from 20.8s to 30.0s"}
  ],
  "videos":["Y6R7T.mp4"]
}
```

#### Example 2 (ActivityNet-Captions).

Original annotation (compact):

```
{"video_id":"v_nwznKOuZM7w",
 "timestamps":[[0,15.51],[11.39,31.65]],
 "sentences":[
   "Two people are seen moving around a kitchen
       quickly performing various tasks and
       sitting down.",
   "They then wax down a ski in the kitchen
       while continuing to move around."
 ]}
```

# Reformatted instances:

```
{"messages":[
  {"role":"user",
   "content":"<video>What is the location of the
       moment: Two people are seen moving around
       a kitchen quickly performing various
       tasks and sitting down."},
  {"role":"assistant","content":"from 0s to
      15.51s"}
 ],
 "videos":["v_nwznKOuZM7w.mp4"]},
{"messages":[
  {"role":"user",
   "content":"<video>Can you tell me the time
       window of this event: They then wax down
       a ski in the kitchen while continuing to
       move around."},
  {"role":"assistant","content":"from 11.39s to
      31.65s"}
 ],
```

```
"videos":["v_nwznKOuZM7w.mp4"]}
```

## 13.3. Highlight Detection Task

Each HD training instance contains at least the video name, a query phrase, relevant temporal segments, and their saliency scores. The prompt templates and output format are summarized in Table [16.](#page-16-0) Specific examples are as follows:

## Example (QVHighlights).

Original annotation (compact):

```
{"qid":10016,
 "query":"Man in baseball cap eats before doing
     his interview.",
 "vid":"j7rJstUseKg_210.0_360.0",
 "relevant_windows":[[96,114]],
 "saliency_scores":[
   [2,3,3],[4,3,2],[2,3,1],[2,3,0],
   [2,3,3],[2,3,2],[2,3,1],[2,3,0],[1,3,3]
 ]}
```

#### Reformatted instance:

{

```
"messages": [
  {"role":"user",
   "content":"<video>Please highlight the most
       exciting parts related to: Man in
       baseball cap eats before doing his
       interview."},
  {"role":"assistant",
   "content":"The highlights are: important
       from 96.0s to 98.0s, 98.0s to 100.0s,
       104.0s to 106.0s; less important from
       100.0s to 102.0s, 102.0s to 104.0s,
       106.0s to 108.0s, 108.0s to 110.0s,
       110.0s to 112.0s, 112.0s to 114.0s"}
],
```

Table 16. Prompt templates and output format for HD task.

<span id="page-16-0"></span>

| Type             | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |  |  |  |  |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
| Prompt Templates | <video>Please highlight the most exciting parts related to:<br/>{query}.<br/><video>Find the most relevant or important moments for:<br/>{query}.<br/><video>Which moments in the video best reflect:<br/>{query}?<br/><video>Highlight the key segments that correspond to:<br/>{query}.<br/><video>Show the most interesting clips about:<br/>{query}.<br/><video>What are the highlight moments for:<br/>{query}?<br/><video>Mark the time intervals that are most significant for:<br/>{query}.</video></video></video></video></video></video></video> |  |  |  |  |
| Expected Output  | The highlights are:<br>very important from {start}s to {end}s,; important from<br>{start}s to {end}s,; less important from {start}s to {end}s,                                                                                                                                                                                                                                                                                                                                                                                                              |  |  |  |  |

```
"videos":["j7rJstUseKg_210.0_360.0.mp4"]
}
```

Based on the above methods, Grounding-FT reformulates heterogeneous VTG annotations into unified, instruction-response pairs. The diversity of prompt phrasing and conversational structure better aligns the dataset with large language model fine-tuning paradigms, leading to improved robustness and generalization.

# 14. Discussion on Early vs. Multi-Stage Token Sampling

In this work, VTS performs query-guided token sampling before multimodal fusion. We adopt this design because it is simple, efficient, and well aligned with the VTG setting: early suppression of query-irrelevant visual content helps reduce noise before constructing the joint representation. This also makes the sampling behavior more interpretable, since token relevance is estimated directly from the text query and visual features prior to deeper cross-modal interactions. At the same time, we acknowledge that later-layer or multi-stage sampling could leverage richer multimodal semantics and potentially improve token selection further. Such designs may offer a different trade-off between efficiency, interpretability, and representational power. We view this as an interesting direction for future work.

# 15. Additional Qualitative Analysis

To complement the qualitative study, we provide additional visualization examples for both GroundVTS-Q and GroundVTS-I. All examples follow the same visualization format as Figure 5 in the main paper, where the bottom curve denotes the normalized token density produced by the VTS module, with higher peaks indicating segments the model regards as more relevant to the query.

Across these cases, GroundVTS consistently exhibits highly accurate temporal localization. In Figure [7\(a\)](#page-17-0) (GT: 23.5–32.0 s), GroundVTS-Q predicts 24.0–31.9 s, aligning almost perfectly with the ground-truth, whereas QwenVL- G and the pretrained Qwen2.5VL shift the interval far earlier and fail to localize the correct moment. A similar trend appears in Figure [7\(b\)](#page-17-0) (GT: 0.0–5.9 s), where GroundVTS-Q outputs 0.0–5.8 s with near-exact precision, while both baselines truncate or deviate from the target boundary.

The InternVL-based examples exhibit the same trend. In Figure [8\(a\)](#page-18-0) (GT: 22.3–30.9 s), GroundVTS-I produces a tightly aligned prediction of 22.0–31.0 s, whereas InternVL-G shortens and shifts the interval (18.0–26.0 s), and the base InternVL3.5 mislocalizes the event to a distant region. In Figure [8\(b\)](#page-18-0) (GT: 0.0–7.0 s), GroundVTS-I again matches the ground-truth boundaries accurately (0.0– 6.9 s), while the baseline models either overextend the span or capture only a partial portion of the event.

These visualizations reveal three consistent advantages of GroundVTS. First, its temporal predictions are markedly more accurate and better aligned with annotated spans, regardless of model backbone or event duration. Second, its predicted spans consistently fall within the regions where the sampled token density reaches (local) maxima. In every example, the model's final prediction aligns with the peaks of the VTS density curve, indicating that GroundVTS relies on the most informative temporal segments identified by the sampling module. Third, the token allocation patterns produced by VTS are adaptive across different scenarios. In some cases (as illustrated in Figure 5 of the main paper), the density distribution forms sharp peaks with strong contrasts between attended and suppressed segments, typically corresponding to short or well-isolated grounding moments. In other cases, the differences between peaks and valleys are more moderate; nevertheless, the VTS curve still places a clear relative emphasis on the correct temporal region. These variations indicate that VTS does not rely on a fixed sparsity pattern but adjusts its sampling behavior according to the temporal structure of each video-query pair.

In addition, Figure [7\(a\)](#page-17-0) visualizes the spatial distribution of visual tokens selected by VTS. It can be observed that VTS mainly focuses on the middle and lower regions of the frames, which correspond to areas around human activities. When the action of watching TV occurs at the end of the

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 7. Additional qualitative comparison between GroundVTS-Q and its base models. Example (a) additionally illustrates spatial token retention maps, which correspond to spatial token selections.

video, the VTS module attends to most regions of the frame.

Overall, by concentrating tokens at the most semantically relevant moments while downweighting redundant frames, VTS enables GroundVTS to encode fine-grained temporal cues more effectively, leading to significantly sharper and more accurate temporal boundaries.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 8. Additional qualitative comparison between GroundVTS-I and its base models.