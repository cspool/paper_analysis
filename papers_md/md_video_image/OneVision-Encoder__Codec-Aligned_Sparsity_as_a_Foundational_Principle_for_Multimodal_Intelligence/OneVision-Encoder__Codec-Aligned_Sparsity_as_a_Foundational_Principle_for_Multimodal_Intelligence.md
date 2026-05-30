# OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence

Glint Lab, AIM for Health Lab, MVP Lab

Hypothesis. Artificial general intelligence is, at its core, a compression problem [\(Sutskever,](#page-28-0) [2023\)](#page-28-0). Effective compression demands resonance: deep learning scales best when its architecture aligns with the fundamental structure of the data. These are the fundamental principles. Yet, modern vision architectures have strayed from these truths: visual signals are highly redundant, while discriminative information, the surprise, is sparse. Current models process dense pixel grids uniformly, wasting vast compute on static background rather than focusing on the predictive residuals that define motion and meaning. We argue that to solve visual understanding, we must align our architectures with the information-theoretic principles of video, i.e., Codecs.

Method. OneVision-Encoder encodes video by compressing predictive visual structure into semantic meaning. By adopting Codec Patchification, OneVision-Encoder abandons uniform computation to focus exclusively on the 3.1%-25% of regions rich in signal entropy. To unify spatial and temporal reasoning under irregular token layouts, OneVision-Encoder employs a shared 3D RoPE and is trained with a large-scale cluster discrimination objective over more than one million semantic concepts, jointly capturing object permanence and motion dynamics.

Evidence. The results validate our core hypothesis: efficiency and accuracy are not a trade-off; they are positively correlated. By resolving the dichotomy between dense grids and sparse semantics, OV-Encoder redefines the performance frontier. When integrated into large multimodal models, it consistently outperforms strong vision backbones such as Qwen3-ViT and SigLIP2 across 16 image, video, and document understanding benchmarks, despite using substantially fewer visual tokens and pretraining data. Notably, on video understanding tasks, OneVision-Encoder achieves an average improvement of 4.1% over Qwen3-ViT. Under attentive probing, it achieves state-of-the-art representation quality, with 17.1% and 8.1% Top-1 accuracy improvements over SigLIP2 and DINOv3, respectively, on Diving-48 under identical patch budgets. These results demonstrate that codec-aligned, patch-level sparsity is not an optimization trick, but a foundational principle for next-generation visual generalists, positioning OneVision-Encoder as a scalable engine for universal multimodal intelligence.

Date: February 27, 2026

Code: <https://github.com/EvolvingLMMs-Lab/OneVision-Encoder>

Data: [https://github.com/EvolvingLMMs-Lab/OneVision-Encoder/blob/main/docs/data\\_card.md](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder/blob/main/docs/data_card.md)

Model: <https://huggingface.co/collections/lmms-lab-encoder/onevision-encoder>

### 1 Introduction

Transformer-based methods have achieved significant improvements in video understanding [\(Carreira et al.,](#page-24-0) [2024;](#page-24-0) [Wang et al.,](#page-28-1) [2024;](#page-28-1) [Soldan et al.,](#page-28-2) [2025;](#page-28-2) [Assran et al.,](#page-24-1) [2025;](#page-24-1) [Shu et al.,](#page-27-0) [2025;](#page-27-0) [Yang et al.,](#page-29-0) [2025\)](#page-29-0). By representing videos as sequences of visual tokens, these models have demonstrated a strong capacity to capture long-range spatial and temporal dependencies [\(Weng et al.,](#page-28-3) [2024;](#page-28-3) [Song et al.,](#page-28-4) [2025\)](#page-28-4). Reconstruction-based self-supervised frameworks (e.g., MAE [\(He et al.,](#page-26-0) [2021\)](#page-26-0), V-JEPA [\(Bardes et al.,](#page-24-2) [2024\)](#page-24-2)) emphasize pixelor feature-level prediction, which is effective for capturing low-level spatial and temporal correlations but often lacks explicit semantic structuring. In contrast, contrastive learning paradigms (e.g., CLIP [\(Radford](#page-27-1) [et al.,](#page-27-1) [2021\)](#page-27-1), SigLIP [\(Zhai et al.,](#page-29-1) [2023\)](#page-29-1)) focus on instance-level discrimination and typically rely on external language supervision to induce semantic separation, limiting their ability to model intra-class consistency and fine-grained inter-class relationships. Recent cluster discrimination methods [\(An et al.,](#page-24-3) [2023,](#page-24-3) [2024;](#page-24-4) [Xie](#page-28-5)

#### **Predictive Video Structure**

### **OneVision Encoder**

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1 Visual intelligence as codec-aligned predictive compression. Visual intelligence as a compression problem, where scalable learning emerges from alignment with the predictive structure of the world. Video exemplifies this principle: most visual content is redundant and predictable, while meaningful information arises sparsely as motion and residual change. Video codecs make this structure explicit by decomposing visual signals into stable spatial context and sparse temporal updates. Grounded in this codec principle, OV-Encoder reframes visual modeling as predictive compression, serving as a scalable engine for universal multimodal intelligence that sees, updates, and reasons over time.

et al., 2025; Tang et al., 2025) address this gap by encouraging semantically related entities to form coherent clusters, pointing toward structured object-centric representation learning. Despite these advances, existing video transformers predominantly rely on representations constructed from sparsely sampled frames, leading to dense token sequences that implicitly assume equivalence across spatial regions and temporal frames. Such a frame-centric design reflects a prevailing modeling assumption in video pretraining and motivates rethinking how visual evidence is structured in video signals.

At its core, general intelligence is a compression problem. Natural videos are highly redundant, exhibiting strong spatial and temporal regularities. As a result, the majority of the visual content in a video is predictable from its surrounding context rather than constituting new discriminative evidence. However, standard video pretraining strategies rely on uniform computation over dense pixel grids, expending substantial capacity on static or easily inferred background regions. Discriminative information, the *surprise*, is sparse. This sparsity is not a modeling artifact, but a property of the signal itself. Video compression makes this explicit. Video codecs such as H.264 and H.265/HEVC (*High Efficiency Video Coding*) decompose video signals into spatially complete intra-coded frames (I-frames) that establish global context and predicted frames (P-frames) that encode inter-frame variations via motion compensation and residuals (Sullivan et al., 2012). This codec-driven formulation reveals that the vast majority of the video signal corresponds to motion-driven incremental updates to existing spatial context rather than independently discriminative visual evidence. In other words, visual understanding is governed by sparse, localized evidence that defines motion and meaning, rather than dense grids of uniformly processed pixels.

These observations lead to a unified conclusion: to scale visual intelligence, architectures must align with the information-theoretic structure of the data. In this work, we present **OneVision-Encoder (OV-Encoder)**, a

HEVC-style Vision Transformer that aligns spatiotemporal representation learning with the intrinsic predictive structure of video signals, as illustrated in Figure [1.](#page-1-0) Rather than uniformly processing dense pixel grids, OV-Encoder explicitly determines which visual signals constitute independent evidence and selectively encodes only the regions rich in signal entropy. To enable this, we introduce Codec Patchification, a codec-inspired input formulation that leverages temporal signals exposed by video codecs to organize visual tokens at the patch level, together with a 3D Rotary Position Embedding (RoPE) [\(Su et al.,](#page-28-8) [2024\)](#page-28-8) that jointly encodes spatial and temporal positions to support coherent attention over irregular spatiotemporal layouts. Specifically, (a) Dense Video-Codec Patchification: a codec-inspired video encoding formulation that leverages motion-centric temporal signals exposed by P-frames to patchify selected visual regions (3.1%-25%) in dense video inputs, while preserving dense temporal coverage. (b) Chunk-wise Patchification: a codec-inspired temporal patchification scheme that partitions video streams into fixed-length chunks and constructs patch-level representations with chunk-level positional encoding. (c) Single-Image Spatial Patchification: a spatial instantiation of Codec Patchification that constructs patch-level representations for single-image input, enabling structured modeling of static visual content. Furthermore, explicitly modeling visual evidence at the patch level requires a training objective that enforces semantic structure. We adopt a self-supervised cluster discrimination objective based on large-scale semantic clustering over a concept bank with more than one million clusters, jointly capturing object-level permanence and motion dynamics. In particular, OV-Encoder provides a bi-directional attention-based vision encoder that effectively supports image and video understanding.

Extensive experiments demonstrate the efficacy of OV-Encoder across both multimodal and representationlevel evaluation protocols. For comparison with SigLIP2 [\(Tschannen et al.,](#page-28-9) [2025\)](#page-28-9), all models are assessed under identical multimodal fine-tuning conditions, employing a 1.5M-scale LLaVA-Next (single-image instruction) [\(Liu et al.,](#page-26-1) [2024a\)](#page-26-1) and LLaVA-Next-Videos [\(Zhang et al.,](#page-29-2) [2024\)](#page-29-2) instruction-tuning corpus, together with a native-resolution processing strategy. Within this experimental configuration, OV-Encoder outperforms SigLIP2 across 16 benchmarks spanning video, image, and document understanding tasks, when evaluated using large multimodal models (LMMs) built upon Qwen3-4B [\(Bai et al.,](#page-24-5) [2025\)](#page-24-5).

For comparison with Qwen3-ViT [\(Bai et al.,](#page-24-5) [2025\)](#page-24-5), we adopt a controlled evaluation protocol. Specifically, we integrate OneVision-Encoder with the Qwen3-1.7B language model and train it under the LLaVA-OneVision-1.5 [\(An et al.,](#page-24-6) [2025\)](#page-24-6) framework, completing both Stage 1 and Stage 1.5 to adapt the encoder to native-resolution inputs. The trained OneVision-Encoder is then decoupled and compared with Qwen3-ViT under the same LLaVA-Next-Videos instruction-tuning training setting, where OV-Encoder outperforms Qwen3-ViT across 16 understanding benchmarks under an LMM built upon Qwen3-4B. In particular, despite having been pretrained on substantially fewer visual–text tokens (approximately 100B caption tokens), OV-Encoder outperforms Qwen3-ViT, which is pretrained on more than 2.1T caption and instruction-aligned tokens.

Under attentive probing on 7 benchmarks, OV-Encoder achieves state-of-the-art performance, including a 17.1% and 8.1% Top-1 accuracy improvements over SigLIP2 and DINOv3 [\(Siméoni et al.,](#page-27-2) [2025\)](#page-27-2), respectively, on Diving-48 under an identical patch budget of 2048. Moreover, OV-Encoder outperforms strong vision baselines such as DINOv3, SigLIP2, MetaCLIP2 [\(Chuang et al.,](#page-25-0) [2025\)](#page-25-0), and AIMv2 [\(Fini et al.,](#page-25-1) [2025\)](#page-25-1) under dense-patch evaluation. We release our data, training protocols, and model parameters to support transparent, reproducible, and cost-effective vision–language research. Our contributions are as follows:

- We present OneVision-Encoder (OV-Encoder), a HEVC-style vision transformer that aligns spatiotemporal representation learning with the intrinsic predictive structure of video signals through Codec patch-level encoding.
- We introduce Codec Patchification, a codec-inspired input formulation that leverages codec-derived temporal signals to selectively encode informative visual patches (3.1%-25%) from dense video, while unifying video, chunk-wise sampling, and single-image inputs with 3D-RoPE.
- We adopt a self-supervised cluster discrimination objective that jointly models object-level and motionlevel semantics with a large-scale concept bank, enabling structured and modality-agnostic visual representation learning.
- Extensive experiments establish the effectiveness of OV-Encoder across evaluation settings. Under LLM-based probing, the model consistently surpasses strong vision backbones, including Qwen3-ViT and SigLIP2, across multimodal benchmarks. Under attentive probing, OV-Encoder outperforms SigLIP2

### 2 Approach

Although most previous video encoders focus on short clips of 16 frames (roughly seconds) (Bardes et al., 2023; Wang et al., 2024), we explore training with longer clips of up to 64 frames at higher spatial resolutions. Let  $\mathcal{V}_i = \{\mathcal{I}_{i,t}\}_{t=1}^{T_i}$  denote the *i*-th input video of spatial size  $H \times W$ , where  $T_i$  is the total number of frames in  $\mathcal{V}_i$ . Our objective is to process raw video inputs within the proposed different input configurations and jointly encode them using a shared ViT for unified spatiotemporal representation learning.

#### 2.1 HEVC Guided Patch Selection

Codec Based Video Factorization. Following the standard High Efficiency Video Coding formulation (Sullivan et al., 2012), each video  $V_i$  is divided into  $N_i$  Groups of Pictures (GOP),  $\{S_{i,n}\}_{n=1}^{N_i}$ . Applying the HEVC codec to each Groups of Pictures yields one intra-coded frame and  $(K_{i,n}-1)$  predicted frames:

$$(F_{i,n}^{\mathrm{I}}, \{F_{i,n,\tau}^{\mathrm{P}}\}_{\tau=1}^{K_{i,n}-1}) = \mathcal{C}_{\mathrm{HEVC}}(\mathcal{S}_{i,n}),$$
 (1)

where  $F_{i,n}^{\rm I} \in \mathbb{R}^{H \times W \times C_{\rm img}}$  is the I-frame (RGB,  $C_{\rm img} = 3$ ), and each  $F_{i,n,\tau}^{\rm P}$  denotes a P-frame, which is represented in the bitstream by motion vectors and a residual signal.  $K_{i,n}$  denotes the GOP length.

Motion and Residual Signals. In HEVC, motion is represented by motion vectors  $d_{i,n,\tau}$  that encode block level displacements between the current frame and its motion compensated prediction from the reference frame. Concretely, P-frames are partitioned into coding units (CUs) with variable sizes ranging from  $4\times4$  to  $64\times64$ , and all pixels within a CU share the same motion vector. To align codec signals with ViT patchification, we first broadcast each CU motion vector to its covered pixels, obtaining a dense pixel level motion field. The magnitude  $\|d_{i,n,\tau}\|_2$  reflects the intensity of local motion, with larger values indicating stronger or more complex dynamics. In addition to motion, each P-frame is associated with a residual signal that captures appearance changes not explained by motion compensation; we decode the luma residual into the pixel domain and measure its energy as a complementary cue for unpredictable visual variation. At the patch level, we aggregate motion magnitude and residual energy over all pixels inside each ViT patch, so the two signals jointly characterize the amount of new visual evidence introduced by a region. We use these codec exposed signals as a principled proxy for unpredictability, enabling the identification of salient regions that contribute new spatiotemporal information.

**Sparse Patch Selection.** Let p denote the patch size (e.g., p=14). We define the patch grid  $\mathcal{G} = \{(y,x) \mid 0 \leq y < H/p, \ 0 \leq x < W/p\}$ , with cardinality  $P_0 = (H/p)(W/p)$ . For each P-frame, we compute a patch level saliency score by aggregating the codec exposed motion magnitude and residual energy defined above. Based on the aggregated saliency score, we construct a binary mask  $\Omega_{i,n,\tau} \subseteq \mathcal{G}$  by selecting a fixed proportion of the most salient patches. The sparsity ratio is fixed throughout training and inference by selecting a fixed proportion r of the most salient patches, i.e.,  $|\Omega_{i,n,\tau}| = |rP_0|$ .

#### 2.2 Codec Patchification

**Dense Video-Codec Patchification.** Following the codec formulation, each video  $\mathcal{V}_i$  is partitioned into  $N_i$  GOP. For the n-th GOP, the HEVC encoder produces one intra-coded frame  $F_{i,n}^{\mathrm{I}}$  and  $(K_{i,n}-1)$  predicted frames  $\{F_{i,n,\tau}^{\mathrm{P}}\}_{\tau=1}^{K_{i,n}-1}$  that are defined by motion vectors and a residual signal after motion compensation. Let  $\Omega_{i,n,\tau}$  denote the codec-derived binary mask that selects dynamically informative patches based on motion magnitude and residual energy, and let  $\Pi_p(\cdot)$  denote patchification with patch size p. The HEVC-compressed input sequence is defined as

$$\mathcal{F}_{i,n}^{(\text{hevc})} \triangleq \left[ \Pi_p(F_{i,n}^{\text{I}}) \oplus \left\{ \Pi_p(\bar{F}_{i,n,\tau}^{\text{P}})[\Omega_{i,n,\tau}] \right\}_{\tau=1}^{K_{i,n}-1} \right], \tag{2}$$

where  $\bar{F}_{i,n,\tau}^{P} \in \mathbb{R}^{H \times W \times C_{\text{img}}}$  denotes the decoded RGB P-frame in the pixel domain obtained by decoding the HEVC bitstream. The binary mask  $\Omega_{i,n,\tau}$  is computed from the codec-exposed motion vectors  $\boldsymbol{d}_{i,n,\tau}$  and

![](_page_4_Figure_0.jpeg)

Utilize video codec structure for efficient spatiotemporal feature extraction.

Build a shared-parameter unified encoder to process video and images.

Through contrastive learning, enable the model to simultaneously master action and category recognition.

Figure 2 Overview of the OneVision-Encoder framework. Left: Input formulation. The framework integrates three Codec Patchification strategies: Dense Video-Codec Patchification, Chunk-wise Patchification, and (Sigle-Image/Frame) Spatial Patchification. All inputs are processed by a shared-parameter OneVision-Encoder. Right: Unified cluster discrimination objective. Image and video embeddings are aligned through contrastive learning against a global set of class centers, jointly optimizing object-level and action-level representations within a single encoder.

the associated motion-compensated residual signal of the same P-frame, and is used only for salient patch selection. Therefore, the tokens fed to the encoder for P-frames are RGB patches from  $\bar{F}_{i,n,\tau}^{P}$  indexed by  $\Omega_{i,n,\tau}$ .  $K_{i,n}$  is the GOP length.  $[\Omega]$  denotes masked patch selection along the patch dimension.  $\oplus$  denotes concatenation along the token dimension. Let  $M_{i,n}$  be the number of tokens in  $\mathcal{F}_{i,n}^{(\text{hevc})}$ ; the total tokens over the video satisfy:

$$M_{i} = \sum_{n=1}^{N_{i}} M_{i,n} = \sum_{n=1}^{N_{i}} \left( P_{0} + \sum_{\tau=1}^{K_{i,n}-1} |\Omega_{i,n,\tau}| \right), \tag{3}$$

which yields a pixel/token compression ratio per GOP,  $\gamma_{i,n} = 1 - M_{i,n}/(K_{i,n}P_0)$ . Under our default setting (64 frames, GOP size 32, token budget 2048,  $P_0 = 256$ ), the overall clip-level token reduction is  $1 - 2048/(64P_0) = 87.5\%$ .

**Chunk-wise Patchification.** To unify Codec patchification with sparse temporal sampling, each video  $V_i$  is uniformly partitioned into C temporal chunks of  $\lfloor T_i/C \rfloor$  consecutive frames. From every chunk, one frame is randomly sampled, resulting in a temporally stratified subsequence  $\pi_i = \{t_c\}_{c=1}^C$ , where  $t_c \in [(c-1)|T_i/C|, c|T_i/C|)$ . The corresponding chunk-wise sampling sequence is defined as

$$\mathcal{F}_i^{\text{(chunk)}} \triangleq \left\{ \Pi_p(\mathcal{I}_{i,t_c}) \mid t_c \in \pi_i \right\} = \Pi_p(\mathcal{V}_i[\pi_i]). \tag{4}$$

Single-Image Spatial Patchification. To achieve spatial scalability, each image  $\mathcal{I}_{i,t} \in \mathbb{R}^{H \times W \times C_{\mathrm{img}}}$  is processed independently as a static image input. Each image is directly patchified in a row-wise manner from top to bottom to ensure a deterministic spatial ordering of visual tokens, preserving the original spatial layout. The resulting patch sequence is defined as

$$\mathcal{F}_i^{\text{(image)}} \triangleq \left\{ \Pi_p(\mathcal{I}_{i,t}) \mid t = 1, \dots, T_i \right\}, \tag{5}$$

where  $\Pi_p(\cdot)$  denotes the patchification operator applied to a single frame, and  $T_i = 1$  for single-image inputs. A detailed spatial bias analysis is provided in the supplementary material.

**Tokenization and Transformer Encoding.** After generating the three types of input sequences, namely the Codec Patchification formulation, we uniformly tokenize and encode them using a shared Transformer backbone  $\phi(\cdot)$ . For each video input vid  $\in$  {hevc, chunk, image}, the token sequence  $\mathcal{F}_i^{\text{vid}} = \{x_{i,k}^{\text{vid}}\}_{k=1}^{M_i^{\text{vid}}}$  is processed by

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

(a) Standard contrastive learning methods

(b) Cluster discrimination learning methods

Figure 3 Contrastive learning vs. cluster discrimination. (a) Standard contrastive learning contrasts samples against batch-local negatives, constraining the view of the embedding space. (b) Cluster discrimination contrasts samples against a global concept bank of clustered centers at scale, yielding discriminative and structurally separated representations.

the encoder  $\phi(\cdot)$  followed by a linear projection head  $W \in \mathbb{R}^{d \times D}$  where d and D denote hidden and latent dimensions of the encoder, respectively, to obtain latent embeddings:

$$E_i^{\text{vid}} = \phi(\mathcal{F}_i^{\text{vid}})W = Z_i^{\text{vid}}W.$$
 (6)

which defines  $Z_i^{\text{vid}} = \phi(\mathcal{F}_i^{\text{vid}}) \in \mathbb{R}^{M_i^{\text{vid}} \times d}$  as the encoder output features before projection. Finally, a function  $f(\cdot)$ , such as attentive pooling, integrates all token embeddings into a compact video-level representation  $e_i^{\text{vid}} = f(E_i^{\text{vid}}) \in \mathbb{R}^D$ .

#### 2.3 Image and Video Clustering

Iterative clustering-discrimination approaches commonly suffer from substantial computational overhead (Caron et al., 2018). To address this issue, we adopt a single-step offline clustering to efficiently capture both object-level semantics from images and motion-level semantics from videos. Notably, embeddings used for clustering are extracted using a separate frozen vision encoder (e.g., metaclip-h14 (Xu et al., 2024))

For the image modality, we follow the clustering formulation in MLCD (An et al., 2024) to capture object-level semantics. Given an image embedding  $e_i^{\text{obj}} \in \mathbb{R}^D$ , we learn a set of object-level semantic centroids  $\mathcal{C}_{\text{obj}} = \{c_k^{\text{obj}}\}_{k=1}^{K_{\text{obj}}} \subseteq \mathbb{R}^D$  by minimizing the within-cluster distance:

$$C_{\text{obj}} = \arg \min_{\{c_k^{\text{obj}}\}} \sum_{i=1}^{N_{\text{obj}}} \min_{1 \le k \le K_{\text{obj}}} \left\| e_i^{\text{obj}} - c_k^{\text{obj}} \right\|_2^2, \tag{7}$$

where  $N_{\text{obj}}$  denotes the number of image samples.

For the video modality, we extend this formulation to model motion-level dynamics. Video embeddings are derived from fixed-length 16-frame inputs, where frame-level features are concatenated to form a single video-level representation  $e_i^{\text{vid}} \in \mathbb{R}^D$ . We then define a set of video semantic centroids  $C_{\text{vid}} = \{c_k^{\text{vid}}\}_{k=1}^{K_{\text{vid}}} \subseteq \mathbb{R}^D$ , and formulate the clustering objective for videos as

$$C_{\text{vid}} = \arg\min_{\{c_k^{\text{vid}}\}} \sum_{i=1}^{N_{\text{vid}}} \min_{1 \le k \le K_{\text{vid}}} \|e_i^{\text{vid}} - c_k^{\text{vid}}\|_2^2,$$
 (8)

where  $N_{\text{vid}}$  denotes the number of video samples. We define a set of shared semantic centroids  $C_{uni} = \{c_k^{\text{obj}}\}_{k=1}^{K_{\text{obj}}} \cup \{c_k^{\text{vid}}\}_{k=1}^{K_{\text{vid}}}$ , and  $K = K_{\text{obj}} + K_{\text{vid}}$  represents the total number of clusters across both image and video modalities. These semantic centroids are later used as supervision signals in the cluster discrimination objective, where image embeddings are supervised by object-level centroids and video embeddings are supervised by motion-level centroids.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 4 3D-RoPE for Codec Patchification. A unified relative positional encoding scheme is adopted for Codec Patchification. (a) encodes full spatiotemporal offsets  $(\Delta t, \Delta x, \Delta y)$  over I/P-frame sequences to preserve motion-driven inter-frame structure. (b) defines temporal offsets at the chunk level, enabling structured reasoning under non-uniform temporal sampling. (c) degenerates the formulation to purely spatial offsets  $(0, \Delta x, \Delta y)$  for static inputs. 3D-RoPE preserves structural consistency, enabling coherent attention over sparse and irregular token layouts.

### 2.4 Training Objective

Visual samples commonly exhibit multiple semantic components, including object-level semantics from images and motion-level semantics from videos, rendering single-label assignments inadequate for unified representation learning. To capture both semantic structures, we introduce a contrastive objective that leverages these semantic clusters as pseudo-label supervision to explicitly enforce structural constraints, as illustrated in Figure 3. Specifically, for each visual embedding  $e_i \in \mathbb{R}^D$ , we identify multiple positive semantic labels from a unified semantic centroid set  $C_{uni}$ , which consists of both object-level centroids  $C_{obj}$  and motion-level centroids  $C_{vid}$ . We compute the training objective separately for each semantic granularity  $m \in \{\text{obj}, \text{vid}\}$ , where negative labels are drawn from the corresponding centroid set  $C_m$ . The remaining centroids within the same granularity are treated as negative labels. Subsequently, the joint multi-label semantic discrimination objective is formulated as:

$$\mathcal{L} = \sum_{m \in \{\text{obj,vid}\}} \mathbb{E}_{(u,k) \sim \mathcal{C}_m} \log \left( 1 + \exp(-y_{u,k}^m \sigma_{u,k}^m) \right), \tag{9}$$

where  $\sigma_{u,k}^m = e_u^\top c_k^m$  denotes the similarity score between the embedding  $e_u$  and its semantic centroid  $c_k^m$ , and u indexes visual embeddings while k indexes centroids in  $\mathcal{C}_m$ . For each granularity m,  $(u,k) \sim \mathcal{C}_m$  samples visual embeddings  $e_u$  and semantic centroids  $c_k^m$  from the corresponding cluster set, with pseudo-labels  $y_{u,k}^m \in \{+1,-1\}$  indicating positive or negative semantic associations. This unified formulation integrates both object- and motion-level clustering signals, enforcing spatiotemporal consistency and promoting discriminative representation learning.

### 2.5 Architecture

For the OneVision-Encoder, we adopt the Vision Transformer (ViT) architecture (Dosovitskiy et al., 2020).

**3D-RoPE for Codec Patchification.** Unlike absolute encoding p=(t,x,y), 3D-RoPE adopts a relative positional scheme (Su et al., 2024) represented as  $\Delta p=(t_1-t_2,x_1-x_2,y_1-y_2)$ , as illustrated in Figure 4. The relative offsets  $\Delta p$  for the three inputs are defined as:

• Dense Video-Codec Patchification:  $\Delta p_{\text{codec}} = (t_i - t_j, x_i - x_j, y_i - y_j)$ , which emphasizes inter-frame (I/P) residual alignment via the temporal offset  $t_i - t_j$ .

Table 1 OneVision-Encoder Pretraining Dataset. The pretraining corpus combines large-scale image and video datasets for unified visual representation learning. Image datasets primarily provide broad visual coverage, while video datasets support temporal modeling and video–language alignment. We use "ExoVideo" to denote large-scale third-person web videos, and "ActionVideo" to denote curated action recognition datasets.

| Source                              | Samples | Type          | Modality | Temporal | Curation |
|-------------------------------------|---------|---------------|----------|----------|----------|
| LAION-400M (Schuhmann et al., 2021) | 250M    | WebImages     | Image    | –        | Yes      |
| COYO-700M (Byeon et al., 2022)      | 400M    | WebImages     | Image    | –        | Yes      |
| OBELICS (Laurençon et al., 2023)    | 15M     | Documents     | Image    | –        | Yes      |
| Zero250M (Xie et al., 2023)         | 15M     | CuratedImages | Image    | –        | Yes      |
| ImageNet-21K (Deng et al., 2009)    | 14M     | Images        | Image    | –        | Yes      |
| HowTo100M (Miech et al., 2019)      | 50M     | ExoVideo      | Video    | Short    | No       |
| Panda-70M (Chen et al., 2024b)      | 50M     | ExoVideo      | Video    | Long     | Yes      |
| Kinetics-710 (Li et al., 2022b)     | 658K    | ActionVideo   | Video    | Short    | Yes      |
| SSV2 (Goyal et al., 2017)           | 221K    | ActionVideo   | Video    | Short    | Yes      |

- Chunk-wise Patchification: for frames t<sup>i</sup> , t<sup>j</sup> belonging to chunks c<sup>i</sup> , c<sup>j</sup> ∈ {1, . . . , C}, the relative offset is defined as ∆pchunk = (∆c, ∆x, ∆y), capturing inter-chunk temporal disparity under non-uniform sampling, where ∆c = c<sup>i</sup> − c<sup>j</sup> .
- Single-Image Spatial Patchification: for two patches within the same frame t with spatial coordinates (x<sup>i</sup> , yi) and (x<sup>j</sup> , y<sup>j</sup> ), the relative positional offset is defined as ∆pspatial = (0, ∆x, ∆y), where ∆x = x<sup>i</sup> − x<sup>j</sup> and ∆y = y<sup>i</sup> − y<sup>j</sup> , encoding spatial positional relationships without temporal shifts.

Attentive Pooling Head. We employ a multi-head attention pooling module, adapted from SigLIP [\(Zhai et al.,](#page-29-1) [2023\)](#page-29-1), to aggregate spatiotemporal tokens into compact class embeddings through learnable token-to-class attention weights, emphasizing salient regions and enabling unified global contextual representation across image and video modalities.

# 3 Pretraining Dataset

Data Annotation and Processing. This section details the annotation and processing pipeline for the OneVision-Encoder pretraining data. Our core objective is to generate high-quality supervision signals for massive-scale data through automated means.

Image Data Annotation. For image data, we primarily process LAION-400M [\(Schuhmann et al.,](#page-27-5) [2022\)](#page-27-5) and COYO-700M [\(Byeon et al.,](#page-24-9) [2022\)](#page-24-9). First, we employ a Union-Find algorithm to strictly deduplicate the dataset. Subsequently, we utilize the metaclip-h14-fullcc2.5b [\(Xu et al.,](#page-28-10) [2024\)](#page-28-10) model to extract image features and cluster all images into two million classes. Based on this clustering, each image sample is annotated with the nearest Top-10 class centers as its multi-label supervision signal. Furthermore, we incorporate the OBELICS [\(Laurençon et al.,](#page-26-2) [2023\)](#page-26-2) and Zero250M [\(Xie et al.,](#page-28-11) [2023\)](#page-28-11) datasets. We utilize PaddleOCR [\(Sarkar](#page-27-6) [et al.,](#page-27-6) [2024\)](#page-27-6) to recognize text within images and perform word segmentation on the recognized content; the resulting vocabulary is used as multi-labels to construct a supervision signal containing exactly 100 fine-grained tags per image.

Video Data Annotation. The video data construction encompasses HowTo100M [\(Miech et al.,](#page-27-4) [2019\)](#page-27-4), Panda-70M [\(Chen et al.,](#page-25-4) [2024b\)](#page-25-4), Kinetics-710 [\(Li et al.,](#page-26-3) [2022b\)](#page-26-3), and Something-Something-V2 (SSV2) [\(Goyal et al.,](#page-25-5) [2017\)](#page-25-5). We uniformly adopt metaclip-h14-fullcc2.5b as the video encoder, performing uniform frame sampling to extract features from a fixed 8-frame clip. In the feature processing stage, we adopt a "L2 Normalize → Concatenate → L2 Normalize" strategy to generate video-level representations. We then cluster the video representations into 400k classes and assign each video clip the nearest Top-10 class centers as its multi-labels.

### 4 Experiments

### <span id="page-8-0"></span>4.1 Pretraining OneVision-Encoder

We adopt a two-stage pretraining pipeline using large-scale image, video, and OCR data, trained on 128 A800 GPUs (16 nodes × 8 GPUs).

Stage 1: For the image model, we use images at a resolution of 224. We adopt the AdamW optimizer with a learning rate of 0.001 and a weight decay of 0.2. The number of classes (k) is two million, the ratio of sampled negative class centers (r) is 0.1, and the number of positive labels (l) assigned to each image (and each video) is 10. In this stage, we trained on 13B samples using only image data.

Stage 2: In the second pre-training stage, we introduced OCR data and video data, training on 4B samples with an image resolution of 448 and video resolution of 224 (frame sampling). For video processing, we randomly adopted one of Codec Patchification formulations. For dense video-Codec inputs, each training sample corresponds to a fixed-length clip of 64 consecutive frames. We follow an HEVC-style GOP configuration with an I-frame every 32 frames, resulting in two I-frames and sixty-two P-frames per clip. All I-frames are fully encoded, retaining all spatial patches to establish complete spatial context. For the remaining P-frames, codec-derived motion vectors and residuals are used to identify temporally salient regions, and patches are selected sparsely across all P-frames. Importantly, the patch budget is enforced at the clip level rather than the GOP level. While GOPs define the I/P frame structure, the total number of visual tokens for the entire 64-frame clip is fixed to 2,048. Specifically, 512 tokens are allocated to the two I-frames (256 patches each), and the remaining 1,536 tokens are distributed across all P-frames by selecting motion-relevant residual patches. This results in an 87.5% reduction compared to dense processing of all 64 frames (16,384 patches), while preserving full temporal coverage. All samples shared the same ViT backbone, with loss calculated separately for each modality based on their annotations. The video-to-image ratio was 1:1, and the learning rate was reduced to 0.0001.

Training Strategy. During the training phase, all data sources share the same OneVision-Encoder for feature extraction. However, when computing the loss, the loss is calculated separately for each data type and then aggregated. Detailed training configurations and implementation specifics are provided in the supplementary material.

### 4.2 LMM Probing Evaluation - Language Alignment

In this section, we evaluate the effectiveness of OV-Encoder when integrated into LMMs. Our goal is to assess whether the learned visual representations transfer robustly to multimodal reasoning settings, while isolating the contribution of the vision encoder from language model capacity and training protocol. We lift these features through alignment tuning to construct a new codec encoder, OV-Encoder-Lang, specialized for MLLMs. The controlled evaluation pipeline is illustrated in Figure [7](#page-19-0) in the supplementary material.

MLLM Evaluation Tasks. We adapt vision encoders to large multimodal models and evaluate downstream performance on a set of image- and video-centric benchmarks using LMMs-Eval [\(Zhang et al.,](#page-29-3) [2025\)](#page-29-3) with the default prompt. Image tasks include ChartQA [\(Masry et al.,](#page-27-7) [2022\)](#page-27-7), DocVQA [\(Mathew et al.,](#page-27-8) [2021\)](#page-27-8), InfoVQA [\(Mathew et al.,](#page-27-9) [2022\)](#page-27-9), AI2D [\(Kembhavi et al.,](#page-26-4) [2016\)](#page-26-4), MMBench-EN [\(Fu et al.,](#page-25-6) [2023\)](#page-25-6), OCRBench [\(Liu](#page-27-10) [et al.,](#page-27-10) [2024b\)](#page-27-10), OCRBench v2 [\(Liu et al.,](#page-27-10) [2024b\)](#page-27-10), MMStar [\(Chen et al.,](#page-25-7) [2024a\)](#page-25-7), and RealWorldQA [\(Corp.,](#page-25-8) [2024\)](#page-25-8). Video tasks include MVBench [\(Li et al.,](#page-26-5) [2023a\)](#page-26-5), MLVU-dev [\(Zhou et al.,](#page-29-4) [2024\)](#page-29-4), NExT-QA [\(Xiao et al.,](#page-28-12) [2021\)](#page-28-12), VideoMME [\(Fu et al.,](#page-25-9) [2024\)](#page-25-9), PerceptionTest [\(Patraucean et al.,](#page-27-11) [2023\)](#page-27-11), TOMATO [\(Shinoda et al.,](#page-27-12) [2025\)](#page-27-12), and LongVideoBench-Val-Video [\(Wu et al.,](#page-28-13) [2024\)](#page-28-13).

Experimental Setup. Following a probing-based fine-tuning paradigm, we keep the language model architecture fixed and vary only the visual encoder. Unless otherwise specified, all experiments are conducted with Qwen3-4B-Instruct2507 as the language backbone. We adopt a two-stage training pipeline: first, a Stage-1 alignment phase, followed by a Stage-2 instruction-tuning phase. The instruction-tuning corpus consists of approximately 1.5M samples, including 740K single-image instruction from LLaVA-Next and 800K samples from LLaVA-Next-Videos. This unified instruction-following setting enables a controlled comparison across different vision encoders under identical multimodal supervision.

<span id="page-9-0"></span>Table 2 Comparison of different vision encoders onmultimodal benchmarks. All models are evaluated on a unified multimodal setting using Qwen3-4B-Instruct2507 as the language backbone. OV-Encoder-Lang denotes the language-aligned variant of the OV-Encoder architecture. Qwen3-ViT is extracted from Qwen3-VL-4B. SigLIP2 uses siglip2-so400mpatch16-naflex. (Codec) indicates codec-guided visual encoding using motion vectors and residual signals, while (Frame) indicates frame-based visual encoding with dense spatial patchification. Bold values indicate the best performance under the same evaluation setting. Results reported in the left columns correspond to encoders trained with caption supervision, whereas results in the right columns correspond to encoders trained without caption supervision.

| Task  | Benchmark                |                            |                      | Qwen3-4B-Instruct2507 |                             |                    |
|-------|--------------------------|----------------------------|----------------------|-----------------------|-----------------------------|--------------------|
|       |                          | OV-Encoder-Lang<br>(Codec) | Qwen3-ViT<br>(Frame) | OV-Encoder<br>(Codec) | OV-Encoder-Frame<br>(Frame) | SigLIP2<br>(Frame) |
|       | MVBench                  | 53.2                       | 47.4                 | 52.4                  | 49.8                        | 47.2               |
|       | MLVU-dev                 | 47.4                       | 47.2                 | 46.3                  | 49.4                        | 48.4               |
|       | NExT-QA (MC)             | 76.1                       | 70.1                 | 75.6                  | 71.9                        | 70.6               |
| Video | VideoMME                 | 54.1                       | 47.2                 | 53.4                  | 49.3                        | 46.8               |
|       | Perception Test          | 60.6                       | 57.1                 | 60.3                  | 56.7                        | 56.0               |
|       | TOMATO                   | 21.8                       | 22.2                 | 22.2                  | 21.8                        | 22.3               |
|       | LongVideoBench-Val-Video | 51.6                       | 45.0                 | 50.4                  | 45.5                        | 45.2               |
|       | AI2D                     | 80.2                       | 77.8                 | 75.7                  | 76.5                        | 78.6               |
|       | ChartQA                  | 80.1                       | 79.6                 | 76.5                  | 77.8                        | 76.4               |
|       | DocVQA                   | 83.2                       | 85.1                 | 78.4                  | 79.5                        | 75.0               |
|       | InfoVQA                  | 51.6                       | 49.0                 | 43.1                  | 45.5                        | 42.0               |
| Image | MMBench-EN               | 80.2                       | 79.4                 | 77.2                  | 78.5                        | 79.6               |
|       | OCRBench                 | 657                        | 706                  | 605                   | 630                         | 621                |
|       | OCRBench v2              | 30.8                       | 30.6                 | 26.3                  | 26.1                        | 26.1               |
|       | MMStar                   | 56.6                       | 56.6                 | 52.1                  | 54.3                        | 55.0               |
|       | RealWorldQA              | 66.1                       | 63.3                 | 60.8                  | 61.2                        | 62.1               |

### 4.2.1 Native-Resolution Evaluation.

We adopt a native-resolution evaluation strategy following LLaVA-Next, with a key distinction: input frames matching the native resolution of the vision encoder are processed directly without spatial tiling or cropping. For video inputs, we use a fixed per-frame resolution, set to 512×512 for SigLIP2 and Qwen3-ViT, and 504×504 for OneVision-Encoder. This design avoids resolution-induced artifacts and enables a direct assessment of the encoder's native-resolution modeling capability under realistic multimodal inference conditions.

Codec-based Patch Sampling. We evaluate the same codec-guided patchification principle as described in Stage 2 of Section [4.1](#page-8-0) under a strictly controlled token budget. For each test video, we uniformly sample 64 frames over the full duration to obtain broad temporal coverage. We do not re-encode or transcode benchmark videos; compressed-domain signals (e.g., motion vectors and residuals) are extracted directly from the original bitstreams, thus preserving the native GOP structures and codec parameters of each dataset. Based on these signals, we compute a lightweight saliency score to estimate temporally informative regions and select patches sparsely across the sampled frames. The selected patches are then packed into a fixed number of visual tokens. Importantly, we keep the total token budget identical to the dense baseline that encodes only 8 frames, ensuring that performance differences reflect improved token allocation rather than increased token count.

Comparison with SigLIP2. We first compare OneVision-Encoder with SigLIP2 under identical multimodal fine-tuning conditions, as shown in Table [2.](#page-9-0) All models share the same instruction-tuning corpus, data preprocessing pipelines, training schedules, decoding strategies, and visual token budgets. Under this setting, OneVision-Encoder consistently outperforms SigLIP2 across 16 video, image, and document understanding benchmarks when integrated into an LMM built upon Qwen3-4B, indicating stronger multimodal transfer from the learned visual representations.

Comparison with Qwen3-ViT. We further conduct a comparison with Qwen3-ViT. Specifically, we integrate

<span id="page-10-0"></span>Table 3 Comparison of OV-Encoder training stages on image understanding benchmarks. In this study, we evaluate two training variants of the same ViT architecture under an identical data scale: OV-Encoder-stage1 is trained with image-only data, while OV-Encoder-stage2 continues training from Stage 1 and incorporates OCR and video data with codec-style patch selection. Bold values indicate the best performance among the compared encoders.

| Benchmark   | OV-Encoder-stage1 | OV-Encoder-stage2 | SigLIP2-sig |
|-------------|-------------------|-------------------|-------------|
| AI2D        | 73.6              | 74.5              | 77.5        |
| ChartQA     | 73.6              | 76.2              | 77.0        |
| DocVQA      | 74.3              | 78.5              | 74.4        |
| InfoVQA     | 34.7              | 41.4              | 37.7        |
| MMBench-EN  | 74.7              | 76.3              | 78.1        |
| MMStar      | 49.3              | 49.7              | 52.6        |
| RealWorldQA | 61.8              | 61.3              | 62.2        |
| OCRBench    | 551.0             | 601.0             | 590.0       |

OneVision-Encoder with the Qwen3-1.7B language model and train it under the LLaVA-OneVision-1.5 framework, completing both Stage 1 and Stage 1.5 to adapt the encoder to native-resolution inputs. After this adaptation, the trained OneVision-Encoder is decoupled and compared with Qwen3-ViT under the same LLaVA-Next-Videos instruction-tuning setting. Under this setting, OneVision-Encoder outperforms Qwen3- ViT across 16 understanding benchmarks when evaluated with an LMM built upon Qwen3-4B, as shown in Table [2.](#page-9-0) Notably, these gains are achieved despite OneVision-Encoder being pretrained on substantially fewer visual–text tokens (approximately 100B caption tokens), whereas Qwen3-ViT is pretrained on more than 2.1T caption and instruction-aligned tokens. This result suggests that the observed improvements arise from more effective visual representation learning, rather than increased pretraining scale or architectural specialization.

#### 4.2.2 Stage-wise Multimodal Training Analysis.

We further analyze how stage-wise multimodal training contributes to the observed performance in LMM probing. In the first stage (OV-Encoder-Stage1), the visual encoder is trained using image-only data, focusing on general-purpose visual representation learning. In the second stage (OV-Encoder-Stage2), OCR and video data are introduced on top of Stage1, together with Codec patch selection and chunk-wise temporal sampling strategies.

As shown in Table [3,](#page-10-0) the Stage2 model consistently outperforms its Stage1 counterpart on multimodal and OCR-related benchmarks, while maintaining strong performance on general visual reasoning tasks. This comparison demonstrates that injecting OCR and video supervision plays a critical role in enhancing the ViT-based encoder's suitability as a unified visual backbone for LMMs. Together with the native-resolution results above, this analysis highlights that stage-wise multimodal training is a key factor enabling both robust native-resolution generalization and effective multimodal reasoning.

### 4.3 Attentive Probing Evaluation

We evaluate the quality of visual representations learned by OV-Encoder using an attentive probing protocol, which has been widely adopted to assess backbone-level spatiotemporal modeling capacity without task-specific adaptation. In this setting, the visual encoder is frozen and a lightweight attention-based classifier head is trained on top of the extracted features for downstream video classification. This protocol isolates the intrinsic representational strength of the encoder and enables fair comparison across architectures with different tokenization and temporal modeling strategies.

Experimental Setup. All models are evaluated under a controlled and unified attentive probing protocol. Following prior work on vision-language pretraining and attentive pooling [\(Tschannen et al.,](#page-28-9) [2025\)](#page-28-9), we employ a multi-head attention pooling classifier to aggregate spatiotemporal features into video-level representations. The same classifier architecture is used for all methods to ensure architectural consistency, and the probing head is trained with an identical number of epochs, optimization settings, and learning rate schedules across

<span id="page-11-0"></span>Table 4 Comparison with state-of-the-art methods on video understanding benchmarks. We report top-1 accuracy (%) using an attentive probe with frozen backbones, evaluated under two input configurations: 8 Frames / 2048 Patches and 16 Frames / 4096 Patches. For OV-Encoder (Codec), inputs are constructed using Dense Video-codec Patchification, which selectively encodes temporally salient patches from dense video inputs under the corresponding patch budgets. Bold indicates the best performance and underline indicates the second-best.

|                          | Model Setup |      | Video Benchmarks (Acc. %) |      |          |                    |             |           |           |              |            |
|--------------------------|-------------|------|---------------------------|------|----------|--------------------|-------------|-----------|-----------|--------------|------------|
| Method                   | Backbone    | Res. | Avg.                      | SSV2 | Diving48 | Test<br>Perception | CharadesEgo | Epic-Verb | Epic-Noun | Kinetics-400 | MDB51<br>H |
| 8 Frames / 2048 Patches  |             |      |                           |      |          |                    |             |           |           |              |            |
| CLIP                     | ViT-L/14    | 224  | 50.5                      | 48.2 | 46.6     | 52.2               | 10.8        | 52.8      | 36.1      | 79.3         | 78.0       |
| SigLIP                   | ViT-L/16    | 256  | 50.1                      | 50.7 | 43.9     | 48.9               | 10.9        | 52.2      | 39.1      | 78.2         | 77.0       |
| MetaCLIP                 | ViT-L/14    | 224  | 48.5                      | 50.6 | 28.9     | 49.8               | 10.4        | 54.1      | 37.1      | 79.6         | 77.1       |
| MetaCLIP2                | ViT-L/14    | 224  | 50.2                      | 47.2 | 48.0     | 47.7               | 11.0        | 48.0      | 40.9      | 82.4         | 76.3       |
| AIMv2                    | ViT-L/14    | 224  | 53.8                      | 55.1 | 43.6     | 55.1               | 12.0        | 56.6      | 45.6      | 81.1         | 81.3       |
| SigLIP2                  | ViT-L/16    | 256  | 53.1                      | 52.6 | 50.1     | 52.7               | 11.6        | 54.2      | 43.8      | 80.9         | 79.1       |
| DINOv3                   | ViT-L/14    | 224  | 58.0                      | 57.4 | 58.6     | 59.3               | 13.2        | 62.5      | 51.7      | 82.9         | 78.6       |
| OV-Encoder (Frame)       | ViT-L/14    | 224  | 58.4                      | 57.7 | 57.6     | 58.3               | 12.1        | 61.4      | 52.5      | 84.3         | 83.1       |
| OV-Encoder (Codec)       | ViT-L/14    | 224  | 60.2                      | 58.5 | 67.2     | 60.0               | 12.3        | 62.3      | 53.9      | 84.4         | 83.4       |
| 16 Frames / 4096 Patches |             |      |                           |      |          |                    |             |           |           |              |            |
| SigLIP                   | ViT-L/16    | 256  | 52.8                      | 52.7 | 54.7     | 51.0               | 11.7        | 54.1      | 40.2      | 79.1         | 78.8       |
| MetaCLIP2                | ViT-L/14    | 224  | 51.0                      | 49.3 | 42.1     | 51.1               | 11.2        | 49.2      | 43.2      | 84.0         | 78.2       |
| AIMv2                    | ViT-L/14    | 224  | 56.4                      | 57.2 | 55.7     | 56.4               | 12.4        | 58.3      | 46.2      | 82.2         | 82.6       |
| SigLIP2                  | ViT-L/16    | 256  | 55.7                      | 58.2 | 56.7     | 53.3               | 11.9        | 56.4      | 45.2      | 82.7         | 81.2       |
| DINOv3                   | ViT-L/14    | 224  | 59.1                      | 58.3 | 61.3     | 60.8               | 14.0        | 63.2      | 51.9      | 83.9         | 79.7       |
| OV-Encoder (Frame)       | ViT-L/14    | 224  | 59.9                      | 58.7 | 63.2     | 60.3               | 12.6        | 62.9      | 54.5      | 85.1         | 81.6       |
| OV-Encoder (Codec)       | ViT-L/14    | 224  | 61.5                      | 60.1 | 69.4     | 60.9               | 12.9        | 63.3      | 54.4      | 85.4         | 85.3       |

all experiments. All experiments are conducted on a cluster of 8 NVIDIA A800 GPUs. We therefore evaluate our model by assessing the quality of the model's learned representation on a set of seven benchmarks: SSV2, Diving48 [\(Li et al.,](#page-26-6) [2018\)](#page-26-6), Perception Test [\(Patraucean et al.,](#page-27-11) [2023\)](#page-27-11), CharadesEgo [\(Sigurdsson et al.,](#page-27-13) [2018\)](#page-27-13), Epic-Kitchens-100 [\(Kay et al.,](#page-26-7) [2017\)](#page-26-7), Kinetics-400 [\(Damen et al.,](#page-25-10) [2022\)](#page-25-10), HMDB51 [\(Kuehne et al.,](#page-26-8) [2011\)](#page-26-8). Batch sizes are determined on a per-dataset basis to balance computational efficiency and training stability: a batch size of 32 is used for SSV2, Diving48, and Perception Test, 16 for HMDB51, and 128 for all remaining datasets. During evaluation, we adopt a single-crop inference protocol, using one temporal crop and one spatial crop per video clip, resulting in a single prediction for each input video.

Input Configuration. For frame-centric baselines, including SigLIP2, DINOv3, and AIMv2, we evaluate both 8-frame and 16-frame inputs using uniform temporal sampling. For OV-Encoder, we evaluate two instantiations under an identical patch budget of 512 visual tokens. Specifically, the codec-guided variant operates on dense 64-frame video inputs, where codec-derived signals determine the visible patch indices, while the chunk-wise variant samples frames within fixed temporal chunks. This design ensures that all methods are compared under a fixed token budget, decoupling representational capacity from raw input resolution or frame count.

Evaluation Protocol. During inference, all crops belonging to the same video are aggregated by averaging logits across crops. Labels are shared across crops of the same video, and no test-time augmentation beyond the single-crop setting is applied. For codec-based models, visible patch indices are provided as part of the input to ensure consistent patch selection across crops.

Results. Table [4](#page-11-0) shows that OV-Encoder consistently outperforms SigLIP2, DINOv3, and AIMv2 across all evaluated video benchmarks under both 8-frame and 16-frame settings. In particular, OV-Encoder achieves a 12.7% absolute Top-1 accuracy improvement over SigLIP2 on Diving48 at an identical patch budget, demonstrating superior motion modeling capability. Notably, these gains are achieved without sacrificing

<span id="page-12-0"></span>Table 5 Effect of patch budget scaling under attentive probing. Patch budgets of 512/1024/2048/4096 correspond to 2/4/8/16 video frames, respectively. Dense SigLIP2 processes all spatial patches per frame, while OV-Encoder (Codec) selectively retains motion-relevant patches guided by codec-derived temporal signals.

| Setting             | Video Benchmarks (Acc. %) |      |          |                    |             |           |           |              |            |
|---------------------|---------------------------|------|----------|--------------------|-------------|-----------|-----------|--------------|------------|
|                     | AVG                       | SSV2 | Diving48 | Test<br>Perception | CharadesEgo | Epic-Verb | Epic-Noun | Kinetics-400 | MDB51<br>H |
| Patch Budget = 512  |                           |      |          |                    |             |           |           |              |            |
| SigLIP2             | 43.4                      | 42.8 | 28.1     | 38.7               | 10.1        | 42.9      | 37.4      | 74.5         | 72.4       |
| OV-Encoder (Codec)  | 50.1                      | 50.0 | 46.5     | 50.5               | 10.6        | 50.3      | 41.7      | 76.7         | 74.7       |
| Patch Budget = 1024 |                           |      |          |                    |             |           |           |              |            |
| SigLIP2             | 50.8                      | 49.0 | 48.7     | 50.1               | 10.8        | 50.6      | 42.0      | 78.8         | 76.6       |
| OV-Encoder (Codec)  | 56.2                      | 56.6 | 54.9     | 58.6               | 11.1        | 58.2      | 48.4      | 81.8         | 80.3       |
| Patch Budget = 2048 |                           |      |          |                    |             |           |           |              |            |
| SigLIP2             | 53.1                      | 52.6 | 50.1     | 52.7               | 11.6        | 54.2      | 43.8      | 80.9         | 79.1       |
| OV-Encoder (Codec)  | 60.2                      | 58.5 | 67.2     | 60.0               | 12.3        | 62.3      | 53.9      | 84.4         | 83.4       |
| Patch Budget = 4096 |                           |      |          |                    |             |           |           |              |            |
| SigLIP2             | 55.7                      | 58.2 | 56.7     | 53.3               | 11.9        | 56.4      | 45.2      | 82.7         | 81.2       |
| OV-Encoder (Codec)  | 61.5                      | 60.1 | 69.4     | 60.9               | 12.9        | 63.3      | 54.4      | 85.4         | 85.3       |

performance on appearance-dominated datasets such as Kinetics-400, confirming that Codec patch-level encoding yields more discriminative and efficient spatiotemporal representations. We further observe that for 64-frame inputs under a fixed token budget, allocating only one to two I-frames is sufficient to establish stable spatial context, with subsequent frames contributing primarily through sparse motion-driven updates.

### 4.4 Patch-Efficient Video Understanding Comparison

We conduct an efficiency analysis comparing SigLIP2 with dense full-frame patch processing and OV-Encoder (Codec) under a fixed token budget, as shown in Table [5.](#page-12-0) It is important to emphasize that OV-Encoder (Codec) does not perform temporal downsampling of the input video. All results are obtained from the same 64-frame (16384 patches) source video, where codec-native motion vectors and residuals are used to selectively extract a fixed number of spatiotemporal patches distributed across the entire temporal extent.

For a fair comparison, SigLIP2 is evaluated under the same token budgets and adopts a traditional frame sampling strategy, where each group of 256 patches corresponds to a contiguous RGB frame. Under a fixed token budget, OV-Encoder (Codec) redistributes patches across time while preserving their spatial positions, enabling long-range temporal coverage. As a result, it outperforms SigLIP2 on Diving48 and Perception Test while reducing patch processing by 75.0%–96.9% compared to dense processing of 16,384 patches. Specifically, the reduction ratio is computed relative to the dense baseline that processes all 64 × 256 = 16,384 patches from the full video. Using token budgets of 4096, 2048, 1024, and 512 patches corresponds to retaining 25.0%, 12.5%, 6.25%, and 3.1% of the dense patches, respectively, yielding a patch reduction of 75.0%–96.9%.

### 4.5 Ablation of Codec-guided Patch Selection

Although OV-Encoder (Codec) consistently outperforms frame-centric baselines under attentive probing, as shown in Table [4,](#page-11-0) performance gains alone do not establish codec-guided patch selection as a functional mechanism. To isolate its causal role, we conduct a set of controlled interventions that explicitly manipulate

<span id="page-13-0"></span>Table 6 Controlled interventions on codec-selected motion patches. All settings use identical token budgets and visual content unless otherwise specified.

| Setting                                                                    | Video Benchmarks (Acc. %) |      |          |                    |             |           |           |              |            |
|----------------------------------------------------------------------------|---------------------------|------|----------|--------------------|-------------|-----------|-----------|--------------|------------|
|                                                                            | AVG                       | SSV2 | Diving48 | Test<br>Perception | CharadesEgo | Epic-Verb | Epic-Noun | Kinetics-400 | MDB51<br>H |
| OV-Encoder (Codec)                                                         | 61.5                      | 60.1 | 69.4     | 60.9               | 12.9        | 63.3      | 54.4      | 85.4         | 85.3       |
| Non-motion Patch Replacement (50%)<br>(Same Video, Same Position)          | 55.4                      | 52.1 | 55.4     | 54.9               | 11.6        | 56.3      | 50.2      | 83.1         | 79.4       |
| Counterfactual Motion Replacement (50%)<br>(Motion Patch from Other Video) | 54.9                      | 50.6 | 57.2     | 53.8               | 11.3        | 55.1      | 49.6      | 82.7         | 79.0       |
| Patch–Position Shuffle<br>(Content Preserved, Positions Shuffled)          | 48.1                      | 41.8 | 46.3     | 45.1               | 8.7         | 48.2      | 42.5      | 78.4         | 73.6       |

patch content while holding token count, spatiotemporal positions, positional encodings, and model parameters fixed.

Patch Content Necessity. We first examine whether the content of codec-selected motion patches is necessary for the observed gains. In this setting, motion-heavy patches identified by the codec are replaced with non-motion patches sampled from the same video, while preserving their original spatiotemporal positions. As shown in Table [6,](#page-13-0) this intervention leads to substantial performance degradation across all benchmarks. The drop is particularly pronounced on motion-sensitive datasets, with accuracy decreasing, while appearance-dominated datasets such as Kinetics-400 exhibit smaller but consistent declines. These results indicate that the benefits of OV-Encoder (Codec) cannot be attributed to token sparsity or positional bias alone, but critically depend on the motion-centric content encoded by the selected patches.

Semantic Specificity of Motion Cues. To further assess whether the model relies on semantically aligned motion rather than generic motion signals, we perform a counterfactual replacement in which codec-selected motion patches are substituted with motion patches drawn from unrelated videos. Despite preserving motion magnitude and patch positions, this intervention results in even larger performance drops on fine-grained temporal benchmarks, as shown in Table [6.](#page-13-0) The consistent degradation relative to non-motion replacement demonstrates that OV-Encoder (Codec) is sensitive to the semantic correctness of motion cues, rather than merely benefiting from the presence of motion energy or stochastic perturbations.

Negative Control: Patch-Position Shuffle. As a sanity check, we additionally evaluate a patch-position shuffle intervention, in which the visual content of codec-selected patches is preserved but their spatiotemporal positions are randomly permuted. This intervention causes a substantially larger performance drop across all benchmarks, as shown in Table [6,](#page-13-0) confirming that coherent spatial and temporal alignment is critical for effective representation learning. Importantly, this experiment is not intended to demonstrate the causal role of codec guidance itself, but rather to rule out degenerate explanations in which patch content alone suffices without positional structure.

Discussion. Taken together, these interventions establish a consistent ordering across benchmarks: preserving semantically correct motion patches yields the strongest performance, followed by non-motion substitutions, while semantically mismatched motion patches and patch-position shuffling are most detrimental. This hierarchy rules out alternative explanations based on regularization, noise injection, or attention disruption, and provides converging evidence that codec-guided patch selection captures motion-centric visual evidence that is both semantically meaningful and structurally aligned with the underlying video content.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 5 Visualization of I- and P-frame decomposition in HEVC. I-frames retain complete spatial structure, whereas P-frames encode motion-compensated residuals highlighting motion. Bright areas denote high residual magnitudes, while dark areas indicate static content.

### 4.6 Qualitative Analysis

To understand how our model leverages temporal information for selective patch processing, we visualize the patch selection mechanism guided by residual magnitudes and motion vectors. As illustrated in Figure [5,](#page-14-0) we compare an I-frame (reference frame) with subsequent P-frames in a video sequence, where patch colors indicate their selection priority based on residual and motion strength. In the I-frame (top-left), all patches are processed uniformly since no temporal prior exists. In contrast, the P-frames show selective emphasis on patches with large residuals and strong motion vectors, corresponding to regions with significant appearance changes or object movements. The highlighted patches, shown in warmer colors (red, orange, yellow), primarily correspond to dynamic foreground objects such as moving pedestrians, whereas static background regions (trees, buildings) appear in cooler colors (blue, green), indicating reduced computational focus. Across the sequence, the model consistently tracks salient motion regions, as pedestrians maintain high activation throughout their trajectories, demonstrating that the residual–motion criterion effectively identifies temporally informative patches. This visualization confirms that our approach achieves spatial selectivity by concentrating computation on motion-rich areas and allocating representational capacity according to temporal saliency, leading to more efficient video understanding.

Figure [6](#page-15-0) contrasts conventional frame-centric video processing with the proposed Codec patch extraction. While dense 64-frame inputs preserve full temporal context at high computational cost, uniform frame sampling reduces computation by sparsely selecting frames but inevitably discards fine-grained inter-frame motion, particularly for fast or subtle actions. Temporal saliency detection instead analyzes all frames to identify motion- and event-centric regions. Leveraging this signal, Codec patch extraction selectively encodes only temporally salient patches using a codec-inspired ordering, achieving substantial token reduction (75%–96.9%) while preserving critical motion dynamics. This formulation decouples temporal coverage from token density, enabling efficient and scalable spatiotemporal modeling without reliance on sparse frame sampling.

<span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

Figure 6 Comparison of video processing pipelines for spatiotemporal representation learning. (a) original dense video input with full temporal context, (b) uniform frame sampling that sparsely selects evenly spaced frames, (c) temporal saliency detection that identifies motion- and event-centric regions across all frames, and (d) Codec patch extraction that selectively retains temporally salient patches under a fixed token budget.

### 5 Related Work

### 5.1 Advances in Visual Representation Learning

Vision Transformers [\(Dosovitskiy et al.,](#page-25-11) [2021;](#page-25-11) [Li et al.,](#page-26-9) [2022a\)](#page-26-9) have emerged as a dominant approach in visual representation learning. DINOv2 [\(Jose et al.,](#page-26-10) [2025\)](#page-26-10) and DINOv3 [\(Siméoni et al.,](#page-27-2) [2025\)](#page-27-2) introduce a self-supervised framework that does not require labels, achieving state-of-the-art results across a wide range of vision tasks. Simultaneously, equivariant self-supervised methods [\(Devillers and Lefort,](#page-25-12) [2023;](#page-25-12) [Park et al.,](#page-27-14) [2022;](#page-27-14) [Garrido et al.,](#page-25-13) [2023;](#page-25-13) [Gupta et al.,](#page-26-11) [2024;](#page-26-11) [Dangovski et al.,](#page-25-14) [2021\)](#page-25-14) have been developed to predict structured transformations consistent with group-theoretic principles. Masked image modeling techniques [\(He et al.,](#page-26-0) [2021;](#page-26-0) [Bao et al.,](#page-24-10) [2021;](#page-24-10) [El-Nouby et al.,](#page-25-15) [2024;](#page-25-15) [Xie et al.,](#page-28-14) [2022\)](#page-28-14) learn visual representations by reconstructing masked regions within the pixel domain. Furthermore, Joint-Embedding Predictive Architecture [\(Assran](#page-24-11) [et al.,](#page-24-11) [2023,](#page-24-11) [2025;](#page-24-1) [Baevski et al.,](#page-24-12) [2022\)](#page-24-12) showed that predicting a learned latent space instead of the pixel space leads to more powerful, higher-level features. Contrastive Language-Image Pretraining (CLIP) [\(Bolya](#page-24-13) [et al.,](#page-24-13) [2025;](#page-24-13) [Sun et al.,](#page-28-15) [2023;](#page-28-15) [Li et al.,](#page-26-12) [2023c;](#page-26-12) [Fang et al.,](#page-25-16) [2024;](#page-25-16) [Xu et al.,](#page-28-10) [2024;](#page-28-10) [Schuhmann et al.,](#page-27-5) [2022\)](#page-27-5) aligns images and texts within a shared embedding space through instance-level contrastive supervision. However, existing methods mainly focus on either pixel-level reconstruction or instance-level contrastive alignment, limiting their ability to capture structured semantic relationships across samples and modalities. In this work, we adopt the self-supervised cluster discrimination paradigm (e.g., UNICOM [\(An et al.,](#page-24-3) [2023\)](#page-24-3), MLCD [\(An](#page-24-4) [et al.,](#page-24-4) [2024\)](#page-24-4), RICE [\(Xie et al.,](#page-28-5) [2025\)](#page-28-5), UniViT [\(Tang et al.,](#page-28-6) [2025\)](#page-28-6)), which learns structured semantics by jointly clustering similar instances and discriminating between clusters.

### 5.2 Efficient Video Representations

(1) Video Sampling. Conventional methods first divide selected frames into fixed patch embeddings (e.g., 16×16 in ViT) before encoding, where frame selection typically follows uniform sampling or motion-based heuristics [\(Xue et al.,](#page-29-5) [2022;](#page-29-5) [Wang et al.,](#page-28-16) [2023\)](#page-28-16). However, such strategies still process all patches per frame, resulting in long token sequences and high preprocessing cost. To improve efficiency, patch-level merging is adopted in AuroraCap [\(Chai et al.,](#page-24-14) [2024\)](#page-24-14) and AuroraLong [\(Xu et al.,](#page-28-17) [2025\)](#page-28-17), while adaptive sampling [\(Kim](#page-26-13) [et al.,](#page-26-13) [2024\)](#page-26-13) allocates more frames to motion-intensive segments but fails to mitigate the quadratic cost of self-attention under large token budgets. Moreover, flexible-FPS variants impose an upper bound on frame count, limiting scalability to high frame rates. Video-LaVIT [\(Jin et al.,](#page-26-14) [2024\)](#page-26-14) explores decomposing videos into keyframes and motion signals to enable efficient tokenization and unified generative pretraining with multimodal LLMs. (2) Token Dropout. A simple strategy to reduce transformer sequence length is token dropout, which removes redundant tokens either randomly [\(Han et al.,](#page-26-15) [2022;](#page-26-15) [Liu et al.,](#page-27-15) [2023\)](#page-27-15) or through learned selection [\(Rao et al.,](#page-27-16) [2021;](#page-27-16) [Yin et al.,](#page-29-6) [2022;](#page-29-6) [Chen et al.,](#page-25-17) [2023\)](#page-25-17). PatchDropout [\(Liu et al.,](#page-27-15) [2023\)](#page-27-15) applies random token removal to standard ViTs for faster training while retaining all tokens at inference, whereas Turbo [\(Han et al.,](#page-26-15) [2022\)](#page-26-15) adopts partial masking for video reconstruction. However, token dropout often removes informative regions and disrupts temporal consistency, leading to unstable optimization and weakened motion modeling. (3) Token Merging. Various approaches have been employed to merge tokens in order to mitigate the information loss associated with dropout. In [\(Liang et al.,](#page-26-16) [2022\)](#page-26-16), inattentive tokens are combined into a singular "background" token, while other studies utilize semantics to enhance the features of interest [\(Zeng et al.,](#page-29-7) [2022;](#page-29-7) [Zhou et al.,](#page-29-8) [2023\)](#page-29-8). Additional research [\(Lee et al.,](#page-26-17) [2024\)](#page-26-17) focuses on optimizing token merging through the use of trainable parameters. The latest work by [\(Koner et al.,](#page-26-18) [2024\)](#page-26-18) trains models using an additional stream of compressed tokens to minimize the overhead of attention. Alternatively, numerous studies have implemented token merging, as detailed in [\(Bolya et al.,](#page-24-15) [2023;](#page-24-15) [Bolya and Hoffman,](#page-24-16) [2023\)](#page-24-16), where image tokens are merged via a weighted average, leveraging attention keys as a similarity metric. In this work, we do not aim to improve compression efficiency or replace existing codecs. Instead, we treat the structural decomposition exposed by modern video codecs as a guiding principle for visual representation learning, where stable spatial context is separated from sparse, motion driven variations. This perspective directly informs the design of our spatiotemporal transformer.

### 5.3 Video Codec Compression

Video codec compression has been extensively studied, with a large body of work devoted to designing efficient and effective video coding systems [\(Girod et al.,](#page-25-18) [2005;](#page-25-18) [Adami et al.,](#page-24-17) [2007;](#page-24-17) [Kumar,](#page-26-19) [2019\)](#page-26-19). Classical standards such as H.264/AVC [\(Sullivan et al.,](#page-28-7) [2012;](#page-28-7) [Zhao and Liang,](#page-29-9) [2006\)](#page-29-9) established the foundation of modern video compression by introducing key techniques including motion compensation, transform coding, and entropy coding, leading to substantial gains in compression efficiency. Building upon this framework, the High Efficiency Video Coding (HEVC) standard, also known as H.265 [\(Sullivan and Wiegand,](#page-28-18) [2005;](#page-28-18) [Sullivan et al.,](#page-28-7) [2012\)](#page-28-7), further improved coding performance through more expressive block partitioning, enhanced motion modeling, and refined entropy coding mechanisms.

Beyond traditional hand-crafted codecs, recent research has explored deep learning-based video compression methods [\(Li et al.,](#page-26-20) [2021;](#page-26-20) [Mentzer et al.,](#page-27-17) [2022;](#page-27-17) [Yang et al.,](#page-29-10) [2021;](#page-29-10) [Li et al.,](#page-26-21) [2023b;](#page-26-21) [Zhang et al.,](#page-29-11) [2023;](#page-29-11) [Wang](#page-28-19) [et al.,](#page-28-19) [2025\)](#page-28-19), which employ neural networks to model spatial and temporal redundancies directly from data. These approaches have demonstrated promising compression performance and have also motivated perceptually driven coding strategies that allocate bits according to human visual sensitivity. Efficiency has also been explored in transformer-based video models, where methods such as Run-Length Tokenization (RLT) [\(Choudhury et al.,](#page-25-19) [2024\)](#page-25-19) exploit temporal redundancy at the token level to reduce input tokens while maintaining performance. Recent work such as EMA [\(Zhao et al.,](#page-29-12) [2025\)](#page-29-12) further leverages compressed video streams by encoding GOP structures with motion-aware mechanisms for efficient video MLLM understanding. Collectively, prior work in video compression has made significant progress toward compactly representing video content by exploiting its inherent temporal predictability and structural redundancy. In this work, we do not aim to improve compression efficiency or replace existing codecs; instead, we draw inspiration from the structural principles underlying modern video codecs, particularly their explicit decomposition of spatial context and temporal variation, to guide the design of spatiotemporal representations for visual transformers.

## 6 Conclusion

In this work, we introduced OV-Encoder, a unified self-supervised vision framework that departs from frame-centric modeling and aligns representation learning with the predictive structure of video signals. By treating discriminative visual evidence as patch-level and motion-centric, OneVision-Encoder selectively encodes informative regions while preserving dense temporal coverage under fixed token budgets. Central to this design is Codec Patchification, which constructs sparse yet structure-preserving spatiotemporal token layouts and naturally extends to chunk-wise temporal modeling and single-image inputs within a unified attention-based encoder, supported by 3D rotary positional encoding. Combined with a cluster discrimination objective that jointly models object-level and motion-level semantics without external supervision, OneVision-Encoder achieves state-of-the-art performance under both LMM probing and attentive probing. These results highlight codec-inspired patch-level sparsity as an effective and scalable foundation for general-purpose visual representation learning.

### 7 Contributors

#### Contributors

core contributors are in bold

- Feilong Tang
- Xiang An
- Yunyao Yan
- Yin Xie
- Bin Qin
- Kaicheng Yang
- Yifei Shen
- Yuanhan Zhang
- Chunyuan Li
- Shikun Feng
- Changrui Chen
- Huajie Tan
- Ming Hu
- Manyuan Zhang

### Project Leaders

- Bo Li
- Ziyong Feng
- Ziwei Liu
- Zongyuan Ge
- Jiankang Deng

### 8 Implementation Details

Model Architecture and Configuration. OneVision-Encoder Large is implemented as a Vision Transformer with 24 transformer layers, a hidden dimension of 1024, and 16 attention heads. The model uses a patch size of 14 × 14 and adopts GELU activations with Layer Normalization throughout the network. Attention computation is accelerated using Flash Attention 2, enabling efficient training and inference at scale. A summary of the core architectural hyperparameters is provided in Table [7.](#page-18-0)

<span id="page-18-0"></span>Table 7 Architecture configuration of OneVision-Encoder Large.

| Component           | Setting                 |
|---------------------|-------------------------|
| Transformer layers  | 24                      |
| Hidden dimension    | 1024                    |
| Attention heads     | 16                      |
| Patch size          | 14 × 14                 |
| MLP expansion ratio | 4×                      |
| Position encoding   | 3D RoPE (T:H:W = 4:6:6) |

Unified Patch-based Input Representation. All inputs are converted into patch tokens and processed by a single ViT backbone. Images are treated as single-frame videos (T=1), while videos are represented in a 5D tensor format. To ensure consistent temporal reasoning, all video inputs are mapped to a virtual temporal grid of 64 frames, regardless of the actual number of frames processed. This mapping enables uniform temporal position encoding across dense, sparse, and Codec inputs.

For inputs that do not cover all 64 frames explicitly, a visible indices mechanism is used to associate each selected patch with its corresponding temporal position in the virtual grid. This design decouples temporal coverage from token density and allows sparse inputs to preserve long-range temporal structure.

Codec-style Patch Selection. Codec-style processing operates on dense videos of 64 frames at a spatial resolution of 224 × 224. Motion vectors and prediction residuals are extracted from the HEVC codec and used to estimate patch-level temporal saliency. Motion vectors capture object displacement at sub-pixel precision, while residuals encode fine-grained appearance changes not explained by motion compensation. These signals are fused into a unified saliency score for each patch across all frames.

Patches are ranked globally by their saliency scores, and only the top-K patches are retained. This selection typically preserves between 3.1% and 25% of all patches, corresponding to a compression ratio of 75%–96.9% relative to dense processing. Selected patches are then reassembled into a compact video representation and passed to the ViT using sparse visible indices. Table [8](#page-18-1) summarizes the codec-style selection procedure.

<span id="page-18-1"></span>Table 8 Codec-style patch selection pipeline.

| Step                | Description                                                |
|---------------------|------------------------------------------------------------|
| Motion extraction   | Decode HEVC motion vectors with camera motion compensation |
| Residual extraction | Obtain prediction residuals for fine-grained changes       |
| Energy fusion       | Combine motion and residual energies into a saliency score |
| Top-K selection     | Retain globally most salient patches across all frames     |
| Sparse encoding     | Process selected patches with sparse visible indices       |

Video Processing Modes and Batch Composition. During pretraining, OneVision-Encoder employs a mixedmodality batch that includes multiple video processing modes. This design exposes the model to diverse temporal structures and encourages robust representation learning. Video samples within a batch are split into three processing modes: Codec patchification, uniform frame sampling, and Tiling-style spatial concatenation. All modes produce inputs that are compatible with the same ViT backbone and position encoding.

Position Encoding Consistency. All input modes share the same 3D Rotary Position Embedding. The temporal

**Table 9** Video processing modes used during training.

| Mode           | Batch Ratio | Input Form                   | Output Shape        |
|----------------|-------------|------------------------------|---------------------|
| Codec          | 50%         | $Dense\ video\ +\ saliency$  | [B, 3, 8, 224, 224] |
| Frame sampling | 37.5%       | Uniform temporal bins        | [B, 3, 8, 224, 224] |
| Tiling         | 12.5%       | Vertical frame concatenation | [B, 3, 1792, 224]   |

dimension of the RoPE encodes the position of each patch within the 64-frame virtual grid, while spatial dimensions encode patch row and column indices. For Codec inputs, patches may originate from arbitrary frames but are positioned correctly via their temporal indices. For uniformly sampled frames, temporal gaps are explicitly encoded. Tiling inputs are treated as single-frame inputs with fixed temporal positions. This unified encoding scheme enables the model to reason coherently over heterogeneous spatiotemporal layouts.

**Training and Inference Behavior.** The model is trained using a unified optimization pipeline across all modalities. No modality-specific parameters or task-specific encoders are introduced. At inference time, the same preprocessing logic is applied, and the model can flexibly switch between Codec sparse processing and conventional frame sampling depending on computational constraints. This design allows efficient deployment across a wide range of image and video understanding tasks without architectural modification.

### 9 Controlled Evaluation Pipeline

As shown in Figure 7, we adopt a controlled evaluation protocol to compare OneVision-Encoder with Qwen3-ViT and SigLIP2 under LMM probing. For comparison with Qwen3-ViT, OneVision-Encoder is first integrated with the Qwen3-1.7B language model and trained through Stage 1 and Stage 1.5 under the LLaVA-OneVision-1.5 framework to adapt the encoder to native-resolution multimodal inputs. After alignment, the trained vision encoder is decoupled and evaluated under the same LLaVA-Next-Videos instruction-tuning setting as Qwen3-ViT, ensuring a fair comparison under identical downstream supervision. For comparison with SigLIP2, all models are directly evaluated under identical multimodal fine-tuning conditions using a unified 1.5M-scale instruction-tuning corpus, while keeping the language model backbone fixed. This decoupled and unified evaluation pipeline isolates the contribution of visual representation learning and avoids confounding effects from language model capacity, instruction data leakage, or differing alignment procedures.

<span id="page-19-0"></span>![](_page_19_Figure_6.jpeg)

Figure 7 Controlled evaluation pipeline decoupling the encoder for fair comparison against Qwen3-ViT and SigLIP2

# 10 Spatial Bias Analysis

As shown in Figure 8, the reported statistics are computed from a random subset of 200,000 video samples drawn from the training data. When using codec-guided patch selection alone, the selected visual tokens exhibit a pronounced spatial center bias, with the majority of tokens concentrated in the central regions of the

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 8 Spatial Bias Analysis of Selected Visual Patches under a given setting.

frame. This behavior reflects intrinsic statistical properties of video data: due to camera framing and subject placement, salient motion cues and residual signals are typically denser near the image center. While such a bias enables the model to focus on regions with strong motion signals, it also leads to insufficient coverage of peripheral areas, thereby weakening the representation of global scene structure and fine-grained action cues.

With the introduction of chunk-wise patchification, the spatial distribution of selected tokens becomes markedly more uniform, redistributing tokens toward the image periphery and boundary regions and effectively mitigating the center bias induced by codec-driven selection. This effect arises because chunk-wise sampling partitions the video along the temporal dimension and performs global selection over visible patch indices, resulting in a more structurally balanced spatial coverage. Importantly, this rebalancing is achieved without increasing the token budget, but through a principled reallocation of patch selection that enhances spatial diversity and complements motion-centric evidence with global contextual information.

# 11 Token Allocation Case Study

For each case, we plot time (x-axis) versus accumulated visual tokens (y-axis) under a fixed token budget. Stepwise growth indicates when a method allocates a non-trivial number of tokens to a timestamp. Vertical rugs at the bottom show the actual timestamps observed by each strategy, and triangles denote I-frame anchors in the codec-style layout. Case 1 uses a 2048-token budget (uniform 8-frame baseline), and Case 2 uses a 4096-token budget (uniform 16-frame baseline). Detailed experiments and evaluation protocols for these two motion regimes will be reported in LLaVA-OneVision 2.

<span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

Figure 9 Case study 1 (Diving): dense evidence for continuous motion. Under a fixed budget of 2048 tokens, codec-style patch extraction distributes tokens over 64 sampled frames to capture dense pose transitions, while uniform sampling allocates the same budget to only 8 frames and may miss brief but critical instants. Phases: P1 0.0–1.4s (takeoff + angular momentum), P2 1.4–3.4s (tuck/pike set; body shape), P3 3.4–11.6s (twist+somersault progression; dense cues), P4 11.6–14.2s (spot water + open/align), P5 14.2–15.0s (entry alignment; line/splash). Diving is high-speed and continuous, so discriminative cues are temporally dense and benefit from dense coverage.

### 11.1 Case 1

We consider videos whose semantics are carried by smooth, persistent motion, where essentially every moment contributes evidence. In this regime, missing intermediate timestamps breaks trajectory continuity and harms recognition. Figure [9](#page-21-0) visualizes this behavior on a Diving example. Under the same fixed token budget of 2048 tokens, uniform 8-frame sampling concentrates all tokens on only eight sparse timestamps (leaving large temporal gaps), whereas codec-style allocation first samples a 64-frame timeline and then reallocates tokens through saliency-guided P-frame patch selection, preserving dense motion evidence across time. This is particularly important for high-angular-velocity actions such as diving, where discriminative cues are temporally dense and distributed across many brief pose transitions rather than confined to a single transition.

### 11.2 Case 2

We consider videos whose semantics concentrate in short, discrete transitions, where the true "key frames" are sparse events such as brief pours, quick tool interactions, or sudden scene changes. In this regime, the primary risk of uniform sampling is misplacement: even with a sufficient token budget, the 16 sampled frames can easily miss a short segment entirely, causing decisive evidence to be absent from the visual input. Figure [10](#page-22-0) illustrates this effect on a 130-second cooking video. Notably, several pour segments are extremely short (e.g., P4: 38.0–39.0s), making them easy to skip under uniform sampling. In contrast, codec-style allocation first samples a 64-frame timeline and then reallocates the same token budget toward high-saliency moments via P-frame patch selection, increasing the probability of capturing the decisive evidence that later multi-turn QA may depend on.

<span id="page-22-0"></span>![](_page_22_Figure_0.jpeg)

Figure 10 Case study 2 (Cooking): sparse key frames under instantaneous motion. We analyze a 130-second cooking video where decisive evidence concentrates in short, discrete transitions (e.g., brief pours) rather than being uniformly distributed over time. Under a fixed budget of 4096 tokens, codec-style allocation spreads tokens over a 64-frame timeline and reallocates P-frame tokens toward high-saliency moments, while uniform sampling allocates the same budget to only 16 frames and may miss brief segments entirely (misplacement). Phases: P1 0.0–25.0s (context), P2 25.0–27.0s (pour raw materials), P3 28.0–37.0s (mix raw materials), P4 38.0–39.0s (pour raw materials), P5 40.0–45.0s (mix raw materials), P6 46.0–48.0s (pour raw materials), P7 49.0–51.0s (mix raw materials), P8 52.0–54.0s (pour raw materials), P9 55.0–71.0s (mix raw materials), P10 72.0–98.0s (pour raw materials), P11 98.0–130.0s (context).

![](_page_23_Figure_0.jpeg)

Figure 11 Comparison of video processing pipelines for spatiotemporal representation learning. (a) original dense video input with full temporal context, (b) uniform frame sampling that sparsely selects evenly spaced frames, (c) temporal saliency detection that identifies motion- and event-centric regions across all frames, and (d) Codec patch extraction that selectively retains temporally salient patches under a fixed token budget.

### References

- <span id="page-24-17"></span>Nicola Adami, Alberto Signoroni, and Riccardo Leonardi. State-of-the-art and trends in scalable video compression with wavelet-based approaches. IEEE Transactions on Circuits and Systems for Video Technology, 17(9):1238–1255, 2007.
- <span id="page-24-3"></span>Xiang An, Jiankang Deng, Kaicheng Yang, Jaiwei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. Unicom: Universal and compact representation learning for image retrieval. In ICLR, 2023.
- <span id="page-24-4"></span>Xiang An, Kaicheng Yang, Xiangzi Dai, Ziyong Feng, and Jiankang Deng. Multi-label cluster discrimination for visual representation learning. In ECCV, pages 428–444. Springer, 2024.
- <span id="page-24-6"></span>Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.
- <span id="page-24-11"></span>Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding. CVPR, 2023.
- <span id="page-24-1"></span>Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv:2506.09985, 2025.
- <span id="page-24-12"></span>Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. arXiv:2202.03555, 2022.
- <span id="page-24-5"></span>Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- <span id="page-24-10"></span>Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv:2106.08254, 2021.
- <span id="page-24-7"></span>Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. V-jepa: Latent video prediction for visual representation learning. arXiv:2404.08471, 2023.
- <span id="page-24-2"></span>Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024.
- <span id="page-24-16"></span>Daniel Bolya and Judy Hoffman. Token Merging for Fast Stable Diffusion. In CVPR Workshops, pages 4599–4603. IEEE, 2023.
- <span id="page-24-15"></span>Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token Merging: Your ViT But Faster. In ICLR. OpenReview.net, 2023.
- <span id="page-24-13"></span>Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv:2504.13181, 2025.
- <span id="page-24-9"></span>Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. <https://github.com/kakaobrain/coyo-dataset>, 2022.
- <span id="page-24-8"></span>Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.
- <span id="page-24-0"></span>João Carreira, Dilara Gokay, Michael King, Chuhan Zhang, Ignacio Rocco, Aravindh Mahendran, Thomas Albert Keck, Joseph Heyward, Skanda Koppula, Etienne Pot, et al. Scaling 4d representations. arXiv:2412.15212, 2024.
- <span id="page-24-14"></span>Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv:2410.03051, 2024.

- <span id="page-25-17"></span>Lei Chen, Zhan Tong, Yibing Song, Gangshan Wu, and Limin Wang. Efficient Video Action Detection with Token Dropout and Context Refinement. In ICCV, pages 10354–10365. IEEE, 2023.
- <span id="page-25-7"></span>Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024a.
- <span id="page-25-4"></span>Tsai-Shien Chen et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. CVPR, 2024b.
- <span id="page-25-19"></span>Rohan Choudhury, Guanglei Zhu, Sihan Liu, Koichiro Niinuma, Kris Kitani, and László Jeni. Don't look twice: Faster video transformers with run-length tokenization. Advances in Neural Information Processing Systems, 2024.
- <span id="page-25-0"></span>Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, et al. Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062, 2025.
- <span id="page-25-8"></span>X.AI Corp. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model. <https://x.ai/blog/grok-1.5v>, 2024.
- <span id="page-25-10"></span>Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100. Int. J. Comput. Vis., 130(1):33–55, 2022.
- <span id="page-25-14"></span>Rumen Dangovski, Li Jing, Charlotte Loh, Seungwook Han, Akash Srivastava, Brian Cheung, Pulkit Agrawal, , and Marin Soljačić. Equivariant contrastive learning. arXiv:2111.00899, 2021.
- <span id="page-25-3"></span>Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.
- <span id="page-25-12"></span>Alexandre Devillers and Mathieu Lefort. Equimod: An equivariance module to improve self-supervised learning. arXiv:2211.01244, 2023.
- <span id="page-25-2"></span>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929, 2020.
- <span id="page-25-11"></span>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- <span id="page-25-15"></span>Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv:2401.08541, 2024.
- <span id="page-25-16"></span>Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. In ICLR, 2024.
- <span id="page-25-1"></span>Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor G Turrisi da Costa, Louis Béthune, Zhe Gan, et al. Multimodal autoregressive pre-training of large vision encoders. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9641–9654, 2025.
- <span id="page-25-6"></span>Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.
- <span id="page-25-9"></span>Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv:2405.21075, 2024.
- <span id="page-25-13"></span>Quentin Garrido, Laurent Najman, and Yann Lecun. Self-supervised learning of split invariant equivariant. Proceedings of Machine Learning Research, pages 10975–10996, 2023.
- <span id="page-25-18"></span>B. Girod, A.M. Aaron, S. Rane, and D. Rebollo-Monedero. Distributed video coding. Proceedings of the IEEE, 93(1): 71–83, 2005.
- <span id="page-25-5"></span>Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzyńska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The "something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017.

- <span id="page-26-11"></span>Sharut Gupta, Joshua Robinson, Derek Lim, Soledad Villar, and Stefanie Jegelka. Structuring representation geometry with rotationally equivariant contrastive learning. In ICLR, 2024.
- <span id="page-26-15"></span>Tengda Han, Weidi Xie, and Andrew Zisserman. Turbo Training with Token Dropout. In BMVC, page 622. BMVA Press, 2022.
- <span id="page-26-0"></span>Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv:2111.06377, 2021.
- <span id="page-26-14"></span>Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, et al. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. arXiv preprint arXiv:2402.03161, 2024.
- <span id="page-26-10"></span>Cijo Jose, Théo Moutakanni, Dahyun Kang, Federico Baldassarre, Timothée Darcet, Hu Xu, Daniel Li, Marc Szafraniec, Michaël Ramamonjisoa, Maxime Oquab, Oriane Siméoni, Huy V. Vo, Patrick Labatut, and Piotr Bojanowski. DINOv2 meets text: A unified framework for image- and pixel-level vision-language alignment. In CVPR, 2025.
- <span id="page-26-7"></span>Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.
- <span id="page-26-4"></span>Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.
- <span id="page-26-13"></span>Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm. IEEE Access, 2024.
- <span id="page-26-18"></span>Rajat Koner, Gagan Jain, Prateek Jain, Volker Tresp, and Sujoy Paul. Lookupvit: Compressing visual information to a limited number of tokens. In ECCV (86), pages 322–337. Springer, 2024.
- <span id="page-26-8"></span>Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In ICCV, 2011.
- <span id="page-26-19"></span>Dr T Senthil Kumar. A novel method for hdr video encoding, compression and quality evaluation. Journal of Innovative Image Processing, 1(2):71–80, 2019.
- <span id="page-26-2"></span>Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. NeurIPS, 2023.
- <span id="page-26-17"></span>Seon-Ho Lee, Jue Wang, Zhikang Zhang, David Fan, and Xinyu Li. Video token merging for long-form video understanding. arXiv:2410.23782, 2024.
- <span id="page-26-20"></span>Jiahao Li, Bin Li, and Yan Lu. Deep contextual video compression. NeurIPS, 34:18114–18125, 2021.
- <span id="page-26-9"></span>Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatiotemporal representation learning. arXiv:2201.04676, 2022a.
- <span id="page-26-3"></span>Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer. arXiv:2211.09552, 2022b.
- <span id="page-26-5"></span>Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark, 2023a.
- <span id="page-26-21"></span>Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In ICCV, 2023b.
- <span id="page-26-12"></span>Xianhang Li, Zeyu Wang, and Cihang Xie. CLIPA-v2: Scaling CLIP training with 81.1% zero-shot imagenet accuracy within a \$10,000 budget; an extra \$4,000 unlocks 81.8% accuracy. arXiv:2306.15658, 2023c.
- <span id="page-26-6"></span>Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European conference on computer vision (ECCV), pages 513–528, 2018.
- <span id="page-26-16"></span>Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations. CoRR, abs/2202.07800, 2022.
- <span id="page-26-1"></span>Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024a.

- <span id="page-27-15"></span>Yue Liu, Christos Matsoukas, Fredrik Strand, Hossein Azizpour, and Kevin Smith. PatchDropout: Economizing Vision Transformers Using Patch Dropout. In WACV, pages 3942–3951. IEEE, 2023.
- <span id="page-27-10"></span>Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 2024b.
- <span id="page-27-7"></span>Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022.
- <span id="page-27-8"></span>Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.
- <span id="page-27-9"></span>Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, 2022.
- <span id="page-27-17"></span>Fabian Mentzer, George Toderici, David Minnen, Sung-Jin Hwang, Sergi Caelles, Mario Lucic, and Eirikur Agustsson. Vct: A video compression transformer. arXiv:2206.07307, 2022.
- <span id="page-27-4"></span>Antoine Miech et al. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. ICCV, 2019.
- <span id="page-27-14"></span>Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan Willem van de Meent, and Robin Walters. Learning symmetric embeddings for equivariant world models. arXiv:2204.11371, 2022.
- <span id="page-27-11"></span>Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, joseph heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Joao Carreira. Perception test: A diagnostic benchmark for multimodal video models. In NeurIPS, pages 42748–42761. Curran Associates, Inc., 2023.
- <span id="page-27-1"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- <span id="page-27-16"></span>Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. NeurIPS, 34:13937–13949, 2021.
- <span id="page-27-6"></span>Oshmita Sarkar, Satyam Sinha, Ajay Kumar Jena, Ajaya Kumar Parida, Nirupama Parida, and Raj Kumar Parida. Automatic number plate character recognition using paddle-ocr. In 2024 International Conference on Innovations and Challenges in Emerging Technologies (ICICET), pages 1–7. IEEE, 2024.
- <span id="page-27-3"></span>Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv:2111.02114, 2021.
- <span id="page-27-5"></span>Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS Datasets and Benchmarks, 2022.
- <span id="page-27-12"></span>Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Saki Mizuno, Keita Suzuki, Ryo Masumura, Hiroaki Sugiyama, and Kuniko Saito. Tomato: Verbalizing the mental states of role-playing llms for benchmarking theory of mind. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1520–1528, 2025.
- <span id="page-27-0"></span>Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26160–26169, 2025.
- <span id="page-27-13"></span>Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charades-ego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626, 2018.
- <span id="page-27-2"></span>Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.

- <span id="page-28-2"></span>Mattia Soldan, Fabian Caba Heilbron, Bernard Ghanem, Josef Sivic, and Bryan Russell. Residualvit for efficient temporally dense video encoding. In ICCV, pages 22305–22315, 2025.
- <span id="page-28-4"></span>Enxin Song, Wenhao Chai, Shusheng Yang, Ethan Armand, Xiaojun Shan, Haiyang Xu, Jianwen Xie, and Zhuowen Tu. Videonsa: Native sparse attention scales video understanding. arXiv:2510.02295, 2025.
- <span id="page-28-8"></span>Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- <span id="page-28-18"></span>Gary J Sullivan and Thomas Wiegand. Video compression-from concepts to the h. 264/avc standard. Proceedings of the IEEE, 93(1):18–31, 2005.
- <span id="page-28-7"></span>Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the high efficiency video coding (hevc) standard. IEEE Transactions on circuits and systems for video technology, 22(12):1649–1668, 2012.
- <span id="page-28-15"></span>Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. EVA-CLIP: Improved training techniques for clip at scale. arXiv:2303.15389, 2023.
- <span id="page-28-0"></span>Ilya Sutskever. An observation on generalization. Talk at the Workshop on Large Language Models and Transformers, Simons Institute for the Theory of Computing, 2023. Accessed: 2023-XX-XX.
- <span id="page-28-6"></span>Feilong Tang, Xiang An, Haolin Yang, Yin Xie, Kaicheng Yang, Ming Hu, Zheng Cheng, Xingyu Zhou, Zimin Ran, Imran Razzak, Ziyong Feng, Behzad Bozorgtabar, Jiankang Deng, and Zongyuan Ge. Univit: Unifying image and video understanding in one vision encoder. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- <span id="page-28-9"></span>Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025.
- <span id="page-28-19"></span>Chenting Wang, Kunchang Li, Tianxiang Jiang, Xiangyu Zeng, Yi Wang, and Limin Wang. Make your training flexible: Towards deployment-efficient video models. arXiv:2503.14237, 2025.
- <span id="page-28-16"></span>Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In ICLR, 2023.
- <span id="page-28-1"></span>Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Jilan Xu, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. InternVideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024.
- <span id="page-28-3"></span>Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. LongVLM: Efficient Long Video Understanding via Large Language Models. In ECCV (33), pages 453–470. Springer, 2024.
- <span id="page-28-13"></span>Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024.
- <span id="page-28-12"></span>Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In CVPR, pages 9777–9786, 2021.
- <span id="page-28-11"></span>Chunyu Xie, Heng Cai, Jincheng Li, Fanjing Kong, Xiaoyu Wu, Jianfei Song, Henrique Morimitsu, Lin Yao, Dexin Wang, Xiangzheng Zhang, et al. Ccmb: A large-scale chinese cross-modal benchmark. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4219–4227, 2023.
- <span id="page-28-5"></span>Yin Xie, Kaicheng Yang, Xiang An, Kun Wu, Yongle Zhao, Weimo Deng, Zimin Ran, Yumeng Wang, Ziyong Feng, Roy Miles, et al. Region-based cluster discrimination for visual representation learning. In ICCV, pages 1793–1803, 2025.
- <span id="page-28-14"></span>Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling, 2022.
- <span id="page-28-10"></span>Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In ICLR, 2024.
- <span id="page-28-17"></span>Weili Xu, Enxin Song, Wenhao Chai, Xuexiang Wen, Tian Ye, and Gaoang Wang. Auroralong: Bringing rnns back to efficient open-ended video understanding. arXiv:2507.02591, 2025.

- <span id="page-29-5"></span>Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- <span id="page-29-0"></span>Haolin Yang, Feilong Tang, Lingxiao Zhao, Xiang An, Ming Hu, Huifa Li, Xinlin Zhuang, Yifan Lu, Xiaofeng Zhang, Abdalla Swikir, et al. Streamagent: Towards anticipatory agents for streaming video understanding. arXiv:2508.01875, 2025.
- <span id="page-29-10"></span>Ren Yang, Fabian Mentzer, Luc Van Gool, and Radu Timofte. Learning for video compression with recurrent auto-encoder and recurrent probability model. IEEE Journal of Selected Topics in Signal Processing, 15(2):388–401, 2021.
- <span id="page-29-6"></span>Hongxu Yin, Arash Vahdat, José M. Álvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-ViT: Adaptive Tokens for Efficient Vision Transformer. In CVPR, pages 10799–10808. IEEE, 2022.
- <span id="page-29-7"></span>Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang. Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer. In CVPR, pages 11091–11101. IEEE, 2022.
- <span id="page-29-1"></span>Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.
- <span id="page-29-11"></span>Guozhen Zhang, Yuhan Zhu, Haonan Wang, Youxin Chen, Gangshan Wu, and Limin Wang. Extracting motion and appearance via inter-frame attention for efficient video frame interpolation. In CVPR, pages 5682–5692, 2023.
- <span id="page-29-3"></span>Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models. In ACL, 2025.
- <span id="page-29-2"></span>Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, 2024.
- <span id="page-29-9"></span>Zhuo Zhao and Ping Liang. A highly efficient parallel algorithm for h. 264 video encoder. In 2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings, pages V–V. IEEE, 2006.
- <span id="page-29-12"></span>Zijia Zhao, Yuqi Huo, Tongtian Yue, Longteng Guo, Haoyu Lu, Bingning Wang, Weipeng Chen, and Jing Liu. Efficient motion-aware video mllm. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24159–24168, 2025.
- <span id="page-29-4"></span>Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv:2406.04264, 2024.
- <span id="page-29-8"></span>Xingyi Zhou, Anurag Arnab, Chen Sun, and Cordelia Schmid. How can objects help action recognition? In CVPR, pages 2353–2362. IEEE, 2023.