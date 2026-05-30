# **DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding**

Jialuo Li<sup>1</sup>*,*2*,*<sup>∗</sup> Bin Li<sup>2</sup> Jiahao Li<sup>2</sup> Yan Lu<sup>2</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Microsoft Research Asia

# **Abstract**

The application of Large Multimodal Models (LMMs) to long-form video understanding is constrained by limited context lengths and the computationally prohibitive cost of processing dense video tokens. Consequently, recent research has focused on query-aware frame selection, methods that often incur significant computational overhead. This paper challenges the assumption that such complex search mechanisms are universally necessary. We first identify and validate a query typology distinguishing between **global query** and **localized query**. We demonstrate that while uniform sampling is both effective and efficient for global queries, localized queries indeed necessitate query-aware selection for optimal performance. Building on this insight, we propose **DIG**, a training-free frame selection framework that adapts its strategy based on the query type. Specifically, **DIG** employs efficient uniform sampling for global queries while activating a specialized pipeline to extract query-relevant frames for localized queries. Experiments on three long-form video understanding benchmarks demonstrate that **DIG** consistently outperforms existing baselines and robustly improves LMM performance, even when scaling the input frame count to 256. The code is available at *<https://github.com/Jialuo-Li/DIG>*.

# **1 Introduction**

In recent years, there has been a rapid advancement in large multimodal models (LMMs) [\[1](#page-10-0)[–5\]](#page-10-1) for open-world visual understanding. A natural and increasingly important direction within this field is the extension of these models to handle video data, thereby enabling them to perform complex video understanding tasks [\[6–](#page-10-2)[14\]](#page-11-0). The common approach [\[15,](#page-11-1) [16\]](#page-11-2) involves representing videos as sequences of individual frames, where visual features are extracted from each frame and concatenated to form a video representation that is subsequently processed by the large language model (LLM). However, due to the limited context length of the LLM and the sheer volume of video tokens, it is impractical to input all frames directly. As a result, only a sampled subset of frames is typically used as input. The predominant method is uniform sampling which selects frames at fixed intervals. While this maximizes temporal coverage, it is query-agnostic, often selecting redundant frames while omitting crucial, query-relevant moments that are essential for accurate reasoning.

To address this limitation, recent work has introduced query-aware adaptive frame selection mechanisms [\[17](#page-11-3)[–21\]](#page-11-4). These methods identify and utilize the most representative frames as input based on the query, but at the cost of significant computational overhead for searching within the video. This high cost motivates a critical question that is frequently overlooked: *Is such a complex search mechanism strictly necessary for all query types?* Our findings indicate that the answer is negative. We first identify the existence of two distinct query categories: *global query*, which requires holistic video understanding, and *localized query*, which targets specific temporal segments. We observe a significant performance disparity in uniform sampling between these categories. As the number of sampled frames increases, performance on localized queries degrades substantially, as irrelevant frames are injected into the context. Conversely, performance on global queries remains stable. This finding validates our query typology. Based on this, our further experiments demonstrate that for global queries, uniform sampling already achieves robust performance. In such

<sup>\*</sup> Work done during Jialuo's internship at MSRA.

cases, deploying more complex selection methods is often inefficient and yields diminishing returns. Conversely, it is for localized queries that advanced, query-aware selection mechanisms are truly impactful, delivering substantial performance gains where uniform sampling fails, highlighting the need for a dynamic, query-dependent sampling strategy.

Building on these findings, we propose DIG, a trainingfree frame selection framework for LMM that adapts its overall strategy based on the specific query type. The framework first employs an LLM to automatically classify a given query as either global or localized. For global queries, standard uniform sampling is employed. For localized queries, a highly targeted multistage pipeline is initiated. This pipeline begins with our proposed Content-Adaptive Frame Selection, a method that leverages pairwise frame similarity based on DINO features [22] to select a set of semantically representative *r-frames*. Subsequently, the LMM itself is then utilized to score these *r-frames*, assigning a relevance reward based on their estimated utility in answering the query. Guided by this reward distribution, a video refinement process identifies and merges the most visually relevant video segments into a more condensed repre-

![](_page_1_Figure_4.jpeg)

**Figure 1:** For global queries requiring holistic understanding (bottom), uniform sampling is both effective and efficient. Conversely, for localized queries targeting specific temporal segments (top), query-based sampling is necessary to ensure high performance.

sentation. Finally, this refined video is uniformly sampled to get the input frames for the LMM, ensuring that final inference is concentrated on the most pertinent temporal segments.

Our main contributions are summarized as follows:

- We identify a query typology (global vs. localized) and demonstrate that the efficacy of frame selection strategies is highly contingent on this classification.
- We propose **DIG**, a training-free frame selection framework that adapts to query type by employing uniform sampling for global queries and a specialized pipeline to extract query-relevant frames for localized queries.
- Experiments on three long-form video understanding benchmarks show that **DIG** consistently outperforms existing baselines and robustly improves LMM's performance, even when scaling input frame count to 256.

### 2 Related Work

#### 2.1 Video-based Large Multimodal Models

The rise of Transformer-based large language models (LLMs) has revolutionized natural language processing, with major advances stemming from increased model scale and larger pre-training datasets [23–30]. Inspired by this success, researchers have begun adapting LLMs to process multiple modalities, particularly integrating visual elements like images and videos [1, 3, 31, 32], leading to the development of LMMs. Through extensive training, these models learn rich, cross-modal representations that effectively connect visual and textual information. This evolution has led to significant improvements across a range of video understanding applications, including tasks such as video captioning [33–37] and video question answering [6, 13, 38–41]. Ongoing research is also focusing on refining model architectures [42–45] and optimizing training strategies [32, 44] to further boost the performance of these systems. Despite their success, LMMs still struggle in video understanding due to the high volume of video tokens and the limited context length [46, 47], as well as the "Needle-in-a-Haystack" issue [20, 48, 49]. These challenges highlight the need for efficient frame selection techniques that capture key visual content without overloading the model.

### 2.2 Video Token Reduction for the VQA Task

In VQA task, uniform frame sampling is a standard technique for video token reduction. However, this method overlooks the query-specific relevance of individual frames. To address this limitation, recent research has focused on adaptive token reduction mechanisms, which are broadly classified into two primary categories.

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Figure 2: Performance trends of various LMMs with respect to input frame counts. We evaluate Qwen2.5-VL-7B [16], InternVL3-8B [5], and LLaVA-OneVision-7B [3] on MLVU [54], LongVideoBench [55], and VideoMME [56]. Results indicate that accuracy peaks at an optimal frame count and subsequently degrades, rather than improving monotonically.

**Token Compression.** This strategy carefully consolidates information within or across video frames to create a significantly more compact yet informative representation, reducing the total number of tokens needed [50]. Various advanced techniques are employed to achieve this, such as using a memory bank [51], reducing temporal redundancy [52, 53], and applying hierarchical compression [49]. Despite their inherent efficiency, token compression techniques may often lead to excessive summarization, resulting in the loss of critical fine-grained visual details. Moreover, highly query-related segments may be either compressed or overly generalized, ultimately compromising the model's capacity to effectively respond to the given query.

Query-Based Frame Selection. Compared with uniform sampling, recent methods employ more refined strategies to select query-relevant frames that typically involve three key steps: (1) uniformly sample candidate frames [17, 18, 21, 57–60] or video segments [61, 62]; (2) assess their relevance to the query using metrics like CLIPScore [63, 64], detector [65] or learned models [19]; (3) apply an algorithm to select the most relevant frames based on these scores. However, uniform sampling often balances poorly between information sparsity and computational load. Furthermore, relevance metrics like CLIPScore [63] can be notoriously unreliable for complex reasoning, and the resulting temporally sparse frames may miss fine-grained details found in continuous clips. We address these limitations by: (1) using content-adaptive frame selection to identify frame candidates much more intelligently; (2) employing inherently more reliable LMMs for relevance assessment; and (3) retrieving and concatenating continuous clips corresponding to candidates before performing final frame selection, ensuring fine-grained information is preserved.

# <span id="page-2-1"></span>3 Revisiting Inference Mechanism of LMM in Video Understanding

Consider a video V with T frames, denoted as  $\{f_i\}_{i=1}^T$ , along with a query Q. In VQA task, the model receives the video V and the query Q as inputs and is tasked with generating a response A that accurately addresses the query. In contemporary approaches, due to computational limitations and the language model's restricted context length L, only a subset of N uniformly sampled frames, denoted as  $\{f_i'\}_{i=1}^N$ , is processed, where  $N \ll T$ . These selected frames are then combined with the query Q and fed into the LMM, which autoregressively generates the answer A:

$$A = LMM([f'_1; f'_2; \dots; f'_N; Q]).$$
(1)

Obviously, a small subset of N frames is often insufficient to capture the full content of a video, particularly in longer sequences. To address this, recent studies [66, 67] have focused on extending model context lengths to allow more frames as input. However, this raises an important question: Does increasing the number of uniformly sampled input frames enhance performance on VQA task?

More frames do not mean improved performance. To investigate this, we conducted an evaluation using three pretrained LMMs: Qwen2.5-VL-7B [16], InternVL3-8B [68], and LLaVA-OneVision-7B [3], across three long-form video understanding benchmarks: MLVU [54], VideoMME [56], and LongVideoBench [55]. We employed uniform frame sampling with varying frame counts to evaluate the impact of frame count on model performance. As illustrated in Figure 2, a consistent pattern emerges across all models and benchmarks: performance initially improves with more input frames but declines beyond a certain point.

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

**Figure 3: Relative accuracy on localized and global queries.** Plotted as the deviation from the initial baseline, the results demonstrate that performance degradation at high frame counts is predominantly attributed to LQ, while GQ remains relatively stable.

**Query classification.** To better understand the underlying causes of this performance degradation, we systematically examined the impact of different query types. Prior studies [17, 18, 69–71] have already identified a class of queries that relate directly to specific, localized segments of a video, such as "What kind of bike is the man riding?", which we now classify as localized queries (LQ). However, these works frequently overlook another important category of queries requiring a comprehensive understanding of the entire video. We define such queries as global queries (GQ), with a typical example being "What title best summarizes this video?"

**Performance trends vary across query types.** Following the definition, we manually categorize queries from MLVU [54] and VideoMME [56], and evaluate the same models on these two query types. As shown in Figure 3, while performance on global queries remains relatively stable with increasing frame count, performance on localized queries drops significantly. We attribute this to global queries benefiting from holistic information, whereas excess frames introduce noise for localized tasks. These results highlight the necessity of pre-classifying query types to optimize efficiency; specifically, global queries can rely on standard uniform sampling, avoiding the computational overhead of key frame search techniques.

### 4 Method: DIG

Overview. In this section, we formally introduce  $\mathbf{DIG}$ , a novel, training-free frame selection framework for LMMs that dynamically adapts to the query type.  $\mathbf{DIG}$  begins by classifying the given query as either localized or global (§4.1). For global queries, the final input frames are uniformly sampled across the entire video. In contrast, for localized queries, we first employ a content-adaptive frame selection method to extract highly representative frames (§4.2), which are then evaluated by the LMM through reward scoring to assess their relevance to the query (§4.3). Then a refined video is carefully constructed through a search procedure guided by these rewards (§4.4) and final input frames are uniformly sampled from the refined video.

### <span id="page-3-1"></span>4.1 Query Type Identification

As established in Section 3, the performance trends vary across query types. Therefore, we first employ a LLM to classify a given query Q as either global or localized (see Appendix  $\mathbb C$  for prompt details). For global queries, the LMM performs direct inference on uniformly sampled frames. Localized queries, in contrast, are addressed using the specialized approach detailed below.

### <span id="page-3-2"></span>4.2 Content-Adaptive Frame Selection (CAFS)

To effectively address the localized query, it is essential to extract relevant frames from the video. However, exhaustive frame-wise analysis of long-form videos is computationally infeasible. This necessitates obtaining a compact yet informative subset of frames. Previous methods typically rely on static sampling (e.g., uniform or fixed-rate) [17, 18, 64, 65]. This static approach presents a dilemma: low-rate sampling may yield a sparse representation that misses critical events, while high-rate sampling produces a large and redundant frame set. To address this, we propose *Content-Adaptive Frame Selection*, a method that adaptively selects representative frames, referred to as *r-frames*, based on high-level semantic content in the video such as objects and scenes.

![](_page_4_Figure_2.jpeg)

**Figure 4:** Overview of DIG. The LLM first classifies the query type. Global queries utilize uniform sampling across the entire video, while localized queries employ CAFS and reward assignment to construct a refined video prior to sampling. The selected frames are subsequently processed by the LMM for final inference.

**Distance calculation.** Given a 2-fps sampled video with M frames  $\{f_{I_i}\}_{i=1}^M$  with their corresponding frame indices  $\{I_i\}_{i=1}^M$ , we first utilize DINOv2 [22] to extract robust visual features from each frame, which results in a sequence of feature vectors  $\{V_{I_i}\}_{i=1}^M$ . To accurately measure the dissimilarity between these consecutive frames, we compute the feature distance  $d_i$  between  $f_{I_i}$  and  $f_{I_{i+1}}$  using the following formula:

$$d_i = 1 - \sin(V_{I_i}, V_{I_{i+1}}), \tag{2}$$

where  $\text{sim}(\cdot,\cdot)$  denotes cosine similarity. This yields a sequence of distances  $\{d_i\}_{i=1}^{M-1}$ .

**R-Frame selection.** Due to frequent scene transitions or camera cuts in long videos, the pairwise frame similarity often exhibits abrupt changes, resulting in numerous peaks in the distance sequence. Specifically,  $d_i$  is identified as a peak if  $d_{i-1} < d_i$  and  $d_{i+1} < d_i$ . To reduce noise effects, only peaks with prominence greater than 0.1 are valid. This threshold has been found effective through empirical observation. We denote the indices of these valid peaks as  $\{K_j\}_{j=1}^N \subset \{I_i\}_{i=1}^M$ , where N < M. These peaks serve as segmentation points, dividing the video into distinct segments. Within each segment, the low pairwise distances between frames indicate visual consistency. Therefore, we select only one frame from each segment to capture its semantic content. For simplicity, we choose the midpoint frame of each segment, resulting in a set of r-frames indexed by  $\{I'_j\}_{j=1}^{N-1} = \{(K_j + K_{j+1})/2\}_{j=1}^{N-1}$ . By aggregating r-frames, we obtain a compact representation that effectively summarizes the essential visual content of the entire video.

### <span id="page-4-0"></span>4.3 Reward Assignment

To accurately identify the relevance of r-frames to the given query Q, existing methods typically use either: (1) multimodal models like CLIPScore [17, 18, 57, 72], or (2) object detection models to localize specific query-related entities in individual frames [20]. However, these traditional methods are often severely constrained by mere surface-level feature matching and reliance on fixed vocabularies, which fundamentally limits their ability to capture complex contextual reasoning and broader world knowledge. To address this, we directly leverage the LMM itself to assess frame relevance by assigning reward scores, with a simplified version of our prompt below.

**Two-dimensional scoring.** Since many queries, particularly those involving "why" or "how", cannot be fully addressed by a single frame, evaluating the relevance of individual frames independently may lead to incomplete or biased assessments. To mitigate this, we design the LMM to consider two complementary factors: (1) the direct relevance of the current frame to the query, and (2) whether the content of the current frame indicates that adjacent frames may contain supplementary information that contributes to a more comprehensive response.

#### Reward Model Prompt (Simplified)

Frame:  $\langle f_i \rangle$ ; Query:  $\langle Q \rangle$ ; Please follow these steps to finish scoring:

- 1. Describe the sampled frame, focusing only on elements relevant to the question, if any.
- 2. Assign a relevance score between 0 and 100 based on: (1) Direct usefulness of the frame for answering the query. (2) Whether it suggests adjacent frames may contain relevant context.

### <span id="page-5-0"></span>4.4 Video Refinement

Building upon the preceding steps, we have obtained the set of peak indices  $\{K_j\}_{j=1}^N$ , the *r-frame* indices  $\{I_j'\}_{j=1}^{N-1}$ , and the reward values  $\{R_j\}_{j=1}^{N-1}$  assigned to these *r-frames*. The next step is to select the most query-relevant *r-frames* based on the reward values  $\{R_j\}_{j=1}^{N-1}$ .

**Iterative reward-guided selection.** In contrast to the commonly employed Top-K selection, which applies a fixed hyperparameter across varying scenarios, we introduce a parameter-free methodology. Given the initial rewards  $\{R_j\}_{j=1}^{N-1}$ , we iteratively refine this set until it stabilizes.

- *Step 1.* Compute the mean of the current reward set:  $\overline{R}$ .
- Step 2. Update each reward value by thresholding below the mean value:

$$R'_{j} = \max(R_{j} - \overline{R}, 0), \quad \forall j = 1, \dots, N - 1.$$

$$(3)$$

Step 3. Finally, let S be the resulting set of candidate indices {j | R'<sub>j</sub> > 0}. Compare S directly with the set of positive indices obtained from the previous iteration. If S is strictly unchanged, terminate the entire iteration process. Otherwise, update the current reward set {R<sub>j</sub>} ← {R'<sub>j</sub>} and repeat from Step 1.

Upon termination, the selected *r-frames*, denoted by  $I_f$ , are formally defined as those *r-frames* whose corresponding reward values in the final iteration are positive:  $I_f = \{I'_j \mid R'_j > 0\}_{j=1}^{N-1}$ . This criterion ensures that all *r-frames* in the final selection set possess a reward larger than average.

**Segment combination.** Since r-frames exhibit high feature similarity with their adjacent frames, it indicates an opportunity to incorporate fine-grained information beyond simply using them as input to the LMM. Specifically, for each selected r-frame indexed by  $I'_j$ , we consider the video segment in the interval  $[K_j, K_{j+1}]$  for richer temporal details. To capture more relevant context, we also consider adjacent r-frames within a window of length wlen, specifically those with index range from  $I'_{j-wlen}$  to  $I'_{j+wlen}$ . This results in the video segment spanning the index range  $[K_{j-wlen}, K_{j+wlen+1}]$ . Then we combine the corresponding video segments of all selected r-frames via union operation, resulting in a refined video containing query-relevant and fine-grained content. Finally, we uniformly sample frames from this refined video as input to the LMM.

### <span id="page-5-1"></span>5 Experiment

### 5.1 Experiment Settings

**Datasets.** We comprehensively evaluate our proposed approach on three benchmarks: MLVU [54], LVB [55], and VideoMME [56], which contain complex videos ranging from several minutes to multiple hours, allowing us to assess long-form video understanding capabilities. For VideoMME [56], we focus only on the medium and long splits. We don't use any subtitles, ensuring that evaluation is strictly based on pure visual understanding. Further benchmark details are provided in Appendix A.

**Implementation details.** The LMMs used are Qwen2.5-VL-7B [16] and Qwen2.5-VL-32B [16]. The LLM used for query identification is Qwen3-Next-80B-A3B [73]. Each input frame is represented using 56 tokens. The hyperparameter *wlen* is set to 2. All experiments are conducted on 8 A100 GPUs within LMMs-Eval [74] framework. Additionally, we utilize vLLM backend [75] to accelerate inference during the query identification and reward assignment stages. As baselines, we choose AKS [18] and Q-Frame [64], and uniform sampling (UNI). Detailed baseline configurations and extended experiments on Qwen3-VL-8B [76] are available in Appendix F.

### 5.2 Main Results

Comparison with existing methods. As shown in Table 1, compared with uniform sampling and competitive baselines including Q-Frame [64] and AKS [18], **DIG** consistently improves performance on both Qwen2.5-VL-32B [16] and Qwen2.5-VL-7B [16] across input frame numbers from 8 to 256. Notably, with 32 frames, **DIG** significantly boosts the accuracy of Qwen2.5-VL-7B [16] by 7.68% on MLVU [54] and 4.51% on LongVideoBench [55] compared to uniform sampling. This superiority extends to the more powerful Qwen2.5-VL-32B [16], where **DIG** achieves better performance across almost all reported settings, effectively enhancing even a strong base model where other methods struggle to show consistent gains.

<span id="page-6-0"></span>**Table 1:** *Performance comparison between different frame selection methods. Base LMMs are Qwen2.5-VL-32B [\[16\]](#page-11-2) (left) and Qwen2.5-VL-7B [\[16\]](#page-11-2) (right). Bold indicates best performance, while Red Box denote results inferior to uniform sampling.*

| Method       | #Frames | MLVU  | LVB   | VideoMME |       |
|--------------|---------|-------|-------|----------|-------|
|              |         |       |       | Medium   | Long  |
| UNI          | 8       | 55.93 | 53.40 | 53.89    | 51.56 |
| Q-Frame [64] | 8       | 56.03 | 53.78 | 54.03    | 49.63 |
| DIG (Ours)   | 8       | 61.55 | 56.77 | 54.12    | 51.21 |
| UNI          | 16      | 58.79 | 54.67 | 55.44    | 53.33 |
| Q-Frame [64] | 16      | 57.73 | 56.62 | 55.09    | 51.11 |
| DIG (Ours)   | 16      | 66.21 | 58.86 | 58.62    | 52.18 |
| UNI          | 32      | 61.91 | 57.89 | 57.89    | 53.33 |
| AKS [18]     | 32      | 66.42 | 59.31 | 59.89    | 56.00 |
| Q-Frame [64] | 32      | 60.95 | 57.37 | 60.43    | 55.90 |
| DIG (Ours)   | 32      | 70.69 | 61.86 | 60.87    | 57.76 |
| UNI          | 64      | 66.24 | 59.01 | 64.33    | 55.67 |
| AKS [18]     | 64      | 69.41 | 61.41 | 64.67    | 58.44 |
| Q-Frame [64] | 64      | 66.05 | 59.61 | 62.80    | 57.72 |
| DIG (Ours)   | 64      | 74.19 | 63.65 | 66.24    | 58.19 |
| UNI          | 128     | 70.24 | 61.78 | 68.89    | 59.67 |
| AKS [18]     | 128     | 72.77 | 62.00 | 68.33    | 61.44 |
| Q-Frame [64] | 128     | 70.10 | 60.06 | 68.21    | 59.28 |
| DIG (Ours)   | 128     | 75.20 | 65.60 | 69.00    | 62.29 |
| UNI          | 192     | 71.76 | 63.80 | 69.56    | 62.00 |
| AKS [18]     | 192     | 73.46 | 62.45 | 69.89    | 61.00 |
| DIG (Ours)   | 192     | 76.66 | 66.42 | 70.11    | 63.42 |

|              |         |       | LVB   | VideoMME |       |
|--------------|---------|-------|-------|----------|-------|
| Method       | #Frames | MLVU  |       | Medium   | Long  |
| UNI          | 8       | 53.64 | 51.23 | 51.36    | 45.84 |
| Q-Frame [64] | 8       | 54.42 | 54.23 | 50.81    | 49.21 |
| DIG (Ours)   | 8       | 58.64 | 55.20 | 54.23    | 46.88 |
| UNI          | 16      | 56.43 | 54.45 | 55.94    | 48.12 |
| Q-Frame [64] | 16      | 56.81 | 57.37 | 53.78    | 49.02 |
| DIG (Ours)   | 16      | 63.98 | 57.89 | 56.81    | 51.93 |
| UNI          | 32      | 59.52 | 56.92 | 59.08    | 52.02 |
| AKS [18]     | 32      | 65.07 | 59.31 | 59.22    | 53.11 |
| Q-Frame [64] | 32      | 60.03 | 56.39 | 56.64    | 51.57 |
| DIG (Ours)   | 32      | 67.20 | 60.43 | 61.62    | 53.24 |
| UNI          | 64      | 63.61 | 58.94 | 61.01    | 51.27 |
| AKS [18]     | 64      | 66.59 | 60.66 | 62.94    | 53.44 |
| Q-Frame [64] | 64      | 63.43 | 57.52 | 61.32    | 53.70 |
| DIG (Ours)   | 64      | 70.65 | 61.41 | 62.61    | 55.30 |
| UNI          | 128     | 67.31 | 61.86 | 65.89    | 54.84 |
| AKS [18]     | 128     | 68.68 | 60.36 | 65.67    | 55.93 |
| Q-Frame [64] | 128     | 68.03 | 59.76 | 65.91    | 54.81 |
| DIG (Ours)   | 128     | 71.40 | 63.13 | 66.78    | 55.69 |
| UNI          | 192     | 69.03 | 61.93 | 67.01    | 55.82 |
| AKS [18]     | 192     | 69.93 | 61.26 | 68.22    | 54.41 |
| DIG (Ours)   | 192     | 72.32 | 64.32 | 68.00    | 58.24 |
| UNI          | 256     | 69.15 | 61.48 | 66.31    | 57.12 |
| AKS [18]     | 256     | 71.50 | 61.03 | 67.56    | 55.11 |
| DIG (Ours)   | 256     | 72.46 | 64.62 | 67.66    | 57.76 |

**Scalability and performance consistency.** In well-resourced environments, performance analysis at minimal frame counts (e.g., 8 or 16) offers limited practical insight, as applications typically seek to maximize frame utilization within given constraints. Therefore, unlike most previous works [\[17](#page-11-3)[–19,](#page-11-10) [21,](#page-11-4) [64\]](#page-14-10) that validate performance in low-frame regimes (*<* 64 frames), we conduct an evaluation that scales inputs to high frame densities (e.g., 256 frames). Under these conditions, as detailed in Table [1,](#page-6-0) AKS [\[18\]](#page-11-9) and Q-Frame [\[64\]](#page-14-10) can exhibit performance degradation relative to uniform sampling as frame counts increase. For instance, when utilizing the Qwen2.5-VL-7B [\[16\]](#page-11-2) with 128 input frames, both AKS [\[18\]](#page-11-9) and Q-Frame [\[64\]](#page-14-10) underperformed uniform sampling by 1–2% on LongVideoBench [\[55\]](#page-14-1). In contrast, **DIG** demonstrates consistent performance gains over uniform sampling across all tested LMMs and most input frame configurations.

# <span id="page-6-1"></span>**6 Discussion and Analysis**

To thoroughly evaluate the specific contributions of each individual module, in this section, we present a detailed analysis of **DIG**. Our evaluation is structured around the following key questions:

- *How does the choice of frame selection strategy impact performance on global versus localized queries?* ([§6.1\)](#page-7-0)
- *How effective is the CAFS module at selecting representative frames, and what is its contribution to the overall performance of DIG?* ([§6.2\)](#page-7-1)
- *How does the LMM-based reward model compare against the CLIPScore [\[63\]](#page-14-9) in reward assignment?* ([§6.3\)](#page-9-0)
- *What is the influence of the temporal window length (wlen) on model performance?* ([§6.4\)](#page-9-1)
- *What is the computational efficiency of DIG?* ([§6.5\)](#page-9-2)

<span id="page-7-2"></span>**Figure 5:** Comparison of our proposed frame selection pipeline (Sections 4.2–4.4) versus uniform sampling across different query types. The base LMMs are Owen2.5-VL-7B [16] and Owen2.5-VL-32B [16].

<span id="page-7-3"></span>![](_page_7_Figure_4.jpeg)

**Figure 6:** GlC and LoC scores across varying video durations. We compare three sampling strategies: FPS, UNI, and CAFS. In each sub-figure, the lines represent the score (left y-axis), while the bars indicate the number of sampled frames (right y-axis).

# <span id="page-7-0"></span>6.1 Frame Selection Strategy vs. Query Type

To examine the impact of different frame selection strategies on global versus localized queries, we leverage the query classifications established in Section 3 and then compare the performance of uniform sampling against our proposed frame selection pipeline on each query type.

**Efficacy of uniform sampling on GQ.** As clearly illustrated in the right two charts of Figure 5, standard uniform sampling consistently achieves performance comparable to, or occasionally even superior to, our complex pipeline on GQs. This observation suggests that global queries generally necessitate comprehensive and temporally diverse information from the video content, which uniform sampling effectively provides.

**Superiority of keyframe selection on LQ.** For LQs, our pipeline consistently outperforms uniform sampling, as shown in the left three charts of Figure 5. This result demonstrates our method's effectiveness in accurately identifying and extracting the specific video segments relevant to localized inquiries. These findings underscore the importance of a query-aware sampling strategy: identifying the query type is essential to determine whether to employ broad sampling for global context or targeted extraction for specific details.

### <span id="page-7-1"></span>6.2 Analysis of CAFS Effectiveness

Let  $f_j$  denote the frame indexed by j, and let  $V_j$  represent its feature vector obtained via DINOv2 [22]. We define the set of r-frames as  $\{f_{I_i}\}_{i=1}^N$  with indices  $\{I_i\}_{i=1}^N$ . To assess their effectiveness in capturing the high-level semantic content within a video, we introduce two quantitative metrics.

**Localized Coverage (LoC).** This metric assesses the effectiveness with which each r-frame captures its local temporal visual context. More specifically, for each r-frame  $f_{I_i}$ , four neighboring frames are sampled uniformly from its surrounding temporal window. The LoC score is then computed as the average similarity between the r-frame and its sampled neighbors across all r-frames.

$$LoC = \frac{1}{4N} \sum_{i=1}^{N} \sum_{j=0}^{3} sim \left( V_{I_i}, V_{M_{i,j}} \right),$$
where  $M_{i,j} = I_i + (j-1.5) \cdot \lfloor (I_{i+1} - I_{i-1})/6 \rfloor$ 

<span id="page-8-1"></span>**Table 2:** Performance comparison with rewards from Qwen2.5-VL-32B [16], Qwen2.5-VL-7B [16] and CLIPScore [63] across various benchmarks. **Bold** indicates best performance. The base LMM used is Qwen2.5-VL-7B [16].

| Method              | #Frames | MLVU | LVB  |             | VideoMME |      |
|---------------------|---------|------|------|-------------|----------|------|
|                     |         |      | 2,2  | Short       | Medium   | Long |
| CLIPScore [63]      | 8       | 57.4 | 52.6 | 62.3        | 51.1     | 49.0 |
| Qwen2.5-VL-7B [16]  | 8       | 58.6 | 55.2 | 63.6        | 54.2     | 46.9 |
| Qwen2.5-VL-32B [16] | 8       | 60.6 | 55.6 | 64.2        | 52.6     | 47.2 |
| CLIPScore [63]      | 16      | 62.2 | 54.3 | 67.2        | 55.9     | 49.4 |
| Qwen2.5-VL-7B [16]  | 16      | 64.0 | 57.9 | 67.8        | 56.8     | 51.9 |
| Qwen2.5-VL-32B [16] | 16      | 64.0 | 59.2 | 68.1        | 57.6     | 50.2 |
| CLIPScore [63]      | 32      | 65.4 | 56.2 | 70.0        | 58.6     | 51.2 |
| Qwen2.5-VL-7B [16]  | 32      | 67.2 | 60.4 | 70.3        | 61.6     | 53.2 |
| Qwen2.5-VL-32B [16] | 32      | 67.9 | 60.6 | 72.6        | 61.4     | 53.1 |
| CLIPScore [63]      | 64      | 67.2 | 59.6 | 72.7        | 62.4     | 54.7 |
| Qwen2.5-VL-7B [16]  | 64      | 70.7 | 61.4 | 73.3        | 62.6     | 55.3 |
| Qwen2.5-VL-32B [16] | 64      | 71.0 | 63.4 | 74.4        | 64.8     | 54.7 |
| CLIPScore [63]      | 128     | 69.6 | 61.0 | 73.3        | 64.0     | 55.8 |
| Qwen2.5-VL-7B [16]  | 128     | 71.4 | 63.1 | 74.9        | 66.8     | 55.7 |
| Qwen2.5-VL-32B [16] | 128     | 72.6 | 65.2 | <b>75.4</b> | 69.2     | 57.1 |
| CLIPScore [63]      | 192     | 71.0 | 62.5 | 74.6        | 63.9     | 54.8 |
| Qwen2.5-VL-7B [16]  | 192     | 72.3 | 64.3 | 75.9        | 68.0     | 58.2 |
| Qwen2.5-VL-32B [16] | 192     | 73.9 | 65.4 | 76.2        | 69.2     | 57.4 |
| CLIPScore [63]      | 256     | 71.2 | 61.9 | 75.0        | 64.7     | 57.0 |
| Qwen2.5-VL-7B [16]  | 256     | 72.5 | 64.6 | 76.3        | 67.7     | 57.8 |
| Qwen2.5-VL-32B [16] | 256     | 74.3 | 64.5 | 76.8        | 68.9     | 59.1 |

<span id="page-8-0"></span>![](_page_8_Figure_4.jpeg)

**Figure 7:** *Performance Comparison of CAFS and uniform sampling in DIG pipeline.* The base LMM is Qwen2.5-VL-7B [16].

**Global Coverage (GIC).** This metric evaluates how well the *r-frames* collectively represent the entire video content. Ideally, each frame in the video should be similar to at least one *r-frame*. To compute it, we randomly sample 200 frames from the video, denoted as  $\{f_x\}_{x\in\mathcal{X}}$ . For each frame  $f_x$ , we find the maximum similarity to any *r-frame* and average these values across all sampled frames:

$$GIC = \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} \max_{i \in [1, N]} sim(V_{I_i}, V_x)$$

$$(5)$$

**Baseline selection.** We evaluate CAFS against two standard baselines: UNI (uniform frame sampling) and FPS (uniform frames-per-second sampling). The assessment is conducted on MLVU [54] and VideoMME [56]. To ensure a fair comparison, the average number of selected frames is kept consistent across all methods.

**Analysis.** As shown in Figure 6, the overall performance of standard uniform sampling declines with increasing video duration. This limitation arises from using a fixed number of frames across videos of varying lengths, which inevitably leads to significant redundancy in short videos and inadequate semantic coverage in long videos. Moreover, while regular fps sampling maintains stable performance, CAFS consistently outperforms it, particularly for videos

<span id="page-9-3"></span>![](_page_9_Figure_2.jpeg)

Figure 8: Performance comparison of different window lengths (wlen) in DIG pipeline. The base LMM is Owen2.5-VL-32B [16].

over 10 minutes. This indicates that key semantic information in videos does not grow linearly with length, and that CAFS is more effective at selecting informative frames.

**Comparison with uniform sampling in DIG.** We compare CAFS with uniform sampling within the **DIG** by replacing CAFS-extracted *r-frames* with standard uniformly sampled ones. As experimentally shown in Figure 7, CAFS robustly outperforms uniform sampling across all benchmarks. In addition, the observed performance gap widens with more input frames, further highlighting the fundamental limitation of uniform sampling: for long videos it cannot sample sufficient frames to adequately cover information for the subsequent reasoning process, while CAFS can effectively adapt to videos of any length and ensures significantly better coverage.

### <span id="page-9-0"></span>6.3 Reward Assignment: LMM vs. CLIPScore

We evaluate the reward assignment mechanism employed by the LMMs in **DIG** by comparing it to a common alternative: computing frame-query similarity using CLIP [72]. Specifically, we substitute all reward values originally assigned by the LMM with corresponding CLIPScore [63].

LMMs exhibit superior capability as reward assigners. As illustrated in Table 2, the rewards generated by LMMs (Qwen2.5-VL-7B/32B [16]) demonstrate superior performance across the benchmarks in most cases, particularly as the number of frames increases. This underscores the LMM's capacity to deliver more precise and semantically rich reward signals through its advanced reasoning abilities and broad world knowledge. In contrast, CLIPScore [63] depends on superficial feature matching and often fails to capture nuanced or visually complex query requirements.

Better LMMs yield superior rewards. The experimental results in Table 2 also clearly indicate that employing the larger Qwen2.5-VL-32B [16] as the reward assigner outperforms the smaller 7B variant, even on a short-video benchmark like VideoMME-short [56]. This confirms that more advanced LMMs provide considerably more precise reward signals, thereby facilitating more accurate identification of query-relevant frames. Furthermore, this directly highlights the inherent flexibility of our framework: we can effectively decouple the reward mechanism from the inference backbone. By leveraging a separate, reasoning-intensive Image-LMM for frame selection, we can significantly enhance the final performance of the primary Video-LMM.

### <span id="page-9-1"></span>6.4 Impact of Window Length

To investigate how different values of wlen affect performance, we conduct an evaluation using settings of  $wlen \in \{0, 2, 4, 8\}$ , while keeping all other settings identical.

<span id="page-9-2"></span>Comparison with different window length. As shown in Figure 8, setting wlen=0 yields the lowest performance across all benchmarks. This deficit is particularly pronounced on LongVideoBench [55], which necessitates reasoning over extended temporal contexts. This indicates that most queries cannot be effectively resolved within only a single scene, but instead require information from the surrounding temporal context. However, performance does not monotonically improve with wlen. When wlen is set to a high value, such as 8, performance degrades compared to wlen=2 and 4. This proves that an excessively large window introduces irrelevant contextual information, creating noise that is detrimental to localized queries. Therefore, wlen=2 appears to strike the optimal balance, achieving the best results across the benchmarks.

### 6.5 Efficiency of DIG

To evaluate computational cost, we measure and compare the FLOPs of our **DIG** pipeline against uniform sampling on LongVideoBench [55]. The reported FLOPs represent the average computation required per QA pair.

**Performance-Efficiency analysis.** As demonstrated in Figure 9, the uniform sampling approach exhibits a clear performance bottleneck. As the number of input frames scales, its accuracy saturates at a peak of 62.5%. Further increases in computation and frame count do not yield better performance. In contrast, **DIG** successfully overcomes this limitation. While operating at a higher computational budget ( $\geq 680$  TFLOPs), **DIG** demonstrates positive performance scaling, surpassing the uniform sampling's peak accuracy once computation exceeds 720 TFLOPs and continuing to improve thereafter.

<span id="page-10-4"></span>![](_page_10_Figure_5.jpeg)

Figure 9: Comparison between Accuracy and FLOPs. The base LMM is Qwen2.5-VL-7B [16].

### 7 Conclusion

In this work, we find that optimal frame selection in video understanding depends on the query type (global vs. localized). Based on this, we propose **DIG**, a training-free framework that adapts to this typology: it employs efficient uniform sampling for global queries while reserving a multi-stage pipeline for localized queries where targeted selection is essential. This dual approach ensures both high performance and efficiency. Extensive experiments across diverse long-form video benchmarks and LMMs validate that **DIG** consistently outperforms baselines and robustly scales LMM performance for inputs from 8 to 256 frames.

#### References

- <span id="page-10-0"></span>[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. URL https://arxiv.org/abs/2304.08485.
- [2] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. URL https://arxiv.org/abs/2406.16860.
- <span id="page-10-3"></span>[3] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. URL https://arxiv.org/abs/2408.03326.
- [4] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. *arXiv preprint arXiv:2408.15998*, 2024.
- <span id="page-10-1"></span>[5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *CVPR*, pages 24185–24198, 2024.
- <span id="page-10-2"></span>[6] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. *arXiv preprint arXiv:2306.05424*, 2023.
- [7] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). *arXiv preprint arXiv:2310.19773*, 2023.
- [8] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. *arXiv preprint arXiv:2312.17235*, 2023.

- [9] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. *arXiv preprint arXiv:2311.10122*, 2023.
- [10] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In *CVPR*, pages 13700–13710, 2024.
- [11] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14313–14323, 2024.
- [12] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. Videollama 3: Frontier multimodal foundation models for image and video understanding, 2025. URL [https://arxiv.org/](https://arxiv.org/abs/2501.13106) [abs/2501.13106](https://arxiv.org/abs/2501.13106).
- <span id="page-11-7"></span>[13] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs, October 2024.
- <span id="page-11-0"></span>[14] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. URL <https://arxiv.org/abs/2410.02713>.
- <span id="page-11-1"></span>[15] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding, 2023. URL <https://arxiv.org/abs/2306.02858>.
- <span id="page-11-2"></span>[16] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL <https://arxiv.org/abs/2502.13923>.
- <span id="page-11-3"></span>[17] Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem. Bolt: Boost large vision-language model without training for long-form video understanding, 2025. URL <https://arxiv.org/abs/2503.21483>.
- <span id="page-11-9"></span>[18] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding, 2025. URL <https://arxiv.org/abs/2502.21271>.
- <span id="page-11-10"></span>[19] Sicheng Yu, Chengkai Jin, Huanyu Wang, Zhenghao Chen, Sheng Jin, Zhongrong Zuo, Xiaolei Xu, Zhenbang Sun, Bingni Zhang, Jiawei Wu, Hao Zhang, and Qianru Sun. Frame-voyager: Learning to query frames for video large language models, October 2024.
- <span id="page-11-8"></span>[20] Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, and Manling Li. Re-thinking temporal search for long-form video understanding, 2025. URL <https://arxiv.org/abs/2504.02259>.
- <span id="page-11-4"></span>[21] Hui Sun, Shiyin Lu, Huanyu Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Ming Li. Mdp3: A training-free approach for list-wise frame selection in video-llms, 2025. URL [https://arxiv.org/abs/2501.](https://arxiv.org/abs/2501.02885) [02885](https://arxiv.org/abs/2501.02885).
- <span id="page-11-5"></span>[22] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. URL <https://arxiv.org/abs/2304.07193>.
- <span id="page-11-6"></span>[23] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, et al. The llama 3 herd of models, August 2024.
- [24] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, et al. Gpt-4 technical report, March 2024.

- [25] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL <https://arxiv.org/abs/2005.14165>.
- [26] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL [https:](https://arxiv.org/abs/2302.13971) [//arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971).
- [27] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4, 2023. URL <https://arxiv.org/abs/2304.03277>.
- [28] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL <https://arxiv.org/abs/2204.02311>.
- [29] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. URL <https://arxiv.org/abs/2210.11416>.
- <span id="page-12-0"></span>[30] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022. URL <https://arxiv.org/abs/2205.01068>.
- <span id="page-12-1"></span>[31] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, October 2023.
- <span id="page-12-2"></span>[32] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input, 2024. URL <https://arxiv.org/abs/2408.15542>.
- <span id="page-12-3"></span>[33] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning, March 2023.
- [34] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. In *NeurIPS*, volume 37, pages 19472–19495, 2024.
- [35] Hao Wu, Huabin Liu, Yu Qiao, and Xiao Sun. Dibs: Enhancing dense video captioning with unlabeled videos via pseudo boundary enrichment and online refinement. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 18699–18708, June 2024.

- [36] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D. Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark, 2025. URL <https://arxiv.org/abs/2410.03051>.
- <span id="page-13-0"></span>[37] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Videococa: Video-text modeling with zero-shot transfer from contrastive captioners, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2212.04979) [2212.04979](https://arxiv.org/abs/2212.04979).
- <span id="page-13-1"></span>[38] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm. *IEEE Access*, 2024.
- [39] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid. Morevqa: Exploring modular reasoning models for video question answering. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 13235–13245, June 2024.
- [40] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2024. URL <https://arxiv.org/abs/2311.10122>.
- <span id="page-13-2"></span>[41] Yaoyao Zhong, Junbin Xiao, Wei Ji, Yicong Li, Weihong Deng, and Tat-Seng Chua. Video question answering: Datasets, algorithms and challenges, 2022. URL <https://arxiv.org/abs/2203.01225>.
- <span id="page-13-3"></span>[42] Min Shi, Shihao Wang, Chieh-Yun Chen, Jitesh Jain, Kai Wang, Junjun Xiong, Guilin Liu, Zhiding Yu, and Humphrey Shi. Slow-fast architecture for video multi-modal large language models, 2025. URL [https://arxiv.](https://arxiv.org/abs/2504.01328) [org/abs/2504.01328](https://arxiv.org/abs/2504.01328).
- [43] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models, 2024. URL [https:](https://arxiv.org/abs/2407.15841) [//arxiv.org/abs/2407.15841](https://arxiv.org/abs/2407.15841).
- <span id="page-13-5"></span>[44] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, and Xide Xia. Apollo: An exploration of video understanding in large multimodal models, 2024. URL <https://arxiv.org/abs/2412.10360>.
- <span id="page-13-4"></span>[45] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring the design space for multimodal llms with mixture of encoders, 2025. URL <https://arxiv.org/abs/2408.15998>.
- <span id="page-13-6"></span>[46] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding, 2024. URL <https://arxiv.org/abs/2307.16449>.
- <span id="page-13-7"></span>[47] Howard Yen, Tianyu Gao, and Danqi Chen. Long-context language modeling with parallel context encoding, 2024. URL <https://arxiv.org/abs/2402.16617>.
- <span id="page-13-8"></span>[48] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue, Longteng Guo, Bingning Wang, Weipeng Chen, and Jing Liu. Needle in a video haystack: A scalable synthetic evaluator for video mllms, 2025. URL [https:](https://arxiv.org/abs/2406.09367) [//arxiv.org/abs/2406.09367](https://arxiv.org/abs/2406.09367).
- <span id="page-13-9"></span>[49] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. Videochat-flash: Hierarchical compression for long-context video modeling, 2025. URL <https://arxiv.org/abs/2501.00574>.
- <span id="page-13-10"></span>[50] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2311.08046) [2311.08046](https://arxiv.org/abs/2311.08046).
- <span id="page-13-11"></span>[51] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 18221–18232, Seattle, WA, USA, June 2024. IEEE. ISBN 9798350353006.

- <span id="page-14-3"></span>[52] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding, October 2024.
- <span id="page-14-4"></span>[53] Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, and Liqiang Nie. Retake: Reducing temporal and knowledge redundancy for long video understanding, 2025. URL <https://arxiv.org/abs/2412.20504>.
- <span id="page-14-0"></span>[54] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding, 2025. URL <https://arxiv.org/abs/2406.04264>.
- <span id="page-14-1"></span>[55] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. URL <https://arxiv.org/abs/2407.15754>.
- <span id="page-14-2"></span>[56] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2024. URL <https://arxiv.org/abs/2405.21075>.
- <span id="page-14-5"></span>[57] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent, 2024. URL <https://arxiv.org/abs/2403.10517>.
- [58] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos, 2025. URL [https:](https://arxiv.org/abs/2405.19209) [//arxiv.org/abs/2405.19209](https://arxiv.org/abs/2405.19209).
- [59] Zeyuan Yang, Delin Chen, Xueyang Yu, Maohao Shen, and Chuang Gan. Vca: Video curious agent for long video understanding, 2025. URL <https://arxiv.org/abs/2412.10471>.
- <span id="page-14-6"></span>[60] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding, 2024. URL <https://arxiv.org/abs/2403.11481>.
- <span id="page-14-7"></span>[61] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Mingchen Zhuge, Jian Ding, Deyao Zhu, Jürgen Schmidhuber, and Mohamed Elhoseiny. Goldfish: Vision-language understanding of arbitrarily long videos. In *ECCV*, pages 251–267. Springer, 2024.
- <span id="page-14-8"></span>[62] Roy Ganz, Yair Kittenplon, Aviad Aberdam, Elad Ben Avraham, Oren Nuriel, Shai Mazor, and Ron Litman. Question aware vision transformer for multimodal reasoning, 2024. URL <https://arxiv.org/abs/2402.05472>.
- <span id="page-14-9"></span>[63] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. URL <https://arxiv.org/abs/2104.08718>.
- <span id="page-14-10"></span>[64] Shaojie Zhang, Jiahui Yang, Jianqin Yin, Zhenbo Luo, and Jian Luan. Q-frame: Query-aware frame selection and multi-resolution adaptation for video-llms, 2025. URL <https://arxiv.org/abs/2506.22139>.
- <span id="page-14-11"></span>[65] Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, and Manling Li. T\*: Re-thinking temporal search for long-form video understanding, 2025. URL <https://arxiv.org/abs/2504.02259>.
- <span id="page-14-12"></span>[66] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. Longvila: Scaling long-context visual language models for long videos, 2024. URL [https:](https://arxiv.org/abs/2408.10188) [//arxiv.org/abs/2408.10188](https://arxiv.org/abs/2408.10188).
- <span id="page-14-13"></span>[67] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision, 2024. URL [https://arxiv.](https://arxiv.org/abs/2406.16852) [org/abs/2406.16852](https://arxiv.org/abs/2406.16852).

- <span id="page-15-0"></span>[68] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL [https:](https://arxiv.org/abs/2504.10479) [//arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479).
- <span id="page-15-1"></span>[69] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding, 2024. URL <https://arxiv.org/abs/2410.17434>.
- [70] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cg-bench: Clue-grounded question answering benchmark for long video understanding, 2024. URL [https:](https://arxiv.org/abs/2412.12075) [//arxiv.org/abs/2412.12075](https://arxiv.org/abs/2412.12075).
- <span id="page-15-2"></span>[71] Tianyuan Qu, Longxiang Tang, Bohao Peng, Senqiao Yang, Bei Yu, and Jiaya Jia. Does your vision-language model get lost in the long video sampling dilemma?, 2025. URL <https://arxiv.org/abs/2503.12496>.
- <span id="page-15-3"></span>[72] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL <https://arxiv.org/abs/2103.00020>.
- <span id="page-15-4"></span>[73] Qwen Team. Qwen3 technical report, 2025. URL <https://arxiv.org/abs/2505.09388>.
- <span id="page-15-5"></span>[74] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. URL <https://arxiv.org/abs/2407.12772>.
- <span id="page-15-6"></span>[75] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- <span id="page-15-7"></span>[76] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL <https://arxiv.org/abs/2511.21631>.
- <span id="page-15-8"></span>[77] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL [https:](https://arxiv.org/abs/2201.11903) [//arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).
- <span id="page-15-9"></span>[78] LLama3 Team. The llama 3 herd of models, 2024. URL <https://arxiv.org/abs/2407.21783>.
- <span id="page-15-10"></span>[79] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL <https://arxiv.org/abs/2508.10925>.
- <span id="page-15-11"></span>[80] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL <https://arxiv.org/abs/2501.12948>.
- <span id="page-15-12"></span>[81] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation, 2022. URL <https://arxiv.org/abs/2201.12086>.

# **Appendix**

We include additional dataset statistics, annotation protocols, prompt designs, and extended experimental analyses to support reproducibility and offer deeper insight into each component of **DIG**. The appendix is organized as follows:

- Section [A](#page-16-0) provides detailed statistics and descriptions of the benchmarks used in our experiments.
- Section [B](#page-17-1) describes the manual annotation protocol for classifying queries across each benchmark.
- Section [C](#page-17-0) presents instructions used for query identification, reward assignment, and direct inference.
- Section [D](#page-18-0) reports additional analysis on the query identification module, including LLM classification accuracy and its alignment with human annotations.
- Section [E](#page-18-1) elaborates on the CAFS algorithm and provides statistical analysis of its output characteristics.
- Section [F](#page-20-0) contains per-task performance breakdowns on each benchmark, extended experiments on Qwen3-VL-8B, and further discussion of the results.
- Section [G](#page-24-0) presents a detailed runtime profiling of **DIG** and analyzes the efficiency gains from the query identification module.

# <span id="page-16-0"></span>**A Benchmark Details**

This section details the benchmarks used in our evaluation. A statistical overview of each dataset is provided in Table [3.](#page-16-1)

**MLVU.** MLVU [\[54\]](#page-14-0) is a multi-task benchmark for long video understanding, comprising 3,102 questions across 9 categories. The dataset is partitioned into a dev set (2,593 questions) and a test set (509 questions). Tasks are categorized into three primary types: 1) holistic analysis, 2) single-detail identification, and 3) multi-detail reasoning. For our evaluation, we utilize only multiple-choice questions from the dev set and exclude open-ended questions.

**LongVideoBench.** LongVideoBench [\[55\]](#page-14-1) is a question-answering benchmark featuring 3,763 web-collected videos and 6,678 human-annotated, multiple-choice questions spanning 17 fine-grained categories. The benchmark is designed to test referring reasoning by requiring models to retrieve and reason over detailed information. In our study, we utilize only the validation set of this benchmark.

**VideoMME.** VideoMME [\[56\]](#page-14-2) is a multi-modal benchmark covering 30 subdomains across 6 primary visual domains. It contains 900 videos, totaling approximately 254 hours, and 2,700 question-answer pairs. The dataset includes multiple modalities (e.g., video, subtitles, audio) and splits videos by duration (short, medium, long). To focus our evaluation on long-form video understanding, we use only the medium and long duration splits. Furthermore, we leverage only the video data and corresponding questions, excluding all other modalities like subtitles.

<span id="page-16-1"></span>**Table 3:** *Dataset Statistics. Overview of the data statistics across LongVideoBench [\[55\]](#page-14-1), MLVU [\[54\]](#page-14-0) and VideoMME [\[56\]](#page-14-2).*

| Dataset                 | Avg. Duration (s) | #QA Pairs |
|-------------------------|-------------------|-----------|
| MLVU [54]               | 636.2             | 2174      |
| LongVideoBench-val [55] | 732.2             | 1337      |
| VideoMME-short [56]     | 80.7              | 900       |
| VideoMME-medium [56]    | 516.8             | 900       |
| VideoMME-long [56]      | 2466.3            | 900       |

# <span id="page-17-1"></span>**B Query Identification by Human Annotator**

In this section, we elaborate on the query identification process described in Section [3,](#page-2-1) detailing the methodology used to classify queries from each benchmark.

**MLVU.** The task structure of MLVU [\[54\]](#page-14-0) maps directly to our proposed query definitions. Queries associated with its "holistic tasks", which necessitate a comprehensive understanding of the entire video's overarching narrative, themes or a summary of its content, are classified as global queries. Conversely, queries within its "single-detail" and "multi-detail" task categories, which inherently demand that the model focus on specific, discrete temporal segments or isolated events, are classified as localized queries. Applying this classification scheme, we identified 462 global queries and 1708 localized queries within MLVU [\[54\]](#page-14-0).

**LongVideoBench.** The design of LongVideoBench [\[55\]](#page-14-1) is centered on "referring reasoning." This evaluation paradigm is explicitly designed to test a model's capacity to ground its reasoning in specific, fine-grained visual information. By their very nature, such queries require pinpointing information within distinct temporal or spatial segments rather than assessing the video as a whole. Consequently, all queries within this benchmark correspond directly to our definition of localized queries.

**VideoMME.** VideoMME [\[56\]](#page-14-2) lacks an intrinsic task classification that aligns with our global-versus-localized classification. To address this gap, we implemented a rigorous manual annotation process. We established a standardized protocol wherein human annotators were provided with detailed instructions and precise criteria (as illustrated in Figure [11\)](#page-25-0) to distinguish between the two query types. To ensure the reliability of these labels and mitigate subjective bias, the final classification for each query was determined by a majority vote consensus. This annotation procedure resulted in the identification of 479 global queries and 2*,* 221 localized queries.

# <span id="page-17-0"></span>**C Prompt Design**

Prompt engineering is a cornerstone of harnessing the sophisticated reasoning capabilities of LLMs and LMMs. For our **DIG** framework, we designed a series of specialized prompts to guide the models through our multi-stage video question-answering pipeline. This section details the design and rationale for the three core prompts: (1) Query Identification, (2) Reward Assignment, and (3) Direct Inference.

**Query identification.** The initial and most critical step in our framework is to determine the type of the user's query. This classification dictates the subsequent processing strategy. As illustrated in Figure [11,](#page-25-0) the prompt leverages a Chain-of-Thought (CoT) strategy [\[77\]](#page-15-8) to deconstruct the classification task into a series of explicit, verifiable reasoning steps. The model is instructed to first analyze the query's intent, then hypothesize the video's genre (e.g., narrative, instructional), identify specific referents (entities, actions, or concepts), and finally synthesize this information to classify the query as either global or localized. This structured approach ensures a robust and transparent classification.

**Reward assignment.** To generate fine-grained feedback for optimizing our video refinement process, we utilize an LMM to assign relevance scores to sampled frames. The prompt, shown in Figure [12,](#page-26-0) presents the LMM with the user's question, a specific video frame, and associated metadata (video duration and frame timestamp). The model is tasked with a two-part CoT process: first, to provide a qualitative description of the frame's content, focusing on elements pertinent to the query, and second, to assign a quantitative reward score from 0 to 100. The reward criteria are carefully defined to capture not only the frame's direct usefulness but also its contextual value, that is, whether the frame suggests that temporally adjacent segments contain the necessary information.

**Direct inference.** For final evaluation, we use a direct inference prompt, exemplified in Figure [13.](#page-26-1) This prompt is designed for a standard multiple-choice question-answering format. It presents the LMM with the question and a set of candidate options (A, B, C, D). Additionally, the prompt instructs the model to return only the letter corresponding to the best answer.

# <span id="page-18-0"></span>**D More Details about Query Identification**

In this section, we evaluate the capability of contemporary LLMs to distinguish between global and localized queries. We assess the alignment between LLM predictions and human annotations by computing classification accuracy across three benchmarks: MLVU [\[54\]](#page-14-0), LongVideoBench [\[55\]](#page-14-1), and VideoMME [\[56\]](#page-14-2). The ground truth labels for these query types are derived from human annotations, as detailed in Section [B.](#page-17-1)

**LLMs exhibit strong alignment with human annotation.** As presented in Table [4,](#page-18-2) nearly all evaluated LLMs achieve an overall classification accuracy exceeding 80%. This indicates that off-the-shelf LLMs possess sufficiently robust reasoning capabilities to effectively differentiate between localized and global queries without extensive fine-tuning when given a proper prompt.

**Localized queries are more readily identifiable.** Table [4](#page-18-2) further reveals that accuracy on localized queries consistently surpasses that of global queries. While GQ accuracy is comparatively lower, this has a negligible impact on final model performance; it primarily incurs a minor computational overhead. This is because, as established previously, performance differences between query-aware frame selection and uniform sampling are minimal for global queries. In addition, the critical metric is LQ accuracy that may influence the final performance. On this metric, almost all LLMs achieve an accuracy greater than 90%, ensuring the final performance is good. And to make a tradeoff between compute cost and final model performance, we choose to use Qwen3-Next-80B-A3B-Instruct [\[73\]](#page-15-4) in our main experiments.

<span id="page-18-2"></span>**Table 4:** *Accuracy (%) of different LLMs in identifying localized queries (LQ) and global queries (GQ) across multiple benchmarks.*

| LLM                               |       | MLVU [54] |         |       |     | LongVideoBench [55] | VideoMME [56] |       |         |  |
|-----------------------------------|-------|-----------|---------|-------|-----|---------------------|---------------|-------|---------|--|
|                                   | LQ    | GQ        | Overall | LQ    | GQ  | Overall             | LQ            | GQ    | Overall |  |
| Qwen3-Next-80B-A3B-Instruct [73]  | 87.02 | 38.26     | 78.52   | 97.53 | N/A | 97.53               | 89.13         | 65.76 | 83.90   |  |
| Llama-3.1-8B-Instruct [78]        | 93.65 | 24.01     | 81.50   | 98.20 | N/A | 98.20               | 96.99         | 34.24 | 82.95   |  |
| GPT-OSS-20B [79]                  | 82.00 | 74.93     | 80.77   | 93.04 | N/A | 93.04               | 89.20         | 69.97 | 84.90   |  |
| DeepSeek-R1-Distill-Qwen-32B [80] | 93.03 | 26.38     | 81.42   | 99.18 | N/A | 99.18               | 97.21         | 52.85 | 87.28   |  |

# <span id="page-18-1"></span>**E More Details about CAFS**

This section provides a detailed examination of the CAFS method. Section [E.1](#page-18-3) formally specifies the algorithm of CAFS, while Section [E.2](#page-19-0) presents a statistical analysis of its output characteristics based on practical application.

# <span id="page-18-3"></span>**E.1 Detailed Algorithm of CAFS**

Algorithm [1](#page-19-1) details our CAFS method. The process is structured into three sequential stages, taking a frame-to-frame distance sequence *d* = [*d*1*, . . . , dM*−1] and their corresponding original frame indices *I* = [*I*1*, . . . , IM*] as input, to produce a final set of *r-frame* indices, r\_idx.

**Initial peak detection.** First, we identify all potential content boundaries. It iterates through the distance sequence, identifying any point *d<sup>i</sup>* that is a local maximum, defined as being greater than its two immediate neighbors (*di*−<sup>1</sup> *< d<sup>i</sup> < d<sup>i</sup>*+1). The indices *i* of all such local maxima are collected into an initial peaks set.

**Topographic prominence filtering.** Second, we prune the peaks set, retaining only the most significant content transitions. For each peak *j* ∈ peaks, it calculates its "prominence" by finding the lowest base levels to its left (*l*min) and right (*r*min). The prominence is then defined as the peak's height *d<sup>j</sup>* minus the higher of its two bases (prominence = *d<sup>j</sup>* − max(*l*min*, r*min)). This metric quantifies how much a peak "stands out" from the surrounding distance signal. Only peaks whose prominence exceeds a threshold (e.g., 0.1) are added to the filtered\_peaks set, effectively discarding minor, localized fluctuations.

**R-Frame selection.** Finally, we generate the output by identifying frames that best represent the stable content *between* these significant transitions. The algorithm iterates through consecutive pairs of prominent peaks (*p*1*, p*2)

# **Algorithm 1:** Content-Adaptive Frame Selection

```
Input: Distance sequence d = [d1, d2, . . . , dM−1], Frame indices I = [I1, I2, . . . , IM]
  Output: Selected r-frame indices Ridx
1 P ← ∅;
2 for i = 2 to M − 2 do
3 if di−1 < di and di > di+1 then
4 P ← P ∪ {i} ; // A peak is a point higher than its neighbors
5 Pvalid ← ∅;
6 foreach j ∈ P do
7 lmin ← dj ;
8 k ← j − 1;
9 while k ≥ 1 and dk ≤ dj do
10 lmin ← min(lmin, dk);
11 k ← k − 1;
12 rmin ← dj ;
13 m ← j + 1;
14 while m ≤ M − 1 and dm ≤ dj do
15 rmin ← min(rmin, dm);
16 m ← m + 1;
17 pprom ← dj − max(lmin, rmin) ; // Calculate topographic prominence
18 if pprom > 0.1 then
19 Pvalid ← Pvalid ∪ {j};
20 Ridx ← ∅;
21 for i = 1 to |Pvalid| − 1 do
22 p1 ← Pvalid[i];
23 p2 ← Pvalid[i + 1];
24 m ← (Ip1 + Ip2
                  )/2 ; // Midpoints between consecutive prominent peaks
25 Ridx ← Ridx ∪ {m};
26 return Ridx;
```

from the filtered set. For each pair, it calculates the temporal midpoint using their associated original frame indices from *I*: midpoint = (*Ip*<sup>1</sup> + *Ip*<sup>2</sup> )*/*2. These midpoints, which correspond to the center of the most stable segments, are aggregated into the final r\_idx set.

# <span id="page-19-0"></span>**E.2 More Results of CAFS**

To further analyze the performance of CAFS on specific examples, we conduct an evaluation about the relationship between the number of *r-frames* and video duration.

**Non-Linear information scaling in videos.** Figure [10](#page-20-1) reveals that the *r-frame* count does not scale linearly with video duration. This non-linearity is prominent in LongVideoBench [\[55\]](#page-14-1): videos in the 0 − 10 minute bracket average 47*.*9 *r-frames*, whereas those in the 10 − 20 minute bracket average 226*.*4. This finding exposes a fundamental limitation of fixed-rate sampling strategies (e.g., *N* frames/video or *M* frames/sec). Such approaches implicitly assume a uniform information distribution, leading to a suboptimal trade-off: sparse sampling risks information loss, while dense sampling incurs high temporal redundancy. CAFS bypasses this limitation by dynamically adapting its selection to the video's content density.

**High context compression efficiency.** CAFS effectively condenses prolonged video-level context into a sparse, salient set of *r-frames*. For instance, on MLVU [\[54\]](#page-14-0), videos in the 10 − 20 minute bracket (12*.*7 min avg.) are reduced to just 180*.*8 *r-frames* on average. This represents a sparse sampling interval of approximately one *r-frame* every 4*.*22 seconds, demonstrating CAFS's capability to efficiently distill essential information from extended video sequences.

<span id="page-20-1"></span>![](_page_20_Figure_2.jpeg)

**Figure 10:** Correlation between video duration and the number of r-frames selected by the CAFS method across different benchmarks.

# <span id="page-20-0"></span>F More Details about Experiment

### F.1 Detailed Experiment Settings

**Baseline setup.** For AKS [18], we adhered to the default configuration: candidate frames were sampled at 1 fps, and frame-question similarity was computed via BLIP [81]. Based on the algorithm's selection logic, we evaluated frame budgets of  $\{32, 64, 128, 192, 256\}$ . We excluded budgets of 8 and 16 as the algorithm occasionally yielded null returns at these low settings. For Q-Frame [64], we employed the default "fixed frame count" strategy. Since this method limits the initial candidate pool to 128 frames, our evaluation was restricted to budgets of  $\{8, 16, 32, 64, 128\}$ .

### F.2 Extended Experiments with DIG

To investigate the scalability of **DIG** in ultra-long context scenarios, we extended our experiments using Qwen3-VL-8B [76], an open-source LMM distinguished for its robust long-context processing capability. We test **DIG** against the uniform sampling baseline and AKS [18].

**Experiment settings.** For **DIG**, the query identification and CAFS configurations align with Section 5, with the exception that we employ Qwen3-VL-8B [76] as the unified backbone for both reward assignment and final inference. Similarly, AKS [18] setup mirrors Section 5 but utilizes Qwen3-VL-8B [76] as the base model. To rigorously test performance across varying context lengths, we scaled input frame counts from 8 to 768, with each frame encoded into approximately 150 tokens. The results are in Table 5.

**DIG delivers consistent performance gains.** As evidenced in Table 5, **DIG** yields substantial improvements across nearly all frame configurations. Notably, with 256 input frames, DIG achieves an 8.6% performance boost on MLVU [54] compared to uniform sampling. Crucially, DIG maintains robustness even at the extreme scale of 768 frames, surpassing the baseline by 4.7% on MLVU [54], 3.7% on LongVideoBench [55], and 3.5% on VideoMME-Medium [56]. In contrast, while AKS [18] remains competitive at lower frame counts ( $\leq 64$ ), it exhibits marked performance degradation as the context length increases, frequently falling below the uniform sampling baseline. Given that practical video understanding tasks necessitate maximizing input frames to capture comprehensive temporal details, AKS [18] demonstrates limited utility for real-world applications. Conversely, **DIG** exhibits superior scalability, effectively delivering sustained performance gains.

### F.3 Detailed Experiment Results & More Analysis

We present detailed performance breakdowns corresponding to the benchmarks discussed in Section 5. Comprehensive quantitative results are in Tables 6, 7, and 8.

**Uniform sampling suffices for global queries.** For global queries, specifically Anomaly Recognition and Topic Reasoning tasks within MLVU [54], all evaluated methods perform comparably to uniform sampling, regardless of the input frame count. This observation reaffirms our previous assertion: uniform sampling is the preferred strategy for global queries, as it achieves sufficient performance while maintaining high efficiency.

Inference for localized queries operates in two distinct stages: query-aware frame selection and subsequent reasoning based on the retrieved content. Without the initial selection stage, evaluating the model's fundamental performance

<span id="page-21-0"></span>**Table 5:** *Performance comparison between different frame selection methods. Base LMM is Qwen3-VL-8B [\[76\]](#page-15-7). Bold indicates best performance, while Red Box denote results inferior to uniform sampling.*

| Method     | #Frames | MLVU | LVB  | VideoMME |      |  |  |
|------------|---------|------|------|----------|------|--|--|
|            |         |      |      | Medium   | Long |  |  |
| UNI        | 8       | 53.4 | 50.3 | 48.4     | 49.2 |  |  |
| DIG (Ours) | 8       | 58.2 | 54.9 | 53.1     | 49.8 |  |  |
| UNI        | 16      | 53.9 | 50.9 | 51.3     | 48.0 |  |  |
| DIG (Ours) | 16      | 58.9 | 53.9 | 52.9     | 49.9 |  |  |
| UNI        | 32      | 53.7 | 51.0 | 49.4     | 48.6 |  |  |
| AKS [18]   | 32      | 57.3 | 54.4 | 52.3     | 50.1 |  |  |
| DIG (Ours) | 32      | 58.7 | 53.2 | 53.9     | 49.4 |  |  |
| UNI        | 64      | 54.7 | 51.2 | 49.9     | 48.9 |  |  |
| AKS [18]   | 64      | 56.3 | 52.7 | 50.9     | 51.6 |  |  |
| DIG (Ours) | 64      | 59.6 | 54.8 | 54.7     | 49.0 |  |  |
| UNI        | 128     | 57.2 | 54.4 | 55.1     | 51.3 |  |  |
| AKS [18]   | 128     | 58.9 | 53.0 | 54.4     | 51.1 |  |  |
| DIG (Ours) | 128     | 64.4 | 58.3 | 58.4     | 51.1 |  |  |
| UNI        | 192     | 58.9 | 57.1 | 57.6     | 51.6 |  |  |
| AKS [18]   | 192     | 61.1 | 54.5 | 61.0     | 53.8 |  |  |
| DIG (Ours) | 192     | 66.8 | 60.4 | 58.2     | 50.7 |  |  |
| UNI        | 256     | 60.4 | 57.6 | 57.8     | 53.4 |  |  |
| AKS [18]   | 256     | 63.8 | 55.6 | 59.2     | 53.2 |  |  |
| DIG (Ours) | 256     | 69.0 | 61.2 | 61.6     | 53.8 |  |  |
| UNI        | 512     | 65.4 | 60.2 | 61.4     | 55.0 |  |  |
| AKS [18]   | 512     | 65.5 | 57.6 | 60.7     | 56.0 |  |  |
| DIG (Ours) | 512     | 71.7 | 63.8 | 65.6     | 56.4 |  |  |
| UNI        | 768     | 67.5 | 60.9 | 64.3     | 56.6 |  |  |
| AKS [18]   | 768     | 65.3 | 58.3 | 62.3     | 57.3 |  |  |
| DIG (Ours) | 768     | 72.2 | 64.6 | 67.8     | 59.0 |  |  |

is challenging, as errors may stem from information-deficient inputs rather than inherent model limitations. By incorporating this stage to ensure the input contains relevant information, we can decouple data retrieval issues from reasoning capabilities. This allows for a more accurate assessment of the model's intrinsic proficiency across different tasks, yielding deeper insights.

**Query-aware selection uncovers intrinsic visual perception capabilities.** As shown in Table [6](#page-22-0) and [8,](#page-23-0) our method significantly and consistently outperforms uniform sampling on localized perception tasks (e.g., PlotQA, NeedleQA, and L1-Perception). Notably, these tasks primarily evaluate fundamental visual perception capabilities. Our findings suggest that LMMs are intrinsically capable of solving such tasks, provided the query-relevant information is effectively supplied. This explains the substantial performance gap: while uniform sampling often introduces significant noise by including irrelevant content, query-aware selection ensures the model is conditioned on relevant frames.

**Temporal reasoning remains a fundamental bottleneck.** Conversely, regarding tasks requiring temporal logic (e.g., Action Order and L2-Relation), performance remains stagnant across all methods. Even when provided with query-relevant visual information, model performance does not improve. This underscores a critical limitation: current LMMs struggle with temporal reasoning and sequencing, a deficiency that persists independently of the quality of visual information retrieval.

<span id="page-22-0"></span>**Table 6:** Performance comparison between different frame selection methods on MLVU. Base LMMs are Qwen2.5-VL-32B [16] (left) and Qwen2.5-VL-7B [16] (right). Bold indicates best performance. The tasks of MLVU [54] are PlotQA (PQA), NeedleQA (NQA), Action Count (AC), Action Order (AO), Ego Reasoning (ER), Anomaly Recognition (AR), Topic Reasoning (TR).

| Method       | <br>#Frames |      |      | M    | LVU [5 | 4]   |      |      |
|--------------|-------------|------|------|------|--------|------|------|------|
|              |             | PQA  | NQA  | AC   | AO     | ER   | AR   | TR   |
| UNI          | 8           | 55.8 | 58.6 | 18.5 | 51.4   | 50.6 | 66.5 | 85.6 |
| Q-Frame [64] | 8           | 51.4 | 63.9 | 18.4 | 60.2   | 50.3 | 70.5 | 76.8 |
| DIG (Ours)   | 8           | 62.3 | 73.0 | 27.2 | 58.3   | 56.2 | 66.0 | 84.0 |
| UNI          | 16          | 59.4 | 63.9 | 18.0 | 54.8   | 52.8 | 69.5 | 86.3 |
| Q-Frame [64] | 16          | 56.4 | 64.8 | 19.9 | 59.5   | 51.4 | 70.5 | 77.7 |
| DIG (Ours)   | 16          | 67.9 | 78.0 | 35.0 | 66.8   | 57.1 | 69.0 | 86.7 |
| UNI          | 32          | 61.8 | 67.9 | 18.5 | 58.7   | 57.4 | 76.0 | 86.7 |
| AKS [18]     | 32          | 66.8 | 73.0 | 40.3 | 56.0   | 59.9 | 74.5 | 90.1 |
| Q-Frame [64] | 32          | 61.4 | 67.9 | 18.9 | 63.8   | 53.1 | 71.5 | 83.3 |
| DIG (Ours)   | 32          | 72.4 | 79.2 | 48.1 | 75.7   | 59.1 | 74.0 | 87.5 |
| UNI          | 64          | 68.5 | 72.1 | 25.7 | 61.4   | 61.1 | 80.0 | 86.7 |
| AKS [18]     | 64          | 73.8 | 76.6 | 40.8 | 58.3   | 63.1 | 75.0 | 88.2 |
| Q-Frame [64] | 64          | 68.5 | 73.2 | 21.8 | 66.0   | 59.7 | 77.5 | 85.9 |
| DIG (Ours)   | 64          | 75.9 | 81.1 | 49.5 | 78.4   | 66.5 | 78.5 | 89.7 |
| UNI          | 128         | 73.5 | 76.3 | 30.6 | 68.7   | 64.2 | 79.0 | 89.4 |
| AKS [18]     | 128         | 78.3 | 80.3 | 42.2 | 61.8   | 69.0 | 77.0 | 87.8 |
| Q-Frame [64] | 128         | 73.1 | 76.1 | 30.1 | 69.1   | 64.5 | 79.5 | 88.6 |
| DIG (Ours)   | 128         | 79.8 | 80.0 | 52.4 | 79.2   | 65.6 | 78.5 | 89.7 |
| UNI          | 192         | 75.0 | 78.0 | 36.9 | 69.9   | 64.5 | 78.0 | 90.9 |
| AKS [18]     | 192         | 77.6 | 81.4 | 47.1 | 63.3   | 68.2 | 77.0 | 89.4 |
| DIG (Ours)   | 192         | 82.6 | 81.4 | 53.9 | 80.7   | 65.3 | 79.0 | 91.6 |
|              |             |      |      |      |        |      |      |      |

| Model        | #Frames   |      |      | M    | LVU [5 | 4]   |      |      |
|--------------|-----------|------|------|------|--------|------|------|------|
| Wodel        | #ITallies | PQA  | NQA  | AC   | AO     | ER   | AR   | TR   |
| UNI          | 8         | 52.1 | 62.5 | 19.4 | 44.0   | 48.6 | 66.5 | 82.9 |
| Q-Frame [64] | 8         | 50.6 | 67.0 | 19.9 | 48.6   | 49.7 | 68.0 | 73.8 |
| DIG (Ours)   | 8         | 57.1 | 73.2 | 31.6 | 48.6   | 51.1 | 66.5 | 82.9 |
| UNI          | 16        | 56.0 | 63.4 | 19.9 | 42.5   | 54.8 | 70.0 | 84.4 |
| Q-Frame [64] | 16        | 55.5 | 67.6 | 20.4 | 50.6   | 49.7 | 70.5 | 78.7 |
| DIG (Ours)   | 16        | 66.4 | 79.7 | 36.9 | 51.7   | 55.4 | 68.5 | 85.2 |
| UNI          | 32        | 59.7 | 69.0 | 22.8 | 51.4   | 54.0 | 74.5 | 84.4 |
| AKS [18]     | 32        | 69.6 | 76.3 | 42.2 | 50.2   | 56.5 | 72.0 | 85.2 |
| Q-Frame [64] | 32        | 60.1 | 67.9 | 23.9 | 54.1   | 54.3 | 70.0 | 83.7 |
| DIG (Ours)   | 32        | 70.3 | 80.6 | 42.2 | 54.4   | 59.1 | 73.0 | 87.5 |
| UNI          | 64        | 64.0 | 74.4 | 26.2 | 51.7   | 59.9 | 76.0 | 87.1 |
| AKS [18]     | 64        | 69.9 | 80.8 | 41.3 | 54.1   | 60.5 | 69.5 | 84.8 |
| Q-Frame [64] | 64        | 63.8 | 73.5 | 26.2 | 56.0   | 58.2 | 72.0 | 85.9 |
| DIG (Ours)   | 64        | 75.3 | 82.8 | 46.6 | 60.2   | 62.2 | 75.5 | 87.8 |
| UNI          | 128       | 71.4 | 79.2 | 34.5 | 58.7   | 61.4 | 73.0 | 86.7 |
| AKS [18]     | 128       | 71.6 | 83.7 | 48.5 | 57.1   | 60.8 | 72.0 | 84.0 |
| Q-Frame [64] | 128       | 71.4 | 79.2 | 34.5 | 58.7   | 61.4 | 73.0 | 86.7 |
| DIG (Ours)   | 128       | 78.3 | 82.3 | 45.6 | 62.5   | 63.6 | 72.0 | 87.8 |
| UNI          | 192       | 72.0 | 80.8 | 40.3 | 61.4   | 63.6 | 73.0 | 87.5 |
| AKS [18]     | 192       | 74.2 | 83.1 | 46.1 | 58.7   | 63.6 | 73.0 | 85.6 |
| DIG (Ours)   | 192       | 78.7 | 84.5 | 47.1 | 63.3   | 65.3 | 72.0 | 87.5 |
| UNI          | 256       | 73.5 | 80.0 | 41.3 | 61.4   | 61.1 | 73.0 | 89.0 |
| AKS [18]     | 256       | 75.3 | 84.2 | 47.3 | 59.5   | 66.2 | 75.5 | 87.8 |
| DIG (Ours)   | 256       | 78.1 | 84.5 | 49.0 | 62.2   | 65.1 | 73.0 | 89.0 |

<span id="page-22-1"></span>**Table 7:** Performance comparison between different frame selection methods on VideoMME. Base LMMs are Qwen2.5-VL-7B [16](left) and Qwen2.5-VL-32B [16](right). Bold indicates best performance. The tasks are Object Reasoning (ORA), Object Recognition (ORC), Action Reasoning (ARA), Information Synopsis (INS), Counting Problem (COP), Temporal Reasoning (TER), Temporal Perception (TEP), Spatial Perception (SPP), Spatial Reasoning (SPR), OCR, Attribute Perception (ATP), Action Recognition (ACR).

| Model        | #Frames |      |      |      |      | V    | 'ideoM | ME [56 | ]    |      |      |      |      |
|--------------|---------|------|------|------|------|------|--------|--------|------|------|------|------|------|
| Model        |         | ORA  | ORC  | ARA  | INS  | COP  | TER    | TEP    | SPR  | SPP  | OCR  | ATP  | ACR  |
| UNI          | 8       | 49.5 | 54.5 | 49.5 | 67.8 | 36.2 | 40.7   | 47.3   | 58.9 | 63.0 | 48.9 | 62.2 | 53.4 |
| Q-Frame [64] | 8       | 50.5 | 50.1 | 50.8 | 64.8 | 36.9 | 36.0   | 43.3   | 62.1 | 33.3 | 44.0 | 57.0 | 52.2 |
| DIG (Ours)   | 8       | 47.6 | 60.2 | 52.3 | 70.0 | 39.6 | 40.7   | 56.4   | 64.3 | 66.7 | 55.4 | 65.3 | 55.9 |
| UNI          | 16      | 53.5 | 58.8 | 51.2 | 74.6 | 37.7 | 43.5   | 63.6   | 64.3 | 72.2 | 54.7 | 67.1 | 56.9 |
| Q-Frame [64] | 16      | 52.4 | 52.2 | 52.2 | 66.0 | 36.8 | 36.1   | 56.9   | 62.1 | 45.9 | 48.8 | 59.0 | 49.0 |
| DIG (Ours)   | 16      | 57.9 | 61.0 | 54.4 | 74.6 | 41.4 | 43.5   | 56.4   | 66.1 | 75.9 | 59.0 | 71.2 | 56.2 |
| UNI          | 32      | 57.5 | 63.6 | 55.8 | 76.5 | 42.5 | 43.5   | 67.3   | 76.8 | 72.2 | 62.6 | 74.8 | 59.7 |
| AKS [18]     | 32      | 56.8 | 66.7 | 51.9 | 80.2 | 42.9 | 49.2   | 60.0   | 76.8 | 72.2 | 66.2 | 74.8 | 59.7 |
| Q-Frame [64] | 32      | 55.3 | 56.6 | 54.2 | 68.9 | 38.3 | 37.2   | 59.6   | 65.5 | 50.1 | 50.1 | 64.0 | 50.6 |
| DIG (Ours)   | 32      | 59.5 | 67.2 | 56.1 | 75.9 | 40.3 | 46.9   | 52.7   | 76.8 | 72.2 | 69.8 | 73.4 | 61.0 |
| UNI          | 64      | 58.1 | 66.7 | 54.0 | 76.8 | 41.4 | 43.5   | 69.1   | 76.8 | 68.5 | 67.6 | 74.8 | 63.9 |
| AKS [18]     | 64      | 58.4 | 67.5 | 56.1 | 79.3 | 44.4 | 52.0   | 72.7   | 76.8 | 72.2 | 72.7 | 74.8 | 61.0 |
| Q-Frame [64] | 64      | 56.0 | 62.4 | 56.0 | 72.6 | 46.5 | 45.2   | 56.8   | 69.1 | 54.2 | 51.3 | 72.0 | 51.2 |
| DIG (Ours)   | 64      | 60.6 | 68.4 | 57.5 | 78.0 | 46.3 | 49.2   | 58.2   | 73.2 | 75.9 | 73.4 | 73.4 | 63.3 |
| UNI          | 128     | 61.2 | 71.2 | 58.9 | 79.9 | 45.1 | 57.1   | 70.9   | 76.8 | 68.5 | 71.2 | 76.6 | 66.1 |
| AKS [18]     | 128     | 61.0 | 68.4 | 59.3 | 80.2 | 44.8 | 57.1   | 76.4   | 75.0 | 68.5 | 77.0 | 77.9 | 61.3 |
| Q-Frame [64] | 128     | 58.7 | 64.5 | 55.6 | 78.0 | 38.9 | 55.6   | 64.9   | 72.4 | 54.3 | 57.4 | 67.0 | 59.3 |
| DIG (Ours)   | 128     | 61.7 | 72.3 | 57.2 | 80.5 | 45.1 | 52.5   | 58.2   | 78.6 | 68.5 | 71.2 | 78.8 | 66.8 |
| UNI          | 192     | 62.3 | 72.0 | 60.7 | 79.9 | 48.5 | 54.2   | 72.7   | 76.8 | 70.4 | 71.9 | 77.5 | 68.1 |
| AKS [18]     | 192     | 60.8 | 69.2 | 57.5 | 79.3 | 45.5 | 57.6   | 81.8   | 76.8 | 72.2 | 76.3 | 77.9 | 63.9 |
| DIG (Ours)   | 192     | 64.5 | 73.7 | 60.0 | 80.2 | 46.3 | 54.2   | 65.5   | 78.6 | 68.5 | 74.1 | 79.3 | 65.8 |
| UNI          | 256     | 62.1 | 71.5 | 59.6 | 82.4 | 46.6 | 57.6   | 76.4   | 75.0 | 66.7 | 71.9 | 77.9 | 68.1 |
| AKS [18]     | 256     | 61.0 | 70.9 | 57.9 | 79.3 | 44.8 | 57.1   | 83.6   | 75.0 | 70.4 | 74.1 | 77.9 | 64.5 |
| DIG (Ours)   | 256     | 63.0 | 72.0 | 62.1 | 82.4 | 46.3 | 54.8   | 67.3   | 78.6 | 66.7 | 73.4 | 80.2 | 67.4 |

| Model        | #Frames |      |      |      |      | V    | ideoM | ME [56 | ]    |      |      |      |      |
|--------------|---------|------|------|------|------|------|-------|--------|------|------|------|------|------|
|              |         | ORA  | ORC  | ARA  | INS  | COP  | TER   | TEP    | SPR  | SPP  | OCR  | ATP  | ACR  |
| UNI          | 8       | 57.2 | 54.8 | 53.4 | 69.7 | 30.1 | 37.8  | 46.0   | 65.5 | 54.2 | 47.6 | 62.0 | 45.6 |
| Q-Frame [64] | 8       | 50.5 | 50.1 | 50.8 | 64.8 | 36.9 | 36.0  | 43.3   | 62.1 | 33.3 | 44.0 | 57.0 | 52.2 |
| DIG (Ours)   | 8       | 55.6 | 58.1 | 53.4 | 69.7 | 31.5 | 40.9  | 46.0   | 72.4 | 62.5 | 54.9 | 62.0 | 48.4 |
| UNI          | 16      | 55.4 | 56.5 | 52.5 | 70.5 | 32.2 | 47.6  | 51.4   | 75.9 | 62.5 | 42.7 | 65.0 | 50.6 |
| Q-Frame [64] | 16      | 52.4 | 52.2 | 52.2 | 66.0 | 36.8 | 36.1  | 56.9   | 62.1 | 45.9 | 48.8 | 59.0 | 49.0 |
| DIG (Ours)   | 16      | 58.3 | 59.7 | 53.4 | 71.0 | 37.8 | 47.0  | 56.8   | 75.9 | 62.5 | 45.1 | 65.0 | 47.3 |
| UNI          | 32      | 55.4 | 55.9 | 54.2 | 76.8 | 35.0 | 40.9  | 62.2   | 75.9 | 62.5 | 47.6 | 66.0 | 51.7 |
| AKS [18]     | 32      | 58.6 | 57.0 | 57.6 | 78.0 | 34.3 | 47.6  | 56.8   | 79.3 | 45.8 | 61.0 | 68.0 | 51.1 |
| Q-Frame [64] | 32      | 55.3 | 56.6 | 54.2 | 68.9 | 38.3 | 37.2  | 59.6   | 65.5 | 50.1 | 50.1 | 64.0 | 50.6 |
| DIG (Ours)   | 32      | 58.8 | 64.0 | 55.5 | 78.0 | 35.7 | 51.8  | 48.7   | 75.9 | 58.3 | 56.1 | 69.0 | 50.0 |
| UNI          | 64      | 61.8 | 62.9 | 58.0 | 76.8 | 34.3 | 44.5  | 64.9   | 79.3 | 62.5 | 58.5 | 69.0 | 59.3 |
| AKS [18]     | 64      | 61.0 | 63.4 | 59.7 | 80.1 | 37.8 | 54.3  | 62.2   | 79.3 | 54.2 | 59.8 | 73.0 | 56.6 |
| Q-Frame [64] | 64      | 56.0 | 62.4 | 56.0 | 72.6 | 46.5 | 45.2  | 56.8   | 69.1 | 54.2 | 51.3 | 72.0 | 51.2 |
| DIG (Ours)   | 64      | 62.6 | 65.6 | 57.6 | 76.4 | 32.9 | 51.8  | 59.5   | 75.9 | 58.3 | 67.1 | 73.0 | 58.2 |
| UNI          | 128     | 63.4 | 69.9 | 63.5 | 81.3 | 42.0 | 54.3  | 59.5   | 86.2 | 58.3 | 68.3 | 73.0 | 57.1 |
| AKS [18]     | 128     | 66.0 | 68.3 | 58.8 | 81.3 | 42.7 | 57.9  | 67.6   | 82.8 | 50.0 | 69.5 | 75.0 | 59.9 |
| Q-Frame [64] | 128     | 58.7 | 64.5 | 55.6 | 78.0 | 38.9 | 55.6  | 64.9   | 72.4 | 54.3 | 57.4 | 67.0 | 59.3 |
| DIG (Ours)   | 128     | 65.2 | 73.1 | 61.8 | 81.7 | 45.5 | 52.4  | 64.9   | 82.8 | 54.2 | 69.5 | 75.0 | 60.4 |
| UNI          | 192     | 64.7 | 69.9 | 63.0 | 81.3 | 44.8 | 55.5  | 73.0   | 86.2 | 62.5 | 68.3 | 75.0 | 62.1 |
| AKS [18]     | 192     | 65.5 | 69.4 | 59.7 | 81.3 | 42.7 | 58.5  | 75.7   | 79.3 | 54.2 | 73.2 | 78.0 | 58.8 |
| DIG (Ours)   | 192     | 66.8 | 74.2 | 62.6 | 81.7 | 42.0 | 57.3  | 64.9   | 79.3 | 58.3 | 73.2 | 80.0 | 61.0 |

<span id="page-23-0"></span>**Table 8: Performance Comparison between Different Frame Selection Methods on LongVideoBench.** Base LMMs are Qwen2.5-VL-7B [16](top) and Qwen2.5-VL-32B [16](bottom). **Bold** indicates best performance.

|              |         |      |      |      |      |         |      |      |      | LongV | ideoBer | nch [55] |      |      |       |        |      |      |      |      |
|--------------|---------|------|------|------|------|---------|------|------|------|-------|---------|----------|------|------|-------|--------|------|------|------|------|
| Model        | #Frames |      |      |      | L1-  | Percept | tion |      |      |       |         |          |      |      | L2-Re | lation |      |      |      |      |
|              |         | S2E  | S2A  | O2E  | T2O  | S2O     | T2E  | E2O  | T2A  | Avg   | TOS     | ЕЗЕ      | SAA  | ОЗО  | ТЗО   | ТЗЕ    | TAA  | SSS  | SOS  | Avg  |
| UNI          | 8       | 57.0 | 51.1 | 62.8 | 56.6 | 45.8    | 58.5 | 56.9 | 49.4 | 54.4  | 38.4    | 62.8     | 47.2 | 45.5 | 47.3  | 47.9   | 46.3 | 34.0 | 64.2 | 48.3 |
| Q-Frame [64] | 8       | 67.7 | 73.9 | 60.9 | 57.9 | 55.6    | 64.6 | 63.1 | 55.7 | 62.7  | 31.5    | 62.8     | 52.8 | 47.0 | 39.2  | 48.0   | 45.1 | 28.9 | 65.4 | 46.8 |
| DIG (Ours)   | 8       | 69.9 | 68.2 | 62.1 | 50.0 | 55.6    | 61.5 | 61.5 | 62.0 | 61.8  | 37.0    | 61.7     | 50.0 | 48.5 | 54.1  | 43.8   | 45.1 | 29.9 | 67.9 | 48.6 |
| UNI          | 16      | 65.6 | 64.8 | 62.8 | 53.9 | 48.6    | 61.5 | 61.5 | 51.9 | 59.0  | 37.0    | 62.8     | 50.0 | 47.0 | 56.8  | 45.2   | 51.9 | 36.1 | 63.0 | 49.3 |
| Q-Frame [64] | 16      | 66.7 | 70.5 | 65.5 | 63.2 | 61.1    | 64.6 | 69.2 | 63.3 | 65.6  | 34.3    | 59.6     | 56.9 | 54.6 | 43.2  | 49.3   | 50.0 | 38.1 | 65.4 | 50.1 |
| DIG (Ours)   | 16      | 72.0 | 71.6 | 59.8 | 65.8 | 54.2    | 69.2 | 69.2 | 63.3 | 65.4  | 37.0    | 57.4     | 52.8 | 43.9 | 56.8  | 52.1   | 43.9 | 36.1 | 74.1 | 50.4 |
| UNI          | 32      | 67.7 | 58.0 | 61.7 | 56.6 | 62.5    | 67.7 | 67.7 | 51.9 | 61.9  | 37.0    | 61.7     | 55.6 | 56.1 | 55.4  | 49.3   | 52.4 | 40.2 | 66.7 | 52.7 |
| AKS [18]     | 32      | 65.6 | 77.3 | 67.8 | 63.2 | 63.9    | 63.1 | 63.1 | 64.6 | 66.1  | 37.0    | 67.0     | 58.3 | 56.1 | 45.9  | 53.4   | 48.8 | 38.1 | 74.1 | 53.2 |
| Q-Frame [64] | 32      | 64.5 | 69.3 | 60.9 | 59.2 | 61.1    | 61.5 | 69.2 | 60.8 | 63.3  | 32.9    | 59.6     | 62.5 | 50.0 | 50.0  | 45.2   | 46.3 | 39.2 | 66.7 | 50.3 |
| DIG (Ours)   | 32      | 72.0 | 78.4 | 63.2 | 68.4 | 62.5    | 67.7 | 67.7 | 62.0 | 68.0  | 41.1    | 62.8     | 52.8 | 51.5 | 55.4  | 50.7   | 48.8 | 36.1 | 70.4 | 52.1 |
| UNI          | 64      | 73.1 | 67.0 | 62.8 | 59.2 | 56.9    | 63.1 | 66.2 | 62.0 | 64.6  | 34.2    | 62.8     | 58.3 | 59.1 | 60.8  | 47.9   | 51.2 | 43.3 | 67.9 | 53.9 |
| AKS [18]     | 64      | 69.9 | 77.3 | 71.3 | 65.8 | 59.7    | 60.0 | 64.6 | 64.6 | 67.2  | 41.1    | 70.2     | 61.1 | 56.1 | 47.3  | 49.3   | 47.6 | 40.2 | 76.5 | 54.5 |
| Q-Frame [64] | 64      | 63.4 | 70.5 | 63.2 | 57.9 | 61.1    | 63.1 | 64.6 | 65.8 | 63.8  | 34.3    | 67.0     | 58.3 | 53.0 | 52.7  | 46.6   | 51.2 | 37.1 | 66.7 | 52.0 |
| DIG (Ours)   | 64      | 69.9 | 78.4 | 66.7 | 69.7 | 55.6    | 67.7 | 72.3 | 68.4 | 68.8  | 38.4    | 66.0     | 58.3 | 62.1 | 59.5  | 53.4   | 46.3 | 42.3 | 69.1 | 54.9 |
| UNI          | 128     | 71.0 | 67.0 | 67.8 | 64.5 | 61.1    | 67.7 | 72.3 | 73.4 | 68.2  | 38.4    | 68.1     | 56.9 | 59.1 | 56.8  | 50.7   | 54.9 | 46.4 | 74.1 | 56.3 |
| AKS [18]     | 128     | 69.9 | 72.7 | 66.7 | 64.5 | 61.1    | 60.0 | 66.2 | 64.6 | 66.1  | 37.0    | 68.1     | 63.9 | 56.1 | 50.0  | 50.7   | 48.8 | 47.4 | 76.5 | 55.6 |
| Q-Frame [64] | 128     | 66.7 | 67.1 | 69.0 | 60.5 | 59.7    | 64.6 | 67.7 | 64.6 | 65.1  | 38.4    | 64.9     | 58.3 | 54.6 | 56.8  | 49.3   | 51.2 | 45.4 | 75.3 | 55.1 |
| DIG (Ours)   | 128     | 72.0 | 79.5 | 66.7 | 71.1 | 61.1    | 69.2 | 75.4 | 70.9 | 70.9  | 38.4    | 68.1     | 61.1 | 65.2 | 56.8  | 57.5   | 47.6 | 45.4 | 67.9 | 56.3 |
| UNI          | 192     | 74.2 | 72.7 | 66.0 | 68.4 | 59.7    | 67.7 | 70.8 | 63.3 | 68.2  | 35.6    | 66.0     | 59.7 | 59.1 | 58.1  | 58.9   | 56.1 | 46.4 | 67.9 | 56.5 |
| AKS [18]     | 192     | 69.9 | 76.1 | 67.8 | 63.2 | 61.1    | 63.1 | 67.7 | 65.8 | 67.2  | 35.6    | 68.1     | 62.5 | 56.1 | 56.8  | 52.1   | 48.8 | 46.4 | 72.8 | 55.6 |
| DIG (Ours)   | 192     | 74.2 | 84.1 | 66.7 | 67.1 | 59.7    | 69.2 | 73.8 | 78.5 | 72.0  | 35.6    | 68.1     | 61.1 | 66.7 | 56.8  | 61.6   | 48.8 | 49.5 | 70.4 | 57.6 |
| UNI          | 256     | 69.9 | 73.9 | 69.1 | 63.2 | 59.7    | 63.1 | 72.3 | 68.4 | 66.7  | 37.0    | 69.1     | 62.5 | 60.6 | 55.4  | 56.2   | 54.9 | 46.4 | 69.1 | 56.9 |
| AKS [18]     | 256     | 68.8 | 72.7 | 70.1 | 65.8 | 61.1    | 66.2 | 63.1 | 65.8 | 67.0  | 35.6    | 69.1     | 62.5 | 56.1 | 56.8  | 52.1   | 48.8 | 43.3 | 71.6 | 55.2 |
| DIG (Ours)   | 256     | 76.3 | 80.7 | 71.3 | 72.4 | 65.3    | 69.2 | 75.4 | 74.7 | 73.4  | 37.0    | 71.3     | 59.7 | 68.2 | 56.8  | 56.2   | 45.1 | 49.5 | 67.9 | 56.9 |

|              |         |      |      |      |      |         |      |      |      | LongVi | ideoBer | nch [55] | ]    |      |       |         |      |      |      |      |
|--------------|---------|------|------|------|------|---------|------|------|------|--------|---------|----------|------|------|-------|---------|------|------|------|------|
| Model        | #Frames |      |      |      | L1-  | Percept | ion  |      |      |        |         |          |      |      | L2-Re | elation |      |      |      |      |
|              |         | S2E  | S2A  | O2E  | T2O  | S2O     | T2E  | E2O  | T2A  | Avg    | TOS     | E3E      | SAA  | ОЗО  | ТЗО   | T3E     | TAA  | SSS  | SOS  | Avg  |
| UNI          | 8       | 62.4 | 61.4 | 65.5 | 54.0 | 51.4    | 58.5 | 58.5 | 51.9 | 58.2   | 30.1    | 63.8     | 55.6 | 42.4 | 47.3  | 49.3    | 45.1 | 46.4 | 58.0 | 49.2 |
| Q-Frame [64] | 8       | 62.4 | 81.8 | 62.1 | 55.3 | 55.6    | 61.5 | 66.2 | 53.2 | 62.6   | 30.1    | 57.5     | 56.9 | 42.4 | 43.2  | 46.6    | 41.5 | 36.1 | 59.3 | 46.1 |
| DIG (Ours)   | 8       | 63.4 | 75.0 | 60.9 | 59.2 | 48.6    | 60.0 | 66.2 | 60.8 | 61.8   | 37.0    | 60.6     | 61.1 | 50.0 | 52.7  | 49.3    | 47.6 | 44.3 | 65.4 | 52.0 |
| UNI          | 16      | 57.0 | 71.6 | 57.5 | 52.6 | 48.6    | 63.1 | 63.1 | 45.6 | 57.4   | 37.0    | 58.5     | 58.3 | 53.0 | 54.1  | 50.7    | 53.7 | 42.3 | 66.7 | 52.7 |
| Q-Frame [64] | 16      | 69.9 | 77.3 | 64.4 | 64.5 | 58.3    | 64.6 | 63.1 | 62.0 | 65.9   | 31.5    | 62.8     | 58.3 | 51.5 | 46.0  | 43.8    | 46.3 | 40.2 | 54.3 | 49.5 |
| DIG (Ours)   | 16      | 66.7 | 80.7 | 63.2 | 55.3 | 62.5    | 66.2 | 64.6 | 68.4 | 65.9   | 34.3    | 56.4     | 63.9 | 54.5 | 56.8  | 54.8    | 46.3 | 39.2 | 67.9 | 52.7 |
| UNI          | 32      | 68.8 | 68.2 | 64.4 | 57.9 | 54.2    | 63.1 | 61.5 | 53.2 | 61.8   | 32.9    | 66.0     | 56.9 | 53.0 | 52.7  | 58.9    | 50.0 | 47.4 | 70.4 | 54.5 |
| AKS [18]     | 32      | 65.6 | 77.3 | 69.0 | 61.8 | 65.3    | 61.5 | 66.2 | 64.6 | 66.7   | 34.3    | 69.2     | 59.7 | 50.0 | 41.9  | 48.0    | 51.2 | 44.3 | 72.8 | 52.8 |
| Q-Frame [64] | 32      | 69.9 | 77.3 | 63.2 | 59.2 | 65.3    | 61.5 | 60.0 | 54.4 | 61.4   | 37.0    | 59.6     | 54.2 | 48.5 | 48.6  | 48.0    | 51.2 | 50.5 | 60.5 | 51.3 |
| DIG (Ours)   | 32      | 71.0 | 75.0 | 65.5 | 57.9 | 65.3    | 72.3 | 67.7 | 60.8 | 66.9   | 35.6    | 70.2     | 66.7 | 62.1 | 56.8  | 56.2    | 43.9 | 51.5 | 71.6 | 57.2 |
| UNI          | 64      | 69.9 | 65.9 | 64.4 | 61.8 | 58.3    | 60.0 | 70.8 | 57.0 | 63.7   | 31.5    | 67.0     | 58.3 | 51.5 | 55.4  | 54.8    | 51.2 | 51.5 | 69.1 | 54.9 |
| AKS [18]     | 64      | 67.7 | 76.1 | 66.7 | 60.5 | 69.4    | 61.5 | 72.3 | 67.1 | 67.8   | 37.0    | 73.4     | 56.9 | 53.0 | 47.3  | 52.1    | 48.8 | 52.6 | 75.3 | 55.8 |
| Q-Frame [64] | 64      | 63.4 | 77.3 | 66.7 | 56.6 | 66.7    | 64.6 | 67.7 | 65.8 | 64.0   | 35.6    | 63.8     | 61.1 | 47.0 | 46.0  | 52.1    | 52.4 | 53.6 | 67.9 | 53.8 |
| DIG (Ours)   | 64      | 69.9 | 79.5 | 65.5 | 64.5 | 65.3    | 66.2 | 72.3 | 67.1 | 68.8   | 34.3    | 68.1     | 73.6 | 62.1 | 59.5  | 54.8    | 46.3 | 57.7 | 72.8 | 55.8 |
| UNI          | 128     | 71.0 | 70.5 | 62.1 | 61.8 | 68.1    | 63.1 | 67.7 | 60.8 | 65.7   | 35.6    | 70.2     | 62.5 | 51.5 | 55.4  | 57.5    | 53.7 | 60.8 | 71.6 | 58.3 |
| AKS [18]     | 128     | 69.9 | 72.7 | 67.8 | 63.2 | 65.3    | 61.5 | 70.8 | 63.3 | 67.0   | 37.0    | 74.5     | 58.3 | 56.1 | 51.4  | 53.4    | 52.4 | 53.6 | 76.5 | 58.6 |
| Q-Frame [64] | 128     | 65.6 | 69.3 | 60.9 | 59.2 | 65.3    | 61.5 | 67.7 | 59.5 | 63.7   | 35.6    | 71.3     | 61.1 | 48.5 | 51.4  | 53.4    | 52.4 | 60.8 | 70.4 | 56.9 |
| DIG (Ours)   | 128     | 76.3 | 83.0 | 65.5 | 65.8 | 70.8    | 67.7 | 80.0 | 63.3 | 71.6   | 34.3    | 72.3     | 68.1 | 68.2 | 66.2  | 57.5    | 45.1 | 55.7 | 74.1 | 60.2 |
| UNI          | 192     | 72.0 | 73.9 | 66.7 | 67.1 | 66.7    | 67.7 | 72.3 | 63.3 | 68.8   | 34.3    | 76.6     | 58.3 | 62.1 | 58.1  | 60.3    | 52.4 | 56.7 | 71.6 | 59.4 |
| AKS [18]     | 192     | 71.0 | 79.5 | 67.8 | 67.1 | 66.7    | 60.0 | 69.2 | 62.0 | 68.3   | 35.6    | 74.5     | 58.3 | 57.6 | 54.1  | 50.7    | 48.8 | 54.6 | 76.5 | 57.3 |
| DIG (Ours)   | 192     | 73.1 | 84.1 | 67.8 | 68.4 | 68.1    | 64.6 | 76.9 | 65.8 | 71.1   | 38.4    | 74.5     | 70.8 | 69.7 | 71.6  | 57.5    | 43.9 | 56.7 | 75.3 | 60.7 |

# <span id="page-24-0"></span>**G More Efficiency Analysis of DIG**

# **G.1 Detailed Runtime Profiling**

We evaluate the computational efficiency of **DIG** compared to distinct baselines, AKS [\[18\]](#page-11-9) and Q-Frame [\[64\]](#page-14-10). The total runtime of each method can be divided into two stages:

- *Key Frame Selection*, where the method identifies optimal indices from raw video.
- *Inference*, where the LMM processes the selected frames to generate a response.

All experiments were conducted on a node equipped with 8 NVIDIA A100 GPUs. To provide a comprehensive analysis, we report the standard LMM inference latency across varying input frame counts in Table [9](#page-24-1) and detail the selection overhead introduced by specific methods in Table [10.](#page-24-1)

**DIG achieves a favorable efficiency-performance trade-off.** As evidenced in Table [10,](#page-24-1) **DIG** offers a significant efficiency advantage over AKS [\[18\]](#page-11-9), reducing computational overhead by an order of magnitude while maintaining superior downstream performance (see Section [5\)](#page-5-1). While **DIG** incurs a marginal increase in processing time compared to Q-Frame [\[64\]](#page-14-10), this cost is justified by substantial robustness gains; specifically, Q-Frame [\[64\]](#page-14-10) fails to outperform uniform sampling as frame counts exceed 32, whereas **DIG** consistently surpasses baselines across all settings. Furthermore, comparing the selection overhead (Table [10\)](#page-24-1) against standard inference latency (Table [9\)](#page-24-1), the additional cost remains within a reasonable range. This confirms that **DIG** effectively balances efficiency and accuracy, serving as a practical, plug-and-play module for enhanced long-form video understanding.

<span id="page-24-1"></span>**Table 9:** *Inference latency analysis. The inference time (in minutes) of the base LMM (Qwen2.5-VL-7B [\[16\]](#page-11-2)) across different input frame counts using standard uniform sampling.*

| Dataset             | # Frames |     |     |      |      |      |      |
|---------------------|----------|-----|-----|------|------|------|------|
|                     | 8        | 16  | 32  | 64   | 128  | 192  | 256  |
| MLVU [54]           | 3.2      | 5.0 | 9.3 | 17.6 | 29.1 | 37.3 | 43.4 |
| LongVideoBench [55] | 1.4      | 2.2 | 4.3 | 8.3  | 14.0 | 19.9 | 25.6 |
| VideoMME [56]       | 3.1      | 4.7 | 8.7 | 15.8 | 26.1 | 36.7 | 46.3 |

**Table 10:** *Comparison of frame selection overhead. The time cost (in minutes) required by different methods to process videos and select key frames. For DIG, we break down the cost into Query Identification (QI), Content-Aware Frame Selection (CAFS), Reward Assignment (RA), and Video Refinement (VR).*

| Dataset             | AKS [18] | Q-Frame [64] | DIG (Ours) |      |       |     |       |
|---------------------|----------|--------------|------------|------|-------|-----|-------|
|                     |          |              | QI         | CAFS | RA    | VR  | Sum   |
| MLVU [54]           | ≥ 720    | 122.1        | 11.3       | 25.9 | 218.9 | 0.2 | 256.3 |
| LongVideoBench [55] | ≥ 720    | 34.5         | 7.6        | 20.8 | 110.4 | 0.1 | 138.9 |
| VideoMME [56]       | ≥ 720    | 94.2         | 11.6       | 31.2 | 264.8 | 0.3 | 307.9 |

# **G.2 Efficiency Gains from Query Identification**

To balance efficiency and accuracy, **DIG** employs a Query Identification module. We apply resource-intensive key frame selection only to localized queries, defaulting to efficient uniform sampling for global ones. This adaptive strategy minimizes computational cost without compromising downstream performance (see Section [6\)](#page-6-1). Table [11](#page-25-1) quantifies these gains by comparing our adaptive approach against the baseline that applies our specific selection universally. On LongVideoBench [\[55\]](#page-14-1), where queries are predominantly localized, the QI module incurs a marginal overhead (3*.*6%) due to the additional classification step. However, on datasets with a diverse mix of query types, such as VideoMME [\[56\]](#page-14-2) and MLVU [\[54\]](#page-14-0), the adaptive strategy yields significant time savings (19*.*9% and 13*.*3%, respectively). This demonstrates that the QI module effectively optimizes resource allocation by bypassing unnecessary computation for global queries.

<span id="page-25-1"></span>**Table 11:** *Impact of Query Identification on efficiency. We compare the frame selection time (in minutes) of applying our specific selection universally (w/o. QI) versus DIG's adaptive approach (w. QI). "Percent" denotes the proportion of localized queries.*

|                     | Percent | w/o QI | w/ QI                 |
|---------------------|---------|--------|-----------------------|
| MLVU [54]           | 82.8    | 295.7  | (↓<br>256.3<br>13.3%) |
| LongVideoBench [55] | 97.8    | 134.1  | (↑<br>138.9<br>3.6%)  |
| VideoMME [56]       | 77.0    | 384.2  | (↓<br>307.9<br>19.9%) |

#### <span id="page-25-0"></span>Query Identification Prompt

You are a helpful assistant in a video-based question-answering process.

### **Core Task & Definitions**

You will classify the given query into one of two categories:

- 1. **Global Query (isGlobal: true):** The query requires going through and understanding the entire video content.
- 2. **Localized Query (isGlobal: false):** The query that can be fully answered by extracting and analyzing several specific segments within the video.

#### **Instructions for Analysis and Response**

In your analysis, please follow this structured reasoning process to classify the query:

- **Step 1. Understand the Query:** First, read the query to understand its general meaning and core intent.
- **Step 2. Infer Video Style (Hypothetically):** Based on the query's phrasing, make a reasonable inference about the style of the video (e.g., is it a narrative film, an educational lesson, a documentary, etc.)?
- **Step 3. Identify Referents:** Analyze if the query has specific referents. A referent is an entity (person, object), action, event, or even a specific piece of information, depending on the type of video you inferred. For instance, in 'What does Professor Smith write about quantum physics?', the referent is 'Professor Smith' and 'quantum physics' since the video style is likely a lesson.
- **Step 4. Evaluate Referents in Context:** Based on the results from step 3 and the criteria below, determine whether the query is Global or Localized.
  - (i) **The query is Global** if it meets either condition:
    - 1. Lacks a specific referent. The examples include: Summary-based: "primary focus," "in summary," "what is the video about?"
    - 2. Has a referent, but answering still requires a holistic understanding from going through the entire video. The examples include: "what is the boy's overall role?"
  - (ii) **The query is Localized** if it has specific referents, and the answer can be found by focusing on specific, related segments where it appears. Here are some examples:
    - Entity-based: "the person in the red shirt," "the black dog," "Professor Smith," "the little girl."
    - Action/Event-based: "what is [X] doing," "how does [X] build,"
    - Temporal/Sequential: "at the beginning," "after the explosion,"

Please provide your answer in the following format: {"analysis\_step1": str, "analysis\_step2": str, "analysis\_step3": str, "analysis\_step4": str, "isGlobal": true/false}

**User Query**: <Question>

**Figure 11:** *Query Identification Prompt. The LLM is first provided with the task definition, followed by an application of the chain-of-thought [\[77\]](#page-15-8) technique to arrive at a judgment.*

#### <span id="page-26-0"></span>Reward Assignment Prompt

You are a reward model for a video-based question-answering system.

### **Task**

You will receive a question and a sampled video frame. Your task is to evaluate the relevance of this frame for answering the question. Please assign a reward score that indicates how useful or informative the provided frame is in the context of the given question.

#### **Instructions for Analysis and Response**

In your analysis, please perform the following steps to finish your evaluation:

- 1. Describe the visual content of the sampled frame, focusing on elements relevant to the question, if such elements are present.
- 2. Assign a relevance reward between 0 and 100 based on: (1) The sampled frame's direct usefulness in answering the question (2) Whether the frame suggests that adjacent frames might provide additional information that help answer the question more effectively.

Please provide your answer in the following format: {"description": str, "reward": int}.

#### **User Input**

Video Duration: <Duration> seconds; Sampled Frame Timestamp: <Timestamp> seconds; Question: <Question>

**Figure 12:** *Reward Assignment Prompt. The LMM is first presented with the task definition and associated metadata. Then, the chain-of-thought reasoning technique [\[77\]](#page-15-8) is applied to assign the reward for the input frame.*

### <span id="page-26-1"></span>Inference Prompt

Question: What is the video mainly about?

- A. Planes invented by the Wright Brothers.
- B. The structural difference between the planes created by Whitehead and planes created by the Wright Brothers.
- C. Who invented the first plane.
- D. How Whitehead and the Wright Brothers cooperated to invent the first motorized flight.

Please select the best answer from the options provided and directly provide the letter representing your choice without giving any explanation.

**Figure 13:** *Prompt Template Example. Example of the prompt template used by LMMs to perform direct inference.*