# <span id="page-0-0"></span>CurveStream: Boosting Streaming Video Understanding in MLLMs via Curvature-Aware Hierarchical Visual Memory Management

Chao Wang1\*, Xudong Tan1\*, Jianjian Cao<sup>1</sup> , Kangcong Li<sup>1</sup> , and Tao Chen1,2†

*Abstract*—Multimodal Large Language Models have achieved significant success in offline video understanding, yet their application to streaming videos is severely limited by the linear explosion of visual tokens, which often leads to Out-of-Memory (OOM) errors or catastrophic forgetting. Existing visual retention and memory management methods typically rely on uniform sampling, low-level physical metrics, or passive cache eviction. However, these strategies often lack intrinsic semantic awareness, potentially disrupting contextual coherence and blurring transient yet critical semantic transitions. To address these limitations, we propose CurveStream, a training-free, curvatureaware hierarchical visual memory management framework. Our approach is motivated by the key observation that high-curvature regions along continuous feature trajectories closely align with critical global semantic transitions. Based on this geometric insight, CurveStream evaluates real-time semantic intensity via a Curvature Score and integrates an online K-Sigma dynamic threshold to adaptively route frames into clear and fuzzy memory states under a strict token budget. Evaluations across diverse temporal scales confirm that this lightweight framework, CurveStream, consistently yields absolute performance gains of over 10% (e.g., 10.69% on StreamingBench and 13.58% on OVOBench) over respective baselines, establishing new state-ofthe-art results for streaming video perception.The code will be released at https://github.com/streamingvideos/CurveStream.

*Index Terms*—Streaming Video Understanding, Multimodal Large Language Models, Visual Memory Management, Curvature-Aware.

## I. INTRODUCTION

While Multimodal Large Language Models (MLLMs) have achieved remarkable success in offline video understanding [\[1\]](#page-8-0)–[\[6\]](#page-8-1), their application to streaming video scenarios is still hindered by fundamental bottlenecks. Streaming videos are theoretically infinite in length, inevitably leading to a linear explosion of visual tokens. Under stringent GPU memory constraints, models are highly susceptible to Out-of-Memory (OOM) errors or suffer from catastrophic forgetting caused by naive truncation strategies [\[7\]](#page-8-2). Consequently, continuously and dynamically managing visual memory within a fixed memory budget emerges as the core challenge in achieving long-term streaming video understanding.

To address the challenge of linear token explosion, existing methods primarily focus on two aspects: visual information retention and long-term memory management. Visual information retention strategies typically utilize uniform sampling [\[8\]](#page-8-3)– [\[10\]](#page-8-4) or low-level difference metrics (including inter-frame similarity [\[11\]](#page-8-5), [\[12\]](#page-8-6) or optical flow [\[13\]](#page-8-7)). However, these approaches are often sensitive to local noise and prioritize low-level physical motion, making it difficult to robustly capture the high-level global semantic transitions required for multimodal reasoning. Building upon these retained visual features, long-term memory management mechanisms further process the context. Mainstream solutions predominantly include rule-based cache eviction [\[7\]](#page-8-2), [\[14\]](#page-8-8)–[\[16\]](#page-8-9), feature clustering and merging, and retrieval paradigms utilizing external storage [\[17\]](#page-8-10).

1

Despite their progress, these visual retention and memory management methods share common limitations that hinder efficient streaming video understanding: 1) Semantic Fragmentation: They mostly employ passive eviction or smoothing compression strategies lacking intrinsic semantic awareness, which disrupts contextual coherence. 2) Information Blurring: During indiscriminate feature compression, they irreversibly blur transient yet critical semantic transition points. 3) Delayed Perception: Retrieval mechanisms conditioned on post-hoc queries restrict the model's capability for real-time, proactive perception in unbounded streaming scenarios.

To overcome these limitations, we re-examine the evolutionary dynamics of video streams within the feature space. We observe a critical phenomenon: when mapping a continuous video stream into a trajectory within the feature space, the high-curvature regions along this trajectory precisely correspond to high-quality visual semantic transitions. Unlike uniform sampling or physical motion metrics that treat frames equally or focus on local noise, curvature geometrically measures the intensity of semantic shifts. A sharp turn (high curvature) in the feature trajectory signifies the emergence of a new event, a sudden viewpoint change, or a critical action boundary. This implies that utilizing "curvature" as an evaluation metric enables the precise extraction of the most valuable contextual information for reasoning, thereby offering a novel perspective for constructing highly efficient, adaptive streaming video memory management systems. As illustrated in Fig. [1](#page-1-0) (b), this geometric approach effectively identifies critical semantic transitions by monitoring the trajectory's curvature peaks.

Building upon this curvature observation, we propose Cur-

<sup>\*</sup>Contributed equally to this work.

<sup>†</sup>Corresponding author.

<sup>1</sup>Chao Wang, Xudong Tan, Jianjian Cao, Kangcong Li are with the College of Future Information Technology, Fudan University, Shanghai, China (e-mail: chaowang25@m.fudan.edu.cn).

<sup>1,2</sup>Tao Chen is with the College of Future Information Technology, Fudan University, Shanghai, China, and also with Shanghai Innovation Institute, Shanghai, China (e-mail: eetchen@fudan.edu.cn).

<span id="page-1-1"></span>![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>Fig. 1. Performance and mechanism of CurveStream. (a) CurveStream achieves state-of-the-art on OVOBench among training-free paradigms, boosting performance by 13.6% over the Qwen2.5-VL-7B baseline. (b) Curvature-aware memory management over infinite streams (t → ∞). By evaluating real-time semantic intensity (blue curve) against a K-Sigma dynamic threshold (pink dashed line), it adaptively filters redundant Low-Semantic Frames. Critical High-Semantic Frames (yellow dots) at curvature peaks are preserved, ensuring optimal visual context retention under strict token limits.

veStream, a training-free, curvature-aware hierarchical visual memory management framework. Diverging from uniform sampling strategies that periodically drop frames, we formulate streaming video processing as a dynamic, semantic-aware memory update process under a fixed token capacity limit (N). Specifically, CurveStream first calculates a Curvature Score in real time to represent the intensity of semantic transitions, integrating motion variation of consecutive frames with the geometric angle between feature displacement vectors. To achieve adaptive memory management in non-stationary video streams, we introduce an online-updating K-Sigma rule (g = µ + kσ). This mechanism dynamically generates an admission threshold based on the running mean and variance of the historical curvature, adaptively categorizing high-value visual tokens into distinct hierarchical states (Clear Memory and Fuzzy Memory). When the memory bank reaches its capacity limit, the system systematically evicts the oldest tokens following strict queue rules. This design ensures that models maintain an acute perception of core visual semantic trajectories under a constant memory footprint.

To comprehensively evaluate CurveStream, we conduct extensive experiments across diverse temporal scales, encompassing 10 Real-Time Visual Understanding tasks in StreamingBench [\[18\]](#page-8-11), 6 Real-Time Visual Perception tasks in OVOBench [\[19\]](#page-8-12), and 3 offline video datasets (15–1200s) [\[20\]](#page-8-13)– [\[22\]](#page-8-14). As a lightweight, model-agnostic module, CurveStream demonstrates broad architectural compatibility across the LLaVA-OneVision and Qwen-VL (2/2.5/3) series at 4B, 7B, 8B, and 32B parameter scales. As shown in Fig. [1a](#page-1-0), integrating our framework into the Qwen2.5-VL-7B baseline yields accuracies of 84.00% and 73.48% on StreamingBench and OVOBench, respectively, delivering absolute performance gains of 10.69% and 13.58%. Furthermore, CurveStream enables 7B-parameter open-source models to consistently surpass closed-source commercial systems, including GPT-4o and Gemini 1.5 Pro, validating its robust generalizability and practical efficacy.

In summary, the main contributions of this paper are as follows:

- 1) Revealing the "curvature" effect in streaming videos. We discover that high-curvature regions in the latent feature space align with critical global semantic transitions, providing a geometric metric for evaluating visual information that overcomes local noise.
- 2) Proposing CurveStream, a training-free hierarchical memory management framework. By integrating realtime curvature scoring with a dynamic K-Sigma threshold, it adaptively routes frames into clear and fuzzy memory states to handle non-stationary streams under fixed token budgets.
- 3) Achieving state-of-the-art performance on streaming benchmarks CurveStream effectively mitigates OOM issues and consistently improves diverse MLLMs by approximately 10% in streaming scenarios, showing broad

<span id="page-2-0"></span>applicability on benchmarks like StreamingBench and OVOBench.

#### II. RELATED WORK

### *A. Existing Visual Information Retention Strategies*

Existing strategies for visual information retention in long videos encompass various directions, with prominent approaches focusing on rule-based token compression and querydriven feature retrieval [\[4\]](#page-8-15), [\[23\]](#page-8-16)–[\[25\]](#page-8-17). Rule-based methods mitigate redundancy by evaluating local feature similarities. AKS [\[26\]](#page-8-18) and M-LLM [\[27\]](#page-8-19) employ adaptive keyframe selection algorithms to maximize video coverage. FLoC [\[28\]](#page-9-0), FlexSelect [\[29\]](#page-9-1), and METok [\[30\]](#page-9-2) dynamically prune redundant tokens during inference utilizing attention weights or facility location functions. Query-driven approaches perform goaloriented extraction by fetching relevant frames conditioned on user instructions. DIG [\[31\]](#page-9-3), APVR [\[32\]](#page-9-4), BOLT [\[33\]](#page-9-5), and MemVid [\[34\]](#page-9-6) compute semantic similarities between post-hoc text queries and visual frames. These paradigms generally rely on delayed user queries or low-level physical metrics (including inter-frame cosine similarity). This makes them susceptible to local motion noise in dynamic scenes and limits their capacity for proactive perception. To address this, our method diverges from traditional metrics by leveraging the "curvature" of feature trajectories in the feature space. This perspective intrinsically captures global semantic transitions, ensuring robust retention that is resilient to local physical disturbances.

## *B. Existing Streaming Video Memory Management Mechanisms*

Processing theoretically infinite streaming videos inherently causes a linear explosion in memory footprint. To circumvent this, current mechanisms explore various solutions, with KV cache eviction and external structured memory being widely adopted [\[17\]](#page-8-10), [\[35\]](#page-9-7), [\[36\]](#page-9-8). KV cache eviction strategies passively discard historical tokens. InfiniPot-V [\[15\]](#page-8-20), Streaming-TOM [\[37\]](#page-9-9), StreamingVLM [\[7\]](#page-8-2), and HERMES [\[14\]](#page-8-8) utilize sliding windows or spatio-temporal redundancy metrics to evict older tokens upon reaching a memory threshold. External memory approaches offload long-term context to expand capacity. StreamForest [\[38\]](#page-9-10), ReKV [\[17\]](#page-8-10), VideoLucy [\[39\]](#page-9-11), and Venus [\[40\]](#page-9-12) organize video segments into hierarchical trees or move features to external storage, utilizing retrieval mechanisms to reactivate necessary context. However, these mechanisms treat memory management as a queue-based smoothing process or an isolated retrieval task. Consequently, they may blur transient semantic shifts and disrupt natural in-context coherence. In contrast, we formulate memory management as a dynamic, semantic-aware, in-context update process. CurveStream incorporates an online K-Sigma rule to actively evaluate historical curvature, adaptively categorizing and replacing clear and fuzzy memory within a strict token limit.

## III. METHODS

To achieve precise understanding of infinitely long streaming videos under strict memory constraints, we propose CurveStream, a training-free vision encoder architecture (illustrated in Fig. [2\)](#page-3-0). The framework operates as an online selective-retention pipeline: it first utilizes a Curvature-Aware Scorer (CAS) to extract semantic transition intensity from the latent feature manifold trajectory, which is then processed by a Hierarchical Visual Memory Management (HVMM) module. Guided by temporally adaptive thresholds derived from online manifold statistics, this mechanism dynamically routes incoming frames into a fixed-capacity memory bank, categorizing them as Clear, Blurred, or Discarded.

## *A. Problem Formulation*

Let V = {It}<sup>∞</sup> <sup>t</sup>=1 be an infinitely long, continuous video stream, where I<sup>t</sup> denotes the visual observation at time step t. Suppose the system receives a natural language query Q regarding the current or historical states at timestamp tq. Due to the large parameter size Θ of Multimodal Large Language Models (MLLMs) and the quadratic complexity of self-attention mechanisms, it is computationally intractable to directly feed the entire historical sequence V<sup>≤</sup>t<sup>q</sup> into the model. Therefore, the system must maintain a dynamic visual memory queue M<sup>t</sup> restricted by a maximum capacity limit Nmax.

We frame the streaming video understanding task as an online information extraction problem within a constrained space. At each time step t, the system needs to derive an efficient memory scheduling policy π. This policy evaluates the informative value of the current frame I<sup>t</sup> and outputs a state tuple (st, rt) containing the retention and resolution decisions to update the memory bank:

$$\mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1}, I_t, s_t, r_t) \tag{1}$$

where s<sup>t</sup> ∈ {Clear,Blurred, Discard} represents the hierarchical routing state, and r<sup>t</sup> denotes the corresponding spatial resolution.

The primary optimization objective of CurveStream is to maximize the conditional probability of the MLLM generating the correct answer A under a strict queue length constraint (|Mt| ≤ Nmax):

$$\max_{\sigma} P(A \mid Q, \mathcal{M}_{t_q}; \Theta) \tag{2}$$

To solve this online decision-making problem lacking direct supervisory signals, we leverage the intrinsic geometric properties of the visual feature manifold to construct a lightweight scheduling policy π, realized through the CAS and HVMM modules described below.

#### *B. Curvature-Aware Scorer (CAS)*

In continuous visual streams, adjacent frames often exhibit high temporal redundancy. Especially in embodied AI or firstperson perspectives, traditional sampling strategies based on simple feature differences are highly prone to overfitting to

![](_page_3_Figure_1.jpeg)

<span id="page-3-0"></span>Fig. 2. Overview of the CurveStream framework. This training-free vision encoder enables infinite streaming video understanding by replacing traditional sampling with a dynamic-retention perception layer designed to prevent Out-of-Memory (OOM) errors in long-term sequences. The Curvature-Aware Scorer (CAS) evaluates semantic transition intensity by fusing first-order motion variation and second-order trajectory curvature within the latent feature manifold, while the Hierarchical Visual Memory Management (HVMM) module dynamically routes incoming tokens into a fixed-capacity ( $N_{\rm max}$ ) queue. By utilizing temporally adaptive K-Sigma thresholds, the encoder adaptively categorizes visual information into Clear, Blurred, or Discard states based on the intensity of semantic shifts, thereby ensuring a constant memory footprint while preserving critical visual anchors for long-term multimodal reasoning.

large translational motions. To accurately localize high-value information, we design the Curvature-Aware Scorer (CAS).

CAS utilizes a frozen visual encoder to extract the global feature representation  $\mathbf{F}_t \in \mathbb{R}^D$  of the input frame  $I_t$ , followed by  $L_2$  normalization. To characterize the evolutionary trajectory of features within the latent space manifold, we integrate both the first-order motion intensity and the second-order geometric curvature. Based on the cosine similarity between consecutive frames, the first-order Motion Variation is defined as:

$$M_t = 1 - \frac{\langle \mathbf{F}_t, \mathbf{F}_{t-1} \rangle}{\|\mathbf{F}_t\| \|\mathbf{F}_{t-1}\|}$$
(3)

To filter out constant-velocity background changes caused by smooth camera movements, we compute an approximation of the second-order partial derivative of the feature trajectory. Let the feature displacement vectors of adjacent time steps be  $\mathbf{d}_1 = \mathbf{F}_{t-1} - \mathbf{F}_{t-2}$  and  $\mathbf{d}_2 = \mathbf{F}_t - \mathbf{F}_{t-1}$ . The local Geometric Curvature of the feature manifold is approximately represented by the angular deviation between these displacement vectors:

$$C_t = 1 - \frac{\langle \mathbf{d}_1, \mathbf{d}_2 \rangle}{\|\mathbf{d}_1\| \|\mathbf{d}_2\|} \tag{4}$$

When  $d_1$  and  $d_2$  are aligned in direction,  $C_t$  approaches 0, indicating a smooth transition period. Conversely, when the

direction of feature evolution changes abruptly (e.g., a new entity intrudes or a sharp viewpoint shift occurs),  $C_t$  increases significantly. The final Curvature Score  $CS_t$  is formulated as a linear combination of the two:

$$CS_t = M_t + \lambda C_t \tag{5}$$

where  $\lambda$  serves as the balancing coefficient for the geometric penalty term.

#### C. Hierarchical Visual Memory Management (HVMM)

After obtaining the  $CS_t$  sequence, the Hierarchical Visual Memory Management (HVMM) module utilizes temporally adaptive dynamic thresholds to route high-value frames into a fixed-capacity memory bank at differentiated resolution levels, effectively suppressing KV Cache bloat.

1) Online Manifold Distribution Estimation: In untrimmed embodied or first-person streaming videos, the temporal pacing typically exhibits significant dynamics. For instance, a subject might suddenly break into a vigorous run after a prolonged period of stationary observation. Under such complex scenarios, employing any a priori static threshold is highly likely to lead to memory bank collapse or severe loss of critical information.

Therefore, HVMM models the filtering of high-value information as an online distribution-aware process. To capture

<span id="page-4-0"></span>the dynamic pacing of the video stream in real time, we update the transient expectation µ<sup>t</sup> and variance σ 2 <sup>t</sup> of the curvature scores using an Exponential Moving Average (EMA) formulation:

$$\mu_t = \gamma \mu_{t-1} + (1 - \gamma) CS_t, \quad \sigma_t^2 = \gamma \sigma_{t-1}^2 + (1 - \gamma) (CS_t - \mu_t)^2$$
(6)

where γ ∈ (0, 1) is the momentum factor controlling the size of the historical observation window. As the time step t advances, the newly observed curvature score CS<sup>t</sup> smoothly calibrates the transient distribution parameters in a recursive manner. Based on this online evolutionary mechanism, we construct Gaussian distribution-aware dynamic dual thresholds: g<sup>1</sup> = µ<sup>t</sup> + k1σ<sup>t</sup> and g<sup>2</sup> = µ<sup>t</sup> + k2σ<sup>t</sup> (k<sup>1</sup> < k2). This design enables CurveStream to adaptively scale its sensitivity to visual shifts according to the current intensity of the scene.

*2) Hierarchical State Transition:* Guided by the adaptive dual thresholds, HVMM executes a resolution-aware hierarchical state transition strategy. Specifically, the retention state s<sup>t</sup> for an incoming frame I<sup>t</sup> is dynamically determined as follows:

$$s_t = \begin{cases} \text{Clear Memory,} & \text{if } CS_t \ge g_2\\ \text{Blurred Memory,} & \text{if } g_1 \le CS_t < g_2\\ \text{Discard,} & \text{if } CS_t < g_1 \end{cases} \tag{7}$$

Clear Memory. Frames satisfying CS<sup>t</sup> ≥ g<sup>2</sup> break through the current local dynamic distribution and capture significant semantic shifts. The system retains their original highresolution features (r<sup>t</sup> = High) and stores them in the memory bank to support subsequent fine-grained visual reasoning. Notably, the current frame It<sup>q</sup> that triggers the query is deterministically assigned this state to ensure immediate context awareness.

Blurred Memory. Frames falling within g<sup>1</sup> ≤ CS<sup>t</sup> < g<sup>2</sup> are identified as intermediate transitional observations consistent with the current dynamic pacing. To preserve necessary temporal causal associations and action coherence while significantly compressing token overhead, these frames are downsampled to a minimal resolution (r<sup>t</sup> = Low) before storage.

Discard. Frames with CS<sup>t</sup> < g<sup>1</sup> represent low-information redundant observations below the local expected mean. The system directly discards these features to protect the scarce memory space.

Finally, to ensure a constant memory footprint without OOM risks, whenever the memory bank exceeds its capacity (|Mt| > Nmax), the system executes a strict First-In-First-Out (FIFO) eviction, removing the oldest tokens from the queue regardless of their retention states.

## IV. EXPERIMENTS

#### *A. Experimental Setup*

*1) Datasets.:* To comprehensively evaluate the effectiveness of the proposed adaptive visual memory framework under various temporal dynamics, we conducted extensive experiments across five mainstream multimodal benchmarks encompassing three video paradigms. As the core of our evaluation for streaming video understanding, we selected StreamingBench [\[18\]](#page-8-11) and OVOBench [\[19\]](#page-8-12). These two benchmarks rigorously test the model's capability for long-range event association and instantaneous dynamic response within continuous data streams. To address complex dynamic scenes, we utilized EgoSchema [\[21\]](#page-8-21), a highly challenging egocentric benchmark that rigorously tests the model's ability to accurately capture micro-actions and perform causal reasoning amidst drastic viewpoint changes and redundant backgrounds. Furthermore, to explore the extreme limits of memory capacity, we introduced VideoMME [\[22\]](#page-8-14), comprehensively examining the model's feature retention and generalizability across short, medium, and extremely long (up to several hours) contexts. Finally, we incorporated the MVBench [\[20\]](#page-8-13) short video benchmark to verify that the system's dynamic frame filtering and resolution reduction strategies do not compromise the model's spatio-temporal perception of fine-grained local actions.

*2) Baselines.:* Our comparative analysis involves two major categories of baseline methods. The first category comprises state-of-the-art open-source Multimodal Large Language Models (Base MLLMs), specifically including LLaVA-OneVision [\[23\]](#page-8-16) and multiple iterations of the Qwen-VL series (i.e., Qwen2-VL [\[2\]](#page-8-22), Qwen2.5-VL [\[3\]](#page-8-23), Qwen3-VL [\[1\]](#page-8-0)). The second category encompasses recent advanced frameworks specifically optimized for streaming video understanding or long-context visual processing (SOTA Streaming Methods), including Flash-VStream [\[36\]](#page-9-8), FreshMem [\[11\]](#page-8-5), HER-MES [\[14\]](#page-8-8), and ReKV [\[17\]](#page-8-10). By integrating our proposed training-free memory module into the base MLLMs, we conduct a direct performance comparison with these specialized SOTA methods under strictly equivalent visual token constraints.

*3) Implementation Details.:* In all comparative experiments, to ensure evaluation fairness and strictly simulate the physical GPU memory constraints inherent in streaming video processing, we establish a uniform memory bank capacity upper limit (i.e., a maximum token budget N) across all methods. At the feature extraction frontend of our framework, we employ the lightweight DINOv2-small model to acquire local geometric representations of temporal features. During the adaptive memory allocation phase, for high-curvature core transition frames that trigger clear memory, the system retains the native dynamic high-resolution input of the base model. Conversely, for blurred memory frames representing smooth transition states, the resolution is uniformly downsampled to a fixed 224 × 224 to conserve memory space. All benchmark evaluations are independently executed on a single inference GPU to fully validate the robustness of our framework under severely limited memory conditions.

## *B. Online Benchmark Results*

Table [I](#page-5-0) presents the quantitative evaluation results of various methods on the streaming video benchmarks. Under strict visual token capacity constraints, our method achieves stable and significant performance leaps across different base models. Specifically, when utilizing Qwen2-VL-7B as the base model, our method achieves accuracies of 81.04% and 70.73% on

TABLE I

<span id="page-5-1"></span><span id="page-5-0"></span>QUANTITATIVE COMPARISON OF AVERAGE ACCURACY ACROSS 10 REAL-TIME VISUAL UNDERSTANDING SUB-TASKS IN STREAMINGBENCH AND 6 REAL-TIME VISUAL PERCEPTION SUB-TASKS IN OVOBENCH. BEST RESULTS ARE HIGHLIGHTED IN BOLD, WITH ABSOLUTE GAINS OVER RESPECTIVE BASELINES IN RED. OUR FRAME COUNT (10-20) REFLECTS THE DYNAMICALLY CHANGING SIZE OF THE ADAPTIVE MEMORY QUEUE.

| Method                                  | Frame  | StreamingBench [18] | OVOBench [19]   |
|-----------------------------------------|--------|---------------------|-----------------|
| Human                                   | -      | 91.46               | 93.20           |
| Proprietary MLLMs                       |        |                     |                 |
| Gemini 1.5 Pro [41]                     | 1fps   | 75.69               | 69.32           |
| GPT-4o [42]                             | 64     | 73.28               | 64.46           |
| Open-source Offline MLLMs               |        |                     |                 |
| Qwen2-VL-7B [2]                         | 64     | 69.04               | 60.65           |
| InternVL-V2-8B [43]                     | 16     | 63.72               | 60.73           |
| Open-Source Online MLLMs                |        |                     |                 |
| Flash-VStream-7B [36]                   | -      | 23.23               | 29.86           |
| VideoLLM-online-8B [44]                 | 2fps   | 35.99               | 20.79           |
| Dispider-7B [45]                        | 1fps   | 67.63               | 54.55           |
| TimeChat-Online-7B [46]                 | 1fps   | 75.36               | 61.90           |
| StreamForest-7B [38]                    | 1fps   | 77.26               | 61.20           |
| Training-free Offline-to-Online Methods |        |                     |                 |
| LLaVA-OneVision-7B [23]                 | 64     | 71.34               | 63.06           |
| + ReKV [17]                             | 0.5fps | 69.22               | 57.33           |
| + HERMES [14]                           | 1fps   | 73.23               | 66.34           |
| + Ours                                  | 10-20  | 75.12 (↑ 3.78)      | 70.57 (↑ 7.51)  |
| Qwen2-VL-7B [2]                         | 1fps   | 69.04               | 60.65           |
| + HERMES [14]                           | 1fps   | -                   | -               |
| + Freshmem [11]                         | 1fps   | 74.20               | 66.67           |
| + Ours                                  | 10-20  | 81.04 (↑ 12.00)     | 70.73 (↑ 10.08) |
| Qwen2.5-VL-7B [3]                       | 1fps   | 73.31               | 59.90           |
| + HERMES [14]                           | 1fps   | 79.44               | 68.98           |
| + Ours                                  | 10-20  | 84.00 (↑ 10.69)     | 73.48 (↑ 13.58) |
| Qwen3-VL-8B [1]                         | 1fps   | 73.20               | 70.1            |
| + Ours                                  | 10-20  | 85.56 (↑ 12.36)     | 80.76 (↑ 10.66) |

StreamingBench and OVO-Bench, respectively, yielding absolute performance gains of 12.0% and 10.08% compared to the uniform sampling baseline.

More importantly, among training-free streaming video understanding frameworks, our method establishes a new stateof-the-art (SOTA). Compared to recent advanced specialized streaming video methods (*e.g.*, FreshMem and HERMES), our framework further achieves absolute accuracy improvements of 6.84% and 4.06% on StreamingBench and OVO-Bench, respectively.

This comprehensively leading performance is directly attributed to our adaptive visual memory mechanism. By introducing manifold curvature as a dynamic prior, the framework not only effectively strips away redundant static backgrounds in long videos but also precisely allocates limited memory resources to high-frequency visual transition points. This strategy, which highly aligns the memory queue with the underlying dynamic evolution of the video, fundamentally overcomes catastrophic forgetting in long-range reasoning, thereby preserving the highest-quality temporal context for the model.

#### *C. Offline Benchmark Results*

Table [II](#page-6-0) presents the evaluation results of our framework on the short-video benchmark (MVBench) and long-video benchmark (VideoMME). Although our adaptive memory framework is specifically designed for streaming video scenarios, it also exhibits strong generalization ability in conventional offline short- and long-video understanding tasks.

As can be observed, our method consistently brings stable performance improvements across different base models. For instance, when built upon Qwen2.5-VL-7B, our method achieves a 1.03% absolute gain (up to 66.03%) on MVBench, a fine-grained action-oriented short-video benchmark, compared with the uniform sampling baseline. Meanwhile, when integrated into LLaVA-OneVision-7B, our method also yields a 1.77% absolute improvement (up to 59.44%) on VideoMME, a comprehensive long-video benchmark.

It is worth noting that for Qwen2.5-VL-7B on VideoMME, there is a slight performance drop (from 64.52% to 62.97%). This is because, to maintain a strictly constant memory footprint without OOM risks over hours-long videos, the system inevitably trades off some fine-grained global details to preserve the most critical semantic transitions. These quantitative results sufficiently verify the universality and effectiveness of the proposed framework in offline settings.

#### *D. Scalability Across Model Parameters*

Fig. [3a](#page-7-0) presents the evaluation results of our framework across the Qwen3-VL series with different parameter scales.

<span id="page-6-3"></span><span id="page-6-0"></span>TABLE II QUANTITATIVE COMPARISON OF THE AVERAGE ACCURACY ON MVBENCH (20 SUB-TASKS), EGOSCHEMA, AND VIDEOMME BENCHMARKS. THE BEST RESULTS ARE HIGHLIGHTED IN BOLD, WITH ABSOLUTE PERFORMANCE GAINS OVER THE RESPECTIVE BASELINES INDICATED IN RED.

| Method                                  | Frame | MVBench [20]     | EgoSchema [21]   | VideoMME [22] |
|-----------------------------------------|-------|------------------|------------------|---------------|
| Proprietary MLLMs                       |       |                  |                  |               |
| GPT4-V [47]                             | 1fps  | 43.7             | 55.6             | 60.7          |
| GPT-4o [42]                             | 64    | 64.6             | 72.2             | 77.2          |
| Open-source Offline MLLMs               |       |                  |                  |               |
| LLaVA-NeXT-Video [5]                    | 32    | 33.7             | 43.9             | 46.5          |
| Qwen2-VL-7B [2]                         | 64    | 67.0             | 66.70            | 69.0          |
| VideoChat2 [20]                         | 16    | 60.4             | 54.4             | 54.6          |
| VideoLLaMA2 [48]                        | 32    | 54.6             | 51.7             | 46.6          |
| Open-Source Online MLLMs                |       |                  |                  |               |
| Dispider-7B [45]                        | 1fps  | -                | 55.60            | 57.20         |
| TimeChat-Online-7B [46]                 | 1fps  | 75.36            | 61.90            | 53.22         |
| StreamForest-7B [38]                    | 1fps  | 70.20            | -                | 61.40         |
| Training-free Offline-to-Online Methods |       |                  |                  |               |
| Qwen2.5-VL-7B [3]                       | 1fps  | 65.00            | 58.47            | 64.52         |
| + HERMES [14]                           | 1fps  | 65.53            | 59.47            | 60.63         |
| + Ours                                  | 1fps  | 66.03(↑<br>1.03) | 64.29(↑<br>5.82) | 62.97         |

TABLE III

<span id="page-6-1"></span>COMPARISON OF FRAME SAMPLING STRATEGIES USING QWEN2-VL-7B UNDER IDENTICAL TOKEN CONSTRAINTS (N=10) ON STREAMINGBENCH. OUR METRIC ACHIEVES OPTIMAL PERFORMANCE BY GEOMETRICALLY LOCATING GLOBAL SEMANTIC TRANSITIONS.

| Sampling Strategy    | Accuracy (%) |  |  |  |  |
|----------------------|--------------|--|--|--|--|
| Uniform Sampling     | 69.04        |  |  |  |  |
| Cosine Similarity    | 73.28        |  |  |  |  |
| Optical Flow         | 46.54        |  |  |  |  |
| Pyramid Optical Flow | 75.69        |  |  |  |  |
| Streamforest (train) | 77.26        |  |  |  |  |
| Ours (Curvature)     | 77.31        |  |  |  |  |

Taking StreamingBench and OVOBench as examples, after being integrated into the 4B, 8B, and 32B versions of Qwen3- VL, our method yields absolute performance improvements of 8.7%, 12.4%, and 11.5% on StreamingBench, respectively, compared to their corresponding uniform sampling baselines. Similarly, it achieves robust gains of 11.2%, 10.7%, and 10.6% on OVOBench.

These consistent quantitative improvements fully demonstrate that our curvature-aware adaptive memory mechanism does not overfit to models of a specific parameter volume. Instead, as a plug-and-play module, it maintains stable positive gains across multimodal base models ranging from small to large parameters, exhibiting exceptionally strong architectural universality and scalability.

TABLE IV

<span id="page-6-2"></span>ABLATION ON CURVATURE SCORE WEIGHTS (λ) USING QWEN2-VL-7B UNDER IDENTICAL TOKEN CONSTRAINTS ON OVOBENCH. CURVESTREAM MAINTAINS STABLE HIGH PERFORMANCE ACROSS DIVERSE WEIGHTS, VALIDATING ITS PLUG-AND-PLAY RELIABILITY.

| Method      | λ   | Accuracy (%) |
|-------------|-----|--------------|
| Qwen2-VL-7B | -   | 60.65        |
| Ours        | 0.2 | 65.83        |
| Ours        | 0.4 | 62.50        |
| Ours        | 0.6 | 63.33        |
| Ours        | 0.8 | 62.50        |
| Ours        | 1.0 | 65.00        |

#### *E. Ablation Studies*

To validate the independent contributions and synergistic effects of the core components in our adaptive memory framework, we conduct systematic ablation analyses on the Qwen-based model.

Effectiveness of Curvature Metric. To evaluate the superiority of manifold curvature in capturing temporal information increments, we compare different frame sampling strategies under identical visual token constraints (see Table [III\)](#page-6-1). The results demonstrate that our curvature metric significantly outperforms both uniform sampling and motion sampling based on cosine similarity. This confirms that pure motion similarity struggles to distinguish redundant smooth-panning shots from sudden semantic shifts. Furthermore, compared to dense optical flow, which is computationally expensive and highly susceptible to pixel noise, temporal manifold curvature

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

<span id="page-7-1"></span>Fig. 3. Scalability and memory allocation analysis. (a) CurveStream consistently delivers significant performance gains across varying model capacities (4B, 8B, 32B) of the Qwen3-VL series. (b) Impact of the clear memory (High-Res) retention ratio on overall accuracy and token cost. An adaptive ~50% ratio achieves the optimal trade-off between semantic integrity and computational overhead.

![](_page_7_Figure_3.jpeg)

<span id="page-7-2"></span>Fig. 4. Ablation on K-Sigma dual thresholds. CurveStream exhibits strong hyperparameter robustness across various  $k_1$  and  $k_2$  configurations on OVOBench. The dynamic mechanism effectively balances memory allocation between High-Res and Low-Res frames, ensuring an optimal accuracy-efficiency trade-off without tedious tuning.

serves as a lightweight second-order geometric prior, enabling more precise and robust localization of core turning points.

Adaptive Hierarchical Visual Memory Management. We further ablate the allocation ratio between clear memory (native high-resolution keyframes) and blurred memory (down-projected low-resolution transition frames) in the memory queue. As illustrated in Fig. 3b, forcing a 100% clear memory strategy accelerates context window depletion, triggering catastrophic forgetting of early memory. Conversely, adopting a 0% clear memory ("all-blur") strategy discards critical spatial details, leading to a drastic performance drop.

In contrast, the content-aware hybrid mechanism of our framework dynamically balances the clear memory ratio at approximately 50% based on temporal dynamics. This approach achieves the best accuracy while substantially reducing computational overhead by about 40%. This indicates that, compared to static constraints, dynamically allocating clear

and blurred memory more effectively strikes a balance between the integrity of long-term context and the capture of fine-grained actions. Specifically, due to the temporal non-stationarity of streaming videos, forcing a fixed high-resolution retention often overfits to local translational motion noise. In contrast, our adaptive 50% dynamic hybrid strategy essentially leverages localized blurred memory to serve as smooth transition states for action continuity, thereby freeing up the most critical clear memory space for high-curvature semantic transitions under the same token budget.

**Hyperparameter Robustness.** To verify the generalization stability of our framework, we evaluate the model's sensitivity to core hyperparameters. As shown in Table IV, when the curvature comprehensive score weight  $\lambda$  varies across a broad range of [0.1,0.4], the model accuracy remains steadily above 62.5%, peaking at 65.83%. The maximum absolute fluctuation is merely 3.33%, and it consistently outperforms

the baseline method. Similarly, the dual-threshold parameters, K SIGMA KEY (k1) and K SIGMA T RANS (k2), maintain highly stable performance and robust frame sampling ratios across different settings (see Fig. [4\)](#page-7-2). Such exceptionally low hyperparameter sensitivity strongly corroborates the intrinsic robustness of our framework as a plug-and-play module, capable of adapting to diverse underlying data streams without tedious heuristic tuning for real-world streaming tasks.

#### V. CONCLUSION

We present CurveStream, a training-free hierarchical memory management framework to boost streaming video understanding in MLLMs by tackling the inherent token explosion and Out-of-Memory (OOM) bottlenecks. Driven by the geometric insight that high-curvature regions in feature trajectories align with critical semantic transitions, CurveStream integrates a real-time Curvature Score with an online K-Sigma threshold. This dynamic mechanism adaptively routes incoming frames into clear or fuzzy memory states, ensuring MLLMs retain essential long-term visual context under strict token budgets.

Extensive experiments demonstrate that this lightweight, model-agnostic module exhibits broad architectural compatibility and consistently yields substantial performance gains over respective baselines. By establishing new state-of-the-art results on challenging benchmarks like StreamingBench and OVOBench, CurveStream offers a robust solution for continuous video perception. Future work will extend this geometric memory paradigm to broader embodied AI applications, such as autonomous navigation and prolonged robotic manipulation, where real-time adaptive reasoning and decision-making are paramount.

## REFERENCES

- <span id="page-8-0"></span>[1] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge *et al.*, "Qwen3-vl technical report," *arXiv preprint arXiv:2511.21631*, 2025. [1,](#page-0-0) [5,](#page-4-0) [6,](#page-5-1) [15,](#page-14-0) [16,](#page-15-0) [18](#page-17-0)
- <span id="page-8-22"></span>[2] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge *et al.*, "Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution," *arXiv preprint arXiv:2409.12191*, 2024. [1,](#page-0-0) [5,](#page-4-0) [6,](#page-5-1) [7,](#page-6-3) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-0)
- <span id="page-8-23"></span>[3] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, "Qwen2.5-vl technical report," *arXiv preprint arXiv:2502.13923*, 2025. [1,](#page-0-0) [5,](#page-4-0) [6,](#page-5-1) [7,](#page-6-3) [14,](#page-13-0) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-0)
- <span id="page-8-15"></span>[4] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, "Videochat: Chat-centric video understanding," *Science China Information Sciences*, vol. 68, no. 10, p. 200102, 2025. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-8-24"></span>[5] Y. Zhang, B. Li, H. Liu, Y. J. Lee, L. Gui, D. Fu, J. Feng, Z. Liu, and C. Li, "Llava-next: A strong zero-shot video understanding model," April 2024. [Online]. Available: [https://llava-vl.github.io/blog/](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) [2024-04-30-llava-next-video/](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) [1,](#page-0-0) [7,](#page-6-3) [17](#page-16-0)
- <span id="page-8-1"></span>[6] B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, "Video-llava: Learning united visual representation by alignment before projection," in *Proceedings of the 2024 conference on empirical methods in natural language processing*, 2024, pp. 5971–5984. [1](#page-0-0)
- <span id="page-8-2"></span>[7] R. Xu, G. Xiao, Y. Chen, L. He, K. Peng, Y. Lu, and S. Han, "Streamingvlm: Real-time understanding for infinite video streams," *arXiv preprint arXiv:2510.09608*, 2025. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-8-3"></span>[8] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, "Adaptive keyframe sampling for long video understanding," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 29 118–29 128. [1](#page-0-0)

- [9] J. Ye, Z. Wang, H. Sun, K. Chandrasegaran, Z. Durante, C. Eyzaguirre, Y. Bisk, J. C. Niebles, E. Adeli, L. Fei-Fei *et al.*, "Re-thinking temporal search for long-form video understanding," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025, pp. 8579–8591. [1](#page-0-0)
- <span id="page-8-4"></span>[10] K. Hu, F. Gao, X. Nie, P. Zhou, S. Tran, T. Neiman, L. Wang, M. Shah, R. Hamid, B. Yin *et al.*, "M-llm based video frame selection for efficient video understanding," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 13 702–13 712. [1](#page-0-0)
- <span id="page-8-5"></span>[11] K. Li, P. Ye, L. Zhang, C. Wang, H. Qin, and T. Chen, "Freshmem: Brain-inspired frequency-space hybrid memory for streaming video understanding," *arXiv preprint arXiv:2602.01683*, 2026. [1,](#page-0-0) [5,](#page-4-0) [6,](#page-5-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-8-6"></span>[12] Y. Wang, Y. Song, C. Xie, Y. Liu, and Z. Zheng, "Videollamb: Long streaming video understanding with recurrent memory bridges," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 24 170–24 181. [1](#page-0-0)
- <span id="page-8-7"></span>[13] H. Xiong, Z. Yang, J. Yu, Y. Zhuge, L. Zhang, J. Zhu, and H. Lu, "Streaming video understanding and multi-round interaction with memory-enhanced knowledge," *arXiv preprint arXiv:2501.13468*, 2025. [1](#page-0-0)
- <span id="page-8-8"></span>[14] H. Zhang, S. Yang, J. Fu, S.-K. Ng, and X. Qiu, "Hermes: Kv cache as hierarchical memory for efficient streaming video understanding," 2026. [Online]. Available: <https://arxiv.org/abs/2601.14724> [1,](#page-0-0) [3,](#page-2-0) [5,](#page-4-0) [6,](#page-5-1) [7,](#page-6-3) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-8-20"></span>[15] M. Kim, K. Shim, J. Choi, and S. Chang, "Infinipot-v: Memoryconstrained KV cache compression for streaming video understanding," in *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025. [Online]. Available: [https://openreview.net/](https://openreview.net/forum?id=hFxOZjHyTg) [forum?id=hFxOZjHyTg](https://openreview.net/forum?id=hFxOZjHyTg) [1,](#page-0-0) [3](#page-2-0)
- <span id="page-8-9"></span>[16] Y. Yang, Z. Zhao, S. N. Shukla, A. Singh, S. K. Mishra, L. Zhang, and M. Ren, "Streammem: Query-agnostic kv cache memory for streaming video understanding," *arXiv preprint arXiv:2508.15717*, 2025. [1](#page-0-0)
- <span id="page-8-10"></span>[17] S. Di, Z. Yu, G. Zhang, H. Li, H. Cheng, B. Li, W. He, F. Shu, H. Jiang *et al.*, "Streaming video question-answering with in-context video kvcache retrieval," in *ICLR*, 2025. [1,](#page-0-0) [3,](#page-2-0) [5,](#page-4-0) [6,](#page-5-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-8-11"></span>[18] J. Lin, Z. Fang, C. Chen, Z. Wan, F. Luo, P. Li, Y. Liu, and M. Sun, "Streamingbench: Assessing the gap for mllms to achieve streaming video understanding," *arXiv preprint arXiv:2411.03628*, 2024. [2,](#page-1-1) [5,](#page-4-0) [6,](#page-5-1) [14](#page-13-0)
- <span id="page-8-12"></span>[19] J. Niu, Y. Li, Z. Miao, C. Ge, Y. Zhou, Q. He, X. Dong, H. Duan, S. Ding, R. Qian *et al.*, "Ovo-bench: How far is your video-llms from real-world online video understanding?" in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 18 902–18 913. [2,](#page-1-1) [5,](#page-4-0) [6,](#page-5-1) [14](#page-13-0)
- <span id="page-8-13"></span>[20] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo *et al.*, "Mvbench: A comprehensive multi-modal video understanding benchmark," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 22 195–22 206. [2,](#page-1-1) [5,](#page-4-0) [7,](#page-6-3) [15](#page-14-0)
- <span id="page-8-21"></span>[21] K. Mangalam, R. Akshulakov, and J. Malik, "Egoschema: A diagnostic benchmark for very long-form video language understanding," *Advances in Neural Information Processing Systems*, vol. 36, pp. 46 212–46 244, 2023. [2,](#page-1-1) [5,](#page-4-0) [7](#page-6-3)
- <span id="page-8-14"></span>[22] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang *et al.*, "Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2025, pp. 24 108–24 118. [2,](#page-1-1) [5,](#page-4-0) [7](#page-6-3)
- <span id="page-8-16"></span>[23] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu *et al.*, "Llava-onevision: Easy visual task transfer," *arXiv preprint arXiv:2408.03326*, 2024. [3,](#page-2-0) [5,](#page-4-0) [6,](#page-5-1) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-0)
- <span id="page-8-25"></span>[24] C. Tu, L. Zhang, P. Chen, P. Ye, X. Zeng, W. Cheng, G. Yu, and T. Chen, "Favor-bench: A comprehensive benchmark for fine-grained video motion understanding," *arXiv preprint arXiv:2503.14935*, 2025. [3,](#page-2-0) [15](#page-14-0)
- <span id="page-8-17"></span>[25] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," 2023. [3](#page-2-0)
- <span id="page-8-18"></span>[26] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, "Adaptive keyframe sampling for long video understanding," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 29 118–29 128. [3](#page-2-0)
- <span id="page-8-19"></span>[27] K. Hu, F. Gao, X. Nie, P. Zhou, S. Tran, T. Neiman, L. Wang, M. Shah, R. Hamid, B. Yin *et al.*, "M-llm based video frame selection for efficient video understanding," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 13 702–13 712. [3](#page-2-0)

- <span id="page-9-0"></span>[28] J. Cho, J. Lee, M. Hayat, K. Hwang, F. Porikli, and S. Choi, "Floc: Facility location-based efficient visual token compression for long video understanding," *arXiv preprint arXiv:2511.00141*, 2025. [3](#page-2-0)
- <span id="page-9-1"></span>[29] Y. Lu, T. Wang, F. Rao, Y. Yang, L. Zhu *et al.*, "Flexselect: Flexible token selection for efficient long video understanding," in *The Thirtyninth Annual Conference on Neural Information Processing Systems*. [3](#page-2-0)
- <span id="page-9-2"></span>[30] M. Wang, S. Chen, K. Kersting, V. Tresp, and Y. Ma, "Metok: Multistage event-based token compression for efficient long video understanding," in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 2025, pp. 18 881–18 895. [3](#page-2-0)
- <span id="page-9-3"></span>[31] S. Di, Z. Yu, G. Zhang, H. Li, T. Zhong, H. Cheng, B. Li, W. He, F. Shu, and H. Jiang, "Streaming video question-answering with in-context video kv-cache retrieval," *arXiv preprint arXiv:2503.00540*, 2025. [3](#page-2-0)
- <span id="page-9-4"></span>[32] H. Gao, Y. Bao, X. Tu, B. Zhong, L. Yue, and M. Zhang, "Apvr: Hourlevel long video understanding with adaptive pivot visual information retrieval," *arXiv preprint arXiv:2506.04953*, 2025. [3](#page-2-0)
- <span id="page-9-5"></span>[33] S. Liu, C. Zhao, T. Xu, and B. Ghanem, "Bolt: Boost large visionlanguage model without training for long-form video understanding," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 3318–3327. [3](#page-2-0)
- <span id="page-9-6"></span>[34] Y. Tang, W. Wang, L. Guo, T. Yue, W. Wang, C. Zhang, and J. Liu, "Divid: Disentangled spatial-temporal modeling within llms for temporally grounded video understanding," in *The Fourteenth International Conference on Learning Representations*. [3](#page-2-0)
- <span id="page-9-7"></span>[35] Y. Chen, X. Bai, Z. Wang, C. Bai, Y. Dai, M. Lu, and S. Zhang, "Streamkv: Streaming video question-answering with segment-based kv cache retrieval and compression," *arXiv preprint arXiv:2511.07278*, 2025. [3](#page-2-0)
- <span id="page-9-8"></span>[36] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, and X. Jin, "Flash-vstream: Efficient real-time understanding for long video streams," in *Proceedings of the IEEE/CVF international conference on computer vision*, 2025, pp. 21 059–21 069. [3,](#page-2-0) [5,](#page-4-0) [6,](#page-5-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-9"></span>[37] X. Chen, K. Tao, K. Shao, and H. Wang, "Streamingtom: Streaming token compression for efficient video understanding," *arXiv preprint arXiv:2510.18269*, 2025. [3](#page-2-0)
- <span id="page-9-10"></span>[38] X. Zeng, K. Qiu, Q. Zhang, X. Li, J. Wang, J. Li, Z. Yan, K. Tian, M. Tian, X. Zhao *et al.*, "Streamforest: Efficient online video understanding with persistent event memory," *arXiv preprint arXiv:2509.24871*, 2025. [3,](#page-2-0) [6,](#page-5-1) [7,](#page-6-3) [14,](#page-13-0) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-11"></span>[39] J. Zuo, Y. Deng, L. Kong, J. Yang, R. Jin, Y. Zhang, N. Sang, L. Pan, Z. Liu, and C. Gao, "Videolucy: Deep memory backtracking for long video understanding," *arXiv preprint arXiv:2510.12422*, 2025. [3](#page-2-0)
- <span id="page-9-12"></span>[40] S. Ye, B. Ouyang, T. Qian, L. Zeng, M. Yuan, X. Chu, W. Hong, and X. Chen, "Venus: An efficient edge memory-and-retrieval system for vlm-based online video understanding," *arXiv preprint arXiv:2512.07344*, 2025. [3](#page-2-0)
- <span id="page-9-13"></span>[41] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, and et al., "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," 2024. [Online]. Available: <https://arxiv.org/abs/2403.05530> [6,](#page-5-1) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-0)
- <span id="page-9-14"></span>[42] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford *et al.*, "Gpt-4o system card," *arXiv preprint arXiv:2410.21276*, 2024. [6,](#page-5-1) [7,](#page-6-3) [15,](#page-14-0) [16,](#page-15-0) [17](#page-16-0)
- <span id="page-9-15"></span>[43] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu *et al.*, "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 24 185–24 198. [6,](#page-5-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-16"></span>[44] J. Chen, Z. Lv, S. Wu, K. Q. Lin, C. Song, D. Gao, J.-W. Liu, Z. Gao, D. Mao, and M. Z. Shou, "Videollm-online: Online video large language model for streaming video," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 18 407–18 418. [6,](#page-5-1) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-17"></span>[45] R. Qian, S. Ding, X. Dong, P. Zhang, Y. Zang, Y. Cao, D. Lin, and J. Wang, "Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 24 045–24 055. [6,](#page-5-1) [7,](#page-6-3) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-18"></span>[46] L. Yao, Y. Li, Y. Wei, L. Li, S. Ren, Y. Liu, K. Ouyang, L. Wang, S. Li, S. Li *et al.*, "Timechat-online: 80% visual tokens are naturally redundant in streaming videos," in *Proceedings of the 33rd ACM International Conference on Multimedia*, 2025, pp. 10 807–10 816. [6,](#page-5-1) [7,](#page-6-3) [15,](#page-14-0) [16](#page-15-0)
- <span id="page-9-19"></span>[47] X. Zhang, Y. Lu, W. Wang, A. Yan, J. Yan, L. Qin, H. Wang, X. Yan, W. Y. Wang, and L. R. Petzold, "Gpt-4v (ision) as a generalist evaluator for vision-language tasks," *arXiv preprint arXiv:2311.01361*, 2023. [7](#page-6-3)

- <span id="page-9-20"></span>[48] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao, and L. Bing, "Videollama 2: Advancing spatial-temporal modeling and audio understanding in videollms," *arXiv preprint arXiv:2406.07476*, 2024. [Online]. Available: <https://arxiv.org/abs/2406.07476> [7](#page-6-3)
- <span id="page-9-21"></span>[49] Anthropic, "Claude 3.5 sonnet," 2024. [Online]. Available: [https:](https://www.anthropic.com/news/claude-3-5-sonnet) [//www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) [15](#page-14-0)
- <span id="page-9-22"></span>[50] ——, "Claude 3.7 sonnet," 2025. [Online]. Available: [https://www.](https://www.anthropic.com/claude/sonnet) [anthropic.com/claude/sonnet](https://www.anthropic.com/claude/sonnet) [17](#page-16-0)
- <span id="page-9-23"></span>[51] B. Lin, B. Zhu, Y. Ye, M. Ning, P. Jin, and L. Yuan, "Video-llava: Learning united visual representation by alignment before projection," *arXiv preprint arXiv:2311.10122*, 2023. [17](#page-16-0)
- <span id="page-9-24"></span>[52] J. Wang, L. Yuan, Y. Zhang, and H. Sun, "Tarsier: Recipes for training and evaluating large video description models," 2024. [Online]. Available: <https://arxiv.org/abs/2407.00634> [17](#page-16-0)
- <span id="page-9-25"></span>[53] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu *et al.*, "Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling," *arXiv preprint arXiv:2412.05271*, 2024. [17](#page-16-0)
- <span id="page-9-26"></span>[54] X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao, K. Li, Y. He, C. Wang, Y. Qiao, Y. Wang, and L. Wang, "Videochat-flash: Hierarchical compression for long-context video modeling," *arXiv preprint arXiv:2501.00574*, 2024. [17](#page-16-0)
- <span id="page-9-27"></span>[55] B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng, Y. Jiang, H. Zhang, X. Li, P. Jin, W. Zhang, F. Wang, L. Bing, and D. Zhao, "Videollama 3: Frontier multimodal foundation models for image and video understanding," *arXiv preprint arXiv:2501.13106*, 2025. [Online]. Available: <https://arxiv.org/abs/2501.13106> [17](#page-16-0)

#### APPENDIX A CURVESTREAM ALGORITHM

In this section, we provide the detailed pseudo-code for the proposed CurveStream framework. As outlined in Al**gorithm 1**, the online memory scheduling process operates sequentially on the incoming video stream without requiring any future context. For each new frame, the system first extracts its feature representation via the frozen visual encoder. Subsequently, the Curvature-Aware Scorer (CAS) evaluates the semantic transition by calculating the feature manifold curvature. Based on this dynamic curvature score and the recursively updated transient distribution, the Hierarchical Visual Memory Management (HVMM) module dynamically routes the current frame into either high-resolution Clear Memory or down-sampled Blurred Memory using dual adaptive thresholds. Finally, a strict First-In-First-Out (FIFO) eviction policy is applied to ensure the maximum memory footprint is strictly bounded.

## APPENDIX B QUALITATIVE CASE STUDIES

To intuitively illustrate the effectiveness of our memory mechanism in handling complex, unconstrained streaming videos, we provide qualitative comparisons between CurveStream and the robust baseline model (Owen3-VL-32B) in Fig. 5 to Fig. 8. We select four highly challenging subtasks from OVOBench: Action Recognition (Fig. 5), Future Prediction (Fig. 6), Attribute Recognition (Fig. 7), and Object Recognition (Fig. 8).

In highly dynamic or visually cluttered scenarios, standard MLLMs often suffer from severe hallucination or catastrophic forgetting. This is primarily because their passive memory eviction policies indiscriminately discard historical tokens, leading to broken causal chains, or their uniform downsampling strategies irreparably blur fine-grained spatial details. As demonstrated in the following cases, CurveStream successfully overcomes these bottlenecks. By monitoring the feature manifold curvature, our framework accurately anchors critical semantic transitions (e.g., the sudden appearance of a small object or a rapid action shift) and intelligently routes them into the high-resolution Clear Memory. This ensures the model maintains a precise, coherent, and hallucination-free understanding across the entire streaming timeline.

#### APPENDIX C

#### THEORETICAL ANALYSIS OF THE GEOMETRIC CURVATURE METRIC

In this section, we provide a rigorous theoretical formulation for the geometric curvature  $(C_t)$  metric introduced in the Curvature-Aware Scorer (CAS). From a discrete geometric perspective, we demonstrate how this metric theoretically decouples core semantic transitions from continuous physical motion noise.

## A. Kinematic Modeling in the Latent Manifold

Let the continuous video stream be mapped into a highdimensional latent feature space. Following  $L_2$  normalization, <span id="page-10-0"></span>Algorithm 1 CurveStream: Curvature-Aware Hierarchical Visual Memory Management

**Require:** Continuous video stream  $\mathcal{V} = \{I_t\}_{t=1}^{\infty}$  and query timestamp  $t_q$ ; target memory capacity  $N_{max}$ ; balancing coefficient  $\lambda$ ; threshold multipliers  $k_1, k_2$  ( $k_1 < k_2$ )

**Ensure:** An adaptively updated visual memory queue  $\mathcal{M}_t$ 

- 1:  $\triangleright$  Initialize the memory queue  $\mathcal{M}_0 \leftarrow \emptyset$ , and the time step
- 2: Initialize transient distribution parameters:  $\mu_0 \leftarrow 0$ ,  $\sigma_0 \leftarrow$
- 3: **while** receiving incoming frame  $I_t$  from stream V **do**
- $\triangleright$  Extract and  $L_2$ -normalize global feature representation  $F_t \in \mathbb{R}^D$  via the frozen visual encoder
- 5: if t > 3 then
- 6:
- Compute first-order Motion Variation:  $M_t$ 7:  $\mathcal{D}_{motion}(F_t, F_{t-1})$
- Compute second-order Geometric Curvature:  $C_t =$ 8:  $\mathcal{K}_{geo}(F_t, F_{t-1}, F_{t-2})$
- 9: Calculate the final Curvature Score:  $CS_t = M_t + \lambda C_t$
- 10: (HVMM)
- Recursively update the transient manifold distribution 11: state:
- $(\mu_t, \sigma_t) \leftarrow \text{UpdateDistributionState}(CS_t, \mu_{t-1}, \sigma_{t-1})$ 12:
- Generate dynamic dual thresholds: 13:
  - $g_1, g_2 \leftarrow \text{CalculateDynamicThresholds}(\mu_t, \sigma_t, k_1, k_2)$
- if  $CS_t \geq g_2$  or  $t == t_q$  then 15:
- ▶ Retain as Clear Memory to capture significant 16: semantic shifts
- 17:  $\mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1}, I_t, s_t = \text{Clear}, r_t = \text{High})$ 18:
  - else if  $g_1 \leq CS_t < g_2$  then
- ▶ Retain as Blurred Memory for intermediate tran-19: sition states
- $\mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1}, I_t, s_t = \text{Blurred}, r_t = \text{Low})$ 20:
- 21:

14:

- 22. ▷ Discard low-information redundant features
- $\mathcal{M}_t = \mathcal{M}_{t-1}$ 23:
- end if 24:
- 25: if  $|\mathcal{M}_t| > N_{max}$  then
- ▶ Execute strict First-In-First-Out (FIFO) eviction 26:
- Remove the oldest tokens from  $\mathcal{M}_t$ 27:
- end if 28:
- 29: end if
- $\triangleright t \leftarrow t + 1$
- 31: end while

the observation of each video frame  $I_t$  is projected onto a unit hypersphere, yielding the feature representation  $F_t \in \mathbb{R}^D$ . The temporal evolution of the video stream constructs a discrete parameterized curve on this hyperspherical manifold.

From a kinematic perspective, the first-order feature displacement vectors  $d_1 = F_{t-1} - F_{t-2}$  and  $d_2 = F_t - F_{t-1}$ represent the discrete velocity vectors of the visual signal at

![](_page_11_Figure_1.jpeg)

<span id="page-11-0"></span>Fig. 5. Action Recognition in dynamic virtual environments. Fast-paced viewpoint shifts often cause baseline models to lose track of transient actions, resulting in severe hallucinations (e.g., misinterpreting the action as setting up a camera). CurveStream captures the sharp curvature peak during the "drinking" animation, preserving it as a key semantic node to deliver an accurate response.

![](_page_11_Figure_3.jpeg)

<span id="page-11-1"></span>Fig. 6. Future Prediction in egocentric views. Predicting future actions requires a complete and unbroken causal chain of past events. While the baseline suffers from context truncation and guesses the next action based on a biased background bias (the chair), CurveStream maintains a coherent sequence of the subject's interactions, correctly inferring the intention to operate the smartphone.

adjacent time steps. Traditional similarity metrics (*e.g.*, interframe cosine similarity) primarily rely on the magnitude of these velocity vectors, which inherently conflates semantic transitions with smooth, continuous camera motions (*e.g.*, panning).

## B. Differential Geometric Perspective of $C_t$

To isolate semantic intensity, we approximate the secondorder geometric curvature of the feature trajectory. In continuous differential geometry, the curvature  $\kappa$  of a parameterized curve measures the rate of change of the unit tangent vector with respect to arc length.

We map this definition onto our discrete manifold. First, we compute the unit tangent vectors (*i.e.*, normalized velocity vectors) at adjacent time steps:

$$T_1 = \frac{d_1}{||d_1||}, \quad T_2 = \frac{d_2}{||d_2||}$$
 (8)

The geometric curvature metric proposed in this paper is defined as the cosine distance between adjacent displacement

![](_page_12_Figure_1.jpeg)

<span id="page-12-1"></span>Fig. 7. Attribute Recognition requiring fine-grained spatial details. Standard memory limits often force base models to downsample past frames uniformly, blurring complex textures. CurveStream dynamically assigns high-resolution Clear Memory to informative frames where the pot's pattern is unobscured, allowing it to correctly identify the nested diamond shapes.

![](_page_12_Figure_3.jpeg)

<span id="page-12-0"></span>Fig. 8. **Object Recognition under severe occlusion.** Tracking small objects (like the fork) is notoriously difficult in long videos. CurveStream registers the semantic shift when the utensil is clearly exposed, safeguarding this vital visual evidence in the memory queue to prevent the baseline's "wooden stick" hallucination.

vectors:

$$C_t = 1 - \frac{\langle d_1, d_2 \rangle}{||d_1|| \cdot ||d_2||} = 1 - \langle T_1, T_2 \rangle$$
 (9)

In Euclidean space, the squared distance between two unit vectors has a strict mathematical equivalence with their inner product:

$$||T_2 - T_1||^2 = ||T_2||^2 + ||T_1||^2 - 2\langle T_1, T_2 \rangle = 2(1 - \langle T_1, T_2 \rangle)$$
(10)

Substituting this into our metric yields the geometric equivalence:

$$C_t = \frac{1}{2}||T_2 - T_1||^2 \tag{11}$$

This theoretical derivation proves that  $C_t$  is strictly equivalent (up to a constant scaling factor) to the squared variation of the unit tangent vector. Thus, as a discrete approximation of manifold curvature,  $C_t$  geometrically evaluates the direc-

<span id="page-13-0"></span>tional derivative of feature evolution instead of a mere scalar displacement.

#### *C. Theoretical Advantages of Semantic Decoupling*

This curvature-based formulation inherently provides two critical theoretical advantages for streaming video understanding:

- Immunity to Constant Velocity Motion Noise: In scenarios with smooth, continuous motion (*e.g.*, stable camera panning), the feature trajectory evolves at a relatively constant velocity. Geometrically, its tangent vectors remain approximately parallel (T<sup>1</sup> ≈ T2), yielding ⟨T1, T2⟩ ≈ 1 and C<sup>t</sup> ≈ 0. Consequently, this geometric penalty inherently suppresses low-level physical motion noise by mechanism.
- Orthogonal Sensitivity to Semantic Transitions: When sudden semantic shifts occur (*e.g.*, shot changes, new entities entering the frame, or sharp action boundaries), the feature trajectory undergoes a drastic directional deviation. The new velocity vector d<sup>2</sup> is projected into a subspace that is nearly orthogonal or even divergent from d1. This forces the inner product ⟨T1, T2⟩ to drop sharply, thereby generating a distinct curvature spike.

By introducing this second-order geometric prior, the CAS module achieves an effective decoupling of core semantic transitions from redundant background dynamics on a mathematical basis, laying a robust theoretical foundation for the subsequent K-Sigma dynamic memory routing mechanism.

## APPENDIX D

## DETAILED PERFORMANCE ON STREAMING BENCHMARKS

We present the comprehensive, fine-grained evaluation results of our proposed curvature-aware hierarchical visual memory management method on streaming video benchmarks. The detailed breakdowns across StreamingBench [\[18\]](#page-8-11) (Table [V](#page-14-1)) and OVO-Bench [\[19\]](#page-8-12) (Table [VI](#page-15-1)) are shown below. We compare our approach against standard MLLMs (*e.g.*, Qwen2.5- VL [\[3\]](#page-8-23)) and state-of-the-art streaming baselines (*e.g.*, Stream-Forest [\[38\]](#page-9-10)).

The comprehensive performance improvements of CurveStream across both streaming benchmarks are primarily attributed to our redesign of the Hierarchical Visual Memory Management mechanism. Confronted with the continuous growth of tokens in long streaming videos, base models are typically bounded by rigid memory mechanisms (*e.g.*, fixed uniform downsampling or passive FIFO cache eviction). This easily leads to the loss of high-value semantic information and the disruption of the model's contextual coherence. CurveStream constructs an adaptive Hierarchical Visual Memory system. We utilize the local curvature on the feature manifold as a perceptual heuristic to guide the dynamic allocation of memory: under the premise of strictly constraining memory overhead, the video stream is intelligently decoupled into highresolution Clear Memory and high-compression-ratio Blurred Memory. This strategy of "semantic-perception-driven memory routing" effectively alleviates the resource allocation bottlenecks of base models in long sequences, providing solid architectural support for the performance leaps across various sub-tasks.

#### *A. Analysis of Improvements on StreamingBench*

CR (Causal Reasoning), EU (Event Understanding) & ACP (Action Perception): One of the core challenges of StreamingBench lies in memory retention under long-term contexts. Constrained by limited context windows, base models often have early key events squeezed out by subsequent redundant frames, leading to difficulties in long-range reasoning. CurveStream's hierarchical architecture provides a viable path to alleviate this issue. Clear Memory focuses on the persistent storage of discrete salient events triggered by high curvature, while Blurred Memory maintains the background context between events at a lower token cost. This macroscopic memory scheduling approach constructs a relatively complete and compact "causal topological chain" for the model, assisting it in better handling complex, long-range logical correlation problems even when operating under severely limited memory capacities.

CT (Counting) & CS (Clips Summarization): In counting and summarization tasks, the loss of historical states is often a critical cause of model output errors. CurveStream's memory management demonstrates strong robustness here. By transforming significant action mutations into discrete keyframe snapshots and retaining them, it essentially compresses the continuous, lengthy video stream into a high-density sequence containing core events. This mechanism provides base models with a more structured and reliable basis for memory retrieval when handling complex frequency statistics, event counting, and global video summarization queries.

## *B. Analysis of Improvements on OVO-Bench*

OCR (Optical Character Recognition) & ATR (Attribute Recognition): These tasks highly rely on the retention of high-resolution visual features. Under memory pressure, base models often resort to global downsampling, easily causing an irreversible loss of fine-grained information. CurveStream's hierarchical memory management adeptly tackles this resource allocation dilemma. When the Curvature-Aware Scorer (CAS) detects significant changes in text or attributes, the system prioritizes allocating the token budget to these key frames, maintaining their native high resolution as Clear Memory. Simultaneously, low-information-density background frames are compressed into Blurred Memory. This dynamic memory scheduling strategy significantly enhances the model's perception of fine-grained information while maintaining a highly stable and consistent overall memory footprint.

ACR (Action Recognition) & FPD (Future Prediction): The sliding window memory mechanism of base models, when constrained by capacity, easily evicts the preceding states of actions, thereby compromising the integrity of temporal logic. CurveStream maps the fluctuations of actions to curvature variations on the feature manifold, utilizing these variations to assist in locating action boundaries and anchoring them as key semantic nodes in the working memory. This mechanism helps ensure that the model is supported by a more coherent

<span id="page-14-1"></span><span id="page-14-0"></span>TABLE V COMPREHENSIVE EVALUATION RESULTS ON THE STREAMINGBENCH BENCHMARK FOR REAL-TIME VISUAL UNDERSTANDING. OUR METHOD CONSISTENTLY IMPROVES THE PERFORMANCE OF BASE MLLMS ACROSS ALL CONTEXTUAL REASONING AND EVENT UNDERSTANDING METRICS, AS INDICATED BY THE "↑" SYMBOL. THE HIGHEST SCORES IN EACH COLUMN ARE MARKED IN BOLD.

| Model                                   | Frame                     | OP    | CR    | CS    | ATP   | EU    | TR    | PR    | SU    | ACP   | CT    | Avg.            |
|-----------------------------------------|---------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----------------|
| Human                                   | -                         | 89.47 | 92.00 | 93.60 | 91.47 | 95.65 | 92.52 | 88.00 | 88.75 | 89.74 | 91.30 | 91.46           |
| Proprietary MLLMs                       |                           |       |       |       |       |       |       |       |       |       |       |                 |
| Gemini 1.5 Pro [41]                     | 1 fps                     | 79.02 | 80.47 | 83.54 | 79.67 | 80.00 | 84.74 | 77.78 | 64.23 | 71.95 | 48.70 | 75.69           |
| GPT-4o [42]                             | 64                        | 77.11 | 80.47 | 83.91 | 76.47 | 70.19 | 83.80 | 66.67 | 62.19 | 69.12 | 49.22 | 73.28           |
| Claude 3.5 Sonnet [49]                  | 20                        | 73.33 | 80.47 | 84.09 | 82.02 | 75.39 | 79.53 | 61.11 | 61.79 | 69.32 | 43.09 | 72.44           |
|                                         | Open-source Offline MLLMs |       |       |       |       |       |       |       |       |       |       |                 |
| Qwen2-VL-7B [2]                         | 32                        | 55.86 | 55.47 | 57.41 | 58.17 | 52.80 | 43.61 | 39.81 | 42.68 | 45.61 | 35.23 | 49.52           |
| InternVL-V2-8B [43]                     | 14                        | 53.68 | 49.22 | 70.98 | 56.86 | 53.42 | 53.89 | 54.63 | 48.78 | 50.14 | 17.62 | 52.32           |
| Open-source Online MLLMs                |                           |       |       |       |       |       |       |       |       |       |       |                 |
| Flash-VStream-7B [36]                   | -                         | 25.89 | 43.57 | 24.91 | 23.87 | 27.33 | 13.08 | 18.52 | 25.20 | 23.87 | 48.70 | 23.23           |
| VideoLLM-online-8B [44]                 | 2 fps                     | 39.07 | 40.06 | 34.49 | 31.05 | 45.96 | 32.40 | 31.48 | 34.16 | 42.49 | 27.89 | 35.99           |
| Dispider-7B [45]                        | 1 fps                     | 74.92 | 75.53 | 74.10 | 73.08 | 74.44 | 59.92 | 76.14 | 62.91 | 62.16 | 45.80 | 67.63           |
| TimeChat-Online-7B [46]                 | 1 fps                     | 80.22 | 82.03 | 79.50 | 83.33 | 76.10 | 78.50 | 78.70 | 64.63 | 69.60 | 57.98 | 75.36           |
| StreamForest-7B [38]                    | 1 fps                     | 83.11 | 82.81 | 82.65 | 84.26 | 77.50 | 78.19 | 76.85 | 69.11 | 75.64 | 54.40 | 77.26           |
| Training-free Offline-to-Online Methods |                           |       |       |       |       |       |       |       |       |       |       |                 |
| LLaVA-OV-7B [23]                        | 32                        | 78.75 | 78.12 | 80.76 | 81.19 | 71.70 | 72.59 | 72.22 | 63.82 | 66.01 | 38.34 | 71.34           |
| + ReKV [17]                             | 0.5 fps                   | 76.02 | 81.25 | 77.92 | 76.90 | 66.04 | 66.04 | 69.44 | 60.98 | 64.31 | 49.22 | 69.22           |
| + HERMES [14]                           | 0.5 fps                   | 79.02 | 81.25 | 87.70 | 80.20 | 69.18 | 71.96 | 73.15 | 66.26 | 69.41 | 43.52 | 73.23           |
| + Ours (CurveStream)                    | 10-20                     | 85.56 | 85.13 | 71.88 | 88.52 | 72.50 | 83.49 | 65.74 | 69.51 | 67.90 | 35.42 | 75.12 (↑ 3.78)  |
| Qwen2-VL-7B [2]                         | 1 fps                     | 77.38 | 76.56 | 73.19 | 75.08 | 75.00 | 67.91 | 73.15 | 65.04 | 66.57 | 35.75 | 69.04           |
| + Freshmem [11]                         | 1 fps                     | 84.47 | 83.59 | 77.60 | 83.28 | 78.12 | 80.37 | 70.37 | 74.39 | 66.86 | 30.05 | 74.20           |
| + Ours (CurveStream)                    | 10-20                     | 88.56 | 77.34 | 88.61 | 89.84 | 76.25 | 92.52 | 76.85 | 76.83 | 76.70 | 45.31 | 81.04 (↑ 12.00) |
| Qwen2.5-VL-7B [3]                       | 1 fps                     | 77.93 | 76.56 | 78.55 | 80.86 | 76.73 | 76.95 | 80.56 | 65.45 | 65.72 | 52.85 | 73.31           |
| + HERMES [14]                           | 1 fps                     | 83.65 | 81.25 | 88.01 | 87.46 | 76.73 | 86.60 | 82.41 | 76.02 | 73.94 | 46.63 | 79.44           |
| + Ours (CurveStream)                    | 10-20                     | 90.19 | 78.12 | 94.94 | 89.51 | 81.25 | 95.02 | 83.33 | 83.74 | 79.26 | 44.79 | 84.00 (↑ 10.69) |
| Qwen3-VL-8B [1]                         | 1 fps                     | 76.84 | 77.22 | 77.29 | 80.74 | 70.35 | 75.21 | 80.56 | 64.23 | 65.76 | 49.22 | 73.2            |
| + Ours (CurveStream)                    | 10-20                     | 90.74 | 79.69 | 95.25 | 93.44 | 81.88 | 95.95 | 85.19 | 79.27 | 85.23 | 47.92 | 85.56 (↑ 12.36) |

and complete history of state transitions when reasoning about current actions or predicting future evolutions, effectively reducing the risk of hallucination caused by context truncation.

STU (Spatial Understanding) & OJR (Object Recognition): Complex spatial structures and target poses constantly change with camera motion. Fixed uniform sampling strategies sometimes fail to retain frames with optimal viewpoints in memory. With the help of the K-Sigma dynamic threshold, CurveStream achieves adaptive memory updating, enabling the system to better adapt to variable camera motion rhythms. It maximizes the retention of frames containing rich spatial topological relations in the core memory area, thereby substantially reducing visual information omissions typically caused by improper or rigid memory scheduling.

## APPENDIX E GENERALIZATION ON OFFLINE VIDEO UNDERSTANDING

Although the CurveStream architecture was primarily designed to alleviate memory bottlenecks in streaming scenarios, its core mechanism—Curvature-Aware Hierarchical Visual Memory Management also provides an efficient representation paradigm for offline long-video understanding. In the offline evaluation setting, confronted with complete video sequences, CurveStream overcomes the limitations of conventional fixed frame sampling. By evaluating the semantic information density across the global temporal axis and utilizing curvature to adaptively route the limited token budget to highly dynamic segments, this mechanism demonstrates highly robust generalization capabilities when evaluated across two major offline video understanding benchmarks.

The details across FAVOR-Bench [\[24\]](#page-8-25) (Table [VII](#page-16-1)) and MVBench [\[20\]](#page-8-13) (Table [VIII](#page-16-2)) are presented below.

MVBench: According to the task definition of MVBench, the core challenge lies in solving "temporal dependencies that cannot be effectively solved with a single frame," such as complex action sequences and object interactions. The high curvature on the feature manifold captured by CurveStream naturally aligns with these state mutation points to some extent. By accurately routing and retaining these key frames in Clear Memory, the model can better construct a visual causal evidence chain, thereby achieving stable performance improvements over baseline models on various sub-tasks heavily reliant on temporal reasoning.

FAVOR-Bench: FAVOR-Bench focuses on the perception of micro-motion dynamics in videos, such as subtle camera motion (CM) or non-subject environmental changes (NSM). These fine-grained motion signals are often transient and sparse in the temporal domain, making them eas-

TABLE VI

<span id="page-15-1"></span><span id="page-15-0"></span>DETAILED PERFORMANCE COMPARISON ON THE OVOBENCH DATASET ACROSS VARIOUS REAL-TIME VISUAL PERCEPTION SUB-TASKS. WE REPORT THE EVALUATION METRIC (*e.g.*, ACCURACY %) FOR BOTH THE STANDARD BASE MODELS AND OUR PROPOSED CURVESTREAM. THE "↑" DENOTES THE ABSOLUTE PERFORMANCE GAIN ACHIEVED BY INTEGRATING OUR CURVATURE-AWARE MEMORY MANAGEMENT INTO THE RESPECTIVE BASE MODELS. BEST RESULTS ARE HIGHLIGHTED IN BOLD.

| Model                                   | Frame                    | OCR   | ACR   | ATR   | STU   | FPD   | OJR   | Avg.                  |  |  |
|-----------------------------------------|--------------------------|-------|-------|-------|-------|-------|-------|-----------------------|--|--|
| Human                                   | -                        | 93.96 | 92.57 | 94.83 | 92.70 | 91.09 | 94.02 | 93.20                 |  |  |
| Proprietary MLLMs                       |                          |       |       |       |       |       |       |                       |  |  |
| Gemini 1.5 Pro [41]                     | 1 fps                    | 85.91 | 66.97 | 79.31 | 58.43 | 63.37 | 61.96 | 69.32                 |  |  |
| GPT-4o [42]                             | 64                       | 69.80 | 64.22 | 71.55 | 51.12 | 70.30 | 59.78 | 64.46                 |  |  |
| Open-source Offline MLLMs               |                          |       |       |       |       |       |       |                       |  |  |
| LLaVA-Video-7B [23]                     | 64                       | 69.80 | 59.63 | 66.38 | 50.56 | 72.28 | 61.41 | 63.34                 |  |  |
| Qwen2-VL-7B [2]                         | 64                       | 69.13 | 53.21 | 63.79 | 50.56 | 66.34 | 60.87 | 60.65                 |  |  |
| InternVL2-8B [43]                       | 64                       | 68.46 | 58.72 | 68.97 | 44.94 | 67.33 | 55.98 | 60.73                 |  |  |
| LongVU-7B                               | 1 fps                    | 55.70 | 49.54 | 59.48 | 48.31 | 68.32 | 63.04 | 57.40                 |  |  |
|                                         | Open-source Online MLLMs |       |       |       |       |       |       |                       |  |  |
| VideoLLM-online-8B [44]                 | 2 fps                    | 8.05  | 23.85 | 12.07 | 14.04 | 45.54 | 21.20 | 20.79                 |  |  |
| Flash-VStream-7B [36]                   | 1 fps                    | 25.50 | 32.11 | 29.31 | 33.71 | 29.70 | 28.80 | 29.86                 |  |  |
| Dispider-7B [45]                        | 1 fps                    | 57.72 | 49.54 | 62.07 | 44.94 | 61.39 | 51.63 | 54.55                 |  |  |
| TimeChat-Online-7B [46]                 | 1 fps                    | 75.20 | 46.80 | 70.70 | 47.80 | 69.30 | 61.40 | 61.90                 |  |  |
| StreamForest-7B [38]                    | 1 fps                    | 68.46 | 53.21 | 71.55 | 47.75 | 65.35 | 60.87 | 61.20                 |  |  |
| Training-free Offline-to-Online Methods |                          |       |       |       |       |       |       |                       |  |  |
| LLaVA-OV-7B [23]                        | 32                       | 67.79 | 55.05 | 72.41 | 48.31 | 72.28 | 62.50 | 63.06                 |  |  |
| + ReKV [17]                             | 0.5 fps                  | 52.35 | 54.13 | 69.83 | 43.26 | 67.33 | 57.07 | 57.33                 |  |  |
| + HERMES [14]                           | 0.5 fps                  | 72.48 | 62.39 | 74.14 | 50.56 | 73.27 | 65.22 | 66.34                 |  |  |
| + Ours (CurveStream)                    | 10-20                    | 84.56 | 66.97 | 77.59 | 53.93 | 74.26 | 70.65 | (↑<br>70.57<br>7.51)  |  |  |
| Qwen2-VL-7B [2]                         | 1 fps                    | 69.13 | 53.21 | 63.79 | 50.56 | 66.34 | 60.87 | 60.65                 |  |  |
| + Freshmem [11]                         | 1 fps                    | 77.18 | 60.55 | 70.69 | 56.74 | 63.37 | 70.65 | 66.67                 |  |  |
| + Ours (CurveStream)                    | 10-20                    | 86.58 | 73.29 | 79.31 | 48.31 | 70.30 | 72.83 | (↑<br>70.73<br>10.08) |  |  |
| Qwen2.5-VL-7B [3]                       | 1 fps                    | 67.79 | 55.05 | 67.24 | 42.13 | 66.34 | 60.87 | 59.90                 |  |  |
| + HERMES [14]                           | 0.5 fps                  | 85.23 | 64.22 | 71.55 | 53.37 | 74.26 | 65.22 | 68.98                 |  |  |
| + Ours (CurveStream)                    | 10-20                    | 87.25 | 70.64 | 79.31 | 57.87 | 76.24 | 73.91 | 73.48<br>(↑<br>13.58) |  |  |
| Qwen3-VL-8B [1]                         | 1 fps                    | 71.14 | 65.14 | 75.86 | 64.61 | 75.25 | 70.65 | 70.10                 |  |  |
| + Ours (CurveStream)                    | 10-20                    | 93.96 | 82.57 | 83.62 | 68.54 | 78.22 | 80.43 | (↑<br>80.76<br>10.66) |  |  |

ily overlooked in conventional downsampling. CurveStream's Curvature-Aware Scorer (CAS) and dynamic threshold mechanism adeptly address this challenge: it can capture local curvature fluctuations triggered by micro-kinematic changes and maximally extract these motion details into the working memory. This capability to account for local high-frequency motions (Clear Memory) while preserving the global macroscopic view (Blurred Memory) indicates that curvature-driven memory management is equally a viable strategy in offline video understanding.

## APPENDIX F EXPERIMENTAL HYPERPARAMETERS

In this section, we detail the core inference hyperparameters used to evaluate the CurveStream framework, as summarized in Table [XI.](#page-17-1) Since our approach is entirely training-free, these parameters strictly govern the online memory scheduling policy during the inference phase. Specifically, we set the maximum visual memory capacity (Queue Size) to 20 frames to simulate stringent memory constraints. For the Curvature-Aware Scorer (CAS), the geometric penalty weight λ is configured to 0.2 to optimally balance first-order motion and second-order curvature. Within the Hierarchical Visual Memory Management (HVMM) module, the K-Sigma dynamic dual thresholds are defined by k<sup>1</sup> = 0.0 and k<sup>2</sup> = 1.0, enabling

<span id="page-16-1"></span><span id="page-16-0"></span>TABLE VII

DETAILED PERFORMANCE COMPARISON ON THE **FAVORBENCH** DATASET. "↑" INDICATES THE PERFORMANCE IMPROVEMENT OF OUR METHOD COMPARED TO THE BASE MODEL.

| Model                         | Frame   | AS    | HAC   | SAD   | MAD   | CM    | NSM   | Avg.                           |
|-------------------------------|---------|-------|-------|-------|-------|-------|-------|--------------------------------|
| Proprietary MLLMs             |         |       |       |       |       |       |       |                                |
| Gemini-1.5-Pro [41]           | 1 fps*  | 49.22 | 53.73 | 48.80 | 54.85 | 41.58 | 56.25 | 49.87                          |
| GPT-4o [42]                   | 1 fps*  | 40.65 | 45.10 | 42.84 | 45.48 | 36.00 | 48.44 | 42.09                          |
| Claude-3.7-Sonnet [50]        | 1 fps*  | 45.20 | 43.02 | 41.82 | 48.05 | 39.07 | 46.88 | 43.73                          |
| Open-source MLLMs             |         |       |       |       |       |       |       |                                |
| Video-LLaVA-7B [51]           | 8 frms  | 24.91 | 21.54 | 25.45 | 30.54 | 26.23 | 21.88 | 25.37                          |
| LLaVA-NeXT-Video-7B [5]       | 8 frms  | 21.27 | 22.45 | 26.05 | 26.72 | 23.07 | 14.06 | 23.45                          |
| LLaVA-NeXT-Video-34B [5]      | 8 frms  | 31.70 | 31.99 | 32.31 | 22.99 | 29.58 | 46.88 | 30.44                          |
| Tarsier-7B [52]               | 8 frms  | 12.55 | 21.16 | 17.87 | 17.93 | 22.23 | 31.25 | 17.46                          |
| Tarsier-34B [52]              | 8 frms  | 28.56 | 34.98 | 26.90 | 31.29 | 31.91 | 37.50 | 30.34                          |
| LLaVA-Video-7B-Qwen2 [23]     | 64 frms | 36.14 | 41.27 | 41.28 | 44.48 | 29.58 | 46.88 | 38.60                          |
| LLaVA-Video-72B-Qwen2 [23]    | 64 frms | 48.35 | 47.50 | 45.25 | 51.70 | 33.02 | 53.12 | 46.08                          |
| InternVL2.5-2B [53]           | 8 frms  | 18.70 | 28.23 | 23.71 | 27.47 | 19.16 | 23.44 | 22.90                          |
| InternVL2.5-8B [53]           | 8 frms  | 31.97 | 38.68 | 38.09 | 37.76 | 26.14 | 35.94 | 34.59                          |
| InternVL2.5-78B [53]          | 8 frms  | 38.38 | 40.62 | 39.05 | 43.65 | 29.40 | 39.06 | 38.54                          |
| VideoChat-Flash-Qwen2-7B [54] | 1 fps   | 41.90 | 48.41 | 42.84 | 50.95 | 35.07 | 50.00 | 43.82                          |
| VideoLLaMA3-2B [55]           | 1 fps   | 28.97 | 36.60 | 34.90 | 38.01 | 28.56 | 40.62 | 32.98                          |
| VideoLLaMA3-7B [55]           | 1 fps   | 40.20 | 44.13 | 42.42 | 48.30 | 31.53 | 42.19 | 41.46                          |
| Qwen2.5-VL-3B [3]             | 1 fps   | 38.45 | 38.22 | 36.64 | 39.75 | 29.77 | 32.81 | 37.05                          |
| Qwen2.5-VL-7B [3]             | 1 fps   | 39.48 | 43.28 | 43.14 | 43.65 | 33.49 | 39.06 | 40.76                          |
| + Ours (CurveStream)          | 10-20   | 48.20 | 51.59 | 47.59 | 53.94 | 30.88 | 51.56 | <b>47.32</b> ( <b>† 6.56</b> ) |

<span id="page-16-2"></span>TABLE VIII

DETAILED PERFORMANCE COMPARISON ON THE **MVBENCH** DATASET ACROSS 19 FINE-GRAINED SUB-TASKS. DUE TO SPACE CONSTRAINTS, THE RESULTS ARE SPLIT INTO TWO BLOCKS. "↑" INDICATES THE PERFORMANCE IMPROVEMENT OF OUR METHOD.

| Model                | Avg.                   | Action<br>Antonym | Action<br>Count | Episodic<br>Reasoning | Action<br>Localization | Action<br>Prediction | Action<br>Sequence | Character<br>Order | Counterfactual<br>Inference | Egocentric<br>Navigation |
|----------------------|------------------------|-------------------|-----------------|-----------------------|------------------------|----------------------|--------------------|--------------------|-----------------------------|--------------------------|
| Qwen3-VL-8B          | 60.17                  | 84.00             | 37.50           | 51.50                 | 34.50                  | 57.49                | 65.95              | 61.50              | 65.50                       | 38.00                    |
| + Ours (CurveStream) | 63.60 († <b>3.43</b> ) | 68.50             | 50.50           | 54.00                 | 39.50                  | 79.00                | 70.50              | 77.50              | 60.50                       | 37.50                    |
|                      |                        |                   |                 |                       |                        |                      |                    |                    |                             |                          |
| Model                | Fine-grained           | Moving            | Moving          | Moving                | Object                 | Object               | Object             | Scene              | State                       | Unexpected               |
| Model                | Action                 | Attribute         | Count           | Direction             | n Existence            | Interaction          | n Shuffle          | Transition         | n Change                    | Action                   |
| Qwen3-VL-8B          | 43.50                  | 85.00             | 63.00           | 64.00                 | 80.80                  | 64.00                | 39.00              | 81.00              | 50.50                       | 76.50                    |
| + Ours (CurveStream) | 48.50                  | 82.00             | 60.50           | 50.00                 | 81.00                  | 74.00                | 39.00              | 90.00              | 63.50                       | 82.50                    |

<span id="page-16-3"></span>TABLE IX

ABLATION STUDY ON **STREAMINGBENCH**. WE EVALUATE THE INDIVIDUAL AND COMBINED EFFECTS OF CAS AND HVMM. RED ARROWS SPECIFICALLY DENOTE THE ABSOLUTE AVERAGE PERFORMANCE IMPROVEMENTS ACHIEVED OVER THE RESPECTIVE BASE MODELS.

| Model Configuration | CAS          | HVMM         | OP    | CR    | CS    | ATP   | EU    | TR    | PR    | SU    | ACP   | CT    | Avg.                       |
|---------------------|--------------|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------------------------|
| Qwen2-VL-7B [2]     |              |              | 77.38 | 76.56 | 73.19 | 75.08 | 75.00 | 67.91 | 73.15 | 65.04 | 66.57 | 35.75 | 69.04                      |
| w/ CAS              | $\checkmark$ |              | 85.29 | 80.47 | 89.59 | 87.58 | 74.53 | 82.24 | 76.85 | 70.73 | 73.94 | 43.52 | 78.16 ( <del>†</del> 9.12) |
| w/ HVMM             |              | $\checkmark$ | 87.19 | 76.56 | 89.59 | 86.93 | 75.16 | 86.92 | 76.85 | 70.33 | 74.50 | 43.01 | 78.80 ( <b>†9.76</b> )     |
| CurveStream         | ✓            | ✓            | 88.56 | 77.34 | 88.61 | 89.84 | 76.25 | 92.52 | 76.85 | 76.83 | 76.70 | 45.31 | 81.04 (†12.00)             |

the adaptive routing of incoming tokens. Furthermore, to effectively compress transitional observations, frames assigned to Blurred Memory are uniformly downsampled to a target spatial resolution of 224 (TRANSITION\_SIZE).

## APPENDIX G ABLATION STUDY

To thoroughly evaluate the independent contributions and synergistic effects of the core components within the Cur-

<span id="page-17-2"></span><span id="page-17-0"></span>TABLE X
ABLATION STUDY ON **OVO-BENCH**. WE REPORT THE PERFORMANCE ACROSS VARIOUS REAL-TIME VISUAL PERCEPTION SUB-TASKS TO VALIDATE THE SYNERGISTIC EFFECT BETWEEN THE PROPOSED MEMORY MODULES.

| Model Configuration | CAS          | HVMM         | OCR   | ACR   | ATR   | STU   | FPD   | OJR   | Avg.                          |
|---------------------|--------------|--------------|-------|-------|-------|-------|-------|-------|-------------------------------|
| Qwen-3VL-8B [1]     |              |              | 71.14 | 65.14 | 75.86 | 64.61 | 75.25 | 70.65 | 70.10                         |
| w/ CAS              | $\checkmark$ |              | 87.92 | 83.49 | 81.90 | 64.04 | 76.24 | 80.98 | 78.49 (†8.39)                 |
| w/ HVMM             |              | $\checkmark$ | 87.25 | 71.56 | 78.45 | 66.29 | 74.26 | 72.83 | 74.79 ( <del>†</del> 4.69)    |
| CurveStream         | $\checkmark$ | $\checkmark$ | 93.96 | 82.57 | 83.62 | 68.54 | 78.22 | 80.43 | <b>80.76</b> ( <b>10.66</b> ) |

<span id="page-17-1"></span>TABLE XI
DETAILED CORE EXPERIMENTAL HYPERPARAMETERS UTILIZED BY THE
CURVESTREAM FRAMEWORK THROUGHOUT THE ENTIRE INFERENCE
PHASE.

| Hyperparameter                     | Value |  |  |  |
|------------------------------------|-------|--|--|--|
| Queue Size $(N_{max})$             | 20    |  |  |  |
| Curvature score weight $(\lambda)$ | 0.2   |  |  |  |
| TRANSITION_SIZE                    | 224   |  |  |  |
| $K_SIGMA_TRANS(k_1)$               | 0.0   |  |  |  |
| K_SIGMA_KEY $(k_2)$                | 1.0   |  |  |  |

veStream architecture, we conducted comprehensive ablation studies on StreamingBench (**Table IX**) and OVO-Bench (**Table X**). Using the passive uniform sampling and FIFO cache of the base model as our baseline, we independently verified the effectiveness of the Curvature-Aware Scorer (CAS) and the Hierarchical Visual Memory Management (HVMM). The experimental results not only validate the performance gains from each individual module but also reveal a significant non-linear synergistic amplification effect when they are combined.

#### A. Effectiveness of CAS: Enhancing Semantic Perception

The integration of the CAS module alone yields average performance improvements of 9.12% and 8.39% on StreamingBench and OVO-Bench, respectively. This significant improvement validates the sensitivity of feature manifold curvature in capturing "Semantic Transitions" within videos. The uniform sampling of traditional base models lacks content awareness, making it prone to missing transient key actions. By evaluating the local curvature in the feature space, the CAS module endows the model with the ability to actively assess information density. Particularly in the real-time dynamic tasks of OVO-Bench (where the gain reaches 18.35% in ACR), CAS successfully locates the curvature peaks triggered by action fluctuations. This demonstrates that using feature manifold curvature as a semantic signal effectively compensates for the omission of key frames caused by the "content-unaware" nature of uniform sampling in dynamic scenes.

## B. Effectiveness of HVMM: Alleviating Forgetting

When only the HVMM module is introduced (i.e., operating without CAS dynamic scoring, degrading to uniform sampling with alternate allocation to Clear and Blurred Memory), the

model achieves stable improvements of 9.76% and 4.69% across the two datasets, respectively. This result indicates that the hierarchical memory architecture inherently possesses advantages in processing long sequences. When facing memory bottlenecks, the FIFO mechanism of base models easily evicts historical features, leading to context truncation. In contrast, HVMM constructs a decoupled binary structure of Clear Memory and Blurred Memory. Without increasing the overall token budget, it leverages the high compression ratio of Blurred Memory to broaden the model's historical context, thereby providing robust architectural support for complex contextual tasks that rely on long-range temporal reasoning.

## C. Synergistic Effect of Perception and Scheduling Loop Mod-

When CAS and HVMM operate jointly (i.e., the complete CurveStream architecture), the model experiences a comprehensive performance leap, with total gains reaching 12.04% and 10.66% on StreamingBench and OVO-Bench, respectively. More importantly, this combined gain significantly exceeds the sum of the individual modules' improvements (e.g., 3.93% > -0.57% + 1.68% in the STU task of the OVO-Bench). This non-linear synergistic amplification profoundly reveals the complementarity of the underlying design of the CurveStream architecture: CAS provides precise "Semantic Awareness," while the HVMM module is responsible for executing the adaptive "Memory Scheduling" strategy."

Without HVMM, the highly dynamic key frames located by CAS might eventually be gradually evicted due to memory capacity constraints. Conversely, without CAS, the alternate allocation of HVMM lacks adaptive perception of the video content, easily degrading into rigid structural segmentation. When the two are combined, CAS is responsible for marking high-curvature transition points across the global temporal axis, while HVMM stores these high-value nodes into Clear Memory and smoothly compresses low-curvature static periods into Blurred Memory. Together, they construct a compact and coherent causal topological chain for the large model, significantly broadening its cognitive boundaries in infinitely long streaming videos.