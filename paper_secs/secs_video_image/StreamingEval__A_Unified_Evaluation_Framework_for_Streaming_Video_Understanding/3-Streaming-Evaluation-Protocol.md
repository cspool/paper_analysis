# 3 Streaming Evaluation Protocol

In this subsection, we present StreamingEval in detail. We first provide a rigorous definition of streaming online inference, and then describe the concrete implementation pipeline of the evaluation framework. Next, we explain how to construct fair and reproducible comparison settings for both offline and online models. Finally, we introduce a set of evaluation metrics that are critical for online inference in streaming scenarios.

#### 3.1 Online Task Definition

We define *streaming online question answering* as a real-time, interactive multimodal dialogue setting in which the model continuously receives video frames along the temporal axis. At any time t, the input may consist of three components: (1) the interaction history between the user and the model up to time t, denoted as C<sup>t</sup> ; (2) the video frame acquired at time t, denoted as V<sup>t</sup> ; and (3) the user query issued at time t, denoted as Q<sup>t</sup> . When the encoded query arrives at the model at time t1, the backbone language model f autoregressively generates a response Rt<sup>1</sup> conditioned on Ct<sup>1</sup> , Venc[0,t1] , Qt<sup>1</sup> , i.e., by maximizing the conditional probability p Rt<sup>1</sup> | Ct<sup>1</sup> , V[0,t1] , Qt<sup>1</sup> . Here, Venc[0,t1] denotes all video frames received up to time t<sup>1</sup> that have already been encoded by the model. This setting emphasizes the model's ability to perform inference and respond at arbitrary time points.

#### 3.2 StreamingEval Framework

To enable streaming online inference, we implement the execution backend of StreamingEval as an asynchronous, time-causal pipeline composed of three decoupled processes that run in parallel, the framework is illustrated in Figure [2.](#page-3-0) Specifically, it consists of a Frame Player, an Encoderand-Memory Updater, and a Responder. These processes communicate via inter-process queues and/or shared buffers, emulating the behavior of an online system in which video frames arrive continuously, the model updates continually, and user queries may occur at any time, without introducing additional synchronization-induced blocking [\(Ma](#page-9-12) [et al.,](#page-9-12) [2021\)](#page-9-12).

Input Stream and Frame Player. For an arbitrary video stream, we represent it as {(v<sup>i</sup> , τi)}<sup>∞</sup> <sup>i</sup>=1, where v<sup>i</sup> denotes the i-th frame and τ<sup>i</sup> denotes its arrival time. The frame player samples frames at a fixed interval ρ and streams them to downstream processes.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Overview of the StreamingEval framework. The framework standardizes streaming video understanding by modeling continuous input ingestion, incremental visual memory updates, and query-driven inference within a unified protocol.

Encoder & Memory Updater. The encoder first maps each incoming frame to a visual representation:

$$z_i = g_{\theta}(v_i), \tag{1}$$

where g<sup>θ</sup> denotes the visual encoding backbone. The memory updater then maintains an online memory state M. For online models, the memory is updated according to the model-specific update rule U by writing the new representation into memory:

$$M_{\tau_i^+} = \mathcal{U}\left(M_{\tau_i^-}, z_i; B, \pi\right),$$
 (2)

where τ − i and τ + i denote the memory states immediately before and after the arrival of the i-th frame at time τ<sup>i</sup> , respectively. B specifies the memory budget (capacity constraint), and π denotes the corresponding write/eviction policy. For offline models, we apply a projection layer to map visual features into the embedding space aligned with the language model, and then store them in a fixedlength visual context window maintained with a first-in-first-out (FIFO) policy.

Responder. A user may launch a query qt<sup>0</sup> at any time t0. Once the query is triggered, the responder first encodes it, and denotes the time when encoding finishes as t1. The responder then reads

the memory snapshot available at time t1, denoted by Mt<sup>1</sup> , and conditions on the dialogue context Ct<sup>1</sup> together with the query qt<sup>0</sup> to autoregressively generate an answer:

$$R_{t_1} \sim p_{\phi}(\cdot \mid q_{t_0}, C_{t_1}, M_{t_1}),$$
 (3)

where Ct<sup>1</sup> represents the interaction history up to time t1, Mt<sup>1</sup> denotes the information updated up to time t1, and p<sup>ϕ</sup> is parameterized by the backbone language model.

#### 3.3 Comparable Online/Offline Setups

In streaming settings, a fair comparison between native online models and general multimodal models is not straightforward: multimodal models typically assume access to the full video, whereas online models must operate under strict causal constraints and rely only on historical information that has already arrived and been processed. Moreover, visual token embeddings differ in dimensionality across models; therefore, constraining the history length solely by the *number* of tokens can yield inconsistent actual memory footprints. To address this, StreamingEval adopts a "two settings, one unified budget" strategy: we preserve the native mechanisms of online models as much as possible, while introducing a resource-constrained offline

adapter for multimodal models that constructs context under the same resource budget, enabling fair comparisons.

Native Online-model Setting. For native online models, we follow the original online mechanisms and configurations in their papers as closely as possible, including incremental encoding, memory/state updates, retrieval policies, and default input resolution and preprocessing. Models run within our multi-process emulator: frames arrive continuously at a fixed frame rate, the model updates its memory on the fly, and when a query is issued at time t, the responder can only access the memory snapshot that is *available up to* t (i.e., processed before t). This ensures strict causality while staying faithful to each model's intended design.

Multimodal-model Adapter Setting. For multimodal models, we introduce a unified boundedmemory adapter. As video frames arrive, each model produces visual representations in its native manner; after a projection layer aligns them to the language model embedding space, the resulting representations are written into a fixed-capacity memory bank. When the memory bank exceeds the budget, we evict the oldest content using a deterministic FIFO policy. When a query is issued, we concatenate the current memory bank with the dialogue context as the model input, thereby simulating a deployable version of multimodal models under strict online constraints.

For offline models, we adopt a fixed-capacity memory bank with a FIFO eviction policy to enable as neutral and reproducible an evaluation as possible under realistic streaming constraints, while avoiding the additional algorithmic gains introduced by compression or summarization modules that could compromise fairness. At the same time, StreamingEval is open to different memorymanagement strategies: methods such as clusteringbased compression, learned summarization, and KV compression can all be readily plugged in, and their effects will be naturally reflected in accuracy, latency, memory usage, and the final StreamingScore.

Unified Resource Budget. To avoid unfair comparisons where different models incur different GPU memory footprints under the same number of visual tokens due to embedding-dimensionality mismatch, we enforce a byte-level resource budget and cap the storage of historical context at

M. We account only for the two components that scale with context length: the projected visualtoken representations cached in the memory bank, and the language Transformer KV cache associated with these visual tokens for incremental inference. [\(Kwon et al.,](#page-9-13) [2023;](#page-9-13) [Dao et al.,](#page-9-14) [2022\)](#page-9-14)Implementation details of the computations are provided in the appendix.

