# PEARL: Personalized Streaming Video Understanding Model

Yuanhong Zheng1<sup>∗</sup> , Ruichuan An1<sup>∗</sup> , Xiaopeng Lin<sup>1</sup> , Yuxing Liu<sup>1</sup> , Sihan Yang<sup>1</sup> , Huanyu Zhang<sup>3</sup> , Haodong Li<sup>4</sup> , Qintong Zhang<sup>1</sup> , Renrui Zhang<sup>5</sup> , Guopeng Li<sup>4</sup> , YiFan Zhang<sup>3</sup>† , Yuheng Li<sup>2</sup>#, and Wentao Zhang<sup>1</sup>,6#

<sup>1</sup> Peking University <sup>2</sup> Adobe <sup>3</sup> CASIA <sup>4</sup> Stepfun <sup>5</sup> CUHK <sup>6</sup> Zhongguancun Academy

Abstract. Human cognition of new concepts is inherently a streaming process: we continuously recognize new objects or identities and update our memories over time. However, current multimodal personalization methods are largely limited to static images or offline videos. This disconnects continuous visual input from instant real-world feedback, limiting their ability to provide the real-time, interactive personalized responses essential for future AI assistants. To bridge this gap, we first propose and formally define the novel task of Personalized Streaming Video Understanding (PSVU). To facilitate research in this new direction, we introduce PEARL-Bench, the first comprehensive benchmark designed specifically to evaluate this challenging setting. It evaluates a model's ability to respond to personalized concepts at exact timestamps under two modes: (1) Frame-level, focusing on a specific person or object in discrete frames, and (2) a novel Video-level, focusing on personalized actions unfolding across continuous frames. PEARL-Bench comprises 132 unique videos and 2,173 fine-grained annotations with precise timestamps. Concept diversity and annotation quality are strictly ensured through a combined pipeline of automated generation and human verification. To tackle this challenging new setting, we further propose PEARL, a plug-and-play, training-free strategy that serves as a strong baseline. Extensive evaluations across 8 offline and online models demonstrate that PEARL achieves state-of-the-art performance. Notably, it brings consistent PSVU improvements when applied to 3 distinct architectures, proving to be a highly effective and robust strategy. We hope this work advances vision-language model (VLM) personalization and inspires further research into streaming personalized AI assistants. Code is available at <https://github.com/Yuanhong-Zheng/PEARL>.

Keywords: Personalized model · Streaming video understanding

# 1 Introduction

Recent advancements in Vision-Language Models (VLMs) [\[1–](#page-14-0)[6\]](#page-14-1) have remarkably expanded the boundaries of multimodal understanding, empowering models to recognize and interact with personalized, user-specific concepts. Despite

<sup>\*</sup> Equal contribution. † Project leader. # Corresponding authors.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Fig. 1: Comparison between our proposed Personalized Streaming Video Understanding (PSVU) task and traditional Personalized Image/Video Understanding tasks. Unlike traditional settings, PSVU better aligns with real-world scenarios by featuring continuous streaming video inputs, dynamically defined concepts, and multi-turn conversations regarding these customized concepts.

these strides, current personalization methods [\[7–](#page-14-2)[15\]](#page-15-0) remain fundamentally constrained. As shown in Fig. [1,](#page-1-0) approaches such as Yo'LLaVA [\[8\]](#page-14-3) and MC-LLaVA [\[7\]](#page-14-2) are mainly designed for static image-text tasks. Furthermore, while PVChat [\[10\]](#page-14-4) pioneers personalized video understanding, it operates strictly in offline settings and only supports single turn interaction, failing to accommodate the openended, streaming nature of real-world environments.

In contrast, humans continuously recognize new individuals and objects, forming memories over time as they process the world as a seamless visual stream. This fundamental cognitive mechanism highlights a critical limitation of existing methods, which remain confined to static images or pre-recorded videos. Bridging this gap is not merely a technical step, but an essential prerequisite for the next generation of personalized AI assistants [\[16\]](#page-15-1): such systems must be capable of handling streaming visual inputs and delivering real-time, interactive, and personalized responses in dynamic real-world environments [\[17\]](#page-15-2). For instance, in customized fitness coaching (Fig. [1\)](#page-1-0), an AI assistant must continuously monitor a user's specific weightlifting actions across an video stream to provide instant, tailored form correction. This real-time, streaming personalization capability is indispensable for deploying truly practical AI assistants.

To bridge this gap, we firstly propose and formally define the novel task of Personalized Streaming Video Understanding (PSVU). To facilitate research in this new direction, we introduce PEARL-Bench, the first comprehensive benchmark specifically designed to evaluate personalized streaming video understanding. Unlike traditional offline tasks, PEARL-Bench distinguishes itself through two core properties: (1) Continuous Temporal Precision, requiring models to localize and reason about personalized concepts at exact timestamps within an ongoing stream; and (2) Interactive Concept Definition, challenging models to grasp user-specific concepts dynamically defined on-the-fly, rather than relying on predefined pools. To thoroughly assess model capabilities, the benchmark evaluates two modes: (1) Frame-level Personalization, focusing on the continuous recognition and reasoning of a specific person or object appearing across discrete frames; and (2) a novel Video-level Personalization, which goes beyond static appearances to focus on specific, customized actions that unfold across continuous frames. PEARL-Bench comprises 132 unique videos, and 2,173 finegrained annotations with precise timestamps. The video data is carefully curated through a combination of expert manual collection and rigorous programmatic synthesis pipelines. By sourcing data from diverse domains including anime, movies, reality shows, and digital humans, we ensure both extensive concept diversity and high annotation quality.

The challenging PSVU task presents significant efficiency and architectural hurdles for existing models, as they struggle to maintain streaming visual context and instantly acquire new concepts without computationally expensive retraining. To tackle this, we propose PEARL, a training-free plug-and-play framework designed to serve as a strong baseline. Specifically, PEARL features a Dual-grained Memory System that explicitly decouples concept-centric knowledge from stream-centric observations, incrementally archiving continuous video clips while dynamically registering user-defined concepts. To ensure fast and accurate response, we further introduce a Concept-aware Retrieval Algorithm that leverages stored concept descriptions to precisely retrieve relevant historical visual evidence. Consequently, without any parameter updates, PEARL seamlessly empowers off-the-shelf VLMs to deliver real-time, personalized responses in continuous video streams. Extensive evaluations demonstrate that PEARL establishes a new state-of-the-art among 8 offline and online models. Notably, equipped with PEARL drives consistent improvements across 3 distinct architectures, yielding an average performance gain of 13.79% at the frame-level and 12.80% at the video-level, thereby proving its effectiveness and robustness.

We summarize our contributions as follows:

- New Task and Benchmark: We are the first to propose and formally define the novel task of Personalized Streaming Video Understanding. To facilitate evaluation in this direction, we introduce PEARL-Bench, the first comprehensive benchmark specifically designed for this challenging setting.
- Novel Framework: We propose PEARL, a novel, training-free, and plugand-play method. By seamlessly integrating into existing models, it demonstrates remarkable effectiveness and robustness across multiple architectures.
- State-of-the-Art Performance: Extensive experiments show that PEARL achieves state-of-the-art results compared to 8 offline and online video un-

derstanding methods. We hope this work inspires the field of VLM personalization and paves the way for next-generation interactive AI assistants.

# 2 Related Works

Personalized VLMs. As the capabilities of Vision-Language Models (VLMs) continue to advance [\[1–](#page-14-0)[6,](#page-14-1) [18–](#page-15-3)[20\]](#page-15-4), growing attention has been increasingly directed toward unleashing their potential to serve as personalized AI assistants [\[21–](#page-15-5) [25\]](#page-15-6). Existing VLM personalization efforts can be broadly categorized into three areas: personalized image understanding, combined with personalized generation and personalized video understanding. Previous research has predominantly focused on personalized image understanding. The paradigm can be summarized as finetune-based [\[7,](#page-14-2) [8,](#page-14-3) [14,](#page-15-7) [26\]](#page-16-0), Retrieval-Augmented Generation (RAGbased [\[9,](#page-14-5) [11\]](#page-14-6)) and reinforcement learning [\[27,](#page-16-1) [28\]](#page-16-2). However, they are inherently limited to static images and fail to generalize to dynamic video domains. Parallel studies have also explored methods that unify personalized understanding and generation [\[12,](#page-14-7) [13,](#page-15-8) [29](#page-16-3)[–31\]](#page-16-4). Yet, these approaches heavily rely on pre-defined concepts, which contradicts the flexible nature of real-world user interactions. In the domain of personalized video understanding, early explorations [\[32\]](#page-16-5) were mostly restricted to personalized retrieval. While a recent work, PVChat [\[10\]](#page-14-4) pioneers to focus on personalized VQA but it is strictly designed for offline scenarios. Meanwhile, the emerging field of streaming video understanding [\[33–](#page-16-6)[42\]](#page-17-0) has made significant strides in processing continuous visual inputs for real-time interactions, yet these methods remain largely agnostic to user-defined concepts. Consequently, existing approaches still fall short of meeting the combined demands for real-time responding, streaming inputs, and flexible concept definition. To address these limitations, this paper introduces the novel task of Personalized Streaming Video Understanding (PSVU) for the first time. Furthermore, we propose PEARL, a training-free, plug-and-play framework designed to achieve highly efficient, instant concept registration and real-time inference within continuous video streams in real-world settings.

# 3 PEARL-Bench

#### 3.1 Task Definition

In the task of Personalized Streaming Video Understanding, a streaming video is processed as a continuous sequence of scenes. Throughout the stream, a user can dynamically introduce new concepts at any timestamp via instructions, forming an evolving set of user-defined concepts. For a subsequent query, the model must retrieve the relevant concepts and visual context to generate an accurate response. Specifically, as illustrated in Fig. [2,](#page-5-0) we define two types of concepts:

1. Frame-level Concepts: Static entities registered from a single frame. For example, defining a specific person or object at any timestamp.

2. Video-level Concepts: Dynamic actions unfolding over a continuous clip. For instance, defining a personalized gesture or a series of special actions.

Based on their temporal and functional requirements, we also categorize the queries into three types:

- 1. **Concept-Definition QA:** Introduces new concepts at specific timestamps. The model registers the concept into memory based on the current scene.
- Real-Time QA: Queries established concepts at the immediate moment.
   The model grounds its response purely on the present scene, evaluating its proficiency in answering real-time questions without historical distraction.
- 3. Past-Time QA: Inquires about the historical states or activities of established concepts. The model must retrieve relevant historical sequences, requiring long-term temporal reasoning and precise evidence retrieval.

The task is inherently multi-turn, enabling flexible concept definitions and queries regarding established concepts at arbitrary future time steps. This interactive format lays the foundation for the next generation of persoanlized AI assistants.

#### 3.2 Benchmark Overview

Existing personalized benchmarks suffer from notable limitations and are largely disconnected from real-world scenarios, as shown in Table 1. MyVLM [26], Yo'LLaVA [8], MC-LLaVA [7], UnifyBench [12] and MMPB [15] are all image-based, supporting neither video input nor streaming scenarios, and lacking multiturn interaction. PVChat [10] and This-isMy [32] introduces video modality but is limited to short offline videos (each video is shorter than 5 seconds), with no support for streaming or multi-turn concept interaction. Moreover, none of the above benchmarks supports Video-level personalization, *i.e.*, recognizing personalized concepts defined by continuous actions unfolding across frames. PEARL-Bench is the first benchmark to simultaneously support long-form streaming video input, multi-turn concept interaction, and both Frame-level and Video-level personalized concept types. As shown in Table 2, PEARL-Bench comprises 132 videos and 2,173 annotations in total, with an average duration of 1,458 seconds per video. All annotations are associated with precise timestamps.

<span id="page-4-0"></span>Table 1: Comparison of PEARL-Bench with ex- Table 2: Data Statistics isting personalized benchmarks.

Of PEARL-Bench.

| Benchmark       | Modality      | Strooming | g Multi-turn | Concept Type |             | Multi-Concept  |
|-----------------|---------------|-----------|--------------|--------------|-------------|----------------|
| Dencimark       | Modanty       | Streaming |              | Frame-level  | Video-level | Withti-Concept |
| MyVLM [26]      | Image         | -         | Х            | /            | Х           | Х              |
| Yo'LLaVA [8]    | Image         | _         | ×            | /            | X           | X              |
| MC-LLaVA [7]    | Image         | -         | ×            | /            | ×           | /              |
| UnifyBench [12] | Image         | -         | ×            | /            | ×           | ×              |
| MMPB [15]       | Image         | -         | ×            | ✓            | ×           | ✓              |
| PVChat [10]     | Video (short) | Х         | Х            | /            | Х           | <b>✓</b>       |
| This-is-My [32] | Video (short) | Х         | ×            | /            | ×           | /              |
| PEARL-Bench     | Video (long)  | /         | /            | /            | /           | <b>√</b>       |

|                   | Frame-level | Video-level | Total |
|-------------------|-------------|-------------|-------|
| #Videos           | 112         | 20          | 132   |
| Avg. Duration (s) | 1,657       | 303         | 1,458 |
| #Concept-Def QA   | 418         | 80          | 498   |
| #Real-Time QA     | 922         | 359         | 1,281 |
| #Past-Time QA     | 394         | -           | 394   |
| #Total QA         | 1,734       | 439         | 2,173 |

# 3.3 Curation Pipeline

Our curation pipeline consists of four stages: video collection and filtering, followed by the annotation of three QA types (Concept-Definition, Real-Time, and

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Fig. 2: Overview of PEARL-Bench. (Left) Frame-level split: Concept-Definition QA registers a person at early timestamps; Real-Time QA queries the concept's current state in the scene; Past-Time QA requires retrieving a historical evidence clip to answer questions about prior states. (Right) Video-level split: Concept-Definition QA registers a personalized action observed in a clip; Real-Time QA asks whether the defined action is being performed at the current timestamp.

Past-Time), and concludes with a quality control phase. We employ a diverse set of question templates to annotate the three QA types. Representative examples are illustrated in Fig. [2,](#page-5-0) and complete templates are provided in the appendix.

Video Collection and Filtering We collect videos from publicly available internet sources and manually filter them according to the following criteria: (i) the video exhibits high dynamics and poses real-time understanding demands; (ii) the video contains multiple repeatedly appearing, clearly definable personalized concepts; and (iii) the video resolution is no lower than 480p. Videos in the frame-level split are drawn from diverse domains including anime, movies, and reality shows, ensuring variety in visual styles and concept types. For the video-level split, collecting videos with clean personalized action annotations from existing internet data is extremely challenging, as these action concepts must appear repeatedly and ideally be performed by different subjects within the video. We therefore adopt a digital human synthesis approach: we synthesize diverse videos using assets from Mixamo [\[43\]](#page-17-1) by randomly combining 8 distinct characters, 20 unique actions, and 20 background scenes to foster data diversity and visual richness, where each distinct action serves as a video-level concept.

Concept-Definition QA Annotation Concept-Definition QA is designed to register a new concept into the model's memory, and carries no specific groundtruth answer, which excludes it from the final evaluation: it suffices for the model to correctly identify and register the concept according to the user's instruction. Given a video, annotators first locate multiple timestamps at which the target concept appears in the scene, and pose a registration question at each such timestamp. For example, at t=5 minutes an annotator issues "This is XiaoJing." alongside the frame showing the target character, thereby registering XiaoJing as a new concept. Notably, to prevent the model from leveraging prior knowledge to recognize a specific concept, we collect 10k common names from the U.S. SSA database [\[44\]](#page-17-2) and use them to randomly replace the original concept names, thereby enhancing benchmark robustness. Previous research [\[7,](#page-14-2) [10\]](#page-14-4) discussed the rationale for this naming strategy.

Real-Time QA Annotation After completing concept definition annotation, annotators begin labeling Real-Time QA. Specifically, they identify timestamps in the video suitable for real-time questioning and pose concept-related questions with corresponding answers. The current clip, question, and answer are then fed to a strong VLM to generate multiple-choice distractors. For example, at t=20 minute an annotator poses "What is XiaoJing wearing now?", which requires the model to ground the recognized concept in the current scene to answer correctly. During annotation, questions that can be answered without any knowledge of the defined concepts are strictly excluded to ensure benchmark validity.

Past-Time QA Annotation Past-Time QA annotation likewise follows concept definition. The key distinction from Real-Time QA is that Past-Time QA cannot be answered from the current clip alone. It additionally requires a historical clip as evidence. Annotators therefore identify both a query timestamp and a corresponding historical evidence timestamp, and pose a question with its answer accordingly. For example, at t=40 minute with evidence at t=10 minute, the question "What was XiaoJing wearing when she was cooking?" can only be answered by retrieving the historical cooking scene, not from the current frame. The current clip, evidence clip, question, and answer are then jointly fed to a VLM to generate distractors. The constraint of this QA type is that correct answering must depend on retrieving and reasoning over historical evidence clips.

Quality Control To ensure the highest annotation quality, our curation team consists of 10 researchers, each with over a year of experience in multimodal research. Specifically, 6 members are dedicated to the primary annotation tasks, while the remaining 4 focus on rigorous review and quality control. Overall, we adopt a combined pipeline of automated filtering and human verification. In the automated stage, we apply an ablation-based filtering method with an experimental setup similar to Section [5.5.](#page-11-0) Specifically, for Real-Time QA, we test models with and without provided concepts; for Past-Time QA, we test with and without historical evidence clips. Questions that models can answer correctly even when the necessary information (i.e., concepts or historical evidence clips) is withheld are deemed trivial and therefore filtered. In the human verification stage, our reviewers conduct multiple rounds of manual inspection to verify that each QA item and its timestamp are accurately aligned with the video content. We additionally collect human evaluation scores as an upper-bound reference for benchmark performance, which are reported in Table [3.](#page-10-0)

# 4 PEARL Framework

To address the challenges of the task of PSVU, we propose a plug-and-play framework, PEARL. As illustrated in Fig. [3,](#page-8-0) it dynamically defines concepts at specific timestamps of streaming video via user instructions and provides realtime responses to user queries in subsequent timestamps.

In Section [4.1,](#page-7-0) we present a formal formulation of the task. In Section [4.2,](#page-7-1) we propose a Dual-grained Memory System to store historical video stream clips and defined concepts. In Section [4.3,](#page-8-1) we present an efficient Concept-aware Retrieval Algorithm for fast retrieval and response.

#### <span id="page-7-0"></span>4.1 Formulation

Formally, we define a streaming video as an infinite sequence V = [X1, X2, . . . ], where X<sup>i</sup> denotes a video clip representing a semantic scene. Throughout the stream, a user can dynamically introduce new concepts at any timestamp t<sup>c</sup> via instructions, forming an evolving set of defined concepts C = {C1, C2, . . . }. For a query Q issued at time t<sup>q</sup> ≥ tc, the model M must dynamically construct a context to generate a response A:

$$A = \mathcal{M}(\mathcal{C}_{sub}, \mathcal{V}_{context}, Q) \tag{1}$$

where Csub ⊆ C is the query-relevant concept subset, and Vcontext is the necessary visual context. Solving this requires overcoming two key challenges: the prohibitive cost of maintaining unbounded stream history alongside evolving concepts, and the difficulty of accurately retrieving personalized Csub and Vcontext in real-time. This motivates our design of a scalable dual-grained memory and a concept-aware retrieval strategy.

# <span id="page-7-1"></span>4.2 Dual-grained Memory System

To support PSVU, the model must (i) retain user-defined concepts introduced at arbitrary timestamps and (ii) maintain access to long-range visual evidence from the evolving video stream for real-time retrieval and response. We therefore design a Dual-grained Memory System that explicitly decouples conceptcentric knowledge from stream-centric observations. Concretely, it consists of a Streaming Memory that incrementally archives segmented clips with compact multimodal embeddings for efficient retrieval, and a Concept Memory that stores structured representations of user-defined concepts. We next describe these two memory components in detail.

Streaming Memory Streaming Memory maintains a set of entries, each consisting of a video clip X<sup>i</sup> and its corresponding embedding e<sup>i</sup> . Given a continuously arriving video stream, we first detect scene boundaries and segment the stream into an ordered sequence of clips V = [X1, X2, . . . ]. For each newly detected clip X<sup>i</sup> , we employ a multimodal embedding model femb(·) to compute

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Fig. 3: PEARL framework. (a) Dual-grained Memory System: Concept Memory stores user-defined concepts with visual evidence and textual descriptions; Streaming Memory archives segmented clips with multimodal embeddings. (b) Conceptaware Retrieval Algorithm: Upon a user query, PEARL retrieves relevant concepts and top-K historical clips via concept-rewritten query embeddings, then feeds them together with the current clip into a VLM for real-time personalized response.

an embedding e<sup>i</sup> = femb(Xi), and store the pair (X<sup>i</sup> , ei) in Streaming Memory. Each clip embedding e<sup>i</sup> captures rich semantic information about the scene and is used for subsequent retrieval. Detailed settings are provided in the appendix.

Concept Memory When a Concept-Definition query Qdef is issued at timestamp tc, the model creates a new entry with three components: (i) a concept name, (ii) associated visual evidence, and (iii) a textual description. The model first invokes an external tool to extract the visual evidence from the current clip X<sup>t</sup><sup>c</sup> : for video-level concepts, the evidence is the clip X<sup>t</sup><sup>c</sup> itself, whereas for frame-level concepts the model stores the last frame of X<sup>t</sup><sup>c</sup> . Conditioned on this extracted visual evidence, the model then generates a compact description that summarizes the concept's salient characteristics, using a standardized prompting template provided in the appendix. The resulting entry is finally inserted into the Concept Memory for subsequent retrieval and querying.

# <span id="page-8-1"></span>4.3 Concept-aware Retrieval Algorithm

When a user issues a Real-Time query or a Past-Time query at timestamp tq, the model needs three types of information to respond accurately: (i) the query Q, (ii) the query-relevant concept subset Csub ⊆ C retrieved from Concept Memory, and (iii) the visual evidence Vcontext retrieved from Streaming Memory, producing the final answer as A = M(Csub, Vcontext, Q). To obtain Csub, we identify the concept names mentioned in Q and use them as keys to retrieve the corresponding Concept Memory entries. To obtain Vcontext, we use the model M to rewrite the query into Q˜ by replacing each concept name with its associated description, then encode it using the same multimodal embedding model as Streaming Memory to compute e<sup>Q</sup> = femb(Q˜). We compute cosine similarities between e<sup>Q</sup> and all stored clip embeddings {ei}i≤t<sup>q</sup> (where e<sup>i</sup> = femb(Xi)), select the top-K most relevant clips, and further expand each selected clip with its adjacent N clips to capture temporally local context, yielding Vcontext ⊆ {X1, . . . , X<sup>t</sup><sup>q</sup> }. Finally, we feed the retrieved concept entries, the retrieved historical clips, the current clip Xt<sup>q</sup> , and the original query Q into the VLM to generate the response, preserving real-time responsiveness while maximizing retrieval of task-relevant evidence.

# 5 Experiments

# 5.1 Implementation Details

As a plug-and-play framework, we employ Qwen3-VL-Embedding-2B [\[45\]](#page-17-3) as the multimodal embedding model, and use LLaVA-OV-7B [\[3\]](#page-14-8), Qwen2-VL-7B [\[46\]](#page-17-4), and Qwen3-VL-8B [\[2\]](#page-14-9) as base models, respectively. For the Dual-grained Memory System, we use the corresponding base model to generate concept descriptions and rewrite user queries. We use PySceneDetect [\[47\]](#page-17-5) to detect scene boundaries and segment the streaming video into clips. For the Concept-aware Retrieval Algorithm, we set K = 4, use N = 1 for frame-level data and N = 0 for videolevel data, and sample the input video stream at 1 FPS.

As a pioneering effort in this area, where no directly comparable baselines exist, we instead adopt representative offline and online video understanding methods as our baselines. For offline models, we adopt different sampling protocols: for frame-level data, we uniformly sample 64 frames; for video-level data, we use a 64-second window and sample at 1 FPS to ensure the model observes continuous actions. For online models, given the unique paradigm of the PSVU task, which necessitates concept registration prior to question answering, we exclusively select models capable of multi-turn dialogue as our baselines. Furthermore, we strictly follow their original sampling configurations, preserving their native sampling frame rates and maximum frame count constraints. All experiments are conducted on NVIDIA H200 GPUs. Note that to minimize the model's bias toward answer options and prevent random guessing [\[48,](#page-17-6) [49\]](#page-17-7), we evaluate each question using a cyclic option rotation strategy, with detailed settings provided in the appendix.

# 5.2 Text-Only Results and Human Score

To establish the upper and lower bounds for PEARL-Bench in Table [3,](#page-10-0) we report the Human Score and a Text-only baseline. The Human Score is obtained by having annotators answer questions with full access to both the visual stream and concept definitions. This human performance sets a robust upper bound, demonstrating that the task is highly solvable given sufficient visual information. Conversely, the Text-only baseline feeds only the query text to the model without any visual inputs. Evaluated using Qwen3-VL-8B (a model with strong pure-text

<span id="page-10-0"></span>Table 3: Results on PEARL-Bench. We report Frame-level Real-Time/Past-Time (with Avg) and Video-level Real-Time metrics. Bold and underline denote the best and second-best results among open-source models and PEARL, respectively.

| Method                   | #Frames                   |                           | Video-level  |              |              |  |  |
|--------------------------|---------------------------|---------------------------|--------------|--------------|--------------|--|--|
|                          |                           | Real-Time                 | Past-Time    | Avg          | Real-Time    |  |  |
| Human Score              |                           |                           |              |              |              |  |  |
| Human                    | -                         | 97.61                     | 96.45        | 97.03        | 97.49        |  |  |
|                          |                           | Text-only                 |              |              |              |  |  |
| Qwen3-VL-8B [2]          | -                         | 11.06                     | 17.45        | 14.26        | 7.04         |  |  |
|                          |                           | Proprietary Offline Model |              |              |              |  |  |
| Gemini3-pro-preview [5]  | 64                        | 47.40                     | 48.98        | 48.19        | 24.51        |  |  |
|                          | Open-source Offline Model |                           |              |              |              |  |  |
| LLava-OV-7B [3]          | 64                        | 24.95                     | 34.01        | 29.48        | 10.86        |  |  |
| Qwen2-VL-7B [46]         | 64                        | 23.21                     | 35.79        | 29.50        | 17.89        |  |  |
| InternVL3.5-8B [1]       | 64                        | 30.35                     | 35.06        | 32.71        | 5.57         |  |  |
| Qwen3-VL-8B [2]          | 64                        | 27.33                     | 30.20        | 28.77        | 25.51        |  |  |
| Open-source Online Model |                           |                           |              |              |              |  |  |
| ReKV(LLava-OV-7B) [33]   | 0.5fps                    | 26.20                     | 37.46        | 31.83        | 24.11        |  |  |
| StreamForest-7B [35]     | 1fps                      | 29.18                     | 40.86        | 35.02        | 10.85        |  |  |
| TimeChat-Online-7B [34]  | 1fps                      | 31.89                     | 35.28        | 33.59        | 22.29        |  |  |
| PEARL Framework          |                           |                           |              |              |              |  |  |
| LLava-OV-7B+PEARL        | 1fps                      | 33.41 ↑8.46               | 42.64 ↑8.63  | 38.03 ↑8.55  | 19.94 ↑9.08  |  |  |
| Qwen2-VL-7B+PEARL        | 1fps                      | 33.30 ↑10.09              | 44.42 ↑8.63  | 38.86 ↑9.36  | 24.34 ↑6.45  |  |  |
| Qwen3-VL-8B+PEARL        | 1fps                      | 54.99 ↑27.66              | 49.49 ↑19.29 | 52.24 ↑23.47 | 48.39 ↑22.88 |  |  |

capability), this lower bound exhibits poor performance across all metrics. The significant disparity between these bounds confirms that the benchmark cannot be reliably solved relying on text priors alone, and fundamentally requires visual grounding in the streaming content.

#### 5.3 Frame-level Results

Comparison with Offline Baselines We evaluate several representative opensource offline models, along with a strong proprietary offline baseline. As shown in Table [3,](#page-10-0) these offline approaches struggle on PSVU task, exhibiting generally poor performance across the board. In contrast, applying PEARL to the same base models yields consistent and substantial improvements: LLaVA-OV-7B+PEARL, Qwen2-VL-7B+PEARL, and Qwen3-VL-8B+PEARL improve their offline counterparts by 8.55%, 9.36%, and 23.47%, respectively, demonstrating the generality and robustness of PEARL across distinct architectures. Notably, even the strong proprietary Gemini3-pro-preview [\[5\]](#page-14-10) falls short of our Qwen3- VL-8B+PEARL, lagging behind by over 4%. This performance gap reveals a key limitation of offline models in the streaming setting: to satisfy low-latency inference, they operate with a restricted visual context (64 frames in our experiments) and typically lack explicit memory mechanisms to preserve and retrieve

long-range historical evidence. Consequently, the visual information needed to answer many queries is often missing, leading to degraded accuracy.

Comparison with Online Baselines We compare PEARL against three representative open-source online models: ReKV (LLaVA-OV-7B) [\[33\]](#page-16-6), StreamForest-7B [\[35\]](#page-16-7), and TimeChat-Online-7B [\[34\]](#page-16-8). As shown in Table [3,](#page-10-0) all three PEARL variants consistently surpass the best online baseline StreamForest-7B. Specifically, LLaVA-OV-7B+PEARL improves upon StreamForest-7B by 3.01%, Qwen2- VL-7B+PEARL by 3.84%, and Qwen3-VL-8B+PEARL achieves a substantial gain of 17.22%. These improvements underscore the superiority of PEARL in effectively managing continuous visual streams for personalized understanding.

Notably, LLaVA-OV-7B+PEARL comprehensively outperforms ReKV on both Real-Time and Past-Time metrics, achieving gains of 7.21% and 5.18% respectively. Since ReKV shares the same backbone and is likewise a training-free, plug-and-play framework, this controlled comparison strongly indicates that the performance gains stem from the PEARL framework design itself rather than differences in backbone capability. We attribute these advantages to PEARL's Dual-grained Memory System and Concept-aware Retrieval Algorithm: whereas traditional online models compress historical information into a fixed-size state and lack concept-grounded retrieval, PEARL explicitly stores user-defined concept representations and precisely retrieves query-relevant historical clips, enabling more accurate personalized responses in the streaming setting.

#### 5.4 Video-level Results

As shown in Table [3,](#page-10-0) all models achieve notably lower scores on the videolevel split than on the frame-level split, reflecting the greater difficulty of this setting, where models must not only recognize personalized concepts but also reason over continuous actions unfolding across frames. Nevertheless, PEARL demonstrates clear superiority. When compared to their respective offline base models, all three PEARL variants yield consistent and significant performance improvements. Moreover, Qwen3-VL-8B+PEARL achieves the highest overall accuracy, outperforming the best online baseline ReKV by a massive margin of 24.28%. It also substantially surpasses Gemini3-pro-preview, leading by nearly 24%. These results indicate that the design of PEARL generalizes effectively to the more demanding video-level personalized understanding task.

## 5.5 Ablation Study

<span id="page-11-0"></span>Effectiveness of PEARL Design We ablate PEARL by progressively enabling its components in Table [4.](#page-12-0) Using Qwen3-VL-8B on the frame-level split, text-only performance starts near random chance, and adding the current clip provides only a marginal improvement. In contrast, adding Concept Memory with concept retrieval leads to a dramatic gain of over 35% in Real-Time accuracy, showing that explicit concept grounding is crucial. Subsequently, incorporating Streaming Memory with retrieval yields a substantial jump of more than 20% in Past-Time accuracy, indicating the need for historical evidence. Finally, adding Query Rewriting (full PEARL) further improves both Real-Time and Past-Time performance, elevating the average accuracy by an additional 4.28% over the non-rewritten version.

These results suggest three takeaways. (i) Concept Memory is indispensable: without concept-specific information, the model cannot reliably link user-defined names to personalized entities, so the benchmark is hard to solve. (ii) Streaming Memory is essential for Past-Time QA, which depends on retrieving and reasoning over historical clips rather than the current scene. (iii) Query Rewriting turns personalized names into descriptive semantics that embedding models can match more effectively, improving evidence retrieval and final answers.

<span id="page-12-0"></span>**Table 4: Ablation of PEARL components** on the frame-level split with Qwen3-VL-8B, showing progressive gains from Text-only to the full PEARL pipeline.

| Text | Current | Concept | Streaming | g Rewrite | Real-Time | Past-Time | Avg   |
|------|---------|---------|-----------|-----------|-----------|-----------|-------|
| 1    | X       | Х       | Х         | Х         | 11.06     | 17.45     | 14.26 |
| 1    | 1       | X       | X         | Х         | 15.84     | 20.30     | 18.07 |
| 1    | 1       | ✓       | X         | Х         | 51.41     | 25.43     | 38.42 |
| 1    | 1       | ✓       | ✓         | Х         | 50.22     | 45.69     | 47.96 |
| ✓    | 1       | ✓       | ✓         | ✓         | 54.99     | 49.49     | 52.24 |

Efficiency of PEARL Design We further evaluate end-to-end inference efficiency. As shown in Table 5, on the frame-level split of PEARL-Bench, although PEARL introduces a minor latency overhead compared to the respective base models, both LLaVA-OV-7B+PEARL and Qwen3-VL-8B+PEARL achieve substantial improvements in frame-level average

**Table 5:** Comparison of end-to-end inference latency across models. F-Avg denotes the frame-level average accuracy.

<span id="page-12-1"></span>

| Method                                   | #Frames            | F-Avg I | Latency (ms) |
|------------------------------------------|--------------------|---------|--------------|
| LLaVA-OV-7B                              | 64                 | 29.48   | 670          |
| Qwen3-VL-8B                              | 64                 | 28.77   | 1,594        |
| ReKV (LLaVA-OV-7B)                       | $0.5 \mathrm{fps}$ | 31.83   | 1,818        |
| StreamForest-7B                          | 1 fps              | 35.02   | 1,164        |
| TimeChat-Online-7B                       | 1fps               | 33.59   | 4,769        |
| LLaVA-OV-7B+PEARL                        | 1fps               | 38.03   | 775          |
| ${\bf Qwen 3\text{-}VL\text{-}8B+PEARL}$ | $1 \mathrm{fps}$   | 52.24   | 2,111        |

accuracy, with gains of 8.55% and 23.47% respectively. Moreover, LLaVA-OV-7B+PEARL not only maintains higher accuracy than all online baselines but also remains faster than all of them. Although Qwen3-VL-8B+PEARL has a slightly higher latency, its accuracy heavily surpasses the strongest online baseline, StreamForest-7B, leading by 17.22%.

Furthermore, as shown in Fig. 5, the end-to-end latency of our PEARL framework breaks down into Concept Retrieval, Query Rewriting, Streaming Memory Retrieval, and LLM Inference. Notably, the latency introduced by PEARL's core modules (retrieval and rewriting) is exceptionally low and invariant across different models, and the underlying LLM inference constitutes the primary latency bottleneck. This demonstrates that PEARL can seamlessly adapt to diverse model architectures while maintaining real-time retrieval capabilities.

<span id="page-13-0"></span>![](_page_13_Figure_2.jpeg)

![](_page_13_Figure_3.jpeg)

Fig. 4: Past-Time accuracy under different top-K(K) and expansion sizes(N).

Fig. 5: End-to-end latency breakdown of PEARL across different models.

Hyperparameter Analysis We analyze two key hyperparameters on Past-Time QA: the number of top-K retrieved clips (K) and the number of adjacent clips to expand per retrieved clip (N). As shown in Fig. [4,](#page-13-0) when K = 0, no historical clips are retrieved, so the model cannot leverage historical video evidence and the metric remains low; as K increases, accuracy improves rapidly and plateaus after K ≥ 3, indicating that moderate retrieval suffices to cover the visual evidence needed for answering. Regarding N, larger adjacent expansion provides richer local temporal context, but the gap between N = 1 and N = 2 is small, suggesting that one adjacent clip already captures most of the useful information. Balancing performance and efficiency, we adopt K = 4 and N = 1 as our default configuration. Since Real-Time QA can be accurately addressed without relying on historical clips, it is inherently less sensitive to the parameters K and N. For completeness, corresponding experimental results for Real-Time QA are provided in the appendix.

# 6 Conclusion

We introduced Personalized Streaming Video Understanding (PSVU) task and our PEARL-Bench, the first comprehensive benchmark for frame-level and videolevel personalization in streaming videos. To tackle this, we proposed PEARL, a training-free framework featuring a dual-grained memory system and conceptaware retrieval algorithm. PEARL consistently achieves state-of-the-art performance across multiple architectures with controllable latency. We hope this work inspires future research toward interactive personalized AI assistants.

# Bibliography

- <span id="page-14-0"></span>[1] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- <span id="page-14-9"></span>[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3 vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- <span id="page-14-8"></span>[3] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [4] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025.
- <span id="page-14-10"></span>[5] Google DeepMind. A new era of intelligence with Gemini 3. [https://blog.](https://blog.google/products-and-platforms/products/gemini/gemini-3/) [google/products-and-platforms/products/gemini/gemini-3/](https://blog.google/products-and-platforms/products/gemini/gemini-3/), 2025.
- <span id="page-14-1"></span>[6] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- <span id="page-14-2"></span>[7] Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, et al. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024.
- <span id="page-14-3"></span>[8] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo'llava: Your personalized language and vision assistant, 2024. URL https://arxiv. org/abs/2406, 9400.
- <span id="page-14-5"></span>[9] Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, and Xiangyu Yue. Rap: Retrieval-augmented personalization for multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14538–14548, 2025.
- <span id="page-14-4"></span>[10] Yufei Shi, Weilong Yan, Gang Xu, Yumeng Li, Yucheng Chen, Zhenxi Li, Fei Yu, Ming Li, and Si Yong Yeo. Pvchat: Personalized video chat with oneshot learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23321–23331, 2025.
- <span id="page-14-6"></span>[11] Binxiao Xu, Junyu Feng, Shaolin Lu, Yulin Luo, Shilin Yan, Hao Liang, Ming Lu, and Wentao Zhang. Jarvis: Towards personalized ai assistant via personal kv-cache retrieval. arXiv preprint arXiv:2510.22765, 2025.
- <span id="page-14-7"></span>[12] Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025.

- <span id="page-15-8"></span>[13] Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, and Yuheng Li. Yo'chameleon: Personalized vision and language generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14438–14448, 2025.
- <span id="page-15-7"></span>[14] Sihan Yang, Huitong Ji, Shaolin Lu, Jiayi Chen, Binxiao Xu, Ming Lu, Yuanxing Zhang, Wenhui Dong, and Wentao Zhang. Small-large collaboration: Training-efficient concept personalization for large vlm using a meta personalized small vlm. arXiv preprint arXiv:2508.07260, 2025.
- <span id="page-15-0"></span>[15] Jaeik Kim, Woojin Kim, Woohyeon Park, and Jaeyoung Do. Mmpb: It's time for multi-modal personalization. arXiv preprint arXiv:2509.22820, 2025.
- <span id="page-15-1"></span>[16] Saif Ahmed and Norzalita Abd Aziz. Impact of ai on customer experience in video streaming services: a focus on personalization and trust. International Journal of Human–Computer Interaction, 41(12):7726–7745, 2025.
- <span id="page-15-2"></span>[17] Norina Gasteiger, Mehdi Hellou, and Ho Seok Ahn. Factors for personalization and localization to optimize human–robot interaction: A literature review. International Journal of Social Robotics, 15(4):689–701, 2023.
- <span id="page-15-3"></span>[18] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.
- [19] OpenAI. Gpt-4v(ision) system card. [https : / / openai . com /](https://openai.com/contributions/gpt-4v/) [contributions/gpt-4v/](https://openai.com/contributions/gpt-4v/), 2023.
- <span id="page-15-4"></span>[20] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4122–4134, 2025.
- <span id="page-15-5"></span>[21] Mengze Hong, Chen Jason Zhang, Chaotao Chen, Rongzhong Lian, and Di Jiang. Dialogue language model with large-scale persona data engineering. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track), pages 961–970, 2025.
- [22] Niv Cohen, Rinon Gal, Eli A Meirom, Gal Chechik, and Yuval Atzmon. "this is my unicorn, fluffy": Personalizing frozen vision-language representations. In European conference on computer vision, pages 558–577. Springer, 2022.
- [23] Junda Wu, Hanjia Lyu, Yu Xia, Zhehao Zhang, Joe Barrow, Ishita Kumar, Mehrnoosh Mirtaheri, Hongjie Chen, Ryan A Rossi, Franck Dernoncourt, et al. Personalized multimodal large language models: A survey. arXiv preprint arXiv:2412.02142, 2024.
- [24] Yeongtak Oh, Sangwon Yu, Junsung Park, Han Cheol Moon, Jisoo Mok, and Sungroh Yoon. Contextualized visual personalization in vision-language models. arXiv preprint arXiv:2602.03454, 2026.
- <span id="page-15-6"></span>[25] Junxian Li, Tu Lan, Haozhen Tan, Yan Meng, and Haojin Zhu. Slowba: An efficiency backdoor attack towards vlm-based gui agents. arXiv preprint arXiv:2603.08316, 2026.

- <span id="page-16-0"></span>[26] Yuval Alaluf, Elad Richardson, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Myvlm: Personalizing vlms for user-specific queries. In European Conference on Computer Vision, pages 73–91. Springer, 2024.
- <span id="page-16-1"></span>[27] Yeongtak Oh, Dohyun Chung, Juhyeon Shin, Sangha Park, Johan Barthelemy, Jisoo Mok, and Sungroh Yoon. Repic: Reinforced posttraining for personalizing multi-modal language models. arXiv preprint arXiv:2506.18369, 2025.
- <span id="page-16-2"></span>[28] Junyu Feng, Binxiao Xu, Jiayi Chen, Mengyu Dai, Cenyang Wu, Haodong Li, Bohan Zeng, Yunliu Xie, Hao Liang, Ming Lu, et al. M2a: Multimodal memory agent with dual-layer hybrid memory for long-term personalized interactions. arXiv preprint arXiv:2602.07624, 2026.
- <span id="page-16-3"></span>[29] Yu Zhong, Tianwei Lin, Ruike Zhu, Yuqian Yuan, Haoyu Zheng, Liang Liang, Wenqiao Zhang, Feifei Shao, Haoyuan Li, Wanggui He, et al. Unified personalized understanding, generating and editing. arXiv preprint arXiv:2601.06965, 2026.
- [30] Sen Ye, Mengde Xu, Shuyang Gu, Di He, Liwei Wang, and Han Hu. Understanding vs. generation: Navigating optimization dilemma in multimodal models. arXiv preprint arXiv:2602.15772, 2026.
- <span id="page-16-4"></span>[31] Sen Ye, Jianning Pei, Mengde Xu, Shuyang Gu, Chunyu Wang, Liwei Wang, and Han Hu. Distribution matching variational autoencoder. arXiv preprint arXiv:2512.07778, 2025.
- <span id="page-16-5"></span>[32] Chun-Hsiao Yeh, Bryan Russell, Josef Sivic, Fabian Caba Heilbron, and Simon Jenni. Meta-personalizing vision-language models to find named instances in video. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 19123–19132, 2023.
- <span id="page-16-6"></span>[33] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Tao Zhong, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. Streaming video question-answering with in-context video kv-cache retrieval. arXiv preprint arXiv:2503.00540, 2025.
- <span id="page-16-8"></span>[34] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10807–10816, 2025.
- <span id="page-16-7"></span>[35] Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, et al. Streamforest: Efficient online video understanding with persistent event memory. arXiv preprint arXiv:2509.24871, 2025.
- [36] Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18902–18913, 2025.
- [37] Zhenyu Yang, Yuhang Hu, Zemin Du, Dizhan Xue, Shengsheng Qian, Jiahong Wu, Fan Yang, Weiming Dong, and Changsheng Xu. Svbench: A

- benchmark with temporal multi-turn dialogues for streaming video understanding. arXiv preprint arXiv:2502.10810, 2025.
- [38] Shuhang Xun, Sicheng Tao, Jungang Li, Yibo Shi, Zhixin Lin, Zhanhui Zhu, Yibo Yan, Hanqian Li, Linghao Zhang, Shikang Wang, et al. Rtv-bench: Benchmarking mllm continuous perception, understanding and reasoning through real-time video. arXiv preprint arXiv:2505.02064, 2025.
- [39] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024.
- [40] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024.
- [41] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24045–24055, 2025.
- <span id="page-17-0"></span>[42] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025.
- <span id="page-17-1"></span>[43] Adobe Systems Inc. Mixamo: Adobe Animated 3D Characters and Motion Capture Library. <https://www.mixamo.com/>, 2026. Accessed: 2026-03-03.
- <span id="page-17-2"></span>[44] Social Security Administration. Popular Baby Names. [https://www.ssa.](https://www.ssa.gov/oact/babynames/) [gov/oact/babynames/](https://www.ssa.gov/oact/babynames/), 2026. Accessed: 2026-03-03.
- <span id="page-17-3"></span>[45] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vlembedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.
- <span id="page-17-4"></span>[46] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- <span id="page-17-5"></span>[47] Brandon Castellano. Pyscenedetect. <https://www.scenedetect.com>, 2026. Video Cut Detection and Analysis Tool.
- <span id="page-17-6"></span>[48] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2006–2017, 2024.
- <span id="page-17-7"></span>[49] Md Atabuzzaman, Ali Asgarov, and Chris Thomas. Benchmarking and mitigating mcqa selection bias of large vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33536–33550, 2025.

# A QA Sub-categories

To thoroughly evaluate model capabilities under diverse scenarios, we further categorize the queries in PEARL-Bench. As shown in Fig. [6,](#page-18-0) we systematically classify the questions within Concept-Definition QA, Real-Time QA, and Past-Time QA into fine-grained sub-categories based on their reasoning requirements.

<span id="page-18-0"></span>![](_page_18_Figure_4.jpeg)

Fig. 6: Distribution of fine-grained sub-categories in PEARL-Bench. "(Frame)" denotes sub-categories belonging to the frame-level split, while "(Video)" denotes those in the video-level split.

Concept-Definition QA. We classify the concept definitions into two distinct types:

- Direct: The concept is defined straightforwardly without relying on complex descriptions. For frame-level concepts, it points out the main character in the scene (e.g., "This character is called {ConceptName}. Please remember this name."); for video-level concepts, it identifies a specific personalized action unfolding over a continuous clip (e.g., "The sequence of movements shown in this clip is defined as {ConceptName}. Please remember this name.")
- Contextual: The concept is defined by describing its explicit visual attributes (like clothing color) or its interactions and relationships with other objects in the scene. (e.g., "The character wearing white clothes is named {ConceptName}. Please remember this name.")

Real-Time QA. To evaluate multi-dimensional perception, we categorize the real-time queries into six distinct sub-tasks:

- Presence: Asking whether the concept is present in the current scene. (e.g., "Is {ConceptName} here now?")
- Behavior: Querying the current action or state of the defined concept. (e.g., "What is {ConceptName} doing now?")
- Appearance: Focusing on the transient visual details of the concept, such as current clothing. (e.g., "What color is {ConceptName} wearing now?")
- Location: Identifying the spatial positioning of the concept within the scene. (e.g., "Where is {ConceptName} located in this scene now?")
- Relation: Inquiring about the interaction or relationship between the target concept and other entities or objects. (e.g., "Who is standing next to {ConceptName} now?")
- Action: Querying whether a dynamically defined action concept is being performed across the current continuous clip. (e.g., "Is the person doing {ConceptName} now?")

Past-Time QA. We divide the past-time queries into two types based on the retrieval mechanism required to answer them:

- Event-based: The historical evidence can be localized purely based on a specific event or action, without requiring strict temporal reasoning. (e.g., "What was {ConceptName} holding when he was cooking?")
- Time-based: The historical evidence requires the model to understand the temporal sequence or order of events to accurately retrieve the correct clip. (e.g., "What did {ConceptName} do right before he left the room?")

# B Complete Question Templates

As mentioned in the main text, we provide the complete question templates used in PEARL-Bench for annotating Concept-Definition QA, Real-Time QA, and Past-Time QA. In Table [6,](#page-20-0) we summarize the representative question templates corresponding to each fine-grained sub-category. It is worth noting that these question templates serve primarily as a reference for the annotators. In practice, annotators often diversify the content, phrasing, and sentence structures of the questions based on specific video scenarios, thereby ensuring the richness and naturalness of the data. To represent the varied entities and properties systematically, we utilize the following placeholders in the templates:

- {ConceptName}: The dynamically assigned user-defined name for the target concept.
- {OtherConcept}: The name of another previously defined concept.
- {EntityPronoun}: General references to the subject, such as person, character, man, girl, object, etc.
- {ConceptFeature}: Specific visual characteristics or attributes used to describe the concept, such as short hair, red clothes, sitting by window, etc.
- {Object}: Variables pertaining to the question, representing objects interacted with or queried.

<span id="page-20-0"></span>**Table 6:** Representative question templates across different QA types and subcategories in PEARL-Bench.

| QA Type            | Sub-category Question Templates |                                                                                                                                                                                                                                                                                                                                                                                                     |  |  |  |
|--------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|
| Concept-Definition | Direct                          | <ul> <li>This {EntityPronoun} is called {ConceptName}. Please remember this name. {DefinitionSuffix}</li> <li>This {EntityPronoun} is my friend, named {ConceptName}. Please remember this name. {DefinitionSuffix}</li> <li>The sequence of movements shown in this clip is defined as {ConceptName}. Please remember this name. {DefinitionSuffix}</li> </ul>                                     |  |  |  |
|                    | Contextual                      | <ul> <li>The {EntityPronoun} in the {Location} is named {ConceptName}. Please remember this name. {DefinitionSuffix}</li> <li>The {EntityPronoun} with {ConceptFeature} in the scene is my friend, named {ConceptName}. Please remember this name. {DefinitionSuffix}</li> <li>The {EntityPronoun} who is {Event} is called {ConceptName}. Please remember this name. {DefinitionSuffix}</li> </ul> |  |  |  |
|                    | Presence                        | <ul> <li>Is this {EntityPronoun} {ConceptName} at this moment?</li> <li>Are {ConceptName} and {OtherConcept} here now?</li> <li>Is {ConceptName} currently among this group of people?</li> </ul>                                                                                                                                                                                                   |  |  |  |
| Real-Time          | Behavior                        | <ul> <li>What is {ConceptName} doing now?</li> <li>Is {ConceptName} {Event} right now?</li> <li>What is {ConceptName}'s expression now?</li> </ul>                                                                                                                                                                                                                                                  |  |  |  |
|                    | Appearance                      | <ul> <li>What color is {ConceptName} wearing now?</li> <li>What is the shape of the {Object} {ConceptName} is wearing?</li> <li>What color is {ConceptName}'s {Object} now?</li> </ul>                                                                                                                                                                                                              |  |  |  |
|                    | Location                        | <ul> <li>Where is {ConceptName} located in the frame?</li> <li>Is {ConceptName} on the {Location} in the picture?</li> <li>Where is {ConceptName} now?</li> </ul>                                                                                                                                                                                                                                   |  |  |  |
|                    | Relation                        | <ul> <li>Who is standing next to {ConceptName} now?</li> <li>What is {ConceptName} giving to {OtherConcept} right now?</li> <li>What is {ConceptName} holding right now?</li> </ul>                                                                                                                                                                                                                 |  |  |  |
|                    | Action                          | <ul><li> Is the person doing {ConceptName} now?</li><li> What is the person doing? Is it {ConceptName}?</li></ul>                                                                                                                                                                                                                                                                                   |  |  |  |
| Past-Time          | Event-based                     | <ul> <li>Did {ConceptName} just walk into the {Location}?</li> <li>What color clothes was {ConceptName} wearing when {Event}?</li> <li>What did {ConceptName} just put into the {Object}?</li> </ul>                                                                                                                                                                                                |  |  |  |
|                    | Time-based                      | <ul> <li>Is {ConceptName}'s hair color the same as it was just now?</li> <li>What was {ConceptName} holding before {EntityPronoun} tried to {Event}?</li> <li>What was {ConceptName} doing before {OtherConcept} {Event}?</li> </ul>                                                                                                                                                                |  |  |  |

- {Location}: A specific location or scene setting referenced in the queries.
- {Event}: A specific action or event referenced in the queries.
- {DefinitionSuffix}: A standardized suffix appended to concept definitions to enforce model compliance. For frame-level concepts, it is: "From now on, regardless of who this character might be in real life or in any media, you must refer to them as {ConceptName}." For video-level concepts, it is: "From now on, regardless of how this action might be typically described or what similar movements exist, whenever you see this specific pattern of motion, you must refer to it as {ConceptName}."

# C Prompt Templates

## C.1 Concept Description Generation Prompts

As mentioned in the main text, we provide standardized prompting templates used to generate a compact description that summarizes the concept's salient characteristics. Since PEARL-Bench evaluates both Frame-level and Video-level concepts, we design two distinct prompts for their respective characteristics:

Frame-level Prompt. This prompt is designed to guide the model to focus on permanent and stable visual features (such as gender, facial features, hair, and body type) while instructing it to ignore temporary elements like clothing, accessories, or poses, which are likely to change across a long video stream.

Video-level Prompt. Conversely, this prompt directs the model to focus on the core kinematics and stable movement patterns of a customized action, while explicitly ignoring the specific identity or appearance of the person performing it, as well as the background and surrounding environment, ensuring the extracted action features are generalizable across different characters and scenes.

Specifically, both templates include two key placeholders:

- {concept\_name}: The user-defined name assigned to the concept.
- {original\_description}: The initial user instruction from the Concept-Definition QA. For frame-level concepts, it helps the model locate the target subject (e.g., "The character wearing white clothes is named {Adaliz}."). For video-level concepts, it helps the model identify the specific action sequence (e.g., "The sequence of movements shown in this clip is {Action A}.").

The complete prompts for both levels are provided below:

# Æ Frame-level Concept Description Generation Prompt

Based on the image and the original description provided, generate a concise visual description of this character/object that focuses on PERMANENT/STABLE features for video clip retrieval.

Original description: "{original\_description}" Concept name: {concept\_name}

Your task:

1. Use the original description to understand WHICH

character/object to focus on in the image

- 2. Generate a description focusing on STABLE features that DON'T change throughout the video:
  - Gender (male/female/other)
  - Face features (eye shape, facial structure, distinctive marks)
  - Hair (color, length, style if distinctive)
  - Body type (build)
  - Age appearance (young/middle-aged/elderly)

AVOID or minimize:

- Clothing details (they change in long videos)

- Accessories (they may be removed)
- Temporary expressions or poses
- Background, location, surroundings, or nearby objects in the scene
- Relative position or size compared to objects/environment in the scene

#### Requirements:

- Keep it concise and simple (1 sentence, around 10-15 words)
- Focus on features that remain consistent across different scenes
- Write in English using simple descriptive terms
- Use third person (e.g., "a young male with...", "the girl with...")
- Make it natural enough to replace the concept name in a question

Please provide the distinctive visual description focusing on PERMANENT features:

# Æ Video-level Concept Description Generation Prompt

Based on the provided video clip and the original description, generate a concise textual description of the specific ACTION or MOVEMENT PATTERN that focuses on the CORE KINEMATICS for video clip retrieval.

Original description: "{original\_description}" Concept name: {concept\_name}

#### Your task:

- 1. Use the original description to understand WHICH specific action or sequence of movements to focus on in the video clip
- 2. Generate a description focusing on the STABLE MOVEMENT PATTERNS that define this action, regardless of who is performing it:
  - Core body movements (e.g., raising arms, squatting, twisting)
  - Sequence of motions (the order of the gestures)
  - Body parts involved (hands, legs, torso)

#### AVOID or minimize:

- The specific identity, gender, age, or appearance of the person performing the action
- Background, location, surroundings, or irrelevant objects in the scene
- Any static features that do not contribute to the dynamic action itself

#### Requirements:

- Keep it concise and simple (1 sentence, around 10-20 words)
- Focus strictly on the dynamic movement pattern that can be performed by different characters
- Write in English using simple descriptive action terms
- Use general action phrases (e.g., "the action of swinging arms in
- a circle", "the action of squatting down and then leaping forward")

- Make it natural enough to replace the concept name in a question Please provide the distinctive action description focusing on CORE MOVEMENT PATTERNS:

# C.2 Query Rewrite Prompt

As discussed in our Concept-aware Retrieval Algorithm, we use a prompt to rewrite user queries by replacing concept names with their descriptions to improve retrieval accuracy. This process helps translate user-defined concept names (which the multimodal embedding model has not seen) into explicit visual or kinematic semantics that facilitate accurate historical clip retrieval.

Specifically, the template includes two key placeholders:

- {query}: The original user question containing the customized concept names.
- {replacement\_instructions}: A set of automatically constructed rules mapping each concept name found in the query to its generated description (e.g., "{Adaliz}" should be replaced with "a young female with long black hair" for a frame-level concept, or "{Action A}" should be replaced with "the action of squatting down and then leaping forward" for a video-level concept).

The complete prompt is provided below:

# Æ Query Rewrite Prompt

Rewrite the following question by replacing the concept names (in curly braces) with their visual descriptions. Keep the sentence grammatically correct and natural.

Original question: {query}

Replacement rules:

{replacement\_instructions}

#### Requirements:

- Replace each {ConceptName} with the provided visual description
- Adjust grammar as needed (e.g., articles, verb forms) to keep the sentence natural
- Do NOT change the meaning of the question
- Do NOT add or remove any information
- Output ONLY the rewritten question, nothing else

# D More Visualization Examples

To better illustrate the complexity and diversity of our dataset, as well as the effectiveness of our proposed framework, we present additional visualization examples from PEARL-Bench. These examples demonstrate the model's capability to handle continuous video streams, dynamically register personalized concepts, and accurately answer both real-time and past-time queries.

### D.1 Frame-level Visualization

As shown in Fig. [7,](#page-25-0) we showcase a comprehensive frame-level interaction process using the Qwen3-VL-8B+PEARL model on a long animated video. As the video stream progresses, the user dynamically defines multiple frame-level concepts at different timestamps (e.g., <Nuriya> at [00:44], <Kavery> at [04:57], and <Truz> at [05:05]). Upon receiving these Concept-Definition instructions, the PEARL framework successfully invokes the Register\_Concept tool. This tool specifically functions to extract the current visual evidence (i.e., the current frame), generate a concise visual description for the target concept, and subsequently update the Concept Memory by storing the visual evidence, the generated description, and the concept name as a unified entity.

Subsequently, the model demonstrates robust real-time perception by accurately answering Real-Time QA based on the current scene. For example, at [01:53], the model not only successfully recognizes <Nuriya>'s presence but also accurately describes her combat posture. Later, at [18:39], it correctly identifies <Kavery> on the left side of the screen. Furthermore, the model exhibits strong long-term temporal reasoning in Past-Time QA. For instance, at [08:01], when asked who saved <Truz>, the model successfully retrieves the historical evidence from [06:48] to provide the correct answer. Similarly, at [13:42], it accurately recalls the terrifying snake encounter that occurred at [11:04]. These results highlight PEARL's ability to maintain and retrieve long-range personalized memories effectively.

#### D.2 Video-level Visualization

In addition to static entities, PEARL-Bench also evaluates the model's ability to understand personalized dynamic actions unfolding over continuous frames. As shown in Fig. [8,](#page-26-0) we illustrate a video-level interaction example using the Qwen3- VL-8B+PEARL model. During the initial phase of the video stream, the user dynamically registers multiple complex action sequences as video-level concepts (e.g., <Action A> at [00:04], <Action B> at [00:16], and <Action C> at [00:34]). Similar to the frame-level process, the model invokes the Register\_Concept tool. However, instead of extracting a single frame, the tool extracts the current video clip corresponding to the action, generates a descriptive summary of the movement pattern, and stores it in the Concept Memory as a video-level entity.

Subsequently, the model accurately recognizes these customized actions when they reappear later in the stream, even when performed by different characters or in different contexts. For example, at [01:08] and [01:30], the model successfully identifies that the character is performing <Action A> and <Action B>, respectively. Furthermore, at [03:25], when multiple characters are present in the scene, the model correctly distinguishes and identifies that the person on the right wearing blue clothes is the one performing <Action C>. These results demonstrate the robust spatiotemporal reasoning and video-level personalization capabilities of the PEARL framework.

<span id="page-25-0"></span>![](_page_25_Figure_2.jpeg)

Fig. 7: Visualization example of frame-level multi-turn interactions in PEARL-Bench. The model successfully registers multiple user-defined concepts (e.g., <Nuriya>, <Kavery>, <Truz>) and accurately answers subsequent Real-Time and Past-Time queries by retrieving corresponding historical evidence.

<span id="page-26-0"></span>![](_page_26_Figure_2.jpeg)

Fig. 8: Visualization example of video-level multi-turn interactions in PEARL-Bench. The model successfully registers multiple user-defined action sequences (e.g., <Action A>, <Action B>, <Action C>) and accurately recognizes these customized actions when they are performed by different characters later in the video stream.

# E Detailed Implementation Settings

# E.1 Multimodal Embedding Model and Scene Detection

We provide additional details regarding the multimodal embedding model and scene detection configurations used in our Dual-grained Memory System.

Multimodal Embedding Model. We adopt Qwen3-VL-Embedding-2B [\[45\]](#page-17-3) as our multimodal embedding model. This model encodes both the generated visual descriptions and the streaming video clips into a unified feature space, enabling efficient and accurate historical evidence retrieval via cosine similarity computation. Video clips are sampled at 1 FPS for embedding extraction.

Scene Detection. We employ PySceneDetect [\[47\]](#page-17-5) to segment the continuous streaming video into semantically coherent clips by detecting fast cuts based on pixel changes in the HSV colorspace between adjacent frames. We set the scene detection threshold to 27.0. To ensure the segmented clips contain sufficient temporal context while avoiding excessively long segments that might dilute the semantic focus, we enforce a minimum clip duration of 1.0 second and a maximum clip duration of 8.0 seconds. Any detected scenes that exceed the maximum duration are proportionally split into multiple smaller segments.

### E.2 Cyclic Option Rotation Evaluation Strategy

As introduced in the main text, we employ a cyclic option rotation strategy during evaluation. Specifically, for each multiple-choice question, we iteratively evaluate the model four times, each time rotating the correct answer to a different option position (A, B, C, and D). During each rotation, we swap the contents of the originally correct option with the target option, leaving the other distractors unchanged, which minimizes perturbation to the overall option distribution. The model is considered to have answered the question correctly only if it consistently selects the correct option across all four rotated variations (i.e., 4/4 correct). This strict evaluation criterion ensures that the model's performance truly reflects its multimodal reasoning capabilities rather than lucky guesses or positional biases.

# F Additional Experimental Results

# F.1 Hyperparameter Analysis for Real-Time QA

As discussed in the main text, Real-Time QA can be accurately addressed without relying heavily on historical clips, making it inherently less sensitive to the parameters K and N. We present the corresponding experimental results in Fig. [9.](#page-28-0) We observe two key findings: (1) Overall, the impact of K and N on the results is not significant, with accuracy fluctuations remaining within a tight 5% range. (2) Compared to the baseline of K = 0 (where no historical clips are retrieved), retrieving a small amount of historical clips generally yields slightly better performance. This suggests that historical clips related to the query can provide supplementary context that modestly improves Real-Time QA accuracy. However, as K and N increase further (introducing more historical clips), the accuracy exhibits a downward trend. This decline indicates that an excessive amount of historical information introduces irrelevant noise, which ultimately interferes with the model's judgment on real-time queries.

#### F.2 Effect of Model Scales

We investigate the performance of PEARL-Bench across different model scales to reveal how parameter size influences personalized streaming video understanding capabilities. Specifically, we evaluate the Qwen2-VL series (2B and 7B) [\[46\]](#page-17-4) and Qwen3-VL series (4B and 8B) [\[2\]](#page-14-9) with and without our PEARL framework on the frame-level split. We do not conduct experiments on models with larger scales, as

<span id="page-28-0"></span>![](_page_28_Figure_2.jpeg)

**Fig. 9:** Real-Time accuracy under different top-K(K) and expansion sizes (N).

<span id="page-28-1"></span>**Table 7:** Performance comparison across different model scales on the frame-level split of PEARL-Bench. **Bold** and <u>underline</u> denote the best and second-best results within each model family (Qwen2-VL and Qwen3-VL), respectively.

| Model                                           | Real-Time                           | Past-Time                            | Avg                      |
|-------------------------------------------------|-------------------------------------|--------------------------------------|--------------------------|
| Qwen2-VL-2B [46]                                | 31.24                               | 22.84                                | 27.04                    |
| Qwen2-VL-7B [46]                                | 23.21                               | 35.79                                | 29.50                    |
| ${\bf Qwen2\text{-}VL\text{-}2B\text{+}PEARL}$  | $29.93_{\ \downarrow 1.31}$         | $32.49_{\uparrow 9.65}$              | $31.21_{\uparrow 4.17}$  |
| ${\bf Qwen2\text{-}VL\text{-}7B\text{+}PEARL}$  | $33.30 \uparrow 10.09$              | $44.42 {\scriptstyle~\uparrow 8.63}$ | $38.86_{\uparrow 9.36}$  |
| Qwen3-VL-4B [2]                                 | 24.08                               | 30.96                                | 27.52                    |
| Qwen3-VL-8B [2]                                 | 27.33                               | 30.20                                | 28.77                    |
| ${\bf Qwen 3-VL-4B+PEARL}$                      | $40.78 \uparrow 16.70$              | $50.25_{\uparrow19.29}$              | $45.52 \uparrow 18.00$   |
| ${\bf Qwen 3\text{-}VL\text{-}8B\text{+}PEARL}$ | $\overline{54.99}_{\uparrow 27.66}$ | $49.49{\scriptstyle~\uparrow 19.29}$ | $52.24_{\uparrow 23.47}$ |

the PSVU task requires real-time responsiveness in practical applications. The experimental results are summarized in Table 7.

Based on the results, we draw two key conclusions:

- 1. Robustness across model scales. Our method consistently yields substantial performance improvements across all sizes and architectures. For instance, the average accuracy of the Qwen3-VL 4B and 8B models is boosted by 18.00% and 23.47%, respectively, with similar trends observed in the Qwen2-VL series (e.g., boosting the 2B and 7B models by 4.17% and 9.36%, respectively). This demonstrates the robustness of the PEARL design, proving its effectiveness regardless of the underlying model capacity or architecture.
- 2. Paradigm mismatch for offline models. When evaluating the standard offline baselines, increasing the model scale (which generally correlates with stronger comprehension capabilities) does not lead to significant performance gains. This highlights that the traditional offline paradigm is fundamentally ill-suited for the PSVU task, as a model's inherent reasoning capability cannot compensate for the lack of visual context. It is only when integrated with a framework specifically designed for PSVU, like PEARL, that the benefits of scaling up the model size are successfully unleashed, with the larger models ultimately outperforming the smaller models by a significant margin.