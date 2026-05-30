# <span id="page-0-0"></span>Speak While Watching: Unleashing TRUE Real-Time Video Understanding Capability of Multimodal Large Language Models

Junyan Lin<sup>1</sup>,2<sup>∗</sup> Junlong Tong<sup>2</sup>,3<sup>∗</sup> Hao Wu<sup>2</sup><sup>∗</sup> Jialiang Zhang<sup>2</sup>,4<sup>∗</sup> Jinming Liu<sup>2</sup>,<sup>3</sup> Xin Jin<sup>2</sup> Xiaoyu Shen<sup>2</sup>,† <sup>1</sup>Department of Computing, The Hong Kong Polytechnic University <sup>2</sup>Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, EIT <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Ocean University of China

[junyan.lin@connect.polyu.hk,](mailto:email@domain) [xyshen@eitech.edu.cn](mailto:email@domain)

# Abstract

*Multimodal Large Language Models (MLLMs) have achieved strong performance across many tasks, yet most systems remain limited to offline inference, requiring complete inputs before generating outputs. Recent streaming methods reduce latency by interleaving perception and generation, but still enforce a sequential perception–generation cycle, limiting real-time interaction. In this work, we target a fundamental bottleneck that arises when extending MLLMs to real-time video understanding: the global positional continuity constraint imposed by standard positional encoding schemes. While natural in offline inference, this constraint tightly couples perception and generation, preventing effective input–output parallelism. To address this limitation, we propose a parallel streaming framework that relaxes positional continuity through three designs: Overlapped, Group-Decoupled, and Gap-Isolated. These designs enable simultaneous perception and generation, allowing the model to process incoming inputs while producing responses in real time. Extensive experiments reveal that Group-Decoupled achieves the best efficiency–performance balance, maintaining high fluency and accuracy while significantly reducing latency. We further show that the proposed framework yields up to 2× acceleration under balanced perception–generation workloads, establishing a principled pathway toward speakwhile-watching real-time systems. We make all our code publicly available:* [https:// github.com/ EIT-](https://github.com/EIT-NLP/Speak-While-Watching)[NLP/Speak-While-Watching](https://github.com/EIT-NLP/Speak-While-Watching)*.*

# 1. Introduction

Modern Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in a wide range of tasks [\[1,](#page-8-0) [17,](#page-8-1) [46\]](#page-10-0). However, the vast majority of existing systems still operate under an *offline* inference paradigm, in which the model must first ingest the entire input sequence before producing any output. While this design aligns well with current benchmark settings [\[11,](#page-8-2) [12,](#page-8-3) [25\]](#page-9-0), it inherently precludes real-time understanding and response. In practical, safety- and time-critical scenarios, such as assistive navigation [\[15\]](#page-8-4), sign language interpretation [\[6\]](#page-8-5), and live video description [\[8\]](#page-8-6), continuous perceptual feedback is essential. Systems that rely on offline processing cannot react promptly to dynamic changes in the environment, limiting their usability in real-world deployments.

To address this, recent studies [\[8,](#page-8-6) [28,](#page-9-1) [43\]](#page-9-2) have attempted to extend MLLMs into a streaming paradigm. However, most of these approaches are essentially interleaved: they alternately process a segment of input and then generate a segment of output. Although this reduces latency compared to fully offline inference, it still behaves like "mini-batch offline" processing and fails to achieve true real-time interaction. For example, in assistive navigation for the visually impaired, the system may be generating a long descriptive output about the next steps to take. If a sudden obstacle or danger appears during this period, an interleaved streaming MLLM, which performs perception and generation alternately rather than concurrently, may fail to detect the hazard in time. Such behavior is clearly unacceptable in safetycritical applications. These limitations stem from the fundamental constraints of decoder-only architectures [\[5,](#page-8-7) [10,](#page-8-8) [40\]](#page-9-3), which are not designed for simultaneous encoding and decoding. Although encoder–decoder architectures [\[16,](#page-8-9) [29\]](#page-9-4) could, in principle, support such parallelism, converting existing MLLMs to this paradigm is highly impractical, as it would require re-establishing large-scale vision–language alignment from scratch.

<sup>∗</sup> Equal contribution. † Corresponding authors.

<span id="page-1-1"></span><span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1. Illustration of different paradigms for video description and positional encoding.The first row shows the offline paradigm, where the model generates the description after observing the entire video, leading to temporal misalignment between narration and visual sequence. The second row presents the interleaved streaming paradigm, which alternates between perception and generation, providing more immediate responses and better temporal coherence, but still suffers from the continuity constraint of positional encoding that prevents full parallelism. The third row illustrates our proposed parallel streaming paradigm, which breaks this continuity, enabling simultaneous perception and generation for true real-time video understanding. The positional IDs serves as a conceptual reference, showing how relaxing positional continuity enables parallel processing between input and output. Please zoom in for a clearer view of details.

We argue that the true bottleneck lies not in the architecture itself, but in the positional encoding design. Current MLLMs enforce a global *continuity* constraint in positional indexing [\[17,](#page-8-1) [32,](#page-9-5) [37\]](#page-9-6). Because future output length is unknown at inference time, the model cannot assign consistent positional indices to incoming inputs while decoding is in progress, thereby preventing concurrent perception and generation. As illustrated in Fig. [1,](#page-1-0) the first row depicts the offline setting for a video description task[\[4\]](#page-8-10), where the model generates the full description after observing the entire video. This often results in temporal misalignment between the narrative and the actual video sequence. For example, the model may start describing "shrimp cooking" even though the shrimp does not appear until the middle of the video. The second row illustrates the interleaved streaming setting, where the description follows the temporal order more naturally, first describing the melting of butter, then the cooking of shrimp, and finally the addition of seasonings. Although this result is more temporally coherent, its latency remains suboptimal because the continuity of positional encoding prevents the model from encoding the next incoming frame until the current text generation is completed.

We observe that such strict positional continuity is not fundamentally required. The essential role of positional encoding is to capture *relative relationships* among tokens, rather than to impose a single, globally continuous index space [\[33,](#page-9-7) [34\]](#page-9-8). This insight allows us to decouple positional assignments across input and output streams while preserving the relational structure necessary for multimodal alignment. Motivated by this perspective, we introduce a *parallel streaming* paradigm that breaks positional continuity and enables true simultaneous encoding and decoding. As illustrated in the third row of Figure [1,](#page-1-0) our approach allows the model to prefill embeddings for incoming visual frames *during* text generation, achieving real-time synchronization between perception and response.

Specifically, we propose three intuitive positional encoding strategies—Overlapped Streaming Position Encoding (OSPE),Group-Decoupled Position Encoding (GDPE), and Gap-Isolated Position Encoding (GIPE). In the OSPE strategy, the model begins encoding the next video frame concurrently with text decoding, assigning the same initial positional indices to both the current response and the next incoming frame. The GDPE strategy, in contrast, separates the input and output streams, assigning each its own positional group that starts from zero independently. Finally, the GIPE strategy extends the group-based design by adding a large numerical offset between input and output positions, thus creating an explicit separation in index space. We conduct extensive experiments under both offline and streaming inference paradigms on Video Description (VD) and Video Question Answering (VQA) tasks, systematically evaluating the proposed positional encoding strategies from the perspectives of performance, robustness, and acceleration potential. Empirically, we observe that in both offline and streaming settings, the original positional embedding scheme can be replaced by our proposed alternatives with only minimal fine-tuning data, while preserving comparable performance across standard evaluation metrics. In terms of robustness, we introduce scheduling perturbations at test time by disrupting the wait-K policy and find that all three proposed strategies consistently yield more fluent and stable language generation than conventional interleaved encoding. Considering both task accuracy and linguistic coherence, Group-Decoupled Position Encoding <span id="page-2-0"></span>(GDPE) emerges as the most balanced and effective design. Finally, we provide a theoretical analysis showing that parallel streaming enables up to 2× acceleration under balanced input–output workloads. Importantly, this theoretical speedup is broadly applicable to nearly any streaming MLLM, offering a plug-and-play pathway toward faster and truly real-time inference.

Our contributions are as follows:

- We identify the key issue preventing true input–output parallelism in current MLLMs: the unnecessary global continuity constraint of position encoding, and propose a novel and intuitive perspective on positional design.
- We introduce three position encoding strategies that enable true parallelism in streaming tasks, allowing simultaneous encoding and decoding without waiting.
- We systematically validate the proposed positional encoding strategies under both offline and streaming paradigms, demonstrating that GDPE provides the most effective balance between performance and fluency for real-time streaming.

# 2. Related Works

Streaming Large Language Models Most existing multimodal large language models (MLLMs) [\[1,](#page-8-0) [17,](#page-8-1) [20,](#page-9-9) [22](#page-9-10)[–24,](#page-9-11) [35,](#page-9-12) [42,](#page-9-13) [46\]](#page-10-0) follow an offline paradigm, where the model observes the entire video before generating responses. However, this approach faces clear limitations in real-world scenarios. For example, when watching a two-hour movie, users naturally expect interactive, real-time responses rather than delayed answers after viewing the whole video. To address this issue, researchers have begun exploring streaming inference, allowing the model to generate outputs continuously during perception. Leveraging powerful visionlanguage pre-training, many studies adopt an interleaved vision-language design to achieve near real-time understanding and generation[\[8,](#page-8-6) [9,](#page-8-11) [19,](#page-9-14) [28,](#page-9-1) [31,](#page-9-15) [39,](#page-9-16) [41,](#page-9-17) [43\]](#page-9-2). For instance, LiveCC [\[8\]](#page-8-6) densely interleaves video frames with automatic speech recognition (ASR) transcripts, enabling real-time commentary.

As the sequence length increases, such interleaved designs suffer from latency accumulation—since prefill and decoding speeds are inversely proportional to the number of tokens—leading to degraded responsiveness in long-video scenarios. Consequently, several recent works have turned to visual token compression and asynchronous perception–generation to improve efficiency. Flash-VStream [\[43\]](#page-9-2) introduces a Flash Memory module that enables real-time reasoning over extremely long videos, while TimeChat-Online [\[41\]](#page-9-17) reduces up to 80% of visual tokens by exploiting temporal redundancy without breaking positional continuity. ViSpeak [\[13\]](#page-8-12) achieves simultaneous input–output by concatenating generated responses with subsequent perceptual inputs, which inevitably mixes heterogeneous semantics within the same embedding space. In contrast, our method achieves the same goal by redesigning the positional encoding scheme rather than altering the input–output format, thereby preserving the LLM's intrinsic feature space while still enabling real-time interaction.

Streaming Tasks In practical applications, many visionlanguage tasks naturally operate in a streaming fashion, where input data arrives continuously and the system must respond in real time. For example, live video description [\[3,](#page-8-13) [44\]](#page-9-18) requires generating descriptive captions for a video stream on the fly, without access to future frames. Similarly, continuous sign language recognition and translation [\[6,](#page-8-5) [47\]](#page-10-1) demands interpreting a signer's continuous video feed into text or speech as it unfolds. In tasks like realtime object tracking [\[7,](#page-8-14) [14\]](#page-8-15), the model needs to continuously localize and describe a target object's state or trajectory in sequential frames, updating its understanding with each new frame. Another illustrative scenario is interactive streaming video question answering [\[9,](#page-8-11) [38\]](#page-9-19), where an agent must answer user queries about a video in real time. In such a setting, a question may be asked before the relevant visual evidence appears, requiring the model to handle temporal asynchrony and retain context until the answer can be given. All these tasks share the characteristic that the input is continuous and time-sensitive. To evaluate the effectiveness of our approach, we conduct experiments on two representative streaming tasks: video description [\[4\]](#page-8-10) and video question answering (QA) [\[38\]](#page-9-19). These tasks are selected for their natural temporal continuity and ease of adaptation to the streaming setting, which allow us to clearly examine the model's ability to understand partial visual context and produce coherent outputs on the fly.

# 3. Position Encoding Strategies

#### 3.1. Limitations of Continuous Position Encoding

Early models such as the LLaVA series [\[23,](#page-9-20) [24\]](#page-9-11) and MiniGPT-4 [\[45\]](#page-9-21) adopt a uniform 1D positional encoding strategy for both visual and textual tokens, following the original design logic of LLMs. While this simplifies training, it overlooks the fact that visual information possesses unique structural dimensions such as height (H), width (W), and temporal axis (T), which differ from text. As a result, recent works increasingly explore 2D or 3D positional encoding strategies (e.g., Qwen2.5-VL [\[1\]](#page-8-0)), enabling the model to better understand the spatial and temporal relationships among tokens. Despite their promising performance, these position encoding strategies all impose a global continuity constraint: every new token must be assigned a position index that strictly follows the used indices. As a result, the position indices of future visual inputs cannot be determined until all previously generated answer tokens have fin-

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2. Comparison of different position encoding strategies, where  $V_i$  represents the video token sequence from the i-th input clip, and  $A_i$  represents the corresponding textual output token sequence. Arrows denote the source dependency for the first generated token of each textual output segment. (a) **Previous Position Encoding**: assigns consecutive positions strictly following the interleaved video-text streaming order; (b) **Overlapped Streaming Position Encoding (OSPE)**: enables video-text streaming parallelism by allowing temporal overlap between encoding and decoding; (c) **Group-Decoupled Position Encoding (GDPE)**: divides video and text into independent groups that maintain intra-group continuity while being inter-group decoupled; (d) **Gap-Isolated Position Encoding (GIPE)**: introduces a fixed gap between groups to fully isolate their index spaces and further reduce cross-modal interference.

ished decoding. Fig. 2(a) illustrates this position encoding paradigm, where  $V_i$  denotes the i-th round of visual input with  $m_i$  visual tokens, and  $A_i$  represents the corresponding textual answer with  $k_i$  text tokens.  $E_i$  indicates the ending token index of either the visual input or textual output in the i-th round. It can be observed that the indexing of the next input or output depends on knowing the length of  $m_i$  or  $k_i$  from the previous round. This creates a hard coupling between prefilling and decoding, forcing the model to alternate between input and output in a strictly sequential manner rather than processing them in parallel.

In summary, continuity in position encoding is the *primary obstacle* preventing streaming MLLMs from achieving real-time interaction. To overcome this issue, we revisit the design of position index allocation and propose a unified framework that relaxes global continuity while preserving intra-modal ordering. We propose three intuitive position encoding strategies: (1) Overlapped Streaming Position Encoding (OSPE), (2) Group-Decoupled Position Encoding (GDPE), and (3) Gap-Isolated Position Encoding (GIPE), which provide alternative ways to relax global continuity and thereby enable genuine input—output parallelism in streaming environments. For illustrative purposes, we describe our methods using a standard 1D positional indexing scheme as a running example.

#### 3.2. Overlapped Streaming Position Encoding

Due to the limitation of position encoding strategies, video segments V and answer tokens A in the previous paradigm

are strictly interleaved. The most intuitive way to break the continuity is to allow the model to continue ingesting  $V_{i+1}$  while generating  $A_i$ , as if  $A_i$  did not occupy additional index space as shown in Fig. 2b. In this case, both  $A_i$  and  $V_{i+1}$  share the same starting position ID, denoted as  $E_i+1$ . The next pair,  $A_{i+1}$  and  $V_{i+2}$ , then start from one greater than the maximum of the end positions of  $A_i$  and  $V_{i+1}$ . In most cases, by the time the model starts generating  $A_{i+1}$ ,  $A_i$  has already been completed, since the number of text tokens is usually much smaller than that of visual tokens.

For subsequent rounds, the same rule applies. The starting index of both  $A_{i+1}$  and  $V_{i+2}$  is assigned as one greater than the maximum  $E_{i+1}$  of the end indices of  $A_i$  and  $V_{i+1}$ :

$$E_{i+1} = \max(E_i + m_{i+1}, E_{i+1} + k_i), \tag{1}$$

where  $E_{i+1} + k_i$  and  $E_i + m_{i+1}$  denote the ending indices of  $A_i$  and  $V_{i+1}$ , respectively. Here,  $k_i$  and  $m_{i+1}$  are the numbers of text tokens and visual tokens in the *i*-th and (i+1)-th rounds. This update rule generalizes the OSPE strategy across all rounds, preserving intra-modal ordering while eliminating the global continuity constraint, thereby enabling true parallel streaming.

#### 3.3. Group-Decoupled Position Encoding

Fig. 2(c) illustrates another possible solution, which divides the entire sequence into two independent groups: one for visual inputs and one for textual outputs. Within each group, position indices are assigned continuously, while continuity across groups is removed. This allows new visual inputs <span id="page-4-1"></span>to be indexed independently of the textual generation process, effectively decoupling perception and language in the positional space. In practice, each newly received visual segment Vi+1 is indexed based only on the end position of the previous visual segment V<sup>i</sup> , and each newly generated answer Ai+1 is indexed based only on the end position of the previous answer A<sup>i</sup> :

$$E_{vi+1} = E_{vi} + m_{i+1},$$
  

$$E_{ai+1} = E_{ai} + k_{i+1},$$
(2)

where mi+1 and ki+1 denote the numbers of visual and text tokens in the (i+1)-th round, respectively.

<span id="page-4-0"></span>![](_page_4_Figure_3.jpeg)

Figure 3. Causal mask visualization. (*left*) Casual mask for previous video-text interleaved streaming paradigm. (*right*) Casual mask for parallel streaming paradigm.

Although each textual output A<sup>i</sup> must still attend to the visual input V<sup>i</sup> within the same round, their positional indices reside in separate continuous spaces. This design preserves intra-modal ordering and cross-modal attention while removing inter-modal positional dependency, enabling the model to process visual and textual streams in parallel without violating contextual consistency.

It is worth noting that during training, the input consists of the complete sequences V1, V2, . . . , V<sup>n</sup> and A1, A2, . . . , An, where n denotes all video segments and their corresponding answers. In this process, the causal mask must be carefully set: Vi+1 should only attend to V<sup>1</sup> through V<sup>i</sup> , while A<sup>i</sup> should only attend to V<sup>1</sup> through V<sup>i</sup> and A<sup>1</sup> through A<sup>i</sup> . A visualization of the causal mask is shown in Figure [3.](#page-4-0)

#### 3.4. Gap-Isolated Position Encoding

While the Group-Decoupled Position Encoding (GDPE) removes cross-modal continuity by assigning independent index spaces to visual and textual groups, their index ranges still remain numerically adjacent within the same overall space. Although it is uncertain whether this adjacency introduces any undesired coupling, we propose Gap-Isolated Position Encoding (GIPE) as a more isolated design that inserts a fixed offset between the two index spaces. Formally, after assigning indices to all visual tokens V1, . . . , Vn, the starting index of the first textual token A<sup>1</sup> is ∆ + 1, where ∆ is a constant gap that isolates the two groups in the positional domain. This ensures that all textual tokens occupy an index range strictly separated from that of visual tokens, making the two modalities positionally disjoint. The causal mask configuration of GIPE remains identical to that of GDPE.

# 4. Experiment

### 4.1. Overview

We conduct a comprehensive evaluation of our three continuity-breaking position encoding strategies—OSPE, GDPE, and GIPE—built upon the representative 3D spatiotemporal encoding used in recent state-of-the-art MLLMs such as Qwen2.5-VL [\[1\]](#page-8-0). Experiments are performed on two tasks, video description and video question answering, to examine how different positional designs affect real-time multimodal understanding.

### 4.2. Tasks

(a) Streaming Video Description. In the streaming scenario, the Video Description task aims to generate natural language descriptions for continuously incoming video streams. Unlike traditional offline captioning, the model must comprehend partial visual context and produce temporally coherent captions on the fly, reflecting real-world applications such as live narration and visual assistance, where minimizing perceptual delay is essential. We adapt the *PE Video Dataset*[\[4\]](#page-8-10), which was originally developed for offline video perception. The PE Video contains high-quality videos with rich motion dynamics and human-refined captions, making it suitable for streaming scenarios.

(b) Streaming Video QA. In the streaming setting, the Video QA task requires the model to answer questions based on continuously arriving video frames rather than full offline clips. The model must reason over partial and evolving visual context, making timely evidence integration essential. We adapt the *FunQA* dataset [\[38\]](#page-9-19), which provides diverse human-annotated videos QA pair. It consists of three subsets: *HumorQA*, *CreativeQA*, and *MagicQA*. For each subset, we evaluate two task types: video description Q&A and counterintuitive reasoning Q&A. This results in six distinct streaming Video QA sub-tasks, allowing us to comprehensively assess the model's ability to perform diverse reasoning under streaming conditions.

### 4.3. Metric

For both PE-Video and FunQA tasks, we follow the standard evaluation metrics widely adopted in video captioning and question-answering, including CIDEr [\[36\]](#page-9-22), BLEU [\[27\]](#page-9-23),

<span id="page-5-1"></span><span id="page-5-0"></span>Table 1. Video Description (VD) and VQA on Qwen2.5-VL. Metrics: CIDEr, BLEU-1, BLEU-4, METEOR, ROUGE-L, BLEURT, and Fluency (higher is better). *VD* denotes video description task and *VQA* denotes video QA task. For the video QA task, we evaluate the model across all six subsets, but report only the average performance here. Detailed per-subset results are provided in the Appendix.

| Category            | Method                      | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT | Fluency |  |  |  |  |
|---------------------|-----------------------------|-------|--------|--------|--------|---------|--------|---------|--|--|--|--|
|                     | Video Description (VD) task |       |        |        |        |         |        |         |  |  |  |  |
| Offline             | Origin                      | 35.44 | 42.36  | 14.45  | 29.18  | 30.47   | 53.21  | 4.84    |  |  |  |  |
|                     | GDPE                        | 30.86 | 40.26  | 13.64  | 28.49  | 34.12   | 53.19  | 4.93    |  |  |  |  |
| Streaming           | Interleave                  | 20.08 | 44.40  | 14.41  | 27.17  | 34.95   | 44.11  | 2.84    |  |  |  |  |
|                     | OSPE                        | 26.32 | 42.14  | 12.78  | 27.92  | 32.29   | 50.62  | 4.48    |  |  |  |  |
|                     | GDPE                        | 12.52 | 26.32  | 7.42   | 30.03  | 27.37   | 51.53  | 4.56    |  |  |  |  |
|                     | GIPE                        | 28.11 | 40.42  | 11.52  | 29.13  | 30.69   | 51.20  | 4.85    |  |  |  |  |
| Video QA (VQA) task |                             |       |        |        |        |         |        |         |  |  |  |  |
| Offline             | Origin                      | 6.98  | 34.47  | 4.74   | 19.67  | 22.43   | 41.34  | 4.70    |  |  |  |  |
|                     | GDPE                        | 7.25  | 35.23  | 5.13   | 19.50  | 22.98   | 42.04  | 4.52    |  |  |  |  |
| Streaming           | Interleave                  | 3.00  | 21.95  | 1.71   | 13.03  | 18.00   | 31.22  | 3.72    |  |  |  |  |
|                     | OSPE                        | 4.22  | 33.40  | 3.68   | 20.23  | 19.61   | 37.38  | 3.98    |  |  |  |  |
|                     | GDPE                        | 3.22  | 31.32  | 3.32   | 21.82  | 18.96   | 41.16  | 4.13    |  |  |  |  |
|                     | GIPE                        | 3.99  | 30.95  | 2.48   | 17.58  | 19.33   | 37.25  | 4.61    |  |  |  |  |

METEOR [\[2\]](#page-8-16), and ROUGE [\[21\]](#page-9-24). To better capture the semantic fidelity between generated and reference texts, we further include BLEURT [\[30\]](#page-9-25) as a sentence-level quality metric, which measures contextual similarity beyond surface n-gram overlap. However, these automatic metrics still fail to reflect the human-perceived fluency and readability of streaming outputs. Therefore, we additionally employ an LLM-as-Judge evaluation [\[18\]](#page-9-26), where GPT-5 [\[26\]](#page-9-27) assesses each generated sentence from a human-like perspective. Specifically, the model rates linguistic fluency on a 1–5 scale, with higher scores indicating more natural, coherent, and well-structured expressions. The detailed prompt design is provided in the supplementary material.

#### 4.4. Baseline and Experimental Setup

We adopt *Qwen2.5-VL* as the baseline in our experiments, which employs explicit three-dimensional positional encoding (x, y, t) for visual tokens, enabling the model to perceive both spatial structures and temporal dynamics. For textual tokens, the three positional dimensions are kept identical, ensuring consistent positional representation across modalities. This 3D positional design allows the model to jointly reason over spatial, temporal, and semantic contexts within a unified embedding space.

We adopt a *streaming* evaluation setting based on a fixed *wait-*K policy: at test time the model consumes one frame and emits exactly K = 3 tokens, matching the average frame–token ratio (≈ 3) observed in PE-Video and FunQA. Unless otherwise specified, all models are trained and evaluated under this default *wait-*K = 3 configuration. To ensure a fair comparison, all streaming variants share identical data, optimization settings, and temporal pacing.

Following the sampling protocol of *Qwen2.5-VL*, we set the frame rate to 2 fps. Videos shorter than 5 seconds or longer than 30 seconds are removed. For each sample from PE-Video or FunQA, we compute the number of text tokens L in its caption/answer and divide it by the video duration Tvid to obtain the average tokens per second. Let M = Tvid × K denote the expected caption length under the *wait-*K setting. We discard samples where the response length L is smaller than M (insufficient supervision) or more than twice M, since extremely long captions lead to most tokens being emitted at the final frame, causing the generation to behave like offline rather than streaming. Finally, we randomly select 20K samples for training. *More details such as dataset examples and additional experimental results are included in the supplementary material.*

#### 4.5. Performance Analysis

Table [1](#page-5-0) summarizes the performance of all methods under *Offline* and *Streaming* settings. Within the Offline category, the Origin model, which fine-tunes Qwen2.5-VL using its native positional encoding, achieves strong results across both Video Description and Video QA. The Offline-GDPE variant replaces the original positional encoding with a GDPE-style layout while keeping the decoding pro-

<span id="page-6-0"></span>Table 2. BLEURT of video description under scheduling disturbance. Models are trained with fixed wait-K=3 and evaluated under both fixed wait-K=3 and test-time Random schedules.

| Setting  | Interleave | OSPE  | GDPE  | GIPE  |
|----------|------------|-------|-------|-------|
| 3→3      | 44.11      | 50.62 | 51.53 | 51.20 |
| 3→Random | 40.56      | 50.71 | 51.76 | 51.56 |

cess fully offline. Its overall performance remains close to that of Origin, indicating that modifying the positional layout alone does not fundamentally disrupt the pretrained visual–language alignment, and that such changes can be successfully compensated through limited fine-tuning.

In the Streaming category, the Interleave model which using native positional encoding shows a severe degradation in linguistic fluency compared with both Offline variants. This degradation arises because visual frames are inserted inside the ongoing text sequence, forcing the model to alternate between writing a partial sentence and processing new visual tokens. As a result, the next generated words no longer attend directly to the preceding text token but first encounter the inserted visual tokens in the attention path. This fragmentation disrupts sentence continuity and leads to substantial drops in fluency-sensitive metrics such as BLEURT, revealing that the interleaving mechanism compromises the continuity and readability of the generated text.

In contrast, our continuity-breaking strategies overcome this issue by restructuring the attention order between input and output tokens, as illustrated in Fig. [3,](#page-4-0) ensuring that visual tokens never interrupt the ongoing textual sequence. Among the three, OSPE resumes each textual segment from the maximum position index of the previous stage's text output and the current stage's visual input, which yields uninterrupted text segments with non-contiguous position indices. GDPE and GIPE enforce an independent and strictly continuous index space for input tokens and output tokens.

Building upon these properties, despite altering the native positional encoding and inference paradigm, the three continuity-breaking strategies achieve competitive performance across the two tasks. Among them, OSPE produces lower BLEURT and fluency scores than GDPE, which is consistent with its non-contiguous index updates that limit coherence. GIPE, on the other hand, benefits from the clear separation between input and output numerical position, as well as the minimized interaction distance between words, allowing it to reach fluency levels close to those of the offline models. However, its ability to capture key semantic content is slightly weaker than GDPE. Considering linguistic quality, GDPE offers the most balanced overall performance and therefore represents the most promising default configuration for future streaming applications.

<span id="page-6-1"></span>![](_page_6_Picture_6.jpeg)

Figure 4. Example of the generated caption by *Interleave* under random scheduling. Duplicated, fragmented, and grammatically broken segments are highlighted in yellow, while correctly recognized key objects and actions are highlighted in red.

### 4.6. Robustness under Scheduling Disturbance

In real streaming scenarios, video frames and user tokens rarely arrive in a perfectly regular pattern. Multiple frames may be buffered together, responses can be delayed, or the emission rate may fluctuate over time. To simulate such irregular behaviors, we train all models with a fixed wait-K=3 configuration and evaluate them under both the same fixed schedule and a test-time Random schedule, where the number of emitted tokens per step is randomly perturbed. Due to BLEURT's sensitivity to sentence-level coherence, we rely on it to assess the impact of scheduling disturbance on streaming generation.

The results in Table [2](#page-6-0) show that the three continuitybreaking strategies remain stable across settings, whereas Interleave experiences a clear drop under the Random schedule. To further illustrate how scheduling disturbance affects generation, we additionally examine representative outputs together with the fluency evaluation. As shown in Fig. [4,](#page-6-1) Interleave frequently produces duplicated, fragmented, or abruptly truncated phrases when evaluated under random scheduling. These failures arise from the repeated alternation between text generation and visual prefilling, which interrupts sentence progression and causes the model to lose track of its prior context. This qualitative breakdown aligns with the fluency results in Fig. [5,](#page-7-0) where Interleave exhibits a pronounced decline under the Random schedule, far larger than that observed for our continuity-breaking strategies. Taken together, these analyses show that Interleave is highly vulnerable to scheduling disturbance, whereas our methods maintain stable and readable outputs even under irregular emission patterns by preserving an uninterrupted textual index space.

#### 4.7. Theoretical Latency and Speedup Analysis

In the previous experiments, we have demonstrated that the proposed OSPE, GDPE, and GIPE strategies maintain stable performance under streaming conditions. Beyond

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5. LLM-as-Judge fluency under scheduling disturbance. The two colors correspond to: (1) trained and evaluated under fixed wait-K=3, and (2) trained with wait-K=3 but evaluated under random scheduling (disturbance setting).

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 6. Theoretical latency and speedup analysis. (a) Parallel streaming overlaps perception and generation to reduce total step latency. (b) The achievable speedup peaks when perception and generation workloads are balanced  $(r \approx 1)$ . Please zoom in for a clearer view of details.

their accuracy, their core advantage lies in enabling parallel perception and generation by breaking the global positional continuity between input and output tokens, thereby substantially reducing end-to-end latency. This subsection further provides a theoretical analysis of the acceleration achieved through such parallelization.

Assume that the entire streaming process consists of N time steps. At each step i, the model receives  $m_i$  visual tokens (perception stage) and generates  $k_i$  textual tokens (generation stage). Let  $R_v$  and  $R_t$  denote the visual processing throughput and text decoding throughput (tokens per second), respectively.

**Interleaved Streaming (Conventional Paradigm).** The total latency for the *i*-th step can be expressed as:

$$T_{\text{interleave},i} = \frac{m_i}{R_v} + \frac{k_i}{R_t}.$$
 (3)

The overall latency across N steps accumulates as:

$$T_{\text{interleave}} = \sum_{i=1}^{N} \left( \frac{m_i}{R_v} + \frac{k_i}{R_t} \right), \tag{4}$$

which implies that each stage must wait until the previous one finishes before proceeding to the next, resulting in strictly serialized perception—generation cycles.

**Parallel Streaming (Our Paradigm).** our OSPE, GDPE, and GIPE strategies allow the model to prefetch visual tokens for the (i+1)-th segment while simultaneously generating textual outputs for the i-th step. Accordingly, the latency per step under ideal parallelization becomes:

$$T_{\text{parallel},i} = \max\left(\frac{m_i}{R_v}, \frac{k_i}{R_t}\right),$$
 (5)

which is evidently smaller than the conventional paradigm. In practice, this formulation can be efficiently implemented on two separate GPUs or computational streams, where the prefill stage and the decode stage operate in parallel with minimal synchronization overhead.

To further quantify the theoretical acceleration, we define the per-step speedup ratio as

$$S_i = \frac{T_{\text{interleave},i}}{T_{\text{parallel},i}} = \frac{\frac{m_i}{R_v} + \frac{k_i}{R_t}}{\max\left(\frac{m_i}{R_v}, \frac{k_i}{R_t}\right)}.$$
 (6)

Let  $r=\frac{m_i/R_v}{k_i/R_t}$  denote the workload ratio between perception and generation, where  $r\gg 1$  indicates vision (input)-dominated latency and  $r\ll 1$  corresponds to text (output)-dominated latency. The relationship between the speedup S and workload ratio r is illustrated in Fig. 6.

This trend can be clearly observed across different tasks. In *video description* tasks, the model processes long video inputs but generates relatively short textual outputs (i.e.,  $r\gg 1$ ), resulting in a vision-dominated runtime and only moderate speedup. In contrast, *video chain-of-thought* (*Video-CoT*) involves both extensive perception and long-form reasoning outputs ( $r\approx 1$ ), placing it near the balanced regime of Fig. 6 and leading to the highest acceleration, where the per-step latency is reduced by nearly half compared with the interleaved baseline.

Overall, the achievable speedup is bounded by approximately  $2\times$  when perception and generation workloads are balanced, whereas the latency asymptotically approaches the perception-only limit as r increases.

# 5. Conclusion

In this work, we revisit the positional encoding design of Multimodal Large Language Models (MLLMs) and reveal that the global positional continuity constraint is the key obstacle to achieving real-time parallel perception and generation. We propose three continuity-breaking strategies, namely Overlapped, Group-Decoupled, and Gap-Isolated positional encodings, which enable simultaneous input and output without altering the model architecture. Extensive experiments demonstrate that the Group-Decoupled strategy (GDPE) achieves the best balance between efficiency, temporal coherence, and robustness, significantly reducing response latency while maintaining comparable accuracy to offline models. Beyond empirical validation, our theoretical analysis confirms that relaxing positional continuity allows genuine "speak-whilewatching" capability, achieving up to 2× theoretical acceleration under balanced perception–generation workloads.

Future Work. Future research can be explored in the following directions: (1) Task-specific parallel scheduling: develop adaptive scheduling strategies tailored to different tasks, enabling the model to dynamically balance performance and latency; (2) Unified streaming framework: extend the proposed streaming strategies to other modalities such as visual generation, action, and multimodal interaction, forming a unified framework for real-time reasoning; (3) Hardware-level parallel optimization: leverage parallel pipelines and multi-GPU execution to further reduce end-to-end latency. Through these directions, we view input–output decoupling not merely as a speedup trick, but as a general design principle for future multimodal systems. Extending this idea beyond video to generation, action, and embodied interaction could enable a new generation of MLLMs that reason continuously over the world while speaking, listening, and acting in real time.

# References

- <span id="page-8-0"></span>[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [1,](#page-0-0) [3,](#page-2-0) [5](#page-4-1)
- <span id="page-8-16"></span>[2] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In *ACL*, pages 65–72, 2005. [6](#page-5-1)
- <span id="page-8-13"></span>[3] Eduardo Blanco-Fernandez, Carlos Guti ´ errez- ´ Alvarez, Na- ´ dia Nasri, Saturnino Maldonado-Bascon, and Roberto J ´ Lopez-Sastre. Live video captioning. ´ *Multimedia Tools and Applications*, pages 1–33, 2025. [3](#page-2-0)
- <span id="page-8-10"></span>[4] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder:

- The best visual embeddings are not at the output of the network. *arXiv preprint arXiv:2504.13181*, 2025. [2,](#page-1-1) [3,](#page-2-0) [5,](#page-4-1) [12](#page-11-0)
- <span id="page-8-7"></span>[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In *NeurIPS*, pages 1877–1901, 2020. [1](#page-0-0)
- <span id="page-8-5"></span>[6] Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, and Richard Bowden. Sign language transformers: Joint endto-end sign language recognition and translation. In *CVPR*, pages 10023–10033, 2020. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-8-14"></span>[7] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. In *CVPR*, pages 9686– 9696, 2023. [3](#page-2-0)
- <span id="page-8-6"></span>[8] Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. Livecc: Learning video llm with streaming speech transcription at scale. In *CVPR*, pages 29083– 29095, 2025. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-8-11"></span>[9] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, Hao Jiang, et al. Streaming video question-answering with in-context video kv-cache retrieval. In *ICLR*, 2025. [3](#page-2-0)
- <span id="page-8-8"></span>[10] Abhimanyu Dubey, Abhinav Jauhri, Abhishek Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv e-prints*, page arXiv:2407.123xx, 2024. [1](#page-0-0)
- <span id="page-8-2"></span>[11] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. In *NeurIPS*, pages 89098–89124, 2024. [1](#page-0-0)
- <span id="page-8-3"></span>[12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In *CVPR*, pages 24108–24118, 2025. [1](#page-0-0)
- <span id="page-8-12"></span>[13] Shenghao Fu, Qize Yang, Yuan-Ming Li, Yi-Xing Peng, Kun-Yu Lin, Xihan Wei, Jian-Fang Hu, Xiaohua Xie, and Wei-Shi Zheng. Vispeak: Visual instruction feedback in streaming videos. *arXiv preprint arXiv:2503.12769*, 2025. [3](#page-2-0)
- <span id="page-8-15"></span>[14] Anfeng He, Chong Luo, Xinmei Tian, and Wenjun Zeng. A twofold siamese network for real-time object tracking. In *CVPR*, pages 4834–4843, 2018. [3](#page-2-0)
- <span id="page-8-4"></span>[15] Bineeth Kuriakose, Raju Shrestha, and Frode Eika Sandnes. Deepnavi: A deep learning based smartphone navigation assistant for people with visual impairments. *Expert Systems with Applications*, 212:118720, 2023. [1](#page-0-0)
- <span id="page-8-9"></span>[16] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-tosequence pre-training for natural language generation, translation, and comprehension. In *ACL*, pages 7871–7880, 2020. [1](#page-0-0)
- <span id="page-8-1"></span>[17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Zi-

- wei Liu, et al. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024. [1,](#page-0-0) [2,](#page-1-1) [3](#page-2-0)
- <span id="page-9-26"></span>[18] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-as-judges: A comprehensive survey on llm-based evaluation methods. *arXiv preprint arXiv:2412.05579*, 2024. [6,](#page-5-1) [12](#page-11-0)
- <span id="page-9-14"></span>[19] Wei Li, Bing Hu, Rui Shao, Leyang Shen, and Liqiang Nie. Lion-fs: Fast & slow video-language thinker as online video assistant. In *CVPR*, pages 3240–3251, 2025. [3](#page-2-0)
- <span id="page-9-9"></span>[20] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In *CVPR*, pages 26763–26773, 2024. [3](#page-2-0)
- <span id="page-9-24"></span>[21] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In *ACL*, pages 74–81, 2004. [6](#page-5-1)
- <span id="page-9-10"></span>[22] Dongyang Liu, Renrui Zhang, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. *arXiv preprint arXiv:2402.05935*, 2024. [3](#page-2-0)
- <span id="page-9-20"></span>[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *NeurIPS*, pages 34892–34916, 2023. [3](#page-2-0)
- <span id="page-9-11"></span>[24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *CVPR*, pages 26296–26306, 2024. [3](#page-2-0)
- <span id="page-9-0"></span>[25] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. *arXiv preprint arXiv:2311.16103*, 2023. [1](#page-0-0)
- <span id="page-9-27"></span>[26] OpenAI. Gpt-5 system card, 2025. [6,](#page-5-1) [12](#page-11-0)
- <span id="page-9-23"></span>[27] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: A method for automatic evaluation of machine translation. In *ACL*, pages 311–318, 2002. [5](#page-4-1)
- <span id="page-9-1"></span>[28] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. In *NeurIPS*, pages 119336–119360, 2024. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-9-4"></span>[29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140):1–67, 2020. [1](#page-0-0)
- <span id="page-9-25"></span>[30] Thibault Sellam, Dipanjan Das, and Ankur Parikh. Bleurt: Learning robust metrics for text generation. In *ACL*, pages 7881–7892, 2020. [6](#page-5-1)
- <span id="page-9-15"></span>[31] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In *CVPR*, pages 18221–18232, 2024. [3](#page-2-0)
- <span id="page-9-5"></span>[32] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In *ICLR*, 2024. [2](#page-1-1)

- <span id="page-9-7"></span>[33] Junlong Tong, Yingqi Fan, Anhao Zhao, Yunpu Ma, and Xiaoyu Shen. Streamingthinker: Large language models can think while reading. *arXiv preprint arXiv:2510.17238*, 2025. [2](#page-1-1)
- <span id="page-9-8"></span>[34] Junlong Tong, Jinlan Fu, Zixuan Lin, Yingqi Fan, Anhao Zhao, Hui Su, and Xiaoyu Shen. Llm as effective streaming processor: Bridging streaming-batch mismatches with group position encoding. *arXiv preprint arXiv:2505.16983*, 2025. [2](#page-1-1)
- <span id="page-9-12"></span>[35] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In *NeurIPS*, pages 87310–87356, 2024. [3](#page-2-0)
- <span id="page-9-22"></span>[36] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In *CVPR*, pages 4566–4575, 2015. [5](#page-4-1)
- <span id="page-9-6"></span>[37] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In *ECCV*, pages 396–416. Springer, 2024. [2](#page-1-1)
- <span id="page-9-19"></span>[38] Binzhu Xie, Sicheng Zhang, Zitang Zhou, Bo Li, Yuanhan Zhang, Jack Hessel, Jingkang Yang, and Ziwei Liu. Funqa: Towards surprising video comprehension. In *ECCV*, pages 39–57. Springer, 2024. [3,](#page-2-0) [5,](#page-4-1) [12](#page-11-0)
- <span id="page-9-16"></span>[39] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. Streaming video understanding and multi-round interaction with memoryenhanced knowledge. *arXiv preprint arXiv:2501.13468*, 2025. [3](#page-2-0)
- <span id="page-9-3"></span>[40] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025. [1](#page-0-0)
- <span id="page-9-17"></span>[41] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In *ACM MM*, pages 10807– 10816, 2025. [3](#page-2-0)
- <span id="page-9-13"></span>[42] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. *arXiv preprint arXiv:2408.04840*, 2024. [3](#page-2-0)
- <span id="page-9-2"></span>[43] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. *arXiv preprint arXiv:2406.08085*, 2024. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-9-18"></span>[44] Xingyi Zhou, Anurag Arnab, Shyamal Buch, Shen Yan, Austin Myers, Xuehan Xiong, Arsha Nagrani, and Cordelia Schmid. Streaming dense video captioning. In *CVPR*, pages 18243–18252, 2024. [3](#page-2-0)
- <span id="page-9-21"></span>[45] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. *arXiv preprint arXiv:2304.10592*, 2023. [3](#page-2-0)

- <span id="page-10-0"></span>[46] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. *arXiv preprint arXiv:2504.10479*, 2025. [1,](#page-0-0) [3](#page-2-0)
- <span id="page-10-1"></span>[47] Ronglai Zuo, Fangyun Wei, and Brian Mak. Towards online continuous sign language recognition and translation. In *EMNLP*, pages 11050–11067, 2024. [3](#page-2-0)

# Appendix

# <span id="page-11-0"></span>A. Overview

In this supplementary material, we provide: (1) the full prompt used for LLM-as-Judge fluency evaluation [\[18,](#page-9-26) [26\]](#page-9-27); (2) concrete input examples from the PE-Video [\[4\]](#page-8-10) and FunQA [\[38\]](#page-9-19) datasets under the streaming protocol; (3) additional results of the 7B backbone on both tasks and (4) funQA sub-task details.

# B. Prompt for Fluency Evaluation

Fig. [7](#page-11-1) shows the system prompt used for LLM-as-Judge (GPT-5) fluency evaluation. The judge receives a single caption and rates *only* its linguistic fluency on a 1–5 scale, returning a JSON dictionary with the score and a short comment. This prompt is used across all settings to ensure consistent evaluation.

```
SYSTEM_PROMPT:
You will be given ONE caption.
Your job is to evaluate ONLY its linguistic fluency and readability.
Scoring:
- Assign an INTEGER score from 1 to 5:
 5 = very fluent, clear, natural
 4 = mostly fluent, minor awkwardness
 3 = understandable but some noticeable issues
 2 = quite awkward or hard to read
 1 = very poor fluency, broken or confusing
Response format (JSON ONLY):
{
 "score": 1-5,
 "why": "short fluency comment"
}
(IMPORTANT: Return JSON ONLY. No explanations outside JSON.)
USER_PROMPT:
Caption: {caption}
```

Figure 7. Full prompt used for LLM-as-Judge fluency evaluation. The judge model receives the task description, the ground-truth caption, and the model output, and then assigns a fluency score from 1 to 5 together with a brief justification.

# C. Streaming Input Examples on PE-Video and FunQA

To better illustrate our streaming protocol, we provide a representative example for each task we test.

PE-Video (Streaming Video Description). Fig. [8](#page-11-2) shows a PE-Video example. The ground-truth captions in this dataset are high-quality and often rely heavily on finegrained temporal cues, making the task naturally compat-

<span id="page-11-2"></span>![](_page_11_Picture_10.jpeg)

Figure 8. PE-Video streaming input example. The model receives frames step-by-step and must produce the caption as the video unfolds.

<span id="page-11-3"></span>![](_page_11_Figure_12.jpeg)

Figure 9. FunQA streaming input example. The question is fixed, while the video evidence arrives over time and must be integrated incrementally.

ible with a streaming formulation where the model must describe the video as frames arrive.

FunQA (Streaming Video QA). Fig. [9](#page-11-3) shows a FunQA sample. Unlike multiple-choice QA, FunQA requires openended, descriptive answers that explain the underlying visual phenomena. This makes its output form closely aligned with PE-Video captions, enabling a consistent streaming setup where the model integrates incoming frames to produce a free-form answer.

# D. Additional Results of the 7B Backbone on Video Description

To assess how our positional strategies scale with model capacity, Table [3](#page-12-0) presents video description results for the 7B Qwen2.5-VL backbone under both offline and streaming settings.

Scaling the backbone from 3B to 7B yields a pronounced increase in CIDEr, while BLEU, METEOR, ROUGE-L, and BLEURT improve by similar margins across all methods. This behavior is expected: CIDEr strongly rewards the recall of salient content words, which larger models capture more reliably, whereas the other metrics remain relatively stable once a reasonable descriptive quality is achieved. Crucially, the relative ranking and overall behav-

Table 3. Video Description results on the Qwen2.5-VL backbone (3B and 7B).

<span id="page-12-0"></span>

| Category  | Method     | Model Size | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|------------|-------|--------|--------|--------|---------|--------|
|           | Origin     | 3B         | 35.44 | 42.36  | 14.45  | 29.18  | 30.47   | 53.21  |
| Offline   | GDPE       | 3B         | 30.86 | 40.26  | 13.64  | 28.49  | 34.12   | 53.19  |
|           | Origin     | 7B         | 42.42 | 40.43  | 12.46  | 27.79  | 32.72   | 53.06  |
|           | GDPE       | 7B         | 38.13 | 39.58  | 11.97  | 27.20  | 31.58   | 52.63  |
|           | Interleave | 3B         | 20.08 | 44.40  | 14.41  | 27.17  | 34.95   | 44.11  |
|           | OSPE       | 3B         | 26.32 | 42.14  | 12.78  | 27.92  | 32.29   | 50.62  |
|           | GDPE       | 3B         | 12.52 | 26.32  | 7.42   | 30.03  | 27.37   | 51.53  |
| Streaming | GIPE       | 3B         | 28.11 | 40.42  | 11.52  | 29.13  | 30.69   | 51.20  |
|           | Interleave | 7B         | 46.94 | 49.02  | 16.13  | 32.24  | 36.29   | 44.78  |
|           | OSPE       | 7B         | 47.49 | 43.85  | 12.10  | 28.05  | 31.86   | 51.71  |
|           | GDPE       | 7B         | 37.78 | 41.01  | 11.25  | 27.48  | 30.52   | 51.18  |
|           | GIPE       | 7B         | 25.70 | 39.09  | 9.85   | 28.71  | 28.82   | 51.16  |

iors of all positional strategies remain consistent between 3B and 7B, indicating that our streaming formulations transfer well across model sizes and maintain their effectiveness at larger scales.

# E. FunQA Sub-task Details

The FunQA dataset contains 12 sub-tasks covering diverse video understanding and reasoning capabilities. In this work, we focus on the six Description & Reasoning tasks: Humor (H2, H3), Creative (C2, C3), and Magic (M2, M3). In the main paper, we report the average performance across these six sub-tasks to provide a concise and unified summary of the model's overall behavior. In this appendix, we further present the detailed per-task results for all six Description & Reasoning tasks. All tables in this section follow the same experimental settings as in the main paper (identical wait-K configuration, sampling strategy, and evaluation protocol). The complete results are provided in Tables [4–](#page-13-0)[9.](#page-14-0)

Table 4. FunQA M2 task performance on Qwen2.5-VL-3B.

<span id="page-13-0"></span>

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 11.48 | 40.54  | 5.93   | 25.23  | 25.70   | 47.33  |
|           | GDPE       | 11.75 | 41.30  | 6.46   | 25.54  | 25.93   | 47.47  |
| Streaming | Interleave | 7.48  | 41.62  | 4.04   | 20.06  | 24.03   | 40.59  |
|           | OSPE       | 1.96  | 31.06  | 3.79   | 26.55  | 21.61   | 41.16  |
|           | GDPE       | 5.77  | 34.66  | 3.19   | 21.92  | 22.12   | 45.47  |
|           | GIPE       | 4.48  | 35.02  | 4.62   | 24.38  | 22.48   | 42.04  |

Table 5. FunQA M3 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 11.61 | 37.26  | 7.02   | 22.95  | 24.15   | 41.69  |
|           | GDPE       | 6.69  | 31.46  | 2.75   | 19.61  | 20.06   | 41.85  |
| Streaming | Interleave | 4.01  | 24.94  | 1.67   | 15.97  | 16.89   | 33.66  |
|           | OSPE       | 0.27  | 16.85  | 1.07   | 19.55  | 13.19   | 35.82  |
|           | GDPE       | 2.86  | 21.16  | 0.82   | 17.28  | 15.18   | 40.43  |
|           | GIPE       | 2.47  | 23.22  | 1.85   | 19.74  | 16.24   | 34.04  |

Table 6. FunQA H2 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 10.26 | 38.40  | 4.80   | 21.07  | 22.91   | 39.12  |
|           | GDPE       | 13.04 | 40.29  | 5.68   | 20.95  | 23.41   | 39.66  |
| Streaming | Interleave | 4.18  | 17.30  | 1.62   | 9.04   | 13.12   | 26.68  |
|           | OSPE       | 3.60  | 30.40  | 3.10   | 22.62  | 19.90   | 36.21  |
|           | GDPE       | 8.81  | 37.61  | 4.71   | 20.74  | 22.60   | 39.77  |
|           | GIPE       | 6.94  | 35.09  | 3.82   | 20.81  | 21.59   | 39.22  |

Table 7. FunQA H3 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 4.71  | 36.52  | 4.08   | 17.45  | 20.55   | 39.80  |
|           | GDPE       | 5.03  | 36.39  | 3.69   | 16.55  | 20.54   | 41.64  |
| Streaming | Interleave | 2.09  | 14.00  | 0.76   | 8.11   | 12.33   | 24.88  |
|           | OSPE       | 1.94  | 27.15  | 1.66   | 19.70  | 16.85   | 38.02  |
|           | GDPE       | 3.63  | 32.22  | 1.77   | 15.85  | 17.09   | 41.57  |
|           | GIPE       | 3.42  | 31.55  | 2.26   | 17.97  | 18.59   | 35.47  |

Table 8. FunQA C2 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 2.14  | 22.16  | 2.24   | 15.17  | 20.68   | 34.97  |
|           | GDPE       | 3.38  | 28.65  | 5.91   | 18.13  | 24.56   | 36.17  |
| Streaming | Interleave | 0.21  | 12.57  | 0.46   | 11.46  | 18.70   | 28.41  |
|           | OSPE       | 8.95  | 45.88  | 8.61   | 22.71  | 22.94   | 33.01  |
|           | GDPE       | 0.14  | 28.59  | 2.76   | 15.56  | 20.90   | 35.34  |
|           | GIPE       | 5.74  | 36.92  | 5.19   | 20.25  | 20.88   | 33.58  |

Table 9. FunQA C3 task performance on Qwen2.5-VL-3B.

<span id="page-14-0"></span>

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 1.66  | 31.98  | 4.39   | 16.17  | 20.60   | 39.05  |
|           | GDPE       | 3.58  | 33.31  | 6.28   | 16.23  | 23.40   | 39.62  |
| Streaming | Interleave | 0.02  | 21.29  | 1.70   | 13.56  | 22.95   | 30.99  |
|           | OSPE       | 2.62  | 36.60  | 1.70   | 19.80  | 19.31   | 34.82  |
|           | GDPE       | 2.77  | 31.51  | 1.64   | 14.18  | 18.10   | 37.65  |
|           | GIPE       | 2.28  | 38.60  | 4.34   | 18.24  | 17.87   | 35.02  |