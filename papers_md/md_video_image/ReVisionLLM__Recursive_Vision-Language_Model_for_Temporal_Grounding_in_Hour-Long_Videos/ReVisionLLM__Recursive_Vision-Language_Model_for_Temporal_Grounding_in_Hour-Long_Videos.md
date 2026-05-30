# <span id="page-0-2"></span>ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

Tanveer Hannan<sup>1</sup>,2\* Md Mohaiminul Islam<sup>3</sup> Jindong Gu<sup>4</sup> Thomas Seidl<sup>1</sup>,<sup>2</sup> Gedas Bertasius<sup>3</sup> <sup>1</sup> LMU Munich <sup>2</sup> MCML <sup>3</sup> UNC Chapel Hill <sup>4</sup> University of Oxford

## Abstract

*Large language models (LLMs) excel at retrieving information from lengthy text, but their vision-language counterparts (VLMs) face difficulties with hour-long videos, especially for temporal grounding. Specifically, these VLMs are constrained by frame limitations, often losing essential temporal details needed for accurate event localization in extended video content. We propose ReVisionLLM, a recursive vision-language model designed to locate events in hour-long videos. Inspired by human search strategies, our model initially targets broad segments of interest, progressively revising its focus to pinpoint exact temporal boundaries. Our model can seamlessly handle videos of vastly different lengths—from minutes to hours. We also introduce a hierarchical training strategy that starts with short clips to capture distinct events and progressively extends to longer videos. To our knowledge, ReVision-LLM is the first VLM capable of temporal grounding in hour-long videos, outperforming previous state-of-the-art methods across multiple datasets by a significant margin (e.g., +2.6% R1@0.1 on MAD). The code is available at <https://github.com/Tanveer81/ReVisionLLM>*

## <span id="page-0-1"></span>1. Introduction

Large language models (LLMs) are particularly adept at handling extensive text documents, such as full-length books, and retrieving relevant information [\[1,](#page-8-0) [2,](#page-8-1) [4,](#page-8-2) [21,](#page-8-3) [34\]](#page-9-0). However, achieving similar capabilities in video, i.e., locating fine-grained temporal events in hour-long videos, remains a critical challenge. This task, known as longvideo temporal grounding, requires accurately identifying the start and end of events based on a user's textual query. This capability could be paramount for video content search, sports analytics, surveillance, and many other applications. However, current vision-language models (VLMs) struggle with this demanding task.

Recently, non-LLM-based models [\[15,](#page-8-4) [16,](#page-8-5) [39,](#page-9-1) [42\]](#page-9-2) have made progress in long temporal video grounding. However,

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

Figure 1. Existing vision-language models (VLMs) such as VTimeLLM [\[18\]](#page-8-6) are not equipped to process hour-long videos effectively and struggle to pinpoint precise temporal boundaries for events within extended video durations. In contrast, ReVision-LLM is the first VLM designed to address this limitation, enabling accurate temporal grounding in hour-long video content.

these methods typically involve multiple networks and complex post-processing steps. Additionally, these models generally lack the flexibility to handle textual user instructions. In contrast, the recent VLMs [\[18,](#page-8-6) [19,](#page-8-7) [43,](#page-9-3) [46\]](#page-9-4) can effectively process textual user queries but are ineffective for temporal localization in long videos (Fig. [1\)](#page-0-0). In particular, such VLM-based approaches tend to underperform even on short video (e.g., 2 minutes) localization tasks compared to non-LLM approaches [\[26,](#page-9-5) [31,](#page-9-6) [35,](#page-9-7) [38\]](#page-9-8).

Extending LLM-based solutions to hour-long video inputs for temporal localization presents several important challenges. Video data is much denser than text, leading to a massive number of input tokens for the LLM. To handle this, many VLMs downsample frames and operate on a limited number of frames [\[18,](#page-8-6) [23,](#page-8-8) [30,](#page-9-9) [58,](#page-10-0) [68\]](#page-10-1), which leads to a significant loss of information in long videos. Moreover, training VLMs on hour-long videos requires immense memory and computational resources, which presents practical challenges for scalability and training efficiency. Furthermore, current VLMs often exhibit poor confidence calibration [\[25,](#page-8-9) [40\]](#page-9-10), leading to frequent false positives with high confidence. This issue is amplified in long videos, where distinguishing actual events from numerous false detections becomes increasingly difficult.

To address these challenges, we introduce the Recursive

<sup>\*</sup>Corresponding author: hannan@dbs.ifi.lmu.de

<span id="page-1-2"></span>Vision-Language Model ReVisionLLM, a VLM with hierarchical perception that processes videos recursively. Cognitive studies [\[6,](#page-8-10) [56\]](#page-10-2) suggest that when searching content, humans maintain a mental representation of the target and direct attention to the most promising areas, refining their search. In line with these principles, given a long video input, our recursive model first identifies broad video segments of interest and then progressively revises its focus, narrowing in on the event's exact temporal boundaries. In Fig. [2,](#page-1-0) we show the operating principle of our model. At the top hierarchy, the model operates broadly, identifying relevant segments (e.g., 5 minutes) from a 2-hour-long video. As it moves down the intermediate hierarchies, it narrows its focus to increasingly fine-grained temporal segments at the lowest hierarchy, pinpointing precise event boundaries (e.g., 3.5 seconds). Such a recursive processing structure of our model allows it to scale effectively to hour-long videos.

We first train our model on short video segments, then progress to training on hour-long videos. In the short video training phase, we introduce contrastive segments (i.e., video clips that do not contain the queried event) to improve confidence calibration. This helps the model learn to identify both the presence and absence of events, enhancing its confidence in visual input and aiding in accurate event localization within long videos. For efficient training on hour-long videos, we employ a temporal feature reduction strategy that compresses video segments into compact representations. This method reduces the input tokens required by the LLM. As a result, our model achieves both high accuracy and efficiency, making it well-suited for analyzing lengthy videos. Our contributions can be summarized as follows:

- We extend the existing VLMs to enable temporal grounding capabilities in hour-long videos.
- We propose a vision-language model that recursively processes hour-long videos for effective and efficient hourlong video processing.
- We propose a progressive training strategy, where the model is first trained to identify events in short video segments, then progressively scales to hour-long videos, enabling it to effectively handle longer, more complex video sequences.
- Our model significantly outperforms previous state-ofthe-art approaches, surpassing specialized models and other Vision-Language Models (VLMs) on multiple datasets by a substantial margin. For instance, ReVisionLLM outperforms the previous best method [\[15\]](#page-8-4) by 2.6% R1@.1 on the MAD [\[48\]](#page-9-11) dataset. Moreover, our model can efficiently solve this task, processing, on average, 43% fewer frames compared to the the existing VLM model [\[18\]](#page-8-6).

<span id="page-1-0"></span>![](_page_1_Figure_6.jpeg)

Figure 2. Recursive Video Grounding. ReVisionLLM is a recursive vision-language model designed for localizing events in hourlong videos. Inspired by human search strategies, it first scans the entire video to identify relevant intermediate segments and then zooms in to precisely locate event boundaries. Here, we show one intermediate hierarchy for brevity.

## <span id="page-1-1"></span>2. Related Work

Vision Language Models. VLMs excel in tasks such as video summarization [\[66,](#page-10-3) [70\]](#page-10-4), decision-making [\[13,](#page-8-11) [52\]](#page-10-5), captioning and general question-answering [\[8,](#page-8-12) [49,](#page-9-12) [68\]](#page-10-1), temporal localization and object trajectory detection [\[18,](#page-8-6) [19,](#page-8-7) [29,](#page-9-13) [46,](#page-9-4) [55,](#page-10-6) [64\]](#page-10-7). Existing VLMs integrate visual input by either training-free methods [\[59\]](#page-10-8), full fine-tuning [\[63\]](#page-10-9), or adapter fine-tuning [\[17,](#page-8-13) [18,](#page-8-6) [27,](#page-9-14) [32,](#page-9-15) [46\]](#page-9-4). However, most VLMs are limited to short videos and lack extensive temporal awareness. Our work extends VLMs to enable capabilities for temporal localization in hour-long videos. We contribute a novel adapter fine-tuning approach for hierarchical perception. Unlike prior models that aggregate frame features through spatial-temporal pooling [\[37\]](#page-9-16) or align visual-text embeddings [\[28\]](#page-9-17), we introduce a temporal feature reduction method to effectively scale training to hour-long videos.

Temporal Grounding VLMs. Most existing VLMs face challenges with temporal grounding. Recent models [\[18,](#page-8-6) [19,](#page-8-7) [29,](#page-9-13) [43,](#page-9-3) [44,](#page-9-18) [46,](#page-9-4) [50,](#page-9-19) [53,](#page-10-10) [55,](#page-10-6) [64\]](#page-10-7) have been specifically developed to address this issue using specialized architectures and datasets. For example, VTimeLLM [\[18\]](#page-8-6) proposes a temporal fine-tuning stage, TimeChat [\[46\]](#page-9-4) integrates timestamps with visual features, LITA [\[19\]](#page-8-7) introduces time tokens for temporal understanding, and Hawkeye [\[55\]](#page-10-6) uses dense video captions for segment matching. However, these models are limited by their training context length, which confines them to short video segments and hinders their ability to handle the complex temporal relationships and redundancies found in longer videos. Although they can process arbitrary user queries, they generally underperform compared to traditional, non-LLM-based models. We overcome these limitations by introducing a new recursive vision-language model that enables long video processing.

<span id="page-2-3"></span>**Long Video Temporal Grounding.** Recent advancements in hour-long video temporal grounding have leveraged datasets like MAD [48] and VidChapters7M [62]. These methods typically follow a two-stage approach: proposalfree methods [5, 33, 47, 67, 69] segment videos to predict candidate moments and rank them, while proposal-based methods [16, 42] generate proposal clips or anchors for grounding models. Approaches like CONE [16] and M-Guidance [5] utilize detection transformers [7] to enhance grounding, while RGNet [15] unifies clip retrieval [57] and grounding with an end-to-end transformer-based approach. Recently, SnAG [39] introduced late fusion techniques for more efficient processing. Prior models lack the instruction-following capability and rely on techniques that match video and text inputs. In contrast, our work integrates VLMs, which naturally allows our model to follow instructions and process textual user queries in the context of a long temporal grounding framework.

#### <span id="page-2-2"></span>3. Method

#### 3.1. Problem Overview

Given a long, untrimmed video input and an event defined by a text query, we aim to predict the precise temporal boundary of the event. Formally, as our inputs, we consider a long-range video sequence  $V = [v^t]_{t=1,\dots,T}$  comprised of T RGB frames, where  $v^t$  is the  $t^{th}$  frame. The event is defined by a query sentence S with  $N_s$  words where the sentence corresponds to a target event's start (s) and end (e) times, denoted as  $\tau = (s, e)$ .

### 3.2. The ReVisionLLM Model

We now describe our proposed ReVisionLLM model, which contains three high-level components: (1) a Multimodal Encoder, (2) a Hierarchical Adapter, and (3) a Large Language Model. We illustrate our approach in Fig. 3 and describe each component below.

**Multimodal Encoder.** We utilize an off-the-shelf video encoder (e.g., CLIP ViT-L/14 [60]) to extract features from an hour-long video. To reduce the input context length and capture the global properties of long video inputs, we extract global features (e.g., CLS token) for each frame,  $f^t \in \mathbb{R}^D$ 

, where D represents the feature dimension. These CLS tokens form a set of temporal features,  $F = [f^t]_{t=1,\dots,T}$  for the whole video. We use the same CLIP ViT-L/14 text encoder to extract textual features  $Q \in \mathbb{R}^{N_s \times D}$  of the query sentence S

**Hierarchical Adapter.** While LLMs excel at retrieving information from long text documents, performing retrieval from a large number of video frames remains challenging, limiting their effectiveness for visual grounding. To address this, we employ a recursive approach that processes

hour-long videos at different temporal resolutions. Initially, the entire video is processed as a whole to identify segments of interest (Fig. 2-top). Next, these segments undergo finer analysis to pinpoint precise event boundaries (Fig. 2-bottom). At the bottom level, we retain the original temporal resolution, while higher levels use compressed representations to maintain manageable visual input lengths for the LLM. Specifically, our Hierarchical Adapter projects initial video features  $\mathcal F$  into dense temporal features  $\mathcal D$  for the bottom hierarchy (Fig. 3-right) and encodes them into downsampled sparse temporal features  $\mathcal S$  for the upper hierarchies (Fig. 3-left).

To obtain both temporal features, we first partition  $\mathcal F$  into sliding windows of length  $L_w$ , producing video segments denoted as  $C = [C^i]_{i=1,\dots,|C|}$ . Each segment  $C^i$  is defined by  $C^i = [f^{s_i+t}]_{t=1,\dots,L_w}$ , where  $s_i$  is the start index of each clip, and  $C^i \in \mathbb R^{L_w \times D}$ . Dense temporal features  $\mathcal D^i$  are derived from each segment  $C^i$  through a linear projection layer,  $h_d$ , such that  $\mathcal D^i = h_d(C^i) \in \mathbb R^{L_w \times D}$ .

To create the sparse features, a two-step process is applied. First, a cross-attention layer (Eq. 1) uses segment feature  $C^i$  as the query and the text feature Q as the key, and outputs text-aligned segment feature  $\tilde{C}^i \in \mathbb{R}^{L_w \times D}$ . This cross-attention mechanism aligns the video segment with the text query, enhancing semantic correspondence between the modalities. Next, a self-attention layer (Eq. 2) takes the concatenation of a sparse feature  $S^i \in \mathbb{R}^D$  and the text-aligned segment feature  $\tilde{C}^i$  as input and condenses the segment into the sparse feature. This sparse feature is a compact, learnable representation similar to the CLS token in BERT [10].

$$\tilde{C}^i = \text{Cross-Attention}(C^i, Q)$$
 (1)

<span id="page-2-1"></span><span id="page-2-0"></span>
$$A = \text{Self-Attention}([S^i; \tilde{C}^i]) \tag{2}$$

For each video segment  $C^i$ , we compute the sparse temporal feature as  $\mathcal{S}^i = A_0$ , which condenses the segment (e.g., 2-minutes) into a compact embedding (e.g., 768-dimensional), substantially reducing the input context length. Collectively, this process yields the dense temporal features  $\mathcal{D} = [\mathcal{D}^i]_{i=1,\dots,|C|}$  and sparse temporal features  $\mathcal{S} = [\mathcal{S}^i]_{i=1,\dots,|C|}$  for all segments.

**Input for the LLM.** We construct a video input feature  $[I^{(\ell)}]_{\ell=1,\dots,L}$  with L hierarchies to capture different temporal scales. As illustrated in Figure 3, the lowest hierarchical feature,  $I^{(1)}$  is set to the dense features  $\mathcal{D}$  while the higher levels  $[I^{(\ell)}]_{\ell=2,\dots,L}$  use sparse features  $\mathcal{S}$ , obtained from the Hierarchical Adapter. We combine this visual input with an instruction prompt for the LLM. We define the instruction as " $\langle video \rangle$  when can we see the  $\langle event \rangle$  happening?". Here  $\langle event \rangle$  is replaced by the textual event description. The word embedding layer of the LLM converts this prompt into token embeddings,  $[w_1, w_2, \dots, w_M]$ . At

<span id="page-3-2"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3. The ReVisionLLM model. (Left) First, we detect segments (e.g., a few minutes) from an hour-long video using sparse temporal features produced by the Hierarchical Adapter. (Right) Then ReVisionLLM produces a precise temporal boundary using dense temporal features within the predicted segments. Note that the green box represents the same event boundary in both sub-figures, zooming in from left to right. The multimodal encoder is omitted for simplicity.

<video> position, we insert the video features to create the final input prompt, P (ℓ) = [I (ℓ) , w1, . . . , wM].

Large Language Model. To perform temporal grounding across multiple hierarchies, we utilize a pre-trained language model (e.g., Vicuna [\[9\]](#page-8-17)) as the temporal grounding decoder, which predicts the event boundaries at each hierarchy. At hierarchy l, the LLM receives input, P^{(l)} and outputs start and end times, {\tau ^{(l)}} , for that level in the form of: *'From* s *to* e*.'*. Here s and e denote the start and end frame indexes of the queried event. If the event is absent from the video segment, the model generates *"Not Present."*. To progressively refine detection, the decoder processes segments features P^{(l)} containing the predicted boundaries {\tau ^{(< l)}} from the previous hierarchy levels. At the initial level ( {\tau ^{(0)}} ), no prior boundaries are provided, which requires the model to scan the entire video for initial boundary prediction. The proposed ReVisionLLM learns the likelihood of a target event's start and end times \tau ^{(l)} conditioned on the hierarchical video representation P (l) through the following training objective:

$$p(T^{(l)}|P^{(l)}) = \prod_{k=1}^{K} p(T_k^{(\ell)}|T_{\leq k}^{(\ell)}, P^{(l)})$$
 (3)

Here, T (ℓ) k denotes the k th language token of the caption, and T (ℓ) <k denotes all preceding tokens.

#### <span id="page-3-1"></span>3.3. Training

Training a hierarchical video-language model for temporal grounding is challenging, particularly with videos of varying lengths and extensive non-relevant content. To tackle this, we employ a progressive training strategy (Fig. [4\)](#page-4-0) where the model initially learns to identify events in short video segments before scaling up to hour-long videos. This approach allows the model to first focus on recognizing key events in shorter segments, then apply that understanding effectively to longer, more complex videos.

Stage 1: Training with Short Segments. Traditional VLMs are generally trained with only positive video segments. For example, VTimeLLM [\[18\]](#page-8-6) trains the model to ground events on video segments that are guaranteed to contain the event (similar to the leftmost subplot in Figure [4\)](#page-4-0). This approach, however, can lead to overconfidence in the model's visual predictions [\[25,](#page-8-9) [40\]](#page-9-10). To address overconfidence—a challenge previously tackled in the text domain through contrastive examples [\[24\]](#page-8-18), we introduce contrastive video segments where the target event is intentionally absent (Fig. [4-](#page-4-0)middle). In an hour-long video, many video segments are naturally present that are unrelated to the queried event. Including these segments in training helps the model better calibrate its confidence. At first, we use dense features to train the model to predict precise temporal boundaries (e.g., *"From* s *to* e*."*) or indicate the absence of events (*"Not Present."*). During this phase, we fine-tune the LLM component using LoRA.

After fine-tuning the LLM, we freeze its weights and proceed to fine-tune only the Hierarchical Adapter module to generate sparse temporal features. These sparse features, which are downsampled versions of the original visual data, are crucial for efficient long-video training (see [3.3\)](#page-3-1). To simplify the training objective for these sparse inputs, we focus on identifying the presence of events rather than lo-

<span id="page-4-2"></span><span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 4. Progressive Training Method. Our model is trained progressively: first on short video segments and then on hour-long videos. (Left) In the first stage, the model learns to detect whether an event is present in the input video and, if so, predicts its precise start and endpoints. Sparse features help determine an event's presence, while dense features additionally facilitate exact localization. (Right) In the second stage, we utilize the sparse features learned in Stage 1 to identify event segments within hour-long videos.

cating precise boundaries. We modify the input prompt as *"*<*video*> *Does the* <*event*> *happen in the video? Answer yes or no."* In this phase, the model learns to respond *"Yes."* for relevant segments or *"No."* for irrelevant ones. Using this contrastive training strategy, we calibrate the model's visual confidence and optimize the sparse features needed for effective hour-long video processing.

Stage 2: Training with Long Videos. In this stage (Fig. [4](#page-4-0) right), we leverage sparse features to localize relevant segments within hour-long videos. These sparse features correspond to segments typically much longer (e.g., 3 minutes) than the actual target events, allowing the model to efficiently scan and identify broader regions of interest. We use the original prompt defined in Section [3.2](#page-2-1) and keep the weights of the hierarchical perception module fixed, finetuning only the same LoRA module used in Stage 1.

## 3.4. Inference with Calibrated Confidence

Previous methods [\[16,](#page-8-5) [42\]](#page-9-2) typically used the CLIP [\[60\]](#page-10-15) similarity score between visual and textual embeddings to rank and select the top-k predictions. In contrast, we rank predictions based on the internal confidence of our LLM. Specifically, we calculate the entropy of the predicted probability distribution for each word generated by the LLM. We do this scoring on the predictions at the bottom hierarchy (l = 0) where we utilize the dense features D. For i th prediction, let p(w | T\_{< k}, \mathcal {D}^{(i)}) denote the probability of the k th word conditioned on prior words T\_{< k} and corresponding segment feature D<sup>i</sup> . We calculate the entropy H^i\_k as:

$$H_k^{(i)} = -\sum_{w} p(w|T_{< k}, \mathcal{D}^{(i)}) \log p(w|T_{< k}, \mathcal{D}^{(i)})$$

Here, w represents each possible word in the vocabulary of our LLM. The overall uncertainty score is obtained by averaging the entropy values across all words in the generated sequence. To convert this into a confidence score (R<sup>i</sup> ), we take the inverse of the mean entropy:

$$R^i = \frac{1}{\frac{1}{K} \sum_{k=1}^K H_k^i}$$

where K is the total number of generated words. We calculate the confidence score for all the predictions [R<sup>i</sup> ]i=1,...,N<sup>p</sup> , where N<sup>p</sup> is the number of predicted boundaries. This confidence score allows us to rank and select the top-K predictions based on the LLM's internal confidence about its outputs. More details are present in the Supplementary Section [S2.](#page-1-1)

## <span id="page-4-1"></span>4. Experimental Setup

## 4.1. ReVisionLLM Baselines

Long video temporal grounding remains largely uncharted territory for vision-large language models, leaving a lack of established baselines for meaningful comparison. To address this, we present the following video-language baselines that we have adapted specifically for this task.

- 1. VTimeLLM [\[18\]](#page-8-6). A fully fine-tuned baseline that takes in hour-long video as input and produces temporal boundary localization.
- 2. VTimeLLM + CONE [\[16,](#page-8-5) [18\]](#page-8-6). This baseline is fully fine-tuned on shorter video segments as ours. We employ CONE's fine-grained ranking method to select the top-k predictions across all segments. First, mean pooling aggregates the CLS features from CLIP [\[60\]](#page-10-15) across all frames to create an average frame representation. The similarity score is then computed with a dot product between this averaged frame feature and the query text CLS feature from CLIP.

#### 4.2. ReVisionLLM Variations

1. ReVisionLLM. Our default model begins by processing hour-long video segments at the top of the hierarchy, then moves down to shorter segments. In this setup, we train two LoRAs within the LLM: one for the bottom hierarchy and another for the higher levels. This approach

<span id="page-5-1"></span><span id="page-5-0"></span>

| Model             | MAD [48] |       |       |       | VidChapters-7m [62] |       |       |       | Average↑ |       |       |       |      |
|-------------------|----------|-------|-------|-------|---------------------|-------|-------|-------|----------|-------|-------|-------|------|
|                   | R1@.1    | R5@.1 | R1@.3 | R5@.3 | R1@.5               | R5@.5 | Avg.↑ | R1@.3 | R1@.5    | R1@.7 | R1@.9 | Avg.↑ |      |
| M-Guide [5]       | 9.3      | 18.9  | 4.6   | 13.1  | 2.2                 | 7.4   | 9.3   | -     | -        | -     | -     | -     | -    |
| CONE [16]         | 8.9      | 20.5  | 6.9   | 16.1  | 4.1                 | 9.6   | 11.0  | -     | -        | -     | -     | -     | -    |
| SOONet [42]       | 11.3     | 23.2  | 9.0   | 19.6  | 5.3                 | 13.1  | 13.6  | -     | -        | -     | -     | -     | -    |
| SnAG [39]         | 10.3     | 24.4  | 8.5   | 20.6  | 5.5                 | 13.7  | 13.8  | -     | -        | -     | -     | -     | -    |
| RGNet [15]        | 12.4     | 25.1  | 9.5   | 18.7  | 5.6                 | 10.9  | 13.7  | -     | -        | -     | -     | -     | -    |
| BERT [10]         | -        | -     | -     | -     | -                   | -     | -     | 0.6   | 0.3      | 0.1   | 0.0   | 0.3   | 0.3  |
| VTimeLLM∗<br>[69] | 1.4      | 3.1   | 1.3   | 2.5   | 0.6                 | 1.1   | 1.7   | 10.6  | 4.1      | 1.6   | 0.2   | 4.1   | 2.9  |
| CLIP [45]         | 6.6      | 15.1  | 3.1   | 9.9   | 1.5                 | 5.4   | 6.9   | 10.7  | 5.2      | 2.3   | 0.5   | 4.7   | 5.8  |
| M-DETR [26]       | 3.6      | 13.0  | 2.8   | 9.9   | 1.7                 | 5.6   | 6.1   | 37.4  | 27.3     | 17.6  | 6.4   | 22.1  | 14.1 |
| Ours†             | 17.3     | 31.4  | 12.7  | 23.5  | 6.7                 | 13.1  | 17.5  | -     | -        | -     | -     | -     | -    |
| Ours              | 15.0     | 25.1  | 11.0  | 18.8  | 5.8                 | 10.5  | 14.4  | 33.8  | 27.4     | 21.8  | 15.2  | 24.6  | 19.5 |

Table 1. Main Results on the MAD and VidChapters-7M Datasets. The best scores are highlighted in bold, while the second-best scores are underlined. ReVisionLLM demonstrates state-of-the-art performance across both datasets. <sup>∗</sup>This paper trains VTimeLLM on the datasets and uses CONE[\[16\]](#page-8-5) ranking method. †ReVisionLLM-I variant processes more frames and achieves higher accuracy.

is highly efficient for inference, as it reduces the number of input frames processed by the LLM.

- 2. ReVisionLLM-U. This is a unified model across all hierarchies with shared weights. With significantly fewer trainable parameters than the default model, it is more efficient to train.
- 3. ReVisionLLM-I. In this variant, the LLM operates in reverse order, starting from the bottom of the hierarchy and progressively increasing the video length. Like the default model, it uses two LoRAs but requires processing all video frames, trading efficiency for improved performance with added computation.
- 4. ReVisionLLM-(U+I). This variant has a unified architecture like ReVisionLLM-U and starts processing from the bottom hierarchy like ReVisionLLM-I.

#### 4.2.1 Datasets and Metrics

MAD Dataset [\[48\]](#page-9-11). This large-scale dataset comprises approximately 1,200 hours of full-length movies, featuring 384,000 natural language queries linked to specific moments within the videos. On average, each video is around 110 minutes long, while the moments are brief—just 4.1 seconds on average—making the moment-to-video ratio very low. This disparity poses a substantial challenge for accurate temporal grounding.

VidChapters-7M [\[62\]](#page-10-11). This is a large-scale, userannotated dataset with over 7 million chapters across 817,000 videos, with the longest 12 hours. Each video includes 2 to 30 chapters with durations from 1 second to 10 minutes, making it a challenging dataset for temporal localization due to its length and variety.

Evaluation Metrics. Following prior work [\[16,](#page-8-5) [48\]](#page-9-11), we use Recall@k at IoU=θ (Rk@θ) as the primary grounding metric. This measures the proportion of test samples where at least one of the top-k predictions achieves an Intersection over Union (IoU) greater than θ with the ground truth. Additionally, to assess generalization to Text-to-Video retrieval, we use the standard Recall at Rank k (R@k) metric [\[14\]](#page-8-19), which measures the percentage of ground truth video in the top-k retrieved ones.

Implementation Details. In our approach, we utilize the 7B version of Vicuna v1.5 [\[9\]](#page-8-17) as the Large Language Model. Training is conducted on the using a total batch size of 128 across 8 A100 GPUs. For optimization, we use AdamW [\[36\]](#page-9-23) with a cosine learning rate decay and an initial warm-up phase. During the adapter training stage, we run 1 epoch with a learning rate of 1 × 10<sup>−</sup><sup>3</sup> . In the subsequent hierarchical stage, we train for 5 epochs for MAD and 1 epoch for VidChapters-7M, with a learning rate of 1 × 10<sup>−</sup><sup>4</sup> . LoRA settings include parameters r = 64 and alpha = 128. Please refer to the Supplementary Section [S1](#page-0-1) for additional implementation details.

## 5. Results

In this section, we present our performance against previous methods, detailed ablation studies, qualitative results, and generalizations of text-to-video retrieval tasks.

Main Results on the MAD Dataset [\[48\]](#page-9-11). ReVisionLLM sets a new state-of-the-art on the MAD dataset, outperforming prior models in temporal grounding (Tab. [1\)](#page-5-0). It surpasses the previous best method, RGNet [\[15\]](#page-8-4), by +2.6% in R1@.1 and +1.5% in R1@.3, achieving competitive scores across other metrics. Our ReVisionLLM-I variant outperforms RGNet by an even larger margin, +4.9% in R1@.1 and +6.3% in R1@.3. As the moments-to-video ratio is extremely low in this dataset, it requires fine-grained event understanding. Our model's recursive architecture effectively narrows the search for relevant segments, handling these challenges well. While existing methods relying on heuristic CLIP [\[60\]](#page-10-15) ranking struggle with numerous false detections (evident by low R1 score across all thresholds) in this dataset, ReVisionLLM reliance on LLM's internal confidence reduces such errors.

Main Results on VidChapters-7M Dataset [\[62\]](#page-10-11). On the

<span id="page-6-3"></span>VidChapters-7M dataset, ReVisionLLM sets a new stateof-the-art (Table [1\)](#page-5-0), significantly outperforming the previous best model, M-DETR [\[26\]](#page-9-5), particularly at higher IoU thresholds (+4.2% in R1@.7 and +8.8% in R1@.9) and showing strong results across other metrics. This precision at stricter thresholds demonstrates the superior ability of ReVisionLLM to localize events accurately. The dataset includes a diverse range of YouTube tutorial videos with userqueried steps, underscoring our model's advancements in improving video content search for online platforms across short clips to extended videos up to 12 hours.

#### 5.1. Ablation Studies

We present ablation studies on each module, model variants, video length, and the number of hierarchies by experimenting on the MAD dataset [\[48\]](#page-9-11).

<span id="page-6-0"></span>

| Modules                  | R1@.1↑ | R5@.1↑ | R1@.3↑ | R5@.3↑ |
|--------------------------|--------|--------|--------|--------|
| Baseline: VTimeLLM [18]  | 0.0    | 0.0    | 0.0    | 0.0    |
| (+) CONE [16]            | 1.4    | 2.4    | 1.3    | 2.5    |
| (+) Contrastive Segment  | 4.8    | 6.7    | 4.2    | 7.2    |
| (+) Calibration (-) CONE | 8.4    | 12.7   | 6.6    | 8.9    |
| (+) Recursive Process∗   | 15.0   | 25.1   | 11.0   | 18.8   |

Table 2. Cumulative Ablation on Proposed Modules. Each of our proposed modules contributes to a significant improvement in grounding capability, with the recursive process achieving the highest gains. <sup>∗</sup> Indicates the ReVisionLLM model.

Modules. Table [2](#page-6-0) highlights the unique contributions of each module in our model for temporal grounding on long videos. We start with the baseline VTimeLLM [\[18\]](#page-8-6) model, trained on the MAD dataset, which scores zero across all recall metrics due to uniform sampling of 100 frames from hours-long videos, causing a complete loss of temporal details. Next, we train and test the VTimeLLM model on shorter video segments and apply CONE's ranking strategy [\[16\]](#page-8-5) for final predictions. This single level of hierarchical processing yields modest improvements (e.g., R1@.1 of 1.4% and R5@.1 of 2.4%).

The addition of the Contrastive Segments marks a notable improvement, raising R1@.1 to 4.8% and R5@.1 to 6.7%; by allowing the model to identify absent segments (e.g., "Not Present"), it narrows the temporal search space and enhances segment selection. Replacing CONE with our Grounding LLM's confidence-based ranking further boosts results (e.g., R1@.1 of 8.4% and R5@.1 of 12.7%), as training on positive and contrastive segments in Stage 1 improves the LLM's calibration, aligning confidence with prediction accuracy and enhancing ranking effectiveness. Finally, the recursive process achieves the highest performance gains, with R1@.1 of 15.0% and R5@.1 of 25.1%, by progressively refining the temporal focus. Each module contributes critically, with the Recursive Process module delivering the largest gains.

<span id="page-6-1"></span>

| Model             |      | Train Params↓ Input Frames↓ R1@.1↑ R5@.1↑ |      |      |
|-------------------|------|-------------------------------------------|------|------|
| VTimeLLM+CONE     | 363M | 100%                                      | 1.4  | 2.4  |
| ReVisionLLM       | 363M | 57%                                       | 15.0 | 25.1 |
| ReVisionLLM-U     | 159M | 58%                                       | 14.4 | 24.7 |
| ReVisionLLM-(U+I) | 159M | 100%                                      | 16.7 | 31.0 |
| ReVisionLLM-I     | 363M | 100%                                      | 17.4 | 31.4 |

Table 3. Ablation on Model Variants. We perform well with fewer frames and trainable parameters than the baseline, which struggles to solve the task effectively. Processing additional frames and more parameters further improves our performance.

Model Variants. Table [3](#page-6-1) compares ReVisionLLM variants to assess the effects of hierarchical processing order, percentage of video input frames for the LLM, and parameter sharing. The baseline (*VTimeLLM*+CONE) performs poorly (R1@.1 = 1.4%), despite processing 100% of video frames as input to the LLM, underscoring the limitations of non-recursive approaches. In contrast, our default Re-VisionLLM processes recursively from top to bottom, balancing accuracy and efficiency, achieving R1@.1 of 15.0% and R5@.1 of 25.1% with processing only 57% of input frames. The ReVisionLLM-U variant enhances training efficiency by sharing weights across all hierarchies, resulting in a slight performance reduction (R1@.1 = 14.4%, R5@.1 = 24.7%) while using fewer trainable parameters (363M vs. 159M). The ReVisionLLM-I variant reaches the highest accuracy (R1@.1 = 17.4%, R5@.1 = 31.4%) by processing in reverse hierarchical order (bottom to top), with more input frames. ReVisionLLM-(U+I) combines reverse processing with parameter sharing, balancing training efficiency, and high performance (R1@.1 = 16.7%, R5@.1 = 31.0%). Overall, the default ReVisionLLM offers strong accuracy with high frame efficiency, while ReVisionLLM-I provides peak accuracy with increased input frames to the LLM.

<span id="page-6-2"></span>![](_page_6_Figure_10.jpeg)

Figure 5. Ablation on Video Length. Our recursive approach maintains strong performance even with videos up to 10 hours long, while the baseline method fails entirely in these cases.

Video Length. Fig. [5](#page-6-2) demonstrates ReVisionLLM's robustness in handling long videos. We extend the videos by repeating them multiple times to create longer sequences, ensuring that the ground truth moment appears only once within the extended video. While the method without recursion fails for 10-hour videos, recursive video processing maintains strong performance. The slight decrease in per-

<span id="page-7-2"></span><span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 6. Qualitative results on MAD. ReVisionLLM accurately locates precise event boundaries that involve intricate actions (top) and complex visual details (bottom) within hour-long movies. In contrast, our VLM baseline fails entirely to capture these events.

formance for longer videos reflects the inherent challenges of temporal grounding in extended content rather than a limitation of the model. Beyond approximately five hours, performance stabilizes, as the increasing diversity of scenes has minimal impact on event localization.

Number of Hierarchies. Table [4](#page-7-0) shows the impact of the number of hierarchies. Without hierarchy, treating the entire video as a single unit prevents the model from capturing event boundaries, highlighting the limitations of current non-recursive VLMs in long video grounding. With one hierarchy, we segment the video and aggregate predictions with calibrated confidence, showing some improvement but still struggling with high false positives. With 2 and 3 hierarchies, we progressively filter out irrelevant regions and revise predictions recursively, resulting in higher accuracy.

<span id="page-7-0"></span>

| Hierarchies | R1@.1↑ | R5@.1↑ | R1@.3↑ | R5@.3↑ |
|-------------|--------|--------|--------|--------|
| 0           | 0.0    | 0.0    | 0.0    | 0.0    |
| 1           | 8.4    | 12.7   | 6.6    | 8.9    |
| 2           | 11.9   | 17.5   | 8.7    | 13.2   |
| 3           | 15.0   | 25.1   | 11.0   | 18.8   |

Table 4. Ablation on number of Hierarchies. The model's performance improves with the number of hierarchies in the recursive structure, becoming more effective with each additional level. Without this hierarchical approach, the model fails on the task.

#### 5.2. Qualitative Results on MAD Dataset

In Figure [6,](#page-7-1) we demonstrate two examples of events accurately localized by ReVisionLLM. In the first example, frequent scenes of office work closely resemble the queried event, but ReVisionLLM successfully identifies the specific instance where a person with distinct attributes answers a phone—a task requiring detailed comprehension of both appearance and action. The second example tests the model's ability to identify a complex visual description spanning 5.8 seconds within a 2-hour movie, highlighting its effectiveness in locating subtle differences within visually similar footage. More qualitative results are included in the Supplementary Section [S4.](#page-4-1)

# 5.3. Generalization: Text-to-Video Retrieval

Text-to-video retrieval is the task of identifying the video corresponding to a textual event description from a large set of different videos. We solve this task with our ReVisionLLM by concatenating all the videos into a single hourlong video and applying our model to locate the index of the predicted video. It outperforms previous state-of-the-art models on the MSRVTT [\[57\]](#page-10-14) dataset, with improvements of +2.2% in R@5 and +0.6% in R@10, while achieving competitive R@1 performance. These results demonstrate Re-VisionLLM's understanding of video-text correspondence, positioning it as a strong model for general multi-modal retrieval tasks, including large-scale video searches. More details are provided in the Supplementary Section [S3.](#page-2-2)

| Method            | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|-------------------|-------|-------|--------|
| X-Pool [14]       | 46.9  | 72.8  | 82.2   |
| DiffusionRet [22] | 49.0  | 75.2  | 82.7   |
| UATVR [11]        | 47.5  | 73.9  | 83.5   |
| TEFAL [20]        | 49.4  | 75.9  | 83.9   |
| CLIP-ViP [61]     | 50.1  | 74.8  | 84.6   |
| T-MASS [54]       | 50.2  | 75.3  | 85.1   |
| Ours              | 49.1  | 77.5  | 85.7   |

Table 5. ReVisionLLM's Generalization. Our model generalizes well to the Text-to-Video retrieval task and performs competitively with state-of-the-art models on the MSRVTT [\[57\]](#page-10-14) dataset.

## 6. Conclusion and Future Work

We introduce ReVisionLLM, the first VLM specifically designed with a recursive structure for temporal event grounding in hour-long videos. Its recursive architecture effectively can locate events within extensive videos and establishes a new state-of-the-art, outperforming specialized models. Future work could focus on integrating audio for better event comprehension and expanding the capabilities to handle even longer videos spanning multiple days.

## References

- <span id="page-8-0"></span>[1] Greg kamradt on x: "pressure testing gpt-4-128k with long context recall 128k tokens of context is awesome - but what's performance like? i wanted to find out so i did a "needle in a haystack" analysis some expected (and unexpected) results here's what i found: Findings: \* gpt-4's recall https://t.co/nhmokmfhw5" / x. [https://twitter.com/GregKamradt/status/](https://twitter.com/GregKamradt/status/1722386725635580292?s=20) [1722386725635580292 ? s = 20](https://twitter.com/GregKamradt/status/1722386725635580292?s=20), . (Accessed on 11/14/2024). [1](#page-0-2)
- <span id="page-8-1"></span>[2] Greg kamradt on x: "claude 2.1 (200k tokens) - pressure testing long context recall we all love increasing context lengths - but what's performance like? anthropic reached out with early access to claude 2.1 so i repeated the "needle in a haystack" analysis i did on gpt-4 here's what i found: https://t.co/b36knjtjme" / x. [https://twitter.com/](https://twitter.com/GregKamradt/status/1727018183608193393) [GregKamradt/status/1727018183608193393](https://twitter.com/GregKamradt/status/1727018183608193393), . (Accessed on 11/14/2024). [1](#page-0-2)
- <span id="page-8-23"></span>[3] Sharegpt: Share your wildest chatgpt conversations with one click. <https://sharegpt.com/>. (Accessed on 11/19/2024). [1](#page-0-2)
- <span id="page-8-2"></span>[4] gkamradt/llmtest needleinahaystack: Doing simple retrieval from llm models at various context lengths to measure accuracy. [https://github.com/gkamradt/LLMTest\\_](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack). (Accessed on 11/14/2024). [1](#page-0-2)
- <span id="page-8-14"></span>[5] Wayner Barrios, Mattia Soldan, Alberto Mario Ceballos-Arroyo, Fabian Caba Heilbron, and Bernard Ghanem. Localizing moments in long video via multimodal guidance. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 13667–13678, 2023. [3,](#page-2-3) [6](#page-5-1)
- <span id="page-8-10"></span>[6] Patrick Bourke, Steven Brown, Elton Ngan, and Mario Liotti. Functional brain organization of preparatory attentional control in visual search. *Brain research*, 1530:32–43, 2013. [2](#page-1-2)
- <span id="page-8-15"></span>[7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In *European conference on computer vision*, pages 213–229. Springer, 2020. [3](#page-2-3)
- <span id="page-8-12"></span>[8] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. *arXiv preprint arXiv:2305.13292*, 2023. [2](#page-1-2)
- <span id="page-8-17"></span>[9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, 2023. [4,](#page-3-2) [6,](#page-5-1) [1](#page-0-2)
- <span id="page-8-16"></span>[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. [3,](#page-2-3) [6](#page-5-1)
- <span id="page-8-21"></span>[11] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Yuxin Song, Weiping Wang, Xiangbo Shu, Xiangyang Ji, and Jingdong Wang. Uatvr: Uncertainty-adaptive text-video retrieval, 2023. [8](#page-7-2)

- <span id="page-8-24"></span>[12] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In *ECCV*, pages 214–229, 2020. [3](#page-2-3)
- <span id="page-8-11"></span>[13] Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. *arXiv preprint arXiv:2306.08640*, 2023. [2](#page-1-2)
- <span id="page-8-19"></span>[14] Satya Krishna Gorti, Noel Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei Yu. X-pool: Cross-modal language-video attention for textvideo retrieval, 2022. [6,](#page-5-1) [8](#page-7-2)
- <span id="page-8-4"></span>[15] Tanveer Hannan, Md Mohaiminul Islam, Thomas Seidl, and Gedas Bertasius. Rgnet: A unified retrieval and grounding network for long videos. *arXiv preprint arXiv:2312.06729*, 2023. [1,](#page-0-2) [2,](#page-1-2) [3,](#page-2-3) [6](#page-5-1)
- <span id="page-8-5"></span>[16] Zhijian Hou, Wanjun Zhong, Lei Ji, Difei Gao, Kun Yan, Wing-Kwong Chan, Chong-Wah Ngo, Zheng Shou, and Nan Duan. Cone: An efficient coarse-to-fine alignment framework for long video temporal grounding. *arXiv preprint arXiv:2209.10918*, 2022. [1,](#page-0-2) [3,](#page-2-3) [5,](#page-4-2) [6,](#page-5-1) [7](#page-6-3)
- <span id="page-8-13"></span>[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [2,](#page-1-2) [1](#page-0-2)
- <span id="page-8-6"></span>[18] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14271–14280, 2024. [1,](#page-0-2) [2,](#page-1-2) [4,](#page-3-2) [5,](#page-4-2) [7](#page-6-3)
- <span id="page-8-7"></span>[19] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. *arXiv preprint arXiv:2403.19046*, 2024. [1,](#page-0-2) [2](#page-1-2)
- <span id="page-8-22"></span>[20] Sarah Ibrahimi, Xiaohang Sun, Pichao Wang, Amanmeet Garg, Ashutosh Sanan, and Mohamed Omar. Audioenhanced text-to-video retrieval using text-conditioned feature alignment, 2023. [8](#page-7-2)
- <span id="page-8-3"></span>[21] Maor Ivgi, Uri Shaham, and Jonathan Berant. Efficient longtext understanding with short-text models. *Transactions of the Association for Computational Linguistics*, 11:284–299, 2023. [1](#page-0-2)
- <span id="page-8-20"></span>[22] Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, and Jie Chen. Diffusionret: Generative text-video retrieval with diffusion model, 2023. [8](#page-7-2)
- <span id="page-8-8"></span>[23] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13700– 13710, 2024. [1](#page-0-2)
- <span id="page-8-18"></span>[24] Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Gordon Wilson. Large language models must be taught to know what they don't know. *arXiv preprint arXiv:2406.08391*, 2024. [4](#page-3-2)
- <span id="page-8-9"></span>[25] Vasily Kostumov, Bulat Nutfullin, Oleg Pilipenko, and Eugene Ilyushin. Uncertainty-aware evaluation for vision-

- language models. *arXiv preprint arXiv:2402.14418*, 2024. [1,](#page-0-2) [4](#page-3-2)
- <span id="page-9-5"></span>[26] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. *Advances in Neural Information Processing Systems*, 34: 11846–11858, 2021. [1,](#page-0-2) [6,](#page-5-1) [7](#page-6-3)
- <span id="page-9-14"></span>[27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730– 19742. PMLR, 2023. [2](#page-1-2)
- <span id="page-9-17"></span>[28] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. *arXiv preprint arXiv:2305.06355*, 2023. [2](#page-1-2)
- <span id="page-9-13"></span>[29] Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, Yiqing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu, et al. Groundinggpt: Language enhanced multi-modal grounding model. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6657–6678, 2024. [2](#page-1-2)
- <span id="page-9-9"></span>[30] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. *arXiv preprint arXiv:2311.10122*, 2023. [1](#page-0-2)
- <span id="page-9-6"></span>[31] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2794–2804, 2023. [1](#page-0-2)
- <span id="page-9-15"></span>[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *arXiv preprint arXiv:2304.08485*, 2023. [2,](#page-1-2) [1](#page-0-2)
- <span id="page-9-20"></span>[33] Naiyuan Liu, Xiaohan Wang, Xiaobo Li, Yi Yang, and Yueting Zhuang. Reler@ zju-alibaba submission to the ego4d natural language queries challenge 2022. *arXiv preprint arXiv:2207.00383*, 2022. [3](#page-2-3)
- <span id="page-9-0"></span>[34] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. corr abs/2307.03172 (2023). *arXiv preprint arXiv:2307.03172*, 10, 2023. [1](#page-0-2)
- <span id="page-9-7"></span>[35] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3042–3051, 2022. [1](#page-0-2)
- <span id="page-9-23"></span>[36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. [6,](#page-5-1) [1](#page-0-2)
- <span id="page-9-16"></span>[37] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. *arXiv preprint arXiv:2306.05424*, 2023. [2](#page-1-2)
- <span id="page-9-8"></span>[38] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In *Pro-*

- *ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 23023–23033, 2023. [1](#page-0-2)
- <span id="page-9-1"></span>[39] Fangzhou Mu, Sicheng Mo, and Yin Li. Snag: Scalable and accurate video grounding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18930–18940, 2024. [1,](#page-0-2) [3,](#page-2-3) [6](#page-5-1)
- <span id="page-9-10"></span>[40] Changdae Oh, Mijoo Kim, Hyesu Lim, Junhyeok Park, Euiseog Jeong, Zhi-Qi Cheng, and Kyungwoo Song. Towards calibrated robust fine-tuning of vision-language models. *arXiv preprint arXiv:2311.01723*, 2023. [1,](#page-0-2) [4](#page-3-2)
- <span id="page-9-24"></span>[41] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. *Proceedings of the ... AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence*, 2015:2901–2907, 2015. [2](#page-1-2)
- <span id="page-9-2"></span>[42] Yulin Pan, Xiangteng He, Biao Gong, Yiliang Lv, Yujun Shen, Yuxin Peng, and Deli Zhao. Scanning only once: An end-to-end framework for fast temporal grounding in long videos. *arXiv preprint arXiv:2303.08345*, 2023. [1,](#page-0-2) [3,](#page-2-3) [5,](#page-4-2) [6](#page-5-1)
- <span id="page-9-3"></span>[43] Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning, 2024. [1,](#page-0-2) [2](#page-1-2)
- <span id="page-9-18"></span>[44] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. Chatvtg: Video temporal grounding via chat with video dialogue large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1847–1856, 2024. [2](#page-1-2)
- <span id="page-9-22"></span>[45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. pages 8748–8763, 2021. [6](#page-5-1)
- <span id="page-9-4"></span>[46] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14313–14323, 2024. [1,](#page-0-2) [2](#page-1-2)
- <span id="page-9-21"></span>[47] Mattia Soldan, Mengmeng Xu, Sisi Qu, Jesper Tegner, and Bernard Ghanem. Vlg-net: Video-language graph matching network for video grounding. In *ICCV*, pages 3224–3234, 2021. [3](#page-2-3)
- <span id="page-9-11"></span>[48] Mattia Soldan, Alejandro Pardo, Juan Leon Alc ´ azar, Fabian ´ Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos from movie audio descriptions. In *CVPR*, pages 5026–5035, 2022. [2,](#page-1-2) [3,](#page-2-3) [6,](#page-5-1) [7](#page-6-3)
- <span id="page-9-12"></span>[49] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18221–18232, 2024. [2](#page-1-2)
- <span id="page-9-19"></span>[50] Yunlong Tang, Daiki Shimada, Jing Bi, and Chenliang Xu. Avicuna: Audio-visual llm with interleaver and contextboundary alignment for temporal referential dialogue. *arXiv preprint arXiv:2403.16276*, 2024. [2](#page-1-2)

- <span id="page-10-18"></span>[51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste ´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. ` Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023. [1](#page-0-2)
- <span id="page-10-5"></span>[52] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. *arXiv preprint arXiv:2304.14407*, 2023. [2](#page-1-2)
- <span id="page-10-10"></span>[53] Junke Wang, Dongdong Chen, Chong Luo, Bo He, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Omnivid: A generative framework for universal video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18209–18220, 2024. [2](#page-1-2)
- <span id="page-10-17"></span>[54] Jiamian Wang, Guohao Sun, Pichao Wang, Dongfang Liu, Sohail Dianat, Majid Rabbani, Raghuveer Rao, and Zhiqiang Tao. Text is mass: Modeling as stochastic embedding for text-video retrieval, 2024. [8](#page-7-2)
- <span id="page-10-6"></span>[55] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training videotext llms for grounding text in videos. *arXiv preprint arXiv:2403.10228*, 2024. [2](#page-1-2)
- <span id="page-10-2"></span>[56] Jeremy M Wolfe and Todd S Horowitz. Five factors that guide attention in visual search. *Nature human behaviour*, 1 (3):0058, 2017. [2](#page-1-2)
- <span id="page-10-14"></span>[57] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 5288–5296, 2016. [3,](#page-2-3) [8](#page-7-2)
- <span id="page-10-0"></span>[58] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. *arXiv preprint arXiv:2404.16994*, 2024. [1](#page-0-2)
- <span id="page-10-8"></span>[59] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. *arXiv preprint arXiv:2407.15841*, 2024. [2](#page-1-2)
- <span id="page-10-15"></span>[60] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment. *ICLR*, 2023. [3,](#page-2-3) [5,](#page-4-2) [6,](#page-5-1) [1](#page-0-2)
- <span id="page-10-16"></span>[61] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment, 2023. [8](#page-7-2)
- <span id="page-10-11"></span>[62] Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vidchapters-7m: Video chapters at scale, 2023. [3,](#page-2-3) [6](#page-5-1)
- <span id="page-10-9"></span>[63] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10714–10726, 2023. [2](#page-1-2)
- <span id="page-10-7"></span>[64] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization

- and question answering. *Advances in Neural Information Processing Systems*, 36, 2024. [2](#page-1-2)
- <span id="page-10-19"></span>[65] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In *ECCV*, pages 471–487, 2018. [3](#page-2-3)
- <span id="page-10-3"></span>[66] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering, 2024. [2](#page-1-2)
- <span id="page-10-12"></span>[67] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. *arXiv preprint arXiv:2004.13931*, 2020. [3](#page-2-3)
- <span id="page-10-1"></span>[68] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. *arXiv preprint arXiv:2306.02858*, 2023. [1,](#page-0-2) [2](#page-1-2)
- <span id="page-10-13"></span>[69] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In *AAAI*, pages 12870–12877, 2020. [3,](#page-2-3) [6](#page-5-1)
- <span id="page-10-4"></span>[70] Yue Zhao, Ishan Misra, Philipp Krahenb ¨ uhl, and Rohit ¨ Girdhar. Learning video representations from large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6586– 6597, 2023. [2,](#page-1-2) [1](#page-0-2)

# ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

# Supplementary Material

Our supplementary materials contain Section [S1:](#page-0-1) Additional Implementation Details, Section [S2:](#page-1-1) Calibration Confidence, Section [S3:](#page-2-2) Generalization: Text-to-Video Retrieval, and Section [S4:](#page-4-1) Aditional Qualitative Results.

## S1. Additional Implementation Details

Multimodal Encoder. To handle multimodal inputs (refer to Fig. [S1\)](#page-11-0), we utilize the Frozen CLIP L/14 model [\[60\]](#page-10-15), a 24-layer transformer pretrained on extensive image-text pairs using a contrastive learning objective [\[70\]](#page-10-4). The vision encoder processes inputs of size 224 × 224, producing both spatial tokens and a global CLS token. For computational efficiency, we use only the CLS token to represent each frame in long videos. Similarly, the CLIP text encoder, a 12-layer transformer, extracts feature representations for the queried event from the input text.

Hierarchical Adapter. As illustrated in Fig. [S1,](#page-11-0) the Hierarchical Adapter processes the i th video segment C i to generate both sparse (S i ) and dense (D<sup>i</sup> ) features. For the MAD dataset, video features are divided into sliding windows of L<sup>w</sup> = 125 seconds, while for the VidChapters-7M dataset, the window length is L<sup>w</sup> = 500 seconds. From each segment, 250 frames are uniformly sampled and used as input to the hierarchical adapter.

Sparse features S i are computed using a combination of cross-attention and self-attention mechanisms, each implemented with two layers (N = 2). This lightweight design ensures minimal computational overhead compared to the 24 transformer layers of the original CLIP Vision Encoder. Dense features D<sup>i</sup> are derived by projecting the CLIP-encoded frame features (dimension 768) into the embedding space of the Large Language Model (dimension 4096) [\[9\]](#page-8-17) using a linear transformation.

Large Language Model. We utilize a pre-trained Vicuna-7B [\[9\]](#page-8-17) model to ground queried events using the adapted visual features. Built upon LLaMA [\[51\]](#page-10-18), this model consists of 32 transformer layers and has been fine-tuned on 70K user-shared conversations from ShareGPT [\[3\]](#page-8-23).

To enhance training efficiency, we adopt Low-Rank Adaptation (LoRA) [\[17\]](#page-8-13), a method commonly used in recent works [\[18,](#page-8-6) [46\]](#page-9-4). LoRA allows us to fine-tune the model without modifying its core weights by introducing lightweight, trainable modules. This significantly reduces computational overhead while retaining the model's flexibility. For our setup, we configure LoRA with a rank of r = 64 and a scaling factor of \alpha = 128 .

Training on ReVisionLLM Model We begin by pretrain-

<span id="page-11-0"></span>![](_page_11_Figure_10.jpeg)

Figure S1. Hierarchical Adapter processes the features extracted by the multimodal encoder, using both the video segments and the textual description of the queried event as inputs. It generates two types of temporal features: sparse and dense. Sparse features are computed through a combination of cross-attention, self-attention, and a feed-forward network, while dense features are generated using a linear projection layer.

ing the Linear Projector (Fig. [S1\)](#page-11-0) using the LCS-558K dataset from LLaVA [\[32\]](#page-9-15). This step aligns the CLS token from the CLIP Vision Encoder with the LLM's embedding space. The projector is trained for 1 epoch with a batch size of 128 and a learning rate of 1 × 10<sup>−</sup><sup>3</sup> .

Following pretraining, we implement a two-stage training pipeline for ReVisionLLM, maintaining a consistent learning rate of 1 × 10<sup>−</sup><sup>4</sup> . We use the AdamW optimizer [\[36\]](#page-9-23) with a warmup ratio 0.03 and a cosine scheduling strategy.

In the first stage, the Linear Projector is frozen, and the LLM is fine-tuned using LoRA on dense features, focusing on the lowest hierarchy level. This stage employs a batch size of 128, spanning 5 epochs for the MAD dataset and 1 epoch for the VidChapters-7M dataset. Sparse temporal features are introduced for upper hierarchies to reduce the LLM's visual input size. To enable sparse feature generation, we freeze the LoRA module and fine-tune the Cross-Attention, Self-Attention, and Feed-Forward layers of the Hierarchical Adapter, using a batch size of 32 for 1 epoch.

The training in this stage incorporates contrastive video segments where the queried event is absent. These segments are randomly sampled from hour-long videos and do not overlap with the temporal boundaries of the ground truth event. By selecting contrastive segments from the same video, the model is trained to handle challenging inference scenarios, where it must distinguish the queried event from visually and contextually similar scenes within the video.

In the second stage, all components of the Hierarchical Adapter are frozen, and a new LoRA module is finetuned for long-video processing on the Stage 2 objective. This stage employs a batch size of 8 and runs for 2 epochs. Two separate LoRA modules are utilized: one optimized for short video training and another adapted for long video processing.

Training on ReVisionLLM-U Model The unified model variant differs from our default model only in its training methodology. Training a unified model with shared parameters across all hierarchical levels poses notable challenges. To address this, we adopt an enhanced two-stage strategy. The first stage, including pretraining, is similar to the procedure used in the ReVisionLLM framework. In the second stage, however, we introduce a dual-training approach, where the ReVisionLLM-U framework is simultaneously trained on both short video clips and hour-long videos. This approach reduces the risk of catastrophic forgetting, ensuring the retention of short-segment representations.

A key challenge arises from the significant differences between short-segment and long-video data. Short segments utilize dense temporal features, while long videos rely on sparse temporal representations, such as CLS features. Additionally, short-segment training involves only video features, whereas long-video descriptions require both video and text features as inputs. To reconcile these differences, we implement an alternating batching strategy. During training, batches of short segments and long videos are alternately sampled, enabling the model to learn effectively from both data types.

This alternating training strategy not only mitigates catastrophic forgetting but also facilitates the successful training of the ReVisionLLM-U framework, which maintains shared parameters across all hierarchical levels. For ReVisionLLM-U, we employ the same hyperparameters as those used in ReVisionLLM , including learning rate, batch size, training epochs, optimizer, and scheduler (as detailed in the previous section).

Inference for ReVisionLLM. During inference, video segments are created using a sliding window approach. For the MAD dataset, each segment spans 125 seconds with a stride of 25 seconds, while for the VidChapters-7M dataset, segments are 500 seconds long with a stride of 100 seconds. From each segment, 250 frames are uniformly sampled. Sparse temporal features are then extracted using the Hierarchical Adapter and provided as input to the LLM to identify relevant video segments.

In our implementation, we employ two hierarchies with long videos. However, it can be extended to more levels based on video length. At the top level, 100 video segments (approx. 150 minutes) are processed simultaneously, while the second level processes 33 segments (approx. 50 minutes) simultaneously. Both hierarchies identify regions of interest, refined at the lowest hierarchical level. In this final hierarchy, all 250 dense temporal features from the selected segments are processed to pinpoint the precise event boundaries.

Inference for ReVisionLLM-I. The inverse model variant differs from our default model only in the inference method. In this variant, the inference begins at the lowest hierarchical level. All video segments are processed together in a single input batch, with their dense temporal features fed into the LLM. The LLM predicts temporal boundaries for multiple segments, often resulting in a high number of false positives. To mitigate this, the false positives are recursively passed through the second and third hierarchical levels, where they are filtered out. These upper levels retain only the most confident predictions, reducing errors. Finally, the confidence scores from the higher hierarchies are used to adjust and normalize the scores of the initial predictions, improving the overall accuracy of the model. We will release the code and pre-trained models for further use.

# S2. Calibration Confidence

Accurate calibration of our model's confidence is crucial for minimizing false positives and improving the overall effectiveness of our approach. To evaluate the impact of our training strategy on model calibration, we compare our method's performance to the baseline VTimeLLM [\[18\]](#page-8-6). A model is considered well-calibrated if its confidence scores match the actual proportion of correct predictions. Calibration is typically measured using the Expected Calibration Error (ECE) [\[41\]](#page-9-24), which quantifies the difference between predicted probabilities and observed outcomes by dividing the predicted confidence into discrete bins. A lower ECE value indicates better calibration, with an ECE of 0 representing perfect calibration.

For a dataset of N video segments and B evenly spaced bins b<sup>j</sup> , the ECE is computed as follows:

$$\widehat{ECE} = \sum_{j=1}^{B} \frac{|b_j|}{N} \left| \operatorname{conf}(b_j) - \operatorname{acc}(b_j) \right|$$

where conf(b<sup>j</sup> ) is the average confidence of samples in bin b<sup>j</sup> , acc(b<sup>j</sup> ) is the accuracy of predictions in bin b<sup>j</sup> , and |b<sup>j</sup> | is the number of samples in bin  $b_j$ . In our experiments,  $\operatorname{conf}(b_j)$  corresponds to the confidence score  $(R^i)$  defined in Section 3.4 of the main paper. A prediction is considered correct if the Intersection over Union (IoU) exceeds a threshold  $\tau_{\text{iou}} \in \{0.1, 0.3, 0.5\}$ .

By comparing ECE values across models, we can assess the effectiveness of our training strategy in improving calibration and generating more reliable confidence estimates.

<span id="page-13-0"></span>

| Model       | ECE @ IoU Thresholds ( $\tau_{\text{IoU}}$ ) |                      |                      |  |  |  |
|-------------|----------------------------------------------|----------------------|----------------------|--|--|--|
| 1,10001     | $\tau = 0.1 \downarrow$                      | $\tau=0.3\downarrow$ | $\tau=0.5\downarrow$ |  |  |  |
| VTimeLLM*   | 0.6231                                       | 0.6233               | 0.6237               |  |  |  |
| ReVisionLLM | 0.4614                                       | 0.4698               | 0.4791               |  |  |  |

Table S1. **Expected Calibration Error (ECE)** comparison between \*Baseline (VTimeLLM+CONE) and Our Model across IoU thresholds ( $\tau_{\text{IoU}}$ ). Our model demonstrates better calibration of confidence compared to the baseline across all IoU thresholds. Lower values indicate superior calibration performance.

Table S1 presents a comparison of Expected Calibration Error (ECE) values between the baseline VTimeLLM+CONE model and our ReVisionLLM across three Intersection over Union (IoU) thresholds:  $\tau_{\text{IoU}=0.1}$ ,  $\tau_{\text{IoU}=0.3}$ , and  $\tau_{\text{IoU}=0.5}$ . Lower ECE values indicate better calibration performance. In this analysis, we set the number of bins, B=10. Our model consistently outperforms the baseline across all thresholds, highlighting the effectiveness of calibrated fine-tuning in improving the reliability of confidence estimates. The increase in error with higher IoU thresholds (+0.01%) is negligible, further validating the robustness of our approach.

#### S3. Generalization: Text-to-Video Retrieval

#### **Problem Statement.**

We show the generalizability of ReVisionLLM on the task of text-to-video retrieval, where the goal is to retrieve the most relevant videos from a given video set  $\mathcal V$  for a provided query text t describing an event. This involves ranking the videos  $v \in \mathcal V$  based on their similarity to the query. For this problem, the input consists of a video v and a text t. We represent a video  $v \in \mathbb R^{T \times 3 \times H \times W}$  as a sequence of T image frames, where  $v = [v^1, v^2, \dots, v^F]^T$ , and each frame  $v^f$  has a spatial resolution of  $H \times W$  with 3 color channels. Text S is represented the queried sentence with  $N_s$  words.

**Task Adaptation for ReVisionLLM.** In our original grounding task, we work with a single long video, which we divide into a set of shorter segments, denoted as C. In contrast, for the text-to-video retrieval task, we handle multiple videos, forming a set  $\mathcal{V}$ . To address this, we combine all

the videos into one long sequence and predict the index of the relevant video. We uniformly sample 100 frames from each video and use our vision encoder to extract features, resulting in video features  $\hat{\mathcal{V}} \in \mathbb{R}^{|\mathcal{V}| \times 100 \times 768}$ . These features serve as our video segments, so  $C = \hat{\mathcal{V}}$ . Additionally, we extract textual features from the query using the text encoder, resulting in  $Q \in \mathbb{R}^{N_s \times 768}$ , where  $N_s$  is the number of words in the query.

For the input prompt, we use: "<video> Does the <event> happen in the video? Answer yes or no." The model is trained to respond "Yes." for relevant videos and "No." for irrelevant ones. In this setup, we use the ReVisionLLM-I variant, where we first process each video at the lowest hierarchical level, then revise our predictions recursively at higher levels. At the upper hierarchies, the prompt becomes: "<video> in which video can we see the <event> happening?" The model responds with "In video v.", where v denotes the index of the relevant video.

Finally, we rank the predicted videos based on the calibrated confidence scores from our LLM, ensuring more accurate retrieval results.

**Dataset Details.** The MSR-VTT dataset contains 10,000 videos, each associated with around 20 human-annotated captions. Notably, the captions for a single video often describe distinct parts of the content, aligning with our goal of matching a specific textual query to the most relevant frames within a video. The videos in this dataset range in duration from 10 to 32 seconds. For training, we use *9k-Train* split, including approximately 9,000 videos as outlined in [12]. Unless specified otherwise, our experiments use the *9k-Train* split for training. To evaluate our models, we adopt the *1k-Test* set from [65], which comprises 1,000 carefully selected video caption pairs.

#### **S4.** Aditional Qualitative Results

In this section we provide additional qualitative results for MAD, VidChapters-7M and MSRVTT datasets.

MAD Dataset: In Figure S2, the qualitative results illustrate ReVisionLLM's ability to localize subtle and tiny moments within extremely long videos, often set in visually similar scenes. For Event 1, ReVisionLLM accurately identifies the brief instance where a woman walks off amidst a dimly lit street setting, despite the challenge of nearly identical surrounding frames. This demonstrates the model's precision in grounding temporal boundaries in extended sequences where minor actions must be differentiated. In Event 2, the model successfully localizes the moment of a hazy orange sunrise over the sprawling streets, slums, and skyscrapers of Mumbai. This event, embedded within a visually repetitive urban setting, showcases ReVisionLLM's capacity to detect subtle temporal shifts in lighting and atmosphere. These examples

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure S2. Additional Qualitative Results for Long video temporal grounding on the MAD dataset. ReVisionLLM effectively identifies moments within hour-long movies by leveraging a recursive processing approach that operates at both the short video segment level and hour-long videos. Our VLM baseline completely fails to locate the events in these scenarios.

<span id="page-14-1"></span>![](_page_14_Picture_2.jpeg)

Figure S3. Qualitative Results for Text-to-video retrieval task on MSRVTT dataset. Here, we only show one representative video frame for four diverse queried events, which our model successfully retrieves.

emphasize the model's ability to handle the intricacies of long-form videos, identifying precise moments even when scenes exhibit minimal variation, thereby enabling enhanced retrieval and understanding of extended video content.

MSRVTT Dataset: In Figure [S3,](#page-14-1) we show a representative video frame for each of the four diverse events that our model successfully retrieved. Events (1) and (2) are visible for short time intervals inside the video, and our model effectively captures these moments due to its ability to focus on fine-grained details. In contrast, events (3) and (4) involve objects with varying speeds and directions, interacting dynamically with their environment. For example, in (3), the dog crosses the road, while in (4), the car moves along the road. These examples demonstrate our model's ability to comprehend and capture both visual and actionrelated details, enabling it to retrieve the most relevant video from a large dataset.

VidChapers-7M Dataset: In Figure [S4,](#page-15-0) the qualitative results from the VidChapters-7M dataset demonstrate ReVisionLLM's ability to enhance online video search and content retrieval across diverse platforms, including YouTube, educational portals, and news archives. In Event 1, ReVisionLLM accurately localizes a video latency test in a product review, capturing fine-grained temporal details essential for identifying technical demonstrations. Event 2 showcases the model's ability to navigate complex, sequential workflows, pinpointing design importing actions in Canva. Event 3 highlights ReVisionLLM's proficiency in localizing a news report on Trans Mountain pipeline construction, effectively distinguishing dynamic scenes involving machinery and landscapes—key for indexing news and documentaries. In Event 4, the model identifies a cameraman's emotional reaction during a podcast, illustrating its capability to understand contextual nuances and interactions. These results emphasize ReVisionLLM's effectiveness in improving content understanding and enabling precise event retrieval in long-form videos, with strong applicability to online video search engines and content recommendation systems.

<span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

Figure S4. Qualitative Results of long video temporal grounding on the VidChapters-7M dataset. ReVisionLLM demonstrates its ability to accurately locate specific events within hour-long YouTube videos across diverse content types, including tutorials, product reviews, news, and podcasts. This precise localization of video chapters has the potential to streamline video search engines and enhance user experience across various online platforms.