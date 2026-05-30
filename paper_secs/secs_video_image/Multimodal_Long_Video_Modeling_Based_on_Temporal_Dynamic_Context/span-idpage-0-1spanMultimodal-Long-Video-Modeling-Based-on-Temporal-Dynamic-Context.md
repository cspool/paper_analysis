# <span id="page-0-1"></span>Multimodal Long Video Modeling Based on Temporal Dynamic Context

Haoran Hao<sup>1,2\*</sup>, Jiaming Han<sup>1\*</sup>, Yiyuan Zhang<sup>1</sup>, Xiangyu Yue<sup>1†</sup>

<sup>1</sup>MMLab, The Chinese University of Hong Kong <sup>2</sup>Nanjing University

#### **Abstract**

Recent advances in Large Language Models (LLMs) have led to significant breakthroughs in video understanding. However, existing models still struggle with long video processing due to the context length constraint of LLMs and the vast amount of information within the video. Although some recent methods are designed for long video understanding, they often lose crucial information during token compression and struggle with additional modality like audio. In this work, we propose a dynamic long video encoding method utilizing the temporal relationship between frames, named Temporal Dynamic Context (TDC). Firstly, we segment the video into semantically consistent scenes based on inter-frame similarities, then encode each frame into tokens using visual-audio encoders. **Secondly**, we propose a novel temporal context compressor to reduce the number of tokens within each segment. Specifically, we employ a querybased Transformer to aggregate video, audio, and instruction text tokens into a limited set of temporal context tokens. Finally, we feed the static frame tokens and the temporal context tokens into the LLM for video understanding. Furthermore, to handle extremely long videos, we propose a training-free chain-of-thought strategy that progressively extracts answers from multiple video segments. These intermediate answers serve as part of the reasoning process and contribute to the final answer. We conduct extensive experiments on general video understanding and audio-video understanding benchmarks, where our method demonstrates strong performance. The code and models are available at https://github.com/Hoar012/TDC-Video.

#### 1. Introduction

Recently, advances in large language models (LLMs) [2, 41, 49] have significantly improved their ability in language processing and generation. Researchers have extended these models to other modalities, such as vision [30, 33, 40, 81], audio [10, 17] and point clouds [22, 23],

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Figure 1. Comparison of Visual and Audio Encoding in Video Modeling. (a) Existing methods encode each modality separately and then concatenate them, leading to inconsistencies and difficulties in handling long videos. (b) We propose Temporal Dynamic Context (TDC) compression, which incorporates both static visual features and dynamic video context to represent videos more effectively. This approach enables better multimodal integration and efficient compression for long videos.

leading to the development of powerful multimodal LLMs (MLLMs). These MLLMs achieve strong performance in various tasks, such as image captioning [6, 24] and question answering [40, 54]. However, video understanding remains a challenging problem due to the interplay of multiple modalities [10, 17] and the complexity of large-scale information [37, 52], especially in long videos.

A key challenge in long video processing is efficiently representing videos to minimize redundancy between frames while preserving crucial details. Some attempts have been made to address this challenge [35, 51, 59, 61], but they typically compress video tokens based

<sup>\*</sup> Equal contribution † Corresponding author

<span id="page-1-0"></span>on visual similarity between frames, while overlooking high-level dynamic semantic information. This limits the model's ability to capture comprehensive video information within a fixed number of tokens, increasing the difficulty of understanding the content.

Another challenge lies in integrating multiple modalities for comprehensive video understanding. For instance, when watching a movie, humans naturally integrate speech, background music, visual scenes, and subtitles to grasp the full context. However, most existing video LLMs are limited to visual and textual modalities and struggle to tackle audio. Although some works [\[10,](#page-9-1) [17,](#page-9-2) [77\]](#page-12-1) represented by Video-LLaMA [\[77\]](#page-12-1) try to incorporate both audio and visual information, their approach of simply concatenating modality tokens often leads to suboptimal performance by treating different modalities separately (refer to Fig. [1](#page-0-0) (a)). Developing a unified representation method that effectively connects information between modalities is essential to improve multimodal video understanding.

To address these challenges, we propose a video representation model that integrates multiple modalities within a unified video context and achieves effective token compression. As shown in Fig. [1](#page-0-0) (b), (1) we segment the video into scenes based on the visual consistency between frames, which are then encoded separately. (2) Each video clip is represented using both static visual features of the key frame and dynamic video context of the video. We first extract per-second features using a visual encoder and an audio encoder. The feature of the first frame is fully retained as a static representation, while the features of subsequent frames are compressed by a Q-Former [\[12\]](#page-9-4), based on their temporal consistency and differences relative to the static frame. This approach enables effective token compression while integrating multiple modalities within the video context. (3) To enhance the effectiveness of the model on extremely long video, we introduce the Long Video Chain-of-Thought (LVCoT) strategy, which guides the model to process long videos step by step before integrating the whole video to generate the final output.

We train models of various sizes using a multi-stage strategy, progressively optimizing them for vision-language alignment, video instruction tuning, and audio-video instruction tuning. We evaluate our models on a range of video benchmarks, including MVBench [\[34\]](#page-10-9), PerceptionTest [\[47\]](#page-11-6), EgoSchema [\[43\]](#page-10-10), MLVU [\[86\]](#page-12-2), and Video-MME [\[16\]](#page-9-5). Furthermore, we assess their performance on audio-video question-answering benchmarks, such as Music-QA [\[31\]](#page-10-11) and AVSD [\[3\]](#page-9-6). Resuls show that our models achieve strong performance on both video and multimodal understanding tasks. Our main contributions include:

• We propose a framework for multimodal video modeling, which represents videos using both static visual features and dynamic multimodal context, effectively

- integrating visual and audio information within a unified video context.
- We introduce the Long Video Chain-of-Thought (LV-CoT), a training-free strategy that enables MLLMs to process and reason over long videos step by step, enhancing the performance of existing models.
- We conduct extensive experiments with MLLMs of various sizes and evaluate them on multiple benchmarks, including general video question answering, long video understanding, and audio-visual video comprehension. Our models achieve strong performance, advancing the field of multimodal long video understanding.

