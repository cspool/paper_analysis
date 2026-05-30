# D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition

### Yiyang Huang, Yizhou Wang, Yun Fu

Northeastern University huang.yiyan@northeastern.edu, yunfu@ece.neu.edu

### Abstract

Video large language models (Vid-LLMs), which excel in diverse video-language tasks, can be effectively constructed by adapting image-pretrained vision-language models (VLMs). However, this adaptation remains challenging, as it requires processing dense and temporally extended visual inputs that exceed the capacity of image-based models. This paper identifies the perception bottleneck and token overload as key challenges in extending image-based VLMs to the video domain. To address these issues, we propose D-CoDe, a training-free adaptation framework that incorporates dynamic compression and question decomposition. Specifically, dynamic compression alleviates the perception bottleneck through adaptive selection of representative frames and content-aware aggregation of spatial tokens, thereby reducing redundancy while preserving informative content. In parallel, question decomposition mitigates token overload by reformulating the original query into sub-questions, guiding the model to focus on distinct aspects of the video and enabling more comprehensive understanding. Experiments demonstrate that D-CoDe effectively improves video understanding across various benchmarks. Furthermore, strong performance on the challenging long-video benchmark highlights the potential of D-CoDe in handling complex video-language tasks. Code is available at <https://github.com/hukcc/D-CoDe>.

### 1 Introduction

Video large language models (Vid-LLMs) integrate video inputs with textual instructions and demonstrate strong performance across a wide range of video-language tasks. However, constructing Vid-LLMs directly from pre-trained large language models is constrained by the scarcity of high-quality video-text data [\(Zhao et al.,](#page-11-0) [2024\)](#page-11-0).

<span id="page-0-0"></span>![](_page_0_Picture_8.jpeg)

Figure 1: Adapting image-pretrained VLMs to video faces two major challenges: the perception bottleneck, in which salient information is unevenly distributed across spatial and temporal dimensions, limiting the effectiveness of static compression in preserving key visual cues; and token overload, where video inputs yield substantially more visual tokens than images, exceeding the model's capacity for comprehensive understanding.

A more data-efficient alternative is to adapt imagepretrained vision-language models (VLMs), leveraging the structural similarity between images and videos.

Approaches for adapting image-pretrained VLMs to video can be broadly divided into trainingrequired and training-free methods. Trainingrequired methods typically fine-tune the visual encoder or cross-modal connector [\(Li et al.,](#page-10-0) [2023c,](#page-10-0) [2024\)](#page-10-1), align visual features between images and videos [\(Lin et al.,](#page-10-2) [2024\)](#page-10-2), incorporate additional modalities to broaden task coverage [\(Zhang et al.,](#page-11-1) [2023;](#page-11-1) [Cheng et al.,](#page-9-0) [2024\)](#page-9-0), or apply techniques like Direct Preference Optimization (DPO) [\(Zhang](#page-11-2) [et al.,](#page-11-2) [2024\)](#page-11-2) and slow-fast architectures [\(Huang](#page-9-1) [et al.,](#page-9-1) [2024\)](#page-9-1) to enhance temporal modeling and factual consistency. Despite their effectiveness, these methods often incur high computational cost. In contrast, training-free methods leverage imagepretrained VLMs without additional tuning, yet

still achieve competitive performance. Representative examples include IG-VLM [\(Kim et al.,](#page-9-2) [2024\)](#page-9-2), which constructs a grid-view image from sampled frames; FreeVA [\(Wu,](#page-11-3) [2024a\)](#page-11-3), which performs frame-level temporal aggregation; SF-LLaVA [\(Xu](#page-11-4) [et al.,](#page-11-4) [2024\)](#page-11-4), which employs a slow-fast architecture; and TS-LLaVA [\(Qu et al.,](#page-10-3) [2024\)](#page-10-3), which adopts a thumbnail-and-sampling strategy to generate compact and informative visual prompts.

However, despite their efficiency, training-free methods face two key challenges that limit scalability: perception bottleneck and token overload, as shown in Figure [1.](#page-0-0) Perception bottleneck arises from static compression strategies such as uniform frame sampling and spatial average pooling, which treat all content equally and discard salient information unevenly distributed across temporal and spatial dimensions, thereby limiting the model's ability to capture fine-grained visual cues. Token overload, in turn, occurs when compressed video inputs still contain substantially more visual tokens than static images, exceeding the processing capacity of image-pretrained VLMs and hindering the modeling of long-range dependencies and complex spatio-temporal structures essential for comprehensive understanding.

To overcome these challenges, we propose D-CoDe, a training-free framework that extends image-pretrained VLMs to video understanding by integrating dynamic compression and question decomposition. Specifically, dynamic compression augments temporal uniform sampling by selecting supplementary frames from segments exhibiting greater semantic variation, then filters out uninformative spatial tokens and merges semantically similar ones, thereby reducing redundancy while preserving informative visual cues. In parallel, question decomposition enhances the model's capacity to interpret dense visual inputs by reformulating complex queries into focused sub-questions, guiding attention to distinct aspects of the video and enabling comprehensive understanding.

Experiments show that D-CoDe consistently improves performance across a range of video understanding benchmarks, including multiple-choice VideoQA (NExT-QA, EgoSchema, IntentQA) and open-ended VideoQA (MSVD-QA, MSRVTT-QA, TGIF-QA, ANet-QA), which cover diverse video types from first- and third-person perspectives and span durations from short clips to long-form content. Notably, D-CoDe is the first training-free method to surpass training-required models on

EgoSchema, a challenging benchmark involving long-form egocentric videos and schema-driven questions.

Our contributions are summarized as follows:

- We analyze the key challenges in adapting image-pretrained VLMs to video understanding, focusing on the perception bottleneck and token overload.
- We introduce D-CoDe, a training-free framework that addresses the perception bottleneck via content-aware dynamic compression and mitigates token overload through question decomposition.
- Extensive experiments across various benchmarks validate the effectiveness of D-CoDe. In particular, strong performance on the longvideo task highlights its potential for complex video-language understanding.

### 2 Related Work

