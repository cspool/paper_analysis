# <span id="page-0-0"></span>Owl-1: Omni World Model for Consistent Long Video Generation

Yuanhui Huang<sup>1</sup> Wenzhao Zheng<sup>1</sup>,\* Yuan Gao<sup>2</sup> Xin Tao<sup>2</sup> Pengfei Wan<sup>2</sup> Di Zhang<sup>2</sup> Jie Zhou<sup>1</sup> Jiwen Lu<sup>1</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Kuaishou Technology

huangyh22@mails.tsinghua.edu.cn; wenzhao.zheng@outlook.com

![](_page_0_Picture_4.jpeg)

Figure 1. Owl-1 approaches consistent long video generation with an omni world model, which models the evolution of the underlying world with latent state, explicit observation and world dynamics variables.

## Abstract

*Video generation models (VGMs) have received extensive attention recently and serve as promising candidates for general-purpose large vision models. While they can only generate short videos each time, existing methods achieve long video generation by iteratively calling the VGMs, using the last-frame output as the condition for the next-round generation. However, the last frame only contains shortterm fine-grained information about the scene, resulting in inconsistency in the long horizon. To address this, we propose an Omni World modeL (Owl-1) to produce longterm coherent and comprehensive conditions for consistent long video generation. As videos are observations of the underlying evolving world, we propose to model the longterm developments in a latent space and use VGMs to film them into videos. Specifically, we represent the world with a latent state variable which can be decoded into explicit video observations. These observations serve as a basis for anticipating temporal dynamics which in turn update* *the state variable. The interaction between evolving dynamics and persistent state enhances the diversity and consistency of the long videos. Extensive experiments show that Owl-1 achieves comparable performance with SOTA methods on VBench-I2V and VBench-Long, validating its ability to generate high-quality video observations. Code:* <https://github.com/huang-yh/Owl>*.*

## 1. Introduction

With the success of image generative models [\[2,](#page-9-0) [12,](#page-9-1) [23,](#page-10-0) [27,](#page-10-1) [28,](#page-10-2) [41\]](#page-11-0), video generation [\[14,](#page-9-2) [15,](#page-10-3) [17,](#page-10-4) [29,](#page-10-5) [32\]](#page-10-6) have also garnered increasing attention. While existing video generation models (VGMs) [\[3,](#page-9-3) [7,](#page-9-4) [26,](#page-10-7) [35\]](#page-10-8) have achieved commercialgrade performance, the durations of videos are still short. The long video generation methods [\[18,](#page-10-9) [20,](#page-10-10) [34,](#page-10-11) [37,](#page-11-1) [43\]](#page-11-2) remedies this issue by focusing on improving the length and consistency of generated videos, facilitating a variety of newly rising tasks such as video extension [\[35\]](#page-10-8), film generation [\[40\]](#page-11-3) and world simulation [\[24\]](#page-10-12).

Despite the promising applications, how to increase the video length while preserving consistency remains an open question. Several work [\[1,](#page-9-5) [43\]](#page-11-2) investigates the 3D vari-

<sup>\*</sup>Project leader.

<span id="page-1-0"></span>ational autoencoder (VAE) which compresses a video in both spatial and temporal dimensions in order to generate long videos in a single denoising process of a latent diffusion model. Although the video consistency is inherently guaranteed in the diffusion process, the length of the generated videos is limited by the computational resources end further expanding the video length requires retraining the diffusion model. Another line of work approaches long video generation through divide-and-conquer, which first generates the key frames of a long video and then interpolates between successive key frames [\[11,](#page-9-6) [38\]](#page-11-4). However, these methods are dependent on the duration of the training video data, thus lacking scalability. In addition, iteratively prompting a video diffusion model for short clip generation is also a promising paradigm to generate long videos [\[9,](#page-9-7) [13,](#page-9-8) [32\]](#page-10-6). To achieve consistency, these approaches design their prompts based on historical clips and texts in each iteration. Nonetheless, current practices for prompt construction usually take the last frames of the direct adjacent clip, which only contain short-term information about the scene, resulting in inconsistency in the long horizon.

In this paper, we propose an Omni World modeL (Owl-1) to produce long-term coherent and comprehensive conditions for consistent long video generation. Since videos are observations of the underlying evolving world, which establishes the temporal consistency of videos, we propose to model the long-term developments in a latent space and use VGMs to film them into videos. To elaborate, we represent the world with a latent state variable which encodes both the current and historical information about the underlying world. Similar to the filming process, the state variable decodes into video clips with VGMs as observations of the world. Based on these observations, we further anticipate future world dynamics which drive the evolution of the world and update the latent state variable. Up to now, we have constructed an autoregressive state-observationdynamics model to simulate the closed-loop evolution of the world, which improves the coherence of long videos with the consistent latent states, and enhances the content diversity with dynamics predictions. To effectively model the relationship of these three components, we employ a pretrained large multimodal model (LMM) to take advantage of its general reasoning ability. Additionally, we adopt a video diffusion model to decode latent states into short video clips. Owl-1 achieves comparable performance with SOTA methods on VBench-I2V and VBench-Long, validating its ability to generate high-quality video observations.

## 2. Related Work

Short video generation. In the realm of computer vision, video generation has emerged as a pivotal area of research, garnering significant attention due to its broad applications. Short video generation investigates how to generate videos

![](_page_1_Picture_4.jpeg)

Figure 2. Iterative long video generation. Conventional iterative long video generation methods use the last-frame output as the condition for the next-round generation, which lacks long-term consistency. Our method constructs an omni world model for comprehensive conditioning.

based on text (and/or image) conditions, where the alignment between the generated video and the given conditions is one of the primary evaluation criteria. For text conditions, most methods [\[3,](#page-9-3) [15,](#page-10-3) [35\]](#page-10-8) encode them with pretrained text encoders [\[22,](#page-10-13) [25\]](#page-10-14), and incorporate the textual features using cross attention. In addition, image-to-video models requires the generated video to incorporate the specified image conditions. In order to effectively fuse the fine-grained visual information, several approaches [\[13,](#page-9-8) [43\]](#page-11-2) directly replace or concatenate the diffusion features with the encoded features of the image condition. Other methods [\[35\]](#page-10-8) also transform the image condition into tokens similar to the textual features, and apply cross attention between the diffusion features and image tokens to preserve coarser level of details such as visual styles and background. Our Owl-1 uses both the latent state and optional image conditions from the last clip for consistent and smooth generation of the next clip.

Long video generation. As an important extension of the application scope of video generation models, long video generation focuses on improving the length and consistency of generated videos. To achieve this, several work attempts to enhance the video durations in a single generation process, by designing 3D VAEs that are able to compress longer videos [\[1,](#page-9-5) [43\]](#page-11-2) or investigating the temporal modules in VGMs for efficient generation [\[35\]](#page-10-8). Although the end-to-end generation pipeline inherently guarantees the video consistency, the length of generated videos is constrained by limited computational resources. To remedy this issue, the divide-and-conquer approach simplifies the task by first identifying key frames that outline the main narrative and then generating the intervening frames to create a cohesive long video. However, these methods are dependent on training video data of long durations which are still

<span id="page-2-2"></span><span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 3. Overall framework. Our Owl-1 models the evolution of the world with the latent state variables s, and film them into video observations o along the generation process. We also incorporate anticipation of the world dynamics d to explicitly drive the evolution.

insufficient, thus lacking scalability.

On the other hand, the temporal autoregressive paradigm adopts a sequential approach to generate short video segments based on prior conditions. Within this paradigm, various models have been employed, including diffusion models [\[13,](#page-9-8) [32\]](#page-10-6), spatial autoregressive models [\[19\]](#page-10-15), and GAN models [\[30\]](#page-10-16). The key challenge here is to ensure the consistency between temporally distant clips to achieve coherent long video generation. Most work directly uses the last frames of the previous generated clip as visual clues for the next-round generation, which only contain short-term information about the scene, resulting in a limited temporal receptive field and inconsistency in the long horizon. In contrast, our Owl-1 employs the latent state variable which encodes both the current and historical information about the underlying world to achieve extensive temporal receptive field and video consistency.

Video generation world models. Video generation models are promising candidates for world models [\[44\]](#page-11-5) which aims to model the evolution of the environment. For videos of short durations, the generated content may reflect certain physical laws [\[21\]](#page-10-17), indicating that the video generation model has learned some general knowledge about the world. For a longer horizon, the emphasis of video generation world models lies in capturing overall dynamics that drive environment evolution [\[44\]](#page-11-5). Although such models have been proposed in autonomous driving [\[33,](#page-10-18) [42\]](#page-11-6) and embodied intelligence [\[5\]](#page-9-9), they can only predict structured actions instead of general world dynamics in the form of natural language. As for general video generation, most existing methods focus on improving the alignment of generated videos and given text conditions, lacking the ability to anticipate the world dynamics. In addition to conditional video generation, our Owl-1 is capable of predicting future dynamics to generate long videos with diverse content.

## 3. Proposed Approach

In this section, we present our method of omni world model for consistent long video generation. To formulate this task mathematically, we aim to generate a long video consisting of a sequence of video clips v = {..., ot−1, ot, ot+1, ...} given a starting image I and a text description d<sup>0</sup> as input.

### 3.1. Omni World Model

Videos are fundamentally recorded observations of the underlying evolving world, whose long-term consistency is inherently guaranteed in the coherence of the world itself. Therefore, maintaining consistency in long videos from the perspective of the implicit world is a more reasonable and essential approach, compared with the explicit pixel-space methods. However, the real world constitutes a complex high-dimensional system, and the cost of directly modeling such a system is unacceptable. Inspired by the world models in the field of embodied intelligence [\[4\]](#page-9-10), we represent the world using a set of latent state variables {..., st−1, st, ...}. Each state s<sup>t</sup> not only encodes information about the world at the current moment t, but also incorporates historical information about the evolution of the world, i.e. {..., st−2, st−1}. Since state variables serve as an purely implicit representation of the world, we introduce a state decoder D to obtain explicit video observations {..., ot−1, ot, ...} from the state variables:

<span id="page-2-0"></span>
$$\mathbf{o}_t = \mathcal{D}(\mathbf{s}_t, \mathbf{o}_{t-1}),\tag{1}$$

where we incorporate the last observation ot−<sup>1</sup> to ensure short-term fine-grained smoothness of successive observations, while the current state s<sup>t</sup> is primarily responsible for the long-term consistency.

Most work approaches long video generation in the same way as short video generation, which overlooks the variation of video content in a long horizon, resulting in repeated generation of homogeneous content. In our omni world world, we explicitly take the world dynamics {..., dt−1, dt, ...} into consideration which drives the evolution of the underlying world and takes the form of texts. To elaborate, we predict the current world dynamics d<sup>t</sup> from state variables and video observations:

<span id="page-3-0"></span>
$$\mathbf{d}_t = f(\mathbf{s}_t, \mathbf{o}_t),\tag{2}$$

where f(·) denote the world dynamics prediction function. Furthermore, current world dynamics, in turn, updates the state variable, advancing the evolution of the world:

<span id="page-3-1"></span>
$$\mathbf{s}_{t+1} = g(\mathbf{s}_t, \mathbf{d}_t),\tag{3}$$

where g(·) represents the world state prediction function. With Eq. [\(1\)](#page-2-0)[\(2\)](#page-3-0)[\(3\)](#page-3-1), we have constructed a state-observationdynamics triplet to simulate the evolution of the world and also obtain consistent video clips along the evolution, as in Figure [3.](#page-2-1) This formulation improves the consistency of long videos by modeling the underlying world and generates diverse video clips through explicit dynamics prediction.

## 3.2. Comprehensive Condition from Latent State

The key challenge in the temporal autoregressive paradigm for long video generation lies in the design of the condition used for generating the next clip. Most existing methods directly take the last frames of the previous clip as condition, which only considers the short-term smoothness between consecutive clips, and overlooks consistency issues in the long-term such as style, character identity, background etc. Our Owl-1 takes the latent state variable as a comprehensive condition for the long-term consistency, because the derivation of the current state s<sup>t</sup> inherently includes the information of all previous observations:

$$\mathbf{s}_{t+1} = g(\mathbf{s}_t, f(\mathbf{s}_t, \mathbf{o}_t)) = h(\mathbf{s}_0, \mathbf{o}_0, ..., \mathbf{o}_{t-1}, \mathbf{o}_t), \quad (4)$$

which is derived by plugging Eq. [\(2\)](#page-3-0) into Eq. [\(3\)](#page-3-1) and iteratively replacing s<sup>t</sup> with st−<sup>1</sup> and ot−1.

For implementation of our Owl-1, we take advantage of a large multimodal model (LMM) to instantiate the functions f(·) and g(·), in order to take advantage of its common knowledge from the large scale pretraining on textual and visual data, and its large receptive field as well. And we instantiate the state decoder D with a pretrained video diffusion model for their capability to generate short videos of high quality. To incorporate the state-observation-dynamics triplet into the framework of LMM, we design the format of the input and output sequences:

<span id="page-3-2"></span>
$$Seq = [..., \mathbf{s}_t, \mathbf{o}_t, \mathbf{d}_t, ...], \tag{5}$$

where we iteratively feed the basic triplet (st, ot, dt) into the LMM. For latent state st, we use a set of Q learnable query embeddings as input tokens to LMM, since it has no ground truth and thus cannot be quantized into discrete tokens. As for video observation ot, we uniformly sample a number of key frames from the current clip, and use the pretrained vector-quantized variational autoencoder (VQVAE) of the LMM to transform the key frames into visual tokens. Moreover, we directly use the text tokenizer of the LMM to convert the textual world dynamics d<sup>t</sup> into discrete input tokens. In summary, we use the LMM to model the closedloop state-observation-dynamics evolution, where the state variable aggregates the information of all previous video observations (Eq. [\(5\)](#page-3-2)) and serves as comprehensive condition for the next-round generation (Eq. [\(1\)](#page-2-0)).

### 3.3. Anticipation of Future Dynamics

In the context of long video generation, anticipating future dynamics is crucial for maintaining consistency and coherence across extended video sequences. Our Owl-1 predicts and integrates these future dynamics d<sup>t</sup> into the evolution of the latent state s, thereby enriching the content diversity and ensuring temporal consistency in the generated videos. As indicated by Eq. [\(2\)](#page-3-0), the prediction of world dynamics d<sup>t</sup> relies on the current video observation o<sup>t</sup> as short-term reference, and the current latent state s<sup>t</sup> as source of long-term information. Once the current dynamics d<sup>t</sup> is predicted, we integrate it into the latent state variable s<sup>t</sup> to update the world state for the next-round generation. To train the dynamics anticipation ability of the LMM, we adopt the nexttoken prediction paradigm and use the textual ground truth dynamics for teacher-forcing supervision.

The anticipation of future dynamics is important for content diversity of generated long videos and world modeling. By enabling the anticipation of subsequent events within a video sequence, our Owl-1 enhances the richness of the generated content, moving beyond repeated generation of homogeneous content to capturing the essence of dynamic scenarios. Furthermore, future dynamics prediction serves as a cornerstone for constructing plausible world models, which are instrumental in simulating and understanding complex environments. Our Owl-1 not only predicts realworld behaviors but also allow for the incorporation of control mechanisms by replacing the anticipated dynamics with user-input control signals, facilitating the generation of content that is not only predictable but also controllable.

## 3.4. Multi-Stage Training

Several challenges exist in the training process of our Owl-1: 1) Since the LMM and the video diffusion model are separately pretrained, it is nontrivial to align these two models. 2) Our Owl-1 is designed for long-term world modeling, which requires video data with long duration and

<span id="page-4-4"></span><span id="page-4-3"></span>

| Table 1. Evaluation results on VBench-I2V. Subj., Bkgd. and Consist. denote Subject, Background and Consistency, respectively. Bold: |
|--------------------------------------------------------------------------------------------------------------------------------------|
| best results. Underline: second best. Our Owl-1 achieves comparable performance with state-of-the-art image-to-video models.         |

| Method                  | Video-Image<br>Subj. Consist. | Video-Image<br>Bkgd. Consist. | Subject<br>Consist. | Bkgd.<br>Consist. | Motion<br>Smoothness | Dynamic<br>Degree | Aesthetic<br>Quality | Imaging<br>Quality | Temporal<br>Flickering | Total<br>Score |
|-------------------------|-------------------------------|-------------------------------|---------------------|-------------------|----------------------|-------------------|----------------------|--------------------|------------------------|----------------|
| VideoCrafter-I2V [7]    | 91.17                         | 91.31                         | 97.86               | 98.79             | 98.00                | 22.60             | 60.78                | 71.68              | 98.19                  | 85.14          |
| ConsistI2V [26]         | 95.82                         | 95.95                         | 95.27               | 98.28             | 97.38                | 18.62             | 59.00                | 66.92              | 97.56                  | 86.84          |
| SEINE-512x512 [9]       | 97.15                         | 96.94                         | 95.28               | 97.12             | 97.12                | 27.07             | 64.55                | 71.39              | 97.31                  | 88.42          |
| I2VGen-XL [39]          | 96.48                         | 96.83                         | 94.18               | 97.09             | 98.34                | 26.10             | 64.82                | 69.14              | 98.58                  | 88.48          |
| Animate-Anything [10]   | 98.76                         | 98.58                         | 98.90               | 98.19             | 98.61                | 02.68             | 67.12                | 72.09              | 98.14                  | 89.76          |
| SVD-XT-1.0 [3]          | 97.52                         | 97.63                         | 95.52               | 96.61             | 98.09                | 52.36             | 60.15                | 69.80              | 99.09                  | 89.87          |
| DynamiCrafter-1024 [35] | 98.17                         | 98.60                         | 95.69               | 97.38             | 97.38                | 47.40             | 66.46                | 69.34              | 97.63                  | 90.25          |
| Owl-1                   | 97.40                         | 97.29                         | 97.28               | 98.54             | 98.92                | 21.63             | 61.89                | 69.66              | 98.69                  | 89.15          |

dense captions. However, given the scarcity of such highquality data, it would be infeasible to train these large models with billions of parameters directly for the purpose of world model. Therefore, we carefully design a multi-stage training scheme for our Owl-1 which consists of alignment, generative pretraining and world model training.

The alignment stage primarily enforces the consistency between the state variables s<sup>t</sup> from the LMM and the textual conditions of the video diffusion model, which serves as a good initialization for the subsequent generative pretraining stage. Specifically, we freeze the video diffusion model to preserve its ability of generating short videos and only trains the LMM at this stage. We use general datasets for video generation in this stage, which provide videos of varying lengths and one single description for each video. For each sample (v, t), we first segment the video into short clips of fixed length v = {..., ot−1, ot, ot+1, ...}, and construct the input sequence as:

<span id="page-4-0"></span>
$$Seq_{align} = [\mathbf{I}, \mathbf{t}, \mathbf{s}_0, \mathbf{o}_0, \mathbf{t}, ..., \mathbf{s}_t, \mathbf{o}_t, \mathbf{t}, ...],$$
(6)

where I represents the first frame, and we use the same text dynamics t for every triplet since the general video generation datasets do not provide dense captions for every clip and the content of the video remains largely unchanged throughout its duration. To train the LMM to align with the textual conditions of video diffusion model, we minimize the L2 distance between the latent state s<sup>t</sup> and the text features from the text encoder of video diffusion model T :

<span id="page-4-1"></span>
$$\mathcal{L}_{align} = \text{MSE}(\mathbf{s}_t, \mathcal{T}(\mathbf{t})).$$
 (7)

The alignment stage enforces the consistency between the state variable and the textual conditions of the video diffusion model, which is pivotal for the stability of subsequent training given the distinction between the LMM and the video diffusion model.

The generative pretraining stage finetunes the LMM and the video diffusion model in a joint manner, to train the ability of the video diffusion model as the state decoder (Eq. [1\)](#page-2-0), which translates the latent state s<sup>t</sup> into explicit video observations ot. We adopt the same general video generation datasets and thus the same input sequence in Eq. [\(6\)](#page-4-0) for this stage. Since the purpose of the MSE loss in the alignment stage (Eq. [\(7\)](#page-4-1)) is only to provide an initialization, we discard it in the generative pretraining stage and substitute the latent state s<sup>t</sup> for the original text condition of the video diffusion model. We supervise these two models with only the denoising target of diffusion models:

<span id="page-4-2"></span>
$$\mathcal{L}_{pretrain} = ||\epsilon - \hat{\epsilon}_{\mathcal{D}}(\mathbf{o}_{t,m}, m, \mathbf{s}_t, \mathbf{o}_{t-1})||_2^2, \quad (8)$$

where m, ot,m represent the denoising timestamp and the noisy video observation, respectively. By training the video diffusion model with the latent state s<sup>t</sup> as conditional input, we turn the video diffusion model into a photographer who films the latent world into explicit videos.

The world model training stage mainly incorporates the prediction of world dynamics d<sup>t</sup> into our Owl-1. It is based on the large scale pretraining of the second stage, which unifies the LMM and the video diffusion model as a preliminary Owl-1 capable of generating latent states s<sup>t</sup> as comprehensive conditions for video clip generation. Now we further finetune the LMM and video diffusion model on a small amount of video data with longer duration and dense captions due to its scarcity. To achieve this, we change the input sequence of the LMM as:

$$Seq = [\mathbf{I}, \mathbf{t}, \mathbf{s}_0, \mathbf{o}_0, \mathbf{d}_0, ..., \mathbf{s}_t, \mathbf{o}_t, \mathbf{d}_t, ...],$$
(9)

which incorporates the provided dense caption of each video clip as world dynamics dt. For supervision, we employ the next-token prediction paradigm and supervise d<sup>t</sup> with its textual ground truth in a teacher-forcing style. Also, we still keep the denoising target in Eq. [\(8\)](#page-4-2) at this stage.

## 4. Experiments

## 4.1. Datasets and Benchmarks

Geneal video generation datasets. We use two general purpose video generation datasets in the first two training stages. The WebVid dataset [\[1\]](#page-9-5) comprises over 10 million captioned videos sourced from the internet, totaling approximately 52K hours of footage. This large-scale text-video

<span id="page-5-1"></span><span id="page-5-0"></span>Table 2. **Evaluation results on VBench-Long.** Subj., Bkgd., Cons., Temp., Flick., Smooth., Relation. and Appear. denote Subject, Background, Consistency, Temporal, Flickering, Smoothness, Relationship and Appearance, respectively. **Bold:** best results. <u>Underline</u>: second best. Our model achieves comparable performance with the open-sourced video generation models.

| Method            | Subj.<br>Cons. | Bkgd.<br>Cons. | Temp.<br>Flick. | Motion<br>Smooth. | Dynamic<br>Degree | Aesthetic<br>Quality | Imaging<br>Quality | Object<br>Class | Multiple<br>Objects | Human<br>Action | Color | Spatial<br>Relation. | Scene        | Appear.<br>Style | -     | Overall<br>Cons. |       |
|-------------------|----------------|----------------|-----------------|-------------------|-------------------|----------------------|--------------------|-----------------|---------------------|-----------------|-------|----------------------|--------------|------------------|-------|------------------|-------|
| Mira [18]         | 96.23          | 96.92          | 98.29           | 97.54             | 60.33             | 42.51                | 60.16              | 52.06           | 12.52               | 63.80           | 42.24 | 27.83                | 16.34        | 21.89            | 18.77 | 18.72            | 71.87 |
| OpenSoraPlan [20] | 95.73          | 96.73          | 99.03           | 98.28             | 47.72             | 56.85                | 62.28              | 76.30           | 40.35               | 86.80           | 89.19 | 53.11                | 27.17        | 22.90            | 23.87 | 26.52            | 78.00 |
| OpenSora [43]     | 96.75          | 97.61          | 99.53           | 98.50             | 42.39             | 56.85                | 63.34              | 82.22           | 51.83               | 91.20           | 90.08 | 68.56                | 42.44        | 23.95            | 24.54 | 26.85            | 79.76 |
| Mochi-1           | 96.99          | 97.28          | 99.40           | 99.02             | 61.85             | 56.94                | 60.64              | 86.51           | 50.47               | 94.60           | 79.73 | 69.24                | 36.99        | 20.33            | 23.65 | 25.15            | 80.13 |
| CogVideoX [37]    | 96.23          | 96.52          | 98.66           | 96.92             | 70.97             | 61.98                | 62.90              | 85.23           | 62.11               | 99.40           | 82.81 | 66.35                | 53.20        | 24.91            | 25.38 | 27.59            | 81.61 |
| Kling             | 98.33          | 97.60          | 99.30           | 99.40             | 46.94             | 61.21                | 65.62              | 87.24           | 68.05               | 93.40           | 89.90 | 73.03                | 50.86        | 19.62            | 24.17 | 26.42            | 81.85 |
| Vchitect-2.0 [34] | 96.83          | 96.66          | 98.57           | 98.98             | 63.89             | 60.41                | 65.35              | 86.61           | 68.84               | 97.20           | 87.04 | 57.55                | 56.57        | 23.73            | 25.01 | 27.57            | 82.24 |
| Gen-3             | 97.10          | 96.62          | 98.61           | 99.23             | 60.14             | 63.34                | 66.82              | 87.81           | 53.64               | 96.40           | 80.90 | 65.09                | <u>54.57</u> | 24.31            | 24.71 | 26.69            | 82.32 |
| MiniMax           | 97.51          | 97.05          | 99.10           | 99.22             | <u>64.91</u>      | <u>63.03</u>         | 67.17              | <u>87.83</u>    | 76.04               | 92.40           | 90.36 | 75.50                | 50.68        | 20.06            | 25.63 | 27.10            | 83.41 |
| Owl-1             | 98.29          | 98.61          | 99.84           | 99.35             | 13.19             | 60.64                | 66.33              | 91.31           | 43.04               | 85.67           | 87.92 | 67.58                | 51.46        | 24.83            | 24.25 | 25.10            | 79.65 |

dataset encompasses a diverse range of content across multiple domains, making it highly suitable for tasks such as video-text retrieval and video generation. We take around 400K randomly sampled videos from this dataset. The Panda70m dataset [8] includes 70 million videos with an average length of 8s along with their high-quality textual captions from an automatic captioning pipeline leveraging multimodal inputs and multiple cross-modal teacher models. We randomly sample 2M videos from this dataset.

Dense video captioning datasets. Due to the lack of datasets specifically focusing on the dynamics driving the progression of videos, we utilize dense video caption datasets as an alternative. The ActivityNet Captions dataset [6] contains 20K YouTube videos with 100K caption annotations and an average duration of 120 seconds. The majority of the videos contain more than three annotated events, each associated with corresponding time span and manually written sentences, averaging 13.5 words per annotation. The Vript dataset [36] represents a large-scale, finegrained video-text dataset comprising 12K high-resolution videos and over 400K segments, which are densely annotated in the form of video scripts. The average lengths of video clips and captions are 11s and 145 words, respectively. We use the training splits of these two datasets.

**VBench.** VBench [17] is a comprehensive and hierarchical benchmark framework, which dissects video generation quality into 16 specific and disentangled dimensions, such as subject identity inconsistency, motion smoothness, temporal flickering, and spatial relationship, each equipped with tailored prompts and evaluation methodologies. VBench possesses three key attributes: its comprehensive coverage of diverse video generation aspects, alignment with human perception, and insights into current models' performance across various dimensions and content.

#### 4.2. Implementation Details

We use the Chameleon model [31] as the LMM, and the DynamiCrafter-1024 [35] as the video diffusion model. For the trainable parameters, we finetune the LMM using

LoRA [16] and finetune all the parameters of the video diffusion model. For the segmentation of videos, we divide each video into equal clips of 4 seconds as observations  $\mathbf{o}_t$ , and sample 2 frames from each clip as input to the LMM. We set the length of learnable state queries  $\mathbf{s}_t$  as 128. For the alignment and generative pretraining stages, we train on a total of 2.4M videos from WebVid and Panda10m for 10K and 10K iterations, respectively. For the world model training stage, we train on a total of 20K videos from ActivityNet Captions and Vript for 1K steps.

#### 4.3. General Video Generation

We evaluate our Owl-1 on two benchmarks of VBench [17], i.e. VBench-I2V and VBench-Long, for its ability of generating short and long videos, respectively. We report the results on VBench-I2V in Table 1, for which we generate 2s short videos. Our Owl-1 achieves comparable performance with state-of-the-art methods for short video generation, excelling at the aspects of motion smoothness, background consistency and temporal flickering. This proves the effectiveness of the state decoding mechanism which films latent state varibles into explicit video observations. However, we do observe a decrease in the score for dynamic degree compared with DynamiCrafter, which we attribute to the lack of training video data with high motion levels.

We report the results on VBench-Long in Table 2, where we generate videos of 7s long, similar to the other methods. Since the video diffusion model we use, i.e. DynamiCrafter, requires both an image and a text description as input to generate the first clip, we adopt an image diffusion model SD2.1-v [27] to generate the first frame of the video from the given text prompt. Our model achieves comparable performance with the open-sourced video generation models, e.g. OpenSora, on this benchmark. Similar to the results on VBench-I2V, our Owl-1 performs better at subject and background consistency, temporal flickering and motion smoothness, while its dynamic degree is lower than other methods, which could be improved through further training with videos of higher motion level.

<span id="page-6-0"></span>![](_page_6_Picture_0.jpeg)

Figure 4. Video frames visualization results for general video generation. We sample 5 frames from each of our generated videos, which lasts 8 seconds. Our Owl-1 generates videos covering various topics with good quality.

We visualize the generated videos of our Owl-1 in Figure [4.](#page-6-0) Each of these generated videos lasts 8 seconds, and we uniformly sample 5 frames from each of them. Owl-1 is able to generate both comprehensive and realistic videos covering various topics, including human actions, animals, natural scenery, etc. Although we do not predict world dynamics when generating these videos, the temporal consistency remains excellent, demonstrating the effectiveness of our proposed conditional video generation approach based on state variables. The second row of Figure [4](#page-6-0) captures the fine-grained detail of the face of the man, validating the ability of our model to generate high resolution videos.

### 4.4. World Model Based Video Generation

Given the current absence of benchmarks for evaluating world models in video generation, we assess the capabilities of our model through qualitative means. We provide the visualization results of generated long videos in Figure [5.](#page-7-0) We generate 3 scenes for a given prompt, and sample 2 frames from each scene. Every scene lasts for 8 seconds, and the whole video is 24 seconds long. When transitioning from one scene to another, we manually discard the image condition from the last frame and depend solely on the latent state variable as condition for geneartion, which is challenging because the latent state has to include information about the style and context of the previous video clips to generate the next clip in a consistent manner. We observe that Owl-1 is able to generate consistent long videos with reasonable dynamics anticipation. The video in the fourth row features a man engaged in gardening, where he utilizes tools to prune branches. The video we generated initially focuses on his hand movements and subsequently showcases the overall

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5. Video frames visualization results for world model based video generation. We generate 3 scenes for each prompt, and sample 2 frames from each scene. Every scene lasts for 8 seconds, and the whole video is 24 seconds long. Our Owl-1 generates consistent long videos with reasonable dynamics anticipation. Blue and red texts denote given prompt and predicted dynamics, respectively.

pruning effect, demonstrating a certain degree of logic. This reflects the modeling and prediction of the evolution of the world. However, we do notice that the predicted dynamics exhibit a certain degree of repetition, which we hypothesize is due to the inherent repetitiveness in the dense captions of the training video data. Even so, the videos generated by Owl-1 maintain good consistency across different scenes.

## 5. Conclusion

In this paper, we have proposed an Omni World Model (Owl-1) for consistent long video generation. Our Owl-1 approaches this task from the perspective of world model, which models the evolution of the world with a sequence of state variables. We have introduced a closed-loop stateobservation-dynamics triplet, in which the latent states encode both current and historical information about the world and serve as comprehensive long-horizon conditions for video generation. Explicit video observations are then decoded from latent state variables with a video diffusion model. To drive the world evolution, we incorporate the anticipation of the world dynamics during the generation process, which is beneficial for the diversity of generated content. Furthermore, we have devised an effective mutli-stage training scheme for our Owl-1 to take advantage of the vast amount of short video data and only finetune on a relatively small amount of long video data which reflects the evolution of the world. Owl-1 shows impressive capabilities in generating long and consistent videos. The visualizations further validate Owl-1's ability to capture fine-grained details and generate videos with reasonable dynamics anticipation.

Limitations and future work. From the evaluation and visualization results, we do notice some limitations of the current Owl-1, especially the decreased dynamic degree after fintuning the video diffusion model and repetitive world dynamics. Future work could investigate into these drawbacks and the scale up of our model on a large amount of high-quality video data with dense captions featuring the evolution process of the world. We believe the proposed paradigm for video generation world model is one of the approaches to realize multimodal general intelligence.

<span id="page-8-2"></span><span id="page-8-1"></span>![](_page_8_Picture_0.jpeg)

Figure 6. Gallery of various video samples of our Owl-1. We take one frame from each of these samples for demonstration.

## A. Video Samples

We provide more samples[1](#page-8-0) generated by our Owl-1 based on a wider range of prompts in Figure [6.](#page-8-1) In addition to the samples shown in the main paper which use the video frames from the validation or test sets of our training datasets as prompts, we also visualize some samples generated according to the standard prompts from the VBench-I2V benchmark [\[17\]](#page-10-4) with higher quality. As shown by Fig-

<span id="page-8-0"></span><sup>1</sup><https://github.com/huang-yh/Owl>

<span id="page-9-14"></span>ure [6,](#page-8-1) our model is able to generate videos covering a variety of topics, and the quality of generated videos generally improves given image prompts of higher quality.

## B. Additional Implementation Details

When finetuning Chameleon [\[31\]](#page-10-19) as the LMM, We employ LoRA [\[16\]](#page-10-20) and set the rank of LoRA to 8, resulting in approximately 798M trainable parameters. Together with all the parameters from DynamiCrafter [\[35\]](#page-10-8) as the video diffusion model, the total amount of trainable parameters is about 2B. We train the Owl-1 using 8 A800 GPUs with 80G memory, and the training time for the three stages is 1 day, 5 days, and 1 day, respectively.

## C. Controllability over Scene Transitions

Due to the scarcity of high quality video data with varying temporal content and devoid of scene transitions, we adopt the datasets of dense video captioning as the training data for the world model training stage. However, these datasets, e.g. Vript [\[36\]](#page-11-8) and ActivityNet Captions [\[6\]](#page-9-13), often incorporate scene transitions in a long video, which poses challenge for the training process. To address this issue, we manually discard the concatenating image conditions when generating the next clip belonging to a new scene during training. This strategy also endows our model with the capability to perform controllable scene transitions. Similar to the training phase, we only need to omit the concatenating image conditions to transit into a new scene. When generating longer videos in Figure [6](#page-8-1) and Fig. 4 in the main paper, we set the interval between scene transitions to about 2 short clips generated by the video diffusion model, resulting in the duration of each scene being about 4 seconds.

## References

- <span id="page-9-5"></span>[1] Max Bain, Arsha Nagrani, Gul Varol, and Andrew ¨ Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In *ICCV*, pages 1728–1738, 2021. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-4)
- <span id="page-9-0"></span>[2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf*, 2 (3):8, 2023. [1](#page-0-0)
- <span id="page-9-3"></span>[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. *arXiv preprint arXiv:2311.15127*, 2023. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-4)
- <span id="page-9-10"></span>[4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana

- Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. *arXiv preprint arXiv:2212.06817*, 2022. [3](#page-2-2)
- <span id="page-9-9"></span>[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv preprint arXiv:2307.15818*, 2023. [3](#page-2-2)
- <span id="page-9-13"></span>[6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In *CVPR*, pages 961–970, 2015. [6,](#page-5-1) [10](#page-9-14)
- <span id="page-9-4"></span>[7] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for highquality video diffusion models, 2024. [1,](#page-0-0) [5](#page-4-4)
- <span id="page-9-12"></span>[8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In *CVPR*, pages 13320–13331, 2024. [6](#page-5-1)
- <span id="page-9-7"></span>[9] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Shortto-long video diffusion model for generative transition and prediction. In *ICLR*, 2023. [2,](#page-1-0) [5](#page-4-4)
- <span id="page-9-11"></span>[10] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Fine-grained open domain image animation with motion guidance. *arXiv e-prints*, pages arXiv–2311, 2023. [5](#page-4-4)
- <span id="page-9-6"></span>[11] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In *ECCV*, pages 102–118. Springer, 2022. [2](#page-1-0)
- <span id="page-9-1"></span>[12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. *arXiv preprint arXiv:2307.04725*, 2023. [1](#page-0-0)
- <span id="page-9-8"></span>[13] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. *arXiv preprint arXiv:2403.14773*, 2024. [2,](#page-1-0) [3](#page-2-2)
- <span id="page-9-2"></span>[14] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P

- Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. *arXiv preprint arXiv:2210.02303*, 2022. [1](#page-0-0)
- <span id="page-10-3"></span>[15] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. *arXiv preprint arXiv:2205.15868*, 2022. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-20"></span>[16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [6,](#page-5-1) [10](#page-9-14)
- <span id="page-10-4"></span>[17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In *CVPR*, 2024. [1,](#page-0-0) [6,](#page-5-1) [9](#page-8-2)
- <span id="page-10-9"></span>[18] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. *arXiv preprint arXiv:2407.06358*, 2024. [1,](#page-0-0) [6](#page-5-1)
- <span id="page-10-15"></span>[19] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, ´ Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. *arXiv preprint arXiv:2312.14125*, 2023. [3](#page-2-2)
- <span id="page-10-10"></span>[20] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. [1,](#page-0-0) [6](#page-5-1)
- <span id="page-10-17"></span>[21] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. *arXiv preprint arXiv:2402.17177*, 2024. [3](#page-2-2)
- <span id="page-10-13"></span>[22] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. *arXiv preprint arXiv:2108.08877*, 2021. [2](#page-1-0)
- <span id="page-10-0"></span>[23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, ¨ and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023. [1](#page-0-0)
- <span id="page-10-12"></span>[24] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video gen-

- eration models as world simulators. *arXiv preprint arXiv:2410.18072*, 2024. [1](#page-0-0)
- <span id="page-10-14"></span>[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, pages 8748–8763. PMLR, 2021. [2](#page-1-0)
- <span id="page-10-7"></span>[26] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. *arXiv preprint arXiv:2402.04324*, 2024. [1,](#page-0-0) [5](#page-4-4)
- <span id="page-10-1"></span>[27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *CVPR*, pages 10684–10695, 2022. [1,](#page-0-0) [6](#page-5-1)
- <span id="page-10-2"></span>[28] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open largescale dataset for training next generation image-text models. *NIPS*, 35:25278–25294, 2022. [1](#page-0-0)
- <span id="page-10-5"></span>[29] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Textto-video generation without text-video data. *arXiv preprint arXiv:2209.14792*, 2022. [1](#page-0-0)
- <span id="page-10-16"></span>[30] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In *CVPR*, pages 3626–3636, 2022. [3](#page-2-2)
- <span id="page-10-19"></span>[31] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. *arXiv preprint arXiv:2405.09818*, 2024. [6,](#page-5-1) [10](#page-9-14)
- <span id="page-10-6"></span>[32] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In *ICLR*, 2022. [1,](#page-0-0) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-10-18"></span>[33] Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, and Jiwen Lu. Occsora: 4d occupancy generation models as world simulators for autonomous driving. *arXiv preprint arXiv:2405.20337*, 2024. [3](#page-2-2)
- <span id="page-10-11"></span>[34] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. *arXiv preprint arXiv:2309.15103*, 2023. [1,](#page-0-0) [6](#page-5-1)
- <span id="page-10-8"></span>[35] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang,

- Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In *ECCV*, pages 399–417. Springer, 2025. [1,](#page-0-0) [2,](#page-1-0) [5,](#page-4-4) [6,](#page-5-1) [10](#page-9-14)
- <span id="page-11-8"></span>[36] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words. *arXiv preprint arXiv:2406.06040*, 2024. [6,](#page-5-1) [10](#page-9-14)
- <span id="page-11-1"></span>[37] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024. [1,](#page-0-0) [6](#page-5-1)
- <span id="page-11-4"></span>[38] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwaxl: Diffusion over diffusion for extremely long video generation. *arXiv preprint arXiv:2303.12346*, 2023. [2](#page-1-0)
- <span id="page-11-7"></span>[39] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. *arXiv preprint arXiv:2311.04145*, 2023. [5](#page-4-4)
- <span id="page-11-3"></span>[40] Canyu Zhao, Mingyu Liu, Wen Wang, Jianlong Yuan, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. *arXiv preprint arXiv:2407.16655*, 2024. [1](#page-0-0)
- <span id="page-11-0"></span>[41] Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster textto-image generation via relay diffusion. *arXiv preprint arXiv:2403.05121*, 2024. [1](#page-0-0)
- <span id="page-11-6"></span>[42] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In *ECCV*, pages 55–72. Springer, 2025. [3](#page-2-2)
- <span id="page-11-2"></span>[43] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. [1,](#page-0-0) [2,](#page-1-0) [6](#page-5-1)
- <span id="page-11-5"></span>[44] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. *arXiv preprint arXiv:2405.03520*, 2024. [3](#page-2-2)