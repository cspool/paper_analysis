# <span id="page-0-1"></span>Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation

Shukang Yin<sup>1†</sup>, Chaoyou Fu<sup>2†‡\*</sup>, Sirui Zhao<sup>1‡†</sup>, Chunjiang Ge<sup>3</sup>, Yan Yang<sup>2</sup>, Yuhan Dai<sup>1</sup>, Yongdong Luo<sup>4</sup>, Tong Xu<sup>1</sup>, Caifeng Shan<sup>2</sup>, Enhong Chen<sup>1‡</sup>

<sup>1</sup>USTC, <sup>2</sup>NJU, <sup>3</sup>THU, <sup>4</sup>XMU

# **Abstract**

Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available here.

## 1. Introduction

The past few years have seen the rapid progress of Multi-modal Large Language Models (MLLMs) [2–9]. Apart from solving traditional vision tasks (such as VQA), these models also excel in following user instructions and generalizing

<span id="page-0-0"></span>![](_page_0_Figure_10.jpeg)

Figure 1. Performance comparison with different schemes of data scaling. We introduce Sparrow, a data-efficient training method for video-LLMs that achieves "Train less, gain more" in both general video and long video understanding performance. 0 sample indicates zero-shot inference with the image-LLM [1].

to new tasks. A mainstream paradigm for developing such models takes a two-stage training strategy. The first stage, pretraining, mainly serves to align vision modality with text and inject various kinds of visual knowledge into the model. In this stage, large-scale datasets of text-image pairs are often used, such as LAION [10] and CC [11], comprising a large proportion of the total compute and injecting abundant vision knowledge into models. Some methods also incorporate OCR and detection-related data to improve foundational capabilities [12, 13]. The second stage, instruction fine-tuning, adapts models to accommodate various tasks and helps generalize to new instructions. Training in this stage typically involves instruction data obtained from self-instruction or adaptation of task-specific datasets (e.g., VQA and chart understanding datasets). Recently, researchers have shifted their focus from single-image models to more advanced ones that support video understanding. Borrowing successful experience from developing image models, some video counterparts are typically trained from scratch, following a similar two-stage training paradigm [14, 15]. Apart from this path, some researchers utilize pre-trained image-LLMs instead. Typical approaches include zero-shot inference [16-18] and further fine-tuning [14, 15, 19, 20].

<sup>\*</sup>Project leader.

<sup>†</sup>Equal contribution.

<sup>‡</sup>Corresponding authors.

<span id="page-1-1"></span>Notably, the success of these models can be largely ascribed to the formidable scaling law, which puts emphasis on scaling up parameter size or data volume for better model performance. For the data aspect, the scaling has mainly been driven by automatic data engines, which synthesize massive amounts of data without human labor. Nevertheless, the characteristics of learning from these synthesized video data stand out as a critical yet underexplored topic. Thus, in this work, we investigate the learning characteristics of video-LLM more deeply from a data scaling perspective. Our preliminary data scaling experiments reveal a low data efficiency problem, that is, the performance gains from utilizing multiple times more data are marginal. An inspection of data characteristics suggests this might be due to a lack of instruction diversity in the training corpus. To address this issue, we propose a data augmentation method, dubbed Sparrow[1](#page-1-0) , to enrich the diversity of instruction. The basic idea is to synthesize video-like samples from textual data and mix these synthetic samples with the video samples. Specifically, we use existing text instruction data whose sample comprises a (long-context, instruction, answer) triplet. The long-context part is split into multiple segments and then further transformed into images, while the instruction and answer stay intact. Processed in this way, the synthetic samples have the same structure as video instruction data and can be incorporated seamlessly.

Comprehensive experiments demonstrate that our methods can facilitate data-efficient fine-tuning of image-LLMs for general video understanding and assist models in the comprehension of long videos. Specifically, using the same number of training samples, our method shows clear advantages over other data schemes. It even surpasses the baselines trained with many more samples, achieving high data efficiency (Fig. [1\)](#page-0-0). The contributions of this work include:

- We investigate the fine-tuning approach for developing video-LLMs from a data perspective, and shed light on possible factors that lead to low learning efficiency.
- We propose a data augmentation method that improves the instruction diversity of training data and facilitates a more efficient training scheme.
- We perform comprehensive experiments to evaluate the proposed method and examine its key properties, paving the way for future research in this line.

# 2. Related Work

# 2.1. Multimodal Large Language Models

Image-LLMs. To develop image-LLMs, the mainstream approach is to build upon powerful pre-trained LLMs and extend LLMs with the capability to perceive and reason with images [\[13,](#page-11-3) [21\]](#page-11-10). Based on a two-stage training recipe, *i.e*., image-text alignment training and instruction tuning, the developed model can fulfill a wide range of multimodal user queries and present its answers in user-friendly natural language sentences.

Video-LLMs. Following the success of image-LLMs, subsequent endeavors aim to expand the triumph to more intricate video understanding. Works like Video-ChatGPT [\[22\]](#page-11-11), VTimeLLM [\[23\]](#page-11-12), PLLaVA [\[24\]](#page-11-13), and LLaVA-NeXT-Video [\[25\]](#page-11-14) attempt to further fine-tune image-LLMs to enhance video understanding capability. Other research [\[14,](#page-11-4) [15,](#page-11-5) [19,](#page-11-8) [20\]](#page-11-9) explores training from pretrained LLM, following the basic alignment-then-finetuning paradigm similar to image-LLM. These approaches usually involve joint training that mixes image and video data in the training corpus. In this study, we build upon pre-trained image-LLMs and enhance video understanding capabilities through fine-tuning.

# 2.2. Long Video Understanding

A fundamental challenge in long video comprehension lies in the effective modeling of long video frame sequences. To tackle this problem, two major technical directions have been actively investigated, namely context window extension and efficient video modeling.

Context Window Extension. To accommodate a longer sequence of video frames with completeness, an intuitive approach is to extend the context window of the LLM backbone. Previous works have mainly adopted continued pretraining on the LLM before video training. For instance, LongVA [\[26\]](#page-11-15), LongVILA [\[27\]](#page-11-16), and Kangaroo [\[28\]](#page-11-17) perform continued training on long text corpora to accommodate more video tokens.

Efficient Video Modeling. With the observation that visual information is often redundant in spatial and temporal dimensions, and that key frames are sparse in videos, another line of work explores efficient modeling of long videos. These works try to compress the video token sequences to fit in the original context window of language models. Specifically, LLaMA-VID [\[29\]](#page-11-18) proposes an attention-based module to compress each video frame into two tokens. MovieChat [\[30\]](#page-11-19) designs a memory mechanism to cache video context and combines both short-term and long-term memory. TimeChat [\[31\]](#page-11-20) combines image Q-Former and video Q-Former to compress video tokens within single frames and sliding windows of video, respectively. Other works seek to retrieve frames most relevant to the user prompt and thus directly cut down the input length. For example, KeyVideoLLM [\[32\]](#page-11-21) utilizes a pretrained CLIP encoder to calculate similarities between the query and video frames, and selects the most relevant frames based on the similarities. Similarly, Video-RAG [\[33\]](#page-11-22) further incorporates external tools to extract auxiliary texts from the video and augments the input with more multimodal context, such as OCR, ASR, and object detection information.

<span id="page-1-0"></span><sup>1</sup> Inspiration taken from the swiftness of sparrows.

## <span id="page-2-0"></span>2.3. Evaluation of Video Understanding

Early methods [16, 22, 34] are generally evaluated on traditional benchmarks like MSVD-QA [35], TGIF-QA [36] and ActivityNet-QA [37]). These benchmarks are generally domain-specific and focus on certain basic skills, such as action recognition and repetition count, which lack comprehensiveness in both length coverage (especially in longer videos) and skill coverage. Moreover, the questions asked often involve shallow perception without deeper reasoning.

Recently, with the rise of benchmarks specifically designed for MLLMs [15, 38–40], a more in-depth and comprehensive evaluation has become more accessible. Compared to previous traditional benchmarks, these newly developed benchmarks are generally more challenging, often entailing composite skills and a finer-grained understanding of the video (*e.g.*, the plot in the movie or causal relationships between events), and can be much longer in duration (*e.g.*, up to 60 minutes in the Video-MME benchmark). In this work, our study adopts these newly developed video benchmarks.

## 2.4. Textual Data for Video Understanding

Since MLLMs are typically built upon LLMs and thus highly compatible with textual data, some works have explored utilizing pure text data to boost the performance of video understanding. Below, we outline the key ideas of these methods and their differences from our method.

 Textual data for context expanding. Previous works have explored utilizing textual data to expand the context window of base LLMs. Specifically, in order to facilitate long video understanding, LLaMA-VID [29] and LongVA [26] incorporate long text data in fine-tuning and continued pretraining stages, respectively, to expand the context window of LLM backbones.

<u>Differences</u>: In this work, we (1) adopt textual data as a data augmentation method and (2) use them in the vision form to accommodate the training format.

• Synthetic textual data for video understanding. This line of work investigates synthesizing textual data that simulates video QA data, aiming to transfer temporal reasoning capabilities from textual training. More specifically, TOPA [41] extracts textual captions and object-level information from video frames, while T3 [42] gathers similar information from multiple different images.

<u>Differences</u>: These two works seek to boost video understanding with synthetic data, while ours aims to enrich the instruction diversity of the training corpus. Moreover, our method does not require calling advanced LLM APIs to build data; instead, our method utilizes existing datasets.

#### 3. Problem Formulation

We focus on mainstream MLLMs architectures [2], which typically consist of a vision encoder, a projector, and an LLM backbone. Given a video  $\mathbf V$  downsampled to T frames as  $\mathbf F = \{f_i\}_{i=1}^T$ , frame-level features are extracted via  $\mathbf E = \{E_i\}_{i=1}^T = \mathtt{ViT}(\mathbf F)$ , where  $\mathtt{ViT}$  denotes the vision encoder (e.g., a pre-trained model such as CLIP [43]). Each  $\mathbf E_i \in \mathbb R^{(H \times W) \times C}$  represents the visual tokens of the i-th frame, with H, W being the spatial dimensions and C the feature dimension. These visual features are then projected into the text embedding space via a projector module, typically an MLP, yielding  $\hat{\mathbf E} = \mathtt{Proj}(\mathbf E)$ . The transformed vision features are subsequently concatenated with the text embeddings of a user query  $\mathbf Q$  to form a joint token sequence  $[\mathbf w_V; \mathbf w_T]$ . This combined sequence is fed into the LLM backbone, which generates a natural language response auto-regressively:

$$p(w_o|\mathbf{w}_V, \mathbf{w}_T) \sim \prod_{t=1}^{L} P(w_t|w_{< t}, \mathbf{w}_V, \mathbf{w}_T),$$
 (1)

where  $\mathbf{w}_o = \{w_{o,t}\}_{t=1}^L$  denotes the output token sequence of length L. Here,  $\mathbf{w}_V$  corresponds to the vision tokens processed through the encoder and projector, while  $\mathbf{w}_T$  refers to the tokenized representation of the user query.

In terms of training data composition, each instance is structured as a triplet  $(\mathbf{V}, \mathbf{Q}, \mathbf{A})$ , where  $\mathbf{Q}$  denotes a natural language instruction and  $\mathbf{A}$  is the corresponding textual answer. Let  $\mathcal{D}$  represent the data distribution over such triplets. The model is trained to generate the answer sequence  $\mathbf{A}$  conditioned on both the video input  $\mathbf{V}$  and the instruction  $\mathbf{Q}$ . Formally, we aim to minimize the expected negative log-likelihood loss over the model answer:

$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{V}, \mathbf{Q}, \mathbf{A}) \sim \mathcal{D}} \left[ -\log p_{\theta}(\mathbf{A} \mid \mathbf{V}, \mathbf{Q}) \right]. \tag{2}$$

where  $\theta$  denotes the model parameters and  $p_{\theta}(\cdot)$  is the conditional output distribution of the MLLM.

## 4. A Probing Study of Data Scaling

To understand the scaling characteristics of training data, our study starts with fine-tuning with different sample sizes and examines the relationship between training sample size and model performance.

In this section, we introduce the study's training and evaluation setup and then illustrate the empirical findings.

# 4.1. Training Setup

## 4.1.1. Model Setup

During our exploration, we mainly utilize two image-LLMs, including Mini-InternVL-Chat-4B-V1.5 [12] (termed as InternVL hereafter), MiniCPM-Llama3-8B-V2.5 [1] (termed as MiniCPM-8B hereafter). These instruction-tuned models are trained with massive image data and equipped with strong foundational capabilities. To support higher-resolution vision input, these models adopt the patchifying

<span id="page-3-1"></span>technique [44–46] with a dynamic resolution scheme, where an image can be cropped into multiple sub-images according to different aspect ratios. Specifically, InternVL supports up to 13 sub-images, each of which is converted into 256 visual tokens; MiniCPM-8B slices images into a maximum of 10 patches, where each is represented by 96 visual tokens. During training and evaluation, we switch off the patchifying option for higher efficiency.

### 4.1.2. Training Configurations

For fairness and ease of reproduction, we follow the official implementations. More specifically, we train the whole model end-to-end (except for InternVL-4B, where we freeze the vision encoder) with a learning rate of 5e-6.

<span id="page-3-0"></span>![](_page_3_Figure_3.jpeg)

Figure 2. Video length statistics of ShareGemini and Video-ChatGPT datasets. Both datasets mostly cover videos shorter than 3.5 minutes. We extract video frames at an FPS of 1 for each video.. For better visibility, we pick samples with frame numbers lower than 99.9 percentile for visualization.

## 4.1.3. Training Datasets

During our investigation, we utilize two representative types of datasets, *i.e.*, video-caption pairs and video instruction data. Specifically, we choose the ShareGemini [47] dataset and the Video-ChatGPT [22] dataset as caption and instruction data, respectively. For each video, frames are extracted at an FPS of 1. In consideration of efficiency, we use up to 64 frames for InternVL-4B and 24 frames for MiniCPM-8B. When the total number of frames exceeds the threshold, we uniformly downsample the video frames. The statistics of video lengths are shown in Fig. 2, and we provide more introduction to the two datasets below.

**ShareGemini-Webvid-core100k.** This dataset comprises 100K video-caption pairs in total. The videos in the dataset are sourced from WebVid [48], a web-scale video-caption dataset spanning diverse open-domain topics. In terms of temporal duration, the majority of videos are short-form, with lengths under 30 seconds.

The captions are annotated by calling the strong Gemini-1.5-Pro [49] API. To ensure the diversity of video content, an advanced clustering algorithm [50] is used to filter out highly similar videos. For simplicity, we refer to this dataset as ShareGemini in the following parts of the paper.

**Video-ChatGPT.** The video instruction dataset contains 100K video-instruction pairs. The videos in this collection are derived from ActivityNet [51]. The dataset's coverage of video duration is larger, yet the average video length is no more than 3.5 minutes. There are broadly three types of instructions: video summarization, questions about video content, and creative/generative tasks.

The dataset is annotated in a semi-automatic manner. A small portion of data samples are manually annotated by human annotators by refining and enriching the video captions. Other instruction data are generated by GPT-3.5 with the aid of off-the-shelf dense prediction and captioning models.

### 4.2. Evaluation Setup

To evaluate the model capabilities in an efficient and comprehensive way, we use Video-MME [38], MVBench [15], and TempCompass [39] as our benchmarks. We do not use traditional video-QA benchmarks (e.g., MSVD-QA [35], TGIF-QA [36], ActivityNet-QA [37]) since these benchmarks are generally limited to a small coverage of domains, task types, and video lengths. Moreover, the questions asked often involve shallow perception without deeper reasoning since early models generally lack reasoning capacity, whereas recent LLM-based models excel. We illustrate more about the benchmarks used as follows:

**Video-MME** is a comprehensive benchmark designed for the evaluation of video-LLMs. For temporal coverage, videos of short length (up to 2 minutes), medium length (4–15 minutes), and longer duration (30–60 minutes) are included. The videos and annotations are manually collected and filtered. We only use the raw frames without the subtitles to focus on the evaluation of video understanding capabilities.

**MVBench** designs a set of 20 video tasks that cover both perception and cognition, such as scene transition and episodic reasoning. Compared to Video-MME, the videos are sourced from existing benchmarks, and the QAs are automatically generated for the 20 pre-defined tasks.

**TempCompass** is designed to evaluate models' temporal understanding capabilities, encompassing temporal aspects such as action, speed, and attribute change. The videos and corresponding meta-information are manually collected, followed by LLM-generated annotations. We use the multiple-choice QA (MCQ) format to align with other benchmarks.

To ensure robust and efficient judging of model answers, we use a combination of exact matching and LLM matching for assessment. More details about the implementation of this evaluation scheme are available in Appendix A.

### 4.3. Main Findings

#### 4.3.1. Low Learning Efficiency Issue

Our experiments start with scaling up the training data volume and evaluating the video understanding performance on different general video understanding benchmarks. The

<span id="page-4-2"></span><span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3. Scaling performance of fine-tuning InternVL with different data volumes and types on different benchmarks. We consider fine-tuning with ShareGemini, Video-ChatGPT, and a mix of both with a 1 : 1 sampling ratio. Training samples are measured in K, where 0 sample indicates zero-shot inference.

<span id="page-4-1"></span>![](_page_4_Figure_2.jpeg)

Figure 4. t-SNE plot of instruction distributions for video datasets – ShareGemini and Video-ChatGPT. We sample 5,000 instructions from each dataset for visualization. The relatively limited instruction diversity observed in both datasets may hinder learning efficiency during fine-tuning.

results are shown in Fig. [3.](#page-4-0) In general, training either with video caption data (ShareGemini), instruction data (Video-ChatGPT), or a mix of both can boost the image-LLM's video understanding performance. Meanwhile, increasing the training volume brings additional gains in accordance with the data scaling law. However, the gains from scaling up quickly reach a plateau. For instance, on the Video-MME benchmark, when training with mixed data, 30K samples improve overall accuracy by 3.1 points, while 100K samples only add another 0.5 points, resembling a logarithmic growth. In view of this quick and early saturation, the learning efficiency with these video datasets can be quite limited. The phenomenon also suggests that there could be high redundancy in the training corpus, and it is possible that we may use less data to achieve a performance comparable to or even better than training with more data samples.

### 4.3.2. Probing of Instruction Diversity

Previous results prompt us to explore the reason for such low learning efficiency. Inspired by prior studies, which have underscored the importance of instruction diversity for finetuning LLMs [\[52\]](#page-12-12) and image-LLMs [\[53\]](#page-12-13), we conduct an inspection of training data in this aspect. Specifically, we follow previous approaches [\[54,](#page-12-14) [55\]](#page-12-15) to visualize the distribution of instructions in the training corpus. 5,000 instructions are sampled from ShareGemini and Video-ChatGPT, respectively. Then, the instructions are embedded and visualized using the t-SNE technique, as shown in Fig. [4.](#page-4-1) Overall, the instruction distribution of these two datasets is not diverse enough, which leads to a low data efficiency: The distribution of ShareGemini exhibits 9 clear clusters in the figure, indicating very similar instructions. This is because this dataset samples from a fixed pool of 9 templates as instructions, each of which is a variant of "Describe this video in detail". On the other hand, the distribution of Video-ChatGPT seems relatively more diverse, as it includes specific questions related to video content and details besides video summarization. Nevertheless, the instruction diversity is still low due to the nature of self-instruction and a few fixed task-specific prompting templates for data curation.

# 5. Methods

# 5.1. Design Concept

Since currently available video data can be limited in instruction diversity, and annotating high-quality video data is costly, we aim to expand the instruction diversity by incorporating new synthetic data. A rich source of instruction data lies in the text domain, and it can effectively complement the vision domain. Nevertheless, there is inherently a modality gap between the text and visual domains. To better utilize these data, we bridge the modality gap by synthesizing images with the text. Fig. [5](#page-5-0) illustrates the overall data synthesis pipeline and characteristics of our scheme. Our proposed scheme enjoys three benefits: (1) Mixing in text data can effectively enrich the instruction diversity (Fig. [6\)](#page-5-1), thus improving the learning efficiency for video fine-tuning; (2) Images synthesized from text can emulate the 1D temporal structure of video frames since text segments are generally

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 5. Conceptual illustration of our proposed scheme. (a) We illustrate the structures of text, synthetic, and video samples. Synthetic video-like samples are generated from textual data, with a structure mimicking real video samples. Specifically, for each (long context, instruction, answer) triplet, the long context part is split into segments, which are then transcribed into a sequence of text-rich images, simulating a video. (b) The domain differences between real and synthetic sources contribute to greater instruction diversity when used jointly for training. (c) The sequential nature of synthetic image sequences aligns well with the structure of real video data, thus enabling a unified training format.

correlated in the context, thus mitigating the gap between common video samples and synthetic ones; (3) Text data is easier to collect than video samples. Thus, utilizing synthetic data can be economical.

<span id="page-5-1"></span>![](_page_5_Figure_3.jpeg)

Figure 6. t-SNE plot of instruction distribution after applying our proposed method.

# 5.2. Implementation Details

Each text sample is a (long-context, question, answer) triplet. For example, the long context can be a section of a book or an academic paper, while the instruction and the answer are centered around the context, *e.g*., an inquiry to give a synopsis, or questions related to the paper. After the data transformation process, each sample is a video-like (images, question, answer) triplet, where long-context information is transformed into a series of images, and the question and answer stay unchanged.

The key to the data synthesis procedure is synthesizing images with pure text. Specifically, for each (long-context, question, answer) triplet, we divide the context information into multiple segments according to word counts (set to 115 empirically) using an open-sourced NLP toolkit[2](#page-5-2) . These text chunks are then transformed into a sequence of images separately. Specifically, each text chunk is embedded into a blank image with a white background. This is achieved using a bitmap font with the ImageFont module of the Pillow library[3](#page-5-3) . Each image is 448x448 pixels in size, and the font

<span id="page-5-2"></span><sup>2</sup><https://www.nltk.org/>

<span id="page-5-3"></span><sup>3</sup><https://pillow.readthedocs.io/en/stable/>

<span id="page-6-1"></span><span id="page-6-0"></span>

| Methods                   | Size | Frames | Short | Medium | Long | Overall |  |
|---------------------------|------|--------|-------|--------|------|---------|--|
| PROPRIETARY MODELS        |      |        |       |        |      |         |  |
| GPT-4V [56]               | N/A  | 10     | 70.5  | 55.8   | 53.5 | 59.9    |  |
| Claude 3.5 Sonnet [57]    | N/A  | 20     | 71.0  | 57.4   | 51.2 | 60.0    |  |
| GPT-4o [58]               | N/A  | 384    | 80.0  | 70.3   | 65.3 | 71.9    |  |
| Gemini 1.5 Pro [59]       | N/A  | 1fps   | 81.7  | 74.3   | 67.4 | 75.0    |  |
| OPEN-SOURCE MODELS        |      |        |       |        |      |         |  |
| VideoChat2 [15]           | 7B   | 16     | 48.3  | 37.0   | 33.2 | 39.5    |  |
| Video-LLaVA [19]          | 7B   | 8      | 45.3  | 38.0   | 36.2 | 39.9    |  |
| Chat-UniVi-v1.5 [20]      | 7B   | 64     | 45.7  | 40.3   | 35.8 | 40.6    |  |
| VideoLLaMA 2 [14]         | 7B   | 16     | 56.0  | 45.4   | 42.1 | 47.9    |  |
| VITA [60]                 | 8x7B | 32     | 65.9  | 52.9   | 48.6 | 55.8    |  |
| Kangaroo [28]             | 8B   | 64     | 66.1  | 55.3   | 46.6 | 56.0    |  |
| VITA-1.5 [61]             | 7B   | 16     | 67.0  | 54.2   | 47.1 | 56.1    |  |
| FT W/ INTERNVL [12]       |      |        |       |        |      |         |  |
| Zero-shot                 | 3.8B | 64     | 61.3  | 51.8   | 44.3 | 52.5    |  |
| 200K video data           | 3.8B | 64     | 66.7  | 54.2   | 48.1 | 56.3    |  |
| Sparrow (30K hybrid data) | 3.8B | 64     | 67.0  | 53.7   | 49.3 | 56.7    |  |

Table 1. Accuracy comparisons of different methods on the Video-MME benchmark. Performance is ranked in ascending order regarding overall performance (The models are grouped according to open-source or not). Our method uses only 15% of the total sample size compared to the full volume video datasets (200K) for fine-tuning and achieves comparable performance. 30K hybrid data comprise 20K data sampled from video datasets and 10K synthesized from our method. Bold digits indicate the best performance within each group.

is 20 pt large, black color, Arial Regular type. We use a bounding box to control the layout, leaving a margin of 20 pixels on each side, so each line of the text has roughly the same width. Following the transformation, the synthetic data aligns structurally with the video samples, enabling its direct and seamless incorporation into the video training sets.

# 6. Evaluation on Proposed Methods

This section includes experimental results and discussions of our proposed method, including (1) a comparison with mainstream methods, (2) an ablation study on data mixes, (3) an examination of key properties, including data scaling performance and gains in long video understanding, and (4) an in-depth analysis of model performance, including qualitative results as well as performance breakdown across various task types.

# 6.1. Comparison with Mainstream Methods

We compare the our method with some representative proprietary models, including GPT-4V [\[56\]](#page-12-16), Claude 3.5 Sonnet [\[57\]](#page-12-17), GPT-4o [\[58\]](#page-12-18), Gemini 1.5 Pro [\[59\]](#page-12-19), and opensource video-LLMs of similar LLM parameter size, including Video-LLaVA [\[19\]](#page-11-8), VideoChat2 [\[15\]](#page-11-5), Chat-UniViv1.5 [\[20\]](#page-11-9), VideoLLaMA 2 [\[14\]](#page-11-4), VITA [\[60\]](#page-12-20), VITA-1.5 [\[61\]](#page-12-21), and Kangaroo [\[28\]](#page-11-17). The results are summarized in Table [1.](#page-6-0)

The table results show that, through zero-shot inference, the image-LLM Intern-VL already outperforms a variety of video-LLMs with larger LLM parameter sizes. This might be due to the rich pre-trained knowledge embedded in the model parameters since the image-LLM has been trained with largescale and high-quality image-text data. The vision prior lays a strong foundation for further video fine-tuning, where models learn temporal and causal concepts from activities, events, *etc*. The model fine-tuned with full video datasets achieves an overall gain of 3.8 points on the image-LLM, closing the gap between open-source models and proprietary ones.

Notably, our methods use only 15% of the total sample size compared to the full volume (200K) for fine-tuning and achieve comparable performance. This result suggests the high data efficiency of our proposed scheme since mixing in synthetic samples mitigates the low instruction diversity issue illustrated in the earlier section.

# 6.2. Ablation on Different Data Compositions

In order to examine the impact of different data compositions and validate the effectiveness of the proposed method, we conduct an ablation study and construct the following settings with the same amount of total data samples:

- 30K video samples from ShareGemini.
- 30K video samples from Video-ChatGPT.

<span id="page-7-0"></span>

| Data Mix                                               | S    | M    | L    | Overall |
|--------------------------------------------------------|------|------|------|---------|
| 30K Share-Gemini                                       | 65.7 | 52.8 | 46.1 | 54.9    |
| 30K Video-ChatGPT                                      | 66.3 | 53.0 | 47.3 | 55.6    |
| 15K Share-Gemini<br>15K Video-ChatGPT                  | 66.2 | 53.3 | 47.4 | 55.7    |
| 10K Share-Gemini<br>10K Video-ChatGPT<br>10K synthetic | 67.0 | 53.7 | 49.3 | 56.7    |
| 10K Share-Gemini<br>10K Video-ChatGPT<br>10K pure text | 67.3 | 52.4 | 47.7 | 55.8    |
| Zero-shot                                              | 61.3 | 51.8 | 44.3 | 52.5    |
| 200K full data                                         | 66.7 | 54.2 | 48.1 | 56.3    |

Table 2. Results of different data compositions on the Video-MME benchmark. Our proposed scheme achieves an overall performance superior to other data mixes of the same amount (30K) and even more data (200K).

- 15K video samples from ShareGemini and 15K from Video-ChatGPT, respectively.
- Our proposed scheme: 10K samples each from ShareGemini and Video-ChatGPT, plus 10K samples synthesized from text data (5K from LongAlpaca and 5K from LongQLora, respectively).
- Same video samples as above (20K in total), plus 10K samples of corresponding pure text data.

Examination of our design choices. As shown in Table [2,](#page-7-0) comparing the first three rows, we can find that when using the same amount of video samples, training only with ShareGemini is not as effective as using more diverse data compositions. Notably, under the same data budget, our proposed scheme (Row 4) achieves the best performance. Furthermore, compared to the full fine-tuning setting with 200K samples, our method achieves comparable performance using only 15% of the number of training samples. The training cost is also significantly reduced from 276.8 GPU hours to 33.6 GPU hours, yielding an 8.2× improvement in efficiency. These results underscore the importance of instruction diversity and validate the effectiveness of our proposed approach.

Notably, replacing the synthetic data with the original pure text counterpart achieves an overall inferior performance. We hypothesize that this is due to the inherent domain gap between vision and text. Thus, to simulate the structure of video frame sequences, transcribing long text into images is necessary.

Can synthetic data help models understand longer videos? Interestingly, in the training stage, we only utilize synthetic samples of long multimodal context instead of authentic long video samples. However, on the long video benchmark set, our proposed method still achieves a score that is 1.2 points higher than the full data training (as shown in the 4th row compared to 200K full data in Table [2\)](#page-7-0). This result suggests that fine-tuning with a *long multimodal context* can enhance the comprehension of longer videos. In the following section, we will present additional results and discussions to illustrate this point further.

# 6.3. Examination of Key Properties

In this section, we further examine the key properties of our proposed method, including its general effectiveness, scaling performance, and effectiveness when applied to long video understanding scenarios.

### 6.3.1. General Effectiveness and Scaling Performance

We further verify the proposed scheme's effectiveness by evaluating our methods on another image-LLM of larger parameter size, *i.e*., MiniCPM-8B, across different benchmarks. Through scaling up with different volumes and types of data, we compare our methods against a pure video data baseline, as well as other relevant methods, including TOPA and T3. Both TOPA and T3 first translate vision information into text, such as captions and relations between objects. Then, synthetic text QAs are constructed to simulate video reasoning samples, aiming to transfer temporal reasoning capabilities from text to video. Note that since the original data format of TOPA is largely different from the current paradigm, we design a template to adapt the samples to the instruction data format (More details are available in Appendix B). The results are summarized in Fig. [7.](#page-8-0)

General effectiveness. When using the same amount of training samples, our methods almost always outperform other methods by a clear margin in all the evaluated benchmarks. Specifically, when using 30K samples, our method achieves an overall accuracy of 52.7, surpassing the baseline by 3.9 points. Notably, it even outperforms the baseline trained with 100K samples by 1.7 points. Similarly, on the MVBench benchmark, with 100K samples, our method attains a 4.3-point absolute gain over the baseline. Overall, the superior performance on different benchmarks showcases the general effectiveness of our proposed method.

Scaling performance. A notable limitation of other methods is that they are more prone to performance saturation when scaling up the data budget. For instance, the baseline method uses 60K samples to improve the Video-MME benchmark by 3.1 points, while using 100K samples only achieves another 0.9 points of absolute gains. In contrast, our proposed scheme shows more stable and consistent improvements when scaling up the data volumes compared with other methods. This underscores the importance of maintaining instruction diversity in video-language training, as insufficient diversity can lead to reduced learning efficiency. In such cases, our data augmentation strategy facilitates a more diverse instructional distribution.

<span id="page-8-2"></span><span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 7. Scaling performance with different data volumes and types on general video benchmarks. Specifically, we scale up the volume of training data until the overall performance gain saturates (empirically set as 1 point). Four settings are considered, *i.e*., fine-tuning with video data only, our hybrid data (video data mixed with our synthetic samples with a ratio of 2 : 1), TOPA data, and T3 data. By default, video data (the caption dataset and the instruction dataset) are mixed with a 1 : 1 sampling ratio. Training samples are measured in K, where 0 sample indicates zero-shot inference. Purple-shaded area indicates gains of our methods over the video data baseline.

<span id="page-8-1"></span>

| Methods    | Samples (K) | Frames | Video-MMEL | LongVideoBench | MLVU        | Overall    |
|------------|-------------|--------|------------|----------------|-------------|------------|
|            | 0           | 24     | 40.1       | 40.0           | 44.5        | 41.6       |
| BASELINE   | 30          | 24     | 44.7       | 39.7           | 45.4        | 43.3       |
|            | 30          | 48     | 45.3       | 39.6           | 45.3        | 43.4       |
|            | 60          | 24     | 46.2       | 42.7           | 46.2        | 45.1       |
|            | 60          | 48     | 46.2       | 42.1           | 45.0        | 44.5       |
|            | 100         | 24     | 46.7       | 44.1           | 45.3        | 45.3       |
|            | 100         | 48     | 44.9       | 40.7           | 45.0        | 43.5       |
|            | 30          | 24     | 45.6(+0.9) | 48.7(+9.0)     | 51.4(+6.0)  | 48.5(+5.2) |
| OUR METHOD | 60          | 24     | 46.2       | 51.2(+8.5)     | 53.2(+7.0)  | 50.2(+5.1) |
|            | 100         | 24     | 48.7(+2.0) | 50.1(+6.0)     | 57.0(+11.7) | 51.9(+6.6) |

Table 3. Long video understanding performance with different data volumes and types. The BASELINE method adopts only video data, while OUR METHOD utilizes a mix of video data and synthetic samples attained from our method. In the BASELINE group, 0 sample indicates zero-shot inference. The performance gains are calculated relative to the video data fine-tuning baseline trained with the same number of samples.

Discussion: Can we scale up only with synthetic textual samples? An intriguing and highly relevant question is whether we can scale up the synthetic samples without using any real video samples. Since the text is more compact and less redundant than a whole video, training in this way is more economical. Unfortunately, the empirical results show that this is probably unfeasible. As shown in Fig. [7,](#page-8-0) scaling with synthetic text data (TOPA and T3) shows undesirable characteristics, *i.e*., this approach can easily reach the saturation point or even slightly downgrade. Other critical issues include the modality gap and special processing of videos in various domains (such as egocentric videos and movies). Besides, since text suffers inevitably from information loss when translated from videos, text data might be better used as a supplement to videos, which helps inject temporal reasoning language prior into the LLM backbones (similar to TOPA or T3) or mixing with video data as a regularization method (as our method does).

### 6.3.2. Long Video Understanding Performance

We adopt tailored benchmarks to evaluate long video understanding capabilities, including LongVideoBench [\[62\]](#page-12-22), MLVU-M [\[63\]](#page-12-23), and the long video set of Video-MME, and report the performance on evaluation sets for the former two benchmarks. Our study focuses on two aspects: (1) performance improvement in terms of long video understanding compared with the video fine-tuning baseline, and (2) frame number (multimodal context) generalization ability in the inference stage.

Performance change w.r.t. training configurations. As presented in Table [3,](#page-8-1) our method consistently outperforms the video-only fine-tuning baseline in long video understanding across varying sample sizes, despite the absence of any long video training data. Remarkably, with the same number of 100K training samples, our hybrid data strategy achieves a performance gain of 6.6 points over the baseline. We attribute this improvement to the transferable reasoning patterns embedded in long-form textual data, which may enhance the

<span id="page-9-1"></span><span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 8. Long video understanding performance evaluated with a different number of frames. "Video context" denotes the max number of video frames used in training, *i.e*., 24 frames; "LLM context" denotes the context window of the LLM backbone. Purpleshaded area indicates performance gains of our proposed method over the video data baseline.

model's temporal comprehension capabilities without requiring vision-derived textual supervision [\[41,](#page-12-1) [42\]](#page-12-2).

We also examine whether adopting a denser sampling scheme can improve long video understanding performance, as this approach can also expand the context window. Specifically, we double the sampling frame limit (*i.e*., from 24 to 48) for authentic videos and find no improvements in long video understanding performance. This is because the training dataset predominantly contains short videos, and the dynamics within each video are relatively small. In this case, sampling more frames does not introduce more information but instead can bring redundancy, and thus, achieve no improvements in long video understanding.

Performance change w.r.t. frame number. We also investigate whether our approach expands the context window. Larger context windows are usually beneficial since video-LLMs derived from large-context LLM backbones often accompany frame number generalization and benefit from inputting more video frames [\[26,](#page-11-15) [62\]](#page-12-22). However, we do not observe this trend. As shown in Fig. [8,](#page-9-0) the model fine-tuned with long synthetic samples still follows a similar pattern, that is, when performing inference beyond the video context of the training stage, the performance stays relatively stable and does not benefit from more frame input. And when the frame number exceeds the LLM context, the performance plunges to a low level. Thus, promising directions for further improvements may include continued pre-training to expand the LLM context window.

# 6.4. Fine-grained Performance Analysis

In this section, we present a more in-depth analysis of model performance. By examining qualitative results and task-wise performance breakdowns, we provide intuitive insights into how our method influences model performance.

### 6.4.1. Qualitative Results

We present qualitative results on Video-MME in Fig. [9,](#page-10-3) with additional examples provided in Appendix C. As shown in the figure, among the three models compared, only our method successfully identifies the number on the back of the basketball player, demonstrating superior OCR capabilities in video-based understanding. Further analyses on this aspect are discussed in the following section.

## 6.4.2. Performance Breakdown

To understand how our method influences model capabilities, we break down the performance across different tasks in Video-MME [\[38\]](#page-11-27). Three broad categories of specific capabilities involved in video understanding are considered: Perception includes Counting Problems, OCR Problems, Object Recognition, Action Recognition, Attribute Perception, Spatial Perception, and Temporal Perception.

Cognition includes Temporal Reasoning, Spatial Reasoning, Action Reasoning, and Object Reasoning.

Information Synopsis. This category of questions emphasizes a model's ability to comprehend the overall narrative or theme of a video rather than fine-grained details.

As shown in Fig. [10,](#page-10-4) beyond overall gains, the incorporation of synthetic data yields notable improvements in OCR-related tasks. In particular, the model trained with augmented synthetic samples outperforms the one trained on 100K real video samples by 10.4% in OCR problems. This notable gain aligns with expectations, as synthetic data enables the model to more effectively recognize and interpret textual elements embedded in text-intensive frames, thereby enhancing its ability to extract relevant information. Interestingly, a significant improvement is also observed in temporal reasoning, with a 7.9% performance gain. This observation suggests that the ability to comprehend textual context may partially transfer to the modeling of temporal relationships across multiple video frames.

# 7. Conclusion

In this paper, we propose Sparrow, a data-efficient training scheme for video-LLMs, which enables training with fewer samples and achieving better video understanding performance. This method derives from our empirical findings that the low learning efficiency in data scaling may be ascribed to a limited instruction diversity in the training corpus. Thus, we design an economical data augmentation method that synthesizes video-like samples rich in instruction diversity. Comprehensive experiments demonstrate the general effectiveness and key properties of our proposed method. We hope this paper's findings can spark more explorations of efficient training and high-quality video training corpora.

<span id="page-10-3"></span>![](_page_10_Picture_0.jpeg)

![](_page_10_Picture_1.jpeg)

**Ours:** The video shows a player wearing the number 6 dunking.

![](_page_10_Picture_3.jpeg)

**Zero-shot:** The player with the number 5 on their jersey is the one who dunks in the video. This can be determined by observing the players on the court and noting the player's jersey number during the dunking action.

![](_page_10_Picture_5.jpeg)

**Video Data Baseline:** The video shows a dunk by the player with number 4.

Figure 9. Qualitative results of OCR-related capabilities. The video features the process of dunking by a basketball player. Three settings are compared, *i.e*., zero-shot (direct inference with image-LLM), video data baseline (trained with 100K video samples), and our Sparrow scheme (trained with 66K video samples and 34K synthetic samples).

<span id="page-10-4"></span>![](_page_10_Figure_8.jpeg)

Figure 10. Model performance across various task categories on Video-MME. The Perception and Cognition categories are highlighted in the outer circle. Performance gains exceeding 5 points over the video data baseline are marked in green.

# References

<span id="page-10-2"></span>[1] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone.

- *arXiv:2408.01800*, 2024. [1,](#page-0-1) [3,](#page-2-0) [14](#page-13-0)
- <span id="page-10-0"></span>[2] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. *National Science Review*, 2024. [1,](#page-0-1) [3](#page-2-0)
- [3] Haomiao Xiong, Yunzhi Zhuge, Jiawen Zhu, Lu Zhang, and Huchuan Lu. 3ur-llm: An end-to-end multimodal large language model for 3d scene understanding. *IEEE Trans. Multimedia*, 27:2899–2911, 2025.
- [4] Binglu Wang, Yao Tian, Shunzhou Wang, and Le Yang. Multimodal large models are effective action anticipators. *IEEE Trans. Multimedia*, 27:2949–2960, 2025.
- [5] Yunxin Li, Baotian Hu, Xinyu Chen, Lin Ma, Yong Xu, and Min Zhang. Lmeye: An interactive perception network for large language models. *IEEE Trans. Multimedia*, 26:10952– 10964, 2024.
- [6] Shidong Cao, Zhonghan Zhao, Shengyu Hao, Wenhao Chai, Jenq-Neng Hwang, Hongwei Wang, and Gaoang Wang. Efficient transfer from image-based large multimodal models to video tasks. *IEEE Trans. Multimedia*, 27:3045–3056, 2025.
- [7] Fukun Yin, Xin Chen, Chi Zhang, Biao Jiang, Zibo Zhao, Wen Liu, Gang Yu, and Tao Chen. Shapegpt: 3d shape generation with a unified multi-modal language model. *IEEE Trans. Multimedia*, 27:4107–4120, 2025.
- [8] Weijia Liu, Bo Miao, Jiuxin Cao, Xuelin Zhu, Jiawei Ge, Bo Liu, Mehwish Nasim, and Ajmal Mian. Context-enhanced video moment retrieval with large language models. *IEEE Trans. Multimedia*, pages 1–11, 2025.
- <span id="page-10-1"></span>[9] Guozhang Li, Xinpeng Ding, De Cheng, Jie Li, Nannan Wang, and Xinbo Gao. Etc: Temporal boundary expand then clarify for weakly supervised video grounding with multimodal large

- language model. *IEEE Trans. Multimedia*, 27:1772–1782, 2025. [1](#page-0-1)
- <span id="page-11-0"></span>[10] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In *NeurIPS*, 2022. [1](#page-0-1)
- <span id="page-11-1"></span>[11] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *ACL*, 2018. [1](#page-0-1)
- <span id="page-11-2"></span>[12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *Science China Information Sciences*, 2024. [1,](#page-0-1) [3,](#page-2-0) [7](#page-6-1)
- <span id="page-11-3"></span>[13] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv:2308.12966*, 2023. [1,](#page-0-1) [2](#page-1-1)
- <span id="page-11-4"></span>[14] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in videollms. *arXiv:2406.07476*, 2024. [1,](#page-0-1) [2,](#page-1-1) [7](#page-6-1)
- <span id="page-11-5"></span>[15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In *CVPR*, 2024. [1,](#page-0-1) [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-1) [7](#page-6-1)
- <span id="page-11-6"></span>[16] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm. *IEEE Access*, 2024. [1,](#page-0-1) [3](#page-2-0)
- [17] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. *arXiv:2407.15841*, 2024.
- <span id="page-11-7"></span>[18] Kai Han, Jianyuan Guo, Yehui Tang, Wei He, Enhua Wu, and Yunhe Wang. Free video-llm: Prompt-guided visual perception for efficient training-free video llms. *arXiv:2410.10441*, 2024. [1](#page-0-1)
- <span id="page-11-8"></span>[19] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In *EMNLP*, 2024. [1,](#page-0-1) [2,](#page-1-1) [7](#page-6-1)
- <span id="page-11-9"></span>[20] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In *CVPR*, 2024. [1,](#page-0-1) [2,](#page-1-1) [7](#page-6-1)
- <span id="page-11-10"></span>[21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *NeurIPS*, 2024. [2](#page-1-1)
- <span id="page-11-11"></span>[22] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In *ACL*, 2024. [2,](#page-1-1) [3,](#page-2-0) [4](#page-3-1)
- <span id="page-11-12"></span>[23] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In *CVPR*, 2024. [2](#page-1-1)

- <span id="page-11-13"></span>[24] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. *arXiv:2404.16994*, 2024. [2](#page-1-1)
- <span id="page-11-14"></span>[25] Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model. [https://llava-vl.github.io/blog/2024-04-](https://llava-vl.github.io/blog/2024-04-30-llava-next-video) [30-llava-next-video](https://llava-vl.github.io/blog/2024-04-30-llava-next-video). [2](#page-1-1)
- <span id="page-11-15"></span>[26] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. *arXiv:2406.16852*, 2024. [2,](#page-1-1) [3,](#page-2-0) [10](#page-9-1)
- <span id="page-11-16"></span>[27] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In *ICLR*, 2025. [2](#page-1-1)
- <span id="page-11-17"></span>[28] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. *arXiv:2408.15542*, 2024. [2,](#page-1-1) [7](#page-6-1)
- <span id="page-11-18"></span>[29] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In *ECCV*, 2024. [2,](#page-1-1) [3](#page-2-0)
- <span id="page-11-19"></span>[30] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In *CVPR*, 2024. [2](#page-1-1)
- <span id="page-11-20"></span>[31] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In *CVPR*, 2024. [2](#page-1-1)
- <span id="page-11-21"></span>[32] Hao Liang, Jiapeng Li, Tianyi Bai, Xijie Huang, Linzhuang Sun, Zhengren Wang, Conghui He, Bin Cui, Chong Chen, and Wentao Zhang. Keyvideollm: Towards large-scale video keyframe selection. *arXiv:2407.03104*, 2024. [2](#page-1-1)
- <span id="page-11-22"></span>[33] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-rag: Visually-aligned retrieval-augmented long video comprehension. *arXiv:2411.13093*, 2024. [2](#page-1-1)
- <span id="page-11-23"></span>[34] Tianming Liang, Linhui Li, Jian-Fang Hu, Xiangyang Yu, Wei-Shi Zheng, and Jianhuang Lai. Rethinking temporal context in video-qa: A comprehensive study of single-frame static bias. *IEEE Trans. Multimedia*, pages 1–15, 2025. [3](#page-2-0)
- <span id="page-11-24"></span>[35] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In *ACM MM*, 2017. [3,](#page-2-0) [4](#page-3-1)
- <span id="page-11-25"></span>[36] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In *CVPR*, 2017. [3,](#page-2-0) [4](#page-3-1)
- <span id="page-11-26"></span>[37] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In *AAAI*, 2019. [3,](#page-2-0) [4](#page-3-1)
- <span id="page-11-27"></span>[38] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui

- Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In *CVPR*, 2025. [3,](#page-2-0) [4,](#page-3-1) [10](#page-9-1)
- <span id="page-12-11"></span>[39] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In *ACL (Findings)*, 2024. [4](#page-3-1)
- <span id="page-12-0"></span>[40] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. *arXiv:2502.04326*, 2025. [3](#page-2-0)
- <span id="page-12-1"></span>[41] Wei Li, Hehe Fan, Yongkang Wong, Mohan S Kankanhalli, and Yi Yang. Topa: Extending large language models for video understanding via text-only pre-alignment. In *NeurIPS*, 2024. [3,](#page-2-0) [10,](#page-9-1) [14](#page-13-0)
- <span id="page-12-2"></span>[42] Lei Li, Yuanxin Liu, Linli Yao, Peiyuan Zhang, Chenxin An, Lean Wang, Xu Sun, Lingpeng Kong, and Qi Liu. Temporal reasoning transfer from text to video. In *ICLR*, 2025. [3,](#page-2-0) [10](#page-9-1)
- <span id="page-12-3"></span>[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021. [3](#page-2-0)
- <span id="page-12-4"></span>[44] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In *EMNLP (Findings)*, 2023. [4](#page-3-1)
- [45] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In *CVPR*, 2024.
- <span id="page-12-5"></span>[46] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. *arXiv:2311.07575*, 2023. [4](#page-3-1)
- <span id="page-12-6"></span>[47] Share14. Sharegemini: Scaling up video caption data for multimodal large language models. [https://github.](https://github.com/Share14/ShareGemini) [com/Share14/ShareGemini](https://github.com/Share14/ShareGemini). [4](#page-3-1)
- <span id="page-12-7"></span>[48] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman. ¨ Frozen in time: A joint video and image encoder for end-toend retrieval. In *ICCV*, 2021. [4](#page-3-1)
- <span id="page-12-8"></span>[49] GeminiTeam. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv:2403.05530*, 2024. [4](#page-3-1)
- <span id="page-12-9"></span>[50] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In *ICLR*, 2023. [4](#page-3-1)
- <span id="page-12-10"></span>[51] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In *CVPR*, 2015. [4](#page-3-1)
- <span id="page-12-12"></span>[52] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. In *NeurIPS*, 2024. [5](#page-4-2)

- <span id="page-12-13"></span>[53] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, Tao Kong, and Ruihua Song. What matters in training a gpt4-style language model with multimodal inputs? In *NAACL*, 2024. [5](#page-4-2)
- <span id="page-12-14"></span>[54] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In *ICLR*, 2025. [5](#page-4-2)
- <span id="page-12-15"></span>[55] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatgpt interaction logs in the wild. In *ICLR*, 2024. [5](#page-4-2)
- <span id="page-12-16"></span>[56] OpenAI. Gpt-4v(ision) system card. [https://cdn.](https://cdn.openai.com/papers/GPTV_System_Card.pdf) [openai.com/papers/GPTV\\_System\\_Card.pdf](https://cdn.openai.com/papers/GPTV_System_Card.pdf), . [7](#page-6-1)
- <span id="page-12-17"></span>[57] Anthropic. Introducing claude 3.5 sonnet. [https://www.](https://www.anthropic.com/news/claude-3-5-sonnet) [anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet). [7](#page-6-1)
- <span id="page-12-18"></span>[58] OpenAI. Hello gpt-4o. [https://openai.com/index/](https://openai.com/index/hello-gpt-4o) [hello-gpt-4o](https://openai.com/index/hello-gpt-4o), . [7](#page-6-1)
- <span id="page-12-19"></span>[59] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. *arXiv:2312.11805*, 2023. [7](#page-6-1)
- <span id="page-12-20"></span>[60] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal llm. *arXiv:2408.05211*, 2024. [7](#page-6-1)
- <span id="page-12-21"></span>[61] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. *arXiv:2501.01957*, 2025. [7](#page-6-1)
- <span id="page-12-22"></span>[62] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. In *NeurIPS*, 2024. [9,](#page-8-2) [10](#page-9-1)
- <span id="page-12-23"></span>[63] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multitask long video understanding. *arXiv:2406.04264*, 2024. [9](#page-8-2)
- <span id="page-12-24"></span>[64] Meta. Introducing llama 3.1: Our most capable models to date. <https://ai.meta.com/blog/meta-llama-3-1>. [14](#page-13-0)
- <span id="page-12-25"></span>[65] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In *ACM MM*, 2024. [14](#page-13-0)

# <span id="page-13-0"></span>A. Answer Judging

We notice that MiniCPM-8B [\[1\]](#page-10-2) often fails to follow instructions properly when we explicitly ask the model to "Answer with the option's letter from the given choices directly", making simple exact matching inaccurate. Specifically, the model often prepends or appends additional text other than the option letters, *e.g*. "Answer: B. Pink.", or gives additional explanations apart from the answer.

To cope with these issues, we adopt a combination of exact matching and LLM matching for assessment. Specifically, we strip the prefixes such as "Answer:" from the prediction and try to use regular expression matching to find the option letter. When the exact matching scheme fails, we use an LLM (Llama-3.1-8B-Instruct [\[64\]](#page-12-24)) to find an option closest to the model prediction. When the LLM matching fails, a placeholder outside of the available options (such as "Z") is returned to denote a wrong answer. Our judging prompt for the LLM is modified from VLMEvalKit [\[65\]](#page-12-25), as shown in Table [4.](#page-14-0)

# B. Reproduction Details of Baseline Methods

Due to an inconsistent formulation between the method TOPA [\[41\]](#page-12-1) and our proposed method, we adapt the implementation for a fair comparison. The original sample comprises a global caption for the whole video and frame-specific information. The frame-related information contains a framelevel caption and some descriptions of key objects in the frame. Thus, we design a prompt template to fit the original textual samples into the unified training format.

A real case of formatting the sample with the devised template is shown in Table [5.](#page-15-0)

# C. More Qualitative Results

In this section, we present more qualitative results, as illustrated in Fig. [11](#page-16-0) and Fig. [12.](#page-16-1)

## <span id="page-14-0"></span>System message

You are an AI assistant who will help me match an answer with several options of a single-choice question.

#### Prompt

You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer.

If the meaning of all options is significantly different from the answer, output Z. You should directly output a single uppercase character, such as A, B, C, D (if they are valid options), and Z, and nothing else. Here are two examples.

Example 1:

Question: What is the main object in the image?

Options: A. teddy bear.

B. rabbit.

C. cat.

D. dog.

Answer: a cute teddy bear

Output: A

Example 2:

Question: What is the main object in the image?

Options: A. teddy bear.

B. rabbit.

C. cat.

D. dog.

Answer: Spider Output: Z

Now here are the question, options, and the answer, you should match and give me the option letter:

Question: {Question} Options: {Options}

Answer: {Model Answer}

Output:

Table 4. Template for prompting LLM to perform option matching. {Question} is the specific question of a benchmark sample, and {Options} are corresponding choices of the question. {Model Answer} is the raw prediction of MLLMs.

#### <span id="page-15-0"></span>User:

You will be provided with some information about a video, including a global caption for the whole video, global captions for each video frame, and descriptions of key objects in each frame. Answer the questions using the information below.

#### Video information:

Caption: This video shows the carpentry process. At first, the person sits on a workbench and measures the length of the plank. Then, he uses a saw to cut the wooden plank into multiple pieces. Then, he uses a hammer to nail two pieces of wood together. Finally, he takes a break and drinks water, and leaves the workshop.

## Frame information:

## Frame 1:

Caption: the person sits on a workbench and holds a hammer in his right hand.

the person: the person is sitting on a workbench in the center of the frame. He is looking at a wooden plank in front of him.

hammer: He has a hammer in his right hand.

workbench: He is sitting on a workbench in the center of the frame.

# Frame 2:

Caption: the person takes a chisel from a red toolbox and starts chiseling the plank.

chisel: There are a chisel and a hammer on the workbench in the center of the frame.

the person: the person is holding a chisel in the center of the frame.

red toolbox: There is a red toolbox in the right corner of the frame.

workbench: He is sitting on a workbench in the center of the frame.

#### Frame 3:

Caption: the person measures the length of the plank with a measuring tape.

the person: the person carries a pencil and a ruler in his right hand.

pencil: There is a pencil in the right hand of the person.

measuring tape: the person is holding a measuring tape in his right hand.

wooden plank: The plank is above the workbench.

...

#### Frame 8:

Caption: the person puts his hammer and chisel into the toolbox and leaves the workshop.

toolbox: the person is putting a hammer and a chisel into a toolbox in the center of the frame.

the person: the person is walking toward the door in the center of the frame.

hammer: There are a hammer and a chisel in the right hand of the person.

workshop: the person is leaving the workshop.

Question: Why does the person use a measuring tape in the third frame?

- A. To hammer the plank
- B. To cut the plank
- C. To measure the length of the plank
- D. To handle the saw
- E. To fix the length of the wooden plank

Answer with the option's letter from the given choices directly.

### Assistant:

C

Table 5. An example of using the template for structuring textual samples into the training format. For simplicity, we only show part of the frame-level information.

<span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

Figure 11. A qualitative example from Video-MME. The video depicts humanoid robots performing a sequence of actions. Our model demonstrates a clear understanding of the video content by accurately detecting both motion patterns and the presence of two distinct robots.

<span id="page-16-1"></span>![](_page_16_Figure_2.jpeg)

Figure 12. A qualitative example from Video-MME. The video presents a tutorial on watercolor painting. In the zero-shot setting, the model exhibits an incorrect reasoning process, ultimately producing a conclusion that contradicts its own reasoning chain. In contrast, only our model demonstrates strong temporal reasoning abilities, successfully arriving at the correct answer.