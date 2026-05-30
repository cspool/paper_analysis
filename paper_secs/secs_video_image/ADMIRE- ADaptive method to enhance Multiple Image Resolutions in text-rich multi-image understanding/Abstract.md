# Abstract

Recent advancements in multi-image understanding have been significant, yet methods that treat all images equally face challenges when handling dense visual OCR information and ultra-long sequences. This results in difficulty focusing on the relevant pages, leading to redundancy and limited effectiveness. To address these issues, we propose an effcient and effective method that enhances multi-image document understanding by dynamically adjusting image resolutions based on attention from large visual language models. This approach enhances both performance and efficiency without modifying the underlying model architecture. Firstly, we propose Text-Guided Key Image Selector to focus on important images. Secondly, we sort visual information by importance score, adaptively increasing the tokens for important images while compressing or dropping the tokens for less important or irrelevant visual inputs. Experimental results across four open datasets and one closed industry dataset, using two mainstream models, demonstrate the efficacy of our approach. We achieve state-of-the-art performance for OCR-free methods, with scores of 82.78 on MP-DocVQA and 56.05 on DUDE (a 7.42-point improvement), using the Qwen2vl-7B model. Despite a slight increase in inference time, our model still shows notable improvements of 4.78 points on MP-DocVQA and 3.28 points on DUDE. Our code will be publicly available for noncommercial use at https://github.com/Alipay-Med/admire.git

## CCS Concepts

• Computing methodologies → Natural language generation; Computer vision; Artificial intelligence.

## Keywords

Large Visual Language Model, Multi-model Representation Learning, Multi-image Understanding, Text-rich Image Understanding

#### ACM Reference Format:

Qipeng Zhu, Xiong Wang, Zhihong Lu, Jiangwei Lao, Congyun Jin, Jie Chen, Yingzhe Peng, Qi Zhu, Lianzhen Zhong, Jiajia Liu, Peng Wei, and Jian Wang. 2025. ADMIRE: ADaptive method to enhance Multiple Image Resolutions in text-rich multi-image understanding. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25), August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, [12](#page-11-0) pages. <https://doi.org/10.1145/3711896.3737187>

## 1 Introduction

The ability to understand high-resolution text-rich multi-image content is essential for a variety of activities, including the analysis of multi-page documents, the interpretation of news videos, and the evaluation of presentation slides. Recently, large visual language models (LVLMs) [\[5,](#page-9-0) [18,](#page-9-1) [24\]](#page-9-2) have demonstrated remarkable abilities in text-rich multi-image comprehension [\[7\]](#page-9-3). However, to achieve greater performance, LVLMs need to accurately recognize all texts and layouts, unlike natural images where each image's composition is relatively straightforward. Efficiently understanding multiple text-rich images without relying on OCR tools [\[19\]](#page-9-4) remains a challenge for LVLMs.

Recent researches [\[6,](#page-9-5) [9\]](#page-9-6) demonstrate inputing images with high resolution improves the performance of LVLMs in text-rich understanding. Despite they achieve certain advancements in text-rich

multi-image understanding, they merely enhance resolution of all images and feed them uniformly into LVLM [\[5,](#page-9-0) [15,](#page-9-7) [24\]](#page-9-2). In real-world scenarios, questions about text-rich multi-image tasks often target only a portion of the content. For example, as shown in Figure [1,](#page-0-0) to answer the question "What is the name of ITC's matches brand? Answer the question using a single word or phrase". Only one image in the provided set is needed to obtain the correct answer.

Existing methods, such as InternVL2, crop 8 high-resolution images into 16 low-resolution sub-images, then resize the original images and sub-images to a fixed 448×448 resolution, resulting in nearly 6.2k visual tokens. As shown in Figure [1,](#page-0-0) only 12.5% of these visual tokens are relevant for generating the correct answer. The remaining tokens are unrelated to the question, which unnecessarily increases computational effort.

Our approach is based on the observation that while upscaling all images can improve prediction accuracy, it may also lead to exceeding the model's token limit. We find that by upscaling only the images related to the answers (Evidence Candidates) and downscaling the others to maintain a balanced token count, we can achieve enhanced prediction accuracy. As shown in Table [1,](#page-1-0) directly doubling the resolution of images related to the answers (evidence candidates) and halving the resolution of irrelevant images leads to an efficient performance boost for Qwen2VL in the MP-DocVQA task. Recent studies [\[25,](#page-9-8) [30,](#page-9-9) [32\]](#page-9-10) have demonstrated that attention maps from pretrained language models capture rich information, making them valuable for identifying critical tokens. Motivated by this insight, we propose a text-guided image scorer that evaluates images based on the attention maps of pretrained LVLMs. By utilizing this scorer, the model can identify evidence candidates and dynamically adjust visual tokens, thus improving model performance while maintaining efficiency.

<span id="page-1-0"></span>Table 1: Comparison between upscaling resolution of all images and evidence images as two times. "Vanilla" refers to the model's direct output, "All" means that all images are upscaled, "Evidence" indicates that only the Evidence images are upscaled and the other images are downscaled, respectively.

|          | InternVL2-8B |       | Qwen2VL-7B |       |  |  |
|----------|--------------|-------|------------|-------|--|--|
|          | VTokens      | ANLS  | VTokens    | ANLS  |  |  |
| Vanilla  | 1509         | 51.53 | 1448       | 72.55 |  |  |
| All      | 2235         | 55.37 | 2738       | 81.59 |  |  |
| Evidence | 1619         | 56.23 | 1688       | 82.88 |  |  |

We propose ADMIRE, a novel method for LVLMs that ADaptively enhances the REsolution of Multiple Images. This approach leverages high-resolution enhancement and token sparsity while addressing their respective challenges. ADMIRE incorporates three modules for adaptive image processing: the Text-Guided Image Scorer (TIE), the Key Image Resolution Enhancer (KIE), and the Dynamic Visual Token Dropper (DVD). The TIE module utilizes attention weights from the first layer of the LVLM to automatically classify images into four categories based on their importance: very important, important, less important, and not important. This

enables the model to focus on crucial information and minimize the generation of redundant tokens. The KIE module upscales the resolution of *very important* images, ensuring that the model's input retains detailed information, thereby generating more accurate answers. Since KIE increases the number of visual tokens, the DVD module compresses these tokens at multiple levels. By dynamically dropping irrelevant visual tokens and compressing less important ones based on their importance scores, our method optimizes token usage efficiency. Finally, all visual tokens are reorganized, concatenated with text tokens and special tokens, and passed into large language models (LLMs). Our approach is plug-and-play compatible with any commonly used pretrained LVLM, offering both efficiency and effectiveness in text-rich multi-image understanding scenarios through its multi-level importance scoring system.

Our contributions are summarized as follows:

- We propose an adaptive resolution enhancement method for multi-image understanding that significantly boosts the performance of multimodal models without the need for additional training.
- We design a multi-level token compression technique that dynamically compresses irrelevant visual tokens or images, effectively balancing performance with computational overhead.
- We perform extensive experiments on four multi-image understanding benchmarks and two widely-used models to assess the efficiency and effectiveness of our approach, achieving state-of-the-art results for OCR-free methods, with scores of 82.78 on MP-DocVQA and 56.05 on DUDE, 69.29 on NewsVideoVQA and 60.54 on SlideVQA.
- Additionally, we applied this method to real-world multipage medical report QA scenarios and achieved a performance improvement of 3.99 points without additional training.

#### 2 Related Work

#### 2.1 OCR-free Visual Document Understanding

Visual document understanding [16, 20, 26] focuses on processing high-resolution, text-rich images, including charts, documents, slides, table images, and news videos. Recent studies have introduced Large Vision-Language Models (LVLMs) for visual understanding in an OCR-free manner. Some approaches crop highresolution images into low-resolution sub-images, resize them to a fixed resolution, and employ a uniform low-resolution visual encoder to process both sub-images and the global image. Notable examples include UReader [28], Monkey [17], LLaVA-Next [15], and InternVL2 [5]. In these cases, Large Language Models (LLMs) are used to understand the relationships between the sub-images. Other methods employ specialized techniques to process either high-resolution or low-resolution images, such as HIRI-VIT [27] and Qwen2VL [24]. However, these approaches tend to consume a large number of visual tokens when applied to multi-image tasks, leading to higher GPU memory usage and longer inference times. Our method strikes a balance between performance and overhead, adaptively applying resolution-enhancement techniques in multiimage scenarios.

### 2.2 Multi-image Understanding

With the advancement of LVLMs [1, 10, 14, 29, 31], some researchers introduce LVLMs to multi-image understanding tasks. They utilize large-scale interleaved image-text copora to train LVLMs. For example, InternVL2 [5] and Qwen2VL [24] demonstrate outstanding results in various multi-image tasks. ADMIRE effectively boosts the performance of pretrained LVLMs in multi-image scenarios without any training.

### 2.3 Visual Token Pruning

Visual token pruning [3, 11, 13, 23] enables Large Vision-Language Models (LVLMs) to process high-resolution images with limited resources by pruning less important or irrelevant visual tokens, thereby reducing the total number of visual tokens. Methods like MniMonkey [8] and SparseVLM [33] focus on pruning visual tokens from a single image, using attention scores computed by pre-trained LVLMs. In contrast, our approach introduces a multi-level visual token pruning strategy, designed specifically for text-rich multi-image understanding, which effectively balances performance with computational efficiency.

#### 3 Preliminaries

LVLMs process images using a visual encoder and decode them through a language model. Let  $\{\mathbf{I}_i \mid 0 \leq i \leq M\}$  represent a sequence of M images. The corresponding visual tokens are encoded as  $\{\mathbf{V}_i \mid 0 \leq i \leq M \land \mathbf{V}_i \in \mathbb{R}^{L_v \times D}\}$ , where  $L_v$  is the length of the visual sequence and D is the dimension of the visual tokens. The text input is tokenized and embedded as  $\mathbf{T} = \{\mathbf{t}_i \mid 0 \leq i \leq L_t\}$ , with  $L_t$  denoting the length of the text sequence. The visual tokens from all M images, along with the text tokens, are concatenated with special tokens to form a unified sequence  $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_L\}$ , where L is the total sequence length. The positions of the text and visual tokens are denoted as  $\mathbf{p}_t$  and  $\mathbf{p}_v^i$ , respectively, as shown in Equation (1).

<span id="page-2-0"></span>
$$\mathbf{p}_t = \{ j \mid x_j \in T \}, \quad \mathbf{p}_v^i = \{ j \mid x_j \in V_i \}.$$
 (1)

The organized token sequence X is fed into LLM to generate the corresponding output as described in Equation (2).

<span id="page-2-1"></span>
$$\hat{y} = LLM(X). \tag{2}$$

Each layer of LLM can be described by the following equations:

$$Z_l = Atten(Norm(H_{l-1})) + H_{l-1},$$
(3)

$$\mathbf{H}_{l} = \mathrm{MLP}(\mathrm{Norm}(\mathbf{Z}_{l})) + \mathbf{Z}_{l},\tag{4}$$

where Atten(·), Norm and MLP(·) represent the attention block, normalization block, and multi-layer perceptron, respectively.  $\mathbf{H}_l$  denotes the hidden states of the l-th layer of the LLM. For the base layer (l=0),  $\mathbf{H}_0=\mathbf{X}+\mathrm{PE}(\mathbf{X})$ , where  $\mathrm{PE}(\cdot)$  refers to the position encoding layer.

#### 4 Method

#### 4.1 Overview

In text-rich multi-image understanding scenarios, LVLMs enhance fine-grained image perception by upscaling the resolution of all images without prior knowledge, which significantly increases the number of visual tokens and inference latency. In real-world,

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 2: ADMIRE adaptively adjusts visual tokens to enhance the performance of pretrained large visual language models in text-rich, multi-image understanding scenarios. Given an input image sequence and a text query, it assesses the importance of each image. Based on this evaluation, the resolution of the images is adaptively adjusted and then concatenated into the large language model (LLM) to generate the corresponding response.

however, images containing critical information to help LVLMs generate accurate answers are often sparse, as shown in Figure 1. A more intuitive and efficient approach would involve identifying and upscaling only those key images that contain the essential information. Motivated by it, we propose a training-free, **AD**aptive method to enhance **Multiple Image RE**solutions, called **ADMIRE**.

ADMIRE comprises three modules: a training-free Text-guided Image scorEr (TIE), a Key Image resolution Enhancer (KIE), and a Dynamic Visual token Dropper (DVD). The overview of the ADMIRE framework is illustrated in Figure 2. First, an input image sequence  $\{I_i \mid 0 \le i \le M\}$  and a text query  $T = \{t_i \mid 0 \le i \le L_t\}$  are fed into TIE to obtain the importance of each image related to the question. We categorize each image into four groups: *very important*, *important*, *less important*, and *not important* based on the their scores. KIE is designed to upscale the resolution of *very important* images. The DVD then compresses those considered *less important*, and discards those considered *not important*.

The processing of images in different important groups can be modeled using Equation (5), where  $\hat{\mathbf{V}}_{i}$  denotes the visual tokens of images processed by ADMIRE. KIE(·) denotes upscaling the resolution of images,  $\mathrm{DVD}_{I}(\cdot)$  and  $\mathrm{DVD}_{V}(\cdot)$ , respectively, denotes drop partial visual tokens of *less important* images and *not important* images. The indexes  $\mathbf{p}_{\mathrm{kie}}$ ,  $\mathbf{p}_{\mathrm{Vdvd}}$ , and  $\mathbf{p}_{\mathrm{Idvd}}$  represent the *very important*, *less important*, and *not important* images, respectively.

<span id="page-3-1"></span>
$$\hat{\mathbf{V}}_{i} = \begin{cases}
KIE(\mathbf{V}_{i}), & i \in \mathbf{p}_{kie} \\
DVD_{V}(\mathbf{V}_{i}), & i \in \mathbf{p}_{Vdvd} \\
DVD_{I}(\mathbf{V}_{i}), & i \in \mathbf{p}_{Idvd}, \\
\mathbf{V}_{i}, & others
\end{cases}$$
(5)

We concatenate these visual tokens and the text tokens with special tokens and feed them into LLM to generate the corresponding response.

## 4.2 Text-Guided Image Scorer (TIE)

Designing an efficient image scorer is important for LVLMs to achieve adaptive adjustment of visual tokens, since ADMIRE requires understanding each image's importance to the given question. Recent studies [25, 32] have demonstrated that text and image data can be used to perform token-level compression on a single image via attention maps generated by pretrained language or visual language models. Inspired by this, we discovered that in multi-image scenarios, this approach is not only applicable for compression but also effective for token importance ranking. Therefore, we effectively score multiple images based on the attention weights of LVLMs by calculating their relevance to the given text query. Therefore, we design a training-free Text-guided Image scorEr (TIE) by reusing the attention weights from the first layer of pretrained LVLMs as described in Figure 3(a).

We consider the input tensor, to efficiently extract the attention score of each image first layer of LLM as X,  $W_Q$  and  $W_K$ , respectively, allowing the query and key matrices to be computed as shown in Equation (6):

<span id="page-3-2"></span>
$$Q = W_O(X + PE(X)), \quad K = W_K(X + PE(X)), \tag{6}$$

where  $PE(\cdot)$  is the position encoder. The attention weight **A** is then computed using the following function:

$$A = Attn(Q, K) = Softmax(\frac{QK^{T}}{\sqrt{D}}).$$
 (7)

We utilize text pooler,  $\operatorname{Pool}_t(\cdot)$  and visual pooler,  $\operatorname{Pool}_v(\cdot)$  to efficiently extract the attention score of each image as shown in Equation (8). Specifically, the attention map  $\mathbf{A}[\mathbf{p}_t]$  corresponding to the text tokens is fed into  $\operatorname{Pool}_t(\cdot)$  to get the text-guided attention score for each visual token, denoted as  $\hat{\mathbf{A}}$ . Then the visual pooler processes it to gain the attention score of each image. For efficient computation of attention scores across all images, we select either

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 3: Details of ADMIRE. ADMIRE is consist of three modules, including (a) Text-guided Image Scorer (TIE), (b) Key Image Resolution Enhancer (KIE) and (c) Dynamic Visual Dropper (DVD). ADMIRE utilizes TIE to score each image and adaptively adjust visual tokens through KIE and DVD.

mean pooling or max pooling for these two pooling functions.

<span id="page-4-1"></span>
$$\hat{\mathbf{A}} = \text{Pool}_t(\mathbf{A}[\mathbf{p}_t]), \quad \hat{\mathbf{A}}_i = \text{Pool}_v(\hat{\mathbf{A}}[\mathbf{p}_v^i]).$$
 (8)

In Equation (9), we normalize the attention score of each image to gain their importance score.

<span id="page-4-2"></span>
$$S = \text{Softmax}\left(\{\hat{\mathbf{A}}_i \mid 0 \le i \le M\}\right). \tag{9}$$

Text-guided Image Scorer utilizes the importance score S to rate each image within a sequence in multi-page document and news video. Guided by the scores, we classify images into different groups. We use  $\mathbf{p}_{\text{kie}}$ ,  $\mathbf{p}_{\text{comp}}$  and  $\mathbf{p}_{\text{drop}}$  to respectively record indices of *very important*, *less important* and *not important* images.

#### 4.3 Key Image Resolution Enhancer (KIE)

In high-resolution text-rich multi-image understanding, upscaling resolution of images relevant to answers, denoted as *very important* images, helps LVLMs to generate correct answers. Most methods [5, 15, 24] increase the resolution of all images to boost the perception of images largely increasing the number of visual tokens. However, lots of visual tokens are not necessary due to the sparsity of *very important* images as shown in Figure 1. Motivated by it, we design an adaptive **Key Image** resolution **Enhancer** (**KIE**), which is easy to apply in the main stream LVLMs to efficiently upscale the resolution of *very important* images.

Due to the sparsity of *very important* images, we set the number of such images as k as shown in Figure 3 (b). To prevent amplifying redundant information, k is set to 3 or 5. Specifically, we employ the Topk(·) method to select those *very important* images with a higher importance score. Their indices are recorded as the set  $\mathbf{p}_{kie}$ . For j in  $\mathbf{p}_{kie}$ , we upscale resolution of the image  $\mathbf{I}_j$  and feed it into a pretrained ViT to generate new visual tokens. These tokens are concatenated with the original visual tokens by being specifically

added before the original, ensuring that the order of the visual tokens remains unchanged.

## 4.4 Dynamic Visual Token Dropper (DVD)

Since very important images are sparse among all images, most of them provide limited help in generating correct answers for LVLMs and merely increase the number of visual tokens. Recent researches [2, 4] compress visual tokens of a single image through selection of visual tokens with high attention score. In multi-image understanding, eliminating irrelevant images to the answers can be more efficient in saving visual tokens compared to relying solely on token selection. By considering both methods, we can significantly reduce computational overhead. Therefore, we propose the Dynamic Visual token Dropper (DVD) to lower the overhead of LVLMs in a multi-level manner as depicted in Figure 3 (c). Different from KIE, we use soft threshold to control the number of not important images and less important images. We use p<sub>Idvd</sub> to record the indices of not important images, which are then dropped to reduce the total number of tokens. Additionally,  $p_{Vdvd}$  represents the indices of less important images, which are downscaled to retain only half of their visual tokens.

We define *less important* images as  $\{\mathbf{I}_j \mid j \notin \mathbf{p}_{kie}\}$ , with their importance scores given by  $\{s_j \mid j \notin \mathbf{p}_{kie} \land s_j \in S\}$ . To ensure that we do not exclude images containing crucial information, we apply varying thresholds to categorize these images. As described in Equation (10), we set the expected importance score,  $\alpha \gamma$ , as the initial threshold to distinguish between important and non-important images.

<span id="page-4-3"></span>
$$\gamma = \mathbb{E}\left(\left\{s_{i} \mid j \notin \mathbf{p}_{kie} \land s_{i} \in \mathbf{S}\right\}\right),\tag{10}$$

$$\mathbf{p}_{\mathrm{Idvd}} = \{k \mid s_j \le \alpha \gamma\},\tag{11}$$

where the hyperparameter  $\alpha$  is set to 0.5 to prevent excessive image filtering. To minimize the overhead of visual language models, we

<span id="page-5-2"></span>Table 2: Comparison KIE with State-of-the-Arts OCR-free Models. "/w. KIE-Topk-XN" using KIE to select the top k images and upscale maximum pixel count for *very important* images by a factor of N. "ANLS" is considered as the measures of performance. The bold font indicates the best performance.

| Model                         | MP-DocVOA | DUDE  | NewsVideoVOA | SlideVOA |
|-------------------------------|-----------|-------|--------------|----------|
| LayoutLMv3 [10]               | 55.1      | 20.3  | ~            |          |
| DocFormerv2 [1]               | 76.4      | 48.4  | _            | -        |
| GPT4(v)                       | -         | 53.9  | _            | _        |
| LongVA-7B [31]                | 60.80     | 38.37 | 50.61        | -        |
| Idefics3-8B [14]              | 67.15     | 38.65 | 60.16        | -        |
| LLaVA-next-interleave-7B [15] | 44.87     | 28.03 | 56.66        | -        |
| DocOwl2-8B [7]                | 69.42     | 46.77 | 64.09        | -        |
| InternVL2-8B [5]              | 51.53     | 37.37 | 65.02        | 54.92    |
| /w. KIE-Top3-X4               | 72.98     | 49.01 | 67.21        | 55.91    |
| /w. KIE-Top5-X4               | 74.59     | 50.12 | 67.13        | 56.91    |
| Qwen2VL-7B [24]               | 72.55     | 48.63 | 68.72        | 58.40    |
| /w. KIE-Top3-X4               | 81.64     | 55.19 | 69.18        | 59.84    |
| /w. KIE-Top5-X4               | 82.78     | 56.05 | 69.29        | 60.54    |

compress these *less important* images. As shown in Equation (12), when  $s_j$  falls in the range of  $\alpha \gamma$  to  $\beta \gamma$ ,  $I_j$  is considered *less important* and is subsequently compressed.

<span id="page-5-0"></span>
$$\mathbf{p}_{\text{Vdvd}} = \{ j \mid \alpha \gamma < s_j \le \beta \gamma \}, \tag{12}$$

where  $\beta$  is the hyper parameter, we set it as 1.5. And the left images are unchanged.

For *less important* images  $\{I_j \mid j \in p_{Vdvd}\}$ , we retain only half of their visual tokens. Considering the knowledge in the attention map, we use the attention score  $\{\hat{\mathbf{A}}_j \mid j \in p_{Vdvd}\}$  to select meaningful visual tokens. As shown in Equation (13), we preserve half of the visual tokens with higher attention scores.

<span id="page-5-1"></span>
$$\hat{\mathbf{V}}_j = \mathbf{V}_j[\operatorname{argsort}(\hat{\mathbf{A}}_j)[:L/2]]. \tag{13}$$

#### 5 Discussion of Efficiency

ADMIRE is a training-free method designed to enhance the performance in high-resolution text-rich multi-image understanding. Its straightforward approach allows for easy application to any visual language model. During inference, ADMIRE leverages the pretrained weights of the first layer of LLMs to assign adaptive importance scores to all images. It upscales the resolution of the very important images, compresses the less important ones, and discards irrelevant ones. The primary computational overhead arises from the scoring process, which has a complexity of  $O((ML_v)^2D)$ . Since only one layer of attention is utilized for scoring, this overhead remains manageable. When employing the Topk( $\cdot$ ) (k = 5) method to select very important images, the maximum number of visual tokens excluding compression and discarding, can be calculated as  $5nL_v + (M-5)L_v$ . In contrast, the resolution enhancement of all images results in a total of  $MnL_v$  visual tokens, which is significantly greater than what our method requires.

