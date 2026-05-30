# Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts

Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, Min Zhang

**Abstract**—Recent advancements in Multimodal Large Language Models (MLLMs) underscore the significance of scalable models and data to boost performance, yet this often incurs substantial computational costs. Although the Mixture of Experts (MoE) architecture has been employed to efficiently scale large language and image-text models, these efforts typically involve fewer experts and limited modalities. To address this, our work presents the pioneering attempt to develop a unified MLLM with the MoE architecture, named **Uni-MoE** that can handle a wide array of modalities. Specifically, it features modality-specific encoders with connectors for a unified multimodal representation. We also implement a sparse MoE architecture within the LLMs to enable efficient training and inference through modality-level data parallelism and expert-level model parallelism. To enhance the multi-expert collaboration and generalization, we present a progressive training strategy: 1) Cross-modality alignment using various connectors with different cross-modality data, 2) Training modality-specific experts with cross-modality instruction data to activate experts' preferences, and 3) Tuning the Uni-MoE framework utilizing Low-Rank Adaptation (LoRA) on mixed multimodal instruction data. We evaluate the instruction-tuned Uni-MoE on a comprehensive set of multimodal datasets. The extensive experimental results demonstrate Uni-MoE's principal advantage of significantly reducing performance bias in handling mixed multimodal datasets, alongside improved multi-expert collaboration and generalization. Our findings highlight the substantial potential of MoE frameworks in advancing MLLMs and the code is available at [https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs)

✦

**Index Terms**—Mixture of Experts, Multimodal Large Language Model, Unified Framework, Training Strategy, Benchmark.

# **1 INTRODUCTION**

R ECENT advancements in open-source Multimodal Large Language Models (MLLMs) [\[1\]](#page-14-0) such as InstructBLIP [\[2\]](#page-14-1) and LLaVA [\[3\]](#page-14-2) present notable successes in image-text understanding tasks [\[4\]](#page-14-3), [\[5\]](#page-14-4). Additionally, there is a growing trend [\[6\]](#page-14-5), [\[7\]](#page-14-6), [\[8\]](#page-14-7), [\[9\]](#page-14-8) toward building a unified MLLM that could comprehend more modalities such as video, audio, and speech, moving beyond the traditional imagetext paradigm. To catch up with superior closed-source MLLMs like GPT-4V [\[10\]](#page-14-9) and Gemini [\[11\]](#page-14-10), the main efforts of open-source community contain enlarging model sizes [\[12\]](#page-14-11), as seen with the expansion of vision foundation models to 6 billion parameters [\[12\]](#page-14-11) and the integration with 70B Large Language models (LLMs) [\[13\]](#page-14-12), [\[14\]](#page-14-13), and enhancing instruction tuning with diverse multimodal datasets [\[3\]](#page-14-2), [\[15\]](#page-14-14), [\[16\]](#page-14-15). These developments underscore the increasing ability of MLLMs to process and reason across multiple modalities, showing the importance of both model scalability and the expansion of multimodal instructional data. However, scaling up model size usually incurs huge computational overhead in both the training and inference phases.

To alleviate this issue, there has been a shift towards integrating the Mixture of Experts (MoE) [\[17\]](#page-14-16), [\[18\]](#page-14-17) architecture

![](_page_0_Picture_14.jpeg)

Fig. 1. **An Illustration of Uni-MoE**. Compared to previous dense MLLMs, it employs the MoE architecture to build a unified MLLM that can handle various modalities. We use the sparse routing control with the lightweight finetuning method LoRA to activate different experts, aiming to reduce computational costs.

<span id="page-0-0"></span>in large models to improve training and inference efficiency. Unlike conventional MLLMs or LLMs, where each input is processed by all model parameters, resulting in a dense computational approach, the MoE architecture only requires activating a subset of expert parameters for each input, as determined by an expert selector or router. Consequently, the MoE approach emerges as a promising strategy for

<sup>•</sup> *Yunxin Li, Shenyuan Jiang, Baotian Hu, Wanqi Zhong, and Min Zhang are with the Department of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China. (e-mail: liyunxin987@163.com, hubaotian@hit.edu.cn, and zhangmin2021@hit.edu.cn)*

<sup>•</sup> *Wenhan Luo is an Associate Professor at the Hong Kong University of Science and Technology, Hong Kong. (e-mail: whluo.china@gmail.com)*

<sup>•</sup> *Lin Ma is a Researcher with Meituan, Beijing, China. (e-mail: forest.linma@gmail.com)*

<sup>•</sup> *Baotian Hu and Longyue Wang are the corresponding authors. (e-mail: hubaotian@hit.edu.cn and vincentwang0229@gmail.com.*

<sup>•</sup> *Project Website:<https://uni-moe.github.io/>*

improving the efficiency of large models while reducing the need for extensive parameter activation. For instance, Jiang *et al.* [\[19\]](#page-14-18) introduces a sparse MoE-based language model named Mixtral-MoE 8x7B with each layer composed of 8 experts. It outperforms Llama2-70B in mathematics, code generation, and multilingual benchmarks while requiring less activated parameters. Lin *et al.* [\[20\]](#page-14-19) developed an MoEenhanced image-text MLLM, MoE-LLaVA, which utilizes about 3 billion activated parameters yet achieves comparable results to the dense 7B model on a variety of image-text understanding benchmarks.

While previous works have demonstrated the successful application of MoE in the construction of text-only and imagetext large models, developing the MoE architecture to construct powerful unified MLLMs remains largely uncharted, e.g., scaling MLLMs to incorporate more than four experts and extending their application to modalities beyond just images and text. In this work, we pioneer the exploration of scaling unified MLLMs with the MoE architecture and present an efficient MLLM named **Uni-MoE** that can leverage sparse MoE to adeptly manage and interpret multiple modalities. Specifically, as depicted in Figure [1,](#page-0-0) we first use modality-specific encoders to obtain the encoding of different modalities and map them into the language representation space of LLMs by the designed various connectors. They contain a trainable transformer model with subsequent linear projection layers to distil and project the output representations of frozen encoders, respectively. Then, we introduce a sparse MoE layer within the internal blocks of dense LLM. Hence, each MoE-based block features a shared self-attention layer applicable across all modalities, diverse experts based on feed-forward networks (FFN), and a sparse router for allocating token-level expertise. In this way, Uni-MoE can understand multiple modalities such as audio, speech, image, video and text, and only need activate partial parameters during inference.

Additionally, to enhance the multi-expert collaboration and generalization of Uni-MoE, we develop a three-stage progressive training approach: Firstly, we use extensive image/speech/audio-to-language pairs to train the corresponding connector, respectively, realizing the unified modality representation in the language space of LLMs. Secondly, we separately train modality-specific experts employing cross-modality datasets to refine each expert's proficiency within its respective domain. Thirdly, we integrate these trained experts into the MoE layer of the LLM and train the whole Uni-MoE framework with mixed multimodal instructional data. To further reduce the training cost, we employ the LoRA [\[21\]](#page-14-20) technique to fine-tune these pre-tuned experts and the self-attention layers. Through the above three-phase training approach, we gain an efficient and stable Uni-MoE that can adeptly manage and interpret multiple modalities.

To verify the effectiveness of Uni-MoE, we compare it with various dense MLLMs across extensive benchmarks such as image-text, video, and audio/speech understanding datasets. Additionally, we also introduce a new benchmark named the English High School Listening Test to assess the model's performance in complex long speech understanding scenarios. The experimental results suggest that leveraging the MoE architecture in constructing multimodal models

- not only outperforms traditional dense model setups in demanding multimodal contexts such as videos and long speech but also enhances stability and robustness across different modalities. In addition, we also discovered that the integration of more modality information into Uni-MoE enhances the performance of single-modality tasks. For instance, incorporating more image-text data improves the output of video QA tasks. Our contributions are as follows:
- 1) *Framework.* We present Uni-MoE (Sec [3.2](#page-3-0) and Table [1\)](#page-5-0), the pioneering sparse MoE-based unified MLLM that integrates multiple modalities including video, images, text, audio, and speech. It is constructed using modality-specific encoders, other modality-to-language connectors, and an LLM equipped with the sparse MoE architecture. During training and inference, we enhance the scalability of the MoE architecture training with an increased number of experts and a diverse set of multimodal data, through the expertlevel model parallel and modality-level data parallel[1](#page-1-0) .
- 2) *Training Strategy*. We introduce a progressive training paradigm (Sec [3.3](#page-4-0) and Algorithm [1\)](#page-4-1): an alignment stage for different modalities to language, training modality-specific experts, and undergoing unified MoE training with LoRA on mixed multimodal data. Our detailed experiments (Table [8\)](#page-12-0) reveal that pre-training experts on individual modalities significantly enhances the collaboration and generalization capabilities of multi-expert systems, surpassing the outcomes of standard MoE tuning where each expert possesses the same initial parameters.
- 3) *Practice*. Uni-MoE consistently outperforms dense MLLMs on almost all evaluation benchmarks, showing its advantages of exceptional ability in handling complex out-domain tasks. We investigate the role of widely-used auxiliary balancing loss [\[22\]](#page-14-21) in training the Sparse MoEbased MLLM with mixed modal data (Tables [8](#page-12-0) and [9\)](#page-12-1), finding that even without this auxiliary loss, Uni-MoE demonstrates superior multi-expert collaboration and generalization. Our results further reveal that as the number of experts and the routing search space expand, the benefits of auxiliary loss become pronounced.

# **2 RELATED WORK**

# **2.1 Unified Multimodal Models**

The advent of the Transformer model, introduced by Vaswani *et al.* [\[23\]](#page-14-22), marked a significant milestone in the field of deep learning, enabling the scalable integration of multiple modalities—including image, language, speech, audio, and video—into a unified representational space. This crossmodal interactive parallelism facilitates a range of generative and interpretive capabilities across different data types after training on extensive multimodal datasets. Recent advancements in generative multimodal large models have further expanded these capabilities [\[24\]](#page-14-23), [\[25\]](#page-15-0), allowing for the perception and generation of diverse modalities [\[6\]](#page-14-5), [\[26\]](#page-15-1), [\[27\]](#page-15-2), [\[28\]](#page-15-3). A notable example of this progress is ImageBind by Girdhar *et al.* [\[29\]](#page-15-4), which pioneers in learning a joint embedding across six distinct modalities, thus opening the door to emergent applications such as cross-modal retrieval,

<span id="page-1-0"></span>1. We have released the two distributed parallel training approaches.

![](_page_2_Figure_2.jpeg)

Fig. 2. **Overview of Uni-MoE Training Methodology**. The progressive training stages contain: 1) Utilize pairs from different modalities and languages to train connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

modality composition through arithmetic operations, crossmodal detection, and generation, directly "out-of-the-box." Similarly, Kirillov *et al.* [\[30\]](#page-15-5) introduce SAM, a foundational model for image segmentation capable of integrating insights from textual descriptions, visual cues, and semantic mappings, thereby demonstrating the versatility and robustness of multi-modal approaches in handling complex tasks. In a parallel vein, Zhang *et al.* [\[27\]](#page-15-2) put forward the Meta-Transformer model, characterized by a unified data Tokenizer, a shared encoder across modalities, and specialized heads for task-specific applications. This model stands out for its ability to conduct unified learning across an impressive array of 12 modalities using unpaired data, showcasing the potential for broad applicability across various domains and tasks [\[31\]](#page-15-6), [\[32\]](#page-15-7). Building on these foundational works, our study aims to construct an efficient unified MLLM leveraging the strengths of large language models and dedicated modality connectors.

# **2.2 Multimodal Instruction Tuning for LLMs**

The recent surge in Natural Language Processing (NLP) research owes much to the evolution of Large Language Models (LLMs) such as Flan-T5 [\[33\]](#page-15-8), GPT-4 [\[10\]](#page-14-9) and LLAMA [\[14\]](#page-14-13). These developments have not only enhanced our understanding and capabilities within the realm of language processing but have also paved the way for the advent of MLLMs. The progression towards these advanced models has been facilitated by two key factors: the broadening of training datasets [\[3\]](#page-14-2) and significant improvements in model architecture and design, as highlighted by recent works [\[8\]](#page-14-7), [\[13\]](#page-14-12). A prominent area of research focuses on leveraging pre-trained LLMs to improve visual instruction tuning, a process that involves refining the models to better understand and execute visually grounded instructions. <span id="page-2-0"></span>This field has seen the emergence of notable models like LLaVA [\[34\]](#page-15-9), MiniGPT-4 [\[35\]](#page-15-10), InstructBLIP [\[2\]](#page-14-1), Qwen-VL [\[36\]](#page-15-11), and LMEye [\[15\]](#page-14-14). A common framework among these models involves integrating a pre-trained visual backbone [\[37\]](#page-15-12) for image and video interpretation, an LLM [\[38\]](#page-15-13) for comprehending textual instructions and generating appropriate responses, and a vision-language cross-modal connector. This connector plays a critical role in harmonizing the capabilities of the vision encoder with the language model, ensuring seamless interaction between the two modalities. Efficient training paradigms [\[39\]](#page-15-14) with data management [\[40\]](#page-15-15) also play an important role in constructing large models. It not only enhances the model's ability to process and understand multimodal inputs but also significantly improves its applicability across a wide range of tasks that require nuanced understanding and generation of responses based on both textual and visual cues.

#### **2.3 Large Models with Mixture of Experts**

Due to the huge overhead caused by training and deploying large models, researchers have been committed to exploring utilizing mixture of experts (MoE) [\[41\]](#page-15-16) architecture to improve the efficiency of large models. This is particularly evident in large models where MoE enables selective activation of different network components, optimizing computational resources and enhancing performance. Pioneering work in this space can be seen with models such as GShard [\[42\]](#page-15-17), which introduced the use of MoE for massively multilingual language models, demonstrating considerable improvements in translation tasks. Switch Transformers [\[22\]](#page-14-21) further refined this concept by scaling up the MoE approach, achieving unprecedented model sizes and training efficiency on visual understanding. Additionally, BASE Layers [\[43\]](#page-15-18) explored the

integration of MoE into bidirectional encoder representations from transformers, further highlighting the potential for MoE to enhance language understanding tasks. The application of MoE in generative models has also been explored, with notable examples such as GLaM [44], showcasing the capability of MoE to handle diverse and complex generative tasks. In this paper, we mainly explore constructing a unifying large multimodal model using the MoE architecture. This body of work collectively underscores the versatility and efficiency gains MoE brings to the realm of large-scale model architectures, pointing towards a future where model size and performance can be enhanced without proportional increases in computational demands.

### 3 Uni-MoE

#### 3.1 Overview

Motivated by the high training and inference costs brought by scaling multimodal large models towards GPT-4V and Gemini and the efficiency of the MoE structure, here, we explore the realization of an efficient and powerful unified MLLM, utilizing the MoE architecture. Figure 2 presents a schematic representation of the designed Uni-MoE, showcasing its comprehensive design that includes encoders for audio, speech, and visuals, along with respective modality connectors. These connectors serve to translate various modality inputs into a unified language space. We then integrate the MoE architecture within the core LLM blocks, which is crucial for boosting the efficiency of both training and inference processes owing to only activating partial parameters. This is achieved through the implementation of a sparse routing mechanism, which is shown in the bottom part of Figure 2. The whole training process of Uni-MoE is divided into three distinct phases: cross-modality alignment, training modality-specific experts, and tuning Uni-MoE using a diverse set of multimodal instruction datasets. In the ensuing subsections, we will delve into the intricate architecture and the progressive training methodology of Uni-MoE in detail.

#### <span id="page-3-0"></span>3.2 Architecture

Connectors. To facilitate the efficient transformation of diverse modal inputs into a linguistic format, our Uni-MoE model is built upon the pretrained visual-language framework LLaVA [45]. This base model integrates the CLIP [46] as the visual encoder, alongside a linear projection layer that converts image features into corresponding soft image tokens within the language domain of Vicuna-LLaMA [47]. For processing video content, we select eight representative frames from each video and transform them into video tokens by employing average pooling to aggregate their frame-based (image) representations. In the audio domain, we enhance feature extraction through the deployment of two distinct encoders: the Whisper encoder, derived from the Whisper-small speech recognition model [48], and the BEATs encoder [49], a sophisticated audio processing tool that generates bidirectional encoder representations from audio transformers. Following a strategy akin to the Qformer approach [50], we then distil fixed-length speech and audio feature vectors respectively and subsequently

map them into soft audio and speech tokens via a linear projection layer. The specific workflow is given in:

$$X_{Input} = [I, V, A, S, T], \tag{1}$$

$$I = MLP(CLIP-V(I)), (2)$$

$$V = \text{Mean}(\text{CLIP-V}([I_1, ..., I_8])),$$
 (3)

$$A = \operatorname{Audio-Qformer}(\operatorname{BEATs}(A)), \tag{4}$$

$$S = \text{Speech-Qformer}(\text{Whisper}(S)), \tag{5}$$

$$T = \text{Word-Embedding}(T),$$
 (6)

where [I,V,A,S,T] represents the image, video, audio, speech, and Text, respectively. "MLP" is the learnable visual-language projection layer. Audio/Speech-Qformer is a four-layer transformer block, where we feed learnable fixed-length query vectors into them and the corresponding audio/speech hidden states as the key and value in the cross-attention layer. By doing such a four-layer calculation, the final outputs can be regarded as the representations of audio and speech inputs. Take the Audio-QFormer as an example, the detailed calculation progress is presented as the following equation:

$$X_Q^A = (h_Q^1, ..., h_Q^{AM}),$$
 (7)

$$h_B^A = \text{BEATs}(A), \tag{8}$$

$$h_S^A = MSA(LN(X_O^A)) + X_O^A, \tag{9}$$

$$h_C^A = \text{MCA}(\text{LN}(h_S^A), h_B^A) + h_S^A, \tag{10}$$

$$h_1^A = \text{MLP}(h_C^A), \tag{11}$$

where  $h_B^A$  is the top output of pretrained audio encoder BEATs and  $h_S^A$  is the multi-head self-attention calculation for the fixed-length query vectors  $X_Q^A$ , where AM refers to the total number of initial vectors.  $h_C^A$  represents the output of the cross-attention module, which is used to distil the main content of input audio. After four layers of the same operation, we apply a learnable linear layer for projecting the last output into the representation space of LLM.

**Uni-MoE**. By the above connectors, we could obtain the encoding tokens of any modality. For any modality inputs, we concatenate the corresponding tokens into one sequence and feed it into the language model. We denote the image, video, text, audio, and speech embedding representations to  $I=(I_1,...,I_N)$ ,  $V=(V_1,...,V_N)$ ,  $T=(T_1,...,T_z)$ ,  $A=(A_1,...,A_k)$ , and  $S=(S_1,...,S_L)$ , respectively, where N, z, k, and L refer to the encoding sequence length of different modalities: visuals, text, audio, and speech. Take understanding a video as an example, the calculation process of the l block configured with MoE is as follows:

$$x_0 = [V_1, ..., V_N; T_1, ..., T_z; A_1, ..., A_k],$$
(12)

$$X_{l}^{s} = MSA(LN(X_{l-1})) + X_{l-1},$$
 (13)

$$X_l^M = \text{MoE}(\text{LN}(X_l^s)) + X_l^s, \tag{14}$$

$$x_l = LN(X_l^M), (15)$$

where "MSA" and "LN" refer to the multi-head self-attention and layer normalization.  $X_{l-1}$  shows the output of l-1 th block. For the MoE layer, we adopt the sparse router to select the corresponding Top-k experts at the token level. When we introduce a set of experts  $E = (e_1, e_2, ..., e_M)$ , the router is a linear function to predict the probability of each token being assigned to each expert. The outputs of selected experts will

### Algorithm 1 Uni-MoE Optimization Framework

```
Require: Set of modalities M_I, pretraining datasets \mathcal{PD}_M, instruction
    tuning datasets \mathcal{D}_M with a set of templates \Pi = \{I_{M_t} : M \in
     M_I, t \in \mathcal{T} for each task t \in \mathcal{T}
 1: Initialize text tokenizer h and corresponding embedding layer E
 2: Stage 1: Cross-Modality Alignment
 3: for each modality M in M_I do
       Initialize modality-specific pre-trained encoder Enc_M
        Initialize Q-Former module QF_A and QF_S for audio and speech
        inputs and modality-specific linear projection layer LP_M for each
 6:
        for each step in a number of iterations do
 7:
           Sample (x, y) from \mathcal{PD}_M
 8:
           x_M \leftarrow Connector(x) {Modality Projection}
 9:
           Prediction \leftarrow LLM(x_M) {Get LLM's prediction}
10:
           Loss \leftarrow \mathcal{L}_{CE}(\text{Prediction}, h(y)) \{\text{Calculate cross-entropy loss}\}
           \theta \leftarrow \theta - \alpha \nabla_{\theta} \text{Loss} {Update Linear and audio/speech Q-Former
           parameters}
12:
        end for
13: end for
    Stage 2: Training Modality-Specific Experts
15: for each modality M in M_I do
        Copy corresponding weights from Stage 1.
16:
17:
        for each step in a number of iterations do
           Sample (x, y) from \mathcal{D}_M
18:
19:
           x_M \leftarrow Connector(x) {Modality Projection}
20:
           Prediction \leftarrow LLM(x_M, E(h(i_M))) {Get LLM's prediction}
21:
           Loss \leftarrow \mathcal{L}_{CE}(\text{Prediction}, h(y)) \{\text{Calculate cross-entropy loss}\}
           \theta \leftarrow \theta - \alpha \nabla_{\theta} \text{Loss} \{ \text{Update Linear and MLP (in LLM) parameters} \}
23:
        end for
24: end for
25: Stage 3: Tuning Uni-MoE
26: for each step in a number of iterations do
        Sample (x, y) from \bigcup \mathcal{D}_M
        X_M' \leftarrow Connector(x)  {Modality Projection}
28:
29:
                          E(h(Context_M))||X_M'||E(h(i_M))||E(h(x)))
        x_{LLM}
        {Inputs Concatenation}
30:
        Prediction \leftarrow LLM(x_{LLM}) {Get LLM's prediction}
        Loss \leftarrow \mathcal{L}_{CE}(\text{Prediction}, h(y)) {Calculate cross-entropy loss}
        \theta \leftarrow \theta - \alpha \nabla_{\theta} \text{Loss} \{ Update \ Linear \ and \ LoRA \ parameters \}
32:
```

be added according to the gating weights. We formulate the whole calculation process as follows:

$$\mathcal{P}(X_l^s)_i = \frac{e^{f(X_l^s)_i}}{\sum_{j}^{M} e^{f(X_l^s)_j}},$$
(16)

$$MoE(X_l^s) = \sum_{i=1}^k \mathcal{P}(X_l^s)_i \cdot e(X_l^s)_i,$$
 (17)

where  $f(x) = \mathbf{W} \cdot x$  is the linear router to produce expert assignment probabilities and  $\mathbf{W} \in \mathbf{R}^{d \times z}$ . d is the last dimension of hidden states and M is the total number of experts.

To speed up the training process, we freeze all parameters of LLMs including added experts, applying Low-Rank Adaption [21] (LoRA) to activate each expert. We denote the tokens of the input for one sequence inputted to the first expert in the MoE layer by  $\mathbf{X}_{E_1}$ . The computed process for this expert could be given in

$$\mathbf{h}_{e_1} = e_1(\mathbf{X}_{E_1}),\tag{18}$$

$$\mathbf{h}_{e_1}^{LoRA} = \text{LoRA-}e_1(\mathbf{X}_{E_1}), \tag{19}$$

$$LoRA(W_0) := W_0X + \Delta WX = W_0x + BAX,$$
 (20)

$$\mathbf{h}_{e_1} = \mathbf{h}_{e_1} + \mathbf{h}_{e_1}^{LoRA} \tag{21}$$

(22)

![](_page_4_Figure_14.jpeg)

<span id="page-4-2"></span>Fig. 3. Comparisons of loss curves under various MoE settings. For MoE-TaskX (Z-Y), "X", "Z", and "Y" represent the specific training tasks presented in Table 2, and this model contains "Z" experts in the MoE layer and selects "Y" experts for each token. "AuxLoss" indicates the widely-used auxiliary balancing loss [42], which is designed to encourage giving all experts the same importance. Uni-MoE variant with MoE-Task3(4-2) (blue line) achieves more stable coverage compared to model variants with fewer experts or identical expert settings.

where B, A are learnable parameters added for each pretrained linear weight  $W_0$  of experts and the final output of this expert is  $\mathbf{h}_{e_1}$ . By adopting this method for each expert, the training process becomes efficient, as it does not require updating the overall parameters of experts.

#### <span id="page-4-0"></span>3.3 Training Strategy

To harness the distinct capabilities of various experts, enhance the efficacy of multi-expert collaboration, and bolster the overall framework's generalization, we introduce a progressive training strategy for the incremental development of Uni-MoE. Algorithm 1 outlines the comprehensive three-stage training protocol designed to actualize the integrated MoE-based MLLM architecture. In the subsequent statements, we will delineate the objectives and elaborate on the specifics of each training stage.

Stage 1: Cross-Modality Alignment. In the initial stage, we aim to establish connectivity between different modalities and linguistics. We achieve this by constructing connectors that translate various modal data into soft tokens within the language space. The primary objective here is minimizing generative entropy loss. As illustrated in the upper section of Figure 2, the LLM is optimized to generate descriptions for inputs across modalities and only the connectors are subject to training. This approach ensures seamless integration of all modalities within a unified language framework, facilitating mutual comprehension by the LLM.

**Stage 2: Training Modality-Specific Experts.** This stage concentrates on developing single-modality experts through dedicated training on specific cross-modal data. The goal is to refine each expert's proficiency within its respective domain, thereby enhancing the overall performance of the MoE system on diverse multimodal data. While maintaining generative entropy loss as the focal training metric, we tailor the FFN to align more closely with the characteristics of the targeted modality.

TABLE 1

<span id="page-5-0"></span>Detailed Architecture of Uni-MoE and Comparison with Visual-Language MoE-LLaVA. Portions of this table are reported by the MoE-LLaVA model [20]. "Width" represents the dimension of the hidden states. "FFN" denotes the dimension of the feed-forward network's intermediate layer. "FFN Factor" represents the quantity of linear layers in the FFN. "Activated" or "Total Param" refers to the activated or total number of parameters. "7B×4-Top2" denotes a dense foundation model with 7B parameters, which is designed to incorporate a total of four experts, with two of them being activated. "†" indicates all layers are equipped with the MoE layer.

| Name Experts Top-k     |   | MoE<br>Layers | Embedding | Width | Layers | FFN | FFN<br>Factor | Heads | Activated<br>Param | Total<br>Param |       |
|------------------------|---|---------------|-----------|-------|--------|-----|---------------|-------|--------------------|----------------|-------|
| Phi2-2.7B [51]         | - | -             | -         | 51200 | 2560   | 32  | 10240         | 2     | 32                 | 2.7B           | 2.7B  |
| MoE-LLaVA-2.7B×4-Top2  | 4 | 2             | 16        | 51200 | 2560   | 32  | 10240         | 2     | 32                 | 3.6B           | 5.3B  |
| MoE-LLaVA-2.7B×4-Top2† | 4 | 2             | 32        | 51200 | 2560   | 32  | 10240         | 2     | 32                 | 4.5B           | 7.8B  |
| OpenChat-7B [52]       | - | -             | -         | 32000 | 4096   | 32  | 14336         | 3     | 32                 | 6.7B           | 6.7B  |
| MoE-LLaVA-7B×4-Top2    | 4 | 2             | 16        | 32000 | 4096   | 32  | 14336         | 3     | 32                 | 9.6B           | 15.2B |
| MoE-LLaVA-7B×4-Top2†   | 4 | 2             | 32        | 32000 | 4096   | 32  | 14336         | 3     | 32                 | 12.4B          | 23.7B |
| Vicuna-7B [47]         | - | -             | -         | 32000 | 4096   | 32  | 11008         | 3     | 32                 | 6.7B           | 6.7B  |
| Uni-MoE-7B×4-Top2      | 4 | 2             | 16        | 32000 | 4096   | 32  | 11008         | 3     | 32                 | 8.9B           | 13.2B |
| Uni-MoE-7B×4-Top2†     | 4 | 2             | 32        | 32000 | 4096   | 32  | 11008         | 3     | 32                 | 11.1B          | 19.7B |
| Uni-MoE-7B×8-Top2      | 8 | 2             | 16        | 32000 | 4096   | 32  | 11008         | 3     | 32                 | 8.9B           | 21.9B |
| Uni-MoE-7B×8-Top2†     | 8 | 2             | 32        | 32000 | 4096   | 32  | 11008         | 3     | 32                 | 11.1B          | 37.0B |

Stage 3: Tuning Uni-MoE. The concluding stage involves integrating the expertly tuned weights from Stage 2 into the MoE layers. We then proceed to jointly fine-tune the MLLMs utilizing mixed multimodal instructional data. The progression of the training process, as reflected by the loss curves, is depicted in Figure 3. Comparative analysis between MoE configurations reveals that experts refined during Stage 2 achieve quicker convergence and display enhanced stability on mixed-modality datasets. Furthermore, in scenarios involving complex mixed-modal data, encompassing video, audio, images, and text, the model employing four experts demonstrates reduced loss variability and more consistent training performance than its two-expert counterpart. Uni-MoE shows better convergence compared to using auxiliary balancing loss, which caused the overall training loss to fluctuate and did not show obvious convergence.

### 4 EXPERIMENTS

### 4.1 Uni-MoE Settings

The Uni-MoE's architectural specifications are presented in Table 1, exhibited along with the recently proposed MoE-LLaVA. Our exploration is anchored on scaling the Uni-MoE model, which is based on the LLaMA-7B architecture. The design and optimization of Uni-MoE are guided by specialized training tasks listed in Table 2. These tasks are instrumental in refining the capabilities of the MLP layers, thereby leveraging their specialized knowledge for enhanced model performance. We undertake eight single-modality expert tasks to elucidate the differential impacts of various training methodologies. Our comprehensive training approach uses the MoE framework to execute six diverse tasks spanning multiple modalities, including video, audio, speech, image, and text. This multi-faceted training task evaluates the Uni-MoE's performance under varied MoE configurations, thus ensuring a robust and versatile modelling approach adaptable to diverse data types and applications.

#### 4.2 Datasets

**Training Datasets**. To endow our model with speech recognition capabilities, we incorporate the **Common Voice** dataset [53] at the cross-modality alignment stage of training.

This dataset comprises short speech clips, each less than 30 seconds in duration, with a cumulative total exceeding 1.7 million instances. Subsequently, we develop a tri-modal dataset derived from LLaVA-Instruct-150K [45], converting user queries into auditory format utilizing sophisticated Text-to-Speech (TTS) technologies from Microsoft Azure [54]. Additionally, the original LLaVA-Instruct-150K dataset is employed in various training tasks to facilitate comparative analyses. To enhance the model's proficiency in understanding extended speech sequences, we integrate and concatenate audio files from the LibriSpeech dataset [55] with short speeches of 30 seconds into longer sound files, each extending up to two minutes in duration. Additionally, we convert the **RACE** dataset [56], a comprehensive reading comprehension collection derived from English examinations in China, from its original textual format into long audio files. These transformed audio files are subsequently presented to the model, enabling it to interpret lengthy speech inputs and accurately determine appropriate responses to questions. For the task of audio captioning, the model is subjected to both crossmodality aligning and single-modality expert training processes utilizing an identical dataset collection. The WavCaps dataset [57], which constitutes a ChatGPT-assisted, weaklylabelled audio captioning collection segmented into four subsets (AudioSet SL subset, SoundBible, FreeSound, and BBC Sound Effects), is employed partially within our optimization framework. Additionally, the AudioCaps dataset [58], a comprehensive corpus comprising approximately 46,000 pairs of audio clips and their corresponding human-generated textual descriptions, is also selectively utilized during the training regimen. The Clotho and ClothoAQA datasets [59], [60] are utilized to bolster the model's proficiency in audiorelated question-answering tasks. Additionally, MELD [61], a dataset designated for audio emotion detection, is employed to augment the diversity of audio-relevant tasks within our framework. In the realm of video-related tasks, which serve as a pivotal component in enhancing models' visual comprehension, the Video-Instruct-Dataset from Video-ChatGPT [62], encompassing 100,000 video-text instructional pairs, is leveraged as the training corpus to advance our models' performance in scenarios involving video content.

Evaluation Datasets. Our models are assessed across various

TABLE 2

<span id="page-6-0"></span>All training tasks. "\*" refers to that the dataset we use is only a subset. "MC" represents the multi-choice setting. "I-A" means image-audio pairs, where the question is converted into the corresponding speech. "T-I" shows the original text-image pairs. "T-A" indicates the contextual paragraph of the RACE dataset is transferred into the long speech. "Pretraining task" represents the tasks included in the previous training stage. "Pure-Task" indicates the MoE-based model contains identical experts and is trained with specific training data.

| Training Tasks               | Data Types                                                                                            | Data Size | Epochs | Trainable Modules                                        | Pretraining tasks                                          |
|------------------------------|-------------------------------------------------------------------------------------------------------|-----------|--------|----------------------------------------------------------|------------------------------------------------------------|
| Audio-Language Pretraining   | WavCaps*,Audiocap*,<br>MELD, ClothoV1                                                                 | 194K      | 2      | Audio Q-former,<br>Audio projection layer                | -                                                          |
| Speech-Language Pretraining  | Common Voice<br>(Short Speech)                                                                        | 1.7M      | 2      | Speech Q-former,<br>Speech projection layer              | -                                                          |
| Single-Modality-Expert-Task1 | LLaVA-Instruct-150K(I-A)                                                                              | 150K      | 1      | LoRA,<br>speech & image projection layer                 | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task2 | LLaVA-Instruct-150K(T-I)                                                                              | 150K      | 1      | LoRA,<br>image projection layer                          | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task3 | LLaVA-Instruct-150K(I-A)                                                                              | 150K      | 1      | LoRA, Audio Q-former,<br>speech & image projection layer | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task4 | LLaVA-Instruct-150K(I-A),<br>RACE(T-A), LibriSpeech                                                   | 271K      | 1      | LoRA,<br>speech & image projection layer                 | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task5 | LLaVA-Instruct-150K(T-I),<br>RACE(T-A), LibriSpeech                                                   | 271K      | 1      | LoRA,<br>speech & image projection layer                 | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task6 | LLaVA-Instruct-150K(I-A),<br>LLaVA-Instruct-150K(T-I),<br>RACE(T-A), LibriSpeech                      | 421K      | 1      | LoRA,<br>speech & image projection layer                 | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task7 | RACE(T-A), LibriSpeech ,<br>RACE(T-A)-MC                                                              | 209K      | 1      | LoRA,<br>speech projection layer                         | Speech-pretrain-task                                       |
| Single-Modality-Expert-Task8 | WavCaps*,Audiocap*,MELD<br>ClothoAQA,ClothoV1                                                         | 203K      | 1      | LoRA,<br>audio projection layer                          | Audio-pretrain-task                                        |
| MoE-Task1                    | LLaVA-Instruct-150K(I-A),<br>LLaVA-Instruct-150K(T-I),<br>RACE(T-A), LibriSpeech<br>RACE(T-A)-MC      | 509K      | 3      | LoRA, Router,<br>speech & image projection layer         | LLaVA-v1.5-LoRA,<br>Single-Modality-Expert-Tasks 2 / 3 / 7 |
| MoE-Task1-short-speech       | LLaVA-Instruct-150K(I-A),<br>LLaVA-Instruct-150K(T-I)                                                 | 300K      | 3      | LoRA, Router,<br>speech & image projection layer         | LLaVA-v1.5-LoRA,<br>Single-Modality-Expert-Tasks 2 / 3 / 7 |
| MoE-Task2                    | Video-Instruct-Dataset,<br>LLaVA-Instruct-150K(T-I),<br>RACE(T-A), LibriSpeech<br>RACE(T-A)-MC        | 459K      | 2      | LoRA, Router,<br>speech & image projection layer         | llava-v1.5-LoRA,<br>Single-Modality-Expert-Tasks 2 / 3 / 7 |
| MoE-Task3                    | Video-Instruct-Dataset,<br>LLaVA-Instruct-150K(T-I),<br>WavCaps*,Audiocap*,MELD<br>ClothoAQA,ClothoV1 | 453K      | 2      | LoRA, Router,<br>audio & image projection layer          | LLaVA-v1.5-LoRA,<br>Single-Modality-Expert-Tasks 2 / 3 / 8 |
| MoE-Task4                    | Video-Instruct-Dataset,<br>LLaVA-v1.5-665K(T-I) [3],<br>RACE(T-A), LibriSpeech<br>RACE(T-A)-MC        | 874K      | 2      | LoRA, Router,<br>speech & image projection layer         | LLaVA-v1.5-LoRA,<br>Single-Modality-Expert-Tasks 2 / 3 / 7 |
| Pure-MoE-Task1               | Video-Instruct-Dataset,<br>LLaVA-Instruct-150K(T-I),<br>WavCaps*,Audiocap*,MELD<br>ClothoAQA,ClothoV1 | 453K      | 2      | LoRA, Router,<br>audio & image projection layer          | LLaVA-v1.5-LoRA                                            |
| Pure-MoE-Task2               | Video-Instruct-Dataset,<br>LLaVA-Instruct-150K(T-I),<br>WavCaps*,Audiocap*,MELD<br>ClothoAQA,ClothoV1 | 453K      | 2      | LoRA, Router,<br>audio & image projection layer          | -                                                          |

benchmarks, reflecting the diversity of specialized tasks they are designed to perform. To evaluate our models' proficiency in short speech recognition and image understanding, we employ modified versions of the A-OKVQA [63], OKVQA [64], and VQAv2 [65] benchmarks, utilizing speech synthesis technologies TTS to convert questions into human speeches. To design tasks with long speech, we leverage the image-text reasoning dataset MMBench [5] and utilize TTS to transform contextual hints into long audios, called MMBench-Audio, along with the speech version of the RACE evaluation set, named RACE-Audio, to assess our models rigorously. More precisely, the proficiency of our models is also evaluated by their performances on the English listening part of the Chinese College Entrance Examination, to check their practical real-world speech recognition capabilities. This compact dataset comprises 150 questions related to long audio segments with an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds. The format of these materials aligns with that of the RACE-Audio evaluation dataset. Additionally, the environmental

audio understanding ability is evaluated by utilizing the test sets of Clotho V1/2 and ClothoAQA. In the context of videorelated tasks, the performance of Uni-MoE is gauged using benchmarks from ActivityNet-QA [66] and MSVD-QA [67], which are instrumental in assessing video comprehension and interaction capabilities.

#### 4.3 Evaluation Metrics

For the visual question-answering benchmarks including A-OKVQA, OK-VQA, and VQA2, we adopt the Exact Match (EM) metric to evaluate if the model's predicted answers align perfectly with the ground truths. For multi-choice formats such as MMBench, RACE, and the English Listening Test, we apply strict accuracy metrics to assess the correctness of chosen responses. In the audio quality assessment for ClothoAQA, the EM accuracy metric is utilized to gauge the precision of responses. For assessing the relevance and quality of generated captions in Clotho, we employ the CIDEr metric. Additionally, for video-related benchmarks like ActivityNet and MSVD-QA, we implement a comprehensive

#### TABLE 3

<span id="page-7-0"></span>**Experimental results on zero-shot speech-image and long speech understanding benchmarks**. The questions of A-OKVQA, OK-VQA, and VQAv2 are converted into speech by the Text-to-Speech technical. MMBench-Audio (Input Type: Text-Speech-Image) and RACE-Audio are used for held-out testing of long speech understanding and reasoning, where "middle" and "high" refer to the Chinese English examinations designed for middle school and high school students. "EHSL" indicates our collected English High School Listening Test (Long Speech) in real world. Empty experimental results indicate that these model variants are not adequate for long-speech reasoning. *"Uni-MoE" in Tables 3-6 refers to Uni-MoE-7B×4-Top2† (speech) and Uni-MoE-7B×4-Top2 (audio)*.

|                                           |         |        |        |               |        | RACE-Audio |        | EHSL   |
|-------------------------------------------|---------|--------|--------|---------------|--------|------------|--------|--------|
| Method                                    | A-OKVQA | OK-VQA | VQAv2  | MMBench-Audio | middle | high       | Long   | Short  |
| Macaw-LLM [7]                             | 1.08%   | 1.06%  | 2.33%  | 1.86%         | 4.04%  | 3.00%      | 0.67%  | 2.00%  |
| X-InstructBLIP [6]                        | 0.91%   | 0.00%  | 26.47% | 11.23%        | 16.33% | 18.88%     | 0.67%  | 2.00%  |
| Single-Modality-Expert-Task1              | 18.38%  | 26.58% | 32.60% | -             | -      | -          | -      | -      |
| Single-Modality-Expert-Task3              | 19.26%  | 27.68% | 32.95% | -             | -      | -          | -      | -      |
| Single-Modality-Expert-Task4              | 9.14%   | 12.26% | 24.30% | 45.96%        | 29.46% | 26.67%     | 12.67% | 6.00%  |
| Single-Modality-Expert-Task5              | 4.56%   | 3.42%  | 5.90%  | 41.30%        | 30.78% | 24.90%     | 9.33%  | 12.00% |
| Single-Modality-Expert-Task6              | 11.40%  | 14.40% | 25.11% | 47.52%        | 32.59% | 29.02%     | 18.67% | 8.00%  |
| Single-Modality-Expert-Task7              | -       | -      | -      | 47.83%        | 46.80% | 48.74%     | 34.67% | 46.00% |
| Uni-MoE w/ MoE-Task1 2 epoch              | 14.80%  | 19.98% | 26.01% | 44.41%        | 47.08% | 47.08%     | 41.33% | 36.00% |
| Uni-MoE w/ MoE-Task1 3 epoch              | 13.58%  | 17.68% | 20.14% | 47.20%        | 50.77% | 48.94%     | 32.67% | 38.00% |
| Uni-MoE w/ MoE-Task1-short-speech 2 epoch | 19.31%  | 26.46% | 31.28% | 37.58%        | -      | -          | -      | -      |
| Uni-MoE w/ MoE-Task1-short-speech 3 epoch | 20.01%  | 29.04% | 30.86% | 34.78%        | -      | -          | -      | -      |
| Uni-MoE w/ MoE-Task2 2 epoch              | 7.76%   | 8.08%  | 12.68% | 49.38%        | 49.65% | 49.37%     | 42.00% | 48.00% |

accuracy verification procedure inspired by the methodology used in Video-ChatGPT [\[62\]](#page-15-37).

#### **4.4 Implementation Details**

We employ the AdamW [\[68\]](#page-15-43) optimizer in conjunction with a cosine learning rate scheduler to train our models in all stages. In the cross-modality alignment phase, we utilize 2 A100 GPUs to separately process 1.7 million short-speech data and 194K short audio captioning data with a global batch size of 32 and a base learning rate of 2e-5. Notably, during this stage, only the Qudio/Speech Q-former and projection layer are trained. Subsequently, we train our model to handle specific tasks using various combinations of data on the same devices, employing a global batch size of 16 and a base learning rate of 4e-5. During fine-tuning audio connectors with audio captioning data, we apply 4e-4 as the LoRA learning rate and 3e-5 as the learning rate of the audio projection layer. In our implementation, we set the LoRA rank to 64 and alpha to 16, and only apply it to tune the MLP in LLMs. Transitioning to the MoE training stage, we utilize a single, eight, or sixteen (two nodes) A800 GPU to train our models. We set the rank to 8 and alpha to 16 in this phase, maintaining a learning rate of 4e-5 for both LoRA parameters and projection layer parameters. Notably, we implement data parallelism and expert parallelism for mixed multi-modal data, as well as multi-nodes and multi-GPUs parallel training for larger models (with more than 8 experts).

# **4.5 Overall Performance**

We evaluate the performance of various model variants across a diverse set of benchmarks, encompassing five speech-related, three audio-understanding, and two challenging video-understanding tasks. We compare the results against two foundational unified multimodal models: **Macaw-LLM** [\[7\]](#page-14-6), which represents a pioneering effort in multi-modal language modeling, integrating image, video, audio, and text data; and **X-InstructBLIP** [\[6\]](#page-14-5), noted for its simplicity and effectiveness in handling multiple modalities simultaneously. Both models serve as strong baselines due to their innovative multimodal integration approaches.

<span id="page-7-1"></span>TABLE 4 **Experimental Results on audio understanding benchmarks (Audio-Text)**. These audio understanding tasks focus on environmental audio and we adopt the CIDEr metric, following X-InstructBLIP [\[6\]](#page-14-5).

| Method                       | ClothoAQA | ClothoV1 | ClothoV2 |
|------------------------------|-----------|----------|----------|
| MWAFM [69]                   | 22.2%     | -        | -        |
| Kim et al. [70]              | -         | -        | 19.2%    |
| X-InstructBLIP [6]           | 20.69%    | 17.8%    | 17.3%    |
| Macaw-LLM [7]                | 16.8%     | 0.2%     | 0.2%     |
| Single-Modality-Expert-Task8 | 32.3%     | 22.3%    | 21.6%    |
| Uni-MoE w/ MoE-Task3 1 epoch | 32.6%     | 25.0%    | 24.7%    |
| Uni-MoE w/ MoE-Task3 2 epoch | 31.1%     | 24.8%    | 25.1%    |

<span id="page-7-2"></span>TABLE 5 **Model performance on image-text understanding and reasoning benchmarks**. Uni-MoE still achieves better performance compared to models with the single expert configuration.

| Method                       | A-OKVQA | OK-VQA | VQAv2  | MMBench |
|------------------------------|---------|--------|--------|---------|
| Macaw-LLM [7]                | 1.90%   | 5.70%  | 20.73% | 3.84%   |
| X-InstructBLIP [6]           | 21.52%  | 30.61% | 37.77% | 8.96%   |
| Single-Modality-Expert-Task2 | 67.07%  | 62.91% | 75.18% | 71.26%  |
| Single-Modality-Expert-Task5 | 58.86%  | 56.01% | 67.35% | 65.80%  |
| Single-Modality-Expert-Task6 | 58.69%  | 57.77% | 68.74% | 67.53%  |
| Uni-MoE w/ MoE-Task1 2 epoch | 61.22%  | 57.63% | 68.42% | 68.15%  |
| Uni-MoE w/ MoE-Task1 3 epoch | 59.91%  | 57.07% | 68.05% | 67.28%  |
| Uni-MoE w/ MoE-Task2 1 epoch | 66.20%  | 63.02% | 74.41% | 71.17%  |
| Uni-MoE w/ MoE-Task2 2 epoch | 65.07%  | 62.10% | 73.87% | 70.50%  |
| Uni-MoE w/ MoE-Task3 1 epoch | 65.07%  | 61.38% | 73.66% | 69.52%  |
| Uni-MoE w/ MoE-Task3 2 epoch | 64.28%  | 61.96% | 73.87% | 69.82%  |

**Speech-Image and Speech-Text Understanding:** We adopt the evaluation of manually constructed datasets of three modalities: text, image, and speech. The speech recognition ability is shown and compared in Table [3.](#page-7-0) Firstly, we observe that previous baselines have inferior performances on speech understanding, which further affects their performances on image-speech reasoning. Additionally, introducing automatically generated audio-image datasets based on text-image instruction data improves the overall performance of short speech and image understanding, e.g., the model trained with Single-Modality-Expert-Task6 significantly outperforms that with Task5 (without introducing audio-image training data) in the audio-image setting of A-OKVQA, OK-VQ, and VQA2. The comparative model

performances trained with Single-Modality-Expert and MoE tasks show that *1) Introducing the MoE architecture can improve the generalization of MLLMs on unseen audio-image reasoning tasks such as MoE-Task2 vs. Single-Modality-Expert-Task5 on A-OKVQA, OK-VQA, VQAv2, and MMBench-Audio; 2) Uni-MoE achieves better performance on challenging long speech understanding and reasoning tasks compared to dense models, e.g., Uni-MoE significantly suppresses other model variants on English High School Listening Test; 3) When the instruction tuning data mixes long and short speech, the performance of Uni-MoE is more stable compared to single-expert model, e.g., Single-Modality-Expert-Task4 vs. MoE-Task1 in Table [3.](#page-7-0) As the training epochs increase, it focuses on improving the performance of challenging long-speech understanding tasks.*

**Audio-Text Understanding:** In Table [4,](#page-7-1) our model significantly surpasses the best performance of baselines by 8.4% on ClothoV1 and 10% on ClothoAQA, suggesting a remarkable capability of understanding audio-related questions. We reproduced the results of two multimodal baselines X-InstructBLIP and Macaw. We also observed that Uni-MoE could achieve improvement in the audio captioning and question-answering tasks compared to previous baselines and that with a single expert tuned with the same audiorelated data, which sufficiently reveals the advancement of the MoE leveraged for audio and text co-reasoning. Concretely, MoE-Task3 surpasses the best result from the fine-tuning stage of ClothoV1 and ClothoV2 by 2.7% and 3.5% respectively.

**Image-Text Understanding:** We present experimental results on image-text understanding in Table [5](#page-7-2) as a sanity check. The best result from our models outperforms all baselines and surpasses the fine-tuning task by an average of 4 points, indicating the enhancement of image and text comprehension from the MoE layer. However, it is also a notable phenomenon that the MoE-Task with all datasets involved shows less capability of image understanding than the one with image data only and the one without long speech data, i.e., MoE-Task1-short and Task2. We hypothesize that this minor decrement in performance is attributed to the mixture of three modality data boosting the understanding of speech with video as well as the image content. Yet, with the presence of long speech data, our models are confused about making the right choice of expert for audio, image, and text modality, which leads to a subtle decline in performance. Overall, *our models show a great ability to interact with other modalities while maintaining the ability of text-image understanding, which makes the method of mixture of experts superior in building multimodal models*.

**Video QA:** Table [6](#page-8-0) shows the performance of our models on zero-shot tri-modality video QA tasks. To speed up the training and inferring process, we use the simple average pooling representation of randomly extracted 8 frames from a video as the soft video tokens, which often leads to visual information loss. It may achieve lower performance compared to strong video-LLMs such as Video-ChatGPT which focuses more on fine-grained visual representations of a video and its temporal encoding. We also fed LLMs with the audio tokens obtained by the audio or speech encoders and connectors. Experimental results indicate that Uni-MoE trained with MoE-Task3 achieves the best performance on Activity-QA among strong video baselines, surpassing Video-ChatGPT

#### TABLE 6

<span id="page-8-0"></span>**Comparative performance on two zero-shot Video QA benchmarks**. The comparative baselines adopt the same video instruction tuning data, yet employ a more powerful video processing method, e.g., Video-ChatGPT contains more visual tokens by introducing two spatial and temporal tokens.

| Method                      | ActivityNet-QA |       | MSVD-QA  |       |  |
|-----------------------------|----------------|-------|----------|-------|--|
|                             | Accuracy       | score | Accuracy | score |  |
| FrozenBiLM [71]             | 24.7%          | -     | 32.2%    | -     |  |
| VideoChat [72]              | 26.5%          | 2.2   | 56.3%    | 2.8   |  |
| LLaMA-Adapter [38]          | 34.5%          | 2.7   | 54.9%    | 3.1   |  |
| Video-LLaMA [73]            | 12.4%          | 1.1   | 51.6%    | 2.5   |  |
| Video-ChatGPT [62]          | 35.2%          | 2.7   | 64.9%    | 3.3   |  |
| Uni-MoE w/ MoE-Task2 epoch1 | 42.7%          | 2.5   | 55.0%    | 3.2   |  |
| Uni-MoE w/ MoE-Task2 epoch2 | 42.2%          | 2.5   | 55.3%    | 3.2   |  |
| Uni-MoE w/ MoE-Task3 epoch1 | 42.8%          | 2.5   | 55.6%    | 3.4   |  |
| Uni-MoE w/ MoE-Task3 epoch2 | 42.7%          | 2.8   | 56.1%    | 3.4   |  |

by 7.5%. Moreover, it also achieves comparative performance on the MSVD-QA dataset. *These findings show that utilizing the MoE approach to integrate the cross-modality reasoning ability, wherein experts concentrate on distinct modalities like audio and image, may effectively improve the overall performance of video understanding.*

# **5 EXPERIMENTAL ANALYSIS**

### **5.1 Visualization Analysis**

We adopt the method of visualizing the routing distributions and token pathways from MoE-LLaVA to check the specific functions of different experts. For each cross-modality or tri-modality pair, we randomly select 200 samples from the corresponding datasets to draw Figures [4,](#page-9-0) [5,](#page-10-0) and [6.](#page-10-1) More visualization and analysis can be found in the Appendix.

**Routing Distributions:** In Figure [4,](#page-9-0) we present the expert loads (leftmost plot) and the modalities preferences of different experts (four subplots on the right) through MoE-Task3 when encountering all combinations of different modality data. When we fed Uni-MoE with audio-text pairs, experts 2 and 4 almost dominated the workload of almost all tokens, while experts 1 and 3 had barely any loads. A similar trend happens in the case of image-text pairs loading to the MoE layers, expert 2 plays a fairly important role and participates deeply in the processing of image information. It is not until the last layer that expert 3 works more predominantly than other experts, which is not an important case for the analysis. However, when we deliver the video data, the workload of all layers becomes seemingly balanced compared with other circumstances. We also observe distinct behaviours: 1) Expert 1 appears less engaged in token distribution compared to the others, indicating a potential inefficiency. 2) Conversely, Expert 2 demonstrates considerable control over token allocation in the initial layers, which shifts as the model deepens. 3) Expert 4 then begins to increasingly manage the tokens, reflecting a gradual assumption of responsibilities. 4) Expert 3 also contributes to token handling as the model progresses, illustrating collaborative task division among the experts. *These behaviours suggest that the experts in Uni-MoE have developed a specific pattern for dividing tasks, particularly notable between Experts 2 and 4. They align with the respective pre-trained specializations in image and audio processing. The provided figures*

![](_page_9_Figure_2.jpeg)

Fig. 4. **Distribution of expert loading with various cross-modality inputs.** The discontinuous lines represent the distribution of tokens among different experts or modalities. The first figure on the left illustrates the workload among experts, while the remaining four figures depict the preferences of experts towards different modalities. Five layers of figures from the top to down refer to text-image, text-audio, text-video, image-audio, and video-audio-text pair data respectively when being fed to the MoE layers of Uni-MoE trained with MoE-Task3. Each expert focuses on different modal information, which could be compared with Pure-MoE-Task1 in Figure [8.](#page-18-0)

*show that Uni-MoE efficiently utilizes expert gates, allocating tokens to the most suitable expert based on their specialized knowledge from fine-tuning tasks. This strategic distribution ensures optimal processing and underscores the model's effective learning and application of task division among experts.*

Moreover, in Figure [5,](#page-10-0) we illustrate how different modalities are distributed among the experts in the Uni-MoE model, revealing distinct preferences. Specifically, text tokens are primarily managed by Expert 4 in scenarios involving audio contexts, but shift towards Expert 2 when image contexts are present. In cases where both video and audio data are input, text tokens are more evenly distributed across experts in the latter layers, indicating their role in cross-modality reasoning. <span id="page-9-0"></span>For audio and visual tokens, there's a notable pattern of distribution: Expert 4 predominantly handles audio tokens in the initial and final layers, whereas Expert 2 takes over in the middle layers. This alternation highlights their specialized functions in processing different data types. Furthermore, the distribution shifts between text-image and text-audio-video scenarios reveal how image and video tokens are managed. Image tokens are mainly processed by Expert 2, reflecting its image-centric specialization, while video tokens are dispersed across multiple experts, underscoring the complex nature of video data that requires integrated processing from various modalities. In conclusion, *this differentiation in token handling among experts underscores the Uni-MoE model's capacity*

![](_page_10_Figure_2.jpeg)

Fig. 5. **Distribution of modalities across different experts**. The discontinuous lines represent the distribution of tokens. Five layers of sub-figures from the top to down refer to text-image, text-audio, text-video, image-audio, and video-audio-text pair data respectively when being fed to the MoE layers of Uni-MoE trained with MoE-Task3. Expert 1 refers to the original MLP layer from LLaVA-v1.5. Expert 2 indicates the MLP is optimized through LLaVA-Instruct-150k(T-I) data, i.e., Single-Modality-Expert-Task2. Expert 3 represents the MLP trained with the audioimage LLaVA-Instruct-150k(I-A) dataset, i.e., Single-Modality-Expert-Task3. Expert 4 focuses on audio understanding, i.e., Single-Modality-Expert-Task8. *Training modality-specific experts is useful for expert assignment learning, facilitating the specialization of experts.*

![](_page_10_Figure_4.jpeg)

<span id="page-10-1"></span><span id="page-10-0"></span>Fig. 6. **Visualization of activated pathways**. We highlight the top 10 activated pathways. Among them, the non-gray paths represent the top 2 paths, while the gray paths represent the remaining 8 paths. All crossmodality data and the Uni-MoE version are identical to Figures [4](#page-9-0) and [5.](#page-10-0) Notably, *the expert numbers are not identical to previous experts because we mainly consider the token-level activated path of experts from the first layer to the last one, adopting the PCA to reduce and sort the tokendimension features.*

#### TABLE 7

<span id="page-11-0"></span>Ablation study about Uni-MoE architectures. For subtables (a), (b), and (d), we employ the MoE-Task2 to train different Uni-MoE variants with varying expert configurations. In subtable (a), two models include four experts and we set different top-k values; In subtable (b), the top-k value is set to 2 for all model variants, which results in the model with two experts operating as a dense model with the same activated parameter size as that with four experts. The model with one expert is also dense. The identical expert in subtable (c) stems from LLaVA-v1.5-7b and it compares the performance impacts of increasing the number of identical pure experts from four to six. Subtable (d) presents the comparative results of Uni-MoE with various injected ways of MoE layers.

(a) The value of top-k (Uni-MoE).

| Top-k | A-OKVOA  | OK-VQA | ActivityN | et-QA | RACE-Audio |        |  |
|-------|----------|--------|-----------|-------|------------|--------|--|
| тор-к | A-OK VQA | OK-VQA | Accuracy  | score | middle     | high   |  |
| 1     | 66.46%   | 62.76% | 42.1%     | 2.5   | 45.13%     | 42.42% |  |
| 2     | 66.20%   | 63.02% | 42.8%     | 2.5   | 46.31%     | 43.71% |  |

(b) The number of experts (Uni-MoE).

| Experts | A-OKVOA | OK-VOA | ActivityN | et-QA | RACE-Audio |        |  |
|---------|---------|--------|-----------|-------|------------|--------|--|
| LAPCITO | A-ORVQA | OR-VQA | Accuracy  | score | middle     | high   |  |
| 1       | 65.68%  | 56.12% | 41.6%     | 2.8   | 42.59%     | 39.02% |  |
| 2       | 65.75%  | 62.04% | 42.9%     | 2.5   | 44.64%     | 43.63% |  |
| 4       | 66.20%  | 63.02% | 42.8%     | 2.5   | 46.31%     | 43.71% |  |

(c) The number of identical experts (Pure MoE).

| Experts | A-OKVOA | OK-VOA | ActivityN      | ClothoV2  |       |
|---------|---------|--------|----------------|-----------|-------|
| Laperts | A-ORVQA | OK-VQA | Accuracy score | Ciotilova |       |
| 1       | 65.24%  | 62.05% | 42.5%          | 2.5       | 23.3% |
| 4       | 64.98%  | 61.67% | 42.3%          | 2.8       | 21.5% |
| 6       | 66.81%  | 61.18% | 43.2%          | 2.6       | 21.8% |

(d) The internal architectures of Uni-MoE.

| Architecture | A-OKVQA | OK-VOA | ActivityN | et-QA | RACE-Audio |        |  |
|--------------|---------|--------|-----------|-------|------------|--------|--|
| Aicintecture | A-ORVQA | OR-VQA | Accuracy  | score | middle     | high   |  |
| First-Half   | 65.68%  | 60.96% | 41.9%     | 2.4   | 38.16%     | 41.14% |  |
| Second-Half  | 63.97%  | 61.33% | 43.2%     | 2.6   | 51.39%     | 52.69% |  |
| Interval     | 64.54%  | 61.77% | 43.3%     | 2.5   | 46.17%     | 46.60% |  |
| All          | 66.20%  | 63.02% | 42.8%     | 2.5   | 46.31%     | 43.71% |  |

for strong multimodal interaction and learning, validating its effectiveness in integrating diverse data types including video, speech, text and image.

**Token Pathways:** As shown in Figure 6, We examine the behaviour of experts at the token level. For all activated pathways, we employ PCA [74] to obtain the top-10 pathways. Notably, the expert indexes in this figure of token pathways have no strict corresponding relationship with the expert tags in the previous figures of routing distributions. Similar to the result from the routing analysis, the pathway of tokens shows the preference of different modalities in various experts of different MoE layers, which again contributes to a better understanding of the advancement in multi-modal related experts and the behaviour of Uni-MoE in multi-modal learning and inferring. Overall, Figures 4, 5, and 6 present the workflow of our Uni-MoE model at the expert-level, modality-level, and token-level perspectives. The above analysis indicates that *Uni-MoE* has learned a certain pattern that allows them to divide multiple-modality tasks in a specific manner.

#### 5.2 Ablation Study

Comparative Analysis of Uni-MoE and Dense Models. In previous Tables 3-6, we compare the performances of dense models (Single-Expert-Tasks) and Uni-MoE (w/ MoE-Tasks). The experimental results show that 1) The performance of Uni-MoE is consistently better than dense models on almost all evaluation benchmarks. By comparing the performances of

Single-Expert-Modality-Task6 and Uni-MoE w/ MoE-Task1 on speech-image, long speech, and image-text performances, where they use the same type of training datasets, we can see that Uni-MoE achieves better performances on all evaluation benchmarks, especially on the long speech understanding tasks. 2) After training on unbalanced mixed-modality data, Uni-*MoE exhibits less performance bias.* For instance, compared to the larger performance drop of Single-Modality-Expert-Task6, Uni-MoE trained with MoE-Task1 shows less performance degradation on short speech-image (in Table 3) and textimage understanding (in Table 5) tasks. It also improves the performances on the long speech understanding benchmarks including RACE-Audio and English High School Listening Test, wherein long speech training data accounts for a small proportion of the training data. 3) *Uni-MoE* shows better generalization than dense models for out-domain inputs when they are trained on the same types of mixed crossmodality data. Compared to Single-Expert-Modality-Task7, Task6 introduces the mixed data of text-image instruction and short speech-image during training. Introducing more types of multimodal data for dense models lowers the performance on long speech and does not enhance the performance on outdomain evaluation benchmark MMBench-Audio. However, when more data types are introduced, Uni-MoE not only maintains the performance in long speech understanding but also improves its performance in extrapolated three-modal input scenarios. Hence, these finding suggests the better ability of Uni-MoE to combine generalization on complex multimodal data

Impact of the Value of Top-k: Our ablation study, detailed in Table 7 (a), investigates the effect of varying the number of activated experts (Top-k) while we set the total number of experts to be identical. We observed that increasing the number of activated experts from one to two enhances model performance, indicating that activating more experts can significantly improve the efficiency of Uni-MoE. The increasing performance also suggests the collaboration ability of modality-specific experts in our model. It is identical to the visualization analysis of Uni-MoE in the bottom part of Figure 4, where it uses more experts at each layer to handle video content. Consequently, we have set the optimal number of activated experts at two to maximize performance across various cross-modality benchmarks in Tables 3- 6.

Scaling up the Number of Experts: We investigate variations in expert numbers while maintaining a constant count of activated experts, detailed in Table 7 (b) and (c). The results illustrate that Uni-MoE utilizing a greater number of sparse experts surpass the performance of the dense expert configurations, particularly excelling in long speech-text scenarios. This enhancement is attributed to the employment of two specialized experts, specifically trained on Single-Modality-Tasks 2 and 3, showcasing significant advancements in visual and speech tasks as demonstrated in Tables 3 and 5. Analysis of routing distributions further confirms the critical role of these single-modality trained experts in their respective fields. For Uni-MoE, we find that scaling number of experts to four can achieve better comprehensive performance on different modality. Conversely, as indicated in Table 7 (c), employing more standard experts from previous configurations without increasing the number of active experts

TABLE 8

<span id="page-12-0"></span>Ablation study about different training strategies and auxiliary balancing loss [42]. All models are trained for one epoch with the same mixed multimodal data from MoE-Task3 or Pure-MoE-Tasks. "mixture(4)" and "pure(4 or 6)" refer to Uni-MoE with four pre-tuned experts from the second training stage and pure MoE with four or six identical MLPs, respectively. The top-k value is set to 2. The "Source" represents which specific model the experts (MLP) are from. "Aux Loss" refers to the classical balancing loss proposed in GShard [42], aiming to encourage giving all experts equal importance. This loss ensures that all experts receive a roughly equal number of training examples.

|      | MoE      | experts    | Source           | Data           | Aux Loss | A-OKVOA | OK-VQA | ActivityNet-QA |       | - ClothoV2 | Avg.  |
|------|----------|------------|------------------|----------------|----------|---------|--------|----------------|-------|------------|-------|
|      | MICE     | experts    | Source           | Data           | Aux Loss | A-OKVQA | OK-VQA | Accuracy       | score | Citilova   | Avg.  |
| (a)  | <b>√</b> | mixture(4) | Training Stage 2 | MoE-Task3      | Х        | 66.20%  | 63.20% | 42.7%          | 2.5   | 24.7%      | 49.2% |
| (a') | 1        | mixture(4) | Training Stage 2 | MoE-Task3      | ✓        | 65.23%  | 61.92% | 42.9%          | 2.5   | 24.3%      | 48.5% |
| (b)  | 1        | pure(4)    | LLaVA-v1.5       | Pure-MoE-Task1 | ×        | 64.98%  | 61.67% | 42.1%          | 2.8   | 21.5%      | 47.5% |
| (b') | 1        | pure(4)    | LLaVA-v1.5       | Pure-MoE-Task1 | ✓        | 65.76%  | 61.99% | 41.9%          | 2.4   | 24.2%      | 48.4% |
| (c)  | 1        | pure(6)    | LLaVA-v1.5       | Pure-MoE-Task1 | ×        | 66.81%  | 61.18% | 43.2%          | 2.6   | 21.8%      | 48.2% |
| (c') | 1        | pure(6)    | LLaVA-v1.5       | Pure-MoE-Task1 | /        | 65.24%  | 61.61% | 42.1%          | 2.7   | 24.5%      | 48.3% |
| (d)  | X        | single     | LLaVA-v1.5       | Pure-MoE-Task1 | ×        | 65.24%  | 62.05% | 42.5%          | 2.5   | 23.3%      | 48.2% |
| (e)  | 1        | pure(4)    | LLaMA            | Pure-MoE-Task2 | Х        | 66.55%  | 57.25% | 41.6%          | 2.8   | 23.8%      | 47.3% |
| (f)  | X        | single     | LLaMA            | Pure-MoE-Task2 | ×        | 65.58%  | 56.12% | 41.3%          | 2.6   | 23.3%      | 46.5% |

TABLE 9

<span id="page-12-1"></span>Ablation study of Uni-MoE performances with adding more image-text training data and modality-specific experts. We only expand the image-text instruction data from LLaVA-150k (MoE-Task 1/2/3) to LLaVA-v1.5-665k (MoE-Task4). For models with XB, X refers to the size of the language model. For Uni-MoE with YE, Y refers to the number of experts. Names are abbreviated due to space limits. I: Image; T: Text; A: Audio; S: Speech; V: Video. AOK: A-OKVQA [63]; OK: OK-VQA [64]; MMB: MMBench [5]; POPE [75]; SEED:SEED-Bench [4]; MMVet [76]; RAudio: RACE-Audio; AN-QA: ActivityNet-QA [66]. "EHSL" refers to the English High School Listening Test we collected, containing long/short speech types. 

† indicates that this model employs enormous or unknown data during training.

| Method                               | AOK    | OK     | VQAv2  | MMB    | POPE   | SEED   | MMVet | RAudio         | EHSL                  | AN-QA |
|--------------------------------------|--------|--------|--------|--------|--------|--------|-------|----------------|-----------------------|-------|
| Any-Modality Understanding           |        |        |        |        |        |        |       |                |                       |       |
| Macaw-LLM [7]                        | 1.90%  | 5.70%  | 20.73% | 3.84%  | -      | -      | -     | 4.04%/3.00%    | 0.67%/2.00%           | -     |
| X-InstructBLIP [6]                   | 21.52% | 30.61% | 37.77% | 8.96%  | -      | -      | -     | 16.33%/18.88%  | 0.67%/2.00%           | -     |
| Dense Model                          |        |        |        |        |        |        |       |                |                       |       |
| AnyMAL-70B (I,T,A,V) [8]             | -      | 42.6%  | 64.2%  | -      | -      | -      | -     | -              | -                     | -     |
| IDEFICS-80B (I,T) [13]               | -      | -      | 60.0%  | 54.5%  | -      | -      | -     | -              | -                     | -     |
| LLaVA-1.5-7B (I,T) [3]               | 70.92% | 55.09% | 75.9%  | 72.2%  | 85.9%  | -      | 30.5% | -              | -                     | -     |
| LLaVA-1.5-13B (I,T) [3]              | 73.54% | 61.93% | 78.9%  | 73.0%  | 85.9%  | -      | 35.4% | -              | -                     | -     |
| BLIP-2(FlanT5-xxl)-11B (I,T) [50]    | 39.06% | 53.7%  | 65.0%  | -      | 85.3%  | -      | -     | -              | -                     | -     |
| InstructBLIP(Vicuna)-13B (I,T) [2]   | 58.30% | 41.02% | -      | -      | 50.7%  | 25.6%  | -     | -              | -                     | -     |
| Shikra-13B (I,T) [77]                | -      | -      | 77.4%  | 58.8%  | -      | -      | -     | -              | -                     | -     |
| LLaMA-VID-13B (I,T,V) [9]            | -      | -      | 80.0%  | 66.6%  | 86.0%  | 62.3%  | -     | -              | -                     | 47.5% |
| MiniGPT-4-7B (I,T) [35]              | 36.06% | 29.31% | -      | 23.0%  | -      | 42.84% | 22.1% | -              | -                     | -     |
| LLaMA-Adapter-v2-7B (I,T) [38]       | -      | -      | -      | 39.5%  | -      | -      | 31.4% | -              | -                     | -     |
| Qwen-VL-7B <sup>‡</sup> (I,T) [78]   | -      | 58.6%  | 78.8%  | 68.2%  | -      | 56.3%  | -     | -              | -                     | -     |
| MobileVLM-3B (I,T) [79]              | -      | -      | -      | 59.6%  | 84.9%  | -      | -     | -              | -                     | -     |
| LLaVA-Phi-3B (I,T) [80]              | -      | -      | 71.4%  | 59.8%  | 85.0%  | -      | 28.9% | -              | -                     | -     |
| Single-Modality-Expert-Task2 (I,T)   | 67.07% | 62.91% | 75.18% | 71.26% | 84.7%  | 60.63% | 27.8% | -              | -                     | -     |
| Single-Modality-Expert-Task5 (I,T,S) | 58.86% | 56.01% | 67.35% | 65.80% | 62.48% | 53.50% | 26.9% | 30.78%/24.90%  | 9.33%/12.00%          | -     |
| Single-Modality-Expert-Task6 (I,T,S) | 58.69% | 57.77% | 68.74% | 67.53% | 65.84% | 55.21% | 25.6% | 32.59%/29.02%  | 18.67%/8.00%          | -     |
| Sparse Model                         |        |        |        |        |        |        |       |                |                       |       |
| MoE-LLaVA-1.6B×4-Top2 [20]           | 63.8%  | 59.9%  | 74.1%  | 69.1%  | 85.7%  | 61.8%  | 28.0% | -              | -                     | -     |
| MoE-LLaVA-2.7B×4-Top2 [20]           | 68.34% | 62.10% | 75.4%  | 70.0%  | 85.5%  | 62.3%  | 31.2% | -              | -                     | -     |
| Uni-MoE w/ MoE-Task1 (4E) (I,T,S)    | 61.22% | 57.63% | 68.42% | 68.15% | 76.67% | 56.58% | 30.1% | 47.08%/47.08%  | 41.33%/36.00%         | -     |
| Uni-MoE w/ MoE-Task2 (4E) (I,T,S,V)  | 65.07% | 62.10% | 73.87% | 70.50% | 85.43% | 60.54% | 28.2% | 49.65% /49.37% | 42.00%/48.00%         | 42.2% |
| Uni-MoE w/ MoE-Task3 (4E) (I,T,A,V)  | 64.28% | 61.96% | 73.87% | 69.82% | 86.10% | 59.16% | 31.7% | -              | -                     | 42.8% |
| Uni-MoE w/ MoE-Task4 (4E) (I,T,S,V)  | 70.22% | 66.02% | 76.4%  | 73.2%  | 86.0%  | 63.4%  | 32.6% | 64.21%/64.72%  | 48.00%/58.67%         | 45.6% |
| + Aux_loss                           | 69.61% | 66.13% | 76.0%  | 72.6%  | 85.0%  | 63.3%  | 31.7% | 63.86%/64.24%  | <b>50.00%</b> /54.00% | 45.2% |
| Uni-MoE w/ MoE-Task4 (8E) (I,T,S,V)  | 70.0%  | 66.2%  | 76.6%  | 73.0%  | 86.2%  | 63.3%  | 32.5% | 63.75%/61.56%  | 48.67%/46.0%          | 46.0% |
| + Aux_loss                           | 70.7%  | 66.4%  | 76.7%  | 73.2%  | 86.3%  | 63.4%  | 32.8% | 62.33%/64.18%  | 42.00%/50.00%         | 46.4% |

leads to marginal improvements in overall performance. This observation underscores the necessity of strategic expert selection and the effectiveness of sparse expert configurations.

Analyzing the Architectures of Uni-MoE: In Table 7 (d), we evaluate four configurations of MoE architecture within the Uni-MoE framework. The "First-Half" configuration applies MoE layers exclusively to the initial segment of the model, maintaining a conventional dense structure in the latter half. Conversely, the "Second-Half" setup incorporates MoE layers in the latter segment while preserving a dense architecture in the initial segment. The "Interval" configuration intersperses MoE and dense layers throughout the model. Lastly, the "All" configuration converts all layers

to sparse MoE layers. We observe that fully converting to MoE ("All") does not exactly lead to superior performance and additionally incurs extended training durations when compared with other configurations. Notably, the "Interval" layout demonstrates the highest average efficacy across all tasks, establishing itself as the most effective architecture among those tested. Furthermore, positioning MoE layers in the latter half of the model significantly enhances the model's capacity for understanding lengthy speech segments, outperforming the second-best configuration by 5% and 6% than middle and high complexity categories, respectively. The above analysis presents that we still need to explore more robust and efficient MoE architectures in building larger MLLMs.

![](_page_13_Picture_2.jpeg)

![](_page_13_Picture_6.jpeg)

![](_page_13_Picture_14.jpeg)

![](_page_13_Picture_18.jpeg)

![](_page_13_Picture_21.jpeg)

![](_page_13_Picture_22.jpeg)

![](_page_13_Picture_24.jpeg)

![](_page_13_Picture_28.jpeg)

<span id="page-13-0"></span>

Fig. 7. **An illustration of various cases generated by Uni-MoE**. Interestingly, Uni-MoE trained with MoE-Task3 could understand the audio content from the video, while the instruction tuning data almost does not contain related data. It also could understand long speech from real English listening tests of high school students.

**Effectiveness of Training Strategy:** We examine the impact of different training strategies on model performance by presenting three distinct model variants in Table [8.](#page-12-0) The comparative analysis between model variant (a) and variants (b), (c) and (d) first reveals that the tri-phase training approach employed in the model (a) facilitates noticeable enhancements across a range of multimodal benchmarks. This underscores the benefit of incorporating specialized training phases for cross-modality data, affirming the strategic advantage of engaging training experts in the process. Further analysis shows that training MoE with identical configurations (see pure models in Table [8\)](#page-12-0) could result in a negligible performance increase compared to a singleexpert approach, as evidenced by models (b), (c), (d), (e), and (f). Interestingly, the performance of models can vary when the number of expert sources is increased, according to the unstable performances of adding identical experts from LLaMA or LLaVA. Experts trained on multimodal data perform better when experts work together. However, integrating an auxiliary balancing loss is a potential solution to mitigate these inconsistencies and stabilize performance. Moreover, visual illustration in Figures [8](#page-18-0) and [10](#page-20-0) (presented in the Appendix) highlight the homogeneity among the experts in model (b), suggesting a need to improve task allocation diversity and expert differentiation. *Overall, initiating singlemodality training proves advantageous for transitioning models from dense to sparse structures, as demonstrated by our approach. This strategy enhances initial model efficiency, facilitating a more effective and rapid adaptation in subsequent training phases, thereby validating our training strategy's effectiveness*.

**Analysis of Balancing auxiliary Loss:** The classical

balancing loss introduced in Gshard [\[42\]](#page-15-17) encourages giving all experts equal importance. This loss ensures that all experts receive a roughly equal number of training examples. In this paper, we also explore the effect of auxiliary balancing loss on model performance. As the experimental results are shown in Tables [8](#page-12-0) and [9,](#page-12-1) our findings indicate that: *1) Employing an auxiliary loss consistently enhances both the synergy among experts and the overall performance of the model across various modalities, when applied to the model with identical (pure) expert; 2) In our Uni-MoE model, as the number of experts expands to eight, the auxiliary loss shows its effectiveness to facilitate expert collaboration. This improvement is primarily attributed to the expanded routing search space resulting from the increased number of experts. The introduction of auxiliary loss at this stage plays a role in optimizing the selection of expert combinations, thereby fully activating the capabilities of the experts*.

**Comparisons of Uni-MoE and Dense Large Visual-Language Models (LVLMs)**. The results presented in Table [9](#page-12-1) indicate that the larger LVLMs, LLaVA-v1.5-13B and LLaMA-VID-13B, outperform Uni-MoE on image-text benchmarks. This superior performance can be attributed to two primary factors. First, these models focus exclusively on visual and language data during training, and they activate more parameters (13B) during inference compared to Uni-MoE's 11B, enhancing their effectiveness in image-text tasks. Second, LLaMA-VID benefits from the inclusion of 703K video data points used in pre-training video-to-language connections, a dataset not utilized by Uni-MoE, giving it an edge in evaluations like ActivityNet-QA. Additionally, for LLaMA-VID, the number of sampling frames is larger than Uni-MoE. Despite these differences, Uni-MoE still excels in imagetext comprehension over similar-sized MLLMs when the same image-text instruction data is incorporated. Moreover, it surpasses well-known unified multimodal models such as X-InstructBLIP and Macaw-LLM in other modal capabilities. *Interestingly, for Uni-MoE, adding image-text data enhances its performance in video understanding, which inspires us to further enhance the performance of video-LLMs by introducing additional image-text data*.

#### **5.3 Case Study**

In our analysis depicted in Figure [7,](#page-13-0) we present the performance of Uni-MoE trained with MoE-Task3 on different modalities. We can see that Uni-MoE could understand any cross-modality inputs and recognize the real long speech produced by humans and speech content in the video outside the training data, as the bottom examples are shown in Figure [7.](#page-13-0) Combining previous quantitative evaluation results and generated cases, we conclude that Uni-MoE shows its power to handle various modalities, trained with small yet diverse mixed multimodal data. Interestingly, Uni-MoE trained with MoE-Task3 could understand the audio content from the video, while the instruction tuning data almost does not contain related data. It indicates the robustness and generalization of utilizing MoE to handle various modalities compared to previous dense baselines such as X-InstructBLIP, which was trained on multiple modalities of data yet achieves inferior performance on speech-image or video understanding tasks.

# **6 CONCLUSION**

In this research, we ventured into expanding the capabilities of large multimodal models through the integration of Mixture of Experts (MoE) architecture, resulting in the development of Uni-MoE. We devised and implemented a novel three-phase training strategy specifically tailored to enhance both the stability and generalization performance of Uni-MoE across a broad spectrum of multimodal scenarios. This approach was rigorously tested against a variety of challenging benchmarks, encompassing both cross-modality comprehension and long-form speech and video reasoning tasks. Our findings reveal that Uni-MoE not only surpasses existing benchmarks in cross-modality and mixed-modality frameworks but also outperforms conventional MoE models equipped with identical experts. Furthermore, our detailed ablation studies validate the significance of our tailored three-phase training strategy, highlighting its critical role in enhancing the robustness and adaptability of MoE-based MLLMs when confronted with diverse multimodal datasets. We hope our work could spark the research of utilizing the MoE architecture to scale up MLLMs.

# **ACKNOWLEDGMENTS**

The authors would like to thank the efforts of editors and reviewers for checking our work.

# **REFERENCES**

- <span id="page-14-0"></span>[1] P. Xu, X. Zhu, and D. A. Clifton, "Multimodal learning with transformers: A survey," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45, no. 10, pp. 12 113–12 132, 2023.
- <span id="page-14-1"></span>[2] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, "Instructblip: Towards general-purpose visionlanguage models with instruction tuning," *NeurIPS*, 2024.
- <span id="page-14-2"></span>[3] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," *arXiv*, 2023.
- <span id="page-14-3"></span>[4] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, "Seed-bench: Benchmarking multimodal llms with generative comprehension," *arXiv preprint arXiv:2307.16125*, 2023.
- <span id="page-14-4"></span>[5] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu *et al.*, "Mmbench: Is your multi-modal model an all-around player?" *arXiv preprint arXiv:2307.06281*, 2023.
- <span id="page-14-5"></span>[6] A. Panagopoulou, L. Xue, N. Yu, J. Li, D. Li, S. Joty, R. Xu, S. Savarese, C. Xiong, and J. C. Niebles, "X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning," *arXiv preprint arXiv:2311.18799*, 2023.
- <span id="page-14-6"></span>[7] C. Lyu, M. Wu, L. Wang, X. Huang, B. Liu, Z. Du, S. Shi, and Z. Tu, "Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration," *arXiv preprint arXiv:2306.09093*, 2023.
- <span id="page-14-7"></span>[8] S. Moon, A. Madotto, Z. Lin, T. Nagarajan, M. Smith, S. Jain, C.-F. Yeh, P. Murugesan, P. Heidari, Y. Liu *et al.*, "Anymal: An efficient and scalable any-modality augmented language model," *arXiv preprint arXiv:2309.16058*, 2023.
- <span id="page-14-8"></span>[9] Y. Li, C. Wang, and J. Jia, "Llama-vid: An image is worth 2 tokens in large language models," *arXiv preprint arXiv:2311.17043*, 2023.
- <span id="page-14-9"></span>[10] OpenAI, "Gpt-4 technical report," *https://arxiv.org/abs/2303.08774*, 2023.
- <span id="page-14-10"></span>[11] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth *et al.*, "Gemini: a family of highly capable multimodal models," *Technical Report*, 2023.
- <span id="page-14-11"></span>[12] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, Z. Muyan, Q. Zhang, X. Zhu, L. Lu *et al.*, "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks," *CVPR*, 2024.
- <span id="page-14-12"></span>[13] H. Laurenc¸on, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela *et al.*, "Obelics: An open web-scale filtered dataset of interleaved imagetext documents," *Advances in Neural Information Processing Systems*, vol. 36, 2024.
- <span id="page-14-13"></span>[14] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *Techinal Report*, 2023.
- <span id="page-14-14"></span>[15] Y. Li, B. Hu, X. Chen, L. Ma, and M. Zhang, "Lmeye: An interactive perception network for large language models," *arXiv preprint arXiv:2305.03701*, 2023.
- <span id="page-14-15"></span>[16] Y. Li, L. Wang, B. Hu, X. Chen, W. Zhong, C. Lyu, and M. Zhang, "A comprehensive evaluation of gpt-4v on knowledge-intensive visual question answering," *arXiv preprint arXiv:2311.07536*, 2023.
- <span id="page-14-16"></span>[17] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparselygated mixture-of-experts layer," *ICLR*, 2017.
- <span id="page-14-17"></span>[18] S. E. Yuksel, J. N. Wilson, and P. D. Gader, "Twenty years of mixture of experts," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 23, no. 8, pp. 1177–1193, 2012.
- <span id="page-14-18"></span>[19] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand *et al.*, "Mixtral of experts," *Techinal Report*, 2024.
- <span id="page-14-19"></span>[20] B. Lin, Z. Tang, Y. Ye, J. Cui, B. Zhu, P. Jin, J. Zhang, M. Ning, and L. Yuan, "Moe-llava: Mixture of experts for large vision-language models," *CoRR*, vol. abs/2401.15947, 2024.
- <span id="page-14-20"></span>[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "Lora: Low-rank adaptation of large language models," *ICLR*, 2022.
- <span id="page-14-21"></span>[22] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *The Journal of Machine Learning Research*, vol. 23, no. 1, pp. 5232–5270, 2022.
- <span id="page-14-22"></span>[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- <span id="page-14-23"></span>[24] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, Z. Yang, Y. Zhang, and D. Tao, "A survey on vision

- transformer," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45, no. 1, pp. 87–110, 2023.
- <span id="page-15-0"></span>[25] J. Wu, X. Li, S. Xu, H. Yuan, H. Ding, Y. Yang, X. Li, J. Zhang, Y. Tong, X. Jiang, B. Ghanem, and D. Tao, "Towards open vocabulary learning: A survey," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, pp. 1–20, 2024.
- <span id="page-15-1"></span>[26] S. Wu, H. Fei, L. Qu, W. Ji, and T.-S. Chua, "Next-gpt: Any-to-any multimodal llm," *arXiv preprint arXiv:2309.05519*, 2023.
- <span id="page-15-2"></span>[27] Y. Zhang, K. Gong, K. Zhang, H. Li, Y. Qiao, W. Ouyang, and X. Yue, "Meta-transformer: A unified framework for multimodal learning," *arXiv preprint arXiv:2307.10802*, 2023.
- <span id="page-15-3"></span>[28] J. Zhan, J. Dai, J. Ye, Y. Zhou, D. Zhang, Z. Liu, X. Zhang, R. Yuan, G. Zhang, L. Li *et al.*, "Anygpt: Unified multimodal llm with discrete sequence modeling," *arXiv preprint arXiv:2402.12226*, 2024.
- <span id="page-15-4"></span>[29] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, "Imagebind: One embedding space to bind them all," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 15 180–15 190.
- <span id="page-15-5"></span>[30] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo *et al.*, "Segment anything," *ICCV*, 2023.
- <span id="page-15-6"></span>[31] X. Li, H. Zhang, R. Wang, and F. Nie, "Multiview clustering: A scalable and parameter-free bipartite graph fusion method," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 44, no. 1, pp. 330–344, 2022.
- <span id="page-15-7"></span>[32] J. Zhang, J. Huang, S. Jin, and S. Lu, "Vision-language models for vision tasks: A survey," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, pp. 1–20, 2024.
- <span id="page-15-8"></span>[33] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma *et al.*, "Scaling instructionfinetuned language models," *arXiv preprint arXiv:2210.11416*, 2022.
- <span id="page-15-9"></span>[34] Y. Zhang, R. Zhang, J. Gu, Y. Zhou, N. Lipka, D. Yang, and T. Sun, "Llavar: Enhanced visual instruction tuning for text-rich image understanding," *NeurIPS*, 2023.
- <span id="page-15-10"></span>[35] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, "Minigpt-4: Enhancing vision-language understanding with advanced large language models," *ICLR*, 2024.
- <span id="page-15-11"></span>[36] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A frontier large vision-language model with versatile abilities," *Technical Report*, 2023.
- <span id="page-15-12"></span>[37] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, "Visual chatgpt: Talking, drawing and editing with visual foundation models," *arXiv preprint arXiv:2303.04671*, 2023.
- <span id="page-15-13"></span>[38] P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou, W. Zhang, P. Lu, C. He, X. Yue *et al.*, "Llama-adapter v2: Parameter-efficient visual instruction model," *ICLR*, 2024.
- <span id="page-15-14"></span>[39] S. Bao, Q. Xu, Z. Yang, X. Cao, and Q. Huang, "Rethinking collaborative metric learning: Toward an efficient alternative without negative sampling," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45, no. 1, pp. 1017–1035, 2023.
- <span id="page-15-15"></span>[40] J. Li, P. Chen, S. Yu, S. Liu, and J. Jia, "Bal: Balancing diversity and novelty for active learning," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 46, no. 5, pp. 3653–3664, 2024.
- <span id="page-15-16"></span>[41] D. Eigen, M. Ranzato, and I. Sutskever, "Learning factored representations in a deep mixture of experts," *ICLR Workshop*, 2014.
- <span id="page-15-17"></span>[42] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *ICLR*, 2021.
- <span id="page-15-18"></span>[43] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer, "Base layers: Simplifying training of large, sparse models," in *International Conference on Machine Learning*. PMLR, 2021, pp. 6265–6274.
- <span id="page-15-19"></span>[44] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat *et al.*, "Glam: Efficient scaling of language models with mixture-of-experts," in *International Conference on Machine Learning*. PMLR, 2022, pp. 5547–5569.
- <span id="page-15-20"></span>[45] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," in *NeurIPS*, 2023.
- <span id="page-15-21"></span>[46] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark *et al.*, "Learning transferable visual models from natural language supervision," in *International conference on machine learning*. PMLR, 2021, pp. 8748–8763.
- <span id="page-15-22"></span>[47] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, "Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality," March 2023.

- <span id="page-15-23"></span>[48] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 2023, pp. 28 492–28 518.
- <span id="page-15-24"></span>[49] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, and F. Wei, "Beats: Audio pre-training with acoustic tokenizers," *PMLR*, 2023.
- <span id="page-15-25"></span>[50] J. Li, D. Li, S. Savarese, and S. Hoi, "Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models," *ICML*, 2023.
- <span id="page-15-26"></span>[51] microsoft, "Phi-2: The surprising power of small language models," 2023.
- <span id="page-15-27"></span>[52] G. Wang, S. Cheng, X. Zhan, X. Li, S. Song, and Y. Liu, "Openchat: Advancing open-source language models with mixed-quality data," *ICLR*, 2024.
- <span id="page-15-28"></span>[53] R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, "Common voice: A massively-multilingual speech corpus," in *Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)*, 2020, pp. 4211–4215.
- <span id="page-15-29"></span>[54] Microsoft, "Text to speech an ai speech feature that converts text to lifelike speech."
- <span id="page-15-30"></span>[55] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: an asr corpus based on public domain audio books," in *ICASSP*. IEEE, 2015, pp. 5206–5210.
- <span id="page-15-31"></span>[56] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, "Race: Large-scale reading comprehension dataset from examinations," *EMNLP*, 2017.
- <span id="page-15-32"></span>[57] X. Mei, C. Meng, H. Liu, Q. Kong, T. Ko, C. Zhao, M. D. Plumbley, Y. Zou, and W. Wang, "Wavcaps: A chatgpt-assisted weaklylabelled audio captioning dataset for audio-language multimodal research," *arXiv preprint arXiv:2303.17395*, 2023.
- <span id="page-15-33"></span>[58] C. D. Kim, B. Kim, H. Lee, and G. Kim, "Audiocaps: Generating captions for audios in the wild," in *ACL*, 2019, pp. 119–132.
- <span id="page-15-34"></span>[59] K. Drossos, S. Lipping, and T. Virtanen, "Clotho: An audio captioning dataset," in *ICASSP*. IEEE, 2020, pp. 736–740.
- <span id="page-15-35"></span>[60] S. Lipping, P. Sudarsanam, K. Drossos, and T. Virtanen, "Clothoaqa: A crowdsourced dataset for audio question answering," in *2022 30th European Signal Processing Conference (EUSIPCO)*. IEEE, 2022, pp. 1140–1144.
- <span id="page-15-36"></span>[61] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, "Meld: A multimodal multi-party dataset for emotion recognition in conversations," *ACL*, 2019.
- <span id="page-15-37"></span>[62] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, "Video-chatgpt: Towards detailed video understanding via large vision and language models," *arXiv preprint arXiv:2306.05424*, 2023.
- <span id="page-15-38"></span>[63] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi, "A-okvqa: A benchmark for visual question answering using world knowledge," *ECCV*, 2023.
- <span id="page-15-39"></span>[64] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, "Okvqa: A visual question answering benchmark requiring external knowledge," in *Proceedings of the IEEE/cvf conference on computer vision and pattern recognition*, 2019, pp. 3195–3204.
- <span id="page-15-40"></span>[65] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the v in vqa matter: Elevating the role of image understanding in visual question answering," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, pp. 6904–6913.
- <span id="page-15-41"></span>[66] B. G. Fabian Caba Heilbron, Victor Escorcia and J. C. Niebles, "Activitynet: A large-scale video benchmark for human activity understanding," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2015, pp. 961–970.
- <span id="page-15-42"></span>[67] D. Chen and W. B. Dolan, "Collecting highly parallel data for paraphrase evaluation," in *Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies*, 2011, pp. 190–200.
- <span id="page-15-43"></span>[68] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.
- <span id="page-15-44"></span>[69] G. Li, Y. Xu, and D. Hu, "Multi-scale attention for audio question answering," *InterSpeech*, 2023.
- <span id="page-15-45"></span>[70] M. Kim, K. Sung-Bin, and T.-H. Oh, "Prefix tuning for automated audio captioning," in *ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2023, pp. 1–5.
- <span id="page-15-46"></span>[71] A. Yang, A. Miech, J. Sivic, I. Laptev, and C. Schmid, "Zeroshot video question answering via frozen bidirectional language

- models," *Advances in Neural Information Processing Systems*, vol. 35, pp. 124–141, 2022.
- <span id="page-16-0"></span>[72] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, "Videochat: Chat-centric video understanding," *arXiv preprint arXiv:2305.06355*, 2023.
- <span id="page-16-1"></span>[73] H. Zhang, X. Li, and L. Bing, "Video-llama: An instruction-tuned audio-visual language model for video understanding," *arXiv preprint arXiv:2306.02858*, 2023.
- <span id="page-16-2"></span>[74] P. Pearson and K. Karl, "Liii. on lines and planes of closest fit to systems of points in space," *Philosophical Magazine Series 1,Philosophical Magazine Series 1*.
- <span id="page-16-3"></span>[75] Y. Li, Y. Du, K. Zhou, J. Wang, X. Zhao, and J.-R. Wen, "Evaluating object hallucination in large vision-language models," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, H. Bouamor, J. Pino, and K. Bali, Eds., 2023, pp. 292–305.
- <span id="page-16-4"></span>[76] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, "Mm-vet: Evaluating large multimodal models for integrated capabilities," *arXiv preprint arXiv:2308.02490*, 2023.
- <span id="page-16-5"></span>[77] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, "Shikra: Unleashing multimodal llm's referential dialogue magic," *arXiv preprint arXiv:2306.15195*, 2023.
- <span id="page-16-6"></span>[78] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang *et al.*, "Qwen technical report," *arXiv preprint arXiv:2309.16609*, 2023.
- <span id="page-16-7"></span>[79] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei *et al.*, "Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices," *Technical Report*, 2023.
- <span id="page-16-8"></span>[80] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang, "Llava-phi: Efficient multi-modal assistant with small language model," *arXiv preprint arXiv:2401.02330*, 2024.

![](_page_16_Picture_12.jpeg)

**Yunxin Li** received the M.S. and B.S. degrees from the Harbin Institute of Technology, in 2019 and 2022. He is currently pursuing the Ph.D. degree with the School of Computer Science, Harbin Institute of Technology, Shenzhen. His research interests include large multimodal models, cross-modal reasoning, and natural language generation.

![](_page_16_Picture_14.jpeg)

**Shenyuan Jiang** earned a Bachelor of Science degree in Computer Science from Harbin Institute of Technology, Shenzhen campus, in 2022. He is now pursuing a Master's degree at the School of Computer Science in Harbin Institute of Technology, Shenzhen. Jiang's research interests are centered around the development and enhancement of multimodal large language models.

![](_page_16_Picture_16.jpeg)

**Baotian Hu** received the M.S. and Ph.D. degrees in computer science from the Shenzhen Graduate School of Harbin Institute of Technology, Shenzhen, China, in 2012 and 2016, respectively. He is currently an Associate Professor with the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen. His current research interests include deep learning and its application in natural language processing and vision-language information processing.

![](_page_16_Picture_18.jpeg)

**Longyue Wang** is a Senior Researcher with the Tencent AI Lab, Shenzhen, China. He received his Ph.D. degree from Dublin City University in 2018. Longyue has studied and practiced in a broad field of Artificial Intelligence especially on Large Language Model, Multimodal, Language Agent, Natural Language Processing, Machine Translation, Deep Learning and AI for Science. He has published 60 papers in leading NLP journals and conferences.

![](_page_16_Picture_20.jpeg)

**Wanqi Zhong** is an undergraduate student majoring in Computer Science and Technology at Harbin Institute of Technology, China. His current research interests include computer vision and deep learning, particularly focusing on visioncentral general perception.

![](_page_16_Picture_22.jpeg)

**Wenhan Luo** (Senior Member, IEEE) is currently an Associate Professor at the Hong Kong University of Science and Technology. Prior to that, he worked as an Associate Professor at Sun Yat-sen University and as a research scientist for Tencent and Amazon. He has published over 80 papers in top conferences and leading journals. He also has served as a reviewer, senior PC member, and Guest Editor for several prestigious journals and conferences. He received Ph.D. degree from Imperial College London in 2016, M.E. degree

from the Institute of Automation, Chinese Academy of Sciences in 2012, and B.E. degree from Huazhong University of Science and Technology in 2009.

![](_page_16_Picture_25.jpeg)

**Lin Ma** received the B. E., and M. E. degrees from Harbin Institute of Technology, Harbin, China, in 2006 and 2008, respectively, both in computer science. He received his Ph.D. degree in the Department of Electronic Engineering at the Chinese University of Hong Kong (CUHK) in 2013. He is now a Researcher with Meituan, Beijing, China. His current research interests lie in the areas of deep learning and multi-modal learning, specifically for image and language.

![](_page_16_Picture_27.jpeg)

**Min Zhang** received the Ph.D., M.S., and B.S. degrees in Computer Science and Technology at the Harbin Institute of Technology, in 1991, 1994, and 1997, respectively. He is currently a Professor at the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen. His research interests include multimodal reasoning, large models, and natural language generation.

# **APPENDIX**

# **Comparative Visualization Analysis of Uni-MoE (4 experts) trained with MoE-Task2 and Pure-Task1**

In this section, we present the routing distributions and token pathways of MoE-Task2 and Pure-MoE-Task1 on five selected combinations of multi-modal data(Image-Text, Audio-Text, Video-Text, Image-Audio, Video-Audio-Text), each figure is shown utilizing 200 of data pair samples. These routing distributions are based on the training up to one epoch.

**Routing Distributions**. In our study, we conducted an ablation analysis by training a model called Pure-MoE-Task1. This model's performance did not meet the levels exhibited by other Uni-MoEs. The routing distribution for Pure-MoE-Task1, as illustrated in Figure [8,](#page-18-0) shows notable differences compared to MoE-Task3 presented in Figure [4.](#page-9-0) Specifically, Pure-MoE-Task1 demonstrated a relatively balanced distribution in terms of both expert loads and their preferences for different modalities. Conversely, MoE-Task3 and other Uni-MoE models, such as MoE-Task2 (illustrated in Figure [9\)](#page-19-0), exhibited distinct preferences among the experts that were fine-tuned during the single-modality optimization phase. For example, MoE-Task2 includes four experts, each fine-tuned for a different purpose: Expert 1 was trained using an audio-relevant dataset derived from an image dataset, Expert 2 was adapted from the fine-tuned LLaVA model's MLP layers, Expert 3 was developed for long speech training tasks, and Expert 4 was optimized for image-related tasks with textual information. The data suggest a strong relationship between the preference of an expert and the single-expert training stage. Specifically, during scenarios involving audio features, the workload of Expert 3, which was fine-tuned for long speech tasks, significantly increases. Similarly, with image inputs, Expert 4's workload exceeds that of all other experts. When handling video files containing both visual and audio content, the workload is almost evenly distributed between Experts 3 and 4, highlighting their significant roles in MoE training for task-specific contexts.

Overall, our findings suggest that the strategy of finetuning individual experts effectively transfers the capabilities of single modalities to enhance the performance of sparse Large Language Models (LLMs) across various tasks. While utilizing identical experts across all modules does not distinctly separate their functions, it inadvertently reveals unique patterns for each expert, which in some cases, proves to be effective. This characteristic pattern underscores the innovative approach of using Mixture of Experts in the Multimodal Large Language Model (MLLM) field. Future research should aim to further leverage the potential of MoE within MLLM to enhance its application and effectiveness.

**Token Pathways**. In Figure [12](#page-21-0) and Figure [13,](#page-21-1) we track the paths of each token for Pure-MoE-Task1 and MoE-Task2, respectively. In general, the overall trends of the token paths align with the analysis in the above routing distributions. The paths of Pure-MoE-Task1 appear more disorderly and diverse, which is attributed to a more balanced expert assignment. On the other hand, MoE-Task2 shows its unique preference for experts.

![](_page_18_Figure_2.jpeg)

<span id="page-18-0"></span>Fig. 8. Distribution of expert loadings and expert preferences on **Pure-MoE-Task1.** For different cross-modality data pairs, different experts from different layers have a high degree of consistency. we fail to observe the modalities for which different experts are primarily responsible. This may be attributed to the fact that experts are initially identical.

![](_page_19_Figure_2.jpeg)

<span id="page-19-0"></span>Fig. 9. Distribution of expert loadings and expert preferences on **MoE-Task2.**

![](_page_20_Figure_2.jpeg)

<span id="page-20-0"></span>Fig. 10. Distribution of modalities across different experts on **Pure-MoE-Task1**.

Fig. 11. Distribution of modalities across different experts on **MoE-Task2**.

![](_page_21_Figure_2.jpeg)

<span id="page-21-0"></span>Fig. 12. Visualization of activated pathways on **Pure-MoE-Task1**. Fig. 13. Visualization of activated pathways on **MoE-Task2**.

<span id="page-21-1"></span>