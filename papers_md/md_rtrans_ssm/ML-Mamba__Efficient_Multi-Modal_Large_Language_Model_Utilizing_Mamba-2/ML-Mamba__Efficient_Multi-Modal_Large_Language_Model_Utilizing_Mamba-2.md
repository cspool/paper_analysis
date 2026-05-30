![](_page_0_Picture_1.jpeg)

Wenjun Huang1,<sup>∗</sup> Jiakai Pan1,<sup>∗</sup> Jiahao Tang<sup>1</sup> Yanyu Ding<sup>2</sup> Yifei Xing<sup>3</sup> Yuhe Wang<sup>1</sup> Zhengzhuo Wang<sup>1</sup> Jianguo Hu1,† <sup>1</sup>Sun Yat-sen University, <sup>2</sup>Dongguan University of Technology <sup>3</sup>University of the Chinese Academy of Sciences huangwj98@mail2.sysu.edu.cn hujguo@mail.sysu.edu.cn Project URL: <https://wenjunhuang94.github.io/ML-Mamba> <sup>∗</sup>These authors contributed equally to this work. †Corresponding author: Jianguo Hu

# Abstract

Multimodal Large Language Models (MLLMs) have attracted much attention for their multifunctionality. However, traditional Transformer architectures incur significant overhead due to their secondary computational complexity. To address this issue, we introduce ML-Mamba, a multimodal language model, which utilizes the latest and efficient Mamba-2 model for inference. Mamba-2 is known for its linear scalability and fast processing of long sequences. We replace the Transformer-based backbone with a pre-trained Mamba-2 model and explore methods for integrating 2D visual selective scanning mechanisms into multimodal learning while also trying various visual encoders and Mamba-2 model variants. Our extensive experiments in various multimodal benchmark tests demonstrate the competitive performance of ML-Mamba and highlight the potential of state space models in multimodal tasks. The experimental results show that: (1) we empirically explore how to effectively apply the 2D vision selective scan mechanism for multimodal learning. We propose a novel multimodal connector called the Mamba-2 Scan Connector (MSC), which enhances representational capabilities. (2) ML-Mamba achieves performance comparable to state-of-the-art methods such as TinyLaVA and MobileVLM v2 through its linear sequential modeling while faster inference speed; (3) Compared to multimodal models utilizing Mamba-1, the Mamba-2-based ML-Mamba exhibits superior inference performance and effectiveness.

# 1 Introduction

The emergence of Large Language Models (LLMs) has profoundly changed the landscape of natural language understanding tasks. Unlike early methods that relied on medium-sized task specific models, recent advances have shifted towards using general large-scale models, especially after the success of systems such as ChatGPT. It has been proven that expanding the scale of language models and increasing data volume can bring many advantages, including enhancing the performance of different tasks and improving the sample efficiency of out of distribution generalization [\[18\]](#page-13-0).

However, traditional LLMs are limited to interacting through language, which limits their adaptability to handling more diverse tasks. Multi modal understanding that integrates visual and textual information is crucial for improving the ability of models to effectively respond to real-world challenges. Therefore, researchers are actively expanding large-scale language models to integrate multimodal

information processing capabilities. Visual language models (VLMs) such as GPT-4 [\[38\]](#page-14-0), LLaMA adapter [\[10\]](#page-12-0), and LLaVA [\[33,](#page-13-1) [32\]](#page-13-2) have been developed to enhance LLM's visual comprehension ability. These VLMs are fundamental models for handling a range of tasks, including visual question answering (VQA), image captioning, and visual content generation.

Despite achieving success, previous research has mainly focused on reducing the parameters of language models while preserving the Transformer architecture. However, this method does not solve the inherent problem of low computational efficiency in Transformer's self attention mechanism, which is quadratic with sequence length. To address this bottleneck, the latest research work has designed a new architecture (Mamba-2), whose core layer is an improvement of Mamba selective SSM. The state space model (SSM) has been widely studied as an effective alternative solution. SSM combines elements of Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs), providing linear scaling of sequence length and effective training and inference. It is 2-8 times faster and continues to compete with Transformers in language modeling.

To this end, this article proposes a new perspective, directly using the state space model (SSM) as the backbone. Specifically, we use the Mamba-2 language model as the basic model of our VLM. In this article, we introduce ML-Mamba, a work that applies state space models to multimodal learning tasks. Our method utilizes a pre-trained Mamba-2 language model as the backbone, replacing traditional Transformer-based models such as LLaMA [\[47\]](#page-14-1). We further enhanced ML-Mamba through a novel multimodal connector called Mamba-2 Scan Connector (MSC) architecture, which includes a Mamba-2 visual selective scanning module (MVSS) and a SwiGLU module specifically both designed for 2D causal modeling of enriched visual sequences. The MVSS module explores two different scanning mechanisms: bidirectional scanning mechanism (BSM) and cross scanning mechanism (CSM). In addition, we investigated the combination of different visual encoders, variants of pre-trained Mamba-2 language models, and multimodal connectors to optimize the integration of visual and linguistic information.

Extensive experiments conducted on a range of multimodal learning benchmarks demonstrate the efficacy of ML-Mamba. Our model not only achieves competitive performance with other similarly sized small multimodal large-scale language models (MLLMs) but also surpasses larger MLLMs on several prominent benchmark tests, including LLaVA v1.5 [\[32\]](#page-13-2) versions 7B and 13b.

The major contributions of this paper are three-fold:

- We propose a novel and efficient method, i.e., ML-Mamba, which explores and utilizes multimodal learning tasks combined with the latest Mamba-2. Compared to the multimodal model adopting the original Mamba, the multimodal large-scale language model based on Mamba-2 has higher inference performance and effectiveness. Meanwhile, ML-Mamba also provides a new framework choice for multimodal large-scale language models beyond Transformer-based architectures.
- We empirically explore the impact of different components in ML-Mamba and propose a novel multimode connector called Mamba-2 Scan Connector (MSC). MSC includes the Mamba-2 Visual Selective Scanning (MVSS) module and the SwiGLU module, which enhance representational capabilities.
- We conduct extensive experiments on different multimodal learning benchmarks. The numerical results show that ML-Mamba achieves competitive performance compared to existing multimodal large-scale language models.

# 2 Related Work

#### 2.1 Large Language Models (LLMs)

In recent years, significant breakthroughs have been made in natural language processing tasks [\[21,](#page-13-3) [24\]](#page-13-4), characterized by large model scales, typically containing billions of parameters, and training using massive datasets. GLM [\[8\]](#page-12-1), LLaMA [\[47\]](#page-14-1), Alpaca [\[45\]](#page-14-2), Vicuna [\[4\]](#page-12-2) and other instruction fine-tuning versions have emerged one after another, with the goal of being comparable to the proprietary InstructGPT model without public access. At the same time, due to the significant computational requirements of large language models, research trends have shifted towards exploring the possibility of smaller scale models, such as Stable LM [\[2\]](#page-12-3), TinyLaMA [\[51\]](#page-14-3), and Phi [\[16,](#page-13-5) [28\]](#page-13-6), which have parameter sizes below 3 billion but can achieve comparable results to large models through high-quality data and feasible training methods.

#### 2.2 State Space Models (SSMs)

State Space Models (SSMs) have demonstrated excellent performance in areas such as long sequence modeling, image generation, and reinforcement learning. A notable feature of SSMs is their ability to perform efficient autoregressive inference like Recurrent Neural Networks (RNNs) while also being able to process entire input sequences in parallel like attention-based Transformers, thus enabling efficient training. Despite their efficiency, SSMs achieve good results in various sequence modeling tasks. Specifically, Albert et al. [\[15\]](#page-12-4) proposed a structured state space sequence model for time series analysis. Goel et al. [\[11\]](#page-12-5) applied SSMs to audio generation and achieved satisfactory performance. Additionally, the H3 model [\[9\]](#page-12-6) was introduced to bridge the gap between SSMs and Transformers in language modeling.

In recent months, a new selective state space model called Mamba [\[14\]](#page-12-7) has been proposed as a strong competitor to the Transformer architecture. Compared to LLMs of the same capacity, language models based on Mamba have shown competitive performance, faster inference speeds, and the ability to scale linearly over time with constant memory usage. In May 2024, the latest Mamba architecture (Mamba-2) [\[7\]](#page-12-8) was introduced, featuring an improved core layer of the Mamba selective SSM, which is 2-8 times faster while continuing to compete with Transformers in language modeling.

#### 2.3 Multimodal Large Language Model (MLLM)

The Multi Modal Large Language Model (MLLM) combines visual and linguistic information and has achieved significant success in various fields [\[48,](#page-14-4) [49,](#page-14-5) [36\]](#page-13-7). However, the basis of these models is usually a known Transformer network, resulting in a square level computational complexity [\[22\]](#page-13-8). In order to improve the efficiency of the base model, ML-Mamba is proposed, which is an MLLM with linear computational complexity. Specifically, ML-Mamba integrates the efficient Mamba-2 language model into visual modalities and explores different modal fusion strategies to create effective multimodal Mamba-2 [\[7\]](#page-12-8). Experiments have shown that ML-Mamba not only competes with current computationally efficient MLLMs such as LLaVA Phi, TinyLaVA, and MobileVLM v2, but also runs faster due to its linear sequence modeling characteristics. Interestingly, the results of the closed set prediction benchmark test show that ML-Mamba performs well in overcoming visual illusions and spatial relationship judgments, even comparable to LLaVA in performance with only 40% of its parameters.

In terms of MLLMs for instruction tuning, recent research [\[23\]](#page-13-9) has questioned the necessity of the pre alignment phase in MLLM training, pointing out that directly fine-tuning the entire LLM backbone and projector may be sufficient. In line with this, ML-Mamba only underwent a small amount of alignment training and then fine-tuned it on a large combination dataset containing visual multi-turn dialogues and visual alignment instructions.

#### 2.4 Mamba in the field of vision

The successful application of Mamba in natural language processing (NLP) has inspired its adoption in visual applications [\[43\]](#page-14-6). Vision Mamba (Vim) [\[56\]](#page-14-7) utilizes Vim blocks composed of pure Mamba layers: each Vim block models bidirectional representations using forward and backward scanning, and alleviates direction sensitivity issues in Mamba. Another approach, VMamba [\[34\]](#page-13-10) utilizes Visual State Space (VSS) blocks that integrate Mamba and 2D convolutional layers, supported by a pyramid architecture similar to Swin Transformer [\[35\]](#page-13-11): each VSS block first models 2D local information through 2D deep convolution as a token mixer, and then processes 2D global information horizontally and vertically through a cross scan module. Mamba ND [\[27\]](#page-13-12) further extends the functionality of Mamba to multidimensional data including images and videos. LocalMamba [\[19\]](#page-13-13) segments the input image into multiple local windows and executes a state space model (SSM) in various directions within these windows to enhance local processing capabilities. EfficientVMamba [\[41\]](#page-14-8) introduced an efficient 2D scanning technique that reduces computational requirements by performing atrous sampling on feature map blocks. In addition to these newly designed Mamba architectures, our work also draws inspiration from VL-Mamba [\[42\]](#page-14-9), a multimodal large language model based on state space models, which has shown great potential for long-sequence modeling with fast inference and

linear scaling in sequence length. Compared with these newly designed Mamba architectures, our architecture closely follows Mamba's design ideas in the field of vision, enhancing the extraction of visual features with the latest Mamba-2 module. Our main goal with the Mamba-2 based architecture is to enhance multimodal representation and inference capabilities.

#### 3 Method

In this section, we first introduce the basic concepts of State Space Models (SSMs) (Sec. 3.1). Subsequently, we provide a detailed description of the proposed ML-Mamba method (Sec. 3.2), which mainly comprises a visual encoder, a multi-modal connector called the Mamba-2 Scan Connector (MSC), an MLP projector, and the Mamba-2 large language model.

#### <span id="page-3-0"></span>3.1 Mamba Preliminaries

The Mamba architecture is derived from state space sequence models [15], which models a 1-D function or sequence  $x(t) \in \mathbb{R} \to y(t) \in \mathbb{R}$  at time t via expanded hidden states  $h_t \in \mathbb{R}^N$ . These hidden states evolve over time according to parameters  $\mathbf{A}, \mathbf{B}, \mathbf{C}$  and are governed by linear ordinary differential equations (ODEs):

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t),$$
  

$$y(t) = \mathbf{C}h(t).$$
(1)

<span id="page-3-2"></span>To discretize parameters in this continuous system, a common approach is to introduce a time scale parameter  $\Delta$  to transform continuous  $\mathbf{A}, \mathbf{B}$  into discrete  $\overline{\mathbf{A}}, \overline{\mathbf{B}}$  using the zero-order hold (ZOH) model [39]:

$$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), 
\overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}.$$
(2)

Using this transformation, Eq. 1 can be rewritten as:

$$h'(t) = \overline{\mathbf{A}}h_{t-1} + \overline{\mathbf{B}}x_t,$$
  

$$y_t = \mathbf{C}h_t.$$
(3)

We then utilize the matrix  $\overline{\mathbf{K}}$  to enable efficient computation:

$$\overline{\mathbf{K}} = (\mathbf{C}\overline{\mathbf{B}}, \mathbf{C}\overline{\mathbf{A}}\overline{\mathbf{B}}, ..., \mathbf{C}\overline{\mathbf{A}}^k \overline{\mathbf{B}}, ...),$$

$$\mathbf{v} = \mathbf{x} * \overline{\mathbf{K}},$$
(4)

where  $k \in [0, L)$  and L is the input sequence length. We also have  $\mathbf{y} = \{y_1, ..., y_L\}$ ,  $\mathbf{x} = \{x_1, ..., x_L\}$ , while  $\overline{\mathbf{K}} \in \mathbb{R}^L$  can be regarded as the convolutional kernel.

By combining the modified parallel Mamba blocks with using SSD as the inner SSM layer, the Mamba-2 architecture is formed (as shown in Fig. 4(a)). The performance of Mamba-2 models of varying sizes on the Pile dataset shows that it matches or outperforms Mamba and other open-source Transformer models on standard downstream evaluations.

#### <span id="page-3-1"></span>3.2 ML-Mamba Model

## 3.2.1 Overall Architecture

The architecture of Mamba consists of four main components: a pre-trained visual encoder, a randomly initialized multi-modal connector called Mamba-2 Scan Connector (MSC), and a pre-trained large language model (Mamba-2 LLM), as shown in Fig. 1. With an image as input, visual features are first extracted through the visual encoder. The extracted sequence of visual features is then fed into the multi-modal connector (MSC), whose output is mapped to the LLM using a multi-layer perceptron (MLP) projector. The output vector from the visual projector is then combined with tokenized text queries and input into the Mamba-2 LLM. Finally, the Mamba-2 LLM generates the corresponding response.

<span id="page-4-0"></span>The image features a city street with a white bus parked on the side of the road. The bus is quite large, occupying a significant portion of the street. There are several people walking around the area, with one person closer to the bus and two others further away......

![](_page_4_Figure_1.jpeg)

Figure 1: The architecture of ML-Mamba (right) uses Mamba-2 as the backbone (left). It includes a visual encoder, a multi-modal connector called the Mamba-2 Scan Connector (MSC), an MLP projector, and a language model. We use the pre-trained Mamba-2 large language model (Mamba-2 LLM) as the language model and a pre-trained visual transformer model as the visual encoder.

<span id="page-4-1"></span>![](_page_4_Figure_3.jpeg)

Figure 2: Three architectures of MultiModal Connector: (a) MLP; (b) MSC-MLP (Basic); (c) MSC-MLP (Advanced).

### 3.2.2 Vision Encoder

We integrate DINOv2 [40] and SigLIP [50] to serve as our vision backbone. The rationale behind this fusion is that combining the low-level spatial features captured by DINOv2 with the semantic features provided by SigLIP enhances performance on downstream tasks [46, 23]. Given an input image  $X_v \in \mathbb{R}^{C \times H \times W}$ , the vision encoder divides the image into  $N_v = HW/P^2$  patches of equal size, where  $P^2$  represents the patch size. Both vision encoders process the patchified image as an input token sequence and concatenate their outputs to form compact visual representations  $V_{img} \in \mathbb{R}^{N_v \times D_v}$ :

$$V_{img} = [\varphi_{\text{SigLIP}}(X_v); \varphi_{\text{DINOv2}}(X_v)], \tag{5}$$

These outputs are then channeled to a dedicated task-specific head, with  $D_v$  representing the dimensionality of the tokens generated as described above.

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 3: Illustration of two different Vision Selective Scan (VSS) Mechanisms: Bidirectional-Scan Mechanism (BSM) (top) and Cross-Scan Mechanism (CSM) (bottom).

#### 3.2.3 MultiModal Connector

Multimodal connectors act between visual features and language models to ensure seamless integration of visual and linguistic information. In this study, we explored a novel multimodal connector called Mamba-2 Scan Connector (MSC) architecture aimed at addressing the challenge of unclear causal relationships in computer vision. The traditional state space model (SSM) is typically used to process sequence data with causal relationships, such as language sequences, but this approach is clearly not applicable to non causal visual sequences generated by visual encoders.

The core of the MSC module is a combination of the two-dimensional Mamba-2 visual selective scanning (MVSS) module and the SwiGLU module. We attempted to integrate this module into the multimodal connector of the ML-Mamba multimodal learning framework.

Specifically, we studied three variants of multimodal connectors:

- MLP: a three-layer Multi-Layer Perceptron (MLP) (see Fig. [2\(](#page-4-1)a)) that aligns the features of vision and text.
- MSC-MLP (Basic): It combines the multimodal connector called the Mamba-2 Scan Connector (MSC) module, which does not include the SwiGLU module and is intended to enhance the processing capability of two-dimensional non-causal visual information. Subsequently, the MLP aligns the features of vision and text (see Fig. [2\(](#page-4-1)b))).
- MSC-MLP (Advanced): This variant combines the MSC module and MLP, where the MSC module includes the SwiGLU (see Fig. [5\)](#page-6-1) module for more complex feature extraction and pattern learning (see Fig. [2\(](#page-4-1)c)).

The MSC module bridges the gap between 1D sequential processing capability (typical of SSM) and 2D non causal visual information by introducing two 2D scanning mechanisms. These scanning mechanisms include:

- Bidirectional-Scan Mechanism (BSM): Scanning the complementary features of the image in both forward and backward directions to capture a broader context without increasing computational complexity (shown at the top of Fig. [3\)](#page-5-0).The corresponding model structure is depicted in Fig. [4\(](#page-6-0)b).
- Cross-Scan Mechanism (CSM): unfolds image patch features into sequences along rows and columns and scans them in four directions (diagonally across the image) (shown at the bottom of Fig. [3\)](#page-5-0). The corresponding model structure is depicted in Fig. [4\(](#page-6-0)c).

After scanning, these feature sequences are processed by the Mamba-2 layer and reshaped into the patch order of the original image, and finally merged into a comprehensive representation for subsequent multimodal learning tasks. The goal of this method is to improve the modeling ability of complex visual data, especially when it involves multimodal input and nonlinear relationship modeling, to enhance the performance and robustness of computer vision tasks.

<span id="page-6-0"></span>![](_page_6_Picture_0.jpeg)

Figure 4: The comparison of block architectures between Mamba-2 block, and Mamba-2 Scan Connector (BSM, With SwiGLU) and Mamba-2 Scan Connector (CSM, With SwiGLU).

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 5: SwiGLU structure in MSC-MLP (Advanced).

As shown in Fig. [2\(](#page-4-1)a), the input of the multimodal connector is the sequential image patch features Vimg extracted from the input images via the transformer-based vision encoder. These feature vectors are then passed into a three-layer Mult-Layer (MLP):

$$V_{out} = \mathbf{MLP}(V_{img}). \tag{6}$$

As shown in Fig. [2\(](#page-4-1)b), the input of the multimodal connector is the sequential image patch features Vimg extracted from the input images via the transformer-based vision encoder. These feature vectors are then passed through a Mamba-2 Scan Connector (MSC) module to obtain the visual scanned feature Vscan. After the MSC module, the output vectors Vscan is then passed into a three-layer Mult-Layer (MLP):

$$V_{scan} = \mathbf{MSC_{Basic}}(\mathbf{V_{img}}),$$

$$V_{out} = \mathbf{MLP}(V_{scan}).$$
(7)

As shown in Fig. [2\(](#page-4-1)c), the feed-forward pass progress can be formulated as follows:

$$V_{scan} = \mathbf{MSC_{Basic}}(\mathbf{V_{img}}),$$

$$V_{scan}^{'} = \mathbf{SwiGLU}(\mathbf{V_{scan}}),$$

$$V_{out} = \mathbf{MLP}(V_{scan}^{'}).$$
(8)

#### 3.2.4 Mamba-2 Large Language Model

The Mamba-2 language model [\[7\]](#page-12-8) serves as the primary language processing component responsible for understanding and generating text. The workflow design of the visual encoder and multimodal connector ensures that visual information can be effectively transmitted to the Mamba-2 language model, enabling the model to process and understand complex multimodal data.

$$R = f_L(V_{out}, f_T(Q)). (9)$$

## 3.2.5 Training Process

We first use a 558K subset of the LAION-CC-SBU dataset to align the Mamba-2 Scan Connector (MSC) and the MLP projector. During the fine-tuning stage, we simultaneously optimized the Mamba-2 Scan Connector (MSC), the projector, and the Mamba LLM. This comprehensive training effort was executed on 8 NVIDIA A100 GPUs. The fine-tuning was conducted over two epochs, randomly sampling from the Mixed Dataset Used in LLaVA v1.5, which includes a total of 665K visual multi-round dialogue samples and pure text dialogue data.

# 4 Experiment

We conducted a comprehensive experimental evaluation of ML-Mamba through four aspects: benchmarking evaluation: We used six commonly used visual language model (VLM) benchmarks to evaluate the effectiveness of the proposed method. These benchmarks include four open-ended visual question answering tasks that require different reasoning abilities, as well as two closed set prediction tasks that involve determining spatial relationships of objects and detecting visual illusions.

- Efficiency evaluation: We conducted a comparative evaluation of ML-Mamba and other Transformer based models at similar model sizes to validate our model's improvement in efficiency.
- Ablation study: We further explored some design choices in the model structure through ablation studies to determine which components have a significant impact on model performance.
- Comparison of answer generation quality: We have provided specific examples to demonstrate the comparison of our model with other models in terms of answer generation quality. Through these experiments, we comprehensively evaluated the performance and advantages of ML-Mamba.

#### 4.1 Experimental Setup

Table [1](#page-8-0) details the hyperparameters of the ML-Mamba model. For the visual encoder part, DINOv2 adopts the same ViT structure as in its original paper, namely a ViT-Large model with 304M parameters, pretrained on the LVD-142M dataset. SigLIP uses a slightly larger shape-optimized version than ViT-Large. The resolution of the input images is set to 384x384, with the number of visual tokens being 729.

The backbone of the LLM is initialized using the pretrained weights from the Mamba-2 model, while the multimodal connectors (MSC) and projectors are always randomly initialized. We chose an open-source model weight from the Huggingface platform to initialize our model as the LLM backbone for our proposed model.

The entire training process took approximately 31 hours on 8 NVIDIA A100 80GB GPUs. During training, we used Pytorch's fully shared data parallel framework [\[53\]](#page-14-14) and adopted automatic mixed precision with FP32 and BF16 for distributed training. The batch size was set to 64. We used the AdamW [\[37\]](#page-13-14) optimizer and updated the network parameters using a learning rate with cosine decay. The learning rate was set to 2 × 10<sup>−</sup><sup>5</sup> , the decay factor was 0.1, and the warm-up ratio was 0.03. The model was trained for 2 epochs with supervised fine-tuning.

# 4.2 Results

In addition, we further evaluated the model on six carefully designed metrics, particularly VizWiz [\[17\]](#page-13-15) and VQAv2 [\[13\]](#page-12-9), for assessing general visual reasoning ability. VizWiz includes common sense

<span id="page-8-0"></span>Table 1: The configuration of the model and hyperparameters for training.

| Configuration                     |                  |
|-----------------------------------|------------------|
| Vision Encoder                    | DINOv2 + SigLIP  |
| LLM init                          | Mamba-2 2.7b     |
| MLP + Mamba-2 Scan Connector init | Random           |
| Image resolution                  | $384 \times 384$ |
| Alignment / Fine-Tuning Samples   | 558K / 665K      |
| Optimizer                         | AdamW            |
| LR schedule                       | Cosine decay     |
| Learning Rate                     | 2e-5             |
| Weight decay                      | 0.1              |
| Warmup ratio                      | 0.03             |
| Alignment and Fine-tuning epochs  | 1 each           |

<span id="page-8-1"></span>Table 2: **Comparison with SoTA methods on six benchmarks:** VQA-v2 [12]; GQA [20]; VQA<sup>T</sup>: TextVQA [44]; POPE [29]; VizWiz [17]; VSR [31]. PT and IT indicate the number of samples in the pretraining and instruction tuning stages, respectively.

| Method           | LLM              | PT    | IT    | VQA <sup>v2</sup> | GQA   | $VQA^T$ | POPE | VizWiz | VSR  |
|------------------|------------------|-------|-------|-------------------|-------|---------|------|--------|------|
| BLIP-2 [26]      | Vicuna-13B       | 129M  | -     | 41.0              | 41.0  | 42.5    | 85.3 | 19.6   | 50.9 |
| MiniGPT-4 [55]   | Vicuna-7B        | 5M    | 5K    | 32.2              | 32.2  | -       | -    | -      | -    |
| InstructBLIP [6] | Vicuna-7B        | 129M  | 1.2M  | _                 | 49.2  | 50.1    | _    | 34.5   | 54.3 |
| InstructBLIP [6] | Vicuna-13B       | 129M  | 1.2M  | _                 | 49.5  | 50.7    | 78.9 | 33.4   | 52.1 |
| Shikra [3]       | Vicuna-13B       | 600K  | 5.5M  | 77.4              | _     | _       | _    | _      | _    |
| IDEFICS-9B [25]  | LLaMA-7B         | 353M  | 1M    | 50.9              | 38.4  | 25.9    | _    | 35.5   | -    |
| IDEFICS-80B [25] | LLaMA-65B        | 353M  | 1M    | 60.0              | 45.2  | 30.9    | _    | 36.0   | -    |
| Qwen-VL [1]      | Qwen-7B          | 1.4B  | 50M   | 78.8              | 59.3  | 63.8    | _    | 35.2   | -    |
| Qwen-VL-Chat [1] | Qwen-7B          | 1.4B  | 50M   | 78.2              | 57.5  | 61.5    | _    | _      | _    |
| LLaVA-1.5 [33]   | Vicuna-7B        | 558K  | 665K  | 78.5              | 62.0  | 58.2    | 85.9 | 50.0   | -    |
| LLaVA-1.5 [33]   | Vicuna-13B       | 558K  | 665K  | 80.0              | 63.3  | 61.3    | 85.9 | -      | -    |
| TinyLLaVA [54]   | Phi2-2.7B        | 1804K | 1330K | 79.9              | 62.0  | -       | 86.4 | -      | -    |
| LLaVA-Phi [57]   | Phi-2-2.7B       | 558K  | 665K  | 71.4              | -     | 48.6    | 85.0 | 35.9   | -    |
| MobileVLM-3B [5] | MobileLLaMA-2.7B | 558K  | 665K  | -                 | 59.0  | 47.5    | 84.9 | -      | -    |
| Cobra [52]       | Mamba LLM-2.8B   | 558K  | 665K  | 75.19             | 58.7  | -       | 87.2 | -      | -    |
| VL-Mamba [42]    | Mamba LLM-2.8B   | 558K  | 665K  | 76.6              | 56.2  | 48.9    | 84.4 | -      | -    |
| ML-Mamba (ours)  | Mamba-2 LLM-2.7B | 558K  | 665K  | 75.26             | 60.68 | 52.2    | 88.3 | 45.17  | 51.5 |

questions and unanswerable questions, requiring the model to avoid incorrect answers to evaluate its reliability. GQA evaluates spatial understanding and multi-step reasoning in real-world images. The issues in TextVQA are related to the text in the image, evaluating the model's optical character recognition (OCR) and inference capabilities. POPE provides a benchmark for evaluating object hallucinations and is a binary classification task that prompts the model to answer whether the object exists. We also introduced two closed set prediction benchmarks consisting of VSR [31] and POPE [30]. VSR evaluates the model's ability to understand spatial relationships between different images, while POPE evaluates the VLM's ability to avoid severe illusion problems. VSR and POPE calculate scores based on the probability of providing the correct answer.

We evaluated VizWiz, VQAv2, and TextVQA using validation sets, while using the recommended test dev partition for GQA, zero sample test partition for VSR, and evaluation partition for POPE.

To demonstrate the model's effectiveness, we compared it with a VLM of the same scale with approximately 3B parameters, or with a larger VLM containing twice the number of parameters. As shown in Table 2, despite having only 40% of LLaVA v1.5 7B's parameters, ML-Mamba performs comparably across multiple benchmarks and even surpasses all models on POPE.

As shown in Table 2, compared with VLM with similar parameter numbers, ML-Mamba consistently achieved better performance than LLaVA Phi in VQAv2, GQA, VQA<sup>T</sup>, POPE and VizWiz. While

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 6: Examples of response generated by ML-Mamba.

VL-Mamba performs better on VQAv2, our ML-Mamba outperforms VL-Mamba on GQA, VQA<sup>T</sup> , and POPE. MobileVLM is another parallel work aimed at producing small-scale LLMs, and is therefore also introduced in experiments. In summary, these results indicate that ML-Mamba matches the performance of state-of-the-art models at the same level (∼3B) on multiple benchmarks and remains competitive when compared to larger scale models (7B and above).

We present some examples to illustrate the qualitative results of ML-Mamba. As shown in Fig. [6,](#page-9-0) ML-Mamba effectively understands the user's questions and responds accurately.

## 4.3 Reasoning speed

In order to evaluate the efficiency advantage of the ML-Mamba model, especially the speed improvement brought by its linear sequence modeling, we conducted a detailed inference speed comparison experiment. In the experiment, we compared ML-Mamba with two baseline models of the same scale parameters, TinyLaVA 3B and MobileVLM v2 3B.

All models were evaluated in the same hardware environment, namely a single Nvidia A100 PCIe 80GB GPU. Each model receives the same example image as input, with a unified image resolution of 336 × 336 pixels, and is processed by a CLIP encoder. For TinyLaVA, the model receives 576 image markers processed by the projector; MobileVLM v2 reduces the number of image labels to 144 through LDP blocks. In contrast, ML-Mamba uses dual encoders to process images with a resolution of 384 × 384, resulting in an increase in the actual number of image labels processed to 729.

Table 3: Latency comparison of small-scale VLMs with ∼3B parameters.

<span id="page-10-0"></span>

| Model        | LM               | Evalavg(tokens/s) | T otal (s) |  |
|--------------|------------------|-------------------|------------|--|
| TinyLLaVA    | Phi-2 2.7B       | 38                | 6.45       |  |
| MobileVLM v2 | MobileLLaMA 2.7B | 50                | 5.15       |  |
| ML-Mamba     | Mamba-2 2.7B     | 171               | 1.47       |  |

In the experiment, all models received the same question: "Provide a detailed description of the image." and set the number of output labels to 256. The total time is the entire process from image encoding until the complete answer is generated.And we calculated the average number of tokens generated per second by Evalavg = 256/Ttotal.

The results from Table [3](#page-10-0) demonstrated that although the number of image markers processed by ML-Mamba significantly increased, it still exhibited extremely fast inference speed. Compared to MobileVLM v2, although the latter has undergone multiple lightweight optimizations, the time required for ML-Mamba to complete inference is only about 30% of the former. This indicates that ML-Mamba not only maintains high speed while processing larger data, but also, thanks to the characteristics of its RNN like model, its memory usage does not significantly increase with the increase of image marker length, as such models maintain a fixed size hidden state to store historical information during the inference process.

The excellent performance of the ML-Mamba model in inference speed proves its advantage in linear sequence modeling, especially when dealing with a large number of image labels. Compared to Transformer based models, ML-Mamba demonstrates significant speed improvements, providing strong support for multimodal tasks that require rapid response.

#### 4.4 Ablation Study

#### 4.4.1 Effects of Language Model Variants

Table [4](#page-10-1) presents the results of ablation experiments evaluating the effectiveness of different language model variants. We conducted experiments on three different variants, namely Mamba-2 with parameters of 780m, 1.3b, and 2.7b, trained on the Pile dataset (containing 300B tokens). Specifically, we constructed a baseline model using the same variant of DINOv2+SigLIP as the visual encoder, Mamba-2 language model as the backbone of a large language model, and a regular MLP multimodal connector without a 2D visual selection scanning module. We can see that as the model size and number of training tokens increase, Mamba2-2.7B outperforms other variants on all benchmarks. Therefore, we chose Mamba2-2.7B for other experiments.

Table 4: Ablation study of the variants of the language model.

<span id="page-10-1"></span>

| Method        | VQAv2 | GQA   | VQAT | POPE | VizWiz | VSR  |
|---------------|-------|-------|------|------|--------|------|
| Mamba2 - 780m | 71.7  | 51.92 | 48.1 | 81.6 | 41.5   | 47.7 |
| Mamba2 - 1.3b | 73.6  | 55.41 | 50.8 | 83.7 | 43.7   | 49.3 |
| Mamba2 - 2.7b | 75.26 | 60.68 | 52.2 | 88.3 | 45.17  | 51.5 |

#### 4.4.2 Effects of Different Visual Encoders

Recent research has found that although language image models similar to CLIP can provide rich semantic information, they may lose detailed information about the image itself. Therefore, we further introduce DINOv2 as a supplementary encoder and connect the visual representations of these two encoders for subsequent LLM. As shown in Table [5,](#page-11-0) the introduction of DINOv2 significantly improved the model performance in six benchmark tests. This result suggests a meaningful principle when selecting a visual encoder for downstream tasks. Therefore, we ultimately chose DINOv2+SigLIP as the visual encoder to construct our model and used it for further experiments. Through this combination, we can achieve better performance on multiple benchmarks.

Table 5: Ablation study of the vision encoder.

<span id="page-11-0"></span>

| Method          | VQAv2 | GQA   | VQAT  | POPE | VizWiz | VSR   |
|-----------------|-------|-------|-------|------|--------|-------|
| DINOv2          | 73.73 | 58.84 | 51.13 | 86.6 | 44.23  | 50.73 |
| SigLIP          | 74.61 | 59.43 | 50.78 | 87.4 | 45.07  | 50.54 |
| DINOv2 + SigLIP | 75.26 | 60.68 | 52.20 | 88.3 | 45.17  | 51.50 |

#### 4.4.3 Ablation on different multimodal connector structures

We also explored the impact of different architectures of multi-mode connectors. We evaluated three different MMC variants: MLP, MSC-MLP (Basic), and MSC-MLP (Advanced). As shown in Table [6,](#page-11-1) by comparing these three architectures, we observed that MSC-MLP (Advanced) performed relatively better on most benchmark tests, especially on VQA, demonstrating the effectiveness of combining MSC modules with swiGLU. Note that these models use DINOv2+SigLIP as the visual encoder, Mamba2-2.7B as the language model, and a bidirectional selective scanning mechanism. Consequently, we ultimately chose MSC-MLP (Advanced) as our model and used it for further experiments.

Table 6: Ablation study of the different architectures of multimodal connector.

<span id="page-11-1"></span>

| Method            | VQAv2 | GQA   | VQAT  | POPE | VizWiz | VSR   |
|-------------------|-------|-------|-------|------|--------|-------|
| MLP               | 73.42 | 58.87 | 50.31 | 86.1 | 43.87  | 50.13 |
| MSC-MLP(Basic)    | 75.09 | 60.14 | 51.72 | 86.5 | 44.57  | 50.76 |
| MSC-MLP(Advanced) | 75.26 | 60.68 | 52.20 | 88.3 | 45.17  | 51.50 |

#### 4.4.4 Under different scanning mechanisms

We compared the bidirectional scanning mechanism (BSM) and cross scanning mechanism (CSM) in MMC modules. As shown in Table [7,](#page-11-2) although BSM and CSM perform similarly in some benchmark tests, such as scoring 76.6 in one test, BSM shows superior performance in most benchmark tests. This highlights its advantages in handling 2D visual information for multimodal learning tasks.

Table 7: Ablation study of the scan mechanisms.

<span id="page-11-2"></span>

| Method                             | VQAv2 | GQA   | VQAT  | POPE | VizWiz | VSR   |
|------------------------------------|-------|-------|-------|------|--------|-------|
| Bidirectional-Scan Mechanism (BSM) | 75.26 | 60.68 | 52.20 | 88.3 | 45.17  | 51.50 |
| Cross-Scan Mechanism (CSM)         | 75.14 | 60.13 | 52.31 | 88.5 | 44.89  | 51.14 |

# 5 Limitation

The training of ML-Mamba relies on specific multimodal datasets, which may have biases or incomplete coverage in certain aspects. Developing more comprehensive and diverse datasets, as well as improving data preprocessing and augmentation techniques, will help enhance the generalization ability and applicability of ML-Mamba in different scenarios.

ML-Mamba currently faces challenges in running on mobile devices, especially in meeting the memory usage requirements of these devices. In order to make ML-Mamba run more smoothly on devices such as smartphones or tablets, further optimization, especially for low memory environments, is necessary.

# 6 Conclusion

This article introduces a novel multimodal learning model, ML-Mamba, which utilizes the latest state space model (SSM) Mamba-2 to solve multimodal learning tasks. It uses a pre-trained Mamba-2 language model as the language model and introduces the multimodal connector Mamba-2 Scan Connector (MSC) module to bridge the gap between 2D non-causal image information and the inherent causal modeling ability of SSM. By conducting comprehensive experiments and ablation studies, ML-Mamba performed well in multimodal benchmark testing, demonstrating its effectiveness and the potential of SSM in multimodal learning. On the other hand, ML-Mamba addresses the efficiency bottleneck of existing multimodal large language models by using models with linear computational complexity. This significantly improves computational efficiency and excels in visual illusion and spatial relationship judgment while reducing the number of parameters. These advancements open new possibilities for deploying high-performance AI models in environments that process visual information at high frequencies.

# References

- <span id="page-12-13"></span>[1] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023.
- <span id="page-12-3"></span>[2] M. Bellagente, J. Tow, D. Mahan, D. Phung, M. Zhuravinskyi, R. Adithyan, J. Baicoianu, B. Brooks, N. Cooper, A. Datta, M. Lee, E. Mostaque, M. Pieler, N. Pinnaparju, P. Rocha, H. Saini, H. Teufel, N. Zanichelli, and C. Riquelme. Stable lm 2 1.6b technical report, 2024.
- <span id="page-12-12"></span>[3] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao. Shikra: Unleashing multimodal llm's referential dialogue magic, 2023.
- <span id="page-12-2"></span>[4] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023.
- <span id="page-12-14"></span>[5] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei, and C. Shen. Mobilevlm : A fast, strong and open vision language assistant for mobile devices. *ArXiv*, abs/2312.16886, 2023.
- <span id="page-12-11"></span>[6] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In *NeurIPS*, 2023.
- <span id="page-12-8"></span>[7] T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. *ArXiv*, abs/2405.21060, 2024.
- <span id="page-12-1"></span>[8] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. GLM: general language model pretraining with autoregressive blank infilling. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pages 320–335, 2022.
- <span id="page-12-6"></span>[9] D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, and C. Ré. Hungry hungry hippos: Towards language modeling with state space models, 2023.
- <span id="page-12-0"></span>[10] P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou, et al. Llama-adapter v2: Parameter-efficient visual instruction model, 2023.
- <span id="page-12-5"></span>[11] K. Goel, A. Gu, C. Donahue, and C. Ré. It's raw! audio generation with state-space models, 2022.
- <span id="page-12-10"></span>[12] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In *CVPR*, 2017.
- <span id="page-12-9"></span>[13] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering, 2017.
- <span id="page-12-7"></span>[14] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023.
- <span id="page-12-4"></span>[15] A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with structured state spaces. *arXiv preprint arXiv:2111.00396*, 2021.

- <span id="page-13-5"></span>[16] S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes, A. D. Giorno, S. Gopi, M. Javaheripi, P. Kauffmann, G. de Rosa, O. Saarikivi, A. Salim, S. Shah, H. S. Behl, X. Wang, S. Bubeck, R. Eldan, A. T. Kalai, Y. T. Lee, and Y. Li. Textbooks are all you need, 2023.
- <span id="page-13-15"></span>[17] D. Gurari, Q. Li, A. J. Stangl, A. Guo, C. Lin, K. Grauman, et al. Vizwiz grand challenge: Answering visual questions from blind people, 2018.
- <span id="page-13-0"></span>[18] L. He, Z. Li, X. Cai, and P. Wang. Multi-modal latent space learning for chain-of-thought reasoning in language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 18180–18187, 2024.
- <span id="page-13-13"></span>[19] T. Huang, X. Pei, S. You, F. Wang, C. Qian, and C. Xu. Localmamba: Visual state space model with windowed selective scan. *arXiv preprint arXiv:2403.09338*, 2024.
- <span id="page-13-16"></span>[20] D. A. Hudson and C. D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019.
- <span id="page-13-3"></span>[21] F. Jia, K. Wang, Y. Zheng, D. Cao, and Y. Liu. Gpt4mts: Prompt-based large language model for multimodal time-series forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 23343–23351, 2024.
- <span id="page-13-8"></span>[22] X. Jiang, H. Tang, J. Gao, X. Du, S. He, and Z. Li. Delving into multimodal prompting for fine-grained visual classification. In *Proceedings of the AAAI conference on artificial intelligence*, volume 38, pages 2570–2578, 2024.
- <span id="page-13-9"></span>[23] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024.
- <span id="page-13-4"></span>[24] Y.-J. Kim, M.-J. Kim, K. An, J. Ahn, J. Kim, Y.-J. Heo, D.-S. Chang, and E.-S. Kim. Structure-aware multimodal sequential learning for visual dialog. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 13193–13201, 2024.
- <span id="page-13-20"></span>[25] H. Laurençon, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-13-19"></span>[26] J. Li, D. Li, S. Savarese, and S. C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *ICML*, 2023.
- <span id="page-13-12"></span>[27] S. Li, H. Singh, and A. Grover. Mamba-nd: Selective state space modeling for multi-dimensional data. *arXiv preprint arXiv:2402.05892*, 2024.
- <span id="page-13-6"></span>[28] Y. Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y. T. Lee. Textbooks are all you need ii: phi-1.5 technical report. *arXiv preprint arXiv:2309.05463*, 2023.
- <span id="page-13-17"></span>[29] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J. rong Wen. Evaluating object hallucination in large vision-language models. 2023.
- <span id="page-13-21"></span>[30] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models, 2023.
- <span id="page-13-18"></span>[31] F. Liu, G. Emerson, and N. Collier. Visual spatial reasoning, 2023.
- <span id="page-13-2"></span>[32] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning, 2023.
- <span id="page-13-1"></span>[33] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning, 2023.
- <span id="page-13-10"></span>[34] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, and Y. Liu. Vmamba: Visual state space model. *ArXiv*, abs/2401.10166, 2024.
- <span id="page-13-11"></span>[35] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *ICCV*, 2021.
- <span id="page-13-7"></span>[36] X. Long, J. Zeng, F. Meng, Z. Ma, K. Zhang, B. Zhou, and J. Zhou. Generative multi-modal knowledge retrieval with large language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 18733–18741, 2024.
- <span id="page-13-14"></span>[37] I. Loshchilov and F. Hutter. Decoupled weight decay regularization, 2019.

- <span id="page-14-0"></span>[38] OpenAI, :, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, et al. Gpt-4 technical report, 2024.
- <span id="page-14-10"></span>[39] A. V. Oppenheim, A. S. Willsky, S. H. Nawab, and J.-J. Ding. *Signals and systems*. Prentice hall Upper Saddle River, NJ, 1997.
- <span id="page-14-11"></span>[40] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, et al. Dinov2: Learning robust visual features without supervision, 2024.
- <span id="page-14-8"></span>[41] X. Pei, T. Huang, and C. Xu. Efficientvmamba: Atrous selective scan for light weight visual mamba. *arXiv preprint arXiv:2403.09977*, 2024.
- <span id="page-14-9"></span>[42] Y. Qiao, Z. Yu, L. Guo, S. Chen, Z. Zhao, M. Sun, Q. Wu, and J. Liu. Vl-mamba: Exploring state space models for multimodal learning, 2024.
- <span id="page-14-6"></span>[43] S. Ren, X. Li, H. Tu, F. Wang, F. Shu, L. Zhang, J. Mei, L. Yang, P. Wang, H. Wang, A. Yuille, and C. Xie. Autoregressive pretraining with mamba in vision, 2024.
- <span id="page-14-15"></span>[44] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. *CVPR*, pages 8309–8318, 2019.
- <span id="page-14-2"></span>[45] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/stanford\\_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- <span id="page-14-13"></span>[46] S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, and S. Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024.
- <span id="page-14-1"></span>[47] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models, 2023.
- <span id="page-14-4"></span>[48] L. Wang, Y. Hu, J. He, X. Xu, N. Liu, H. Liu, and H. T. Shen. T-sciq: Teaching multimodal chain-ofthought reasoning via large language model signals for science question answering. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 19162–19170, 2024.
- <span id="page-14-5"></span>[49] F. Ye, G. Liu, X. Wu, and L. Wu. Altdiffusion: A multilingual text-to-image diffusion model. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 6648–6656, 2024.
- <span id="page-14-12"></span>[50] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training, 2023.
- <span id="page-14-3"></span>[51] P. Zhang, G. Zeng, T. Wang, and W. Lu. Tinyllama: An open-source small language model, 2024.
- <span id="page-14-19"></span>[52] H. Zhao, M. Zhang, W. Zhao, P. Ding, S. Huang, and D. Wang. Cobra: Extending mamba to multi-modal large language model for efficient inference, 2024.
- <span id="page-14-14"></span>[53] Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, et al. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023.
- <span id="page-14-17"></span>[54] B. Zhou, Y. Hu, X. Weng, J. Jia, J. Luo, X. Liu, et al. Tinyllava: A framework of small-scale large multimodal models, 2024.
- <span id="page-14-16"></span>[55] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. *arXiv preprint arXiv:2304.10592*, 2023.
- <span id="page-14-7"></span>[56] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. *ArXiv*, abs/2401.09417, 2024.
- <span id="page-14-18"></span>[57] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang. Llava-phi: Efficient multi-modal assistant with small language model, 2024.