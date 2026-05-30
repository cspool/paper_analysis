# VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models

Haowen Hou\* and Peigen Zeng<sup>+</sup> and Fei Ma\* and Fei Richard Yu+†

\*Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Shenzhen, China <sup>+</sup>College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China †Shool of Information Technology, Carleton University, Canada {houhaowen, mafei, yufei}@gml.ac.cn \*

# Abstract

Visual Language Models (VLMs) have rapidly progressed with the recent success of large language models. However, there have been few attempts to incorporate efficient linear Recurrent Neural Networks (RNNs) architectures into VLMs. In this study, we introduce VisualRWKV, the first application of a linear RNN model to multimodal learning tasks, leveraging the pre-trained RWKV language model. We propose a data-dependent recurrence and sandwich prompts to enhance our modeling capabilities, along with a 2D image scanning mechanism to enrich the processing of visual sequences. Extensive experiments demonstrate that VisualRWKV achieves competitive performance compared to Transformer-based models like LLaVA-1.5 on various benchmarks. Compared to LLaVA-1.5, VisualRWKV has a speed advantage of 3.98 times and can save 54% of GPU memory when reaching an inference length of 24K tokens. To facilitate further research and analysis, we have made the checkpoints and the associated code publicly accessible at the following GitHub repository: [https://github.com/howard-hou/VisualRWKV.](https://github.com/howard-hou/VisualRWKV)

# 1 Introduction

Large Language Models (LLMs) have demonstrated exceptional performance in natural language processing tasks [\(Touvron et al.,](#page-11-0) [2023b;](#page-11-0) [Brown et al.,](#page-8-0) [2020\)](#page-8-0). Extending LLMs to support visual inputs has garnered significant attention in the research community [\(OpenAI,](#page-10-0) [2023\)](#page-10-0). Visual Language Models (VLMs) inherit powerful capabilities from LLMs, such as strong instruction following, zero-shot generalization, and in-context learning [\(Liu et al.,](#page-10-1) [2023b;](#page-10-1) [Zhu et al.,](#page-11-1) [2024a\)](#page-11-1). By integrating

visual and textual information, VLMs not only enhance the understanding of visual content but also provide richer context for language understanding and generation. VLMs hold tremendous potential for solving visual problems and advancing various vision-language tasks.

However, despite the excellent performance of existing LLMs and VLMs, their inherent computational and memory complexity due to the selfattention mechanism in the Transformer architecture results in quadratic growth in computation and memory requirements with the increase in sequence length [\(Katharopoulos et al.,](#page-9-0) [2020\)](#page-9-0). This leads to high inference costs and limits the deployment and application of Transformer-based VLMs on edge devices.

The Receptance Weighted Key Value (RWKV) model, a novel Recurrent Neural Network (RNN) architecture, presents a promising solution to the bottleneck of long-sequence modeling [\(Peng et al.,](#page-10-2) [2023a\)](#page-10-2). It surpasses Transformers in large-scale data performance and exhibits linear scalability with sequence length, positioning itself as a promising successor to Transformers in language modeling [\(Peng et al.,](#page-10-3) [2023b\)](#page-10-3).

Currently, there is a notable gap in research exploring how this efficient architecture can be leveraged for multimodal tasks. In this study, we introduce the VisualRWKV model, marking the first application of a linear RNN model to multimodal learning tasks. Specifically, we utilize the pre-trained RWKV language model as the foundational language model and explore several novel mechanisms applied to VisualRWKV.

VisualRWKV introduces: (1) an innovative datadependent recurrence to enhance the capabilities and capacity of the RWKV model. (2) a novel sandwich prompt designed to provide richer conditions when processing visual sequences. (3) a new 2D image scanning mechanism to enhance the 2D modeling capabilities of visual sequences.

<sup>\*</sup>This work is supported in part by Shenzhen Science and Technology Program under Grant ZDSYS20220527171400002, the National Natural Science Foundation of China (NSFC) under Grants 62406197, 62271324, 62231020 and 62371309. Corresponding author: F. Richard Yu.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: **VisualRWKV** outperforms the SoTA LLaVA-1.5 (Liu et al., 2023a) on 4 tasks (a), with high computational efficiency (b) and low, stable memory usage (c).

(c) GPU Memory Comparison

Extensive experiments on various multimodal learning benchmarks validate the effectiveness of VisualRWKV, as shown in Figure 1. Compared to other Transformer-based models of similar size, such as LLaVA-1.5 (Liu et al., 2023a), Visual-RWKV demonstrates competitive performance, achieving outstanding results on multiple popular benchmarks.

In summary, this study presents the VisualR-WKV model, explores the impact of various novel designs on VisualRWKV, introduces the innovative sandwich prompt to enhance representation capabilities, and conducts extensive experiments across diverse multimodal learning benchmarks.

#### 2 Related Works

#### 2.1 Visual Language Models

Following the success of LLMs, recent research has pivoted towards VLMs (Achiam et al., 2023; Team et al., 2023) for enhancing visual understanding and reasoning capabilities. Expanding on various pre-trained LLM architectures, researchers have proposed diverse methodologies for incorporating visual information. Flamingo (Alayrac et al., 2022) and BLIP-2 (Li et al., 2023c) introduce distinct techniques for modality fusion, integrating visual tokens with frozen large language models through gated attention or query transformers. Building on the effectiveness of instruction tuning, LLaVA (Liu et al., 2023b,a) and MiniGPT-4 (Zhu et al., 2024a; Chen et al., 2023a) utilize visual instruction tuning to align visual input with LLMs, showcasing notable achievements. Recent advancements, such as Kosmos-2 (Peng et al., 2023c) and Shikra (Chen et al., 2023b), further enhance VLMs with grounded visual understanding capabilities. Despite their promising potential for general-purpose visual reasoning and planning tasks, these models are generally expensive and challenging to train and deploy.

#### 2.2 Linear RNN Large Language Model

Recent advancements in LLMs, such as GPT (Radford et al., 2019; Brown et al., 2020; Achiam et al., 2023), LLaMA (Touvron et al., 2023a,b), and PaLM (Anil et al., 2023; Chowdhery et al., 2023), have showcased remarkable prowess across various natural language processing tasks. However, traditional Transformer-based LLMs suffer from quadratic complexity  $O(L^2)$  issues in both computation and memory, prompting the emergence of linear RNNs as potential successors.

RNNs model sequential data with temporal dependencies by generating a hidden state  $h_t$  at each time step, which is then utilized as input for the subsequent step. Classical RNN variants like LSTM (Hochreiter and Schmidhuber, 1997) and GRU (Cho et al., 2014) excel in inexpensive inference, operating typically at O(1) time complexity per step relative to sequence length. Nonetheless, their older designs often pose challenges in parallelization across time dimensions during training.

Linear RNNs present themselves as promising successors to the Transformer, offering a more efficient token mixing method. They enable a space complexity of O(L) and an inference complexity

of O(1). Leveraging Parallel Prefix Sum Scan (Harris et al., 2007) for acceleration can further enhance their efficiency. The RWKV (Peng et al., 2023b; Hou and Yu, 2024), a linear RNN-based LLM, has showcased competitive performance compared to GPT models of similar scale. RWKV introduces temporal decay to gradually reduce the influence of past information, implicitly incorporating positional information. Additionally, it integrates a token-shift mechanism facilitating linear interpolation between current and previous inputs. This allows the model to naturally aggregate and regulate information within input channels. Furthermore, RWKV boasts a time complexity of O(L) and an inference complexity of O(1), ensuring consistent inference time per token. As a result, the overall inference duration scales linearly with sequence length. The memory footprint of RWKV remains constant, regardless of sequence length, contributing to its efficiency and scalability.

#### 3 Methods

In this section, we initially introduce the fundamental concepts of the RWKV language model. (Section 3.1). Following that, we elaborate on the transformation of the RWKV language model into our proposed VisualRWKV visual language model (Section 3.2), which mainly includes data-dependent recurrence, sandwich prompting, and image scanning.

#### <span id="page-2-0"></span>3.1 Preliminaries

The RWKV(Peng et al., 2024) backbone is structured using stacked residual blocks, with each block containing a time-mixing and a channel-mixing sub-block. These components embody recurrent structures designed to leverage past information.

**Data-independent Token Shift** As shown in Figure 3, trainable variable  $\mu_g$ ,  $\mu_r$ ,  $\mu_k$ ,  $\mu_v$  are used in a linear combination of  $x_t$  and  $x_{t-1}$ , to achieve a simple mixing, which interpolate between the inputs of the current and previous time-steps. The combination of shifted previous step and current step was linear projected through projection matrix within the block:

$$\alpha_t = (\mu_\alpha \odot x_t + (1 - \mu_\alpha) \odot x_{t-1}) W_\alpha \tag{1}$$

where  $\alpha$  serves as a notation for the variables r, g, k, and v, given that they are subject to an identical linear combination formula. Please note that the linear combination used here is data independent,

meaning the value of  $\mu_{\alpha}$  is not dependent on  $x_t$  or  $x_{t-1}$ .

**Data-independent Time Mixing** In vanilla RWKV, the time mixing is articulated through the update of the WKV vectors and the WKV operator is input-data independent. The formula of single head WKV operator is given by:

<span id="page-2-3"></span>
$$wkv_t = \operatorname{diag}(u) \cdot k_t^{\mathrm{T}} \cdot v_t + \sum_{i=1}^{t-1} \operatorname{diag}(w)^{t-1-i} \cdot k_i^{\mathrm{T}} \cdot v_i \tag{2}$$

where w and u are two trainable parameters. The parameter u serves as a term weight for the current token when the model encounters it for the first time. It enables the model to efficiently process the token by focusing more on important tokens and quickly filtering out unimportant ones. Another important parameter is w, which is a channel-wise time decay vector per head. Furthermore, we transform parameter w by  $w = \exp(-\exp(w))$ . This transformation ensures that all values of w are within the range (0,1), ensuring that  $\operatorname{diag}(w)$  represents a contraction matrix.

The output from the single-head WKV operator undergoes processing by the layer normalization and the SiLU activation. Then, all outputs are concatenated to form the output vector  $o_t$ :

$$o_t = concat(SiLU(g_t) \odot LayerNorm(r_t \cdot wkv_t))W_o$$
 (3)

where LayerNorm operates on each head separately. For further details and formulas of the models, one can refer to Peng et al. (2024) and Hou and Yu (2024).

### <span id="page-2-1"></span>3.2 VisualRWKV

<span id="page-2-2"></span>

| Method                 | Size | VQA   | SQA   | TQA   | GQA   |
|------------------------|------|-------|-------|-------|-------|
| VisualRWKV-Base        | 1.6B | 51.08 | 41.94 | 35.19 | 48.09 |
| +Data-dep Recurrence   | 1.6B | 65.82 | 46.55 | 40.26 | 49.06 |
| +Bidirection +Sandwich | 1.6B | 64.96 | 56.72 | 41.94 | 48.04 |
| +Better Learning Rate  | 1.6B | 69.42 | 59.05 | 43.57 | 55.23 |
| +Scale up to 3B        | 3B   | 71.52 | 65.34 | 48.68 | 59.56 |
| +Scale up to 7B        | 7B   | 75.82 | 68.22 | 51.01 | 64.27 |
|                        |      |       |       |       |       |

Table 1: **Scaling results** on model. We choose to conduct experiments on VQA-v2(VQA), ScienceQA(SQA), TextVQA(TQA) and GQA to examine model's capabilities.

#### 3.2.1 VisualRWKV Baseline

VisualRWKV is a follow-up work to RWKV. RWKV paper (Peng et al., 2024) proposed a simplified version of VisualRWKV that employed data-independent recurrence (Fig. 3), unidirection image scanning (Fig. 4), and image first prompting

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: VisualRWKV architecture overview and three prompting method. Image First Prompt: place image tokens before instruction tokens; Image Last Prompt: place image tokens after instruction tokens; Sandwich Prompt: place image tokens in the middle of instruction tokens. Red words indicate the key contributions.

(Fig. [2\)](#page-3-0). We used that version of VisualRWKV as the baseline and starting point for our research, as shown in Table [1.](#page-2-2) We denote this initial model without any modifications as VisualRWKV-Base.

### <span id="page-3-1"></span>3.2.2 Data-dependent Recurrence

The Data-dependent Recurrence mechanism introduces two key enhancements: the Data-dependent Token Shift and the Data-dependent Time Mixing.

Data-dependent Token Shift First, we define low-rank adaptation (lora) and data-dependent linear interpolation (ddlerp) as follow:

$$lora_{\alpha}(x) = \lambda_{\alpha} + \tanh(xA_{\alpha})B_{\alpha} \tag{4}$$

$$ddlerp_{\alpha}(a,b) = a + (b-a) \odot lora_{\alpha}(a + (b-a) \odot \mu_x)$$
 (5)

Then, the Data-dependent Token Shift is defined as:

$$\alpha_t = \mathrm{ddlerp}_{\alpha}(x_t, x_{t-1})W_{\alpha} \tag{6}$$

where α serves as a notation for the variables r, g, k, and v. Aα, Bα, λ<sup>α</sup> and W<sup>α</sup> are trainable parameters. The data-dependent token shift seeks to broaden the model's capacity. It dynamically allocates the ratio of new to existing data per channel, depends on the input at both current and previous time steps.

Data-dependent Time Mixing The key improvement over data-independent time mixing (Eq. [2\)](#page-2-3) lies in the evolution of the time decay vector from a fixed parameter w to a dynamic one w<sup>t</sup> that reacts to the input data x<sup>t</sup> at time step t. The dynamic nature of w<sup>t</sup> allows the model to adjust more nimbly

to diverse input data, unbound by rigid, predefined structures. Equations are as follow:

$$d_t = \operatorname{lora}_d(\operatorname{ddlerp}_d(x_t, x_{t-1})) \tag{7}$$

$$w_t = \exp(-\exp(d_t)) \tag{8}$$

$$wkv_t = \operatorname{diag}(u) \cdot k_t^{\mathrm{T}} \cdot v_t + \sum_{i=1}^{t-1} \operatorname{diag}\left(\bigcap_{j=1}^{i-1} w_j\right) \cdot k_i^{\mathrm{T}} \cdot v_i \tag{9}$$

The LoRA mechanism utilizes vectors learned from data-independent time mixing and enhances them at a low cost with additional offsets modulated by the incoming input. It should be noted that the computation of the new time-varying decay w<sup>t</sup> employs a token-shifted value ddlerp<sup>d</sup> (x<sup>t</sup> , xt−1) as its input, not just the current token x<sup>t</sup> . As shown in Table [1,](#page-2-2) the VisualRWKV equipped with data-dependent recurrence exhibits a significant improvement in performance.

### 3.2.3 Sandwich Prompt

The motivation for designing the sandwich prompt is as follows: Unlike the attention mechanism in Transformers, RNN models such as RWKV, due to their sequential nature, cannot revisit historical information repeatedly. Instead, they must decide immediately whether to store a token or image token in memory upon encountering it. Therefore, carefully designing tailored prompts is essential for enhancing VisualRWKV's ability to effectively acquire and utilize information. For this purpose, we have specifically designed three types of prompting methods, as shown in Figure [2:](#page-3-0)

• Image First Prompt: Place image tokens prior to the instruction tokens.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3: Data-dependent recurrence. Top: Semantic diagram of the time-mixing block; Bottom: Time-mixing block as an RNN cell. Dashed arrows represent connections in data-dependent recurrence, not present in data-independent recurrence.

- Image Last Prompt: Place image tokens following the instruction tokens.
- Sandwich Prompt: Insert image tokens between the instruction tokens.

The sandwich prompt is designed to provide optimal conditions that assist the model in making these decisions more effectively. Specifically, the first prompt helps the model efficiently extract relevant information from the image, while the second prompt focuses on improving the model's ability to answer questions.

For instance, the Image Last Prompt can cause the model to occasionally forget the question embedded in the prompt, while the Image First Prompt may result in the model processing the image without considering the question, hindering its ability to analyze the image contextually. In contrast, the sandwich prompt resolves these issues and achieves a synergistic effect, enabling the model to perform better than the sum of the individual prompts. The experimental results show that the Sandwich

Prompt achieves the best performance, as presented in Table [3.](#page-7-0)

### 3.2.4 Image Scanning

The motivation for designing the image scanning techniques is as follows: Language is inherently unidirectional, while images are multidirectional by nature. As a result, unidirectional language models face inherent limitations when processing visual information. By implementing bidirectional or multidirectional image scanning strategies, these challenges can be effectively mitigated.

Vanilla RWKV is designed for 1D sequential data with causal relationships, such as language sequences. However, the visual sequences generated by vision encoders are non-causal. To bridge this gap, we propose a 2D scanning mechanism to improve VisualRWKV's performance on visual tasks. This work integrates the 2D scanning mechanism into RWKV blocks, exploring three variants of multimodal RWKV blocks, which are illustrated in Figure [4:](#page-5-0)

- Unidirectional Blocks: Only containing the Forward Scanning Block, which is the basic scanning pattern of RWKV and other linear RNN models. This serves as the Base.
- Bidirectional Blocks: Comprising both Forward Scanning and Backward Scanning Blocks, arranged in an alternating fashion.
- Multidirectional Blocks: Including blocks for Forward Scanning, Backward Scanning, Upward Scanning, and Downward Scanning, with the sequence of Forward, Backward, Upward, and Downward arranged in an alternating order.

Our design alternates different scanning directions within layers, which does not introduce additional computational overhead and preserves the efficiency of the architecture. The experimental results (Table [4\)](#page-7-1) have also verified the effectiveness and necessity of such scanning techniques in enhancing the model's ability to handle and understand visual sequences, thereby improving the overall performance of VisualRWKV in visual language processing tasks.

# 4 Experiments

The following section is dedicated to showcasing the key experiments and outcomes related to Visu-

<span id="page-5-0"></span>![](_page_5_Picture_0.jpeg)

Figure 4: Illustration of 3 different multimodal RWKV Blocks: Unidirectional Blocks (left), Bidirectional Blocks (middle), and Multidirectional Blocks (right). The four scanning modes are also depicted at the top.

alRWKV. All results presented in this section are derived from a single run.

### 4.1 Experiment Setup

Following [Liu et al.](#page-9-1) [\(2023a,](#page-9-1)[b\)](#page-10-1), the training process of VisualRWKV consists of two stages: vision-andlanguage alignment pretraining and visual instruction tuning. In the pretraining stage, the vision encoder and RWKV LLM are frozen, with only the projector being updated. During the visual instruction tuning stage, we finetune both the projector and the RWKV LLM, as shown in Figure [2.](#page-3-0) Details of training data and hyper-parameters can be found in Appendix [A.](#page-12-0)

# 4.2 Benchmarks

We evaluated VisualRWKV across 8 benchmark tests tailored to assess the model's performance in academic tasks.

For assessing visual perception capabilities, VQA-v2 [\(Goyal et al.,](#page-9-9) [2017\)](#page-9-9) and GQA [\(Hudson](#page-9-10) [and Manning,](#page-9-10) [2019\)](#page-9-10) presented open-ended short questions. Following the methodology outlined in LLaVA [\(Li et al.,](#page-9-11) [2023b\)](#page-9-11), we utilized the image subset of ScienceQA [\(Lu et al.,](#page-10-9) [2022\)](#page-10-9) to gauge the model's zero-shot generalization in answering scientific questions via multiple-choice questions. TextVQA [\(Singh et al.,](#page-10-10) [2019\)](#page-10-10) focused on visual question answering with rich text content.

Regarding benchmarks tailored for VLMs, var-

ious assessments evaluated the model's performance across diverse domains and applications, encompassing different response formats. MME-Perception [\(Fu et al.,](#page-9-12) [2023\)](#page-9-12) scrutinized the model's visual perception abilities through true/false questions. MMBench [\(Liu et al.,](#page-10-11) [2023c\)](#page-10-11) assessed the robustness of the model's answers by rigorously shuffling multiple-choice options. MMBench-CN, the Chinese counterpart of MMBench, was employed to evaluate the model's multilingual capabilities. POPE [\(Li et al.,](#page-9-13) [2023d\)](#page-9-13) assesses the model's hallucination degree on three sampled subsets of COCO [\(Lin et al.,](#page-9-14) [2014\)](#page-9-14): random, common, and adversarial, reporting the average F1 score across all three splits.

### 4.3 Quantitative Evaluation

### 4.3.1 Main Results

Table [2](#page-7-2) presents a comparison of our proposed VisualRWKV model with some state-of-the-art (SOTA) multimodal large language models. VisualRWKV achieved the best performance in 3 out of 8 benchmarks and came in second place in SQA benchmark. Compared with LLaVA-1.5 7B, which has similar scale parameters and the same amount of multimodal training data, Our model(VisualRWKV-7B) outperformed it in 4 benchmarks: SQA (68.2 vs. 66.8), GQA (64.3 vs. 62.0), MMB (65.8 vs. 64.3), and MMB-cn (63.7 vs. 30.5). It is noteworthy that VisualRWKV and

LLaVA-1.5 used completely identical training data. Yet, on the MMB-cn Chinese test set, VisualR-WKV showed a substantial lead. This may indicate that the RWKV language model has stronger multilingual capabilities. These promising results not only confirm the effectiveness of the VisualRWKV model, but also highlight the significant potential of the Linear RNN model in multimodal learning tasks.

### 4.3.2 Gain Analysis on Different Benchmarks

VisualRWKV excels in academic benchmarks like VQA, GQA, and SQA, where both the questions and answers are short texts. The model faces no fundamental obstacles in handling such tasks, leading to significant performance improvements. As a result, VisualRWKV achieves results that are comparable to, and even surpass, the Transformerbased LLaVA-1.5 on these benchmarks.

Although VisualRWKV shows notable improvements on the TextVQA (TQA) benchmark, it still lags behind LLaVA-1.5 in this task (51.0 vs. 58.2). TextVQA requires recalling information from images, which is similar to the Multi-Query Associative Recall (MQAR) task [\(Arora et al.,](#page-8-5) [2023\)](#page-8-5), which is often a limitation for RNN-like architectures. However, our latest work, VisualRWKV-HD/UHD [\(Li and Hou,](#page-9-15) [2024\)](#page-9-15), has shown that higher resolution and better-quality image features can significantly alleviate these limitations.

### 4.4 Ablation Study

### 4.4.1 Ablation on Data-dependent Recurrence

To verify the effectiveness of data-dependent recurrence described in Section [3.2.2,](#page-3-1) we conducted a rigorous ablation study, ensuring that the model size, training data, environment, and all hyperparameters were strictly consistent. As depicted in Table [1,](#page-2-2) the outcomes demonstrate significant enhancements in the data-dependent VisualRWKV across the four monitored benchmarks, affirming that data-dependence is essential for the success of linear RNN-type models in the VLM domain.

### 4.4.2 Ablation on Prompting Method

As shown in Table [3,](#page-7-0) among the three prompting approaches, the sandwich prompt outperforms the others, followed by the image-first prompt, with the image-last prompt being the least effective. The effectiveness of the sandwich prompt is attributed to its ability to allow the model to review the instructions before engaging with the image, enabling a

more targeted extraction of information and enhancing the conditional aspects of image information retrieval.

However, simply placing the instructions before the image is insufficient. The image-last prompt performs poorly because linear RNN models tend to forget the instructions after processing the image, making it necessary to repeat the instructions for better results. Additionally, our research shows that the sandwich prompt can effectively mitigate information loss even with a reduced number of image tokens, maintaining robust performance. Further experimental results and analyses can be found in Appendix [E.](#page-15-0)

# 4.4.3 Ablation on Scanning Method

We compared three image scanning mechanisms: Uni-directional scanning (UniDir), Bi-directional scanning (BiDir), and Multi-directional scanning (MultiDir). As shown in Table [4,](#page-7-1) UniDir performs the worst because it is inherently unsuitable for 2D visual information. BiDir and MultiDir show comparable outcomes across various benchmark assessments, but BiDir outperforms in the majority, highlighting its strength in handling 2D visual information for multimodal learning tasks.

The image scanning techniques are applied during both training and inference, and it is essential to maintain train-test consistency. We have made simple attempts to rearrange the order of layers with different directions, but the performance was not robust. Specific layers have already been specialized to process image information from particular directions.

### 4.4.4 Ablation on Learning Rate

As shown in Table [1,](#page-2-2) correct learning rate is crucial for the performance of benchmarks. Table [10](#page-16-0) shows a comparison of our model with different learning rate. From the Table, it can be observed that a higher initial learning rate has a significant impact on the model's performance. Our hypothesis is that the substantial divergence in tasks from the textual to the visual domain necessitates a higher learning rate to facilitate the model's adaptation.

It has been observed that there is a substantial discrepancy between the optimal learning rates of VisualRWKV and LLaVA[\(Liu et al.,](#page-9-1) [2023a\)](#page-9-1), with the optimal initial learning rate for LLaVA-1.5-7B being 2e −5 and for VisualRWKV-7B being 4e −5 . This caused considerable difficulties in our work

<span id="page-7-2"></span>

| Method                              | LLM        | Res. | PT/IT     | VQA  | GQA  | SQA  | TQA  | POPE | MME    | MMB  | MMB-cn |
|-------------------------------------|------------|------|-----------|------|------|------|------|------|--------|------|--------|
| BLIP-2 (Li et al., 2023c)           | Vicuna-13B | 224  | 129M/ -   | 41.0 | 41.0 | 61.0 | 42.5 | 85.3 | 1293.8 | _    | 22.4   |
| MiniGPT-4 (Zhu et al., 2024a)       | Vicuna-7B  | 224  | 5M/5K     | -    | 32.2 | -    | -    | -    | 581.7  | 23.0 | -      |
| InstructBLIP (Dai et al., 2023)     | Vicuna-7B  | 224  | 129M/1.2M | _    | 49.2 | 60.5 | 50.1 | _    | _      | 36   | 26.2   |
| InstructBLIP (Dai et al., 2023)     | Vicuna-13B | 224  | 129M/1.2M | _    | 49.5 | 63.1 | 50.7 | 78.9 | 1212.8 | _    | 25.6   |
| Shikra (Chen et al., 2023b)         | Vicuna-13B | 224  | 600K/5.5M | 77.4 | _    | -    | -    | _    | _      | 58.8 |        |
| Otter (Li et al., 2023a)            | LLaMA-7B   | 224  | -         | -    | -    | -    | -    | -    | 1292.3 | 48.3 | 24.6   |
| mPLUG-Owl (Ye et al., 2023)         | LLaMA-7B   | 224  | 2.1M/102K | -    | -    | -    | -    | -    | 967.3  | 49.4 | -      |
| IDEFICS-9B (IDEFICS, 2023)          | LLaMA-7B   | 224  | 353M/1M   | 50.9 | 38.4 | -    | 25.9 | _    | _      | 48.2 | -      |
| IDEFICS-80B (IDEFICS, 2023)         | LLaMA-65B  | 224  | 353M/1M   | 60.0 | 45.2 | -    | 30.9 | _    | _      | 54.5 | -      |
| Qwen-VL (Bai et al., 2023)          | Qwen-7B    | 448  | 1.4B/50M  | 78.8 | 59.3 | 67.1 | 63.8 | _    | _      | 38.2 | -      |
| Qwen-VL-Chat (Bai et al., 2023)     | Qwen-7B    | 448  | 1.4B/50M  | 78.2 | 57.5 | 68.2 | 61.5 | _    | 1487.5 | 60.6 | -      |
| LLaVA-1.5 (Liu et al., 2023a)       | Vicuna-7B  | 336  | 558K/665K | 78.5 | 62.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 30.5   |
| LLaVA-Phi (Zhu et al., 2024b)       | Phi2-2.7B  | 336  | 558K/665K | 71.4 | -    | 68.4 | 48.6 | 85.0 | 1335.1 | 59.8 | 28.9   |
| MobileVLM-3B (Chu et al., 2023)     | LLaMA-2.7B | 336  | 558K/665K | -    | 59.0 | 61.2 | 47.5 | 84.9 | 1288.9 | 59.6 | -      |
| VL-Mamba (Qiao et al., 2024)        | Mamba-2.8B | 224  | 558K/665K | 76.6 | 56.2 | 65.4 | 48.9 | 84.4 | 1369.6 | 57.0 | 32.6   |
| VisualRWKV-Base (Peng et al., 2024) | RWKV5-1.6B | 336  | 558K/665K | 51.1 | 48.1 | 41.9 | 35.2 | 73.1 | -      | -    | -      |
| VisualRWKV                          | RWKV6-1.6B | 336  | 558K/665K | 69.4 | 55.2 | 59.1 | 43.6 | 83.2 | 1204.9 | 55.8 | 53.2   |
| VisualRWKV                          | RWKV6-3B   | 336  | 558K/665K | 71.5 | 59.6 | 65.3 | 48.7 | 83.1 | 1369.2 | 59.5 | 56.3   |
| VisualRWKV                          | RWKV6-7B   | 336  | 558K/665K | 75.8 | 64.3 | 68.2 | 51.0 | 84.7 | 1387.8 | 65.8 | 63.7   |

Table 2: **Comparison with SoTA methods on 8 benchmarks.** Due to space constraints, benchmark names are abbreviated. VQA (Goyal et al., 2017); GQA (Hudson and Manning, 2019); SQA: ScienceQA-IMG (Lu et al., 2022); TQA: TextVQA (Singh et al., 2019); POPE (Li et al., 2023d); MME (Fu et al., 2023); MMB: MMBench (Liu et al., 2023d); MMB-cn: MMBench-CN (Liu et al., 2023d). PT and IT denote the quantity of samples involved in the pre-training and instruction-tuning phases. "Res." stands for "Resolution.

<span id="page-7-0"></span>

| Method          | Size | Prompt   | VQA   | SQA   | TQA   | GQA   |
|-----------------|------|----------|-------|-------|-------|-------|
| VisualRWKV-Base | 7B   | First    | 67.93 | 65.59 | 47.13 | 48.52 |
| VisualRWKV-Base | 7B   | Last     | 63.07 | 57.66 | 48.52 | 44.19 |
| VisualRWKV-Base | 7B   | Sandwich | 69.71 | 65.20 | 50.25 | 50.50 |

Table 3: Results for three prompting method.

<span id="page-7-1"></span>

| Method          | Size | Scanning | VQA   | SQA   | TQA   | GQA   |
|-----------------|------|----------|-------|-------|-------|-------|
| VisualRWKV-Base | 1.6B | UniDir   | 51.03 | 41.94 | 35.19 | 48.09 |
| VisualRWKV-Base | 1.6B | BiDir    | 65.62 | 47.30 | 37.13 | 48.60 |
| VisualRWKV-Base | 1.6B | MultiDir | 66.04 | 44.03 | 35.84 | 49.95 |
| VisualRWKV      | 1.6B | BiDir    | 69.26 | 57.61 | 43.17 | 54.85 |
| VisualRWKV      | 1.6B | MultiDir | 69.20 | 57.31 | 42.97 | 54.63 |

Table 4: Results for three scanning methods.

at the beginning and also confirmed the significant divergence between the RWKV architecture and the Transformer architecture.

### 4.5 Efficiency Analysis

As shown in Figure 1, we compared the inference speed and GPU memory consumption directly with LLaVA-1.5 of the same parameter size. VisualRWKV has a constant single token inference speed, while the inference speed of a single token in LLaVA-1.5 slows down as more tokens are generated. On the other hand, VisualRWKV has a constant GPU memory consumption, while the mem-

ory consumption of LLaVA-1.5 increases linearly. In practice, compared to LLaVA-1.5, VisualRWKV has a speed advantage of 3.98 times and can save 54% of the GPU memory when reaching an inference length of 24576 tokens. Since VisualRWKV retains a fixed state size throughout inference, GPU memory usage remains nearly constant, which is illustrated as a straight line in Figure 1(c).

### 4.6 Text-only Capability

According to Lin et al. (2024), LLMs face the issue of degraded text capabilities after visual instruction tuning. As shown in Table 5, no degradation of text abilities was observed in VisualRWKV. Conversely, enhancements in performance were noted across various text-only English datasets, which we credit to the integration of a large set of English samples in our fine-tuning dataset. Furthermore, it was observed that VisualRWKV did not face text ability degradation across multiple languages, as shown in Table 5. The capabilities were fundamentally aligned with those of the text-only RWKV. This may be due to the incorporation of the multilingual ShareGPT4. More details about text-only capability can be found in Appendix G.

Besides the results previously stated, we also compared the outcomes of single-stage and two-

<span id="page-8-7"></span>

| Method     | Size | LAMBADA | English | MultiLang |
|------------|------|---------|---------|-----------|
|            |      | ppl     | avg%    | avg%      |
| RWKV       | 1.6B | 4.63    | 59.82   | 59.97     |
| VisualRWKV | 1.6B | 4.15    | 61.01   | 59.83     |

Table 5: Results for text-only capability: The English score is the average of 10 English benchmarks, while the Multilingual score is the average of 4 Multilingual benchmarks.

stage training approaches; conducted ablation studies on the method of cross-entropy loss reduction; assessed the influence of Weight Decay on the model; and explored a basic hybrid model known as VisualRWKV Hybrid. Due to space limitations, we have included these contents in the Appendix.

# 5 Conclusions

In this paper, we introduce for the first time VisualRWKV, which explores the construction of a visual language model using the linear RNN model RWKV. VisualRWKV incorporates three innovative designs: data-dependent recurrence to enhance the model's information extraction capabilities, sandwich prompt for better conditioning, and bidirectional scanning for more effective extraction of 2D visual information. We conducted extensive experiments on eight multimodal benchmarks and achieved comparable performance with some of the most advanced VLMs; we also carried out ablation studies to evaluate the effectiveness of data-dependent recurrence, prompting methods, and various scanning mechanisms. The results validate the effectiveness of our proposed model and demonstrate the potential of applying RNNs to VLMs.

# Limitations

Despite the encouraging results achieved by VisualRWKV, several limitations must be acknowledged. Firstly, due to the lack of data following such instructions and the limited context length, VisualRWKV is currently unable to process multiple images. Secondly, although VisualRWKV shows good performance on academic datasets, its ability to handle certain tasks, such as TextVQA, may be constrained by the limitations in the recall ability of efficient language models [\(Arora et al.,](#page-8-5) [2023\)](#page-8-5). These constraints can potentially be mitigated by further architectural improvements. Lastly, to maintain consistency with LLaVA-1.5, this study did

not investigate the effects of the choice of vision encoder or the quality of training data on VisualRWKV. In the future, we aim to explore more advanced visual encoders and utilize higher-quality training data to further enhance its performance.

Risks Although VisualRWKV significantly reduces the occurrence of hallucinations, it can still generate hallucinations and occasionally disseminate misinformation. Therefore, its application in critical fields, such as the medical industry, should be approached with great caution.

# Acknowledgments

Thanks to Peng Bo, the author of RWKV, for participating in the discussion and providing valuable suggestions for modifications.

# References

<span id="page-8-1"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*.

<span id="page-8-2"></span>Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. *Advances in Neural Information Processing Systems*, 35:23716–23736.

<span id="page-8-4"></span>Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. *arXiv preprint arXiv:2305.10403*.

<span id="page-8-5"></span>Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. 2023. Zoology: Measuring and improving recall in efficient language models. *arXiv preprint arXiv:2312.04927*.

<span id="page-8-6"></span>Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*.

<span id="page-8-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901.

<span id="page-8-3"></span>Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed

- Elhoseiny. 2023a. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. *arXiv preprint arXiv:2310.09478*.
- <span id="page-9-3"></span>Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023b. Shikra: Unleashing multimodal llm's referential dialogue magic. *arXiv preprint arXiv:2306.15195*.
- <span id="page-9-6"></span>Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.
- <span id="page-9-4"></span>Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240):1–113.
- <span id="page-9-19"></span>Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, and Chunhua Shen. 2023. [Mobilevlm : A fast, strong and open vision language](https://api.semanticscholar.org/CorpusID:266573855) [assistant for mobile devices.](https://api.semanticscholar.org/CorpusID:266573855) *ArXiv*, abs/2312.16886.
- <span id="page-9-16"></span>Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. 2023. [Instructblip: Towards general-purpose](https://api.semanticscholar.org/CorpusID:258615266) [vision-language models with instruction tuning.](https://api.semanticscholar.org/CorpusID:258615266) *ArXiv*, abs/2305.06500.
- <span id="page-9-12"></span>Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv preprint arXiv:2306.13394*.
- <span id="page-9-9"></span>Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6904–6913.
- <span id="page-9-7"></span>Mark Harris, Shubhabrata Sengupta, and John D Owens. 2007. Parallel prefix sum (scan) with cuda. *Graphics Processing Unit Gems*, 3(39):851–876.
- <span id="page-9-5"></span>Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. *Neural computation*, 9(8):1735– 1780.
- <span id="page-9-8"></span>Haowen Hou and F. Richard Yu. 2024. [Rwkv-ts: Be](https://api.semanticscholar.org/CorpusID:267027925)[yond traditional recurrent neural network for time](https://api.semanticscholar.org/CorpusID:267027925) [series tasks.](https://api.semanticscholar.org/CorpusID:267027925) *ArXiv*, abs/2401.09093.
- <span id="page-9-10"></span>Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *CVPR*.

- <span id="page-9-18"></span>IDEFICS. 2023. Introducing idefics: An open reproduction of state-of-the-art visual language model. <https://huggingface.co/blog/idefics>.
- <span id="page-9-21"></span>Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. 2024. [Prismatic vlms: Investigating the design space](https://api.semanticscholar.org/CorpusID:267627175) [of visually-conditioned language models.](https://api.semanticscholar.org/CorpusID:267627175) *ArXiv*, abs/2402.07865.
- <span id="page-9-0"></span>Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In *International conference on machine learning*, pages 5156–5165. PMLR.
- <span id="page-9-17"></span>Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023a. [Otter: A](https://api.semanticscholar.org/CorpusID:258547300) [multi-modal model with in-context instruction tuning.](https://api.semanticscholar.org/CorpusID:258547300) *ArXiv*, abs/2305.03726.
- <span id="page-9-11"></span>Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2023b. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. *arXiv preprint arXiv:2306.00890*.
- <span id="page-9-2"></span>Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023c. [BLIP-2: Bootstrapping language-image pre](https://proceedings.mlr.press/v202/li23q.html)[training with frozen image encoders and large lan](https://proceedings.mlr.press/v202/li23q.html)[guage models.](https://proceedings.mlr.press/v202/li23q.html) In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 19730–19742. PMLR.
- <span id="page-9-13"></span>Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023d. Evaluating object hallucination in large vision-language models. *arXiv preprint arXiv:2305.10355*.
- <span id="page-9-15"></span>Zihang Li and Haowen Hou. 2024. [Visualrwkv-hd and](https://api.semanticscholar.org/CorpusID:273350770) [uhd: Advancing high-resolution processing for visual](https://api.semanticscholar.org/CorpusID:273350770) [language models.](https://api.semanticscholar.org/CorpusID:273350770) *ArXiv*, abs/2410.11665.
- <span id="page-9-20"></span>Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models. *CVPR*.
- <span id="page-9-14"></span>Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft COCO: Common objects in context. In *ECCV*.
- <span id="page-9-22"></span>Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Ves Stoyanov, and Xian Li. 2021. [Few-shot learning with mul](https://api.semanticscholar.org/CorpusID:260651613)[tilingual language models.](https://api.semanticscholar.org/CorpusID:260651613) *ArXiv*, abs/2112.10668.
- <span id="page-9-1"></span>Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning.

- <span id="page-10-1"></span>Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. [Visual instruction tuning.](https://openreview.net/forum?id=w0H2xGHlkw) In *Thirtyseventh Conference on Neural Information Processing Systems*.
- <span id="page-10-11"></span>Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2023c. Mmbench: Is your multi-modal model an all-around player? *arXiv preprint arXiv:2307.06281*.
- <span id="page-10-13"></span>Yuanzhan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2023d. [Mmbench: Is your multi-modal model](https://api.semanticscholar.org/CorpusID:259837088) [an all-around player?](https://api.semanticscholar.org/CorpusID:259837088) *ArXiv*, abs/2307.06281.
- <span id="page-10-9"></span>Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*.
- <span id="page-10-14"></span>Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In *Conference on Computer Vision and Pattern Recognition (CVPR)*.
- <span id="page-10-17"></span>Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2022. [Crosslingual general](https://arxiv.org/abs/2211.01786)[ization through multitask finetuning.](https://arxiv.org/abs/2211.01786) *Preprint*, arXiv:2211.01786.
- <span id="page-10-0"></span>OpenAI. 2023. Gpt-4v(ision) system card. [https://cdn.openai.com/papers/GPTV\\_](https://cdn.openai.com/papers/GPTV_System_Card.pdf) [System\\_Card.pdf](https://cdn.openai.com/papers/GPTV_System_Card.pdf).
- <span id="page-10-2"></span>Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023a. [Rwkv:](https://arxiv.org/abs/2305.13048) [Reinventing rnns for the transformer era.](https://arxiv.org/abs/2305.13048) *Preprint*, arXiv:2305.13048.
- <span id="page-10-3"></span>Bo Peng, Eric Alcaide, Quentin G. Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, G Kranthikiran, Xuming He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hay- ´ den Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan Sokrates Wind, Stansilaw Wozniak, Ruichong

- Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui Zhu. 2023b. Rwkv: Reinventing rnns for the transformer era. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-10-8"></span>Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy Ferdinan, Haowen Hou, Przemys l aw Kazienko, G Kranthikiran, Jan Koco'n, Bartlomiej Koptyra, Satyapriya Krishna, Ronald McClelland, Niklas Muennighoff, Fares Obeid, Atsushi Saito, Guangyu Song, Haoqin Tu, Stanislaw Wo'zniak, Ruichong Zhang, Bingchen Zhao, Qihang Zhao, Peng Zhou, Jian Zhu, and Ruijie Zhu. 2024. [Eagle and](https://api.semanticscholar.org/CorpusID:269010053) [finch: Rwkv with matrix-valued states and dynamic](https://api.semanticscholar.org/CorpusID:269010053) [recurrence.](https://api.semanticscholar.org/CorpusID:269010053) *ArXiv*, abs/2404.05892.
- <span id="page-10-5"></span>Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023c. Kosmos-2: Grounding multimodal large language models to the world. *arXiv preprint arXiv:2306.14824*.
- <span id="page-10-15"></span>E. Ponti, Goran Glavavs, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen. 2020. [Xcopa: A](https://api.semanticscholar.org/CorpusID:218470125) [multilingual dataset for causal commonsense reason](https://api.semanticscholar.org/CorpusID:218470125)[ing.](https://api.semanticscholar.org/CorpusID:218470125) In *Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-10-12"></span>Yanyuan Qiao, Zheng Yu, Longteng Guo, Sihan Chen, Zijia Zhao, Mingzhen Sun, Qi Wu, and Jing Liu. 2024. [Vl-mamba: Exploring state space models for](https://api.semanticscholar.org/CorpusID:268537285) [multimodal learning.](https://api.semanticscholar.org/CorpusID:268537285) *ArXiv*, abs/2403.13600.
- <span id="page-10-6"></span>Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.
- <span id="page-10-10"></span>Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8317–8326.
- <span id="page-10-4"></span>Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*.
- <span id="page-10-16"></span>Alexey Tikhonov and Max Ryabinin. 2021. [It's all in the](https://arxiv.org/abs/2106.12066) [heads: Using attention heads as a baseline for cross](https://arxiv.org/abs/2106.12066)[lingual transfer in commonsense reasoning.](https://arxiv.org/abs/2106.12066) *Preprint*, arXiv:2106.12066.
- <span id="page-10-7"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

- <span id="page-11-0"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-11-2"></span>Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. *arXiv preprint arXiv:2304.14178*.
- <span id="page-11-1"></span>Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2024a. [MiniGPT-4: Enhancing](https://openreview.net/forum?id=1tZbq88f27) [vision-language understanding with advanced large](https://openreview.net/forum?id=1tZbq88f27) [language models.](https://openreview.net/forum?id=1tZbq88f27) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-11-3"></span>Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. 2024b. [Llava-phi: Efficient](https://api.semanticscholar.org/CorpusID:266755915) [multi-modal assistant with small language model.](https://api.semanticscholar.org/CorpusID:266755915) *ArXiv*, abs/2401.02330.

# Supplementary Material for VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models

# <span id="page-12-0"></span>A Data and Hyperparameters

Training Data The data used in this study is strictly aligned with LLaVA-1.5. The training of VisualRWKV is composed of two phases: (1) Feature Alignment Phase: Utilizing our 558K subset from the LAION-CC-SBU dataset, we link a pretrained, frozen vision encoder to a frozen Large Language Model (LLM); (2) Visual Instruction Tuning Phase: We employ 150K of GPT-generated multimodal instruction-following datasets, supplemented by approximately 515K Visual Question Answering (VQA) datasets from academically oriented tasks [\(Marino et al.,](#page-10-14) [2019;](#page-10-14) [Singh et al.,](#page-10-10) [2019;](#page-10-10) [Hudson and Manning,](#page-9-10) [2019;](#page-9-10) [Goyal et al.,](#page-9-9) [2017\)](#page-9-9), to instruct the model in adhering to multimodal directives. For more details, one can refer to the paper on LLaVA-1.5 [\(Liu et al.,](#page-9-1) [2023a\)](#page-9-1). All the data used in this paper are consistent with their intended use. We carefully identified and handled all personally identifiable information and offensive content. We started with automated screening to flag sensitive data, followed by manual review for precision. Anonymization methods like data masking and pseudonymization were applied to protect sensitive information. Strict data protection protocols were followed throughout.

Evaluation Benchmarks Additional details on Benchmarks are provided here. The VQA-v2 reports its metrics based on the test-dev split. Similarly, GQA's metrics are on the test-dev split. The metrics for TextVQA are reported on the validation set. ScienceQA's metrics are based on the test set. POPE's metrics are also reported on the test set. The MMBench metrics are reported on the development set. MME has a unique test-set, thus there is no ambiguity.

Data Language Firstly, the training data includes academic Visual Question Answering (VQA) datasets and ShareGPT data. The primary language of the VQA academic datasets is English, while the ShareGPT data is multilingual, encompassing mainstream languages, but derived from contributions by users worldwide, it is not feasible to count the total number of languages. Among the evaluation benchmarks, MMBench-cn is the only Chinese dataset; the rest are English datasets. Concurrently, we evaluated the model's text-only capabilities in multiple languages, with the specific languages detailed in Appendix [G.](#page-16-1)

Hyperparameters The hyperparameters here were used for the training of a range of VisualRWKV models, from 1.6B to 7B parameters, as illustrated in Table [2.](#page-7-2) We show the training hyperparameters for both first-stage vision-language alignment pretraining and the second-stage visual instruction tuning in Table [6.](#page-12-1)

<span id="page-12-1"></span>

| Hyperparameter  | 1.6B-Pretrain | 1.6B-Finetune | 3B-Pretrain  | 3B-Finetune  | 7B-Pretrain  | 7B-Finetune  |
|-----------------|---------------|---------------|--------------|--------------|--------------|--------------|
| batch size      | 256           | 128           | 256          | 128          | 256          | 128          |
| lr init         | 1e-3          | 6e-5          | 1e-3         | 5e-5         | 1e-3         | 4e-5         |
| lr end          | 1e-5          | 1.5e-5        | 1e-5         | 1.25e-5      | 1e-5         | 1e-5         |
| lr schedule     | cosine decay  | cosine decay  | cosine decay | cosine decay | cosine decay | cosine decay |
| lr warmup ratio | 0             | 0             | 0            | 0            | 0            | 0            |
| weight decay    | 0             | 0             | 0            | 0            | 0            | 0            |
| epoch           | 1             | 2             | 1            | 2            | 1            | 2            |
| optimizer       | AdamW         | AdamW         | AdamW        | AdamW        | AdamW        | AdamW        |
| DeepSpeed stage | 1             | 1             | 1            | 1            | 1            | 2            |

Table 6: Hyperparameters of VisualRWKV.

Licenses VisualRWKV is licensed under the Apache-2.0 license. The RWKV language model is also under the Apache-2.0 license. The LLaVA model is licensed under the Apache-2.0 license. The VQA-v2 dataset is licensed under the Commons Attribution 4.0 International License. MMBench is licensed under the Apache-2.0 license. TextVQA data is available under the CC BY 4.0 license. ScienceQA is licensed under the MIT License, and POPE is also under the MIT license.

# B Model and Computation

LLM Model The LLM foundation model is primarily based on two families: the RWKV-5 series[1](#page-13-0) and the RWKV-6 series[2](#page-13-1) . Both the RWKV-5 and RWKV-6 series consist of models with 1.6 billion, 3 billion, and 7 billion parameters respectively. In this research, the RWKV-5 series is mainly applied in the VisualRWKV-Base, and the RWKV-6 series acts as the LLM backbone for VisualRWKV.

Model Size The vision encoders utilized in this paper are based on the CLIP-L model, which features 0.3 billion parameters. In contrast, the RWKV models vary in size: the RWKV 7B has 7.6 billion parameters, the RWKV 1.6B has 1.6 billion parameters, and the RWKV 3B has 3.1 billion parameters. Consequently, the VisualRWKV variants have different total parameter counts: the VisualRWKV 1.6B encompasses 1.9 billion parameters, the VisualRWKV 3B includes 3.4 billion parameters, and the VisualRWKV 7B comprises 7.9 billion parameters.

Computing Infrastructure A range of computational resources were employed in the study. The standard training and benchmark evaluation were conducted using 8 NVIDIA A100-80GB GPUs. The VisualRWKV 7B model is trained with 6 A100 GPUs due to insufficient memory capacity with 8 GPUs. For the efficiency analysis, a GPU with L20-48GB of memory was employed.

Computing Budget Training an epoch of VisualRWKV 1.6B with 8 A100 GPUs takes 6.7 hours, equivalent to 53.6 GPU hours; Training an epoch of VisualRWKV 3B with 8 A100 GPUs takes 11.3 hours, equivalent to 90.4 GPU hours; Training an epoch of VisualRWKV 7B with 6 A100 GPUs takes 26.5 hours, equivalent to 159 GPU hours.

Packages Version The main experimental environment for this study is the NVIDIA PyTorch NGC Container (23.07-py3) with lightning1.9.5 and deepspeed0.12.6. For updates, please refer to our codebase (currently anonymized, will be released later).

# C Single-Stage Training vs. Two-Stage Training

The research conducted by [Karamcheti et al.](#page-9-21) [\(2024\)](#page-9-21), suggests that including a distinct projector pretraining phase may not be essential. Their study indicates that a single-stage training process can lead to improved performance outcomes. Omission of the pretraining phase results in a significant cost reduction of about 20 to 25 percent and avoids the need for stage-specific data collection.

To validate these insights, we conducted a series of experiments using the VisualRWKV framework. The results, as illustrated in Figure [5,](#page-14-0) show that the two-stage training outperforms single-stage training, indicating that the two-stage approach is still very necessary. The different results associated with single-stage training could be due to the diverse training setups used by various researchers. Given these results, we have made a strategic decision to adopt a two-stage training protocol for all subsequent experiments in this paper.

# D Influence of Cross-Entropy Loss Reduction

In the experiment, we found that using zero1 for training with a batch size of 1 and gradient accumulation of 16; and using zero2 for training with a batch size of 1 and gradient accumulation of 1; These two settings are not equivalent, with different losses, leading to significantly disparate outcomes for the final model. Therefore, we conducted an in-depth analysis and study.

For illustrative purposes, consider a simple thought experiment with four samples: the first sample consists of 100 tokens, the second of 200 tokens, the third of 300 tokens, and the fourth of 400 tokens. Consequently, the total length sums up to 1000 tokens. When these samples are batched together(batch size of 4 and gradient accumulation of 1), each token is normalized by a factor of 1000. We refer this process as batch-level reduction. Please note that the batch-level reduction is highly dependent on the

<span id="page-13-0"></span><sup>1</sup> <https://huggingface.co/BlinkDL/rwkv-5-world>

<span id="page-13-1"></span><sup>2</sup> <https://huggingface.co/BlinkDL/rwkv-6-world>

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 5: Single-Stage Training vs. Two-Stage Training. We conducted a comparative analysis between the twostage training and single-stage training, with the latter omitting the vision-language alignment phase. Our findings reveal that the single-stage training yields inferior performance outcomes. This suggests that the vision-language alignment, integral to the two-stage training, significantly contributes to enhanced performance.

batch size. As the batch size varies, the total batch length by which each token's loss is divided can differ significantly.

An alternative approach, termed sample-level reduction, normalizes each sample by its length. This sample-level reduction is independent of the batch size and introduces a different loss re-weighting compared to batch-level reduction. Continuing our thought experiment, we apply sample-level reduction with a batch size of 1 and gradient accumulation of 4. The first sample undergoes a sequential division by 100 (its length) and then by 4, culminating in an effective division by 400. The second sample is adjusted by a factor of 800, the third by 1200, and the fourth by 1600. This scaling mechanism inherently leads to a larger loss for shorter texts and a smaller loss for longer texts compared to batch-level reduction.

Our findings underscore the importance of accurate reduction and loss re-weighting for the performance of certain downstream tasks. Table [7](#page-14-1) presents a comparative analysis between our model's performance under batch-level and sample-level reduction. Notably, we have found that using sample-level reduction yields better results on 5 benchmarks. In contrast, batch-level reduction performs better on 2 benchmarks. Among them, sample-level reduction significantly outperforms on the ScienceQA benchmark. On the MME benchmark, batch-level reduction takes the lead. After an in-depth investigation, we discovered that the score in the Celebrity domain within MME has significantly improved, while other domains show varying degrees of success.

<span id="page-14-1"></span>

| Reduction    | VQAv2 | ScienceQA | TextVQA | GQA    | VizWiz | MME     | POPE |
|--------------|-------|-----------|---------|--------|--------|---------|------|
| Sample-Level | 67.54 | 56.62%    | 42.18%  | 52.82% | 26.03  | 1111.66 | 0.82 |
| Batch-Level  | 66.85 | 47.94%    | 41.79%  | 52.56% | 27.02  | 1173.42 | 0.79 |

Table 7: Study comparing batch-level reduction and sample-level reduction across 7 Visual Language benchmarks. Loss reduction method is crucial for performance. Model used here is VisualRWKV 1.6B.

Furthermore, we conducted a comparison of the textual abilities resulting from sample-level and batch-level reduction, as shown in Table [8.](#page-15-1) It was observed that sample-level training exhibited superior English capabilities, whereas the batch-level training demonstrated enhanced multilingual abilities. This is due to the higher loss weight assigned to the multilingual long texts of ShareGPT4 data in the batch-level training.

In general, we consider sample-level reduction to be the better approach. On one hand, the performance is better, whether in visual-linguistic abilities or pure textual capabilities. On the other hand, sample-level reduction is invariant to batch size. When the sample-level reduction-based training protocol is migrated across various GPUs, it does not suffer from inconsistencies due to batch size variations, which could <span id="page-15-1"></span>otherwise lead to divergent outcomes.

| Reduction    | LAMBADA(ppl) | English(avg%) | MultiLang(avg%) |
|--------------|--------------|---------------|-----------------|
| Batch-Level  | 4.499        | 59.89         | 59.97           |
| Sample-Level | 4.145        | 61.01         | 59.84           |

Table 8: Study comparing batch-level reduction and sample-level reduction across language benchmarks. Model used here is VisualRWKV 1.6B.

# <span id="page-15-0"></span>E Further details on the Prompting Method

In this section, we will further discuss three types of prompt methods. As shown in Table [9,](#page-15-2) we found that as the number of image tokens decreases, the effectiveness of the image first prompt and sandwich prompt also monotonically decreases, which is intuitively expected as fewer image tokens contain less pictorial information. Nonetheless, the image last prompt does not exhibit a strictly decreasing trend; it initially increases and subsequently decreases, achieving optimal performance at the point of 145 image tokens. The effect is especially evident in scenarios of train-test mismatch. We term this the information barrier formed by image tokens, which hinders the model's information transfer.

<span id="page-15-2"></span>An additional observation indicates that the sandwich prompt is capable of mitigating information loss, sustaining good performance even with a limited number of image tokens. In contrast, the other two types of prompt methods fail to achieve this.

| Method                                                | Size | Prompt   | Image Tokens | ScienceQA | TextVQA | GQA    |
|-------------------------------------------------------|------|----------|--------------|-----------|---------|--------|
| VisualRWKV-Base<br>VisualRWKV-Base<br>VisualRWKV-Base |      |          | 577          | 65.59%    | 47.13%  | 48.52% |
|                                                       |      |          | 145          | 64.14%    | 42.91%  | 45.99% |
|                                                       |      |          | 65           | 64.01%    | 40.67%  | 44.08% |
|                                                       |      |          | 37           | 62.87%    | 39.90%  | 43.44% |
|                                                       | 7B   | First    | 17           | 61.23%    | 39.96%  | 43.31% |
|                                                       |      |          | 10           | 60.29%    | 39.65%  | 43.23% |
|                                                       |      |          | 5            | 59.35%    | 39.80%  | 43.16% |
|                                                       |      |          | 1            | 57.11%    | 39.34%  | 43.53% |
|                                                       |      |          | 577          | 57.66%    | 48.52%  | 44.19% |
|                                                       |      | Last     | 145          | 58.75%    | 45.29%  | 42.93% |
|                                                       |      |          | 65           | 56.07%    | 43.89%  | 42.38% |
|                                                       |      |          | 37           | 53.35%    | 43.03%  | 42.07% |
|                                                       | 7B   |          | 17           | 50.37%    | 42.50%  | 42.03% |
|                                                       |      |          | 10           | 50.72%    | 42.18%  | 42.10% |
|                                                       |      |          | 5            | 49.23%    | 41.20%  | 41.80% |
|                                                       |      |          | 1            | 50.67%    | 41.19%  | 41.93% |
|                                                       |      |          | 577          | 65.20%    | 50.25%  | 50.50% |
|                                                       |      |          | 145          | 64.90%    | 46.38%  | 47.47% |
|                                                       |      |          | 65           | 64.40%    | 44.58%  | 45.09% |
|                                                       |      |          | 37           | 64.11%    | 44.01%  | 44.78% |
|                                                       | 7B   | Sandwich | 17           | 63.86%    | 43.61%  | 44.57% |
|                                                       |      |          | 10           | 63.26%    | 43.27%  | 44.37% |
|                                                       |      |          | 5            | 62.87%    | 43.03%  | 44.08% |
|                                                       |      |          | 1            | 60.34%    | 41.72%  | 36.09% |

Table 9: Full Results for three prompting method.

# F Study on Learning Rate

In this section, We will explore the effect of learning rates on VisualRWKV. Setting different initial learning rates and using a cosine learning rate scheduler, the performance of the model on multiple <span id="page-16-0"></span>benchmarks is shown in the Table [10.](#page-16-0)

| Method     | Size | Learning Rate    | VQAv2 | ScienceQA | TextVQA | GQA   | MME     |
|------------|------|------------------|-------|-----------|---------|-------|---------|
| VisualRWKV | 1.6B | 2e-5 to 2e-5     | 66.85 | 57.51     | 41.85   | 52.07 | 1080.77 |
| VisualRWKV | 1.6B | 3e-5 to 1e-5     | 67.25 | 53.40     | 41.84   | 52.49 | 1115.70 |
| VisualRWKV | 1.6B | 3e-5 to 1.5e-5   | 67.54 | 56.62     | 42.18   | 52.82 | 1111.66 |
| VisualRWKV | 1.6B | 4e-5 to 1.5e-5   | 68.51 | 55.68     | 43.73   | 54.31 | 1151.20 |
| VisualRWKV | 1.6B | 5e-5 to 1.5e-5   | 69.26 | 57.61     | 43.17   | 54.85 | 1208.96 |
| VisualRWKV | 1.6B | 6e-5 to 1.5e-5   | 69.42 | 59.05     | 43.57   | 55.23 | 1204.90 |
| VisualRWKV | 1.6B | 1e-4 to 1.5e-5   | 70.02 | 55.58     | 42.24   | 55.72 | 1212.52 |
| VisualRWKV | 1.6B | 1.5e-4 to 1.5e-5 | 68.89 | 55.63     | 41.90   | 54.09 | 1249.51 |
| VisualRWKV | 3B   | 4e-5 to 1e-5     | 68.65 | 65.99     | 48.46   | 54.40 | 1323.18 |
| VisualRWKV | 3B   | 5e-5 to 1.25e-5  | 71.52 | 65.34     | 48.68   | 59.56 | 1369.19 |
| VisualRWKV | 7B   | 2e-5 to 2e-5     | 68.31 | 68.91     | 50.09   | 52.80 | 1340.44 |
| VisualRWKV | 7B   | 4e-5 to 1e-5     | 75.82 | 68.22     | 51.01   | 64.27 | 1387.75 |

Table 10: Impact of Learning Rate on the Performance of the VisualRWKV on 5 benchmarks.

# <span id="page-16-1"></span>G Improvement on Text-only Capability

In this section, you can find full results on text-only capability, as shown in the Table [11](#page-16-2) and Table [12.](#page-16-3)

<span id="page-16-2"></span>

| Method     | Size | LBD  | Eng   | LAM   | PIQA  | SC16  | HSW   | WG    | ARC-C | ARC-E | HQA   | OBQA  | SCIQ  |
|------------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|            |      | ppl  | avg%  | acc   | acc   | acc   | acc-n | acc   | acc-n | acc   | acc-n | acc-n | acc   |
| RWKV       | 1.6B | 4.63 | 59.82 | 67.39 | 74.37 | 74.50 | 61.06 | 60.93 | 33.70 | 64.18 | 35.22 | 37.4  | 89.40 |
| VisualRWKV | 1.6B | 4.15 | 61.01 | 67.64 | 73.44 | 75.09 | 61.50 | 61.95 | 38.31 | 67.88 | 36.46 | 38.0  | 89.80 |

Table 11: The table lists the English performance metrics for various benchmarks: LBD (LAMBADA), PIQA, SC16 (StoryCloze16), HSW (Hellaswag), WG (WinoGrande), ARC-C (arc\_challenge), ARC-E (arc\_easy), HQA (headQA\_en), OBQA (openbookQA), SCIQ. Metric units are ppl (perplexcity), acc (accuracy) and acc-n (normalized accuracy).

For multilingual evaluations, we assess LAMBADA in English, French, German, Italian, and Spanish. We evaluate StoryCloze as per [\(Lin et al.,](#page-9-22) [2021\)](#page-9-22) in Arabic, English, Spanish, Basque, Hindi, Indonesian, Burmese, Russian, Swahili, Telugu, and Chinese. COPA is evaluated in Estonian, Haitian Creole, Indonesian, Italian, Cusco-Collao Quechua, Kiswahili, Tamil, Thai, Turkish, Vietnamese, and Chinese, following [\(Ponti et al.,](#page-10-15) [2020\)](#page-10-15). We also evaluate multilingual WinoGrande in English, French, Japanese, Portuguese, Russian, and Chinese, as demonstrated in [\(Tikhonov and Ryabinin,](#page-10-16) [2021;](#page-10-16) [Muennighoff et al.,](#page-10-17) [2022\)](#page-10-17).

| Method     | Size | MultiLang | xLBD  | xSC   | xWG   | xCOPA |  |
|------------|------|-----------|-------|-------|-------|-------|--|
|            |      | avg%      | acc   | acc   | acc   | acc   |  |
| RWKV       | 1.6B | 59.97     | 47.17 | 58.24 | 76.46 | 58.03 |  |
| VisualRWKV | 1.6B | 59.83     | 46.73 | 58.90 | 75.07 | 58.65 |  |

<span id="page-16-3"></span>Table 12: The table lists the Multi-Language performance metrics for various benchmarks: xLBD (Multilingual LAMBADA), xSC (Multilingual StoryCloze), xWG (Multilingual WinoGrande), xCOPA (Multilingual COPA).

# H Study on Weight Decay

Having established the best learning rate, we conducted additional investigations into weight decay. Weight decay was imposed solely on the model's linear layers. The Table [13](#page-17-0) illustrates that, currently, the

<span id="page-17-0"></span>

| Model           | Weight Decay | Learning Rate  | VQA   | SQA    | TQA   | GQA   | VizWiz | MME     |
|-----------------|--------------|----------------|-------|--------|-------|-------|--------|---------|
| VisualRWKV 1.6B | 0            | 6e-5 to 1.5e-5 | 69.42 | 59.05  | 43.57 | 55.23 | 29.84  | 1204.90 |
| VisualRWKV 1.6B | 0.1          | 6e-5 to 1.5e-5 | 68.48 | 58.85% | 41.58 | 54.34 | 28.05  | 1173.03 |
| VisualRWKV 1.6B | 0.01         | 6e-5 to 1.5e-5 | 68.53 | 59.40% | 42.24 | 54.24 | 27.86  | 1154.52 |

Table 13: Impact of Weight Decay on the Performance of the VisualRWKV on 6 benchmarks.

best outcomes are achieved without weight decay. The role of weight decay is complex and may require further exploration in the future.

# I VisualRWKV Hybrid

<span id="page-17-1"></span>We have preliminarily explored the feasibility of the VisualRWKV hybrid model. The hybrid model refers to the combined use of RWKV and Attention. As shown in the Figure [6,](#page-17-1) we have simply added a layer of Tiny Attention on the top of the RWKV blocks. The parameter count of Tiny Attention is smaller than that of the standard Attention, and it does not include an FFN layer.

![](_page_17_Figure_5.jpeg)

Figure 6: VisualRWKV Hybrid: Add a Tiny Attention Layer on the top of RWKV Blocks.

The results of the VisualRWKV hybrid are presented in Table [14.](#page-18-0) It can be observed that there is an improvement over the baseline model without tiny attention. Considering the minimal increase in the number of parameters, this improvement is quite significant. Additionally, we found that the hybrid model equipped with tiny attention is more robust to the number of image tokens. These results suggest the incorporation of more Attention modules in future work may lead to further enhancements and enable the construction of superior Hybrid models.

# J Use of AI Assistants

In this research, an AI writing assistant is solely employed for the purposes of paraphrasing, spell-checking, and enhancing the author's original content, and it does not introduce any novel content.

<span id="page-18-0"></span>

| Method            | Size | Image Tokens | ScienceQA | TextVQA | GQA   |
|-------------------|------|--------------|-----------|---------|-------|
|                   |      | 577          | 65.2      | 50.25   | 50.5  |
|                   |      | 145          | 64.90     | 46.38   | 47.47 |
|                   |      | 65           | 64.40     | 44.58   | 45.09 |
|                   | 7B   | 37           | 64.11     | 44.01   | 44.78 |
| VisualRWKV-Base   |      | 17           | 63.86     | 43.61   | 44.57 |
|                   |      | 10           | 63.26     | 43.27   | 44.37 |
|                   |      | 5            | 62.87     | 43.03   | 44.08 |
|                   |      | 1            | 60.34     | 41.72   | 36.09 |
|                   |      | 577          | 67.38     | 50.97   | 49.96 |
|                   |      | 145          | 66.83     | 47.13   | 46.20 |
|                   | 7B   | 65           | 65.44     | 45.63   | 45.03 |
|                   |      | 37           | 65.39     | 45.47   | 44.81 |
| VisualRWKV-Hybrid |      | 17           | 64.40     | 45.07   | 44.65 |
|                   |      | 10           | 64.06     | 44.79   | 44.44 |
|                   |      | 5            | 63.26     | 44.75   | 43.98 |
|                   |      | 1            | 63.11     | 44.71   | 43.76 |

Table 14: Results of VisualRWKV Hybrid model on 3 benchmarks. The prompting method used here is the sandwich prompt.