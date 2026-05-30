## LONGMAMBA: ENHANCING MAMBA'S LONG CON-TEXT CAPABILITIES VIA TRAINING-FREE RECEPTIVE FIELD ENLARGEMENT

Zhifan Ye1<sup>∗</sup> , Kejing Xia1<sup>∗</sup> , Yonggan Fu1,<sup>2</sup> , Xin Dong<sup>2</sup> , Jihoon Hong<sup>1</sup> , Xiangchi Yuan<sup>1</sup> , Shizhe Diao<sup>2</sup> , Jan Kautz<sup>2</sup> , Pavlo Molchanov<sup>2</sup> , Yingyan (Celine) Lin1,<sup>2</sup> <sup>1</sup>Georgia Institute of Technology <sup>2</sup>NVIDIA

{zye327,kxia39,yfu314,jhong392,xyuan300,celine.lin}@gatech.edu {xind,sdiao,jkautz,pmolchanov}@nvidia.com

## ABSTRACT

State space models (SSMs) have emerged as an efficient alternative to Transformer models for language modeling, offering linear computational complexity and constant memory usage as context length increases. However, despite their efficiency in handling long contexts, recent studies have shown that SSMs, such as Mamba models, generally underperform compared to Transformers in long-context understanding tasks. To address this significant shortfall and achieve both efficient and accurate long-context understanding, we propose LongMamba, a trainingfree technique that significantly enhances the long-context capabilities of Mamba models. LongMamba builds on our discovery that the hidden channels in Mamba can be categorized into local and global channels based on their receptive field lengths, with global channels primarily responsible for long-context capability. These global channels can become the key bottleneck as the input context lengthens. Specifically, when input lengths largely exceed the training sequence length, global channels exhibit limitations in adaptively extend their receptive fields, leading to Mamba's poor long-context performance. The key idea of LongMamba is to mitigate the hidden state memory decay in these global channels by preventing the accumulation of unimportant tokens in their memory. This is achieved by first identifying critical tokens in the global channels and then applying token filtering to accumulate only those critical tokens. Through extensive benchmarking across synthetic and real-world long-context scenarios, LongMamba sets a new standard for Mamba's long-context performance, significantly extending its operational range without requiring additional training. Our code is available at <https://github.com/GATECH-EIC/LongMamba>.

## 1 INTRODUCTION

The rapid advancement of large language models (LLMs) has demonstrated significant capabilities across a diverse array of real-world tasks, ranging from question answering [\(Zhuang et al., 2023\)](#page-12-0) and document summarization [\(Jin et al., 2024\)](#page-11-0) to code completion [\(Li et al., 2022\)](#page-11-1). These tasks often involve processing long input sequences, such as extensive documents and sizable codebases, thereby increasing the demand for LLMs to manage increasingly longer context lengths. Contemporary commercial LLMs, including Mistral Large 2 [\(MistralAI, 2024\)](#page-11-2) and GPT-4 [\(Achiam et al.,](#page-10-0) [2023\)](#page-10-0), feature context windows of up to 128,000 tokens.

Despite their capabilities, Transformer-based LLMs encounter significant scalability issues as sequence lengths increase [\(Katharopoulos et al., 2020\)](#page-11-3). This is primarily due to their quadratic computational complexity and linear memory complexity as context length increases. In contrast, Mamba [\(Gu & Dao, 2023\)](#page-10-1), one of the representative state space models (SSMs) [\(Gu et al., 2021;](#page-10-2) [2022a](#page-10-3)[;b\)](#page-10-4), offers a recurrent computation mechanism that maintains linear computational complexity and constant memory with fixed-size hidden states, enabling efficient long-context processing.

<sup>∗</sup> Equal contribution.

However, SSMs fall short in achievable accuracy on long-context tasks compared to similarly sized Transformers, as highlighted in recent empirical studies [\(Waleffe et al., 2024;](#page-11-4) [Ben-Kish et al., 2024\)](#page-10-5).

To understand the cause of Mamba's failure to generalize to long-context lengths, we analyze the per-channel attention patterns [\(Ali et al., 2024\)](#page-10-6) of Mamba. We find that the channels in Mamba have distinct receptive field lengths: most channels, termed *local channels*, focus on local contexts, while others, termed *global channels*, have receptive fields that extend as long as the training sequence, enabling them to capture global information from the input context. More importantly, we find that these global channels are primarily responsible for Mamba's long-context capability and can become the key bottleneck for this capability as their receptive fields fail to generalize to new sequence lengths. This failure stems from cumulative state decay that increases exponentially with context length, rendering the global channels incapable of memorizing past tokens with large decay.

Inspired by these findings and analyses, we propose LongMamba, a training-free method designed to significantly enhance the receptive fields of the identified global channels when the sequence length far exceeds the training sequence. Specifically, LongMamba enlarges the receptive fields of the identified global channels by adaptively adjusting the decay based on the target context length. This is achieved by applying token filtering to accumulate only critical tokens in the global channels' hidden state memory. This enlargement ensures that these channels can maintain their function as global information processors when exposed to much longer sequences than they were originally trained on, thereby substantially extending the functional range of Mamba models. Our contributions can be summarized as follows:

- Through visualization and analysis, we find that hidden state channels in Mamba SSMs have distinct receptive field lengths, allowing them to be categorized into local channels and global channels. We identify that the inability of global channels to capture global information when exposed to much longer sequences than they were originally trained on is the key bottleneck that limits Mamba's performance on long-context tasks.
- Building upon our findings, we propose LongMamba, a training-free method that enhances Mamba's long-context performance by effectively enlarging the receptive fields of global channels when the sequence length exceeds the training sequence. We achieve this by identifying and removing less important tokens in global channels, thereby preventing the accumulation of unimportant tokens in their hidden state memory.
- Through comprehensive benchmarking on both synthetic and real-world long-context tasks, we demonstrate that LongMamba can significantly extend the operational range of pre-trained Mamba models, outperforming previous methods aimed at enhancing Mamba's context-handling capabilities. For instance, on the widely used LongBench-E [\(Bai et al., 2023\)](#page-10-7) dataset, our method improves task accuracy by up to 4.8× compared to the vanilla Mamba models and up to 2.6× over the previous approach.

## 2 RELATED WORKS

State Space Models (SSMs). SSMs provide a framework for representing dynamic systems through a temporal sequence of latent states, where the system's output is derived from these states [\(Durbin](#page-10-8) [et al., 2012\)](#page-10-8). In the realm of deep learning, SSMs have emerged as a promising alternative to Transformer-based architectures for sequential data processing. Initial efforts to integrate SSMs into deep learning architectures encountered significant obstacles, such as stability issues during training. The Structured State Space Sequence model (S4) [\(Gu et al., 2022b\)](#page-10-4) marks a pivotal advancement in addressing these challenges, enabling the stable training of SSMs in deep neural networks. However, early deep SSM implementations still lack a crucial feature inherent to attention mechanisms: inputdependent information selection. Mamba [\(Gu & Dao, 2023\)](#page-10-1) addresses this limitation by introducing selective SSM layers with input-dependent update mechanisms. A subsequent follow-up, Mamba-2 [\(Dao & Gu, 2024b\)](#page-10-9), further refined this approach, demonstrating competitive performance as compared to Transformers. However, the difficulty of effectively handling very long-range dependencies in SSM-based models like Mamba remains a key challenge in modern language modeling, particularly when processing extended contexts beyond their initial training lengths [\(Ben-Kish et al., 2024;](#page-10-5) [Waleffe et al., 2024\)](#page-11-4).

Mamba Models. The Mamba architecture's efficiency and potential drive its adaptation across diverse applications. In computer vision, Vim (Zhu et al., 2024) uses bidirectional state space modeling for managing long-range dependencies in images, while VMamba (Liu et al., 2024) enhances selective SSMs with novel scanning algorithms for better information flow. DiM (Teng et al., 2024) customizes Mamba for high-resolution image diffusion. The need for extended context modeling in video, point cloud, and graph sequences boosts the demand for Mamba solutions, spurring further research. VideoMamba (Li et al., 2024) and Graph-Mamba (Wang et al., 2024a) exemplify this by applying Mamba's long temporal and spatial sequence handling. Ongoing advancements and applications further fuel the demand for Mamba's long-context capabilities. Hybrid Mamba-Attention models like Jamba (Lieber et al., 2024), Zamba(Glorioso et al., 2024b), and Hymba (Dong et al., 2024) attempt to combine the benefits of attention mechanisms with Mamba's efficiency in long-range modeling (Lieber et al., 2024; Glorioso et al., 2024b).

Language Models for Long-Context Understanding. Language models trained on length-limited contexts often experience performance degradation when extrapolated to longer sequences. Previous research attempted to address this problem through various approaches, including positional interpolation (Peng et al., 2024; Wang et al., 2024b), improvements to the attention mechanism (Xiao et al., 2024b; Yao et al., 2024), and external memory integration (Xiao et al., 2024a; Bulatov et al., 2022). Despite these advancements, such Transformer-based solutions frequently encounter computational and memory constraints as context lengths increase significantly. Furthermore, these methods cannot be directly applied to Mamba models due to the fundamental architectural differences between Transformers and SSMs, particularly the absence of explicit attention mechanisms in Mamba's recurrent structure. To close this gap, DeciMamba (Ben-Kish et al., 2024) is the first to explore context-extension capabilities of Mamba models. Specifically, DeciMamba employs a token pruning mechanism that progressively reduces sequence length in deeper layers by selectively removing less critical tokens, using empirically determined pruning ratios that vary across datasets and tasks. In contrast, our approach identifies the existence of global channels and their inability to capture global information when exposed to longer sequences as the key bottleneck that limits Mamba's performance on long-context tasks. Consequently, our method leverages this observation by enlarging the receptive fields of global channels, eliminating the need for meticulous layer-specific adjustments, and consistently surpassing DeciMamba in performance across diverse benchmarks.

#### <span id="page-2-1"></span>3 PRELIMINARIES OF MAMBA MODELS

In this section, we provide the background of the Mamba model design and review previous efforts (Ali et al., 2024) in measuring the attention score of Mamba models, which lays the groundwork for our analysis in Sec. 4.

**Mamba model design.** Given an input sequence of L tokens  $I \in \mathbb{R}^{L \times d_m}$  ( $d_m$  is the input channel dimension), a Mamba block maps the input sequence to output sequence  $O \in \mathbb{R}^{L \times d_m}$  through the following computation:

$$X = \sigma(\text{Conv1D}(\text{Linear}_1(I))) \in \mathbb{R}^{L \times d_e}$$
(1)

$$Y = SSM(X) \in \mathbb{R}^{L \times d_e} \tag{2}$$

$$O = \operatorname{Linear}_{3}(\sigma(\operatorname{Linear}_{2}(I)) \odot Y) \in \mathbb{R}^{L \times d_{m}}$$
(3)

where Linear<sub>1</sub>, Linear<sub>2</sub> and Linear<sub>3</sub> are regular linear projections, Conv1D is a 1D causal convolution with a causal mask,  $\sigma$  is an activation function, and  $\odot$  represents element-wise product. SSM is a state-space machine that performs a recurrent computation on the input sequence  $X = (X_1, X_2, ..., X_L) \in \mathbb{R}^{L \times d_e}$  ( $d_e$  is the output dimension of Linear<sub>1</sub>):

$$H_t = \bar{A}_t \odot H_{t-1} + \bar{B}_t \odot X_t \in \mathbb{R}^{d_s \times d_e}$$
(4)

<span id="page-2-0"></span>
$$Y_t = C_t^T H_t \in \mathbb{R}^{d_e} \tag{5}$$

where  $H_t \in \mathbb{R}^{d_s \times d_e}$  is the hidden state at time step t,  $\bar{A}_t \in (0,1)^{d_s \times d_e}$  is a decay factor on the hidden state,  $\bar{B}_t \in \mathbb{R}^{d_s \times d_e}$  determines the hidden state update at step t, and  $C_t \in \mathbb{R}^{d_s}$  is a perchannel output scaling factor.

The key innovation of Mamba is making  $\bar{A}_t$ ,  $\bar{B}_t$  and  $C_t$  time variant (i.e., predicted from the input of the t-th token  $X_t$ ). Specifically, they can be formulated as:

$$\Delta_t = \text{Softplus}(X_t), \qquad B_t, C_t = \text{Linear}_4(X_t) \in \mathbb{R}^{d_s}, \mathbb{R}^{d_s}$$
 (6)

$$\bar{A}_t = \exp(\Delta_t \odot A), \qquad \bar{B}_t = \Delta_t \otimes B_t$$
 (7)

where  $\Delta_t \in \mathbb{R}_{>0}^{d_e}$  is a per-channel positive factor, while  $A \in \mathbb{R}_{<0}^{d_s \times d_e}$  is a negative learnable matrix, which makes the hidden state decay factor  $\bar{A}_t$  always smaller than 1 (i.e., continuously decaying the previous hidden state  $H_{t-1}$ ). Finally,  $\otimes$  denotes outer product operation.

**Attention Score of Mamba-based SSMs.** We can quantify the contribution of the j-th token's input  $X_j$  to the hidden state at time step i by expanding the recurrent computation in Eq. 4 across time steps:

$$H_i = \sum_{j=1}^i (\prod_{k=j+1}^i \bar{A}_k) \odot \bar{B}_j \odot X_j \tag{8}$$

therefore for the output at time step i:

$$Y_i = C_i^T \Sigma_{j=1}^i (\Pi_{k=j+1}^i \bar{A}_k) \odot \bar{B}_j \odot X_j \tag{9}$$

<span id="page-3-2"></span>
$$= \sum_{i=1}^{i} \alpha_{i,j} \odot X_j \tag{10}$$

where we have:

$$\alpha_{i,j} = C_i^T (\Pi_{k=j+1}^i \bar{A}_k) \odot \bar{B}_j \in \mathbb{R}^{d_e}$$
(11)

is the weighting factor of the contribution of the j-th token's input  $X_j$  to the i-th token's output  $Y_i$ , which has a similar function as the attention score in a Transformer model. Therefore, (Ali et al., 2024) proposes to regard  $\alpha_{i,j}$  as the attention score between the i-th token and the j-th token. In the next section, we analyze the attention patterns of different hidden state channels (i.e., along the  $d_e$  dimension). For the simplicity of notation, variables in the following sections only refers to the values at a single hidden state dimension unless otherwise noted.

