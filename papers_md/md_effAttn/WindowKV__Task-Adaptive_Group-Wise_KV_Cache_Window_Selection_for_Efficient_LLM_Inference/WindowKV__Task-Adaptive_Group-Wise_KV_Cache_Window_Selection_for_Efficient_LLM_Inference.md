## WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

Youhui Zuo<sup>1</sup> , Sibo Wei<sup>2</sup> , Chen Zhang<sup>1</sup> , Zhuorui Liu<sup>1</sup> , Wenpeng Lu<sup>2</sup> , Dawei Song1 (B) <sup>1</sup>Beijing Institute of Technology, Beijing, China <sup>2</sup>Qilu University of Technology (Shandong Academy of Sciences), Jinan, China (B) Corresponding author.

## Abstract

With the advancements in long-context inference capabilities of large language models (LLMs), the KV cache has become one of the foundational components. However, its substantial GPU memory consumption makes KV cache compression a key technique for enabling efficient LLM inference in industrial scenarios. While recent studies have focused on optimizing the memory occupied by the KV cache, they overlook two critical factors: preserving semantic coherence and considering task-specific characteristic during compression. To address these limitations, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to taskspecific characteristics, ensuring the retained KV cache captures continuous, essential context. Additionally, we introduce an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency. We rigorously evaluate WindowKV on the Long-Bench benchmark, and the results demonstrate that it maintains a performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Furthermore, our method also achieves state-of-the-art results in the Needle-in-a-Haystack evaluation, highlighting its effectiveness and robustness. [1](#page-0-0)

## 1 Introduction

Tasks requiring long-context understanding, such as long-text reading comprehension [\(Trivedi et al.,](#page-7-0) [2022\)](#page-7-0), in-context learning [\(Dong et al.,](#page-6-0) [2024b\)](#page-6-0), document summarization [\(Huang et al.,](#page-6-1) [2021\)](#page-6-1) and code completion [\(Zheng et al.,](#page-7-1) [2023\)](#page-7-1), have gained significant prominence in the era of LLMs. As a result, LLMs that are capable of processing extended context lengths have become increasingly

prevalent [\(Huang et al.,](#page-6-2) [2023\)](#page-6-2). For example, models like GPT-4 and DeepSeek-V3 support context lengths of up to 128K tokens, while Claude-3.5 and Yi extend this capability to 200K tokens [\(Achiam](#page-6-3) [et al.,](#page-6-3) [2023;](#page-6-3) [Liu et al.,](#page-6-4) [2024b;](#page-6-4) [Fu et al.,](#page-6-5) [2024\)](#page-6-5). However, the self-attention mechanism in transformer architectures exhibits quadratic complexity with respect the context length [\(Vaswani,](#page-7-2) [2017\)](#page-7-2), leading to significant increases in inference latency for long-context scenarios. One potential method to mitigate this latency is to cache the key and value (KV) states of previous tokens, thereby avoiding the recomputation of historical contexts [\(Wadding](#page-7-3)[ton et al.,](#page-7-3) [2013\)](#page-7-3). Nevertheless, as both the input context length and the number of layers increase, the memory required to store the KV states increases substantially [\(Luohe et al.,](#page-7-4) [2024\)](#page-7-4). For instance, storing a KV cache for 100K tokens in the LLaMA2-7B model [\(Touvron et al.,](#page-7-5) [2023\)](#page-7-5) demands over 50GB of memory, whereas a 2K token context requires less than 1GB [\(Wu et al.,](#page-7-6) [2024\)](#page-7-6). Overall, KV cache compression is essential to addressing issues such as memory demands, computational efficiency, energy consumption, and costs in LLMs, directly impacting their deployment and application effectiveness in industrial scenarios.

Recent studies have alleviated the aforementioned memory constraints by modifying attention architectures [\(Shazeer,](#page-7-7) [2019;](#page-7-7) [Ainslie et al.,](#page-6-6) [2023;](#page-6-6) [Liu et al.,](#page-6-7) [2024a\)](#page-6-7), or by implementing cross-layer sharing of the KV cache [\(Brandon et al.,](#page-6-8) [2024;](#page-6-8) [Sun et al.,](#page-7-8) [2024;](#page-7-8) [Wu and Tu,](#page-7-9) [2024\)](#page-7-9). However, these approaches require additional training. In contrast, another line of research has focused on compressing the KV cache during the inference phase. For example, some approaches evict the KV states of non-essential tokens under a fixed layer budget [\(Zhang et al.,](#page-7-10) [2023;](#page-7-10) [Xiao et al.,](#page-7-11) [2024;](#page-7-11) [Adnan et al.,](#page-6-9) [2024;](#page-6-9) [Ge et al.,](#page-6-10) [2024\)](#page-6-10). However, these methods overlook differences in attention sparsity between layers. PyramidInfer [\(Yang et al.,](#page-7-12)

<span id="page-0-0"></span><sup>1</sup>Our code is available at [GitHub.](https://github.com/optim996/WindowKV)

[2024b\)](#page-7-12) and PyramidKV [\(Cai et al.,](#page-6-11) [2024\)](#page-6-11) observe that dense attention is particularly prevalent in the lower layers, while sparse attention dominates in the higher layers [\(Wang et al.,](#page-7-13) [2023\)](#page-7-13). They propose allocating varying KV cache sizes across layers to maintain a pyramid-like structure.

However, the aforementioned methods individually select KV cache tokens, and the selected discrete tokens disrupt the semantic consistency of the context. This also contradicts human reading habits. When processing long texts, humans typically do not read token by token but rather process information in windows [\(Rayner,](#page-7-14) [1998\)](#page-7-14). Moreover, these methods employ a uniform compression strategy across all tasks, limiting their task-specific adaptability and overall performance. In fact, based on human experience, the information processing approaches for different tasks vary significantly. For instance, in a question-answering task, this can be seen as an information localization task [2](#page-1-0) , where the entire window is processed to capture comprehensive semantic details, thus facilitating accurate answer generation. In contrast, in a summarization task, the goal is information aggregation [3](#page-1-1) , where extract the most salient information from each window to generate a concise summary. The challenges outlined above motivate us to propose a task-adaptive KV cache window selection method. Additionally, to enhance computational efficiency, we integrated intra-group layer KV cache indices sharing strategy to better sustain the model's performance under constrained memory budgets.

In summary, our contributions are as follows:

- Different from previous KV cache compression methods that select discrete tokens, we introduce a task-adaptive window selection method, WindowKV. WindowKV dynamically compresses the KV cache based on taskspecific characteristics while preserving the semantic coherence within local windows.
- Additionally, we propose an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency.

• Extensive experiments are conducted on the LongBench and Needle-in-a-Haystack benchmarks. The results demonstrate that WindowKV achieves the highest number of stateof-the-art results across various backbone LLMs and KV cache configurations on Long-Bench, while also surpassing other baseline methods on the Needle-in-a-Haystack test.

## 2 Methodology

#### 2.1 Overview of KV Cache Compression

In autoregressive transformer-based LLMs, the generation of the i-th token requires the attention module to access the KV states of all preceding i − 1 tokens. To enhance the computational efficiency and avoid redundant calculations, these KV states are stored in the KV cache upon their initial computation. This caching mechanism significantly accelerates the inference process by eliminating the need for repeated computations. However, the KV cache can impose substantial memory demands, particularly for lengthy contexts. As a potential method, KV cache compression has been proposed to save memory while minimizing information loss as much as possible.

## 2.2 WindowKV

In this section, we introduce WindowKV, a novel approach that employs a window level KV cache selection method according to the specific requirements of the task, as shown in Figure [1](#page-2-0) (d). Unlike previous token level methods, our window selection method enhances semantic coherence in long-context inputs, by dynamically adapting to the task's characteristic, prioritizing relevant context, and closely mimicking human information processing. Additionally, to balance performance and efficiency, we integrate an intra-group layer KV cache indices sharing strategy.

## <span id="page-1-2"></span>2.2.1 Task-Adaptive Window Selection

In our method, we retain the last α tokens as an observation window, while the remaining context, referred to as the review context, is divided into multiple review windows. The observation window is utilized to identify important review windows for caching across all layers, tailored to the specific task.

For clarity, we illustrate the attention mechanism using a single head. In a standard LLM, this is

<span id="page-1-0"></span><sup>2</sup> Information Localization Task: Identify the critical paragraphs within the given context, and then answer the relevant questions based on the information provided in these critical paragraphs.

<span id="page-1-1"></span><sup>3</sup> Information Aggregation Task: Extract essential information from each paragraph and compile it into a cohesive summary.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: Comparison of WindowKV with state-of-the-art KV cache compression methods. (a) Full KV retains all tokens in the KV cache for each layer, with cache size growing linearly with input length. (b) H2O maintains a fixed cache size across layers, selecting tokens based on attention scores. (c) PyramidKV adopts a pyramid-shaped cache structure, allocating varying cache budgets to different layers. These methods uniformly apply token level selection strategies across all tasks. (d) WindowKV, in contrast, introduces a task-adaptive window selection method combined with intra-group layer KV cache indices sharing strategy, dynamically allocating group budgets across different groups.

computed as follows:

$$\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q} \cdot \mathbf{K}^{\mathsf{T}}}{\sqrt{d_k}}\right),\tag{1}$$

where  $d_k$  denotes the dimension of the key states.

To assess the importance of each token within the review context, we compute the attention score for each token based on the attention it receives from the observation window. This is formally expressed as:

$$t_j = \sum_{i \in [n-\alpha,n]} \mathbf{A}_{ij}, \quad j \in [0,n-\alpha], \quad (2)$$

where  $t_j$  represents the score of the j-th token in the review context, n denotes the input context length.

However, this token selection approach disrupts semantic coherence. To preserve semantic coherence and accommodate the variability in human reading patterns across different tasks, we further propose a task-adaptive window selection method. Specifically, in question-answering tasks, which can be viewed as **Information Localization**, humans must comprehend the semantic content of the entire review window to give accurate responses. In contrast, for tasks such as summarization and code completion, which can be viewed as **Information Aggregation**, humans only need to focus on the most critical and relevant context within individual review windows.

The review windows in the context can be repre-

sented as follows:

$$\mathbb{W}_k = \{t_j, \cdots, t_{j+\omega-1}\}, \quad k \in \left[1, \left\lceil \frac{n-\alpha}{\omega} \right\rceil \right],$$
(3)

where  $\omega$  denotes the review window size, window  $\mathbb{W}_k$  consists of tokens  $t_j, \dots, t_{j+\omega-1}$ , and  $\lceil \cdot \rceil$  denotes the ceiling function.

To facilitate this process, a task-adaptive classifier is trained, with detailed training procedures described in Appendix A.3 and A.5. Consequently, the scoring function for evaluating the importance of review windows in the context for a specific task is defined as follows:

$$s_k = \frac{1}{\min(p, \omega)} \cdot \text{sum}(\text{Top-}p(\mathbb{W}_k)), \quad (4)$$

where  $\operatorname{Top-}p(\mathbb{W}_k)=\{t_0',t_1',\cdots,t_{p-1}'\}$  represents the selection of the p tokens with the highest scores from the w tokens in the window. When  $p=\omega$ , it aligns with the information localization task. When  $p<\omega$ , the scenario aligns with the information aggregation task. The task type, identified by the task-adaptive classifier from the input context, is used to invoke the corresponding window selection method. Based on the allocated budget, a subset of high-scoring windows is retained from the review context to maintain the desired budget of KV cache. The detailed budget allocation strategy is described in Section 2.2.3.

However, performing review window selection at every layer is computationally expensive. Ma et al. (Ma et al., 2024) demonstrate that the attention scores of adjacent layers in LLMs exhibit similarity. Additionally, Liu et al. (Liu et al., 2025) proposed a layer-wise index reuse method under fixed layer budgets, which further validates the inter-layer similarity in LLMs. Inspired by the inter-layer similarity characteristics of LLMs, we propose the intra-group layer KV cache indices sharing strategy to optimize the trade-off between performance and efficiency in WindowKV.

## 2.2.2 Efficient Intra-Group Layer KV Cache Indices Sharing Strategy

To enhance the efficiency of review window selection, an intra-group layer KV cache indices sharing strategy is employed.

Assume that the transformer layers of a model are denoted as  $\mathbb{L}=\{l_0,l_1,\ldots,l_{m-1}\}$ , where m represents the number of layers in the model. The layers in  $\mathbb{L}$  are divided into  $H=\frac{m}{\gamma}$  groups, each containing  $\gamma$  layers. For a given group  $\mathbb{G}=\{l_g,l_{g+1},\ldots,l_{g+\gamma-1}\}$ , we apply the method from Section 2.2.1 to perform task-adaptive window selection on the first layer  $l_g$ , obtaining the KV cache indices  $\mathbb{I}_{l_g}$  to be retained for that layer. For the remaining layers in the group  $\{l_{g+1},\ldots,l_{g+\gamma-1}\}$ , they share the same KV cache indices  $\mathbb{I}_{l_g}$  as  $l_g$ .

By adopting this approach, the computational cost can be significantly reduced, as the window selection algorithm is executed only once per group.

#### <span id="page-3-0"></span>2.2.3 Dynamic Budget Allocation

Inspired by PyramidKV (Cai et al., 2024), we allocate budgets to each group using an arithmetic sequence. The total budget for all groups is defined as:

<span id="page-3-2"></span>
$$b^{\text{total}} = \sum_{h \in [0, H-1]} b^h, \tag{5}$$

where H represents the number of groups.

For all groups  $\{\mathbb{G}_0, \dots, \mathbb{G}_{H-1}\}$ , we first compute the budgets for the top group  $\mathbb{G}_{H-1}$  and the bottom group  $\mathbb{G}_0$  as:

$$b^{H-1} = \frac{b^{\mathrm{total}}}{\lambda \cdot H}$$
 and  $b^0 = \frac{2 \cdot b^{\mathrm{total}}}{H} - b^{H-1},$  (6)

where  $\lambda$  is a hyperparameter that controls the shape of the pyramid. The budgets for the remaining groups are calculated using the following equation:

<span id="page-3-3"></span>
$$b^{h} = b^{0} - \frac{b^{0} - b^{H-1}}{H-1} \times h.$$
 (7)

Finally, the budget for each group is averagely distributed across all layers within the group.

#### 3 Experiments

## 3.1 Experimental Setup

#### 3.1.1 Backbone LLMs & Benchmarks

Due to computational constraints, we evaluate WindowKV against baseline methods using state-of-the-art open-source LLMs, specifically Qwen2.5-1.5B-Instruct (Yang et al., 2024a) and LLaMA3-8B-Instruct (Touvron et al., 2023). LongBench (Bai et al., 2024) and Needle-in-a-Haystack (Fu et al., 2024) are two widely used benchmarks for evaluating KV cache compression methods. LongBench is specifically designed to assess the ability of language models to handle extended contexts. Needle-in-a-Haystack evaluates a model's ability to locate key information within long input sequences, testing the in-context retrieval capabilities of LLMs across various KV cache compression methods.

#### 3.1.2 Baseline Methods

We compare WindowKV with three state-of-theart methods: **StreamingLLM** (**SLM**) (Xiao et al., 2024), **Heavy Hitter Oracle** (**H2O**) (Zhang et al., 2023) and **PyramidKV** (**PKV**) (Cai et al., 2024), as well as the use of full KV. Among these, SLM and H2O allocate a uniform KV cache size across all layers, while PKV assigns different KV cache sizes to different layers. Each method adopts a distinct KV cache compression strategy. For more detailed information, please refer to Appendix A.2.

# 3.2 Analysis on Intra-Group Layer KV Cache Indices Sharing Strategy

<span id="page-3-1"></span>![](_page_3_Figure_21.jpeg)

Figure 2: Similarity of Intra-Group Layer KV Cache Indices.

In this section, we present some experiments to further analyze and validate the feasibility of

<span id="page-4-0"></span>

| Method | Information Localization Task |        |       |                   |         | Information Aggregation Task |               |           |                   |              |           |           |          |          |          |                  |       |
|--------|-------------------------------|--------|-------|-------------------|---------|------------------------------|---------------|-----------|-------------------|--------------|-----------|-----------|----------|----------|----------|------------------|-------|
|        | Single-Document QA            |        |       | Multi-Document QA |         |                              | Summarization |           | Few-shot Learning |              |           | Synthetic |          | Code     |          | Avg.             |       |
|        | MINOA                         | Qasper | MF-en | HotpotQA          | 2WikiMQ | Musique                      | GovReport     | QMSum     | MultiNews         | TREC         | TriviaQA  | SAMSun    | PCount   | PRE      | Loc      | $RB_{\cdot}^{P}$ | g.    |
|        | Fl                            | F1     | F1    | F1                | F1      | F1                           | R-L           | R-L       | R-L               | Acc (CLS)    |           | R-L       | Acc (EM) | Acc (EM) | Edit Sin | Edit Sin         | ı     |
|        |                               |        |       |                   | Qv      | ven2.5-1.5                   | B-Instruct    | , Max Inj | out Length        | = 15500, K   | V Size =  | = Full    |          |          |          |                  |       |
| FKV    | 17.51                         | 25.44  | 41.80 | 39.33             | 32.89   | 20.05                        | 28.17         | 20.49     | 21.07             | 68.00        | 81.70     | 39.05     | 2.17     | 19.50    | 37.02    | 43.90            | 33.63 |
|        |                               |        |       |                   | Qw      | en2.5-1.5                    | B-Instruct,   | Max Inp   | ut Length         | = 15500, K   | V Size =  | 2048      |          |          |          |                  |       |
| SLM    | 14.41                         | 20.61  | 29.97 | 31.27             | 31.28   | 13.38                        | 20.44         | 18.81     | 20.63             | 63.00        | 81.00     | 37.72     | 3.50     | 12.00    | 37.26    | 42.94            | 29.89 |
| H2O    | 17.06                         | 23.58  | 38.14 | 36.83             | 31.94   | 17.66                        | 24.07         | 18.62     | 21.18             | 67.50        | 78.21     | 35.73     | 3.00     | 11.06    | 36.27    | 40.66            | 31.34 |
| PKV    | 16.93                         | 21.66  | 39.32 | 40.23             | 32.08   | 19.46                        | 20.64         | 19.36     | 20.80             | 67.00        | 80.69     | 37.75     | 2.56     | 18.50    | 36.11    | 41.71            | 32.18 |
| Ours   | 17.50                         | 25.39  | 41.83 | 39.86             | 31.44   | 20.99                        | 19.09         | 20.71     | 20.09             | 67.50        | 80.99     | 37.13     | 3.00     | 17.00    | 36.98    | 44.53            | 32.75 |
|        |                               |        |       |                   | Qw      | en2.5-1.5                    | B-Instruct,   | Max Inp   | ut Length         | = 15500, K   | V Size =  | 1024      |          |          |          |                  |       |
| SLM    | 12.01                         | 13.50  | 24.69 | 29.30             | 29.68   | 11.07                        | 17.17         | 16.80     | 19.20             | 56.50        | 78.34     | 37.79     | 3.00     | 8.00     | 36.82    | 40.95            | 27.18 |
| H2O    | 16.20                         | 21.02  | 33.60 | 35.96             | 30.69   | 15.62                        | 22.42         | 18.68     | 20.37             | 67.00        | 75.45     | 32.98     | 3.00     | 11.05    | 34.87    | 38.67            | 29.85 |
| PKV    | 16.79                         | 19.41  | 38.01 | 39.25             | 32.23   | 17.12                        | 18.06         | 18.66     | 19.14             | 66.00        | 80.32     | 36.78     | 2.62     | 14.08    | 36.67    | 39.11            | 30.89 |
| Ours   | 16.63                         | 22.12  | 42.16 | 39.49             | 31.56   | 19.63                        | 15.00         | 19.61     | 18.71             | 67.50        | 79.40     | 36.77     | 3.00     | 14.50    | 37.48    | 41.79            | 31.58 |
|        |                               |        |       |                   | Q۱      | ven2.5-1.5                   | B-Instruct    | , Max Inj | out Length        | t = 15500, K | (V Size = | = 512     |          |          |          |                  |       |
| SLM    | 11.35                         | 11.89  | 23.58 | 27.79             | 27.62   | 11.05                        | 14.67         | 16.75     | 16.68             | 50.50        | 76.10     | 37.89     | 3.00     | 6.00     | 35.60    | 39.23            |       |
| H2O    | 15.37                         |        | 30.45 | 36.39             | 29.42   | 16.54                        | 20.56         | 17.82     | 19.60             | 66.00        | 74.40     | 31.45     | 3.00     | 6.92     | 33.28    | 34.81            |       |
| PKV    | 16.98                         | 17.84  | 36.25 | 37.87             | 30.61   | 15.82                        | 16.17         | 17.88     | 16.89             | 64.50        | 78.80     | 35.00     | 2.62     | 12.50    | 34.72    | 36.69            | 29.45 |
| Ours   | 16.32                         | 17.57  | 40.05 | 38.26             | 30.03   | 17.18                        | 12.58         | 19.18     | 15.79             | 63.50        | 78.85     | 36.96     | 3.00     | 11.00    | 36.89    | 41.59            | 29.92 |
|        |                               |        |       |                   |         |                              |               |           |                   | = 7950 , K   |           |           |          |          |          |                  |       |
| FKV    | 25.59                         | 32.04  | 39.67 | 43.61             | 35.29   | 21.30                        | 28.64         | 23.15     | 26.69             | 71.50        | 90.48     | 42.59     | 4.86     | 69.75    | 56.84    | 52.16            | 41.51 |
|        |                               |        |       |                   | LI      | LaMA3-8I                     | 3-Instruct,   | Max Inp   | ut Length         | = 7950 , KV  | V Size =  | 2048      |          |          |          |                  |       |
| SLM    | 24.00                         | 24.00  | 30.18 | 39.03             | 31.59   | 17.82                        | 24.92         | 21.59     | 26.30             | 68.00        | 89.62     | 41.65     | 5.58     | 69.67    | 58.78    | 56.13            | 39.30 |
| H2O    | 26.07                         | 28.95  | 37.19 | 42.62             | 32.97   | 19.77                        | 27.40         | 22.71     | 26.65             | 71.00        | 90.93     | 42.13     | 5.88     | 70.00    | 57.52    | 55.42            | 41.08 |
| PKV    | 25.41                         |        | 38.62 | 43.37             | 35.83   | 21.97                        | 26.94         | 23.09     | 26.10             | 70.50        | 90.56     | 42.37     | 5.13     | 69.75    | 57.88    |                  | 41.32 |
| Ours   | 26.61                         | 29.27  | 38.95 | 44.26             | 34.76   | 21.17                        | 25.47         | 22.89     | 25.86             | 71.50        | 90.48     | 41.29     | 5.43     | 70.00    | 58.08    | 55.63            | 41.35 |
|        |                               |        |       |                   | LI      | LaMA3-8I                     | 3-Instruct,   | Max Inp   | ut Length         | = 7950 , KV  | V Size =  | 1024      |          |          |          |                  |       |
| SLM    | 20.94                         |        | 29.49 | 39.29             | 29.44   | 16.01                        | 23.27         | 21.12     | 25.95             | 67.00        | 84.44     | 41.33     | 5.87     | 70.00    | 58.16    | 53.32            | 37.50 |
| H2O    |                               | 27.26  | 35.05 | 42.71             | 30.48   | 18.91                        | 26.28         | 22.80     | 26.07             | 70.50        | 91.21     | 41.07     | 5.55     | 69.53    | 57.77    | 54.85            | 40.32 |
| PKV    | 26.02                         | 27.31  | 37.10 | 43.85             | 33.86   | 21.18                        | 24.71         | 23.21     | 25.26             | 70.00        | 90.56     | 41.28     | 5.58     | 69.75    | 57.33    | 53.29            | 40.64 |
| Ours   | 25.48                         | 24.04  | 39.41 | 43.42             | 32.70   | 20.96                        | 23.90         | 22.41     | 25.14             | 69.00        | 89.84     | 40.81     | 5.68     | 70.00    | 58.58    | 56.79            | 40.51 |
|        |                               |        |       |                   |         | LaMA3-8                      | B-Instruct    |           |                   | = 7950 , K   |           |           |          |          |          |                  |       |
| SLM    | 20.70                         | 12.14  | 22.08 | 35.14             | 27.06   | 15.54                        | 21.01         | 20.92     | 23.84             | 60.50        | 83.49     | 40.32     | 5.79     | 68.22    | 58.59    | 53.36            |       |
| H2O    | 23.29                         | 20.89  | 31.38 | 40.47             | 30.30   | 17.50                        | 24.71         | 21.80     | 25.78             | 67.50        | 90.67     | 39.67     | 5.81     | 68.49    | 59.26    | 54.63            |       |
| PKV    | 24.25                         | 23.19  | 36.17 | 43.06             | 32.13   | 20.41                        | 23.31         | 22.48     | 24.23             | 70.00        | 90.61     | 40.79     | 5.83     | 70.00    | 57.13    | 53.70            | 39.83 |
| Ours   | 23.74                         | 21.30  | 37.27 | 44.03             | 32.28   | 20.26                        | 21.42         | 22.19     | 23.28             | 63.50        | 88.18     | 40.55     | 5.42     | 70.00    | 58.65    | 57.25            | 39.33 |

Table 1: Performance comparison of WindowKV (Ours) with StreamingLLM (SLM), H2O, PyramidKV (PKV), and FullKV (FKV) on LongBench. WindowKV achieves the highest number of state-of-the-art results across various backbone LLMs and KV cache sizes. The best performance is highlighted in bold text.

intra-group layer KV cache indices sharing strategy. First, we sampled multiple data points from each dataset in LongBench (Bai et al., 2024) and conducted a comparative analysis on the similarity among intra-group layer KV cache indices. Using the budget allocation strategy outlined in Section 2.2.3, we divided the 32 layers of LLaMA3-8B-Instruct into 4 groups, assigning distinct budgets to each group. After averagely distributing the budget across layers within each group, we applied the window selection method described in Section 2.2.1 to identify the retained windows and their corresponding KV cache indices for each layer. We then computed the Jaccard similarity of the KV cache indices between layers within the same group. The results, illustrated in Figure 2, are presented in a heatmap where each cell represents the similarity between the retained KV cache indices of two layers within the same group. The experimental findings reveal that the KV cache indices of layers within the same group in WindowKV exhibit significant similarity, thereby validating the effec-

tiveness of our KV cache indices sharing strategy. For further analysis, please refer to Appendix A.4.

#### 3.3 Main Results

The evaluation results for LongBench are presented in Table 1. We report the performance of Owen2.5-1.5B-Instruct and LLaMA3-8B-Instruct across three KV cache sizes: 512, 1024, and 2048. As shown in Table 1, the datasets in LongBench are categorized into two types: information localization task and information aggregation task. WindowKV achieves the highest number of state-ofthe-art results across various backbone LLMs and KV cache configurations, demonstrating its superior adaptability and robustness across a wide range of tasks. The performance of WindowKV on some datasets is comparable to or slightly inferior to that of PyramidKV. This may be attributed to the fact that PyramidKV's token selection method, although disrupting semantic coherence, is able to consistently identify important tokens. In contrast, WindowKV maintains semantic coherence within

a fixed-size window but may lose crucial tokens, making it challenging to achieve optimal performance across all datasets. Therefore, exploring adaptive review window size is a critical issue in our future work. For implementation details, please refer to Appendix A.3.

#### 3.4 Results on Needle-in-a-Haystack

We use Needle-in-a-Haystack to evaluate the long-context retrieval capabilities of LLMs. The Rouge-1 F1 metric is applied to assess the accuracy of the retrieved information. Several KV cache compression methods are evaluated. Figure 3 presents the benchmark results for LLaMA3-8B-Instruct, with the context length set to 8k tokens, which corresponds to the maximum length on the horizontal axis. The vertical axis represents the depth percentage. The results demonstrate that WindowKV outperforms other KV cache compression methods.

<span id="page-5-0"></span>![](_page_5_Figure_3.jpeg)

Figure 3: Needle-in-a-Haystack for LLaMA3-8B-Instruct with 512 KV cache size at 8K context length.

#### 3.5 Throughput Test

Table 2 compares the throughput and latency of Vanilla, WindowKV, and WindowKV + Classifier. Compared to Vanilla, the Vanilla + WindowKV + Classifier configuration achieves a throughput increase of 117 tokens/s and a latency reduction of 0.17 ms/token. Moreover, the results indicate that incorporating the classifier does not significantly degrade efficiency.

#### 3.6 Discussion and Analysis

This section examines the necessity of task-adaptive window selection method. When the task-adaptive classifier identifies the input context as an information localization task, an information localization-based window selection method

<span id="page-5-1"></span>

| Model                           | Throughput (token/s) | Latency (ms/token) |
|---------------------------------|----------------------|--------------------|
| Vanilla                         | 764                  | 1.31               |
| Vanilla + WindowKV              | 894                  | 1.12               |
| Vanilla + WindowKV + Classifier | 881                  | 1.14               |

Table 2: Throughput test results. Vanilla refers to LLaMA3-8B-Instruct. The prefill length and generation length are 7,950 and 242, respectively. The experiment is conducted on a single A100 40G GPU with a KV cache size of 512 and repeated 10 times, with the results averaged.

<span id="page-5-2"></span>![](_page_5_Figure_11.jpeg)

Figure 4: Impact of task-adaptive window selection and review window size on WindowKV performance.

outperforms an information aggregation-based approach, and vice versa, as shown in Figure 4. Additionally, the figure illustrates the impact of different review window sizes on WindowKV's performance. In our experiments, the review window size varies among {8, 16, 32, 64, 128}. The Qwen2.5-1.5B-Instruct model achieves optimal performance across all tasks with a window size of 32. For the LLaMA3-8B-Instruct model, optimal performance is attained with a window size of 8 for information localization tasks and a window size of 16 for information aggregation tasks.

#### 4 Conclusion

In this work, we present WindowKV, a method designed to address two issues in existing methods: preserving semantic coherence and considering task-specific characteristics during compression. Evaluations on the LongBench demonstrate that WindowKV achieves performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Moreover, it outperforms other baselines in the Needle-in-a-Haystack test.

## Ethical Considerations

We highly prioritize ethical considerations and strictly adhere to the ACL Ethics Policy. In this paper, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to task-specific characteristics, ensuring the retained KV cache captures continuous, essential context. During the inference phase, WindowKV uses only 12% of the original KV cache, significantly reducing memory requirements and improving inference speed. The methods and resources presented in this paper are open-source and widely used by researchers in the field of KV cache compression. The research results and conclusions presented in this paper are accurate and objective reports.

## References

- <span id="page-6-3"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. *arXiv preprint arXiv:2303.08774*.
- <span id="page-6-9"></span>Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant Nair, Ilya Soloveychik, and Purushotham Kamath. 2024. Keyformer: KV cache reduction through key tokens selection for efficient generative inference. In *Proceedings of the 7th Machine Learning and Systems*, pages 114–127.
- <span id="page-6-6"></span>Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 4895– 4901.
- <span id="page-6-12"></span>Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A bilingual, multitask benchmark for long context understanding. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, pages 3119– 3137.
- <span id="page-6-8"></span>William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan Kelly. 2024. Reducing transformer key-value cache size with cross-layer attention. In *Proceedings of the 38th Conference on Neural Information Processing Systems*, pages 1–31.
- <span id="page-6-11"></span>Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao

- Chang, Junjie Hu, et al. 2024. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. *arXiv preprint arXiv:2406.02069*.
- <span id="page-6-13"></span>Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. 2024a. Get More with LESS: Synthesizing recurrence with KV cache compression for efficient LLM inference. In *Proceedings of the 41st International Conference on Machine Learning*, pages 1–16.
- <span id="page-6-0"></span>Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. 2024b. A survey on in-context learning. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 1107–1128.
- <span id="page-6-5"></span>Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024. Data engineering for scaling language models to 128k context. In *Proceedings of the 41st International Conference on Machine Learning*, pages 1–10.
- <span id="page-6-10"></span>Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024. Model Tells You What to Discard: Adaptive KV cache compression for LLMs. In *Proceedings of the 12th International Conference on Learning Representations*, pages 1– 14.
- <span id="page-6-1"></span>Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. 2021. Efficient attentions for long document summarization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1419–1436.
- <span id="page-6-2"></span>Yunpeng Huang, Jingwei Xu, Junyu Lai, Zixu Jiang, Taolue Chen, Zenan Li, Yuan Yao, Xiaoxing Ma, Lijuan Yang, Hao Chen, et al. 2023. Advancing Transformer Architecture in Long-Context Large Language Models: A comprehensive survey. *arXiv preprint arXiv:2311.12351*.
- <span id="page-6-14"></span>Chen Liang, Simiao Zuo, Qingru Zhang, Pengcheng He, Weizhu Chen, and Tuo Zhao. 2023. Less is More: Task-aware layer-wise distillation for language model compression. In *Proceedings of the 40th International Conference on Machine Learning*, pages 20852–20867.
- <span id="page-6-7"></span>Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024a. DeepSeek-v2: A strong, economical, and efficient mixture-of-experts language model. *arXiv preprint arXiv:2405.04434*.
- <span id="page-6-4"></span>Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024b. DeepSeek-v3 technical report. *arXiv preprint arXiv:2412.19437*.

- <span id="page-7-18"></span>Ruikang Liu, Haoli Bai, Haokun Lin, Yuening Li, Han Gao, Zhengzhuo Xu, Lu Hou, Jun Yao, and Chun Yuan. 2024c. IntactKV: Improving large language model quantization by keeping pivot tokens intact. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 7716–7741.
- <span id="page-7-16"></span>Xiang Liu, Zhenheng Tang, Peijie Dong, Zeyu Li, Bo Li, Xuming Hu, and Xiaowen Chu. 2025. ChunkKV: Semantic-preserving KV cache compression for efficient long-context LLM inference. *arXiv preprint arXiv:2502.00299*.
- <span id="page-7-19"></span>Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time. In *Proceedings of the 37th Conference on Neural Information Processing Systems*, pages 1–23.
- <span id="page-7-4"></span>Shi Luohe, Hongyi Zhang, Yao Yao, Zuchao Li, and Hai Zhao. 2024. Keep the Cost Down: A review on methods to optimize LLM's KV-cache consumption. In *Proceedings of the 1st Conference on Language Modeling*, pages 1–19.
- <span id="page-7-15"></span>Da Ma, Lu Chen, Situo Zhang, Yuxun Miao, Su Zhu, Zhi Chen, Hongshen Xu, Hanqi Li, Shuai Fan, Lei Pan, et al. 2024. Compressing KV cache for longcontext LLM inference with inter-layer attention similarity. *arXiv preprint arXiv:2412.02252*.
- <span id="page-7-20"></span>Xiao Pu, Tianxing He, and Xiaojun Wan. 2024. Style-Compress: An LLM-based prompt compression framework considering task-specific styles. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 14533–14549.
- <span id="page-7-14"></span>Keith Rayner. 1998. Eye movements in reading and information processing: 20 years of research. *Psychological bulletin*, page 372.
- <span id="page-7-21"></span>Shivam Shandilya, Menglin Xia, Supriyo Ghosh, Huiqiang Jiang, Jue Zhang, Qianhui Wu, and Victor Rühle. 2024. TACO-RL: Task aware prompt compression optimization with reinforcement learning. *arXiv preprint arXiv:2409.13035*.
- <span id="page-7-7"></span>Noam Shazeer. 2019. Fast Transformer Decoding: One write-head is all you need. *arXiv preprint arXiv:1911.02150*.
- <span id="page-7-8"></span>Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. 2024. You Only Cache Once: Decoder-decoder architectures for language models. *arXiv preprint arXiv:2405.05254*.
- <span id="page-7-5"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

- <span id="page-7-0"></span>Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop questions via single-hop question composition. *Transactions of the Association for Computational Linguistics*, 10:539–554.
- <span id="page-7-2"></span>A Vaswani. 2017. Attention is all you need. In *Proceedings of the 31st Conference on Neural Information Processing Systems*, pages 1–11.
- <span id="page-7-3"></span>Daniel Waddington, Juan Colmenares, Jilong Kuang, and Fengguang Song. 2013. KV-Cache: A scalable high-performance web-object cache for manycore. In *IEEE/ACM 6th International Conference on Utility and Cloud Computing*, pages 123–130.
- <span id="page-7-13"></span>Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. 2023. Label Words are Anchors: An information flow perspective for understanding in-context learning. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 9840–9855.
- <span id="page-7-9"></span>Haoyi Wu and Kewei Tu. 2024. Layer-condensed KV cache for efficient inference of large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, pages 11175–11188.
- <span id="page-7-6"></span>Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2024. Retrieval head mechanistically explains long-context factuality. *arXiv preprint arXiv:2404.15574*.
- <span id="page-7-11"></span>Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In *Proceedings of the 12th International Conference on Learning Representations*, pages 1–21.
- <span id="page-7-17"></span>An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024a. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*.
- <span id="page-7-12"></span>Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. 2024b. PyramidInfer: Pyramid KV cache compression for high-throughput LLM inference. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 3258–3270.
- <span id="page-7-10"></span>Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. 2023. H2O: Heavy-hitter oracle for efficient generative inference of large language models. In *Proceedings of the 37th Conference on Neural Information Processing Systems*, pages 34661–34710.
- <span id="page-7-1"></span>Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Lei Shen, Zihan Wang, Andi Wang, Yang Li, et al. 2023. CodeGeeX: A pre-trained model for code generation with multilingual benchmarking on humaneval-x. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 5673–5684.

## A Appendix

## A.1 Related Work

#### KV Cache Efficiency in Training and Inference

Extensive previous research has explored methods for modifying transformer architecture to reduce KV cache size, including head-wise [\(Shazeer,](#page-7-7) [2019;](#page-7-7) [Ainslie et al.,](#page-6-6) [2023;](#page-6-6) [Liu et al.,](#page-6-7) [2024a\)](#page-6-7) and layer-wise approaches [\(Brandon et al.,](#page-6-8) [2024;](#page-6-8) [Sun](#page-7-8) [et al.,](#page-7-8) [2024;](#page-7-8) [Wu and Tu,](#page-7-9) [2024\)](#page-7-9). However, such modifications often require substantial computational resources for model retraining, making them less practical in settings with limited GPU resources. An alternative line of research focuses on compressing the KV cache during the inference phase. Related approaches in this area include low-rank decomposition [\(Dong et al.,](#page-6-13) [2024a\)](#page-6-13), quantization [\(Liu](#page-7-18) [et al.,](#page-7-18) [2024c\)](#page-7-18), token selection methods [\(Liu et al.,](#page-7-19) [2023;](#page-7-19) [Zhang et al.,](#page-7-10) [2023;](#page-7-10) [Xiao et al.,](#page-7-11) [2024;](#page-7-11) [Yang](#page-7-12) [et al.,](#page-7-12) [2024b;](#page-7-12) [Cai et al.,](#page-6-11) [2024\)](#page-6-11). Among these methods, StreamingLLM [\(Xiao et al.,](#page-7-11) [2024\)](#page-7-11) maintains a fixed KV cache size by retaining the KV states of both the initial and most recent tokens. Building on this idea, H2O [\(Zhang et al.,](#page-7-10) [2023\)](#page-7-10) and Scissorhands [\(Liu et al.,](#page-7-19) [2023\)](#page-7-19) use fixed-length KV cache size, selectively preserving the KV states of important tokens while evicting less critical ones. Extending this line of research, PyramidInfer [\(Yang](#page-7-12) [et al.,](#page-7-12) [2024b\)](#page-7-12) and PyramidKV [\(Cai et al.,](#page-6-11) [2024\)](#page-6-11) highlight that using the same KV cache size across all layers often leads to suboptimal performance. To address this issue, they propose a KV cache budget allocation strategy that assigns varying cache budgets to different layers, forming a pyramid structure. Despite these advances, most existing methods rely on token-by-token selection of KV states, disrupting the semantic coherence of the context. This deviates from human reading behavior, where information is retrieved at the window level, rather than at the token level, particularly in long-context scenarios. In this work, we focus primarily on layer-wise window selection methods.

#### Task-Adaptive Compression Methods

In the field of model compression, TED [\(Liang](#page-6-14) [et al.,](#page-6-14) [2023\)](#page-6-14) addresses the challenge of layer-wise distillation by introducing task-aware filters that align the hidden representations of the student and teacher models. These filters extract task-relevant knowledge, reducing the knowledge gap and enabling the student model to better adapt to the target task. In the field of prompt compression, Style-Compress [\(Pu et al.,](#page-7-20) [2024\)](#page-7-20) argues that dif-

ferent tasks favor compressed prompts in distinct styles (e.g., extractive or abstractive), and optimizing compression performance requires identifying the most effective style for each task. Building on this insight, they introduce Style-Compress, a lightweight framework that enables smaller models to compress prompts for larger models across various downstream tasks without requiring additional training. Moreover, TACO-RL [\(Shandilya](#page-7-21) [et al.,](#page-7-21) [2024\)](#page-7-21) critiques existing compression techniques for relying on suboptimal metrics or treating the task as task-agnostic. It proposes a novel taskaware method using reinforcement learning with task-specific rewards, guided by the lightweight REINFORCE algorithm. However, task-adaptive approaches in KV cache compression field remain unexplored.

#### <span id="page-8-0"></span>A.2 Baselines

SLM maintains efficient long-context modeling by enabling LLMs trained with finite attention windows to generalize to infinite sequence lengths without fine-tuning. It leverages the attention sink phenomenon, where preserving the KV states of initial tokens largely restores the performance. In our experiments, for consistency with other methods, SLM retains the KV cache for the most recent α tokens and the initial b − α tokens, where b denotes the per-layer KV cache size.

H2O enhances KV cache efficiency by dynamically balancing the retention of recent tokens and Heavy Hitter (H2) tokens. It is based on the observation that a small subset of tokens contributes to most of the attention scores. H2O maintains the KV cache for the most recent tokens and the identified H2 tokens, where the eviction policy is guided by average attention scores computed across all queries.

PKV enhances KV cache management by dynamically adjusting the cache size across layers, leveraging the pyramidal information funneling effect in LLMs. PKV allocates more KV cache to lower layers and less to higher layers, deviating from previous approaches that use a uniform cache size. Furthermore, instead of aggregating attention across all queries, PKV captures attention signals based on patterns from instruction tokens, enabling more targeted and efficient compression.

Full KV (FKV) serves as the upper bound. It stores all keys and values for every token at every layer. All other methods need to be compared with Full KV.

#### <span id="page-9-0"></span>A.3 Implementation Details

In our method, for all tasks in LongBench, we use the prompts recommended by LongBench and follow its standard evaluation metrics [\(Bai et al.,](#page-6-12) [2024\)](#page-6-12). To eliminate variability introduced by sampling-based decoding, we employ greedy decoding for answer generation in both the Qwen2.5- 1.5B-Instruct [\(Yang et al.,](#page-7-17) [2024a\)](#page-7-17) and the LLaMA3- 8B-Instruct [\(Touvron et al.,](#page-7-5) [2023\)](#page-7-5) model. Specifically, for Qwen2.5-1.5B-Instruct [\(Yang et al.,](#page-7-17) [2024a\)](#page-7-17), the number of shared layers is set to 7, and the review window size to 32. The observation window size for the information localization and aggregation tasks are 4 and 16, respectively. Due to computational constraints, the maximum input length for Qwen2.5-1.5B-Instruct is limited to 15,500 tokens. For LLaMA3-8B-Instruct [\(Tou](#page-7-5)[vron et al.,](#page-7-5) [2023\)](#page-7-5), the number of shared layers is set to 8. For the information localization task, the review window size and observation window size are 8 and 16. For the information aggregation task, the review window size and observation window size are 16 and 32. λ is used to control the shape of the pyramid, and it is 14 for all experiments. Additionally, the classifier is trained on a dataset created by us, which consists of 9,551 samples divided into training, validation, and test sets with a ratio of 8:1:1. The task-adaptive classifier is based on the bert-base-cased model and is trained with the following hyperparameters: batch size = 16, learning rate = 1e-6, dropout rate = 0.5, and 10 epochs. Experiments are conducted using 8× A100 GPUs with 40 GB of memory.

## <span id="page-9-2"></span>A.4 Supplementary Analysis on Intra-Group Layer KV Cache Indices Sharing

We conducted an experimental evaluation of the performance of WindowKV with various shared layer configurations on the Qwen2.5-1.5B-Instruct and LLaMA3-8B-Instruct models, as detailed in Table [3.](#page-9-3) The results indicate that performance variations are negligible when the number of shared layers is set to 1, 4, or 7 for Qwen2.5-1.5B-Instruct. However, a significant decline in performance is observed when the number of shared layers is increased to 14 for Qwen2.5-1.5B-Instruct. Moreover, according to Equation [\(5\)](#page-3-2)-[\(7\)](#page-3-3), under a fixed total budget, as the number of shared layers increases, the budget allocated to each group becomes more evenly distributed. Specifically, when the number of shared layers is set to 1, the budget

distribution is lopsided, resulting in more budget allocated to the earlier layers and significantly less to the later layers. Conversely, when the number of shared layers is set to 14 for Qwen2.5-1.5B-Instruct (and 16 for LLaMA3-8B-Instruct), the budget becomes overly even, disrupting the pyramid-shaped distribution. To balance performance and computational efficiency, we opted for 7 shared layers for Qwen2.5-1.5B-Instruct (and 8 for LLaMA3- 8B-Instruct). This configuration ensures both the preservation of the pyramid-shaped distribution and a evenly budget allocation across layers.

<span id="page-9-3"></span>

| WindowKV, KV Size = 2048 | LongBench Avg Score                                  |  |  |  |  |  |  |
|--------------------------|------------------------------------------------------|--|--|--|--|--|--|
|                          | γ = 1<br>γ = 4<br>γ = 7<br>γ = 8<br>γ = 14<br>γ = 16 |  |  |  |  |  |  |
| Qwen2.5-1.5B-Instruct    | 32.13 32.40 32.75<br>-<br>27.83<br>-                 |  |  |  |  |  |  |
| LLaMA3-8B-Instruct       | 40.93 40.78<br>-<br>41.35<br>-<br>40.67              |  |  |  |  |  |  |

Table 3: LongBench performance with different layersharing scales, where γ denotes the number of layers shared in each group.

### <span id="page-9-1"></span>A.5 Effect of the Task-adaptive Classifier

The task-adaptive classifier analyzes the input context to determine whether it corresponds to an information localization or aggregation task. The evaluation results for LongBench [\(Bai et al.,](#page-6-12) [2024\)](#page-6-12), as illustrated in Table [4,](#page-9-4) indicate that the task-adaptive classifier achieves high accuracy with simple finetuning.

<span id="page-9-4"></span>

| Model           | LongBench |        |       |  |  |  |  |
|-----------------|-----------|--------|-------|--|--|--|--|
|                 | Acc       | Recall | F1    |  |  |  |  |
| bert-base-cased | 92.69     | 95.19  | 94.75 |  |  |  |  |

Table 4: Classifier Test Result.

#### A.6 Limitations

This study is limited to investigating layer-wise KV cache compression and does not explore headwise approaches, which represent another highly active and promising research direction. Future work should extend this research by investigating head-wise compression techniques. Additionally, while the current study focuses on long-context input scenarios with compression applied exclusively during the prefilling phase, subsequent research could expand the scope to include KV cache compression in long-output generation scenarios.