# MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT

Omkar Thawakar<sup>1</sup>\* , Ashmal Vayani<sup>1</sup>\* , Salman Khan<sup>1</sup>,<sup>2</sup> , Hisham Cholakal<sup>1</sup> , Rao M. Anwer<sup>1</sup>,<sup>3</sup> , Michael Felsberg<sup>5</sup> , Tim Baldwin<sup>1</sup>,<sup>4</sup> , Eric P. Xing<sup>1</sup> , Fahad Shahbaz Khan<sup>1</sup>,<sup>5</sup>

<sup>1</sup>Mohamed bin Zayed University of AI, <sup>2</sup>Australian National University, <sup>3</sup>Aalto University <sup>4</sup>The University of Melbourne, <sup>5</sup>Linköping University

## Abstract

'*Bigger the better*' has been the predominant trend in recent Large Language Models (LLMs) development. However, LLMs do not suit well for scenarios that require on-device processing, energy efficiency, low memory footprint, and response efficiency. These requisites are crucial for privacy, security, and sustainable deployment. This paper explores the '*less is more*' paradigm by addressing the challenge of designing accurate yet efficient Small Language Models (SLMs) for resource constrained devices. Our primary contribution is the introduction of an accurate and fully transparent open-source 0.5 billion (0.5B) parameter SLM, named *MobiLlama*, catering to the specific needs of resource-constrained computing with an emphasis on enhanced performance with reduced resource demands. *MobiLlama* is a SLM design that initiates from a larger model and applies a careful parameter sharing scheme to reduce both the pre-training and the deployment cost. Our work strives to not only bridge the gap in open-source SLMs but also ensures full transparency, where complete training data pipeline, training code, model weights, and over 300 checkpoints along with evaluation codes is available at : [https://github.com/](https://github.com/mbzuai-oryx/MobiLlama) [mbzuai-oryx/MobiLlama](https://github.com/mbzuai-oryx/MobiLlama).

# 1 Introduction

Recent years have witnessed a tremendous surge in the development of Large Language Models (LLMs) with the emergence of prominent closedsource commercial models such as ChatGPT, Bard, and Claude. These LLMs exhibit surprising capabilities, typically called emergent abilities, towards solving complex tasks. Most existing popular LLMs follow a similar trend that bigger is always better, where scaling model size or data size typically provides improved model capacity and performance on downstream tasks. For instance,

the recent Llama-2 70 billion (70B) model [\(Tou](#page-10-0)[vron et al.,](#page-10-0) [2023\)](#page-10-0) is considered more favorable in different chat applications due to its effectiveness towards handling dialogues, logical reasoning, coding, compared to its 7B counterpart which is typically better suited for basic tasks such as categorization or summaries. While these LLMs demonstrate impressive performance in handling complex language tasks, a key limitation is their size and computational requirements. For instance, the large-scale Falcon [\(Almazrouei et al.,](#page-8-0) [2023\)](#page-8-0) 180B model was trained using 4096 A100 GPUs and requires large memory and compute for deployment with dedicated high-performance servers and scalable storage systems.

Recently, Small Language Models (SLMs) have shown potential in terms of providing decent performance with emergent abilities achieved at a significantly smaller scale compared to their large-scale LLM counterparts. Modern SLMs like Microsoft's Phi-2 2.7 billion [\(Li et al.,](#page-9-0) [2023b\)](#page-9-0) highlight the growing focus in the community on achieving more with less. SLMs offer advantages in terms of efficiency, cost, flexibility, and customizability. With fewer parameters, SLMs offer significant computational efficiency in terms of fast pre-training and inference with reduced memory and storage requirements. This is critical in real-world applications where efficient resource utilization is highly desired. It particularly opens up possibilities in resource-constrained computing, where the models are required to be memory efficient to operate on low-powered devices (e.g., edge). SLMs support on-device processing that enhances privacy, security, response time, and personalization. Such an integration can lead to advanced personal assistants, cloud-independent applications, and improved energy efficiency with a reduced carbon footprint.

The landscape of language models, especially SLMs, is currently marked by a notable lack of open-source availability. While LLMs have gar-

<sup>\*</sup>Equal contribution.

nered significant attention, the proprietary nature of most models has led to limited transparency and accessibility, particularly in the realm of SLMs. This gap hinders the scientific and technological exploration of these more efficient, compact and performant models. Recognizing this, there's a growing need in the community for fully transparent opensource SLMs, which would facilitate a deeper understanding of their capabilities and limitations and spur innovation by allowing broader community access to their architecture and reproducible training methodologies. We argue that bridging this gap is crucial for democratizing access to collaborative advancement for SLMs. Therefore, we investigate the problem of designing accurate yet efficient SLMs from scratch with the intention to provide full transparency in the form of access to entire training data pipeline and code, model weights, more than 300 checkpoints along with evaluation codes.

When designing a SLM from scratch it is desired that the resulting model is accurate, while maintaining efficiency in terms of pre-training and deployment. A straightforward way is to scaledown a larger LLM design to the desired model size (e.g., 0.5B) by reducing either the size of the hidden dimension layers or the number of layers. We empirically observe both these design strategies to provide inferior performance. This motivates us to look into an alternative way of designing a SLM from scratch that is accurate yet maintains the efficiency, while offering full transparency.

## Contributions:

We introduce a SLM framework, named *MobiLlama*, with an aim to develop accurate SLMs by alleviating the redundancy in the transformer blocks. Different to the conventional SLM design where dedicated feed forward layers (FFN) are typically allocated to each transformer block, we propose to employ a shared FFN design for all the transformer blocks within SLM. Our *MobiLlama* leveraging a shared FFN-based SLM design is accurate and maintains efficiency, while offering full transparency in terms of data pipeline, training code, model weights and extensive intermediate checkpoints along with evaluation codes.

We empirically show that our *MobiLlama* performs favorably compared to conventional SLMs design schemes when performing pre-training from scratch. Our *MobiLlama* 0.5B model outperforms existing SLMs of similar size on nine different benchmarks. *MobiLlama* 0.5B achieves a gain of 2.4% in terms of average performance on nine

<span id="page-1-0"></span>![](_page_1_Figure_5.jpeg)

Figure 1: Comparison of our *MobiLlama* 0.5B and 0.8B models with recent OLMo-1.17B [\(Groeneveld](#page-9-1) [et al.,](#page-9-1) [2024\)](#page-9-1) and TinyLlama-1.1B [\(Zhang et al.,](#page-10-1) [2024\)](#page-10-1) in terms of pre-training tokens, pre-training time and memory, model parameters, overall accuracy across nine benchmarks and on-device efficiency (average battery consumption and average token/second on a PC with RTX2080Ti). Our *MobiLlama* achieves comparable accuracy while requiring significantly fewer pre-training data (1.2T tokens vs. 3T tokens), lesser pre-training time and GPU memory along with being efficient in terms of deployment on a resource constrained device.

benchmarks, compared to the best existing 0.5B SLM in the literature. We further develop a 0.8B SLM that originates from our 0.5B model by utilizing a wider shared-FFN scheme in transformer blocks, achieving top performance among existing SLMs falling under less than 1B parameters category. Lastly, we build multimodal models on top of our SLM to showcase visual perception and reasoning capabilities. Fig. [1](#page-1-0) shows a comparison of our *MobiLlama* with recent fully transparent relatively larger SLMs in terms of accuracy, pre-training complexity and on-board deployment cost.

# 2 Related Work

While LLMs have gained tremendous popularity [\(Zhao et al.,](#page-10-2) [2023\)](#page-10-2), one of their key limitations is the size and computational requirements both during pre-training and deployment. Another issue is limited availability of fully transparent openssource LLMs that provide complete access to data pipeline, training code along with checkpoints and evaluation protocols. Prior works explore making several components of LLM framework efficient such as, attention mechanism [\(Dao,](#page-8-1) [2023\)](#page-8-1) and optimization strategies [\(Loshchilov and Hutter,](#page-9-2) [2017\)](#page-9-2). Further, existing efforts also include exploring posttraining sparsification schemes [\(Ashkboos et al.,](#page-8-2) [2024\)](#page-8-2) or quantization [\(Hoefler et al.,](#page-9-3) [2021;](#page-9-3) [Zhu](#page-10-3) [et al.,](#page-10-3) [2023;](#page-10-3) [Xiao et al.,](#page-10-4) [2023\)](#page-10-4) of computationally

<span id="page-2-0"></span>

| Model      | #Params | Training Time | GPU Hours | GPU memory | No. of layers | Hidden dim size |
|------------|---------|---------------|-----------|------------|---------------|-----------------|
| baseline1  | 0.54B   | 7.5 days      | 28.8K     | 3.2 GB     | 22            | 1024            |
| baseline2  | 0.52B   | 7 days        | 26.9K     | 3 GB       | 8             | 2048            |
| large-base | 1.2B    | 12 days       | 46.1K     | 6 GB       | 22            | 2048            |
| MobiLlama  | 0.52B   | 7 days        | 26.6K     | 3 GB       | 22            | 2048            |

Table 1: Comparison of our *MobiLlama* with the two baselines and the large-base model. We show the comparison in terms of total number of parameters, training time, total GPU hours, GPU memory, number of transformer layers and the hidden dimension size in each layer. The numbers are computed on A100 GPUs with 80 GB memory each. Compared to *large-base*, our *MobiLlama* reduces the GPU training hours by 42% along with a significant reduction in GPU memory with the same design configuration (number of layers and hidden dimension size etc.). Further, our *MobiLlama* possesses increased model capacity in terms of number of layers and hidden dimension size while maintaining comparable training cost and parameters, compared to *baseline1* and *baseline2*.

expensive LLM. In several cases, such a post-hoc sparsification can reduce the performance of LLMs with more on-device memory consumption, compared to a SLM trained from scratch. Further, these techniques typically employ LLMs with limited transparency and accessibility.

Recently, designing SLMs from scratch have gained attention [\(Biderman et al.,](#page-8-3) [2023;](#page-8-3) [Wu et al.,](#page-10-5) [2023;](#page-10-5) [Zhang et al.,](#page-10-1) [2024;](#page-10-1) [Li et al.,](#page-9-4) [2023a;](#page-9-4) [Lin et al.,](#page-9-5) [2021b;](#page-9-5) [Shoeybi et al.,](#page-10-6) [2019;](#page-10-6) [Zhang et al.,](#page-10-7) [2022\)](#page-10-7). SLMs have shown potential as an alternative especially in case of limited pre-training compute as well as deployment in resource-constrained environments (e.g., edge devices). Further, SLMs can support on-device processing which in turn can enhance security, privacy, response efficiency, and personalization. Here, we strive to construct fully transparent accurate yet computationally efficient SLMs by maintaining the model's capacity to capture complex patterns and relationships in data while reducing the redundancy often present in the parameters of SLMs. Prior works [\(Frantar et al.,](#page-8-4) [2022;](#page-8-4) [Gholami et al.,](#page-9-6) [2022;](#page-9-6) [Pires et al.,](#page-10-8) [2023;](#page-10-8) [Pan](#page-9-7) [et al.,](#page-9-7) [2023;](#page-9-7) [Bhojanapalli et al.,](#page-8-5) [2021\)](#page-8-5) exploring alleviating redundancy in transformer design either focusing on the attention mechanism or on the single feed-forward layer in BERT style architectures. Different from these approaches, we explore alleviating the redundancy in the SLM architectures with an LLM objective function by focusing on the sharing mechanism of MLP blocks having multiple feed-forward network (FFN) layers.

## 3 Method

#### 3.1 Baseline SLM Design

We first describe our baseline 0.5B SLM architecture that is adapted from recent TinyLlama [\(Zhang](#page-10-1) [et al.,](#page-10-1) [2024\)](#page-10-1) and Llama-2 [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0). The baseline architecture comprises N layers,

where each layer consists of hidden dimensions of M and intermediate size (MLPs) of 5632. The vocabulary size is 32K and max. context length is C. We consider two different design choices when constructing a 0.5B model from scratch. In first design choice, named baseline1, the number of layer is set to N = 22 and hidden size of each layer is set to M = 1024. In second design choice, named baseline2, we set the number of layer to N = 8 and hidden size of each layer is set to M = 2048.

We note that both the aforementioned baseline designs struggle to strike an optimal balance between accuracy and efficiency. While a reduced size of hidden dimensions (1024) in case of baseline1 aids in computational efficiency, it can likely hamper the model's capacity to capture complex patterns within the data. Such a reduction in dimension can potentially lead to a bottleneck effect, where the model's ability to represent intricate relationships and nuances in the data is constrained, thereby affecting the overall accuracy. On the other hand, reducing the number of hidden layers (22 to 8), as in the baseline2, affects the model's depth that in turn hampers its ability to learn hierarchical representations of the language. Achieving superior performance on tasks requiring deeper linguistic comprehension and contextual analysis likely requires combining the advantages of the two aforementioned baselines. However, increasing the model capacity of baseline1 and baseline2 into a single model (22 layers and hidden dimension size of 2048) results in a significantly larger parameterized model of 1.2B with increased training cost (see Tab. [1\)](#page-2-0). We name this larger model as *largebase*. Next, we present our proposed *MobiLlama* 0.5B model design that does not reduce hidden dimension size in each layer (baseline1) or the total number of layers (baseline2), while maintaining a comparable training efficiency (see Tab. [1\)](#page-2-0).

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Illustrative comparison of our *MobiLlama* with the two baselines. For each case, we show two transformer blocks denoted by different self-attention layers. In the case of both *baseline1* and *baseline2*, a dedicated MLP block comprising three FFN layers is utilized for each transformer layer. In contrast, our *MobiLlama* utilizes a single MLP block (highlighted by the same color) that is shared across different transformer layers. This enables to increase the capacity of the network in terms of layers and hidden dimension size without any significant increase in the total number of trainable parameters.

#### 3.2 Proposed SLM Design: MobiLlama

The proposed approach, *MobiLlama*, constructs a SLM of desired sizes (e.g., 0.5B model) by first initiating from a larger model size design, *largebase*. Then, we employ a careful parameter sharing scheme to reduce the model size to a pre-defined model configuration, thereby significantly reducing the training cost. Generally, both SLMs and LLMs typically utilize a dedicated multilayer perceptron (MLP) block comprising multiple feed forward network (FFN) layers within each transformer block. In such a configuration (e.g., *large-base*), the FFN layers account for a substantial 65% of the total trainable parameters, with attention mechanisms and heads contributing 30% and 5%, respectively. As a consequence, a significant number of parameters are concentrated within the FFN layers, thereby posing challenges during pre-training with respect to computational cost and the model's ability to achieve faster convergence. To address these issues, we propose to use a sharing scheme where the FFN parameters are shared across all transformer layers within the SLM. This enables us to significantly reduce the overall trainable parameters by 60% in our *MobiLlama*, compared to the *large-base*. Such a significant parameter reduction also enables us to increase the model capacity in terms of number of layers and hidden dimension size without any substantial increase in the training cost (see Tab. [1\)](#page-2-0).

Fig. [2](#page-3-0) compares our architecture design with two baselines. In case of both baselines, a dedicated MLP block that consists of multiple FFN layers is used in each transformer layer. Instead, our efficient *MobiLlama* design utilizes a single MLP block which is shared across different layers of transformer within the SLM. This helps in increasing the model capacity without any increase in the total number of trainable parameters in the model.

<span id="page-3-1"></span>

| Subset        | Tokens (Billion) |
|---------------|------------------|
| Arxiv         | 30.00            |
| Book          | 28.86            |
| C4            | 197.67           |
| Refined-Web   | 665.01           |
| StarCoder     | 291.92           |
| StackExchange | 21.75            |
| Wikipedia     | 23.90            |
| Total         | 1259.13          |

Table 2: Data mix in Amber-Dataset.

<span id="page-3-2"></span>

| Hyperparameter              | Value    |
|-----------------------------|----------|
| Number Parameters           | 0.5B     |
| Hidden Size                 | 2048     |
| Intermediate Size (in MLPs) | 5632     |
| Number of Attention Heads   | 32       |
| Number of Hidden Layers     | 22       |
| RMSNorm ϵ                   | −6<br>1e |
| Max Seq Length              | 2048     |
| Vocab Size                  | 32000    |

Table 3: *MobiLlama* architecture & hyperparameters.

#### 3.3 Towards Fully Transparent MobiLlama

As discussed earlier, fully transparent open-source SLM development is desired to foster a more inclusive, data/model provenance, and reproducible collaborative SLM research development environment. To this end, we present here pre-training dataset

| Model Name         |       | #Params HellaSwag Truthfulqa MMLU Arc_C CrowsPairs |       |       |       |       | piqa | race              | siqa | winogrande Average |       |
|--------------------|-------|----------------------------------------------------|-------|-------|-------|-------|------|-------------------|------|--------------------|-------|
| gpt-neo-125m       | 0.15B | 30.26                                              | 45.58 | 25.97 | 22.95 | 61.55 |      | 62.46 27.56 40.33 |      | 51.78              | 40.93 |
| tiny-starcoder     | 0.17B | 28.17                                              | 47.68 | 26.79 | 20.99 | 49.68 |      | 52.55 25.45 38.28 |      | 51.22              | 37.86 |
| cerebras-gpt-256m  | 0.26B | 28.99                                              | 45.98 | 26.83 | 22.01 | 60.52 |      | 61.42 27.46 40.53 |      | 52.49              | 40.69 |
| opt-350m           | 0.35b | 36.73                                              | 40.83 | 26.02 | 23.55 | 64.12 |      | 64.74 29.85 41.55 |      | 52.64              | 42.22 |
| megatron-gpt2-345m | 0.38B | 39.18                                              | 41.51 | 24.32 | 24.23 | 64.82 |      | 66.87 31.19 40.28 |      | 52.96              | 42.81 |
| LiteLlama          | 0.46B | 38.47                                              | 41.59 | 26.17 | 24.91 | 62.90 |      | 67.73 28.42 40.27 |      | 49.88              | 42.26 |
| gpt-sw3-356m       | 0.47B | 37.05                                              | 42.55 | 25.93 | 23.63 | 61.59 |      | 64.85 32.15 41.56 |      | 53.04              | 42.48 |
| pythia-410m        | 0.51B | 40.85                                              | 41.22 | 27.25 | 26.19 | 64.20 |      | 67.19 30.71 41.40 |      | 53.12              | 43.57 |
| xglm-564m          | 0.56B | 34.64                                              | 40.43 | 25.18 | 24.57 | 62.25 |      | 64.85 29.28 42.68 |      | 53.03              | 41.87 |
| Lamini-GPT-LM      | 0.59B | 31.55                                              | 40.72 | 25.53 | 24.23 | 63.09 |      | 63.87 29.95 40.78 |      | 47.75              | 40.83 |
| MobiLlama (Ours)   | 0.5B  | 52.52                                              | 38.05 | 26.45 | 29.52 | 64.03 |      | 72.03 33.68 40.22 |      | 57.53              | 46.00 |
| Lamini-GPT-LM      | 0.77B | 43.83                                              | 40.25 | 26.24 | 27.55 | 66.12 |      | 69.31 37.12 42.47 |      | 56.59              | 45.49 |
| MobiLlama (Ours)   | 0.8B  | 54.09                                              | 38.48 | 26.92 | 30.20 | 64.82 |      | 73.17 33.37 41.60 |      | 57.45              | 46.67 |

Table 4: State-of-the-art comparisons with existing *< 1B params models* on *nine* benchmarks. In case of around 0.5B model series, our *MobiLlama* achieves a substantial gain of 2.4% in terms of average performance on nine benchmarks. Further, our *MobiLlama* 0.8B model achieves an average score of 46.67.

and processing details, architecture design configuration with training details, evaluation benchmarks and metrics. In addition, we will publicly release complete training and evaluation codes along with intermediate model checkpoints.

Pre-training Dataset and Processing: For pretraining, we use 1.2T tokens from LLM360 Amber dataset [\(Liu et al.,](#page-9-8) [2023b\)](#page-9-8). The Amber dataset provides a rich and varied linguistic landscape having different text types, topics, and styles. Tab. [2](#page-3-1) shows the data mix from Amber dataset gathered from various sources.

*Arxiv (30 Billion Tokens)* subset is drawn from the repository of scientific papers, provides complex, domain-specific language and technical terminology, enriching the understanding of academic prose. *Book (28.9 Billion Tokens)* subset comprises tokens from a broad range of literature with diverse narrative styles, cultural contexts, and rich vocabulary, deepening the grasp of storytelling and language nuances. *C4 (197.7 Billion Tokens)* is the Colossal Clean Crawled Corpus (C4) that offers a vast and cleaned selection of web text, providing a broad linguistic foundation that includes various registers, styles, and topics. *Refined-Web (665 Billion Tokens)* subset is a curated web crawl and offers the model exposure to contemporary, informal, and varied internet language, enhancing the relevance and applicability to modern communication. *StarCoder (291.9 Billion Tokens)* subset is a vast collection used for code understanding featuring 783GB of code across 86 programming languages. It includes GitHub issues, Jupyter notebooks, and commits, totaling approximately 250 billion tokens. These are meticulously cleaned and de-duplicated for training efficiency. *StackExchange (21.8 Bil-* *lion Tokens)* is from the network of Q&A websites, this subset aids the model in learning questionanswering formats and technical discussions across diverse topics. *Wikipedia (23.9 Billion Tokens)* is an encyclopedia collection, it offers well-structured and factual content that helps the model to learn encyclopedic knowledge and formal writing styles.

From the above-mentioned subsets, Arxiv, Book, C4, StackExchange and Wikipedia are sourced from RedPajama-v1 [\(Computer,](#page-8-6) [2023\)](#page-8-6). The Amber dataset uses RefinedWeb [\(Penedo et al.,](#page-9-9) [2023\)](#page-9-9) data to replace common\_crawl subset of RedPajama-v1. These subsets amount to 1259.13 billion tokens.

Initially, raw data sourced from the above sources is tokenized using Huggingface LLaMA tokenizer [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0). Subsequently, these tokens are organized into sequences with each containing 2048 tokens. To manage data, these sequences are merged to the token sequences and divided the amalgamated dataset into 360 distinct segments. Each data segment, structured as a jsonl file, carries an array of token IDs along with a source identifier that denotes the originating dataset. Each data sample is designed to have 2049 tokens. Architecture Design: Our *MobiLlama* 0.5B comprises a hidden size of 2048, an intermediate size of 5632 in its MLPs, and operates with 32 attention heads across 22 hidden layers. It is designed to handle sequences up to 2048 tokens long, supported by a vocabulary size of 32,000. The precision in normalization is ensured by an RMSNorm epsilon of 1e −6 to obtain a more stable training. We utilize RoPE (Rotary Positional Embedding) [\(Su](#page-10-9) [et al.,](#page-10-9) [2024\)](#page-10-9) to encode positional information in our *MobiLlama*. Similar to [\(Zhang et al.,](#page-10-1) [2024\)](#page-10-1), we employ a combination of Swish and Gated Lin-

<span id="page-5-1"></span>

| Model     | HellaSwag Truthfulqa MMLU Arc_C Average |       |       |       |       |
|-----------|-----------------------------------------|-------|-------|-------|-------|
| baseline1 | 42.44                                   | 38.46 | 25.08 | 26.18 | 33.04 |
| baseline2 | 42.15                                   | 38.70 | 25.73 | 26.10 | 33.17 |
| MobiLlama | 44.47                                   | 40.12 | 26.48 | 26.53 | 34.40 |

Table 5: Baseline comparison on four benchmarks. Here, both the baselines and our *MobiLlama* comprise the same parameters (0.5B) and are pre-trained on 100B tokens from Amber. Our *MobiLlama* achieves favorable performance compared to the two baselines, while operating on a similar training budget.

ear Units together as activation functions. Tab. [3](#page-3-2) presents details of our model configuration. We also derive a 0.8B version from our *MobiLlama* by widening the shared FFN design. Compared to the 0.5B model, our 0.8B design increases the hidden dimension size to 2532 and the intermediate size to 11,080 while the rest of the configuration is same.

For pre-training of our *MobiLlama*, we use a public cluster having 20 GPU nodes each equipped with 8 NVIDIA A100 GPUs with 80 GB memory each and 800 Gbps interconnect for model training. Each GPU is interconnected through 8 NVLink links, complemented by a cross-node connection configuration of 2 port 200 Gb/sec (4× HDR) InfiniBand, optimizing the model's training process. To further enhance the training efficiency, we employ flash-attention mechanism and follow the pre-training hyper-parameters established by the LLaMA [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0) model. Our *MobiLlama* model's training is performed using the AdamW optimizer, leveraging hyperparameters β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.95, with an initial learning rate of η = 3e −4 . This rate follows a cosine learning rate schedule, tapering to a final rate of η = 3e −5 . We further incorporate a weight decay of 0.1 and apply gradient clipping at 1.0 with a warm-up period over 2, 000 steps. Adapting to our hardware configuration of 20 GPU nodes, we optimize the pre-training batch size to 800 (160 × 5), achieving a throughput of approximately 14k-15k tokens per second on a single GPU. During our model pretraining, we save intermediate checkpoints after every 3.3B tokens which will be publicly released.

#### Evaluation Benchmarks and Metrics:

For a comprehensive performance evaluation, we use nine different benchmarks from the Open LLM Leaderboard[1](#page-5-0) .

HellaSwag [\(Zellers et al.,](#page-10-10) [2019\)](#page-10-10) assesses the model's ability to predict the correct ending to a scenario from a set of possible continuations,

thereby testing common sense reasoning. TruthfulQA [\(Lin et al.,](#page-9-10) [2021a\)](#page-9-10) evaluates the model to provide truthful answers, focusing on its understanding of facts and its ability to avoid deception. MMLU [\(Hendrycks et al.,](#page-9-11) [2020\)](#page-9-11) measures the model's broad knowledge across numerous subjects such as, humanities, science, technology, engineering and management. ARC\_Challenge [\(Clark](#page-8-7) [et al.,](#page-8-7) [2018\)](#page-8-7) tests complex reasoning with science questions. CrowsPairs [\(Nangia et al.,](#page-9-12) [2020\)](#page-9-12) evaluates the model's biases by comparing sentences that differ only by the demographic group mentioned, aiming for fairness. PIQA [\(Bisk et al.,](#page-8-8) [2020\)](#page-8-8) evaluates the model's physical commonsense knowledge, requiring understanding of everyday physical processes. Race [\(Lai et al.,](#page-9-13) [2017\)](#page-9-13) assesses reading comprehension through multiple-choice questions based on passages. SIQA [\(Sap et al.,](#page-10-11) [2019\)](#page-10-11) focuses on the model's social commonsense reasoning and its understanding of social dynamics. Winogrande [\(Sakaguchi et al.,](#page-10-12) [2021\)](#page-10-12) evaluates the model's ability to resolve ambiguities in text, testing its commonsense reasoning.

Following the Analysis-360 framework [\(Liu](#page-9-8) [et al.,](#page-9-8) [2023b\)](#page-9-8) that is built on llm-harness [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9), we conduct extensive evaluations under the standard settings with varying shots for detailed assessments, validating the model's robustness and adaptability across diverse linguistic tasks. Following the standard evaluation protocol, our evaluation setting consists of 10, 25, 5 and 5 shot evaluation for Hellaswag, ARC\_Challenge, Winogrande and MMLU, while zero-shot for rest of the benchmarks.

# 4 Results

Baseline Comparison: We first present a comparison with the two baselines in Tab. [5\)](#page-5-1) for 0.5B model series. For the baseline evaluation, we pretrain all the models on the same 100B tokens from the Amber dataset and report the results on four benchmarks: HellaSwag, TruthfulQA, MMLU, and Arc\_C. Our *MobiLlama* achieves favourable performance compared to the two baselines by achieving an average score of 34.4 over the four benchmarks. We note that this performance improvement is achieved without any significant increase in the training cost (see Tab. [1\)](#page-2-0), highlighting the merits of the proposed SLM design.

State-of-the-art Comparison: We compare our *MobiLlama* 0.5B and 0.8B with existing SLMs having comparable (less than 1B) parameters: gpt-

<span id="page-5-0"></span><sup>1</sup> [https://huggingface.co/spaces/HuggingFaceH4/open\\_llm\\_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

<span id="page-6-0"></span>

| Platform       | Model      | #Params (↓) | Precision | Avg Tokens/Sec (†) | Avg Memory<br>Consumption (↓) | Avg Battery Consumption /1k Tokens (\( \psi \)) | CPU<br>Utilization (↓) |
|----------------|------------|-------------|-----------|--------------------|-------------------------------|-------------------------------------------------|------------------------|
|                | Llama2     | 7B          | bf16      | 14.85              | 27793 MB                      | 135.51 mAH                                      | 31.62%                 |
| RTX2080Ti      | Phi2       | 2.7B        | bf16      | 32.19              | 12071 MB                      | 59.13 mAH                                       | 24.73%                 |
| K1 A208011     | large-base | 1.2B        | bf16      | 50.61              | 6254 MB                       | 18.91 mAH                                       | 18.25%                 |
|                | MobiLlama  | 0.5B        | bf16      | 63.38              | <b>3046</b> MB                | <b>8.19</b> mAH                                 | 14.79%                 |
|                | Llama2     | 7B          | 4bit      | 5.96               | 4188 MB                       | 73.5 mAH                                        | 49.16%                 |
| CPU-i7         | Phi2       | 2.7B        | 4bit      | 22.14              | 1972 MB                       | 27.36 mAH                                       | 34.92%                 |
| CPU-1/         | large-base | 1.2B        | 4bit      | 29.23              | 1163 MB                       | 10.81 mAH                                       | 30.84%                 |
|                | MobiLlama  | 0.5B        | 4bit      | 36.32              | <b>799</b> MB                 | <b>4.86</b> mAH                                 | 24.64%                 |
|                | Llama2     | 7B          | 4bit      | 1.193              | 4287 MB                       | 10.07 mAH                                       | 77.41%                 |
| Constant COF   | Phi2       | 2.7B        | 4bit      | 2.882              | 1893 MB                       | 14.61 mAH                                       | 56.82%                 |
| Snapdragon-685 | large-base | 1.2B        | 4bit      | 6.687              | 780 MB                        | 6.00 mAH                                        | 17.15%                 |
|                | MobiLlama  | 0.5B        | 4bit      | 7.021              | <b>770</b> MB                 | <b>5.32</b> mAH                                 | 13.02%                 |

Table 6: Comparison in terms of efficiency and resource consumption on different low-end hardware devices. We show the comparison on: a PC with RTX-2080Ti GPU, a laptop with i7 CPU and a smartphone with Snapdragon-685 processor. In addition to our *large-base* model, we also present the comparison with Llama2 7B and Phi2 2.7B. In case of CPU and smartphone, we use 4-bit GGUF format of the corresponding models, whereas the original models are deployed and tested on PC with RTX-2080Ti GPU. The different metrics measure the model's operational efficiency, model's footprint in the device's RAM and the energy efficiency of processing 1,000 tokens. Our *MobiLlama* performs favorably in terms of efficiency on these low-end hardware devices. We note that both Phi2 and Llama2 are not fully transparent in that the complete data pipeline for pre-training is not publicly available.

<span id="page-6-1"></span>

| Model      | #Slice | #Params | HellaS | Arc_C | piqa  | wino  | Average |
|------------|--------|---------|--------|-------|-------|-------|---------|
| OPT-1.3B   | 30%    | 0.91B   | 39.81  | 25.77 | 60.77 | 54.7  | 45.26   |
| OPT-6.7B   | 30%    | 4.69B   | 54.56  | 29.01 | 68.61 | 60.69 | 53.21   |
| Llama-2-7B | 30%    | 4.9B    | 49.62  | 31.23 | 63.55 | 61.33 | 51.43   |
| Phi2-2.7B  | 30%    | 1.89B   | 47.56  | 30.29 | 65.94 | 63.14 | 51.73   |
| MobiLlama  | Dense  | 0.5B    | 52.52  | 29.52 | 72.03 | 57.53 | 52.90   |
| MobiLiama  | Dense  | 0.8B    | 54.09  | 30.20 | 73.17 | 57.45 | 53.72   |

Table 7: Comparison on 4 open LLM benchmarks when parameters are sliced down to 30% using Wiki2Text dataset, following (Ashkboos et al., 2024).

neo (Black et al., 2021), tiny-starcoder (Li et al., 2023a), cerebras-gpt (Dey et al., 2023), opt (Zhang et al., 2022), megatron-gpt-2 (Shoeybi et al., 2019), LiteLlama, gpt-sw3, pythia (Biderman et al., 2023), xglm (Lin et al., 2021b), Lamini-LM (Wu et al., 2023). Among existing methods falling around 0.5B model series category, pythia-410m achieves an average score of 43.57. Our MobiLlama 0.5B model achieves superior performance with an average score of 46.0, outperforming pythia-410m by 2.4% in terms of average performance on nine benchmarks. Notably, MobiLlama achieves superior performance on the HellaSwag benchmark which is designed to evaluate the model's capabilities in the NLP text completion task. Further, MobiLlama also performs favorably on commonsense reasoning tasks with superior results on piqa and winogrande benchmarks. Further, our MobiLlama 0.8B model achieves an average score of 49.06.

**Efficiency Comparison:** We present the comparison of our model in terms of efficiency and re-

<span id="page-6-2"></span>

| Model       | GQA  | SQA  | TextQA | MME    |
|-------------|------|------|--------|--------|
| MobiLlama-V | 58.5 | 53.1 | 41.4   | 1191.9 |

Table 8: Quantitative performance of our multimodal design, *MobiLlama-V* 0.8B, on different benchmarks.

source consumption on various low-end hardware platforms: a PC with RTX-2080Ti GPU, a laptop with i7 CPU, and a smartphone with Snapdragon-685 processor. Tab. 6 shows the comparison of our *MobiLlama* 0.5B with *large-base* 1.2B, Llama2-7B (Touvron et al., 2023) and Phi2-2.7B (Li et al., 2023b) model, in terms of the average processing speed in tokens per second (Average Tokens/Sec), average memory consumption (Avg Memory Consumption) in megabytes (MB), and the average battery consumption (Average Battery Consumption/1000 Tokens) in milliampere-hours (mAH). Our *MobiLlama* performs favorably in terms of efficiency across different hardware platforms.

We further perform an efficiency comparison to a recent post-training sparsification scheme (Ashkboos et al., 2024), where each weight matrix is substituted with a smaller (dense) matrix, thereby reducing dimensions of the embeddings in the model. In such a scheme, the parameters of the original LLM are reduced significantly up to 70% followed by post-slicing fine-tuning using a dataset such as WikiText-2 (Merity et al., 2016). Tab. 7 shows the comparison of our *MobiLlama* with existing LLMs (e.g., Llama-2-7B, OPT-6.7B) on four benchmarks following (Ashkboos et al., 2024). Our *MobiL*-

<span id="page-7-2"></span>

| Model          |      | #Params HellaSwag Truthfulqa MMLU Arc_C CrowsPairs |       |       |       |       | piqa | race              | siqa | winogrande Average |       |
|----------------|------|----------------------------------------------------|-------|-------|-------|-------|------|-------------------|------|--------------------|-------|
| Boomer         | 1B   | 31.62                                              | 39.42 | 25.42 | 22.26 | 61.26 |      | 57.99 28.99 40.32 |      | 50.98              | 39.80 |
| Pythia-Dedup   | 1B   | 49.63                                              | 38.92 | 24.29 | 29.09 | 67.11 |      | 70.23 32.44 42.63 |      | 53.98              | 45.36 |
| Falcon-RW      | 1B   | 63.12                                              | 35.96 | 25.36 | 35.06 | 69.04 |      | 74.10 36.07 40.23 |      | 61.88              | 48.98 |
| TinyLlama      | 1.1B | 60.22                                              | 37.59 | 26.11 | 33.61 | 70.60 |      | 73.28 36.45 41.65 |      | 59.18              | 48.74 |
| OLMo           | 1.2B | 62.50                                              | 32.94 | 25.86 | 34.45 | 69.59 |      | 73.70 36.74 41.14 |      | 58.90              | 48.42 |
| Cerebras-GPT   | 1.3B | 38.51                                              | 42.70 | 26.66 | 26.10 | 63.67 |      | 66.75 30.33 42.42 |      | 53.59              | 43.41 |
| Lamini         | 1.3B | 38.05                                              | 36.43 | 28.47 | 26.62 | 64.62 |      | 67.89 33.39 43.19 |      | 50.59              | 43.25 |
| OPT            | 1.3B | 54.50                                              | 38.67 | 24.63 | 29.6  | 70.70 |      | 72.47 34.16 42.47 |      | 59.74              | 47.43 |
| GPT-NEO        | 1.3B | 48.49                                              | 39.61 | 24.82 | 31.31 | 65.67 |      | 71.05 34.06 41.81 |      | 57.06              | 45.98 |
| Pythia-Deduped | 1.4B | 55.00                                              | 38.63 | 25.45 | 32.59 | 67.33 |      | 72.68 34.64 42.68 |      | 56.90              | 47.32 |
| large-base     | 1.2B | 62.99                                              | 35.90 | 24.79 | 34.55 | 68.49 |      | 75.57 35.31 41.96 |      | 62.03              | 49.06 |

Table 9: Comprehensive comparisons with existing *< 2B params fully open-source LLM models* on *9* benchmarks. Our 1.2B *large-base* model pre-trained on 1.2T tokens achieves superior performance compared to both the recent OLMo 1.17B model [\(Groeneveld et al.,](#page-9-1) [2024\)](#page-9-1) and TinyLlama 1.1B model [\(Zhang et al.,](#page-10-1) [2024\)](#page-10-1), which are pre-trained on a substantially larger data of 3T tokens.

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 3: Example responses from our *MobiLlama* across a variety of tasks, including creative storytelling, coding exercises, economic analysis, and cooking instructions. The responses highlight the models' ability to engage with both abstract concepts and practical, step-by-step processes, demonstrating its broad applicability.

<span id="page-7-1"></span>![](_page_7_Figure_4.jpeg)

Figure 4: Example responses of *MobiLlama*-V in responding to visual stimuli across a range of scenarios.

*lama* 0.5B and 0.8B models perform favorably against representative LLMs, with an average score of 53.72 computed over four benchmarks. These results highlight the potential of designing new fully transparent SLMs that can achieve comparable capabilities of their larger sliced model counterparts.

Multimodal MobiLlama: We further build a multimodal model on top of our *MobiLlama* by combining it with a vision encoder to develop a generalpurpose visual assistant having visual reasoning capabilities. Our multimodal model, *MobiLlama*-V , is trained by bridging the visual encoder of CLIP [\(Radford et al.,](#page-10-13) [2021\)](#page-10-13) with the language decoder of our *MobiLlama*, and fine-tuning it in an end-to-end fashion on a 665k vision-language instruction set [\(Liu et al.,](#page-9-15) [2023a\)](#page-9-15). We conduct

evaluation on GQA [\(Hudson and Manning,](#page-9-16) [2019\)](#page-9-16), SQA [\(Lu et al.,](#page-9-17) [2022\)](#page-9-17), TextQA [\(Singh et al.,](#page-10-14) [2019\)](#page-10-14), and MME [\(Fu et al.,](#page-8-12) [2023\)](#page-8-12). Tab. [8](#page-6-2) shows the performance of *MobiLlama*-V 0.8B model.

Qualitative Analysis: Fig. [3](#page-7-0) shows example responses obtained when interacting with *MobiLlama* 0.5B with conversation capabilities. We show examples covering different tasks such as, text completion, code generation and conversation capabilities. Our model generates faithful responses to these diverse interactions. Fig. [4](#page-7-1) shows examples demonstrating visual reasoning capabilities of our multimodal *MobiLlama*-V . For instance, *MobiLlama*-V accurately describes the atypical aspects of the image when asked to describe the given image.

Evaluating Large-base Model: As discussed ear-

lier, we strive to develop fully transparent models for democratization of SLMs and fostering future research. To this end, we compare our *large-base* 1.2B with existing fully transparent SLMs falling within the less than 2B category. Tab. [9](#page-7-2) shows that compared to recent OLMo and TinyLlama that are pre-trained on a larger dataset of 3T tokens, our *large-base* 1.2B model pre-trained on 1.2T tokens achieves favourable results with an average score of 49.06 over nine benchmarks. We hope that our *large-base* model will serve as a solid baseline and help ease future research in SLM development.

## 5 Conclusion

We present a fully transparent SLM, *MobiLlama*, that alleviates redundancy in the transformer block. Within *MobiLlama*, we propose to utilize a shared FFN design for all the blocks within the SLM. We evaluate *MobiLlama* on nine benchmarks, achieving favourable results compared to existing methods falling under less than 1B category. We also build a multimodal model on top of *MobiLlama* SLM to demonstrate visual reasoning capabilities. Limitation and Future Direction: A potential direction is to further improve *MobiLlama* for enhanced context comprehension. While *MobiLlama* offers a fully transparent SLM framework, a followup study to understand any misrepresentations and biases is desired to improve model's robustness.

## 6 Acknowledgement

The computations were enabled by the Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre. We thank Sahal Shaji Mullappilly and Muhammad Maaz for their support in the evaluations on mobile platform and VLM training.

## References

- <span id="page-8-0"></span>Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. [The falcon series of open language](http://arxiv.org/abs/2311.16867) [models.](http://arxiv.org/abs/2311.16867)
- <span id="page-8-2"></span>Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. *arXiv preprint arXiv:2401.15024*.

- <span id="page-8-5"></span>Srinadh Bhojanapalli, Ayan Chakrabarti, Andreas Veit, Michal Lukasik, Himanshu Jain, Frederick Liu, Yin-Wen Chang, and Sanjiv Kumar. 2021. Leveraging redundancy in attention with reuse transformers. *arXiv preprint arXiv:2110.06821*.
- <span id="page-8-3"></span>Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In *International Conference on Machine Learning*, pages 2397–2430. PMLR.
- <span id="page-8-8"></span>Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.
- <span id="page-8-10"></span>Sid Black, Gao Leo, Phil Wang, Connor Leahy, and Stella Biderman. 2021. [GPT-Neo: Large](https://doi.org/10.5281/zenodo.5297715) [Scale Autoregressive Language Modeling with Mesh-](https://doi.org/10.5281/zenodo.5297715)[Tensorflow.](https://doi.org/10.5281/zenodo.5297715) If you use this software, please cite it using these metadata.
- <span id="page-8-7"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*.
- <span id="page-8-6"></span>Together Computer. 2023. [Redpajama: An open source](https://github.com/togethercomputer/RedPajama-Data) [recipe to reproduce llama training dataset.](https://github.com/togethercomputer/RedPajama-Data)
- <span id="page-8-1"></span>Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*.
- <span id="page-8-11"></span>Nolan Dey, Gurpreet Gosal, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, Joel Hestness, et al. 2023. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster. *arXiv preprint arXiv:2304.03208*.
- <span id="page-8-4"></span>Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*.
- <span id="page-8-12"></span>Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv preprint arXiv:2306.13394*.
- <span id="page-8-9"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. [A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836) [evaluation.](https://doi.org/10.5281/zenodo.10256836)

- <span id="page-9-6"></span>Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. 2022. A survey of quantization methods for efficient neural network inference. In *Low-Power Computer Vision*, pages 291–326. Chapman and Hall/CRC.
- <span id="page-9-1"></span>Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, A. Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Daniel Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hanna Hajishirzi. 2024. [Olmo:](https://api.semanticscholar.org/CorpusID:267365485) [Accelerating the science of language models.](https://api.semanticscholar.org/CorpusID:267365485) *arXiv preprint*.
- <span id="page-9-11"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*.
- <span id="page-9-3"></span>Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. 2021. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *The Journal of Machine Learning Research*, 22(1):10882–11005.
- <span id="page-9-16"></span>Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6700–6709.
- <span id="page-9-13"></span>Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations. *arXiv preprint arXiv:1704.04683*.
- <span id="page-9-4"></span>Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis,

- Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023a. [Starcoder:](http://arxiv.org/abs/2305.06161) [may the source be with you!](http://arxiv.org/abs/2305.06161)
- <span id="page-9-0"></span>Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023b. Textbooks are all you need ii: phi-1.5 technical report. *arXiv preprint arXiv:2309.05463*.
- <span id="page-9-10"></span>Stephanie Lin, Jacob Hilton, and Owain Evans. 2021a. Truthfulqa: Measuring how models mimic human falsehoods. *arXiv preprint arXiv:2109.07958*.
- <span id="page-9-5"></span>Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Veselin Stoyanov, and Xian Li. 2021b. [Few-shot learn](http://arxiv.org/abs/2112.10668)[ing with multilingual language models.](http://arxiv.org/abs/2112.10668) *CoRR*, abs/2112.10668.
- <span id="page-9-15"></span>Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning.
- <span id="page-9-8"></span>Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, and Eric P. Xing. 2023b. [Llm360: Towards](http://arxiv.org/abs/2312.06550) [fully transparent open-source llms.](http://arxiv.org/abs/2312.06550)
- <span id="page-9-2"></span>Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
- <span id="page-9-17"></span>Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521.
- <span id="page-9-14"></span>Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*.
- <span id="page-9-12"></span>Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R Bowman. 2020. Crows-pairs: A challenge dataset for measuring social biases in masked language models. *arXiv preprint arXiv:2010.00133*.
- <span id="page-9-7"></span>Bowen Pan, Rameswar Panda, Rogerio Schmidt Feris, and Aude Jeanne Oliva. 2023. Interpretability-aware redundancy reduction for vision transformers. US Patent App. 17/559,053.
- <span id="page-9-9"></span>Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. [The RefinedWeb dataset](http://arxiv.org/abs/2306.01116) [for Falcon LLM: outperforming curated corpora](http://arxiv.org/abs/2306.01116)

- [with web data, and web data only.](http://arxiv.org/abs/2306.01116) *arXiv preprint arXiv:2306.01116*.
- <span id="page-10-8"></span>Telmo Pessoa Pires, António V Lopes, Yannick Assogba, and Hendra Setiawan. 2023. One wide feedforward is all you need. *arXiv preprint arXiv:2309.01826*.
- <span id="page-10-13"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR.
- <span id="page-10-12"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106.
- <span id="page-10-11"></span>Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. 2019. Socialiqa: Commonsense reasoning about social interactions. *arXiv preprint arXiv:1904.09728*.
- <span id="page-10-6"></span>Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*.
- <span id="page-10-14"></span>Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8317–8326.
- <span id="page-10-9"></span>Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063.
- <span id="page-10-0"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-10-5"></span>Minghao Wu, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, and Alham Fikri Aji. 2023. [Lamini-lm: A diverse herd of distilled models from](http://arxiv.org/abs/2304.14402) [large-scale instructions.](http://arxiv.org/abs/2304.14402) *CoRR*, abs/2304.14402.
- <span id="page-10-4"></span>Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*, pages 38087–38099. PMLR.
- <span id="page-10-16"></span>Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. *arXiv preprint arXiv:2304.12244*.

- <span id="page-10-10"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*.
- <span id="page-10-1"></span>Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. *arXiv preprint arXiv:2401.02385*.
- <span id="page-10-7"></span>Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. [Opt: Open pre](http://arxiv.org/abs/2205.01068)[trained transformer language models.](http://arxiv.org/abs/2205.01068)
- <span id="page-10-2"></span>Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. *arXiv preprint arXiv:2303.18223*.
- <span id="page-10-15"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-10-3"></span>Xunyu Zhu, Jian Li, Yong Liu, Can Ma, and Weiping Wang. 2023. A survey on model compression for large language models. *arXiv preprint arXiv:2308.07633*.

## A Appendix

## A.1 MobiLlama-Chat

We also developed the fine-tuned versions of MobiLlama-0.5B and MobiLlama-1.2B, namely MobiLlama-0.5B-Chat and MobiLlama-1.2B-Chat. We used the similar fine-tuning dataset and hyperparameters as AmberChat [\(Liu et al.,](#page-9-8) [2023b\)](#page-9-8) to train our models. We use FastChat [\(Zheng et al.,](#page-10-15) [2024\)](#page-10-15) to fine-tune our models on [WizzardLM-](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)[196K](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k) as used by WizardLM [\(Xu et al.,](#page-10-16) [2023\)](#page-10-16). We trained our base models for 3 epochs on 8 A100s (80G), keeping the learning rate as 2 × 10−<sup>5</sup> , gradient accumulation steps as 16, and warmup ratio as 0.04.

### A.2 Qualitative Examples

## Reasoning:

Human: What are the key benefits of practicing mindfulness meditation?

## MobiLlama-0.5B:

Practicing mindfulness meditation can have several benefits, including:

- 1. Reduced stress and anxiety: Mindfulness meditation can help reduce stress and anxiety by focusing on the present moment and letting go of distractions.
- 2. Improved focus and concentration: Mindfulness meditation can help improve focus and concentration by focusing on the present moment and letting go of distractions.
- 3. Better sleep: Mindfulness meditation can help improve sleep by reducing stress and anxiety and promoting relaxation.
- 4. Better mood: Mindfulness meditation can help improve mood by reducing stress and anxiety and promoting relaxation.
- 5. Better relationships: Mindfulness meditation can help improve relationships by reducing stress and anxiety and promoting relaxation.

#### Knowledge:

Human: What are some ways to reduce the carbon footprint of your home?

## MobiLlama-0.5B:

Reducing the carbon footprint of your home can be achieved through several ways. Here are some tips:

- 1. Use energy-efficient lighting and appliances.
- 2. Install energy-efficient windows and doors.
- 3. Use energy-efficient appliances and lighting.
- 4. Reduce the use of paper and other non-biodegradable materials.
- 5. Use energy-efficient appliances and lighting.
- 6. Reduce the use of plastic and other single-use materials.
- 7. Use energy-efficient appliances and lighting.
- 8. Reduce the use of paper and other single-use materials.

## Reasoning:

Human: How do diverse ecosystems contribute to the overall health of the planet?

# MobiLlama-0.5B:

Different cultures and traditions contribute to the diversity of the world in many ways. For example, the Chinese culture is known for its emphasis on family, tradition, and hospitality. The Japanese culture is known for its emphasis on individualism, creativity, and innovation. The Indian culture is known for its emphasis on tradition, devotion, and spirituality.

#### Story Creation:

Human: Create the opening paragraph of a suspenseful thriller: In the dimly lit alley, Detective Miller knew something was off. The air was thick with tension, and as he turned the corner, he discovered a hidden door leading to a world of

#### MobiLlama-0.8B:

icy secrets that would soon be revealed in his own life."

"The first thing I noticed about this book is how well-written it really felt like an actual story from beginning to end," said one reviewer on Goodreads who also gave her rating for "10 out of ten stars". Another reviewed by another author wrote:"This novel has been described many times over but never quite captured my attention until now...I'm not sure if you can put your finger on what makes me love these books so much or maybe just because they are such good reads! They have all sorts of twists along their way which keeps them fresh every time!"

## Technological Evolution:

Human: Once people considered Artificial Intelligence is something impossible to achieve, but now

#### MobiLlama-0.8B:

The future of artificial intelligence: AI will be used in every aspect from medicine and healthcare through automation for manufacturing industries like robotics or even autonomous driving systems that can drive themselves without human intervention (or at least with less risk).

#### Analytical Thinking:

Human: Create a concept for an album cover that represents the concept of time travel through music

# MobiLlama-0.8B:

and art, or something similar to this:

Time Travel Through Music – A song with lyrics about how you can go back in history if only one person would listen. . . or maybe it's just me! I don't know what my favorite is yet but there are so many songs out now on YouTube where people talk about going backwards into their past lives (like "Time Machine" by The Beatles) which makes sense because they were talking about getting ridiculous amounts of money from someone who was dead when we get here today; however some artists have gone further than simply saying "I want all your stuff" like Kanye West did last year...he actually went as far as creating his own musical instrument called 'The Tidal Wave'. It basically looks exactly like he does at times during his live shows except instead of being able to see himself perform inside another body while singing along, users will be transported directly onto stage via wave technology using headphones attached to them..and then afterward everyone gets together again once more!! This could also mean anything really though since most musicians do not even realize themselves performing outside of reality anymore anyway....but whatever works best for YOU!!!

#### Programming and Logic Building:

Human: Generate a Python code snippet that implements a function to calculate the Fibonacci sequence using recursion. Bonus points if you can make it more efficient using memoization.

### MobiLlama-1.2B:

Sure, here's a Python code snippet that implements a function to calculate the Fibonacci sequence using recursion:

```
"'
def fibonacci(n):
  if n <= 1:
    return n
  else:
    return fibonacci(n-1) + fibonacci(n-2)
```

To make this function more efficient using memoization, we can store the previous values of the Fibonacci sequence in a dictionary. Here's an updated version of the function that uses memoization:

```
def fibonacci(n, memo={}):
  if n in memo:
    return memo[n]
  elif n <= 1:
    return n
  else:
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
```

This function takes a number n as input and returns the nth Fibonacci number, or the value n if it's already been calculated. If n is less than or equal to 1, the function returns the value n. Otherwise, it recursively calls itself with the previous two numbers in the sequence and adds them together. If the value of n is already in the dictionary, the function returns the value of n directly. Otherwise, it adds the value of n to the dictionary and returns the value of n with the memoized value.

This updated function is more efficient because it avoids calculating the same value multiple times, which can be a bottleneck in recursive functions.

## Innovative Thinking:

Human: Construct a sentence that uses every letter of the alphabet at least once.

## MobiLlama-1.2B:

I am just a 10-year-old's birthday party planner. I love organizing and planning fun and memorable events for kids. I am always up for a challenge and love to create unique and exciting experiences for my clients.

#### Creativity and Imagination:

Human: Write a role-play with a tourist and a travel guide for a finctional planet.

#### MobiLlama-1.2B:

Sure, here's a role-play:

### Tourist: Hi, I'm a tourist from Earth. I'm here to explore the finctional planet. Can you tell me more about it?

### Travel Guide: Sure, the finctional planet is a unique and fascinating place. It's a world where science and magic collide, where the laws of physics are different from our own.

### Tourist: That sounds amazing! Can you tell me more about the planet's history and culture? ### Travel Guide: Sure, the planet was created by a group of scientists who wanted to explore the limits of human knowledge. They created a simulation to test their theories, and the simulation went wrong. The simulation created a world where science and magic intertwined, and the planet became a place where people could live and work together.

### Tourist: That sounds amazing! Can you tell me more about the planet's technology and infrastructure?

### Travel Guide: Sure, the planet has advanced technology that allows people to travel through time and space. It also has a network of underground tunnels that connect different parts of the planet.

### Tourist: That sounds amazing! Can you tell me more about the planet's natural resources and resources?

### Travel Guide: Sure, the planet has a vast array of natural resources that are essential for the planet's survival. It has a unique ecosystem that is home to a variety of species, and it has a thriving economy that relies on the planet's resources.

### Tourist: That sounds amazing! Can you tell me more about the planet's history and culture? ### Travel Guide: Sure, the planet has a rich history that dates back to the beginning of time. It has a complex society that is based on a balance between science and magic.

### Tourist: That sounds amazing! Can you tell me more about the planet's future?

### Travel Guide: Sure, the planet is in the process of creating a new civilization that will be based on a fusion of science and magic. It will be a place where people can live and work together, and it will be a place where science and magic will be in harmony.

### Tourist: That sounds amazing! Can you tell me more about the planet's future?

### Travel Guide: Sure, the planet is in the process of creating a new civilization that will be based on a fusion of science and magic. It will be a place where people can live and work together, and it will be a place where science and magic will be in harmony.