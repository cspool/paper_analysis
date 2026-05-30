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

