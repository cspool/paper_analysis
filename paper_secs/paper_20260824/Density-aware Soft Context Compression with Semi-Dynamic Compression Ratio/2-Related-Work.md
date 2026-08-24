# **2 Related Work**

**Hard Prompt Compression** Hard compression methods, such as LLMLingua [\(Jiang et al.,](#page-9-4) [2023;](#page-9-4) [Pan et al.,](#page-10-3) [2024\)](#page-10-3), operate directly within the discrete text space to prune redundant tokens. While these approaches avoid extensive model training, they are inherently bounded

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> last tokens compression tokens mean pooling Encoder compression feature instruction context Decoder answer
![](_page_2_Figure_1.jpeg)

Figure 1: Three typical feature extraction mechanisms for soft context compression.

by the discrete nature of the vocabulary, struggling to achieve extreme compression ratios without severe information loss.

**Soft Context Compression** Soft compression maps discrete token sequences into shorter, continuous latent representations. Early explorations like xRAG [\(Cheng et al.,](#page-9-1) [2024\)](#page-9-1) and 500xCompressor [\(Li et al.,](#page-10-0) [2024\)](#page-10-0) aggressively compressed entire documents into a single token embedding, which inevitably caused massive information loss for lengthy documents. Intermediate methods like ICAE [\(Ge et al.,](#page-9-3) [2024\)](#page-9-3) and PCC [\(Dai et al.,](#page-9-0) [2025\)](#page-9-0) popularized the "compression tokens" paradigm. However, these frameworks typically require massive text-reconstruction pre-training and often freeze the decoder, resulting in semantic misalignment. Methods such as Mean-pooling Context Compression [\(Feldman & Artzi,](#page-9-2) [2025\)](#page-9-2) discard heavy pre-training in favor of knowledge distillation. Conversely, Cascade Context Compression [\(Liu & Qiu,](#page-10-1) [2025\)](#page-10-1) utilizes 1 million pages of diverse OCR data (encompassing both Chinese and English documents) alongside text reconstruction tasks for its pre-training phase. Concurrently, approaches like Arcaligner [\(Li et al.,](#page-10-4) [2026\)](#page-10-4) introduce specialized decoder modules, while CLaRa [\(He et al.,](#page-9-5) [2025\)](#page-9-5) utilizes high-quality synthetic data to jointly train the compressor and generator over fixed-length targets.

**Dynamic and Adaptive Compression** While most soft compression techniques enforce rigid ratios, some recent works explore text-adaptive strategies. Dynamic Large Concept Models [\(Qu et al.,](#page-10-5) [2025\)](#page-10-5) attempt to chunk text into semantic concepts based on adjacenttoken similarity, subsequently applying mean-pooling to each individual chunk to extract its features. However, its chunking strategy is somewhat heuristic and it lacks a mechanism for user-controlled global compression scaling. Similarly, REFRAG [\(Lin et al.,](#page-10-6) [2025\)](#page-10-6) employs a reinforcement learning-trained selector for a binary routing decision (compress entirely or leave uncompressed) for each document block. In contrast, our work introduces a semidynamic, continuous-to-discrete selection mechanism that seamlessly adapts to varying densities while providing explicit, continuous control over the global compression scale.

### **3 Methodology**

To systematically address the context compression bottleneck, we first review existing feature extraction paradigms to explain why a fully dynamic compression ratio would lead to infinite structural hyperparameters. Building upon this, we detail the core failure of fully dynamic compression, which motivates our Semi-Dynamic framework.

### **3.1 Re-evaluating Feature Extraction Methods**

Soft context compression relies on a feature extraction mechanism to derive a compressed latent representation from an encoder's hidden states. Given an input context of length *Lctx*,

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Input Context Hidden States Encoder Compression Features S=2 Mean Pooling Linear Head S=4 Mean Pooling Router scale Mean Pooling
![](_page_3_Figure_1.jpeg)

Figure 2: Our semi-dynamic context compression method, utilizing mean-pooling as the optimal structural backbone.

the goal is to extract a representation of reduced length *M*, i.e. *M* latent tokens. As shown in Figure 1, we categorize existing extraction operations into 3 primary paradigms:

- **Last Tokens:** A naive approach that directly extracts the hidden states of the final *M* tokens of the original sequence. The structural hyperparameter is the target token count *M*.
- **Compression Tokens:** The widely adopted paradigm that appends *M* learnable tokens to the end of the context for information gathering. After encoding, the hidden states corresponding to these tokens are extracted. The structural hyperparameter is also *M*.
- **Mean-Pooling:** A chunking-free approach that partitions the encoded sequence into non-overlapping windows. By applying mean-pooling over the hidden states within each window, it produces the compressed vectors. The structural hyperparameter here is the pool size *S*.

A fundamental tension arises when controlling the compression behavior of these methods. For token-based methods (last tokens and compression tokens), maintaining a specific compression ratio r requires the hyperparameter M to be strictly dependent on the input context length (i.e.,  $M \approx r \cdot L_{ctx}$ ), or manually split the context into chunks of fixed length. Conversely, for mean-pooling, outputting a fixed compressed length M requires the pool size S to be dynamically dependent on  $L_{ctx}$  (i.e.,  $S \approx L_{ctx}/M$ ). Consequently, without dynamic hyperparameters, token-based methods must be inherently fixed-length, while mean-pooling is inherently fixed-ratio.

#### 3.2 The Pitfall of Continuous Structural Hyperparameters

The dependency issues outlined above naturally motivate *fully dynamic* compression, where structural hyperparameters (M or S) are dynamically computed to adapt to varying information densities. However, theoretically, LLMs map inputs to fixed computational sub-graphs. When a hyperparameter dictates the structure of the graph, such as dynamically determining the exact number of tokens M to append or the exact stride S of a pooling window as a continuous function of  $L_{ctx}$ , it creates an infinite spectrum of computational variations, making optimization highly unstable.

Our empirical investigations confirm this: when models are forced into fully continuous dynamic setups (the "continuous" here is not its mathematical meaning, but refers to a too vast variety of variations, thus can be considered relatively "continuous" in integer space), they suffer severe accuracy degradation. Conversely, training a model simultaneously on a small, discrete set of fixed operations (e.g.,  $S \in \{2,4,8,16\}$ ) maintains near-optimal accuracy. This definitive contrast highlights that models can robustly learn a finite set of distinct structural operations, but fail against the infinite variations of a continuous dynamic parameter.

#### 3.3 The Semi-Dynamic Compression Framework

Guided by the necessity for finite structural operations, we propose the **Semi-Dynamic Context Compression** framework (Figure 2). It retains the flexibility of density-aware compression while actively avoiding the continuous hyperparameter pitfall.

**Discrete Ratio Selector (DRS)** To bridge the gap between continuous density prediction and discrete structural execution, we propose **Discrete Ratio Selector (DRS)**, a rule-based module between the encoder and decoder. At its core, the DRS functions mathematically as a scalar quantizer: it maps a continuous predicted signal into a predefined, finite set of discrete states.

Initially, the encoder's regression head outputs a continuous value  $\hat{y}$ , representing the predicted compression ratio in logarithmic space ( $\log_2$ ). To enable zero-shot controllable inference, we introduce a user-defined hyperparameter, *scale*, which acts as an additive bias to the head's prediction:

$$\hat{y}^{scaled} = \hat{y} + scale \tag{1}$$

By adjusting *scale* at inference, users can smoothly shift the overall distribution toward better fidelity (negative scale) or better efficiency (positive scale). The continuous predicted compression ratio is then recovered via exponentiation:

$$\hat{r} = 2^{\hat{y}^{scaled}} \tag{2}$$

The subsequent quantization process branches based on the chosen structural backbone:

**Case 1: Ratio-Based Quantization (e.g., Mean-Pooling).** We define a predefined candidate set of discrete ratios  $\mathcal{R} = \{r_1, r_2, \dots, r_k\}$  (e.g.,  $\{0.125, 0.25, 0.5\}$ ). The continuous ratio  $\hat{r}$  is quantized to the nearest discrete candidate  $r_{target}$ :

$$r_{target} = \underset{r \in \mathcal{R}}{\arg\min} |\hat{r} - r| \tag{3}$$

The discrete pooling window size S is then deterministically computed as  $S = \text{int}(1/r_{target})$ , ensuring a valid, finite structural operation.

Case 2: Length-Based Quantization (e.g., Compression Tokens). We define a candidate set of discrete token counts  $\mathcal{M} = \{m_1, m_2, \dots, m_k\}$  (e.g.,  $\{16, 32, 64, 128\}$ ). For a given context of length  $L_{ctx}$ , we calculate the continuous target token count  $\hat{m}$  and quantize it to the nearest available discrete count  $M_{target}$ :

$$\hat{m} = \hat{r} \cdot L_{ctx} \tag{4}$$

$$M_{target} = \underset{m \in \mathcal{M}}{\arg \min} |\hat{m} - m| \tag{5}$$

By decoupling the continuous density prediction from the discrete structural execution through this DRS quantization, the model operates exclusively within the finite set of structural parameters it can reliably learn.

**Single-Stage Architecture and Dynamic Expansion** To ensure computational efficiency, we designed a single-stage architecture that completes density prediction and compression in a single encoding pass. Given the context, the encoder first produces hidden states  $H \in \mathbb{R}^{L_{ctx} \times d}$  (d is hidden size). We extract the hidden state of the final token,  $h_{last}$ , passing it through a linear regression head to predict the continuous compression target  $\hat{y}$ . Next,  $\hat{y}$  is routed through the Discrete Ratio Selector (DRS) to determine the exact discrete parameter ( $r_{target}$  or  $M_{target}$ ). Only after this parameter is selected does the model execute the structural compression over H to extract the condensed representations. Finally, these representations are mapped into the decoder's input embeddings via an MLP projector.

To simplify the user prompt, we introduce *dynamic single-placeholder expansion*. The user inserts only a single placeholder token to replace the original context. When preparing the input for the decoder, this token is dynamically expanded to the required length dictated by  $r_{target}$  (or  $M_{target}$ ), and its input embeddings are replaced by the projected compression features.

#### 3.4 Density-Aware Data Synthesis and Label Generation

Unlike previous density-aware methods (Lin et al., 2025) relying on complex Reinforcement Learning (RL) pipelines (like PPO), we propose a pure Supervised Fine-Tuning (SFT) approach driven by synthetic data. This avoids the optimization instabilities inherently associated with RL.

Motivation for the Density Proxy Our approach relies on the intuition that the length of a highly condensed summary reflects the original text's information density. While an imprecise heuristic, the discretized nature of our framework means the continuous proxy label does not need flawless precision; it only needs to provide a rough indicator to steer the prediction into the correct discrete bucket.

**Dual-Phase Data Synthesis** We perform synthetic data generation in two phases using a teacher LLM (e.g., Qwen3-30B-A3B-Instruct) on seed contexts from the UltraFineWeb (Wang et al., 2025) dataset. This dataset comprises a robust mixture of bilingual pre-training data, where the English subset is rigorously filtered from Fineweb-v1.4 (Penedo et al., 2024) and the Chinese subset is filtered from Chinese-Fineweb-V2 (Yu et al., 2025).

- Phase 1: Task Synthesis for Generative Loss. We generate standard QA pairs and summaries to compute the causal language modeling loss ( $\mathcal{L}_{LM}$ ), jointly optimizing the encoder, projector, and decoder.
- Phase 2: Ultra-Concise Synthesis for Density Labels. We prompt the teacher LLM to generate extremely concise summaries omitting all redundant words, whose lengths ( $L_{sum}$ ) are used as the intrinsic density proxy for training label creation.

**Label Formulation and Joint Optimization** For a context of length  $L_{ctx}$  and ultra-concise summary length  $L_{sum}$ , the target density label in logarithmic space is defined as:

$$y = \log_2\left(\frac{L_{ctx}}{L_{sum}}\right) \tag{6}$$

The logarithmic transformation is critical for optimization stability. Taking the base-2 logarithm ensures that the label distribution remains roughly uniform across a linear space. Without it, as the summary length  $L_{sum}$  linearly decreases for highly compressible texts, the raw ratio  $\frac{L_{ctx}}{L_{sum}}$  would rapidly expand following an inverse proportional curve. This would result in an heavily skewed target distribution dominated by excessively large label values, leading to inherently biased model predictions. Finally, the joint model is optimized using the LM loss and Mean Squared Error (MSE) for the prediction head:

$$\mathcal{L}_{total} = \mathcal{L}_{LM} + \lambda \cdot MSE(\hat{y}, y) \tag{7}$$

#### 4 Experiments

#### 4.1 Experimental Setup

**Training Data** We construct a synthetic dataset of 10 million samples, whose seed contexts are sampling from UltraFineWeb (Wang et al., 2025) with context lengths between 128 and 1,300 tokens. Using Qwen3-30B-A3B-Instruct, we generate context-based NLP tasks encompassing summarization, single/multi-document QA, and multi-hop reasoning in English and Chinese. All of our training experiments are based on this synthetic dataset.

**Evaluation Benchmarks** We construct a mixed dataset for evaluation, from four standard reading comprehension benchmarks (filtered under 2,048 tokens), uniformly sampling 1,000 instances from: HotpotQA (Yang et al., 2018), SQuAD (Rajpurkar et al., 2016), Natural Questions (NQ) (Kwiatkowski et al., 2019), and AdversarialQA (Bartolo et al., 2020).

**Evaluation Metrics** We evaluate mainly using 2 metrics: answer accuracy and average compression ratio. For accuracy, we use *substring accuracy*: a score of 1 is awarded if the exact reference answer appears anywhere within the output, which is more intuitive than F1 and more aligns with human assessment than exact-match. For average compression ratio, it is calculated as the sum of the original context lengths of all the samples which are corrected answered divided by the sum of their compressed lengths. Noteworthy, here we apply a strict *validity filter*: count **only** for instances answered correctly. This prevents the samples that are aggressively compressed by the model but fail to generate correct answers from artificially inflating the average compression ratio.

### **4.2 Implementation Details**

We employ the Qwen3 family, initializing the encoder from Qwen3-0.6B and the decoder also from Qwen3-0.6B. For SFT, we apply LoRA (*r* = 16, alpha 128 for encoder, 64 for decoder) on all the linear modules with a global batch size of 80. For the discretized mechanism, ratio-based candidate sets are R = {2×, 4×, 8×, 16×, 32×}. We append an < *eos* > token to the context to let the encoder know the last token's hidden state is specifically used for ratio prediction. The converter is a 2-layer MLP with the intermediate size of 4,096. For mean-pooling compression, the encoder's attention is turned to bidirectional.

#### <span id="page-6-0"></span>**4.3 Main Results**

> **[图片提取文字 (无描述)]:**
> mean pooling (fixed ratio) 60 mean pooling (fixed ratio 5-ratio-in-1) mean pooling (fixed len) 50 last tokens (fixed len) last tokens (fixed ratio) Acc(%) memory tokens (fixed ratio) 40 memory tokens (fixed len) 30 20 1.5 2.0 2.5 3.0 3.5 4.0 log2(ratio)
![](_page_6_Figure_5.jpeg)

<span id="page-6-1"></span>Figure 3: Accuracy vs. Average Compression Ratio across three feature extraction methods (mean-pooling, last tokens, compression tokens), evaluated under fixed-ratio and fixedlength settings.

> **[图片提取文字 (无描述)]:**
> - fixed-ratio 65 semi-dynamic - fully-dynamic 60 baseline 55 Acc(%) 50 45 40 35 30 5.0 2.0 2.5 3.0 4.5 1.0 1.5 3.5 4.0 log2(ratio)
![](_page_6_Figure_7.jpeg)

Figure 4: Accuracy vs. Average Compression Ratio for fixed-ratio vs. semidynamic mean-pooling. For the semi-dynamic, the *scale* parameter is varied across {−2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4} to achieve gradually growing compression ratio. The baseline dashed line represents not using compression.

<span id="page-7-0"></span>> **[图片提取文字 (无描述)]:**
> 0.8 Acc Improvement --- variance 4 0.7 0.6 3 Acc diff (%) 0.5 2 0.4 0.3 0 0.2 -1 0.1 2.5 5.0 1.0 1.5 2.0 3.0 3.5 4.0 4.5 log2(ratio)
![](_page_7_Figure_1.jpeg)

Figure 5: Variance of selected compression ratios (log<sup>2</sup> ) and absolute accuracy improvement of semi-dynamic method over the fixed-ratio baseline.

> **[图片提取文字 (无描述)]:**
> semi-dynamic (2-stage) semi-dynamic (1-stage) 60 Acc(%) 50 40 30 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 1.0 log2(ratio)
![](_page_7_Figure_3.jpeg)

Figure 7: Accuracy vs. Average Compression Ratio for 2-stage and 1-stage semidynamic mean-pooling compression.

> **[图片提取文字 (无描述)]:**
> --- fixed-ratio 80 semi-dynamic 75 baseline 70 (%) 65 60 60 55 50 45 2.5 1.0 1.5 2.0 3.0 3.5 4.0 4.5 5.0 log2(ratio)
![](_page_7_Figure_5.jpeg)

Figure 6: Accuracy vs. Average Compression Ratio for Qwen3-4B-Instruct with fixed-ratio and semi-dynamic meanpooling compression.

> **[图片提取文字 (无描述)]:**
> 65 causal (fixed-ratio) bidir (fixed-ratio) 60 causal (semi-dynamic) bidir (semi-dynamic) 55 Acc(%) 50 45 40 35 4.5 1.5 2.0 3.0 3.5 1.0 2.5 4.0 log2(ratio)
![](_page_7_Figure_7.jpeg)

Figure 8: Accuracy vs. Average Compression Ratio for mean-pooling with causal or bidirectional attention.

### *4.3.1 Backbone Comparisons and the Hyperparameter Pitfall*

Our first experiments isolate the architectural mechanisms by training and evaluating with the 3 feature extraction, on 4 different settings: fixing the compression ratio at 4 and 16 respectively, and fixing the compression length to 32 and 128 respectively. The results are shown in Figure [3,](#page-6-0) where each method corresponds to 2 average compression ratios.

**Dominance of Mean-Pooling** When comparing feature extraction methods, at equivalent average compression rates, we find mean-pooling consistently outperforms both token-based methods. Surprisingly, the widely adopted compression tokens paradigm is significantly outperformed even by naive last tokens extraction. This may be due to the fact that the additional trainable parameters required for compression tokens do not provide additional useful information and are only for the formal rationality of the structure, struggling to converge optimally alongside the rest of the model, unless adopting a heavy pretraining phase.

**The Hyperparameter Space Pitfall** When evaluating fixed-ratio versus fixed-length settings, the empirical data highlights a strong inverse correlation between the size of the structural hyperparameter space and downstream accuracy. Given our training context lengths (128–1,024):

- 1. A **Fixed Ratio** regime for token-based methods forces the target token count *M* to fluctuate widely (e.g., between 32 and 256, based on the context length of the training sample), exposing the model to over 200 distinct hyperparameters of operations and causing massive quality drops compared to the fixed-length regime where the hyperparameter is fixed.
- 2. A **Fixed Length** regime for mean-pooling forces the stride *S* to fluctuate (e.g., 4 to 32), a hyperparameter set of 28 distinct operations, resulting in a slight but noticeable

- accuracy drop compared to the fixed-ratio regime where the hyperparameter is fixed.
- 3. The **5-Ratio-in-1 Mean-Pooling** setup explicitly constrains the hyperparameter set to just five discrete values ( $\mathcal{R} = \{2,4,8,16,32\}$ ) (for each training batch, we randomly fix one of these 5 ratios; in evaluation we fix the ratio to 4 and 16 respectively). Because the model multiplexes between only 5 fixed hyperparameters of operations, the accuracy drop is minimal, which is consistent with the findings of Mean-pooling Context Compression (Feldman & Artzi, 2025).

This confirms that while LLMs can effectively multiplex a small set of discrete structural operations (semi-dynamic), navigating the vast hyperparameter space (fully-dynamic) causes severe optimization failures.

### 4.3.2 Semi-Dynamic vs. Fixed-Ratio Compression

We evaluate our Semi-Dynamic framework against standard fixed-ratio baselines (training an individual model for each ratio) using the mean-pooling backbone (since it is just proved optimal). As the results shown in Figure 4, we can find that:

- 1. **Semi-Dynamic Outperforms Static:** At identical average compression ratios, the density-aware semi-dynamic method maintains higher accuracy than static ratio method across the entire evaluated spectrum, except for the lowest ratio. While the fully dynamic method performs even worse than the static ratio method.
- 2. **Maximum Gain at Moderate Scales:** The performance gap is most pronounced at moderate *scale* biases. As illustrated in Figure 5, the variance of the model's selected compression ratio peak at moderate average compression ratios (typically ranging from 4 to 16), showing basically the same trend as the accuracy improvements over the fixed-ratio baseline. While at extreme high/low scales, almost all samples are forced into the maximum/minimum discrete bucket, stifling the advantage. This further confirms that our framework's superiority indeed stems directly from its **dynamic adaptability**, but not more training data or training tricks.

#### 4.4 Supplementary Experiments

**Scaling to Larger Base Models** As shown in Figure 6, replicating the experiments with Qwen3-4B-Instruct (for initializing both the encoder and decoder) confirms that the 4B models exhibit a significantly higher overall accuracy at any setting. Yet the relative performance gap between static and semi-dynamic methods persists, proving our semi-dynamic framework scales effectively with model capacity.

**1-Stage vs. 2-Stage** As shown in Figure 7, we compare a 2-stage pipeline (using a standalone, isolated regression model for ratio prediction) against our single-stage joint model. The single-stage model's performance is very close to the 2-stage pipeline, indicating that jointly training 2 functions (compression and ratio prediction) into one encoder poses no harm, and achieves higher efficiency.

**Bidirectional vs. Causal Attention** As shown in Figure 8, for fixed-ratio mean-pooling compression, comparing a standard causal encoder against a bidirectional encoder (causal mask disabled) reveals negligible differences at low compression ratios  $(2\times,4\times)$ . While at higher compression ratios  $(\geq 16\times)$ , the bidirectional encoder's global visibility provides a distinct advantage in determining salient features during aggregation. For the semi-dynamic setting, bidirectional encoder is always slightly better than causal encoder. This confirms the necessity of bidirectional attention.

#### 5 Conclusion

In this work, we introduce the Semi-Dynamic Context Compression framework to address the core inefficiencies of rigid, fixed-ratio soft compression. By identifying the intrinsic inability of LLMs to optimize over continuous structural hyperparameters, we propose

the Discrete Ratio Selector (DRS). This novel quantization mechanism successfully bridges continuous density prediction with discrete, learnable structural execution. Implemented within a highly efficient single-stage architecture, our approach dynamically adapts to the intrinsic information density of varying texts while offering users smooth control over global compression aggressiveness via a simple scaling parameter. Furthermore, we established a streamlined, pure-SFT training pipeline that utilizes summary length as a highly effective density proxy, eliminating the need for complex reinforcement learning or expensive textreconstruction pre-training. Extensive empirical evaluations confirm that our density-aware framework outperforms static baselines, successfully leveraging text diversity to establish a robust new Pareto frontier for context compression techniques.

