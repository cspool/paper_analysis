# 2 Methodology

In this section we review linear transformers, and describe the linearization technique of Kasai et al. (2021) as it lays the groundwork for our approach. Finally, we present SUPRA, our method for uptraining large transformers into RNNs.

#### <span id="page-2-1"></span>2.1 Background: Linear Attention

*Linear Transformers* (Katharopoulos et al., 2020) establish a connection between transformers and RNNs, generalizing the definition of attention by replacing the softmax dot-product attention  $\mathbf{v}'$  with a more generic similarity function  $\operatorname{sim}(\mathbf{q}, \mathbf{k})$  between the queries  $\mathbf{q}$  and keys  $\mathbf{k}$ :

$$\mathbf{v}_i' = \frac{\sum_{j=1}^i \operatorname{sim}(\mathbf{q}_i, \mathbf{k}_j) \mathbf{v}_j}{\sum_{j=1}^i \operatorname{sim}(\mathbf{q}_i, \mathbf{k}_j)}.$$
 (1)

Standard softmax attention is a special case, using  $sim(\mathbf{q}, \mathbf{k}) = exp\left(\frac{\mathbf{q}^T\mathbf{k}}{\sqrt{d}}\right)$ .

The authors explore several alternative functions for  $sim(\mathbf{q}, \mathbf{k})$ , including a linear kernel. Their main architecture uses the similarity function  $sim(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q}) \cdot \phi(\mathbf{k})$  with a fixed exponential linear unit kernel  $\phi(x) = elu(x) + 1$ . They show the computational benefits of linear attention and, more importantly for this work, they demonstrate how such a model can be expressed as an RNN in the case of attention with causal masking.

**Recurrent Inference** Linear attention can be expressed as an RNN that updates a state  $\mathbf{s}_i$  and a normalization factor  $\mathbf{z}_i$  at each time step. Katharopoulos et al. (2020) call these terms the *attention memory* and *normalized memory*. This RNN formulation is mathematically equivalent to linear

attention, allowing the user to choose the most efficient one for a given task and hardware. Consider a stream of tokens we want to generate  $X = [x_1, x_2, x_3, ...]$ . At inference time, we use the following update rule, where subscripts denote timestep in the recurrence (calling  $\mathbf{k}_i = W_K \mathbf{x}_i$ , etc):

$$\mathbf{s}_0 = 0 \quad \mathbf{z}_0 = 0 \tag{2}$$

$$\mathbf{s}_i = \mathbf{s}_{i-1} + \phi(\mathbf{k}_i)\mathbf{v}_i^T \tag{3}$$

$$\mathbf{z}_i = \mathbf{z}_{i-1} + \phi(\mathbf{k}_i) \tag{4}$$

$$\mathbf{v}_i' = \frac{\phi(\mathbf{q}_i)^T \mathbf{s}_i}{\phi(\mathbf{q}_i)^T \mathbf{z}_i} \tag{5}$$

The state  $s_i$  acts as a constant-size KV cache. Instead of appending new values to the cache, the state is updated. This allows for inference cost that is constant in the number of generated tokens.

#### 2.2 Finetuning a Transformer into an RNN

Kasai et al. (2021) introduced a linear transformer uptraining procedure that converts a pre-trained softmax transformer into an RNN by *approximating* the attention computation with multi-layer perceptrons (MLPs). The method (T2R) starts with a softmax attention model, and linearizes the softmax operation. Recall the kernel linear attention similarity function:

$$sim(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) \cdot \phi(\mathbf{y}). \tag{6}$$

Instead of choosing  $\phi$  as a simple non-linearity, the authors use a trainable layer:

$$\phi(\mathbf{x}) = \text{relu}(W\mathbf{x} + \mathbf{b}). \tag{7}$$

The weights are shared between keys and queries for a given attention head. By using  $\phi$  and rearranging the operations, attention can be written as:

$$\mathbf{v}_i' = \frac{\phi(\mathbf{q}_i)^T \sum_{j=1}^i \phi(\mathbf{k}_j) \mathbf{v}_j^T}{\phi(\mathbf{q}_i)^T \sum_{j=1}^i \phi(\mathbf{k}_j)}.$$
 (8)

This allows the recurrent inference described in Section 2.1. However, this formulation has a number of drawbacks. First, it requires a significant re-training of the model, using approximately 20% of pre-training tokens for conversion, while suffering a 5-10% drop in performance on language benchmarks. Furthermore, this approach was tested on relatively small models ( $\approx 100M$  scale). Because it mimics the attention formulation closely, it suffers from stability issues at larger scales. To address these issues, we modify the approach to adapt it to large-scale model uptraining.

#### 2.3 SUPRA: Scalable UPtraining for Recurrent Attention

Rather than pre-training linear models from scratch, we choose to instead *uptrain* state-of-the-art transformers. Leveraging models that take advantage of high-quality (but proprietary) pre-training datasets, we linearize them using a modest fraction of pre-training data (see Figure 1). We build on T2R, identifying two major issues and proposing SUPRA, an approach to fine-tuning very large transformers into RNNs.

We first follow the literature in linear transformers and identify the **normalization factor** in linear attention as unstable (e.g. TransNormer (Qin et al., 2022a)). In Section 3.3 we show that uptraining a 1B model following the procedure in T2R causes a large drop in performance. We instead follow Retentive Networks (Sun et al., 2023) and replace the normalization with a GroupNorm operation.

Next we note that linear attention suffers more with absolute positional encoding than softmax attention, and a modern relative positional encoding scheme like RoPE [\(Su et al., 2021\)](#page-12-4) is crucial for competitive performance. Rather than training a linear transformer from scratch incorporating these findings (RetNet, TransNormer) we use MLP kernels to *convert* large language models into RNNs.

Starting with the pre-trained model, we add weights shared between keys and queries *W*, *ϕ*(**x**) = relu(*W***x** + **b**) and use the rotary positional embedding (RoPE [\(Su et al., 2021\)](#page-12-4)) such that the similarity function becomes

$$sim(\mathbf{q}_i, \mathbf{k}_j) = RoPE(\phi(\mathbf{q}_i)) \cdot RoPE(\phi(\mathbf{k}_j)). \tag{9}$$

We normalize the output with a GroupNorm [\(Wu & He, 2018\)](#page-13-2) instead of dividing by the sum of sim(**q***<sup>i</sup>* , **k***j*) (as in T2R). We use a fixed decay vector *γ* ∈ (0, 1) *h* , with *h* heads, as in [Sun et al.](#page-12-1) [\(2023\)](#page-12-1). This leads to the following attention formulation (see Figure [2](#page-2-0) for a graphical representation):

$$\mathbf{v}_{i}' = \text{GroupNorm}\left(\sum_{j=1}^{i} \gamma^{i-j} \text{sim}(\mathbf{q}_{i}, \mathbf{k}_{j}) \mathbf{v}_{j}\right).$$
 (10)

These new parameters are trained jointly with the rest of the network; at test time, we use the recurrent formulation for inference.

### **3 Experiments**

We uptrain a variety of models from the 1B to 7B range into RNNs (Llama2 [\(Touvron et al., 2023\)](#page-12-3) and Mistral [\(Jiang et al., 2023\)](#page-11-5)), and evaluate our models in two settings: standard language understanding benchmarks and long-context evaluations. We compare the results of different architectural choices and training strategies, and then show the limitations of linear models on various benchmarks, describing the persistent gap between vanilla attention and recurrence. We choose Llama2-7B and Mistral-7B as our base models for uptraining, but our recipe is general to any transformer model.

We compare our procedure to a variety of pre-trained recurrent models. Given that the largest available state-space models are at the 2.8B scale, we also train a Mamba model on the Refined-Web [\(Penedo et al., 2023\)](#page-11-6) dataset from scratch for 1.2T tokens, to serve as a strong baseline for a pre-trained recurrent model [2](#page-4-0) .

We use [a fork of OpenLM](https://github.com/TRI-ML/linear_open_lm) [\(Gururangan et al., 2023\)](#page-11-7) for all training and fine-tuning. Please see Section [7](#page-9-0) for hyperparameters and further details on reproducibility.

**Language Modeling.** In Table [1](#page-5-0) we report results on standard NLU evaluations using the Eleuther evaluation harness [\(Gao et al., 2023\)](#page-10-3). We primarily compare to transformers and linear models at the 7B scale, and we train a Mamba model at 7B for comparison with RWKV-5. As our model is initialized from strong pre-trained transformers (Llama2 and Mistral-7B), it preserves performance on most benchmarks (except MMLU; see Section [4](#page-7-0) for a discussion below). Our technique outperforms RWKV-5 with minimal uptraining and is competitive with our 7B Mamba trained from scratch on 1.2T tokens.

<span id="page-4-0"></span><sup>2</sup>The [Mistral-SUPRA](https://huggingface.co/TRI-ML/mistral-supra) and [Mamba-7B](https://huggingface.co/TRI-ML/mamba-7b-rw) models are released along with the [code.](https://github.com/TRI-ML/linear_open_lm)

<span id="page-5-0"></span>

| Model         | Size | Tokens  | HellaSwag | PIQA | WG   | ARC-E | ARC-C | MMLU | Average |
|---------------|------|---------|-----------|------|------|-------|-------|------|---------|
| StableLM2     | 1.6B | 2000    | 69.0      | 76.7 | 63.6 | 68.6  | 38.9  | 38.4 | 59.2    |
| StableLM      | 3B   | 1000    | 73.8      | 79.3 | 65.8 | 72.1  | 40.0  | 44.2 | 62.5    |
| Gemma         | 2B   | 2000    | 71.4      | 78.6 | 64.4 | 74.0  | 41.5  | 41.2 | 61.9    |
| Mamba         | 1.4B | 600     | 59.0      | 73.9 | 61.4 | 65.5  | 32.9  | 25.2 | 53.0    |
| RWKV-5        | 1.5B | 1100    | 53.1      | 71.6 | 59.0 | 62.2  | 32.7  | 26.2 | 50.8    |
| Mamba         | 2.8B | 600     | 66.2      | 75.8 | 63.4 | 69.7  | 36.3  | 26.3 | 56.3    |
| Llama2        | 7B   | 2000    | 76.0      | 79.1 | 69.1 | 76.3  | 46.3  | 45.9 | 65.4    |
| Gemma         | 7B   | 6000    | 80.7      | 81.9 | 73.7 | 81.1  | 53.2  | 62.9 | 72.2    |
| Mistral       | 7B   | 8000(?) | 81.0      | 82.1 | 74.0 | 80.9  | 53.8  | 62.4 | 72.4    |
| RetNet        | 6.7B | 200     | 60.7      | 75.4 | 58.1 | –     | –     | –    | –       |
| RWKV-5        | 7B   | 1100    | 70.9      | 77.2 | 67.4 | 71.8  | 43.6  | 31.0 | 60.3    |
| RWKV-5-1.7T   | 7B   | 1700    | 73.0      | 78.6 | 72.9 | 75.8  | 45.6  | 34.9 | 63.5    |
| Mamba (ours)  | 7B   | 1200    | 77.9      | 81.0 | 71.8 | 77.5  | 46.7  | 33.3 | 64.7    |
| Llama2-SUPRA  | 7B   | +20     | 71.8      | 78.6 | 65.8 | 71.1  | 39.5  | 24.9 | 58.6    |
| Mistral-SUPRA | 7B   | +20     | 74.8      | 80.1 | 67.4 | 74.6  | 42.3  | 28.0 | 61.2    |
| Mistral-SUPRA | 7B   | +100    | 77.1      | 80.4 | 70.3 | 75.9  | 45.8  | 34.2 | 64.0    |

Table 1: Linear models (RNNs and SSMs) highlighted in gray. 5-shot results are used for MMLU. Norm results are used for PIQA, HellaSwag, ARC-C. RetNet results taken from RetNet paper.

<span id="page-5-1"></span>

| Model             | Size | Train   | Qasper (2-shot) |       |       | NarrativeQA (0-shot) |       |       |       |       |
|-------------------|------|---------|-----------------|-------|-------|----------------------|-------|-------|-------|-------|
|                   |      | Context | 2048            | 4096  | 8192  | 16384                | 2048  | 4096  | 8192  | 16384 |
| Llama1-7B         | 7B   | 2048    | 24.43           | 7.23  | 5.08  | 4.88                 | 21.44 | 1.03  | 0.0   | 0.0   |
| Llama2-7B         | 7B   | 4096    | 23.26           | 27.26 | 6.20  | 5.49                 | 21.32 | 22.61 | 0.0   | 0.0   |
| Llama2-7B*        | 7B   | 4096    | 23.26           | 27.26 | 31.46 | 25.52                | 21.32 | 22.61 | 23.0  | 14.27 |
| Mistral-7B        | 7B   | 8196    | 21.53           | 25.50 | 33.61 | 6.88                 | 24.94 | 26.90 | 25.93 | 0.63  |
| RecurrentGemma-2B | 2.7B | 8192    | 22.44           | 13.16 | 13.42 | 12.66                | 19.80 | 11.59 | 12.93 | 12.95 |
| RWKV-5-1.7T       | 7B   | 2048    | 22.28           | 23.87 | 22.30 | 20.35                | 17.77 | 18.65 | 17.81 | 16.00 |
| Mamba (ours)      | 7B   | 2048    | 19.68           | 5.58  | 5.90  | 6.32                 | 19.70 | 0.28  | 0.0   | 0.0   |
| Mistral-SUPRA     | 7B   | 2048    | 19.44           | 17.13 | 17.11 | 17.22                | 18.99 | 17.76 | 17.75 | 17.74 |

Table 2: Long context evaluations. Performance at various context size cutoffs for Qasper (2-shot) and NarrativeQA (0-shot). \* denotes linear RoPE scaling with YaRN [\(Peng et al., 2023b\)](#page-12-5).

**Long Context.** Recurrent models were thought to perform well on long-context tasks because of their ability to preserve performance beyond their training sequence size. However, their downstream performance on long-context tasks has not been well-documented. Prior studies either do not conduct long-context evaluations [\(Katharopoulos et al., 2020;](#page-11-1) [Kasai et al., 2021\)](#page-11-4), evaluate only on perplexity [\(Sun et al., 2023;](#page-12-1) [De et al., 2024;](#page-10-1) [Gu & Dao, 2023\)](#page-11-3), or evaluate on datasets which require task-specific training [\(Peng et al., 2023a\)](#page-12-0). Instead, we consider downstream *natural language* tasks from the SCROLLS benchmark [\(Shaham et al., 2022a\)](#page-12-6). Specifically, in Table [2](#page-5-1) we present two tasks – Qasper [\(Dasigi et al., 2021\)](#page-10-4) and NarrativeQA [\(Kocisk](#page-11-8) ˇ y et al., [2018\)](#page-11-8) – from the set of tasks evaluated ´ in the Llama2-Long report [\(Xiong et al., 2023\)](#page-13-3). We evaluate both tasks with an input context cut-off at different lengths. A strong long-context model should perform better given more context. However, the training context lengths for these models do not go beyond 8k tokens. Transformer models show the strongest results up to the context length they were trained for but degrade beyond that. Interestingly, applying the YaRN trick [\(Peng et al., 2023b\)](#page-12-5) enables transformers to scale beyond their training context quite well. RWKV shows a strong ability to handle much longer context than its training. Our Mamba model on the contrary is not able to generalize beyond its training context length. Surprisingly, the RecurrentGemma model [\(Griffin Team et al., 2024\)](#page-10-2) shows degrading

performances even within its training context length. Finally, our Mistral-SUPRA model preserves some performance at larger context lengths but we believe it to result from the decay along the context length that shortens the effective context. This is discussed in more details below. We find a significant gap in performance between transformers and available linear models, including models uptrained from strong long-context transformers. We speculate that more sophisticated recurrent state update rules may be required to perform well at this task. Ideas such as gating strategies [\(De](#page-10-1) [et al., 2024\)](#page-10-1), higher order linear attention [\(Mercat, 2020\)](#page-11-9), or associative binding [\(Munkhdalai et al.,](#page-11-10) [2024\)](#page-11-10) could be explored.

**Decay factors.** The default decay factors proposed in [Qin et al.](#page-12-7) [\(2024\)](#page-12-7) gives better results than no decay on short context benchmarks but at a long range, the decay cancels out the influence of the context (max(*γ*) <sup>2048</sup> = 3.35*e* −4 ). This can be related to a smooth version of window attention [\(Beltagy et al., 2020\)](#page-10-5). However, as more context is given to the model, the long-range evaluation performances plateau. When using the values proposed in [Sun et al.](#page-12-1) [\(2023\)](#page-12-1), that allow longer range attention, we observe a performance drop on short-context benchmarks and no substantial improvement on long-context evaluation.

<span id="page-6-0"></span>

| Model                      | Size | Tokens         | HellaSwag | ARC-E | ARC-C |
|----------------------------|------|----------------|-----------|-------|-------|
| Mamba                      | 1B   | 100B           | 58.3      | 62.7  | 29.1  |
| SUPRA (from scratch)       | 1B   | 100B           | 54.7      | 59.6  | 27.8  |
| T2R (from scratch)         | 1B   | 100B           | 55.2      | 61.0  | 28.4  |
| Transformer                | 1B   | 100B           | 55.9      | 60.4  | 29.3  |
| Transformer*               | 1B   | 1.6T           | 62.1      | 66.2  | 34.3  |
| Fine-tune only new weights | 1B   | (1.6T)+10B     | 33.2      | 37.8  | 22.6  |
| 2-step fine-tune           | 1B   | (1.6T)+10B+10B | 56.1      | 62.2  | 30.4  |
| T2R                        | 1B   | (1.6T)+10B     | 40.6      | 53.8  | 27.9  |
| SUPRA                      | 1B   | (1.6T)+10B     | 57.0      | 62.4  | 31.6  |
| SUPRA                      | 1B   | (100B)+10B     | 51.7      | 57.5  | 27.4  |

Table 3: Ablating different choices for linear uptraining: note the importance of normalization. For the second half of the table, we uptrain a transformer trained on 1.6T tokens on 10B further tokens. \*This model was trained on a different mix of data.

**Ablations.** Table [3](#page-6-0) compares transformers pre-trained on 100B tokens to Mamba [\(Gu & Dao, 2023\)](#page-11-3), T2R [\(Kasai et al., 2021\)](#page-11-4), and our approach. At this scale, with 100B tokens of training, the Mamba model performs best and other models show similar performance. The second half of Table [3](#page-6-0) shows results for uptraining from a pre-trained transformer. TheT2R [\(Kasai et al., 2021\)](#page-11-4) uptraining was unstable, yielding poor results compapred to SUPRA. This confirms that normalization is key for maintaining performance of the base LLM when uptraining.

To test the hypothesis that linear attention approximates the softmax attention, we experimented with a 2-step approach. The first step trains only the new parameters such that the model could learn to approximate the softmax. The second step fine-tunes all the weights. The results show no benefit from the two steps approach and indicates that the softmax is not approximated. See Appendix [A](#page-14-0) for a different approach to compare softmax attention and linear attention.

Finally, we compare the results of SUPRA uptrainings from two pre-trained softmax models. It appears that pre-training a linear model for a 100B token yields better results than fine-tuning a softmax model that was trained with the same budget. These results also shows, along with the comparison of LLama2-SUPRA and Mistral-SUPRA in Table [1,](#page-5-0) that SUPRA benefits significantly from a stronger pre-trained transformer. Thus, given a limited training budget, using SUPRA from a strong pre-trained model is the best option.

### <span id="page-7-0"></span>**4 Discussion**

**Comparison to pre-training SSMs/RNNs.** With only 20B tokens of training, which represents 2 − 10% of RWKV and RetNet training cost, we obtain a model that outperforms both on HellaSwag and that is competitive on other benchmarks (see Table [1\)](#page-5-0). Given the existing performance gap between the strongest transformer models and the most performant linear models, SUPRA is a simple recipe for conversion, allowing the study of strong RNNs with limited uptraining.

**Comparison to Transformers on Short-Context Tasks.** Our approach does not explicitly approximate attention from the base transformer model (see Appendix [A\)](#page-14-0), we do see a modest drop in performance across all benchmarks compared to softmax transformers. This could be partially explained by the lower quality of our data compared to the pre-training mix used to train models like Mistral-7B. It is also likely that linear transformers are inherently less expressive. However, the performance drop is relatively modest on most benchmarks, and significantly smaller than the drop from T2R uptraining, which shows the relevance of our approach.

**Long Context Comparisons.** Prior work on linear attention showcased similar or better validation set perplexity to transformer models over long context (e.g. [Sun et al.](#page-12-1) [\(2023\)](#page-12-1)) but did not evaluate linear models on *natural language* long-context evaluations like SCROLLS [\(Shaham et al., 2022b\)](#page-12-8). The results in Table [2](#page-5-1) show that recurrent models generally maintain performance beyond their training context (except for Mamba-7b) while transformers (without modification) do not. However, Table [2](#page-5-1) also demonstrates that simple linear scaling of the rotary positional embedding [\(Peng et al.,](#page-12-5) [2023b;](#page-12-5) [emozilla, 2023\)](#page-10-6) can allow for context scaling beyond the window used for training a given transformer model, effectively nullifying the performance edge of these linear models. Furthermore, transformers generally outperform linear models at their maximum training context length. Further research is needed into extending linear models to long-context inference to take full advantage of the lower inference cost relative to vanilla transformers.

**Limitations.** Since our method relies on initializing with strong pre-trained transformers, our models inherit any of the biases and weaknesses of their base models. Additionally, models that are already instruct-tuned do not linearize as well as base models. Our models suffer from poor performance on MMLU which requires in-context learning (5-shot), a weakness of linear models [\(Akyurek et al., 2024\)](#page-10-7). We leave the investigation of these weaknesses of linear models to ¨ future work and hope that our proposed uptraining approach can help facilitate and accelerate the research in this area.

