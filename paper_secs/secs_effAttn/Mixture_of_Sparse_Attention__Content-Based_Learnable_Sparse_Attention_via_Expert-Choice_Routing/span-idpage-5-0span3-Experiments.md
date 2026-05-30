# <span id="page-5-0"></span>3 Experiments

<span id="page-5-1"></span>

| Model size | #Params Dense | Dense ppl ↓ | MoSA Best ppl ↓      | Fixed Best ppl ↓ | Routing Best ppl ↓ |
|------------|---------------|-------------|----------------------|------------------|--------------------|
| Tiny       | 28M           | 22.46       | 16.39 (-27.0%)       | 23.28 (+3.7%)    | 23.33 (+3.9%)      |
| Small      | 113M          | 16.01       | 12.85 (-19.7%)       | 16.51 (+3.1%)    | 16.43 (+2.6%)      |
| Medium     | 210M          | 13.95       | 11.06 (-20.7%)       | 14.35 (+2.9%)    | 14.21 (+1.9%)      |
| Large      | 516M          | 12.20       | $10.58 \; (-13.3\%)$ | 12.40 (+1.6%)    | $12.24 \ (+0.3\%)$ |

Table 1: Comparing dense and sparse models (Fixed, Routing, MoSA) under a fixed computational budget (see Section 3.2). For sparse models, the table contains the best perplexity across all sparsities bigger than 1. The results for sparse models were selected as the best of all sparsities. Relative difference to the dense baseline is displayed in the parentheses. MoSA significantly outperforms the dense baseline, reducing perplexity by up to 27%. The fixed and the Routing Transformer baselines both fail to reach the performance of the dense model.

In this section, we empirically demonstrate MoSA's performance in different settings. We compare MoSA to dense and sparse baselines introduced in Section 3.1. In Section 3.2, we evaluate all the methods on language modeling under a fixed FLOP budget. In Section 3.3 we demonstrate the practical benefits of MoSA by measuring wall-clock time, memory usage, and KV cache size in a

perplexity-matched setup. In Section 3.4 we investigate the performance of MoSA on long sequences. Finally, in Section 3 we show the performance of different models in downstream zero-shot tasks.

We use four model sizes for our experiments: *Tiny*, *Small*, *Medium* and *Large*. Each size is defined by the *FLOP count* of the forward pass of the corresponding dense transformer baseline. The parameter count of dense models associated with each size is: 28M for *Tiny*, 113M for *Small*, 210M for *Medium*, and 516M for *Large*.

Implementation details We use the SentencePiece [42] tokenizer based on sub-word units [43, 44] a vocabulary size of 8000. All our models are trained on the C4 [45] dataset for 100k batches, with batch size B=64 and sequence length T=1024. This means that we train on the  $10^5SB\approx 6.5B$  tokens from the dataset. We use the Adam [46] optimizer with a learning rate of 0.00025, gradient clipping above the norm of 0.25, and a linear warmup for 4k steps. For detailed hyperparameters, please refer to Appendix C.

#### <span id="page-6-0"></span>3.1 Baselines

Apart from a dense baseline, we compare MoSA with two sparse attention methods: static, position-based sparse attention, and content-based sparse attention.

**Fixed Sparse Attention.** Position-based static attention patterns have been shown to be a strong sparse attention variant [18], outperforming strided sliding window attention. Fixed sparse attention for a sparsity  $\rho$  selects  $k = \frac{T}{\rho}$  tokens with stride  $\rho$ . Using the notation introduced in Section 2.2, fixed sparse attention can be written as a special case of MoSA, where  $I = [0, \rho, 2\rho, ..., T - \rho]$  and  $\mathbf{r} = \mathbf{1}$ .

Fixed sparse attention reduces computational complexity in two ways. First, it decreases the  $O(T^2)$  cost of the full attention matrix by limiting attention to predefined token positions. Second, since only these pre-selected tokens participate in attention calculations, the query, key, value, and output transformations need only be computed for this subset rather than all tokens.

However, this approach introduces information flow constraints. Pre-selected tokens must aggregate necessary information in earlier layers. Furthermore, in the subsequent layers they have to be routed back to the positions where they are most useful. This additional overhead in information routing limits the model's representational capacity and overall expressiveness.

The Routing Transformer. We also compare MoSA to the content-based attention proposed in the Routing Transformer [25]. The Routing Attention is the most similar method to MoSA we found in the literature. It groups tokens with online K-means into  $\rho$  clusters of size  $k=\frac{T}{\rho}$  inside each head. This is implemented during training by the top-k tokens most similar to the cluster centers using the dot-product distance metric. Cluster centers are learned using a moving average of the most similar tokens.

The Routing Attention might resemble the Expert-Choice selection with MoSA. There are, however, several crucial differences that, as our experiments show, lead to significant differences in the performance of MoSA in comparison to the Routing Transformer. Specifically, online K-means, used for clustering in the Routing Transformer is known for suffering from an extremely slow convergence rate [26]. It is also unclear if clustering keys and queries is well aligned with the language modeling objective. In contrast, the learned dynamic matching mechanism of MoSA is directly optimized by the same objective as the model.

MoSA benefits from the sparsity in the  $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O$  transformations, which need to be computed only for selected tokens. In contrast, the Routing Transformer has to compute all keys and queries before the clustering step. MoSA's efficiency enables the use of more heads with specialized weights in a smaller subset of tokens. Its selection can also lead to dynamic compute allocation, where some more important tokens are processed by more heads than less important tokens.

Last but not least, the Routing Transformer performs best in language modeling when the clusters share the same destination (query) tokens and source (keys and values) tokens. In our experiments, we also found that MoSA performs better if the same tokens are selected for the source and destination sides. However, to enforce this in the Routing Transformer, they require to set  $\mathbf{W}^Q = \mathbf{W}^K$ . In MoSA, however, the same selection for source and destination side can be enforced with  $\mathbf{W}^Q$  different from  $\mathbf{W}^K$ , allowing greater flexibility.

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 3: Perplexity  $(\downarrow)$  of FLOP matched models under different sparsities. Each plot corresponds to a specified FLOP budget per step. The number in parenthesis is the number of parameters of the dense baseline. Sparsity 1 represents the dense baseline. As sparsity increases, MoSA's perplexity improves monotonically until reaching a saturation point around sparsity 32-64, beyond which performance deteriorates. This is likely because at very high sparsity levels, each attention head selects only a few tokens, which is insufficient to capture the complex relations. On the other hand, other sparse methods fail to reach the perplexity of the dense baseline in the IsoFLOP setting. We explore fewer sparsity levels for larger models due to excessive memory requirements.

We visualize typical schematic attention patterns of the baselines and MoSA in Fig. 2. Note that several previous works proposed combining different types of sparse attention to achieve synergic performance in long-sequence tasks [20, 19, 47]. In this work, we focus on investigating sparse attention methods in combination with a few dense attention heads, but without combining multiple sparse attention types. We leave combining MoSA with other sparse-attention methods for future work.

#### <span id="page-7-0"></span>3.2 Main Results

To evaluate sparse methods, we evaluate multiple models with a gradually increasing sparsity rate  $\rho=\frac{T}{k}$ . This reduces the compute requirements for each sparse head. We use the saved budget to increase the number of sparse heads. Specifically, we choose the number of sparse heads to be the maximum such that the FLOPs of the sparse model do not exceed the FLOPs of the baseline model for a given size. All sparse models include four dense heads that we keep (see Section 2.2), and are included in the FLOP calculations.

Note that increasing the number of attention heads also increases the memory requirements of all methods. Consequently, for the larger FLOP-matched models, we restricted the explored sparsity values to ensure that the models fit in the memory budget dictated by our hardware.

In the IsoFLOP experiments, following observations from StreamingLLM [48] on the importance of first tokens in the attention mechanism, we always include the first token in all MoSA heads. The head selects k-1 tokens based on their router scores and the first token. The representation of the first token, just like the others, is multiplied by its router score after the attention mechanism.

<span id="page-7-2"></span>**FLOPs Calculation.** Let T be the sequence length, h the hidden dimension of the model, h' the hidden dimension in each head (after passing through the query, key or value projection), k the number of tokens selected for each head, and the sparsity rate  $\rho = \frac{T}{k}$ .

FLOPs cost of a single head is equal to:

$$FLOP_{dense} = \underbrace{8hh'T}_{Q,K,V,O \text{ mappings}} + \underbrace{4h'T^2}_{Attention}$$

$$FLOP_{mosa} = \underbrace{8hh'k}_{Q,K,V,O \text{ mappings}} + \underbrace{4h'k^2}_{Attention} + \underbrace{2hT + h'k}_{routing \text{ overhead}}$$

$$FLOP_{fixed} = \underbrace{8hh'k}_{Q,K,V,O \text{ mappings}} + \underbrace{4h'k^2}_{Attention}$$

$$FLOP_{routing} = \underbrace{6hh'T}_{Q=K,V,O \text{ mappings}} + \underbrace{4h'k^2\rho}_{Attention} + \underbrace{2h'T}_{cluster \text{ selection}} = \rho(6hh'k + 4h'k^2) + 2h'T$$

The detailed derivation of FLOP costs of the attention and the entire models can be found in App. A. Note that, typically k << T, hence the MoSA head is significantly cheaper compared to a dense head.

The selection mechanism in MoSA introduces an additional overhead of 2hT + h'k (2hT comes from token scoring and h'k comes from multiplying the output by the scores), which is small compared to the rest. As a consequence, the cost of the MoSA head is comparable to that of the fixed sparsity attention head, while allowing content-based dynamic sparsity.

In contrast to MoSA and fixed attention, the Routing Transformer must compute all tokens by query, key, value, and output transformations. However, in the Routing Transformer for autoregressive text K=Q, therefore, only 3 projections need to be computed. Hence, the projection cost is equal to 6hh'T. The attention in the Routing Transformer has multiple clusters inside each head. More specifically, it has  $\rho$  clusters of size k, and therefore the attention cost of the head is equal to the attention cost of the cluster multiplied by the number of clusters. The Routing Transformer has an additional layer normalization inside the head, which we omitted for simplicity.

FLOP-wise, one Routing Attention head more or less corresponds to  $\rho$  fixed attention or  $\rho$  MoSA heads. Loosely speaking, MoSA with  $\rho$  heads is similar to the Routing Attention head, where each cluster has its own custom linear transformation, rather than a single one shared among clusters.

**IsoFLOP Curves.** Starting from sparsity 1, which corresponds to the dense model, we gradually increase the sparsity and measure the test-set perplexity of FLOP-matched models. Table 1 lists the best results for each model class and size. Across all model sizes tested, MoSA achieved significantly better perplexity within fixed FLOP budgets compared to dense baselines. All MoSA hybrids reduce the perplexity of the baseline, sometimes by 27%. On the other hand, the sparse baselines for all sparsities  $\rho > 1$  perform worse than the dense baseline.

Figure 3 illustrates the IsoFLOP curves of the models with varying degrees of sparsity. For MoSA, performance steadily improves as sparsity increases, reaching optimal results at approximately  $\rho=64$ . Beyond this threshold, performance begins to decline, creating a "U" shape in the curve. This is likely because the excessively high sparsity values limit the model's ability to capture complex attention patterns. For example, at  $\rho=256$  with a sequence length of T=1024, only k=4 tokens are selected to participate in each attention head.

For some configurations, MoSA turns proves to be more efficient than the dense model even in a parameter-matched setting. For example, Medium model with sparsity 8 has 442M parameters and perplexity 12.16, while the Large baseline model has 516M parameters and perplexity 12.20. This shows that a higher specialization of the heads might lead to improved performance even when we discard computational benefits. Detailed results for different MoSA sparsity configurations, together with the total number of parameters and the number of heads, are listed in the Appendix 5.

In contrast to MoSA, both fixed sparse attention and the Routing Attention consistently underperform the dense baseline across all sparsity levels. They exhibit relatively constant, but worse, perplexity across different sparsity values, with only minor fluctuations that reveal no discernible trend.

#### <span id="page-9-0"></span>3.3 Resource Optimization

The previous section demonstrates MoSA's ability to achieve better perplexity than dense transformers with an identical compute budget. In this section, we examine MoSA's practical efficiency gains. Specifically, we match the perplexity scores between the MoSA and the dense baseline to measure wall-clock time, memory, and KV-cache size savings.

To find the perplexity-matched comparison, we select sparsity to be equal to 32 for model sizes Tiny, Small and Medium. For Large we select  $\rho=16$  to keep sparsity closer to the range investigated in Section 3.2. Then, we gradually increase the number of MoSA heads until the perplexity matches the dense baseline. We do it for all four model scales defined in Section 3.2.

The results are shown in Table 2. MoSA can match the dense baseline, while being faster in wall-clock time and using less memory at the same time. These findings show that MoSA not only improves model quality in the FLOP-matched setting but can also be used to reduce computational and memory requirements when targeting the same performance level. Furthermore, it shows that MoSA uses computation more effectively than standard dense attention across all efficiency metrics.

MoSA achieves this without a specialized CUDA kernel using only PyTorch-level operations. We expect that designing a specialized kernel would result in additional significant efficiency gains.

In addition to the speed and memory used for the training, we report the total number of key-value pairs (KV) used, calculated as  $KV = TH_{dense} + kH_{mosa}$ , where  $H_{dense}$  and  $H_{mosa}$  represent the number of dense and sparse heads, respectively. KV directly corresponds to the size of the costly KV-Cache in the autoregressive setting. KV cache optimization has been the goal of many post-training sparse-attention methods[32, 33, 34]. Our results demonstrate that MoSA offers a significant reduction in KV-cache size while simultaneously improving speed and memory requirements.

<span id="page-9-2"></span>

|                                  | T     | iny    | S     | mall   | Me    | dium   | Large |        |
|----------------------------------|-------|--------|-------|--------|-------|--------|-------|--------|
|                                  | Dense | MoSA   | Dense | MoSA   | Dense | MoSA   | Dense | MoSA   |
| Dense Heads                      | 9     | 4      | 9     | 4      | 9     | 4      | 16    | 4      |
| MoSA Heads                       | 0     | 17     | 0     | 14     | 0     | 12     | 0     | 16     |
| Perplexity $(\downarrow)$        | 22.46 | 22.40  | 16.02 | 16.01  | 13.94 | 13.76  | 12.20 | 12.16  |
| Wall-time/step $\downarrow$ (ms) | 137   | 127    | 326   | 319    | 619   | 592    | 807   | 703    |
| Wall-time/step gain (%)          | _     | -7.3%  | _     | -2.1%  | _     | -4.4%  | _     | -12.9% |
| Memory $\downarrow$ (GB)         | 21.1  | 19.0   | 32.4  | 31.4   | 50.2  | 49.4   | 104.1 | 94.5   |
| Memory gain (%)                  | _     | -10.0% | _     | -3.1%  | _     | -1.6%  | _     | -9.2%  |
| $KV Total \downarrow (K)$        | 9.2   | 4.5    | 9.2   | 4.4    | 9.2   | 4.4    | 16.4  | 5.0    |
| KV Total gain (%)                | -     | -51.1% | _     | -52.2% | –     | -52.2% | -     | -69.5% |

Table 2: Resource usage reduction from perplexity-matched MoSA models. KV is the KV-cache size, representing the total number of key-value pairs required (in thousands). MoSA models match the perplexity of dense baselines while at the same time improving wall-clock time, using less memory, and significantly smaller KV cache for all model sizes. Resource usage was measured on a single A100 GPU for *Tiny, Small* and *Medium* models and on two A100 GPUs for *Large*.

#### <span id="page-9-1"></span>3.4 Scaling with Sequence Length

Traditionally, sparse attention methods have been introduced as a necessity when sequence length makes dense attention computationally prohibitive. After demonstrating MoSA's effectiveness in standard-length sequences, we now investigate whether MoSA's benefits are retained or amplified in this long sequence setup.

In contrast to previous sections, here we combine MoSA or a baseline method with local attention [18, 20]. We use local attention instead of dense attention because even a small number of dense attention heads would result in prohibitive memory usage in a longer context scenario. This is a standard practice in the sparse attention literature [18, 25]. Local attention preserves local dependencies, while global, sparse attention enables efficient processing of long dependencies.

We scale our sequence length from 1024 to 8192 tokens and keep the k constant equal to 64. Hence, the sparsity increases from  $\rho=16$  for T=1024 to  $\rho=128$  for T=8192. Contemporary sparse attention methods for long sequences are trained in longer sequences [37]. However, due to our

<span id="page-10-0"></span>![](_page_10_Figure_0.jpeg)

Figure 4: Perplexity of sparse-attention methods (MoSA, Fixed, and Routing) as sequence length increases. Each method has a fixed size window size (cluster size for the Routing Transformer, number of tokens selected for each head in MoSA and Fixed) regardless of total sequence length. MoSA matches the computational cost of the fixed sparsity baseline while requiring fewer FLOPs than the Routing Attention and consistently achieves the lowest perplexity.

limited hardware budget, we restrict our experiments to a sequence length of 8192. We treat this investigation as a preliminary analysis that demonstrates the potential of MoSA for long sequences. Importantly, it demonstrates that MoSA performs well when combined with local attention, which is a typical long-sequence setup.

As in the previous section, we compare MoSA with fixed sparse attention and the Routing Attention. All long sequence models have 6 layers and hidden dimension size of 1024. The Routing Transformer has 4 local attention heads and 4 Routing Transformer heads in all layers, whereas the fixed sparse attention and MoSA have 60 sparse heads and 4 local attention heads. We chose 60 sparse heads to roughly FLOP match all models for T=1024. However, as we keep k constant, for longer sequences with 2048,4096 and 8192 tokens, the FLOP cost for fixed attention and MoSA will be much lower than for the Routing Attention. For T=8192 FLOP cost of 60 MoSA's heads is equal to only 22.99% of 4 Routing Transformer heads.

The results are shown in Fig. 4. MoSA significantly outperforms other sparse attention methods across all sequence lengths. This is true even at length 8192, where MoSA uses only a small fraction of the computational cost of the Routing Transformer.

The significant performance gap in the results demonstrates the potential of MoSA for ultra-long sequences [49, 37, 50]. Given our limited resources, we leave the investigation of MoSA in this context for future work.

#### <span id="page-10-1"></span>3.5 Downstream Tasks

We evaluate the zero-shot downstream performance of MoSA on six established benchmarks: LAMBADA [51], WinoGrande [52], BLiMP [53], HellaSwag [54], PIQA [55] and AI2ARC [56]—covering tasks from cloze-style completion to commonsense reasoning.

During training, MoSA operates on sequences of more or less constant size T=1024. However, for downstream tasks, some inputs will be much shorter. For example, most datapoints in the BLiMP dataset do not exceed 10 tokens. In order to handle such situations, we adaptively choose the number of tokens for each input to be  $k=\max(\lfloor \frac{T}{\rho}\rfloor,2)$  tokens for each head. This simulates the ratio of tokens selected for the attention head during the training. Moreover, it ensures that at least 2 tokens are selected, which is the minimum necessary for the attention to model any cross-token dependencies.

For each scale and sparse model type, we select the model with sparsity  $\rho > 1$  that produced the best perplexity in the IsoFLOP scenario (Sec. 3.2). We also include the dense baseline for each size. Table 3 reports the performance across the tasks. The best result for a given task across model types is bold.

For *Tiny*, *Small*, and *Medium* scales, MoSA generally outperforms other models. BLiMP stands as a notable exception, where MoSA consistently underperforms. This weak performance on BLiMP can

be attributed to the extremely short length of most examples in the dataset. With longer sequences seen during training, each MoSA head can selectively process only the tokens it handles well. However, in short sequences, the shortage of tokens forces MoSA heads to operate on tokens outside their training distribution. Furthermore, when ⌊ T ρ ⌋ = 1, resulting in only 2 tokens being selected, there is a significant discrepancy between the percentage of selected tokens compared to training conditions. Models with a high sparsity factor of 64 typically select only 1.56% tokens in a sequence for each attention head. Yet for a sequence length of T = 10, 2 selected tokens represent 20% of the sentence, creating a distribution mismatch.

Moreover, in *Large* scale, the Dense baseline outperforms MoSA despite having much higher perplexity. We attribute the downstream performance gap of MoSA to two main factors. First, MoE architectures have been shown to suffer from expert overspecialization, which often leads to decreased performance in downstream tasks [\[28,](#page-15-9) [57\]](#page-17-4). Instruction tuning has been shown to mitigate this issue [\[58\]](#page-17-5).

Furthermore, content-based sparse attention methods tend to struggle on shorter sequence[‡](#page-11-2) . Our experiments confirm this pattern, as MoSA outperforms the Routing Attention in most tasks. Furthermore, some runs of the Routing Attention were unstable in context of downstream tasks (Medium scale of the Routing Attention). Practitioners report that extending training by additional epochs on truncated sequences can mitigate the issues of sparse attention methods on short sequences[‡](#page-0-0) .

<span id="page-11-1"></span>

|        | Model   | LAMBADA | WinoGrande | BLiMP | HellaSwag | PIQA | AI2ARC |
|--------|---------|---------|------------|-------|-----------|------|--------|
| Tiny   | Dense   | 18.7    | 50.3       | 72.0  | 27.5      | 59.4 | 28.0   |
|        | MoSA    | 25.4    | 51.9       | 64.6  | 29.1      | 59.4 | 28.6   |
|        | Routing | 14.0    | 51.3       | 66.2  | 27.8      | 57.1 | 25.9   |
|        | Fixed   | 17.1    | 50.6       | 72.5  | 27.7      | 58.6 | 28.1   |
| Small  | Dense   | 25.8    | 52.1       | 76.2  | 30.9      | 62.4 | 30.1   |
|        | MoSA    | 30.7    | 48.5       | 62.8  | 31.8      | 60.4 | 30.2   |
|        | Routing | 19.2    | 50.7       | 70.2  | 28.0      | 57.6 | 27.3   |
|        | Fixed   | 24.6    | 51.6       | 75.3  | 30.1      | 63.2 | 30.2   |
| Medium | Dense   | 31.4    | 51.2       | 77.8  | 33.8      | 64.5 | 31.5   |
|        | MoSA    | 27.6    | 52.2       | 75.1  | 33.9      | 65.1 | 31.6   |
|        | Routing | 10.2    | 51.5       | 65.9  | 30.3      | 57.8 | 27.8   |
|        | Fixed   | 29.4    | 51.4       | 77.3  | 33.0      | 64.6 | 31.5   |
| Large  | Dense   | 36.2    | 52.5       | 80.4  | 38.7      | 67.1 | 33.8   |
|        | MoSA    | 32.3    | 52.8       | 77.2  | 36.6      | 65.0 | 32.2   |
|        | Routing | 27.5    | 51.1       | 76.5  | 36.2      | 64.1 | 32.5   |
|        | Fixed   | 32.3    | 51.7       | 79.6  | 35.9      | 66.0 | 32.2   |

Table 3: Accuracy on downstream zero-shot tasks. Each model is selected with the best sparsity in the IsoFLOP comparison. Note that on downstream tasks, the token selection mechanism of MoSA operates out of distribution. Despite this, MoSA often outperforms the dense baseline. Even when it doesn't, the performance gap is usually small.

