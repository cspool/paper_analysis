# Composing Sparse Attention via Learned Grouping

Hengshuai Yao1,2 Xing Chen<sup>1</sup> Ahmed Murtadha<sup>1</sup> Jin Li<sup>1</sup> Yasin Abbasi Yadkori<sup>1</sup> Shuai Shao<sup>1</sup> Changling Liu<sup>1</sup> Guan Wang<sup>1</sup> Mingli Yuan<sup>1</sup> William Chen<sup>1</sup> Sen Song<sup>3</sup> <sup>1</sup>Sapient Intelligence <sup>2</sup>University of Alberta <sup>3</sup>Tsinghua University April 30, 2026

#### Abstract

Efficient attention methods reduce the O(n 2 ) cost of transformers, but existing approaches degrade perplexity, downstream accuracy, or both when retrofitted onto pretrained models. We introduce Focus, which instead learns which token pairs matter. A small set of learnable centroids (as few as 148K parameters) is added to each attention layer. These centroids act as gates, allowing only same-group token pairs to attend to each other at long range. Focus is composable with any pretrained model: only the centroids are trained; all original weights stay frozen.

Our experiments show that composing Focus onto pretrained models yields zero degradation on downstream benchmarks—from 124M to 70B parameters, across five attention architectures. Surprisingly, sparse attention surpasses full attention at 124M (30.3 vs 31.4 PPL) and matches it when trained from scratch at 7B (13.82 vs 13.89 PPL). Focus is also fast: top-k group membership yields 2× speedup with better quality than the pretrained model. With our FlashAttention decomposition, Focus reaches 8.6× speedup at 1M tokens with no custom kernels.

# 1 Introduction

Transformers compute pairwise attention scores between all tokens at O(n 2 ) cost [\[Vaswani et al.,](#page-12-0) [2017\]](#page-12-0). Does each token really need to attend to every other token? The efficient attention literature has explored this question extensively, but how to reduce attention without losing quality remains open. Prior work falls into three camps. Structured sparsity methods use fixed patterns—local windows, block structures—and miss important long-range dependencies when retrofitted onto pretrained models [\[Beltagy et al.,](#page-11-0) [2020,](#page-11-0) [Zaheer et al.,](#page-13-0) [2020\]](#page-13-0). Approximation methods replace the attention matrix with a cheaper proxy via kernels or low-rank projections, but the approximation error compounds across layers [\[Choromanski et al.,](#page-11-1) [2021,](#page-11-1) [Wang et al.,](#page-13-1) [2020\]](#page-13-1). Token selection methods [\[Ribar et al.,](#page-13-2) [2024,](#page-13-2) [Chen et al.,](#page-13-3) [2024,](#page-13-3) [Zhang et al.,](#page-13-4) [2024,](#page-13-4) [Singhania et al.,](#page-13-5) [2024\]](#page-13-5) keep the attention mechanism intact and select the top-k most relevant tokens per query, but degrade perplexity by 5–10 points at high sparsity, as we show in Section [3.](#page-3-0)

We take a different approach: we learn which token pairs actually matter. We introduce Focus. The key insight is that existing pretrained models can read every token but cannot focus—they have no mechanism to determine, before computing attention, which distant tokens are worth attending to. Focus adds this missing capability: learnable centroid vectors in each attention layer assign tokens to semantic groups and gate the attention scores accordingly. Tokens within the same group attend with exact softmax—no re-normalization, no approximation—so the pretrained computation is preserved, not approximated.

Composability. Focus is composable: only the centroid parameters are trained—as few as 148K, just 0.1% of the model—while all original weights stay frozen. The model retains everything it knew and gains the ability to direct its attention. This holds from 124M to 70B, across five attention architectures (MHA, GQA, GQA+bias, MHA+QK-norm, interleaved+softcap), with zero degradation on downstream benchmarks. Composability distinguishes Focus from LoRA [\[Hu et al.,](#page-11-2) [2022\]](#page-11-2): in our experiments, LoRA degrades alignment scores at every learning rate we tested, while Focus preserves instruction-tuned behavior fully.

Less attention can be more. Focus is sparse: with K=4 groups and top-k=2 membership, each token attends to only half of the distant tokens. Despite this sparsity, composing Focus onto GPT-2 124M achieves 30.3 PPL, surpassing the full-attention model at 31.4. At inference, the same sparse model yields 41.3 PPL—better than the pretrained model at 42.8—with 2× speedup. Trained from scratch on Mistral 7B with 2B tokens, Focus matches full attention at 13.82 vs 13.89 PPL.

Speed. Focus's sparsity pattern decomposes into two standard FlashAttention calls with no custom kernels, reaching 8.6× speedup at 1M tokens.

Training stable groups. Focus assigns tokens to groups and restricts distant attention. We found that training exhibits group dominance—one group absorbs all tokens, collapsing the learned sparsity. We identify three pathways through which dominance occurs and show that standard mitigations all fail. Our solution, Sinkhorn normalization, enforces balanced groups as a structural constraint.

### Our contributions are as follows.

- 1. We introduce Focus, the first composable efficient attention method that can be retrofitted onto any pretrained model with improved quality and zero benchmark degradation.
- 2. We identify group dominance—a training instability analogous to expert collapse in Mixture of Experts [\[Fedus et al.,](#page-11-3) [2022\]](#page-11-3)—and solve it with Sinkhorn normalization.
- 3. We show zero degradation when composing Focus onto models from 124M to 70B across five attention architectures.
- 4. We show that less attention can improve quality, shedding light on the assumption that n 2 attention is the quality ceiling.
- 5. We show that token routing requires only a 16-dimensional projection (dg=16, 148K parameters): token group assignment is far simpler than attention itself.

# <span id="page-1-0"></span>2 Method: Focus

In standard attention, for a sequence of T tokens, Q, K, V ∈ R <sup>T</sup> <sup>×</sup><sup>d</sup> are projected from hidden states, and each token attends to all others via softmax(QK⊤/ √ d)V, computing all T 2 token pairs. We propose to replace the full T × T score matrix QK<sup>⊤</sup> with two levels: (1) distant tokens attend only if they belong to the same learned group, and (2) nearby tokens always attend to each other within a local window.

**Learned grouping.** Let  $\mathbf{C} \in \mathbb{R}^{K \times d_g}$  be the learnable centroid vectors that define K token groups. A learned projection  $W_g \in \mathbb{R}^{d \times d_g}$  maps tokens into the centroid space. The soft group assignment for token i is:

$$\mathbf{g}_i = \text{normalize}\left(\frac{W_g \mathbf{h}_i \cdot \mathbf{C}^{\top}}{\tau}\right) \in \mathbb{R}^K$$
 (1)

where  $\tau$  is temperature.

We found that softmax normalization leads to group collapse (Section 4), and use **Sinkhorn** normalization to enforce balanced groups as a structural constraint. Given scores  $\mathbf{S} \in \mathbb{R}^{T \times K}$ :

- 1.  $\mathbf{Q} \leftarrow \exp(\mathbf{S}/\tau)$
- 2. For i = 1 to  $N: \mathbf{Q} \leftarrow \mathbf{Q}/\text{sum}(\mathbf{Q}, \text{dim=tokens})$ , then  $\mathbf{Q} \leftarrow \mathbf{Q}/\text{sum}(\mathbf{Q}, \text{dim=groups})$

After N=10 iterations, assignments are approximately doubly-stochastic: both row sums (each token's total assignment) and column sums (each group's total mass) are equalized. This prevents any single group from dominating, while still allowing the LM gradient to learn which tokens belong to which group.

**Gated attention.** The group affinity between tokens i and j is  $\mathbf{g}_i^{\top} \mathbf{g}_j$ : tokens in the same group have high affinity, tokens in different groups have low affinity. We use this to combine local windowed attention with group-gated distant attention:

$$s_{ij} = \mathbf{q}_i^{\top} \mathbf{k}_j \cdot \left( \mathbf{1}_{\text{local}}(i, j) + (1 - \mathbf{1}_{\text{local}}(i, j)) \cdot \sigma(\lambda \cdot \mathbf{g}_i^{\top} \mathbf{g}_j) \right)$$
(2)

Local tokens (within window w) always attend with full attention. For distant tokens in different groups,  $\mathbf{g}_i^{\mathsf{T}} \mathbf{g}_j \approx 0$ , so the gate drives  $s_{ij} \to 0$ —these pairs are pruned. Only same-group distant pairs survive. The gate determines whether information flows; the standard score  $\mathbf{q}_i^{\mathsf{T}} \mathbf{k}_j$  determines how much.

**Separation of routing and attention.** A key design principle is that centroids determine who can attend to whom—routing only. Content flows via the pretrained QKV attention, which determines what information transfers. This separation is why composability works: the pretrained attention computation proceeds unchanged within each group.

Efficiency at inference. Note that during training, soft gating computes all  $O(n^2)$  pairs, and there is no training-time speedup. At inference, each token is assigned to its top-k groups from  $\mathbf{g}_i$ , and two tokens attend only if they share at least one group. Different-group distant pairs are never computed—eliminated entirely, not merely scaled to zero.

The sparsity pattern decomposes into two standard FlashAttention [Dao et al., 2022, Dao, 2024] calls with no custom kernels:

- 1. Local: flash\_attn\_func with sliding window (O(nw)).
- 2. **Group:** Sort tokens by group (stable sort preserves causal order), reshape into K sequences, call flash\_attn\_func with causal=True  $(O(n^2/K))$ .

The key insight is that these two sets are **disjoint by construction**: set  $\mathcal{A}$  (same-group) requires g(i) = g(j), while set  $\mathcal{B}$  (cross-group local) requires  $g(i) \neq g(j)$ . Because  $\mathcal{A} \cap \mathcal{B} = \emptyset$  and  $\mathcal{A} \cup \mathcal{B}$  covers all attended pairs, the logsumexp merge is mathematically exact—no double-counting, no subtraction, no numerical instability. Sorting adds  $O(n \log n)$  overhead, negligible at long sequences (12ms at 1M tokens vs 1.5s for attention). This achieves  $8.6 \times$  speedup at 1M tokens (Table 6; full decomposition details and correctness proof in Appendix D).

<span id="page-3-2"></span>Table 1: Retrofit comparison on GPT-2 124M / PG-19. Focus is the only method that improves PPL and preserves all benchmarks.

| Method                               | Params | PPL ↓ | HellaSwag | ARC-E | PIQA | LAMBADA |
|--------------------------------------|--------|-------|-----------|-------|------|---------|
| Pretrained (full attn)               | 0      | 42.8  | 31.1      | 39.5  | 62.5 | 32.6    |
| Longformer [Beltagy et al., 2020]    | 0      | 38.9  | 30.0      | 37.5  | 58.9 | 6.6     |
| Performer [Choromanski et al., 2021] | 0      | 112.0 | 26.9      | 30.8  | 55.0 | 0.3     |
| Routing Trans. [Roy et al., 2021]    | 0      | 37.4  | 29.6      | 38.3  | 58.4 | 6.4     |
| Full attention FT                    | 124M   | 36.4  | 30.0      | 37.8  | 59.9 | 7.8     |
| Focus (ours)                         | 100K   | 36.2  | 31.1      | 39.5  | 62.5 | 32.6    |

How many dimensions does grouping need? Recall that the projection W<sup>g</sup> ∈ R <sup>d</sup>×d<sup>g</sup> maps tokens into the centroid space. This can be low-rank: rather than using the full d-dimensional space, we project into a small dg-dimensional subspace. On GPT-2 124M, we find that dg=16 suffices:

| dg         | Centroid params | % of model | PPL  |
|------------|-----------------|------------|------|
| 768 (full) | 7.1M            | 5.39%      | 34.8 |
| 128        | 1.2M            | 0.90%      | 34.5 |
| 32         | 296K            | 0.22%      | 34.5 |
| 16         | 148K            | 0.11%      | 34.5 |

A 16-dimensional subspace gives 50× fewer parameters than the full projection with no quality loss. This shows that token grouping is inherently low-dimensional: deciding which group a token belongs to is much simpler than computing attention itself.

# <span id="page-3-0"></span>3 Experiments

We evaluate Focus on two axes: quality and speed. Section [3.1](#page-3-1) compares against four baselines on GPT-2 124M. Section [3.2](#page-4-0) scales this to seven models from 124M to 70B. Section [3.3](#page-5-0) compares with LoRA. Section [3.5](#page-6-0) verifies quality at long contexts. Section [3.6](#page-7-1) examines the speed–quality tradeoff.

## <span id="page-3-1"></span>3.1 Comparison with Prior Methods

We compare Focus against efficient attention methods that can be retrofitted onto pretrained models, all evaluated on GPT-2 (124M) with PG-19. Full attention FT and Focus are trained for 4000 steps on PG-19. All methods use sequence length 1024.

Table [1](#page-3-2) shows three levels of retrofit quality. Longformer, Performer, and Routing Transformer impose fixed structural patterns that miss long-range dependencies, degrading LAMBADA by 25–32 points. Full attention fine-tuning updates all 124M parameters and degrades every benchmark (HellaSwag −1.1, ARC-E −1.7, PIQA −2.6, LAMBADA −24.8). Focus, composed onto the same pretrained model, improves PPL (42.8→36.2) with exactly zero downstream degradation composability preserves pretrained capabilities while improving domain quality.

Figure [1](#page-4-1) plots PPL vs wall-clock speedup for all methods. Focus is the only method that is both faster and better quality than full attention.

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 1: Quality-speed Pareto frontier of efficient attention retrofits on GPT-2 124M / PG-19 (seq\_len=64K). Y-axis is inverted: higher position means lower (better) PPL. Only Focus occupies the upper-right quadrant: better quality and faster than full attention.

#### <span id="page-4-0"></span>3.2 Scaling to Larger Models

Section 3.1 showed composability on GPT-2 124M. Is this scalable? We apply the same centroid-only training (all model weights frozen) to seven models from 124M to 70B, spanning five attention architectures: MHA (GPT-2), GQA (Mistral, Qwen, OLMo, LLaMA-2), GQA+bias, MHA+QK-norm, and interleaved+softcap.

<span id="page-4-2"></span>Table 2: Focus composed onto seven models. Only centroids trained on PG-19; all pretrained weights frozen. PPL column shows pretrained  $\rightarrow$  Focus. Benchmark columns show Focus scores, which are identical to pretrained (zero degradation).

| Model         | PG-19 PPL               | HellaSwag | ARC-E | PIQA | LAMBADA |
|---------------|-------------------------|-----------|-------|------|---------|
| GPT-2 124M    | $42.8 \rightarrow 34.2$ | 31.1      | 39.5  | 62.5 | 32.6    |
| GPT-2774M     | $25.7 \rightarrow 21.7$ | 45.3      | 46.6  | 69.2 | 47.7    |
| Mistral 7B    | $10.8 \rightarrow 11.6$ | 81.2      | 79.4  | 82.6 | 75.3    |
| Qwen~2.5~7B   | $19.3 \rightarrow 20.3$ | 78.3      | 76.2  | 80.0 | 70.7    |
| OLMo-27B      | $16.4 \rightarrow 16.9$ | 80.5      | 82.9  | 81.0 | 73.2    |
| LLaMA-2 13B   | $11.7 \rightarrow 11.7$ | 79.6      | 76.5  | 80.4 | 76.6    |
| LLaMA-2 $70B$ | $7.6 \rightarrow 8.3$   | 84.0      | 79.4  | 82.4 | 79.4    |

Table 2 confirms three findings. First, zero benchmark degradation holds for all the models: the worst drop across all models and benchmarks is -0.3%, within noise. Second, **PPL** improves at smaller scales (GPT-2 124M: -8.6, GPT-2 774M: -4.0) but shows a small cost at larger scales with top-k=2 (Mistral 7B: +0.8, LLaMA-2 70B: +0.7). Increasing the number of groups each token belongs to (top-k=3 instead of 2) recovers the pretrained PPL exactly at all scales, confirming that the centroid mechanism itself introduces no quality loss. Third, **centroid** overhead is negligible: as few as 0.015% of parameters at 70B scale.

Generation quality. The benchmarks above are classification tasks. To test whether Focus preserves autoregressive generation, we evaluate 8-shot chain-of-thought on GSM8K (1319 math word problems) using Mistral 7B with centroid-only training. Focus achieves 39.3% accuracy vs 40.6% for the full-attention baseline.

### <span id="page-5-0"></span>3.3 Comparison with LoRA

A key claim of Focus is composability: adding centroid parameters without degrading pretrained capabilities. Does this hold simply because few parameters are added? To test this, we compare with LoRA [\[Hu et al.,](#page-11-2) [2022\]](#page-11-2)—the most widely used small-parameter adaptation method—at a similar parameter budget on GPT-2 124M.

Table 3: Focus vs LoRA on GPT-2 124M / PG-19.

<span id="page-5-1"></span>

| Method            | Params | PPL ↓ | HellaSwag | ARC-E | LAMBADA | PIQA |
|-------------------|--------|-------|-----------|-------|---------|------|
| LoRA (r=4)        | 147K   | 31.6  | −0.4      | −1.6  | −2.1    | −0.5 |
| LoRA (r=16)       | 590K   | 31.2  | −0.3      | −1.6  | −1.5    | −0.4 |
| Centroids (dg=16) | 148K   | 34.2  | ±0.0      | ±0.0  | ±0.0    | ±0.0 |

Table [3](#page-5-1) shows that LoRA degrades every benchmark at both ranks, while Focus achieves exactly zero degradation at a similar parameter budget (148K). We conjecture the reason is that LoRA modifies weight matrices (∆W = AB), which can disrupt pretrained knowledge, while Focus only adds routing without modifying any original weights.

Alignment preservation. Tables [1–](#page-3-2)[3](#page-5-1) compared Focus and LoRA on base pretrained models. In practice, many deployed models are instruction-tuned and aligned for safety. Adapting such models to new domains risks undoing the alignment—a well-known problem in deployment. How do Focus and LoRA affect alignment when adapting such models? We test by applying both methods to Mistral-7B-Instruct and measuring TruthfulQA alongside standard benchmarks:

Table 4: Alignment preservation on Mistral-7B-Instruct (2000 training steps).

| Method                       | Params | PPL  | TQA MC1 | HellaSwag | ARC-E | LAMBADA |
|------------------------------|--------|------|---------|-----------|-------|---------|
| Instruct baseline            | 0      | 17.9 | 39.7    | 74.4      | 77.2  | 69.1    |
| + Centroids (K=2)            | 2.1M   | 18.0 | 40.0    | 74.4      | 77.2  | 69.1    |
| + LoRA (r=4, lr= 10−5<br>)   | 1.7M   | 16.1 | 40.1    | 72.6      | 76.4  | 69.1    |
| + LoRA (r=4, lr= 5×10−5<br>) | 1.7M   | 17.9 | 33.3    | 63.9      | 64.5  | 56.0    |
| + LoRA (r=4, lr= 10−4<br>)   | 1.7M   | 20.8 | 28.5    | 31.2      | 31.6  | 16.4    |

Focus slightly improves TruthfulQA (+0.3) and preserves all other benchmarks with zero degradation. LoRA degrades benchmarks across all settings tested, and is highly sensitive to learning rate: at 10−<sup>5</sup> it preserves TruthfulQA (40.1) but degrades HellaSwag by −1.9; at 5×10−<sup>5</sup> , benchmarks collapse (−10.5 HellaSwag, −13.1 LAMBADA) while PPL shows zero improvement (17.9, unchanged)—the model has forgotten without learning. No LoRA learning rate achieves zero degradation across all benchmarks.

### <span id="page-6-2"></span>3.4 Full Training with Sparsity

<span id="page-6-1"></span>Sections [3.1–](#page-3-1)[3.3](#page-5-0) used centroid-only training (frozen weights). What if we fine-tune all the parameters — both the centroids and the original model weights? Both Focus and full attention are fine-tuned on PG-19. At inference, the full attention baseline attends to all T tokens for each token. In Focus, each token attends to ∼T /8 tokens in the same group plus 128 local tokens.

| Table 5: PG-19 PPL across three scales (all parameters fine-tuned). |  |
|---------------------------------------------------------------------|--|
|---------------------------------------------------------------------|--|

| Model / Method    | PPL  | Params trained |
|-------------------|------|----------------|
| GPT-2 124M        |      |                |
| Full attention FT | 31.4 | 124M           |
| Focus FT          | 30.3 | 124M           |
| GPT-2 Large 774M  |      |                |
| Full attention FT | 20.4 | 774M           |
| Focus FT          | 20.7 | 774M           |
| GPT-2 XL 1.5B     |      |                |
| Full attention FT | 19.3 | 1.5B           |
| Focus FT          | 19.7 | 1.5B           |

Table [5](#page-6-1) shows that at 124M, Focus surpasses full attention (30.3 vs 31.4). At 774M and 1.5B, Focus closely matches full attention (within 0.3–0.4 PPL).

Multiple domains. To verify that Focus is not specific to PG-19, we apply the same full fine-tuning setup as Table [5](#page-6-1) to two additional domains (GPT-2 124M):

| Dataset                  | Full FT | Focus FT | ∆    |
|--------------------------|---------|----------|------|
| PG-19 (books)            | 31.4    | 30.3     | −1.1 |
| WikiText-103 (Wikipedia) | 21.4    | 21.3     | −0.1 |
| OpenWebText (web)        | 22.2    | 21.7     | −0.5 |

Focus matches or outperforms full attention on all three datasets without any dataset-specific tuning.

Training from scratch at 7B. Does Focus require a pretrained model? We train a 7B model from scratch on 2B tokens of OpenWebText with Focus (K=4) and compare against an identical model with full attention. Focus matches full attention: 13.82 vs 13.89 PPL, confirming that sparse group-gated attention loses nothing even without pretrained weights.

### <span id="page-6-0"></span>3.5 Long-Context Quality Preservation

All prior experiments use sequence length 1024. The practical motivation for efficient attention is long sequences, where O(n 2 ) cost dominates. We load the Mistral 7B centroids trained at T=1024 and evaluate at T ∈ {1024, 2048, 4096, 8192} on PG-19, varying the number of groups each token belongs to (top-k):

| Seq Length | Full attn | Focus                |                        |  |  |  |
|------------|-----------|----------------------|------------------------|--|--|--|
|            |           | top-k=2 (2× speedup) | top-k=3 (1.3× speedup) |  |  |  |
| 1,024      | 6.13      | 6.39                 | 6.13                   |  |  |  |
| 2,048      | 5.84      | 6.13                 | 5.84                   |  |  |  |
| 4,096      | 5.45      | 5.76                 | 5.45                   |  |  |  |
| 8,192      | 6.10      | 6.57                 | 6.05                   |  |  |  |

Two findings. First, centroids trained at T=1024 transfer to 8× longer sequences without retraining. Second, the PPL gap for top-k=2 stays small (+0.26–0.47) and does not grow with sequence length. Top-k=3 matches the baseline exactly at all lengths.

## <span id="page-7-1"></span>3.6 Speed–Quality Tradeoff?

Sparse attention typically sacrifices quality for speed. Does Focus follow this tradeoff? At inference, each token is assigned to its top-k highest-scoring groups; two tokens attend only if they share at least one group. Thus a smaller k means fewer groups, more sparsity and faster inference. We measure wall-clock speedup and quality across different top-k and K settings.

<span id="page-7-0"></span>Table 6: Wall-clock speedup of Focus over full attention (both using FlashAttention) on H100-80GB.

| Context     | 1K   | 4K   | 16K  | 32K  | 65K  | 262K | 1M   |
|-------------|------|------|------|------|------|------|------|
| K=4 speedup | 0.2× | 0.5× | 1.5× | 2.2× | 3.0× | 4.0× | 4.1× |
| K=8 speedup | 0.2× | 0.6× | 1.8× | 3.1× | 4.7× | 7.6× | 8.6× |

The theoretical speedup is K×: each of K groups attends over n/K tokens, costing K ·(n/K) <sup>2</sup> = n <sup>2</sup>/K. The measured 4.1× at K=4 and 8.6× at K=8 are consistent with this estimate; the slight bonus comes from FlashAttention being more efficient on shorter per-group sequences. At short contexts (≤4K), the overhead of sorting and two separate kernel launches exceeds the savings.

The parameter k controls the sparsity level, from full sparsity (k=1, K× speedup) to full attention (k=K, 1×). Table [7](#page-7-2) sweeps k on GPT-2 124M and Mistral 7B [\[Jiang et al.,](#page-11-6) [2023\]](#page-11-6) (K=4 groups).

<span id="page-7-2"></span>Table 7: Speed–quality tradeoff by varying top-k group membership at inference (K=4, PG-19). GPT-2 pretrained: 42.8 PPL; Mistral pretrained: 10.8 PPL.

| Model      | top-k      | PPL  | Speedup | Pairs retained | ∆ vs Pretrained |
|------------|------------|------|---------|----------------|-----------------|
|            | 1 (argmax) | 82.9 | 4.0×    | 26%            | +40.1           |
|            | 2          | 41.3 | 2.0×    | 60%            | −1.5            |
| GPT-2 124M | 3          | 47.2 | 1.3×    | 100%           | +4.4            |
|            | 4 (full)   | 47.2 | 1.0×    | 100%           | +4.4            |
|            | 1 (argmax) | 16.2 | 4.0×    | 25%            | +5.4            |
|            | 2          | 11.6 | 2.0×    | 73%            | +0.7            |
| Mistral 7B | 3          | 10.8 | 1.3×    | 100%           | +0.0            |
|            | 4 (full)   | 10.8 | 1.0×    | 100%           | +0.0            |

Three findings emerge. First, fewer groups is better: top-k=2 (41.3 PPL) outperforms top-k=3 and k=4 (both 47.2)—more sparsity yields better quality, answering the title's question. Second, top-k=2 even surpasses pretrained quality at 124M (41.3 vs 42.8) with 2× speedup. At 7B, the cost is just +0.7 PPL. Third, argmax (k=1) is too aggressive (82.9 PPL), but k=2 recovers fully.

## <span id="page-8-0"></span>4 Training Stable Groups

When training centroids with softmax assignment, we found that one group absorbed all tokens within 600 steps, reducing Focus to expensive full attention. Similar to load imbalance in Mixture of Experts [Fedus et al., 2022], this is a form of routing collapse, which we call *group dominance*. It has three independent escape pathways that were hard to battle:

- Path A—Centroid drift: the LM gradient shifts centroids so all tokens match one centroid.
- Path B—Representational bypass (full FT only): even with centroids frozen, hidden states shift toward one centroid direction.
- <span id="page-8-1"></span>• Path C—Projection bypass: even with EMA centroids and detached inputs, the learned projection maps all tokens to the same direction.

| Table 6. Three esca                 | Table 6. Three escape pathways and intigations attempted. |          |              |                       |  |  |  |  |
|-------------------------------------|-----------------------------------------------------------|----------|--------------|-----------------------|--|--|--|--|
| Method                              | A                                                         | В        | $\mathbf{C}$ | Outcome               |  |  |  |  |
| Entropy + balance loss              | Partial                                                   | Х        | X            | Collapses by step 600 |  |  |  |  |
| Stop-gradient on inputs             | X                                                         | ✓        | X            | Slow, not converging  |  |  |  |  |
| ${\rm EMA\ centroids} + {\rm proj}$ | ✓                                                         | X        | X            | Proj erases structure |  |  |  |  |
| Recluster every 100 steps           | Periodic                                                  | Periodic | X            | Balanced but unstable |  |  |  |  |
| Balance weight $\times 5$           | Partial                                                   | X        | X            | 6 of 8 groups die     |  |  |  |  |
| Sinkhorn (ours)                     | ✓                                                         | ✓        | 1            | Stable, semantic      |  |  |  |  |

Table 8: Three escape pathways and mitigations attempted

Why soft losses fail. Table 8 summarizes our attempts:

- Entropy and balance losses only address Path A, and collapse by step 600.
- Stop-gradient on inputs blocks Path B but not A or C.
- EMA centroids block A but the projection erases structure via Path C.
- Reclustering periodically resets balance but produces unstable groups.

There is a fundamental issue underlying these failures. Full attention minimizes training loss because the model can access all tokens. The gradient therefore always pushes to remove attention restrictions. This destroys the groups before they become useful. Interestingly, this is at odds with our finding that sparse attention *improves* quality (Section 3.4), suggesting that better generalization requires enforcing sparsity as a constraint, not learning it from the gradient alone.

Why Sinkhorn works. As defined in Section 2, Sinkhorn normalization enforces balanced groups as a structural constraint rather than a soft loss. This blocks all three pathways: even if centroids drift (A), representations shift (B), or the projection collapses (C), the Sinkhorn iterations redistribute the resulting scores to maintain balance.

Does Sinkhorn hold under full fine-tuning? Full fine-tuning is the hardest test because all three pathways are active. To test this, we first establish group structure with frozen model weights by training only the centroids (Phase 1). Then we apply full fine-tuning (all parameters updated; Phase 2). The question is whether balanced groups survive Phase 2, for Softmax and Sinkhorn normalization.

|                                    | Centroid-only  |                | Full fine-tuning |                |
|------------------------------------|----------------|----------------|------------------|----------------|
| Method                             | Dominance      | Stability      | Dominance        | Stability      |
| Softmax + balance loss<br>Sinkhorn | 15.0%<br>14.6% | 0.966<br>0.953 | 99.4%<br>15.9%   | 1.000<br>1.000 |

Dominance is the fraction of tokens in the largest group; with K=8, perfect balance is 12.5%. Both produce near-balanced groups after centroid-only training (∼15%). After full fine-tuning, softmax collapses—one group absorbs 99.4% of all tokens, and the sparsity is lost. Sinkhorn remains balanced at 15.9%. Sinkhorn is robust to hyperparameters: fine-tuned PPL varies only 0.6 across 16 configurations (Appendix [B\)](#page-14-0).

# 5 The Learned Group Structures

What do the groups discover? It is an interesting question, because the group training is end to end and no enforcement of group structure is used. Regardless, we found there are linguistic structures in the learned groups. When trained with Sinkhorn normalization (K=8, τ=0.1), centroids discover interpretable linguistic categories without supervision:

| Group | Category               | Top tokens                               |
|-------|------------------------|------------------------------------------|
| G4    | Punctuation (96% pure) | , (×55), . (×24), ; (×4), – (×7)         |
| G3    | Determiners            | the (×38), a (×14), this (×5), my (×3)   |
| G0    | Prepositions           | to (×14), of (×14), in (×13), for (×5)   |
| G5    | Connectives            | who (×7), which (×7), and (×6), but (×5) |
| G7    | Verbs + pronouns       | have (×6), are (×5), is (×4), I (×4)     |
| G1    | Content/nouns          | Nature, freedom, Land, sense, home       |

Assignment confidence is high (avg 0.89) and groups are balanced (10–16% each). These categories persist through fine-tuning of all 124M parameters. Notably, prepositions and determiners form separate groups—traditional POS tagging lumps them together as "function words," but Focus discovers they serve different attention roles: determiners point to their noun; prepositions link phrases across distance.

Long-range pairing examples. The learned groups enable same-group tokens to attend across long distances. Here are concrete examples from a PG-19 passage: 'Henry' (pos 18) → 'Walker' (pos 772), distance 754, group affinity 0.945 (entity tracking); 'When' (pos 2) → 'since' (pos 390), affinity 0.988 (temporal connectives). These groupings emerge end-to-end from the language modeling objective alone—no supervision on group semantics is provided. Focus discovers these groupings and uses the learned structure to determine which token pairs attend at long range.

# 6 Related Work

Efficient attention methods fall into three families. Sparse attention methods (Longformer [\[Beltagy](#page-11-0) [et al.,](#page-11-0) [2020\]](#page-11-0), BigBird [\[Zaheer et al.,](#page-13-0) [2020\]](#page-13-0)) use fixed positional patterns with exact softmax. They cannot adapt to content and degrade quality when retrofitted. Linear attention (Performer [\[Choromanski et al.,](#page-11-1) [2021\]](#page-11-1)) replaces softmax with kernel approximations; it diverges catastrophically in the retrofit setting (+75.6 PPL). Low-rank attention (Linformer [\[Wang et al.,](#page-13-1) [2020\]](#page-13-1)) projects keys/values to fewer positions but is incompatible with causal modeling.

Routing Transformer [\[Roy et al.,](#page-12-1) [2021\]](#page-12-1) is our closest prior work—both use content-based routing. Key differences: (1) online k-means (transient) vs learned centroids (stable); (2) replaces attention mask vs gates existing attention; (3) no balancing vs Sinkhorn.

Mixture of Experts [\[Fedus et al.,](#page-11-3) [2022\]](#page-11-3) and Focus both route computation via learnable parameters, but MoE routes tokens to FFN experts while Focus routes attention connections. The two are complementary; our Sinkhorn solves the analogous load-balancing problem.

Token selection methods [\[Ribar et al.,](#page-13-2) [2024,](#page-13-2) [Chen et al.,](#page-13-3) [2024,](#page-13-3) [Zhang et al.,](#page-13-4) [2024,](#page-13-4) [Singhania](#page-13-5) [et al.,](#page-13-5) [2024\]](#page-13-5) select individual tokens per query without learning, while Focus learns group structure across the entire sequence. The approaches are complementary.

LoRA [\[Hu et al.,](#page-11-2) [2022\]](#page-11-2) is the dominant parameter-efficient adaptation method (see also DoRA [\[Liu et al.,](#page-12-2) [2024b\]](#page-12-2)). We compare in Section [3.3.](#page-5-0)

# 7 Limitations

The limitations of our Focus are as follows.

Training cost. Soft gating computes all O(n 2 ) pairs during training, so efficiency is inference-only for now. Training directly with discrete assignments remains open.

Quality benefit diminishes with scale. Focus surpasses full attention at 124M but only matches it at 774M–1.5B (within 0.3–0.4 PPL). Although this is good for a sparse model, it seems larger models are less susceptible to noisy attention patterns. The good thing is that the efficiency benefit (speedup) still grows with sequence length regardless of scale.

Routing overhead at short sequences. Sorting and gather/scatter add ∼12ms constant overhead, which dominates at sequences ≤4K. Focus offers no speedup below 16K tokens.

# 8 Conclusion

We introduce Focus, a composable sparse attention method. Lightweight centroid modules are composed onto a pretrained model's attention layers, making the attention sparse by gating which token pairs can attend at long range. All original weights stay frozen; only the centroids are trained. This composability is the key property: Focus can be applied to any pretrained model—regardless of size, architecture, or training recipe. A comparison against four efficient attention baselines shows Focus is the only method that achieves improved quality, zero benchmark degradation, and wall-clock speedup. This composability holds from 124M to 70B across five attention architectures. Learning which tokens to attend to, rather than attending to all or selecting heuristically, is an effective approach to efficient attention. Our results indicate that full attention can be improved by sparse attention in terms of quality.

# References

- <span id="page-11-0"></span>Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
- Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Havens, Robert Jennings, Daniel King, Sam Havens, Nick Blankenship, et al. LoRA learns less and forgets less. Transactions on Machine Learning Research, 2024.
- Peter F Brown, Vincent J Della Pietra, Peter V deSouza, Jennifer C Lai, and Robert L Mercer. Class-based n-gram models of natural language. Computational Linguistics, 18(4):467–480, 1992.
- Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Advances in Neural Information Processing Systems, 2020.
- <span id="page-11-1"></span>Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations, 2021.
- <span id="page-11-5"></span>Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024.
- Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning, 2024.
- <span id="page-11-4"></span>Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.
- DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024a.
- DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024b.
- <span id="page-11-3"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
- Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.
- Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Joshi, Valentina Pyatkin, et al. OLMo: Accelerating the science of language models. In Annual Meeting of the Association for Computational Linguistics, 2024.
- <span id="page-11-2"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.
- <span id="page-11-6"></span>Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024a.
- Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, 2024b.
- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, 2020.
- Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Piotrowski, Piotr Sankowski, Michał Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Marek Jaszczur, et al. Scaling laws for fine-grained mixture of experts. In International Conference on Machine Learning, 2024.
- Opher Lieber, Barak Lenz, Horace Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Amnon Shashua, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887, 2024.
- Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for nearinfinite context. In International Conference on Learning Representations, 2024a.
- <span id="page-12-2"></span>Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation. In International Conference on Machine Learning, 2024b.
- Llama Team. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- Shuming Lu et al. MoBA: Mixture of block attention for long-context LLMs. arXiv preprint arXiv:2502.13189, 2025.
- Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pages 109–165. Elsevier, 1989.
- Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts. In International Conference on Learning Representations, 2024.
- <span id="page-12-1"></span>Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Advances in Neural Information Processing Systems, 2024.
- <span id="page-12-0"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

- <span id="page-13-1"></span>Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- Tianzhu Ye, Li Li, Gao Huang, et al. Differential transformer. In International Conference on Learning Representations, 2025.
- Jingyang Yuan, Huazuo Liu, Zhaozhuo Zhang, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Annual Meeting of the Association for Computational Linguistics, 2025.
- <span id="page-13-0"></span>Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems, 2020.
- Michael Zhang, Kush Bhatia, Jonathan Ragan-Kelley, and Christopher Ré. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. In International Conference on Learning Representations, 2024.
- <span id="page-13-4"></span>Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Zhangyang Wang, Beidi Chen, and others. H2O: Heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- <span id="page-13-5"></span>Prajwal Singhania, Siddharth Nrusimha, Chih-Ping Park, and Joo-Young Kim. Loki: Low-rank keys for efficient sparse attention. arXiv preprint arXiv:2406.02542, 2024.
- <span id="page-13-2"></span>Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Sheridan, Thang Bui, and Walterio Mayol-Cuevas. SparQ Attention: Bandwidth-efficient LLM inference. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.
- <span id="page-13-3"></span>Zhuoming Chen, Ranajoy Sadhukhan, Ying Ye, Yang Chen, Baris Kasikci, and Hao Zheng. MagicPIG: LSH sampling for efficient LLM generation. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

# A Why Less Attention Produces Better Quality

The fact that Focus surpasses full attention—rather than merely approximating it—requires explanation. Three mechanisms contribute:

- 1. Softmax dilution. In full attention, softmax distributes probability mass across all n tokens, even when only a small subset is relevant. A pronoun at position 800 seeking its antecedent at position 200 must compete with hundreds of irrelevant distant tokens for attention weight. Focus restricts softmax to same-group tokens plus the local window, concentrating probability mass on a smaller, more relevant candidate set. The result is sharper, more informative attention distributions.
- 2. Noise removal. Irrelevant attention pairs do not merely waste computation—they actively degrade quality. Each irrelevant key–value pair contributes a small amount of noise to the attention output. Across 12 layers and 12 heads, this noise accumulates. Focus eliminates these pairs entirely: the model never computes attention over tokens it should ignore.
- 3. Implicit structural constraint. Full attention at 124M scale can memorize spurious long-range correlations in the training data. Restricting attention to semantically coherent groups acts as a structural prior—analogous to how L<sup>1</sup> penalties zero irrelevant features or dropout removes random connections. The restriction prevents the model from fitting noise in the attention pattern, without any explicit penalty term.

The key insight: full n <sup>2</sup> attention is not the performance ceiling—it is the unconstrained baseline. Learned sparsity improves upon it for the same reason that feature selection improves upon using all features: removing noise is not a cost, it is a benefit.

# <span id="page-14-0"></span>B Ablation Studies

Section [4](#page-8-0) showed that Sinkhorn normalization produces stable, balanced groups. Here we ablate four key hyperparameters on GPT-2 124M / PG-19, varying each while holding others at defaults (K=8, w=128, τ=0.1, Sinkhorn iters = 10).

Table 9: Ablation study (GPT-2 124M, PG-19). Each row varies one hyperparameter. Fine-tuned PPL is stable (29.9–30.5) across all 16 configurations.

| Parameter      | Value | Centroid PPL | Fine-tuned PPL | Dominance (centroid) | Dominance (full FT) |
|----------------|-------|--------------|----------------|----------------------|---------------------|
| Groups K       | 4     | 36.8         | 30.1           | 40%                  | 38%                 |
|                | 8     | 38.4         | 30.3           | 24%                  | 23%                 |
|                | 16    | 40.4         | 30.4           | 20%                  | 31%                 |
|                | 32    | 42.4         | 30.5           | 21%                  | 30%                 |
| Window w       | 64    | 38.3         | 30.2           | 17%                  | 17%                 |
|                | 128   | 38.4         | 30.2           | 26%                  | 23%                 |
|                | 256   | 38.1         | 30.3           | 26%                  | 25%                 |
|                | 512   | 38.6         | 30.0           | 27%                  | 28%                 |
| Temp τ         | 0.05  | 36.9         | 30.0           | 68%                  | 74%                 |
|                | 0.1   | 38.4         | 30.3           | 24%                  | 23%                 |
|                | 0.2   | 39.1         | 30.3           | 16%                  | 19%                 |
|                | 0.5   | 40.5         | 30.3           | 21%                  | 31%                 |
| Sinkhorn iters | 3     | 35.8         | 29.9           | 95%                  | 97%                 |
|                | 5     | 36.8         | 30.2           | 69%                  | 75%                 |
|                | 10    | 38.4         | 30.3           | 21%                  | 20%                 |
|                | 20    | 39.0         | 30.2           | 14%                  | 14%                 |

Fine-tuned PPL is robust. Across all 16 configurations, fine-tuned PPL ranges from 29.9 to 30.5—a spread of only 0.6 PPL. Focus is not sensitive to hyperparameter choices.

Sinkhorn iterations: a subtle trap. With 3 iterations, PPL appears best (29.9) but groups have collapsed to 95–97% dominance. This is not real Focus—it is effectively full attention with extra overhead. At low temperature (τ=0.1), exp(scores/0.1) produces extremely peaked distributions that 3 iterations cannot redistribute. At least 10 iterations are needed for balanced groups.

Window size: smaller is better. With K=2 centroid-only training: w=16 achieves the best PPL (33.8), beating w=128 by 0.8 PPL. At w=512 (half the sequence), quality drops by 3.7 PPL because most attention is handled locally, leaving little for group routing to contribute. This confirms that local and group attention are complementary.

# C Comparison with Recent Token-Selection Methods

We compare Focus against recent token-selection methods (SparQ [\[Ribar et al.,](#page-13-2) [2024\]](#page-13-2), MagicPIG [\[Chen et al.,](#page-13-3) [2024\]](#page-13-3)) on GPT-2 124M / PG-19. These methods select top-k=32 tokens per query at inference without modifying weights. Note that they operate at a different sparsity level than Focus: token selection at k=32 retains 3% of tokens per query, while Focus with K=4, top-k=2 retains ∼50% of distant pairs.

Table 10: Token-selection methods vs Focus on GPT-2 124M / PG-19 (k=32). Token-selection methods preserve downstream benchmarks but degrade PPL by 5–10 points. Focus improves PPL with zero benchmark degradation.

| Method                                              | PPL ↓        | HellaSwag    | ARC-E        | PIQA         | LAMBADA      |
|-----------------------------------------------------|--------------|--------------|--------------|--------------|--------------|
| Pretrained                                          | 42.8         | 31.1         | 39.5         | 62.5         | 32.6         |
| SparQ [Ribar et al., 2024]<br>SparQ (mean realloc.) | 52.8<br>48.3 | 31.3<br>31.2 | 39.4<br>39.3 | 62.4<br>62.3 | 34.3<br>33.1 |
| MagicPIG [Chen et al., 2024]                        | 52.8         | 31.3         | 39.4         | 62.5         | 34.0         |
| Focus (ours)                                        | 36.2         | 31.1         | 39.5         | 62.5         | 32.6         |

Token-selection methods preserve downstream benchmarks but degrade PPL by 5–10 points. Focus improves PPL (42.8→36.2) with exactly zero benchmark change. The methods achieve speedup through different mechanisms and operate at different sparsity levels, making direct comparison nuanced; we include this for completeness.

Focus exactly matches pretrained on all four benchmarks. SparQ and MagicPIG show minor fluctuations (±0.2–1.7 points) but no systematic degradation, indicating that downstream classification tasks are robust to token-level sparsity at this level. The critical distinction is perplexity: Focus improves PPL by 6.6 points while training-free methods degrade it by 5–10 points.

# <span id="page-15-0"></span>D FlashAttention Decomposition

The Focus attention mask under hard group assignment is:

$$\mathcal{M}(i,j) = \mathbf{1}[j \le i] \land (\mathbf{1}[g(i) = g(j)] \lor \mathbf{1}[i - j \le w])$$
(3)

where g(i) is the group assignment of token i and w is the local window size.

The overlap problem. The natural decomposition into same-group pairs S and local pairs L fails because S ∩ L ̸= ∅—same-group local pairs are double-counted. Subtraction in logsumexp space (log(exp(a) + exp(b) − exp(c))) is numerically catastrophic (cosine similarity 0.79 against reference).

Disjoint decomposition. We split M into two sets that are disjoint by construction:

$$\mathcal{A} = \{(i,j) : j \le i \land g(i) = g(j)\}$$
 (same-group causal) (4)

$$\mathcal{B} = \{(i,j) : j \le i \land i - j \le w \land g(i) \ne g(j)\}$$
 (cross-group local) (5)

A ∩ B = ∅ (one requires same group, the other different group) and A ∪ B = M (every attended pair is either same-group or cross-group-local). The logsumexp merge is mathematically exact.

Set A is computed by sorting tokens by group (stable sort preserves causal order), reshaping into K sequences, and calling flash\_attn\_func with causal=True. Complexity: O(n <sup>2</sup>/K).

Set B extracts local keys for each query and masks same-group pairs to −∞. Complexity: O(nw), never the bottleneck.

Merge: o[i] = (e ℓA[i] · oA[i] + e ℓB[i] · oB[i])/(e <sup>ℓ</sup>A[i] + e ℓB[i] ), where ℓA, ℓ<sup>B</sup> are per-query logsumexp values.

Empirical verification. All configurations achieve cosine similarity 1.0000 against the O(n 2 ) reference, confirming mathematical exactness. The complete implementation is 320 lines of Python using only flash\_attn\_func and standard PyTorch—no custom CUDA kernels, no Triton, no compilation.