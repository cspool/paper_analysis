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

