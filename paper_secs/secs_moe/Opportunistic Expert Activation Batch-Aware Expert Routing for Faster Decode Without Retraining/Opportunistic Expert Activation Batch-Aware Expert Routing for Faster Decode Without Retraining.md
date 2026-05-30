## OPPORTUNISTIC EXPERT ACTIVATION: BATCH-AWARE EXPERT ROUTING FOR FASTER DECODE WITHOUT RETRAINING

Costin-Andrei Oncescu 1 2 Qingyang Wu <sup>2</sup> Wai Tong Chung <sup>2</sup> Robert Wu <sup>2</sup> Bryan Gopal <sup>2</sup> Junxiong Wang <sup>2</sup> Tri Dao 3 2 Ben Athiwaratkun <sup>2</sup>

## ABSTRACT

An increasing number of LLMs employ Mixture-of-Experts (MoE) architectures where the feed-forward layer is replaced by a pool of experts and each token only activates a small subset of them. During autoregressive generation, these models often enter a memory-bound regime even for moderate batch sizes because the average expert load grows more slowly than in an equivalent dense feedforward layer. Consequently, MoE latency is governed by the number of activated experts. We introduce a framework for dynamically re-routing token-toexpert mapping to lower this number (and thus, the decode latency) while preserving a comparable quality. Our best results use a batch-aware routing that works by having tokens piggyback experts that have already been loaded into memory due to being crucial to other tokens within the same batch. Empirically, we evaluate our method on the Qwen3-30B and Qwen3-235B models with a batch size of 16. Without any statistically significant loss in accuracy, our approach achieves latency reductions of 39% and 15% in the MoE layer decode latency, respectively.

## 1 INTRODUCTION

Mixture-of-Experts (MoE) architectures have contributed significantly to the state-of-the-art in language modeling [\(Liu et al.,](#page-10-0) [2024a;](#page-10-0) [Kimi Team et al.,](#page-9-0) [2025;](#page-9-0) [Yang et al.,](#page-10-0) [2025a\)](#page-10-0). They replace the feedforward layer with a pool of experts — smaller feedforward layers — and route each input to only a small subset of the pool. By employing this sparse and conditional computation, MoEs decouple model size from the computation cost, allowing for more amenable model scaling.

When deploying these models, serving frameworks [\(Kwon](#page-10-0) [et al.,](#page-10-0) [2023;](#page-10-0) [Zheng et al.,](#page-10-0) [2024\)](#page-10-0) usually batch several requests and proceed in two steps: prefill and decode. During the prefill stage, prompts are processed together in parallel across sequence length, much as a normal forward pass would. Then, decoding is the process of sequentially (autoregressively) generating one new token at a time, in parallel across a batch. This stage has a lower arithmetic intensity than prefill and is often memory-bound [\(Rajbhandari](#page-10-0) [et al.,](#page-10-0) [2022\)](#page-10-0), where runtime is limited by data movement bandwidth rather than arithmetic throughput.

Because decoding dominates serving time for long se-

quences and interactive workloads, reducing its latency directly improves user experience and cost efficiency.

The problem. During decode, it takes a larger batch size to get into a regime where experts are not memory-bound. This is because when each token activates k experts out of N, the average per-expert load increases only at a rate of k/N which is low by design in MoEs (e.g. 1/16 in Qwen3). Coupled with the arithmetic intensity being roughly 100- 200 [\(NVIDIA,](#page-10-0) [2022\)](#page-10-0), the sparsity factor N/k results in required batch sizes of order of thousands for MoEs to be in compute bound regime (e.g. ≈ 1.6k for Qwen3). Hence, for moderate batch sizes, the latency of an MoE layer is not dominated by the computational load of individual experts, but rather by the overhead of fetching the weights of all activated experts from the high-bandwidth memory (HBM) to the on-chip one (SRAM) [\(Rajbhandari et al.,](#page-10-0) [2022\)](#page-10-0). Consequently, latency becomes effectively linear in the number of unique activated experts, a number that can grow quickly with batch size in spite of each token activating only a few experts; this is because we need to activate the *union* of all these small sets of experts.

This paper introduces Opportunistic Expert Activation (OEA), a batch-aware routing framework designed to lower decode latency by explicitly minimizing the number of unique active experts per batch during inference. OEA operates without any model retraining and comprises two stages:

<sup>1</sup>Harvard University. Part of the work was done when Costin was interning at Together AI. <sup>2</sup>Together AI <sup>3</sup> Princeton University. Correspondence to: Costin-Andrei Oncescu <concescu@g.harvard.edu>.

#### <span id="page-1-0"></span>Algorithm 1 Simplified OEA Routing Algorithm

```
1: Input: Token embeddings x_{1..B}, Initial number of ex-
     perts per token k, Sorted expert indices e_{i,j} for each
     token i and rank j. Hyperparameter: k_0.
 2: {Phase 1: Determine Baseline Experts}
 3: for i = 1 to B do
       S_i^{\text{base}} \leftarrow \{e_{i,1}, \dots, e_{i,k_0}\}
 7: {Phase 2: Opportunistic Piggybacking}
 8: S^{\text{base}} \leftarrow \bigcup_{i=1}^{B} S_i^{\text{base}} {Union of all required experts}
    for i = 1 to B do
        S_i \leftarrow S_i^{\text{base}} {Initialize final set with baseline}
10:
        for j = k_0 + 1 to N do
11:
12:
           if |S_i| > k then
              break
13:
14:
           end if
           if e_{i,j} \in S^{\mathrm{base}} then
15:
              S_i \leftarrow S_i \cup \{e_{i,j}\}
16:
17:
18:
        end for
19: end for
20: Output: Final expert sets S_1, \ldots, S_B
```

- Firstly, OEA sets a minimum quality baseline for each token by keeping the first few of its expert choices to guarantee crucial computations take place.
- Then, OEA opportunistically augments this baseline by routing tokens to additional, lower-priority experts only if those experts already need to be loaded due to another token's baseline requirement within the same batch.

This "piggybacking" mechanism allows the model to recover some of the performance that is potentially lost due to activating fewer experts, practically for free since it preserves the number of activated experts (and thus latency).

Relation to Prior Work. OEA is complementary to approaches that reduce the number of experts activated *per token* (Lu et al., 2024). In contrast, our piggybacking phase can be applied on top of such methods at no added cost. Moreover, unlike prior dynamic batch-aware routing strategies (Gupta et al., 2024), OEA guarantees a batch-independent quality baseline for every token, ensuring consistent per-token computation regardless of batch composition.

#### **Contributions** Our contributions are as follows:

 We formalize the MoE decode latency problem under a memory-bound roofline model, showing that reducing the number of unique active experts is the primary

![](_page_1_Figure_9.jpeg)

Figure 1. Mean MoE latency as a function of the number of activated experts within a decode batch. The average is computed over all layers and decode steps across a GPQA evaluation of the vanilla Qwen3-30B-A3B model.

optimization target.

- We propose OEA, a dynamic routing algorithm that provides a tunable trade-off between model quality and system performance.
- We evaluate OEA on the Qwen3-30B and Qwen3-235B models, demonstrating its ability to substantially reduce the number of active experts and, consequently, MoE latency, while maintaining performance on both downstream tasks and language modeling perplexity. At a batch size of 16, OEA achieves latency reductions of 39% on the 30B model and 15% on the 235B model.

## 2 BACKGROUND AND MOTIVATION

Modern state-of-the art MoE models such as Kimi K2 (Kimi Team et al., 2025), Deepseek-V3 (Liu et al., 2024a) and Qwen3 (Yang et al., 2025a) fundamentally incorporate the same setup (excluding potentially shared experts) popularized by Shazeer et al. (2017) — namely, they replace the feedforward layer of a transformer with sets of N experts  $E_1, \ldots E_N: \mathbb{R}^D \to \mathbb{R}^D$  (where D is the embedding dimension) and a router scoring function  $R: \mathbb{R}^D \to \Delta^N$  that assigns a normalized score  $R(x)_i$  to each expert i. The output of the MoE module is then computed via:

$$moe(\mathbf{x}) = \sum_{i \in S} \frac{R(\mathbf{x})_i}{\sum_{j \in S} R(\mathbf{x})_j} E_i(\mathbf{x})$$
 (1)

where  $S=\operatorname{Top}_k(R(\boldsymbol{x}))$  is the set of indices of top-k values of the router's scores. The extra normalization factor is optional, but enabled in Qwen3, the model we evaluate. Henceforth, we use B for batch size.

Typically, serving these models is done by batching requests and proceeding in the following two stages:

1. The **prefill stage**, where activations and KV caches are computed for the prompts. This passes over entire

<span id="page-2-0"></span>prompts' at once, thus increasing the effective (token) batch size (*i.e.*, sequence length × batch size) of the MoE layers, resulting in heavier loads for each expert.

2. The (iterative) decode stage where, at a given decode step, exactly one token of each sequence in the batch is processed for next-token prediction. Crucially, the effective batch size passed to the MoE layers is now only equal to the batch size.

Note that the low effective (token) batch size seen during decoding is further exacerbated by the fact that the average expert load only increases at a rate of k/N per token. This raises the threshold batch size for reaching the computebound regime, implying that even moderately-sized batches can still result in memory-bounded experts. In this regime, the time to fetch expert's weights from HBM into on-chip SRAM dominates the time needed to compute Ei's outputs. Consequently, for each expert, latency depends primarily on whether it is activated at all: if no token is routed to it, its weights need not be fetched and thus incur no latency, whereas once it is activated and fetched, the marginal cost to serving additional tokens is negligible. Therefore, when experts are not executed in parallel, overall MoE latency scales with the number of activated experts.

To illustrate how quickly this quantity grows, consider, for example, a setting where each token activates k = 8 experts out of N = 128 total ones (as is the case for Qwen3). For a batch size of 16 tokens, any number of experts between 8 and 128 could be employed. Assuming uniform routing (which the models are trained to balance), the expected number of activated experts is 82<sup>1</sup> . Note that this represents an increase of up to 10× over a batch size of 1 (where the token only triggers k = 8 experts). This is not the case in non-MoE architectures where both a batch size of 1 and one of 16 would deem a memory-bound regime and thus incur a fixed one-time fetching cost.

In summary, reducing the number of activated experts directly targets the dominant term of MoE decode latency in the memory-bound regime. We cover the rest of the related work in Section [5](#page-7-0) where we also put our method in perspective.

## 3 OUR FRAMEWORK

#### 3.1 Latency and Number of Activated Experts

To formalize the argument introduced in Section [2,](#page-1-0) we adopt a simplified latency model for the computation performed by one expert. Let f(n) represent the time it takes an expert to process n tokens and let it be given by f(0) = 0 and f(n) = an + b for n > 0. Here, b is the cost of fetching the expert's weights from the high-bandwidth memory (HBM) into on-chip SRAM, while a is the computation time it takes to process one token. It follows that the total latency of a whole MoE block is given by:

$$\sum_{i=1}^{N} f(\operatorname{cnt}_{i}) = \sum_{i=1}^{N} b \cdot \mathbb{1}_{\operatorname{cnt}_{i} > 0} + a \cdot \operatorname{cnt}_{i}$$
$$= b \cdot T + a \cdot Bk$$
 (2)

where cnt<sup>i</sup> is the number of tokens routed to expert E<sup>i</sup> ; T is the number of experts that have at least one token routed to them; B is the batch size; N is the total number of experts; and k is the number of experts activated per token.

Equation 2 shows that the overall latency is given by a memory-bound term linear in the number of active experts T and a compute-bound term linear in the total computation load Bk. Whether we are in a compute- or memory-bound regime only indicates which of these terms dominates, but as a general statement, it directly follows that reducing T lowers the latency. If the loads cnt<sup>i</sup> are small enough to be in a memory-bound regime [\(Rajbhandari et al.,](#page-10-0) [2022\)](#page-10-0), the total latency is dominated by b · T and thus we can expect almost proportional gains to the drop in T.

While this description is a simplification — it does not account for system-level effects such as kernel launch overhead, padding to equalize expert loads, or the use of optimized kernels like Grouped GEMM [\(Hejazi,](#page-9-0) [2024\)](#page-9-0) — these factors do not alter the constraint. Grouped GEMM can improve efficiency by batching computations for different experts, but it still requires all activated expert weights to be loaded into on-chip memory, meaning the latency remains fundamentally tied to T in the memory-bound regime. We confirm this empirically (Figure [1\)](#page-1-0).

Finally, note that cnt<sup>i</sup> ≤ B, since each token can route to an expert at most once; this holds for any potential rerouting as well. Furthermore, for the original top-k routing, if we are to further assume it to be uniform, it follows that E[cnt<sup>i</sup> ] = Bk/N which is much lower and thus increases the threshold for B to be in a compute-bound regime. We henceforth turn our focus on optimizing T and assume we are in a regime where this translates (as shown empirically) to lower overall latency.

To achieve this, we modify token routing during inference while preserving empirical performance. This approach is motivated by recent studies demonstrating the robustness of MoE models to re-routing [\(Li et al.,](#page-10-0) [2025;](#page-10-0) [Gupta et al.,](#page-9-0) [2024\)](#page-9-0). While several approaches have explored *static expert pruning* [\(Lu et al.,](#page-10-0) [2024;](#page-10-0) [Liu et al.,](#page-10-0) [2024b\)](#page-10-0) — permanently removing experts to save memory — this inevitably constrains the model's capacity. In contrast, our goal is to develop methods that maintain a minimum level of perfor-

<sup>1</sup>The exact formula is N(1 − (1 − k N ) <sup>B</sup>) where B is the batch size.

<span id="page-3-0"></span>Table 1. Ablation across k0: Benchmark accuracies for Phase 1 (pruned, top-k0) vs simplified OEA routing (top-k0+piggybacking) on Qwen3-30B-A3B. Pruned refers to using top k<sup>0</sup> experts per token, OEA does additional piggybacking and vanilla represents the default model. Results averaged over 4 runs. Setups that are no worse than vanilla (standard-error adjusted) are in bold.

|                                 | k0                   | = 3                  | k0                   | = 4                  | k0                   | = 5                  | k0                   | = 6                  | k0                   | = 7                  | VANILLA              |
|---------------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| BENCHMARK                       | PRUNED               | OEA                  | PRUNED               | OEA                  | PRUNED               | OEA                  | PRUNED               | OEA                  | PRUNED               | OEA                  |                      |
| AIME24<br>GPQA<br>LIVECODEBENCH | 51.2<br>45.7<br>37.4 | 80.0<br>58.6<br>61.2 | 75.8<br>54.3<br>58.2 | 81.9<br>59.3<br>62.7 | 80.6<br>56.2<br>63.2 | 81.5<br>61.1<br>62.0 | 80.2<br>58.3<br>63.1 | 80.8<br>62.2<br>63.1 | 82.5<br>59.7<br>63.0 | 78.5<br>60.6<br>62.5 | 80.4<br>60.2<br>62.1 |
| MATH 500                        | 91.1                 | 93.5                 | 92.7                 | 93.1                 | 92.6                 | 93.3                 | 93.1                 | 93.1                 | 93.3                 | 93.2                 | 92.8                 |

Table 2. Ablation across k0: Benchmark accuracies for Phase 1 (pruned, top-k0) and simplified OEA routing (top-k0+piggybacking) on Qwen3-235B-A22B. Pruned refers to using top k<sup>0</sup> experts per token, OEA does additional piggybacking and vanilla represents the default model. Results averaged over 3 runs. Setups that are no worse than vanilla (standard-error adjusted) are in bold.

|                                             | k0                          | = 3                          | k0                           | = 4                          | k0                           | = 5                          | k0                           | = 6                          | VANILLA                      |
|---------------------------------------------|-----------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| BENCHMARK                                   | PRUNED                      | OEA                          | PRUNED                       | OEA                          | PRUNED                       | OEA                          | PRUNED                       | OEA                          |                              |
| AIME24<br>GPQA<br>LIVECODEBENCH<br>MATH 500 | 17.5<br>43.8<br>5.7<br>80.9 | 81.4<br>66.3<br>63.4<br>94.4 | 69.4<br>56.4<br>27.4<br>93.3 | 82.5<br>67.7<br>67.1<br>94.8 | 81.9<br>60.6<br>53.5<br>94.5 | 83.6<br>67.5<br>66.1<br>94.7 | 82.8<br>64.1<br>60.8<br>94.5 | 83.6<br>67.5<br>66.1<br>94.3 | 85.0<br>68.4<br>68.5<br>94.7 |

.

mance in the worst case while enabling full recovery of the model's original performance in the best case.

#### 3.2 The Proposed Routing Algorithm

Why two algorithms? While Algorithm [2](#page-4-0) describes our method — OEA — in its *full generality*, following exhaustive experiments, we conclude that a *simplified* version of it, described in Algorithm [1,](#page-1-0) recovers its performance while requiring fewer hyperparameters. We hereby describe its most general form and then touch on how to simplify at the end of Section [4.1.](#page-4-0)

Following Section [3.1'](#page-2-0)s argument, OEA aims to minimize the number of activated experts T within a decode batch. Its core constraint is to ensure that the overall response quality, for any given sequence in the batch, does not significantly degrade. This motivates a two-stage approach that works by first establishing a quality baseline for each token *independently*, and then opportunistically recovering lost performance by exploiting the shared computation within the batch.

Notation. Suppose the B tokens in the batch are x<sup>1</sup> . . . xB, and that their sorted expert index scores are ei,j where e<sup>i</sup> is a permutation such that,

$$R(\boldsymbol{x}_i)_{e_{i,1}} \geq R(\boldsymbol{x}_i)_{e_{i,2}} \geq \cdots \geq R(\boldsymbol{x}_i)_{e_{i,N}},$$

where ei,j represents the j th expert choice of i th token. In particular, the default router (as described in Section [2\)](#page-1-0) routes token x<sup>i</sup> to experts in the set Top<sup>k</sup> (R(xi)) =

{ei,1, . . . , ei,k}. Our target is to decide sets S1, . . . , S<sup>B</sup> ⊆ {1, . . . N} where S<sup>i</sup> represents the set of experts that the i th token routes to.

Phase 1: Baseline expert selection. The first phase guarantees a minimum foundation for each token, irrespective of how it is batched. For each token x<sup>i</sup> , we create this baseline by activating the first n<sup>i</sup> experts, thus creating a base set of experts S base <sup>i</sup> = {ei,1, . . . , ei,n<sup>i</sup> }. This is motivated by empirical findings that the top-ranked experts are disproportionately critical to output quality [\(Gupta et al.,](#page-9-0) [2024\)](#page-9-0). The number of base experts is determined by two hyperparameters: (1) a fixed upper bound k<sup>0</sup> ∈ {1 . . . N}; and (2) a cumulative score p ∈ (0, 1], following n<sup>i</sup> = min(k0, ti) where t<sup>i</sup> is the minimum number of experts it takes to reach a cumulative mass of p, such that,

$$\sum_{j=1}^{t_i - 1} R(\mathbf{x}_i)_j$$

Intuitively, t<sup>i</sup> is a function of the normalized scores — it is defined exactly as in [Huang et al.](#page-9-0) [\(2024a\)](#page-9-0). While their work pretrained a model with a regularizer factor to ensure that ni is low on average, no such guarantee is assumed here. And thus, we decide to further cap n<sup>i</sup> by k0. In general, it should never help to set k<sup>0</sup> > k where k is the model's default configuration. Finally, note that by setting p = 1, we essentially have a fixed k<sup>0</sup> and by setting k<sup>0</sup> = N, we practically have the top-p method of [Huang et al.](#page-9-0) [\(2024a\)](#page-9-0),

#### <span id="page-4-0"></span>Algorithm 2 OEA Routing Algorithm

```
1: Input: Token embeddings x_{1...B}, Router scores R(x_i),
      Sorted expert indices e_{i,j} for each token i and rank j.
      Hyperparameters: k_0, p, k^{\text{max}}, \text{maxP}.
 2: {Phase 1: Determine Baseline Experts}
 3: for i = 1 to B do
        Find t_i = \min\{t' \mid \sum_{j=1}^{t'} R(\boldsymbol{x}_i)_{e_{i,j}} \geq p\}
n_i \leftarrow \min\{k_0, t_i\} {Number of baseline experts}
 5:
         S_i^{\text{base}} \leftarrow \{e_{i,1}, \dots, e_{i,n_i}\}
 7: end for
 8:
 9: {Phase 2: Opportunistic Piggybacking}
10: S^{\text{base}} \leftarrow \bigcup_{i=1}^{B} S_i^{\text{base}} {Union of all required experts}
11: for i = 1 to B do
         S_i \leftarrow S_i^{\text{base}} {Initialize final set with baseline}
12:
13:
         for j = n_i + 1 to maxP do
14:
            if |S_i| > k^{\max} then
15:
                break
            end if
16:
            if e_{i,j} \in S^{\text{base}} then
17:
18:
                S_i \leftarrow S_i \cup \{e_{i,j}\}
19:
         end for
20:
21: end for
22: Output: Final expert sets S_1, \ldots, S_B
```

so we generalize and abstract on both methods. We decided to adopt this approach to allow the number of experts to be adaptive to the router scores so that harder instances can demand more experts.

This  $(k_0, p)$ -heuristic can select experts deemed critical to at least one token's predictions, and therefore the set of all essential experts  $S^{\text{base}} = \bigcup_{i=1}^{B} S_i^{\text{base}}$  to activate.

**Phase 2: Opportunistic piggybacking.** Instead of adding any new experts into the mix, this second phase opportunistically recovers some performance by allowing tokens to piggyback onto experts already included in  $S^{\text{base}}$ , thus maintaining the number of activated experts  $T = |S^{\text{base}}|$ . For each token i, we traverse experts in decreasing order of their scores and select those in  $S^{\text{base}}$  provided that (1) the number of selected experts does not exceed  $k^{\text{max}}$  and (2) the expert's rank does not fall below a threshold position maxP. These constraints ensure that the selected experts do not degrade performance, either by over-diversifying expert usage or by selecting experts poorly aligned with the current token.

Weighting after rerouting. Once the routing sets  $S_i$  are chosen, we keep the model's original router scores and renormalize them following Equation (1). Intuitively, this preserves the model's learned preferences among the experts we keep, while ensuring mixture weights still sum to 1.

Other choices like using the weights of top-k are possible, but we leave such optimization to future work.

