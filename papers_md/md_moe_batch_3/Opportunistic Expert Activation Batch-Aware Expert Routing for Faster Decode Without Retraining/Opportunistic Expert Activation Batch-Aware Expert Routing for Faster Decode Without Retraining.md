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

# 4 EXPERIMENTAL SETUP AND EMPIRICAL RESULTS

Hardware and model. Unless otherwise specified, all our experiments were performed on one NVIDIA H100 80GB GPU each, while using Qwen3-30B-A3B (Yang et al., 2025a) under bfloat16 precision. This model has 48 layers, each with N=128 experts of which k=8 are activated per token, 32 query heads and 4 KV heads, an embedding dimension of 2048 and per-expert hidden dimension of 768; each expert uses SwiGLU-based (Shazeer, 2020) feedforward network which entails 3 matrix multiplications of sizes  $2048 \times 768$ .

#### 4.1 Cross-Entropy Experiments

**Motivation.** We use cross-entropy on a pretraining dataset as a granular proxy for the compound effect of our router intervention. We do this because:

- It is much cheaper to measure cross-entropy than downstream performance thanks to its parallel computation. Thus, we can perform a large hyperparameter sweep to determine the *optimal* setting of OEA. Based on these findings, we suggest a simplified version of OEA (Algorithm 1) that we then evaluate on standard benchmarks in Section 4.2.
- Unlike benchmarks, cross-entropy provides a more statistically reliable estimate of modeling quality since it provides dense, per-token signal rather than sparse, task-level evaluation.

**Dataset.** In the first round of experiments, we evaluated cross-entropy loss on a subset of the FineWeb-Edu dataset (Penedo et al., 2024), which we selected as a high-quality and diverse proxy since Qwen3's pretraining data is not public. We randomly selected 2048 sequences, each containing at least 8192 tokens to ensure a fixed batch size across positions since OEA is sensitive by construction to batch size. In particular, a batch size of 1 deems the piggy-backing redundant.

**Methodology.** We simulate L decoding steps (sequence length) but execute them efficiently in parallel. At step t, we form a batch from the t-th token of each sequence and run routing *only within that step*: both the Phase 1 pruning and Phase 2 piggybacking are computed using tokens that share the same position t. No information (experts or scores) is shared across different positions, so piggybacking never crosses decode steps. We then process all steps in parallel

by grouping expert workloads post-routing, which yields the same routing decisions as true sequential decode while enabling a fast, batched implementation for measurement. Throughout this computation, we track the *average number of activated experts* across positions and layers, as well as *average cross-entropy*.

**Experiments.** The parallel speedup allows for a comprehensive sweep of hyperparameters:  $k_0 \in \{4,5,6,7,8\}$ ,  $k^{\max} \in \{7,8,9,10,11\}$ ,  $p \in \{0.4,0.5,0.6,0.7,0.8,0.9,1\}$  and  $\max P \in \{8,16,32,128\}$ . On top of these, we also considered stopping after Phase 1 (for the same ranges of  $k_0,p$ ) and forgoing the piggybacking — we refer to this as *Phase I* or *pruned* routing. For each such routing algorithm, we swept  $B \in \{8,16,32,64\}$ . We used 128 sequences and the full length of 8192 for B=8 and cut sequence length by half for every batch doubling to keep the activation memory fixed and able to fit in the memory of a GPU. We also doubled the number of sequences for every batch doubling to keep the overall number of tokens fixed at 1M.

Equipped with these runs, we can investigate the effects of our design choices. There are four degrees of freedom corresponding to the four hyperparameters:  $k_0$  and p determine the pruning extent while  $k^{\max}$  and  $\max$ P control whether adding an extra expert starts hurting.<sup>2</sup>

**Ablations.** For each experiment, we define its performance as a trade-off between cross-entropy and the average number of activated experts — the objective is to minimize both. To assess the effect of each hyperparameter value on performance, we plot the Pareto frontier of all experiments conducted with that value. As our purpose is to limit cross-entropy degradation, we plot the increase in cross-entropy with respect to a vanilla MoE and track across runs. Furthermore, as a lot of runs present negligible differences in cross-entropy and average number of activated experts, we round the increase in cross-entropy to the closest multiple of 0.005 and the average number of activated experts to the closest multiple of 0.1 to avoid crowding the plots.

**Piggybacking gains.** Since our algorithm's core addition over a form of adaptive pruning is the piggybacking phase, the salient question is whether Phase 2 truly adds value: we answer this in the affirmative, as shown in Figure 2.

We find three consistent patterns regarding hyperparameter choice. The ablation plots corresponding to them are available in the Appendix A.

1. Using p < 1 does not help. Setting p = 1 (equivalent to using top- $k_0$  in Phase 1) performs on par with p < 1

![](_page_5_Figure_9.jpeg)

Figure 2. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The two types of dots correspond to the Pareto frontiers of pruned and OEA experiments at batch size B=16. OEA consistently performs better.

- 1 (Figure 9). This holds across both OEA and the partial "pruned" (Phase 1 only) case. We considered employing a top-p scheme to allow the choice of base quality to be a function of the router scores, but there is no significant marginal gain from this adaptivity.
- 2.  $k^{\max} = k$  works best. Interestingly, we find that bounding the number of experts per token at exactly k works better than both smaller and larger choices (Figure 7). Naturally, one expects more experts to help but interestingly, using  $k^{\max} = 9$  experts does not really improve above  $k^{\max} = 8$ ; in fact, further increasing to  $k^{\max} = 10, 11$  actually results in degradation.
- 3. Setting maxP < N does not help. Our ablation over maxP (Figure 6) shows that refraining from piggybacking onto an activated expert due to it being too far down a token's preference list is detrimental for the optimal values of  $k^{max}$ . It is worth noting that maxP could only make a difference when we do not have  $k^{max}$  activated experts in the top-maxP preferences of a token, which becomes less likely with the increase in B (and thus  $|S^{base}|$ ). Finally, one important consequence of maxP = 8 strictly hurting is proving that using out-of-policy experts confers a strict advantage. This is contrary to the thesis that those experts are not trained to be useful for this one token.

**Simplifying OEA** Putting these together, we conclude that we can drop the usage of top-p in Phase 1 and that of maxP in Phase 2, as well as set  $k^{\max}$  to be k. This leaves us with a simplified version of the OEA routing which is presented in Algorithm 1. Figure 3 shows that our hyperparameter findings are jointly consistent, suggesting the

<sup>&</sup>lt;sup>2</sup>Note that setting p = 1 is equivalent to not using it at all and so is the case for maxP = 128.

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Figure 3. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The two types of dots correspond to the Pareto frontiers of simplified OEA and the rest of experiments at batch size B=16. Simplified OEA performs comparably to the best hyperparameter choices.

simplified algorithm to be as performant as its general counterpart. A major benefit is therefore the reduced cost of hyperparameter sweeps prior to deploying the model:  $k_0$  itself controls both the guaranteed baseline quality and the drop in activated experts.

#### 4.2 Downstream Evaluations

Qwen3-235B-A22B model. We benchmark our approach on both the Qwen3-30B and Qwen3-235B-A22B models. The latter doubles the number of layers (96), embedding dimension (4096) and expert hidden dimension (1536) while the same attention head configuration top-8/128 routing. All experiments are performed under tensor parallelism across 8 H100 GPUs within a single HGX H100 node interconnected via NVSwitch (18 NVLink per GPU pair).

**Setup.** We conducted downstream evaluation on four benchmarks: AIME 24, MATH 500 (Hendrycks et al., 2021), GPQA (Rein et al., 2024) and LiveCodeBench (Jain et al., 2024). All accuracy reported is an average over four runs of each for Qwen3-30B (and three runs for Qwen3-235B), with the exception of AIME 24 which we evaluate four times more runs (since it only has 30 data points and, thus, higher variance). For all runs, we use a temperature of 0.6, top-p sampling with p = 0.95 and generate up to 32768 tokens. We integrate our router into the SGLang framework (Zheng et al., 2024). For each run, we track the batch size, number of activated experts and the latency for every layer and decode step. Note that, throughout the serving process, batch size can and does vary as requests are finished, enqueued or retracted. Since our routing algorithm only benefits latency in moderate batch size regimes, we use it only during decode, not prefill.

**Experiments.** Informed by our findings from Section 4.1, we only used the recommended settings and thus only tried the simplified algorithm (1) parameterized only by  $k_0$ . We tested all values of  $k_0 \in \{3,4,5,6,7\}$  on Qwen3-30B, and all except  $k_0 = 7$  on Qwen3-235B due to computational constraints. We also evaluated post-Phase 1 (pruned) routing (as in Section 4.1) for the same set of  $k_0$  values. As batch size cannot be fixed in SGLang, we use its --max-running-requests option to set a maximum batch size. We could not do this for a batch size of 32 due to the large KV cache size, so we report OEA numbers for a batch size bounded at 16.

**Piggybacking gains.** Tables 1 and 2 report the average performances of the pruned (Phase 1) approach, of the simplified OEA, and of the base model ( $k_0$ =8) for the 30B and 235B model, respectively. We find that:

- Qwen3-30B: Across all benchmarks, except GPQA, using top-5 rather than top-8 experts does not make a statistically significant difference<sup>3</sup>. However, further lowering it to 4 and 3 starts showing substantial degradation. OEA manages to recover lost performance even for  $k_0 = 3$  while fundamentally not incurring any extra cost over its pruned counterpart; this is the marginal gain of piggybacking and our main contribution.
- Qwen3-235B: In the basic pruning setup, performance drops sharply at k<sub>0</sub> = 5 falling below the base model by 15% on LiveCodeBench, 8% on GPQA. In contrast, OEA at k<sub>0</sub> = 5 maintains performance on all benchmarks except for LiveCodeBench where its accuracy declines slightly by 2%.

Relationship between latency and the number of activated experts. The central hypothesis of this work (introduced in Section 3.1) is that for moderate batch sizes, the MoE latency scales linearly with the number of activated experts. To confirm this, we tracked all the (T, latency) pairs obtained at all decode steps and all layers. Figure 1 shows the average latency for a fixed number of activated experts across the whole GPQA run of the vanilla Qwen3-30B model (across decode steps and layers); the standard errors are all less than  $2 \cdot 10^{-4}$  indicating that latency is well predicted by these means. Since the MoE module itself was left unchanged, this trend is independent of the routing strategy. Strikingly, the linear trend fits regression at  $R^2 > 0.99$ , thus affirming the thesis of this work: latency is linearly controlled by the number of activated experts.

**Latency gains via reducing active experts.** Building on the above, we examine how OEA's reduced expert activation

 $<sup>^3</sup>$ A result  $\mu \pm se$  is considered standard-error adjusted worse than  $\mu_{\text{vanilla}} \pm se_{\text{vanilla}}$  when  $\mu + se < \mu_{\text{vanilla}} - se_{\text{vanilla}}$ 

<span id="page-7-0"></span>translates to practical latency reductions. For the Qwen3- 30B model, Table [4](#page-8-0) reports the average number of active experts (aggregated over layers and decode steps) as a function of k0, with k<sup>0</sup> = 3 halving the number of activated experts. As shown in Table [3,](#page-8-0) this corresponds to latency reductions of 39% for k<sup>0</sup> = 3 and 23% for k<sup>0</sup> = 5. For Qwen3-235B, the results in Table [5](#page-9-0) show a 15% speedup at k<sup>0</sup> = 5; we attribute this smaller relative reduction to the additional overhead of tensor parallel's all-reduce.

## 5 RELATED WORK

The challenge of optimizing MoE inference latency is an active area of research [\(Liu et al.,](#page-10-0) [2024c\)](#page-10-0). OEA builds upon lessons from works on alternative routing mechanisms, architectural innovations, and other dynamic inference-time strategies, while distinguishing in several ways.

## 5.1 Foundational MoE Routing and Its System-Level Challenges

The modern MoE paradigm in Transformers was established by Shazeer et al. with the Sparsely-Gated MoE layer [\(2017\)](#page-10-0), which employs a trainable gating network to route each token to a top-k subset of experts. Their goal was to decouple the model parameters from the computational cost of training to enable scaling of larger LLMs.

However, this approach introduces significant systems-level challenges. The most notable is load imbalance (where the router ends up favoring a subset of "popular" experts) leads to router collapse, leaving other experts and their associated parameters and hardware underutilized. To address this matter, load-balancing losses are employed during training [\(Shazeer et al.,](#page-10-0) [2017;](#page-10-0) [Fedus et al.,](#page-9-0) [2022;](#page-9-0) [Liu et al.,](#page-10-0) [2024a\)](#page-10-0).

More central to our work are the issues that arise in batched inference for large sparsity. Serving systems [\(Kwon et al.,](#page-10-0) [2023;](#page-10-0) [Zheng et al.,](#page-10-0) [2024\)](#page-10-0) rely on batching to achieve high throughput, but this forces the activation of the union of all experts selected by any token in the batch, quickly negating MoE's sparsity and deeming MoE layers to be memorybound for moderate batch sizes.

#### 5.2 Alternative Routing and Architectural Paradigms

To address these fundamental issues, paradigms going beyond token-centric top-k routing have been explored.

Expert choice routing. [Zhou et al.](#page-10-0) [\(2022\)](#page-10-0) inverted the selection logic, allowing each expert to select its top-k preferred tokens from the batch. This is an inherently batchaware mechanism that guarantees perfect load balancing by design, eliminating the need for auxiliary losses. While it enables a variable number of experts per token, its purpose is optimizing throughput via load balancing rather than

minimizing the number of active experts to reduce latency.

Architectural solutions (shared experts). Models like DeepSeek-V3 [\(Liu et al.,](#page-10-0) [2024a\)](#page-10-0) and Kimi K2 [\(Kimi Team](#page-9-0) [et al.,](#page-9-0) [2025\)](#page-9-0) incorporate a hybrid architecture with both "routed" and "shared" experts. The shared experts process every token in the batch, providing a form of guaranteed computational reused core. This allows for system cooptimization, such as hiding communication latency behind the shared expert's computation. Such approaches represent a static design solution to shared computation, contrasting with our dynamic, runtime approach that requires no architectural modifications.

#### 5.3 Dynamic Inference-Time Optimizations

OEA best fits under the paradigm of dynamic, inferencetime optimizations that modify MoE behavior without retraining. Such approaches are crucially different from static pruning methods that permanently remove experts they expect to not be crucial to performance. While effective for compression, the behavior of the dropped experts cannot be recovered if a token depended on them. Such issues could in principle be mitigated by making exclusion decisions adaptive to the batch, although then the model size cannot be reduced.

Token-centric dynamic skipping. One category of dynamic methods operates on a per-token basis. Lu et al. [\(2024\)](#page-10-0) proposed dynamically skipping a secondary expert if its router score is significantly lower than the primary one, saving computation on a per-token basis. Similarly, the Online Dynamic Pruning (ODP) technique identifies less important tokens and assigns them fewer experts [\(Huang](#page-9-0) [et al.,](#page-9-0) [2024b\)](#page-9-0). These methods are not explicitly batch-aware and thus miss opportunities for shared computation.

Expert offloading and prefetching. In environments that are memory constrained environments, model weights are stored on CPU and offloaded on a need basis. Systems like Pre-gated MoE [\(Hwang et al.,](#page-9-0) [2024\)](#page-9-0) employ predictive prefetching, where information from the current layer is used to anticipate and pre-load experts for the next layer while the current layer's computation is underway. These systems optimize for memory transfer costs across time (inter-batch), whereas our method optimizes for computational reuse within a single batch.

Comparison to related work. The most closely related work is Lynx, a framework that also uses a batch-aware routing to reduce active experts [\(Gupta et al.,](#page-9-0) [2024\)](#page-9-0). They employ a fundamentally different, subtractive approach: first the union of experts that would normally be activated is computed; then, the least popular among these experts are

<span id="page-8-0"></span>Table 3. Average MoE layer latency (in microseconds) when using simplified OEA (top-k<sup>0</sup> + piggybacking) on Qwen3-30B-A3B.

|                    | k0<br>= 3 | k0<br>= 4 | k0<br>= 5 | k0<br>= 6 | k0<br>= 7 | VANILLA |
|--------------------|-----------|-----------|-----------|-----------|-----------|---------|
| AIME24             | 97.9      | 110.5     | 122.0     | 138.4     | 147.4     | 158.0   |
| GPQA               | 111.0     | 125.4     | 143.2     | 159.0     | 172.5     | 184.1   |
| LIVECODEBENCH      | 102.7     | 117.1     | 132.2     | 146.5     | 157.3     | 170.8   |
| MATH 500           | 115.6     | 130.4     | 146.7     | 161.5     | 174.7     | 189.9   |
| AVERAGE            | 106.8     | 120.9     | 136.0     | 151.3     | 163.0     | 175.7   |
| NORMALIZED AVERAGE | 0.61      | 0.69      | 0.77      | 0.86      | 0.93      | 1.00    |

Table 4. Average number of activated experts when using simplified OEA (top-k<sup>0</sup> + piggybacking) on Qwen3-30B-A3B.

|                    | k0<br>= 3 | k0<br>= 4 | k0<br>= 5 | k0<br>= 6 | k0<br>= 7 | VANILLA |
|--------------------|-----------|-----------|-----------|-----------|-----------|---------|
| AIME24             | 22.2      | 26.5      | 30.5      | 36.0      | 39.2      | 43.0    |
| GPQA               | 26.5      | 31.4      | 37.5      | 42.9      | 47.6      | 51.6    |
| LIVECODEBENCH      | 23.8      | 28.7      | 33.9      | 38.7      | 42.5      | 47.2    |
| MATH 500           | 27.9      | 33.0      | 38.6      | 43.7      | 48.3      | 53.5    |
| AVERAGE            | 25.1      | 29.9      | 35.1      | 40.3      | 44.4      | 48.8    |
| NORMALIZED AVERAGE | 0.51      | 0.61      | 0.72      | 0.83      | 0.91      | 1.00    |

dropped. Crucially, this risks removing an expert that, while unpopular across the batch, is critical to a single token's accuracy. In contrast, OEA is an additive and opportunistic framework. It first guarantees the critical computation for every token, ensuring a baseline quality. It then opportunistically augments this baseline by "piggybacking" on experts that are already active, recovering model capacity at *zero* additional latency cost. This additive approach provides a more robust trade-off between performance and accuracy, particularly for modern models with a high number of activated experts (k > 2), where the hierarchical importance of experts observed by Lynx may be less pronounced. Finally, note that our piggybacking phase can be added to any routing-changing approach (static or dynamic) — including Lynx — to gain *free* quality recovery.

## 6 DISCUSSION

Effect of batch distribution. OEA's effectiveness depends strongly on the tokens' distribution within a batch. When tokens come from similar distributions, they tend to overlapping experts resulting in a smaller S base which limits piggybacking's gains. This is the regime that our benchmarks represent, making the reported performance a conservative estimate. In contrast, the cross-entropy experiments in Section [4.1](#page-4-0) correspond to a more diverse token distribution which enlarges S base and allows piggybacking to recover more of the base model's performance.

A note on padding. During our experiments, under default configuration of SGLang, we noticed the average number of tokens (and average latency) in batches of size 7

exceed that of batches of size 8, which was counter-intuitive. This is because SGLang captures CUDA Graphs for a set of batch-sizes and when it needs to process a certain batch size B, it looks up the smallest B′ > B that has been captured and pads the batch up to size B′ . While for classic feedforward networks and attention, the specific contents of the batch do not influence the kernel's runtime, this is not the case for MoEs, especially under the memory-bound batch regime we operate. What happened was that the padding token activated on average more experts "out-of-distribution" (that were not activated by real tokens already) than an 8 th realistic one would. Thus this seemingly inoffensive padding ended up costing more than processing an extra real token, when it should ideally not add any more experts. In our experiments we simply fixed this by capturing CUDA Graphs up to size 16 (thus ensuring no padding), but we do make the general note that there is value in adding a padding mask and using it to zero out the padding tokens' expert choices.

## 7 CONCLUSION AND FUTURE DIRECTIONS

In this work, we introduce OEA, a new expert routing algorithm targeting the problem of memory-bound MoE decoding under moderate batch sizes. OEA achieves this by reducing the number of activated experts per decode batch. To do this, it first activates a few top experts per token deemed crucial to its performance and then opportunistically piggybacks some more that were crucial to other tokens in the batch. This approach results in MoE latency speed-ups of 39% and 15% for the Qwen3-30B and Qwen3-235B models, respectively, without statistically significant degradation in benchmark accuracy.

<span id="page-9-0"></span>

| Table 5. Average MoE layer latency (in microseconds), including all-reduce when using simplified OEA (top-k0<br>+ piggybacking) on |
|------------------------------------------------------------------------------------------------------------------------------------|
| Qwen3-235B-A22B.                                                                                                                   |
|                                                                                                                                    |
|                                                                                                                                    |

|                    | k0<br>= 3 | k0<br>= 4 | k0<br>= 5 | k0<br>= 6 | VANILLA |
|--------------------|-----------|-----------|-----------|-----------|---------|
| AIME24             | 86.4      | 92.6      | 98.8      | 105.7     | 118.4   |
| GPQA               | 86.7      | 93.8      | 99.5      | 104.7     | 116.0   |
| LIVECODEBENCH      | 88.2      | 95.3      | 102.8     | 108.6     | 121.1   |
| MATH 500           | 89.6      | 97.4      | 104.4     | 108.7     | 122.2   |
| AVERAGE            | 87.7      | 94.8      | 101.4     | 106.9     | 119.4   |
| NORMALIZED AVERAGE | 0.73      | 0.79      | 0.85      | 0.90      | 1.00    |

Batch adaptivity. The cross-entropy's analysis (Figure [5d](#page-13-0)) shows negligible degradation at B = 64. Larger batches naturally increase S base, enabling piggybacking to approximate the original routing more closely. This observation suggests an approach where the routing scheme is a function of the batch-size (e.g. using a bigger (safer) k<sup>0</sup> at a lower batch size). We leave determining such batch-sizedependent k0-choice as an open problem.

Extension to expert parallelism. Although OEA assumes experts are not executed in parallel, adapting to the expertparallel setting is possible — the latency is then driven by the *maximum* number of activated experts per machine. Hence, one immediate equivalent to our method is to do piggybacking independently on each machine while potentially increasing k<sup>0</sup> in the machines that activate fewer experts (have a smaller S base).

Layer heterogeneity. Empirically, we observe that the average number of active experts varies significantly across layers. This, coupled with previous empirical findings (Gupta et al., 2024; [Yang et al.,](#page-10-0) [2025b\)](#page-10-0), suggests that adapting our method's hyperparameters independently for each layer could further improve its performance.

Routing robustness and model co-designing. At its core, our method relies on pretrained models' routing being robust to slight changes. Therefore, further understanding or quantifying the routing robustness limits, or even having models' pretraining be co-designed with this purpose in mind could further increase router flexibility and thus improve OEA's scope. This includes tackling the challenge mentioned in Section [3.2](#page-3-0) of how to adapt expert weights when expert routing is modified.

## REFERENCES

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.

Gupta, V., Sinha, K., Gavrilovska, A., and Iyer, A. P. Lynx: Enabling efficient moe inference through dynamic batchaware expert selection. *arXiv preprint arXiv:2411.08982*, 2024.

Hejazi, B. Introducing grouped gemm apis in cublas and more performance updates. [https://developer.](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) [nvidia.com/blog/introducing-grouped-g](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) [emm-apis-in-cublas-and-more-perform](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) [ance-updates/](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/), June 2024. NVIDIA Developer Blog, June 12, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.

Huang, Q., An, Z., Zhuang, N., Tao, M., Zhang, C., Jin, Y., Xu, K., Chen, L., Huang, S., and Feng, Y. Harder tasks need more experts: Dynamic routing in moe models. *arXiv preprint arXiv:2403.07652*, 2024a.

Huang, W., Liao, Y., Liu, J., He, R., Tan, H., Zhang, S., Li, H., Liu, S., and Qi, X. Mixture compressor for mixture-of-experts llms gains more. *arXiv preprint arXiv:2410.06270*, 2024b.

Hwang, R., Wei, J., Cao, S., Hwang, C., Tang, X., Cao, T., and Yang, M. Pre-gated moe: An algorithm-system codesign for fast and scalable mixture-of-expert inference. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, pp. 1018–1031. IEEE, 2024.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. *arXiv preprint arXiv:2403.07974*, 2024.

Kimi Team, Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi K2: Open agentic intelligence. *arXiv preprint arXiv:2507.20534*, 2025.

- <span id="page-10-0"></span>Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th symposium on operating systems principles*, pp. 611–626, 2023.
- Li, Z., Li, Z., and Zhou, T. R2-t2: Re-routing in testtime for multimodal mixture-of-experts. *arXiv preprint arXiv:2502.20395*, 2025.
- Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseekv3 technical report. *arXiv preprint arXiv:2412.19437*, 2024a.
- Liu, E., Zhu, J., Lin, Z., Ning, X., Blaschko, M. B., Yan, S., Dai, G., Yang, H., and Wang, Y. Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs. *arXiv preprint arXiv:2407.00945*, 2024b.
- Liu, J., Tang, P., Wang, W., Ren, Y., Hou, X., Heng, P.-A., Guo, M., and Li, C. A survey on inference optimization techniques for mixture of experts models. *arXiv preprint arXiv:2412.14219*, 2024c.
- Lu, X., Liu, Q., Xu, Y., Zhou, A., Huang, S., Zhang, B., Yan, J., and Li, H. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. *arXiv preprint arXiv:2402.14800*, 2024.
- NVIDIA. H100 tensor core gpu architecture white paper. Technical report, NVIDIA Corporation, 2022. URL [https://resources.nvidia.com/en-us-h](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) [opper-architecture/nvidia-h100-tenso](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) [r-c](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c).
- Penedo, G., Kydl´ıcek, H., Lozhkov, A., Mitchell, M., Raffel, ˇ C. A., Von Werra, L., Wolf, T., et al. The fineweb datasets: Decanting the web for the finest text data at scale. *Advances in Neural Information Processing Systems*, 37: 30811–30849, 2024.
- Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., Rasley, J., and He, Y. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In *International conference on machine learning*, pp. 18332–18346. PMLR, 2022.
- Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In *First Conference on Language Modeling*, 2024.
- Shazeer, N. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.
- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025a.
- Yang, H., Shi, L., Li, Q., Li, Z., Wang, P., Du, B., Shen, M., and Zhao, H. Faster moe llm inference for extremely large models. *arXiv preprint arXiv:2505.03531*, 2025b.
- Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al. Sglang: Efficient execution of structured language model programs. *Advances in neural information processing systems*, 37:62557–62583, 2024.
- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A. M., Le, Q. V., Laudon, J., et al. Mixture-ofexperts with expert choice routing. *Advances in Neural Information Processing Systems*, 35:7103–7114, 2022.

## A MORE BENCHMARK RESULTS

Tables [6](#page-11-0) and [8](#page-12-0) show the benchmark accuracies of *simplified OEA* together *with the standard errors* across the 4 Qwen3- 30B, respectively 3 Qwen3-235B runs.

Tables [7](#page-11-0) and [9](#page-12-0) show the benchmark accuracies for the *pruned* routers (top-k0) *with the standard errors* across the 4 Qwen3-30B, respectively, 3 Qwen3-235B runs.

Furthermore, Table [10](#page-12-0) shows the average number of activated experts while using simplified OEA routing on Qwen3- 235B.

For the Qwen3-235B model, we report, in Figure [4,](#page-11-0) the average latency (across decode steps and layers) for a fixed number of activated experts across a complete run of GPQA.

## B CROSS ENTROPY ABLATION PLOTS

#### B.1 Pruned vs OEA ablation

We performed ablations for all batch sizes showing the Pareto frontier of OEA-based experiments in contrast to simple pruning (based on phase 1). The results are shown in Figure [5](#page-13-0) and confirm the piggybacking's (phase 2) gains.

## B.2 Ablation over maxP

Pareto frontiers corresponding to each of the values of maxP are displayed in Figure [6](#page-14-0) for all batch sizes B and support the fact that maxP = 128 is optimal while maxP = 8 is strictly worse.

<span id="page-11-0"></span>Table 6. Ablation across  $k_0$ : Benchmark accuracies, standard error included, of **simplified OEA** routing (top- $k_0$ +piggybacking) on Qwen3-30B-A3B. Vanilla represents the default model.

|               | $k_0 = 3$       | $k_0 = 4$       | $k_0 = 5$       | $k_0 = 6$       | $k_0 = 7$       | VANILLA         |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| AIME24        | $80.0 \pm 0.59$ | $81.9 \pm 0.71$ | $81.5 \pm 0.62$ | $80.8 \pm 0.90$ | $78.5 \pm 1.33$ | $80.4 \pm 0.99$ |
| GPQA          | $58.6 \pm 1.22$ | $59.3 \pm 0.15$ | $61.1 \pm 1.69$ | $62.2 \pm 0.56$ | $60.6 \pm 0.36$ | $60.2 \pm 0.83$ |
| LIVECODEBENCH | $61.2 \pm 0.94$ | $62.7 \pm 0.53$ | $62.0 \pm 0.33$ | $63.1 \pm 0.57$ | $62.5 \pm 0.47$ | $62.1 \pm 0.94$ |
| MATH_500      | $93.5 \pm 0.29$ | $93.1 \pm 0.29$ | $93.3 \pm 0.27$ | $93.1 \pm 0.30$ | $93.2 \pm 0.08$ | $92.8 \pm 0.22$ |

Table 7. Ablation across  $k_0$ : Benchmark accuracies, *standard error included*, of **pruned** routing (top- $k_0$ ) on Qwen3-30B-A3B. Vanilla represents the default model.

|                                 | $k_0 = 3$                                             | $k_0 = 4$                                             | $k_0 = 5$                                             | $k_0 = 6$ | $k_0 = 7$                                             | VANILLA                                               |
|---------------------------------|-------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|-----------|-------------------------------------------------------|-------------------------------------------------------|
| AIME24<br>GPQA<br>LIVECODEBENCH | $51.2 \pm 1.42$<br>$45.7 \pm 0.84$<br>$37.4 \pm 0.83$ | $75.8 \pm 0.83$<br>$54.3 \pm 0.53$<br>$58.2 \pm 1.41$ | $80.6 \pm 0.86$<br>$56.2 \pm 0.43$<br>$63.2 \pm 0.95$ |           | $82.5 \pm 0.83$<br>$59.7 \pm 2.01$<br>$63.0 \pm 0.77$ | $80.4 \pm 0.99$<br>$60.2 \pm 0.83$<br>$62.1 \pm 0.94$ |
| MATH_500                        | $91.1 \pm 0.37$                                       |                                                       | $92.6 \pm 0.13$                                       |           | $93.3 \pm 0.26$                                       | $92.8 \pm 0.22$                                       |

### **B.3** Ablation over $k^{\text{max}}$

Ablations over the value of  $k^{\rm max}$  are depicted in Figure 7, with a different Pareto frontier computed for each  $k^{\rm max}$ . Note that  $k^{\rm max}=8$  and  $k^{\rm max}=9$  perform comparably with others being strictly worse.

#### **B.4** Simplified OEA contrasted with other settings

Figure 8 contrasts the Pareto frontier of simplified OEA (Algorithm 1) with all the other settings (pruned and general OEA together). It shows no meaningful trade-off losses.

#### **B.5** Ablation over p

We grouped experiments by whether p=1 (thus having a static  $k_0$  core experts per token) or p<1, as well as whether they use a pruned (phase-1) routing or an OEA-based routing. Pareto frontiers for these 4 groups are depicted in Figure 9. Note that within both pruned and OEA, it consistently holds that p=1 approximately recovers performance of p<1.

![](_page_11_Figure_11.jpeg)

Figure 4. Mean MoE latency as a function of the number of activated experts within a decode batch. The average is computed over all layers and decode steps across a GPQA evaluation of the vanilla Qwen3-235B-A22B model (under a tensor parallel degree of 8).

<span id="page-12-0"></span>Table 8. Ablation across k0: Benchmark accuracies, *standard error included*, of simplified OEA routing (top-k0+piggybacking) on Qwen3-235B-A22B. Vanilla represents the default model.

|                       | k0<br>= 3                  | k0<br>= 4                  | k0<br>= 5                  | k0<br>= 6                  | VANILLA                    |
|-----------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| AIME24                | 81.4 ± 1.21                | 82.5 ± 0.48                | 83.6 ± 1.39                | 83.6 ± 1.11                | 85.0 ± 0.96                |
| GPQA<br>LIVECODEBENCH | 66.3 ± 0.67<br>63.4 ± 0.75 | 67.7 ± 1.05<br>67.1 ± 0.73 | 67.5 ± 0.67<br>66.1 ± 0.63 | 67.5 ± 0.45<br>66.1 ± 0.12 | 68.4 ± 0.34<br>68.5 ± 0.21 |
| MATH 500              | 94.4 ± 0.46                | 94.8 ± 0.31                | 94.7 ± 0.13                | 94.3 ± 0.18                | 94.7 ± 0.18                |

Table 9. Ablation across k0: Benchmark accuracies, *standard error included*, of pruned routing (top-k0) on Qwen3-235B-A22B. Vanilla represents the default model.

|                                 | k0<br>= 3                                | k0<br>= 4                                 | k0<br>= 5                                 | k0<br>= 6                                 | VANILLA                                   |
|---------------------------------|------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| AIME24<br>GPQA<br>LIVECODEBENCH | 17.5 ± 0.96<br>43.8 ± 0.61<br>5.7 ± 0.72 | 69.4 ± 0.73<br>56.4 ± 0.73<br>27.4 ± 1.19 | 81.9 ± 1.39<br>60.6 ± 1.54<br>53.5 ± 0.73 | 82.8 ± 1.21<br>64.1 ± 1.01<br>60.8 ± 0.24 | 85.0 ± 0.96<br>68.4 ± 0.34<br>68.5 ± 0.21 |
| MATH 500                        | 80.9 ± 0.18                              | 93.3 ± 0.27                               | 94.5 ± 0.44                               | 94.5 ± 0.18                               | 94.7 ± 0.18                               |

Table 10. Average number of activated experts when using simplified OEA (top-k<sup>0</sup> + piggybacking) on Qwen3-235B-A22B.

|                    | k0<br>= 3 | k0<br>= 4 | k0<br>= 5 | k0<br>= 6 | VANILLA |
|--------------------|-----------|-----------|-----------|-----------|---------|
| AIME24             | 27.5      | 32.9      | 38.4      | 43.9      | 53.2    |
| GPQA               | 27.4      | 33.3      | 38.6      | 43.1      | 51.6    |
| LIVECODEBENCH      | 28.8      | 34.9      | 41.2      | 45.8      | 55.1    |
| MATH 500           | 29.6      | 36.2      | 42.4      | 46.2      | 56.0    |
| AVERAGE            | 28.3      | 34.4      | 40.2      | 44.7      | 54.0    |
| NORMALIZED AVERAGE | 0.53      | 0.64      | 0.74      | 0.83      | 1.00    |

<span id="page-13-0"></span>![](_page_13_Figure_1.jpeg)

Figure 5. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The two types of dots correspond to the Pareto frontiers of pruned and OEA experiments at all batch-sizes B. OEA consistently performs better.

<span id="page-14-0"></span>![](_page_14_Figure_1.jpeg)

Figure 6. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The four types of dots correspond to the Pareto frontiers of experiments using different values of maxP. maxP consistently performs best, whereas maxP = 8 is strictly worse than it.

<span id="page-15-0"></span>![](_page_15_Figure_1.jpeg)

Figure 7. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The five types of dots correspond to the Pareto frontiers experiments using different values of  $k^{\max}$ .  $k^{\max} \in \{8,9\}$  perform best while all others perform strictly worse.

<span id="page-16-0"></span>![](_page_16_Figure_1.jpeg)

Figure 8. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). The two types of dots correspond to the Pareto frontiers of simplified OEA and the rest of experiments at all batch-sizes B. Simplified OEA performs comparably to the best hyperparameter choices.

<span id="page-17-0"></span>![](_page_17_Figure_1.jpeg)

Figure 9. The y-axis shows the cross-entropy delta relative to the baseline (lower left is better). We split dots as per the legend (by whether p=1 and whether they use pruned or OEA-based routings) and report the Pareto frontiers of each group for all batch sizes B. Always using p=1 does not compromise substantial performance within either group.