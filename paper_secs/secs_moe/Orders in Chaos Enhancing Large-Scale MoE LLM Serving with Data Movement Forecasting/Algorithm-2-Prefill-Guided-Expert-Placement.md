# Algorithm 2: Prefill-Guided Expert Placement

```
Input: Prefill traces \mathcal{D}, GPU count G, extra slots per GPU R
    Output: Per-layer expert-to-GPU assignment \{S_q\}_{q}^{q}
 1 Notation: E: total experts; f_{l,e}: freq of expert e at layer l; L_g: load
     of GPU g; r_g: remaining slots on GPU g; \delta_{e,g}: \max_{q'} L_{q'}
     change after copying expert e to GPU g
2 Function remap_based_placement (\mathcal{D}, G):
3
          for each layer l do
                Compute f_{l,e} from \mathcal{D}; sort exps by decreasing Cost(f_{l,e});
                L_g \leftarrow 0 \text{ for all } g;
 5
                for each expert e in sorted order do
                      Assign e to least-loaded GPU g^* s.t. |S_{q^*}| < E/G;
                        L_{g^*} += \operatorname{Cost}(f_{l,e});
          return \{S_q\} for each layer;
9 Function dup_based_placement (\mathcal{D}, G, R):
          for each layer l do
10
11
                Compute f_{l,e} from \mathcal{D}; generate default placement \mathcal{S}_q;
                r_g \leftarrow R \text{ for all } g;
12
                L_g \leftarrow \sum_{e \in S_g} \operatorname{Cost}(f_{l,e}) for all g; for i \leftarrow 1 to R \cdot G do
13
14
                      (e^*,g^*) \leftarrow \mathop{\arg\min}_{e,g: \ r_g > 0, \ g \not\in \text{hosts}(e)} \delta_{e,g};
15
                      Assign e^* to \mathcal{S}_{g^*}; r_{g^*} \leftarrow r_{g^*} - 1;
16
                      update affected L_g;
17
          return \{S_g\} for each layer;
```

As shown in Figure 16, we design two placement algorithms (details in algorithm 2). The *Remap-based* algorithm keeps the number of experts per GPU unchanged and reassigns experts across GPUs for a more balanced workload: it sorts experts by decreasing roofline cost and greedily assigns each to the least-loaded GPU, subject to a uniform capacity of E/G experts per GPU. The *Duplication-based* algorithm reserves extra expert slots on each GPU and uses prefill traces to duplicate hot experts, thereby avoiding congestion: starting from the default contiguous layout (e.g., experts 0–15 on GPU,0, 16–31 on GPU,1, etc.), it greedily adds up to R extra replicas per GPU, selecting at each step the (expert, GPU) pair that maximally reduces the bottleneck load  $\max_g load_g$ ; tokens of a replicated expert are evenly split among all its copies. Both algorithms use a roofline-based cost model to estimate per-GPU load.

#### B. Methodology

We deploy Qwen3-235B with SGLang on  $8\times H100$  GPUs with NVLink. We build a distributed profiler by inserting cuda. Event timers into SGLang to measure individual operations (attention, top-k, all-to-all, and MoE) on each GPU independently. We manipulate expert placement through SGLang's init\_expert\_location interface and use DeepEP as the MoE backend. The ep\_dispatch\_algorithm is set to "dynamic" so that tokens are evenly distributed across replicas of a duplicated expert.

**Metric.** We report MoE computation time, i.e., all three expert linear layers, excluding attention, all-to-all, and top-k.

**Model and Benchmark.** We evaluate on Qwen3-235B (94 MoE layers, 128 experts per layer, 8 selected). We use MMLU and Global-MMLU datasets, following the original ordering. Batch sizes range from 64 to 16,384.

![](_page_12_Figure_0.jpeg)

<span id="page-12-0"></span>Figure 17. Performance of our prefill-aware expert placement.

**Baselines.** Default is the standard contiguous placement used by Qwen and SGLang (experts 0–15 on GPU-0, 16–31 on GPU-1, etc.). Best and Worst are the theoretically optimal and worst placements generated with oracle decodestage selections (not available in practice). Remap and Dup are our two prefill-guided strategies. For Dup, we use one extra slot per GPU, yielding 128+8=136 experts per layer.

#### C. Results

As shown in Figure 17, Remap and Dup achieve speedups of 15.5% and 12.5% over Default, respectively, and deliver over 2× speedup compared with Worst. Both remain within 10% of Best, which exploits oracle decode-stage information unavailable in practice, which demonstrates the effectiveness of our approach. Since the two algorithms perform comparably, one can choose between them to fit different memory and system constraints.

We note that our 8-GPU EP scale inherently limits the achievable improvement: with EP8, each GPU holds 16 experts per layer, so every GPU likely contains a mix of hot and cold experts, naturally yielding a relatively balanced workload even under the Default layout (the max/min execution-time ratio is only about 1.3×). We expect greater speedups at larger EP scales where load imbalance is more pronounced.

#### VII. DISCUSSION

Both the wafer-scale GPU architecture and the prefill-guided expert placement strategy serve as case studies demonstrating the practical applicability of our profiling insights, which constitute the paper's primary contribution. Specifically, the wafer-scale GPU design follows <a href="Insight 3">Insight 3</a> for task allocation and leverages temporal relation insights (Insight 1 and part of Insight 2) to build a data-driven predictor. The prefill-guided placement strategy utilizes <a href="Insight 1">Insight 1</a> to guide decode-stage expert placement using information collected during prefill.

Importantly, our insights extend far beyond these two case studies and can benefit a wide range of MoE serving systems, including multi-GPU clusters (Multi-Node DGX [9], [15], [70] and NVL72 [71]), CXL-/CPU-based memory disaggregation [12], [31], flash-based multi-tier systems [72], [73], PIM architectures [33], [74], [75], and other emerging platforms.

