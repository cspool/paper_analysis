# <span id="page-19-0"></span>**G Clarification on Terminology and Computational Concepts**

In this section, we provide formal definitions for the terminology used in our method. These terms describe novel computational behaviors in MHLA that lack direct analogues in prior linear attention formulations.

## <span id="page-19-1"></span>**G.1 Concept 1: query-conditioned**

The phrase "query-conditioned" describes a mechanism where the aggregation of contextual information is dynamic and specific to each query instance, distinct from the fixed recurrence found in standard linear attention.

Specifically, the process operates as follows:

- Each query token is associated with a unique vector of mixing coefficients.
- These coefficients are used to weight and aggregate all local KV summaries independently for every query position.

<span id="page-19-2"></span>Consequently, the adaptation occurs per query, rather than globally or via a shared recursive rule.

### **G.2 Concept 2: KV Summary vs. Hidden States**

We introduce the term KV Summary tos strictly distinguish our approach from the Hidden State found in traditional linear attention papers. While the KV summary may seemingly resemble Hidden States in notation, the underlying computation and dependency graphs are structurally different in two key aspects:

<span id="page-20-1"></span>**Table 11** Comparison with LiT. We report the FID scores (mean  $\pm$  std) over three independent runs for MHLA to demonstrate result stability.

| Model                                         | FID (mean $\pm$ std) |
|-----------------------------------------------|----------------------|
| LiT-S/2                                       | 63.21                |
| DiT-S/2 with MHLA                             | $59.744 \pm 0.100$   |
| ${\rm LiT-B/2}$                               | 40.86                |
| $\mathrm{DiT}\text{-}\mathrm{B}/2$ with MHLA  | $37.519 \pm 0.039$   |
| $\rm LiT$ - $\rm L/2$                         | 24.04                |
| DiT-L/2 with MHLA                             | $21.426 \pm 0.051$   |
| $\rm LiT\text{-}XL/2$                         | 20.66                |
| $\mathrm{DiT}\text{-}\mathrm{XL}/2$ with MHLA | $19.164 \pm 0.031$   |

<span id="page-20-2"></span>**Table 14** Profiling results of MHLA under varying sequence length N and token-level head number M. Left: DiT-S/2. Right: DeiT-S/16.

1024

104 imgs/s 11.0G 89 imgs/s 18.0G

8.9G

9.4G

 $124~\rm imgs/s$ 

 $118~\rm imgs/s$ 

| $M\backslash N$ | 256                    | 1024                   | 4096                                                                    | $M \setminus N$      | 256                                                   |
|-----------------|------------------------|------------------------|-------------------------------------------------------------------------|----------------------|-------------------------------------------------------|
| 16              | 40ms 3.9G<br>39ms 4.8G | 51ms 7.2G<br>52ms 8.0G | 147ms 20.8G<br>145ms 21.0G<br>148ms 21.7G<br>157ms 25.4G<br>219ms 40.0G | 4<br>16<br>64<br>256 | 129 imgs/s 3.4G<br>118 imgs/s 3.8G<br>150 imgs/s 5.7G |

- Unlike the strict recursive chain in traditional linear attention where  $h_t$  relies on  $h_{t-1}$ , MHLA computes each Global KV Summary  $(S_g)$  independently, eliminating state propagation across positions.
- While traditional states are derived via a one-to-one update from the previous step, MHLA follows a many-to-one aggregation pattern, where each  $S_g$  is computed from all local summaries using specific mixing coefficients.

By avoiding the rigid inheritance of history inherent to hidden states, MHLA's KV summaries achieve greater expressivity and flexibility.

