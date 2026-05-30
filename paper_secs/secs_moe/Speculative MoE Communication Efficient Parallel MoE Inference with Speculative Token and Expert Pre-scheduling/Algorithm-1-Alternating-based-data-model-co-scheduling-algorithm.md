# Algorithm 1: Alternating-based data-model co-scheduling algorithm

```
input: n steps: number of iteration steps;
  Cp: the token-2-expert confidence table; a: the token frequency;
  r: requests list; K: number of requests;
  N: number of experts per layer; E: number of co-clusters;
  t: number of tokens;
  output: E: expert labels; T : token labels;
  Tp: confidence of tokens choosing specific experts
1 p matrix ep opt ← zeros(N, E) / E
2 p matrix req opt ← zeros(K, E) / E
3 Function expert place(Cp, p matrix req, αe, βe):
4 loads ← compute per expert load by (Cp)
5 sort by load(e, loads)
6 mask, cnter ← ones(E), zeros(E)
7 EAfE, EAfR ← zeros(N, E), zeros(N, E)
8 p matrix ep ← zeros(N, E) / E
9 loads cls ← zeros(E)
10 for e in e do
11 EAfE[e] ← compute expert-expert affinity by (mask, p matrix ep, Cp)
12 EAfR[e] compute req-expert affinity by (mask, p matrix req, Cp)
13 aff score ← αr ∗ EAfE[e] + βr ∗ EAfR[e] − γe ∗ loads cls
14 clse ← arg maxcls aff score
15 p matrix ep[e][clse] ← 1
16 cnter[clse] ← cnter[clse] + 1
17 if cnter[clse] >= N/E then
18 maks[clse] ← 0
19 loads cls[clse] ← loads cls[clse] + loads[e]
20 repeat
21 cls1, cls2 ← randomly select a cluster in JEK
22 e1, e2 ← randomly select experts in p matrix ep[:][cls1] and p matrix ep[:][cls2]
23 if aff gain(e1, e2, cls1, cls2) > 0 then
24 swap(e1, e2, p matrix ep)
25 until iterating for f t steps steps
26 return p matrix eq
27
28 Function request schedule(Cp, p matrix ep, αr, βr):
29 sort by len(r)
30 mask, cnter ← ones(E), zeros(E)
31 RAfR, RAfE ← zeros(K, E), zeros(K, E)
32 p matrix req ← zeros(K, E) / E
33 for r in r do
34 RAfR[r] ← compute req-req affinity by (mask, p matrix req, Cp)
35 RAfE[r] ← compute req-expert affinity by (mask, p matrix ep, Cp)
36 aff score ← αr ∗ RAfR[r] + βr ∗ RAfE[r]
37 clsr ← arg maxcls aff score
38 p matrix req[r][clsr] ← 1
39 cnter[clsr] ← cnter[clsr] + 1
40 if cnter[clsr] >= K/E then
41 maks[clsr] ← 0
42 return p matrix req
43
44 p matrix req ← cluster based on expert affinity
45 repeat
46 p matrix ep ← expert place(Cp, p matrix req, αe, βe)
47 p matrix req ← request schedule(Cp, p matrix ep, αr, βr)
48 scores ← summation the max load and communication cost given p matrix eq and p matrix req
49 better scheduling ← samples with scores
50 update p matrix ep opt and p matrix req opt
51 until iterating for n steps steps
52
53 E ← argmax(p matrix ep opt, axis=1)
54 p matrix tk opt ← count the tokens per req in p matrix req opt
55 T , Tp ← argmax with values(p matrix tk opt, axis=1)
```

### B.1 MODELING INTER-LAYER ACTIVATION CONJUGACY

Leveraging the conditional probability model described in § A.2, we use a simple probability-based first-order Marcov chain to model the inter-layer activation conjugacy. To reduce the combination space, we model the activation device sequence rather than the activation expert sequence, because we only care about the device-level token rebatching. When looking back l layers, we construct a table shaped like  $[E^l, E]$ , where the row of the table indicates the sequence of devices selected at the previous l layers and the column indicates the probability of activating the E devices in the current layer. Like the § B shows, we also calculate the activation sequence to device table  $\mathcal{A}$  and the confidence table  $\mathcal{A}_p$ . In practice, we set the number of looking-back layers as 2.

**Algorithm 2:** Online request scheduling based on fast lookup

```
input: \mathcal{R} \in \mathbb{N}^n: Input requests; \mathcal{T}: token-to-expert-cluster Schedule Table; E: number of DP size

1
```

#### B.2 Speculative Token Shuffling on the Fly Based on Fast Lookup

To reduce the combination space, we model the activation device sequence rather than the activation expert sequence, because we only care about the device-level token rebatching. We implement a fast online token re-batching mechanism based on fast looking-up tables in both Attention-DP and Attention-TP (Algorithm 2 & Algorithm 3).

**Data Scheduling: Attention-DP Scenarios.** The algorithm 2 queries the token-to-expert-cluster scheduling table  $\mathcal{T}$  based on the token IDs appearing in the request  $\mathcal{R}$ , and aggregates the results to obtain a score for each device for that request (line 3). Then  $\mathcal{R}$  is scheduled to the device with the max valid score (line 5). To prevent requests biased toward a subset of experts, which could skew the load during the decoding phase, we introduce a  $dev\_mask$ . The device is masked after it is allocated (line 4-5). Once a round of allocation is completed and all devices are masked, the  $dev\_mask$  is reset and enters a new round (line 7-8). This ensures that Sem-MoEachieves expert affinity while maintaining load balance across devices.

**Data Scheduling: Attention-TP Scenarios.** The algorithm 3 queries the token-to-expert-cluster scheduling table  $\mathcal{T}$  and expert-cluster-sequence-to-expert-cluster table  $\mathcal{S}$ , together with their confidences first. Then, the table with higher confidence is adopted to obtain the device ID list to which the current batch token needs to be shuffle (line 2). The algorithm performs the argsort operation to obtain the shuffle indicators (line 3) of the token. Then, the final shuffle indicators are obtained by grouping, aligning, and concatenation, and the token is shuffled (line 4 to line 7). After rebatching is complete, Sem-MoE calls the reduce-scatter operation. After MoE computing is complete, Sem-MoE runs the allgather operation to collect tokens. Finally, the order of tokens are shuffled back based on the previously calculated shuffle indicators (lines 14-18).

The both algorithm do not involve complex load calculation and decision-making. They are directly completed by querying tables. The runtime overhead mainly involves large token matrix shuffling, which we optimize via high-performance kernels. The memory occupation of the scheduling tables is negligible. For example, for DeepSeek-V2, the memory space that the token-to-device table  $\mathcal{T}$  occupies is  $\frac{102400 \times 60 \times 2}{1024^2} \approx 11.72 MB$  (assuming the data format is int16).

