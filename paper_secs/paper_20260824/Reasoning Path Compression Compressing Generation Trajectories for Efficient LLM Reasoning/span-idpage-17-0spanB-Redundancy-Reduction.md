# <span id="page-17-0"></span>B Redundancy Reduction

### B.1 Details of Embedding-based Redundancy Measurements

This section provides details of the embedding-based analysis described in Section [4.2](#page-6-3) used to demonstrate the redundancy-reducing effect of RPC.

For this analysis, we generated outputs for each sample in the AIME 2024 dataset [\[13\]](#page-10-3) using both the full KV baseline and RPC. Generation was stopped after the 4128th decoding step, where the first compression of RPC is completed, given P = 4096 and R = 32. The generated outputs are separated into individual sentences, and each sentence is embedded using the all-MiniLM-L6-v2 model with Sentence Transformers library [\[36\]](#page-11-6). The model maps each token sequence to a 384-dimensional vector.

Pairwise cosine similarities were computed between all sentence embeddings within the same generated output. Two sentences with cosine similarity above 0.75 were considered semantically similar. Finally, we defined the *redundancy rate* as the proportion of sentences that have more than N semantically similar counterparts within the same output, where N ∈ 1, 2, 4.

### B.2 Visualized Examples of Token Selection with RPC

In this section, we provide qualitative examples illustrating how RPC selects tokens to be retained in the KV cache.

In the following examples, token colors indicate how frequently each token is retained across layers: tokens preserved by a larger proportion of layers appear in warmer colors, following the scale defined in the color legend below.

### Legend:

- Black: Unselected or low selection (<40%)
- Yellow: Medium selection (40-60%)
- Orange: High selection (60-80%)
- Red: Very high selection (≥80%)

Examples [B.1](#page-17-1) and [B.2](#page-19-0) are sampled from outputs of DeepSeek-R1-Distill-Qwen-7B [\[3\]](#page-9-2) and QwQ-32B [\[12\]](#page-10-2), respectively, on the AIME 2024 dataset.

While it is natural that recent tokens tend to exhibit higher retention frequencies due to LLMs' tendency to assign larger attention weights to local tokens, the selection pattern from a semantic perspective can be broadly categorized into two types:

- (1) Information-bearing tokens: Tokens containing critical information that advances the reasoning process, such as conclusions of intermediate steps.
- (2) Transition tokens: Tokens indicating shifts or boundaries between reasoning stages, which, although not semantically rich on their own, play an important contextual role through the attention mechanism by aligning and integrating information across reasoning steps.

This qualitative analysis demonstrates that RPC effectively preserves essential reasoning content while filtering out redundant or outdated derivations, thereby maintaining the continuity and integrity of the reasoning trajectory.

### <span id="page-17-1"></span>Example B.1

Problem. Let △ABC have circumcenter O and incenter I with IA ⊥ OI, circumradius R = 13, and inradius r = 6. Find AB · AC.

DeepSeek-R1-Qwen-7B Output. *<think>* Okay, so I need to find the product AB times AC in triangle ABC, given that the circumradius is 13, the inradius is 6, and the segment IA is perpendicular to OI, where O is the circumcenter and I is the incenter. Hmm, that seems a bit involved, but let me try to break it down step by step.

First, let me recall some properties and formulas related to triangle centers, circumradius, inradius, and maybe some coordinate geometry. Since the problem involves perpendicularity between IA and OI, maybe coordinate geometry could be a good approach here. Alternatively, there might be some formula that connects these elements directly. Let me think.

I know that in any triangle, the distance between the incenter and circumcenter can be expressed by the formula: OI² = R(R - 2r) where R is the circumradius and r is the inradius.

Wait, is that correct? Let me verify. Hmm, actually, I think the formula is OI² = R² - 2Rr. Yes, that's right. So OI squared equals R squared minus 2 times R times r. So substituting the given values, R is 13 and r is 6, so OI² = 13² - 2\*13\*6 = 169 - 156 = 13. Therefore, OI is sqrt(13). Hmm, but the problem states that IA is perpendicular to OI. Maybe this can help relate IA and OI? [...]

The inradius r is related to the area and semiperimeter: r = area / s, where s = (a + b + c)/2. The circumradius R is related to the sides by the formula a = 2R sin A, and similar for other sides. Also, the distance between O and I is given by OI² = R² - 2Rr, which we already computed as 13. So OI = sqrt(13). But we also have that IA is perpendicular to OI. So maybe we can use vectors or coordinate geometry to relate these.

Alternatively, perhaps using the formula for the distance from the incenter to a vertex. The length IA can be expressed in terms of the angles and inradius. Let me recall that in a triangle, the distance from the incenter to vertex A is given by: IA = \{}sqrt{r^2 + \{}left( s - a \{}right)^2 }

Wait, is that correct? Let me see. The formula for the distance from the incenter to vertex A: yes, in triangle ABC, the distance from I to A is sqrt(r² + (s - a)²), where s is the semiperimeter and a is the side BC. Similarly for other vertices. [...]

But since A is a vertex of the triangle on the circumradius of 13, so OA = 13, so x\_A² + y\_A² = 13² = 169.

Hmm, this might get complicated with too many variables. Maybe there is a better way.

Alternatively, since O is the circumcenter, coordinates can be set such that triangle ABC is inscribed in a circle of radius 13. Let me perhaps place point A at (13, 0) for simplicity, so coordinates of A are (13, 0). Then O is at (0, 0). Then points B and C are somewhere on the circle x² + y² = 169. [...]

Expressed as vectors, OI is (I - O) and IA is (A - I). Since O is the origin, OI is just the coordinates of I, and IA is (A - I). So their dot product is I • (A - I) = 0.

Therefore, I • A - |I|² = 0.

Therefore, I • A = |I|². Hmm, that's an equation relating the coordinates of I and A. Since A is on the circumcircle, |A| = R = 13. [...]

Therefore, IA² = 169 - 13 = 156, so IA = sqrt(156) = 2\*sqrt(39). Hmm, that seems useful.

So IA is sqrt(156). But we also have IA = r / sin(α/2). Since r = 6, so:sqrt(156) = 6 / sin(α/2)

Therefore,  $\sin(\alpha/2) = 6$  /  $\operatorname{sqrt}(156) = 6$  /  $(2*\operatorname{sqrt}(39)) ) = 3$  /  $\operatorname{sqrt}(39) = \operatorname{sqrt}(39)/13$ . Wait, let's compute that again:

Wait 6 divided by  $\operatorname{sqrt}(156)$  is equal to 6 divided by  $\operatorname{(sqrt}(4*39)) = 6/(2*\operatorname{sqrt}(39)) = 3/\operatorname{sqrt}(39)$ . Rationalizing the denominator:  $3/\operatorname{sqrt}(39) = 3*\operatorname{sqrt}(39)/39 = \operatorname{sqrt}(39)/13$ . So  $\sin(\alpha/2) = \operatorname{sqrt}(39)/13$ .

Therefore,  $\sin(\alpha/2) = \text{sqrt}(39)/13$ . Let me note that sqrt(39)/13 is equal to sqrt(39)/13, which is approximately  $6.245/13 \approx 0.48$ , which is plausible.

Then,  $\cos(\alpha/2)$  would be  $\operatorname{sqrt}(1 - (39/169)) = \operatorname{sqrt}(130/169) = \operatorname{sqrt}(130)/13$ .

Therefore,  $cos(\alpha/2) = sqrt(130)/13$ .

Then,  $\sin \alpha = 2 \sin(\alpha/2) \cos(\alpha/2) = 2*(\sqrt{39})/13)*(\sqrt{130})/13) = 2*\sqrt{39*130}/(\frac{13^2}{130})$ .

Compute 39\*130: 39\*100 = 3900, 39\*30 = 1170, so total 3900 + 1170 = 5070. So sqrt(5070).

Hmm, 5070 factors: 5070 = 10\*507 = 10\*3\*13\*13. Wait, 507 divided by 13 is 39, so 507 = 13\*39 = 13\*3\*13.

Wait, 39 is 3\*13, so 5070 = 10\*3\*13\*13. Therefore  $sqrt(5070) = sqrt(10*3*13^2)$  ) = 13\*sqrt(30). Because 10\*3 = 30. So sqrt(5070) = 13\*sqrt(30).

Therefore,  $\sin \alpha = 2*(13*\operatorname{sqrt}(30))/(13^2) = (2 \operatorname{sqrt}(30))/13$ .

Thus  $\sin \alpha = (2 \text{ sqrt}(30))/13$ .

Hmm, so angle A has sine equal to  $2\sqrt{30/13}$ .

Therefore, the side BC can be found via the Law of Sines:  $a = BC = 2R \sin \alpha = 2*13*(2\sqrt{30/13}) = 4\sqrt{30}$ . [...]

### <span id="page-19-0"></span>Example B.2

**Problem.** Let A, B, C, and D be points on the hyperbola  $\frac{x^2}{20} - \frac{y^2}{24} = 1$  such that ABCD is a rhombus whose diagonals intersect at the origin. Find the greatest real number that is less than  $BD^2$  for all such rhombi.

**QwQ-32B Output.** <think> Okay, so I have this problem here about a hyperbola and a rhombus inscribed in it. The hyperbola equation is given as  $x^2/20 - y^2/24 = 1$ . The rhombus ABCD has its diagonals intersecting at the origin, and I need to find the greatest real number that is less than BD<sup>2</sup> for all such rhombi. Hmm, okay, let me break this down step by step. [...]

But first, let me think about coordinates. Let me denote points A and C as (x1, y1) and (-x1, -y1), and points B and D as (x2, y2) and (-x2, -y2). Since all four points lie on the hyperbola, they must satisfy the equation  $x^2/20 - y^2/24 = 1$ . So for points A and C,  $(x1)^2/20 - (y1)^2/24 = 1$ , and similarly for points B and D,  $(x2)^2/22 - (y2)^2/24 = 1$ . [...]

Let me consider parametrizing the hyperbola. The hyperbola equation is  $x^2/20 - y^2/24 = 1$ . For hyperbola parametrization, we can use hyperbolic functions. Let me recall that a standard hyperbola  $x^2/a^2 - y^2/b^2 = 1$  can be parametrized as  $(a \sec \theta, b \tan \theta)$ , or using hyperbola parameters. Alternatively, using coordinates:

Let me set x = sqrt(20) secθ, y = sqrt(24) tanθ. That would satisfy the hyperbola equation. Alternatively, perhaps using coordinates in terms of parameters t and s for points A and B.

Alternatively, since the hyperbola is symmetric, maybe we can set point A as (x, y) and point B as (u, v), such that x²/20 - y²/24 =1 and u²/20 - v²/24 =1. Then, the condition is x u + y v =0 (from x1x2 + y1y2 =0). Then, BD² would be the squared distance between points B and D. Since D is (-u, -v), BD is the distance between (u, v) and (-u, -v), which is sqrt[(2u)^2 + (2v)^2] so BD squared is (2u)^2 + (2v)^2 =4(u² + v²). [...]

Alternatively, maybe use Lagrange multipliers to minimize u² + v² subject to the constraints:

```
1. u²/20 - v²/24 =1,
```

2. x u + y v =0,

and also x²/20 - y²/24 =1.

But this might get complicated with multiple variables. Let me see if I can express variables in terms of each other.

Let me suppose that OA is (x,y) and OB is (u,v). Since OA and OB are perpendicular, then u = k y, v = -k x for some scalar k. Wait, because if (x,y) and (u,v) are perpendicular, then (u,v) is a scalar multiple of (-y, x) or (y, -x). Let me check:

If OA is (x,y), then a perpendicular vector would be (-y, x) or (y, -x). So, (u,v) must be a scalar multiple of that. Let me suppose (u,v) = λ\*(-y, x). Then, substituting into the hyperbola equation for B:

$$u^2/20 - v^2/24 = ((\lambda^2 y^2)/20) - ((\lambda^2 x^2)/24) = \lambda^2 (y^2/20 - x^2/24) = 1.$$

Wait, but the hyperbola equation for point A is x²/20 - y²/24 =1, so y²/24 = (x²/20 -1). Hmm, so let me rearrange that.

Wait, let me see:

Given (u,v) = λ\*(-y, x). Then, substituting into the hyperbola equation:

$$u^2/20 - v^2/24 = (\lambda^2 y^2)/20 - (\lambda^2 x^2)/24 = \lambda^2 (y^2/20 - x^2/24) = 1.$$

But from point A's equation, x²/20 - y²/24 =1, which can be rearranged to y²/24 = (x²/20 -1). Therefore, y² =24\*(x²/20 -1). Let me plug that into the expression above.

So, 
$$y^2/20 - x^2/24 = (24*(x^2/20 - 1))/20 - x^2/24$$
.

Wait, let me compute that step by step.

First, compute y²/20:

$$y^2/20 = [24*(x^2/20 - 1)]/20 = 24/(20)*(x^2/20 - 1) = (24/20)*(x^2/20 - 1)$$

Wait, perhaps this is getting too convoluted. Let me think differently.

Let me denote S = u² + v². Since BD squared is 4S, we need to find the minimal value of S given that (u,v) lies on the hyperbola and (u,v) is perpendicular to (x,y) which is also on the hyperbola.

Alternatively, since (u,v) is along the direction perpendicular to (x,y), let me parametrise

```
(u,v) as a scalar multiple of (-y, x). So let me write u = -λ y, v = λ x. Then, substituting into
the hyperbola equation for B:
```

$$u^2/20 - v^2/24 = (\lambda^2 y^2)/20 - (\lambda^2 x^2)/24 = \lambda^2 (y^2/20 - x^2/24) = 1.$$

From point A's hyperbola equation, x²/20 - y²/24 =1, so rearranged: y²/24 = x²/20 -1, so y² =24\*(x²/20 -1). Let me substitute that into the expression above.

First, compute y²/20 - x²/24:

$$y^2/20 = [24*(x^2/20 - 1)]/20 = (24/20)*(x^2/20 - 1) = (6/5)*(x^2/20 - 1)$$

x²/24 is just x²/24.

Therefore, y²/20 - x²/24 = (6/5)(x²/20 -1) - x²/24

Let me compute this term:

First, 
$$(6/5)(x^2/20 - 1) = (6x^2)/(100) - 6/5 = (3x^2)/50 - 6/5$$

Then subtract x²/24:

$$(3x^2/50 - 6/5) - x^2/24 = (3x^2/50 - x^2/24) - 6/5$$
 [...]

### C Extra Experimental Results

### C.1 Output Length Statistics across Evaluation Datasets

Table 3 summarizes the output length statistics of DeepSeek-R1-Distill-Qwen-7B [3] and QwQ-32B [12], measured under the full KV cache setting. For both models, AIME 2024 [13] and LiveCodeBench [25] exhibit long reasoning outputs (over 10K tokens on average), whereas IFEval produces much shorter outputs. Accordingly, compression in RPC is triggered multiple times for long-context benchmarks but rarely for short ones, which justifies using a consistent P value (1024 or 4096) across all datasets to prevent excessive KV cache growth.

| Table 3: Output length distr | ibution under f | iıll KV | cache setting. |
|------------------------------|-----------------|---------|----------------|
|------------------------------|-----------------|---------|----------------|

<span id="page-22-1"></span>

| Dataset                              | DeepS                        | DeepSeek-R1-Distill-Qwen-7B |                         |                            | QwQ-32B                      |                    |                         |                            |
|--------------------------------------|------------------------------|-----------------------------|-------------------------|----------------------------|------------------------------|--------------------|-------------------------|----------------------------|
| Dutuset                              | Mean                         | Min                         | Max                     | Std                        | Mean                         | Min                | Max                     | Std                        |
| AIME 2024<br>LiveCodeBench<br>IFEval | 13668.6<br>11889.1<br>1778.4 | 2413<br>809<br>190          | 32768<br>32768<br>32768 | 9356.4<br>7066.2<br>5073.3 | 13834.6<br>13454.6<br>1336.9 | 2747<br>491<br>144 | 32768<br>32768<br>32768 | 7365.5<br>9692.1<br>1844.3 |

#### C.2 Granularity of Attention Score Aggregation for Token Selection

To examine how the granularity of attention score aggregation for token selection impacts accuracy, we compare three aggregation schemes: *layer-wise*, *group-wise* (key-value group), and *head-wise* (no aggregation). In our approach, attention scores are aggregated at the layer level, meaning that saliency is averaged across all heads within each layer.

Table 4 presents results for DeepSeek-R1-Distill-Qwen-7B on AIME 2024 dataset under different aggregation granularities. The results show that layer-wise aggregation consistently yields the highest accuracy, indicating that averaging attention scores across heads helps stabilize the estimation of token importance and preserve overall performance after compression.

<span id="page-22-2"></span>Table 4: AIME 2024 (pass@1) results for DeepSeek-R1-Distill-Qwen-7B with different attention score aggregation granularities.

| P    | Layer Aggregation | <b>Group Aggregation</b> | Head (No Aggregation) |
|------|-------------------|--------------------------|-----------------------|
| 4096 | 52.9              | 50.8                     | 49.6                  |
| 1024 | 50.4              | 50.4                     | 47.5                  |

Layer-wise aggregation offers a coarser yet more reliable estimation of token saliency than head-level aggregation, which often suffers from high variance and instability across heads. This observation is consistent with previous work. TOVA [11] reported that layer-level token selection outperformed head-level selection in terms of perplexity.

Moreover, in models using grouped-query attention (GQA) [37], multiple attention heads share a single KV head. Performing token selection separately for each head in such architectures would require maintaining distinct KV caches per head, introducing significant memory overhead and defeating the purpose of compression. Therefore, even aside from accuracy, head-level selection is impractical for GQA-based models.

Overall, the layer-level token selection strategy adopted in RPC offers a practical and stable solution for compressing the KV cache of reasoning LLMs.

#### <span id="page-22-0"></span>**C.3** Efficiency Evaluation

We evaluate the efficiency gains of RPC by comparing decoding throughput and peak memory usage against the inference with full KV cache. Specifically, we report results for the default  $4\times$  compression setting of RPC with two compression intervals, P=1024 and P=4096. Throughput is measured in tokens per second, and peak memory reflects the maximum GPU memory consumption during generation.

Measurements were conducted using two reasoning models: DeepSeek-R1-Distill-Qwen-7B and QwQ-32B. For DeepSeek-R1-Distill-Qwen-7B, evaluations were performed on a single NVIDIA H100 SXM GPU, while QwQ-32B was tested using 4 NVIDIA H100 SXM GPUs in parallel. We fix the input length to 128 tokens and vary the generation length across 4096, 8192, 16384, and 32768 tokens. Batch size is varied across 8, 16, and 32 to assess scalability under different workloads.

Results for DeepSeek-R1-Distill-Qwen-7B are shown in Table [5,](#page-23-0) and the corresponding results for QwQ-32B are reported in Table [6.](#page-24-0)

<span id="page-23-0"></span>Table 5: DeepSeek-R1-Distill-Qwen-7B's throughput and peak memory usage by batch size and generation length.

| Metric                | Batch Size | 4096           | 8192    | 16384   | 32768  |  |  |  |
|-----------------------|------------|----------------|---------|---------|--------|--|--|--|
| Full KV Cache         |            |                |         |         |        |  |  |  |
|                       | 8          | 401.50         | 368.72  | 330.41  | 256.92 |  |  |  |
| Throughput (tokens/s) | 16         | 669.53         | 653.04  | 504.50  | 342.88 |  |  |  |
|                       | 32         | 1328.58        | 1031.51 | 671.83  | OOM    |  |  |  |
|                       | 8          | 19.20          | 22.95   | 30.47   | 45.50  |  |  |  |
| Peak Memory (GB)      | 16         | 23.09          | 30.60   | 45.63   | 75.70  |  |  |  |
|                       | 32         | 30.86          | 45.89   | 75.96   | OOM    |  |  |  |
| RPC (P = 1024)        |            |                |         |         |        |  |  |  |
|                       | 8          | 448.19         | 428.31  | 407.00  | 385.00 |  |  |  |
| Throughput (tokens/s) | 16         | 848.75         | 794.62  | 751.69  | 650.20 |  |  |  |
|                       | 32         | 1504.40        | 1499.80 | 1288.51 | 977.11 |  |  |  |
|                       | 8          | 17.08          | 18.02   | 20.27   | 24.75  |  |  |  |
| Peak Memory (GB)      | 16         | 18.86          | 20.74   | 25.15   | 34.20  |  |  |  |
|                       | 32         | 22.40          | 26.16   | 35.00   | 53.08  |  |  |  |
|                       |            | RPC (P = 4096) |         |         |        |  |  |  |
|                       | 8          | 406.43         | 420.62  | 385.75  | 362.75 |  |  |  |
| Throughput (tokens/s) | 16         | 753.11         | 708.95  | 671.38  | 575.21 |  |  |  |
|                       | 32         | 1318.33        | 1247.44 | 1064.05 | 883.43 |  |  |  |
|                       | 8          | 19.20          | 20.14   | 22.02   | 25.77  |  |  |  |
| Peak Memory (GB)      | 16         | 23.09          | 24.96   | 28.72   | 36.24  |  |  |  |
|                       | 32         | 30.86          | 34.62   | 42.13   | 57.16  |  |  |  |

The results show that RPC consistently improves decoding efficiency over full KV cache inference across various batch sizes and generation lengths. As the batch size increases, both the throughput gains and peak memory reductions become more pronounced. This is because larger batches amplify the memory bottleneck imposed by the growing KV cache, allowing RPC's compression to better utilize available GPU compute resources. Notably, full KV cache inference results in out-of-memory (OOM) errors for DeepSeek-R1-Distill-Qwen-7B when the batch size is 32 and the generation length reaches 32768, and for QwQ-32B when the batch size is 16 at 32768 tokens or 32 at 16384 tokens or longer. In contrast, RPC enables successful generation under all of these settings.

When comparing compression intervals, P = 1024 achieves slightly higher throughput and lower peak memory than P = 4096 across both models. While P = 1024 offers stronger compression, it may come at a modest accuracy cost, as shown in Section [4.3.](#page-6-0) Therefore, P = 1024 and P = 4096 can be considered complementary settings: the former prioritizes efficiency, and the latter provides a more balanced trade-off between performance and accuracy.

<span id="page-24-0"></span>Table 6: QwQ-32B's throughput and peak memory usage by batch size and generation length.

| Metric                  | Batch Size | 4096      | 8192   | 16384  | 32768  |  |  |  |
|-------------------------|------------|-----------|--------|--------|--------|--|--|--|
| Full KV Cache           |            |           |        |        |        |  |  |  |
|                         | 8          | 128.79    | 109.80 | 93.28  | 64.85  |  |  |  |
| Throughput (tokens/s)   | 16         | 213.75    | 173.99 | 117.51 | OOM    |  |  |  |
|                         | 32         | 351.34    | 228.59 | OOM    | OOM    |  |  |  |
|                         | 8          | 83.40     | 100.58 | 134.94 | 203.66 |  |  |  |
| Peak Memory (GB)        | 16         | 101.14    | 135.50 | 204.22 | OOM    |  |  |  |
|                         | 32         | 136.61    | 205.33 | OOM    | OOM    |  |  |  |
| <b>RPC</b> $(P = 1024)$ |            |           |        |        |        |  |  |  |
|                         | 8          | 135.79    | 131.97 | 124.45 | 111.84 |  |  |  |
| Throughput (tokens/s)   | 16         | 238.76    | 229.22 | 176.73 | 178.06 |  |  |  |
|                         | 32         | 411.42    | 392.04 | 328.56 | 246.81 |  |  |  |
|                         | 8          | 73.75     | 78.42  | 89.19  | 111.84 |  |  |  |
| Peak Memory (GB)        | 16         | 81.81     | 91.24  | 112.77 | 155.81 |  |  |  |
|                         | 32         | 97.95     | 116.70 | 159.78 | 245.94 |  |  |  |
|                         | RPC (      | P = 4096) |        |        |        |  |  |  |
|                         | 8          | 126.59    | 113.32 | 115.75 | 102.57 |  |  |  |
| Throughput (tokens/s)   | 16         | 214.27    | 207.28 | 187.97 | 147.26 |  |  |  |
|                         | 32         | 345.02    | 314.67 | 279.34 | 208.48 |  |  |  |
|                         | 8          | 83.40     | 87.70  | 96.28  | 114.53 |  |  |  |
| Peak Memory (GB)        | 16         | 101.14    | 109.73 | 126.90 | 163.40 |  |  |  |
| • , ,                   | 32         | 136.61    | 153.79 | 188.14 | 261.13 |  |  |  |

### C.4 Effect of Aggressive Compression

To assess the robustness of RPC under extreme compression, we evaluate its performance with a target compression ratio of  $8\times$ . This setting represents a highly aggressive compression scenario where only one-eighth of the generated tokens' KV entries are retained over time. Table 7 shows the resulting performance across the three benchmark datasets.

Table 7: Accuracy (%) of RPC  $8\times$  compared to RPC  $4\times$  and full KV cache.

<span id="page-24-1"></span>

| Method                                           | DeepSe                | ek-R1-Distill-Qwe         | en-7B              | VB QwQ-32B            |                           |                    | QwQ-32B |  |  |
|--------------------------------------------------|-----------------------|---------------------------|--------------------|-----------------------|---------------------------|--------------------|---------|--|--|
|                                                  | AIME 2024<br>(pass@1) | LiveCodeBench<br>(pass@1) | IFEval<br>(pass@1) | AIME 2024<br>(pass@1) | LiveCodeBench<br>(pass@1) | IFEval<br>(pass@1) |         |  |  |
| Full KV Cache                                    | 55.5                  | 37.6                      | 55.1               | 79.5                  | 63.4                      | 83.9               |         |  |  |
| RPC $4 \times$ Best<br>RPC $8 \times (P = 4096)$ | 52.9<br>47.5          | 35.9<br>32.8              | 57.3<br>55.1       | 78.3<br>72.1          | 62.2<br>57.2              | 82.6<br>84.3       |         |  |  |
| <b>RPC 8</b> × $(P = 1024)$                      | 37.5                  | 27.2                      | 58.4               | 72.1                  | 57.4                      | 82.8               |         |  |  |

Both models exhibit a notable performance drop on AIME 2024 and LiveCodeBench under  $8\times$  compression, compared to the default  $4\times$  setting, indicating the difficulty of preserving reasoning fidelity under extreme compression. Nevertheless, the stronger reasoning model QwQ-32B demonstrates greater robustness, maintaining pass@1 scores close to the results of RPC  $4\times$  across both benchmarks. In contrast, on IFEval, a benchmark characterized by lower reasoning difficulty, the performance remains stable or even improves slightly for both models, suggesting that light-weight instruction-following tasks are less sensitive to aggressive KV cache compression.