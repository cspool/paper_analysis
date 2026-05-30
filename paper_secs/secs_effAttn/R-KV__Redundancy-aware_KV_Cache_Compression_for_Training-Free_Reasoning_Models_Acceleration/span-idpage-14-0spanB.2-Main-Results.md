# <span id="page-14-0"></span>B.2 Main Results

See [Table 2.](#page-16-2)

<span id="page-15-0"></span>**Algorithm 1** R-KV:  $Q_{\text{obs}}$  are query states for  $\alpha$  observation tokens,  $K_{\text{full}}$ ,  $V_{\text{full}}$  are the full KV cache states of length  $L_{\text{full}}$ .

```
1: procedure R-KV((K_{\text{full}}, V_{\text{full}}), L_{\text{full}}, L_{\text{budget}}, Q_{\text{obs}}, \alpha, B_{\text{budget}}, B_{\text{buffer}}, T, \beta, \lambda, \epsilon, H, d_k)
                                                                                                                 ▷ Check if compression is triggered
  2:
              if L_{\text{full}} - L_{\text{budget}} < B_{\text{buffer}} then
  3:
                     return (\boldsymbol{K}_{\text{full}}, \boldsymbol{V}_{\text{full}})
  4:
  5:
               (\mathbf{K}_{\text{obs}}, \mathbf{V}_{\text{obs}}) \leftarrow \text{last } \alpha \text{ tokens of } (\mathbf{K}_{\text{full}}, \mathbf{V}_{\text{full}})
  6:
               (\mathbf{K}_{\text{cand}}, \mathbf{V}_{\text{cand}}) \leftarrow \text{first } (L_{\text{full}} - \alpha) \text{ tokens of } (\mathbf{K}_{\text{full}}, \mathbf{V}_{\text{full}})
              N_c \leftarrow L_{\text{full}} - \alpha
  7:

    Number of candidate tokens

  8:
              if N_c \leq B_{\text{budget}} then
  9:
                     return (K_{\text{full}}, V_{\text{full}})
                                                                                         ▶ Not enough candidates to prune beyond budget
10:
              for each head h = 0 \dots H - 1 do
11:
                     Compute attention matrix A^h \in \mathbb{R}^{\alpha \times N_c} using Q^h_{obs} and K^h_{cand} \triangleright Handles MHA/GQA as
12:
       per Eqs. (1)-(3) from text
                    \begin{array}{c} \textbf{for } k = 0 \dots N_c - 1 \textbf{ do} \\ I'_{k,h} \leftarrow \frac{1}{\alpha} \sum_{q=0}^{\alpha-1} (\boldsymbol{A}^h)_{qk} \end{array}
                                                                                                                             \triangleright For each candidate token k
13:
14:
                                                                                                     \triangleright q: observation token, k: candidate token
                     end for \{I_{k,h}\}_{k=0}^{N_c-1} \leftarrow \text{1D-Pooling}(\{I'_{k,h}\}_{k=0}^{N_c-1})
15:
16:
17:
               \begin{aligned} & \textbf{for} \text{ each head } h = 0 \dots H - 1 \textbf{ do} \\ & \boldsymbol{K}^h_{\text{norm}} \in \mathbb{R}^{N_c \times d_k}; \text{For } k = 0 \dots N_c - 1, \boldsymbol{K}^h_{\text{norm},k} \leftarrow \boldsymbol{K}^h_{cand,k} / (\|\boldsymbol{K}^h_{cand,k}\|_2 + \epsilon) \end{aligned}
18:
19:
                     \boldsymbol{S}^h \leftarrow \boldsymbol{K}^h_{\text{norm}} (\boldsymbol{K}^h_{\text{norm}})^{\top}
                                                                       ▷ Cosine Similarity Matrix Computation, similarity matrix
20:
       \boldsymbol{S}^h \in \mathbb{R}^{N_c \times N_c}
21:
                    for k = 0 ... N_c - 1 do
                                                                                                                               ▶ Prevent Self-Redundancy
                          (\mathbf{S}^h)_{kk} \leftarrow 0
22:
                     end for
23:
                     B^h_{uv} \leftarrow ((\mathbf{S}^h)_{uv} > T?1:0) for u,v \in \{0,\dots,N_c-1\} \triangleright Identify Highly Similar Pairs
24:
                     for u = 0 ... N_c - 1 do T_u^h \leftarrow \{v \mid B_{uv}^h = 1, v \in \{0, ..., N_c - 1\}\}
25:

26:
                           T_{u,\beta}^h \leftarrow \text{subset of } T_u^h \text{ with up to } \beta \text{ largest indices } v.
27:
                           for v' \in T_{u,\beta}^h do
28:
                                  (\mathbf{S}^h)_{u,v'} \leftarrow 0
29:
                                                                                                                                         \triangleright S^h is now modified
30:
                           end for
31:
                     end for
                    Let \bar{\mathbf{S}}^h \in \mathbb{R}^{N_c} where (\bar{\mathbf{S}}^h)_u \leftarrow \frac{1}{N_c} \sum_{v=0}^{N_c-1} (\mathbf{S}^h)_{uv}
32:
                     for u = 0 \dots N_c - 1 do
33:
                           R_{u,h} \leftarrow (\operatorname{softmax}(\bar{\mathbf{S}}^h))_u
34:
35:
36:
              end for
              for each head h = 0 \dots H - 1 do
37:
                     for k = 0 ... N_c - 1 do
38:
                           Score_{k,h} \leftarrow \lambda I_{k,h} - (1-\lambda)R_{k,h}
39:
40:
                     end for
              end for
41:
              Let AggScore \in \mathbb{R}^{N_c}
42:
              for k = 0 ... N_c - 1 do
43:
44:
                     AggScore_k \leftarrow mean_h(Score_{k,h})

    ▶ Aggregate scores across heads

45:
46:
              Idx_{sel} \leftarrow \text{indices of top-} B_{budget} \text{ tokens from } \{0, \dots, N_c - 1\} \text{ based on AggScore}
              K_{cand\_sel} \leftarrow K_{cand}[Idx_{sel}]; V_{cand\_sel} \leftarrow V_{cand}[Idx_{sel}]
47:
              K_{comp} \leftarrow \text{concatenate}(K_{cand\_sel}, K_{obs})
                                                                                                                                              ▷ Order might vary
48:
              V_{comp} \leftarrow \text{concatenate}(V_{cand\_sel}, V_{obs})
49:

    □ Update length for next cycle

50:
              L_{prev\_comp} \leftarrow B_{budget} + \alpha
              return (\boldsymbol{K}_{comp},\boldsymbol{V}_{comp})
51:
52: end procedure
```

<span id="page-16-2"></span>

| Model     | Benchmark | Method | 128   | 256   | 512   | 768   | 1 024 | 1 536 | 2 048 | 2 5 6 0 | 3 072 | 4 096 |
|-----------|-----------|--------|-------|-------|-------|-------|-------|-------|-------|---------|-------|-------|
| Llama3-8B | МАТН      | FullKV | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | _       | _     | _     |
|           |           | R-KV   | 51.08 | 67.39 | 76.92 | 80.21 | 81.34 | 82.34 | 82.65 | _       | _     | _     |
|           |           | SnapKV | 32.53 | 50.07 | 64.03 | 70.81 | 74.43 | 78.43 | 80.50 | -       | _     | -     |
|           | AIME24    | FullKV | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79   | 49.79 | _     |
|           |           | R-KV   | 0.42  | 10.21 | 29.48 | 40.31 | 45.26 | 51.56 | 52.29 | 53.85   | 53.13 | _     |
|           |           | SnapKV | 0.16  | 0.94  | 4.53  | 11.20 | 15.73 | 26.04 | 32.76 | 39.43   | 41.93 | -     |
| Qwen-14B  | МАТН      | FullKV | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | _       | _     | _     |
|           |           | R-KV   | 56.21 | 73.33 | 84.77 | 88.79 | 90.72 | 92.72 | 93.62 | -       | _     | -     |
|           |           | SnapKV | 26.32 | 43.93 | 77.93 | 82.52 | 86.63 | 90.86 | 92.73 | -       | -     | -     |
|           | AIME24    | FullKV | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | _       | 65.68 | 65.68 |
|           |           | R-KV   | 0.57  | 7.92  | 24.53 | 36.25 | 42.66 | 55.00 | 56.09 | _       | 64.32 | 67.45 |
|           |           | SnapKV | 0.26  | 2.86  | 12.86 | 16.30 | 25.00 | 36.41 | 46.56 | _       | 52.86 | 54.32 |

Table 2: Accuracy (%) of **Llama3-8B** and **Qwen-14B** on the MATH and AIME24 benchmarks under different memory-optimization methods across context lengths. "—" denotes configurations that were not evaluated.

