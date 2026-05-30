# Algorithm 3: Online token re-batching based on fast lookup

```
input: \mathcal{B} \in \mathbb{N}^n: Input token IDs; \mathcal{T}: token-to-expert-cluster Schedule Table;
     A: expert-cluster-sequence-to-expert-cluster Schedule Table
 1 Function rebatch_tokens (\mathcal{B}, \mathcal{T}):
          dev_{-}ids \leftarrow cond\left(\mathcal{T}_{p}\left[\mathcal{B}\right] > \mathcal{A}_{p}\left[\mathcal{B}\right], \mathcal{T}\left[\mathcal{B}\right], \mathcal{A}\left[\mathcal{B}\right]\right)
 2
          shf\_indices \leftarrow argsort (dev\_ids)
          g\_shf\_indices \leftarrow group\_by\_key(shf\_indices)
 4
          g\_shf\_indices \leftarrow align(g\_shf\_indices)
 5
          shf\_indices \leftarrow \texttt{concat}(g\_shf\_indices)
          \mathcal{B} \leftarrow \mathcal{B} [shf\_indices]
          return shf\_indices
10 Function resume_tokens (\mathcal{B}, shf\_indices):
11
         r\_shf\_indices \leftarrow argsort (shf\_indices)
         \mathcal{B} \leftarrow \mathcal{B}\left[r\_shf\_indices\right]
12
14 shf\_indices \leftarrow rebatch\_tokens(\mathcal{B}, \mathcal{T})
15 \mathcal{B}_{local} \leftarrow \text{reduce\_scatter}(\mathcal{B})
16 executing MoE layer
17 \mathcal{B} \leftarrow \text{allgather}(\mathcal{B}_{local})
18 resume_tokens (B,shf_indices)
```

### C EXTENSIVE EXPERIMENTS

#### <span id="page-18-1"></span>C.1 Cross-dataset Validation of Expert Routing Prediction

To assess the generalization of our method, we conducted cross-domain experiments using two large language models: DeepSeek-V2-Lite and Qwen3-30B. Specifically, we trained Sem-MoE's activation predictor on a single source dataset and evaluated the quality of its token-expert co-scheduling decisions, measured by the Local Activation Rate (LAR), on the other two unseen, out-of-distribution target datasets. Note that LAR measures the proportion of tokens routed to local experts, reflecting a reduced communication volume to experts on remote devices. Therefore, a higher LAR indicates lower cross-device communication overhead—and is better.

The results reveal robust zero-shot transfer performance for our method.

<span id="page-18-2"></span>Table 3: Cross-dataset evaluation of zero-shot transfer for DeepSeek-V2-Lite

| LAR (p50)                                                       | ShareGPT             | Lmsys-Chat-1M           | MMLU             |
|-----------------------------------------------------------------|----------------------|-------------------------|------------------|
| Sem-MoE trained on ShareGPT<br>Sem-MoE trained on Lmsys-Chat-1m | <b>46.49%</b> 40.68% | 41.25%<br><b>47.19%</b> | 41.55%<br>41.40% |
| Baseline: SGLang                                                | 24.98%               | 25.01%                  | 24.97%           |

<span id="page-18-3"></span>Table 4: Cross-dataset evaluation of zero-shot transfer for Qwen3-30B

| Method LAR (p50)                 | ShareGPT | Lmsys-Chat-1M | MMLU   |
|----------------------------------|----------|---------------|--------|
| Sem-MoE trained on ShareGPT      | 46.88%   | 39.28%        | 35.30% |
| Sem-MoE trained on Lmsys-Chat-1m | 36.57%   | 43.61%        | 33.37% |
| Baseline: SGLang                 | 25.00%   | 25.01%        | 25.00% |

As shown in Table 3, for DeepSeek-V2-Lite, the predictor trained on ShareGPT achieves an indistribution LAR of 46.49%. When applied without retraining or fine-tuning to Lmsys-Chat-1M, it still attains 41.25% LAR—only a modest drop from the best in-domain result on that dataset (47.19%, achieved when training directly on Lmsys-Chat-1M). More importantly, this out-of-distribution performance is 1.65× higher than the SGLang's default scheduling setting (25.01%). Similarly, on

MMLU—a domain with very different content and structure—the same ShareGPT-trained predictor yields 41.55% LAR, far surpassing SGLANG defaults (24.97%) and remaining close to in-distribution levels.

As shown in Table 4, the trend is consistent for Qwen3-30B: the ShareGPT-trained predictor achieves 39.28% LAR on Lmsys-Chat-1M and 35.30% on MMLU, compared to peak in-domain scores of 43.61% and 36.57% (when trained on Lmsys), and both are substantially above the SGLang baseline (25%). A similar trend holds when transferring reservedly from Lmsys-Chat-1M to ShareGPT and MMLU.

These results demonstrate that when fed with real-world, representative datasets, Sem-MoE tends to capture the invariability in token-routing patterns that remain effective across different linguistic styles, task types, and knowledge domains. While training on the target distribution offers marginal improvements, it is not required except for extreme performance pursuits.

#### <span id="page-19-0"></span>C.2 THROUGHPUT IMPROVEMENT OF MOONLIGHT-16B

To address the issue of model diversity, we evaluate the Moonlight-16B model. The results align closely with the performance trends observed for DeepSeek-V2-Lite and Qwen3-A30B, as reported in the § 4. In summary, our approach Sem-MoE, consistently outperforms SGLANG (MoETuner): it achieves an average performance improvement of  $1.12\times(1.22\times)$  for prefill and  $1.10\times(1.14\times)$  for end-to-end inference, respectively. Detailed results are provided in Table 5 and Table 6.

<span id="page-19-1"></span>Table 5: Throughput Optimization Effect under TTFT SLO

| Throughput (Req/s) Dataset | SGLang | MoETuner | Sem-MoE | Speedup<br>v.s. SGLang | Speedup<br>v.s. MoETuner |
|----------------------------|--------|----------|---------|------------------------|--------------------------|
| Lmsys-Chat-1M              | 32.7   | 28.6     | 38.6    | 1.18x                  | 1.35x                    |
| MMLU                       | 32.2   | 30.1     | 38.6    | 1.20x                  | 1.28x                    |
| ShareGPT                   | 2.9    | 2.8      | 2.9     | 1.00x                  | 1.02x                    |
| Average                    | 22.6   | 20.5     | 26.7    | 1.12x                  | 1.22x                    |

<span id="page-19-2"></span>Table 6: Throughput Optimization Effect under E2E latency SLO

| SGLang | MoETuner            | Sem-MoE                                      | Speedup<br>v.s. SGLang                                                                                                                                          | Speedup<br>v.s. MoETuner                                                                                                                                                                                                              |
|--------|---------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 37.2   | 35.5                | 38.6                                         | 1.04x                                                                                                                                                           | 1.09x                                                                                                                                                                                                                                 |
| 37.9   | 35.3                | 38.6                                         | 1.02x                                                                                                                                                           | 1.10x                                                                                                                                                                                                                                 |
| 1.9    | 1.9                 | 2.4                                          | 1.23x                                                                                                                                                           | 1.23x                                                                                                                                                                                                                                 |
| 25.7   | 24.2                | 26.5                                         | 1.10x                                                                                                                                                           | 1.14x                                                                                                                                                                                                                                 |
|        | 37.2<br>37.9<br>1.9 | SGLang MoETuner  37.2 35.5 37.9 35.3 1.9 1.9 | SGLang         MoETuner         Sem-MoE           37.2         35.5         38.6           37.9         35.3         38.6           1.9         1.9         2.4 | SGLang         MoETuner         Sem-MoE         Speedup v.s. SGLang           37.2         35.5         38.6         1.04x           37.9         35.3         38.6         1.02x           1.9         1.9         2.4         1.23x |