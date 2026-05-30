# 5 ABLATION STUDY

#### 5.1 SHARING BY KV CACHE SIMILARITY OR DISSIMILARITY?

We adopt a counterintuitive sharing strategy by compressing during inference through sharing dissimilar KV cache, rather than the intuitive approach of sharing similar KV cache. This section will experimentally demonstrate that sharing based on KV cache dissimilarity performs better.

Specifically, we modify Algorithm [1](#page-3-0) by changing the descending order based on euclidean distance to ascending order, so that KV caches are sorted from high to low similarity while keeping all other steps unchanged. We then conduct experiments on the four models used in the main experiment.

Figure [6](#page-7-1) compares the models' PPL when sharing based on similarity versus dissimilarity. The results indicate that, for each model, at the given compression rates, the PPL of the similaritybased sharing strategy is significantly higher, often nearly twice as high or more than that of the dissimilarity-based strategy. Therefore, the method proposed in this paper is founded on sharing through dissimilarity.

<span id="page-8-0"></span>Table 3: Model performance at a 25% compression rate using Wikipedia and BookCorpus as calibration dataset. For each model, using a subset of the BookCorpus dataset as the calibration dataset has little impact on *KVSharer* compared to using a subset of the Wikipedia dataset.

| LLM           | Calibration Dataset | BoolQ | PIQA<br>HeSw<br>PPL    |
|---------------|---------------------|-------|------------------------|
| Llama2-7B     | Wikipedia           | 72.39 | 74.37<br>63.97<br>9.39 |
|               | BookCorpus          | 72.01 | 74.10<br>64.05<br>9.15 |
| Llama2-13B    | Wikipedia           | 78.20 | 76.71<br>72.40<br>9.11 |
|               | BookCorpus          | 78.34 | 76.81<br>72.18<br>9.17 |
| InternLM2-7B  | Wikipedia           | 80.37 | 79.49<br>73.22<br>9.78 |
|               | BookCorpus          | 80.37 | 79.49<br>73.22<br>9.78 |
| InternLM2-20B | Wikipedia           | 80.61 | 80.96<br>75.84<br>7.05 |
|               | BookCorpus          | 81.08 | 80.53<br>75.46<br>7.01 |

### 5.2 EFFECT OF DIFFERENT CALIBRATION DATASETS

To investigate the impact of different calibration datasets, we replace the Wikipedia dataset with a randomly selected, equally sized subset of the BookCorpus dataset [\(Kiros et al., 2015\)](#page-11-17). We set the compression rate to 25% and rerun the experiments, keeping all other settings unchanged.

The results are shown in Table [3.](#page-8-0) The findings indicate that using the two different calibration datasets has almost no impact on model performance, with only minimal differences in performance across several benchmarks and PPL. For InternLM2-7B, the same sharing strategy is identified with both datasets, further indicating that *KVSharer* is not sensitive to the calibration dataset. We also conduct an ablation study on calibration dataset size in Appendix [A.3,](#page-14-3) Table [8,](#page-14-4) and find that the size has little impact.

#### 5.3 RANDOM SHARING V.S. KVSHARER

*KVSharer* compresses KV cache through a highly counterintuitive strategy of sharing dissimilar KV caches, which leads us to explore whether KV caches can be shared arbitrarily to achieve compression effect. Thus, we conduct comparative experiments. Specifically, we randomly select some layers' KV caches to replace others, set the compression rate to 25%, keep other settings unchanged, and evaluate the models' performance on multiple benchmarks and their PPL. We repeat the experiments three times and take the average of the results.

<span id="page-9-0"></span>Table 4: Model performance using KVSharer and random sharing strategies at a 25% compression rate.

| LLM           | Strategy | BoolQ | PIQA  | HeSw  | PPL   |  |
|---------------|----------|-------|-------|-------|-------|--|
| Llama2-7B     | KVSharer | 72.39 | 74.37 | 63.97 | 9.39  |  |
|               | Random   | 50.67 | 59.15 | 44.97 | 21.29 |  |
| Llama2-13B    | KVSharer | 78.20 | 76.71 | 72.40 | 9.11  |  |
|               | Random   | 40.69 | 51.21 | 42.99 | 51.41 |  |
| InternLM2-7B  | KVSharer | 80.37 | 79.49 | 73.22 | 9.78  |  |
|               | Random   | 63.33 | 61.73 | 58.13 | 13.58 |  |
| InternLM2-20B | KVSharer | 80.61 | 80.96 | 75.84 | 7.05  |  |
|               | Random   | 61.43 | 64.11 | 58.39 | 18.50 |  |

We present the results in Table [4.](#page-9-0) The results indicate that, compared to KVSharer's PPL of under 10, the randomly selected sharing strategy causes a significant increase in the model's PPL, reaching as high as 50 for Llama2-13B.

Across different benchmarks, the randomly selected strategy also reduces the model's performance, typically by about 30%. This set of experiments demonstrates that a randomly selected sharing strategy cannot maintain model performance, while *KVSharer*, with its search-based approach, can find a more effective sharing strategy.

However, the results also contain some surprising findings. In the case of randomly sharing the KV cache, the model's performance does not drop to zero, and the PPL does not explode to over a hundred. This suggests that there may be redundancy in the KV cache, or that the impact of the self-attention keys and values on the subsequent hidden-state calculations is not as significant as we initially thought. We will continue to explore this in the future.

<span id="page-9-1"></span>Table 5: Comparison of performance on different benchmarks and PPL between Chat and Base versions of the models at the same compression rate.

| LLM               |                  |       | Llama2-7B   |      |             |      | Llama2-13B  |      | InternLM2-7B |       |             |      | InternLM2-20B |      |             |      |
|-------------------|------------------|-------|-------------|------|-------------|------|-------------|------|--------------|-------|-------------|------|---------------|------|-------------|------|
| Version           |                  | Base  |             | Chat |             | Base |             | Chat |              | Base  |             | Chat |               | Base |             | Chat |
| Layer             | 32               | 24    | 32          | 24   | 40          | 30   | 40          | 30   | 32           | 24    | 32          | 24   | 48            | 36   | 48          | 36   |
| BoolQ 70.67 69.27 |                  |       | 70.67 72.39 |      | 71.50 65.63 |      | 81.56 78.20 |      | 71.28 70.40  |       | 83.21 80.37 |      | 65.44 54.04   |      | 81.71 80.61 |      |
|                   | PIQA 78.18 76.66 |       | 78.18 74.37 |      | 79.71 75.35 |      | 78.24 76.71 |      | 80.30 79.00  |       | 79.60 79.49 |      | 82.10 81.23   |      | 81.39 80.96 |      |
| HeSW              | 71.28 69.43      |       | 71.35 63.97 |      | 74.83 67.81 |      | 75.41 72.40 |      | 73.43 72.46  |       | 73.30 73.22 |      | 75.46 74.99   |      | 76.57 75.84 |      |
| PPL               | 5.25             | 11.13 | 6.62        | 9.39 | 4.32        | 7.73 | 5.99        | 9.11 | 7.27         | 10.59 | 6.99        | 9.78 | 5.13          | 7.38 | 5.67        | 7.05 |

### 5.4 EFFECT OF KVSHARER ON DIFFERENT MODEL VERSIONS

Since the models used in our main experiments are all Chat versions, we also want to explore whether *KVSharer* can be effective on the Base versions of the models. We conduct comparative experiments using the Base versions of different models, setting the compression rate at 25%, and also comparing the results with those of the full KV cache.

We show the results in the Table [5.](#page-9-1) As shown in the result, *KVSharer* also works for Base models, as it similarly maintains a minor impact on both various tasks and PPL, comparable to its effect on the Chat model. This also demonstrates that *KVSharer* has strong generalizability.

