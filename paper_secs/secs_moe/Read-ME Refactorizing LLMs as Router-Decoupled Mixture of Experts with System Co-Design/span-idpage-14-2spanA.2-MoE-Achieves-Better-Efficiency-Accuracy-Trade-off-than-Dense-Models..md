# <span id="page-14-2"></span>A.2 MoE Achieves Better Efficiency-Accuracy Trade-off than Dense Models.

Prior compression-based works [\[42,](#page-12-9) [43,](#page-12-10) [44,](#page-12-11) [45,](#page-12-12) [46\]](#page-12-13) focus on converting a large *dense* pre-trained model into a smaller *dense* model. However, we argue that a smaller *MoE* model (i.e. the MoE model with the smaller number of activation parameters) is a better target architecture. To ensure a fair comparison, we (1) derive a small dense model with 4.7B parameters, matching the size of a single expert network, using the same amount of data, and (2) fine-tune the obtained dense model for an equivalent number of steps. As shown in Table [5,](#page-14-4) refactorizing the pre-trained model into an MoE structure, rather than a smaller dense variant, leads to significant performance improvement. The models are evaluated based on performance on the MMLU [\[16\]](#page-10-15), and perplexity across seven data domains included in RedPajama [\[35\]](#page-12-2).

<span id="page-14-4"></span>Table 5: We compare the *Read-ME*performance with dense model, and report the MMLU performance and perplexity on 7 data domains. By adopting an MoE as the target structure instead of dense model, our model achieve significantly better overall performance.

| Evaluation<br>Arxiv | Books<br>C4 |       | Common Crawl | Github | StackExchange | Wikipedia<br>MMLU |  |  |
|---------------------|-------------|-------|--------------|--------|---------------|-------------------|--|--|
| Dense<br>5.63       | 1.94        | 11.78 | 9.68         | 3.75   | 13.42         | 6.24<br>27.1%     |  |  |
| Read-ME<br>4.18     | 1.31        | 10.57 | 7.72         | 2.39   | 12.52         | 3.94<br>38.9%     |  |  |

#### A.3 *Read-ME* Remains Effective without Prior Knowledge of the Training Domain

We additionally use the Mistral [\[65\]](#page-13-14) model as the pre-trained dense model, and convert it to the MoE structure, with the proposed method. The task is challenging because we do not have prior knowledge on the Mistral original training data, and our experiment in Table [6](#page-14-5) shows that our method remains effective without the prior knowledge of the original training domain.

Table 6: Ablation study on Mistral [\[65\]](#page-13-14) pre-trained model.

<span id="page-14-5"></span>

| Method                     | Pre-trained<br>Domain    | Fine-tune<br>Domain | #Param           | MMLU           | Hell.          | Wino.          | ARC-E          | ARC-C          | LogiQA         | CoQA<br>avg.                     |
|----------------------------|--------------------------|---------------------|------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------------------------|
| Read-ME-Llama-2<br>Llama-2 | Red-pajama<br>Red-pajama | Red-pajama<br>-     | 4.7B-17B<br>6.9B | 38.9%<br>45.3% | 68.5%<br>78.6% | 67.7%<br>69.3% | 66.6%<br>76.4% | 42.3%<br>53.0% | 29.7%<br>31.0% | 74.8%<br>55.5%<br>75.9%<br>61.4% |
| Read-ME-Mistral<br>Mistral | N/A<br>N/A               | Red-pajama<br>-     | 4.7B-17B<br>6.9B | 39.2%<br>62.1% | 79.1%<br>84.5% | 68.2%<br>79.3% | 77.1%<br>82.7% | 49.3%<br>63.7% | 30.9%<br>33.5% | 76.2%<br>60.0%<br>80.3%<br>69.4% |

#### A.4 Computational Cost of Auto-regressive Router

For a detailed cost analysis of auto-regressive router that we introduced, we added: (1) FLOPs comparison, (2) latency, and (3) latency breakdown with a larger batch size (high-throughput scenarios) of a Traditional Router (TR) and an Autoregressive Router (AR). To focus solely on the router's impact on latency, we controlled other variables (e.g., the number of activated parameters) to be the same.

Note that the computational cost of both the traditional router and the autoregressive router is theoretically *linear to batch size*. Therefore, when the batch size is high (in high-throughput

Table 7: Flops comparison between Traditional router and Auto-regressive router

|              | Traditional Router | Auto-regressive Router |
|--------------|--------------------|------------------------|
| Flops/sample | 4.7 KFLOPs         | 3 KFLOPs               |

Table 8: Latency [ms] comparison between Traditional router and Auto-regressive router

|            | bsz=5 | bsz=5 | bsz=10 | bsz=10 | bsz=20 | bsz=20 | bsz=30 | bsz=30 |
|------------|-------|-------|--------|--------|--------|--------|--------|--------|
|            | TR    | AR    | TR     | AR     | TR     | AR     | TR     | AR     |
| Router     | 1.76  | 0.61  | 1.80   | 0.61   | 1.78   | 0.61   | 1.93   | 0.61   |
| Attention  | 18.13 | 18.18 | 18.28  | 18.13  | 18.49  | 18.36  | 19.59  | 19.66  |
| Expert/MLP | 22.43 | 21.75 | 24.59  | 22.53  | 24.97  | 22.99  | 30.17  | 28.31  |
| Sum        | 42.31 | 40.55 | 44.67  | 41.27  | 45.23  | 41.96  | 51.69  | 48.59  |

Table 9: Latency breakdown comparison between Traditional router and Auto-regressive router

<span id="page-15-0"></span>

|            | bsz=5   | bsz=5   | bsz=10  | bsz=10  | bsz=20  | bsz=20  | bsz=30  | bsz=30  |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|
|            | TR      | AR      | TR      | AR      | TR      | AR      | TR      | AR      |
| Router     | 4.15%   | 1.50%   | 4.02%   | 1.48%   | 3.93%   | 1.46%   | 3.74%   | 1.26%   |
| Attention  | 42.85%  | 44.85%  | 40.92%  | 43.92%  | 40.87%  | 43.75%  | 37.90%  | 40.47%  |
| Expert/MLP | 53.01%  | 53.65%  | 55.06%  | 54.59%  | 55.20%  | 54.80%  | 58.36%  | 58.27%  |
| Sum        | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

scenarios), the cost increases linearly. In both cases, the computation can be parallelized, so this remains negligible in end-to-end latency even in high-throughput scenarios. In fact, we would like to clarify that the bottleneck in high-throughput scenarios is actually the expert layers, as seen in Table [9](#page-15-0) – Expert/MLP row. This issue can be addressed by the methods discussed in Section 4. *Traditional layerwise routers do not allow for efficient system design*, which underscores the need for a careful co-design of routers.

