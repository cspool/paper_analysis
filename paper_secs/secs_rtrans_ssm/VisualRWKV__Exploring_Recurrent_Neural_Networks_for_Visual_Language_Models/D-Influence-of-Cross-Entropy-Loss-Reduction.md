# D Influence of Cross-Entropy Loss Reduction

In the experiment, we found that using zero1 for training with a batch size of 1 and gradient accumulation of 16; and using zero2 for training with a batch size of 1 and gradient accumulation of 1; These two settings are not equivalent, with different losses, leading to significantly disparate outcomes for the final model. Therefore, we conducted an in-depth analysis and study.

For illustrative purposes, consider a simple thought experiment with four samples: the first sample consists of 100 tokens, the second of 200 tokens, the third of 300 tokens, and the fourth of 400 tokens. Consequently, the total length sums up to 1000 tokens. When these samples are batched together(batch size of 4 and gradient accumulation of 1), each token is normalized by a factor of 1000. We refer this process as batch-level reduction. Please note that the batch-level reduction is highly dependent on the

<span id="page-13-0"></span><sup>1</sup> <https://huggingface.co/BlinkDL/rwkv-5-world>

<span id="page-13-1"></span><sup>2</sup> <https://huggingface.co/BlinkDL/rwkv-6-world>

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 5: Single-Stage Training vs. Two-Stage Training. We conducted a comparative analysis between the twostage training and single-stage training, with the latter omitting the vision-language alignment phase. Our findings reveal that the single-stage training yields inferior performance outcomes. This suggests that the vision-language alignment, integral to the two-stage training, significantly contributes to enhanced performance.

batch size. As the batch size varies, the total batch length by which each token's loss is divided can differ significantly.

An alternative approach, termed sample-level reduction, normalizes each sample by its length. This sample-level reduction is independent of the batch size and introduces a different loss re-weighting compared to batch-level reduction. Continuing our thought experiment, we apply sample-level reduction with a batch size of 1 and gradient accumulation of 4. The first sample undergoes a sequential division by 100 (its length) and then by 4, culminating in an effective division by 400. The second sample is adjusted by a factor of 800, the third by 1200, and the fourth by 1600. This scaling mechanism inherently leads to a larger loss for shorter texts and a smaller loss for longer texts compared to batch-level reduction.

Our findings underscore the importance of accurate reduction and loss re-weighting for the performance of certain downstream tasks. Table [7](#page-14-1) presents a comparative analysis between our model's performance under batch-level and sample-level reduction. Notably, we have found that using sample-level reduction yields better results on 5 benchmarks. In contrast, batch-level reduction performs better on 2 benchmarks. Among them, sample-level reduction significantly outperforms on the ScienceQA benchmark. On the MME benchmark, batch-level reduction takes the lead. After an in-depth investigation, we discovered that the score in the Celebrity domain within MME has significantly improved, while other domains show varying degrees of success.

<span id="page-14-1"></span>

| Reduction    | VQAv2 | ScienceQA | TextVQA | GQA    | VizWiz | MME     | POPE |
|--------------|-------|-----------|---------|--------|--------|---------|------|
| Sample-Level | 67.54 | 56.62%    | 42.18%  | 52.82% | 26.03  | 1111.66 | 0.82 |
| Batch-Level  | 66.85 | 47.94%    | 41.79%  | 52.56% | 27.02  | 1173.42 | 0.79 |

Table 7: Study comparing batch-level reduction and sample-level reduction across 7 Visual Language benchmarks. Loss reduction method is crucial for performance. Model used here is VisualRWKV 1.6B.

Furthermore, we conducted a comparison of the textual abilities resulting from sample-level and batch-level reduction, as shown in Table [8.](#page-15-1) It was observed that sample-level training exhibited superior English capabilities, whereas the batch-level training demonstrated enhanced multilingual abilities. This is due to the higher loss weight assigned to the multilingual long texts of ShareGPT4 data in the batch-level training.

In general, we consider sample-level reduction to be the better approach. On one hand, the performance is better, whether in visual-linguistic abilities or pure textual capabilities. On the other hand, sample-level reduction is invariant to batch size. When the sample-level reduction-based training protocol is migrated across various GPUs, it does not suffer from inconsistencies due to batch size variations, which could <span id="page-15-1"></span>otherwise lead to divergent outcomes.

| Reduction    | LAMBADA(ppl) | English(avg%) | MultiLang(avg%) |
|--------------|--------------|---------------|-----------------|
| Batch-Level  | 4.499        | 59.89         | 59.97           |
| Sample-Level | 4.145        | 61.01         | 59.84           |

Table 8: Study comparing batch-level reduction and sample-level reduction across language benchmarks. Model used here is VisualRWKV 1.6B.

