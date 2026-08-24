# A.3 Are the True/False claims distinguishable without the book texts?

We ask the question of whether distinguishing between True and False claims is inherently too easy. If so, then the high performance of the fine-tuned models may be attributed merely to their ability to detect formatting cues rather than actually reasoning. To investigate this, we prompt both baseline and fine-tuned models to verify claims without providing any book texts or metadata. Our hypothesis is that if a model performs better than random chance under these conditions, then the claims are likely too easily distinguishable based on their formatting alone.

| Models           | Before SFT | After SFT |
|------------------|------------|-----------|
| ProLong-Instruct | 0.0%       | 25.2%     |
| Llama-Instruct   | 20.2%      | 13.8%     |
| Qwen-Instruct    | 21.7%      | 22.9%     |

<span id="page-16-4"></span>Table 7: Accuracy on CLIPPER's test set (no book text or metadata provided).

<span id="page-16-2"></span>As shown in Table 7, even after fine-tuning, the models perform only marginally above random guessing. We conclude that, without the contextual information from the book text, True/False claims are not easily distinguishable.

