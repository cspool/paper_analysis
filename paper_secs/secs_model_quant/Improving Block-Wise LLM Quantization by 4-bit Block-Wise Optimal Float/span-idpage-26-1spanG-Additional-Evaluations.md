# <span id="page-26-1"></span>G Additional Evaluations

In Tab. [9,](#page-27-1) the quantization error and perplexity results for the Llama-3.1 8B and the Qwen-2.5 7B models are shown. Note that Tab. [9](#page-27-1) corresponds to Tab. [1,](#page-6-0) the latter just showing larger models. Similar to the larger models, we observe that our BOF4(-S) quantizers perform at least as well and usually better than the baseline methods in the quantization error metric (MAE or MSE) for which the particular codebook is optimized. Furthermore, the BOF4-S quantizers using our signed absmax normalization significantly improve all metrics over BOF4 with absolute absmax normalization. When additionally applying OPQ to the BOF4-S quantizers, performance in all metrics improves further. *The lowest errors are achieved by BOF4-S +OPQ, using the codebook optimized for the particular error metric.* Interestingly, for the Qwen-2.5 3B model, our MAE-optimized methods

<span id="page-27-1"></span>Table 9: Quantization error (MAE and MSE) and perplexity (PPL) on WikiText-2 of quantization methods applied to the network weights of two 3B regime LLMs with block size I = 64. Best result in each column in bold, second best underlined.

|              |               | Llama-3.1 3B  |       | Qwen-2.5 3B   |               |       |  |
|--------------|---------------|---------------|-------|---------------|---------------|-------|--|
|              | MAE ↓<br>1e−3 | MSE ↓<br>1e−6 | PPL ↓ | MAE ↓<br>1e−3 | MSE ↓<br>1e−6 | PPL ↓ |  |
| NF4          | 1.399         | 3.333         | 10.72 | 1.822         | 5.722         | 12.13 |  |
| AF4          | 1.441         | 3.588         | 10.71 | 1.862         | 6.118         | 13.48 |  |
| BOF4 (MAE)   | 1.399         | 3.302         | 10.72 | 1.821         | 5.670         | 12.16 |  |
| BOF4 (MSE)   | 1.424         | 3.191         | 10.73 | 1.862         | 5.526         | 12.46 |  |
| BOF4-S (MAE) | 1.341         | 3.071         | 10.68 | 1.746         | 5.274         | 12.07 |  |
| + OPQ        | 1.316         | 2.971         | 10.63 | 1.689         | 5.026         | 12.05 |  |
| BOF4-S (MSE) | 1.367         | 2.936         | 10.66 | 1.788         | 5.087         | 12.41 |  |
| + OPQ        | 1.336         | 2.791         | 10.64 | 1.719         | 4.739         | 12.36 |  |

generally achieve better perplexity, suggesting that the target error metric for optimization, which leads to the best performance, may vary depending on the LLM.

Fig. [12](#page-28-0) shows perplexity results for BOF4 on the WikiText-2 and LAMBADA datasets. Note that Fig. [12](#page-28-0) corresponds to Fig. [3,](#page-7-1) the latter reporting on BOF4-S, however. *We observe that for most block sizes* I*, the perplexity of BOF4 optimized w.r.t. MAE or MSE is similar to that of the best-performing baseline method. Adding OPQ significantly reduces perplexity, particularly when used with MSE optimized codebooks and at large block sizes.*

Tab. [10](#page-28-1) displays additional perplexity and accuracy measurements for the larger Llama-3 8B, Qwen-2.5 7B, and the tiny Qwen-2.5 0.5B. Note that Tab. [10](#page-28-1) corresponds to Tab. [2,](#page-8-0) the latter reporting on small 3B models. Our best BOF4 quantization method consistently outperforms the baseline methods, NF4 and AF4, in terms of perplexity on WikiText-2 and LAMBADA, except when applied to the Qwen-2.5 7B model, where NF4 achieves a surprisingly low perplexity on LAMBADA—surpassing even the performance of the unquantized BF16 model. Note, however, that for the Qwen-2.5 7B model, each of our four proposed BOF4(-S) methods performs as well as or better than NF4 in the NLP benchmarks' normalized average accuracy (NAV) metric. Overall normalized average accuracy (NAV) results from the language modeling benchmarks do not indicate a single quantizer or approach that consistently performs best.

The OPQ variant proves particularly effective for the smaller Qwen-2.5 0.5B model, where it significantly improves perplexity over the respective quantizer without OPQ and both baselines NF4 and AF4.

