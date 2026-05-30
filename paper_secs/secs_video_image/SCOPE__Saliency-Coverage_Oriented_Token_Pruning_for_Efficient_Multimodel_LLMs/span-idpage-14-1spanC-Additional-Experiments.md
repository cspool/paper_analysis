# <span id="page-14-1"></span>C Additional Experiments

In the main paper, we present experiments on LLaVA 1.5 7B and LLaVA-Next 7B. To further demonstrate the generalizability of our method across model scales, we provide additional results on LLaVA 1.5 13B and LLaVA-Next 13B. We also provide the results on more MLLMs such as Qwen2-VL and more OCR-related benchmarks.

#### <span id="page-14-0"></span>C.1 Results on LLaVA 1.5 13B

As shown in Table [6,](#page-14-2) our method consistently outperforms VisionZip [\[43\]](#page-12-6) across all token budgets. With 192 tokens, our approach achieves 100.2% of the upper bound's average performance, slightly higher than VisionZip [\[43\]](#page-12-6) (98.7%). The advantage becomes more evident as the token count decreases: at 64 tokens, our method retains 96.9% performance, compared to VisionZip's 93.7%. Notably, on benchmarks like MMVet [\[45\]](#page-12-12) and POPE [\[22\]](#page-11-13), our method even surpasses the original model's performance. These results demonstrate that our joint saliency-coverage strategy better preserves essential information under aggressive token pruning.

