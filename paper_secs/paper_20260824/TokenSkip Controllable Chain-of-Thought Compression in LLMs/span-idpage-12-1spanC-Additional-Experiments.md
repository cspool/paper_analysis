# <span id="page-12-1"></span>C Additional Experiments

## C.1 Out-of-domain Evaluation

To assess the generalizability of TokenSkip beyond the training domain data, we conducted an additional out-of-domain evaluation. Specifically, we fine-tuned LLaMA-3.1-8B-Instruct on the MATH training data and evaluated TokenSkip on both the in-domain MATH-500 and two outof-domain benchmarks, GSM8K and MMLU-STEM [\(Hendrycks et al.,](#page-9-19) [2021a\)](#page-9-19). MMLU-STEM includes a diverse set of STEM subjects from the full MMLU dataset.

The results in Table [4](#page-12-3) suggest that TokenSkip maintains strong generalizability on out-of-domain scenarios. The model adheres closely to specified compression ratios while preserving accuracy. Notably, on the MMLU-STEM test set, TokenSkip exhibits comparable performance to the original LLM with 40% token trimming. Even at a compression ratio of 0.5, the model maintains strong reasoning capabilities, with only 0.4% absolute performance degradation.

