# <span id="page-16-2"></span>A.7 EVALUATION SETTINGS

Perplexity Evaluation. We use the test split in GPTQ to measure the perplexity on WikiText2 and C4. The sequence length is 8192 for LLaMA-3 and Qwen3 models, and 4096 for LLaMA-2 models. Note that the perplexity results of Qwen3 family in Table [1](#page-7-0) are from the pre-trained "Base" models (*e.g*., Qwen/Qwen3-8B-Base from Hugging Face), not the models after post-training (*e.g*., Qwen/Qwen3-8B).

Reasoning Task Evaluation. We follow [Liu et al.](#page-10-5) [\(2025a\)](#page-10-5) and use Lighteval 0.8.1 [\(Habib et al.,](#page-10-16) [2023\)](#page-10-16) with vLLM 0.10.1 [\(Kwon et al.,](#page-10-9) [2023\)](#page-10-9) to evaluate the reasoning tasks in Table [2.](#page-7-1) We evaluate GPQA Diamond, AIME-24, and AIME-25 with three different seeds (42, 0, 1) and report the average accuracy to reduce variations, and evaluate MMLU-Pro with one seed (42).

Non-Reasoning Task Evaluation. We use the Language Model Evaluation Harness library (lm eval) version 0.4.9.1 [\(Gao et al.,](#page-9-14) [2024\)](#page-9-14) to evaluate the tasks in Table [3](#page-8-1) with the default settings of the library and a batch size of 32.

