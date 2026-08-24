# <span id="page-8-0"></span>5.2. Combination with Different Eviction Policies

Our method could either work alone or be integrated with any token eviction policy. In Table [6,](#page-8-3) to extend the maximum context length of Llama-2-7B to 8192 tokens, we showed our core idea of token merging via convolutional heads (1) works well alone; (2) could be combined with StreamingLLM [\(Xiao et al.,](#page-11-3) [2023\)](#page-11-3), by additionally storing the initial tokens, as known as "attention sink"; and (3) could be further augmented by heavy hitters[\(Zhang et al.,](#page-11-2) [2023b\)](#page-11-2), the "important tokens" identified by accumulated attention scores. All variants of our methods show superior performance to solely using the previous token eviction method [\(Zhang et al.,](#page-11-2) [2023b\)](#page-11-2).

### <span id="page-8-1"></span>5.3. Effectiveness under Different Kernel Sizes

Longer convolutional kernels may also present challenges in optimization. With the Llama-2-7B model [\(Touvron et al.,](#page-11-14) [2023\)](#page-11-14), we extend the context length to 8192, employing kernel sizes ranging from 3 to 21. We evaluate the finetuned model on Proof-Pile-2 [\(Azerbayev et al.,](#page-9-22) [2023\)](#page-9-22), using a context length of 8192. The results are summarized in Table [7.](#page-8-4) We observe stable performance for most size choices, although there are degradations with extremely small kernel sizes. That suggests LoCoCo can work well with moderately

<span id="page-8-2"></span>sized convolutions, without visible optimization hurdles.

<span id="page-8-4"></span>

| Table 7. Ablation with different kernel sizes. |      |      |      |      |      |      |      |      |
|------------------------------------------------|------|------|------|------|------|------|------|------|
| Kernel                                         | 3    | 7    | 17   | 21   | 31   | 41   | 51   | 61   |
| PPL                                            | 3.68 | 3.57 | 3.53 | 3.53 | 3.54 | 3.53 | 3.57 | 3.58 |

