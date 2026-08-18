# *B. Attention-Score Quantization*

In order to fully exploit the hardware efficiency of KV-cache quantization, it is essential to quantize additional operands, e.g., attention-scores, within the self-attention module. Although attention-scores have little impact on the overall memory footprint (see Section [III-A\)](#page-2-1), they play a critical role in the computation flow. Consider a low-precision PCU supporting 8 bit multiplication, if attention-scores remain in FP16, the PCU cannot be fully utilized to accelerate self-attention. Instead, the quantized value cache must be transferred to NPU and rely on the FP16 compute units to perform multiplication with attention-scores, thus diminishing the bandwidth advantages of PIM for LLM decoding. A straightforward solution is to quantize attention-scores with INT8 [\[95\]](#page-15-9), but this introduces two drawbacks. First, INT8 attention-scores cause noticeable perplexity degradation as shown in Table [II.](#page-5-0) Second, as we will demonstrate in Section [IV-D,](#page-5-1) INT8 can lead to significant accuracy loss for activation quantization. Since attention-scores behave the same as activations during the MAC operation on hardware, it is desirable to identify a good numerical format that balances quantization accuracy and hardware complexity.

<span id="page-5-2"></span>TABLE III. Wikitext-2 and C4 perplexity (↓) of different weightactivation quantization methods. For activations, we compare INT8 based SmoothQuant (SQ) and direct FP8-E4M3 quantization. For weights, we examine 4-bit BitMoD. The context length is 4K.

|        | Precision  | 2-7B |      | 2-13B |      | 3.1-8B |      | 3.2-3B |       |
|--------|------------|------|------|-------|------|--------|------|--------|-------|
| Weight | Activation | Wiki | C4   | Wiki  | C4   | Wiki   | C4   | Wiki   | C4    |
| 16     | 16         | 5.12 | 6.63 | 4.57  | 6.05 | 5.84   | 8.43 | 7.28   | 10.01 |
| 16     | INT8-SQ    | 5.15 | 6.67 | 4.61  | 6.09 | 5.92   | 8.54 | 7.34   | 10.08 |
| 16     | FP8-E4M3   | 5.12 | 6.63 | 4.58  | 6.05 | 5.85   | 8.46 | 7.31   | 10.03 |
| 4      | INT8-SQ    | 5.37 | 6.95 | 4.76  | 6.27 | 6.36   | 9.12 | 7.87   | 10.73 |
| 4      | FP8-E4M3   | 5.24 | 6.78 | 4.66  | 6.15 | 6.16   | 8.90 | 7.64   | 10.48 |

To address this, we propose an unsigned 8-bit floatingpoint format, FP8-S0E4M4, that contains a 4-bit mantissa and a 4-bit exponent with an exponent bias of −15. There are two insights that drive our design choice. First, because the attention-scores are produced after softmax, its numerical range always lies between 0 and 1, eliminating the need for a sign bit in its encoding. Second, recall FP16 has a 5-bit exponent with a bias of −15, which provides an exponent range of [−14, 15]. However, since attention-scores are always less than 1, there is no need to use positive exponent values, leading to an effective exponent range of [−14, −1] with 14 distinct values. Thus, a 4-bit exponent is sufficient to represent the attention-score range, leaving 4 bits to store mantissa with better numerical fidelity. Table [II](#page-5-0) reflects that FP8-S0E4M4 outperforms both INT8 and FP8-E4M3 for attention-score quantization, and achieves near-lossless model performance.

