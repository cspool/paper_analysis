# <span id="page-10-1"></span>A.1. Case Study: Detailed Analysis of MxMoE W5A5 Scheme

We conduct in-depth analysis of MxMoE's W5A5 mechanism and its accuracy advantages. As shown in Tab. [4,](#page-10-2) reducing activation bitwidth from 5 to 4 bits causes significant model quality degradation, demonstrating critical quantization sensitivity around 4-bit activation precision. This can be attribute to the massive outlier observed in the input activation of Down proj [\(Sun et al.,](#page-9-21) [2024\)](#page-9-21), where heavy-tailed activation distributions require higher precision preservation. MxMoE dynamically identifies these quantization-sensitive components (whose bitwidth reduction causes substantial model quality degradation) and allocates elevated bitwidth accordingly. The heterogeneous bitwidth allocation strategy for Qwen1.5-MoE is visualized in Tab. [7.](#page-11-0)

To further validate mixed-precision benefits, we compare against QuaRot [\(Ashkboos et al.,](#page-8-11) [2024\)](#page-8-11) in Tab. [5.](#page-10-2) While uniform bitwidth scaling shows similar perplexity improvement trends in QuaRot, practical deployment remains constrained by the capacity of model hardware, on which 5-bits operation is not supported. In contrast, MxMoE achieves better accuracy while maintaining hardware compatibility through mixed-precision allocation that leverages existing low-bitwidth arithmetic units.

<span id="page-10-2"></span>Table 4. WikiText2 perplexity under different weight-activation bitwidth (RTN-token/channel quantization). Column: activation bitwidth, row: weight bitwidth.

| #Bits | 4         | 5      | 6      | 7     | 8     |
|-------|-----------|--------|--------|-------|-------|
| 4     | 68079.039 | 41.433 | 11.298 | 9.406 | 8.068 |
| 5     | 12305.585 | 38.707 | 9.715  | 8.169 | 7.335 |
| 6     | 14251.822 | 26.297 | 9.216  | 8.196 | 7.204 |
| 7     | 18151.474 | 34.775 | 9.747  | 8.182 | 7.325 |
| 8     | 19091.917 | 38.990 | 9.525  | 8.260 | 7.278 |

Table 5. WikiText2 perplexity of Qwen1.5-MoE under different quantization bitwidth settings. Both MxMoE and QuaRot employ RTN weight quantization.

| Setting      | w4a4   | w5a5  | w6a6  | w7a7  | w8a8  |
|--------------|--------|-------|-------|-------|-------|
| QuaRot (Uni) | 36.385 | 7.998 | 6.990 | 6.852 | 6.814 |
| MxMoE (Mix)  | -      | 7.160 | -     | -     | -     |

