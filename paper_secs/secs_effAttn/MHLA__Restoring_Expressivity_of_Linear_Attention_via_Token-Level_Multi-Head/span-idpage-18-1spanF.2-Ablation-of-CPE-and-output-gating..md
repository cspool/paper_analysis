# <span id="page-18-1"></span>**F.2 Ablation of CPE and output gating.**

We conducted a detailed analysis of the effects of CPE and Output Gating when combined with MHLA in the DiT-S model, as shown in Tab. [12.](#page-18-6) Our findings show that, in smaller models, CPE and Output Gating serve as orthogonal optimizations of MHLA, effectively enhancing the expressive ability when the model size is insufficient. However, our experiments in Tab. [3a](#page-8-0) indicate that the performance gains from CPE and Output Gating diminish as the model size increases. In the DiT-XL model, adding CPE alone actually leads to a performance decrease. In contrast, MHLA consistently provides significant improvements in expressivity, regardless of model size.

<span id="page-18-6"></span>**Table 12** Ablation study of MHLA with CPE and output gating.

| Setting            | FID  |
|--------------------|------|
| Linear Attention   | 89.7 |
| MHLA w/ None       | 76.4 |
| MHLA w/ CPE        | 64.0 |
| MHLA w/ Gating     | 68.5 |
| MHLA w/ CPE+Gating | 59.8 |

### <span id="page-18-4"></span>**F.3 Classification results on Higher Resolutions**

We further conducted additional experiments at resolutions of 384×384 and 512×512, using the DeiT-T model to verify the effectiveness of MHLA on high-resolution classification tasks. Results are shown in Tab. [13.](#page-18-7)

