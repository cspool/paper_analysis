# <span id="page-11-4"></span>7.3 Image Quality

Table 2 compares the image generation quality of MoDM with various baselines on the DiffusionDB and MJHQ-30k

<span id="page-11-2"></span>**Table 2.** Image quality on DiffusionDB and MJHQ-30k (Vanilla model: SD3.5L). Higher is better for CLIP, IS, and Pick: lower is better for FID.

|                  | DiffusionDB   |       |       | MJHQ-30k |               |       |       |        |
|------------------|---------------|-------|-------|----------|---------------|-------|-------|--------|
| Baseline         | <b>CLIP</b> ↑ | FID ↓ | IS ↑  | Pick ↑   | <b>CLIP</b> ↑ | FID ↓ | IS ↑  | Pick ↑ |
| Vanilla (SD3.5L) | 28.55         | 6.29  | 15.52 | 21.44    | 28.77         | 5.16  | 25.84 | 21.67  |
| SDXL             | 29.30         | 16.29 | 16.90 | 21.45    | 29.66         | 12.67 | 25.82 | 21.55  |
| SD3.5L-Turbo     | 27.23         | 14.63 | 15.38 | 21.45    | 27.84         | 10.68 | 23.70 | 21.59  |
| SANA             | 28.08         | 19.96 | 12.20 | 20.78    | 28.83         | 16.31 | 21.90 | 21.32  |
| Nirvana          | 28.02         | 9.01  | 15.38 | 21.28    | 28.57         | 5.37  | 25.04 | 21.59  |
| PINECONE         | 25.98         | 14.18 | 15.09 | 20.80    | 27.20         | 6.80  | 25.99 | 21.27  |
| MoDM-SDXL        | 28.70         | 11.85 | 15.27 | 21.00    | 28.79         | 6.87  | 25.46 | 21.33  |
| MoDM-SANA        | 28.01         | 16.96 | 12.67 | 20.79    | 28.82         | 9.96  | 22.25 | 21.28  |

<span id="page-11-3"></span>**Table 3.** Image quality on DiffusionDB (Vanilla model: FLUX). Higher is better for CLIP, IS, and Pick; lower is better for FID.

| Baseline       | CLIP ↑ | FID ↓ | IS ↑  | Pick ↑ |
|----------------|--------|-------|-------|--------|
| Vanilla (FLUX) | 26.82  | 6.02  | 16.69 | 21.29  |
| SDXL           | 29.30  | 17.60 | 16.90 | 21.45  |
| SD3.5L-Turbo   | 27.23  | 15.11 | 15.38 | 21.45  |
| SANA           | 28.08  | 24.37 | 12.20 | 20.78  |
| Nirvana        | 26.01  | 9.07  | 15.44 | 21.06  |
| PINECONE       | 24.37  | 19.41 | 16.08 | 20.63  |
| MoDM-SDXL      | 28.41  | 10.74 | 15.61 | 21.13  |
| MoDM-SANA      | 27.59  | 16.84 | 12.70 | 20.84  |

datasets, using SD3.5L as the vanilla large model. Across both datasets, MoDM achieves CLIP, IS, and Pick scores comparable to the Vanilla baseline and NIRVANA, demonstrating strong semantic alignment, perceptual diversity, and human preference alignment. Importantly, MoDM obtains substantially lower FID scores compared to standalone small or distilled models like SDXL, SD3.5L-Turbo, and SANA, indicating it preserves a high-quality distribution similar to the large model and avoids the occasional distortions and defects typical of small model outputs. In contrast, the PINECONE baseline shows noticeably lower CLIP scores, reflecting weaker imagetext alignment and highlighting the limitations of retrieval-only methods without generative refinement—emphasizing the effectiveness of MoDM's refinement approach.

Separately, Table 3 reports image quality on DiffusionDB with FLUX as the vanilla large model. MoDM balances quality and efficiency, achieving CLIP, IS, and Pick scores close to the strong FLUX baseline while improving upon the FID of standalone small or distilled models, confirming its generalization across large model backbones. Fig. 20 in Appendix shows example outputs, highlighting its high-quality image generation.

