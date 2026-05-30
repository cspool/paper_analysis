# A. Appendix

### A.1. Visible Comparison Examples

![](_page_8_Figure_3.jpeg)

<span id="page-8-1"></span>Figure 7. Visible examples of using SageAttention2++ on image generation.

### <span id="page-8-0"></span>A.2. Datasets and Metrics in Experiments

Datasets. Text-to-text models are evaluated on: WikiText [\(Merity et al., 2022\)](#page-6-14) to assess the model's prediction confidence, LAMBADA [\(Paperno et al., 2016\)](#page-6-15) for contextual understanding, and Needle-in-A-Haystack (NIAH) task [\(Kamradt, 2023\)](#page-5-15). Text-to-video models are evaluated using the open-sora [\(Zheng et al., 2024\)](#page-7-9) prompt sets. Text-to-image models are assessed on COCO annotations [\(Lin et al., 2014\)](#page-5-16).

End-to-end metrics. For text-to-text models, we use perplexity (Ppl.) [\(Jelinek et al., 1977\)](#page-5-17) for WikiText, accuracy (Acc.) for LAMBADA and NIAH. For text-to-video models, following [Zhao et al.](#page-7-10) [\(2025\)](#page-7-10), we evaluate the quality of generated videos on five metrics: CLIPSIM and CLIP-Temp (CLIP-T) [\(Liu et al., 2024\)](#page-6-16) to measure the text-video alignment; VQA-a and VQA-t to assess the video aesthetic and technical quality, respectively; and Flow-score (FScore) for temporal consistency [\(Wu et al., 2023\)](#page-6-17). For text-to-image models, generated images are compared with the images in three aspects: FID [\(Heusel et al., 2017\)](#page-5-18) and sFID [\(Salimans et al., 2016\)](#page-6-18) for fidelity evaluation, *Clipscore* (CLIP) [\(Hessel et al., 2021\)](#page-5-19) for text-image alignment, and *ImageReward* (IR) [\(Xu et al., 2023\)](#page-6-19) for human preference.

Accuracy metrics. We use three metrics to assess the accuracy of quantized attention output O′ compared to attention output in full-precision O. First, we flatten O′ and O into vectors in the shape of 1 × n. Then, Cosine similarity: CosSim = POO′/ pPO<sup>2</sup> pPO′<sup>2</sup>, Relative L1 distance: L1 = P|O − O′ |/ P|O|, Root mean square error: RMSE = p (1/n) P(O − O′) 2.