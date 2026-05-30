# A More Background

### A.1 Image-pretrained VLMs

Image-pretrained vision-language models (VLMs) combine large language models (LLMs)[\(Bai et al.,](#page-9-7) [2023a;](#page-9-7) [Brown et al.,](#page-9-8) [2020;](#page-9-8) [Chiang et al.,](#page-9-9) [2023;](#page-9-9) [Chowdhery et al.,](#page-9-10) [2023;](#page-9-10) [Gilardi et al.,](#page-9-11) [2023;](#page-9-11) [Tou](#page-10-16)[vron et al.,](#page-10-16) [2023\)](#page-10-16) with visual encoders such as CLIP[\(Radford et al.,](#page-10-10) [2021\)](#page-10-10), enabling effective image-text alignment and multimodal understanding. As a pioneer, Flamingo [\(Alayrac et al.,](#page-9-12) [2022\)](#page-9-12) introduces interleaved vision-language modeling for open-ended generation. BLIP-2 [\(Li et al.,](#page-10-17) [2023b\)](#page-10-17) employs Q-Former to align frozen vision and language models. LLaVA and its extensions [\(Liu et al.,](#page-10-5) [2023,](#page-10-5) [2024\)](#page-10-8) integrate lightweight connectors with instruction tuning to bridge modalities efficiently. Qwen-VL [\(Bai et al.,](#page-9-13) [2023b;](#page-9-13) [Wang](#page-10-18) [et al.,](#page-10-18) [2024a;](#page-10-18) [Bai et al.,](#page-9-14) [2025\)](#page-9-14) connects a visual encoder to the Qwen language model via crossattention, supporting high-resolution and multilingual reasoning. Most recently, MM1 [\(McK](#page-10-19)[inzie et al.,](#page-10-19) [2024\)](#page-10-19) provides a systematic analysis of model scaling and data design for efficient VLM training.

### B Hyper-Parameters Ablation

### B.1 Effect of α in Dynamic Compression

Table [9](#page-12-0) presents the results of an ablation study on α, which controls the uniform sampling ratio in temporal dynamic compression. The results show that accuracy on the EgoSchema benchmark increases significantly as α decreases from 0.95 to 0.85. This suggests that selecting supplementary key frames to emphasize informative segments enhances the video understanding ability of image-pretrained VLMs. However, when α continues to decrease, the number of uniformly sampled frames becomes insufficient, weakening the model's global perception of the video and leading to performance degradation.

### B.2 Effect of β in Dynamic Compression

Table [10](#page-12-0) presents the results of an ablation study on β, which controls the proportion of spatial tokens retained during pruning. A larger β corresponds to more aggressive pruning, keeping only highly activated tokens. As β increases, the accuracy on the EgoSchema benchmark first rises and then falls, indicating that retaining too few tokens harms the model's understanding of visual content, while retaining too many introduces redundant noise.

### B.3 Effect of τ in Dynamic Compression

Table [11](#page-12-0) presents the results of an ablation study on τ , the cosine similarity threshold used for token merging. A smaller τ results in more tokens being grouped and merged. As τ increases, accuracy on the EgoSchema benchmark follows a rise-then-fall trend. This is because merging low-similarity tokens can blur critical visual details, while merging only highly similar tokens leads to token redundancy, both of which hinder model performance.

### B.4 Effect of t in Question Decomposition

Table [12](#page-12-0) presents the results of an ablation study on t, which controls the diversity of sub-questions generated during question decomposition. A larger t results in more diverse sub-questions. Accuracy on the EgoSchema benchmark initially improves as t increases, but eventually declines. This indicates that both insufficient and excessive diversity can impair the model's ability to comprehensively interpret large volumes of visual tokens.

