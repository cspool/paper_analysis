# B Experimental Details

## <span id="page-11-1"></span>B.1 Implementation Details

We utilize LLMLingua-2 [\(Pan et al.,](#page-10-10) [2024\)](#page-10-10) as the token importance metric to generate our compressed CoT training data. The compression ratio γ is randomly selected from {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} for each training sample. We adopt LoRA [\(Hu](#page-9-11) [et al.,](#page-9-11) [2022\)](#page-9-11), an efficient and reproducible approach that has been widely verified as effective in LLM fine-tuning, to train our models. The rank r is set

to 8, and the scaling parameter α is set to 16. We train the models for 3 epochs on both datasets. The peak learning rate is set to 5e-5, following a cosine decay schedule. We use AdamW [\(Loshchilov and](#page-9-18) [Hutter,](#page-9-18) [2019\)](#page-9-18) for optimization, with a warmup ratio of 0.1. We implement our training process using the LLaMA-Factory [\(Zheng et al.,](#page-10-15) [2024\)](#page-10-15) library. Inference for both our method and all baselines is performed using the Huggingface transformers package. During inference, the maximum number of tokens max\_len is set to 512 for GSM8K and 1024 for MATH[4](#page-11-5) . All experiments are conducted using Pytorch 2.1.0 on 2×NVIDIA GeForce RTX 3090 GPU (24GB) with CUDA 12.1, and an Intel(R) Xeon(R) Platinum 8370C CPU with 32 cores.

