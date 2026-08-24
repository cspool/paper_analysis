# 5 Implementation

#### 5.1 Model Architecture

Despite the reduction in model parameters, OpenBA-V2 maintains the same model architecture as OpenBA [\(Li et al., 2023b\)](#page-16-1), including an Encoder-Decoder model structure, rotary embedding scheme [\(Su et al., 2024\)](#page-17-14),and the SwiGLU Activation Function [\(Shazeer, 2020\)](#page-17-15).

#### <span id="page-8-0"></span>5.2 Training

We first use a relatively small amount of tokens to compress the 15B model to 3.8B without a significant loss in model capability. Subsequently, we use a large number of tokens to train the model further for better performance. The entire process can be divided into multiple stages, and table [5.2](#page-8-0) illustrates the model sizes and training objectives at different stages. We use a cosine scheduler for stages 1-4 with the max learning rate 1e-4 and the min learning rate 5e-5. For stage 5, we use the max learning rate 5e-5 and the min learning rate 1e-5. After pruning, We directly prune 140,000 tokens from the vocabulary, reducing the model size from 3.8B to 3.4B.

