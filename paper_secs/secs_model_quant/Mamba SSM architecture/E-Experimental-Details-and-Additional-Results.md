# E Experimental Details and Additional Results

### <span id="page-28-0"></span>**E.1** Synthetic Tasks

**Selective Copying.** Our setting is on sequences of length 4096, with a vocab size of 16 possible tokens (including the white "noise" token from Figure 2) and requiring models to memorize 16 "data" tokens. We use 2 layer models with a model dimension of D = 64.

Models are trained for 400K steps at a constant learning rate of 0.0001 with a batch size of 64.

**Induction Heads.** Training consists of randomly generating data every step, with a batch size of 8. We choose an "epoch" size of 8192 steps, and track the accuracy on fixed validation sets (also randomly generated) of each target sequence length. For the MHA-Abs and Mamba models, results are reported after the 25th epoch ( $8192 \times 25 = 204800$  steps). For the MHA-RoPE and MHA-xPos models, results are reported after the 50th epoch ( $8192 \times 50 = 409600$  steps). For the LTI H3 and Hyena models, results are reported after the 10th epoch (81920 steps) because they had converged by then and failed to improve further.

We use the Adam optimizer with no weight decay. All models are trained at constant learning rates 2e - 4 and 1e - 3, and the better results are reported for each model (2e - 4 for all models except Mamba). The attention and Hyena models did not learn at LR 1e - 3. H3 learned at both LRs, but interestingly generalized better to shorter sequences at the smaller LR of 2e - 4. Mamba learned at both LRs, but extrapolated better at the larger LR of 1e - 3.

### <span id="page-28-2"></span>E.2 Language Modeling

#### **E.2.1 Scaling Law Details**

Scaling law experiments generally followed the GPT3 recipe. All models were trained on the Pile with the GPT2 tokenizer.

**Model Sizes.** Table 12 specifies the model sizes we use for scaling laws. This is taken directly from the GPT3 specifications (Brown et al. 2020), with very minor modifications. First, we changed the batch size of the 1.3B model from 1M tokens to 0.5M tokens, since we did not use enough parallelization to require the larger batch size. Second, we changed the number of training steps and total tokens to roughly match Chinchilla scaling laws (Hoffmann et al. 2022), which specify that training tokens should increase proportionally to model size.

Training Recipes. All models used the AdamW optimizer with

<span id="page-29-1"></span>Table 12: (Scaling Law Model Sizes.) Our model sizes and hyperparameters for scaling experiments. (Model dimension and number of heads applies only to Transformer models.)

| Params | n_layers | d_model | n_heads<br>/<br>d_head | Training steps | Learning Rate | Batch Size  | Tokens |
|--------|----------|---------|------------------------|----------------|---------------|-------------|--------|
| 125M   | 12       | 768     | 12 / 64                | 4800           | 6e-4          | 0.5M tokens | 2.5B   |
| 350M   | 24       | 1024    | 16 / 64                | 13500          | 3e-4          | 0.5M tokens | 7B     |
| 760M   | 24       | 1536    | 16 / 96                | 29000          | 2.5e-4        | 0.5M tokens | 15B    |
| 1.3B   | 24       | 2048    | 32 / 64                | 50000          | 2e-4          | 0.5M tokens | 26B    |

- gradient clip value 1.0
- weight decay 0.1
- no dropout
- linear learning rate warmup with cosine decay

By default, the peak learning rate is the GPT3 specification.

We give several models an "improved recipe", inspired by changes adopted by popular large language models such as PaLM (Chowdhery et al. [2023\)](#page-17-6) and LLaMa (Touvron et al. [2023\)](#page-21-6). These include:

- linear learning rate warmup with cosine decay to 1 − 5, with a peak value of 5× the GPT3 value
- no linear bias terms
- RMSNorm instead of LayerNorm
- AdamW hyperparameter = (.9, .95) (the GPT3 value) instead of the PyTorch default of = (.9, .999)

