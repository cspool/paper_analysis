# <span id="page-23-1"></span>D.2 Hyperparameters

ELF pipeline hyperparameters. Tab. [4](#page-24-3) summarizes the main hyperparameters used in the ELF pipeline, covering model architecture, diffusion settings, conditioning and guidance, and optimization details. Unless noted otherwise, all experiments in the paper follow this default configuration. We include these settings for completeness and to facilitate reproducibility.

Inference-time settings for system-level comparison. For system-level comparison in Fig. [7,](#page-7-0) we use SDE sampling with time schedule enabled for all step budgets. We set the CFG scale to 3 for 8-, 16-, and 32-step generation. For SDE sampling, we use a stronger noise injection scale of γ = 2 in the very few-step regimes of 8 and 16 steps, and reduce it to γ = 1.5 for 32 steps, as longer denoising trajectories require less stochastic correction. For the system-level comparison in Tab. [1,](#page-8-1) we use 64-step ODE sampling with time schedule. We set the self-conditioning CFG scale to 1 and the input-condition CFG scale to 2.

<span id="page-24-3"></span><span id="page-24-2"></span>

| Model Architecture            |          | Denoising and Decoding Config |                 |
|-------------------------------|----------|-------------------------------|-----------------|
| Model                         | ELF-B    | Time schedule                 | logit normal    |
| Model size                    | 105M     | Denoiser (Pmean, Pstd)        | (−1.5, 0.8)     |
| Encoder backbone              | T5-small | Denoiser noise scale          | 2.0             |
| Embedding dimension           | 512      | Decoder (Pmean, Pstd)         | (0.8, 0.8)      |
| Bottleneck dimension          | 128      | Decoder noise scale           | 5.0             |
| Model dimension               | 768      | Denoiser vs. decoder prob.    | 0.8 vs. 0.2     |
| Sequence length               | 1024     |                               |                 |
| Conditioning and Guidance     |          | Optimization and Training     |                 |
| Self-conditioning probability | 0.5      | Optimizer                     | Muon            |
| Self-conditioning CFG range   | [0.5, 5] | Learning rate                 | 0.002           |
| Num. of time tokens           | 4        | Weight decay                  | 0               |
| Num. of model-mode tokens     | 4        | Training epochs               | 5               |
| Num. of CFG tokens            | 4        | Global batch size             | 512             |
| SDE γ                         | 1.0      | Learning rate schedule        | constant        |
|                               |          | Warmup epochs                 | 0.5             |
|                               |          | EMA decay                     | 0.9999          |
|                               |          | Training device               | TPU v5p × 64    |
|                               |          | Training time                 | 1.5 h per epoch |

Table 4: Default training hyperparameters and setup for ELF-B on the OpenWebText dataset. Unless noted otherwise, all experiments in the paper follow this default configuration.

<span id="page-24-1"></span>

| Method           | Base training   | Distillation training | Effective tokens | Ratio |
|------------------|-----------------|-----------------------|------------------|-------|
| MDLM [56]        | 512 × 1M × 1024 | -                     | 524.3B           | 11.6× |
| Duo [57]         | 512 × 1M × 1024 | -                     | 524.3B           | 11.6× |
| MDLM + SDTT [56] | 512 × 1M × 1024 | 512 × 10K × 5 × 1024  | 550.5B           | 12.2× |
| Duo + DCD [57]   | 512 × 1M × 1024 | 512 × 10K × 5 × 1024  | 550.5B           | 12.2× |
| FLM [30]         | 512 × 1M × 1024 | -                     | 524.3B           | 11.6× |
| FMLM [30]        | 512 × 1M × 1024 | 512 × 100K × 1024     | 576.7B           | 12.8× |
| LangFlow [10]    | 512 × 1M × 1024 | -                     | 524.3B           | 11.6× |
| ELF (ours)       | 5 × 9.04B       | -                     | 45.2B            | 1.0×  |

Table 5: Estimated effective training tokens for ELF and the prior DLM baselines used in our systemlevel comparison (Fig. [7c](#page-7-0)). We estimate base-training tokens as batch size×steps×sequence length; distillation / flow-map stages are added on top where applicable.

Training-token budget for system-level comparison. Tab. [5](#page-24-1) reports the estimated effective training tokens used by ELF and each baseline in Fig. [7c](#page-7-0). We estimate base-training tokens as batch size × steps × sequence length and add distillation or flow-map stages on top where applicable. The OWT dataset contains roughly 9.04B tokens. With our default training schedule of 5 epochs, ELF therefore uses 45.2B effective training tokens. Thus, ELF requires roughly an order of magnitude fewer effective training tokens than the compared DLMs.

### <span id="page-24-0"></span>D.3 Ablation Studies Setting

We evaluate several choices of embedding representations for ELF, and report the implementation details as below. We also try two-stage training with a separate decoder. Unless specified, we keep other settings the same as the default ELF configuration.

Scratch encoder. We train an encoder from scratch on OpenWebText [\[18\]](#page-10-14) by following the original T5-small training pipeline [\[53\]](#page-11-15). The encoder is trained for 5 epochs with a learning rate of 1 × 10<sup>−</sup><sup>3</sup> , cosine learning rate schedule, 0.4 epoch warmup, and a batch size of 512. During ELF training, we apply channel-wise normalization to the encoder outputs.

| Steps | SC CFG | γ   | Gen. PPL ↓   | Entropy ↑    |
|-------|--------|-----|--------------|--------------|
| 8     | 3      | 2.0 | 67.32 ± 2.25 | 5.14 ± 0.085 |
| 16    | 3      | 2.0 | 33.66 ± 1.09 | 5.16 ± 0.026 |
| 32    | 3      | 1.5 | 24.08 ± 0.16 | 5.15 ± 0.002 |

<span id="page-25-1"></span><span id="page-25-0"></span>Table 6: System-level ELF performance reported as mean ± standard error (SE) over 6 independent evaluation runs (seeds 0–5; n = 6).

| Sampler | SC CFG | ELF-B 105M |         | ELF-M 342M |         | ELF-L 652M |         |
|---------|--------|------------|---------|------------|---------|------------|---------|
|         |        | Gen. PPL   | Entropy | Gen. PPL   | Entropy | Gen. PPL   | Entropy |
|         | 0.5    | 36.77      | 5.28    | 39.21      | 5.35    | 37.50      | 5.41    |
|         | 1.0    | 29.50      | 5.23    | 33.45      | 5.30    | 31.82      | 5.37    |
|         | 1.5    | 25.25      | 5.18    | 28.42      | 5.26    | 28.72      | 5.35    |
| SDE     | 2.0    | 22.53      | 5.14    | 25.34      | 5.23    | 26.47      | 5.32    |
|         | 3.0    | 19.72      | 5.10    | 21.69      | 5.18    | 23.31      | 5.28    |
|         | 3.5    | 37.56      | 5.30    | 36.48      | 5.34    | 22.28      | 5.27    |
|         | 4.0    | 36.50      | 5.29    | 34.93      | 5.33    | 21.37      | 5.26    |
|         | 0.5    | 104.29     | 5.51    | 88.51      | 5.51    | 68.27      | 5.52    |
|         | 1.0    | 65.30      | 5.40    | 62.47      | 5.44    | 49.72      | 5.45    |
|         | 1.5    | 44.85      | 5.31    | 46.71      | 5.37    | 39.97      | 5.40    |
| ODE     | 2.0    | 34.65      | 5.23    | 37.66      | 5.32    | 33.72      | 5.36    |
|         | 3.0    | 26.62      | 5.15    | 28.80      | 5.24    | 26.57      | 5.29    |

Table 7: Scaling performance of generative perplexity (Gen. PPL) and unigram entropy for ELF models of different sizes under SDE and ODE samplers with 64 sampling steps. The effect of self-conditioning (SC) CFG scaling diminishes beyond 3.

Pretrained embedding layer. We use the frozen embedding table from the T5-small encoder as the token embedding layer. The embedding layer matrix is normalized, and the unembedding layer is trained separately.

Gaussian embedding layer. We randomly initialize and freeze an embedding layer from a Gaussian distribution, with token-wise embedding mean 0 and standard deviation 1. The unembedding layer is trained separately using the decoder mode.

Learnable embedding layer. We jointly train the embedding layer together with the denoiser and decoder modes. The unembedding layer is tied with the embedding layer: denoiser-mode updates affect the embedding layer, while decoder-mode updates affect the unembedding layer. To stabilize training, we apply normalization directly on the unembedding layer matrix at every step.

Separate decoder. For the separate-decoder setting, we use a randomly initialized decoder architecture obtained by mirroring the T5-small encoder. We keep the encoder fixed, mask 20% of the input tokens, and add logit-normal noise to the latent representations with Pmean = 0.5 and Pstd = 1.0. The model is trained for 3 epochs with a learning rate of 3×10<sup>−</sup><sup>4</sup> and a cosine learning-rate schedule. The relative noise scale with respect to the normalized latent representations is set to 5.0.

### D.4 Reported Numbers

System level comparison. Across 6 independent evaluation seeds, ELF shows highly consistent system-level behavior, as shown in Tab. [6.](#page-25-1) As the number of sampling steps increases from 8 to 32, the standard error (SE) decreases. The small standard errors—especially at 32 steps—suggest that these gains are robust to random seed variation and that the overall trend is reliable across runs. See Tab. [6](#page-25-1) for detailed numbers.

Scaling behavior with CFG scales. The default setting for both sampling methods uses 64 sampling steps with time schedule. For the SDE sampler, we set γ = 1.0. The exact numbers are reported in

<span id="page-26-1"></span><span id="page-26-0"></span>

| Config            | AR             | MDLM                    | E2D2                    | D           | uo          |
|-------------------|----------------|-------------------------|-------------------------|-------------|-------------|
| Architecture      |                |                         |                         |             |             |
| Codebase          | E2D2           | E2D2                    | E2D2                    | Duo         | Duo         |
| Tokenizer         | Qwen3-0.6B     | Qwen3-0.6B              | Qwen3-0.6B              | T5-small    | T5-small    |
| Hidden Size       | 256            | 256                     | 256                     | 768         | 768         |
| Intermediate Size | 768            | 768                     | 768                     | _           | -           |
| #Layers / Blocks  | 28             | 28                      | enc=20, dec=8           | 12          | 12          |
| Sequence Length   | 64             | 64                      | 64                      | 64          | 64          |
| Max Cond Length   | 1024           | 1024                    | 1024                    | 1024        | 64          |
| Cond Embed        | _              | _                       | _                       | T5-small    | T5-small    |
| Training          |                |                         |                         |             |             |
| Dataset           | XSum           | XSum                    | XSum                    | XSum        | De-En       |
| Learning Rate     | 3e-4           | 3e-4                    | 3e-4                    | 3e-4        | 3e-4        |
| LR Scheduler      | const          | const                   | const                   | const       | const       |
| Warmup Steps      | 1000           | 1000                    | 1000                    | 2500        | 2500        |
| Global Batch Size | 128            | 128                     | 128                     | 512         | 512         |
| Optimizer         | DecoupledAdamW | DecoupledAdamW          | DecoupledAdamW          | AdamW       | AdamW       |
| Loss Type         | NLL            | MDLM ELBO               | E2D2 ELBO               | Duo ELBO    | Duo ELBO    |
| Train Steps       | 500K           | 500K                    | 500K                    | 1M          | 1 <b>M</b>  |
| Evaluation        |                |                         |                         |             |             |
| Sampling Strategy | greedy         | predict_and_noise       | predict_and_noise       | Duo sampler | Duo sampler |
| Sampling Steps    | L = 64  (AR)   | $\approx L$ (first-hit) | $\approx L$ (first-hit) | 1000        | 1000        |
| Block size        | 1              | 32                      | 8                       | -           | -           |
| CFG Scale         | _              | _                       | _                       | 1.0         | 1.5         |
| Checkpoint        | best           | best                    | best                    | best        | best        |
| EMA               | true           | true                    | true                    | true        | true        |

Table 8: **Detailed training and evaluation configurations for conditional generation tasks** of our reproduced AR, MDLM, E2D2, and Duo baselines. AR, MDLM, and E2D2 are reproduced on XSum using the E2D2 [4] codebase and follow the configurations reported in the E2D2 paper. For Duo, we build on the original Duo [57] repository, add cross-attention conditioning and CFG, adapt the T5-small encoder to match our setting, and tune the hyperparameters to obtain the strongest reproduced results.

Tab. 7. Larger CFG scales improve generation quality by reducing Gen. PPL within a certain range. The effect of CFG scaling reverses beyond 3. Only ELF-L benefits from increasing the CFG scale from 3 to 4. Thus, in most default ablation studies, we only consider CFG scales from 0.5 to 3.

#### **D.5** Conditional Generation

Specifically, the WMT14 results for AR, MDLM, and E2D2 are taken from the E2D2 [4] paper, the SeqDiffuSeq result is taken from the LD4LG [41] paper, and the CDCD result is taken from the original CDCD [13] paper. For reproduced results, Duo [57] is implemented using the Duo codebase<sup>4</sup>, while AR, MDLM, and E2D2 are reproduced using the E2D2 codebase<sup>5</sup>.

For a fair comparison, we reproduce all baselines using settings that are as close as possible to their original implementations, as summarized in Tab. 8. For AR, MDLM, and E2D2, we use the E2D2 codebase and follow the training and evaluation configurations reported in the E2D2 paper on XSum. Note that although E2D2 is primarily designed for semi-autoregressive generation, we find that MDLM also achieves its best performance under a semi-autoregressive setting (*i.e.*, block size 32 with two-block generation); using single-block diffusion without semi-autoregressive generation degrades performance. For Duo, we start from the official Duo repository and adapt it to our conditional generation setting by adding cross-attention conditioning and classifier-free guidance, and by using a T5-small encoder for the conditioning input. During inference, we generate without

<span id="page-26-2"></span><sup>4</sup>https://github.com/s-sahoo/duo

<span id="page-26-3"></span><sup>5</sup>https://github.com/kuleshov-group/e2d2

<span id="page-27-1"></span>

| t<br>=<br>0 | strength | will    | building | building     | building   | building | back        | played | playedband | bit     | choiceband | bitband | played | playedband | played | bit        | bit    |
|-------------|----------|---------|----------|--------------|------------|----------|-------------|--------|------------|---------|------------|---------|--------|------------|--------|------------|--------|
|             |          |         |          |              |            |          |             |        |            |         |            |         |        |            |        |            |        |
|             | The      | results | was      | ab           | disturbing |          | EFuture     | after  | watching   | various | games      | ,       |        | I<br>was   | pretty | fierce     | withLI |
|             |          |         |          |              |            |          |             |        |            |         |            |         |        |            |        |            |        |
|             | The      | results | were     | flat         | striking   |          | Immediately | after  | watched    | the     | games      | ,       |        | I<br>was   | pretty | determined | with   |
|             |          |         |          |              |            |          |             |        |            |         |            |         |        |            |        |            |        |
| t<br>=<br>1 | The      | results | were     | particularly | striking   |          | Immediately | after  | watching   | the     | games      | ,       |        | I<br>was   | very   | concerned  | about  |

Figure 17: Denoising trajectory of ELF-B. As t increases from 0 to 1, ungrammatical sentences are progressively refined into fluent and grammatical text.

semi-autoregressive decoding. We tune the main sampling and guidance hyperparameters and report the best reproduced results we obtain.

