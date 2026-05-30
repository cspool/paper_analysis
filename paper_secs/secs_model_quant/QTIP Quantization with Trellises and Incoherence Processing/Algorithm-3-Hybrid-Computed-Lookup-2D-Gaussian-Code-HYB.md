# Algorithm 3 Hybrid Computed-Lookup 2D Gaussian Code "HYB"

```
input L-bit 0 left-padded integer x, codebook C \in \mathbb{R}^{2^Q \times (V=2)}. x \leftarrow x \cdot x + x \mod 2^{32} {calculate hash} v \in \mathbb{R}^2 \leftarrow C[(x >> (15-Q)) \& 2^Q - 1] {lookup from symmetric codebook} v \leftarrow v \mod (x \& (1 << 15)) {apply sign flip} output Pseudorandom approximate Gaussian vector v.
```

in total. Table 2 shows that in practice, Algorithm 4 can find close-to-optimal tail-biting sequences while being significantly cheaper to run than other tail-biting approximation algorithms [29].

### 4 Experiments

Here, we present experiments quantizing the Llama family of models with QTIP [32, 33, 25]. These models offer strong performance across a wide range of sizes, allowing us to compare how different quantization methods perform and scale. We primarily compare QTIP against QuIP# and AQLM. For Llama 1, we include GPTVQ-2D instead of AQLM since AQLM does not publish Llama 1 numbers [36]. GPTVQ-2D performs 2D VQ inside GPTQ and offers strong performance. These methods outperform scalar quantization methods including GPTQ, AWQ, and OmniQuant; comparisons to those methods can be found in QuIP# and AQLM [19, 14, 28, 34, 11]. We mainly focus on the hybrid code (Section 4.2) since it is tailored for modern GPUs, and present a full suite of results for it. For the computed codes (Section 4.1), we present results for Llama 2.

Since the proxy error is not an additive distortion metric, we cannot minimize it by quantizing W as one sequence. Instead, for all experiments, we use QTIP as a quantizer in QuIP#'s BlockLDLQ, which allows us to simultaneously achieve high dimensionality and low proxy error [34]. Specifically, we quantize a block of  $T_x \times T_y$  weights as a sequence, where  $T_x$  and  $T_y$  span the output and input dimensions of W, respectively. Since BlockLDLQ only specifies feedback along the input dimension, this is equivalent to BlockLDLQ with  $g = T_y$  but a vector dimension of  $T_x T_y \gg T_y$ . This has the benefit of limiting the effect of g in BlockLDLQ's error bound  $gm\mu^2\sigma^2{\rm tr}(H^{1/2})^2/n$  while achieving a high dimension for TCQ. Algorithm 5 in the Appendix describes this in more detail.

#### <span id="page-6-2"></span>4.1 Lookup-Free Computed Codes

Here, we use 1MAD and 3INST with  $L=16, V=1, T_x=T_y=16$ . Setting  $T_x=T_y=16$  enables using a  $16\times 16$  MMA tile per trellis sequence to perform matrix multiplication during inference.  $16\times 16$  MMA tiles form the basis of many types of "AI hardware," making fast decoding relatively simple [6]. We do not perform fine-tuning since the codes themselves are not tunable, but these codes are fully compatible with QuIP#-style fine-tuning (recall that QuIP#'s codebook is also not tunable). Table 3 shows that both 1MAD and 3INST significantly outperform QuIP# without fine-tuning (AQLM does not have numbers without fine-tuning). Even at 4 bits, where all methods are close to lossless, QTIP results in significant improvements. Notably, the computed-code QTIP variants without fine-tuning outperforms both QuIP# and AQLM with fine-tuning on almost all models and sizes, showing that fine-tuning is not a silver bullet.

### 4.2 Hybrid Lookup-Computed Codes

### Algorithm 4 Tail-biting Trellis Approx.

<span id="page-7-0"></span>input Sequence  $S \in \mathbb{R}^T$ , (L, k, V) Trellis G.  $S' \leftarrow \text{Rotate } S \text{ to the right by } |T/2|$ 

 $\hat{S}' \leftarrow \text{Viterbi}(S', G)$ 

 $O \leftarrow L - kV$  bit overlap of  $\hat{S}'_{|T/2|} \hat{S}'_{|T/2|+1}$ 

 $\hat{S} \leftarrow \text{Viterbi}(S, G)$  with start/end overlap = O**output** Tail biting  $\hat{S}$ 

<span id="page-7-1"></span>Table 2: Quantizing 4K T=256 i.i.d Gaussian seqs. with a tail-biting (12, k, 1) trellis.

| $\overline{k}$ | Alg. 4 MSE | Optimal MSE |
|----------------|------------|-------------|
| 1              | 0.2803     | 0.2798      |
| 2              | 0.0733     | 0.0733      |
| 3              | 0.0198     | 0.0198      |
| 4              | 0.0055     | 0.0055      |

<span id="page-7-3"></span>Table 3: Wikitext2 and C4 perplexity (↓), ctx. 4096, QTIP with pure-computed codes. Even without fine-tuning, pure-computed QTIP outperforms QuIP# and AQLM, both of which use fine-tuning, at almost all models sizes.

|      |    |      | 4 B  | іт No | FT    | ≈4    | BIT  | 3 B  | іт No | FT    | ≈3    | BIT  | 2 B  | it No | FT    | $\approx 2$ | BIT          |
|------|----|------|------|-------|-------|-------|------|------|-------|-------|-------|------|------|-------|-------|-------------|--------------|
|      |    | FP16 | 1MAD | 3INST | QuIP# | QuIP# | AQLM | 1MAD | 3INST | QuIP# | QuIP# | AQLM | 1MAD | 3INST | QuIP# | QuIP#       | AQLM         |
| 2-7  | W2 | 5.12 | 5.17 | 5.17  | 5.22  | 5.19  | 5.21 | 5.38 | 5.40  | 5.60  | 5.41  | 5.38 | 7.05 | 6.82  | 8.22  | 6.19        | 6.14         |
| 2-1  | C4 | 6.63 | 6.71 | 6.71  | 6.79  | 6.75  | 6.75 | 6.99 | 7.01  | 7.34  | 7.04  | 7.01 | 9.14 | 8.96  | 11.0  | 8.16        | 6.14<br>8.09 |
| 2-13 | W2 | 4.57 | 4.62 | 4.62  | 4.65  | 4.63  | 4.64 | 4.74 | 4.74  | 4.90  | 4.78  | 4.78 | 5.59 | 5.52  | 6.06  | 5.35        | 5.33<br>7.19 |
| 2-13 | C4 | 6.05 | 6.10 | 6.10  | 6.15  | 6.13  | 6.14 | 6.28 | 6.28  | 6.50  | 6.35  | 6.33 | 7.46 | 7.39  | 8.07  | 7.20        | 7.19         |
| 2-70 | W2 | 3.12 | 3.16 | 3.16  | 3.18  | 3.18  | 3.19 | 3.27 | 3.27  | 3.41  | 3.35  | 3.36 | 3.87 | 3.90  | 4.16  | 3.91        | 3.83<br>5.62 |
| 2-70 | C4 | 4.97 | 5.00 | 5.00  | 5.02  | 5.02  | 5.03 | 5.09 | 5.09  | 5.20  | 5.15  | 5.17 | 5.70 | 5.69  | 6.01  | 5.71        | 5.62         |

Here, we use the hybrid lookup-computed code with  $L = 16, V = 2, T_x = T_y = 16, Q = 9$ . Setting Q = 9gives a 2KiB codebook, which fits in L1 cache even after duplication for bank conflicts (32 $\times$ ) on modern GPUs. This codebook is differentiable, so we can finetune it: to evaluate this, we fine-tune using QuIP#'s methodology, tuning both the codebook entries and the as-yet-unquantized weights in a blockwise fashion. Table 5 shows the perplexity of quantized Llama 1 and 2 models. In all cases, QTIP outperforms the other vector quantization-based methods. Even at 3 and 4 bits, where QuIP# and AQLM are close to lossless,

<span id="page-7-2"></span>Table 4: Batch size 1 decoding throughput on a RTX6000 Ada (960GB/s mem. BW).

| Метнор | BITS | 2-7B Tok/s | 2-70B Tok/s |
|--------|------|------------|-------------|
| FP16   | 16   | 55.9       | OOM         |
| AQLM   | 2    | 81.5       | 8.78        |
| QuIP#  | 2    | 186        | 22.2        |
| QTIP   | 2    | 188        | 23.5        |
| QTIP   | 3    | 161        | 19.1        |
| QTIP   | 4    | 140        | 16.3        |

QTIP roughly halves the perplexity gap. These results also show the importance of dimensionality. Note that the 3- and 4-bit Llama 2 70B numbers here match those in 3. Since Table 3 uses a purecomputed code without fine-tuning, fine-tuning has no effect in these regimes and the improvement over QuIP# is purely from dimensionality.

Table 6 shows zeroshot results computed with LM Eval, which are slightly random; QTIP generally matches or exceeds QuIP# and AQLM on these tasks [15]. Table 7 contains results on Llama 3. In our experiments, we observed that quantizing layer 0 v of all Llama 3 70B variants (including 3.1, 3.3, and all instruct models) resulted in catastrophic collapse of zeroshot performance. Since the focus of this work is on what to round with and not how to round, Table 7 includes results with quantizing 0 v. This catastrophic collapse can be remedied by prepending special tokens (e.g. \n), applying the chat template for instruct models (Table 8), or just not quantizing 0 v ([35]). Regardless, QTIP significantly improves upon QuIP# at all model sizes and bitrates, once again showing the dimensionality advantage of TCQ over VQ.

Table 8 shows results for Llama 3.1 instruct-tuned models, including Llama 3.1 405B. At all sizes, QTIP achieves strong results. Notably, QTIP is able to match or exceed PV-Tuning, a recent quantization method that focuses on better fine-tuning algorithms [21]. However, PV-Tuning is based off of AOLM and inherits its slow inference speed, making it significantly slower than OTIP. Finally, Table 9 shows results for quantizing Llama 3.2 instruct-tuned models to 4 bits. Since the embedding layers are very large relative to the decoder layers for small Llama 3 models ( $\approx 500 - 750 \text{MB}$ ), quantizing the decoder layers to fewer than 4 bits does not make a significant difference on the final model size. Here, QTIP is still able to achieve a meaningful end-to-end compression rate (2.5-3X) without degrading the final model.

<span id="page-8-0"></span>Table 5: Wikitext2 and C4 perplexity (↓), QTIP with the hybrid-computed code. QTIP enables highdimensional quantization and outperforms state-of-the-art vector quantization approaches.

|             |          |     | CTX. 2048, X = GPTVQ, Y = 0.13 |                |      |     |           |                | CTX. 4096, X = AQLM, Y ≈ 0 |     |           |           |     |           |           |
|-------------|----------|-----|--------------------------------|----------------|------|-----|-----------|----------------|----------------------------|-----|-----------|-----------|-----|-----------|-----------|
|             | WIKTEXT2 |     |                                |                |      | C4  |           |                |                            |     | WIKITEXT2 |           |     | C4        |           |
| METHOD BITS |          | 1-7 |                                | 1-13 1-30 1-65 |      | 1-7 |           | 1-13 1-30 1-65 |                            | 2-7 |           | 2-13 2-70 | 2-7 |           | 2-13 2-70 |
| FP16        | 16.0     |     | 5.68 5.09                      | 4.10           | 3.53 |     | 7.04 6.61 | 5.98           | 5.62                       |     | 5.12 4.57 | 3.12      |     | 6.63 6.05 | 4.97      |
| X           | 4+Y      |     | 5.94 5.20                      | 4.18           | 3.64 | –   | –         | –              | –                          |     | 5.21 4.65 | 3.19      |     | 6.75 6.14 | 5.03      |
| QUIP#       | 4.00     |     | 5.76 5.17                      | 4.18           | 3.60 |     | 7.18 6.67 | 6.03           | 5.66                       |     | 5.19 4.63 | 3.18      |     | 6.75 6.13 | 5.02      |
| QTIP        | 4.00     |     | 5.72 5.15                      | 4.15           | 3.58 |     | 7.13 6.65 | 6.01           | 5.64                       |     | 5.17 4.61 | 3.16      |     | 6.69 6.09 | 5.00      |
| X           | 3+Y      |     | 6.32 5.31                      | 4.38           | 3.79 | –   | –         | –              | –                          |     | 5.38 4.78 | 3.36      |     | 7.01 6.33 | 5.17      |
| QUIP#       | 3.00     |     | 5.98 5.31                      | 4.36           | 3.70 |     | 7.39 6.83 | 6.17           | 5.77                       |     | 5.41 4.78 | 3.35      |     | 7.04 6.35 | 5.15      |
| QTIP        | 3.00     |     | 5.85 5.24                      | 4.26           | 3.68 |     | 7.26 6.74 | 6.09           | 5.71                       |     | 5.28 4.69 | 3.26      |     | 6.87 6.22 | 5.08      |
| X           | 2+Y      |     | 9.64 6.58                      | 5.63           | 4.91 | –   | –         | –              | –                          |     | 6.14 5.33 | 3.83      |     | 8.09 7.19 | 5.62      |
| QUIP#       | 2.00     |     | 6.86 5.97                      | 5.02           | 4.36 |     | 8.36 7.48 | 6.71           | 6.19                       |     | 6.19 5.35 | 3.91      |     | 8.16 7.20 | 5.71      |
| QTIP        | 2.00     |     | 6.52 5.80                      | 4.83           | 4.21 |     | 7.99 7.31 | 6.56           | 6.08                       |     | 5.86 5.11 | 3.70      |     | 7.73 6.85 | 5.48      |

Table 6: Zeroshot accuracy (↑), QTIP with the hybrid-computed code.

<span id="page-8-1"></span>

|                                                                                  |    |      | 2-70 |      |      |      |      | 2-13 |      |      |      |      | 2-7  |      |      |
|----------------------------------------------------------------------------------|----|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| MTHD. BITS ARCC ARCE PIQA WINO BITS ARCC ARCE PIQA WINO BITS ARCC ARCE PIQA WINO |    |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| FP16                                                                             | 16 | 51.1 | 77.7 | 81.1 | 77.0 | 16   | 45.6 | 73.3 | 73.5 | 69.6 | 16   | 40.0 | 69.3 | 78.5 | 67.3 |
| AQLM 4.14                                                                        |    | 50.7 | 77.3 | 81.5 | 76.5 | 3.94 | 44.8 | 73.3 | 78.4 | 69.9 | 4.04 | 41.0 | 70.2 | 78.2 | 67.3 |
| QUIP#                                                                            | 4  | 50.5 | 77.7 | 81.4 | 77.3 | 4    | 43.6 | 71.3 | 78.7 | 69.6 | 4    | 40.4 | 68.6 | 78.5 | 67.4 |
| QTIP                                                                             | 4  | 50.0 | 77.8 | 81.3 | 76.9 | 4    | 44.8 | 73.6 | 78.9 | 69.9 | 4    | 40.0 | 68.9 | 78.4 | 67.1 |
| AQLM 3.01                                                                        |    | 50.3 | 78.0 | 80.7 | 75.3 | 3.03 | 42.8 | 72.9 | 78.5 | 68.8 | 3.04 | 38.5 | 66.8 | 77.3 | 65.4 |
| QUIP#                                                                            | 3  | 50.9 | 77.6 | 81.4 | 76.1 | 3    | 44.0 | 72.5 | 78.4 | 69.1 | 3    | 39.2 | 68.4 | 77.3 | 66.5 |
| QTIP                                                                             | 3  | 50.3 | 78.2 | 80.6 | 77.0 | 3    | 44.0 | 72.8 | 78.0 | 69.5 | 3    | 38.9 | 68.1 | 78.1 | 66.9 |
| AQLM 2.07                                                                        |    | 47.9 | 77.7 | 80.4 | 75.9 | 1.97 | 38.8 | 69.3 | 75.9 | 68.8 | 2.02 | 32.8 | 63.7 | 74.8 | 65.7 |
| QUIP#                                                                            | 2  | 47.6 | 77.1 | 79.5 | 74.6 | 2    | 39.6 | 69.0 | 77.3 | 67.4 | 2    | 35.2 | 65.3 | 75.4 | 64.9 |
| QTIP                                                                             | 2  | 48.0 | 76.3 | 80.2 | 75.1 | 2    | 41.4 | 70.8 | 77.3 | 67.6 | 2    | 35.7 | 65.6 | 75.9 | 64.7 |

### 4.3 Inference Speed

Table [4](#page-7-2) shows the batch size 1 inference speed of QTIP, QuIP#, and AQLM on Llama 2 7B and 70B with matrix fusion. Here, the design choices of QTIP and QuIP# become apparent. Whereas AQLM uses a codebook that is too large to fit in cache and thus prevents fast inference, both QTIP and QuIP# achieve significant speedups over FP16. Furthermore, while it is impressive that both QuIP# and QTIP are > 2× faster than AQLM, it is even more impressive that QTIP is able to match QuIP#'s throughput with an effective dimension size of 256, or 32× larger than QuIP#'s. This means that the improved quantization quality of QTIP comes with *no additional inference-time cost*. Although our empirical throughput numbers were timed on NVIDIA GPUs, QTIP can be fast on a broad class of accelerators due to its flexibility. QTIP only requires generating a pseudorandom Gaussian efficiently, and can work on devices with no cache as well as devices with lookup hardware. For example, if we were using a ARMv8 CPU, we could use the vqtbl4q\_u8 NEON intrinsic to look up 16 indices in a 64-entry codebook. This would let us use a 6 bit 1D codebook with the HYB code (Q=6, V=1). Quantizing Llama 2 7B to 2 bits with this setup and w/out fine-tuning gives 6.89 Wikitext2 perplexity – essentially the same state-of-the-art quality as 3INST.

