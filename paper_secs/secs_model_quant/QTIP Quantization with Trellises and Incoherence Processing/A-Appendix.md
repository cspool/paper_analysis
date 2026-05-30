# A Appendix

### A.1 Additional Results

### A.1.1 Ablations on Trellis Size

Table [10](#page-14-0) shows an ablation on L for quantizing Llama 2 7B with K = 2, V = 1, the bitshift trellis, a pure-lookup codebook, and no fine-tuning. L = 8 is the largest L achievable if we had to store the trellis and codebook in the same amount of cache as the HYB code (2KiB). L = 10 is the largest L achievable if we only had to store the codebook. As expected, increasing L improves quality. Table [10](#page-14-0) also shows very little difference between an equal-sized LUT codebook and QTIP's codes, meaning that QTIP isn't sacrificing quality for speed. However, an equal-sized LUT would need > 10× more cache than the latest GPUs have, making the bitshift trellis and compute-based codes necessary to achieve both quality and speed. Table [11](#page-14-1) shows an ablation on V with L = 12 and 16, K = 2, and the same settings as Table [10.](#page-14-0) Increasing V generally decreases quality, but this can be recovered with a larger L. It is hard to measure V 's impact on decoding speed since this is highly implementation and hardware dependent, so V is more of a user-chosen hyperparameter.

<span id="page-14-0"></span>Table 10: Ablation on L when quantizing Llama 2 7B to 2 bits (K = 2 and V = 1).

| L     | Trellis Size | CB size  | total size | W2   | C4   |
|-------|--------------|----------|------------|------|------|
| QuIP# | -            | 8Kb      | 8Kb        | 8.22 | 11.0 |
| 8     | 8.19 Kb      | 4.10 Kb  | 12.29 Kb   | 7.83 | 10.3 |
| 10    | 40.96 Kb     | 16.38 Kb | 57.34 Kb   | 7.49 | 9.67 |
| 12    | 196.61 Kb    | 65.54 Kb | 262.14 Kb  | 6.97 | 9.21 |
| 16    | 4.19 Mb      | 1.05 Mb  | 5.24 Mb    | 6.83 | 8.92 |
| 16    | Bitshift     | 3INST    | 0Kb        | 6.82 | 8.96 |
|       |              |          |            |      |      |

<span id="page-14-1"></span>Table 11: Ablation on V when quantizing Llama 2 7B to 2 bits (K = 2).

| Codebook         | L  | V | W2   | C4   |
|------------------|----|---|------|------|
| LUT              | 12 | 1 | 6.97 | 9.21 |
| LUT              | 12 | 2 | 7.09 | 9.24 |
| LUT              | 12 | 4 | 7.55 | 9.88 |
| LUT              | 16 | 1 | 6.83 | 8.92 |
| LUT              | 16 | 2 | 6.79 | 8.97 |
| QTIP HYB (no FT) | 16 | 2 | 6.83 | 8.97 |
| LUT              | 16 | 4 | 6.92 | 9.07 |
|                  |    |   |      |      |

### A.1.2 Zeroshot Results

Table 12: Zeroshot results for the 1MAD code.

|      | Bits | ArcC (acc) | ArcE (acc) | BoolQ (acc) | PiQA (acc) | Wino (acc) |
|------|------|------------|------------|-------------|------------|------------|
| 2-7  | 16   | 39.9       | 69.3       | 71.1        | 78.4       | 67.2       |
| 2-7  | 4    | 39.0       | 69.4       | 72.0        | 78.4       | 67.9       |
| 2-7  | 3    | 38.8       | 68.0       | 68.2        | 77.6       | 68.4       |
| 2-7  | 2    | 32.1       | 63.5       | 66.3        | 73.3       | 62.7       |
| 2-13 | 16   | 45.6       | 73.3       | 69.1        | 78.7       | 69.7       |
| 2-13 | 4    | 45.6       | 72.9       | 68.1        | 78.7       | 70.3       |
| 2-13 | 3    | 42.2       | 71.0       | 69.9        | 78.6       | 69.8       |
| 2-13 | 2    | 38.5       | 71.5       | 71.4        | 75.9       | 68.9       |
| 2-70 | 16   | 51.2       | 77.7       | 76.7        | 81.1       | 76.9       |
| 2-70 | 4    | 51.1       | 77.8       | 75.2        | 81.5       | 77.0       |
| 2-70 | 3    | 50.8       | 77.8       | 77.9        | 80.7       | 76.3       |
| 2-70 | 2    | 49.3       | 77.7       | 83.3        | 80.4       | 75.7       |

Table 13: Zeroshot results for the 3INST code.

|      | Bits | ArcC (acc) | ArcE (acc) | BoolQ (acc) | PiQA (acc) | Wino (acc) |
|------|------|------------|------------|-------------|------------|------------|
| 2-7  | 16   | 39.9       | 69.3       | 71.1        | 78.4       | 67.2       |
| 2-7  | 4    | 40.2       | 68.5       | 70.3        | 78.0       | 67.7       |
| 2-7  | 3    | 40.2       | 68.6       | 73.0        | 77.5       | 65.4       |
| 2-7  | 2    | 32.9       | 61.9       | 65.5        | 74.5       | 65.0       |
| 2-13 | 16   | 45.6       | 73.3       | 69.1        | 78.7       | 69.7       |
| 2-13 | 4    | 45.4       | 72.7       | 67.9        | 78.5       | 69.9       |
| 2-13 | 3    | 44.5       | 72.6       | 70.1        | 78.5       | 69.4       |
| 2-13 | 2    | 38.7       | 68.2       | 63.6        | 75.6       | 68.7       |
| 2-70 | 16   | 51.2       | 77.7       | 76.7        | 81.1       | 76.9       |
| 2-70 | 4    | 50.3       | 77.9       | 77.3        | 80.7       | 76.5       |
| 2-70 | 3    | 50.9       | 78.3       | 78.8        | 81.1       | 77.5       |
| 2-70 | 2    | 48.0       | 76.5       | 76.7        | 80.1       | 77.6       |

Table 14: Llama 1 Zeroshot results for the Hybrid code

|      | Bits | ArcC (acc) | ArcE (acc) | BoolQ (acc) | PiQA (acc) | Wino (acc) |
|------|------|------------|------------|-------------|------------|------------|
| 1-7  | 16   | 38.2       | 67.4       | 73.1        | 78.4       | 67.0       |
| 1-7  | 4    | 38.8       | 67.1       | 74.2        | 78.3       | 67.1       |
| 1-7  | 3    | 37.0       | 65.7       | 74.1        | 77.7       | 67.3       |
| 1-7  | 2    | 35.3       | 64.9       | 72.9        | 76.1       | 65.4       |
| 1-13 | 16   | 43.9       | 74.6       | 68.5        | 78.8       | 70.1       |
| 1-13 | 4    | 43.4       | 73.7       | 68.2        | 79.1       | 70.1       |
| 1-13 | 3    | 42.2       | 74.2       | 68.0        | 78.7       | 70.5       |
| 1-13 | 2    | 39.7       | 72.1       | 66.6        | 77.6       | 68.9       |
| 1-30 | 16   | 46.7       | 75.4       | 68.4        | 81.0       | 72.6       |
| 1-30 | 4    | 46.7       | 75.4       | 69.9        | 81.0       | 73.3       |
| 1-30 | 3    | 47.8       | 75.0       | 70.0        | 80.4       | 73.6       |
| 1-30 | 2    | 44.0       | 72.7       | 72.8        | 78.7       | 71.7       |
| 1-65 | 16   | 47.0       | 75.3       | 82.3        | 81.5       | 77.2       |
| 1-65 | 4    | 46.8       | 74.5       | 82.8        | 81.4       | 76.6       |
| 1-65 | 3    | 46.8       | 75.3       | 83.0        | 81.3       | 75.9       |
| 1-65 | 2    | 44.4       | 74.2       | 83.1        | 80.4       | 75.7       |

### A.1.3 Lookup-Only Codes

<span id="page-15-0"></span>Table 15: Wikitext2 and C4 perplexity (↓), ctx. 4096, QTIP with a size 2 <sup>14</sup> LUT codebook. This codebook is too large (32KB) for current GPU L1 caches, but could fit on near-future hardware.

|      |    |      | ∼4 Bit |      |      | ∼3 Bit |      |                                                      | ∼2 Bit |      |      |
|------|----|------|--------|------|------|--------|------|------------------------------------------------------|--------|------|------|
|      |    |      |        |      |      |        |      | FP16 QTIP QuIP# AQLM QTIP QuIP# AQLM QTIP QuIP# AQLM |        |      |      |
| 2-7  | W2 | 5.12 | 5.16   | 5.19 | 5.21 | 5.30   | 5.41 | 5.46                                                 | 5.89   | 6.19 | 6.64 |
|      | C4 | 6.63 | 6.68   | 6.75 | 6.75 | 6.86   | 7.04 | 7.08                                                 | 7.78   | 8.16 | 8.56 |
| 2-70 | W2 | 3.12 | 3.15   | 3.18 | 3.19 | 3.26   | 3.35 | 3.36                                                 | 3.77   | 3.91 | 3.94 |
|      | C4 | 4.97 | 4.99   | 5.02 | 5.03 | 5.07   | 5.15 | 5.17                                                 | 5.55   | 5.71 | 5.72 |

Here, we use a pure-lookup code ∼ N (0, 1) with L = 14, V = 1, T<sup>x</sup> = 32, T<sup>y</sup> = 8, and QuIP#'s fine-tuning scheme. These parameters show what performance QTIP could achieve if we did not care about fast inference *today*. Specifically, a pure-lookup codebook is tunable, and setting T<sup>y</sup> = 8 reduces the BlockLDLQ group size while maintaining high dimensionality (256). This codebook uses 32KB; this only fits in GPU L1 cache with bank conflicts. Setting T<sup>x</sup> = 32, T<sup>y</sup> = 8 corresponds to using a larger MMA tile size than current GPUs allow for. The largest tile size is usually 16 in the T<sup>x</sup> dimension, meaning that a 32 × 8 trellis needs two tiles. Thankfully, hardware required to serve

Table 16: Wikitext2 and C4 zeroshot accuracy (↑), QTIP with a size 2 <sup>14</sup> LUT codebook. This codebook is too large (32KB) for current GPU L1 caches, but could fit on near-future hardware.

|      | Bits | ArcC (acc) | ArcE (acc) | BoolQ (acc) | PiQA (acc) | Wino (acc) |
|------|------|------------|------------|-------------|------------|------------|
| 2-7  | 16   | 40.0       | 69.3       | 71.0        | 78.5       | 67.3       |
| 2-7  | 4    | 40.3       | 69.2       | 73.0        | 78.1       | 67.5       |
| 2-7  | 3    | 39.1       | 69.3       | 69.6        | 77.8       | 66.3       |
| 2-7  | 2    | 37.0       | 64.6       | 67.2        | 75.6       | 66.9       |
| 2-70 | 16   | 51.1       | 77.7       | 76.6        | 81.1       | 77.0       |
| 2-70 | 4    | 50.1       | 77.5       | 76.4        | 81.3       | 77.3       |
| 2-70 | 3    | 50.6       | 77.9       | 78.0        | 81.1       | 76.1       |
| 2-70 | 2    | 47.1       | 76.9       | 79.5        | 80.1       | 76.3       |

such a model quickly is likely only a few years away, as these parameters are only slightly outside of what today's hardware is capable of.

Table [15](#page-15-0) shows that QTIP outperforms both QuIP# and AQLM at all compression ratios, with 3 bit QTIP achieving similar quality as 4 bit AQLM. While it is not fair to compare this QTIP setup with QuIP#, since QuIP# was designed for fast inference, we note that AQLM's VQ codebook uses 2 <sup>16</sup> × 8 × 2 = 1 MiB. This is 32 times larger than the QTIP codebook here, and would require 32 MiB of L1 cache to read from without bank conflicts. Not only is this orders of magnitude larger than current L1 caches (256KB on the H100), it is even larger than many L2 caches!

### A.1.4 Decoding Speed on Different GPUs

Table 17: Decoding speed on different Ampere and Lovelace GPUs.

| GPU Model        | Model | 2-bit tok/s | 3-bit tok/s | 4-bit tok/s | FP16 tok/s |
|------------------|-------|-------------|-------------|-------------|------------|
| RTX 3090         | 2-7   | 127         | 119         | 109         | 52.5       |
| RTX 3090         | 2-70  | 15.3        | OOM         | OOM         | OOM        |
| RTX A6000 Ampere | 2-7   | 116         | 106         | 95          | 43.5       |
| RTX A6000 Ampere | 2-70  | 15.0        | 13.1        | 11.7        | OOM        |
| RTX 6000 Ada     | 2-7   | 188         | 161         | 140         | 55.9       |
| RTX 6000 Ada     | 2-70  | 23.5        | 19.1        | 16.3        | OOM        |

### A.2 QTIP with BlockLDLQ

Here, we detail how we use TCQ within BlockLDLQ to produce our experimental setup. Essentially, QTIP is used as a high dimensional TxT<sup>y</sup> quantizer within BlockLDLQ and is a drop-in replacement for vector quantization in BlockLDLQ. The regular blockLDLQ step Q(W + (W − Wˆ )A) is exactly the same, and the only difference is in how Q rounds. Instead of rounding each row of x = W +(W − Wˆ )A independently, it groups T<sup>x</sup> rows into a block to round as m/T<sup>x</sup> high-dimensional sequences.

### A.3 Implementation Details

### A.3.1 Code

Our code is available at <https://github.com/Cornell-RelaxML/qtip>.

