# 5 Conclusion

We present QTIP, a weight-only post-training quantization algorithm that achieves state-of-theart results through the use of trellis-coded quantization (TCQ). TCQ enables tractable ultra-high dimensional quantization, significantly reducing quantization distortion over vector quantization (VQ). However, naive TCQ does not admit fast inference due to sequential bottlenecks during decoding and needing to store a large codebook. QTIP solves this problem through a novel combination of incoherence processing, the hardware-efficient bitshift trellis, and fast computed codes. Specifically, QTIP introduces a series of compute-based pseudorandom Gaussian codes that, when used in

<span id="page-9-0"></span>Table 7: QTIP vs. QuIP#, Llama 3 (ctx. 8192 for perplexity). The 70B results quantize layer 0 v, which catastrophically degrades zeroshot performance without special prefix tokens (e.g. \n). This can be remedied by prepending the prompt with \n, applying the chat template as in Table 8, or not quantizing 0 v (see [\[35\]](#page-13-5)). Regardless, the lower distortion of TCQ in QTIP improves over the lowdimensional VQ in QuIP#.

|            |      |      | 3-70 PPL (↓) |      |      | 3-70 ZEROSHOT ACC (↑)     |           |      | 3-8 PPL (↓) |      |      | 3-8 ZEROSHOT ACC (↑)      |           |
|------------|------|------|--------------|------|------|---------------------------|-----------|------|-------------|------|------|---------------------------|-----------|
| MTHD. BITS |      | W2   | C4           |      |      | ARCC ARCE BOOLQ PIQA WINO |           | W2   | C4          |      |      | ARCC ARCE BOOLQ PIQA WINO |           |
| BF16       | 16.0 | 2.59 | 5.78         | 60.5 | 86.9 | 85.3                      | 82.4 80.3 | 5.54 | 7.10        | 50.2 | 80.1 | 81.0                      | 79.7 72.9 |
| QUIP#      | 4.00 | 2.99 | 5.96         | 35.0 | 67.3 | 84.7                      | 71.9 76.7 | 5.81 | 7.32        | 50.2 | 79.7 | 81.3                      | 79.7 73.1 |
| QTIP       | 4.00 | 2.75 | 5.83         | 56.1 | 83.9 | 85.8                      | 81.3 80.6 | 5.67 | 7.20        | 50.2 | 79.6 | 79.5                      | 79.4 73.4 |
| QUIP#      | 3.00 | 3.59 | 6.18         | 31.1 | 36.6 | 85.7                      | 58.8 76.4 | 6.27 | 7.71        | 46.4 | 77.4 | 79.9                      | 77.9 72.9 |
| QTIP       | 3.00 | 3.18 | 5.98         | 48.6 | 77.8 | 85.0                      | 77.8 79.7 | 6.01 | 7.48        | 49.2 | 79.3 | 80.0                      | 79.2 74.5 |
| QUIP#      | 2.00 | 5.77 | 7.46         | 18.3 | 32.2 | 82.1                      | 54.7 68.9 | 7.84 | 9.06        | 39.2 | 72.9 | 76.6                      | 75.6 68.2 |
| QTIP       | 2.00 | 4.97 | 6.80         | 28.0 | 35.2 | 83.6                      | 57.1 72.6 | 7.33 | 8.62        | 44.2 | 75.2 | 76.7                      | 77.6 70.7 |

<span id="page-9-1"></span>Table 8: Llama 3.1 instruct-tuned model results (ctx. 8192 for perplexity). QTIP performs well at all model sizes and generally outperforms PV-Tuning, a recent quantization method that focuses on finetuning algorithms. The zeroshot results in this table use LM Eval 0.4.4 and the "standard" versions of each task instead of the Meta versions in [\[25\]](#page-12-13).

|                |            |                  | PPL. (↓)<br>ZEROSHOT (↑) |      |      |                      |      |
|----------------|------------|------------------|--------------------------|------|------|----------------------|------|
|                |            | BITS             | W2                       |      |      | ARCC ARCE HSWAG PIQA |      |
|                | META "FP8" | 16 ATTN. / 8 MLP | 1.70                     | 61.6 | 81.4 | 67.1                 | 83.8 |
|                | QTIP       | 4                | 1.79                     | 61.3 | 80.9 | 66.7                 | 84.2 |
| 3.1 405B INST. | QTIP       | 3                | 2.05                     | 61.5 | 81.4 | 66.8                 | 83.5 |
|                | QTIP       | 2                | 3.29                     | 60.7 | 81.1 | 65.4                 | 82.2 |
|                | BF16       | 16               | 3.52                     | 56.7 | 75.6 | 61.5                 | 82.8 |
|                | QTIP       | 4                | 3.73                     | 56.3 | 75.8 | 61.4                 | 83.0 |
| 3.1 70B INST.  | QTIP       | 3                | 4.12                     | 55.1 | 75.1 | 60.8                 | 82.6 |
|                | QTIP       | 2                | 5.08                     | 54.4 | 72.6 | 59.4                 | 82.5 |
|                | PV-TUNING  | 2.01             | 5.70                     | 52.7 | 72.2 | 60.2                 | 82.6 |
|                | BF16       | 16               | 6.50                     | 51.6 | 77.8 | 57.7                 | 80.0 |
|                | QTIP       | 4                | 6.61                     | 50.7 | 78.0 | 57.5                 | 80.1 |
| 3.1 8B INST.   | QTIP       | 3                | 6.80                     | 50.4 | 77.7 | 56.9                 | 79.3 |
|                | QTIP       | 2                | 7.82                     | 45.1 | 75.6 | 54.5                 | 79.0 |
|                | PV-TUNING  | 2.07             | 8.45                     | 46.2 | 75.4 | 54.4                 | 78.7 |

<span id="page-9-2"></span>Table 9: Llama 3.2 instruct-tuned results when quantizing to 4 bits (ctx. 8192 for perplexity). Even on extremely small models, QTIP is still able to achieve meaningful compression without sacrificing quality. This table uses the same LM Eval setup as Table [8.](#page-9-1)

|    |      |           | PPL (↓) |      |      | ZEROSHOT (↑) |      |
|----|------|-----------|---------|------|------|--------------|------|
|    |      | SIZE (GB) | W2      | ARCC | ARCE | HSWAG        | PIQA |
| 3B | BF16 | 6         | 9.58    | 43.3 | 74.3 | 52.2         | 75.7 |
|    | QTIP | 2.1       | 9.77    | 43.5 | 74.3 | 51.9         | 75.1 |
| 1B | BF16 | 2.4       | 11.57   | 36.0 | 68.5 | 45.2         | 74.2 |
|    | QTIP | 0.97      | 11.93   | 34.8 | 68.4 | 44.5         | 73.3 |

conjunction with the bitshift trellis and incoherence processing, simultaneously achieves state-ofthe-art PTQ quality and fast inference. QTIP improves quantization quality at all tested bitrates over the latest VQ-based PTQ methods, QuIP# and AQLM, further pushing the boundary of LLM PTQ. QTIP's codes use as few as 2 instructions per weight during decoding, enabling matrix-vector multiplication to run at over 80% of peak memory bandwidth on modern GPUs. Altogether, our results indicate that high dimensional quantization is necessary for high-quality compression, and QTIP is the first LLM PTQ method to scale to ultra-high dimensions while supporting fast inference.

