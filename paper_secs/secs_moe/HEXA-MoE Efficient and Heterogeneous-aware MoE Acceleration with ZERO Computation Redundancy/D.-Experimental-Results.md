# **D.** Experimental Results

We provide the exact values for both memory footprint analysis and average latency analysis in Table 7 and 8, respectively. Specifically, for latency analysis, we provide the average value for each case with 0.5k, 1k, 1.5k and 2k total steps.

<span id="page-15-0"></span>Table 7. Memory analysis with Tutel, MegaBlocks and **HEXA-MoE** on Swin-Transformer-MoE benchmark (Base and Small). Experiments are conducted on 2 homogeneous GPUs with automatic mixed precision in PyTorch and batch size 40 for all the experiments. We set the number of global experts to 8, and record the average GPU memory footprint (GB) on each device.

|       | Method               | top-1 | top-2 | top-3 | top-4 | top-5 | top-6 | top-7 | top-8 |
|-------|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|       | Tutel                | 12.7  | 13.9  | 15.3  | 16.0  | 17.4  | 19.0  | 20.3  | 21.8  |
| Base  | MegaBlocks (MoE)     | 13.1  | 13.8  | 14.8  | 15.4  | 16.0  | 16.9  | 17.8  | 18.7  |
|       | MegaBlocks (dMoE)    | 12.9  | 13.7  | 14.9  | 15.6  | 16.7  | 17.8  | 18.6  | 19.7  |
|       | Ours (data-centric)  | 10.9  | 11.2  | 11.3  | 11.7  | 12.0  | 12.0  | 12.3  | 12.4  |
|       | Ours (model-centric) | 10.0  | 10.2  | 10.3  | 10.6  | 10.5  | 10.7  | 10.9  | 11.4  |
|       | Tutel                | 9.0   | 10.0  | 11.0  | 11.6  | 12.7  | 13.8  | 15.0  | 16.0  |
| Small | MegaBlocks (MoE)     | 9.2   | 9.8   | 10.2  | 10.8  | 11.2  | 11.8  | 12.5  | 13.0  |
|       | MegaBlocks (dMoE)    | 9.0   | 9.7   | 10.4  | 11.4  | 12.0  | 12.4  | 13.3  | 13.9  |
|       | Ours (data-centric)  | 8.1   | 8.3   | 8.2   | 8.5   | 8.6   | 8.7   | 9.0   | 9.2   |
|       | Ours (model-centric) | 7.7   | 7.8   | 7.8   | 8.0   | 7.9   | 8.2   | 8.3   | 8.5   |

<span id="page-16-0"></span>Table 8. Latency analysis for Tutel, MegaBlocks and HEXA-MoE on Swin-Transformer-MoE benchmark with base and small scale. Experiments are conducted on 4 homogeneous GPUs with 4 experts. We set different batch size (bs) for different models under different routing strategy to maximize the utilization of GPU memory. We record the average latency of one step (s) during training.

|       |              | Method               | 0.5k | 1k   | 1.5k | 2k   |              | 0.5k | 1k   | 1.5k | 2k   |
|-------|--------------|----------------------|------|------|------|------|--------------|------|------|------|------|
|       | 0            | Tutel                | 2.96 | 2.90 | 2.47 | 2.40 | 0            | 2.14 | 2.59 | 2.73 | 2.66 |
|       | top-1,bs=110 | MegaBlocks (MoE)     | 2.10 | 2.66 | 2.57 | 2.43 | top-2,bs=100 | 2.40 | 2.58 | 2.51 | 2.49 |
|       | -dc          | MegaBlocks (dMoE)    | 2.06 | 2.02 | 2.13 | 2.19 | p-2,l        | 2.47 | 2.72 | 2.63 | 2.55 |
| Base  | =            | Ours (model-centric) | 1.52 | 1.51 | 1.51 | 1.51 | t [          | 1.61 | 1.60 | 1.60 | 1.60 |
| В     |              | Ours (data-centric)  | 1.01 | 0.99 | 0.99 | 0.99 |              | 1.17 | 1.16 | 1.16 | 1.16 |
|       |              | Tutel                | 2.23 | 2.43 | 2.48 | 2.54 |              | 2.27 | 2.34 | 2.20 | 2.18 |
|       | top-3,bs=90  | MegaBlocks (MoE)     | 2.09 | 2.31 | 2.35 | 2.17 | )8=sq        | 2.11 | 2.18 | 2.05 | 2.03 |
|       | <br>pp-3,l   | MegaBlocks (dMoE)    | 2.55 | 2.53 | 2.44 | 2.41 | top-4,bs=80  | 2.00 | 2.19 | 2.23 | 2.12 |
|       | =            | Ours (model-centric) | 1.70 | 1.69 | 1.69 | 1.69 |              | 1.83 | 1.82 | 1.82 | 1.82 |
|       |              | Ours (data-centric)  | 1.32 | 1.31 | 1.31 | 1.30 |              | 1.42 | 1.41 | 1.41 | 1.41 |
|       | 0            | Tutel                | 3.59 | 3.63 | 3.62 | 3.01 | 0            | 2.83 | 2.96 | 2.72 | 2.53 |
|       | <br> S=14    | MegaBlocks (MoE)     | 3.41 | 3.64 | 3.72 | 3.23 | S=13         | 3.04 | 3.26 | 3.11 | 3.02 |
|       | top-1,bs=140 | MegaBlocks (dMoE)    | 3.51 | 3.58 | 3.76 | 3.47 | top-2,bs=130 | 2.25 | 3.10 | 3.01 | 2.87 |
| Small | 2            | Ours (model-centric) | 1.34 | 1.34 | 1.33 | 1.33 | =            | 1.49 | 1.49 | 1.48 | 1.48 |
| Sī    |              | Ours (data-centric)  | 0.69 | 0.68 | 0.68 | 0.68 |              | 0.87 | 0.86 | 0.86 | 0.85 |
|       | 0            | Tutel                | 2.23 | 2.92 | 3.17 | 3.28 | 0            | 3.09 | 2.86 | 3.03 | 3.00 |
|       | top-3,bs=120 | MegaBlocks (MoE)     | 2.63 | 2.79 | 2.92 | 3.07 | top-4,bs=110 | 3.04 | 3.09 | 3.01 | 2.91 |
|       |              | MegaBlocks (dMoE)    | 3.08 | 3.24 | 3.38 | 3.09 | p-4,k        | 3.03 | 3.19 | 3.18 | 2.91 |
|       | tc           | Ours (model-centric) | 1.61 | 1.60 | 1.59 | 1.59 | t t          | 1.67 | 1.67 | 1.68 | 1.68 |
|       |              | Ours (data-centric)  | 1.07 | 1.05 | 1.05 | 1.05 |              | 1.09 | 1.08 | 1.07 | 1.07 |