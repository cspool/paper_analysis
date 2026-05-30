# C. Additional Optimizations: QKV Caching for Multi-Scale Attention

A naive implementation of Multi-Scale Attention (MSA) would require recomputing Query, Key, and Value (QKV) projections for each window involved in cross-attention operations across different scales. Consider a window  $W^{(l)}$  at scale l performing cross-attention with windows at coarser scales  $\{W^{(l+1)},\ldots,W^{(L)}\}$  in the top-down pathway. In a naive implementation, the QKV for each window  $W^{(l)}$  would be recalculated for every cross-attention instance, even if the underlying feature representation of  $W^{(l)}$  remains unchanged. This repeated computation becomes increasingly inefficient as the number of scales and windows grows.

To overcome this challenge, we introduce a QKV cache mechanism within MSA. During both the top-down and bottom-up pathways, we maintain a cache at each scale l to store the QKV projections for all windows  $\{W_{ij}^{(l)}\}$ . When a window at scale l needs to perform cross-attention, it first queries this cache. If a valid QKV set for the current version of  $W^{(l)}$  is available, it is directly retrieved from the cache. The cache is updated only when the feature representation of a window at a given scale is modified. This occurs after self-attention at the coarsest scale L, and after each dense cross-attention operation in the top-down and parent cross-attention in the bottom-up pathways. By reusing QKV projections, our cache significantly accelerates MSA

in long sequences where cross-scale attention operations are frequent.

<span id="page-11-0"></span>Table 6. Comparison of vision models across different image resolutions. Each model has two rows: one for runtime (in minutes) and one for Top-1 accuracy (in %). We trained all models for 50 epochs for each resolution. We limited each experiment to a maximum runtime of 24hrs on an 8 × H100 GPU node and report "–" for experiments that could not be complete within our runtime limit.

| Model                            |                  | Runtime (min) ↓ |       |        | Top-1 Accuracy (%) ↑ |       |       |        |        |
|----------------------------------|------------------|-----------------|-------|--------|----------------------|-------|-------|--------|--------|
|                                  |                  | 256px           | 512px | 1024px | 2048px               | 256px | 512px | 1024px | 2048px |
| Transformer-Based                | ViT-B/16         | 18              | 51    | 247    | 3480                 | 63.68 | 72.60 | 69.42  | –      |
|                                  | WViT-B/16        | 18              | 44    | 137    | 638                  | 64.21 | 68.95 | 63.61  | 53.93  |
| Convolutional                    | ConvNext-B/16    | 66              | 237   | 955    | 3825                 | 78.84 | 75.94 | 67.50  | –      |
| Sparse-Transformer               | FasterViT-4      | 49              | 168   | 675    | 2400                 | 77.64 | 74.40 | 53.62  | –      |
|                                  | LongViT-B/16     | 39              | 116   | 442    | 2000                 | 55.20 | 51.88 | 45.32  | –      |
| Mamba-Based                      | MambaVision-B/16 | 21              | 56    | 197    | 750                  | 73.10 | 69.94 | 51.68  | 24.64  |
| Multi-Scale Attention Atlas-B/16 |                  | 25              | 54    | 198    | 786                  | 80.05 | 83.75 | 82.73  | 74.74  |