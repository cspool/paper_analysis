# 3 Algorithm Optimizations of MCBP

Based on the three challenges, we propose three corresponding optimization strategies: BRCR, BSTC, and BGPP. Fig. 6 depicts the overall execution flow of MCBP. Model weights are offline-compressed into a bit-level (BL) sparsity format (BSTC, §3.2). During inference, the BL-compressed weights are loaded and decompressed, then sent for GEMM acceleration (BRCR, §3.1), the BL KV cache are on-demand fetched to predict attention sparsity (BGPP, §3.3).

