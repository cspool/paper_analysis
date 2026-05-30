# **C** Efficiency

## <span id="page-16-0"></span>C.1 Complexity Analysis of Memory and Computation

**Memory Saving** As discussed in §3.1, we need to allocate memory for the KV cache budget  $M_{\text{budget}} \in \mathbb{R}^{b \times B_{\text{budget}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$  to retain  $B_{\text{budget}}$  KV cache tokens, and for the buffer  $M_{\text{buffer}} \in \mathbb{R}^{b \times B_{\text{buffer}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$  to store  $B_{\text{buffer}}$  newly generated KV cache tokens during the generation of a text segment. Here, b is the batch size,  $N_{\text{layer}}$  is the number of Transformer layers,  $N_{\text{head}}$  is the number of attention heads, and d is the dimension of attention heads. In addition, we also need to allocate memory for the model weight  $M_{\theta}$ . During decoding, the previous query states are typically discarded by default, so we use a query cache to store the last  $\alpha$  tokens in the query state, consuming memory of  $M_{\alpha} \in \mathbb{R}^{b \times \alpha \times N_{\text{layer}} \times N_{\text{head}} \times d}$ . In summary, R-KV requires memory of  $M_{\text{total}} = M_{\theta} + M_{\text{budget}} + M_{\text{buffer}} + M_{\alpha}$  during generation. In comparison to FullKV without KV cache compression, generating  $B_{\text{full}}$  tokens requires memory of  $M_{\text{full}} \in \mathbb{R}^{b \times B_{\text{full}} \times N_{\text{layer}} \times N_{\text{head}} \times D_{\text{head}}}$  to retain  $B_{\text{full}}$  KV tokens, and memory of the model weight  $M_{0}$ . Therefore, the memory saved by our method w.r.t. FullKV is:  $M_{\text{saving}} = M_{\text{full}} - M_{\text{budget}} - M_{\text{buffer}} - M_{\alpha}$ .

Computation Overhead The computational complexity of importance scoring (See §3.2) is  $O(\alpha B_{\mathrm{budget}})$  while redundancy estimation (see §3.3) has complexity  $O(B_{\mathrm{budget}}^2)$ . Thus, the total overhead incurred during each generation segment is  $O(\alpha B_{\mathrm{budget}} + B_{\mathrm{budget}}^2)$ . The generation complexity without KV cache compression is  $O(B_{\mathrm{full}}B_{\mathrm{buffer}})$ , whereas the complexity with KV cache compression is  $O((B_{\mathrm{budget}} + B_{\mathrm{buffer}})B_{\mathrm{buffer}})$ . For reasoning models,  $B_{\mathrm{full}}$  tends to be large because of the long generation length, and using a relatively small  $B_{\mathrm{budget}}$  value can efficiently reduce computation cost. The effectiveness of this approach depends on depends on whether the speedup gained by attending over a reduced KV cache outweighs the overhead of computing the compression scores—i.e., the combined cost of importance and redundancy scores,  $(O(\alpha B_{\mathrm{budget}}) + O(B_{\mathrm{budget}}^2))$ .

## <span id="page-16-1"></span>C.2 Detailed Analysis of Throughput Results

We analyze the end-to-end throughput from two perspectives: ratio budget and fixed budget.

**Ratio Budget:** section 4.2 indicates that for DeepSeek-R1-Distill-Llama-8B, lossless compression (i.e., model performance equivalent to no KV compression) is achievable when the KV budget ratio, relative to the output length, is between 10% and 34%. For DeepSeek-R1-Distill-Qwen-14B, this range for lossless compression is 25% to 54% of the output length. Consequentlywe investigated the maximum achievable batch size and corresponding throughput for R-KV at compression ratios of 10%, 34%, and 54%, comparing these against the maximum batch size and throughput of FullKV using DeepSeek-R1-Distill-Llama-8B. In 8K sequence length setting, at a 54% compression ratio, R-KV allows for a batch size  $1.7 \times larger$  than FullKV, resulting in  $1.5 \times the$  throughput. At a  $10\% to the compression ratio, R-KV achieves a <math>1.7 \times the compression$  ratio, at  $1.5 \times the$  throughput compared to FullKV. For a  $1.5 \times the$  sequence length setting, at  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression.

<span id="page-17-0"></span>

| Gen. Length | Method | Budget              | Mem. Saving (%) | Batch     | Throughput (tok/s) | Tokens Gen. | Dec. Time (s) |
|-------------|--------|---------------------|-----------------|-----------|--------------------|-------------|---------------|
|             | FullKV | -                   | -               | 1         | 75.44              | 8 094       | 107.30        |
|             |        | -                   | -               | 62 (max)  | 849.13             | 501 828     | 590.99        |
|             | SnapKV | Fixed - 1024        | 87.50           | 1         | 81.26              | 8 094       | 99.60         |
| 8K          |        | Fixed - 1024        | 87.50           | 402 (max) | 3 253.93           | 3 253 788   | 999.96        |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 525.25           | 2 322 978   | 919.90        |
|             |        | Fixed - 3072        | 62.50           | 150 (max) | 1 527.67           | 1 214 100   | 794.74        |
|             |        | Ratio - 10% - 819   | 90.00           | 479 (max) | 3 808.81           | 3 877 026   | 1017.91       |
|             |        | Ratio - 34% - 2785  | 66.00           | 167 (max) | 1 625.46           | 1 351 698   | 831.58        |
|             |        | Ratio - 54% - 4423  | 46.00           | 105 (max) | 1 269.68           | 849 870     | 669.36        |
|             | R-KV   | Fixed - 1024        | 87.50           | 1         | 80.46              | 8 094       | 100.60        |
|             |        | Fixed - 1024        | 87.50           | 402 (max) | 3 251.52           | 3 253 788   | 1 000.70      |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 525.75           | 6 546 972   | 919.72        |
|             |        | Fixed - 3072        | 62.50           | 150 (max) | 1 520.99           | 1 214 100   | 798.23        |
|             |        | Ratio - 10% - 819   | 90.00           | 479 (max) | 3 809.15           | 3 877 026   | 1017.82       |
|             |        | Ratio - 34% - 2785  | 66.00           | 167 (max) | 1 608.01           | 1 351 698   | 840.61        |
|             |        | Ratio - 54% - 4423  | 46.00           | 105 (max) | 1 257.83           | 849 870     | 675.66        |
|             | FullKV | _                   | _               | 1         | 69.41              | 16 286      | 234.65        |
| 16K         |        | -                   | -               | 30 (max)  | 347.03             | 488 580     | 1 407.89      |
|             | SnapKV | Fixed - 1024        | 87.50           | 1         | 81.03              | 16 286      | 200.99        |
|             |        | Fixed - 1024        | 87.50           | 402 (max) | 3 202.17           | 6 546 972   | 2 044.54      |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 449.02           | 4 674 082   | 1 908.56      |
|             |        | Fixed - 3072        | 81.25           | 150 (max) | 1 413.84           | 2 442 900   | 1 727.84      |
|             |        | Ratio - 10% - 1638  | 90.00           | 271 (max) | 2 306.26           | 4413506     | 1913.71       |
|             |        | Ratio - 34% - 5 570 | 66.00           | 82 (max)  | 798.42             | 1 335 452   | 1 672.61      |
|             |        | Ratio - 54% - 8 847 | 46.00           | 46 (max)  | 586.43             | 749 156     | 1 277.48      |
|             | R-KV   | Fixed - 1024        | 93.75           | 1         | 80.95              | 16 286      | 201.18        |
|             |        | Fixed - 1024        | 93.75           | 402 (max) | 3 188.82           | 6 546 972   | 2 053.10      |
|             |        | Fixed - 1536        | 90.63           | 287 (max) | 2 447.61           | 4 674 082   | 1 909.65      |
|             |        | Fixed - 3072        | 81.25           | 150 (max) | 1 406.28           | 2 442 900   | 1 737.13      |
|             |        | Ratio - 10% - 1638  | 90.00           | 271 (max) | 2 300.28           | 4413506     | 1918.68       |
|             |        | Ratio - 34% - 5570  | 66.00           | 82 (max)  | 797.43             | 1 335 452   | 1 674.70      |
|             |        | Ratio - 54% - 8847  | 46.00           | 46 (max)  | 584.77             | 749 156     | 1 281.12      |

Table 3: Memory-saving, throughput, and decoding-time comparison for LLAMA3-8B under various generation lengths and KV-cache compression budgets.

that of FullKV, and the throughput is  $1.7 \times$  higher. At 10% compression, R-KV supports a  $9 \times$  larger batch size, delivering  $6.6 \times$  the throughput. We observe that for smaller batch sizes (e.g., less than 128), throughput scales nearly linearly with increasing batch size. However, for larger batch sizes this linear scaling diminishes as inference on the NVIDIA A100 GPU becomes compute-bound.

**Fixed Budget:** We also conducted an analysis under a fixed KV cache budget. With an output length of 8K and a fixed budget  $B_{\rm budget} = 1024$ , R-KV enables a batch size  $6.48 \times$  larger than FullKV, yielding  $3.8 \times$  the throughput. At  $B_{\rm budget} = 1536$ , the batch size is  $4.6 \times$  larger, and throughput is  $3 \times$  that of FullKV. For an output length of 16K and  $B_{\rm budget} = 1024$ , R-KV achieves a  $13.4 \times$  increase in batch size and a  $9.19 \times$  increase in throughput. With  $B_{\rm budget} = 1536$ , the batch size is  $9.6 \times$  larger, and throughput is  $7.1 \times$  higher. In the fixed budget scenario, the advantage of R-KV becomes more pronounced with longer generation lengths. This is because the KV cache size for R-KV under a fixed budget does not increase with the sequence length, unlike FullKV where the memory footprint grows linearly with the generation length, thus more severely limiting its maximum batch size.

#### C.3 Results

Full results could be found at Table 3. While R-KV incurs a minor computational overhead for redundancy estimation compared with SnapKV, this results in a throughput that is only slightly lower, with a negligible difference of less than 1%.

### **D** Limitations

One limitation of our proposed KV cache compression method is its current compatibility with certain advanced attention mechanisms, such as paged attention. Adapting our compression technique to seamlessly integrate with such mechanisms presents a non-trivial challenge and may require further investigation. Additionally, the implementation of KV cache compression within existing serving frameworks can encounter practical difficulties, particularly if these frameworks lack native support or flexible interfaces for KV cache compression. In serving frameworks that do not offer specialized KV cache compression interfaces, the performance benefits of our method might be less pronounced. Without such interfaces, implementing KV cache compression may necessitate reallocating memory

to store the compressed KV cache and subsequently deallocating the memory used for the original, uncompressed cache. This process of memory reallocation can introduce significant overhead, potentially offsetting some of the acceleration gains. In contrast, serving frameworks equipped with dedicated KV compression interfaces can handle these operations much more efficiently, avoiding such costly memory management tasks.