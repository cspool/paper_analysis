# <span id="page-8-0"></span>5.3 Hardware Utilization

To better understand Quantix's performance gains, we analyze GPU hardware utilization for a single 12,288×12,288 linear layer on the L40 GPU using NVIDIA Nsight [\[30\]](#page-11-30).

Compute and Memory. Fig. [13a](#page-8-2) compares the compute and memory utilization of Quantix and the 16-bit cuBLAS baseline. The 16-bit baseline operates in a memory-bound regime for batch sizes up to 32, where its memory utilization exceeds 80%. By contrast, Quantix maintains a much more balanced resource utilization, exhibiting significantly higher compute utilization while keeping memory utilization substantially lower. This demonstrates that Quantix effectively avoids the "memory wall" that limits the baseline and leverages the GPU's compute capabilities more efficiently, especially at smaller batch sizes. However, Quantix's compute utilization does not increase at larger batch sizes due to the overhead of dequantization, as further discussed below.

ALU and Tensor. The compute utilization reported by Nsight aggregates the activity of arithmetic logic units (ALUs),

<span id="page-8-3"></span>![](_page_8_Figure_10.jpeg)

Figure 14. Performance of 2/4-bit quantization for the 4 linear layers of LLaMA-65B on L40.

Tensor Cores, and other functionalities such as branching and load/store operations. To assess actual computing usage, we profile ALU and Tensor Core utilization, as plotted in Fig. [13b.](#page-8-2) Both Quantix and the 16-bit baseline increasingly rely on Tensor Cores as batch size grows. The baseline incurs minimal ALU usage. By contrast, Quantix shows high ALU utilization for small batches (<32) due to dequantization, but then declines for larger batches. This drop is caused by register pressure from in-register dequantization: larger batches require more registers than an SM can provide, causing register spilling and stalling the ALUs.

Cache and Throughput. Fig. [13c](#page-8-2) shows the cache efficiency and overall throughput of Quantix and the 16-bit baseline. Quantix maintains a cache hit rate above 90% across all batch sizes, a key factor contributing to its high throughput. In contrast, the baseline's cache hit rate drops sharply with increasing batch size, falling to nearly 0%. Leveraging its advantages in compute utilization and memory-cache efficiency, Quantix consistently achieves higher throughput than the FP-16 baseline at all batch sizes.

