# VII. CONCLUSION

We propose MECLA, which is, to our knowledge, the first memory-compute-efficient accelerator for cloud-to-edge LLM inference. We first propose the SSMP matrix partition method, which splits a large parameter matrix into two tinyscale source sub-matrix and derived sub-matrix, and the latter can be obtained by scaling the former. Also, we propose a finetuning method to turn a pre-trained LLM into SSMP format. Then, we propose an efficient hardware processor, MECLA, to exploit the PSum reuse to save computation effort. MECLA effectively reduces the memory footprint and computation by 69.2-83.6% and 65.5-72.2% on an average of 20 benchmarks. Compared to SOTA accelerators, it is 113.4×, 12.99×, and 1.62× more energy saving compared to V100 GPU, SpAtten, and FACT.

