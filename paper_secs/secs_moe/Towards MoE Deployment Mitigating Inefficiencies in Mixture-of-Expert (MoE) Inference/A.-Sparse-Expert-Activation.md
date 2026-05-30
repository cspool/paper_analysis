# *A. Sparse Expert Activation*

Our investigation in the expert activation pattern shows that although, in every batch, there exists some experts that are inactive, all experts have been activated a few times across time and batches. Pruning out experts that are not frequently active can potentially hurt model accuracy. However, we can offload the less frequently accessed experts to CPU memory and use the GPU memory for hot and active experts.

We propose the expert buffering mechanism to exploit expert sparsity and reduce static memory allocation. Figure [11](#page-8-1) illustrates the mechanism, which reduces static memory consumption by offloading expert parameters to CPU memory. Since CPU is much slower than GPU for matrix multiplication, we only use CPU memory to hold the experts but do not offload the computation. We use GPU memory to cache active experts and perform computation.

