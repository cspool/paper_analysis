# VI. RELATED WORK

Several works have focussed on efficient software-based synchronization between threads of the same CUDA kernel for irregular applications [9], [16], [17]. Li et. al. [9] developed an approach for inter-thread synchronizations by reassembling the micro-instructions of shared memory atomic operations in an efficient manner. Kai et. al. [16] presented a hierarchical synchronization approach for irregular applications by synchronizing thread blocks using global memory and threads of a thread block using shared memory. Xu et. al. [17] present a

<span id="page-10-0"></span>TABLE V: EXECUTION TIMES OF TILESYNC WITH OPTI-MIZATIONS IN GPT-3'S MLP AND CONV2DTILESYNC IN RESNET FOR SMALLER GRID SIZES. *+W* AVOIDS THE WAIT-KERNEL. *+WR* ALSO REORDERS THE TILE LOADING. *+WRT* ALSO AVOIDS CUSTOM TILE ORDER.

(a) EXECUTION TIMES IN µs OF TILESYNC OF GEMM KERNELS IN GPT-3'S MLP WITH AND WITHOUT OPTIMIZATIONS FOR DIFFER-ENT BATCH SIZES.

| B    | TileSync |     |     |      |  |  |  |
|------|----------|-----|-----|------|--|--|--|
|      | Vanilla  | +R  | +WR | +WRT |  |  |  |
| 1–64 | 378      | 365 | 360 | 355  |  |  |  |

(b) EXECUTION TIMES µs OF CONV2DTILESYNC OF RESNET-38'S CONV2D WITH AND WITHOUT OPTIMIZATIONS FOR ALL CHANNELS AND SMALL BATCH SIZES.

| C   | B | Conv2DTileSync |     |     |      |  |  |
|-----|---|----------------|-----|-----|------|--|--|
|     |   | Vanilla        | +R  | +WR | +WRT |  |  |
| 64  | 1 | 50             | 45  | 41  | 37   |  |  |
| 128 | 1 | 60             | 56  | 50  | 45   |  |  |
| 256 | 1 | 65             | 61  | 56  | 51   |  |  |
|     | 1 | 100            | 94  | 89  | 85   |  |  |
| 512 | 4 | 135            | 128 | 120 | 115  |  |  |

lock design that uses lock stealing to avoid deadlocks. CO-CONET [\[8\]](#page-11-5) performs synchronization between computation and communication kernel to overlap the communication transfers with the computation. cuSync targets synchronization between threads of multiple CUDA kernels and provide abstraction to easily design several synchronization policies, both of these are missing from above mentioned works. Moreover, some works have focussed on hardware-supported synchronization primitives for inter-kernel threads. GLocks [\[2\]](#page-11-6) is the first hardware supported implementation for highly-contented locks using message passing. HQL [\[18\]](#page-12-6) is a hardware-accelerated fine-grained lock scheme for GPUs, which adds support for queuing locks in L1 and L2 caches and uses a customized communication protocol for faster lock transfer and reduced lock retries. ElTantway et. al. [\[5\]](#page-11-7) propose a hardware warp scheduling policy that reduces lock retries by de-prioritizing warps whose threads are waiting in their spin lock. They also propose a hardware mechanism for accurately detecting busywait synchronization on GPUs. Dalmia et. al. [\[3\]](#page-11-8) designed multi-level barrier and priority mechanisms for semaphores for GPU based synchronization primitives. cuSync is a software solution for synchronizing threads of multiple CUDA kernels and these hardware-supported mechanisms are complementary to cuSync.

Lingqi et. al. [\[19\]](#page-12-7) studied the performance and pitfalls of several CUDA synchronization methods for reduction operations. Sinclair et. al. [\[14\]](#page-12-8) presented a benchmark suite to measure the performance of synchronization primitives for different coherence protocols and consistency models.

Stream-K [\[10\]](#page-11-0) is a GeMM implementation that improves the utilization of SMs of a GPU by dividing the workload among all SMs. However, Stream-K is not straightforward to apply to computations other than GeMMs. In contrast, cuSync fits thread blocks of multiple kernels in each wave and is applicable to any tile based computations.

## VII. CONCLUSION

State-of-the-art ML models consist of thousands of individual computations that are executed on one or more GPUs. However, these models under-utilize the GPUs because individually each of these computations cannot completely utilize a GPU and these models largely consists of dependent computations. In this paper, we presented cuSync, a framework for finegrained synchronization of tiles of dependent computations. By synchronizing only, the dependent tiles, our framework allows concurrent execution of independent tiles, thus improving the utilization of GPU. Our experiments show that synchronizing computations of existing machine learning models using cuSync can reduce inference times of these models significantly.

## APPENDIX

The artifact [\[7\]](#page-11-9) contains cuSync CUDA implementation and scripts to reproduce all of our results. The artifact provides both a Dockerfile, which contains all prerequisites installed, and source code. Latest source code is available at [https://github.](https://github.com/microsoft/cusync) [com/microsoft/cusync.](https://github.com/microsoft/cusync) The artifact reproduces Figure [6,](#page-8-0) [7,](#page-9-0) and [8](#page-9-1) in Section [V.](#page-7-0)

System We executed our experiments on a NVIDIA DGX-2 system containing 8 NVIDIA Tesla V100 GPUs connected using NVLINK. Our experiments will run on any system with a GPU, however, the end-to-end inference results in Figure [8](#page-9-1) might not be reproducible on another system.

Extract Artifact Download the artifact from [\[7\]](#page-11-9) and extract the zip file.

```
unzip cusync-cgo-24.zip
cd cusync-cgo-24
```

