# *A. A Brief Background on GPU Architecture*

GPU Architecture. Throughout this paper, we use terminology from the NVIDIA ecosystem of GPU software and hardware. As shown in Figure 1, GPUs comprise several Graphics Processing Clusters (GPCs), which share an L2 cache and are themselves composed of several Texture Processing Clusters (TPCs). A TPC comprises two Streaming Multiprocessors (SMs). The recent NVIDIA Blackwell B200 GPU has 9 GPCs comprising 74 TPCs, for a total of 148 SMs. Each SM runs at the same clock frequency. Blackwell additionally brings faster frequency scaling, with transition latencies of ≈10–100 µs compared with Hopper's ≈10–100 ms [55]. By default, the GPU runs at maximum frequency, scaling down when power exceeds the Thermal Design Power (TDP).

GPU Programming. GPU kernels are structured as a computational grid of many thread blocks, both of which have dimensions specified by the programmer. A kernel's work is divided among the thread blocks, which are executed on SMs and consist of multiple SIMD threads.

