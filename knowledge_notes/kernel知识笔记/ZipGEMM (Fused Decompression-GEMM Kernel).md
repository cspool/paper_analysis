## ZipGEMM (Fused Decompression-GEMM Kernel)

术语是什么？通过联网搜索让回答具体和精准。
ZipGEMM是ZipServ提出的fused decompression-GEMM CUDA kernel。它将TCA-TBE格式压缩权重的解压与Tensor Core矩阵乘法融合为单一kernel，实现"load-compressed, compute-decompressed"执行模型。权重从DRAM以压缩格式加载→在register file内解压→直接送入Tensor Core mma.m16n8k16指令计算，消除传统decoupled pipeline中intermediate global memory buffer的redundant memory traffic。ZipGEMM在RTX4090上达1.31× average/1.71× peak speedup over cuBLAS，L40S上达2.21× peak，是首个超越高度优化的cuBLAS Tensor Core GEMM的压缩推理kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ZipGEMM kernel伪代码（split-K tiling，per thread block）：
```
for k_tile in range(0, K, K_TILE):
    // Stage 1: Tile Loading (async)
    cp.async.load(compressed_weight_tile + activations → shared_mem)
    cp.async.wait_group<0>(); __syncthreads()

    // Stage 2: Warp-Level Decoding (per warp, 32 threads)
    for each assigned 8×8 FragTile:
        M = B1 | B2 | B3  // spatial indicator (64-bit)
        for i in {0,1}:  // two elements per thread
            pos = 2*lane_id + i
            if (M >> pos) & 1:  // compressed
                idx_H = popc(M & ((1<<pos)-1))  // dynamic addressing
                val = PackedSignMantissa[start_H + idx_H]
                codeword = (B3[pos]<<2) | (B2[pos]<<1) | B1[pos]
                exponent = base_exp + codeword  // implicit lookup
                bf16_val = MakeBF16(val.sign, exponent, val.mantissa)
            else:  // fallback
                idx_L = pos - idx_H  // complementary offset
                bf16_val = FullValue[start_L + idx_L]

    // Stage 3: Activation Register Transfer
    LDSM.M88(activation_tile → registers)  // layout matches mma requirement

    // Stage 4: Tensor Core Computation
    mma.m16n8k16(weight_regs, activation_regs, accum_regs)

    // Next k_tile iteration...
```
关键micro-architecture优化：
- LDGSTS.128 bypass L1 cache直接写shared memory
- cp.async + __syncthreads() barrier做tile double buffering
- Fine-level: slice-wise interleaving（Tensor Core算slice i时ALU load+decompress slice i+1）
- 全程无shared memory bank conflict（仅~4.7K触发，vs DietGPU百万级）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZipGEMM以CUDA C++实现（约2.5K行），编译为独立.so库（nvcc 12.4/12.8）。使用方式：
1. 编译：`mkdir build && cd build && cmake .. && make` → 生成libzipgemm.so
2. 调用：通过C++ API传入TCA-TBE格式压缩权重buffer、激活tensor、matrix dimensions (M,N,K)
3. 集成：PyBind11桥接到vLLM的linear layer execution
4. Profiling：Nsight Compute (NCU) 分析micro-architecture counters（DRAM read volume, ALU utilization, Tensor Core utilization, bank conflicts）
适用场景：memory-bound decode阶段效果最好（如batch 8-32的token generation）。compute-bound prefill阶段回退到decoupled pipeline以避免ALU overhead超过memory saving。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

