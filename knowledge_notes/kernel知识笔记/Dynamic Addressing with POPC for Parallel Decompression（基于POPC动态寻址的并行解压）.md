## Dynamic Addressing with POPC for Parallel Decompression（基于POPC动态寻址的并行解压）

术语是什么？通过联网搜索让回答具体和精准。
Dynamic addressing with POPC是ZipGEMM Decompressor的核心技术：在GPU warp内并行解压时，每个线程需要确定其负责元素在compact value buffer中的偏移量。由于TCA-TBE以bitmap存储每个元素的存储模式（1=压缩存储在PackedSignMantissa buffer，0=fallback存储在FullValue buffer），偏移量通过POPC（population count，即__popc() intrinsic）在spatial indicator mask上做并行prefix sum得出。例如线程i处理tile内位置2i的元素，需计算indicator mask中bits[0, 2i-1]内值为1的个数（compressed元素偏移）或值为0的个数（fallback元素偏移）。这种将非均匀索引转换为确定性SIMT-friendly前缀和的技巧，是TCA-TBE实现无分支并行解码的关键。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Decompressor的dynamic addressing过程（per thread, per FragTile）：
```
Input: thread lane_id l (0-31), spatial indicator M (64-bit), 
       bitmap B1/B2/B3 (each 64-bit), base_exp, H_buffer_ptr, L_buffer_ptr

For k in {0, 1}:  // two assigned elements a0, a1
    pos = 2*l + k  // global position in 8×8 tile
    mask = (1 << pos) - 1
    idx_H = __popc(M & mask)  // count 1s before pos = compressed elem count
    if (M >> pos) & 1:  // this element is compressed
        val = H_buffer[start_H + idx_H]  // read packed sign+mantissa
        c = (B3[pos]<<2) | (B2[pos]<<1) | B1[pos]  // reconstruct codeword
        exp = base_exp + c  // arithmetic recovery
        result[k] = MakeBF16(val.sign, exp, val.mantissa)
    else:  // this element is fallback
        idx_L = pos - idx_H  // count 0s = total positions - count 1s
        result[k] = L_buffer[start_L + idx_L]  // read full BF16
```
关键insight：idx_H（compressed count）和idx_L（fallback count）通过popc一次计算得出，idx_L = pos - idx_H（因为前pos个元素中非0即1）。这避免了两次popc调用，仅需一次__popc() + 一次整数减法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖NVIDIA GPU intrinsic：
- `__popc(unsigned int)`：统计32-bit word中1的个数，映射为单条POPC PTX指令
- `__shfl_sync()`：warp shuffle，用于跨线程共享prefix sum结果
- 该技术完全在register内操作，无shared memory往返
- 前提条件：tile大小≤64元素（对应64-bit bitmap）使单warp内popc足够；更大tile需分层prefix sum（先warp内reduction再跨warp）
该技术可推广至其他需要sparse/non-uniform decompression的场景（如sparse matrix decompression、bitmap-based activation sparsity）。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

