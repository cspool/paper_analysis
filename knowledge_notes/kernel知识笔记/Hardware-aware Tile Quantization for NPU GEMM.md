## Hardware-aware Tile Quantization for NPU GEMM

术语是什么？通过联网搜索让回答具体和精准。

Hardware-aware tile quantization是一种将fine-grained group quantization的weight layout重新排列以对齐矩阵乘法单元（如HMX）tile-level数据布局的技术。核心思想：先按矩阵单元期望的tile memory order重排权重，再在memory order上做group quantization，使quantization group的连续性与硬件tile的连续性一致。对HMX FP16：tile layout为32×32 column-major + tile内2-row permutation；在此order上做group size=32的4-bit量化等价于以2×16 tile片段为量化组。量化后通过group coalescing将8个group的INT4值合并为128-byte super-group（恰好填满一个HVX 1024-bit register），scale连续存放。相比conventional column-major group layout需要runtime scatter到TCM（开销极大），tile quantization使weight tile可连续写入TCM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Offline: Weight preparation
W = load_hf_weights()                   // [K, N], FP16
W_tile = permute_to_hmx_layout(W)       // tile级column-major + tile内2-row perm
for group g in range(num_groups):       // group size 32, 在tile memory order连续
    scale = max(abs(W_tile[g*32:(g+1)*32])) / 7.0
    W_q[g*32:(g+1)*32] = round(W_tile[...]/scale)

// Post-quant: Coalesce 8 groups → super-block
// 8 × 32 INT4 = 256 INT4 = 128B = 1 HVX register; scales 8 × 2B = 16B contiguous
for super_g in range(num_groups // 8):
    dst_q[super_g*128:(super_g+1)*128] = concat(groups 0..7 quantized)
    dst_s[super_g*16:(super_g+1)*16] = concat(groups 0..7 scales)

// Runtime: Dequantized GEMM kernel
for tile_idx in range(num_tiles):
    dma_load(W_q[tile_idx], TCM)       // DMA搬weight+activation tile入TCM
    dma_load(A_tile, TCM)
    for super_g in tile:
        W_fp16 = vlut16(W_q[super_g], LUT_INT4_TO_FP16)  // INT4→FP16
        scales = vlut16(const_indices, scale_LUT)          // broadcast 4 scales
        W_fp16 *= scales
    O_tile += hmx_mma(W_fp16, A_tile)  // HMX 32×32 tile MM, FP32 accum
    dma_store(O_tile, DDR)
```

相比conventional layout (column-major group quant + runtime scatter)，加速9.65-19.04×；相比仅HMX layout无coalesce，再加速1.82-3.45×；仅比no-dequantization上界慢~27%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 离线转换脚本：HuggingFace weight→HMX tile layout GGUF→llama-quantize (Q4_0/Q8_0)；(2) 不同NPU的tile layout不同，需为每种硬件定制permutation；(3) 通用性：AMX (Intel)、SME (ARM)等CPU矩阵单元也有类似tile layout，核心思想可迁移；(4) 精度：tile-group quant与conventional group quant的WinoGrande/MMLU/Wiki PPL差异远小于量化本身损失。开源实现见llama.cpp-npu。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

