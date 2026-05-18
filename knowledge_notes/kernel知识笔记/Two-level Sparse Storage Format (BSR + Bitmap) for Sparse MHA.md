## Two-level Sparse Storage Format (BSR + Bitmap) for Sparse MHA

术语是什么？通过联网搜索让回答具体和精准。

Two-level Sparse Storage Format是STOF提出的用于在GPU MHA kernel中统一表示任意sparse attention mask的两级存储格式，结合Block Compressed Sparse Row (BSR)和bitmap。该格式将mask矩阵按两级抽象组织：OuterTile (OT) 级——每个OT为64个8×8 InnerTile (IT)，OT作为coarse-grained skipped block单元；InnerTile (IT) 级——每个IT为8×8元素块，内部64个元素的mask pattern用一个uint64 bitmap_mask值精确表示。OT被分为"full"（内部所有IT均非空）和"part"（内部至少一个IT含mask元素但非全满）两类。存储结构由6个数组组成：full_row_ptr/full_col_idx（CSR格式表示full OTs位置）、part_row_ptr/part_col_idx（CSR格式表示part OTs位置）、load_row_ptr/load_col_idx（将full和part OTs按row-major合并的统一加载索引）、bitmap_mask（每个part OT对应64个uint64值，每个表示一个8×8 IT的mask pattern）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Two-level storage format在block-wise MHA kernel中的使用流程（STOF Algorithm 1）：

```
// ===== Mask Preprocessing (offline) =====
// Input: mask matrix M ∈ {0,1}^{seq_len × seq_len}
// OT_Size_M = OT_Size_N = 64 (64个8×8 IT = 512×512 elements per OT)
//
// 1. Partition M into OT grid
// 2. For each OT: if all 64 ITs non-empty → "full"
//               elif at least one IT non-empty → "part"
//               else → skip (empty)
// 3. Build arrays: full_row_ptr, full_col_idx, part_row_ptr, part_col_idx,
//    bitmap_mask[part_ot_idx][64] (uint64 per IT)
//    load_row_ptr/load_col_idx = merge of full + part

// ===== Kernel Runtime =====
for i in [0, ceil(seq_len / OT_Size_M)):        // Row-Parallel Dimension
    Q_i = Load_from_HBM(Q_HBM_i)                  // Q_i in registers (resident)
    load_num = load_row_ptr[i+1] - load_row_ptr[i]
    part_num = part_row_ptr[i+1] - part_row_ptr[i]
    tmp_part_idx = 0
    
    for kv_idx in [0, load_num):
        j = load_col_idx[load_row_ptr[i] + kv_idx]
        K_Tj = Load_from_HBM(K_HBM_j)
        V_j = __async_memcpy(Load_from_HBM(V_HBM_j))  // async: overlap with GEMM
        
        P_ij = Compute_GEMM(Q_i, K_Tj)  // register × SMEM → register
        
        // Fine-grained mask for "part" OTs
        if tmp_part_idx < part_num and j == part_col_idx[part_row_ptr[i] + tmp_part_idx]:
            Apply_Mask(P_ij, bitmap_mask[tmp_part_idx])  // uint64[64] per part OT
            tmp_part_idx++
        
        S_ij, alpha = Softmax(P_ij)       // Online Softmax with scaling
        O_i = O_i × alpha + Compute_GEMM(S_ij, V_j)
    
    result_HBM = Write_back_to_HBM(O_i)
```

8×8 IT设计理由：正好匹配NVIDIA Tensor Core的mma.m16n8k16操作粒度；64个elements恰好填满一个uint64（每element 1 bit），bitwise操作（POPC/OR/shift）可直接用于mask判断无额外memory lookup。IT列主序存储消除bank conflict（同warp内线程访问不同列映射到不同bank），OT行主序支持Softmax的iterative row-wise reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF以CUDA/C++实现（约2,500 LOC），基于FA2的CuTe结构扩展。预处理在Python端将mask转换为6个数组通过torch::Tensor传入GPU。BSR格式是NVIDIA cuSPARSE原生支持的稀疏矩阵格式（提供cusparseXcsr2bsrNnz等API）。bitmap_mask存储为uint64一维数组。该格式的通用性：causal、sliding window、Longformer、Bigbird等任意mask pattern均可用同一套数据结构和kernel处理——仅预处理不同。相比之下FlashMask的column-range数组仅支持column-continuous mask。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

