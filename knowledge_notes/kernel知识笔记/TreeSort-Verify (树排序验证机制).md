## TreeSort-Verify (树排序验证机制)

术语是什么？通过联网搜索让回答具体和精准。
TreeSort-Verify是DFVG提出的高效tree-based speculative decoding验证机制。传统tree-based验证需为每个token sequence维护复杂拓扑感知causal mask（irregular sparse pattern），导致GPU attention计算中memory access不规整，无法充分利用向量化计算能力。TreeSort-Verify通过path-packing对token tree节点重排序，将irregular causal mask转换为block-diagonal lower triangular矩阵形式，使tree attention计算分解为K个独立block的标准attention，每个block直接调用高度优化的cuBLAS GEMM kernel，显著提升GPU验证计算效率。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TreeSort-Verify的kernel级执行流程：
```
Input: token tree T = {t_1, t_2, ..., t_n}, draft tokens with positions

// Step 1: Path-Packing Reordering
π: T → {1,2,...,n}  // 定义排序函数
// 约束: 若t_i是t_j的ancestor，则π(t_i) < π(t_j)
// 效果: 所有ancestor在其所有descendant之前
// 实现: DFS/BFS遍历tree，按深度优先顺序分配全局index

// Step 2: 构建Block-Diagonal Causal Mask
// 重排序后的mask M_reordered:
// M[i,j] = 1 if π(t_j) ≤ π(t_i) AND t_j∈ancestors(t_i), else 0
// 性质: M_reordered是block-diagonal lower triangular

// Step 3: Block Decomposition
// 将重排序序列划分为K个连续block {B_1, B_2, ..., B_K}
// B_k = tokens with indices [start_k, end_k]
// 每个B_k内causal mask是标准lower triangular

// Step 4: Parallel Block Attention
Att_tree = ⊕_{k=1}^{K} Att_block(Q_Bk, K_Bk, V_Bk, M_Bk)
// M_Bk为标准lower triangular mask
// ⊕表示按原始index顺序recombine
// 每个block独立调用cuBLAS GEMM
```
效率来源：(1) block内标准causal mask → 直接调用cuBLAS，无需custom sparse kernel；(2) block-diagonal结构天然支持GPU SM间pipeline并行；(3) 连续block布局improve memory locality，compact KV-cache存储减少bandwidth waste。FPGA侧配套的Multi-Branch Mapping：shared prefix使多branch共享weight loading→Q×K^T复用prefix KV仅改loading address→S×V最后round accumulation归并。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TreeSort-Verify在DFVG的GPU verify path中实现（CUDA C++）。每轮iteration：FPGA通过PCIe发送token tree→GPU host code执行path-packing排序→按block划分→每个block launch cuBLAS GEMM→结果按原始index顺序recombine→acceptance decision。TreeSort-Verify在ablation中贡献2.21×→2.46× speedup（相比仅HW-Branch）。对比传统tree-based验证（SpecInfer的irregular mask），TreeSort-Verify消除sparse mask的memory divergence和vectorized computing underutilization。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

