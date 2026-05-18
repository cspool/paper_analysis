## Block-Diagonal Causal Mask (块对角因果掩码)

术语是什么？通过联网搜索让回答具体和精准。
Block-Diagonal Causal Mask是DFVG的TreeSort-Verify机制将tree-based speculative decoding的irregular causal attention mask转换后的矩阵形式。传统token tree的attention mask因树结构不规则而呈sparse pattern（如SpecInfer中的topology-aware mask），导致GPU memory access不规整。TreeSort-Verify通过path-packing重排序使mask变为block-diagonal lower triangular——即矩阵由K个沿对角线排列的稠密lower triangular子块组成，块间为零。每个block内部是标准因果attention mask，可直接调用cuBLAS高度优化的GEMM kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block-Diagonal Causal Mask的构建和使用：
```
// 原始token tree T，节点按ancestor关系排序后:
// T_sorted = [t_π(1), t_π(2), ..., t_π(n)]
//      其中π满足: t_i是t_j的ancestor ⇒ π(t_i) < π(t_j)

// 排序后causal mask M_reordered具有block-diagonal结构:
//
//     B1  │      │      
//     ────┼──────┼──────
//         │  B2  │      
//     ────┼──────┼──────
//         │      │  B3  
//
// 其中每个Bi是mi×mi的下三角全1矩阵（标准causal mask）
// Bi之间可能有零或少量cross-block dependency（由tree topology决定）

// Block-Diagonal在GPU attention中的使用:
for each block B_k in parallel:
    Q_k = Q[T_sorted[start_k : end_k]]
    K_k = K[T_sorted[start_k : end_k]]
    V_k = V[T_sorted[start_k : end_k]]
    // 标准attention计算（无稀疏mask overhead）
    attn_k = softmax(Q_k × K_k^T / √d) × V_k  // 完全在cuBLAS GEMM中

// 跨block零区域天然skip，无额外masking开销
// 最终recombine按原始tree index顺序
```
关键特性：(1) 密度→每个block内部mask全为1（lower triangle），无sparsity overhead；(2) 规整性→每个block shape对齐cuBLAS tile size偏好；(3) 并行性→K个block可在GPU不同SM上并行执行。与FlashInfer的融合prefill+decode attention类似，通过mask结构变换将irregular pattern转为硬件友好形式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在DFVG中，Block-Diagonal Causal Mask通过path-packing排序自动产生（无需手动构造mask矩阵）。排序后连续block内的tokens天然具有标准因果依赖（ancestor在前，descendant在后，但同block内关系简单）。实现为GPU host code中的预处理步骤（token tree→重排序→block划分），开销极低（token数少，通常≤64）。该技术可推广至任何需要tree-structured attention的场景。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

