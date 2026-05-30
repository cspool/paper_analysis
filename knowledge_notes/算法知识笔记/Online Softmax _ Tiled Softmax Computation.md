## Online Softmax / Tiled Softmax Computation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Online Softmax（在线softmax，又称tiled softmax或streaming softmax）是一种允许在不一次性访问全部输入数据的情况下精确计算softmax的增量算法。源于Milakov & Gimelshein (2018)的"Online normalizer calculation for softmax"。标准safe softmax需要两次遍历：第一次找全局最大值m，第二次计算$\exp(x_i-m)$并求和，再归一化。Online softmax通过维护running state $(m, \ell)$（running max和running sum）在一次遍历中完成计算，每接收一个data block时更新状态：$m' = \max(m_A, m_B)$，$\ell' = \ell_A \cdot \exp(m_A-m') + \sum \exp(x_B - m')$。关键在于softmax对输入平移不变（$x - c$不改变结果），online版本不断将"参考坐标系"（最大值）重设并施加代数修正。最终结果与标准两次遍历完全等价，非近似。FlashAttention利用online softmax实现block-wise attention计算：每个(i,j) block pair的softmax计算仅需维护当前row的$(m_i, \ell_i)$状态，使得N×N attention矩阵永远不需要整体存在。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Online softmax在FlashAttention的tiled attention中的核心流程（per query row）：
```
# 状态初始化（每query row i维护）
m_i = -inf           # running max of attention scores
l_i = 0              # running sum of exp(scores - running max)
O_i = 0              # running weighted sum of V (output accumulator)

# 对每个KV block j（按序处理）：
S_ij_block = Q_i @ K_j.T           # [B_r, B_c] attention scores
m_ij = rowmax(S_ij_block)          # block local max per query row
m_new = max(m_i, m_ij)             # 更新global max
# 重缩放旧累加值（补偿max变化）：
l_i = l_i * exp(m_i - m_new)       # rescale old exp-sum
O_i = O_i * exp(m_i - m_new)       # rescale old output
# 添加新block的贡献：
P_ij = exp(S_ij_block - m_new)     # [B_r, B_c] unnormalized softmax
l_i += rowsum(P_ij)                # update exp-sum
O_i += P_ij @ V_j                  # accumulate weighted V
m_i = m_new                        # update running max
# 处理完所有KV blocks后：
O_i = O_i / l_i                    # 最终归一化得到exact softmax output
```
该算法的正确性保证了最终O_i与标准两次遍历softmax完全一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Online softmax已在FlashAttention系列（v1-v4）、FlashInfer、xFormers等GPU attention库中广泛实现。实现细节：(1) 使用base-2 scaling（exp2替代exp）以利用硬件MUFU.EX2指令和与FFMA（fused multiply-add）的编译器融合；(2) 处理全mask行：当rowmax为-inf时替换为0避免NaN；(3) 反向传播时，存储前向的LogSumExp $L_i = m_i + \ln(\ell_i)$（每query row一个scalar），反向在SRAM中重计算$P_{ij} = \exp(S_{ij} - L_i)$来求梯度。Flash-D (2025)进一步提出用sigmoid替代softmax division，完全消除max subtraction步骤。Online softmax的流式处理模式不仅限于softmax——LayerNorm/RMSNorm、running statistics (Adam/RMSProp)等也可用类似模式实现tiled/streaming计算。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

**MetaAttention 的 RowNorm Online 泛化**：MetaAttention 将 online softmax 思想泛化为通用的 RowNorm Online 接口，支持任意 row-wise normalization（softmax、sigmoid、ReLU norm、RetNet reduceAbsSum norm 等），而非仅局限于 softmax。该接口定义为三段式：(1) online_prologue——初始化归一化状态变量（如 row_max=-inf, row_sum=0 或 row_sum_wo_clamp=0）；(2) online_forward——对每个 block 更新归一化状态，计算 rescale factor 传给 aggregation 阶段用于修正已累积输出；(3) online_epilogue——最终归一化。在 MetaAttention 的 scheduling 中，RowNorm Online 产生的中间状态变量（row_max, row_sum 等）作为 IntermediateTensor 纳入调度（通常分配在 register），elementwise/scaling 操作被 SIMT fused，reduce 操作使用 intra-warp reduction。这使得 MetaAttention 能在一个框架内同时支持 parallel pattern（如 FlashAttention-like online softmax）和 recurrent pattern（如 chunk-parallel state update）的 online normalization。

**FlashAttention-2 的算法改进**：FlashAttention-2对online softmax做了两项关键tweak来减少non-matmul FLOPs：
1. **Un-scaled output maintenance**：FlashAttention v1在每次内迭代都做`O_i = diag(ℓ)^{-1} @ O_tilde` rescale。FlashAttention-2改为维护un-scaled output $\tilde{\mathbf{O}}^{(j)} = \operatorname{diag}(e^{m^{(j-1)}-m^{(j)}})\tilde{\mathbf{O}}^{(j-1)} + e^{\mathbf{S}^{(j)}-m^{(j)}}\mathbf{V}^{(j)}$，仅在所有KV blocks处理后一次性做`diag(ℓ)^{-1}` rescale得到最终O。消除每次迭代对已累积output的elementwise rescale（non-matmul operation）。伪代码对比：
```
# FlashAttention v1 (每次迭代rescale):
O_i = diag(ℓ)^{-1} @ (diag(exp(m_old-m_new)) @ (diag(ℓ_old) @ O_old) + P_tilde @ V_j)

# FlashAttention-2 (维护un-scaled, 最终rescale):
O_tilde = diag(exp(m_old-m_new)) @ O_tilde + P_tilde @ V_j
# ... 循环结束后:
O_i = diag(ℓ)^{-1} @ O_tilde
```
2. **仅存LogSumExp L**：FlashAttention-2反向仅需`L = m + log(ℓ)`（每行一个scalar），替代FlashAttention v1的(m, ℓ) pair。反向从L重建softmax denominator：$P_{ij} = \exp(S_{ij} - L_i)$。减少register压力和对non-matmul计算的需求。
