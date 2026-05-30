## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- baseline方法是什么？
  Baseline方法：(1) **标准Transformer MoE推理**（HuggingFace Transformers）：MoE层执行input permutation（tokens按routing结果重排到各expert tensor，产生额外内存分配memcpy）、各expert独立执行dense GEMM（cuBLAS）、output un-permutation（expert输出写回global memory后加权求和，产生额外I/O）。(2) **vLLM fused MoE kernel**：将多expert计算融合为单个kernel，消除部分permutation开销，但不利用权重稀疏性。(3) **VENOM（SOTA structured sparse）**：利用SpTC加速sparse-dense矩阵乘法，支持灵活V:N:M稀疏格式，但仅处理单端权重稀疏——当输入也稀疏时，跳过行导致I/O amplification和uncoalesced memory access（Figure 6中②③④格式），性能退化严重。
  全栈执行例子（Baseline: vLLM-DS + VENOM权重稀疏，单个token decode）：
  ```
  Prompt请求 → Transformer decoder layer
  ├─ Attention层：FlashAttention2（已优化）
  └─ MoE层（瓶颈，80%+时间）：
      ├─ Router: token → gating scores → select top-k experts
      ├─ Input Permutation: 创建per-expert tensor → memcpy tokens（GPU GMEM allocation+copy）
      ├─ Expert计算（VENOM kernel，单端稀疏）:
      │   - gate_proj: sparse(W_gate_VENOM) × dense(input) → dense output
      │   - up_proj: sparse(W_up_VENOM) × dense(input) → dense output  
      │   - SiLU(gate) * up（separate kernel launch）
      │   - down_proj: sparse(W_down_VENOM) × dense(hidden) → dense output
      │   ★ 问题：input tensors中大量零行（未路由token）仍参与计算
      │   ★ I/O amplification：SEL跳过列导致加载多余数据（②③），或非连续访问（④）
      ├─ Output Un-permutation: expert_outputs → GMEM → reload → weighted_sum（额外I/O roundtrip）
      └─ 结果：内存管理开销+冗余计算+非连续访存 → VENOM在高稀疏输入时性能退化
  ```
  Baseline缺陷：(1) 输入permutation/un-permutation产生额外GMEM分配和数据搬运；(2) 单端稀疏无法利用MoE路由产生的激活稀疏性；(3) VENOM的dual-side稀疏场景下存在I/O amplification和uncoalesced access；(4) 稀疏格式与SpTC硬件未充分对齐，导致硬件利用率不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Samoyeds系统——提出双端结构化稀疏数据格式+定制sparse-sparse kernel+四项系统优化。
  核心设计：(1) **双端稀疏格式（N:M:V）**：权重端=(M×V block内N个Sub-Row) × (2:4 element-wise per Sub-Row)；激活端=vector-wise稀疏通过SEL数组记录routing结果。数学等价于原始MoE计算。(2) **定制sparse-sparse kernel**：基于PTX mma.sp指令直接调用SpTC执行双端稀疏MMA，pipeline机制overlap fetch和compute。(3) **四项优化**：3-step hierarchical tiling、data stationary ($C_{IR}$中间寄存器shuffle)、data packing（metadata 2-bit→32-bit映射）、optimized layout（offline weight transpose + in-kernel input transpose + compressed output）。
  全栈执行例子（Samoyeds，同场景单个token decode）：
  ```
  Prompt请求 → Transformer decoder layer（Samoyeds优化后）
  ├─ Attention层：FlashAttention2（不变）
  └─ MoE层（Samoyeds优化）：
      ├─ Router: token → gating scores → 生成SEL（仅记录indices，无内存拷贝）
      ├─ Expert计算（Samoyeds kernel，双端稀疏+fused）:
      │   - 权重预编码（offline）：原始W → (data, indices, metadata) Samoyeds格式
      │   - Kernel执行（单个kernel覆盖gate+up+down+act+acc）:
      │       1. 加载SEL→SMEM，识别有效token columns
      │       2. Pipeline: 异步加载A_tile(编码权重), B_tile(仅有效token列), indices→SMEM
      │       3. 3-step tiling: block_tile(m_b×n_b)→warp_tile(m_w×n_w)→SpTC_tile(m16×n8×k32)
      │       4. ldmatrix按SpTC spec排列数据到register
      │       5. mma.sp: SpTC执行 M_sparse × N_sparse → P（硬件2:4加速）
      │       6. 每V/k_h步shuffle C register（data stationary，避免写回GMEM）
      │       7. gate_proj→SiLU→×up_proj→down_proj 全在kernel内fused
      │       8. Weighted accumulation fused: output += router_score * expert_C
      │       9. 压缩output写入GMEM（仅非零行）
      └─ 结果：无input permutation开销 + 无output roundtrip + SpTC双端加速
  ```
  对比baseline的关键改进：
  - 输入permutation消除 → 直接通过SEL在kernel内索引，零内存拷贝（对比baseline的GMEM alloc+memcpy）
  - 双端稀疏 → weight稀疏（2:4 SpTC加速 × N:M:V灵活稀疏比） + input稀疏（仅计算路由到的tokens），解决VENOM在dual-side稀疏时I/O amplification问题
  - Operator fusion → activation+weighted accumulation+matmul融合，消除多次kernel launch和中间GMEM roundtrip
  - Data stationary → $C_{IR}$中间寄存器避免C频繁写回，保持C在register中跨越Sub-Row边界
  - Packing + Layout → metadata对齐32-bit transaction，B矩阵转置packing coalesced access，offline weight transpose消runtime开销
  - 量化收益：kernel级up to 1.99× vs VENOM，模型级up to 1.58× vs vLLM-DS，最大batch size 4.41× average boost
