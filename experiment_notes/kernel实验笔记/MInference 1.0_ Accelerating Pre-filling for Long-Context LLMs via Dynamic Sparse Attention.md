## MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了三种针对不同稀疏注意力模式的优化 GPU kernel：(1) A-shape kernel（静态稀疏，保留 global tokens + local windows）；(2) Vertical-Slash kernel（混合 block-sparse + column-sparse，两个子 kernel：VS Sparse Index kernel + VS FlashAttention kernel）；(3) Block-Sparse kernel（基于 block-level top-k 选择）。Kernel 基于 Triton 语言 + PIT 动态稀疏编译器 + FlashAttention 实现，针对 A100 GPU 优化。

  实验比较：(a) 三种 kernel vs FlashAttention 的 micro-benchmark latency（Fig. 10）：Block-Sparse 最快（1M context 下 30× speedup），A-shape 次之（10K 下 <1ms，1M 下 164ms），Vertical-Slash 最慢但仍有 13× speedup；(b) 端到端 pre-filling latency breakdown（Fig. 1b）：100K→1.8×, 300K→4.1×, 500K→6.8×, 1M→10× speedup；(c) 三种模式的 sparsity distribution（Fig. 12）：>200K 时实际计算稀疏度 >90%，>500K 时 >95%。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU（bfloat16）。Kernel 基于 Triton，可移植到 H100、MI300X。单 A100 优化：Tensor Splitting（按 head 拆分 Attention、按 sequence 维度拆分 MLP）、消除中间变量（mask logic 直接在 kernel 内实现 causal mask）、仅计算最后 token 的 logits。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashAttention Triton 实现 + PIT（Permutation Invariant Transformation）动态稀疏编译器。核心修改：

  1. **Block-Sparse FlashAttention kernel**（Appendix C.4.1）：以 selected block index 为额外输入，每个 thread block 循环遍历每行的 top-K blocks。速度比 $s_p = S / (2B \times k_b)$，B=64 为 block size。

  2. **Vertical-Slash Sparse Index kernel**（Algorithm 4）：对每行 blocks 构建稀疏索引——point-range two-way merge 算法，垂直索引视为 points、斜线索引按行索引转为 ranges。输出 merged ranges（block indexes）+ separate column indexes。时间复杂度 O(k_v + k_s) per row，GPU 并行化。

  3. **Vertical-Slash FlashAttention kernel**（Algorithm 5）：混合 kernel——先循环 block indexes（Block-Sparse FlashAttention 方式），再循环 column indexes grouped by block size（PIT sparse attention 方式）。PIT 将稀疏数据通过 Permutation Invariant Transformation 加载到 dense compute blocks。

  4. **A-shape kernel**：静态稀疏掩码（固定保留 1K global tokens + 4K local window tokens），直接使用 FlashAttention 但仅计算静态掩码内的区域。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://aka.ms/MInference（GitHub）。基于 PyTorch + Triton + PIT + FlashAttention。

  **Kernel 执行全流程（以 Vertical-Slash kernel 为例，LLaMA-3-8B, 128K context）**：

  ```
  输入：
    Q, K, V ∈ R^{131072×128}（S=128K, d_h=128）
    i_v ∈ N^{30}（top-30 垂直列索引）
    i_s ∈ N^{2000}（top-2000 斜线索引）
    block_size B = 64

  Step 1: 在线稀疏索引构建（Vertical-Slash Sparse Index kernel, Algorithm 4）
    N = ceil(S/B) = 2048 行 blocks
    GPU 并行 for i ← 1 to N:
      # 找到当前行 i 对应的斜线范围
      j_s ← biset_left(i_s, i×B)         # 二分查找第一条穿过行 i 的斜线
      r_start ← (i-1)×B - i_s[j_s]
      r_end ← i×B - i_s[j_s]
      # Point-range two-way merge
      while 垂直列和斜线范围存在:
        如果垂直列在范围外 → 记录 column index
        如果斜线范围结束 → 记录 block index 并更新范围
      输出: c_blk^i（block count）, i_blk^i（block indices）,
            c_col^i（column count）, i_col^i（column indices）

  Step 2: 稀疏 FlashAttention（Vertical-Slash FlashAttention kernel, Algorithm 5）
    GPU 并行 for i ← 1 to N (2048 行 blocks):
      Load Q_chip ← Q[i×B:(i+1)×B]  [B=64, 128]
      Init O_chip = 0, m = -inf, l = 0

      # Part A: Block-sparse attention（循环 block indexes）
      for j ← 1 to c_blk^i:
        s ← i_blk[i, j]                      # block 起始位置
        Load K_chip ← K[s:s+B]               # [64, 128]
        Load V_chip ← V[s:s+B]               # [64, 128]
        S ← τ × Q_chip @ K_chip^T            # [64, 64], τ=1/√128
        S ← mask(S)                           # causal mask
        m_new ← max(m, rowmax(S))
        S ← S - m_new; P ← exp(S)
        l_new ← rowsum(P(S))
        α ← exp(m - m_new)
        l ← α·l + l_new
        O_chip ← α·O_chip + P @ V_chip

      # Part B: PIT column-sparse attention（循环 column indexes）
      j ← 0
      while j < c_col^i:
        cols ← i_col[i, j:j+B]              # [B] column indices
        Load K_chip ← K[cols]               # [64, 128]
        Load V_chip ← V[cols]               # [64, 128]
        S ← τ × Q_chip @ K_chip^T           # 同上流程
        S ← mask(S); m_new ← max(m, rowmax(S))
        S ← S - m_new; P ← exp(S)
        l_new ← rowsum(P(S))
        α ← exp(m - m_new)
        l ← α·l + l_new
        O_chip ← α·O_chip + P @ V_chip
        j ← j + B

      O_chip ← diag(l)^{-1} × O_chip         # 归一化
      Save O[i×B:(i+1)×B] ← O_chip

  输出：O ∈ R^{131072×128}
  ```

  **延迟分解**：
  - 动态索引构建时间：Vertical-Slash ~5-15%，Block-Sparse ~25%（主要开销来自 MeanPooling + block-level matmul）
  - 稀疏计算时间：占总时间主要部分
  - Memory overhead：1M context 下 <160MB（LLaMA-3-8B）

  **实际加速比**（端到端 pre-filling，单 A100）：
  - 10K context：接近 FlashAttention（索引构建开销占比高，~30%）
  - 100K context：1.8× speedup
  - 500K context：6.8× speedup
  - 1M context：10× speedup（从 30 min 降至 3 min）
