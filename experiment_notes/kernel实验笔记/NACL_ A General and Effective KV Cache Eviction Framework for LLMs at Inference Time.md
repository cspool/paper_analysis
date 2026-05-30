## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了 Reduce Attention Scores CUDA kernel，与 FlashAttention-2 兼容，用于在 encoding 阶段高效计算 attention scores 的列向归约（per-token importance scores），避免重新计算完整 attention matrix。两种实现方式：(1) **重计算方式**：利用 FlashAttention-2 forward 返回的 log-sum-exp（LSE），按 backward pass 的方式重算 attention scores 矩阵，再做 column-wise sum 得到 reduced attention scores（Algorithm 2）；(2) **小矩阵重计算方式**：仅对 proxy tokens（~10% tokens）重新计算 attention scores，因 proxy token 数量远小于总 token 数，额外开销可忽略。128K context 下 evict 20% 维持 15GB 稳定显存。实验：128K context 推理的显存和速度兼容性验证。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU。基于 FlashAttention-2 的 CUDA kernel 实现。128K context length 测试。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashAttention-2 的 backward pass 逻辑实现。核心修改：
  1. **Reduce Attention Scores Kernel（Algorithm 2）**：利用 FlashAttention-2 forward 输出的 LSE vector L ∈ R^{N_q}，在 SRAM 上分 tile 重算 P_i^{(j)} = exp(S_{ij} - L_i)，再做 column-wise reduce R_j += Reduce(P_i^{(j)})，输出 per-key 的累积 attention scores O ∈ R^{N_k}。分块策略与 FlashAttention-2 一致（T_r = ceil(N_q/B_r), T_c = ceil(N_k/B_c)）。
  2. **小矩阵重计算方式**：仅对 proxy token subset P 和完整 K 计算 attention scores，Q_proxy ∈ R^{|P|×d}，K ∈ R^{N_k×d}，计算量仅 O(|P|·N_k·d) 而非 O(N_q·N_k·d)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL

  **Reduce Attention Scores Kernel 执行全流程（Algorithm 2, FlashAttention-2 兼容）**：

  ```
  输入：
    Q ∈ R^{N_q×d}, K ∈ R^{N_k×d} in HBM
    Logsumexp L ∈ R^{N_q} in HBM（FlashAttention-2 forward 输出）
    block sizes B_c, B_r
  输出：O ∈ R^{N_k}（per-key 的 reduced attention scores）

  Step 1: 分块
    Q → T_r = ceil(N_q/B_r) blocks
    K → T_c = ceil(N_k/B_c) blocks
    L → T_r blocks
    O = zeros(N_k) in HBM → T_c blocks

  Step 2: 逐 K block 计算（外层循环）
    for j = 1..T_c:
      Load K_j from HBM → SRAM     # K_j ∈ R^{B_c×d}
      R_j = zeros(B_c) on Register  # per-block reduced scores

  Step 3: 逐 Q block 计算（内层循环）
      for i = 1..T_r:
        Load Q_i, L_i from HBM → SRAM  # Q_i ∈ R^{B_r×d}, L_i ∈ R^{B_r}
        S_i^{(j)} = Q_i @ K_j^T          # ∈ R^{B_r×B_c}, on-chip matmul
        P_i^{(j)} = exp(S_i^{(j)} - L_i) # ∈ R^{B_r×B_c}, online rescale
        R_j = R_j + Reduce(P_i^{(j)})    # column-wise sum, ∈ R^{B_c}

  Step 4: atomicAdd to HBM
      atomicAdd(O_j, R_j)               # 累加到全局输出

  返回 O ∈ R^{N_k}，即每个 key token 的累积 attention score
  ```

  **评估原理**：128K context 下 NACL evict 20% KV cache，测量 GPU 显存使用量（维持 ~15GB 稳定），确认 kernel 开销不显著影响推理吞吐。小矩阵重计算方式因 |P| ≪ N_q 而开销可忽略。
