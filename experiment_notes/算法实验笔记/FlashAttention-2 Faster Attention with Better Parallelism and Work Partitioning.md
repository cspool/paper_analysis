## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- 属于算法pipeline的实现是什么？实验比较什么？
  对FlashAttention v1的online softmax算法做两项tweak以减少non-matmul FLOPs：(i) **前向**——不再对output累加的每一项做`diag(ℓ)^{-1}` rescale，改为维护"un-scaled" output并在所有KV blocks处理完后一次性做`diag(ℓ)^{-1}` rescale，消除每次内迭代对之前已累积output的rescale操作；(ii) **反向**——只存储logsumexp `L = m + log(ℓ)`（每行一个scalar, O(N) extra memory）替代同时存储rowwise max m和rowwise sum ℓ，反向从L恢复softmax denominator。
  实验比较：与FlashAttention v1 forward+backward speed对比（相同算法逻辑，不同non-matmul FLOPs数量）。与FlashAttention Triton实现对比。与xformers cutlass实现对比。与PyTorch标准attention对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4 GPU：FP16/BF16 matmul 312 TFLOPs/s，non-matmul FP32 19.5 TFLOPs/s（matmul与non-matmul吞吐比16:1，non-matmul FLOP实质比matmul FLOP贵16×）
  - NVIDIA H100 GPU（初步benchmark，未用TMA和FP8）

- 模型是什么。数据集和bench分别是什么。
  - 端到端训练模型：GPT3-1.3B（24 layers, hidden_dim=2048, 16 heads, head_dim=128）和GPT3-2.7B（32 layers, hidden_dim=2560, 32 heads, head_dim=80），sequence length 2k和8k
  - Benchmark数据集：论文未明确说明端到端训练使用的具体数据集（仅测量training throughput TFLOPs/s，非下游任务accuracy）
  - Attention benchmark：变sequence length（512-16K）、head dim（64/128）、causal/non-causal mask

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD license）。安装：`pip install flash-attn`。

  **算法pipeline（以forward pass, 2 blocks简化为例, Q/K/V ∈ R^{B_r×d}）**：

  伪代码（对应Algorithm 1）：
  ```
  # FlashAttention-2 Forward (one row block)
  O_tilde = zeros(B_r, d)       # un-scaled output
  l = zeros(B_r)                # running sum of exp
  m = -inf * ones(B_r)          # running max
  
  for j = 1 to T_c:             # loop over KV blocks
      S = Q @ K_j.T             # [B_r, B_c] matmul (Tensor Core)
      m_new = max(m, rowmax(S)) # [B_r]
      m_rescale = exp(m - m_new) # [B_r], rescale factor for old values
      P_tilde = exp(S - m_new)  # [B_r, B_c] pointwise exp (non-matmul)
      l_new = m_rescale * l + rowsum(P_tilde)  # [B_r] (non-matmul)
      
      # Key difference from FlashAttention v1:
      # O_tilde stays un-scaled; no diag(l)^{-1} applied here
      O_tilde = diag(m_rescale) @ O_tilde + P_tilde @ V_j  # matmul
      
      m = m_new; l = l_new
  
  # Final rescale (only once, at the end):
  O = diag(l)^{-1} @ O_tilde
  L = m + log(l)  # logsumexp for backward
  ```
  vs FlashAttention v1（每次迭代都rescale output）：
  ```
  O = diag(l)^{-1} @ (diag(exp(m_old - m_new)) @ (diag(l_old) @ O_old) + P_tilde @ V_j)
  ```
  FlashAttention-2避免了每次迭代中`diag(l)^{-1}`对O_old的rescale操作（non-matmul elementwise multiply），改为最后一次统一rescale。同时反向只需L而非(m, l) pair，减少了register压力和non-matmul运算。

  **Non-matmul FLOPs对比（per iteration, per row block）**：
  - FlashAttention v1: rescale m_old→m_new (B_r mul) + rescale l_old (B_r mul) + exp(S-m_new) (B_r×B_c exp) + rowsum (B_r×B_c add) + rescale O_old by l_old/l_new (B_r×d mul) + rescale P_tilde by 1/l_new (B_r×B_c mul) ≈ B_r×(2 + 2B_c + 2d) non-matmul FLOPs
  - FlashAttention-2: rescale m_old→m_new (B_r mul) + rescale l_old (B_r mul) + exp(S-m_new) (B_r×B_c exp) + rowsum (B_r×B_c add) + rescale O_tilde by m_rescale (B_r×d mul, no l-based rescale) ≈ B_r×(2 + 2B_c + d) non-matmul FLOPs
  - 减少约 B_r×d 次non-matmul multiply per iteration。
