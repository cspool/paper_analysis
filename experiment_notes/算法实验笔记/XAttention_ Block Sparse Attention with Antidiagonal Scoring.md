## XAttention: Block Sparse Attention with Antidiagonal Scoring

- 属于算法pipeline的实现是什么？实验比较什么？
  实现 XAttention，一种 training-free 的 block-sparse attention 框架。核心创新是用注意力矩阵的反对角线和（antidiagonal sum）作为 block 重要性的轻量级代理指标，通过三步流程加速长上下文推理：(1) Strided Antidiagonal Scoring：按步长 S 沿反对角线采样元素求和作为 block 得分；(2) Threshold Block Selection：选择累计反对角线 softmax 概率之和超过阈值 τ 的最小 block 集合；(3) Minimum Threshold Prediction：通过动态规划为每个注意力头预测最优阈值，进一步优化稀疏度。

  实验比较：(1) RULER benchmark 上对比 Full Attention (FlashInfer/FlashAttention)、FlexPrefill、MInference、SeerAttn，在 4k-128k 序列长度下，XAttention (S=8) 平均分 88.47 vs Full 87.52，且优于 FlexPrefill (87.72)；(2) LongBench 真实长文本任务对比 MInference 和 FlexPrefill，XAttention 取得最高平均分；(3) Video-MME 视频理解任务上对比 Full Attention、MInference、FlexPrefill，XAttention 在长视频上优于 Full Attention；(4) VBench 视频生成任务（HunyuanVideo），XAttention + 5-step warmup 达到 PSNR 23.5 / SSIM 0.822 / LPIPS 0.155，密度仅 45.5%；(5) 效率对比：256k 上下文下 prefill 注意力加速最高 13.5×（S=16，密度 7.32%），pattern selection 比 MInference 快 24.9×、比 FlexPrefill 快 5.9×。

- 硬件平台是什么，配置是什么。
  NVIDIA GPU（论文致谢 NVIDIA DGX 服务器捐赠）。使用 FlashInfer 框架进行注意力计算。具体 GPU 型号论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（文本任务）、Qwen2-VL-7B-Instruct（视频理解）、HunyuanVideo（视频生成，DiT 架构，non-causal attention）。
  数据集/Benchmark：RULER（合成长上下文 benchmark，4k-128k）、LongBench（真实长文本任务含 Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Code）、Video-MME（900 视频、254 小时）、VBench（946 GPT-augmented 文本提示词）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/mit-han-lab/x-attention

  算法 Pipeline（XAttention 三步流程）：

  ```
  # 输入: Q, K ∈ R^{L×d}，block size B，stride S，threshold τ
  # 输出: Sparse mask M, Sparse attention output O

  # === Step 1: Strided Antidiagonal Scoring ===
  N_B = L // B  # number of blocks
  for b = 0 to N_B - 1:
      # 沿反对角线 reshape Q 和 K
      Q_reshaped = []  # shape: [S, B//S, d] per block
      for i = S-1 down to 0:
          Q_reshaped.append(Q[b*B:(b+1)*B, :][i::S, :])
      K_reshaped = []
      for i = 0 to S-1:
          K_reshaped.append(K[i::S, :])

      # 计算近似注意力分数
      A_approx = Softmax((Q_reshaped @ K_reshaped^T) / sqrt(d_h) / S)
      # A_approx 的反对角线和作为 block 重要性代理
      M_b = find_blocks(A_approx, τ)

  # find_blocks: 选择累计概率超过 τ 的最小 block 集合
  # find_blocks(A, τ) = argmin_{B} |B|  s.t. Σ_{b∈B} Σ_{(i,j)∈b} A_{i,j} ≥ τ

  M = concat(M_0, ..., M_{N_B-1})  # 稀疏 mask

  # === Step 2: Threshold Block Selection ===
  # 使用累积反对角线 softmax 概率超过 τ 的 block 作为选中 block
  # 不同注意力头可设置不同 τ，实现动态稀疏度

  # === Step 3: Minimum Threshold Prediction (Optional) ===
  # 动态规划为每个头寻找最优阈值
  # D[h][m]: h 个头、m 次调整的最佳性能
  # D[h][m] = max(D[h-1][m], P(h, m))
  # t_h(m) = t_h(m-1) * 0.9  # 每次调整降低 10%

  # === Step 4: Sparse Attention Computation ===
  # 仅对 M 中标记的 block 执行精确注意力计算
  O = BlockSparseAttention(Q, K, V, mask=M)
  ```

  关键洞察：反对角线交叉每个 block 内所有可能的垂直和斜线注意力模式（vertical & slash patterns），确保不遗漏任何关键模式。每个 token 至少参与一条反对角线，保证信息完整性。

  加速来源：Block 级计算从 O(L²) 降至 O(L² × density)，如 128k 时密度仅 6.89%（S=8），理论加速 ~14.5×。
