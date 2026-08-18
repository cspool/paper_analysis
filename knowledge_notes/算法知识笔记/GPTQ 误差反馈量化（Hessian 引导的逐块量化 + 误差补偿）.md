## GPTQ 误差反馈量化（Hessian 引导的逐块量化 + 误差补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPTQ（Frantar et al., ICLR 2023）是一种基于近似二阶信息（Hessian）的 LLM 训练后量化方法：用少量 calibration 激活估计权重对输出影响的 Hessian $H=2X X^T$（X 为校准激活），逐列/逐块贪心量化权重，并把当前块的量化误差通过 $H^{-1}$ 补偿（"错误反馈"，类似最优脑手术 OBS 的层内误差传播）到尚未量化的权重上，使整体量化误差最小。SingularBit 不直接量化原权重，而是把 GPTQ 错误反馈框架应用到 SVD 分解后的 U 与 V^T 分量上：先按 rank 边界逐块量化 U（Hessian $H_U=x^Tx$），再推导计入已量化 $\hat{U}$ 的有效 Hessian $H_{V^T}=S\hat{U}^T H_U \hat{U}S$ 量化 V^T，让后量化的 V^T 能补偿前序 U 的累积误差。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流程（论文 Algorithm 1）：量化 U 时逆序（idx: ich→0 步长 blocksize）逐块处理，每块 QUANTIZE 输出行/列缩放因子与二进制基 $\{\alpha_{r,i},\alpha_{c,i},B_i\}$，重建 $\hat{U}=\sum\alpha_{r,i}\alpha_{c,i}B_i$，误差 $E_u=U-\hat{U}$ 经 $H_U^{-1}$ 作用后加到剩余未量化参数（$U\leftarrow U-E_uH_U^{-1}$）；V^T 正序（idx: 0→r）同法，用有效 Hessian $H_{V^T}=S\hat{U}^TH_U\hat{U}S$。伪代码：
  ```
  H_U = x^T @ x                              # 校准激活的二阶信息
  for idx in range(ich, 0, -blocksize):      # 逆序量化 U（先量化"更重要"的列？实际按块序）
      alpha_r, alpha_c, B = QUANTIZE(U, boundaries, idx)   # rank 边界决定该块位宽
      U_hat = sum_i alpha_r[i] * alpha_c[i] * B[i]
      E_u = U[:, idx:idx+b] - U_hat
      U[:, :idx] -= E_u @ inv(H_U)           # 误差反馈到已处理(未量化)部分
  H_VT = S @ U_hat.T @ H_U @ U_hat @ S       # 有效 Hessian（计入已量化 U）
  for idx in range(0, r, blocksize):         # 正序量化 V^T
      ...  # 同 U：重建 → 误差 → H_VT^{-1} 反馈
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：标准 GPTQ 有开源实现（https://github.com/IST-DASLab/gptq，AutoGPTQ 等），按列/块分组（group size 128）量化 + $H^{-1}$ 误差补偿；SingularBit 将其改造为"对 SVD 分量"执行并以 rank 边界替代均匀位宽。与标准 GPTQ 的差异：①量化对象是 U/V^T 而非原 W；②位宽逐 rank 区域变化（4/3/2/1-bit）而非整层均匀；③V^T 的 Hessian 需经 $S\hat{U}^T$ 变换以把 U 的量化误差纳入考量。论文数据：标准 GPTQ 在 2-bit 均匀量化下 LLaMA-7B Wiki 困惑度 44.01（严重退化），而 SingularBit-W（GPTQ 框架 + rank-aware 混合精度）达 7.56，说明错误反馈机制+重要性感知位宽缺一不可。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
