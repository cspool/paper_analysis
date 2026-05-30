## Scaling Law for Quantization-Aware Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出统一的 QAT（Quantization-Aware Training）缩放定律，将 W4A4 量化误差 δ_p 建模为模型参数量 N、训练数据量 D、量化粒度 G 三个变量的函数：δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}。在 Chinchilla 缩放定律 L(N,D) 的基础上叠加该量化误差项。关键发现：(1) δ_p 随 N 增大而减小、(2) 随 D 增大而增大、(3) 随 G 变粗而增大。进一步将 W4A4 量化误差分解为权重量化误差 δ_{W4A16} 和激活量化误差 δ_{W16A4} 两个分量，发现激活量化误差是主要瓶颈（尤其在 FC2 Proj 输入层，因其来自 SwiGLU 输出，峰度高达 89 远高于其他层）。提出混合精度方案：对 FC2 输入用 8-bit 量化，可消除激活瓶颈，使权重和激活误差贡献趋于均衡（ratio R 从 1.67 降至 0.85–1.10）。
  - 实验比较：(1) 与现有 QAT 缩放定律 [Frantar et al. 2025, Kumar et al. 2024] 对比，ours 将 N、D、G 统一建模为单条曲线（vs baseline 需为每种 G 单独拟合），W4A16 相对误差从 19.3% → 5.2%，W4A4 从 8.5% → 4.7%；(2) 消融实验：在 W4A4 精度下去除 D 项，预测相对误差从 4.7% 升至 8.6%（W4A4）和 5.2% 升至 13.8%（W4A16）；(3) INT4 vs FP4 量化精度对比（INT4 略优于 FP4）；(4) 不同量化器对比（AbsMax/LWC/LSQ 权重量化差异 <0.003，LAC 在 G>256 时显著优于 AbsMax 激活量化）；(5) FC2 8-bit 混合精度消融：G=32 量化误差降 20.5%，G=256 降 42.9%；(6) 973M 模型外推验证：缩放定律对更大模型和数据量准确外推。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 GPU，总计 268 次 QAT 实验消耗 276K GPU-hours
  - 训练框架：PyTorch，基于 OLMo2 训练超参数（AdamW β=(0.9, 0.95), weight decay 0.1, gradient clip 1.0, cosine LR schedule, warmup 500 steps, sequence length 2048）
  - 混合精度训练：BFloat16

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama3-style decoder-only Transformer，参数规模 N ∈ {74M, 145M, 297M, 595M} + 973M（外推验证），架构含 GQA + SwiGLU + RMSNorm + RoPE。详细架构配置：74M (12 layers/768 hidden/16 attn heads/4 KV heads), 145M (12/1024/16/4), 297M (12/1536/24/6), 595M (24/1536/24/6), 973M (16/2048/32/8)。Chinchilla 缩放定律拟合额外引入 2.8B/6.5B/12.7B 模型（OLMo-2 官方 release）。
  - 数据集：OLMo2-Mix-1124 pretraining 数据集，训练 token 数 D ∈ {10B, 20B, 50B, 100B} + 200B（外推验证）
  - Benchmark/评估指标：smoothed training loss 作为验证损失的无偏估计（与 Chinchilla 一致），量化误差 δ_p 定义为 loss_bf16 − loss_W4A4

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明代码开源。实验基于 PyTorch + HuggingFace Transformers，使用 OLMo2 开源训练框架。

  **算法 pipeline（QAT 训练 + 缩放定律拟合流程）：**

  ```
  # === Step 1: 全精度 Chinchilla 缩放定律拟合 ===
  # 用 BF16 模型训练数据拟合 L(N,D) = A/N^α + B/D^β + E
  # 使用 Huber loss + L-BFGS 优化器估计 (E, A, α, B, β)
  # 约束 α = β (与 Chinchilla 原文一致)
  # 输入: 145M-12.7B 模型的训练 loss，输出: 拟合参数

  # === Step 2: W4A4 QAT 训练 ===
  for each (N, D, G) in experiment_grid:
      model = Llama3Style(N)  # N ∈ {74M, 145M, 297M, 595M, 973M}
      for step in range(D):
          # 前向传播（量化插入在 Linear 层前后）:
          X_int4 = quantize_activation(X, G, AbsMax/LAC)
          W_int4 = quantize_weight(W, G, AbsMax)
          Y = INT4_GEMM(X_int4, W_int4) * s_X * s_W  # Fake-quantized forward
          loss = CrossEntropy(Y, labels)
          # STE 反向传播（通过量化器的直通估计器）
          loss.backward()
          optimizer.step()
      # 记录最终量化误差:
      δ_W4A4 = loss_bf16 - loss_W4A4  # 量化误差 ground truth

  # === Step 3: W4A16/W16A4 解耦训练（误差分解用） ===
  for each (N, D, G):
      # W4A16: 仅权重量化，激活保持 BF16
      # W16A4: 仅激活量化，权重保持 BF16
      # 分别记录 δ_W4A16 和 δ_W16A4
  # 验证 δ_W4A4 ≈ k · (δ_W4A16 + δ_W16A4)，k≈0.906

  # === Step 4: 缩放定律拟合 ===
  # δ_p(N,D,G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}
  # 用 80 次 W4A4 QAT 实验数据拟合 k, γ_N, γ_D, γ_G
  # 使用 Huber loss + L-BFGS

  # === Step 5: FC2 瓶颈分析与混合精度 ===
  # 分析各层 kurtosis → FC2 Proj 输入 kurtosis=89（vs 其他层 <10）
  # 原因: FC2 输入来自 SwiGLU 的 gating + 非线性变换，产生 outlier
  # 方案: FC2 Proj 输入保持 8-bit 量化，其余保持 4-bit
  # 效果: δ_W16A4(FC2 8-bit) 与 δ_W4A16 的 ratio R ∈ [0.85, 1.10]
  ```

  **量化器实现细节：**
  - 权重 AbsMax：s = M / max(|W|), W_int = clamp(round(W/s), -2^{b-1}, 2^{b-1}-1)
  - 激活 AbsMax（G<256）：同上，按 group 计算
  - 激活 LAC（G≥256）：s = M / (max(|X|)·γ)，γ 为可学习 clipping factor，同 group index 共享
  - W4A4 使用 INT4（16 个可表示值），优于 FP4 E2M1（15 个可表示值），尤其在 per-channel/token 粒度下差距 0.015 loss

  **关键结果：** 
  - 量化误差趋势：N 从 74M→594M，δ_{W4A4} 平均降 34%；D 从 10B→100B，δ_{W4A4} 平均升 22%；G 从 finest→coarsest，δ 差 0.037（约半数粗粒度误差）
  - 误差分解：δ_{W4A16} 对 D 的敏感度 γ_D=0.1610 远大于 δ_{W16A4} 的 γ_D=0.0331；δ_{W16A4} 对 G 的敏感度 γ_G=0.9812 远大于 δ_{W4A16} 的 γ_G=0.3533
  - 激活量化误差始终大于权重量化误差（ratio R>1），但随着 D/N 增大差距缩小
  - FC2 8-bit 后，W4A4 EPM（Effective Parameter Multiplier）提升 0.06–0.14，W4A4 EPM 始终 >0.5（即 W4A4 优于此 W8A8 的 cost-accuracy trade-off）
  - 973M/200B tokens 外推验证缩放定律准确预测趋势
