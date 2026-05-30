## S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出S²Q-VDiT，一个面向视频扩散模型（V-DMs）的后训练量化（PTQ）框架。核心包含两个创新组件：(1) **Hessian-aware Salient Data Selection (SDS)**：联合评估扩散信息量（diffusion salience）和量化敏感度（quantization salience）两个维度，通过Hessian矩阵近似计算每个校准样本的重要性分数并进行min-max归一化后的乘积作为统一salience得分，从候选池中筛选高质量校准数据。扩散信息量C_diff = ||x_t - x_{t-1}||²/||x_t||²衡量相邻去噪步的表示变化；量化敏感度C_quant = ||x_t^T x_t||_2基于Levenberg-Marquardt近似的Hessian矩阵衡量样本对量化扰动的敏感度。(2) **Attention-guided Sparse Token Distillation (STD)**：利用V-DMs中空间-时间注意力固有的稀疏模式，通过注意力图中token-wise的注意力分布计算每个token的重要性权重λ_j = normalize(Σ_{h,i} A_{h,i,j})，将量化损失重加权为L_quant = (1/n) Σ_j λ_j||θ^f(x_{j,:}) - θ^q(x_{j,:})||²，使模型在优化中聚焦高影响力token。

  实验比较W4A6（4-bit权重6-bit激活）和W4A4（4-bit权重4-bit激活）两种量化设置下的视频生成质量。对比方法包括：Q-DiT、PTQ4DiT、ViDiT-Q（扩散模型PTQ）；SmoothQuant、QuaRot（LLM PTQ baseline）。评估指标：VBench的8个维度（Imaging Quality, Aesthetic Quality, Motion Smoothness, Dynamic Degree, Background Consistency, Subject Consistency, Scene Consistency, Overall Consistency）和EvalCrafter的5个指标（CLIPSIM, CLIP-Temp, VQA-Aesthetic, VQA-Technical, FLOW Score）。

- 硬件平台是什么，配置是什么。
  所有校准实验在单张NVIDIA A800 GPU上完成。量化部署效率测试同样在单张NVIDIA A800 GPU上使用CUDA实现（基于ViDiT-Q和FlatQuant的CUDA kernel）。量化框架基于PyTorch实现。

- 模型是什么。数据集和bench分别是什么。
  模型：CogVideoX-2B、CogVideoX-5B (Yang et al., 2024)、HunyuanVideo-13B (Kong et al., 2024)。校准数据：10个随机prompt生成候选校准样本，最终每个方法选取40个样本（trade-off 性能vs校准时间后选择）。校准数据集尺寸对比实验（20/40/80样本）。评估benchmark：VBench (Huang et al., 2024) 和 EvalCrafter (Liu et al., 2024)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码（论文声明）：https://github.com/wlfeng0509/s2q-vdit
  
  算法Pipeline（以CogVideoX-2B W4A6量化为例）：

  **阶段1：Hessian-aware Salient Data Selection (SDS)**
  ```
  # 输入：预训练V-DM模型θ，候选prompt集合P，总去噪步数T
  # 输出：校准数据集D_calib，大小为N=40

  for each prompt p in P:
      for each timestep t in [1..T]:
          x_t = 模型在prompt p、timestep t的隐层表示  # x_t ∈ R^{n×d}, n=s×t
          计算扩散salience: C_diff(x_t) = ||x_t - x_{t-1}||² / ||x_t||²
          计算量化salience: C_quant(x_t) = ||x_t^T x_t||_2  # Levenberg-Marquardt Hessian近似
      end
  end

  # min-max归一化到[0,1]
  C̅_diff = (C_diff - C_diff_min) / (C_diff_max - C_diff_min)
  C̅_quant = (C_quant - C_quant_min) / (C_quant_max - C_quant_min)

  # 联合salience分数（乘积形式，算术-几何平均不等式确保双高才得分高）
  for each candidate x_t:
      C_sample(x_t) = C̅_diff(x_t) · C̅_quant(x_t)
  end

  # 按C_sample降序排列，选Top-N样本构成D_calib
  D_calib = top_N_samples_by_C_sample
  ```

  **阶段2：Block-wise Post-Training Quantization with STD**
  ```
  # 量化方案：uniform per-channel weight量化 + dynamic per-token activation量化
  # 使用channel-wise scale、rotation matrix、learnable clipping threshold
  # 基于GPTQ weight quantizer

  # 量化参数：对称量化
  # x_int = clamp(round(x/Δ), -2^{N-1}, 2^{N-1}-1), Δ = max(|x|)/(2^{N-1}-1)

  for each transformer block in model:
      # 前向计算FP模型的attention map用于后续STD
      A = block.attention(x)  # A ∈ R^{H×n×n}, H为head数

      # 计算每个token的重要性权重λ_j
      for each token j in [1..n]:
          S_j = Σ_{h,i} A_{h,i,j}  # token j从所有token和head获得的attention权重之和
      end
      λ_j = (S_j - min(S)) / (max(S) - min(S)) * (λ_max - λ_min) + λ_min

      # Block-wise量化优化（30个样本，15 epochs，AdamW optimizer，cosine LR）
      for epoch in [1..15]:
          for each sample x in D_calib（随机选30个）:
              # 量化损失：重加权的MSE
              L_quant = (1/n) Σ_{j=1}^{n} λ_j · ||θ^f_block(x_{j,:}) - θ^q_block(x_{j,:})||²
              # 更新量化参数：diag-balancing scale (lr=5e-3), rotation matrix (lr=5e-3),
              # learnable clipping threshold (lr=5e-2)
          end
      end

      # 吸收量化参数到权重（weight folding），无额外推理负担
      # 激活量化采用online dynamic quantization
  end
  ```

  **阶段3：部署推理**
  ```
  # W4A6量化后推理
  # 权重：per-channel INT4存储，推理时dequantize到FP16
  # 激活：per-token online dynamic INT6量化
  # CUDA实现基于ViDiT-Q [62] 和 FlatQuant [47]
  # 结果（CogVideoX-5B）：模型存储 2.633GB (3.94×压缩)，推理显存 10.145GB (1.56×节省)，延迟 203.2s (1.28×加速)
  ```

  **消融实验关键结果**：
  - SDS vs 随机采样：W4A4 CogVideoX-2B下SDS的Imaging Quality=52.95±0.69（方差仅0.69），而ATOP仅51.65±1.76（方差1.76），证明SDS在性能和稳定性上均优于随机采样。
  - STD vs 无STD：λ_min=0.5时获得最佳性能平衡，所有λ_min取值均能提升量化性能，证明STD的鲁棒性。
  - 校准数据量：40样本 vs 20样本显著提升，40→80样本提升微小，选40为统一设置。
  - 校准资源：W4A4 CogVideoX-2B下GPU Memory 35.68GB, GPU Time 2.88h, 相比PTQ4DiT仅增加2GB和0.63h。
