## XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**XStreamVGGT**，一个 tuning-free 的 KV cache 压缩方法，无缝集成 pruning 和 quantization 两种技术，用于 StreamVGGT 的流式 3D 视觉几何推理。由两个核心组件构成：

  **(1) 基于 Query 引导的 KV Cache Pruning**（消除多帧冗余）：
  - 对当前帧的 Query tokens 按固定组大小 g 分组并平均池化：`Q_pooled = concat(Q_special, GroupAvg(Q_normal, g))`，然后跨 attention heads 平均得到 `Q̄_t ∈ R^{N_pooled × C}`。
  - 对中间帧的 Key tokens 也跨 heads 平均：`K̄_prunable ∈ R^{T_prunable × C}`。
  - Token 重要性分数通过 query-key 内积计算：`S_matrix = Q̄_t · (K̄_prunable)^T`，然后沿 Query 维度平均：`S = (1/N_pooled) · Σ_i S_matrix[i,:]`。
  - 基于重要性分数做 top-k 选择，总 cache 长度控制在预算 `L_max` 以内（实验中设为 2K tokens）。始终保留第一帧 KVs（作为几何参考）和当前帧 KVs。第一帧和当前帧 token 之外的历史帧才参与剪枝。
  - 剪枝在每层 temporal global attention 完成后触发。分组池化设计保持与 FlashAttention 等高效 attention kernel 的兼容性。

  **(2) 维度自适应 KV 量化**（基于 KV 分布特性）：
  - 分析发现：Key tensors 存在显著的 channel-wise outliers（少量 channel 的数值远大于其他 channel），而 Value tensors 的分布更均匀。
  - 量化策略：对 Keys 使用 **per-channel 量化**（每个 channel 独立计算 scale s_c 和 zero-point z_c），对 Values 使用 **per-token 量化**（每个 token 独立计算 scale s_t 和 zero-point z_t）。
  - 采用非对称均匀量化：`x̂ = clamp(⌊x/s⌋ + z, 0, 2^b-1)`，使用 KIVI 方案，INT4 精度，group size 64。
  - 量化紧耦合在 pruning 之后：先对 pruned KV cache 进行量化存储，attention 计算时 dequantize 回浮点精度。

  实验比较：对比以下方法：
  - **VGGT** (CVPR 2025)：Offline 推理，使用全局 Alternative-Attention，无 KV cache 限制。
  - **StreamVGGT** (2025)：Online streaming 推理，frame-wise causal attention + unbounded KV cache。

  评估任务：3D 重建（7-Scenes, NRGBD），相机姿态估计（TUM Dynamics, ScanNet），单目/视频深度估计（Sintel, Bonn, KITTI）。
  效率评估：50-1000 帧输入序列，测量 GPU 内存消耗和推理速度（FPS）。
  消融实验：cache length（2K/4K/6K/8K），pruning 和 quantization 的独立效果。

- 硬件平台是什么，配置是什么。
  **单张 NVIDIA A100 GPU (80GB)**。所有效率实验在此配置上完成。StreamVGGT 和 VGGT 随帧数增加出现 FPS 显著下降并快速 OOM，XStreamVGGT 可稳定运行 1000 帧以上不 OOM。

- 模型是什么。数据集和bench分别是什么。
  模型：**StreamVGGT**（基于 VGGT，1.2B 参数，Alternating-Attention 架构，L 层 spatio-temporal transformer encoder，每层含 frame-wise spatial self-attention + temporal causal attention）。输入图像处理遵循 Point3R 协议：变长宽比处理，resize 最大边长 ≤ 518 pixels。token 序列含 camera token (1)、register tokens (R)、patch tokens (N)，每帧总长 1+R+N。

  数据集和 Benchmark：
  - **3D 重建**：7-Scenes [22], NRGBD [2]。指标：Accuracy (Acc↓), Completion (Comp↓), Normal Consistency (NC↑)。
  - **相机姿态估计**：TUM Dynamics [24], ScanNet [8]。指标：ATE↓, RPE_trans↓, RPE_rot↓。
  - **深度估计**（单目+视频）：Sintel [3], Bonn [19], KITTI [13]。指标：Abs Rel↓, δ<1.25↑。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ywh187/XStreamVGGT/

  算法 pipeline 伪代码：

  ```
  # 初始化：XStreamVGGT 推理循环，每帧 t 执行以下步骤

  # 第一步：Token 化
  F_t = PatchEmbed(I_t)           # I_t ∈ R^{3×H×W} → F_t ∈ R^{N×C}
  Input_t = [g_t; r_t; F_t]        # camera token + register tokens + patch tokens
                                   # Input_t ∈ R^{(1+R+N)×C}

  # 第二步：逐层 Transformer 处理（L 层 Alternating-Attention）
  for ℓ = 1 to L:
      # 2a. 帧内空间 self-attention（无 KV cache）
      H_t^(ℓ) = SpatialSelfAttn(Input_t^(ℓ-1))

      # 2b. 时序 causal attention（使用 KV cache）
      K_t^(ℓ), V_t^(ℓ) = Proj_KV(H_t^(ℓ))  # 新计算当前帧 K, V
      Q_t^(ℓ) = Proj_Q(H_t^(ℓ))

      # 2c. 拼接历史 cache 和新 K/V，causal attention
      K_all = concat(Cache.K_{1:t-1}^(ℓ), K_t^(ℓ))
      V_all = concat(Cache.V_{1:t-1}^(ℓ), V_t^(ℓ))
      Out_t^(ℓ) = FlashAttn(Q_t^(ℓ), K_all, V_all, causal_mask=True)

      # 2d. KV Cache 剪枝（当 cache 长度超过 L_max 时触发）
      if len(K_all) > L_max:
          # 分组池化 Query（保留特殊 token，patch token 分组平均）
          Q_pooled = concat(Q_special, GroupAvg(Q_normal, g=16))
          Q̄ = mean(Q_pooled, dim=heads)     # 跨 head 平均

          # 提取中间帧 prunable Key，跨 head 平均
          K̄_prunable = mean(K_{first+1 : t-1}, dim=heads)

          # 计算重要性分数
          S = mean(Q̄ @ K̄_prunable^T, dim=query)  # 沿 query 维度平均

          # Top-k 选择 + 保留首帧和当前帧
          I_middle = TopK(S, k = L_max - T_first - T_current)
          I_keep = {1..T_first} ∪ I_middle ∪ {T-T_current+1..T}

          # 同步剪枝 K 和 V
          Cache.K_{1:t}^(ℓ) = Cache.K_{1:t}^(ℓ)[I_keep]
          Cache.V_{1:t}^(ℓ) = Cache.V_{1:t}^(ℓ)[I_keep]

      else:
          # 未达预算，直接追加
          Cache.K_{1:t}^(ℓ).append(K_t^(ℓ))
          Cache.V_{1:t}^(ℓ).append(V_t^(ℓ))

      # 2e. 维度自适应量化（剪枝后对 cache 量化存储）
      for each channel c in Cache.K:
          s_c = (K_max[c] - K_min[c]) / (2^4 - 1)
          z_c = round(-K_min[c] / s_c)
          K̂_c = clamp(round(K_c / s_c) + z_c, 0, 15)  # INT4

      for each token i in Cache.V:
          s_i = (V_max[i] - V_min[i]) / (2^4 - 1)
          z_i = round(-V_min[i] / s_i)
          V̂_i = clamp(round(V_i / s_i) + z_i, 0, 15)   # INT4

      # 存储量化后的 K̂, V̂ 及 scale/zero-point 参数

  # 第三步：任务头预测
  CameraParams = Head_camera(Out_t^(L))
  PointMap = Head_pointmap(Out_t^(L))
  DepthMap = Head_depth(Out_t^(L))
  ```

  关键超参数：pooling group size g=16，cache budget L_max=2K，KIVI INT4 量化 + group size 64。
  剪枝和量化均为 tuning-free，无需额外训练。
