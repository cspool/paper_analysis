## PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PTQ4ARVG 是首个面向 ARVG（AutoRegressive Visual Generation）模型家族的 training-free PTQ 框架，包含三个核心组件：
    (1) **GPS (Gain-Projected Scaling)**：通过 Taylor 展开量化激活和权重的量化损失，定义 scaling gain（激活量化损失减少量减去权重量化损失增加量），对 gain 函数求导得到闭式最优 scaling factor：s_i = s_k · √(Σ_j |ΔW_{i,j}·x_i|) / √(Σ_j |W_{i,j}·Δx_i|)，其中 s_k 为最大激活 range 通道的 scaling factor，由 s_k = √(R_x^k / R_W^k) 确定。GPS 是首个基于数学优化的量化 scaling 策略，无需训练。
    (2) **STWQ (Static Token-Wise Quantization)**：利用 ARVG 的固定 token 长度和跨样本位置不变分布两大特性，离线分配 per-token 量化参数。对 AdaLN 模块沿 token 序列分配量化参数；对线性层的 sink token（首 token）和 normal token 分别分配量化参数。使用百分位数（percentile）校准确保精度，无在线校准开销。
    (3) **DGC (Distribution-Guided Calibration)**：基于 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 评估每个样本对分布熵的贡献，选择 top 50% 高熵样本作为校准集，消除样本间分布不匹配导致的量化参数校准偏差。
  - 实验比较：PTQ4ARVG vs SmoothQuant、OS+、RepQ*、OmniQuant（training-based）、QuaRot（rotation-based）、SVDQuant（low-rank decomposition），在 W8A8 和 W6A6 两种位宽下，对 VAR（d16/d20/d24/d30）、RAR（B/L/XL/XXL）、PAR（XL-4×/XXL-4×/3B-4×/3B-16×）、MAR（B/L/H）四个 ARVG 模型家族进行量化，比较 FID、sFID、IS、Precision 四个图像生成质量指标。额外消融验证 GPS/STWQ/DGC 各自贡献，对比 GPS vs 其他 scaling 方法（SmoothQuant, RepQ*, OS+, SQ+RepQ*），对比 STWQ vs DTWQ（dynamic token-wise quantization）的精度和 speedup，对比 DGC vs random/uniform 采样的校准效果。W4A8 实验展示更低 bit-width 下的优势。在 RTX 3090 上部署 8-bit 量化模型评估真实加速比和内存节省。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 3090 用于实际部署评估（测量推理延迟、峰值内存和加速比）。校准和量化过程在 GPU 上完成（基于 baseline 对比中 OmniQuant 需 A100-80G 的说明，推断使用高性能 GPU 进行量化实验）。
  - 校准集：128 张 ImageNet 图像（DGC 从更大校准池中选择 top 50% 高熵样本）。
  - 量化配置：W8A8 和 W6A6，所有线性层和矩阵乘法均量化（含 KV cache）。权重采用 channel-wise 非对称量化，激活采用 layer-wise 非对称量化（经 STWQ 后为 per-token static）。scaling factor 离线融合到网络权重中，推理时零额外开销。
  - 评估套件：ADM's TensorFlow evaluation suite *guided-diffusion* 计算 FID/sFID/IS/Precision。生成 50K 张 ImageNet 图像评估质量。
  - 实际部署测试：标准 CUDA kernel 部署 decoder 网络，batch size=100，不同 token 序列长度（64/128/256/512/1024）下测试延迟和峰值内存。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：VAR（d16, d20, d24, d30，最大 2B 参数）、RAR（B, L, XL, XXL，最大 1.5B 参数）、PAR（XL-4×, XXL-4×, 3B-4×, 3B-16×，最大 3B 参数）、MAR（B, L, H，最大 1B 参数）。所有预训练模型来自各官方仓库。
  - **数据集**：ImageNet（ILSVRC 2012, Deng et al. 2009），用于生成 50K 图像评估质量，128 张图像用于校准。
  - **Benchmark/指标**：FID↓（Fréchet Inception Distance）、sFID↓、IS↑（Inception Score）、Precision↑。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/BienLuky/PTQ4ARVG
  - PTQ4ARVG 量化流程（以 RAR-B 单层线性层为例，W6A6）：
    ```
    输入: 预训练 ARVG 模型, 校准数据 X_cal (128 张 ImageNet)
    输出: W6A6 量化模型

    === Step 1: DGC - 分布引导校准集选择 ===
    计算校准池中所有样本的均值 u 和协方差 S
    for each sample x in calibration pool:
        ρ(x) = sqrt((x-u)^T · S^{-1} · (x-u))  // Mahalanobis 距离
    选择 ρ(x) 最大的 top 50% 样本作为校准集 X_DGC

    === Step 2: GPS - 增益投影缩放 ===
    for each linear layer (qkv, fc1) in ARVG blocks:
        // 量化当前权重和激活，计算量化误差
        X_q = Q(X), W_q = Q(W)
        ΔX = X - X_q, ΔW = W - W_q

        // 搜索激活 range 最大的通道 k
        R_x = max(X, dim=0) - min(X, dim=0)   // per-channel activation range
        k = argmax(R_x)

        // 计算 s_k：使该通道激活和权重 range 对齐
        s_k = sqrt(R_x^k / R_W^k)

        // 计算剩余通道的 scaling factors (闭式解, Eq. 16)
        for i = 1 to n (n = input channels):
            if i != k:
                s_i = s_k * sqrt(Σ_{j=1}^m |ΔW_{i,j} · x_i|) /
                           sqrt(Σ_{j=1}^m |W_{i,j} · Δx_i|)

        // 应用等效缩放 (Eq. 2)，缩放因子离线融合到权重
        X' = X ⊘ s        // activation 逐通道除以 s
        W' = s ⊙ W        // weight 逐通道乘以 s

    === Step 3: STWQ - 静态逐 token 量化 ===
    for each block in decoder:
        // (a) AdaLN 模块：沿 token 序列分配 per-token 量化参数
        for t = 1 to T (fixed token length):
            δ_AdaLN[t] = PercentileCalib(X_AdaLN[:, t, :])

        // (b) 线性层 (qkv, fc1, etc.)：sink token 与 normal token 分别量化
        for each linear layer input X_lin ∈ R^{T×n}:
            X_sink = X_lin[0, :]               // 首 token (sink, 含条件信息)
            X_normal = X_lin[1:, :]            // 其余 token
            δ_sink = PercentileCalib(X_sink)
            δ_normal = PercentileCalib(X_normal)

    === Step 4: 量化推理 (无在线校准开销) ===
    for each inference step (token generation):
        // 使用 Step 3 中预设的静态 per-token 量化参数 δ
        X_int = clamp(round(X / δ) + z, 0, 2^b - 1)
        // INT 矩阵乘法 + 反量化
        Y = dequant(X_int · W_int, δ_x, δ_w)
    ```
  - GPS 数学推导核心（Taylor 展开 + 求导得闭式解）：
    - 量化损失分解（Eq. 3）：E(x,W) ≤ E_x + E_W = E[L(x̂,W) − L(x,W)] + E[L(x,Ŵ) − L(x,W)]
    - Taylor 展开（Eq. 4-6）：E_W ≈ ½·ΔW^T·H^(W)·ΔW，E_x ≈ ½·Δx^T·H^(x)·Δx，用 MSE 近似 Hessian
    - 引入 scaling 后（Eq. 11-12）：E'_x < E_x（激活量化损失降低），E'_W > E_W（权重量化损失增加）
    - 缩放增益函数（Eq. 13）：g(s₂) = g_x − g_W ∝ s₂²/s₁²·W²Δx² + s₁²/s₂²·ΔW²x²
    - 对 s₂ 求导得闭式解（Eq. 14-16）：∂g/∂s₂ = 0 ⇒ s₂ = s₁·√(Σ|ΔW·x|)/√(Σ|W·Δx|)
  - 推理加速实测（8-bit RAR-L, RTX 3090, batch=100, seq_len=256）：PTQ4ARVG 达 2.87× speedup（FP: 3722ms → Quant: 1297ms），峰值内存从 4241MB 降至 2186MB（1.94× 压缩）。对比 QuaRot 因在线 Hadamard 旋转导致 0.70× slowdown（6062ms vs FP 3722ms）。
