## PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PMQ-VE 是一个面向多帧视频增强模型的 PTQ 量化框架（W/A 联合量化），包含粗-细两阶段过程：(1) **BMFQ（Backtracking-based Multi-Frame Quantization）**——粗阶段：对多帧激活张量 X∈R^{N×C×H×W} 进行 per-frame 独立量化，为每帧 X_i 搜索最优裁剪边界 (lb_i, ub_i)。采用百分位数初始化（lb∈[p0.1, p10], ub∈[p90, p99.9]），通过回溯搜索算法（BTBI）在候选空间中递归评估量化误差，剪枝后回溯以寻找最优解；(2) **PMTD（Progressive Multi-Teacher Distillation）**——精阶段：层次化蒸馏框架，训练低比特模型时使用 FP 教师和中间比特教师（如 INT8）共同监督。损失函数 L_PMTD = (L_INT + α(t)·L_FP) / (1+α(t))，α(t) 线性增长，L_INT 和 L_FP 各自包含输出重建损失（L2）和中间特征匹配损失（MSE, λ=5）。量化函数为 fake quantization（uniform quantizer + STE），量化 Linear 和 MatMul 层。
  - 实验比较：PMQ-VE (W2A2, W4A4) vs **OpenVINO** [Gorbachev et al. 2019], **TensorRT** [Vanholder 2016], **SNPE** [Ignatov et al. 2018], **Percentile** [Li et al. 2019], **MinMax** [Jacob et al. 2018], **NoisyQuant** [Liu et al. 2023], **DBDC+Pac** [Tu et al. 2023], **2DQuant** [Liu et al. 2024]，在三个视频增强任务（STVSR, VSR, VFI）的多个 benchmark 上比较 PSNR、SSIM、LPIPS、NIQE。

- 硬件平台是什么，配置是什么。
  - 8 张 NVIDIA V100 GPU。PyTorch 框架实现。训练：Adam 优化器，初始 lr=2×10^-4，Cosine Annealing，20000 次迭代。初始化和蒸馏微调阶段 batch size 分别为 8 和 2 per GPU。数据增强：随机裁剪、旋转、翻转。

- 模型是什么。数据集和bench分别是什么。
  - 模型/backbone：**RSTT** [Geng et al. 2022]（STVSR 任务）、**MIA** [Zhou et al. 2024]（VSR 任务）、**EMA-VFI** [Zhong et al. 2024]（VFI 任务）。均为 Transformer-based 视频增强模型。
  - 数据集/benchmark：**Vimeo-90K** [Xue et al. 2019] 训练集用于所有任务训练；**Vid4** [Liu & Sun 2013] 和 Vimeo-90K 测试集（含 Vimeo-Fast/Medium/Slow）作为评估 benchmark。评估指标：PSNR、SSIM（Y 通道 YCbCr 色彩空间）、LPIPS [Zhang et al. 2018]、NIQE [Mittal et al. 2012]。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/xiaoBIGfeng/PMQ-VE（论文承诺开源），PyTorch 实现。
  - BMFQ 伪代码（BTBI 算法）：
    ```
    输入: 多帧激活张量 X ∈ R^{N×C×H×W}, 步长 ΔL, ΔU, 阈值 ε
    输出: 每帧最优边界 (lb_i*, ub_i*), i=1..N
    
    For each frame i in 1..N:
        X_i = X[i,:,:,:]
        lb_0 = p_0.1(X_i), ub_0 = p_99.9(X_i)   // 百分位初始化
        visited = {}, error_min = ∞
        
        Function Backtrack(lb, ub):
            if (lb, ub) in visited or out of [p0.1,p10]×[p90,p99.9]:
                return
            visited = visited ∪ {(lb, ub)}
            X_q = Quantize(X_i, lb, ub)           // clamp + round + dequantize
            err = ||X_i - X_q||_2
            if err > error_min + ε: return        // 剪枝
            if err < error_min:
                error_min = err; lb_i* = lb; ub_i* = ub
            foreach (δ_l, δ_u) in {±ΔL, ±ΔU}:
                Backtrack(lb+δ_l, ub+δ_u)
        
        Backtrack(lb_0, ub_0)
    ```
  - PMTD 蒸馏流程：
    ```
    // 训练 4-bit 模型
    // Teacher 1: FP32 教师模型; Teacher 2: INT8 中间教师模型
    for t in 1..T:
        // 前向传播
        out_student = QuantizedModel_4bit(x)
        out_int8 = INT8_Teacher(x)           // 中间教师输出
        out_fp = FP32_Teacher(x)             // FP 教师输出
        
        // 损失计算
        L_rec_int = ||out_student - out_int8||^2   // L2 重建损失
        L_feat_int = MSE(feat_student, feat_int8)  // 特征匹配
        L_INT = L_rec_int + λ * L_feat_int         // λ=5
        
        L_FP = L_rec_FP + λ * L_feat_FP
        
        α(t) = min(1, t/T_warmup)           // 线性增长权重
        L = (L_INT + α(t)*L_FP) / (1+α(t))
        
        // 反向传播（STE 梯度）
        L.backward()
        optimizer.step()
    ```
  - 量化公式（fake quantization）：
    ```
    x_clip = clamp(x, lb, ub) = min(max(x, lb), ub)
    Δ = (ub - lb) / (2^N - 1)
    x_int = round((x_clip - lb) / Δ)
    x̂ = x_int * Δ + lb
    // 梯度通过 STE: ∂x̂/∂x = 1 if x∈[lb,ub] else 0
    ```
  - 消融实验效果（STVSR 2-bit 设置）：Baseline（无任何核心模块）12.67dB → +Per-Frame Quantization 19.64dB → +BMFQ 27.56dB → +PMTD 30.33dB。
