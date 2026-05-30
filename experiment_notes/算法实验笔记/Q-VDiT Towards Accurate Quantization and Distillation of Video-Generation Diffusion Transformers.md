## Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-VDiT 是一种面向视频生成 Diffusion Transformer (V-DiT) 的后训练量化（PTQ）框架。核心包含两个组件：(1) **Token-aware Quantization Estimator (TQE)**——利用量化误差比原始权重具有更低信息熵的理论（Theorem 3.2，H(Δ)≤H(W)），使用两组低维向量参数 α∈R^{d_in} 和 β∈R^{d_out} 进行低秩误差估计（rank=1），从 token 维度和 feature 维度正交地补偿量化误差。修订后的前向计算：X·W^T ≈ Q̂(X)·Q̂(W)^T + Δ̂·β，其中 Δ̂_{[f_i+1:f_i+s,:]} = (M_i ⊙ Q̂(X)_{[f_i+1:f_i+s,:]})·α，M∈R^t 是 token-aware 缩放因子，对每帧 token 进行选择性缩放，M_i = η_i/ω_i 由量化误差权重 η_i 和显著度量 ω_i 初始化（Eq. 9）；(2) **Temporal Maintenance Distillation (TMD)**——构建 FP 模型中帧间相似度分布 D^{FP}_i = softmax([T^{FP}_{i,1},...,T^{FP}_{i,t}]) 作为先验，用 KL 散度对齐量化模型的帧间时序分布 D^{Q}_i，使每个 frame 的优化考虑整体视频特征分布，弥补 MSE loss 无法捕捉帧间信息的问题。总损失 L_total = L_task + γ·L_temporal，γ=100。
  - 实验比较（VBench benchmark，Tab. 1）：Q-VDiT (W4A6/W3A8/W3A6) vs Q-DiT、PTQ4DiT、SmoothQuant、Quarot、EfficientDM、SVDQuant、ViDiT-Q。8 个维度：Imaging Quality、Aesthetic Quality、Motion Smoothness、Dynamic Degree、Background Consistency、Subject Consistency、Scene Consistency、Overall Consistency。Q-VDiT 在 W3A6 下 Scene Consistency 达 23.40，超过 SOTA（ViDiT-Q 11.99 / EfficientDM 12.04）近 1.9×。W3A8 下 Overall Consistency 达 22.39 vs SOTA 18.53。
  - 实验比较（OpenSORA prompt set 多指标评估，Tab. 2）：CLIPSIM、CLIP-Temp、VQA-Aesthetic、VQA-Technical、ΔFLOW Score、Warping Error。W4A6 下几乎无损（VQA-Aesthetic 67.05 vs FP 66.91）。W3A6 下 VQA-Technical 从 SOTA 29.58 提升到 59.10，CLIPSIM 从 0.1768 提升到 0.1785。
  - 实验比较（Higher bits, Tab. 3）：W8A8/W6A6/W4A8 vs Q-Diffusion、Q-DiT、PTQ4DiT、SmoothQuant、Quarot、ViDiT-Q。W4A8 下 VQA-Aesthetic 达 71.32，甚至超过 FP 模型（66.91）。
  - 实验比较（Latte model on UCF-101, Tab. 6）：Naive-PTQ、ViDiT-Q vs Q-VDiT。指标：FVD、FVD-FP16、CLIPSIM、CLIP-Temp、VQA-Aesthetic、VQA-Technical、ΔFLOW Score、Temporal Flickering。W3A6 下 VQA-Aesthetic 达 19.79 vs ViDiT-Q 16.83。
  - 消融实验（Tab. 4）：PTQ4DiT baseline → +TQE(w/o M) → +TQE(w M) → +TMD → 完整 Q-VDiT，各组件均正向贡献。γ 超参数不敏感（Fig. 6, γ ∈ {0.1,1,10,100,500,1000} 均显著优于 PTQ4DiT baseline）。
  - 定性对比（Fig. 5, Fig. 7-15）：W3A6 下其他方法无法生成有意义图像，Q-VDiT 仍能生成清晰图像且帧间有显著连贯运动变化。

- 硬件平台是什么，配置是什么。
  - GPU 显存校准阶段约 16600-19460 MB（Tab. 5），具体 GPU 型号论文未明确说明。W8A8 校准耗时约 12.5-12.9 小时。推理效率（Tab. 7）：W4A8 下相比 FP16 达到 2.40× 显存节省和 1.35× 推理加速。使用 LoRunner Kernel（来自 SVDQuant）将 TQE 低秩分支（rank=1）与量化 kernel 融合，额外延迟仅约 5%。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Open-SORA (HPC-AI, 2024) —— 基于 PixArt-α 架构的视频 DiT；Latte (Ma et al., 2024) —— class-conditioned 视频 DiT（UCF-101 训练）。仅量化线性层权重（channel-wise quantization），激活使用动态 token-wise quantization。位宽：W4A6、W3A8、W3A6（主打 hard settings），以及 W8A8、W6A6、W4A8（higher bit settings）。
  - **校准数据集**：Open-Sora 提供的 10 个 prompt，均匀选择 50 个去噪步。校准迭代：6-8 bit 为 5k iters，4-bit 为 10k iters，3-bit 为 15k iters；batch size=4；学习率 lr=1e-6（权重量化参数），lr=1e-5（TQE 参数）。
  - **Benchmark**：VBench（8 维度综合视频生成评估）；OpenSORA prompt set（CLIPSIM、CLIP-Temp、DOVER VQA-Aesthetic/Technical、ΔFLOW Score、Warping Error）；UCF-101（FVD、FVD-FP16、Temporal Flickering）。
  - **推理设置**：Open-SORA 用 100-step DDIM，CFG scale 4.0；Latte 用 20-step DDIM solver，CFG scale 7.0。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wlfeng0509/Q-VDiT（MIT License，ICML 2025），基于 Open-Sora v1.0 和 ViDiT-Q，含 calibrate 和 inference 脚本。
  - Q-VDiT 量化 pipeline（以 Open-SORA 单层 Linear 为例，W4A6）：
    ```
    # === 输入: FP 模型权重 W ∈ R^{d_out×d_in}, 视频 latent Z ∈ R^{n×d}, n=s×t ===

    # === Step 1: TQE 参数初始化 (Eq. 8-9) ===
    for each linear layer with weight W:
        α ∈ R^{d_in}: Kaiming normal init
        β ∈ R^{d_out}: zero init (LoRA convention)
        M ∈ R^{t}: token-aware scaling init
          for frame i in [0..t-1]:
            η_i = exp[1-ρ(X_i, Q̂(X)_i)] / Σ_v exp[1-ρ(X_v, Q̂(X)_v)]
            ω_i = Σ_τ |X_{i,τ}| / Σ_v Σ_τ |X_{v,τ}|
            M_i = η_i / ω_i

    # === Step 2: 量化前向传播 (TQE, Eq. 8) ===
    for each linear layer:
        Q_X = quantize(X, s_x, z_x)    # token-wise dynamic quantization
        Q_W = quantize(W, s_w, z_w)    # channel-wise quantization

        # TQE 误差补偿: r=1 低秩近似
        for frame i in [0..t-1]:
            f_i = i * s
            Δ̂[f_i:f_i+s, :] = (M_i ⊙ Q_X[f_i:f_i+s, :]) @ α   # ∈ R^{s×1}

        Y = Q_X @ Q_W^T + Δ̂ @ β^T                              # ∈ R^{n×d_out}

    # === Step 3: TMD 损失计算 (Eq. 13-15) ===
    # S^{FP}, S^{Q} ∈ R^{n×d}, n=s×t
    for frame i, j in [0..t-1]:
        T_fp[i,j] = cosine_sim(S_fp_i, S_fp_j)   # ρ(·,·)
        T_q[i,j]  = cosine_sim(S_q_i, S_q_j)
    for frame i in [0..t-1]:
        D_fp_i = softmax([T_fp[i,0], ..., T_fp[i,t-1]])  # ∈ R^t
        D_q_i  = softmax([T_q[i,0],  ..., T_q[i,t-1]])
    L_temporal = Σ_i KL(D_fp_i || D_q_i)

    # === Step 4: 校准训练 ===
    for iter in 1..total_iters:
        Y_q = quantized_forward(X_calib)   # TQE 修正前向
        Y_fp = fp_forward(X_calib)         # 全精度参考

        L_task = ||Y_q - Y_fp||²           # Eq. (11)
        L_temporal = compute_TMD(Y_q, Y_fp) # Eq. (15)
        L_total = L_task + 100 * L_temporal # γ=100

        # 更新 TQE (α,β,M) 和量化参数 (s,z)
        optimizer.step(L_total)

    # === Step 5: 推理 ===
    # LoRunner Kernel 融合: TQE branch (r=1) + quant GEMM
    # Y = fused_quant_linear(X)  # 额外延迟<5%
    ```
  - 关键理论（Theorem 3.2）：量化误差 Δ = W − Q̂(W) 的信息熵 H(Δ) ≤ H(W)，因为 round-to-nearest 舍入的 decimal truncation 是 surjection (Lemma A.1)。因此 Δ 可在 rank=1 空间估计，参数从 d_out×d_in 降到 d_out+d_in。
  - TMD 梯度分析（Eq. 16-18）：∂L_temporal/∂S^{Q}_i 包含 Σ_j [·T^{Q}_{i,j} + ·T^{Q}_{j,i}] 双向项，且 ∂L_temporal/∂T^{Q}_{i,j} = Σ_k D^{FP}_{i,k}·D^{Q}_{i,j} − D^{FP}_{i,j}（Eq. 17），确保任意帧对的相关性受所有帧共同影响。
