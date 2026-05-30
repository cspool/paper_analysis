## QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuantSparse 是一个统一的后训练量化（PTQ）+ 注意力稀疏化框架，针对 Video Diffusion Transformer（如 HunyuanVideo-13B、Wan2.1-1.3B/14B）。核心算法 pipeline 包含两大组件：(1) **Multi-Scale Salient Attention Distillation (MSAD)**：在校准阶段，通过 Global Guidance（对 Q/K 做 average pooling stride=s 下采样后计算低分辨率 attention，MSE 蒸馏 FP 与量化模型的 attention map）和 Local Guidance（根据 token saliency 分布选择 top-k salient queries 做高分辨率 attention 蒸馏）两种互补机制，高效对齐量化前后的 attention 分布（内存从 O(L²) 降至 O(L̃²+kL)）。(2) **Second-Order Sparse Attention Reparameterization (SSAR)**：在推理阶段，利用量化噪声在扩散过程中呈缓变随机过程的特性，发现二阶残差 Δ̃_quant^(t)=Δ_quant^(t)−Δ_quant^(t-1) 在不同时间步间具有显著高于一阶残差的时间稳定性；缓存参考时间步的一阶+二阶残差，在稀疏 attention 输出上叠加缓存残差近似全 attention；进一步对二阶残差做 SVD 投影到 top-r 主成分上抑制时间方差。整体流程：校准阶段 block-wise PTQ → MSAD 优化量化参数（minimize L_quant + λ_global L_global + λ_local L_local）→ 推理阶段量化稀疏 attention + SSAR 残差校正（cache interval=5）。量化粒度：channel-wise weight + dynamic token-wise activation，uniform symmetric quantization。稀疏 mask 使用 SVG（SparseVideoGen）的静态 spatial-temporal pattern。
  - 实验比较：(a) QuantSparse vs 纯量化 baseline（PTQ4DiT, Q-DiT, SmoothQuant, QuaRot, ViDiT-Q, Q-VDiT）在 W6A6/W4A8 下；(b) QuantSparse vs 纯量化+稀疏化 naive 组合（QuaRot+DFT/Jenga/SVG, Q-VDiT+DFT/Jenga/SVG）在多种 density 下；(c) 多指标评估：CLIPSIM, VQA, FlowScore（绝对质量）+ PSNR, SSIM, LPIPS（与 FP16 的差异）；(d) VBench benchmark 8 维度评估（IQ, AQ, MS, DD, BC, SuC, ScC, OC）；(e) 消融：MSAD（global/local 各自贡献）、SSAR（无/first-order/second-order/SVD 各级贡献）、cache interval、attention density、pooling stride s、salient token k、λ 权重、SVD rank r、full attention distillation vs MSAD 效率对比；(f) 效率分析：模型存储、显存消耗、DiT 推理时间、端到端加速比；(g) 与其他加速技术组合：SageAttention + TeaCache；(h) 校准资源开销。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA A800 80GB GPU，CUDA 12.4。INT 矩阵乘法使用 CUTLASS 基于 PyTorch 实现。校准和推理均在单 A800 上完成。

- 模型是什么。数据集和bench分别是什么。
  - 模型：HunyuanVideo-13B (Kong et al. 2024)、Wan2.1-1.3B (Wan et al. 2025)、Wan2.1-14B (Wan et al. 2025)。图像生成扩展：Hunyuan-DiT 1.5B (Li et al. 2024c)。采样步数 50。
  - Calibration：20 个随机生成样本，每个 transformer block 训练 15 epoch。
  - 评估数据：OpenSORA prompt sets（与 Zhao et al. 2024, Feng et al. 2025c 相同）。
  - Benchmarks：CLIPSIM (Wu et al. 2021), VQA (Wu et al. 2023), FlowScore (Liu et al. 2024b), PSNR/SSIM/LPIPS (Zhang et al. 2018), VBench 8 维度 (Huang et al. 2024b)。
  - Baseline 量化方法：PTQ4DiT (Wu et al. 2024), Q-DiT (Chen et al. 2024), ViDiT-Q (Zhao et al. 2024), Q-VDiT (Feng et al. 2025c), SmoothQuant (Xiao et al. 2023a), QuaRot (Ashkboos et al. 2024)。
  - Baseline 稀疏化方法：DiTFastAttn/DFT (Yuan et al. 2024, cache-based), Jenga (Zhang et al. 2025d, dynamic pattern), SparseVideoGen/SVG (Xi et al. 2025, static pattern)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wlfeng0509/QuantSparse（MIT License，代码尚未发布，README 声明"will be released soon"）
  - 算法 pipeline 张量计算流程（以 HunyuanVideo-13B W4A8 + 15% attention density 为例）：

  **阶段一：校准（Calibration）**
  1. 从预训练 FP16 video DiT 加载权重。初始化量化参数 {s, z}：channel-wise scale for weights, token-wise dynamic scale for activations。
  2. 采样 20 个 calibration samples，block-wise 逐 block 优化：
     a. FP 前向：X ∈ R^(L×d_in) → Q_fp = X W_q^T, K_fp = X W_k^T, V_fp = X W_v^T（FP16）。
     b. 计算 FP attention: A_fp = softmax(Q_fp K_fp^T / √d_k) ∈ R^(h,L,L)。
     c. 计算 token saliency: s_j = Σ_h Σ_i A_fp[h,i,j]，选 top-k=256 salient queries I。
     d. 量化前向：Q_quant = Q(X) Q(W_q)^T, K_quant = Q(X) Q(W_k)^T, V_quant = Q(X) Q(W_v)^T。其中 Q(·): X_Q = clip(⌊X/s⌋+z, 0, 2^b-1), dequant = s·(X_Q−z)。
     e. Global Guidance: Q̃ = AvgPool(Q_quant, s=128), K̃ = AvgPool(K_quant, s=128) → A_global = softmax(Q̃ K̃^T / √d_k) → L_global = MSE(A_global^FP, A_global^quant)。
     f. Local Guidance: A_local = softmax(Q_quant[I,:] K_quant^T / √d_k) → L_local = MSE(A_local^FP, A_local^quant)。
     g. Total loss: L_distill = L_quant + λ_global·L_global + λ_local·L_local（Wan2.1: λ=1e-4, HunyuanVideo: λ_global=1.0, λ_local=1e2）。
     h. AdamW 优化 s, z（scale LR=5e-2, channel-wise scale/rotation matrix LR=5e-3, cosine scheduler）。
  3. 吸收量化参数到权重中，得到量化模型 M_quant。

  **阶段二：推理（Inference）**
  1. 加载 M_quant，输入 prompt P，初始化缓存 C = {Δ_quant^(t_ref), Δ̃_quant^(t_ref)}。
  2. for t = 0 to T-1（T=50 去噪步）:
     a. 量化稀疏 attention：Q_quant = Q(X_t) Q(W_q)^T, K_quant = Q(X_t) Q(W_k)^T → A_s,q^(t) = SparseAttention(Q_quant, K_quant, V_quant; M) where M 为 SVG spatial-temporal mask（density=15%/25%）。
     b. if t - t_ref ≤ τ（τ=5，cache refreshing interval）:
        - 复用缓存残差: Δ_curr = Δ_quant^(t_ref) + Δ̃_quant^(t_ref)。
     c. else:
        - 计算全 attention A_full^(t)（无 mask）。
        - 更新一阶残差: Δ_quant^(t) = A_full^(t) − A_s,q^(t)。
        - 更新时间步 t-1 全 attention A_full^(t-1) 及其稀疏版 A_s,q^(t-1)。
        - 计算二阶残差: Δ̃_quant^(t) = Δ_quant^(t) − Δ_quant^(t-1)。
        - SVD 投影: SVD(Δ̃_quant) = S U V^T → Δ̃_quant_proj = S_{:,:r} U_{:r,:r} V^T_{:,:r}（r=16）。
        - 更新缓存 C = {Δ_quant^(t), Δ̃_quant_proj}, t_ref = t。
     d. 精炼 attention: Ã^(t) = A_s,q^(t) + Δ_curr（稀疏输出 + 缓存残差 = 近似全 attention）。
     e. Ã^(t) @ V_quant → 后续 transformer block 计算。
  3. 解码 latent → 输出视频 Y。

  **关键张量形状**（HunyuanVideo-13B, 720×1280p, frames=60）：
  - X ∈ R^(L×d_in), L ≈ 10^4+ tokens
  - A_full ∈ R^(h×L×L) → 单层 attention 矩阵内存 ~6.82GB（不可承受）
  - MSAD Global: Q̃,K̃ ∈ R^(L̃×d_k), L̃ = L/s² → attention ~O(L̃²)，s=128 时仅需 ~0.14GB
  - MSAD Local: k=256 个 salient queries → attention ~O(kL)，高效
  - SSAR cache: Δ_quant + Δ̃_quant（first+second order），与 first-order 缓存等量存储
