## QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 QuantCache 框架开发了 optimized GEMM CUDA kernels，通过 kernel fusion 技术将三个运行时计算组件融合为高效的单次 kernel 调用：(1) **Quantization + Rotation kernel fusion**：将 activation quantization（uniform min-max quantize，online 计算 s_X = (max(X)-min(X))/(2^b-1)）与 channel-balancing rotation 变换（R @ X_balanced）融合为单一 CUDA kernel，避免中间结果写回 global memory；(2) **Intermediate feature caching kernel**：在 GPU shared memory / L2 cache 中缓存 HLC 判定可复用的 intermediate features，跨 timestep 直接复用，减少 HBM 访问；(3) **Scaling factor absorption**：受 QServe、SmoothQuant、ViDiT-Q 启发，将 channel-balancing 的 scaling factors offline 吸收到前层权重中（W'_prev = S ⊙ W_prev），消除推理时的额外 scaling 开销。Kernel 融合后，QuantCache 的量化过程不再引入额外的 kernel launch 和 global memory round-trip，使整个 DiT 推理的 CUDA kernel launch 次数显著减少。
  - 实验比较：(a) Speedup 对比：QuantCache 6.72× vs Open-Sora baseline 1.00× on A800-80GB；(b) 对比 T-Gate (1.10×), PAB (1.34×), ViDiT-Q (1.71×), AdaCache-slow (1.46×), AdaCache-fast (2.24×) 的端到端加速比。论文未提供逐 kernel 的 micro-benchmark。

- 后端平台是什么，配置是什么。
  - NVIDIA A800-80GB GPU（Ampere 架构，80GB HBM2e），CUDA 12.1。CUDA kernel 实现受 QServe (Lin et al., MLSys 2025)、SmoothQuant (Xiao et al., ICML 2023)、ViDiT-Q (Zhao et al., ICLR 2025) 启发，吸收了 scaling factor absorption 和 kernel fusion 技术。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：Open-Sora 1.2 推理代码（基于 PyTorch + custom CUDA kernels）。论文未明确说明评估框架名称（论文未使用标准 serving framework）。
  - 修改内容：(1) 将 Open-Sora DiT blocks 中的标准 FP16 GEMM 层替换为 QuantCache 的量化 GEMM kernel（支持 W8A8 和 W4A6 低精度 GEMM）；(2) 在 kernel 内集成 online activation quantization（动态计算 min/max → 计算 scale → quantize → GEMM）；(3) 融合 rotation transformation 到量化 kernel（避免额外 kernel launch）；(4) 实现 HLC 缓存逻辑：在 kernel 输入侧检查 D_t^(l) 是否低于阈值，若是则从 cached buffer 直接读取 feature（存储在 GPU global memory 的 dedicated cache buffer）；(5) 实现 SRAP 剪枝逻辑：计算 S_t^(l,l+1) → 若超过阈值则跳过当前 kernel launch（kernel 调用侧逻辑）；(6) Speedup 在单张 A800-80GB 上测量 end-to-end latency（包含 VAE encode/decode + DiT denoising + 所有 quantization/caching/pruning 开销），100 timesteps。
  - 评估方式：测量 end-to-end video generation wall-clock time（从输入 prompt 到输出 512×512×64 frames 视频），speedup = baseline Open-Sora latency / QuantCache latency。论文未提供 per-kernel 级别的 profiling 数据或 roofline model 分析。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/JunyiWuCode/QuantCache（论文声明 code and models will be available）
  - Kernel 输入到性能输出全过程（以 Open-Sora DiT block Single Kernel Call with W4A6 为例）：
    1. **Kernel Launch 准备**：CPU 侧根据 timestep t 和当前 layer l → 读取 D_t^(l)（HLC 决策）→ 如果命中缓存 → 跳过 kernel launch，直接从 cache buffer 读取 output（HLC 缓存命中 path）。否则继续执行量化 GEMM kernel。计算 S_t^(l,l+1)（SRAP 决策）→ 如果 S > τ_high → 跳过当前 kernel launch（SRAP 剪枝 path）。
    2. **Kernel 输入**：FP16 输入激活 X ∈ R^{seq_len × d_model}（global memory） + 4-bit packed weights W̄ ∈ R^{d_model × d_ff}（global memory） + per-channel weight scales s_W（global memory） + offline fused scaling factors S_absorbed（已吸收到 W̄ 中） + rotation matrix R。
    3. **Kernel 内执行**（单次 kernel launch，fused）：(a) 从 global memory 加载 X tile 到 shared memory；(b) Online activation quantization: 在 shared memory 中计算 min(X_tile) / max(X_tile) → 计算激活 scale s_X = (max-min)/(2^6 - 1)（W4A6 配置）→ quantize: X̄ = clamp(round(X/s_X) + z_X, 0, 63) → 6-bit INT8-compatible representation；(c) 加载 4-bit packed W̄ tile → dequant to INT8: W_deq = W̄ × s_W_tile；(d) Rotation transform fused：X_rot = R @ X̄（轻量 rotation，O(d²)，在 shared memory 中完成）；(e) INT8 Tensor Core GEMM: Y = W_deq @ X_rot（利用 A800 Tensor Core INT8 算力）；(f) Dequant output: Y_FP16 = Y × (s_X × s_W)（fused output scaling）。
    4. **Kernel 输出**：FP16 输出激活 Y_FP16 ∈ R^{seq_len × d_ff} → 写回 global memory → 同时写入 HLC cache buffer（如果 D_t^(l) < δ_1，标记该 feature 为可缓存）。
    5. **性能输出**：end-to-end video generation latency = Σ(kernel launch overhead + kernel compute time + cache hit skip time)。总 speedup = 6.72×（包含所有 HLC cache hit、AIGQ low-bit compute、SRAP skip 的累积收益）。CUDA kernel fusion 使单项 kernel launch overhead 从 3 次（quantize + rotate + GEMM）降为 1 次。
