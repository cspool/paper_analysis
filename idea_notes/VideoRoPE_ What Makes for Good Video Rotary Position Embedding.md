## VideoRoPE: What Makes for Good Video Rotary Position Embedding

- baseline方法是什么？
  **M-RoPE** (Wang et al., 2024a, Qwen2-VL) 是当前 Video LLM 最广泛采用的 3D position embedding 方案。它将 d=128 维 head 分为三组：前 32 维（高频率，θ_n = β^{-2n/d}）用于 temporal t，中间 48 维（中频）用于 horizontal x，后 48 维（低频）用于 vertical y。这种设计的核心缺陷通过 V-NIAH-D 任务暴露：

  **缺陷 1 — 时间维度高频分配导致周期性振荡**：低维对应的高频 θ_n 产生短单调区间，cos(θ_n·t) 在远距离上周期性重复。M-RoPE 中 t 使用前 16 个旋转角（如 θ_13, θ_14, θ_15），当帧号从 0 到 3000 时，cos(θ_n·t) 多次经过 0 产生"hash collision"——距离很远的两个位置有几乎相同的 temporal embedding。这使 distractor 帧（距 needle 200 帧插入）在 temporal 维度与 needle 不可区分，模型被误导。注意力可视化显示 M-RoPE 的 temporal 注意力集中在对角线附近（仅关注局部帧），实际定位 needle 依赖的是 vertical 维度而非 temporal 维度。

  **缺陷 2 — 无空间对称性（Spatial Asymmetry）**：M-RoPE 的 visual token 位置索引在每帧内从 (0,0) 到 (W-1,H-1)，导致每帧最后一个 visual token 总在 (W-1,H-1) 处形成"corner stack"，preceding text end 到 visual start 的距离 ≠ visual end 到 subsequent text start 的距离。

  **缺陷 3 — 无时间索引缩放**：M-RoPE 中所有维度使用相同的 index increment=1，不区分 temporal frame spacing 和 spatial pixel spacing 的差异。

  全栈执行例子（M-RoPE + Qwen2-VL-7B）：
  - 模型推理算法层：视频 → ViT 编码 → 每帧 W×H patches → 拼合文本→visual→文本序列 → M-RoPE 分配 position IDs：t 维度前 32 维（高频），x 维度中间 48 维，y 维度后 48 维 → 计算 RoPE rotation → Qwen2-7B LLM 前向 → 生成答案。
  - 系统框架层：vLLM 推理（>32K token 序列），无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，FlashAttention 加速，无自定义 kernel。
  - 硬件架构层：NVIDIA A100 GPU 推理。704 GPU hours fine-tuning。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoRoPE** 通过三个协同设计模块解决 M-RoPE 的缺陷：

  **(1) Low-frequency Temporal Allocation (LTA) → 解决缺陷 1**

  Baseline (M-RoPE) → t 使用低维（高频 θ_n），cos(θ_n·t) 在远距离上周期性重复。
  VideoRoPE → t 使用高维（低频 θ_n，dims 48-63 即 θ_48 到 θ_63），θ 值极小（如 θ_63 ≈ 10000^{-126/128} ≈ 0.00011），cos(θ_n·t) 在 3000 帧范围内几乎单调不减，不会产生 hash collision。x 和 y 维度交叉排列在低维（dim 0-47），因为它们处理的是分辨率受限的空间信息，高频足以覆盖所有空间位置。效果：V-NIAH-D 87.11%（M-RoPE 74.67%），temporal 维度在注意力中成功捕获长程 needle。

  **(2) Diagonal Layout (DL) → 解决缺陷 2**

  Baseline (M-RoPE) → 每帧 visual token 从 (0,0) 到 (W-1,H-1) 排列，corner stack。
  VideoRoPE → 整个输入沿对角线排列。第 0 帧中心 patch 坐标为 (Ts, Ts, Ts)，第 τ 帧中心 = (Ts+δτ, Ts+δτ, Ts+δτ)，其他 patch 偏移 ±(w-W/2, h-H/2)。preceding text (0..Ts-1 → 0..Ts-1) 到 visual start (Ts) 的距离 = visual end 到 subsequent text start 的距离，满足 Eq.5 对称性。

  **(3) Adjustable Temporal Spacing (ATS) → 解决缺陷 3**

  Baseline (M-RoPE) → 所有维度 index increment=1。
  VideoRoPE → 引入 δ=2，帧间 temporal index 增量为 2，而 spatial 和 text index 增量为 1，解耦时间与空间尺度差异。t = T_s + δ(τ-T_s)，x = t + w - W/2，y = t + h - H/2。

  全栈执行例子（VideoRoPE + Qwen2-VL-7B）：
  - 模型推理算法层：视频(128帧, 2FPS) → ViT 编码 → 构建 3D position IDs(t=T_s+2·f_idx, x=t+w-W/2, y=t+h-H/2) → d=128: dims[0:48] 交叉 x/y, dims[48:64] 为 t（低频 LTA）→ 计算 RoPE 旋转 → Qwen2-7B LLM → 生成。
  - 系统框架层：vLLM Serve-API 推理，Qwen2-VL fine-tuning pipeline（LR=1e-5, cosine scheduler, warmup=0.01）。训练 8K context，推理支持 128K。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers + FlashAttention。VideoRoPE 仅修改 position ID 计算和 RoPE dimension allocation，不涉及 kernel 修改。
  - 硬件架构层：训练 704 NVIDIA A100 GPU hours，推理单 A100。128K 推理 vLLM 支持。

  对比 baseline 的解决效果：
  | 缺陷 | M-RoPE 表现 | VideoRoPE 解决方式 | 效果 |
  |------|-------------|-------------------|------|
  | 时间维度高频振荡 | V-NIAH-D 74.67% (-4.0 vs V-NIAH) | LTA: 低频分配, 避免 hash collision | V-NIAH-D 87.11% (+12.44) |
  | 无空间对称性 | Corner stack, text-visual 非对称 | DL: 3D 对角线排列 | 施加式对称性 |
  | 时间/空间尺度统一 | index incr=1 所有维度 | ATS: δ=2 temporal scaling | Avg 60.92 (δ=2) |
  | LongVideoBench 64K | 54.35 | LTA+DL+ATS | 57.26 (+2.91) |
  | MLVU 64K | 61.10 | LTA+DL+ATS | 65.56 (+4.46) |
  | Temporal Hallucination | 29.0 | VideoRoPE | 58.5 (+29.5) |
  | 128K extrapolation | 51.45 (LVB) | VideoRoPE | 55.64 (+4.19) |
