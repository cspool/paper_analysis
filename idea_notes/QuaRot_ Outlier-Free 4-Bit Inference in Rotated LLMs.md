## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- baseline方法是什么？
  - Baseline: 传统 LLM 激活量化方法（SmoothQuant, OmniQuant, QUIK, Atom）采用"识别并特殊处理离群值"的策略——用校准集识别离群值特征通道（outlier features），将这些通道保持在高精度（FP16 或 INT8），其余通道量化到低比特。SmoothQuant 通过 per-channel scaling 将量化难度从激活值迁移到权重，解决了 8-bit 量化问题，但在 4-bit 失效。QUIK 和 Atom 在 4-bit 下仍需保留部分高精度通道或使用复杂混合精度矩阵乘法 kernel。这些方法的根本缺陷：(1) **治标不治本**：依赖校准集识别离群值，未从根本上消除离群值产生的原因；(2) **混合精度 kernel 开销大**：需要特殊内存布局分离离群值/正常通道，增加 kernel 复杂度和延迟；(3) **无法全 4-bit**：始终有部分计算或参数保持更高精度，限制了内存节省和加速的理论上限；(4) **KV cache 离群值问题未系统解决**：KV cache 量化（如 KVQuant, KIVI）需要 feature-wise 量化、非均匀表示、保留高精度离群值等复杂机制。
  - 全栈执行例子（Baseline: 4-bit Atom 量化 LLAMA2-7B, NVIDIA RTX 3090）：
    - **算法pipeline**：FP16 LLAMA2-7B → 校准集推理识别激活值离群通道 → 离群通道保留 FP16，非离群通道 per-token 4-bit 量化 → 离线权重量化（GPTQ-128G, group=128）→ 推理时混合精度 MatMul：离群通道 FP16×FP16 + 非离群通道 INT4×INT4 → 需 special reordering kernel 分离两类通道 → KV cache: 论文未明确说明 4-bit KV cache 量化方案 → WikiText-2 PPL: 6.03 (7B), 5.26 (13B)。
    - **系统框架**：Hugging Face Transformers + PyTorch。论文未明确说明 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Atom 混合精度 MatMul kernel：按 outlier channel mask 将输入 X 分拆为 X_outlier (FP16) 和 X_normal (INT4)，分别执行 FP16 GEMM 和 INT4 GEMM → 合并结果。需要特殊的 weight reordering 预处理。离群值 mask 需校准集确定。
    - **硬件架构**：NVIDIA RTX 3090 GPU，Tensor Cores 加速统一精度 GEMM。混合精度路径需额外的 memory reordering 和 kernel launch overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuaRot 从根源上消除离群值——通过随机 Hadamard 旋转将权重和激活值"失相关"（incoherence processing），利用计算不变性定理将旋转矩阵融入网络权重中，使激活值分布均匀化，无需识别和特殊处理任何离群通道。具体设计映射：(1) **旋转消除离群值（对应"治标不治本"）**：随机 Hadamard 矩阵 Q 具有"扩散"效应——Q^T 将单通道的大值均匀分布到所有通道（每通道的 Hadamard 变换是 ±1 加权求和），图 1 验证变换后激活值从长尾分布变为类高斯分布，无任何离群值；(2) **计算不变性保证等价性（对应"混合精度 kernel"）**：利用 RMSNorm 旋转等变性 RMSNorm(X) = RMSNorm(XQ^T)Q，将 Q 融入相邻权重矩阵 W ← Q^T W，前向网络数学上完全等价，因此无需混合精度——所有 MatMul 均为统一 INT4×INT4；(3) **每层仅 1.5 次在线 Hadamard（对应"复杂 kernel"）**：对比 QuIP# 每权重矩阵需 2 次 Hadamard 变换，QuaRot 将大部分 Hadamard 融入权重，仅保留 down-projection 和 out-projection 前的在线变换。Walsh-Hadamard 变换 O(d log d) 极快，在线开销仅 ~7%（Table 14 验证）；(4) **KV cache 全量化（对应"KV cache 未解决"）**：head-wise Hadamard 旋转消除 Key 和 Value 中的离群值，Post-RoPE 在线旋转使量化 KV cache 在 4-bit 下困惑度几乎无损（+0.04 on 7B, +0.03 on 13B, +0.01 on 70B, Table 6），keys 比 values 更敏感（K3V4 困惑度 5.65 vs K4V3 的 5.54）；(5) **无需校准集的无损 8-bit（附加优势）**：RTN 8-bit 量化完全无需校准数据，困惑度 5.50 vs FP16 5.47（Table 3），同时 Hadamard 旋转提升 weight-only 量化质量：4-bit GPTQ 从 8.25 → 5.60 (7B)，2-bit 从 NaN → 22.07（Table 7）。
  - 全栈执行例子（QuaRot: 4-bit LLAMA2-7B, NVIDIA RTX 3090）：
    - **算法pipeline**：FP16 LLAMA2-7B → 离线：生成 Q = H_4096 diag(s), s_i∈{±1}（利用 Walsh-Hadamard O(d log d) 结构和已知 Hadamard 矩阵库 Sloane 2024 处理非 2^n 维度）→ 吸收 RMSNorm α 到相邻权重 → 所有 "输入侧" 权重左乘 Q^T：W_gate/up/k/q/v ← Q^T diag(α) W → 所有 "输出侧" 权重右乘 Q: W_down ← H W_down Q, W_out ← H(I⊗H_{128})W_out Q → W_v 右乘 (I⊗H_{128}), W_out 左乘 (I⊗H_{128})（利用 identity H = (I⊗H_{d_h})(H_{n_h}⊗I)）→ GPTQ 权重量化（128 calib samples, per-column symmetric INT4, MSE-optimal clipping）→ 推理：X 经 RMSNorm → per-token 量化（s_x = max(|X|)×0.9/7, X_q=round(clip(X/s_x,-7,7))) → CUTLASS INT4 GEMM (W_gate/up) → dequant FP16 → SiLU gate → 在线 Hadamard → per-token 量化 → CUTLASS INT4 GEMM (W_down) → dequant → 输出 YQ → 无任何离群值通道，无混合精度。
    - **系统框架**：Hugging Face Transformers + PyTorch。论文未明确说明 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：CUTLASS INT4 TensorCore GEMM（sub-byte packed activations+weights, INT32 accumulator, per-token+per-column scale dequant output）→ 在线 Walsh-Hadamard fast kernel（O(d log d), FP16/FP32, ~7% overhead）→ FlashInfer 量化 KV cache kernel（Append: quantize→pack→store; Decode: load→dequant→FP16 dot product with online softmax）。LLAMA2-7B prefill 加速 2.16×（batch=64, seq=2048），LLAMA2-70B 达 3.33×。解码内存节省 3.63×−3.89×。
    - **硬件架构**：NVIDIA RTX 3090 GPU（Ampere Tensor Cores），FP16/INT4/INT32 精度层次。B200 GPU 的 FP4 硬件支持与本方法的 INT4 路径可类比（论文 conclusion 提出）。
