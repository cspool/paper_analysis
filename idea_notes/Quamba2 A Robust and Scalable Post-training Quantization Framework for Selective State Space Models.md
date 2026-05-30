## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- baseline方法是什么？
  - Baseline 方法：(a) **Quamba**（Chiang et al. 2025）：仅支持 W8A8，仅 Mamba1 backbone。使用 percentile clipping 量化 SSM 输入 x_t，对 output projection input 应用 online Hadamard transform 消除 outlier。不支持 4-bit 权重、不支持 Mamba2、不做 head-to-toe 量化（embedding 和 lm_head 保持 FP16）。(b) **MambaQuant**（Xu et al. 2025）：支持 W8A8 和 W4A8，但仅 Mamba1 backbone。使用 variance-aligned rotation 方法量化。W4A8 下性能显著下降（Mamba1-2.8B W4A8 58.5% avg vs FP16 62.2%）。
  - Baseline 在模型推理全栈的执行例子（以 Quamba W8A8 Mamba2-8B 为例）：
    - **算法pipeline**：FP16 Mamba2-8B → 收集 calibration stats → percentile clipping 量化 x_t（在线执行）→ online Hadamard transform on output proj input → W8A8 per-tensor/per-channel 量化 weights → embedding/lm_head 保持 FP16。量化粒度粗（per-tensor/channel），导致 Mamba2-8B W8A8 仅 64.8% avg accuracy（vs FP16 70.7%），差距 5.9%。
    - **系统框架**：论文未明确说明 Serving 框架集成。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT8 GEMM kernel + online FWHT kernel + percentile clipping kernel（在线排序+截断，引入额外延迟）。无 4-bit kernel 支持。
    - **硬件架构**：NVIDIA A5000 GPU。W8A8 Mamba2-8B TPOT = 14.12ms, TTFT = 124.01ms。FP16 embedding/lm_head 阻止在 Orin Nano 8G 部署（OOM）。
  - Baseline 的核心缺陷：(1) **bit-width 单一**：Quamba 仅 W8A8，MambaQuant W4A8 精度差，无法覆盖不同部署场景（W4A8 大 batch 高吞吐 vs W4A16 单用户低延迟）；(2) **Mamba2 精度差**：Quamba 的 clipping+Hadamard 在 Mamba2 上效果差（W8A8 仅 64.8% vs FP16 70.7%），因为未利用 SSD 的 channel order preserving 和 activation persistence 特性；(3) **不做全模型量化**：embedding/lm_head 保持 FP16 导致显存瓶颈，无法在边缘设备部署；(4) **SSM 输入参数量化粗糙**：per-tensor/channel 量化 x_t/B_t/C_t 对 SSM 线性递归的误差极度敏感。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Quamba2 提出两大核心技术：(1) **Sort-and-cluster** 量化 x_t：利用 SSM 的 **channel order preserving**（SSD 计算 channel-wise，输入 channel 顺序=输出 channel 顺序）和 **channel persistence**（各 channel 激活幅度在不同输入间保持一致），先 offline 校准 channel max → 排序 channel → 聚类 head（m 组）→ 每组 head 内再聚类 channel（n 组）→ 共 m×n 个 scaling factor 精细量化 x_t 到 8-bit，配合 offline cluster-aware weight reordering 保证计算正确性。(2) **Per-state-group quantization** 量化 B_t/C_t：利用 **state persistence**（B 和 C 中激活的 state group 在时间步和样本间一致），对每组 state group 使用独立 scaling factor，大幅提升小数值 group 的量化精度。
  - 全栈执行例子（Quamba2 W4A8 Mamba2-8B on A5000）：
    - **算法pipeline**：FP16 Mamba2-8B → Pile 512 句 calibration → (a) 记录 x 各 channel max，排序，对 head 聚类 m=4 组，每组内 channel 聚类 n=4 组 → 16 个 scale per layer 量化 x_t；(b) 记录 B/C state group 激活模式 → per-state-group scale；(c) offline cluster-aware reorder: W_in 列/W_out 行/W_conv channel/W_norm 全部按 cluster indices 重排；(d) offline Hadamard fusion: W_in^H=W_in@H^T, W_out^H=H@W_out@H^T；(e) GPTQ 优化 4-bit weights per-group；(f) W4AX 进化搜索自动选择每层 W4A8/W4A16。效果：W4A8 Mamba2-8B 69.1% avg（vs FP16 70.7%, -1.6%），W4A16 69.8%（-0.9%）。
    - **系统框架**：集成 vLLM，替换所有 projection/SSD/conv/embed 层为 Quamba2 量化 kernel。支持 head-to-toe 量化使 Mamba2-8B 部署在 Orin Nano 8G（13 TPS）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：W4A8 input proj: 4-bit weights + 8-bit acts → per-group dequant → INT8 Tensor Core matmul → fused output scale → INT8 output。SSD kernel: 8-bit A/B/C/x/cached states 加载 → INT8 scan → 写回 8-bit cached states（HBM traffic 减半）。FWHT kernel: 内联 scaling factor s_y 避免额外计算。W4A16 gen: 4-bit weights → dequant → FP16 matmul（memory-bound, TPOT 7.58ms）。W4A8 gen: 8-bit INT8 matmul（TPOT 7.43ms + state 压缩带来更大 batch 支持）。
    - **硬件架构**：NVIDIA A5000 GPU 24GB (cloud), Orin Nano 8G (edge)。Quamba2 支持所有 bit-width 在 roofline model frontier：小 batch→W4A16 memory-bound 最优，中 batch→W4A8 平衡，大 batch→W8A8 compute-bound 吞吐最高。4× memory reduction（15.7GB→1.4GB for Mamba2-2.7B W4A8），1.3× prefilling speedup，3× generation speedup。
