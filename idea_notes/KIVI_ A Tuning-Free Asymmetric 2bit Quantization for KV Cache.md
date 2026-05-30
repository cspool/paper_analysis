## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- baseline方法是什么？
  KV Cache 量化的 naive baseline 是对 key cache 和 value cache **统一使用 per-token 量化**（如 FlexGen 的 4bit group-wise per-token 量化）。方法是将 KV cache 沿 token 维度分组做 round-to-nearest quantization，新到达的 KV tensor 直接 append 到已有 quantized cache 沿 token 维度。这种流式兼容性良好，但存在根本问题：
  - **Key cache per-token 量化**：由于 key cache 中某些固定 channel 存在极大 magnitude outlier（如图2所示），per-token 量化时这些 outlier 的误差会污染同一 group 内的所有 channel，导致 attention score 相对误差高达 47%（vs per-channel 的 9.6%）。将精度降到 INT2 时，LM-Eval 准确率大幅下降（Llama-2-13B CoQA: 66.37→52.93）。
  - **Value cache per-channel 量化**：由于 attention output 是 value cache 的加权求和（attention score 极为稀疏），per-channel 量化导致 token 间量化误差互相混合，attention output 相对误差比 per-token 高约 15×。INT2 per-channel value 量化导致 CoQA 准确率塌陷至 2.88%。
  - **全栈执行例子（baseline FlexGen 4bit per-token）**：
    - 算法层：统一 per-token group-wise INT4 量化 key 和 value cache
    - 系统框架层：Hugging Face Transformers PyTorch 前端，KV cache 按 token 存储量化张量
    - 编译框架层：论文未明确说明
    - kernel调度层：标准 PyTorch 反量化后 matmul，无反量化融合，量化-反量化在 Python 层面完成
    - 硬件架构层：NVIDIA A100 GPU，标准 HBM→SRAM 加载全精度 KV cache

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **KIVI 提出非对称 2bit KV Cache 量化**：key cache 沿 channel 维度做 group-wise per-channel 量化，value cache 沿 token 维度做 group-wise per-token 量化。关键设计：
  1. **异维度量化**：利用 key/value cache 不同的元素分布特征选择不同的量化轴，同时在各自有利的维度上实现 INT2 精度。
  2. **Grouped + Residual 分割**：为解决 per-channel 量化不兼容流式 append 的问题，将 KV cache 分为 grouped 部分（量化存储）和 residual 部分（FP16 保留，滑动窗口大小 ≤ R=128）。grouped 部分每 G=32 个元素一组量化，residual 部分提供全精度局部上下文，对 GSM8K 等困难任务至关重要。
  3. **Tiled Matrix Multiplication**：将 grouped（量化）和 residual（FP16）两部分用分块矩阵乘法分别计算 attention 后拼接，配合 fused dequantization+MatMul CUDA kernel 和 Triton group-wise quantization kernel。
  - **全栈执行例子（KIVI 方法）**：
    - 算法层：非对称量化——key cache per-channel (沿特征维度分组)、value cache per-token (沿序列维度分组)。prefill 时全精度 key/value 传至下一层，仅保留量化版本在内存。decoding 时新 token 先加入 residual FP16 buffer，residual 满（R=128）后量化并移入 grouped 部分。
    - 系统框架层：基于 Hugging Face Transformers 修改 attention 层 KV cache 管理，使用 grouped+residual 分块数据结构，兼容 weight-only 量化（如 GPTQ/AWQ），可实现 2.6× 峰值内存缩减（Llama-2-7B）。
    - 编译框架层：论文未明确说明
    - kernel调度层：CUDA 实现 Q_MatMul（fused dequantization + tiling matmul，避免 FP16 中间结果写回 HBM），Triton 实现 group-wise quantization kernel（在线计算 min/max → scale/zero-point → round-to-nearest INT2）。
    - 硬件架构层：NVIDIA A100 GPU (80GB)，batch size 增大 4×（相同内存限制下），吞吐量提升 2.35× ∼ 3.47×（ShareGPT workload）。
