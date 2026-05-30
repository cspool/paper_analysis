## PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  PM-KVQ 提出三个关键技术实现 KV Cache 后训练量化以降低长 CoT 推理的内存开销：(1) **Progressive Quantization**：逐步降低 KV Cache 位宽（16→8→4→2 bit），充分利用目标硬件内存预算，而非每步直接量化到目标位宽；(2) **Block-wise Memory Allocation**：对不同敏感度的 transformer block 分配不同位宽，通过一阶泰勒近似估计敏感度，建模为整数规划问题并用 CVXPY 求解；(3) **Calibration with Positional Interpolation**：对短标定数据施加位置插值（RoPE 中位置索引乘以缩放因子 s），在不增加标定开销的情况下近似长上下文数据分布。
  实验比较对象为 RotateKV、MiKV、KIVI，在 AIME-2024/2025、CMIMC-2024、LiveCodeBench 上评测数学推理和代码生成能力。

- 硬件平台是什么，配置是什么。
  性能评测使用 8×A100-80G GPU 服务器进行 fake quantization 实验。目标 GPU 配置取决于模型规模：DeepSeek-Qwen-7B 使用 1×4090-24G；DeepSeek-LLaMA-8B 使用 1×4090-24G；DeepSeek-Qwen-14B 使用 1×A100-40G；DeepSeek-Qwen-32B、QwQ-32B 使用 1×A100-80G；DeepSeek-LLaMA-70B 使用 1×A100-80G（论文未明确说明 70B 的具体 GPU 配置，但从上下文推断为 A100-80G）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Qwen-7B/14B/32B、DeepSeek-R1-Distill-LLaMA-8B/70B、QwQ-32B。
  标定数据集：RedPajama arXiv 子集，随机选取 512 个样本，每个长度 2048 tokens。位置插值 s=4（嵌入 8192 上下文到 2048 tokens），α 通过网格搜索在 [0,1] 区间寻优（grid size=20），最小化自注意力算子的重建损失。
  评测 Benchmark：AIME-2024、AIME-2025（各30道数学竞赛题）、CMIMC-2025（代数/组合/几何，各10道标准题）、LiveCodeBench（2025年1月1日至4月6日的代码生成题）。数学题每道采样 16 个回答，代码题每道采样 4 个回答，temperature=0.6，top-p=0.95，最大输出长度 32768 tokens。指标为 pass@1 和 Voting accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/thu-nics/PM-KVQ

  算法 Pipeline（三步，推理前预处理 + 推理时执行）：

  **Step 1（预处理）—— Block-wise Memory Allocation：**
  对每个 transformer block i 和每个候选位宽 b，用校准数据前向传播记录 KV Cache 梯度 G_Ki, G_Vi。
  计算敏感度 s_{i,b} = ||G_Ki ⊙ (Ki - Q_b(Ki))||_1 + ||G_Vi ⊙ (Vi - Q_b(Vi))||_1
  求解整数规划：min Σ_i Σ_b x_{i,b} · s_{i,b}, s.t. Σ_b x_{i,b}=1, Σ_i Σ_b x_{i,b} · Mem(Q_b(Ki)+Q_b(Vi)) ≤ M。用 CVXPY 求解器（几秒内完成）。
  可选位宽集合 B：DeepSeek-LLaMA-8B 使用 {4,8}，其他 LLM 使用 {2,4}。

  **Step 2（预处理）—— Calibration with Positional Interpolation：**
  在 RoPE 中对位置索引 m 乘以缩放因子 s：θ' = s · m · θ^{-2i/d}
  然后用修正后的 RoPE 进行通道级重参数化标定：
  Λ = diag(λ_i)，λ_i = (max_m K_{m,i})^α
  P = (QΛ) · Q((KΛ^{-1})^T)，将 Key 中的 outlier 迁移到 Query 中。

  **Step 3（推理时）—— Progressive Quantization + Equivalent Right Shift：**
  初始阶段：以 FP16/INT16 存储 KV Cache。
  当内存预算耗尽时，执行位宽收缩：
  从 16bit → 8bit → 4bit → 2bit（Fbit），每次用 Equivalent Right Shift：
  X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b
  保持零点不变 (Z_b = Z_{2b})，缩放因子放大为 S_b = (2^b + 1)S_{2b}。
  同时保留首 token 为 INT16，最近 128 tokens 用滑动窗口保留 INT16（继承 KIVI/SKVQ 策略）。
  量化方式：非对称分组量化（group size=128），如公式 X_asym = ⌊(X_FP16 - Z) / S_asym⌋, S_asym = (max(X_FP16) - Z) / (2^N - 1)。

  **性能结果：**
  DeepSeek-Qwen-7B (2-bit): PM-KVQ pass@1 40.00% vs KIVI 32.08% on AIME-2024，提升 ~8%。
  DeepSeek-LLaMA-8B (4-bit): PM-KVQ pass@1 47.71% (BS=6, block-wise) vs KIVI 41.25%，甚至超过 16-bit 的 44.17%。
  DeepSeek-LLaMA-70B (2-bit): PM-KVQ pass@1 64.79% vs KIVI 51.88%，提升 12.91%。
