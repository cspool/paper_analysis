## PM-KVQ Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- baseline方法是什么？
  - **KIVI**（代表主流 post-training KV Cache 量化方法）：对 Key Cache 做 per-channel 量化、Value Cache 做 per-token 量化，使用 group-wise 非对称量化，保留首 token 和最近窗口内 token 的 FP16 全精度。在 long-CoT 场景下的两个核心缺陷：
    - **(1) 大累积误差**：KIVI 在每次解码步直接把新生成的 KV Cache 量化为目标位宽（如 2-bit），且采用 uniform bit-width 分配。如图 1(a) 左所示，在生成初期显存大量闲置（因为 token 数远未达到最大上下文长度），但这些本可用来以高精度存储早期 token 的显存被浪费了。在 32K 上下文 long-CoT 推理过程中，每个解码步的量化误差累积，导致随 token 增多推理质量急剧下降。
    - **(2) 短上下文校准无法反映长上下文数据分布**：RoPE 将位置信息通过不同频率的正弦/余弦注入 Key Cache 各通道，低频通道（如 DeepSeek-R1-Distill-Qwen-7B 的最低频通道周期达 54410 tokens）在短校准数据（512 tokens）下只能观察一小段正弦曲线，无法获得准确的 channel-wise reparameterization factor λ_i = (max_m K_{m,i})^α，导致 outlier channel 被错误平滑。
  - 全栈执行例子（以 KIVI 执行 2-bit DeepSeek-R1-Distill-Qwen-7B long-CoT 推理为例）：
    - **算法层**：prefill 阶段计算 K = X·W_K, V = X·W_V → Key Cache 做 per-channel group-wise 量化（G=128）→ Value Cache 做 per-token group-wise 量化 → 首 token INT16 保留 → 每个 decoding step：新 token 直接量化为 2-bit → 最近 128 token 保留 INT16 → 注意力计算时反量化到 FP16 做 softmax attention → 输出。全程 uniform 2-bit，显存利用率低。
    - **系统框架层**：论文未明确说明 serving 框架。评测使用 HuggingFace Transformers + fake quantization（不实际节省显存，仅模拟量化误差），在 8×A100-80G 服务器上运行。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PM-KVQ 通过三项创新解决 baseline 累积误差和校准偏差问题：
    - **(1) 渐进量化（Progressive Quantization）→ 解决累积误差**：不再在每次解码步直接量化到目标 Fbit，而是先从 16-bit 开始存储 KV Cache，当显存预算被占满后，通过"等价右移"逐步将已存储的 KV Cache 位宽从 16→8→4→2 缩减（`X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b`），为新 token 腾出空间。在 32K 上下文的例子中（Fbit=2），第 1-2K token 以 16-bit 存储（零量化误差），第 2K-4K token 以 8-bit 存储（极低误差），以此类推。只有最后一部分 token 以 2-bit 存储。这使得 long-CoT 推理过程的前期（最多 token）享受低误差，后期才承受高误差——远优于 KIVI 的全程 2-bit。
    - **(2) 块级内存分配（Block-wise Memory Allocation）→ 解决 uniform bit-width 浪费**：用一阶泰勒近似估计每个 transformer block 的 KV Cache 敏感度 `s_{i,b}`，将位宽分配形式化为 Integer Programming 问题：`min Σ_i Σ_b x_{i,b}·s_{i,b}` s.t. `Σ_i Σ_b x_{i,b}·(Mem(Q_b(K_i)) + Mem(Q_b(V_i))) ≤ M`。CVXPY 在数秒内求解，为深层 block（更敏感）分配更高 Fbit。当 batch size 从 40 减少到 32（单样本显存更多但仍不足以统一升到 4-bit）时，PM-KVQ 将多余显存分配给敏感 block，额外提升 0.84%。
    - **(3) 位置插值校准（Calibration with Positional Interpolation）→ 解决短校准数据偏差**：在 RoPE 旋转矩阵中引入位置缩放因子 s：`cos(s·mθ_i)`，使 2048 token 的校准数据模拟长上下文（s=4 → 有效 8192 token）的位置分布。如图 1(c) 底所示，低频通道的完整正弦周期得以在短校准数据中展现，λ_i 校准更准确。消融实验：2048 token + s=4 的 pass@1（48.33%）与直接使用 8192 token 校准（48.33%）持平，远超无插值 baseline（46.67%）。
  - 论文方法全栈执行例子（以 PM-KVQ 执行 2-bit DeepSeek-R1-Distill-Qwen-7B long-CoT 推理，batch=40，Fbit=2 为例）：
    - **算法层**：
      - 离线阶段：加载校准数据（512 seqs × 2048 tokens，arXiv RedPajama）→ 逐 block 计算 s_{i,2} 和 s_{i,4} → CVXPY 求解 ILP 得到每个 block 的 Fbit（如 block 1 最敏感→4-bit，block 28 最敏感→4-bit，其他→2-bit）→ 位置插值校准 position scaling s=4（有效 8192）→ 计算 channel-wise reparameterization λ_i → 应用式(9)将 Key Cache outlier 迁移到 Query。
      - 推理阶段：prefill 计算 K, V → 首 token INT16 → 渐进量化循环：t=1..2048 以 16-bit 存储（显存未满）→ t=2049 触发 16→8 bit 缩减（等价右移，更新 S 和 Z）→ 腾出空间继续 8-bit 存储 → 类似地 8→4→2 逐步缩减直到 32K 上下文 → 注意力计算时混合精度：首 token INT16 + 滑动窗口 128 token INT16 + 渐进量化部分按各自当前位宽反量化到 FP16 做 attention。
    - **系统框架层**：8×A100-80G GPU 服务器，HuggingFace Transformers，fake quantization 评测。论文明确声明未与系统级优化和推理引擎结合（Limitations 章节）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PM-KVQ 是纯算法方法，位宽缩减通过整数移位操作实现，不涉及自定义 CUDA kernel 或硬件修改。

  **所有三个技术的协同关系**：渐进量化解决单体请求内的累积误差（时间维度）；块级内存分配解决跨层的显存分配优化（空间维度）；位置插值校准解决校准阶段的值域估计准确性（数据维度）。三者正交且互补，共同将 long-CoT LLM 的 2-bit KV Cache 推理从接近随机（RotateKV/MiKV pass@1≈0%）提升到接近 FP16 水平（差距 < 5%）。
