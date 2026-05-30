## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- baseline方法是什么？
  Baseline 为 MoE 推理时不使用任何 prediction 的标准 Expert Parallelism (EP) 方案。以 Mixtral 8×7B（32 layers, 8 experts/layer, Top-K=2）在 4×A100 NVLink 上的 prefill 推理（batch=1, seq_len=512）为例说明全栈执行路径：
  - **算法层（MoE Routing）**：Gating network 接收 self-attention 输出 → Linear projection → Softmax → SelectTopK(k=2)。路由结果固定（inference 时不可修改 token-to-expert 映射），导致 skewed distribution：如 Expert 1 承接 75% tokens（skewness=3.0）。**缺陷**：(1) FFN compute imbalance——GPU 1（hosts Expert 1）成为 compute bottleneck，延迟被最慢 GPU 主导，放大倍数为 skewness；(2) All-to-All communication imbalance——GPU 1 接收最多 token → 通信 bottleneck 也被 skewness 放大：$(N-1)·skewness/N^2$ vs 平衡时的 $(N-1)/N^2$；(3) 无动态调整机制，skewness 随 workload 变化而无法适应。
  - **系统框架层**：标准 EP 推理 pipeline：Attention（TP, 含 Ring All-Reduce）→ Gating → All-to-All Scatter → Expert FFN → All-to-All Gather → 下一层。无 predictor，无 expert duplication 机制，expert 在每个 GPU 上静态驻留。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch/NCCL 通信 kernel（All-to-All, All-Reduce），无自定义 kernel。
  - **硬件架构层**：4×A100 NVLink 3.0 (2TB/s) fully connected。skewness 导致的 compute + communication imbalance 与硬件 topology 无关——即使 NVLink bandwidth 充足，bottleneck GPU 的 compute delay 仍构成硬上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-GPS——一个系统性能模拟框架，用于量化 MoE 推理中不同 expert prediction 策略的 runtime trade-off，指导选择最优 predictor 设计。核心方法包括两种预测策略（Distribution-Only 和 Token-to-Expert）配合 dynamic expert duplication（Algorithm 1）。
  - **算法层（Expert Prediction Strategies）**：
    1. **Distribution-Only Prediction**：使用 Multinomial Distribution + MLE 对每层 expert 激活概率建模（$\hat{p}_i^l = n_i^l/N$），仅预测 coarse-grained token 分布比例。配合 Algorithm 1 将热门 expert 复制到 underloaded GPU → FFN compute 均衡化。**解决 baseline 缺陷(1)**：通过 expert duplication 打破单 GPU 上的 expert compute bottleneck，skewness 越高收益越大。**代价**：不减少通信（仍做 All-to-All Scatter）。**优势**：zero predictor overhead（offline MLE 估计）。
    2. **Token-to-Expert Prediction**：将 expert selection 建模为分类问题（Probability / Conditional Probability / FFN / LSTM 四类 predictor），预测每个 token 的目标 expert → Direct Routing 跳过 Scatter 阶段。**解决 baseline 缺陷(1)+(2) 同时**：expert duplication 均衡 compute + 跳过 All-to-All Scatter 节省通信。**代价**：predictor inference overhead，accuracy 越高通常 overhead 越大（U-shape trade-off）。
    3. 两种策略的选择由 MoE-GPS simulator 在给定 hardware + model + workload 下自动决策。**解决 baseline 缺陷(3)**：系统可根据实时条件切换策略。
  - **系统框架层（MoE-GPS Simulation Framework）**：以 LLMCompass（ISCA 2024 block-level simulator, silicon-validated）为基础，增强：(1) MoE + EP 模块（custom EP communication + FFN workload 建模）；(2) Mixtral 架构支持（GQA, SwiGLU, Sliding Window）；(3) Prediction strategy 建模（tunable accuracy + overhead）。以 Mixtral 8×7B, 4×A100 NVLink, skewness=1.4, batch=1, seq_len=512 为例：
    - **Distribution-Only 路径**：Offline MLE → 预测 expert 分布 → Algorithm 1 决定 duplication → Expert copy over NVLink (~0.1ms, hidden by Attention) → Attention (TP, ~12ms) → Scatter → Balanced FFN Compute → Gather → 23% speedup vs Token-to-Expert best config。
    - **Token-to-Expert 路径**：Predictor inference (overhead) → Algorithm 1 duplication → Attention → Direct Route (skip Scatter) → Balanced FFN Compute → Gather。Overhead trade-off 导致 U-shape 性能曲线（Figure 6b, 6d）。
    - **决策准则**（Figure 7）：
      - Distribution-Only 更优：low skewness OR high-bandwidth interconnect (NVLink)——因为通信不是瓶颈，predictor overhead 不值得。
      - Token-to-Expert 更优：high skewness（预测更容易，accuracy/overhead 比更优）AND low-bandwidth interconnect（PCIe）——因为节省的通信远大于 predictor overhead。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用 LLMCompass 模拟的 GEMM + communication + element-wise 操作，不涉及自定义 kernel。
  - **硬件架构层**：4×A100 NVLink 3.0 (2TB/s) 和 PCIe 4.0 (32GB/s) 两种配置。核心 insight：interconnect bandwidth 直接决定两种策略的盈亏平衡点——高带宽下 Distribution-Only 几乎始终更优，低带宽+高 skewness 下 Token-to-Expert 有机会反超（Figure 7 的 32GB/s PCIe 场景）。论文还在 Appendix C 验证了 LLaMA-MoE 和 Switch Transformer 上的一致性趋势。
