## ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

- baseline方法是什么？
  **Baseline 是现有 MoE offloading 推理系统（Cache-MoE, SE-MoE, Pregated-MoE）**，它们在单 GPU 内存受限场景下将 inactive experts 卸载到 CPU 内存并按需加载。

  **Baseline 全栈执行例子（Cache-MoE 处理两个 batch 的 MoE 推理）**：

  - **算法 Pipeline 层**：Standard MoE 推理 —— router 计算 `softmax(xW_g)` 选 top-k experts，weighted sum 组合 expert FFN 输出。Router 在原模型内，无外部预测器。
  - **Serving 框架层**：Cache-MoE 使用固定 per-layer expert cache + LRU 替换策略。每层独立管理 cache，无视跨层 routing 模式。Token batch 保持原始顺序，不进行重组。Pregated-MoE 用 MLP 逐层预测 expert（sequential dependency），SE-MoE 用 ring-buffer 预取连续两层全部 experts（内存膨胀）。
  - **编译框架层**：论文未明确说明（使用 PyTorch 框架）。
  - **Kernel 调度层**：CPU→GPU expert 加载与 GPU compute 无法充分重叠。LRU 按 recency 驱逐，不感知 routing 模式，导致缓存命中率不稳定。SE-MoE 的 ring-buffer 在 Switch-128 等大 expert 数场景下加载大量 inactive experts 造成带宽浪费。Expert kernel 在少量 token 下 near-constant cost（roofline 模型的 memory-bound 区域），token 稀疏分布导致低计算效率。
  - **硬件架构层**：单 NVIDIA A40 GPU (48GB) + Intel Xeon Gold 6338 CPU。PCIe 带宽限制 CPU-GPU 传输。Mixtral-8×7B 在 All-in-GPU 下 OOM。

  **Baseline 的三个核心缺陷**：
  1. **Inefficient Expert Prediction**：回归式方法（gate score 近似）误差累积需要 fine-tuning 修复；启发式方法（token-expert 统计）无法捕获 input-dependent routing；学习式方法（ProMoE 逐层预测）限制调度灵活性。
  2. **Low Expert Utilization**：decoding 阶段 token 分布极度不均衡，部分 expert 仅收到单 token，expert kernel 在少量 token 下 near-constant cost，GPU 计算效率低。
  3. **Ineffective Expert Caching**：LRU 仅按 recency 驱逐，无视动态 routing pattern；SE-MoE ring-buffer 在大量 expert 场景下内存膨胀且重复加载 inactive experts。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文提出 ExpertFlow，通过三个协同组件解决 Baseline 三大缺陷**。

  **ExpertFlow 全栈执行例子（对比 Baseline）**：

  - **算法 Pipeline 层**：
    RPP（T5-style encoder-decoder）替代 baseline 的无预测/逐层预测/回归预测。在单次前向传播中输出所有 token 在所有层的 expert 激活概率矩阵 (B, S, L, E)。训练使用 binary cross-entropy 多标签分类：`L = (1/LE) * Σ[r*log(p) + (1-r)*log(1-p)]`。模型仅 7.21 MB，in-domain accuracy >90%，跨域下降 5-10%。

    **解决缺陷 1**：一次性全局预测（vs ProMoE 逐层预测）提前暴露完整 routing plan，支持早期 prefetch 和调度；T5 encoder-decoder 捕获输入语义（vs 启发式方法），accuracy 比 TLP/SLP baselines 高 60-80%。

  - **Serving 框架层**：
    (a) TS 用 K-means 聚类将两个 batch 的 token 按 routing path 相似度重新分组为两个新 batch：`min Σ Σ (R1 + R2)`，其中 `R_k = OR(r_i for i in T_k)`。在 CPU 上 <10ms 完成聚类。Dual-Batch Pipeline 将 RPP+TS 与 MoE 执行重叠，消除调度 overhead。
    (b) ECE/PLEC 基于预测自适应分配各层 cache slot（如 layer_1 需求 3 experts、layer_2 需求 2 experts、总 cache 容量 4 → 分配 3:1），预取最可能需要的 experts，early-layer expert 完成后释放 slot 供后续层复用。Real-time Correction 在 GPU compute 期间异步加载误预测遗漏的 expert，重叠 I/O 与 compute。

    **解决缺陷 2**：TS 将相似 routing 的 token 聚集到同一 batch，减少 active expert 数（per-batch），增加 per-expert token 量（从单 token 到多 token），使 expert kernel 从 memory-bound 移向 compute-bound 区域。Switch-128 上 throughput 提升 1.17×。

    **解决缺陷 3**：PLEC 替换 LRU/ring-buffer，预测驱动槽位分配 + 运行时复用 → cache hit ratio 91.90%（CS=16, BS=4），比 LRU 高 15-36%。Real-time Correction 的 async load 与 compute overlap 消除 cache miss 延迟。

  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Dual-Batch Pipeline 将 RPP+TS 的 CPU 计算与 GPU 的 MoE 执行交叠。ECE 的异步 CPU→GPU expert 加载与当前 running expert 的 GPU compute 并行。论文未明确说明是否使用 CUDA stream 或自定义 kernel。
  - **硬件架构层**：单 NVIDIA A40 (48GB)。与 baseline 相同硬件，但通过预测驱动缓存和 token 重排将 GPU 显存从 15.26GB (Switch-128 AIG) 降至 1.03GB（最大 93.72% 降低），Mixtral-8×7B 从 OOM 降至 15.99GB 完成推理。

  **三个缺陷的对应解决**：
  | Baseline 缺陷 | 论文解决方案 |
  |---|---|
  | Inefficient Expert Prediction（逐层预测/回归近似/启发式不准确） | RPP 一次性全局预测所有层所有 expert 激活，T5 encoder-decoder 捕获输入语义，>90% accuracy |
  | Low Expert Utilization（token 分散到不同 expert，per-expert 单 token，kernel 低效） | TS 按 routing 相似度 rebatch token，减少 active expert 数并增加 per-expert token load，1.17× 提升 |
  | Ineffective Expert Caching（LRU 无视 routing，ring-buffer 内存膨胀） | PLEC 预测驱动 adaptive slot 分配 + runtime slot 复用 + async correction，hit ratio 91.90%，比 LRU 高 61.15% |

  实验效果：(a) Switch-128 CS=4 达 9.99× throughput vs SE-MoE；(b) GPU memory 最高降低 93.72%；(c) Mixtral-8×7B AIG OOM → ExpertFlow 15.99GB；(d) Qwen1.5 cross-domain 达 2.21× vs Cache-MoE；(e) RPP 准确率 >95% on Qwen1.5，跨域仅降 5-10%。
