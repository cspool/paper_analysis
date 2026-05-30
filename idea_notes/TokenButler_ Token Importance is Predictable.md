## TokenButler: Token Importance is Predictable

- baseline方法是什么？
  Baseline 方法分为三类：(1) 静态策略：StreamingLLM 用 recency-based sliding window + attention sinks 固定 KV-cache budget，完全缺乏 query-awareness；(2) 自适应驱逐：H2O 累积注意力分数永久驱逐低分 token，SnapKV 在固定窗口上池化注意力分数驱逐 token，一旦 token 被驱逐就永久丢失，在 co-referential 场景（如对话中早先提到的实体被后续引用）中失败；(3) 自适应动态策略：Quest 保留完整 KV-cache 但以 page 粒度（chunk）选择性加载 token，用 query 与 page 内 min-max token 幅值的点积作为 page 重要性代理。page 粒度过粗，当关键 token 跨越 page 边界时会丢失部分信息，在 context-dense 任务中精度不足。TokenSelect 保留全量 token 并用 Q·K 点积选择重要 token，但需在完整嵌入维度 E 上计算，开销高。
  
  全栈执行例子（以 H2O / SnapKV 为代表的自适应驱逐 Baseline）：
  - 算法层：在每个 decode step，对每个 head 计算 query 与 KV-cache 中所有 key 的注意力分数，累积近期窗口内的分数作为 "长期重要性" 指标，对低分 token 永久驱逐出 KV-cache。使用 GQA (Grouped Query Attention) 时由 KV head 共享。
  - 系统/Serving 层：论文未明确说明具体 Serving 框架修改。驱逐逻辑通常在模型 forward 内部实现，拦截 attention 计算后的注意力权重，排序并裁剪 KV-cache。
  - 编译框架层：论文未明确说明。
  - Kernel 调度层：驱逐后的 KV-cache 变为非连续存储（碎片化），需要 gather/scatter 操作或重新打包。H2O/SnapKV 使用标准 FlashAttention kernel，sink token + 保留 token 可能不连续。
  - 硬件架构层：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TokenButler 训练一个轻量级预测器（二层 MLP，<1% LLM 参数量）通过 attention distillation 学习预测 token 重要性。核心设计：(1) 在固定深度间隔 G 的 producer layer 处从 hidden states 预测低维 importance queries（d'=16），与学习投影后的真实 KV-cache keys（同样 d' 维）做点积得到细粒度 token 重要性分数，替代 H2O/SnapKV 的启发式注意力累积；(2) 不驱逐任何 token，保留完整 KV-cache，而是每步动态选择 top-k token——解决 H2O/SnapKV 永久驱逐导致 co-referential retrieval 失败的缺陷；(3) 逐 token 粒度选择（非 page 粒度）——解决 Quest 跨 page 丢失 token 的缺陷；(4) prediction interval + neighbor fetching 摊销预测器开销——每 N 步运行一次预测器并通过空间邻居覆盖重要性漂移，解决 TokenSelect 每步计算高维 Q·K 点积的高开销问题。

  全栈执行例子（TokenButler 方法）：
  - 算法层：Producer layer（每 G=4 层一个）的 hidden states H ∈ R^{B×L×E} 经 LayerNorm + 二层 MLP（hidden=512）预测 slot-specific importance queries Q_imp ∈ R^{(B·H)×G×L×d'}（d'=16）。对每个 consumer layer l，其真实 key cache K ∈ R^{B×H_kv×L×D} 通过学习投影矩阵 W_K^{(l)} ∈ R^{D×d'} 降维为 K_proj ∈ R^{B×H_kv×L×d'}。计算 scores = Q_imp[slot] · K_proj^T，取 top-B token 组成 Important Buffer。最终 attention 输入 = [Sink(128) | Important(B) | Local_Window(256)]，调用标准 FlashAttention。训练目标：minimize CE(softmax(A_teacher_masked), softmax(A_pred_masked))，teacher 为冻结 LLM 每层的 masked causal attention logits。训练数据 1K seq len，预测器通过 key-cache 投影自动泛化到 64K。
  - 系统/Serving 层：KV-cache 组织为三个连续 buffer（Sink / Important / Local Window），避免碎片化。延迟投影：新 token 在 Local Window 中停留 N 步后才批量投影 key（利用 HBM 带宽）。Prediction interval i=N：预测器每 N 步运行一次，中间 N-1 步复用上次选择。Neighbor fetching 基于聚类感知算法扩展选中 token 的空间邻居，2B 个唯一位置。集成 TokenSelect 代码库进行端到端 throughput 评测。
  - 编译框架层：论文未明确说明。
  - Kernel 调度层：Attention kernel 为标准 FlashAttention，输入为三 buffer 拼接。Importance Score Computation（低维 Q·K 点积，d'=16）随 context 增长但远小于原始 attention（D=128）。Timing breakdown 显示 Attention Kernel 耗时恒定（因 sparse budget 固定），Importance Score Computation 随 context 线性增长但斜率低（低维运算）。KV gather 耗时与 budget B 成正比。CPU offloading 场景（>=256K context）：仅传输 sparse 选中的 KV pairs 从 CPU 到 GPU，减少数据传输量 8×，latency 从 Dense 的 3.2s/token 降至 0.6s/token (7.6×)。
  - 硬件架构层：论文未明确说明。

  Baseline 缺陷 → TokenButler 解决方案对照：
  1. **静态策略无 query-awareness** → 学习预测器从 hidden states 动态预测每个 query 的 token 重要性
  2. **驱逐策略永久丢失 token** → 保留完整 KV-cache，仅选择性访问，co-referential 场景 near-oracle 精度
  3. **Page 粒度过粗（跨 page token 丢失）** → 逐 token 细粒度选择，synthetic co-reference coverage 84-95% vs Quest 19-58%
  4. **启发式重要性指标不准确** → attention distillation 直接学习真实 attention distribution，Recall@50% 达 67-81%
  5. **每步预测开销高** → prediction interval (up to 16× amortization) + neighbor fetching，精度仅降 1.1%
