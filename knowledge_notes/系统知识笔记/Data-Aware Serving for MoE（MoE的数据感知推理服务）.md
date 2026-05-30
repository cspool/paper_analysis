## Data-Aware Serving for MoE（MoE的数据感知推理服务）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Data-Aware Serving 是 SiDA-MoE (MLSys '24) 提出的一种 MoE 模型推理服务范式，核心思想是：推理系统的行为（如 expert 的加载/卸载决策、路由选择）应根据输入数据的特征预先确定，而非在推理过程中在线计算。这与传统 MoE serving（每次前向在线执行 router → expert selection）形成对比。

SiDA-MoE 的 data-aware 设计包含三个层面：(1) **预知能力**——通过 offline-trained hash 函数在推理前预测 expert 激活模式，使系统提前知晓 GPU 需要哪些 expert；(2) **适应能力**——系统行为随输入数据分布变化（不同 batch 激活不同 expert 集合），而非固定策略；(3) **结构保持**——hash 函数预测保留了每个样本的独特特征（通过 sparse attention 关注关键 token），避免粗粒度 offloading 策略破坏模型对特定输入的处理能力。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

SiDA-MoE Data-Aware Serving 的端到端流程（双线程并行）：

```
时间线（自上而下）：
═══════════════════════════════════════════════════════════════

Hash-building 线程 (CPU)          │  Inference 线程 (GPU)
                                  │
Batch X_0 到达                     │  等待 H_0...
  hash_func(X_0) → H_0            │  
  push H_0 → shared queue ────────→ pop H_0
                                  │  load experts(H_0, X_0)  # CPU→GPU
Batch X_1 到达                     │  forward(X_0, H_0)       # GPU compute
  hash_func(X_1) → H_1            │  offload inactive(H_0)    # GPU→CPU, FIFO
  push H_1 → shared queue ────────→ pop H_1
                                  │  load experts(H_1, X_1)
Batch X_2 到达                     │  forward(X_1, H_1)
  hash_func(X_2) → H_2            │  offload inactive(H_1)
  ...                             │  ...
═══════════════════════════════════════════════════════════════

Hash table H_i 结构（每个 batch）：
H_i[layer_idx][token_idx] = {
    expert_id: int,           # 预测激活的 expert
    scaling_factor: float     # 预测的 α 缩放因子
}
```

Data-Aware Serving 与非 data-aware 方法（Deepspeed/Tutel/Standard）的本质区别：

| 维度 | 非 Data-Aware (Deepspeed/Tutel) | Data-Aware (SiDA-MoE) |
|------|-------------------------------|----------------------|
| Expert选择时机 | 在线（inference时router计算） | 离线/半在线（hash线程预先预测） |
| GPU内存管理 | 全模型常驻GPU或粗粒度offload | 基于预测的per-expert精细offload |
| Expert选择开销 | 计入推理延迟（>72% MoE overhead） | 从推理路径中完全移除 |
| 对输入的适应性 | 固定调度策略 | 每个batch定制化expert集合 |
| 并行度 | 仅device间并行（通信优化） | CPU预测+GPU推理完全并行 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Data-Aware Serving 的实现涉及三个核心组件：

1. **Offline-trained Hash Function 训练**：使用预训练 MoE 模型的训练集收集 ground truth expert 激活模式（router 输出）→ 训练轻量预测器（LSTM+Sparse Attn+TKD）→ 在推理前加载训练好的 hash 函数权重到 CPU。

2. **双线程运行时系统**：Hash-building 线程（CPU，轻量，通常比 GPU 推理更快）和 Inference 线程（GPU，计算密集）。共享队列同步——hash table 是一个 FIFO 队列，hash-building 作为生产者，inference 作为消费者。由于 GPU 推理耗时 >> CPU hash 预测耗时（SiDA-MoE 论文观察），inference 线程几乎从不需要等待 hash 线程。

3. **FIFO Expert Eviction**：有限的 GPU 内存预算下，当所有 expert slot 被占用时，最早加载到 GPU 的 expert 被最先驱逐（FIFO）。论文讨论部分提到可升级为更复杂策略（如 LRU、priority-based）和层级 offloading（GPU → DRAM → SSD）。

SiDA-MoE 开源实现：https://github.com/timlee0212/SiDA-MoE，基于 HuggingFace Transformers 构建。

涉及论文标题：
- SiDA Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models
