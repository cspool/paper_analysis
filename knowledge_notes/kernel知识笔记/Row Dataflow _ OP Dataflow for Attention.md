## Row Dataflow / OP Dataflow for Attention

术语是什么？
Row Dataflow 和 OP (Output-stationary) Dataflow 是 VAR-Turbo 论文中 Unified Attention Core 支持的两种可重构 attention 执行数据流。Row Dataflow 将 PE Cluster 按 attention head 分配，每个 Cluster 独立处理一个 head 的完整 K×V 矩阵乘加，适合大矩阵的 Big Attention（全局建模）。OP Dataflow 将 PE Cell/Node 切换为 output-stationary 模式，每个 PE 固定负责特定的输出位置，输入数据流经 PE array，适合小矩阵的 Small Attention（local window 内 token 聚合）。两种 dataflow 通过 Snooper+Fat Tree 动态切换。

从kernel调度角度拆解术语：
Row Dataflow 执行 Big Attention 的计算过程：
```
// PE Cluster 按 attention head 分配
for each head h:
  Q_h, K_h, V_h = get_qkv(h)  // 分配到 PE Cluster[h]
  Scores = Q_h × K_h^T        // PE Cell 间按 row 并行
  Attn = softmax(Scores)       // Row-wise normalization
  Out_h = Attn × V_h           // PE Cell 间按 row 并行
```

OP Dataflow 执行 Small Attention 的计算过程：
```
// PE Cell/Node 固定输出位置，输入数据流动
for each local window w:
  Q_w, K_w, V_w = get_qkv(w)  // 小矩阵（如 window_size=2, d=128）
  // K_w^T 广播到所有 PE Cell → Q_w 沿 PE array 流动
  Scores = Q_w × K_w^T         // output-stationary: 每个 PE 固定输出元素
  Coeff = mean(softmax(Scores), axis=0)  // 平均 attention coefficient
  Rep = Coeff × V_w            // 聚合为 representative token
```

Dataflow 切换机制：
- Snooper 读取当前层类型（Learning Region vs Inert Region）→配置 PE Cell 的 packet ID 映射
- Fat Tree 根据 packet ID 路由数据到对应 PE Cell→实现单 cycle 内 Row ↔ OP dataflow 切换
- Row Dataflow 优势：高吞吐（大数据矩阵），适合 Big Attention
- OP Dataflow 优势：低延迟（小数据矩阵）、减少中间数据移动，适合 Small Attention

术语一般如何实现？如何使用？
作为 VAR-Turbo accelerator 的 Unified Attention Core 的一部分以 SystemVerilog RTL 实现，TSMC 28nm 综合。Row+OP MAC 通过 Divide-and-Conquer 技术将大矩阵乘法分解为子块，Fluid Zone Detection 动态调整 FP 累加精度边界，shared FP accumulator 降低面积和功耗。Row/OP dataflow 的设计思路也可应用于其他需同时处理异构 attention pattern 的 accelerator。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

