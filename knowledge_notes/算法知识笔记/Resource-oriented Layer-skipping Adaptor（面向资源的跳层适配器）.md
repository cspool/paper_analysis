## Resource-oriented Layer-skipping Adaptor（面向资源的跳层适配器）

术语是什么？通过联网搜索让回答具体和精准。

Resource-oriented Layer-skipping Adaptor 是 LEGO 提出的算法-系统协同设计中的算法侧核心组件：在游戏-LLM 共置场景中，当 GPU 资源不足以运行完整 LLM 推理时，必须跳过若干 Transformer 层以满足资源预算。该 adaptor 是一个小型 FFN（Feed-Forward Network），用于近似被跳过层段的知识变换，将资源约束驱动的跳层造成的精度损失降到最低。其设计受知识蒸馏（Knowledge Distillation）启发，但采用自蒸馏（self-distillation）方式——adaptor 从同一模型中被跳过的层学习知识表示，而非从外部 teacher model 学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Adaptor 的算法 pipeline 分为离线准备和在线推理两个阶段：

**离线准备流程：**
```
Step 1: Profile 游戏 rendering headroom 范围
  - 采集代表性 gameplay 期间的 rendering task 执行时间
  - 计算最小/最大可用 headroom H_min, H_max
  - 根据 H_min/H_max 确定必须跳过的最小层数 M 和最多层数 N
  - 因此需要准备 N-M+1 种跳层配置

Step 2: 构建 Layer Similarity Heatmap
  - 用训练数据（如 WebInstruct）对 LLM 做前向推理
  - 提取每个 Transformer layer L_i 的输出 tensor T_i
  - 对所有层对 (L_i, L_j) 计算 cosine similarity: sim(T_i, T_j)
  - 构建 similarity heatmap 矩阵

Step 3: 选择跳过层段
  - 当需跳过 N 层时，沿 heatmap 对角线寻找连续层区间 [L_k, L_{k+n}]
  - 选择该区间内平均相似度最高的配置
  - 如 Llama3-8B 跳 4 层: 选 L25-L29（highest similarity）
  - 如 Llama3-8B 跳 8 层: 选 L23-L31

Step 4: 训练 Adaptor
  - Adaptor 是一个单层 FFN: FFN^{k+n}_k
  - 输入: 第 k 层输出 f_k
  - 目标: 逼近第 k+n 层原始输出 f_{k+n}
  - Loss: L_mse = ||f_{k+n} - FFN^{k+n}_k(f_k)||²
  - 仅更新 adaptor 权重（268.8 MB/adaptor）
  - 不同跳层配置可复用中间层输出，减少冗余计算
  - BlackMyth 最多需 14 个 adaptor，总训练时间约 36 小时
```

**在线推理流程：**
```
1. Scheduler 根据 headroom 预测决定跳过 N 层
2. 将对应层段 [L_k, L_{k+n}] 的 Transformer layers 替换为已训练的 adaptor
3. 推理时：输入 f_k → adaptor FFN → 直接映射到 f_{k+n}
4. 剩余层（L_1 到 L_{k-1}, L_{k+n+1} 到 L_last）正常执行
5. decode 阶段以 Transformer layer (~0.4ms) 为粒度调度跳层后推理
6. prefill 阶段以 self-attention (0.5ms) 和 FFN sublayer (1.0ms) 为粒度
```

关键设计决策：
- **连续跳层而非离散跳层**：heatmap 显示后层高相似度但最后层与倒数第二层低相似（最后层编码与 output layer 对接的关键知识，不应跳过）。连续跳层避免 disruption of inter-layer coherent representations
- **资源驱动而非 confidence 驱动**：传统 layer-skipping（LITE/CALM）按 token confidence 决定早退，无法为每个请求提供资源预算保证。LEGO 反转逻辑：资源预算决定跳层数，adaptor 补偿精度
- **跳层上限**：≤12 层时 LEGO 精度优于小模型 baseline（Llama3-3B），跳 13-14 层时精度显著下降但仍优于 LITE

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **Adaptor 架构**：标准 FFN layer（两层全连接 + 激活函数），输入输出维度与 Transformer hidden dimension 相同。以 Llama3-8B（hidden=4096）为例，单个 adaptor 参数量约 268.8 MB（FP16）
- **训练数据**：游戏公司使用自己的 fine-tuned LLM + 私有数据集训练 adaptor。论文使用 WebInstruct 作为 upstream training set
- **内存开销**：12 个 adaptor 合计约 3.23 GB；intermediate-result tensor 占 67.5 MB（推理本身需要，无额外开销）
- **兼容性**：与静态量化（如 INT4）、静态 sparsity 兼容；不与动态加速方法叠加（引入执行时间不确定性）
- **跨模型适用**：论文验证 adaptor 在 Llama3-8B、Mistral-7B、DeepSeek-V2-Lite（MoE）、Mixtral-8x7B（MoE）上均有效
- **部署方式**：游戏公司离线训练 adaptor → 与游戏 + LLM 打包 → 用户下载后本地部署；也可集成到云游戏平台如 NVIDIA GeForce NOW

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU

