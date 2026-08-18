## 架构感知先验适配（Architecture-aware Prior Adaptation / 硬件感知 Prompt 自进化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QiMeng-Tensify（ISCA'26）的离线机制：把 LLM 的先验"适配"到目标硬件架构，使 LLM 给出硬件特定的调度指导。核心是两件事：(1) Prompt 自进化（Hardware-aware Prompt Self-evolving，Algorithm 3）——用 meta prompt 把"输入 TensorIR、调度策略、奖励、硬件性能指标"的完整搜索轨迹反馈给 LLM，迭代蒸馏出自然语言启发式规则，把 base prompt 进化为硬件感知的 learned principles（如图 6：injective block 优先 AutoInline、SSR-shaped 计算用 tensor-core-aware 多级 tiling、长 elementwise 链用 ComputeAtLocation 融合、长 reduction 用 CrossThreadReduction、常量标量用 InlineConstantScalar、寄存器逼近上限时抑制 vectorize/unroll 等）；(2) 离线成本模型训练——用同一批性能数据训练 XGBoost cost model 作为在线搜索的初始预测器。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 3（论文）：输入 base prompt P_b 与代表子图池 T；迭代 E 个 epoch，每 epoch 采样 M 个子图，用当前 prompt 并行跑 LLM-guided MCTS 收集完整轨迹（输入 IR、采用的策略、SM Efficiency/Shared Utilization/Achieved Occupancy/Instructions per Warp/Tensor Precision FU Utilization/Warp Execution Efficiency 等硬件指标，Table V），汇总反馈 F 后 UpdatePrompt 进化 prompt 并训练 cost model。实验：用 7 个代表子图（DoubleMatmul、Conv2d+Bias+Relu、Matmul、LSTM、Matmul+Relu、MLP、Softmax）收集 12,500 次测量，一次性离线开销约 30 小时。消融（Fig.7）：Static Prompt+Online-Only / Static+Offline / Evolved+Online-Only / Evolved+Offline（QiMeng-Tensify 全量），两者单独有效且协同最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LLM 根据"硬件反馈"（profiler 指标）总结性能提升根因 → 生成/更新自然语言规则注入 prompt；可移植性分析显示该先验不严重过拟合：L0 软件版本（CUDA 11.8→12.4）差距平均 0.7%、L1 同代 SKU（A100→A30）差距 1.3%、L2 跨代（A100→H100）差距 7.7%（H100 有更大 L2 与更强 TensorCore，native 适配仍建议）。使用方式：编译一次模型前先做一次性离线适配（~30h），之后在线 LLM-guided MCTS 获得 warm start；新硬件上重新适配或直接迁移（迁移可行但性能有差距）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
