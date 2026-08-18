## Coupled Performance Predictor（耦合性能预测器，XGBoost cost model）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCC 用 XGBoost（learning_rate=0.1、max_depth=8、num_boost_round=5000）联合预测一个 tiling draft 的端到端执行时间（数据重排时间 + 计算时间耦合建模），代替对成千上万候选 draft 的逐一实机/仿真 profile，选出最优配置。选择学习型模型的原因：解析模型需要 PIM 设备特定公式与参数、跨后端难泛化；相比神经网络预测器（如 [72]），XGBoost 内存与计算开销轻量。训练：对每个 PIM 后端，采样候选 draft 的子集在目标后端 profile 执行时间作标签（draft 配置 + 后端信息作特征），训练一次即可覆盖该后端全部 kernel/张量尺寸；静态尺寸离线训好查表，动态尺寸（LLM token 变化）运行时在线生成 draft 并预测，记录进 lookup table，可选在线增量训练。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
工作流：schedule generator 产出（剪枝后）draft 集 → 15% 随机采样 profile 出标签 → 训练 XGBoost → 对剩余 85% draft 预测端到端时间 → 选预测最优者写 lookup table。精度（Table 4）：单 kernel 平均 89.28% 命中真最优（AttAcc+ATTN 因融合 GEMV+softmax 仅 74.63%）、LLM 97.37%；预测选错时平均仍有真最优 96.56% 的性能，LLM 端到端总性能损失仅 ~0.2%；预测器使编译时间平均加速 3.81×。在线场景：MT-NLG-310B 随机 token 尺寸 2000 批查询中，未见尺寸仅前 ~160 查询触发在线编译，之后收敛到离线水平。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：sklearn/XGBoost 2.1.4（artifact 依赖 scikit-learn 1.3.2）；特征为 draft 的 [组数、核数、representative mapping、tile 尺寸] 等配置与后端参数，标签为仿真器测得的执行时间。使用：DCC 离线训练一次（全部 workload 约 67s），多用户推理请求摊销；新张量尺寸出现时在线生成+预测。对比参照：TVM 的 cost model 只估计算时间（compute-centric，见 ATiM 条目），DCC 的耦合预测是联合 co-optimization 的最后闭环。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
