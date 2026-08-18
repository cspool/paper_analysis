## 激活重计算（Activation Recompute / Activation Checkpointing，激活重算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Activation Recompute（激活重计算，又名 gradient checkpointing/激活重物化）是训练期的显存优化技术：前向传播时丢弃中间激活（activations），反向传播需要梯度时重新计算这些激活，用额外计算换取显存。STAGE（ISCA'26）将其建模为 workload 生成选项（对应 Korthikanti 等 Reducing Activation Recomputation 与 Grattafiori 等 LLaMA 系列论文的机制），可对同一模型+并行策略生成有/无重计算两种 workload。vault 证据：paper_secs 本论文 A.-Impact-of-Parallelism-Strategies.md（Observation 4，score 47.6）与 DeepSeek-AI.md（HC 超连接论文中提及需要 gradient checkpointing，75.9）；知识库_编译框架.md 另有 "Activation Rematerialization（激活重物化）与 ILP 图调度（训练内存优化）" 条目（同义概念）。
- 从算法pipeline角度拆解：在训练 pipeline 中，前向每一层保留激活供反向使用，显存随层数线性增长；激活重计算把"保留"改为"丢弃+重算"——选择 checkpoint 点（典型每 N 层或按内存预算选层），checkpoint 点处保留激活，其余层激活丢弃；反向时从最近的 checkpoint 重放前向到目标层获得激活。STAGE 的案例（LLaMA-7B, batch=1, TP=8, w/ SP）：峰值显存从 7042.5 MB 降至 6107.0 MB（降低 13.3%），执行时间增加 20.3%（图 11）。显存降低可支撑更大的 DP 度，从而可能整体更快。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- STAGE 通过命令行开关 --activation_recompute 生成对应 workload；框架层实现（PyTorch）用 torch.utils.checkpoint.checkpoint 包裹模块，设置 checkpoint 策略（selective/full）；Megatron-LM/DeepSpeed 提供 activation checkpointing 与 recompute 层选择。STAGE 的建模意义：在部署前模拟"显存-时间"权衡曲线，帮助选择是否启用重计算以及是否因显存释放而采用更高 DP 度。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
