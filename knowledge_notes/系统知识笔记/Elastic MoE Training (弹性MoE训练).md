## Elastic MoE Training (弹性MoE训练)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Elastic MoE Training 是 MoESys 针对 multi-task MoE 训练中负载不均衡问题提出的动态节点扩展/收缩策略。在 multi-task MoE 训练（如 UFO 模型含多个不同 batch size 的 task）中，不同 task 的输入数据量差异导致各 GPU 的计算时间不均——重 task 节点处理时间长，轻 task 节点完成后空闲等待（称为"木桶效应"或 "bubble"），整体 FLOPS 利用率低。Elastic MoE Training 根据各 task 的 workload 估算，动态调整训练节点分配：轻量 task 合并到更少节点（combine nodes），重量 task 增加节点并通过 data parallelism 分割数据（add nodes + partition）。策略分为 upscaling（增加设备提升整体 throughput）和 downscaling（减少设备控制成本），两者均有效减少 bubble 时间、提升 FLOPS 利用率。

从系统架构角度拆解术语：
Elastic MoE Training 的决策流程：
1. **Workload 评估**：Gate 网络 AlltoAll 收集各 task 的 expert 选择结果 → 估算各 task 的计算量。
2. **节点重分配**：轻量 task（如 batch=128 的 task 3 和 task 4）合并到 2 GPU → 2 task per GPU。重量 task（如 batch=512 的 task 1）拆分到 4 GPU → data parallelism partition → 每 GPU batch=128。
3. **训练执行**：各 GPU 按新分配的 task 和 data shard 执行训练，forward/backward 后通过 AlltoAll 同步 sparse 参数梯度、AllReduce 同步 dense 参数梯度。
4. **成本感知**：upscale 优先吞吐，downscale 优先成本，根据实际需求二选一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 UFO 模型（4 tasks, batch sizes 512/256/128/128）的实验中：load imbalanced 配置（4 GPU, 1/1/1/1 per task）per-GPU throughput = 62.6 samples/s；load balanced (8 GPU, 4/2/1/1 per task) per-GPU throughput = 74.0 samples/s（+18.2%）。
- 在 VIMER-UFO 2.0 (32× A100) 上对比 PyTorch v1.10：throughput 从 425 images/s 提升至 697 images/s (+64%)，memory 从 55GB 降至 45GB per GPU (-18%)。
- 该策略的关键挑战是避免因节点数变化导致的参数同步开销大于 bubble 消除收益——论文通过资源感知的成本模型判断 upscale/downscale 的利弊。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
