## HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HD-MoE 在 kernel 调度/运行时计算层面的实现是 **Dynamic Placement Strategy（在线动态专家调度）**，包含三个关键组件：
    1. **Priority Detection and Computation Prediction**：利用相邻层专家激活的时间局部性预测下一层计算热点。对每个节点 c 上的专家 i，计算优先级分数 `prio_ic = 2 * P_ic * f̂_i * IS / comp`，其中 f̂_i 是预测的激活频率。选择最高负载节点上优先级最高的专家作为预广播候选。
    2. **Optimal Broadcast Chunk Size**：基于 α-β 通信模型推导最优广播 chunk size `c = sqrt(α * h * IS / (2 * β * k * sqrt(D)))`，在给定 runtime window（由上一层推理延迟决定）内最大化预广播效率。
    3. **Communication-Efficient Dispatch**：预广播后，每个 token 被路由到持有其激活专家的任意节点中当前计算负载最低的节点，在不引入额外通信开销的前提下最小化负载不均衡。
  - 实验比较：静态部署策略 vs 动态调度策略在不同推理场景（math/coding/reasoning 等 MT Bench 问题类型）下的延迟和加速比。两种配置：(5 TFLOPS, 50 GB/s, batch=512, 预广播 2 experts/layer) 和 (2.5 TFLOPS, 75 GB/s, batch=512, 预广播 5 experts/layer)。广播 2 experts 时动态策略平均加速 1.15×，广播 5 experts 时平均加速 1.25×。

- 后端平台是什么，配置是什么。
  - 模拟的 3D NMP 加速器，具有可配置的计算吞吐和通信带宽：2.5 TFLOPS / 75 GB/s，5 TFLOPS / 50 GB/s，10 TFLOPS / 25 GB/s。
  - 2D mesh NoC 拓扑：4×4, 4×8, 8×8 节点网格。
  - 模型：Mixtral-8x7B-Instruct, DeepSeek-V2-Lite-Chat, Qwen2-57B-A14B-Instruct。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文自建 Python 离散事件模拟器，模拟 2D mesh NoC 中不规则 all-to-all 通信。实现 XY routing + 优先级队列事件调度 + 链路占用追踪。
  - 线性规划（LP）求解器用于 Node Balance 优化；Bayesian Optimization 用于 Link Balance 物理映射。
  - 验证工具：ASTRA-sim [27] 用于验证 ring all-reduce 延迟模型的准确性。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/angerybob/HD-MoE
  - 动态调度评估原理：
    1. **输入**：专家激活 trace（从 MT Bench 数据集不同问题类型采样），placement matrix P_ic（offline 确定），预测的激活频率 f̂_i。
    2. **Runtime 调度流程**：每层推理开始时 → Priority Detection 扫描各节点负载，识别计算热点 → 选择最高负载节点上优先级最高的 expert → 按最优 chunk size 将其预广播到所有节点 → 每个 token 根据其激活的 experts 从候选节点中选择当前负载最低的节点 dispatch → 节点执行本地 expert 计算 → 进入下一层。
    3. **性能测量**：模拟器记录每层的 computation latency（max across nodes）和 communication latency（discrete-event 调度时间线），计算 MoE Decomposed Latency = t_comp + t_comm。Normalized TBT = 当前策略 TBT / TP baseline TBT。
    4. **动态 vs 静态对比**：静态策略使用 reasoning 问题确定固定 placement；动态策略在运行时根据不同问题类型（math/coding/writing 等）自适应调整广播和调度决策。记录各场景下 per-MoE-layer latency 和 speedup。
