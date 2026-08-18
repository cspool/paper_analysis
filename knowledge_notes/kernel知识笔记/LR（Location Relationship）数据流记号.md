## LR（Location Relationship）数据流记号

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LR（Location Relationship）记号是 BusyBarn 的形式化数据流表示，用数据位置与生产者-消费者关系描述 LLM 的算子级并行：对给定数据组，标注所有数据切片在片上的位置、生产者与消费者，覆盖自注意力的张量/序列并行（TP/SP）与相邻层间的流水并行（PP）。它把"数据标注"与"模型架构和互连拓扑"解耦，从而可系统地生成统一与非统一的并行模式、跨模型与跨硬件平台可扩展兼容。作用：把 LLM 推理负载转成可调度的事件序列——用 LR 刻画每层算子的输入/输出数据切片→按目标并行度把数据切成细粒度切片（依赖执行函数的基本单元）→追踪数据依赖得到函数并行计算 DAG→由 DAG 系统生成对应通信事件，作为层次化映射与通信调度的输入。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FFN 层例子（Fig.3）：FFN = LN + 两个线性层（PyTorch 中为 Conv1d）+ 下一 block 的 LN。
```
Act0 = LN(x)                      # 两个设备都需要的输入
Comm 0-0, Comm 0-1                # LN 输出广播到两个设备（TP 分片）
Act1-1 = Conv1d_a(Act0_slice1)    # 设备 1 的部分和
Act1-2 = Conv1d_b(Act0_slice2)    # 设备 2 的部分和
Reduce(Act1-1, Act1-2) -> Act1    # TP reduce 得到完整输出
Comm + reduce -> LN1 的输入切片    # 下一 LN 的数据布局决定归约结果切分
```
运转流程：LR 记号明确每个算子输入/输出数据切片与其位置/生产者/消费者 → 追踪依赖生成函数并行计算 DAG → 依 DAG 生成通信事件 → BusyBarn 的 Event Synthesizer 把记号转成已映射、已调度的事件集合（Notation Building → Hierarchical Mapper → Communication Scheduler 迭代优化，见"BusyBarn Overview"）。对比：它类似数据流 IR/调度描述（如 Alpa 的 tensor-splitting 框架），但显式解耦数据标签与硬件拓扑，兼容多样模型与平台。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BusyBarn 框架内的 Notation Building 阶段（输入模型参数 JSON 与硬件配置 JSON），产出计算事件与数据依赖，供 SA 映射与 BALD 调度消费。使用：指定混合并行度（SP/CP/TP/PP 组合）后自动生成调度事件集；端到端经事件驱动后端评估。信息缺口：论文未给出 LR 记号的语法/文件格式规范（artifact 代码中体现）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
