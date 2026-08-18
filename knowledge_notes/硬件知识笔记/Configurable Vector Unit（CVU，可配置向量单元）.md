## Configurable Vector Unit（CVU，可配置向量单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CVU 由模块化单功能 vector 算术算子组成，每个算子取一个/两个输入 vector 流、产出一个输出流；TPB 指令可配置 CVU 把输入路由经单个算子或构建带中间缓冲的多级流水。高效处理基础 vector 操作与常见 AI 任务（pooling、softmax、layer normalization）。复杂 vector 操作无法完全流水时，可分多阶段（每条 TPB 指令一个阶段）处理，吞吐略降但性能仍相当于或优于传统 vector core。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 例子（softmax 的 CVU 流水配置）：输入 x_i 流 → 算子 1（逐元素求 max，得到局部 max）→ 算子 2（exp(x - max)）→ 算子 3（累加 rowsum）→ 算子 4（除法 prob = exp(x-max)/rowsum），各算子经中间缓冲级联、数据流同步由 SU 协调。配置空间大，可适配 transformer 等模型中多样的 vector 计算模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模块化 vector 算术算子 + 可配置互连/中间缓冲 + 配置寄存器（TPB 指令配置）。使用：编译器把 vector 算子序列映射为单条/多条 TPB 指令配置；与 TCU（tensor 缩并）互补覆盖非矩阵类算子。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
