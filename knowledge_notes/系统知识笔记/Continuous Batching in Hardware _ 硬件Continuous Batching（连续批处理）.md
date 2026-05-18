## Continuous Batching in Hardware / 硬件Continuous Batching（连续批处理）

术语是什么？
Continuous Batching in Hardware是将软件级Continuous Batching（如Orca论文提出的iteration-level调度）直接实现在硬件中的技术。HNLPU的Control Unit管理硬件batching调度：prefill阶段token间无依赖可大规模并行（所有token流经pipeline stage-by-stage），decode阶段由于auto-regressive特性每sequence一次产生一个token但不同sequence间独立。通过inter-layer pipelining（36层层间pipeline）和intra-layer pipelining（6级层内pipeline），HNLPU最多同时处理216个请求（6×36=216），动态调度新sequence填充释放的pipeline slots。

从系统架构角度拆解：
HNLPU的Continuous Batching硬件调度流程：
```
Pipeline: Stage 1(QKV) → Stage 2(Allreduce) → Stage 3(Attention) → Stage 4(Xo) → Stage 5(Router+Up/Gate) → Stage 6(SwiGLU+Down) → Next Layer Stage 1

Prefill batch (无序列间依赖):
- Token-0到Token-N可同时在pipeline中流动
- 每stage处理不同token的计算（时间轴并行）
- 最大可同时处理216个prefill token

Decode batch (序列间独立，序列内auto-regressive):
- 每sequence每次仅一个decode token
- 不同sequence的decode token可在同一pipeline stage共存
- 最大216个并发sequence

Continuous Batching 动态调度:
- 当某个sequence完成（生成EOS token）→ pipeline slot释放
- Control Unit检测空闲slot → 从waiting queue分配新sequence
- 新sequence的prefill tokens插入pipeline（与已有decode tokens混合）
```
关键设计：(1) 混合prefill/decode token在同一pipeline中流动——Control Unit跟踪每个pipeline stage中token的类型和状态；(2) 权重固化使每个layer有独立计算资源（HN Array per layer），layer间可全流水；(3) 216最大batch受pipeline depth限制而非内存带宽（因为无weight fetching）。

术语一般如何实现？如何使用？
在HNLPU中，Control Unit是纯硬件实现（Verilog RTL），无需软件调度器。与GPU软件Continuous Batching（vLLM Scheduler在CPU上运行）相比：硬件方案零调度延迟（cycle-deterministic）、无CPU-GPU通信开销、但策略固定不可软件修改。该方案是HNLPU"零软件栈"理念的一部分——将传统Serving框架的所有功能（调度、batching、内存管理）都硬化到芯片中。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates
