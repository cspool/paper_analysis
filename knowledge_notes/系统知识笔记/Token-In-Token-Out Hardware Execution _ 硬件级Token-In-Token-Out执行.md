## Token-In-Token-Out Hardware Execution / 硬件级Token-In-Token-Out执行

术语是什么？
Token-In-Token-Out是HNLPU提出的纯硬件LLM推理执行范式——硬件系统直接接收token ID作为输入，经过内部推理计算，直接输出token ID，全程无软件栈（无OS、runtime、library、compiler、framework）。与GPU方案（需CUDA/Triton编译kernel、PyTorch/TensorRT-LLM框架、OS调度等）形成对比。该设计提供两个收益：(1) 消除软件开发和维护成本；(2) 消除软件造成的性能抖动（deterministic execution behavior）。论文中的HNLPU在硬件中直接实现了Continuous Batching、6级pipeline、inter-chip CXL 3.0通信，所有推理逻辑由Control Unit以硬件状态机管理。

从系统架构角度拆解：
Token-In-Token-Out在HNLPU中的请求处理流程：
```
1. Host发送token ID → HNLPU Control Unit接收
2. Control Unit查HBM embedding table → token embedding (1, 2880)
3. embedding进入36层transformer pipeline:
   - 每层: HN Array(QKV/FFN weight) → Interconnect Engine(all-reduce) → VEX(attention/nonlinear)
   - 6级intra-layer pipeline × 36层 = 216最大并发batch
   - Continuous Batching: 新请求在slot空闲时动态插入
4. 最终output vector → Unembedding HN Array → logits
5. VEX multinomial sampling → output token ID
6. Host接收token ID
```
无软件栈意味着：(1) 无可编程性——模型功能100%由硬件决定；(2) 无运行时决策——所有调度（batching、pipeline、通信）由硬件FSM固定策略执行；(3) 无性能抖动——每个操作的cycle数完全确定。Continuous Batching通过Control Unit以硬件实现：prefill阶段token间无依赖→所有token并行流经pipeline；decode阶段auto-regressive→每sequence一个token但不同sequence间并行。

术语一般如何实现？如何使用？
HNLPU将整个推理流程（从token ID到token ID）以RTL（Verilog）在5nm工艺实现，经Synopsys EDA工具综合、布局布线和签核。实际部署中，HNLPU作为PCIe/CXL设备接入服务器，host通过标准接口发送prompt token IDs并接收generated token IDs。该范式的trade-off：极致效率（5,555× H100吞吐，1,047×能效）换灵活性的完全牺牲（模型不可软件更新）。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates
