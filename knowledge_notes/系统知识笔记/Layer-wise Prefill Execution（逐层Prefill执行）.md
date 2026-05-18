## Layer-wise Prefill Execution（逐层Prefill执行）

术语是什么？

Layer-wise Prefill Execution是将一个Prefill阶段按**Transformer Layer**拆分为多个可独立调度的执行单元。每个执行单元对应一个Transformer层的完整计算（attention + FFN）。MuxWise称其为Prefill Layer (PL)，Bullet称其为layer-wise prefill scheduling。两个系统都利用这一粒度进行与decode的并发调度，但实现有所不同。

从系统架构角度拆解术语：

Layer-wise Prefill在MuxWise调度流程中的作用：

1. **拆分Prefill**：新请求到达后，multiplex engine将其Prefill计算按transformer layer数（如Llama-70B的80层）拆分为NT个PL。每个PL包含该层的QKV投影、attention和FFN计算。
2. **发射数量计算**：dispatcher调用estimator，用公式 `NPL = ⌈(Td × NT) / TP⌉` 计算本轮需发射的PL数量——其中Td为decode迭代估计延时，TP为整个prefill估计延时，NT为模型总层数。目标是让NPL个PL的执行时间大致覆盖一个decode迭代窗口。
3. **异步发射**：decode iteration优先发射（CUDA graph launch <0.5ms），然后将NPL个PL发射到prefill stream。两个stream在不同GreenContext绑定的SM分区上并行执行。
4. **抢占支持**：因为PL是独立单元，当短请求的TTFT SLO面临风险时，可暂停当前长请求的PL发射，插入短请求的PL（非递归抢占，即短请求被抢占后不允许再被其他请求抢占）。
5. **Bubble消除**：decode迭代完成时若prefill stream上还有PL在执行，后续PL可在新建GreenContext中继续执行，避免等待decode完成产生bubble。

术语一般如何实现？如何使用？

MuxWise实现依赖CUDA Graph：decode侧用单一CUDA graph，prefill侧用piecewise CUDA graph（按layer切分）。Piecewise graph每次launch约10ms开销（Llama-70B 8×A100），但prefill kernel执行时间远长于launch开销。对SGLang的具体修改在multiplex engine模块中，通过GreenContext的stream同步支撑快速SM重配。

Bullet在SGLang v0.4.6上的实现：prefill engine以若干layer为粒度运行，每若干层后同步回CPU等待scheduler决策；decode engine使用CUDA Graph发射一个完整的decode step。两个engine作为独立SGLang worker进程运行，共享GPU memory pool（CUDA IPC）和CPU metadata buffer。Bullet的layer-wise执行粒度对齐prefill layer group而非单个layer，通过调整group大小在调度频率和kernel efficiency之间trade-off。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

Laser在layer-wise prefill执行基础上进一步提出完整的**layer-level chunked prefill**机制：(1) prefill Scheduler在每层后评估新请求的TTFT slack (SLO deadline − estimated remaining prefill time)，决定是否在当前layer边界抢占、合并或继续执行；(2) Executor维护GPU intermediate cache保存被抢占请求的hidden states，支持细粒度恢复；(3) 与MuxWise/Bullet的空间分区并发不同，Laser的layer-wise prefill专注于**时间域调度**——在同一prefill instance内通过layer边界的抢占和合并优化multi-SLO compliance，不涉及SM分区。Laser的prefill intermediate cache配置为16384 tokens (<256MB for Llama-70B)，switching overhead <1.5%。

---
