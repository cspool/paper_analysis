## Prediction-Verification Decoupled Asynchronous Pipeline（预测-验证解耦异步流水线）

术语是什么？通过联网搜索让回答具体和精准。
Prediction-Verification Decoupled Asynchronous Pipeline是SADDLE提出的系统架构设计，将speculative decoding中传统串行执行的DLM prediction和TLM verification完全解耦并行执行。与传统系统（DLM完成所有请求drafting→TLM验证→DLM等待→下一轮）不同，SADDLE的异步pipeline通过Shared Pool和Eager Pool实现：(1) DLM不必等待TLM验证即可继续生成新draft tokens；(2) TLM不必等待所有请求完成DLM prediction即可开始在Shared Pool累积的tokens上验证；(3) 两阶段以乐观假设（当前正在验证的tokens将被接受）协调，Eager Pool tokens在验证通过时无缝融入下一轮verification。SADDLE实现1.73×端到端延迟降低，prediction和verification latency分别降低1.18×和1.23×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。
异步pipeline的timeline对比：
```
// 传统串行Pipeline (如PIM-SD):
Time →  |DLM_prediction|wait|TLM_verification|wait|DLM_prediction|...
         ↑ DLM→TLM同步barrier → pipeline bubble

// SADDLE异步Pipeline:
Time →  |DLM_μB0|DLM_μB1|TLM_Verify_Pool0|DLM_μB0|DLM_μB1|TLM_Verify|...
Shared: [tokens accumulate]|== verify ==|[tokens accumulate]|== verify ==|
Eager:                     |== DLM continues ==|→migrate/discard|
```
SADDLE Manager维护两个独立execution context：Draft Generator array (per micro-batch) 和TLM Verification Engine (Shared Pool触发)。调度决策基于简单的threshold检查（Shared Pool fill level ≥ C, GPU idle flag），决策延迟仅占0.83% latency。

术语一般如何实现？如何使用？
SADDLE Manager以专用硬件实现异步pipeline控制：Draft Generator array独立执行DLM prediction；Shared Pool状态机监控fill level并触发verification；Eager Pool状态机处理acceptance results后的migration/discard；Scheduler接收triggers协调operator mapping。与DFVG的FPGA-GPU异步pipeline（interrupt-driven, PCIe通信）不同，SADDLE的异步pipeline基于PIM+GPU的Shared/Eager Pool机制，通过Manager的on-chip pools和hardware triggers协调，无需host CPU介入。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
