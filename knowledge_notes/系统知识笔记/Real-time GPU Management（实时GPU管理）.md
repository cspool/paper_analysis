## Real-time GPU Management（实时GPU管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Real-time GPU Management 是指为满足安全关键系统（如自动驾驶汽车）中 GPU-using task 的实时性要求而设计的一系列 GPU 资源调度与管理技术。核心理念：通过 bound GPU-using task 的 response time（最坏情况完成时间），确保 task 在 deadline 前完成。Bakita & Anderson 将现有方法分为三类：(i) mutual-exclusion-based management (GPUSync)；(ii) preemption-based management (Capodieci et al.)；(iii) management-free response-time analysis (Yang et al.)。所有三类方法均因对 NVIDIA GPU 硬件调度行为的不完整理解而产生 unsafe response-time bound。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

基于 Bakita & Anderson 论文总结的三类 GPU 管理方法和其硬件假设缺陷：

```
三类GPU管理方法的架构对比:

① Mutual-Exclusion Management (GPUSync [14], Elliott et al. [2]):
   架构: Task → acquire(engine_lock) → use GPU → release(engine_lock)
   硬件假设: copy engines 独立 → R6, R8推翻 (共runlist + PCE共享)
   安全条件: 每个独立PCE一个lock, 需nvdebug验证

② Preemptive EDF (Capodieci et al. [3]):
   架构: Task → CPU EDF scheduler → insert channels into runlist
   硬件假设: 仅一个runlist, 重置runlist=完全抢占 → R5, R7推翻 (多runlist独立)
   安全条件: 抢占所有相关runlist, 需nvdebug获取完整runlist拓扑

③ Management-Free Analysis (Yang et al. [4]):
   架构: 无管理中间件, GPU-using task直接使用GPU → response-time analysis
   硬件假设: kernel按FIFO dequeue → R2推翻 (超channel数时非FIFO)
   安全条件: 限制stream数 ≤ channel数 (默认x86_64上≤8)

共同必要条件 (Bakita & Anderson):
  所有GPU管理方法必须遵守8条调度规则(R1-R8),
  这些规则跨越channel → runlist → engine三个层次
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

正确实现 real-time GPU management 需要：(i) 使用 nvdebug 等工具在 target GPU 上验证硬件调度配置（engine-runlist 拓扑、PCE-LCE 映射、channel 数量）；(ii) 选择与硬件调度行为一致的管理方法（如在高 end GPU 上用 per-engine locking，在嵌入式单 runlist 平台上用 runlist 抢占）；(iii) 在 response-time analysis 中考虑 channel 限制（R2）、runlist timeslicing 间隔（R4）、跨 engine PCE 干扰（R8）等硬件行为。Bakita & Anderson 的工具套件（nvdebug + gpu-microbench）为这些验证提供了开源基础设施。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
