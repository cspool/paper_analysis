## GPU Task Preemption for Inference

术语是什么？通过联网搜索让回答具体和精准。
GPU Task Preemption for Inference是Infera推理系统中实现的任务级抢占机制：当高优先级任务（deadline或real-time）到达时，系统发送preemption signal中断当前低优先级任务的执行，保存GPU kernel队列上下文（HKQ、DKQ、shared memory queue），切换到高优先级任务执行；待高优先级任务完成后恢复低优先级任务上下文继续执行。Infera实现两种preemption模式：Infera-P（完整保存暂停~10μs）和Infera-R（仅保存不暂停~5μs）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Infera GPU task preemption的运转流程：
1. **触发**：高优先级任务（如ddl任务）到达→TSU检测优先级高于当前运行任务→发送preemption signal给Task Executor
2. **Host-side响应**：kernel selector停止选择新kernel→kernel fuser继续将已选定kernel入队HKQ→host launcher (1)暂停HKQ→DKQ传输 (2)保存HKQ中所有kernel（标记为off-device）
3. **Device-side响应**：daemon kernel (1)保存DKQ中所有kernel (2)保存shared memory queue中所有kernel
4. **In-flight kernel处理**：正在执行的fused kernel通过preemption flag检测信号→尽快终止执行。由于DNN kernel执行被设计为idempotent（幂等），即使少数kernel未能及时停止也不会破坏系统状态
5. **Context切换**：TSU和TEU同步→切换到高优先级runqueue→新任务context加载→开始执行
6. **恢复**：高优先级任务完成后→低优先级任务context恢复→HKQ/DKQ/shared memory kernel恢复→继续执行

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera的GPU preemption利用其自研scheduler和kernel管理基础设施实现。与CUDA stream层面的kernel preemption不同（GPU hardware scheduler不支持细粒度抢占），Infera在software scheduler层面实现preemption：通过cooperative compilation and scheduling的tile-level granularity，在scheduling cycle边界和fused kernel的flag检查点实现安全抢占。Infera-P比REEF-N快约2.5×，比EffiSha快超一个数量级。Preemption不依赖GPU hardware context switching，而是依赖kernel idempotence保证正确性。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
