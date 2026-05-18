## Cooperative Compilation and Scheduling（协同编译与调度）

术语是什么？通过联网搜索让回答具体和精准。
Cooperative Compilation and Scheduling是Infera提出的端到端DNN模型服务架构范式：将传统系统中独立运行的编译和调度从"动态耦合"改为"静态提供调度空间、运行时动态选择"。编译期离线生成多版本tile级micro-kernel（提供不同ILP/TLP/intensity trade-off），推理期在线根据任务优先级、GPU占用状态、kernel hazard和data dependency动态选择、融合和发射kernel。该范式将Halide的algorithm/schedule解耦思想提升到推理时调度层面，而非编译时固定schedule。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Infera系统架构中cooperative compilation and scheduling的运转流程：
1. **离线编译阶段**：用户调用Infera compiler→输入ONNX model→TVM Relay computation graph→tile-based partition（大operator切为micro operators，小operator合并为shepherd operator）→为每个micro operator生成多种ILP/TLP/intensity配置的kernel candidates→cut-and-patch instruction scheduling优化SASS级ILP→打包为CUDA binary static library→注册到inference server model pool。
2. **在线推理阶段**：Job Dispatch Unit (JDU)将inference job按GPU可用显存和estimated remaining time分发→Task Schedule Unit (TSU)按priority（deadline EDF/real-time FIFO/normal GCFS）生成Virtual Task with Budget (VTB)→Task Execution Unit (TEU)三阶段执行：SelectKernels（DAG中选zero in-degree且最大化asynchronous wavefront的data blocks→在线回归模型选最优kernel版本）→FuseKernels（CUDA binary level warp-level horizontal fusion）→LaunchKernel（HKQ→GDRCopy→DKQ→daemon kernel CDP launch）。
3. **协同点**：编译器提供的multi-version kernels + kernel metadata (#inst, resource usage) 使调度器可根据运行时GPU并发状态选择最优kernel配置；调度器的VTB instruction budget使编译器可为不同优先级任务生成合适粒度的kernel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera的实现：compiler基于TVM 0.16.0（约17k LoC C++ kernel-space module），inference server从零开发。编译时完全并行化（zero-tuning，无需GPU profiling）；推理时所有latency-sensitive操作（kernel fusion、launch）使用real-time scheduling或isolated CPU core。daemon kernel独占一个SM用于device-side kernel管理和launch。GDRCopy实现低延迟host-device kernel传输。与operator-based compilation+scheduling相比，cooperative approach在multi-model mixed serving中speedup至少1.6×（最高3.5×）。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
