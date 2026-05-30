## FaaS-based Remote Kernel Evaluation

术语是什么？
FaaS (Function-as-a-Service)-based Remote Kernel Evaluation是KernelEvolve的kernel评估架构模式：将kernel generation（CPU-bound LLM推理）和kernel evaluation（accelerator-bound correctness验证+profiling）解耦到不同的硬件资源上执行。Generation在CPU host上完成（prompt synthesis、knowledge retrieval、LLM invocation），evaluation通过Meta's FaaS platform异步dispatch到remote accelerator pools（NVIDIA/AMD/MTIA），利用FaaS的Thrift auto-generated server interfaces和Tasklet resource model（扩展自CPU/RAM到GPU资源）。

从系统架构角度拆解术语：
KernelEvolve的tree search每个node expansion包含两个阶段：
1. Generation phase: prompt synthesis + knowledge base retrieval + LLM invocation——纯CPU-bound操作，不需要accelerator。
2. Evaluation phase: kernel compilation + TritonBench correctness验证 + Torch Profiler timeline capture + NCU/Proton/MPP/MTIA Insight profiling——需要target hardware accelerator。

Generation-evaluation workload asymmetry驱动了FaaS disaggregation的设计动机：单个host可运行数百个generation agents但仅拥有有限accelerator（8 GPUs或24 MTIA devices per host）。无FaaS分离时，agents串行通过本地hardware——每个agent占用device 8-12分钟（大部分时间idle等待generation），其他agents排队。FaaS evaluation提供：(1) resource decoupling——generation在本地CPU执行，evaluation dispatch到remote accelerator pools；(2) elastic capacity——evaluation分布在拥有数百个GPU/MTIA设备的FaaS worker pools上，而非串行通过本地硬件。

FaaS worker通过Conveyor continuous deployment自动接收pre-deployed Bento interpreter environments（bento_kernel_*），消除了per-kernel evaluation的环境配置和编译开销（从≥10分钟降至秒级）。

术语一般如何实现？如何使用？
论文使用Meta内部的FaaS platform（XFaaS），其Tasklet resource model从CPU/RAM扩展到GPU。KernelEvolve的evaluation code generator生成平台特定的evaluation scripts（TritonBench harness、Torch Profiler script、NCU/MPP instrumentation），FaaS worker加载pre-deployed interpreter环境（包含完整的Triton compiler、profiling frameworks、runtime libraries），执行evaluation harnesses，返回结构化结果（correctness、speedup、profiling metrics）供context memory sub-agent消费。这种架构实现了CPU (generation)和accelerator (evaluation)的独立scaling。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
