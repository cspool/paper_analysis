## AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  AccelOpt 使用 LLM agentic workflow（Planner-Executor-Summarizer）+ beam search + optimization memory，在 Trainium 加速器上自动生成和优化 NKI kernel。优化技术包括：loop invariant code motion（循环不变量外提）、loop fusion（循环融合）、tile size 增大（256→512 以利用硬件 optimal throughput 配置 128×128 stationary + 128×512 moving）、algebraic simplification（如 θ − γλθ → (1 − γλ)θ）、intrinsic fusion（如 reciprocal(sqrt) → rsqrt，x/(1+e^(-x)) → x·sigmoid(x)）、memory spilling 消除（通过 recomputation 减少 off-chip memory 访问）等。
  实验比较：(1) AccelOpt vs Claude Sonnet 4 重复采样（peak throughput percentage），(2) beam search vs 重复采样，(3) beam search + optimization memory vs beam search only，(4) 不同 executor 模型（Qwen3-Coder-30B、gpt-oss-120b、Qwen3-Coder-480B）及 model ensemble，(5) 不同 memory 配置（TopK, ExpN）的 cost-benefit trade-off，(6) Reflexion-style baseline，(7) AccelOpt vs human experts（Mamba 和 RoPE kernel），(8) AccelOpt 在 H100 GPU Triton kernel 上的泛化实验。

- 后端平台是什么，配置是什么。
  Amazon Trainium 1 (trn1.32xlarge) 和 Trainium 2 (trn2.48xlarge) EC2 实例。Trainium 芯片包含 Tensor Engine、Vector Engine、Scalar Engine（三者并发运行），通过 kernel-managed on-chip memory（SBUF 和 PSUM）与 HBM 通信。NKI（Neuron Kernel Interface）是 Python-embedded kernel 编程语言。GPU 泛化实验使用 NVIDIA H100 GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  自建 NKIBench benchmark suite（14 个 NKI kernel，来自 Qwen3、DeepSeek-V2.5/V3/MoE、Falcon-40B 等真实 LLM workload），分布式 profiling service 基于 Neuron Profile 工具。AccelOpt 系统实现了 beam search 算法 + optimization memory curation（Algorithm 1 & 2）。评估使用 Roofline 模型计算 peak throughput percentage: T = max(Traffic_Min/Bandwidth, FLOPs_MM/Peak_MM, FLOPs_Vec/Peak_Vec)，性能指标为 T/t（百分比）。GPU 泛化验证使用 FlashInfer-Bench 的 24 个 Triton kernel。Agent 请求通过 vLLM 服务 open-source 模型，使用 Logfire 记录 LLM query 信息。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址: https://github.com/zhang677/AccelOpt
  评估原理:
  1. NKIBench 提供 14 个 baseline NKI kernel 及其对应 ML operator 问题描述（Matmul、BatchMatmul+Softmax、Group Query Attention、Mamba block、LoRA、RoPE、SiLU、SwiGLU、AdamW 等），每个 kernel 关联 config（shape）、profiling 数据和 peak throughput 计算。
  2. AccelOpt 每轮迭代: Planner 为每个 candidate kernel 生成 N 个优化计划 → Executor 每个计划实现 K 次，生成 B×N×K 个新 kernel → 分布式 profiling service 在 Trainium 硬件上运行所有 kernel，收集 latency、HBM read/write bytes、tensor/vector/scalar engine utilization、spill bytes 等指标 → Summarizer 从超过 speedup 阈值的 slow-fast kernel pairs 中提炼 experience items → 更新 optimization memory → Beam search candidate selection function β 选择 Top-B kernels 进入下一轮。
  3. 最终评估以 peak throughput percentage（= 理论最优时延 T / 实测时延 t）衡量 kernel 质量，T 基于 Roofline 模型取 memory bandwidth bound、tensor engine bound、vector engine bound 三者的最大值。Traffic_Min 为所有输入+输出 tensor 的 byte 总量。
  4. 从 baseline kernel 输入到优化后 kernel 性能输出全过程: NKI kernel 源码 → Planner 分析 profile 瓶颈（如低 HFU、高 spill、高 memory write）→ Executor 实现 loop transformation / tiling / memory layout 优化 → Neuron Compiler 编译 → Trainium hardware 执行 → Neuron Profile 采集性能数据 → Summarizer 提炼通用优化策略 → optimization memory 积累经验 → 下一轮迭代。共运行 T=16 轮，最终输出最优 kernel 和 peak throughput percentage。
