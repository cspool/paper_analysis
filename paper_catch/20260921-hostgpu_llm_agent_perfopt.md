# 20260921 下载清单：host+消费级GPU 本地LLM推理 & Agent自动性能优化
## 用法：python3 scripts/paper_download.py --file paper_catch/20260921-hostgpu_llm_agent_perfopt.md --output papers_pdf/paper_catch_20260921 --dry-run
## 规则：`#` 行为说明（下载器跳过）；`- [标题](链接)` 行按链接下载；无链接的纯标题行由下载器按 arXiv/OpenAlex/DBLP 检索
## 已去重：本地 paper_secs / papers_md 已有的论文未列入（去重依据为 Obsidian 目录清单，2026-09-21）
## 来源：paper_catch 镜像（AmberLJC/LLMSys-PaperList、byungsoo-oh/ml-systems-papers）+ web 搜索
## 说明：云端 Serverless LLM serving（ServerlessLLM/Aegaeon/PipeBoost 等）不属于本清单主题，未列入

# ===== 主题一：host（CPU+DRAM/SSD）+ 单张桌面/消费级 GPU 的 LLM 推理 =====

## A. 权重/张量 offloading 与 CPU-GPU 混合执行（dense 为主）
- [FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865)
- [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456)
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514)
- [TwinPilots: A New Computing Paradigm for GPU-CPU Parallel LLM Inference](https://dl.acm.org/doi/pdf/10.1145/3688351.3689164)
- [NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference](https://arxiv.org/abs/2411.01142)
- [PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices](https://arxiv.org/abs/2504.03664)
- [SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices](https://arxiv.org/abs/2505.10259)
- [prima.cpp: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters](https://arxiv.org/abs/2504.08791)
- [Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures](https://arxiv.org/abs/2504.11750)
- [Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference](https://arxiv.org/abs/2505.06461)
- [Distributed LLM Serving on Consumer-Grade GPUs by Reconciling Computation and Communication](https://aclanthology.org/2025.findings-emnlp.957.pdf)
- [Memory Offloading for Large Language Model Inference with Latency SLO Guarantees](https://arxiv.org/abs/2502.08182)
- [DynamicInfer: Runtime-Aware Sparse Offloading for LLMs Inference on a Consumer-Grade GPU](https://openreview.net/forum?id=CvjmvjlczZ)
- [Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices](https://arxiv.org/abs/2607.10183)
- [DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference](https://arxiv.org/abs/2604.26074)
- [PipeMax: Enhancing Offline LLM Inference on Commodity GPU Servers](https://arxiv.org/abs/2605.02189)
- [SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference](https://arxiv.org/abs/2506.22033)
- [Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption](https://arxiv.org/abs/2605.19593)
- [Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference](https://arxiv.org/abs/2603.22774)
- [OpenJarvis: Personal AI, On Personal Devices via LLM-Guided Spec Search and Local-Cloud Collaboration](https://arxiv.org/abs/2605.17172)

## B. CPU-GPU 异构投机解码
- [SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices](https://arxiv.org/abs/2406.02532)
- [Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference](https://arxiv.org/abs/2412.18934)
- [Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism](https://arxiv.org/abs/2505.23219)

## C. KV cache 放 host DRAM（长上下文、单卡）
- [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/abs/2406.19707)
- [HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading](https://arxiv.org/abs/2502.12574)
- [Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU](https://arxiv.org/abs/2506.20187)
- [CLO: Efficient LLM Inference System with CPU-Light KVCache Offloading via Algorithm-System Co-Design](https://arxiv.org/abs/2511.14510)
- [No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs](https://www.usenix.org/conference/osdi26/presentation/luo)
- [ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- [HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval](https://arxiv.org/abs/2606.21633)
- [RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU](https://arxiv.org/abs/2504.15302)

## D. 本地 CPU+GPU 上的 MoE（本地库已有 17 篇，仅列缺的）
- [Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU-GPU Hybrid Design](https://arxiv.org/abs/2606.10493)
- [TriMoE: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP for High-Throughput MoE Inference via Offloading](https://arxiv.org/abs/2603.01058)
- [Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking](https://arxiv.org/abs/2603.12831)
- [Efficient LLM Serving on Commodity GPU Clusters: Data-Reduced Cross-Instance Orchestration for LLM Serving](https://www.usenix.org/conference/osdi26/presentation/du)

## E. 相关（偏硬件/表征，可选）
- [Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM](https://arxiv.org/abs/2409.15654)
- [A Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/abs/2512.01644)

# ===== 主题二：LLM/Agent 自动做性能优化 =====

## F. Agent 生成 / 调优 GPU kernel（按时间倒序）
- [SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System](https://arxiv.org/abs/2608.05033)
- [CommBench: Can LLMs Write Correct and Efficient GPU Communication Code?](https://arxiv.org/abs/2608.04450)
- [KernelBrain: Coarse-to-Fine, Budget-Aware Search for Agentic GPU Kernel Optimization](https://arxiv.org/abs/2608.02611)
- [Kernel Forge: An Agent Harness for LLM-based Generation and Optimization of CUDA Kernels](https://arxiv.org/abs/2607.24762)
- [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979)
- [KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?](https://arxiv.org/abs/2607.16241)
- [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541)
- [SOLAR: AI-Powered Speed-of-Light Performance Analysis](https://arxiv.org/abs/2606.26383)
- [SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation](https://arxiv.org/abs/2606.17518)
- [AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis](https://arxiv.org/abs/2606.09682)
- [Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](https://arxiv.org/abs/2605.26720)
- [Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU](https://arxiv.org/abs/2605.26118)
- [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215)
- [PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization](https://arxiv.org/abs/2605.15222)
- [VibeServe: Can AI Agents Build Bespoke LLM Serving Systems?](https://arxiv.org/abs/2605.06068)
- [KEET: Explaining Performance of GPU Kernels Using LLM Agents](https://arxiv.org/abs/2605.04467)
- [Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance](https://arxiv.org/abs/2603.29010)
- [Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization](https://arxiv.org/abs/2603.28342)
- [AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search](https://arxiv.org/abs/2603.21331)
- [Improving Coherence and Persistence in Agentic AI for System Optimization](https://arxiv.org/abs/2603.21321)
- [SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits](https://arxiv.org/abs/2603.19173)
- [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://arxiv.org/abs/2603.02637)
- [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286)
- [Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations](https://arxiv.org/abs/2602.05885)
- [Towards Automated Kernel Generation in the Era of LLMs](https://arxiv.org/abs/2601.15727)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
- CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning
- GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks
- Astra: A Multi-Agent System for GPU Kernel Performance Optimization
- Autocomp: LLM-Driven Code Optimization for Tensor Accelerators

## G. Agent 自动优化系统 / 编译 / 训练推理配置
- [InferenceBench: A Benchmark for Open-Ended LLM Inference Optimization by AI Agents](https://arxiv.org/abs/2607.20468)
- [AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning](https://arxiv.org/abs/2606.20373)
- [AutoLLMResearch: Training Research Agents for Automating LLM Experiment Configuration - Learning from Cheap, Optimizing Expensive](https://arxiv.org/abs/2605.11518)
- [GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon](https://arxiv.org/abs/2602.15241)
- [ASAP: an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training](https://arxiv.org/abs/2511.03844)
- [Learning to Shard: RL for Co-optimizing the Parallelism Degrees and Per-operator Sharding Dimensions in Distributed LLM Inference](https://arxiv.org/abs/2509.00217)
- [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)
- [Let the Barbarians In: How AI Can Accelerate Systems Performance Research](https://arxiv.org/abs/2512.14806)
