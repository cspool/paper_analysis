## NKIBench (NKI Kernel Optimization Benchmark)

术语是什么？
NKIBench 是由 AccelOpt 论文构建的 NKI kernel 优化 benchmark suite，是第一个针对 Amazon Trainium 加速器的 NKI kernel 优化基准。它包含 14 个从真实 LLM workload 中提取的代表性 NKI kernel（来自 Qwen3 0.6B/1.7B/32B、DeepSeek-V2.5/V3/MoE-16B、Falcon-40B 等模型），涵盖从单算子（Matmul、BatchMatmul）到多算子链（Matmul+Add+RMSNorm、BatchMatmul+Softmax）和更大 building block（Group Query Attention、Mamba block）的广泛范围，涉及 inference 和 training kernel。区别于传统 kernel benchmark（仅测量相对 speedup），NKIBench 的关键创新是为每个 kernel 提供基于 Roofline 模型的 Peak Throughput Percentage 指标（= 理论最优时延 T / 实测时延 t），提供绝对性能坐标系。

从kernel调度角度拆解术语：
NKIBench 的 14 个任务及其配置和性能瓶颈：

| Name                   | Source Workload   | Config                          | Latency (ms) | Bottleneck |
|------------------------|-------------------|---------------------------------|--------------|------------|
| AdamW                  | DeepSeek-MoE-16B  | M=10944, N=2048                 | 2.00         | Memory BW  |
| Add+RMSNorm+Matmul     | Qwen3 0.6B        | K=1024, M=4096, N=2048          | 1.22         | Tensor Eng |
| BatchMatmul            | Falcon-40B        | B=16, K=64, M=4096, N=4096      | 4.61         | Tensor Eng |
| BatchMatmul+Softmax    | Falcon-40B        | K=64, M=4096, N=4096            | 12.02        | Vector Eng |
| Group Query Attention  | Qwen3 0.6B/1.7B   | B=1, D=128, KH=8, N=4096, QH=16 | 19.12        | Tensor Eng |
| LoRA                   | DeepSeek-V2.5     | K=5120, M=4096, N=12288, R=128  | 30.17        | Tensor Eng |
| Mamba block            | Synthesized       | C=256, M=7168, S=16             | 2.89         | Vector Eng |
| Matmul+Add+RMSNorm     | Qwen3 1.7B        | K=2048, M=4096, N=2048          | 2.67         | Tensor Eng |
| Matmul                 | DeepSeek-V2.5     | K=5120, M=4096, N=12288         | 35.27        | Tensor Eng |
| RMSNorm+Matmul         | Qwen3 0.6B        | K=1024, M=4096, N=2048          | 1.06         | Tensor Eng |
| RoPE                   | Qwen3 32B         | B=1, D=128, H=64, N=4096        | 4.33         | Memory BW  |
| SiLU                   | DeepSeek-V3 671B  | M=4096, N=7168                  | 1.33         | Memory BW  |
| SwiGLU                 | Qwen3 0.6B        | K=1024, M=4096, N=3072          | 4.22         | Tensor Eng |
| Transpose+Matmul       | DeepSeek-MoE-16B  | K=2048, M=4096, N=10944         | 9.61         | Tensor Eng |

Baseline kernel 由 Neuron Compiler 自动生成（10/14）或基于 NKI 官方 example 人工编写（4/14），初始性能差异大（从 ~9% 到 ~83% peak throughput）。

术语一般如何实现？如何使用？
NKIBench 由两部分组成：(1) 结构化 kernel 存储——每个 kernel 关联 operator config、baseline 源码、profiling 数据和 peak throughput 计算；(2) 分布式 profiling service——利用 Trainium core-level 和 machine-level 并行度批量测评 kernel，通过共享网络文件系统和 centralized manager 调度。Correctness check: `||output - cpuref|| < tol × ||cpuref||`（不同 kernel 独立设定 tol），performance measurement: warmup + 多轮平均取最小差异轮。机器的 cores 定期轮换以缓解性能波动。NKIBench 是持续的社区项目，将继续扩充 benchmark 内容。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
