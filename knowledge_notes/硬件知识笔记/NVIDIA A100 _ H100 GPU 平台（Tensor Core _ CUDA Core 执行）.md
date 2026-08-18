## NVIDIA A100 / H100 GPU 平台（Tensor Core / CUDA Core 执行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA A100（Ampere 架构，40GB PCIe 版）与 H100（Hopper 架构，80GB PCIe 版）是 QiMeng-Tensify（ISCA'26）的评估目标 GPU：FP16 推理/训练在 Tensor Core 上执行（张量核的矩阵乘加速），FP32 在传统 CUDA Core 上执行。两者微架构差异显著——H100 有更大的寄存器文件、更大 L2 cache 与更高的 Tensor Core 吞吐（FP16 Tensor Core 吞吐约 2 倍于 A100），这直接驱动论文的架构感知设计：同一程序在 A100 上搜索更常采样 memory-centric 规则（AutoInline、ComputeAtLocation 以降寄存器压力、提升 L2 局部性），在 H100 上更常采样 compute-centric 规则（MultiLevelTiling、ParallelizeVectorizeUnroll、CrossThreadReduction 以利用更大寄存器文件与更高 tensorcore 吞吐）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的运转流程（GatedMLP kernel 执行）：生成的 CUDA kernel 由主机端（Intel Xeon Gold 6330，112 cores @ 2.0GHz，跑 MCTS 编排）启动 → GPU 的 SM 调度 warp → 每 warp 在 CUDA Core（FP32 模式）或 Tensor Core（FP16 模式）执行 tile 计算 → 多级 tiling 使数据流经 全局内存→L2→共享内存→寄存器→Tensor Core 的层次，global-to-shared 异步拷贝与上一 tile 计算重叠 → 执行后用 profiler 采集 SM Efficiency/Shared Utilization/Achieved Occupancy 等指标反馈给搜索。H100 的更高寄存器容量允许更深 unroll/向量化而不 spill（故 compute-centric 规则更优）；A100 寄存器压力更紧则 memory-centric 规则（inline/compute_at）更优——这就是"架构感知"的硬件根源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为编译框架的目标后端（TVM/TensorIR 的 CUDA codegen 输出在 A100/H100 上运行），论文用 A100（40GB PCIe）与 H100（80GB PCIe）双平台对比；精度分 FP32（CUDA core）与 FP16（TensorCore）两档（子图级全部 9 个 baseline 都按此配置）。使用方式：子图级与网络级（Chameleon-7B/LLaMA3-8B/GPT-3-7B-LoRA/nGPT-1B，batch 1/8、seq 4096）都在两平台测量；结果 H100 上相对 PyTorch/TensorRT-LLM/Mirage 加速（1.78×/1.29×/1.30×）高于 A100（1.56×/1.22×/1.30×）；可移植性实验还涉及 A30（同代 SKU）与 CUDA 11.8/12.4 版本切换。注意：本条目是"评估平台"术语，与既有条目 "Tensor Core 与 SM 内管线划分"（描述 SM 内 tensor pipeline vs INT/LSU pipeline 的硬件结构）互补不重复。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，8×A100 SXM 80GB MLLM serving）：RESONATOR 部署在单服务器 8×A100 SXM 80GB + Intel Xeon Gold 6430，GPU 间 NVLink；LLM backbone TP=4 on 4 A100（Qwen2-VL-7B、Kimi-VL-16B）、TP=8 on 8 A100（Qwen2-VL-72B）；encoder 每 GPU 全量预载（HBM 开销 1.6%/1.0%）；A100 的 SM 分区粒度（108 SM、ΔSM 级）决定 Atlas 的 SM 分配层级集合 S 与 SM_dec_min 配额粒度；A100 FP16 峰值 312 TFLOPS 用作 logical sharding MFU 归一化。
涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
