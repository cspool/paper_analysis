## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Lens 提出一个面向资源受限环境的 CPU-GPU 混合高吞吐 MoE LLM 推理系统，核心包含五个组件：
    1. **两阶段 holistic performance model**：Stage 1 模型基于 CPU memory capacity、GPU compute、workload 特征（prompt/generation length）推导 PME（Parallelism-Memory Efficiency）指标和理论上界；Stage 2 模型引入 bounded batch size K、paged KV cache、prefill/decode overlapping 调度策略，精确预测端到端 wall-clock 时间（94% 准确率）。
    2. **Resource-Aware Scheduler**：包含 Prefill Scheduler 和 Decode Scheduler 双调度器，均运行在 GPU 上以 GPU memory 维护调度状态。支持 Normal Inference Mode（prefill/decode 并行调度）和 Preemption Mode（KV cache 不足时抢占 decode 序列、释放 KV cache blocks、将抢占序列重新注入 prefill 阶段）。
    3. **Pipeline Profiler**：基于 Equation 2 估算 GPU compute 饱和所需 token 阈值 $n_{real}$，通过测量不同 token 数量的 GPU 时长和单层 weight 传输时间来校准。Scheduler 确保调度 token 数不超过 $n_{real}$。
    4. **VSLPipe 执行引擎**：将 MoE transformer layer 的 compute graph 重组为 `GA (QKV proj + GPU Flash Attn)` → `C (CPU Decode Attn + KV cache store)` → `GB (O proj + MoE layer)`，跨 layer 重组成 execution stage（CPU-only phase → GPU-only phase）。采用 software pipeline（prologue → N-1 main stages → epilogue）将 prefill 和 decode tokens 分两组 $\alpha$/$\beta$ 交替执行，CPU attention 与 GPU GEMM 重叠。
    5. **Contiguous Data Mover**：独立线程运行的 C++ PyTorch extension，以 100MB packet size 分批传输 weight，避免与 PyTorch 计算传输的头线阻塞。Weight Buffer 大小为 $2 \times$ per-layer weight size，仅为原模型大小的几个百分比。
    6. **CPU Decode Attention**：手工 AVX512 SIMD intrinsics 实现的 decode attention kernel（§6.6），manual vectorization + loop unrolling + data prefetching，单线程 4.7×、全线程 3.1× 高于 auto-vectorized baseline。
  - 实验比较：(1) MoE-Lens vs MoE-Lightning 和 vLLM（CPU offload）的 generation throughput（tokens/sec）；(2) 不同模型（Mixtral8x7B 94GB、Mixtral8x22B 282GB、DBRX 264GB）下的吞吐对比；(3) 不同 KV cache 大小（70GB、210GB）对 throughput 的影响；(4) 不同数据集（MTBench 多轮对话、RAG 长 prompt、AIME2024 长生成）下的吞吐对比；(5) 不同 generation length（32、64、128、256 tokens）对 throughput 的影响；(6) 性能模型预测精度（94% accuracy）；(7) 详细执行状态分析（prefill/decode throughput timeline、GPU/CPU/IO 时间分解、preemption 频率、bandwidth contention）。

- 硬件平台是什么，配置是什么。
  - Dual-socket Intel Platinum 8380 CPU（每 socket 8×DDR4-3200 channels，总计 750GB），使用 numactl 限制到单 socket + 单 GPU。实测单 socket CPU memory bandwidth ~150GB/s。
  - NVIDIA A40 GPU（48GB），通过分配随机 tensor 模拟 T4/L4 级别内存（16-24GB effective GPU memory）。
  - PCIe 互连，实测 $B_{IO} \approx 19.5$ GB/s（1GB tensor transfer）。
  - KV cache 配置：70GB-210GB（模拟不同 CPU memory capacity）。

- 开源Serving框架是什么。修改了什么。
  - 论文自身未开源（2504.09345，2025年4月，无公开代码）。
  - 对比 baseline 的 serving 框架：
    - **MoE-Lightning** [9]：state-of-the-art 资源受限 MoE 推理系统，基于 Hierarchical Roofline Model（HRM）将 decode attention offload 到 CPU。开源实现参考 [8]（https://github.com/caoshiyi/artifacts/tree/asplos25）。
    - **vLLM** [26]：基于 paged attention 的 LLM serving 系统，使用 CPU offload 选项运行超大模型。开源 https://github.com/vllm-project/vllm。
  - MoE-Lens 相比 MoE-Lightning 的核心修改：
    1. 用 holistic two-stage model 替代 HRM（HRM 仅建模 arithmetic intensity 和 IO bandwidth，忽略 CPU memory capacity 和 request 特征）。
    2. Resource-Aware Scheduler 替代 MoE-Lightning 的独立 prefill/decode 调度（前者重叠 prefill/decode，后者分离执行）。
    3. VSLPipe + Contiguous Data Mover 替代 MoE-Lightning 的 pipelining（前者最大化 IO bandwidth 利用率，后者存在 IO stall）。
  - 相比 vLLM 的核心修改：vLLM 在 GPU 上计算所有 GEMM 和 attention，仅 page KV cache 到 CPU，受限于 PCIe 带宽。MoE-Lens 将 attention 完全 offload 到 CPU 执行，避免 KV cache 传输。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？
  - **MoE-Lens 未开源**。论文 arXiv: 2504.09345。
  - **框架使用流程（基于论文描述）**：
    1. **部署阶段**：Pipeline Profiler 对目标 GPU 和模型进行 profiling——Equation 2 估算 $n_{real}$，变参测量 GPU time 和 weight transfer time，拟合线性关系得到精确的饱和 token 阈值。
    2. **请求到达**：incoming requests 进入 Prefill Scheduler 队列。
    3. **Normal Inference Mode**：
       - Decode Scheduler 估算现有 decode sequences 的 KV cache block 需求（基于 paged KV cache，每 block b tokens）。
       - 若 KV cache 充足，Decode Scheduler 率先调度所有 decode sequences。
       - Prefill Scheduler 读取活跃 decode 数量，计算可额外调度的 prefill tokens 数（不超过 $n_{real}$），从队列头部调度 prefill requests。
    4. **VSLPipe 执行**：每个 stage 包含 CPU-only phase + GPU-only phase。$\alpha$ 组和 $\beta$ 组交替执行：一组在 GPU 做 GEMM 时另一组在 CPU 做 attention。每个 stage 开始前 Contiguous Data Mover 预取下一 stage 的 weights。
    5. **Preemption Mode**：若 decode 所需 KV cache blocks 不足，preempt 部分 decode sequences → 回收 KV cache → Prefill Scheduler 将抢占序列作为新 prefill 序列重新注入，利用 prefill/decode overlapping 隐藏重计算开销。
    6. **完成**：Decode Scheduler garbage collection 回收 KV cache blocks。
  - **作用**：最大化 CPU memory capacity 利用率（KV cache），通过 prefill/decode overlapping 平滑 GPU 和 PCIe 利用率，减少 GPU idle time。在 MTBench 上以 70GB KV cache 可达 ~90% GPU utilization（$g_{max}=32$），较 MoE-Lightning 平均 4.6× 加速。
