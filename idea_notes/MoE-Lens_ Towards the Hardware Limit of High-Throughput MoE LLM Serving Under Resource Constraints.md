## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- baseline方法是什么？
  Baseline 是 **MoE-Lightning** [9]，state-of-the-art 资源受限 MoE LLM 推理系统，基于 Hierarchical Roofline Model（HRM）指导系统设计。以 Mixtral8x7B（32 layers, 8 experts/layer, k=2, 94GB BF16）在 A40 GPU（16GB effective）+ CPU（750GB DRAM）上的离线批量推理执行路径为例：
  - **算法层（MoE Routing）**：标准 top-k gating，Router 对每层 self-attention 输出计算 logits → Softmax → SelectTopK(k=2)。每个 token 分配给 2 个 expert。**缺陷(1)**：HRM 仅建模 arithmetic intensity 和 IO bandwidth，忽略了 CPU memory capacity 对并行 token 数量的约束——Table 1 显示 MoE-Lightning 的 CPU memory utilization 仅 35%-56%，大量 KV cache 容量未被利用，导致 GPU 端并发 token 数不足。
  - **系统框架层**：MoE-Lightning 使用 HRM 指导下的 CPU-GPU hybrid 执行——decode attention offload 到 CPU，避免 KV cache 传输。prefill 和 decode 作为两个独立阶段串行执行。**缺陷(2)**：prefill/decode 分离导致资源利用不均衡——prefill 阶段 IO 仅 23.9% 活跃、decode 阶段 GPU 仅 16.5% 利用率（图 1）。**缺陷(3)**：独立阶段执行使 KV cache 的峰值内存占用为 $p+g$（最大序列长度），而非 prefill/decode 重叠下的平均占用 $p+g/2$，浪费了 CPU memory capacity 的有效利用率。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA。
  - **kernel调度层**：MoE-Lightning 使用标准 PyTorch/CBLAS 进行 CPU attention 计算，未针对 CPU 向量单元优化。**缺陷(4)**：auto-vectorized CPU attention 无法充分利用向量单元，达不到系统所需的 attention throughput。
  - **硬件架构层**：NVIDIA A40 GPU + CPU DRAM + PCIe 互连。IO bandwidth $B_{IO} \approx 19.5-32$ GB/s。MoE-Lightning 的 pipeline 中 weight transfer API 调用嵌入在执行流水线内，未与 PyTorch 操作和 attention 同步数据做隔离，导致 head-of-line blocking。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-Lens**，核心是 holistic two-stage performance model + model-informed system design（Resource-Aware Scheduler + VSLPipe + Contiguous Data Mover + hand-optimized CPU attention kernel）。以 Mixtral8x7B 在 A40（16GB effective）+ CPU（750GB）上运行 MTBench（p=98, g=32, 70GB KV cache）的执行路径为例：
  - **算法层（MoE Routing 不变）**：Router 逻辑不变，但通过 system-level optimization 使 GPU 端并发 token 数大幅提升。PME（Equation 3）量化了不同 prompt/generation length 下的 memory efficiency——g=32 时 PME 较高，GPU utilization 可达 ~90%；g=256 时 PME 显著下降，GPU utilization 受限。**解决缺陷(1)**：通过 Stage 1 model 的 CPU memory capacity 分析（Equation 2, Table 2），识别出饱和 GPU compute 所需的 KV cache 大小，指导 system 配置 KV cache capacity 以最大化并发 token 数。
  - **系统框架层（MoE-Lens System）**：
    1. **Stage 2 performance model** (§5.5)：将 bounded batch size K、paged KV cache（block size b）、prefill/decode overlapping 调度策略纳入模型，预测真实 throughput（94% accuracy）。**解决缺陷(2)+(3)**：prefill/decode overlapping 调度（Equation 7-13）不仅平衡了 prefill 和 decode 阶段的资源利用（GPU 利用率从 MoE-Lightning 的 16.5% 提升到 ~90%），还通过重叠执行使 KV cache 有效容量从 $C_{KV}$ 扩展为 $\frac{p+g}{p+g/2}C_{KV}$（Equation 7），降低了峰值内存占用。
    2. **Resource-Aware Scheduler** (§6.2)：Normal Inference Mode 下 Prefill Scheduler 和 Decode Scheduler 并行调度——Decode Scheduler 先调度所有 decode sequences → Prefill Scheduler 根据 Pipeline Profiler 的 $n_{real}$ 阈值补充 prefill tokens。KV cache 不足时进入 Preemption Mode 抢占部分 decode sequence、回收 KV cache、重新注入 pipeline。**解决缺陷(2)**：重叠调度最大化 GPU 利用率，preemption 机制保障 KV cache constrained 场景下的鲁棒性。
    3. **VSLPipe 执行引擎** (§6.4)：将每层计算图重组为 GA→C→GB，跨 layer 合并成 execution stage（CPU-only phase → H2D/D2H → GPU-only phase）。$\alpha$/$\beta$ 两组交替执行，CPU attention 与 GPU GEMM 完全重叠。每个 stage 开始时 Contiguous Data Mover 预取下一 stage weights。**解决缺陷(3)+(5)**：Contiguous Data Mover 将 weight transfer 从执行流水线中解耦，独立线程以 100MB packet size 分批传输，避免 head-of-line blocking 和与 PyTorch compute transfer 的竞争。
    4. **Hand-optimized CPU Decode Attention** (§6.6)：AVX512 intrinsics + loop unrolling + data prefetching，单线程 4.7× auto-vectorized baseline。**解决缺陷(4)**：CPU attention 达到系统所需 throughput（满足 KV cache = 2× model size 时的 attention 计算需求），使 CPU 不成为 bottleneck。
  - **编译框架层**：论文未明确说明。使用 C++ PyTorch extension（Contiguous Data Mover）和手写 SIMD kernel。
  - **kernel调度层**：Contiguous Data Mover 作为独立线程调度 CPU→GPU weight transfer packets，100MB packet size 平衡带宽和竞争。CPU attention kernel 在 VSLPipe CPU phase 执行，与 GPU GEMM 并行。
  - **硬件架构层**：同 baseline（A40 GPU + CPU + PCIe），但利用率大幅提升：GPU utilization 从 16.5% → ~90%（70GB KV cache, g=32），平均 throughput 4.6× MoE-Lightning（up to 25.5× on RAG）。
