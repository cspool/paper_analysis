## Expert Offloading（专家参数卸载）

术语是什么？
Expert Offloading 是一种在资源受限 GPU 上部署大规模 MoE 模型的系统策略。由于 MoE 模型的 expert 参数量巨大（如 Switch-XXL 在 FP32 下需 >1.5 TB 内存），远超单 GPU 显存（如 H200 141 GB），Expert Offloading 将大部分 expert 参数存储在 host CPU DRAM 中，仅将推理时 gating network 选中的少数 activated experts 按需传输到 GPU 显存执行计算。计算完成后，专家参数可被保留（caching）或驱逐（eviction）。核心思想是利用 MoE 的稀疏激活特性（每个 token 仅激活 top-k experts）缩减 GPU 内存需求。

从系统架构角度拆解术语：
在 Diff-MoE 描述的场景中，Expert Offloading 的系统流程为：

1. **初始化**：非 MoE 参数（attention weights, embeddings, layernorm 等）常驻 GPU。所有 expert 参数保留在 host DRAM。
2. **按需加载（On-Demand Fetch）**：Gating network 输出当前层的 top-K activated expert IDs 后，系统检查每个 activated expert 是否已在 GPU 缓存中。若缺失（cache miss），触发 `cudaMemcpy` 从 host DRAM → GPU memory 的 PCIe 传输。
3. **计算**：Expert FFN 在 GPU 上并行执行（batch matmul, SwiGLU）。
4. **驱逐/缓存策略**：计算完成后，根据缓存策略决定保留（如 LRU、LRU+priority）或驱逐 expert 以释放 GPU 空间。
5. **预取**：部分方案（如 Pre-gated MoE、Diff-MoE）在当前层计算期间异步预取下一层所需 experts，重叠通信与计算。

以 batch_size=64, Switch-Base (7B), top-1 gating 为例：128 experts 中约 30-34 个被激活，每个 expert 约 85 MB，总传输量约 2.9 GB。PCIe 5.0 双向 128 GB/s，纯传输约需 23 ms。

当前 SOTA Expert Offloading 方案分为三类：
- **On-demand only**（DeepSpeed-Offload）：计算后再驱逐，无缓存无预取。通信开销最大。
- **Prefetch-based**（Pre-gated MoE、MoESys、Fiddler）：提前预取下一层 experts，重叠传输与计算。在大 batch 下计算窗口仅能覆盖 1-2 个 expert 的传输。
- **Cache-based**（MoE-Infinity、ProMoE）：在 GPU 中缓存频繁使用的 experts，减少重复传输。在大 batch 下全局共享缓存命中率崩塌。
- **Hybrid cache+prefetch**（Diff-MoE）：差分缓存层级（per-layer HPC+MPC+共享 LPC）+ GRU predictor 预取。

术语一般如何实现？如何使用？
主要实现方式：
- **DeepSpeed-Inference**：通过 `deepspeed.init_inference()` 的 `mpu` 参数配置 expert offloading。将 MoE 参数标记为可卸载，运行时按需加载后计算，计算完立即释放。
- **FasterTransformer + 自定义 offload 模块**：Diff-MoE 基于 FasterTransformer 实现，通过将 HuggingFace 模型 bin 文件拆分为 per-expert 细粒度文件，注入缓存管理层。
- **PyTorch `accelerate`**：通过 `device_map="auto"` 实现简单的 per-layer offloading，但 MoE 的 per-expert 粒度需要额外处理。
- **llama.cpp MoE**：支持 MoE 模型的 CPU/GPU 混合推理，通过 `--n-gpu-layers` 控制 GPU 层数。

关键配置约束：PCIe 带宽（PCIe 4.0: ~32 GB/s, PCIe 5.0: ~64 GB/s 单向）、GPU 显存预算（需为缓存 + 激活中间结果 + 非 MoE 参数预留空间）、batch size（决定一次性激活的 expert 数量）。

**ES-MoE 的 Training Expert Offloading**：ES-MoE (ICML '24) 首次将 expert offloading 应用于 MoE **训练**场景（非推理）。与推理时 offloading 的核心区别：(1) **双向数据流**：训练需要 forward pass 上传 experts (CPU→GPU) + backward pass 下载 gradients (GPU→CPU) + CPU optimizer 更新参数；(2) **Optimizer States Offloading**：Adam optimizer states（12 bytes/param, fp32）占用 6× 于 model params（2 bytes/param, fp16），offload 到 CPU 后由 CPU Adam optimizer 更新，避免 GPU→CPU 梯度下载后再上传 optimizer states；(3) **Expert-wise Optimizer**：每个 expert backward 完成即触发 CPU Adam step，与后续 layers 的 GPU forward/backward 重叠——这与推理 offload 的"仅 forward 时预取"根本不同；(4) **SSD 扩展**：当 CPU RAM 不足时（cloud VM 限制），ES-MoE 使用 LRU cache + DMA-able pinned memory + prefetching 将 experts 扩展到 SSD。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
- SiDA-MoE: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models
- SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget

**SwapMoE 的 Amortized Expert Offloading on Edge Devices**：SwapMoE 提出了一种专为边缘设备设计的 expert offloading 策略。与现有方案在 per-layer or per-token 粒度加载 experts 不同，SwapMoE 维护跨层共享的 Virtual Experts 集合，每隔 N 个 sample（batch size=1 边缘场景）才触发一次 expert 更新：(1) 基于 runtime 收集的 expert importance scores 对全层 experts 排序；(2) 仅加载 top-k 最重要的 experts 到 GPU memory；(3) 卸载不再重要的 experts；(4) async copy engine 与 computation pipeline overlap。由于 activation locality（连续 sample 激活相似 experts），每次更新只需替换少量 experts（~1-2 per layer），实测 peak IO overhead ~40 MiB/s, mean ~20 MiB/s，适合 Jetson Nano/AGX ORIN 等边缘设备的 PCIe 和 CPU-SSD IO 路径。其 offline memory planning 通过 genetic algorithm + profiling data 搜索最优的 per-layer expert 数量和 update frequency。

**SiDA-MoE 的 Data-Aware FIFO Offloading**：SiDA-MoE (MLSys '24) 首次将 data-aware 视角引入 expert offloading。与 Pre-gated MoE 和 Diff-MoE 的 per-layer token-by-token 在线预测不同，SiDA-MoE 通过 offline-trained LSTM+Sparse Attention hash 函数在推理前一次性预测整个 batch 在所有 MoE 层的 expert 激活模式。系统包含两个并行线程：(1) Hash-building 线程在 CPU 上预测下一 batch 的 expert 激活并写入 hash table；(2) Inference 线程根据 hash table 动态加载激活 expert（CPU→GPU）并卸载未激活 expert（GPU→CPU），采用 FIFO 驱逐策略。所有 router 函数被 offload 到 CPU 内存，不参与推理前向。这使得 expert 选择开销完全从推理关键路径中移除。GPU 内存节省达 80%（SST2 短句场景），Hash 函数 Top-3 预测准确率达 99%。

Pre-gated MoE (ISCA '24) 将 Expert Offloading 从被动策略升级为主动策略。通过 pre-gate function 提前知道下一个 MoE block 需要的 experts，在当前 block 计算期间异步迁移仅激活的 experts（而非全部），传输量比 MoE-Prefetch 减少 ~100×。Peak GPU memory 公式：Peak_GPU_mem = max(Non_MoE_M + Σ_{L=N}^{N+1} Act_Exp_L)。

**Remoe 的 Serverless Expert Offloading**：Remoe 将 expert offloading 推广到 serverless 场景。与传统 offloading（所有 inactive experts 仍常驻 CPU 内存）不同，Remoe 将低频 "remote experts" 部署为独立 serverless function（Kubernetes Pod），按需冷启动，pay-per-use 计费。高频 "local experts" 与主模型同容器常驻 CPU。关键区别：(1) Remote experts 不在主模型容器中占用任何内存——它们仅在接收到 token 数据时才启动并计费；(2) Expert 选择由 SPS 算法（prompt 语义相似度预测）在推理开始前完成，而非 token-by-token 在线预测，避免 serverless 冷启动开销；(3) Remote expert Pods 可多 replica 并行执行（LPT 划分），与 local experts 的 CPU 计算重叠。Remoe 在 Deepseek-v2-lite 上实现 57.1% 推理成本降低，冷启动时间减少 47%。
