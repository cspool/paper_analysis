## Host-GPU Heterogeneous Architecture for LLM Inference（LLM推理的Host-GPU异构架构）

术语是什么？
Host-GPU Heterogeneous Architecture 指在 LLM 推理中，将模型参数分布存储在 CPU 主机内存（host DRAM）和 GPU 设备显存（device HBM）之间的计算架构。由于大模型参数规模常超过单 GPU 显存限制，该架构通过 PCIe/NVLink 互联实现 host→GPU 参数传输，利用 GPU 的高并行计算能力执行推理。在 MoE 推理中，非 MoE 参数（attention, embedding）常驻 GPU，大量 expert 参数存储于 host DRAM，仅按需通过 PCIe 加载激活的 experts。

从硬件架构角度拆解术语：
在 Diff-MoE 的硬件配置中（H200 GPU + 2×Xeon Gold 6430 + 1TB DRAM，PCIe 5.0 128 GB/s 双向）：

1. **GPU 端 (H200, 141 GB HBM)**：执行推理计算。141 GB HBM 中，约 16% 分配给 Diff-MoE 的三级缓存（HPC+MPC+LPC，α=5%），其余用于非 MoE 参数（attention weights, embeddings, KV cache）和计算中间结果。HBM 带宽 4.8 TB/s（远高于 PCIe 128 GB/s），因此计算速度取决于数据是否已在 HBM 中。
2. **Host 端 (1 TB DRAM)**：存储所有 expert 参数。Switch-Base (7B) 的 MoE 参数约 6.8 GB（128 experts × 6 layers × ~8.7 MB/expert），Switch-Large (26B) 约 26 GB。Host DRAM 容量大但带宽低（DDR5 ~50-100 GB/s），且数据需经 PCIe 传到 GPU 才能计算。
3. **PCIe 5.0 互联**：双向 128 GB/s 带宽。这是主要瓶颈：batch=64 时约传输 2.9 GB expert 数据（~23 ms），远超 GPU 计算时间（~2.5 ms）。
4. **内存层次等效**：实际形成 Host DRAM (TB, slow) → PCIe (128 GB/s) → GPU HBM (141 GB, 4.8 TB/s) 的两级物理存储层次。Diff-MoE 的软件缓存层级（LPC→MPC→HPC）通过在 GPU HBM 内进一步划分来缓解 PCIe 的带宽瓶颈。

关键瓶颈分析：GPU 计算时间随 batch size 增长缓慢（batch 1→64 约 1.26×），而 PCIe 传输量随 batch size 线性增长（因更多 experts 被激活），导致大 batch 下通信时间占比急剧上升至 >97%。

术语一般如何实现？如何使用？
常见实现方式：
- **PyTorch `device_map`**：HuggingFace `accelerate` 提供 `device_map="auto"`，自动将模型 layer 分配到 CPU/GPU。MoE 场景需额外处理 expert 粒度。
- **DeepSpeed-Inference**：提供 `--offload` 参数，将指定模块（如 MoE layer）卸载到 CPU，运行时注入 CPU→GPU 传输。
- **FasterTransformer + 自定义 offloading**：手动实现 per-expert 粒度的 CPU↔GPU 传输管理。
- 硬件互联选择：PCIe 4.0 (32 GB/s 单向) vs PCIe 5.0 (64 GB/s 单向) vs NVLink-C2C (900 GB/s for Grace-Hopper) vs CXL 共享内存（未来的低延迟方案）。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
