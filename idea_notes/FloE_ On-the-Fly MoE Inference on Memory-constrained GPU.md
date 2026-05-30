## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- baseline方法是什么？
  - **Naive MoE Offloading（DeepSpeed-MII ZeRO-Infinity）**：非 expert 权重常驻 VRAM，expert 权重全部 offload 到 DRAM。每 token 推理时，router 确定激活 expert → PCIe 传输对应 expert 权重（~300MB/expert，各投影矩阵 FP16 全量）→ GPU 执行完整 dense GEMV。瓶颈：MoE 推理从 memory-bound 转变为 I/O-bound，PCIe 4.0 单向带宽仅 ~32GB/s，而 VRAM 带宽 ~300GB/s+，单 expert 传输耗时 ~15ms vs 计算 ~5ms，3:1 传输-计算比。
  - **Advanced MoE Offloading（Mixtral-Offloading, Fiddler）**：增加 expert 预测器 + LRU cache + uniform INT3 量化。Mixtral-Offloading 用 intermediate results 预测下一层 expert 并预取+缓存，配合 uniform 量化压缩传输。Fiddler 将部分计算卸载到 CPU 以减少传输需求。痛点：(1) uniform quantization 对所有投影矩阵同等对待，gate/down 对量化敏感，INT2 时 perplexity 暴涨；(2) 预测-传输-计算串行执行，无法 pipeline；(3) 学习型稀疏预测器额外消耗 2.19~9GB VRAM。
  - 全栈执行例子（Mixtral-Offloading baseline）：token 嵌入 → GPU 计算 Attention → Router 选择 expert E_i → CPU 从 DRAM 加载 INT3 E_i 权重（gate/up/down 统一量化）→ PCIe 传输 → GPU dense GEMV（gate→SiLU, up→×, ×→down）→ 输出。传输与计算串行，吞吐受 I/O 限制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - FloE 通过三个核心设计解决 baseline 缺陷：
    1. **Hybrid Compression（混合压缩）**：不再 uniform 量化，而是利用 expert 内部参数的差异化敏感性——gate/down 用 contextual sparsification（按 up projection 输出幅值剪枝对应通道，理论证明 L_down ≤ L_up < L_gate），仅保留 ~10% 通道；up projection 用 INT2 HQQ（因 up 对量化天然不敏感）。实现 9.3× 压缩比而无 uniform INT3/INT2 的精度崩溃。
    2. **Dual Sparsity Predictors（双稀疏预测器）**：利用相邻层 hidden state 相似度 >0.95 的观察，inter-expert predictor（MLP）预测下一层激活 expert，intra-expert predictor（复用 W_up 做矩阵乘）预计算稀疏掩码。两个预测器均用当前层 hidden state 预测下一层，打破串行依赖，实现传输-计算 pipeline。intra predictor 零额外参数，相比学习型 baseline（PowerInfer 9GB / SparseInfer 2.19GB）极大降低内存开销。
    3. **System Co-optimization（系统协同优化）**：Sparse GEMV kernel（Triton, 列主序转置+选择性列加载+SiLU 融合）消除稀疏带来的计算开销；Compact async transfer（AVX-512 SIMD+多线程+pinned memory+多 stream）将传输利用率从 ~7%（PyTorch 原生）提升至 88% PCIe 峰值。
  - 全栈执行例子（FloE 第 i 层推理）：
    - **算法层**：hidden state x_i → inter-expert predictor MLP(x_i) → 预测层 i+1 的激活 expert 索引 + intra-expert predictor(x_i, W_up_{i+1}) → 预计算稀疏掩码。
    - **系统框架层**：FloE scheduler 根据预测结果触发 compact async transfer，从 DRAM 预取层 i+1 的压缩 expert（gate/down 仅 ~10% 列/行转置 + up INT2）。
    - **编译框架层**：论文未明确说明（使用 PyTorch/Triton 标准编译栈，无自定义编译 pass）。
    - **Kernel 调度层**：Sparse GEMV kernel 在 GPU 执行——加载 x_i → W_up 全精度 GEMV → 阈值掩码 → 选择性加载 W_gate[mask] 列和 W_down^T[mask] 列 → 融合 SiLU+Hadamard+sparse GEMV → 输出。同时 AVX-512 多线程在 CPU 端打包下一层 expert 权重到 pinned memory，CUDA stream 异步传输。
    - **硬件架构/芯片设计层**：论文未明确说明（使用 consumer GPU + CPU + PCIe 标准硬件）。
  - 关键对比：baseline 中单 expert 传输 15ms + 计算 5ms（串行 20ms），FloE 中压缩后传输 ~1.6ms + 计算 ~3ms（pipeline 后有效延迟 ~3ms），实现 48.7× vs DeepSpeed-MII 的端到端加速。
