## CPU-GPU Heterogeneous MoE Inference Orchestration

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CPU-GPU Heterogeneous MoE Inference Orchestration 是一种面向资源受限环境的 MoE 推理策略，将 MoE 模型的 expert 计算动态分配到 CPU 和 GPU 两种异构计算设备上执行。核心思想是：当 GPU 显存不足以容纳全部 expert 参数时，不采用单一的"全部 CPU 计算"或"全部 GPU offloading"策略，而是根据每个 expert 在每个推理步骤的实际输入量 s，在三种策略中动态选择最优方案：(a) 若 expert 权重已在 GPU，直接在 GPU 执行；(b) 若输入量 s 较大，将 expert 权重从 CPU 经 PCIe 拷贝到 GPU，在 GPU 执行；(c) 若输入量 s 较小，将 activation 从 GPU 拷贝到 CPU，在 CPU 执行后传回。决策基于 latency model：GPU 延迟近乎恒定（受限于内存带宽），CPU 延迟随输入量线性增长（受限于计算能力），PCIe 传输延迟恒定。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

以 Mixtral-8x7B 16-bit 在 Quadro RTX 6000 24GB 上 single-batch decode 为例的系统架构执行流程：

```
┌──────────────────────────────────────────────────────────────┐
│ 初始化阶段                                                     │
│   Non-expert 层 (Attention/Embedding/Norm, ~2B params): GPU    │
│   Expert 层 (32 layers × 8 experts = 256, ~45B params):        │
│     - Offline 热门度 profiling (ShareGPT calibration)          │
│     - Top-56 热门 expert → GPU memory                          │
│     - 其余 200 expert → CPU pinned memory                      │
│   Latency model 校准: cpu_lat(s), gpu_lat(s), trans_lat()     │
├──────────────────────────────────────────────────────────────┤
│ 推理执行阶段 (per-layer, per-token)                            │
│   1. Attention: GPU 直接执行（权重常驻 GPU）                    │
│   2. MoE Gate: GPU 直接执行 → top-2 expert index + inp_size[] │
│   3. Fiddler Algorithm 1 (per-expert 决策):                    │
│      for each expert j in top-2:                              │
│        s = inp_size[j]                                        │
│        if is_at_gpu(l, j):                                    │
│          → Strategy (a): GPU 直接计算 (无 PCIe 传输)            │
│        elif cpu_lat(s) > gpu_lat(s) + trans_lat():            │
│          → Strategy (b): PCIe copy weight → GPU 计算           │
│        else:                                                   │
│          → Strategy (c): PCIe copy activation → CPU AVX512    │
│            → PCIe copy output back to GPU                      │
│   4. Aggregation: Σ gate_weight * expert_output               │
└──────────────────────────────────────────────────────────────┘
```

Strategy (b) vs (c) 的权衡分析：
- Strategy (b) 传输量：~300MB/expert（3个 4096×14336 矩阵 at 16-bit），恒定
- Strategy (c) 传输量：s × 4096 × 2 bytes（activation），随 s 线性增长
- 对于 s=1（decode）：activation = 8KB << 300MB weight → (c) 更优
- 对于 s=1024（prefill）：activation ≈ 8MB，但 CPU 计算 s× 开销 >> PCIe 传输 → (b) 更优

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fiddler 基于 PyTorch 实现（开源在 https://github.com/efeslab/fiddler）：
- **Expert 权重管理**：CPU 侧使用 pinned memory（连续分配），支持 cudaMemcpyAsync 异步传输
- **CPU 计算 kernel**：自实现 AVX512_BF16 expert FFN kernel（替代 PyTorch 默认 CPU GEMM）
- **Latency model**：初始化阶段运行 microbenchmark 测量 cpu_lat(s)、gpu_lat(s)、trans_lat() 三个函数所需的常数参数（32 层平均），>99% R² 拟合
- **适用场景**：GPU 显存不足以容纳全量 expert 参数的 resource-constrained 场景；Single-batch decode、long-context prefill、beam search 均适用
- **通用性**：已在 Mixtral-8x7B 和 Phi-3.5-MoE 上验证，论文声明方法适用于 MoE 模型家族

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
