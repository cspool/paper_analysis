## Mixed Precision Expert Offloading

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixed Precision Expert Offloading 是一种 MoE 推理加速技术，在传统 Expert Offloading（将 non-expert 权重 + 部分 hot expert 缓存于 GPU，其余 expert 卸载到 CPU/SSD）的基础上，为不同 expert 维护多个精度版本（如 FP16 + INT4 或 INT8 + INT2），运行时根据 expert 重要性动态选择加载精度。核心思想：MoE 中不同 expert 对输出贡献不同，不重要的 cache-miss expert 可用低精度版本替代，大幅减少 PCIe/SSD 传输量（FP16→INT4 减少 4×），同时保持模型精度（精度下降 <1%）。该技术由 HOBBIT (Tang et al., 2024) 首次系统化提出，后续 MoE-APEX 等系统沿用并扩展。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

HOBBIT 中 Mixed Precision Expert Offloading 的系统架构流程（以 Mixtral-8x7B + RTX 4090 为例）：

1. **权重分布**：所有 non-expert 权重（Attention + LayerNorm 等，约 4% 参数）常驻 GPU memory。Expert 权重存储于 CPU memory 或 SSD，每个 expert 维护两个精度版本（FP16 高精度 + INT4 低精度）。GPU memory 中维护两个分离的 expert cache：High-Precision Cache（较大）和 Low-Precision Cache（较小）。
2. **运行时精度决策**：每 token 的 MoE 层 gating 计算完成后，对所有 top-K experts 按 ||G(x)|| 降序排列，计算 unimportance degree score s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||（已归一化）。双阈值 T1/T2 决定精度：
   - s ≤ T1 → 高精度加载（FP16，~10.5MB/expert）
   - T1 < s ≤ T2 → 低精度加载（INT4，~2.6MB/expert）
   - s > T2 → 跳过该 expert
3. **异步多精度加载**：Dynamic Expert Loader 的 Expert Scheduler 通过 read() 系统调用从 CPU memory 异步加载对应精度 expert 权重到 GPU cache。由于传输量减少，PCIe 带宽利用率提升，总加载延迟降低。
4. **混合精度缓存管理**：Multidimensional Cache Manager 使用 LHU + LRU + LFU + FLD 加权策略管理高/低精度分离 cache，优先保留高精度高频使用的 expert。

配置示例（Mixtral-8x7B, T1=0.6, T2=0.9）：67% expert 加载为高精度，30% 加载为低精度，3% 跳过。top-1 expert 始终得分 0（保持高精度），所有 top-2 experts 的分布满足精度保持要求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现基础：基于 Llama.cpp（C++/C），HOBBIT 增加 8000 行代码。将原始的 per-layer 权重加载改为 per-expert 多精度加载。
- 低精度 expert 的生成：使用标准后训练量化（如 GPTQ 或 RTN round-to-nearest），对每个 expert 独立量化为 INT4/INT2 并存储于 CPU memory。
- 运行时切换：无需修改模型架构，仅需在加载路径中添加精度选择逻辑。Expert Loader 维护 Task Queue，按优先级异步加载。
- 阈值选择：通过 profiling 一次推理的 ||G(x)|| 分布确定 T1/T2（如 Mixtral-8x7B 的 T1=0.6, T2=0.9），跨模型和硬件可复用。
- 效果：RTX 4090 上 vs MoE-Offloading decoding speedup 3.21× (Mixtral) / 3.29× (Phi-MoE)，vs MoE-Infinity speedup 2.30× / 3.92×。Jetson Orin 上 vs Llama.cpp speedup 13.0× / 18.9×。

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading
