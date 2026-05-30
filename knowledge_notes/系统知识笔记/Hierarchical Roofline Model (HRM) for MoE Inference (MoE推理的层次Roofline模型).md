## Hierarchical Roofline Model (HRM) for MoE Inference (MoE推理的层次Roofline模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Roofline Model (HRM) 是 MoE-Lightning 提出的 CPU-GPU 混合系统性能模型，扩展自经典 Roofline Model [Williams et al. 2009]。核心公式为式 (7)：$P_x^i = \min(P_{peak}^i, B_{peak}^i \times I_x^i, B_{peak}^{j,i} \times I_x^j)$，其中 level j 为 CPU、level i 为 GPU。即某计算任务 x 在 GPU 上的峰值性能受限于三者的最小值：(1) GPU 计算峰值 $P_{peak}^i$，(2) GPU HBM bandwidth roof $B_{peak}^i \times I_x^i$，(3) PCIe bandwidth roof $B_{peak}^{j,i} \times I_x^j$。对于就地计算无需跨层数据传输的场景，简化为式 (8)：$P_x^i = \min(P_{peak}^i, B_{peak}^i \times I_x^i)$，即经典 Roofline Model。

HRM 引入两个 Turning Points 和一个 Balance Point：(1) $P_1$ (Eq. 9)——当 $I_x^j$ 低于 $P_1$ 的 critical intensity 时，不值得将数据从 CPU 传输到 GPU 计算，应就地计算（例如 decode attention 在 CPU 执行）；(2) $P_2$ (Eq. 10)——低于此 intensity 时系统受限于 PCIe 带宽，需增大 batch size 或静态放置部分 weights；(3) Balance Point (Eq. 11)——$B_{peak}^i \times I_x^i = B_{peak}^{j,i} \times I_x^j$，GPU BW 与 PCIe BW 达到平衡，资源利用最大化。

例如：Mixtral 8x7B decode attention 的 operational intensity 极低（GEMV, < 1 FLOP/Byte），低于 $P_1$ → 不值得 KV cache H2D 到 GPU 做 attention → 在 CPU 执行。MoE FFN 的 operational intensity 随 batch size N 增大而增大，最终在 Balance Point 达到峰值。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HRM 分析流程（以 Mixtral 8x7B on L4 为例）：(1) 对每个算子根据模型配置 M 计算理论 FLOPs 和内存访问字节数 → 计算 $I_x^{GPU}$（regarding GPU data）和 $I_x^{CPU}$（regarding CPU data）；(2) 绘制 HRM 图（Fig. 4 为 attention block，Fig. 5 为 MoE FFN block）——水平线 = compute roofs (GPU/CPU peak FLOPS)，斜线 = memory roofs (GPU BW, CPU BW, PCIe BW)；垂直虚线 = 不同配置下的 operational intensity；(3) 比较当前 intensity 与 turning points 位置判断 bottleneck (memory-bound / PCIe-bound / compute-bound)；(4) 若受限于 PCIe bandwidth，增大 N（需要更多 CPU memory）或放置部分 weights 在 GPU（需要更多 GPU memory）；(5) 若受限于 GPU memory capacity → 系统 throughput 上界由 GPU memory 决定 → TP 增加 GPU 数量可 linear/super-linear 提升上限。

MoE-Lightning 基于 HRM 构建 per-layer decode latency 模型 $T = \max(comm^{cpu\_to\_gpu}, T_{cpu}, T_{gpu})$，其中 $T_{gpu} = T_{attn}^g + T_{ffn}^g$，每个 $T_x = \max(comm_x, comp_x)$。MILP optimizer 在 CPU/GPU memory 约束下搜索最优 6 元组策略。

**已知局限**（来自 MoE-Lens 分析）：仅建模 operator-level arithmetic intensity vs bandwidth，(a) 不建模 CPU memory capacity 对 batch size 的硬限制，(b) 不建模 prompt/generation length 对 memory efficiency 的影响，(c) 不建模 prefill/decode overlapping，(d) 不建模 paged KV cache fragmentation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Lightning 使用理论 FLOPs/bytes 计算 + 硬件峰值参数 profiling，MILP 搜索时间 < 1 分钟（无需 FlexGen 的数小时 data fitting）。
- HRM 为通用模型——可扩展至任意多层内存层次（Disk→CPU→GPU 或 HBM→L2→SMEM）。论文仅建模 CPU-GPU 两层。
- MoE-Lens 用更全面的 Two-Stage Holistic Model 替代 HRM，将 CPU memory capacity utilization 从 35-56% 提升到 ~100%。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
