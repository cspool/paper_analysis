## Concat Unit（拼接单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Concat Unit 是 OASIS PE Line 中最基础的计算单元（§IV-A）：接收两个 4-bit 索引（激活索引与权重索引），拼接成一个 8-bit 寄存器值并输出，即 concat_idx = (idx_A << 4) | idx_W。设计哲学：极简位操作（无乘法、无浮点），面积远小于 FP16 MAC——在 compute-intensive 的 prefill 阶段芯片可放置极多 Concat Unit 提供高计算并行度（每条 PE Line 4096 个）；在 memory-intensive 的 decode 阶段轻量单元占用极小面积，把芯片面积让给额外 I/O pin 提升带宽、缓解访存瓶颈。硬件配置：每条 PE Line 4096 个 Concat Unit（每芯片 16×4096=65536 个），单 Concat Unit 面积 8.68×10⁻² mm²/功耗 8.36×10⁻² W 级（Table II，按每 line 计 8.68×10⁻²）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Concat Unit 在 OASIS 主分支流水中的位置（图8 步骤③）：Clustering Unit 把激活聚类成 4-bit 索引存入 Activation Index Buffer → 激活索引广播到 16 条 PE Line → 每条 PE Line 的 4096 个 Concat Unit 并行把广播的激活索引与 Weight Index Buffer 中对应列的 4-bit 权重索引拼接 → 8-bit 拼接索引送 Index Counter。示例（W4A4，K=4096, N=4096）：第 n 输出通道的 Concat Unit 并行处理 K 个位置，每个单元把 idx_A[k] 与 idx_W[k,n] 拼成 8-bit，一周期内 4096 个单元完成整列拼接，产出 4096 个拼接索引交给 32 个 Index Counter 统计。设计动机：WAQ LUT-GEMM 的核心洞察——量化后乘法退化为索引拼接 + 查表，因此"拼接"是唯一需要的 per-element 运算，其位操作本质支撑 16× FLOPs 下降与 1024× 并行度提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：两个 4-bit 寄存器/锁存 + 位拼接逻辑（移位+OR），输出 8-bit 寄存器；多路复用/广播网络把激活索引分发到全部 Concat Unit。使用场景：双测量化 LUT-GEMM 加速器的标准前端算子；与 GPU 上"打包索引 + 查表"kernel（LUT-GEMM/FLUTE/T-MAC 的 in-register table）同思路，但 Concat Unit 是硬连线、无 bank conflict。无公开 RTL；面积/功耗来自 TSMC 28nm 综合（Table II）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
