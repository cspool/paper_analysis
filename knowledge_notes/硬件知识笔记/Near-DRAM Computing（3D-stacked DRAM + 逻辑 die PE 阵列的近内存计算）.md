## Near-DRAM Computing（3D-stacked DRAM + 逻辑 die PE 阵列的近内存计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Near-DRAM computing 指把计算单元放在 DRAM 侧（而非 CPU/NPU 侧）的近内存计算形态：DRAM die 负责缓存/存储，紧邻其下的逻辑 die 放置 PE 阵列执行计算，用极高内部带宽消除数据搬运。DIAMoND 的 near-DRAM 模块（Tab.II/Fig.5d-e）：3D-stacked DRAM 共 1.5GB、4 层、每层 2×2 tiles（每 tile 96MB），层间用 hybrid bonding + mini-TSV 互连，带宽 1620GB/s；底部 Logic & Control die 分 4 个 tile（对齐 DRAM 2×2 tile 布局），每 tile 含 PE 阵列（每 PE 十六个 1×16 MAC 阵列，8-bit VMM，1GHz 峰值合计 2 TOPs）+ SRAM I/O buffer + I/O 控制器 + softmax/SiLU 专用单元；DRAM 功耗 3.6W、面积 48mm²，逻辑部分 2.035mm²/1.7W。与 bank 内 PIM 的区别：计算在逻辑 die 上（非 DRAM bank 内），形态接近 Duplex（MICRO'24）的 Logic-PIM——把 GEMM 模块、buffer、softmax/激活单元放在 logic die、用 TSV 提 4× 带宽。HBM 为何不用：容量 4~80GB 与功耗 15~45W 均超边缘约束，且易失性需外挂 SSD 存权重。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DIAMoND 中 near-DRAM 的职责分工：FFN 专家矩阵（动态选中、权重巨大）交给 in-NAND 算（占 76% 负载），self-attention 与 KV cache 交给 near-DRAM 算（占 24%）——因为 KV cache（128~224MB）需动态写、NAND 耐久不足，而 DRAM 高耐久低 P/E 延迟。执行流程（QKV-Attention Joint Caching Pipeline）：W_Q 权重驻 DRAM、W_K/W_V 留 NAND → near-DRAM 先算 Q 投影 → 得到 Q 后立即对历史 K（缓存于 DRAM）做 Softmax(QK^T) 主体计算 → 同时 NAND 侧并行算 K/V 投影 → 只等最终 K 结果补齐 attention 尾部 → 延迟 −13.5%。每 token 解码的硬件级路径：attention 输出 h → 路由打分（NPU/逻辑侧）→ AES 电路选专家 → in-NAND 算 FFN（3 cycles）→ 结果与 near-DRAM 侧状态汇合输出 logits。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现配置（DIAMoND Tab.II）：3D-stacked DRAM 采 UniIC SeDRAM 多层阵列路线（VLSI'23，135GBps/Gbit、0.66pJ/bit，fine-pitch HB + mini-TSV）；逻辑 die 的 PE/加法单元 Verilog + Synopsys DC 28nm PDK 综合，SRAM 用 CACTI 建模，softmax/SiLU 单元采 [46][62]。使用方式：作为边缘 LLM 加速器的"活跃状态层"——KV cache、attention、非线性算子与路由；同族工作：Duplex（MICRO'24，expert/attention co-processing，专家按负载分 xPU/Logic-PIM）、NeuPIMs（ASPLOS'24，NPU+PIM 异构）、Stratum（tiered monolithic 3D DRAM）、InstAttention（in-storage attention offload）。评估：基于 SSDsim 基座的 cycle-accurate 模拟器建模 DRAM 时序与能耗。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
