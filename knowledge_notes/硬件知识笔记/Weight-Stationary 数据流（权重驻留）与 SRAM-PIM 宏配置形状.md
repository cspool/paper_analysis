## Weight-Stationary 数据流（权重驻留）与 SRAM-PIM 宏配置形状

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-stationary = 权重固定在计算单元（CIM 宏）内不动、激活/输入流经单元——权重复用最大化，适合 batch 大、权重复用高的 GeMM。CompAir 用它判定 SRAM-PIM 的适用边界：纯 SRAM-PIM 全权重驻留算 FC（GPT3-175B 需不可行宏数量、功耗超 A100 三个数量级）→ 必须按 batch 复用换取收益：batch=1 时频繁权重写回无优势，batch=32 时 Q/K/V 6.3×。宏配置形状 (512,8) vs (256,16)：4 个 128 输入 8 输出宏的拼接方式（输入维扩展 vs 输入/输出各分），决定 DRAM→SRAM 带宽压力与归约开销的权衡。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 中数据流例子：权重 W 驻留 4×8KB 宏（拼成 (512,8)），每 batch 输入 X 经 DRAM→HB 写入宏输入寄存器，宏内 128×8 BF16 MAC 并行算 Y=WX 并写回 DRAM；每推理只需搬运输入/输出、权重仅重载时搬。形状选择依据：均值不等式下输入输出维相近时带宽需求最小；(256,16) 引入输入切分的小幅归约但大幅降低 DRAM→SRAM 带宽压力。DSE（图 22）发现宏形状存在分歧点：之前性能受输入带宽主导（不同电压无差别）、之后受 SRAM-PIM 延迟主导；宽输入形状在高带宽下更好。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CIM 宏一般支持 weight/input/output/row-stationary 多种数据流，LLM 场景重量复用故 weight-stationary 主流。使用方式：按 batch/复用率决定放 SRAM（batched FC）还是 DRAM（attention）；按宏形状与带宽匹配选择拼接配置；GQA 的 K/V head 共享使 attention 也出现权重复用机会，QK^T 是否驻留 SRAM 依 TP/seqlen 判断。

HybridSpec 补充视角（ISCA'26，HB 逻辑块内的 GEMM/attention 数据流）：因 decode 激活尺寸小，HybridSpec 采用 weight-stationary GEMM 与 KV-cache stationary attention——Opr1×Opr2=Res 模式、Opr2 驻留（权重从 HB DRAM 直接映射上计算阵列、KV cache 驻留），Opr1（激活）从缓冲读出广播到各 array element，element 用 adder tree 累加部分和、块级缓冲聚合（数据流沿用既往加速器 [7][36][52][61]）。与 SRAM-PIM 宏的差别：Opr2 驻留在逻辑 die 的分布式 weight SRAM（512KB/block）而非 CIM 宏内，MAC 阵列 80×64 FP16/BF16；目的同样是最大化权重/KV 复用、最小化 HB DRAM 读流量。

从硬件架构角度拆解（一次 decode GEMV）：draft token 激活 x∈R^d 读入 block 激活缓冲 → 广播到 80×64 MAC 阵列 → 每 element 用驻留权重 tile 与广播输入乘加 → adder tree 累加部分和 → 块级缓冲聚合输出；attention 时 KV tile 驻留、Q 广播。块间再配 tile 化 TP 的 ring 通信（见 kernel 层 TP 条目）。

实现与使用：数据流选择由激活/权重相对尺寸决定——decode 激活小故权重（KV）驻留最省带宽；实现要点是 Opr2 的 DRAM→阵列映射与 Opr1 广播网络 + 部分和聚合树；评估经 silicon-derived 参数注入事件驱动模拟器。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
