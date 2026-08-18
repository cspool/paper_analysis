## SRAM-PIM（SRAM 存内计算 / CIM 宏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SRAM-PIM/CIM 在 SRAM 阵列内完成乘加（数字域加法树或模拟域电流叠加），一个宏既是存储也是矩阵乘法单元。CompAir 采用 ISSCC'23 的 28nm 64kb 数字域浮点 CIM 宏（Guo 等）：64kb/array、BF16、t_access=6.8–14.1ns、14.4–31.6 TOPS/W（0.9–0.6V 低压下能效更高）、单个 8KB 宏面积 0.136mm²。优势：亚 10ns 矩阵延迟、>30 TFLOPS/W 能效（多数 NPU <5 TFLOPS/W）。劣势：宏容量小（KB 级），权重驻留模式下堆宏导致面积/功耗爆炸——CompAir 测算纯 SRAM-PIM 跑 GPT3-175B 的 FC 层需要不可行的宏数量、功耗超 A100 三个数量级。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 中 SRAM-PIM = 效率组件（compute-bound GeMM）：每 bank 逻辑 die 上 4 个 8KB 宏，每宏是 128 输入 8 输出 BF16 矩阵单元，可拼成 (512,8)（输入维扩展）或 (256,16)（输入/输出各分）配置。执行流程（batched FC）：权重写入宏并跨 batch 驻留 → 每 token 输入向量经 HB 从 DRAM 载入 → 宏内矩阵乘 → 结果写回 DRAM；DRAM 只承担输入/输出搬运，batch 越大权重复用越高（batch=32 时 Q/K/V 比 DRAM-PIM 快 6.3×，batch=1 无收益）。瓶颈：DRAM→SRAM 带宽（bank 读带宽 32GB/s + HB 6.4Gbps），宏形状存在分歧点——之前性能受输入带宽主导（与电压无关）、之后受宏延迟主导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：数字域 CIM（精度无损、加法树占面积）或模拟域（面积小、受噪声/工艺偏差影响）；数据流以 weight-stationary 为主（见本条目相关"权重驻留"）。落地：逻辑 die 上放宏阵列 + HB 与 DRAM die 堆叠（CompAir；H2-LLM 等 hybrid-bonding 系工作）。使用方式：只给 compute-bound、有 batch 权重复用的投影/FFN 层；input-dependent 的 attention 矩阵（K^T/V）避免使用；GQA 下 K/V 有 head 间复用才考虑。注意 K/V 每次推理变化使驻留失效。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
