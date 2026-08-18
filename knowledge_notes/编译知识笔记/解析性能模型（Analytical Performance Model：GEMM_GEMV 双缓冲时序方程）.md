## 解析性能模型（Analytical Performance Model：GEMM/GEMV 双缓冲时序方程）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 解析性能模型是用闭合形式方程在运行详细模拟之前估计硬件执行时间的抽象层，用于快速扫设计空间。SHyLA 把每个 MCM chiplet 抽象为"统一 MAC 阵列 + 三个统一缓冲（input/weight/output）"的计算 die，配以容量与有效读写带宽（硬件带宽 × 利用率）描述的混合内存；把 LLM workload 表示为 GEMM/GEMV 序列，用双缓冲把 Weight/KVCache 加载与计算重叠（Fig. 8），忽略尾部计算（仿真验证未掩蔽 stall 占关键路径 <20%）。片内执行时间（GEMM，IA 在 DRAM、Weight 在 NVM）：
$$T_{intra} = \lceil I/B_I \rceil \cdot [ \frac{B_I \cdot J \cdot IA_bw}{DRAM_BW \cdot util_{DRAM_RD}} + \lceil K/B_K \rceil \cdot ( \frac{J \cdot B_K \cdot Weight_bw}{NVM_BW \cdot util_{NVM_RD}} + \frac{B_I \cdot B_K \cdot IA_bw}{DRAM_BW \cdot util_{DRAM_WR}} ) ]$$
其中 $B_I/B_K$ 为 tiling 因子（片上 buffer 容量决定），$bw$ 为各数据类别字节宽，$util$ 为带宽利用率（仿真标定：DRAM 读/写 90%、NVM 读 70%、NVM 写 10%）。片间通信（Dense 每 Transformer block 2 次 tensor 并行通信，Attention Output 与 FFN2 输出处）：
$$T_{inter} = \frac{2 \cdot I \cdot d \cdot ACC_bw \cdot (1-1/p_t)}{ICNT_BW} + \frac{2 \cdot I \cdot d \cdot IA_bw \cdot (1-1/p_t)}{ICNT_BW}$$
端到端延迟 = 跨所有推理迭代累加片内+片间时间。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 解析模型扮演"离线 cost model"角色：它支撑两阶段 DSE 的 Stage 1 对混合内存空间穷举（每次配置评估 = 代入 T_intra/T_inter 方程而非跑 GPGPU-Sim），以及 Stage 2 进化算法的每个个体适应度评估（系统吞吐/每用户吞吐）。运转流程：输入配置（面积比 → NVM/DRAM 带宽-容量点、部署参数 pp/pt/b）→ 由模型配置表（Table IV：tile 数 20、weight buffer/tile 1.2MB、ICNT_BW 429GB/s 等）代入方程 → 输出系统吞吐估计 → 喂给 DSE 选优。与 GPGPU-Sim 的关系：解析模型负责设计空间扫描，GPGPU-Sim 对选中的代表性配置做 request 级校准（验证带宽利用率 90/70/10% 假设、stall 占比 <20%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为论文自研（未开源，联网未找到仓库），核心是带宽利用率为系数的分层访存时间累加：IA 行按 tile 分批从 DRAM 读（DRAM 读带宽），每批内 Weight 块从 NVM 读（NVM 读带宽）并与 IA 写回（DRAM 写带宽）并行/流水；tiling 因子 B_I/B_K 由 weight buffer/tile（1.2MB）容量约束。这种"带宽利用率 × 硬件带宽"的有效带宽抽象与 Roofline 模型一脉相承，区别在于显式建模混合内存的读写非对称（NVM 写 10% 利用率）与双缓冲重叠。SHyLA 用其把 DSE 总时间压到约 12 分钟（96 核 Xeon Gold 5418Y），并以 GPGPU-Sim 交叉验证。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
