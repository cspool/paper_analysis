## Dfloat（Dynamic Floating-Point Representation，动态浮点表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dfloat 是 NASZIP 的位级压缩技术：每个特征用可配置位宽 1+n_exp+n_man（符号位+自适应指数位+尾数位，∈[12,32] bit）的动态浮点表示，不同特征段采用不同位宽，使每个 DRAM burst（DDR5 每 device 128 bit）装入更多特征、减少访问向量所需的 burst 数。与传统 BF16/FP16/FP8 均匀量化的关键区别：FEE-sPCA 变换后各维贡献不均（低维承载更多信息），Dfloat 按段适配位宽、对敏感的低维段保留更多位（如 SIFT 128 维三段 18/14/16 bit），保持 recall 的同时最大化压缩。通用背景：动态浮点/自适应浮点（自适应指数/尾数位宽）在神经网络训练/量化中有成熟先例（Schrödinger's FP MLSys'24、AFP ICCV'21、"Be Like Water" ICML'22 等，Web 证据），本论文把该思路与 NDP 突发访问对齐并做设计空间搜索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dfloat 配置搜索（Algorithm 1）：对可能的每向量 burst 数 N_burst ∈ [N_min=d/(B_burst/32), N_max=d/(B_burst/12)] 做二分 + 枚举验证，目标是 min N_burst 且 recall@k≥R_target；验证用 host CPU 位掩码模拟各配置精度损失（无需重建索引）。约束：同一 burst 内特征同格式；位宽随特征索引增大而递减；N_burst 须是每 sub-channel device 数的倍数（device 同步工作）；B_burst 依 DDR 代际（DDR5=128bit、DDR4=64bit）。伪代码：
```
N_burst_min ← d/(B_burst/32); N_burst_max ← d/(B_burst/12)
while N_burst_min < N_burst_max:
  N_burst ← ⌊(N_burst_min+N_burst_max)/2⌋
  C ← cfg-validate(N_burst)          # 枚举所有满足约束的 {n_exp,n_man} 分段配置
  for C_i in C: if R(C_i) ≥ R_target and R(C_i) > R(C_opt): C_opt ← C_i; N_burst_min ← N_burst
  else: N_burst_max ← N_burst
return C_opt
```
张量示例（SIFT 128 维，图 11）：段 1~42 / 43~74 / 75~128 分别 18/14/16 bit → 每段 6/4/6 个 burst，4 个 device 交叉并行，每个 burst 一次取 4 device×128bit。Dfloat 值进 FPU 前零填充回 FP32，因此不改变计算单元（可移植性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Dfloat 打包在离线预处理阶段完成，与具体浮点格式无关、可套在现有 FP 表示上；硬件解码由 VPE 的 Dfloat 处理模块完成（16-to-1 MUX 逐 cycle 装 burst 进 128-bit 寄存器 → barrel shifter 按偏移寄存器抽 n-bit 元素 → 零填充 FP32）。ECC 兼容性：不改 DDR5 die 结构，on-die ECC 与 side-band ECC、内存控制器 ECC 均不受影响。开源仓库提供 FEE-sPCA 与 Dfloat 算法源码及配置搜索脚本。适用：需在固定 DRAM 带宽下检索海量向量的场景（BigANN 1B 等），与 FEE-sPCA 正交叠加（Dfloat 额外 1.79× 距离延迟削减）。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
