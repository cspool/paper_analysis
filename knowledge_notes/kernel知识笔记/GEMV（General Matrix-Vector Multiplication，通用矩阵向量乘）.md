## GEMV（General Matrix-Vector Multiplication，通用矩阵向量乘）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMV 是矩阵×向量的线性代数原语：Y = A·x（A∈R^{M×K}，x∈R^K，Y∈R^M）。自回归 decode 每步只处理 1 个 token，attention（QK^T、SV）与 FFN 均为 GEMV 形态——算术强度极低、输出计算量 O(MK) 而每个输出只需读整行权重，典型 memory-bound，是"decode 内存墙"的直接来源。RAG 中 decode 更糟：query KV 每步新增但文档 KV（数千到上万 token）完全复用，GEMV 的 K 维被文档 KV 拉长、且 batch 小无法用权重批复用补救。MERIDIAN 把文档侧 GEMV 移到 PIM 就地执行（KV 静止），decode 每步只传 query 向量并回收紧凑统计量。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一次 decode 步的文档注意力 GEMV（单 head，d_model=3584，文档 14630 token）：
s_d = q[1,3584] @ K_d[14630,3584]^T      # GEMV：1×3584 @ 3584×14630，输出 14630
# MERIDIAN PU 执行（bank 内）：
#   K_d 按 head shard 静止在各 bank；q 经 CXL 广播到设备
#   每 bank 的 PU：16-lane FP16 乘加，256-bit/周期消费数据，All-Bank-Mode 并行
#   结果与 o_d/m_d/l_d 紧凑摘要经 NMU（channel 内归约）与 BOOMv2 核（跨设备）聚合
# 对比集中式：K_d（~96MB FP16）从 host DRAM 经 PCIe 搬上 GPU 再算 GEMV
```
decode 的 QK^T/SV 都随 query 变化、无 batch 复用，因此即便 H100 高算力也无法加速——roofline 上始终 memory-bound（MERIDIAN 图 3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上为高带宽受限 kernel（FlashInfer/vLLM 的 decode attention、cuBLAS 的 GEMV 路径；低比特变体如 BitBLAS INT2 GEMV kernel 用 Tensor Core）；PIM 上为 bank 级 MAC 阵列（HBM-PIM/AiM 的标准 GEMV 形态），MERIDIAN 进一步支持内存侧非线性与专用 softmax。使用场景：任何自回归 decode、RAG 的文档注意力（MERIDIAN 的 DocumentAttention 分支）、小 batch FFN。调度要点：GEMV 适合 PIM（数据就地、免搬运）；若批变大 FFN 可转为 GEMM（权重复用）、attention 仍 GEMV（KV query 相关）。

PuD 视角（ISCA'26，PuDGhost 论文）：GEMV 是 PuD 的主要目标负载（LLM 推理的 memory-bound 层）。PuDGhost 论文在真实 DDR4 上以 MAJ3 位串行实现 GEMV：8-bit 精度、4096×N（N∈{4,8,16,32,64}）随机矩阵，用 32768 列（4096×8 bits）通过筛选的列执行；MAJ3 用 8 行 SiMRA（3 操作数×2 冗余 + 常量 0/1 行）。PuDGhost 使长 MAJ3 链累积错误——Base-worst（screening 全 1、执行全 0 相邻行）NMSE 达 2.2×10⁻²（N=16）与 5.8×10⁻²（N=64）；配合 CS-1/CS-2 + 隔离行布局缓解后 NMSE 全维度 <10⁻³（相对 Base-worst 413× 降 @N=32、114× 降 @N=64，BER 1.3×10³× 降）。
  - SHyLA 补充：decode 阶段以 GEMV 为主（memory-bound，逐 token 用 KVCache + Weight 计算），是混合内存系统的带宽瓶颈。SHyLA 的 ATTN 层在 decode 用"GEMV 配对"：每个 tile 处理一对 fused ATTN 的 GEMV（Q·K^T 与 S·V 连续），中间 QK^T 结果留在片上不写 DRAM；KVCache 单 memory plane 放置以支持并行 GEMV（溢出跨两 plane，因 decode 占端到端运行时间主导、prefill 影响小）。tile 数 = 微批 b × 每 die attention head 数。

XtraMAC 补充视角（ISCA'26，FPGA 混合精度 GEMV kernel）：GEMV 是 FPGA 带宽受限负载——kernel 吞吐上限 ≈ HBM 带宽/权重字节数。XtraMAC 论文实现 tile 并行混合精度 GEMV kernel：M 个 tile 各映射到一个连独立 HBM channel 的 PE，权重存 HBM、激活片上缓冲；HBM 每 channel 512-bit 接口字按 per-lane 拆成权重段分发到 PE 内级联的 XtraMAC 链，per-tile datatype 控制信号与操作数同步传播，逐 lane 部分和经级联 MAC 链累加写回。每 channel 级联数 N_MAC = BitWidth_channel/(BitWidth_weight×P)：INT4 权重 + P=2 lanes 时单 channel 512/(4×2)=64 个 MAC 输入/cycle，32 HBM channel 理论 2048 个、实际 1920 个（30 活跃 channel，留 1 读激活 + 1 写回保证布线收敛）。结果：≤1024 实例维持 300 MHz、1920 实例 250–270 MHz（HBM 接口路由拥塞）；512-XtraMAC 占 LUT 98.5%/FF 95.6%/DSP 100%；U55c（460 GB/s）相对 CUTLASS H100（2 TB/s）GEMV 1.2× 低时延（0.0246 vs 0.0294 ms @4096²）、1.9× 能量效率（85 vs 135 W），FPGA 维持 ~74% 有效 HBM 利用率、接近带宽 roofline——靠 2× lane 打包（每 DSP 每 cycle 2 个 INT4 权重）与无格式转换开销抵消带宽差距。
涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
