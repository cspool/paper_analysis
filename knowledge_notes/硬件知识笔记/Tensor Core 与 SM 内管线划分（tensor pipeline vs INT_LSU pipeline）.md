## Tensor Core 与 SM 内管线划分（tensor pipeline vs INT/LSU pipeline）

术语解释
NVIDIA GPU 自 Volta 起的矩阵乘硬件单元；SM 分 4 个子分区，每分区有 warp scheduler + 多条执行管线（fma/alu/xu/LSU/mma），tensor core 走独立 MMA 管线，整数/访存指令走 ALU/LSU/MIO 管线，不同管线指令可由 warp scheduler 交错/同周期 co-issue。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tensor core 是混合精度矩阵乘加（MMA）硬件：A100 每 SM 4 个（每子分区 1 个），支持 mma.sync.m16n8k16（bf16/fp16/tf32/int8）；Hopper 引入异步 wgmma（直接从 shared memory 读操作数）+ TMA（Tensor Memory Accelerator，异步 global→shared 拷贝）；Blackwell 进一步解耦为 tcgen05/UM tensor engine。SM 微架构上，指令按管线归类：fma（FP32/INT32 乘加）、alu（整数/位操作）、xu（超越函数）、LSU（全局/局部/shared 访存，经 MIO 队列）、mma（tensor）——各管线有独立 issue 能力，warp scheduler 每周期每 sub-partition 发 1 条指令，可跨管线交错（如 FFMA 与 HMMA 交替发射）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文利用"管线物理分立"实现融合内核的重叠（Section V-D）：解码 warp 只发整数指令（rANS 状态更新、查表）+ LSU 指令（coalesced 补位 load、shared memory store），GEMM warp 只发 shared load + MMA；两 warp 的指令落在互不占用的管线上，warp scheduler 同周期 co-issue——解码与矩阵乘在 SM 内真正并行，而非时分复用。硬件约束同时决定 tile 几何与流水形态：A100 用 32×128 tile、H200 因更大 shared memory 用 64×256 tile（更大 tile → 元数据开销更低 0.015%–0.072%）；H200 更大片上容量使其重叠效果更好（最高 1.2× 超越 CUTLASS，A100 为 1.0–1.1× 内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编程接口：CUTLASS（Mma_Atom/TiledMma 抽象 wgmma/mma）、WMMA API、内联 PTX mma.sync/wgmma；分析工具：Nsight Compute 的 pipe utilization（wavefront-bound vs bandwidth-bound 区分）。使用：混合精度 GEMM、融合类 kernel（解压+GEMM、量化+GEMM、attention Flash 系列）的管线划分依据——把非 MMA 工作（解码、反量化、地址计算）派给整数/LSU 管线、与 tensor pipeline 重叠，是 GPU 融合 kernel 的通用优化范式。
- HyperDrive 补充视角（ISCA'26，FP64 Tensor Core 与 FHE NTT）：A100（sm_80）的 FP64 Tensor Core 支持 mma.sync.aligned.m8n8k4.f64（8×8×4 形状，每 TC 每周期 32 FP64 FLOPS = 16 FMA，IEEE 754 合规），sm_90+（Hopper/Blackwell）才支持 m16n8k4/m16n8k8/m16n8k16；论文以"8×4×8 MMA 维度"描述（即 m8n8k4），用它作为 radix-64 Inner-NTT 的基例：FP64 53-bit 尾数可直接承载 32-bit 模乘的 2 次乘法（轻量 MPA，见 kernel 层 MPA 条目），比 INT8 TCU 方案的 16 次子乘法大幅降 MPA 开销。FP64 MMA fragment 数据分布（每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 D 均分）被逆向分析用于寄存器级数据访问（TLMOP）与隐式转置（TransOP），使 64 点 Inner-NTT 全程零中间 SMEM。FP64 TCU 吞吐规格：A100 19.5、H100 67、B200 37、Rubin 200 TFLOPS（Table XI），HyperDrive 在 H100 上 NTT 吞吐 1669.5 KOPS 验证跨代可扩展。

- MNEMOS 补充视角（ISCA'26，FP64 Tensor Core 上的 TFHE FFT）：MNEMOS 是首个把 TFHE 的高精度 FFT 映射到 FP64 Tensor Core 的工作（WMMA m8n8k4.f64，A100 每 TC 每周期 32 FP64 FLOPS），并批判性检验 CKKS 方案的 Tensor Core NTT 映射（INT8/FP16）为何不能直接照搬：TFHE PBS 内部 FFT 的数值精度分析（噪声公式 n·2^ω·ℓ·2^(2β)·N²·(k+1)，对照 FPT 理论界）表明 4-bit 明文正确性需 ≥30 小数位（常 >35），而 FP32（24 尾数位）/FP16（11 尾数位）不足 → 必须用 FP64（53 尾数位）。硬件前提：FP64 吞吐仅数据中心旗舰不受阉割——A100/H100 为 FP32 的 1/2，消费级 GPU 为 1/64；FP64 Tensor Core 自 A100 起是旗舰标配，Blackwell（B200 37 TFLOPS FP64 TCU）延续，故该优化对数据中心 GPU 世代有效。实现：复数乘分解为 4 个实数 WMMA、8 点 FFT 基例（8×8×4 匹配原生形状）、四步 FFT radix-8/64 分层、64 点 fragment 布局免转置；性能上 PBS 吞吐 A100 最高 3.01×、H100 最高 2.86× vs ZAMA baseline。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
