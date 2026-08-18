## NTT（Number Theoretic Transform，数论变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NTT 是定义在有限域（素数模 q 上）的离散傅里叶变换（DFT）变体，用单位原根与模算术代替复指数，把多项式乘法（卷积）从 O(N²) 降到 O(N log N)，是 RLWE/Ring-LWE 类 FHE 方案（BGV/BFV/CKKS）中多项式算术的性能基石。iNTT（逆 NTT）把乘积域结果变回系数域。在 HE-CNN 推理中，NTT/iNTT 是乘法（CMult/PMult）、旋转（keyswitch 的 automorphism）、bootstrapping 等几乎所有原语的共同子操作。
- 本论文角色：NTT 是旋转（含 keyswitch）的 dominant 消耗者——Ring-LWE 方案的 keyswitch/automorphism 需多次正向与逆变换在分解基间切换（Fig.1(a)：旋转 4.8ms vs PMult 0.15ms）；每次 (i)NTT 调用伴随大量 twiddle factor 与密文系数的内存搬运，是 FHE 加速器内存带宽瓶颈的来源之一。FEnc² 通过减少旋转/keyswitch/NTT 的"数量"（相对 HELayers 最多降 94%）来降低 NTT 单元需求，属应用层优化与 NTT 硬件加速（ARK/Neo/TensorFHE 等）正交互补。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次长度为 N 的 NTT kernel（Cooley-Tukey 基 2，GPU 上按 stage 调度）：
```
input:  a[0..N-1] (系数域), w = primitive N-th root mod q
for stage s in 1..log2(N):
    len = 2^s; half = len/2
    for i in 0..N-1 step len:            # 每 block 一个 warp/thread block
        for j in 0..half-1:
            u = a[i+j]; v = a[i+j+half] * w^{j*N/len} (mod q)
            a[i+j] = u+v; a[i+j+half] = u-v
bit_reverse(a)
```
- Annotations：kernel 输出喂给 keyswitch（与旋转求值密钥做点积再逆 NTT）或 CMult；FEnc² 的 GPU 评测（Table VI）显示 kernel 调用从 48,015 降到 5,775（-88%）、GPU 内存传输从 12,021MB 降到 1,461MB（-87.8%），即 NTT kernel 的调用次数与随之而来的内存流量同时被压缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：SEAL/OpenFHE/Liberate-FHE/TenSEAL 内嵌的 NTT 实现（GPU 上做多多项式批量 NTT、radix-2/4/8 混合、双缓冲掩盖传输）；专用 GPU 库 TensorFHE/HE-Booster/Cheddar 用 tensor core 或 CUDA core 加速 NTT。硬件：ARK、Neo、FAST、EFFACT、CraterLake、F1 等加速器内置专用 NTT/iNTT 单元（多 lane、多基分解）。使用场景：任何 RLWE 多项式乘/keyswitch/自举，是 HE 加速器面积与功耗的主要构成；本论文的架构启示是"算法层面削减 NTT 需求后可缩减 NTT 簇规模、降低面积功耗并缓解内存带宽"。

GenZA 补充视角（ISCA'26，面向 ZKP 的 NTT 映射）：ZKP 的 NTT 大小可达 2^23（Groth16 29.32% 时间、Plonky2 多项式乘），stride 不规则重排（stage i 的 stride 2^{N-1-i}）是大 N 下 off-chip 访问的主要挑战。GenZA 的 NTT 映射：(1) 2D-NTT 分解（four-step NTT）——把数据当 2D、每维做 √N 大小（如 2^13）子-NTT 片内完成，维度间转置用全局 transpose buffer；(2) MDC（Multipath Delay Commutator）流水线——每逻辑级 = radix-2 蝴蝶 + FIFO 延迟缓冲（容量∝stride），沿 PE 行实例化，PE scratchpad 分区作 FIFO；(3) 折叠流水线 + scratchpad 借贷——L=13 长 MDC 管线映射到 2×8 PEs，SRAM 饥饿的首段向邻近空余 PE 借 FIFO 空间（借/贷 PE 距离 ≤2 hops、FIFO 访问下同时至多一对活跃、NoC 流量 ≤2× 前向数据），免去 LegoZK 的 3D 分解（3 次 off-chip 往返）与 UniZK 昂贵的专用 transpose buffer；2^23 NTT 流量 7.4→3.0 GB、PE 利用率 16%→38%、时间 27.1→11.4 ms；(4) 小 bitwidth（64-bit Goldilocks）整条 MDC 管线合并进单 PE（32 lanes 够两条 L=13 管线，只用内部 crossbar/forward chain，完全避免 NoC 压力）；(5) NoC 带宽分析：每蝴蝶 II≈6.75（256/384-bit）或 10.125（768-bit）PE 周期（KO 乘+Montgomery 归约），worst-case 256-bit 约 152 GB/s = per-hop 容量 30%（32×64-bit links @ 2 GHz）。NTT 算术强度低（每元素 1 模乘），本质访存受限，故融合/流水线（见"核间融合与流水线"条目）进一步消除中间数据传输。
- HyperDrive 补充视角（ISCA'26，GPU FP64-TCU 上的分层 NTT kernel）：把 NTT kernel 拆为 Inner-NTT（radix-64 基例，全片上）与 Outer-NTT（EWMult、Residual NTT、转置、GMEM 搬运）两级两 kernel（Kernel 1/2）。Inner-NTT 用 FP64 TCU 的 8×4×8 MMA（PTX m8n8k4）以 warp 粒度执行——每个 warp 用 4 次 MMA（MMA1/2 乘 twiddle factor matrix 的高/低 16-bit 分量、MMA3/4 完成第二级 radix-8）+ Bit-Merge + ModRed + EWMult 完成一个 64 点 NTT，寄存器内完成（TLMOP），零中间 SMEM；转置用 MMA 的 Fragment A/B/D 跨线程分布隐式完成（TransOP）。Outer-NTT 用 RowMaj（预转置布局，3/4 GMEM 访问变行主序全 coalesced）+ TFOP（negacyclic 顺序读、TF-XY 预排序、TF256/TFM 驻 SMEM）。效果（N=2^16、36 limbs）：SMEM stall -33.2%、occupancy 55.0%→76.8%（NTT+ 92.5%）、GMEM stall -59.5%、总延迟 -61.1%、吞吐 932.6 KOPS（2.0× WarpDrive、5.5× Neo，图 15-18）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
