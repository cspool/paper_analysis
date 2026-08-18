## Key switching（密钥切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Key switching 是把密文从一把秘密钥的密钥域转换到另一把密钥域的 FHE 操作：给定旧钥 s 加密的密文 c，用 key-switching key（新钥对旧钥的加密）做一次密文域线性变换，输出新钥 s' 加密的同一消息。TFHE 自举收尾必用：盲旋转使密文落到大维度密钥域，须切回原 n 维 TLWE；在 BGV/BFV/CKKS 中其特例 relinearization 把乘法后的 (1,s,s²) 密文拉回 (1,s)，旋转（Galois 自同构）也需要它。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算形式 = 密文向量 x key-switching key 矩阵（向量-矩阵乘）：
```
# c = (a, b) 在密钥 s'（维度 kN+1），目标密钥 s（维度 n+1）
a_dec = GadgetDecomp(a)          # L 层分解
a_new = K_a * a_dec              # 矩阵乘，输出 n 维
b_new = b + K_b * a_dec
return (a_new, b_new)
```
- Annotations：本论文 key-switching 模板集成向量单元 + 累加器，PoK = 向量单元并行度；放置于 bootstrapping 下游（TFHE 协议要求自举后切换回原密钥域）；分解层数 L 越大噪声越小但计算与密钥存储越大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 库实现：TFHE-rs/OpenFHE 的 keyswitch 函数（Gadget 分解 + 矩阵乘）；硬件实现为向量 MAC 阵列（本论文模板）。优化：延迟 relinearization（加法无需先切回，编译器层面推迟）、fused key-switching（合并乘-旋转序列，Transformer 隐私推理场景）、HEIR 的 optimize-relinearization ILP pass。AutoFHE 把 key-switching 作为 CPE 模板之一，与 arithmetic+bootstrapping 组成统一 CPE lane。
- FlashTFHE 补充视角（ISCA'26）：PBS 中 key-switching 约占运行时间 10%，为第二大耗时步骤，主要作用是把 LWE 维数从长切短（如 ~30000 降到 ~1000），从而减少盲旋转的串行迭代数。硬件实现在 LPU（LWE Processing Unit）中：8 个独立可寻址、可时钟门控的 lane，每 lane 处理 32 个并行 64-bit 值（匹配 2^64 torus modulus），含向量加/乘单元、decomposer 与 rotator；lane partitioning 使小向量 KS 与 native LWE 运算不需要占满整条 8-lane 流水线。FlashTFHE 采用 key-switching-first 执行顺序，使 KS 结果可被多个后续盲旋转复用（KS-dedup，最多省 47.12% KS 操作）——这是 Boolean TFHE 的 blind-rotation-first 顺序做不到的。
- HE² 补充视角（ISCA'26，CKKS 方案）：CKKS 中 keyswitch 占据约 80% 计算量，是所有乘与旋转共用的核心原语。其数据流为交替的 ComOps/MemOps 序列 ModUp→IP→ModDown：密文在模数 Q 下分解为 dnum 组，经 ModUp 提升到 PQ·dnum 域、与 evk 多项式做内积（IP）、再经 ModDown 降回 Q，结果加回原密文。keyswitch 的中间密文（ModUp 输出与 IP 结果）尺寸大（单次传输最高 144 MB 量级），在异构加速器中若走 IRF 数据流（IP 放近存 xMU）则这些传输落在关键路径上——这是 HE² 的核心优化对象（HERO DFG 优化降通信频率 + 双级流水 xPU 隐藏通信延迟，见本库"xPU-xMU 异构架构""EVF/IRF 数据流"与编译框架层"HERO"条目）。

CASCADE 补充视角（ISCA'26）：CASCADE 把 key-switching 与 sample extraction、同态加法、标量乘等轻量操作交给集成在 HC0 的 VPU（Vector Processing Unit）执行——因为这些操作只占 BSP 计算的一小部分（相对 n 次 HMUX 的盲旋转），VPU 用并行乘法器/加法器 + 局部 buffer 实现、与 HMUX 流水并行工作，避免打断高吞吐的 HMUX 流水线。BSK 之外，HC0 额外 0.5 MB SRAM（12 MB vs 普通 HC 11.5 MB）存储 KSK。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
