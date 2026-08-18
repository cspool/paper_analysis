## Bootstrapping（自举，含可编程自举 PBS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bootstrapping 是 FHE 中同态执行自身解密电路的过程：对噪声接近阈值、无法再运算的密文 c，利用 bootstrapping key 中加密的秘密钥位做一次"密文域的解密-再加密"，输出噪声重置到低水平、内容不变的新密文。它是所有已知 FHE 方案实现任意深度计算的前提，也是性能大头（FHE 开销常占 50–90%）。TFHE 的可编程自举 PBS 更进一层：把一元函数 f 编码进 test vector，噪声刷新的同时同态求值 f——相当于免费 LUT 查表。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 自举主体 = n 次串行迭代的密文处理（n 为 TFHE 参数），每次迭代含：多项式缩放 → external product → FFT/IFFT → 分解。本论文 bootstrapping 模板的模块化参数即这一算法结构：PoV（多项式缩放向量宽度）、PoD（分解并行 lane）、BFU/IBFU（FFT/IFFT butterfly 数）、PoE（external product MAC 数）。
```
for i in 0..n-1:                          # n 次串行迭代
    decomposed = GadgetDecomp(BK_i)       # 分解
    prod = ExternalProduct(ACC, decomposed)   # FFT/IFFT 加速的多项式乘
    ACC = ACC + X^(a_i) * prod
```
- Annotations：n 次迭代决定自举延迟；FFT 次数 = n（bootstrapping unrolling 可降到 n/r）；放置策略——TFHE 协议规定 arithmetic 下游实例化 bootstrapping 硬件单元刷新噪声（AutoFHE 自动实例化；算法级"何时自举"由 FHE 算法设计者决定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs 的 programmable_bootstrapping API；GPU 用批量自举摊薄密钥带宽（BOLT-FHE 等）。硬件：MATCHA 的 38-bit DVQTF 整数近似 FFT（自举后噪声被刷新故可容忍近似）、FPT 的单 CMUX 流水 PE、Strix CPE 内 FFT/IFFT + vector MAC。加速方向：自举展开、多值自举（MOSFHET）、NTT 替代浮点 FFT。在 FEnc² 中，深层 CNN（SqueezeNet/ResNet18/MobileNet）推理需要自举，采用 GPU 优化的 NEXUS bootstrapping（每次消耗 14 个密文 level）；FEnc² 通过减少卷积层旋转来压缩自举之外的 HE 负载，即使自举开销不可避免（ResNet18/MobileNet 分别 +191s/+105s）仍显著加速。

FlashTFHE 补充视角（ISCA'26，multi-bit 参数下的 PBS 分解与执行顺序）：PBS 运行时间分解 = key-switching ~10%（第二大耗时）→ modulus-switching <1% → blind rotation ~90%（n 次串行外部乘积迭代）→ sample extraction <1%。执行顺序两种等价选择：key-switching-first（FlashTFHE 采用）先 KS 后盲旋转，使 KS 结果可在 fanout 结构（多个 LUT 作用同一 ciphertext）中跨多个盲旋转复用（KS-dedup，最多省 47.12% KS 操作）；blind-rotation-first 单次 PBS 计算量相同但无此复用。位宽越宽 PBS 频率越低：8-bit 相对 4-bit 模拟，DNN 类 workload 原生运算占 ~99.5%、PBS 数量级下降带来 6.8–8.1× 加速；非 DNN（XGBoost/DecisionTree/KNN）PBS 占比更高，8-bit LUT 需 ≥32 个 4-bit LUT 模拟（信息论下界），LUT 模拟成为瓶颈。硬件实现：FlashTFHE 单 ciphertext bootstrapping 延迟 6.16–34.67ms（高 bit-width 参数集）、CNN-20/50 单 batch 0.28/0.85ms。

MNEMOS 补充视角（ISCA'26，GPU 端 PBS 内存优化）：PBS 四阶段为 Modulus Switching → Blind Rotation → Sample Extraction → Key Switching（Algorithm 1）；MNEMOS 的剖析量化了各阶段的 GPU 开销——盲旋转（含 Decompose+FFT+MAC+IFFT 的 n 次迭代）主导执行时间，其中 MAC 阶段因需取 (k+1) 倍于 GLWE 体积的 BSK 而成为 memory-bound 瓶颈，stall_long_scoreboard >50% + stall_MIO_throttle >60%。优化后（BSK 分块复用 + Tensor Core FFT + 跨迭代融合）stall_long_scoreboard 降到约 20%，PBS 吞吐在 128-bit 参数下最高 3.01×（A100，Para-D）/2.86×（H100），应用端到端平均 1.96×（最高 2.23×，batch 4096 的 VGG-9 达 2.21×）。正确性前提：内部 FFT 需 FP64（≥30 小数位），与 FPT 的固定点位宽分析（[36]）互为印证。

CASCADE 补充视角（ISCA'26，自举的跨 HMUX 流水线化）：CASCADE 的自举（Algorithm 1）把 BSP 拆为 c←(2N/q)·c_in → ACC←X^(-b)·c_T → n 次 HMUX 迭代（Line 5：BSK_i←(X^(-a_i)−1)·BSK_i；Line 6：ACC_i←BSK_i⊡ACC_{i-1}，即外积）→ SampleExtract → key-switching（KSK 标量乘）。自举的 n 次 HMUX 是主要瓶颈：串行执行吞吐 ≈1/(n·t_HMUX)，流水线并行后稳态吞吐 ≈1/t_HMUX（最高 n× 提升），但每个 HMUX 需要访问唯一的 BSK（GGSW 矩阵），流水线化使并发 BSK 访问成为内存带宽瓶颈（算术强度低于 A100 平衡点 440 GOPS/s）。CASCADE 通过把全部 BSK 驻留在分布式 SRAM（BSK-distributed）消除片外 BSK 搬运，并把 n 个 HMUX 用 Interleaved-Fusion 策略融合/交错映射到 12 个 HMUX Chiplet（HC）执行；每个 HC 的流水线完成一个 HMUX 的时延 ≈ 最长流水级，BSP 时延参数集 I/II/III 为 0.01/0.02/0.04 ms。消融显示：Monolithic（单 chiplet+HBM3 串行）→ 细粒度流水架构 13.2× → 加 OIFS 调度再 4.1×，共 53.5×。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
