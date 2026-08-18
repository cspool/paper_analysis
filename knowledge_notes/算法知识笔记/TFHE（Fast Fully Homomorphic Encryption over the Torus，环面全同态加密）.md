## TFHE（Fast Fully Homomorphic Encryption over the Torus，环面全同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TFHE（Chillotti, Gama, Georgieva, Izabachène, JoC 2020）是基于环面（torus）的 FHE 方案族，核心卖点：极快自举（单次 <0.1 s 量级）与可编程自举 PBS，能同态评估任意一元函数，是隐私推理与加密查找的主流选择（ZAMA 商业化部署）。密文结构三层：TLWE（环面 LWE，(a,b)，b=⟨a,s⟩+m+e，二元消息编码为 ±1/8 或 ±1/4）；TRLWE（环上多项式版本，模 X^N+1）；TRGSW（秘密钥的 gadget 加密，构成 bootstrapping key BK）。
- 本论文表 I 参数集：n（TLWE 维度 500–630）、N（TRLWE 多项式度 1024/2048）、L（分解层数 2–3）、k（TRGSW 层数 1），对应 80/110/128-bit 安全级别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TFHE 一次 bootstrapping = 三步（本论文 CPE 模板对应的算法结构）：
```
ACC = X^(-b) * test_vector(f)              # TRLWE 累加器初始化
for i in 0..n-1:                           # 盲旋转：每个 TLWE 密钥位一次 CMux
    ACC = CMux(BK_i, X^(a_i)*ACC, ACC)     # external product，主开销
c' = SampleExtract_0(ACC)                  # TRLWE -> TLWE（取常数项）
c  = KeySwitch(c', KSK)                    # 切回原 n 维密钥域
```
- Annotations：CMux 每步一次 external product（TRGSW x TRLWE），FFT/IFFT 把多项式乘从 O(N^2) 降到 O(N log N)，占自举延迟约 80%；PBS = 把一元函数 f 编码进 test vector 系数，自举同时完成查表求值（同态激活 ReLU/sign、加密表查找都靠它）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 库：原版 C++ TFHE、TFHEpp（纯 C++）、TFHE-rs（Zama 生产级）、cuFHE/nuFHE（GPU）。加速器：Strix（eNPU）、MATCHA（整数近似 FFT + bootstrapping-key unrolling）、FPT（单 CMUX 流式 PE）、OFHE（光电 FFT）。本论文 AutoFHE 以 TFHE 为代表性方案：arithmetic/bootstrapping/key-switching/HMUX 四类 CPE 模板即 TFHE 算子集的硬件参数化；使用场景：DeepCNN 隐私推理、加密 ALR（算术/逻辑/关系）、index 加密查找。

MNEMOS 补充视角（ISCA'26，GPU 内存访问优化框架）：MNEMOS 系统性剖析 ZAMA 的 TFHE-rs/Concrete GPU 实现，确认 TFHE 相比 CKKS 更适合 GPU 加速的控制密集型/bit 级/逻辑重负载（隐私保护量化神经网络要求精确而非近似计算），但代价是更高的计算与内存需求。其关键新发现：(1) PBS 在真实 TFHE workload 中严重 memory-bound——stall_long_scoreboard 超 50% 执行时间（加 stall_MIO_throttle 超 60%），根源是庞大的 bootstrapping key（BSK）作为"热数据"被多 SM 同时访问、仅获得 L2 级复用而无法驻留 SM；(2) 首次研究把高精度 TFHE FFT 映射到 FP64 Tensor Core（WMMA 8×8×4），并指出 CKKS 方案的 Tensor Core NTT 映射（低精度 INT8/FP16）因 TFHE 的 FP64 精度要求不能直接照搬——精度分析表明 4-bit 明文正确性需 ≥30 小数位（常 >35），FP32（24 尾数位）/FP16（11 尾数位）不足；(3) 噪声公式 n·2^ω·ℓ·2^(2β)·N²·(k+1)（ω≈2·(64−53)−2.6，64-bit 密文空间）显示尾数位宽影响呈指数级，N 为二次、n/ℓ/k 线性，因此 ZAMA 参数集正是在 FP64 舍入误差模型下联合优化的。参数集（Table II）：Para-A~D 由 Concrete 编译器为 CNN 生成（N=512/1024、k=2/4、ℓ=1/2/11、n=532~728、128-bit），Para-E 来自 tfhe-rs benchmark、Para-F 来自 Morphling，另有 Para-I/II 用于跨平台对比（80/110-bit）。

CASCADE 补充视角（ISCA'26，跨 HMUX 流水线并行的 TFHE 加速器）：CASCADE 把 TFHE 的瓶颈定位为 BSP 中 n 次串行 HMUX（盲旋转）迭代——n 为加密参数，为保 128-bit 安全需大 n（参数集 III：n=592、N=2048、L=3、k=1），CPU 实测 n 次 HMUX 占 BSP 执行时间 79%、BSK 搬运占总数据搬运 80%，且 n-HMUX 算术强度远低于 A100 平衡点（440 GOPS/s）。先前加速器（MATCHA/Strix/Morphling）全部串行执行 n 次 HMUX，吞吐受 Thp_seq ≈ 1/(n·t_HMUX) 硬约束；CASCADE 首次利用跨 HMUX 流水线并行（理论 Thp_pipe ≈ 1/t_HMUX，最高 n× 提升），但流水线并行要求 n 个 HMUX 并发访问各自的 BSK（GGSW 高阶多项式），产生集中式 HBM 无法支撑的带宽需求（Morphling 单 HBM stack 约 30W ≈ 加速器 die 功耗 56%）——CASCADE 用分布式 SRAM（BSK-distributed，126 MB）驻留全部 BSK 解决。四参数集（I/II/III/IV，80/110/128/128-bit）见表 I。评估：参数集 I/II/III 吞吐 2,133,624/1,235,248/416,408 BSP/s，Speedup/Area 相对 MATCHA/Strix/Morphling 30.5×/15.6×/3.1×。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
