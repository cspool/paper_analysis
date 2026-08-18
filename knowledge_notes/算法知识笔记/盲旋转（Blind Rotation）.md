## 盲旋转（Blind Rotation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 盲旋转是 TFHE 可编程自举（PBS）中最核心、最耗时的步骤（约占 PBS 运行时间 90%）：根据输入 LWE 密文的整数化分量（modulus-switching 到 Z 后），把 GLWE 密文（编码 test vector/LUT）整体旋转对应步数，从而"盲"地（在密文域、不泄露明文）完成一次多项式位移，等价于对 LWE 的每个分量做一次多项式幂乘 X^{a_i}·prod 并累加。它由 n 次串行迭代的外部乘积组成（n 为 LWE 维数，key-switching 后通常从 ~30000 降到 ~1000），每次迭代 = gadget 分解 + 多项式乘（FFT/IFFT）+ 累加。旋转完成后，test vector 中由明文决定的项被移到常数项位置，再经 sample extraction 取回 LWE 密文，同时噪声被刷新到低水平——这就是 PBS 既能求 LUT 又能刷新噪声的原因。
- 在 multi-bit TFHE 中盲旋转的输入/输出都是更大参数（N 至 2^16、l_b 8–10），因此它主导了整个加速器设计：外部乘积吞吐、FFT 单元、BSK 流式访存都围绕它优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PBS 四步（论文 Figure 3，key-switching-first 顺序）：
```
PBS(c_lwe, f):                      # c_lwe: n 维 LWE，f: LUT
  (A) c' = KeySwitch(c_lwe)         # KSK 把 n 从长切短（~30000→~1000），~10% 时间
  (B) mu = ModSwitch(c')            # torus → 整数，<1%
  (C) ACC = BlindRotation(mu, test_vector_f)
      for i in 0..n'-1:             # n' 次串行外部乘积迭代
          ACC = X^{mu[i]} * (GadgetDecomp(BSK_i) □ ACC)   # FFT/IFFT 多项式乘
      # 占 ~90% 时间
  (D) out = SampleExtract(ACC)      # 取 GLWE 常数项回 LWE，<1%
```
- Annotations：盲旋转的迭代次数 = key-switching 后的维数 n'；每迭代的算力 = (k+1)·l_b 个多项式乘；FlashTFHE 把 C 步骤的数据通路设计成外层循环扫 BSK chunk、内层 round-robin 扫 ciphertext（时间域复用），与空间架构"BSK 单遍流经 PE 阵列"相反。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs/Concrete 的 blind rotation 用 FFT 加速多项式乘；硬件：加速器把盲旋转映射到 FFT 流水线 + VecMAC。FlashTFHE 的 BRU 内：FFT-A（256-point）/FFT-B（128-point）混合基双实数 FFT 集群 + VecMAC（512 coef/cycle）+ 9.2MB 累加 buffer + 共享 I-FFT（FFT:I-FFT 操作数比 l_b:1，两个 BRU 共用一个 I-FFT）。执行顺序选择：key-switching-first（FlashTFHE 采用）允许 key-switching 结果跨多个后续盲旋转复用（KS-dedup），而 blind-rotation-first 单次 PBS 计算量相同但无此复用机会。

MNEMOS 补充视角（ISCA'26，GPU 端盲旋转的迭代结构与跨迭代融合）：MNEMOS 的 Algorithm 1 逐迭代结构为 Rotation（ACC_rotate = X^{ã_i}·ACC − ACC）→ Decompose（按基 β 位切片成 ℓ 组）→ FFT（Tangent FFT，N/2 点复数 FFT）→ MAC（ACC_fourier ⊙ BSK）→ IFFT + 累加，共 n 次迭代。GPU 剖析显示 BSK 访问导致的 stall_long_scoreboard 是盲旋转最大瓶颈（超 50%）。MNEMOS 两个针对性设计：(1) MAC 阶段 BSK 分块复用（见"LWE / GLWE / GGSW 密文与外部乘积"条目补充）；(2) 跨迭代 kernel 融合——由于 FFT 与 IFFT 使用同一组 twiddle factors / precomputation factors 的共轭版本，把迭代 i 的尾部与迭代 i+1 的头部（即 IFFT 后接下一次 FFT）融合为单个 kernel，两套系数在片上跨迭代复用、消除主循环内对这些系数的冗余全局载入，收益随分解层数 ℓ 增大（ℓ 越大融合窗口越宽）。

CASCADE 补充视角（ISCA'26，盲旋转 = n 次 HMUX 的流水线化）：CASCADE 论文把盲旋转明确表述为 Algorithm 1 第 4-6 行的 n 次 HMUX 迭代（与 FlashTFHE/MNEMOS 的"n 次外部乘积迭代"是同一结构）。CASCADE 的洞察是：这 n 次迭代并非不可流水——把 HMUX_i 的 ACC 输出在算完多项式系数后立即流向 HMUX_{i+1}（inter-HC 系数粒度流水），稳态吞吐可从 1/(n·t_HMUX) 提升到 1/t_HMUX。但盲旋转流水化引出两个此前被忽视的硬件挑战：(1) n 个 HMUX 并发访问各自 BSK 造成极端带宽压力（集中式 HBM 无法支撑）；(2) 每个 HMUX 依赖前一 HMUX 输出的中间密文（ICT），跨 chiplet 的 ICT 传输造成 D2D 通信瓶颈（D2D 时延 > HMUX 计算时间时 HC 严重欠利用）。CASCADE 分别用 BSK-distributed 分布式 SRAM 与 Interleaved-Fusion（融合+交错）策略 + OIFS 调度解决。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
