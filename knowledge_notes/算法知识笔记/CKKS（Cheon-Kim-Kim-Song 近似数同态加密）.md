## CKKS（Cheon-Kim-Kim-Song 近似数同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CKKS（Cheon, Kim, Kim, Song，ASIACRYPT 2017）是面向近似实数的 Leveled 全同态加密方案，专为加密数值计算（尤其是神经网络推理）设计：它支持固定点数/浮点数编码、SIMD 风格的向量化并行、以及加/乘/旋转等近似运算，解密结果与明文计算相比仅含可容忍的近似误差。其核心机制：明文消息 m 被编码为次数 N 的多项式（环 R=Z[X]/(X^N+1)），密文是环上的两个多项式 (c_0, c_1)（外加一个 a），满足 c_0 + c_1·s ≈ m（s 为秘密钥）；实际数值被放入多项式的"槽（slot）"中，N 次多项式可承载 N/2 个复数槽，全部槽上的运算并行执行（SIMD）。
- 关键参数与符号（本论文 Table I）：N=多项式次数（coefficient 数），N/2=可用槽数；Δ（scale factor，本论文用 Δ=2^40 即 40 位）用于把实数定标为整数编码并控制精度；模数链 {q_0,...,q_L} 构成 level 预算。乘法后噪声与规模增长，需 Rescale（除以 2^Δ、截断模数）管理噪声、消耗一个 level；level 耗尽前必须 Bootstrapping 刷新噪声（深层 CNN 如 SqueezeNet/ResNet18/MobileNet 需要）。安全基于 Ring-LWE 困难问题，本论文所有模型保证 λ≥128 bits。
- 本论文中 CKKS 是 FEnc² 的底层方案：RNS-CKKS 实现（GPU 端用 Liberate-FHE），输入图像在客户端编码为多项式并加密，服务端在密文域执行全部 CNN 电路。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 CKKS 编码-加密-运算-解密 pipeline（以单张图像的一个卷积输出为例）：
```
1) encode:  m = [x_0, x_1, ..., x_{N/2-1}]   # N/2 个实数槽（本论文打包 4D 块元素）
            m_poly = CanonicalEmbedding(m) * Δ   # 定标到整数
2) encrypt: ct = (c_0, c_1) = (a·s + m_poly + e, -a)   # a 随机多项式，e 小噪声
3) compute: ct_out = PMult(ct, plaintext_weight)  # 密文×明文权重（卷积核）
            ct_out = Add(ct_out, ...)             # 累加
            ct_out = Rot(ct_out, k)               # 槽循环移位（对齐聚合）
            ct_out = Rescale(ct_out, Δ)           # 每次乘法后截位、耗 1 level
4) decrypt: m_out = (c_0 + c_1·s) / Δ             # 近似恢复结果
```
- Annotations：第 1 步的槽布局（哪些标量放哪个槽）即"密文打包"问题，决定后续旋转次数与槽利用率；Rot 是本论文主要优化目标（4.8ms vs PMult 0.15ms，Fig.1(a)）；Rescale 消耗 level，深层网络需在耗尽前插入 Bootstrapping（NEXUS GPU 自举每次耗 14 个 level）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件库：SEAL（微软，CKKS 原生实现）、HElib、OpenFHE、TenSEAL；GPU 库：Liberate-FHE（本论文 GPU 后端，纯 Python+CUDA 多 GPU，RNS-CKKS，BSD-3-Clause-Clear，现已弃用、继任 DESILO FHE https://fhe.desilo.dev/）、HEonGPU、TensorFHE、Cheddar。使用流程：设置参数（logN、logQ、scale、安全级别）→ 生成密钥（sk/pk/evk 旋转与重线性化密钥）→ 客户端编码加密 → 服务端同态计算 → 客户端解密。典型场景：加密 CNN/Transformer 推理、加密矩阵乘法、隐私梯度聚合。
- HE² 补充视角（ISCA'26，面向加速器的算子分类）：HE² 按算术强度（AI，ops/byte，SHARP 参数下）把 CKKS 算子分为两类：ComOps（计算密集型）= NTT（0.89）、BConv（1.60）、ModUp（3.38）、ModDown（2.92），复杂计算模式、由定制 ASIC 模块（xPU）加速；MemOps（内存密集型）= IP（0.12）、PMul（0.09）、CAdd（0.07）、Rescale（0.11），内存足迹大、由近存模块（xMU）加速。该分类是"ASIC-NMP 异构加速"的硬件分工依据（参数 N=2^16、L=35、L_eff=8、k=12、α=12、dnum=3、λ=128-bit，见本库"xPU-xMU 异构架构"与"ModUp/ModDown"条目）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
