## Slot Rotation（槽旋转）与 Key-Switching（密钥切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 槽旋转 Rot(ct,k) 是 CKKS 的 SIMD 原语：把密文 N/2 个槽的内容循环移位 k 位（等价于对消息多项式做自同态 X→X^k），用于对齐数据做聚合（如卷积的通道/空间累加）。旋转实现 = 自同态（automorphism） + 密钥切换（key-switching）两步：自同态把系数映射 X^i→X^{ik}，得到的是用"旋转后密钥"加密的密文，必须用旋转求值密钥 evk_rot^k 做 key-switching 换回原密钥才能继续运算；keyswitch 占据旋转延迟主体（本论文式 (1)：Rot(ct,k)=(c(X^{ik}),0)+P^{-1}(a(X^{ik})·evk_rot^k)）。
- 性能：旋转 + CMult 远贵于 PMult/Add（Fig.1(a)：4.8ms vs 0.15ms），每次旋转含多次 NTT/iNTT 与大规模向量 shuffle，应用级端到端延迟约 70% 来自旋转（Fig.1(b)）。因此"减少旋转次数"是 HE-CNN 打包优化的首要目标。
- 两类旋转（本论文定义）：内旋转（inner rotation）= 空间聚合，每输入密文需 (K²−1) 次生成 K×K 卷积的移位副本；外旋转（outer rotation）= 通道聚合，每个输出密文需 (α−1) 次（α=每密文打包通道数）对齐通道。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次多通道卷积的旋转流水（本论文 Fig.2，4 输入/输出通道、3×3 核、BSGS）：
```
# 内旋转：生成 K²=9 个移位副本做空间 MAC
for i,j in 0..K-1:
    X_shifted = Rot(ct_X, offset(i,j))        # 内旋转 ×(K²-1)
    acc += PMult(X_shifted, plaintext_w[i,j]) # 与明文核相乘
# 外旋转：跨 α 个通道对齐并累加
for c in 1..α-1:
    acc += Rot(acc_c, channel_offset(c))      # 外旋转 ×(α-1)
```
- Annotations：旋转次数被 FEnc² 的块大小 S 控制：内旋转复杂度在 K>S 时为 (⌈K/S⌉²−1)/密文、K≤S<M 时为 4(S−1)/S²；外旋转为 N_out/α×(αS²/BS−1)；最优 S* = ⌈(K²N_in/(αN_out))^(1/4)⌉（Theorem 1，式 (8)）。Conv-aware Encoding 使旋转复杂度从 O(K²)（乃至 Hyena+ 的 O(K⁴)）降到 O(K)（Table III）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：任何 CKKS 库（SEAL/OpenFHE/Liberate-FHE/TenSEAL）的 rotate 接口 + 预先生成的旋转密钥（每偏移一个 evk）。硬件/系统级：keyswitch 依赖 NTT/iNTT 在分解基间变换（本论文指出旋转是 NTT 单元的最大消耗者，减旋转即减 NTT 压力）；GPU 实现把旋转映射为 ciphertext 系数 shuffle + 多项式乘 kernel。使用场景：卷积/矩阵乘/FFT 在密文域的对齐聚合，以及密文内数据重排（如 FEnc² 的 rot-mask-add 重打包）。优化方向：预旋转副本复用、BSGS 拆分、减少旋转次数（本论文）、keyswitch 密钥复用（ARK）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
