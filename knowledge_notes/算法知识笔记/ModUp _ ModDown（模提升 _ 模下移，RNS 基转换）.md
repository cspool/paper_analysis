## ModUp / ModDown（模提升 / 模下移，RNS 基转换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ModUp / ModDown 是 CKKS keyswitch 的核心基转换算子，在 RNS（Residue Number System）表示下把密文多项式在不同模数基之间转换。keyswitch 中密文在模数 Q 下分解为 dnum 个 group，ModUp 把每组提升（lift）到 PQ·dnum 域（在原始模数基之上附加特殊模数 P 的基），使密文与 evk 能在足够大的域中做内积；ModDown 把 IP 结果从 PQ·dnum 域降回 Q 域并完成模归约，与原始密文相加完成 keyswitch。二者都是计算密集型算子（算术强度 3.38/2.92 ops/byte，Table I），计算模式复杂（BConv 依赖：ModUp = 原始基下常数乘 + 目标基下常数乘与归约，复杂度 O(l1·l2·N)）。
- 模数可交换性质：EWO/Autom 可与 ModUp/ModDown 交换顺序（ModUp(PMul(ct,pt))=PMul(ModUp(ct),PModUp(pt))；ModUp(CAdd(ct,ct'))=CAdd(ModUp(ct),ModUp(ct'))），这是 hoisting 能把 ModUp/ModDown 合并提取到 PKB 首尾的数学基础——代价是交换后 MemOps 的模域从 Q 升到 PQ 或 PQ·dnum、计算量增大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 keyswitch 的 ModUp-IP-ModDown 数据流（HE² 论文 Sec. II-B1）：
```
# 输入：ct（模 Q）、分解组数 dnum、特殊模 P、evk（PQ·dnum 域）
for g in 0..dnum-1:
    ct_g = ModUp(ct, g)            # 提升到 PQ·dnum 域（BConv 模式）
    ip_g = IP(ct_g, evk_g)         # 与 evk 内积（MemOps，内存密集）
    acc  += ip_g                   # 累加
out = ModDown(acc) + ct            # 降回 Q 域并加回原密文
```
- Annotations：ModUp 输出与 IP 结果是异构加速器（xPU 做 ComOps、xMU 做 MemOps）中最大的中间结果传输（单次最高 144 MB 量级），且落在 keyswitch 关键路径上——这是 HE² 通信优化的直接对象；hoisting 后 ModUp 由 n 次共享为 1 次、ModDown 聚合为 1 次（见"Hoisting"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库（SEAL/OpenFHE/Liberate-FHE 等）以 BConv + 常数乘实现；硬件上 ModUp/ModDown 各走 INTT→BConv→NTT 流水（见"BConv"与"NTTU/BConvU"条目）。使用：每个 keyswitch（乘与旋转都依赖）必经 ModUp→IP→ModDown；HE² 中 xPU 主要承担 ModUp/ModDown（含 INTT-Resident 流水：把 INTT→BConv→NTT 拆成并行 BConv→NTT 与 NTT 两路提升并行），MemOps 卸到 xMU。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
