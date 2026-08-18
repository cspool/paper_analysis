## Modulus Switching 与 Sample Extraction（PBS 的缩放取整与降维提取阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Modulus Switching（模数切换）是 TFHE PBS 的第一阶段：把输入 LWE 密文 c=(a_0,…,a_{n-1},b) 的每个分量按模 2N 缩放取整，即 $\tilde{a}_i=\lfloor 2N a_i\rceil_{2N}$、$\tilde{b}=\lfloor 2N b\rceil_{2N}$，把连续 torus 值映射到整数域，从而让盲旋转能用整数旋转量决定测试多项式的位移步数。Sample Extraction（样本提取）是盲旋转之后的阶段：从包含 (k+1) 个 N 次多项式的 GLWE 累加器 ACC_n 中取出第 0 个明文分量，把 GLWE 密文（形状 (k+1)×N）还原为 LWE 密文（维度 kN+1），本质是一系列按式 $SE^i((A_0,A_1,\dots,A_{n-1},B))=((a_{0,0},\dots,a_{0,i},-a_{0,N-1},\dots,-a_{0,N-i-1}),\dots,(b_i))$ 的系数置换。两者都是开销极小的"管道"阶段（FlashTFHE 剖析各占 PBS 时间 <1%），但却是 PBS 算法链中不可省略的正确性环节。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 位置与数据流（Algorithm 1）：MS 在最前（LWE→整数化）→ 盲旋转 n 次迭代 → SE 在盲旋转后（GLWE→LWE）→ Key Switching 收尾（kN+1→n+1）。伪代码：
```
# Modulus Switching：把 torus 密文整数化到模 2N
for i in 0..n:
    a_tilde[i] = round_to_nearest(2N * c[i]) mod 2N   # 含 b 分量
# Blind Rotation 使用 a_tilde 作为旋转量（X^{a_tilde[i]}）

# Sample Extraction：取 GLWE 累加器 ACC 的常数项回 LWE
out = []
for poly in ACC[0..k]:                     # k+1 个多项式，各 N 系数
    out += [poly[0], poly[1], ..., poly[i],        # 正序前 i+1 项
            -poly[N-1], -poly[N-2], ..., -poly[N-i]] # 负号折叠后 N-i-1 项
out += [ACC_body[0]]                        # 提取的 body 项 b'
# 得到维度 kN+1 的 LWE 密文，交由 Key Switching 切回 n+1 维
```
- Annotations：`a_tilde` 的量化步长 1/2N 决定旋转精度（更大 N 更准但更贵）；SE 的输出维度 kN+1 直接决定 Key Switching 的矩阵规模（k、N 越大 KS 越重）；MNEMOS 中 MS/SE/KS 在 GPU 上与盲旋转同流水执行，SE 的置换在 kernel 内完成、不引入额外全局内存往返。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TFHE-rs/Concrete 的 PBS 流水内嵌实现（MS 为逐元素缩放取整，SE 为寄存器/共享内存内置换）；硬件加速器在专用单元中实现（如 FlashTFHE 的 LPU 处理 SE 与 KS，盲旋转在 BRU）。使用要点：MS 必须在盲旋转前完成（旋转量需为整数）；SE 后密文维度膨胀到 kN+1，必须紧跟 Key Switching 恢复 n+1 维；位宽管理（bit-removal rounding，ZAMA）也通过 PBS 实现、同样走这套 MS→BR→SE→KS 流程。MNEMOS 在 A100/H100 上整套流水全程 GPU 执行（修改 Concrete 后端把 PBS 独占 offload 到 GPU）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
