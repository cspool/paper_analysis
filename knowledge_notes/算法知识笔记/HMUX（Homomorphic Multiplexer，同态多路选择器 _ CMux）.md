## HMUX（Homomorphic Multiplexer，同态多路选择器 / CMux）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HMUX/CMux 是 TFHE 的基本选择原语：HMUX(a,b,s) = a·s + b·(1−s)，在密文选择位 s 控制下从两个密文 a、b 中选出其一，选择位本身加密（外部无法得知选中者）。实现上它是 bootstrapping 一次迭代的同构操作：CMux 用 GGSW（TRGSW）加密的选择位做一次 external product。TFHE 借此支持"index 加密查找"——用 HMUX 树对加密索引逐位选择表项，同时隐藏访问地址与输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 加密查找（本论文 PPGNN/查找引擎场景）：
```
# 表 T 有 2^d 项，加密索引 idx = (s_0..s_{d-1})
out = T[0]
for j in 0..d-1:
    out = CMux(s_j, out, SelectRotated(out, T, j))   # 每层一次 external product
# out = 加密的 T[idx]
```
- Annotations：d 层 HMUX 树 = d 次 external product；选择位为 GGSW 密文；与 PBS 盲旋转共享同一计算内核——本论文 HMUX 模板直接用 bootstrapping 模板实例化（"HMUX 算法流程类似一次自举迭代"）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 用法：加密图查找（PPGNN）、加密数据库索引（ArcEDB）、隐私推理激活函数（HMUX 编码 ReLU/sign，ZAMA 方案把整个激活单元替换为 HMUX CPE）、盲旋转内部逐位 CMux。软件对应：TFHE-rs 的 cmux 函数、WOPPS 的 cmux-tree 大位宽查表。AutoFHE 算子映射规则：Chisel 两输入 MUX → HMUX 模板。

CASCADE 补充视角（ISCA'26，n 次 HMUX 迭代作为自举性能瓶颈）：CASCADE 中 HMUX 是自举（盲旋转）的迭代单位：每个 HMUX_i = BSK 旋转（Line 5：BSK_i←(X^(-a_i)−1)·BSK_i）+ 外积（Line 6：ACC_i←BSK_i⊡ACC_{i-1}），外积是矩阵-向量乘（L×(k+1)×(k+1) 多项式矩阵 × (k+1) 向量），占计算成本主体，用 FFT/IFFT 把多项式乘从 O(N²) 降到 O(NlogN)。关键性质：n 个 HMUX 严格串行（每个依赖前一 ACC），且每个 HMUX 访问唯一的 BSK（BSK 不能跨 HMUX_i 复用，只能跨多个 BSP 复用）——这两个性质分别是串行吞吐瓶颈与并发 BSK 带宽瓶颈的根源。CASCADE 的 HMUX Chiplet（HC）把一次 HMUX 实现为 Rotation→Decomposition→FFT→VMA→IFFT 的系数粒度流水（一个 HMUX 时延≈最长流水级），用 Interleaved-Fusion 把连续 HMUX 融合成组（组内回馈本地执行、减少 D2D 通信）并跨 chiplet 交错以保持流水并行。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
