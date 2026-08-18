## Bootstrapping Key（BSK，自举密钥）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSK 是 TFHE 可编程自举（PBS/盲旋转）的密钥材料：把秘密钥 s 的每个分量（n 个）加密成 GGSW（TRGSW）密文组成的大密钥集合，用于在密文域执行"解密-再加密"式旋转。每个 BSK_i 是 GGSW 密文 = L×(k+1)×(k+1) 多项式矩阵（每个元素为 N 阶多项式；L 为 gadget 分解层数、k 为 GLWE 维数、N 为多项式度）。BSK 总量约 O(n·N·L) 多项式系数：TFHE 为 10s–100s MB 量级（CASCADE 参数集 III：112 MB、参数集 IV：90 MB），远小于 CKKS 的 GB 级密钥。关键复用性质：BSK 是 BSP 的参数、可在同参数下跨多次 BSP 执行复用，但不能在同一 BSP 内跨 HMUX_i 复用（每个 HMUX 需要唯一的 BSK_i）——这一"每 HMUX 唯一 + 跨 BSP 复用"的组合决定了它的存储/带宽策略（可常驻内存，但并发访问量大）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BSK 在自举 pipeline 中的使用（CASCADE Algorithm 1，参数集 I：n=500、N=1024、L=2、k=1）：
```
for i in 1..n:                       # 盲旋转 = n 次 HMUX 迭代
    BSK_i = (X^(-a_i) - 1) * BSK_i   # 对 BSK_i 做旋转/多项式减法
    ACC_i = BSK_i ⊡ ACC_{i-1}        # 外积：L×(k+1)×(k+1) GGSW 矩阵 × (k+1) RLWE 向量
```
- Annotations：每个 HMUX_i 从 BSK 集合中取唯一 BSK_i（n 个 BSK 各用一次）；外积经 FFT 变逐系数乘后由 VMA 单元执行；BSK 以 GGSW 多项式矩阵驻留（CASCADE 中 126 MB 分布式 SRAM）；流水线并行时 n 个 HMUX 需并发访问 n 个不同 BSK——这就是"并发 BSK 访问带宽"问题（集中式 HBM 无法支撑，CASCADE 用 BSK-distributed 把访问局限在各 chiplet 本地）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 生成：密钥生成时把秘密钥位用 gadget 加密为 GGSW 密文，做一次同态"密钥位查询"（bit extraction）；软件库（TFHE-rs/Concrete）在 keygen 阶段生成并常驻内存。硬件使用：片上/片外驻留策略决定带宽——Morphling 用 HBM+片上 buffer（BSK 复用依赖 batching）；MNEMOS 在 GPU 上把 BSK 分块（TBSK）到共享内存跨 batch 复用（见"LWE/GLWE/GGSW 密文与外部乘积"条目）；CASCADE 把全部 BSK 驻留分布式 SRAM（每 HC 10.5 MB BSK buffer、共 126 MB），消除 BSK 片外搬运，并把并发访问分散/限制在各 chiplet 内（BSK-stationary 数据流）。设计要点：BSK 容量（决定可驻留的最大安全参数）与并发访问带宽（决定可支撑的流水线并行度）是两个相互独立的架构约束。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
