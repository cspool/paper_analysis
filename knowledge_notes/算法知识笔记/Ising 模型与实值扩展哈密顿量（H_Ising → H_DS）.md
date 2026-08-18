## Ising 模型与实值扩展哈密顿量（H_Ising → H_DS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ising 模型源自统计物理（Ising 1924，研究铁磁性），描述全局相互作用的二值自旋系统，能量函数（哈密顿量）为 $$H_{Ising} = -\sum_{i \neq j}^{N} J_{ij}\sigma_i\sigma_j - \sum_{i}^{N} h_i\sigma_i$$：J_ij 是自旋 i、j 间耦合强度，h_i 是作用在自旋 i 上的外场（偏置）。物理实现的 Ising 机求该哈密顿量的最低能态，对应映射到模型上的组合优化问题之解（实现路线：量子退火、光学系统、耦合振荡器、CMOS）。DS-ISA 论文的核心算法扩展是把线性自相互作用项换成二次项，得到实值节点哈密顿量 $$H_{DS} = -\sum_{i \neq j} J_{ij}\sigma_i\sigma_j + \frac{1}{2}\sum_i h_i\sigma_i^2$$：二次项 h_iσ_i²/2 作为能量调节器（self-coupling），阻止能量发散、让连续实值节点稳定在有效平衡点而非饱和到边界，从而把二值优化机器推广为可承载实值 ML/科学计算的动力系统处理器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DSU 上求解 pipeline（以实值 H_DS 的一次推断为例，对应 DS-ISA 的 A1 模式）：
```
# 映射：变量 → 节点，交互 → 耦合
for (i,j) in model_edges:            # 模型权重编码为耦合电导
    J_ij = weight[i][j]              # C_LOAD 写入耦合组
for i in input_nodes:                # 输入特征编码为节点电压
    sigma_i = feature[i]; lock(i)    # N_LOAD + N_LOCK 锁定边界
h_i = self_coupling[i]               # 自耦合防止发散
# 演化：梯度下降动力学，能量单调下降 dH/dt ≤ 0
while not converged(time_limit):     # N_EVOLVE 按指定时长触发
    for i in free_nodes:
        d(sigma_i)/dt ∝ sum_j((J_ij+J_ji)*sigma_j) - h_i*sigma_i
    # 硬件上：耦合电导电流 I_in^i = Σ J_ij σ_j 对节点电容充放电
# 输出：平衡点 sigma* = 自然给出的解（N_STORE 读回）
```
二值 Ising 版（优化，B1 模式）为同一 pipeline 取 σ∈{−1,+1}：H_Ising 最低能态即 Max-Cut/SAT 等组合优化解；实值扩展后能量景观变成连续二次型，梯度流保证 dH/dt ≤ 0 收敛，可用于 GNN 层前向（DS-GL）、LLM 层映射（DS-LLM）、微分方程对齐求解（DS-TIDE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BRIM 约定下节点状态 = 电容电压、耦合参数 = 可编程电导、h_i 为对角自耦合（图 2 对角线）；实值扩展工作有 DS-GL（ISCA'24，图学习）、DS-TPU（ISCA'25，Chebyshev 多项式非线性节点交互 + 片上训练）、DS-TIDE（MICRO'25，时不变 PDE）、DS-LLM（ICLR'24，LLM 训练/推理）、InstaTrain（ICLR'25）。使用方式：任何可写成"变量交互 + 边界条件"的能量最小化/梯度动力学问题都可映射（优化、图学习、DE、EBM/Hopfield 类双向网络）；训练侧把节点锁真值、让耦合电导在电流误差反馈下演化（见 Electric-Current Loss 条目）。注意 J_ij+J_ji 对称项出现在双向演化中；节点被钳制时反向项 J_ji 可省略（A1 单向模式）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
