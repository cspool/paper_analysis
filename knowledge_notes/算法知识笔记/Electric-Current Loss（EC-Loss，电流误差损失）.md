## Electric-Current Loss（EC-Loss，电流误差损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EC-Loss 是 DSU 片上训练（耦合演化模式）的损失机制：训练时输出节点被钳制（locked）到真值数据，输出节点内部电流 I_R^i = h_iσ_i 随之保持恒定，而输入节点经耦合流入的电流 I_in^i = Σ J_ijσ_j 由数据与当前耦合决定，两者之差 I_loss^i = I_in^i − I_R^i 直接就是该节点的误差/失配电流。反馈回路用 I_loss 调整可编程电导（耦合参数），使 |I_loss| 最小化，等效于最小化对应训练损失——把损失函数的梯度下降转化为模拟电流误差驱动的电导自适应，不需要数字反向传播。该机制由 [36]（DS-TPU）引入，并被扩展到多层网络（DE 对齐 [15] DS-TIDE、LLM 训练 [29] DS-LLM）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
片上训练 pipeline（对应 DS-ISA 的 C1 耦合演化模式 + Evolve-Load 循环）：
```
# 映射：输入变量→输入节点，输出变量→输出节点，可训练参数→两集合间耦合
C_LOAD J_ij                          # 初始化耦合电导
for iteration in 1..T:               # DS-ISA 评估用 T=100，每轮演化 10ns
    N_LOAD 钳制输入节点 = batch 输入
    N_LOCK 钳制输出节点 = batch 真值        # 节点全锁
    C_EVOLVE [GM=耦合子集, Time=10ns]      # 标签-触发：仅耦合演化
        for (i,j) in trainable_couplings:
            I_in^i = sum_j J_ij * sigma_j      # 输入侧电流
            I_R^i  = h_i * sigma_i             # 真值钳制下的恒定内部电流
            I_loss^i = I_in^i - I_R^i          # 逐节点误差电流
            J_ij -= eta * f(I_loss^i)          # 反馈回路调电导，min |I_loss|
C_STORE J_ij                          # 保存训练后的权重
```
与推断的对称关系：推断 = 锁耦合（权重恒定）演化节点；训练 = 锁节点（数据恒定）演化耦合。微调（C2 部分耦合演化）只需 CLM 掩码只解锁待训练耦合子集，其余锁定——同一机制直接支持 fine-tuning。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：模拟域反馈回路（I_loss 电流直接驱动电导编程，无数字梯度计算）；在 DS-ISA 下由 C_LOCK（CLM 掩码选择演化耦合子集）+ C_EVOLVE（GM + Time 触发）实现，节点侧由 N_LOCK 全部钳制。使用方式：DSU 原生训练/终身学习（lifelong learning：耦合常驻、持续观察新数据演化，绕过每次耦合重载的 O(N²) 开销）；评估中每迭代 10ns、训练 100 轮，C_EVOLVE 是训练 workload 的主要成分（Fig.12）。局限性：论文以推理/训练/优化/DE 四类控制负载为评估对象，EC-Loss 的收敛精度对照数字训练未在本文评估（沿用 [14][36] 的物理时间尺度假设）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
