## LLR 软判决解调与 layered min-sum 迭代解码

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLR（对数似然比）是软判决接收的核心量：对收到的每个比特 b_k，L_ch(b_k)=log[P(y|b_k=0)/P(y|b_k=1)]，符号给极性（>0 判 0、<0 判 1）、幅值给置信度（如 |L|=3.0 高置信、0.2 低置信）。PAM4 Gray 映射下，接收符号 y 相对符号子集 X_k^(0)/X_k^(1) 计算 LLR，常用 min 近似 L_ch(b_k)≈(1/2σ²)(min_{x∈X_k^(1)}(y−x)² − min_{x∈X_k^(0)}(y−x)²)。layered min-sum 是 LDPC 解码的低延迟调度：按 H 的行（层/check node）逐层更新——check-node 更新用"exclude-self"规则（符号取邻居符号积、幅值取邻居 LLR 绝对值最小），variable-node 做增量累加 L(v_i)←L(v_i)+m_{cn→v}，后续层立刻复用本层更新的 LLR（比 flooding 收敛更快）；每轮后做硬判决 ĉ 与 syndrome 检查 H·ĉᵀ≡0，为 0 则终止，否则继续迭代直至预算 N 或触发重传。解码是 NP-hard，迭代式解码是该困难性的工程出路（Web 证据：layered min-sum 广泛用于 5G NR LDPC、NAND flash LDPC，早期终止标准基于 syndrome/LLR 可靠性）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DICE 接收端解码流水线伪代码：
```
# 初始化
for v_j: L(v_j) = L_ch(v_j)            # 信道 LLR 输入
# 迭代（每轮 sweep 所有层）
for t in 1..N:
    for layer i in 0..m-1:
        for edge (cn_i, v_j):
            m_cn_to_v = prod(sign(L_other)) * min(|L_other|)   # check-node，exclude-self
            L(v_j) += m_cn_to_v                                # variable-node，增量累加
    hard_decision: ĉ_j = 0 if L(v_j)>=0 else 1
    if H @ ĉ == 0 (mod 2): return SUCCESS                    # syndrome 检查
return NACK_RETRANSMIT                                          # 预算 N 用完仍未收敛
```
Annotations：DICE 标定 N=4（35 dB 下所有码率 ≤2 迭代收敛，Fig.10）；每迭代 1 cycle（含全层 LLR 更新）、syndrome 1 cycle，总延迟 = (N+1)·L_syn + N·L_iter = 2N+1 cycles（≤9 cycles）。示例（论文）：y=[-45,-171,+137,+158]mV → L_ch=[+22.8,-27.8,+122.5,+35.9,-88.1,+18.7,-109.4,+29.4] → 一轮三层 sweep 后 L=[69.3,-80.0,170.6,58.7,-117.5,69.3,-132.2,80.0] → 硬判决 [0,1,0,0,1,0,1,0] 通过 syndrome。迭代预算与 SNR 的耦合：15 dB 噪声下 2B 奇偶只能纠少量错（2 迭代内可纠的都纠了）、更多奇偶需更多迭代——迭代预算-码率-噪声三者构成解码成本环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上 check-node 单元做符号乘积 + 最小值树，LLR 缓存 + 分层调度器；DICE 在 Verilog 实现后经 Yosys/OpenSTA 合成标定 1 cycle/iteration 时序。使用方式：作为接收端 flit 恢复的前置（S2P 之后），成功则 ACK 释放发送缓冲、失败则 NACK 仅重传该 flit；gem5 开销大头来自该迭代解码（占总开销主导，DICE 平均开销 9.2%），论文提出 memoization 缓存符号→LLR 模式作为未来优化。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
