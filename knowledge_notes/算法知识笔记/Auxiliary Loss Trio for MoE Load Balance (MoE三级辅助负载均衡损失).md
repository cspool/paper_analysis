## Auxiliary Loss Trio for MoE Load Balance (MoE三级辅助负载均衡损失)

术语解释
DeepSeek-V2 提出的一套三层辅助损失函数体系，分别从 Expert 级、Device 级和 Communication 级三个粒度控制 MoE 训练中的负载均衡，配合 Token-Dropping Strategy 实现软硬结合的负载管理。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
DeepSeek-V2 训练时采用 8-way Expert Parallelism (D=8)，160 个 routed experts 均匀分配到 8 个设备上，每个 token 激活 K_r=6 个 experts。三层辅助损失分别为：

**(1) Expert-Level Balance Loss (L_ExpBal)**：经典 MoE balance loss（Fedus et al. 2021, Lepikhin et al. 2021），用于防止 routing collapse。L_ExpBal = α1 × Σ(f_i × P_i)，其中 f_i 为 expert i 的实际负载占比，P_i 为 expert i 的平均路由概率。DeepSeek-V2 中 α1=0.003。

**(2) Device-Level Balance Loss (L_DevBal)**：DeepSeek-V2 新增设计，确保各 device 计算量均衡。将 routed experts 分区 D 组 {E_1,...,E_D}，每组部署在一个 device。L_DevBal = α2 × Σ(f'_i × P'_i)，其中 f'_i 为 device i 上所有 experts 的平均负载，P'_i 为 device i 上所有 experts 的总路由概率。DeepSeek-V2 中 α2=0.05（权重最高，因 device 级均衡对计算效率最关键）。

**(3) Communication Balance Loss (L_CommBal)**：DeepSeek-V2 新增设计，确保各 device 收发 token 量均衡。虽然 Device-Limited Routing (M=3) 限制了发送量，但若某 device 收到远超平均的 token，all-to-all 通信效率仍受影响。L_CommBal = α3 × Σ(f''_i × P''_i)，其中 f''_i 为归一化的 device i 接收 token 占比。DeepSeek-V2 中 α3=0.02。

为什么需要三层？单层 expert-level loss 不感知分布式拓扑——expert 级均衡不等于 device 级均衡（一个 device 上多个 expert 可能整体偏载）。device 级均衡不保证通信均衡（发送 bounded ≠ 接收均衡）。三层各司其职。

从算法pipeline角度拆解术语：
```
=== Auxiliary Loss Computation (per training step) ===

Input: batch of T tokens, N_r=160 experts, D=8 devices, K_r=6

// Expert-Level Balance Loss
for expert i in 1..160:
    f_i = (160 / (6*T)) * count(token selects expert i)  // actual load ratio
    P_i = (1/T) * sum_t s_{i,t}                           // mean routing prob
L_ExpBal = 0.003 * sum_i f_i * P_i

// Device-Level Balance Loss  
for device d in 1..8:
    f'_d = (1/20) * sum_{i in E_d} f_i                    // avg on-device expert load
    P'_d = sum_{i in E_d} P_i                              // total routing prob
L_DevBal = 0.05 * sum_d f'_d * P'_d

// Communication Balance Loss
for device d in 1..8:
    f''_d = (8 / (3*T)) * count(token received by device d)
    P''_d = sum_{i in E_d} P_i
L_CommBal = 0.02 * sum_d f''_d * P''_d

L_total = L_main + L_ExpBal + L_DevBal + L_CommBal
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
所有三轮损失在 HAI-LLM 训练框架中实现，与 expert parallel all-to-all 通信层、device-limited routing 和 token dropping 配合构成完整的负载管理方案。α2=0.05 高出一个数量级以上（device 均衡是分布式训练的关键性能瓶颈）。DeepSeek-V3 后续改为 Auxiliary-Loss-Free Load Balancing（bias-based），取消了此三层损失机制。DeepSeek-V2-Lite 仅用简化的 expert-level loss（α1=0.001，无 device/comm loss），因其所有 experts 部署在同一 device。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
