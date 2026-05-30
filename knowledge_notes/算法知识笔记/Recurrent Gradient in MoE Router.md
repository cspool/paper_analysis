## Recurrent Gradient in MoE Router

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Recurrent Gradient 是 RMoE 论文提出的概念，指引入跨层 GRU 后为 MoE router 提供的额外反向传播梯度路径。标准 SMoE 中 router 梯度仅来源于 expert weight score g_n 对 LM loss 的偏导和 load balance loss 的偏导。引入 GRU 后，第 i 层 GRU hidden state h_i 通过跨层连接 (h_{i-1} -> h_i -> h_{i+1}) 传递梯度，形成 Recurrent Gradient。消融实验验证：(1) RMoE + detach h_{i-1}（切断梯度但保留前向信息）：test BPC 从 1.116 退化到 1.133，甚至差于完全无跨层连接 RMoE-NP (1.123)，表明仅有前向信息不足；(2) RMoE-NP + routing logits residual（无 GRU 但有 logits 残差的梯度路径）：test BPC 1.124-1.126，优于纯 NP 但不如完整 RMoE；(3) 更深模型上 RMoE vs NP 的 gap 随深度增大，支持 Recurrent Gradient 缓解深层 router 梯度消失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Recurrent Gradient 的梯度流分解

# 标准 SMoE router 梯度（仅 per-layer）
dL/dG_i = dL/dy_i * dy_i/dscore_i * dscore_i/dG_i

# RMoE router 梯度（含 Recurrent Gradient）
dL/dG_i = dL/dy_i * dy_i/dscore_i * dscore_i/dh_i * dh_i/dG_i   # 直接路径
         + dL/dy_{i+1} * dy_{i+1}/dscore_{i+1} * dscore_{i+1}/dh_{i+1}
           * dh_{i+1}/dh_i * dh_i/dG_i                            # Recurrent Gradient
         + ...  (更后层继续反向传播)

# 消融设置对比
# (a) RMoE:              h_i = GRU(x_i', h_{i-1})          # 完整
# (b) RMoE + detach:     h_i = GRU(x_i', h_{i-1}.detach()) # 有前向无梯度
# (c) RMoE-NP:           h_i = GRU(x_i', h_0)              # 无前向无梯度
# (d) RMoE-NP + r-α:     g_i += α * g_{i-1}                # logits残差梯度
```

结果 (Enwiki8 test BPC): SMoE=1.128, RMoE=1.116, RMoE+detach=1.133, RMoE-NP=1.123, RMoE-NP+r-0.5=1.124。关键洞察：仅有前向信息而无 Recurrent Gradient (detach) 甚至不如完全无跨层连接 (NP)，说明 Recurrent Gradient 是核心贡献者。本质原理与 ResNet 残差连接类似——GRU 为深层 router 创建了直接的梯度传播路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Recurrent Gradient 不是手动实现的，而是 PyTorch autograd 引擎通过 GRU 跨层 hidden state 连接自动构建计算图并反向传播。实现要求：(1) forward 时不 detach h_{i-1}；(2) 跨层共享 GRU 参数（使 RNN cell 权重梯度从所有层累积）。该技术可推广：任何跨层路由连接（routing logits residual、attention-based cross-layer routing）都可能通过类似机制提供额外梯度路径，关键是确保跨层连接不被 detach 且连接权重可学习。

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts
