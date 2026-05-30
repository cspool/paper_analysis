## Local Model Parameter Updates (LMPUs / 本地模型参数更新量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMPUs (Local Model Parameter Updates) 是联邦学习中的核心通信单元。在每轮通信 t 中，client i 经过 K 步 local training 后，计算 LMPU 为 Δw_i^t = w_i^t - w_g^{t-1}，即本地训练后的 LMP (Local Model Parameter) 与上一轮 GMP (Global Model Parameter) 的差值（向量）。LMPU 包含两个角色：(1) 携带 client i 从本地数据 D_i 中学到的梯度信息——保持 data locality；(2) 作为 uplink 通信的主要 payload——每个 client 将 LMPU 传输到 server，server 聚合所有 LMPU 更新 GMP。在量化 FL 中，LMPU 是量化压缩的目标张量。FedWSQ 利用 LMPU 的一个关键统计性质：由于 LMPU 是模型参数的差值，在大规模网络中其逐元素值近似服从正态分布 ∼ N(0, σ²)，其中 σ 随层次和训练阶段变化。DANUQ 正是基于这一正态性假设来设计 QLs。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMPU 在 FedAvg/FedWSQ pipeline 中的角色：

```python
# Round t, Client i:
# 1. Receive GMP and (optionally) global scale
w_g = receive_from_server()       # GMP: shape matches model
s_g = receive_from_server()       # global scale vector (FedWSQ only)

# 2. Local training (K steps with WS if FedWSQ)
w_i = w_g.clone()
for k in range(K):
    w_i = local_sgd_step(w_i, D_i)    # WS applied in forward if FedWSQ

# 3. Compute LMPU (the communication payload)
Δw_i = w_i - w_g                 # LMPU: same shape as model

# 4. Quantize LMPU (FedWSQ/DANUQ)
Δw̄_i = danuq_quantize(Δw_i / s_g)   # B-bit per element
s_i = compute_std_per_layer(Δw_i)   # local scale vector

# 5. Upload (Δw̄_i, s_i) to server
upload_to_server(Δw̄_i, s_i)

# Server-side:
# 6. Aggregate LMPUs
Δ = Σ h_i * dequantize(Δw̄_i, s_i)   # weighted sum
w_g = w_g + Δ                         # GMP update
```

**Annotations**: LMPU 的大小 = 模型参数量（如 ResNet-18 约 11M）。不压缩时 uplink 通信 = 4 bytes × 11M = 44MB/client/round。4-bit DANUQ 压缩后 = 0.5 bytes × 11M + scale overhead = ∼5.5MB。LMPU 的正态性假设是 DANUQ 设计的理论基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMPU 是 FL 算法的标准抽象——所有 FL 方法都需要传输某种形式的模型更新。实现上，LMPU 是 client 在 local training 结束后用当前 LMP 减初始 GMP 得到的一个与模型同形的张量集合（每层一个 tensor）。在 PyTorch FL 实现中，通常用 `[p.clone() for p in model.parameters()]` 保存初始 GMP，训练后用 `[p - gmp for p, gmp in zip(model.parameters(), gmp)]` 计算 LMPU。量化 LMPU 时，FedPAQ 使用 absmax + uniform quantizer + 概率舍入，FedWSQ 使用 σ-scaling + DANUQ non-uniform quantizer。关键区别：FedWSQ 仅传输量化后的 LMPU (B-bit integers) 和 per-layer scale (1 float/layer)，无需传输额外的量化参数（zero point, 学习到的 scale 等）。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
