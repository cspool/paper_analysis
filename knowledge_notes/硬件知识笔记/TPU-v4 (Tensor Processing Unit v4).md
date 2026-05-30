## TPU-v4 (Tensor Processing Unit v4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TPU-v4 是 Google 2021 年发布的第四代 AI 加速器 ASIC，7nm 制程，每芯片含两个 TensorCore，每个 TensorCore 包含 MXU (Matrix Multiply Unit) 用于高效 bfloat16/float32 矩阵乘法。单芯片实测系统功耗 326W，32 GB HBM/chip。芯片间通过 ICI (Inter-Chip Interconnect) 以 300 GB/s 双向带宽连接到 2D/3D torus 拓扑。

在 GLaM 中，最大模型用 1,024 TPU-v4 chips 训练，2D torus 拓扑匹配 GSPMD 2D sharding 的 device mesh。bfloat16 激活值 + float32 权重混合精度。训练能耗：280B tokens 213 MWh，600B tokens 456 MWh，PUE=1.11（数据中心）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# GLaM 在 TPU-v4 集群上的单 MoE 层数据流
# Device mesh: 32×32 2D torus, 1,024 chips

# Step 1: 输入激活
# x[B/32, S, M/32] per device (2D sharding)

# Step 2: 64 experts → 16 chips/expert
# Expert weight [M/?, H/32] 在 16 chips 上进一步划分

# Step 3: All-to-all token dispatch via ICI
# Gate 选 top-2 expert → token 经 ICI 发到 expert group
# ~B·S·M / (N/E) bytes per all-to-all

# Step 4: Expert FFN on local chips via MXU
# bfloat16 matmul, float32 accumulate
# ~4·M·H·(tokens_for_expert) FLOPs per expert

# Step 5: All-to-all combine via ICI
# Expert 输出经 ICI 返回原 device
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Google Cloud 提供 TPU-v4 Pod（最多 4,096 chips）。编程：TensorFlow + XLA 或 JAX。MXU 支持 bfloat16 × bfloat16 → float32 混合精度。ICI 形成 2D/3D torus 拓扑。GLaM 训练中 TPU-v4 的计算利用率达 50-62%，远高于 GPU 集群上类似规模训练的典型利用率。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
