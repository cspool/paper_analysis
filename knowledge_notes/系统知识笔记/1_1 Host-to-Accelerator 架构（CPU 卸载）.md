## 1:1 Host-to-Accelerator 架构（CPU 卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
1:1 Host-to-Accelerator 是 MTIA 300 的系统架构选择：每 compute blade 一个 CPU（512 GB RAM）+ 一个 MTIA 300（对比 H100/H200 系统典型的 1:8 加速器/host）。动机：DLRM 训练需要不可预测的 CPU 工作（Shampoo 优化器的矩阵特征分解等算子 offload 到 host 执行），1:1 保证 CPU 算力充足且避免 PCIe 争用；论文实测若用 1:8 比例（H100 式）把 Shampoo 特征分解放加速器上算会损失 7.8% 性能。MTIA 300 的 host 接口为 16× PCIe Gen5（64 GB/s），配合 16 ME 卸载通信数据面（host 不参与 collective 数据路径）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
一次 DLRM 训练迭代的 host/device 分工：
```python
# Host CPU（512 GB RAM, 每 MTIA 300 一个）:
#   - Shampoo 优化器: 矩阵特征分解（eigendecomposition）在 host 算（cuSOLVER 等价物）
#     → 保证数值精度（MTIA 300 上实现有性能-精度权衡）
#   - 模型开发/调试 eager 模式、job dispatch（WQE 快速派发）
# Accelerator（MTIA 300）:
#   - GEMM/SIMD/embedding 计算（72 PE）+ collective 通信（16 ME/NMC）
#   - 数据面通信不经过 host（内置 NIC chiplet）
# 通信: 16× PCIe Gen5 64 GB/s（仅控制/少量数据）
```
对比：H100 8:1 时 Shampoo 用 cuSOLVER 在 GPU 上算（1:8 损失 7.8%）；MTIA 300 1:1 使 host offload 成为可行 co-design 策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：compute blade（1 CPU + 1 MTIA 300）+ 液冷，chassis 16 compute 槽；host 接口 PCIe Gen5 DMA + secure boot 处理器 + debug 接口。使用场景：DLRM 训练的 CPU 卸载类算子（Shampoo 特征分解）、eager 模式开发调试；成本/功耗代价是每加速器配一颗 host CPU（912W 加速器 + 1500W host vs H100 500W/6500W per-8-accelerator）。信息缺口：论文未给出 host CPU 型号与 CPU offload 的通信开销。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
