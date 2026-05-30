## Model FLOPs Utilization (MFU)（模型 FLOPs 利用率）

术语是什么？
Model FLOPs Utilization (MFU) 是衡量大模型训练硬件效率的核心指标，定义为模型实际达到的 FLOPs 与硬件理论峰值 FLOPs 的比值：MFU = Actual TFLOPS / Peak TFLOPS。MFU 考虑了训练中所有开销（通信、kernel launch、pipeline bubble、重计算等），是端到端训练效率的综合度量。Megatron-LM (Narayanan et al. 2021) 引入这一指标并给出 dense LLM 的 MFU 上限约 52-57%（A100 上）。

从算法pipeline角度拆解术语：
MFU 的计算过程：
```
1. 模型每步 FLOPs：基于模型参数和 micro-batch 配置的理论计算量
   - MoE 模型：仅计算激活的 expert FLOPs（稀疏计算），而非全部参数 FLOPs
   - MFU = 实际 TFLOPS / 峰值 TFLOPS → 其中"峰值"通常取 BF16 理论峰值（H100: 989.5 TFLOPS/GPU）

2. 通信模型对 MFU 的衰减：
   - TP 通信: 2 × bsh (n-1)/n 字节（AG + RS），恒定占比
   - EP 通信: 2 × k/n × bsh (n-1)/n 字节（A2A × 2），随 n 增大而减小
   - MFU ≈ T_compute / (T_compute + T_comm + T_bubble)
   其中 T_compute ∝ FLOPs / Peak, T_comm ∝ Communication_vol / Bandwidth

3. 本论文结果：
   - Mixtral 8x22B (w/ Folding): 49.3% MFU，128 H100 GPU
   - Qwen2-57B-A14B (w/ Folding): 39.0% MFU，64 H100 GPU
   - Fine-grained MoE 的 MFU 低于 coarse-grained: 更小的 expert hidden size → 更低 GEMM 效率 + 更高通信占比
```

术语一般如何实现？如何使用？
- 开源计算脚本：https://github.com/NVIDIA/Megatron-LM 中的 MFU 计算逻辑
- 使用 benchmark 模式（token-dropping, CF=1）消除负载不均的性能抖动，获取稳定的 MFU 读数
- MFU 用于指导并行策略选择：比较不同 (tp, ep, cp, pp) 配置下的 MFU，选最优配置
- 对 MoE 模型，MFU 远低于 dense 模型（因稀疏激活 + all-to-all 通信开销）

涉及论文标题：
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
