## Hardware-Aware Bitwidth Allocation via ILP（基于整数线性规划的硬件感知位宽分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hardware-Aware Bitwidth Allocation via ILP 是 MxMoE 提出的混合精度量化方案选择机制。给定一个 MoE block（E 个 expert，每个 expert N 个 linear block），以及硬件支持的量化方案集合 S（如 {W2A16, W4A16, W4A4, W4A4-g128, W8A8}），目标是为每个 linear block (i,j) 分配最优量化方案 k ∈ S，最小化联合目标 L^r · T^{1-r}。其中 L = Σ Δ_{i,j,k} · x_{i,j,k} 为量化输出扰动（per-block Euclidean distance 加权求和），T = (1/P) · Σ c_{i,j,k,t} · y_{i,j,k,t} · x_{i,j,k} 为基于 tile 级 profiling 的执行时间近似。变量 x_{i,j,k} ∈ {0,1} 表示方案选择，y_{i,j,k,t} ∈ {0,1} 表示 tile 配置选择。约束条件包括：每个 linear block 恰好选一个方案、每个选中的方案恰好配一个 tile 配置、量化后权重总量不超过内存预算 M。超参数 r 控制精度-性能权衡。该 ILP 问题离线求解（使用标准 ILP solver），输出最优 {x_{i,j,k}} 和 {y_{i,j,k,t}}，指导后续 GPTQ 量化和 kernel 生成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MxMoE 中 ILP bitwidth allocation 的编译时流程：

```
=== 离线编译时阶段 ===
输入: MoE 模型结构 (E experts, N=3 linear blocks each), 
      硬件平台信息 (SM 数量 P, 内存预算 M),
      支持的量化方案 S = {W2A16, W4A16, W4A4-g128, W8A8, ...}

1. 校准统计收集:
   for each linear-block (i,j) in MoE block:
       for each scheme k in S:
           # 量化该 linear block 并在校准集上测量输出扰动
           W_q = GPTQ_quant(W_{i,j}, scheme k)
           Δ_{i,j,k} = ||MoE_block(W_q at ij) - MoE_block(FP16)||₂

2. Tile profiling (ahead-of-time):
   for each (linear-block-shape, scheme, tile-config):
       在目标 GPU 上 profile 单 tile 执行时间 c_{i,j,k,t}
       记录 tile 数量 n_t

3. ILP 求解:
   variables:
     x_{i,j,k} ∈ {0,1}  // 方案选择
     y_{i,j,k,t} ∈ {0,1}  // tile 配置选择
   
   minimize: (Σ Δ·x)^r · ((1/P)·Σ c·y·x)^{1-r}
   
   s.t.:
     Σ_k x_{i,j,k} = 1  ∀i,j  // 每 block 一方案
     Σ_t y_{i,j,k,t} = 1  ∀i,j,k
     Σ W_{i,j,k}·x_{i,j,k} ≤ M  // 内存预算
   
   输出: 最优 {x*}, {y*}

4. 根据 ILP 输出生成 kernel 配置:
   for each selected scheme k:
       编译对应 micro-kernel
   生成 mixed-precision Group-GEMM kernel with tile scheduler
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. 校准数据使用 WikiText2 的 128 条 sequence（长度 4096），校准耗时从数分钟到数小时
2. ILP 使用标准 solver（如 Gurobi, OR-Tools）；问题规模：对 64-expert MoE，约 64×3×|S| 个变量
3. Tile profiling 在目标 GPU 上一次性完成，结果缓存供所有层复用
4. r 选择：weight-only 极低比特时 r=1（精度优先，资源受限环境），weight-activation 时 r=0.75（平衡）
5. 与 HAQ（Wang et al. 2019）类似但扩展到 MoE 的 linear-block 粒度，并联合优化执行时间而非仅精度

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
