## Near-Data Processing (NDP) for MoE

术语解释
近数据处理（NDP）是一种将计算资源放置在数据存储附近的架构范式。在MoE推理中，NDP通过在内存侧（而非GPU侧）处理"cold"expert来避免通过PCIe/CXL传输大量expert参数，替代传统的Parameter Movement范式。

术语是什么？
MoE推理的瓶颈之一是expert参数的传输——当expert不在GPU显存中时需要从CPU内存/SSD加载。
NDP的核心思想：与其将expert参数移动到GPU计算，不如将计算移动到expert参数所在的位置（内存附近）。
- **MoNDE**（Mixture of Near-Data Experts）：CXL-based NDP方案，GPU处理"hot"expert，NDP单元处理"cold"expert，以Activation Movement替代Parameter Movement。NDP核通过CXL连接，使用LPDDR SDRAM提供高带宽。

从硬件架构角度拆解术语。
MoNDE的架构与数据流：
```
传统Parameter Movement模式：
[CPU Memory/SSD] --(expert weights)--> [GPU] --(compute)--> output
瓶颈：expert权重体积大，传输延迟高

Activation Movement模式（MoNDE NDP）：
[GPU] --(activations)--> [CXL Fabric] --> [NDP Core + LPDDR SDRAM]
  |                          |
  | (hot expert本地计算)    | (cold expert NDP计算)
  v                          v
  y_hot                      y_cold
         \                  /
          --> 合并 --> output
```

数据流：
1. Router计算完成后，hot expert在GPU本地执行
2. Cold expert的激活值通过CXL发送到NDP核
3. NDP核在LPDDR SDRAM内读取expert权重并执行FFN计算
4. NDP结果通过CXL返回GPU汇总

优势：
- 激活值 << expert权重 → 传输数据量大幅减少
- 利用空闲的内存带宽进行计算
- GPU和NDP可并行执行

术语一般如何实现？如何使用？
- CXL协议：Compute Express Link，提供低延迟的CPU-GPU-内存互联
- NDP核：嵌入在CXL控制器中的轻量级计算单元
- 需要Hot/Cold expert分类器（基于访问频率统计）
- 适用于：数据密集型MoE推理（expert多但每次只激活少数）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

**补充（来自 Context-Aware MoE on CXL-NDP）**：该论文将 NDP 概念推进为 context-aware 系统——在 prefill 阶段收集每 expert 的激活频率 P_{l,e} 和路由评分 W_{l,e}，计算归一化重要性分数 S_{l,e} = αP̃_{l,e} + (1-α)W̃_{l,e}，依此动态决定 hot/cold 分类。与 MoNDE 的 context-agnostic 全局频率统计不同，本论文的 per-sequence 重要性评分捕捉了 context-dependent expert 激活模式（prefill-decode cosine sim ~0.89）。同时引入 Activation Movement vs Parameter Movement 的量化对比：per-token activation ~8KB vs per-expert weight ~170MB (Mixtral-8×7B)，减少约 2×10^4× 数据移动量。

---
