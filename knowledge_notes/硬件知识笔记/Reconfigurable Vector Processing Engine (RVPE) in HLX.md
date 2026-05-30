## Reconfigurable Vector Processing Engine (RVPE) in HLX

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

RVPE（Reconfigurable Vector Processing Engine）是 HLX URSC 中的向量处理和特殊函数引擎，位于两个 DPE 之间（DPE#0 → RVPE → DPE#1），负责所有非 MatMul 的向量级计算。由一个 VMEM（向量内存，存预处理中间数据）和两个 RVPU（Reconfigurable Vector Processing Unit）组成。每个 RVPU 包含：(i) add/sub 单元（256 元素宽度），(ii) rowsum/cumsum 单元（分别用于 FA-2 的 rowsum 和 SSD 的 cumsum），(iii) 两个乘法单元（element-wise multiplication），(iv) SFU（Special Function Unit：reciprocal, exp, max, log, sqrt, SiLU）。核心特色是**可重构局部 NoC（local NoC/Reconfigurable Network）**——支持 4 种操作模式，在不同数据流中动态切换引擎内部的数据路径。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

RVPE 的 4 种可重构操作模式及数据路径：

```
Mode 1: "PipeFlash local softmax" (attention)
  Input: score[2, block_size] from DPE#0
  Dataflow: add/sub unit (score - max) → exp (SFU) → add/sub unit (rowsum) 
            → reciprocal (SFU) → multiply unit (prob = exp(score-max) / rowsum)
            → rescale_factor computation (exp(prev_max - current_max) in SFU)
  Output: prob → DPE#1, rescale_factor → UpE

Mode 2: "PipeSSD pre-processing" (dA related)
  Input: dt, A, dt_bias from GS
  Dataflow: add/sub unit (dt + dt_bias) → SFU (softplus) → multiply unit (sdt × A)
            → cumsum unit (dA_CS = cumsum(sdt × A))
  Output: sdt, dA_CS → stored in VMEM

Mode 3: "Y_Diag element-wise multiplication"
  Input: CB_T from DPE#0 (via NoC), L, dt from VMEM
  Dataflow: multiply unit (CB_T × L) → multiply unit (× dt) → CB_TLdt
  Output: CB_TLdt → DPE#1 for Y_Diag = CB_TLdt @ x

Mode 4: "Y_Off / states_N element-wise multiplication"
  Input: C, exp(dA_CS) from VMEM
  Dataflow: multiply unit (dC_Off = C × exp(dA_CS)) → mux (to DPE#0)
            multiply unit (dBdt_T = (B × dt)^T) → demux (to DPE#1)
  Output: dC_Off → DPE#0, dBdt_T → DPE#1
```

mux/demux 在 Mode 4 中的关键作用：Stage 3 开始时 RVPE 同时产生两路输出（dC_Off 和 dBdt^T），通过配置 mux/demux 分别路由至 DPE#0 和 DPE#1，使两个 MatMul 并行执行。这是 PipeSSD 实现 concurrency（Y_Off ∥ states_N）的硬件基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RVPE 单 core 面积 1.76mm²（14nm），功耗 0.85W。与 DPE（面积 2.48+2.44mm²）相比轻量得多，因为它主要处理 element-wise 操作而非大规模 MatMul。RVPE 的 4-mode 可重构网络通过 local NoC 的交叉开关实现，切换开销极小（config register write）。HLX 的 Top Controller 根据当前 layer 类型（attention vs Mamba-2）和数据流阶段动态配置 RVPE 模式。RVPU 的 cumsum 单元专为 SSD 设计（cumsum 是 SSD pre-processing 的核心操作），而在 GPU 上 cumsum 通常需要多 kernel 或 warp shuffle 实现。SFU 中的 softplus 和 SiLU 直接支持 Mamba-2 特有的激活函数。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models
