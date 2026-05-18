## HBM-PIM PE and SFU for Speculative Decoding（面向投机解码的HBM-PIM计算单元与专用功能单元）

术语是什么？通过联网搜索让回答具体和精准。
HBM-PIM PE (Processing Element) 是集成在HBM DRAM bank附近的轻量级计算单元，专为memory-bound GEMV操作设计。SADDLE继承并实例化AttAcc/SpecPIM的PE设计：每个HBM bank配1个PE，每PE含16个FP16 multipliers、16个FP16 adders和registers，每cycle处理2个256-bit operands（从bank local row buffer和pCH global buffer获取），所有PE在一个pCH内跨bank并行操作最大化内部带宽。SFU (Specialized Functional Unit) 是SADDLE在buffer die上新增的专用计算单元，负责transformer中的非矩阵运算：softmax（exponential + adder tree + divider）、layer normalization、activation functions (如SwiGLU)、residual addition。SFU使PIM可独立完成完整transformer layer而无需回传中间结果到GPU/xPU处理非GEMM操作，消除了跨芯片通信等待。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
SADDLE PIM chip内部结构和数据流：
```
┌────────────────── HBM Stack ──────────────┐
│  DRAM Die 7 (8 pCHs)                       │
│  DRAM Die 6                                  │
│  ...                                         │ ← TSV vertical interconnect
│  DRAM Die 0                                  │
├────────────────── Buffer Die ──────────────┤
│  ┌────────────────────────────┐             │
│  │ SFU (per stack)             │             │
│  │  softmax | LN | activation │             │
│  └────────────────────────────┘             │
│  Global Buffer | Accumulator | Router       │
└────────────────────────────────────────────┘

Per pCH detail:
  Bank Group 0: Bank0+PE, Bank1+PE, Bank2+PE, Bank3+PE
  Bank Group 1: ...
  Bank Group 2: ...
  Bank Group 3: ...
  Global Buffer ←→ Accumulator ←→ SFU (via buffer die)

PE内部datapath:
  [Bank local row buffer]─256b→[16 FP16 MUL]→[16 FP16 ADD]→[Reg]→[Global Buffer]
  [pCH global buffer]────256b→                                                  
```
PE执行GEMV流程：activation vector broadcast到pCH global buffer → weight matrix rows从各bank row buffer读取 → 每个PE执行16-wide dot product → partial sums经adder tree累加 → accumulator聚合bank group内结果 → SFU处理non-linear operations → 结果写回global buffer或经TSV到DRAM。SFU在buffer die上、PE在DRAM die上——这种物理分离允许PE紧邻DRAM arrays（最小化read latency+最大化内部带宽），而SFU利用buffer die的逻辑工艺优势容纳更复杂的计算逻辑（如exponential unit）。

术语一般如何实现？如何使用？
SADDLE PE采用FP16 MAC (28nm综合，1GHz，面积0.116mm²/PE，缩放到DRAM 1z-nm process)。每个pCH含16 PEs + 4 accumulators，DRAM die area overhead 13.4%（121mm² HBM3 die，PE+accumulator占~16.24mm²）。PE面积拆分为：arithmetic units 57%、on-chip buffers 16%、control logic 27%（专为GEMV优化，比通用PIM PE紧凑）。SFU在7nm buffer die上，softmax accelerator + accumulator 1.44mm² + 0.18mm²。SADDLE共8 PIM devices × 5 HBM3 stacks，总640GB HBM容量（与A100 DGX 640GB对齐）。该设计与AttAcc的attention-only PIM offloading和SpecPIM的static mapping PIM形成演进：SADDLE新增SFU启用完整end-to-end transformer layer在PIM内执行，消除inter-chip non-GEMM通信。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

