## Semantic Concentrator (SEC)

术语解释
Semantic Concentrator (SEC) 是Focus架构中负责语义级token pruning的硬件模块，集成在systolic-array accelerator的attention层pipeline中，基于cross-modal attention scores动态选择和保留与文本prompt语义相关的visual tokens。

术语是什么？通过联网搜索让回答具体和精准。
SEC是Focus Unit的一个子模块，在硬件accelerator的attention layer中执行prompt-aware token pruning。它由三个紧密协调的组件构成：(1) Importance Analyzer：使用并行max units处理attention SoftMax输出的text-to-image (T×M) cross-modal attention scores，对每个image token计算其从所有text tokens和attention heads中接收到的最大attention score作为importance；(2) a-way Streaming Bubble Sorter：将importance analyzer的a个max units级联为a-way bubble sorter，以M·a·k cycles完成top-k selection（M为image tokens数，k为保留数），与image attention GEMM (M·(M+T)·h·n/(a·b) cycles)完全重叠，在attention GEMM完成前结束；(3) Offset Encoder：用sliding window为每对保留token记录相对位置offset（small integer），使下游SIC能恢复prune后token的原始(Frame, Height, Width)空间坐标。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SEC在Focus accelerator的硬件执行流程：
```
1. Attention SoftMax(QK^T) stream from systolic array →
   SEC importance analyzer intercepts text-to-image columns
2. a parallel max units concurrently process a attention scores per cycle
   → Two dataflows supported:
   - Parallel (spatial) stream: attention columns directly streamed into max units
   - Orthogonal (temporal) stream: attention rows buffered locally for column-wise reduction
3. 1×M importance vector stored in 25KB on-chip buffer
4. a-way streaming bubble sorter refines top-a tokens incrementally →
   top-k selection over M candidates in M·a·k cycles
5. Sorter output → pruning mask applied to P(i)×V input loading
   (only retained tokens loaded, pruned tokens never accessed)
6. Offset encoder writes compact position offsets for retained tokens →
   streamed alongside GEMM output to SIC
```
Key: SEC的top-k sorter完全与image attention GEMM重叠（ratio: (M+T)·h·n/(k·b) ≫ 1），因为h·n (3584) ≫ b (32)且k < M+T，SEC不在critical path上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SEC用SystemVerilog实现为Focus Unit的子模块。Area: SEC占总Focus Unit面积1.9%（约0.061 mm² in 28nm）。On-chip buffer: 25KB for importance vector storage。SEC pruning ratio per layer: [3: 40%, 6: 30%, 9: 20%, 18: 15%, 26: 10%]，通过配置寄存器设置。SEC modular design意味着可以独立启用/禁用，不影响core compute pipeline。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

