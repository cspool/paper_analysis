## Focus Unit

术语解释
Focus Unit是Focus架构的核心硬件模块，作为一个modular component嵌入systolic-array accelerator的memory interface附近，包含SEC和SIC两个streaming子模块，在GEMM/attention/FC执行流中执行on-chip multilevel concentration。

术语是什么？通过联网搜索让回答具体和精准。
Focus Unit是Focus提出的专用硬件加速单元，定位在systolic-array accelerator的计算核心与off-chip DRAM之间。它由Semantic Concentrator (SEC)和Similarity Concentrator (SIC)两个完全on-chip、streaming的子模块组成。SEC在attention layer中做cross-modal prompt-aware token pruning（token-level）；SIC在FC/FFN/projection layers中做vector-level similarity-based compression + scatter/gather reconstruction。Focus Unit是modular设计：类似pooling或activation function，不修改core compute pipeline（PE array, weight stationary dataflow, etc.），拦截compute stage间的数据流进行压缩，压缩后数据才写回DRAM。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Focus Unit在accelerator中的集成位置和数据流：
```
Off-Chip DRAM ←→ On-Chip Memory (734KB: input 128KB + weight 78KB
    + output 512KB + layouter buffer 16KB)
       ↕
[Focus Unit] ← positioned near memory interface
  ├── SEC: intercepts attention SoftMax → importance + top-k + offset
  └── SIC: intercepts GEMM tile output → similarity gather/scatter
       ↕
Systolic Array (32×32 PE, FP16 mul/FP32 acc, weight stationary)
```
Focus Unit总面积3.21 mm² (TSMC 28nm)，总功耗736 mW。相比vanilla systolic array仅增加2.7% area和0.9% power。SEC占1.9%面积，SIC占0.8%面积。On-chip buffer总计734KB，off-chip memory使用DDR4 4Gb×16, 2133R, 4 channels, 64GB/s。Target clock 1.32ns (≈757 MHz)，500MHz place-and-route有34% timing margin。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Focus Unit用SystemVerilog RTL实现，on-chip SRAM由TSMC N28HPC+ Memory Compiler生成，Synopsys Design Compiler在28nm SS corner (0.81V, 125°C)综合。Focus Unit的modular设计使其可扩展：可配置PE array size、tile size (m, n)、block size、vector length、similarity threshold和per-layer SEC pruning ratio。性能指标：4.47× speedup vs dense SA，7.90× vs GPU (A100)，2.37× vs GPU+FrameFusion。Energy efficiency: 4.67× vs SA，17.09× vs GPU。DRAM traffic: 4.9× reduction vs dense SA。开源实现含algorithm/simulator/rtl/evaluation_scripts: https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

