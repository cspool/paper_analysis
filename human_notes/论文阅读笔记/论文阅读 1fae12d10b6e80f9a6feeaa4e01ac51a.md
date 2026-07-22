# 论文阅读

- ReSA: Reconfigurable Systolic Array for Multiple Tiny DNN Tensors
    - 运行时配置的SA,但需要控制核host辅助控制
    - 对不同数据流的分析可以一看
- ISARA: An Island-Style Systolic Array Reconfigurable Accelerator Based on Memristors for Deep Neural Networks
    - 运行时重配置不同精度下的数据流,降低功耗
    - 片上网络的设计讲的很清楚
- Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators
    - 基于性能建模后算子设计参数搜索,自动生成(编译)算子到DLAs(加速器)的高效实现
    - 处理不同DLAs内在约束难以精确描述,导致自动生成的算子库效率不高
    
    [笔记](%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/%E7%AC%94%E8%AE%B0%201fbe12d10b6e80cda21aff1bb550750e.md)
    
- Soter: Analytical Tensor-Architecture Modeling and Automatic Tensor Program Tuning for Spatial Accelerators
    - 基于autoTuning的编译算子?

感觉都侧重编译?那体系结构的侧重点是?可重配置?

软硬件co-design,本质是编译(软件)和运行时(硬件)的协作.

CGRA的动态编译本质也是运行时的重配置

[四大会论文23-25](%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/%E5%9B%9B%E5%A4%A7%E4%BC%9A%E8%AE%BA%E6%96%8723-25%20203e12d10b6e8033a75dc2d243318ca1.md)