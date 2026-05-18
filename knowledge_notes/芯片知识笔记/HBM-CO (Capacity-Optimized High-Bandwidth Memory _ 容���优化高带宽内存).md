## HBM-CO (Capacity-Optimized High-Bandwidth Memory / 容���优化高带宽内存)

术语是什么？通过联网搜索让回答具体和精准。

HBM-CO（Capacity-Optimized High-Bandwidth Memory）是RPU论文提出的一种面向低延迟LLM推理的定制化HBM（High-Bandwidth Memory）内存设计。传统HBM（如HBM3e）以高带宽伴随高容量为设计目标（典型HBM3e stack: 1280GB/s, 48GB, BW/Cap≈27），适合训练和高吞吐推理。HBM-CO通过选择性缩减主要贡献容量而非带宽的DRAM内部结构（减少ranks/banks per group/channels per layer/subarrays），在保留HBM的shoreline bandwidth（102.5 GB/s/mm）和IO接口前提下，换取更高的bandwidth-to-capacity ratio（BW/Cap）。候选Pareto-optimal HBM-CO配置：768MB容量、256GB/s带宽、BW/Cap=341。相对HBM3e：energy per bit从~3.44降至1.45pJ/bit（~2.4× improvement），带宽/美元提高5×，总模块成本降低35×（去除了192×未使用容量），代价是每GB成本高1.81×。

从芯片设计角度拆解：

HBM-CO的芯片设计基于标准HBM core-die floorplan（如HBM3 [35][47][54]），核心修改在于DRAM阵列区域和channel组织的结构缩减，保留TSV、command和peripheral logic区域不变。具体操作（从HBM3e出发）：
1. **Banks per group**: 从4缩减到1（每bank group内bank数减少→减少row buffer和sense amplifier面积）
2. **Ranks**: 从4缩减到1（减少独立rank的存储阵列和rank-level控制逻辑）
3. **Channels per layer**: 从4缩减到1（减少pseudo-channel的独立I/O gating和地址逻辑）
4. **Subarrays**: 按需同比例缩减（缩小每bank内的subarray阵列面积）

物理实现：DRAM array region和channel shoreline按比例缩小，TSV/command/peripheral logic region不缩放（约占1/3总die area）。每HBM-CO chiplet提供dual 256 GB/s memory shorelines。在chiplet-based系统中，HBM-CO modules可混合搭配（mix-and-match），不同BW/Cap SKUs覆盖不同部署规模需求（edge: BW/Cap=38-227, datacenter: BW/Cap=170-682）。

术语一般如何实现？如何使用？

HBM-CO通过分析模型（analytical modeling approach based on [45]）从HBM core-die floorplan的wire-length scaling trends估算energy per bit和cost per module。系统层面，HBM-CO chiplets与compute chiplets通过EMIB [40]或CoWoS-L [25]共封装。部署时根据target workload（model size, CU count, batch size, sequence length）从Pareto frontier选择最小capacity满足需求的最高BW/Cap配置。例如：64-CU RPU运行Llama3-405B, BS=1, 8K seq len→optimal HBM-CO=192MB/core (BW/Cap=171)→energy per inference降低1.7× vs HBM3e。论文未提供HBM-CO物理实现或流片验证，为分析模型层面的设计。

涉及论文标题：
- RPU - A Reasoning Processing Unit
