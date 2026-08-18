## Last-Level Cache（LLC，最后一级缓存）与 GPU 大容量 LLC

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LLC 是 CPU/GPU 内存层次中最靠近主存的一级缓存，承担"过滤绝大多数主存访问、为数据密集型负载提供高容量片上复用"的职责。现代 GPU 中 LLC 即 L2（NVIDIA H100 50MB、AMD MI300X 256MB Infinity Cache）；CPU 中通常是 L3。扩大 LLC 容量是提升数据密集型负载（LLM、图分析、稀疏线性代数）能效的主流方向——数据在 LLC 命中可避免昂贵的片外 DRAM/HBM 访问。TDMSim 论文以 32MB 6T-SRAM LLC 为 baseline（30nm 工艺、CGP 与 2D tape-out 匹配），对比 Silicon-1T1C/2D-1T1C/2D-3T0C DRAM cache 等替代 LLC 方案：把 LLC 容量提升到 128MB 时，KMeans（1e6 个 10 维点）可全部驻留而达 2.46× 加速、Llama 解码期权重与部分 KV cache 驻留而达 1.8× 加速。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TDMSim 的 gem5 MI300X 单 XCD 系统（Table II：40 CU、1GHz、L1 32KiB/55cyc、L2 4MiB/100cyc、LLC 32MiB/200cyc、treePLRU、128B line、16-way、2 HBM3 stack 660GB/s、FR-FCFS）中，GPU kernel 的访存流经 L1→L2→LLC→HBM：命中 LLC 直接服务、未命中才访问 HBM。把 LLC 换成 2D-1T1C 128MB DRAM cache 后，KMeans 的全部聚类中心与数据点、Llama 的部分权重与 KV cache 驻留 LLC，命中率上升转化为 speedup（2D 1T1C 128 平均 28.8%）。容量之外，LLC 的 cell 技术决定延迟/能量/面积：6T-SRAM 低延迟高带宽但面积大；DRAM 密度高但需刷新（见 DRAM Cache 与 Access Interference 条目）；2D 材料 DRAM 以低泄漏换取长 retention 与高密度，在等面积下把 LLC 容量扩到 512MB 且静态功率仅约 SRAM 同容量 87%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上 LLC 是片上 SRAM 阵列或 die-stacked DRAM（商用如 Intel Sapphire Rapids eDRAM L4、AMD 3D V-Cache 等），由 cache 控制器/替换策略（LRU/treePLRU）管理；GPU 中 LLC 还承担跨 SM 一致性点角色。TDMSim 用 TDM-Memory（修改版 CACTI）在 32MB SRAM 面积预算下做 cell/array 设计空间探索，选出最优 LLC 组织（LH-cache 结构、16-way、16 bank、单 DRAM row 一组），再用 gem5 全系统模拟评估真实 workload 下的 LLC 命中/干扰/能耗。评估指标：workload speedup、access interference rate、总 cache 能量（静态+刷新+动态）。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
