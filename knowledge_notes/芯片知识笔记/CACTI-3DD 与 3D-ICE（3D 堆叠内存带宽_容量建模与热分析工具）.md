## CACTI-3DD 与 3D-ICE（3D 堆叠内存带宽/容量建模与热分析工具）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CACTI-3DD（Chen et al., DATE 2012，"Cacti-3dd: Architecture-level modeling for 3d die-stacked dram main memory"）是 CACTI 面向 3D die 堆叠 DRAM 主存的架构级建模扩展，按面积/工艺参数推导带宽-容量-timing；SHyLA 用它推导 DRAM 与 PCM 混合内存的硬件带宽与容量（共享外围电路、NVM cell density 4× DRAM、4-Hi 堆叠、TSV pitch 10μm）。3D-ICE（3D-ICE: 面向 3D IC 的热仿真器，Georgia Tech，https://github.com/nycu-eda/3D-ICE）做粗粒度稳态/瞬态热分析；SHyLA 用性能仿真的平均功耗评估 4 层 NVM-DRAM 堆叠在计算 die 上的热（液冷热沉 h=2×10^-7 W/(μm²·K)、环境 300K → DRAM 315.8-322.4K、NVM/计算 die 326.1-344.5K，均安全）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：CACTI-3DD 输入 400mm² 面积预算、NVM:DRAM 面积比、堆叠层数/TSV 参数 → 输出每 plane 的 (带宽, 容量) 设计点（Table III）→ 有效带宽按利用率折算（DRAM RD/WR 90%、NVM RD 70%、NVM WR 10%）→ 喂入 DSE/解析模型选点 → GPGPU-Sim 按所选带宽配置 channel 数做性能仿真 → 输出平均功耗 → 3D-ICE 以功耗为输入算稳态温度（Table X）验证热可行性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- CACTI-3DD 为研究扩展（论文 [11] 引用，无公开仓库链接，联网无法确认）；3D-ICE 开源（https://github.com/nycu-eda/3D-ICE）。两者均论文未说明被修改。作用：在无流片的情况下把 3D 混合内存的带宽-容量物理约束与热约束量化，支撑"4:1 NVM:DRAM 面积比 + 每 die 2344GB/s/66GB"的架构结论（up to 5.84× over DRAM-only）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
