## PCM（Phase-Change Memory，相变存储器）与 NVM 技术选型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PCM（相变存储器）通过硫系材料（如 Ge2Sb2Te5）在晶态/非晶态间的相变存储数据：晶态低阻（SET）、非晶态高阻（RESET），读为低延迟非破坏、写需长恢复（tWR）。SHyLA 选 PCM 作为 NVM 原型：高密度（cell density 4× DRAM）、天然 3D 可堆叠、商业成熟（Intel Optane 即 3D XPoint 类），制造就绪度与 DRAM 工艺对齐。其读写不对称（读带宽高、写带宽 10% 利用率、tWR=1000 vs DRAM tWR=9）正是混合内存设计的关键约束；写入每 entry 一次（KVCache 写事务少）使 PCM 写压力可控。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级运转：PCM 4-Hi 堆叠、4× DRAM 密度 → 同面积下更高容量或更高读带宽 → SHyLA 把容量/读密集、写稀疏的 Weight 与 KVCache 放 PCM、写密集 IA 放 DRAM → DSE 在带宽-容量曲线上选点（Table III 中选 (112GB/s, 1GB/plane)，每 die 4-Hi × 20 plane × 1.6/2.0 面积占比 = 64GB/1792GB/s）。PCM 时序（tCL=14、tRCD=120、tRP=14）与 DRAM 同频率 541MHz；4× tRCD 仅 ~20% 降速（大块传输与流式读隐藏延迟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 参数取自 CACTI-3DD 推导 + 文献（PCM RD/WR energy 3.4/17.84pJ/bit、BG energy 4.23mW/GB）；耐久建模用 ECP+BBM 5% overprovisioning、Wmax 扫 10^8–10^9（[39] 报告 PCM 可达 10^12）。技术无关 DSE：低密度 NVM（如 MRAM）→ 设计点偏向容量；高延迟 → 偏向带宽，框架按输入 NVM 参数自动调整。SHyLA 架构未开源（联网未找到）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
