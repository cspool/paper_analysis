## NVM 写耐久与可靠性（Write Endurance / ECP / BBM / 写速率-寿命）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVM（如 PCM）写耐久指每 cell 可承受的编程（写）次数上限（~10^8，DRAM >10^16 [103]）。ECP（Error-Correcting Pointers，[77]）为每个 block 分配备用位纠正坏 cell；BBM（Bad Block Management，[100]）用 overprovisioned 容量重映射失效 block——两者可改善可靠性但难突破 per-cell 耐久本征极限。写放大（WA）指物理写入与逻辑写入之比；SHyLA 中 KVCache 更新粗粒度（head 维 128 → KV 向量 ≥256B ≥ NVM 物理编程单元），WA≈1.0、无 read-modify-write。数据写强度 = 单位时间写事务/容量。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级流程：SHyLA 只把 KVCache 动态写 NVM（IA 写全留 DRAM），配合模型/部署变更（tensor/pipeline 并行调整）时的 Weight/KVCache 区域周期互换做磨损均衡 → NVM 写速率 ~15× 低于 NVM-only baseline（NVM-only 把 IA+KVCache 都写 NVM）；用 ECP+BBM（5% overprovisioning）+ 均匀磨损假设估算写速率 W_rate → 扫 Wmax=10^8–10^9 得系统寿命 T：SHyLA 寿命最长 15.4× 于 NVM-only，NVM-only 保守耐久下一年内失效（不可持续）。写延迟 tWR=1000 对系统吞吐影响小（KVCache 写少、IA 写在 DRAM）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 写速率用性能仿真统计（GPGPU-Sim 流量 + 数据放置）结合磨损假设（理想均匀 vs 集中写）评估；寿命 = f(写速率, Wmax, ECP/BBM 冗余)。SHyLA 通过"IA 写留在 DRAM + KVCache 少量写 + 周期换区"实现写压力最小化，这是相对 NVM-only 可行性的核心论证（NVM-only 写速率高、寿命短）。SHyLA 评估脚本未开源（联网未找到）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
