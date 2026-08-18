## Access Count Monitor（访问计数监视器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CDFD 的本地访问统计硬件（256 项，每项 36-bit VPN + 8-bit 访问计数器，共 1,408 B）：跟踪"远端更新最频繁的重复页"的本地访问次数，供 CDB 计算去重收益。所有 SM 的 L1 TLB 访问都转发给它：从 VA 提取 VPN 与监控集匹配则计数器 +1；周期与 CDB 同步——本地计数右移 1 位 + 新增量写回 CDB 对应子项（融合长期与短期访问模式），远端更新计数同样右移衰减；随后用 CDB 当前子项 VPN 集替换监控集进入下一周期。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
避免为全部重复页维护全量计数：只监控 CDB 候选集（远端更新频繁者），256 项即可覆盖；8-bit 计数 + 周期右移衰减在精度与存储间平衡。监控集与 CDB 双向同步：ACM 给 CDB 供本地访问数，CDB 给 ACM 供待监控 VPN（CDB 项可能被换出，需刷新监控集）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
可复用 Volta+ 既有每页 32-bit 访问计数器（GMMU 在 TLB 查找时自动更新，经 fetch_access_counter_buffer_entries() 读回），论文称只需小幅扩展；能耗用 CACTI(32nm) 估 0.00998 nJ/access（含 CDB），是 CDFD 额外功耗 +5.58 W 的组成部分之一。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
