## 投机度过滤器（Speculation Degree Filter）

术语解释
Revelator 硬件投机引擎中的动态决策模块：用内存利用率监控（各分配 tier 的实际成功率）与带宽监控（当前内存带宽压力）决定每次 L2 TLB miss 实际发出多少个候选 PA 的投机取数请求，在翻译覆盖与内存系统压力之间权衡。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
投机度（speculation degree）= 每次 L2 TLB miss 从 N 个哈希候选 PA 中实际转为投机内存请求的个数。大的 degree（如 N=4）覆盖更多 tier（覆盖利用率升高时由后 tier 分配或被常规分配器回退的页）但每多一个 tier 就多一个投机请求，占用内存队列/互连/DRAM 带宽并可能延迟 demand 请求；小的 degree（如 N=1）在大多数页由第一 tier 分配时避免多余流量，但会漏掉后 tier 分配的页（利用率升高后更常见）。Revelator 用投机度过滤器动态选择：①memory utilization monitor——跟踪 OS 报告的各 tier 成功分配比例，丢弃分配页数低于阈值的 tier；②bandwidth monitor——对剩余 tier 按可用带宽自适应调节投机度（带宽充足时投机更多候选、带宽稀缺时收敛到少数候选）。这是 Revelator 控制"误投机代价"的关键组件：论文敏感性研究显示投机度应跟随内存利用率下 tier 的成功概率（0% 利用率 1 tier 最优、40% 2 tiers、80% 用 4 tiers 反而因冗余投机慢 4%），degree filter 在保持性能的同时减少被浪费的投机带宽与 L2 缓存污染。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（Fig.6）：L2 TLB miss → 投机引擎用 CityHash 重算 N 个候选 PPN/PA → utilization monitor 读取各 hash 计数器（OS 侧每个哈希函数维护的分配成功计数器，硬件通过轻量接口获取）→ 丢弃分配率低于阈值的 tier → bandwidth monitor 读内存控制器/互连的带宽占用 → 按剩余带宽决定在剩余 tier 上发几个投机取数请求 → 选中的候选 PA 按 tier 顺序（最可能的 H_1 优先）发投机取数。整体实现为小型硬件：哈希电路 + 每 hash 计数器 + 两个 monitor 与决策逻辑，RTL 综合（Chisel + Yosys + Nangate 45nm）仅 0.0149 mm²、14.723 mW（0.02% 面积/0.03% 功耗 vs Cascade Lake core）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件侧为 MMU 内小模块（monitor 计数器 + 阈值比较 + 带宽估算），OS 侧需为每个哈希函数维护分配成功计数器（Revelator 在 Linux 6.10.8 buddy allocator 集成时加入，属于论文所说的"OS 小改动"之一）。使用：模拟评估中作为 Revelator 的默认配置（N=3 + degree filter on），对照无 filter 的固定 degree 变体评估带宽/污染收益（论文附加敏感性研究：degree filtering 减少浪费取数而不牺牲性能）。评估平台：Virtuoso（Sniper 之上）模拟 128GB DDR4-2400 4-channel 系统，Table 2 配置。信息缺口：论文正文未给出 utilization 阈值与带宽阈值的具体数值（见 extended version）。

涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
