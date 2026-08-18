## MMAT / MMP（Multi-way Multi-stride Access Table 与 Multi-way Multi-stride Prefetcher）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LIBRA 每 GPU 新增的硬件页面预取组件（论文唯一新增片上硬件）。MMP 由两部分组成：Triggered Table（按 VPN 索引、缓存近期已触发预测的 far-fault 以去重，避免多 SM 冗余预测，周期性清空）与 MMAT（压缩格式维护每 SM 多 way 多 stride 访问模式，生成高精度、成本感知的预取请求）。MMAT 每 SM 4 行（4 way），每行：36-bit last VPN + 8-bit access counter + 4 组(6-bit stride + 6-bit occurrence counter) + 8-bit sum + 36-bit monitored VPN。面积：6500 bytes/GPU（100 SM×4 way，约 7.8×10^4 NAND2 门）；CACTI 建模 data array 0.00964041mm² + tag array 0.0031502mm² = 0.01279061mm²，读 0.0051nJ/access、写 0.0062nJ/access。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
工作流（图 9）：far-fault VPN 查 Triggered Table（已存在则跳过）→ 路由到 MMAT 对应 SM 的 way（按 VPN 差阈值 512 匹配，多匹配取差最小；无匹配且已 4 way 则淘汰最低 access counter 的 way）→ 动态深度预测生成含页 VPN 列表与估计未来访问数的预取请求 → 送 CPU 侧 PPC → PPC 决策返回 GMMU 解析 far-fault 并迁移全部/部分/零个请求页。学习路径（L3 TLB miss → MMAT）：miss VPN 与 4 way 的 last VPN 比较，差 < 阈值则归为该 way，差为 stride——已存在则计数 +1，否则替换最不频繁 stride；更新 sum 与 last VPN。访问计数回填：GMMU 每页访问计数达阈值中断 → CPU 侧 UVM support 调修改后的 fetch_access_counter_buffer_entries(.) 比对 MMAT monitored VPN 并累加（图 11 只改 step 4）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
片上 SRAM 表（每 GPU 6500B），与 GMMU/TLB 流程耦合；进程级上下文切换（不同进程 CTA 在同一 SM）时重载该进程的 MMAT 状态（NVIDIA SM 进程级上下文切换粗粒度且低频 [31]）。CACTI 评估能耗/面积；论文未开源实现（无法确认）。

涉及论文标题：
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
