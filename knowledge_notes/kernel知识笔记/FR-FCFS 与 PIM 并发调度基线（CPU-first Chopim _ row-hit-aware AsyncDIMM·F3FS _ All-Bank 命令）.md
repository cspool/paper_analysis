## FR-FCFS 与 PIM 并发调度基线（CPU-first Chopim / row-hit-aware AsyncDIMM·F3FS / All-Bank 命令）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FR-FCFS（First-Ready, First-Come-First-Serve）= 经典内存调度：行命中（ready）请求优先、其余按到达序——最大化行缓冲局部性，但突发高局部性流量会饿死需行冲突的请求（内存性能攻击，[55–57]）。PIM 并发调度的三大基线：(1) All-Bank 命令（HBM2 FiM [42]）：一条命令触发全部 PIM 单元，执行期无 bank 可供 CPU → 时间片轮转（COSM 设 95%CPU/5%PIM + 理想化零切换开销），PIM 阶段外部带宽空转、PIM 命令须等全部 bank 就绪；(2) CPU-first（Chopim [39]，ISCA'20）：bank 的 CPU 队列非空即阻塞 PIM——CPU 延迟最优但浪费 ACT→数据访问间的内部带宽，命令长=tBL 使命令总线饱和（PIM 吞吐仅 1.9×）；（Web: arXiv 1908.06362）Chopim 还有随机 NDA 写节流与 next-rank 预测抑制读改写干扰、bank 分区（host-reserved/shared）；(3) row-hit-aware（AsyncDIMM [38]，HPCA'25：PRE 触发 CPU/PIM 队列切换 + rank 内 relay 控制器；F3FS [40]，ISPASS'25：FR-FCFS 前加 mode 仲裁 + 跨 mode 请求 CAP 防饿死）——行命中率最优但随机访问 CPU 任务延迟恶化（AsyncDIMM-Bank 实测使 CPU -89.9%）。三者共同缺陷：无 CPU-mediated 传输的专门调度（与 CPU 访问同走 FR-FCFS）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
并发时序对比（COSM Fig.3）：FR-FCFS CPU-only：行命中连续服务、per-bank 队列不均；All-Bank：PIM 命令插入 CPU 流但全 bank 被占 → 外部带宽空闲① + PIM 等全部 bank 就绪②；CPU-first：请求到 Row1/Bank1 立即停 PIM → ACT 与数据访问间内部带宽浪费③；row-hit-aware：PRE 强制切队列 → CPU 延迟④。COSM 的取舍：保留 CPU-first 原则（CPU/refresh > PIM_Pause > PIM 命令的严格仲裁），用 IWE 补足窗口利用、可抢占命令补足长命令吞吐——PIM 吞吐较 All-Bank 6.0×、较 Chopim 2.8×，CPU 降速 <2.0%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FR-FCFS 实现于各内存控制器（COSM 保留为 CPU 侧调度器，PIM 侧另设 PIM scheduler，Command Arbiter 三源仲裁）；Chopim/AsyncDIMM/F3FS 构成并发调度设计空间的两个极端（偏 CPU / 偏 PIM）与折中。使用：CPU-PIM 并发系统的 baseline 选择与消融对照；F3FS 的 mode CAP 与 COSM 的空闲窗口插入是两种正交的防饿死思路；All-Bank 时间片是"无并发能力接口"的对照基线。ComPASS [41] 的批量调度是另一路线（粗粒度时间片、非真正并发）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
