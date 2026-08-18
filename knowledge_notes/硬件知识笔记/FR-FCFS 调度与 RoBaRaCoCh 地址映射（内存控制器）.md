## FR-FCFS 调度与 RoBaRaCoCh 地址映射（内存控制器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FR-FCFS（First-Ready, First-Come-First-Serve）：经典 DRAM 内存控制器调度策略——"就绪"（行命中，数据已在行缓冲）的请求优先于未就绪请求，同优先级内按到达序（FCFS）服务；目标是最小化行切换、最大化行缓冲局部性与吞吐。RoBaRaCoCh：物理地址位段从高位到低位按 Rank-Bank-Row-Column-Channel 顺序映射的 DRAM 地址交织方案（gem5 等模拟器支持的映射族之一，另有 RaBaChCo/RaBaCoCh/CoRaBaCh 等），决定请求在 bank 间的分布（bank 级并行）与同 bank 内的行命中率。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 两者共同决定 ColumnKeeper 预防性刷新插入命令流后的实际性能影响，是本论文评估的基础环境配置。运转例子：请求进入 64 项读写队列 → FR-FCFS 仲裁（行命中请求连续服务，行冲突请求按到达序排后）→ RoBaRaCoCh 把连续物理页交织到不同 rank/bank 以利用并行、同页连续地址落在同一行以利用行命中 → ColumnKeeper 的预防性刷新作为额外 ACT+PRE 与正常请求一起参与仲裁。由此：工作负载按 RBMPKI（每千条指令行缓冲 miss 数）分类成为分析 ColumnKeeper 开销的关键维度——高 RBMPKI 负载行冲突频繁、防御注入的刷新与其争抢 bank，开销最大（如 16K 阈值下高 RBMPKI 类最大能耗增量达 2.00× 以上）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：FR-FCFS 在内存控制器调度器（gem5 mem/dram_ctrl.cc 提供 FCFS/FR-FCFS 选项，open/closed 页策略可配）；RoBaRaCoCh 由地址位段重排实现（gem5 提供 RaBaChCo/RaBaCoCh/CoRaBaCh 等，开放页策略适配 RaBaChCo 类映射）。使用：内存系统模拟的标准默认配置；选择映射与页策略需配合（开放页+按 bank 交织 vs 关闭页+按 channel 交织）；在防御评估中，调度/映射决定注入刷新的排队位置与 bank 冲突率，是开销敏感度分析（§6.5 subarray 大小、§6.6 页分配器、§6.7 SALP）的基线。

PVAC 视角（ISCA'26）：PVAC 的内存控制器配置同为 FR-FCFS 调度，但地址映射用 MOP4CLXOR [34]（与 ColumnKeeper 的 RoBaRaCoCh 同族不同位段顺序；MOP4CLXOR 通过 XOR 折叠相邻地址位把连续流量交织到多 bank，提升 bank 级并行），搭配 64-entry 读/写队列与 4 核系统。在攻击模式评估中，MC 的 FR-FCFS 仲裁与 MOP4CLXOR 交织决定 round-robin 攻击行（n 行、stride 1–5）在 bank 间的分布，进而影响 PVAC 的 victim 计数器累积与 RFM 触发频率（stride≥3 时无自重置、RFM 最多）。防御机制（PRAC/Chronus/QPRAC/MOAT/PVAC）作为缓解逻辑叠加在 FR-FCFS 之上，Alert→ABO→RFM 的 350ns 阻塞窗口与正常命令一起参与仲裁。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems

- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
