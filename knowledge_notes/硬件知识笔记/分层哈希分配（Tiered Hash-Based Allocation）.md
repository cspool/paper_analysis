## 分层哈希分配（Tiered Hash-Based Allocation）

术语解释
OS 物理页分配策略：对每个虚拟页号（VPN），用同一哈希函数配合多个 tier 专属 seed 依次计算 N 个候选物理页号（PPN），分配第一个空闲候选帧，全部占用才回退常规分配器——从而建立硬件可重算的可预测 VA-to-PA 映射，且成功率只取决于内存利用率、与碎片结构无关。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
分层哈希分配是 Revelator（本文）的核心 OS 侧机制，把"硬件可预测放置"从依赖连续物理内存（大页/VA-to-PA 连续）改为依赖哈希这一 per-page 不变量。分配流程：tier i 计算 PPN_i = CityHash(VPN, PID, seed_i)，按 tier 顺序检查候选帧是否在 buddy allocator 空闲列表中，空闲即分配（建立可预测映射），占用则进入下一 tier；N 个候选全占用才回退常规全相联分配（该 VPN 映射不可预测）。概率分析：每 tier 独立、候选帧均匀分布，单次成功概率 p=空闲页数/总页数，N 次中至少一次成功概率 P_S=1-(1-p)^N，全部失败概率随 N 指数下降；tiering bias 使第 i 个哈希尝试的分配概率 p(1-p)^(i-1) 几何递减（H_1 最多、H_2 次之），硬件据此按 tier 顺序发投机取数。Revelator 还把这个分配扩展到末级页表帧（PT frame 单次尝试 H_1(VPN>>9)，因每个 PT frame 覆盖 512 个 VPN），使 VA→PTE 也可预测、硬件可在 PTW 开始时就投机取末级 PTE。真实 Linux 6.10.8 原型实测：3 个哈希可成功哈希分配 70–85% 的页，minor fault 开销仅 0.078% 总执行时间；用 2MB 页时先哈希 2MB 分配、失败回退 THP、再回退 4KB 哈希（Revelator+THP 四级分配：哈希 2MB→THP→4KB 哈希→buddy）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件侧的作用：L2 TLB miss 后，投机引擎用同一哈希函数（2 cycle 电路）与同一 seed 集合重算 N 个候选 PPN，得到硬件侧与 OS 分配时完全相同的候选集——这是"OS 引导"的本质：OS 与硬件共享哈希函数和 tier seed，硬件无需存储每个页的映射元数据，只需哈希电路 + 每 hash 计数器（跟踪各 tier 实际分配成功率供 degree filter 用）。运转例子：某 VPN=0x1234 缺页 → OS 算 PPN_1=CityHash(0x1234,pid,seed_1) 空闲 → 分配，建立映射；之后该 VPN 触发 L2 TLB miss → 硬件 2 cycle 重算 PPN_1..N，按 tier 顺序发投机取数 → 若数据实际在 PPN_1，取数在 PTW 完成前到达（隐藏翻译延迟）。多核下 16 核系统投机准确率稳定 87–88%。NUMA 变体：policy-first node-scoped hashing——先按 OS NUMA 策略选目标节点 n*，只在 n* 地址范围内做分层哈希放置，保留 OS 局部性决策。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux buddy allocator 扩展（Linux 6.10.8，O(1) 每分配：最多 N 次哈希 + 常量时间元数据检查）；哈希用 CityHash（github.com/google/cityhash，非加密、快速，OS 与硬件共用同一实现保证确定性）。使用：在 Virtuoso 模拟器中选 allocator=Revelator（N=3 默认、degree filter on）与 ReserveTHP/SpOT/ASAP 等对比；也可在真实 Linux 内核跑原型（分配成功率与 minor fault 开销测量）。设计要点：N 的选择权衡——N 大提高哈希分配成功率但增加分配延迟与投机带宽（论文敏感性：0% 利用率 1 tier 最优、40% 利用率 2 tiers 最优、80% 利用率 4 tiers 反而慢 4%）。信息缺口：论文未公开单独的内核 patch 仓库，Revelator 分配器随 Virtuoso 的 revelator-artifact-release 分支发布。

涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
