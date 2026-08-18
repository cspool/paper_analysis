## Store-Load Bounce（存储-加载弹跳法）与 MDP 组织表征

术语解释
SSBench 提出的自动生成不同 IP 的 store-load 对、进而逆向 MDP 预测表组织（hash 函数、表大小、相联度、index/tag 位、替换策略）的微基准技术：在 stld 函数中把 store/load 替换为跳转到指令页的分支，通过预填充指令页高效地在受限搜索空间内生成大量不同 IP 的 store-load 对。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
硬件实现中 IP 在索引表项前常被压缩（hash 函数），SSBench 用 store-load bounce 自动反推：① hash 函数逆向——固定 store-load 对 Pair_{x0}^y 饱和状态机后切换 IP 执行 Pair_{x1}^y，若 T(N_P)=B 说明 x0/x1 hash 碰撞；收集碰撞地址集 X，构造差分矩阵 R（R_{i,j}=地址对 XOR 的第 j 位），nullspace N(R)={x|Rx=0} 的基向量即 hash 输出位（维数=hash 输出位数），零维 nullspace 表示非线性 hash（本论文实测全部线性）。② 相联度/eviction set——训练 Pair_{x0}^y 后用不同 hash 值的 Pair 依次 prime，最后测 T(N_P)：若为 S 说明训练态被驱逐，最小驱逐数即相联度；用 2 的幂倍增避免触发全局禁用 MDP。③ 表结构与 index/tag——Algorithm 2：给定 entry y 的最小 eviction set E={x0..x_{k-1}}，扩展为更大集合 S（每个地址都能驱逐 y、同映射一个 set），分析 S 中所有地址相同的 hash 位 = index 位，其余为 tag/offset 位。④ 替换策略——最小 eviction set + 4 项，测"插入 x3,x2,x1,x0 后访问 xi 谁被保留"的置换模式 Π，与 Table I 的 FIFO/LRU/Tree-PLRU/NLRU 模式匹配。实测（Table II）：Intel 256/512 项 direct-mapped（index 低 8/9 位）；AMD Zen3 z3-mdp2 32 项 2 路 LRU（tag 4-11、index 0-3、物理 IP 12-bit stride hash）、Zen4 32 项 2 路 FIFC、Zen5 64 项 4 路 NLRU；Arm A72/A73 16 项 16 路 PLRU/FIFO（tag 6-15）；Apple M2 P 核 90 项全相联 LRU。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
以 AMD Zen3 为例：用 JIT 在指令页预填 store/load 地址，代码跳转执行不同 IP 的 store-load 对 → 对 Pair 饱和状态机（D_P 反复）→ 换 IP 测 N_P 判定是否同 hash（B=碰撞）→ 收集碰撞集 → 差分矩阵 nullspace 得 12-bit stride hash（物理地址位 [0,12,24,36] 等分组）→ 构造 eviction set 得 32 项 → Algorithm 2 判定 index 位 0-3、tag 位 4-11、2 路 → 置换模式匹配判 LRU。输出组织参数（characterization.json org 字段：eviction_set_size、size/set/set_index、replacement_policy 与 confidence）。组织参数直接决定攻击可行性：如 Intel direct-mapped 512 项使 MDP-Gates 表项可持久；AMD 物理 IP 索引使 MDP-CF 可构造跨进程 hash 碰撞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SSBench 中 hash.py/org.py + src/ 微基准（store-load bounce 的 JIT 指令页）；哈希求解用 numpy 矩阵运算 + 内核接口记录虚拟/物理碰撞地址（虚拟 hash 无解时回退物理地址）。使用方式：对目标 CPU 运行组织阶段（耗时最长：hash 测试占 i13ra 总时长 95.5%、org 占 Apple a4p 80.9%），输出 hash 矩阵（hash_func 位分组 + hash_va 是否虚拟地址）、表大小/相联度/替换策略。用途：理解 MDP 表资源（容量、冲突行为、替换行为）以构造碰撞/驱逐原语，支撑 MDP 侧信道攻击与 μWM 构造。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
