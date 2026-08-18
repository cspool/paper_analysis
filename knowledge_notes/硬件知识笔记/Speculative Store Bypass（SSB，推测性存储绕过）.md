## Speculative Store Bypass（SSB，推测性存储绕过）

术语解释
乱序 CPU 中当 load 被 MDP 预测为与更老 store 无数据依赖时，load 在 store 地址未解析前就乱序执行（绕过 store）的微架构行为。SSB 提升指令级并行，但其误预测是 Spectre-V4 瞬态攻击的根源之一，也是 MDP 定时侧信道（S/B/R 时序）的物理基础。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
乱序核中，年轻 load 与更老 store 的地址重叠未知时有两种处理：保守地阻塞（依赖预测）或推测地绕过执行（独立预测）。SSB 即后者——MDP 预测独立后 load 提前取数，若 store 地址最终解析为重叠，必须 squash 全部后续指令并重执行（比直接阻塞更慢）。因此 load 的实际执行时间呈现三种可观测类型（SSBench 的 S/B/R 模型）：S（bypass，独立且正确，最快）、B（block，依赖预测，load 等待）、R（rollback，误预测独立后回滚，最慢）。乘法指令延迟 store 地址生成可放大三者时序差，是 MDP 逆向与侧信道攻击的标准探测手段（Fig.1/Fig.6）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
流程（SSBench 微基准）：st 指令写 st_addr（乘法链延迟其地址）→ ld 指令读 ld_addr（受 dep 控制与 st_addr 是否重叠）→ 若 MDP 预测独立：ld 立即执行，若实际重叠则检测到后回滚并重执行（R 时延）→ 若预测依赖：ld 被阻塞（B 时延）→ 若确实独立：正常绕过（S 时延）。攻击场景（Spectre-V4）：受害者代码中 load 依赖 secret 控制的数据，MDP 被训练成预测独立，攻击者触发误预测让 load 越界读取 secret 并作为地址访问自身缓存行，随后经缓存侧信道恢复。SSB 与 MDP 表项更新条件耦合：Intel 上仅 load 地址先于 store 解析时更新（SSBench 用它构造 MDP-Gates 的 NOT/NOR/NAND 门）；AMD/Apple 上更宽松的更新条件支撑单 load 控制流攻击与瞬态隐蔽信道。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于乱序核的 load-store 队列与 MDP 协同：MDP 表项命中且预测独立 → load 绕过（SSB）；预测依赖 → 等待。SSB 的开启/关闭由微码控制（Intel/AMD 的 SSBD——Speculative Store Bypass Disable、Armv8.5+ 的 SSBS 位），关闭即强制依赖阻塞（性能损失见 MDP 条目）。使用方式（安全研究）：以 SSB 为探测信道测量 S/B/R 时序恢复 MDP 状态（SSBench exist.py 固定 IP 执行 100 N_P+100 D_P 取 200 样本，DBSCAN 聚类判定）；以 SSB 误预测为瞬态执行原语构造 Spectre-V4 泄漏 gadget。攻击缓解：SSBD/SSBS 关闭、软件 constant-time 与屏障。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
