## CityHash

术语解释
Google 发布的快速非加密哈希函数族（Pike & Alakuijala，2011），针对哈希表等性能敏感场景优化，比 MurmurHash 快约 30%+；Revelator 选它作为 OS 与硬件共享的哈希函数（PPN_i=CityHash(VPN,PID,seed_i)），硬件实现 2-cycle 延迟。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CityHash 是 Google 的开源哈希库（github.com/google/cityhash，MIT）：CityHash64 针对短字符串（<64 字节，哈希表键场景）优化、CityHash128 针对较长字符串、CityHashCrc128/256 用 CRC32 硬件指令加速；设计上让多数步骤含至少两个独立数学操作以利用 64-bit CPU 的指令级并行。Revelator 用 CityHash 的原因：①OS 与硬件必须用同一哈希实现才能让硬件确定性重算 OS 分配时的候选 PPN；②非加密、快速（硬件 2-cycle，远小于 PTW 的 40–100 cycles）；③已被先前哈希型翻译方案（Utopia [17]、ECH [19]、DPT [146]）使用，成熟可靠。输入为 (VPN, PID, tier seed)：PID 入哈希保证不同进程的相同 VPN 映射到不同候选 PPN（避免跨进程可预测性/冲突），tier seed 生成多个独立候选（每个 tier 一个）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的运转：投机引擎内置 CityHash 电路（2 cycle），L2 TLB miss 时用 VPN、PID 与 N 个 tier seed 并行/顺序重算 N 个候选 PPN = CityHash(VPN, PID, seed_i)，每个候选拼 page offset 得候选 PA 按 tier 顺序投机取数；末级 PTE 帧预测用 CityHash(VPN>>9)。安全考量：论文用 secret per-process hash key 防攻击者构造 VA 碰撞故意把投机引向目标地址。RTL 综合（Chisel + Yosys + Nangate 45nm）显示含哈希电路的投机引擎整体仅 0.0149 mm²、14.723 mW。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：C++ 参考实现（CityHash64/128/Crc128/Crc256），多语言移植广泛（Python cityhash、Rust simplehash、JS @anglinb/city-hash、Go go-faster/city）；注意 CityHash v1.0.3 与 v1.1+ 输出不同、FarmHash 是其继任但输出不兼容。使用：作为哈希型地址翻译/页表方案的通用哈希原语（Utopia、ECH、Revelator 等）；OS 侧在 buddy allocator 分配路径调用、硬件侧用组合逻辑实现同一函数，两侧必须保持输入格式（VPN+PID+seed）与输出语义完全一致。局限：非加密，不能用于安全敏感/碰撞抵抗场景。

涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
