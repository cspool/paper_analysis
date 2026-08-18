## 瞬态执行（Transient Execution）与 Spectre-V4

术语解释
CPU 在推测/乱序执行路径上执行、最终被 squash 的指令序列（不提交、无架构可见效果，但会留下微架构状态痕迹，如缓存/MDP 表项更新）。Spectre-V4（Speculative Store Bypass 变体）利用 MDP 的独立误预测制造瞬态越界访问，MDP 因此是瞬态攻击的核心泄漏源；MDP-CC 更进一步把瞬态访问的数据编码进 MDP 表项。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
瞬态执行基础（Spectre 类）：分支/值/MDP 等预测器误预测 → CPU 执行错误路径上的指令 → 结果被检测后 squash。攻击链：误预测使越界访问 arr[x]（secret）→ secret 作为索引访问自身地址空间 cache 行 → 经缓存侧信道恢复 secret。Spectre-V4 的误预测来源正是 MDP 预测"load 与 store 独立"（SSB），实际 store/load 地址重叠，load 读到旧值（瞬态）并继续传播。SSBench 对瞬态与 MDP 的交互给出新证据：① Intel 上 MDP 更新不依赖瞬态执行（MDP-Gates 借此免去推测窗口开销）；② AMD 上无前置 store 的 load 可更新 MDP（MDP-CF 扩大可探测 load 范围）；③ Apple 上未提交（瞬态）的 store-load 对即可更新 MDP——推翻了此前 [36] "Apple MDP 无推测更新"的结论（源于其漏判 SL 设计），据此实现 MDP-CC：瞬态执行中按 secret bit 决定 store-load 依赖与否，更新 MDP 计数器（bit1→counter 3、bit0→counter 0），瞬态结束后探测 MDP 恢复 secret。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
MDP-CC 瞬态信道流程（Apple M1-M4 效率核）：攻击者定位与瞬态 store-load 对 hash 碰撞的地址，用独立 store-load 对初始化该 MDP 表项 → 触发瞬态执行：secret bit x 编码进 load 地址，若 x=0 store/load 依赖（计数器更新到 3），x=1 则保持 0 → 瞬态路径被 squash（架构无痕迹）→ 攻击者探测 MDP 表项推断 bit。实测真容量 41129–152144 bps（M1-M4）、bit error rate ≤0.06、cache miss/inst≈0.01、TLB miss≈0（对 kperf 等性能计数器检测隐形），优于同平台 cache/TLB 隐蔽信道（cache 真容量 ≤139320 bps 但 miss 0.27；TLB ≤1145 bps）。内核到用户信道（M2 kext）：trojan 在内核按 bit 决定是否执行碰撞 store-load 对（bit1→counter 3），spy 用户态探测，真容量 159578.30 bps。Spectre-V4 侧：MDP 误预测产生瞬态 load，是跨进程（AMD）/跨 SGX（Intel MDPeek）泄漏的基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：瞬态执行是乱序核原生行为（无需特殊硬件），攻击者用预测器训练（重复 N_P 训练 MDP 独立预测）+ 触发（插入依赖对）制造误预测。使用方式（安全研究）：以 MDP 表项为瞬态泄漏的存储介质——瞬态期间更新、提交后探测（MDP-CC）；或利用瞬态访问的缓存痕迹（传统 Spectre）。缓解：关闭 MDP（SSBD/SSBS）、constant-time 编程、OS 上下文切换刷新 MDP 状态；这些缓解对 MDP-Gates（非瞬态 μWM）无效。论文未明确说明对瞬态执行本身的硬件实现细节（属乱序核标准行为）。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
