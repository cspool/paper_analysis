## Magic State Cultivation (MSC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Magic State Cultivation（魔态培育，Gidney, Shutty & Jones, arXiv:2409.17595）是低开销制备高保真 T 魔态（magic state）的技术——T 门是非 Clifford 门，FTQC 中需通过魔态蒸馏/培育获得；MSC 声称"培育 T 态像 CNOT 门一样便宜"。论文 [16] 用 statevector 模拟验证 d=3 的 MSC 电路正确性，更大码距只能靠启发式猜测。
- 本论文把 MSC 作为 TUSQ 的 FTQC 验证用例：MSC 电路含 mid-circuit measurement，TUSQ 以 DFTT+Caching 支持；用 18-qubit、d=3 的 MSC 电路（p=10^-4）对比 [16] 原代码库（数据在 Zenodo 10.5281/zenodo.13777072）：原代码 1166.69s → TUSQ 2.24s，520× 加速。这展示了 TUSQ 对"可扩展模拟器使更大码距 MSC 验证成为可能"的贡献。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MSC 验证流水线：① 用 Stim 等工具在物理层得到逻辑错误率 → ② 对含非 Clifford 门的逻辑级电路做 noisy statevector 模拟（此时 circuit 含 MCM/培育流程）→ ③ 验证培育出的 T 态保真度。TUSQ 的角色在②：逻辑级模拟是深电路、多逻辑比特、time+memory critical，且含非幺正 MCM 边——正好落入 TUSQ（ECM+DFTT+Caching）的最优区间。
- 在 TUSQ 内的执行：ECM 消除冗余电路实例 → DFTT+Caching 沿树遍历（MCM 边取缓存）→ 输出分布与 [16] 原实现对比，同输入下 2.24s vs 1166.69s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：MSC 电路与数据来自 Gidney 等（Zenodo 10.5281/zenodo.13777072）；Stim 可用于物理层模拟。TUSQ 论文用它论证"TUSQ 可加速 FTQC 逻辑级子程序验证"（类似用途还有 FTQC 逻辑级噪声模拟：物理层 Clifford 用 Stim、逻辑层非 Clifford 深电路用 TUSQ）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
