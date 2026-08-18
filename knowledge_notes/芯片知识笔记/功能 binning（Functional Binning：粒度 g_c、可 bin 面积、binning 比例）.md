## 功能 binning（Functional Binning：粒度 g/c、可 bin 面积、binning 比例）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
功能 binning 是半导体良率工程实践：把 die 的部分面积模块化（CPU/GPU 的核、cache slice、内存控制器、HBM PHY 等），晶圆测试后禁用落在缺陷上的模块、保留其余功能，得到低一档 SKU，从而提升有效良率（且对性能无影响——binned 版本等价于低配版）。三个关键参数：binning 粒度 g/c（c 个可 bin 模块中至少 g 个功能正常，如 14/15）、binning 比例 g/c、可 bin 面积占比（1−η，η 为不可 bin 的互联/电源/时钟区域比例）。CAPA 的 binning 良率公式源自 Stow et al.（ICCAD 2017）多 die core-binning 成本模型：P_defect(d)（Eqn. 7）× 模块级缺陷 Poisson 分布 P_bin（Eqn. 8，用 Stirling 数 S(d,c−g) 计数"d 个缺陷恰好落在 c−g 个坏模块上"的分法）对 d 求和（Eqn. 9）；支持多区域扩展（Eqn. 10-11，如核 + 内存控制器 + cache 三区域）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
binning 通过提高 Y_die 降低 C_die=CPW/(N_die×Y_die) 的分母。具体例子：A100 建模为 5/6 binning + 75% 可 bin 面积 → GPU die 良率 45%→72.6%，die 碳 −38%、整包 −9.5%；SPR 56 核 SKU 为 14/15 binning + 47% 可 bin 面积 → CPU die 良率 66%→79%、CPU 碳 −16%、整包 −15%。粒度效应：同样 binning 比例下粒度越细良率越高（600mm² die 的 5/6 良率 80%，10/12 达 94%）。可 bin 面积效应：SPR 的 47%→97% 假想扫描使渐进整包碳从 40 降到 34.6 kgCO2eq（−14%）——因为当 binning 比例降到极限时，良率提升受不可 bin 面积限制（"良率地板"）。设计启示：可 bin 面积、粒度、binning 比例是架构师手中的三个降碳旋钮，大 die 收益更大（A100 −38% vs SPR −16%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上以模块为最小单位做可测性设计（DFT）+ 熔丝/寄存器禁用缺陷模块，形成 SKU 分级（SPR 60/56/52/48/44/40 核 SKU 系列，每 die 15 核）。CAPA 里用户输入 g/c 与 1−η 两个参数即可：如 A100 用一个"5/6、75% 可 bin"近似复杂的真实策略（真实 A100 是禁用 1/6 HBM 堆 + 对应 PHY/控制器/cache slice/核的整链 binning）。与性能 binning（同 die 分频率/功耗档）不同，功能 binning 不改变性能定位。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
