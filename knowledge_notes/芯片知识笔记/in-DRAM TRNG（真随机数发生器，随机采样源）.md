## in-DRAM TRNG（真随机数发生器，随机采样源）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- in-DRAM TRNG 是在 DRAM 芯片内生成真随机比特的硬件模块，作为概率性 RowHammer 缓解（MINT、PrISM、DSAC 等）随机采样的随机源。PrISM 论文按 MINT 的假设采用 7-bit TRNG（Katz 等混沌差分电流模式设计，[39]），消耗 90µW 静态 + 200µW 动态（共 290µW），比 DRAM 芯片功耗（Micron 功率计算器估计约 245mW）低三个数量级。除专用 TRNG 电路外，近年研究利用 DRAM 自身随机物理现象生成高熵随机数：QUAC-TRNG（ISCA 2021，[67]，四次行激活的随机失败）、D-RaNGE（HPCA 2019，[41]，降低激活延迟诱导失败）、以及多行同时激活的 in-DRAM TRNG 设计（[65]），PrISM 指出可复用这些设计。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PrISM 中的运转：每个 mitigation window 开始时，芯片内 TRNG 输出随机比特 → 选择 R 个激活槽位（W=72 中选 7 个）→ 采样到的行地址记入 SSQ。TRNG 的随机性质量决定采样均匀性与安全：若采样可预测，攻击者可定向躲避采样/交集。实现上放在 DRAM 芯片内、随激活逻辑工作（per-activation 逻辑落在 tRC 内、无 DRAM 时序开销），功耗极低（290µW vs 245mW 芯片功耗，三个数量级差）。
- 与缓解结构的关系：TRNG 是概率缓解"随机选择"的物理实现——MINT 用它选 1 个激活槽，PrISM 用它选 R 个激活槽；在-DRAM 实现避免了 host 侧随机源的调用延迟，且随机性不可被 host 预测（对抗知道防御、能构造访问模式的威胁模型）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现两类：(i) 专用 TRNG 电路（混沌/电流模式差分设计，7-bit 输出，290µW 总功耗）；(ii) 利用 DRAM 随机物理现象（如 QUAC-TRNG 的四行同时激活失败、D-RaNGE 的降低激活延迟诱导失败），无需额外模拟电路。PrISM 评估按 MINT 假设用 7-bit TRNG 计功耗，并指出可切换为 in-DRAM 现象型 TRNG 设计。使用场景：任何需要芯片内真随机数的安全/缓解机制（概率采样、随机化映射等）。

PuDGhost 视角（ISCA'26）：SiMRA-based TRNG 依赖"源列"（固定 SiMRA 输入下输出非确定、因电荷共享/感知近阈值 metastability 波动的列）作为熵源；PuDGhost 使 TRNG 熵受相邻行与非源列数据影响——相邻行干扰最坏 Row-1 把 Norm. Entropy 降到 0.35，并发列干扰最坏 Col-1 降到 0.07（即损失约 93% 熵），Col-0/Col-rand 亦降到 0.12/0.15。论文提出 Fixed 条件（相邻行与非源列数据与模式搜索阶段保持一致，即"固定数据"缓解）可保留熵，说明 SiMRA-TRNG 设计若不考虑 PuDGhost 会无意损失大部分随机性。
涉及论文标题：
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
