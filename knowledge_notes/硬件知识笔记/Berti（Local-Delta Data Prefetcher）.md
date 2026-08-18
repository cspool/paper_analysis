## Berti（Local-Delta Data Prefetcher）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Berti（Navarro-Torres、Panda、Alastruey-Benedé、Ibáñez、Viñals-Yúfera、Ros，MICRO 2022）是面向 L1D 的本地 delta 数据预取器：用 IP 定位（IP-localization）方法按指令分析每个 IP 的 delta 序列特征，只选最佳本地 delta（local delta：同一指令发出的 demand 访问间的差，区别于 BOP/MLOP 的全局 delta）。仅 2.55KB 存储，是"紧凑且高效"的 SOTA 记忆式 L1D 预取器代表（在 Moirai 对比中单核 10.48%、多核 7.3% 平均 speedup，DRAM traffic 最保守仅 +6.5%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Berti 运转流程：LSU 产生地址+PC → 按 IP 查表定位该指令的 delta 历史 → 统计/匹配本地 delta 模式（如 +64）→ 按匹配的 delta 生成预取。Moirai 用它做泛化 vs 记忆化的微观案例（Figure 15，页 0x1c757cf8000，429.mcf-192B）：在线性 stride 段 Berti 与 Moirai 都命中；在复杂多相/噪声段 Berti 的有用预取变稀疏（局部记忆模型被噪声/非线性打败），Moirai 继续命中——直接可视证据证明泛化优势。
- 对比：Berti 靠精确本地 delta 匹配，存储 2.55KB 小于 IPCP；在 0.8KB 约束下与 IPCP 一样性能骤降；Berti 的保守策略（最低 traffic）vs Moirai 的覆盖导向（+56.6% DRAM traffic 与 IPCP 相当，但需求流量降至 0.56×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：per-IP 状态表 + 本地 delta 统计/选择逻辑，ChampSim 实现（公开），评估用 SPEC CPU2017 + GAP；相对 IP-stride baseline +8.5%、相对 IPCP +3.5%，且因高准确率在内存层次节能 33.6%。使用：作为 SOTA 记忆式 L1D 预取 baseline（Moirai 选它正是因其 inner-cache 可部署性与高效率）。局限：同属记忆范式，复杂/新颖模式覆盖有限；PC 表开销在亚 KB 预算下受限。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
