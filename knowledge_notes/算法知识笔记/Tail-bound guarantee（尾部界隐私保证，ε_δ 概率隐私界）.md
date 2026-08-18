## Tail-bound guarantee（尾部界隐私保证，ε/δ 概率隐私界）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tail-bound guarantee 是 PML 原文献（Saeidian et al., ISIT 2022, arXiv 2205.04935）提出的隐私保证形式：由于 PML 是观测 y 的函数、y 的分布是 P_Y，PML 可视为随机变量，隐私保证即对 PML 分布施加统计约束。tail-bound 要求"泄漏超过 ε 的观测出现概率低于 δ"：$P_Y[\ell(Y) \le \epsilon] \ge 1 - \delta$，即把观测划分为"good"（低 PML）与"bad"（高 PML）两类，bad 类总概率受 δ 约束。Helium 是第一个把 tail-bound 应用于硬件侧信道泄漏量化并给出具体 (ε,δ) 构造方法的工作。
- 逻辑链：平均类指标（互信息/最大泄漏）给单值平均、掩盖低概率高泄漏事件；tail-bound 把保证表述为"高泄漏以低概率发生"——更贴合安全实践者"程序必须以极高概率泄漏极少"的直觉。Helium 的核心贡献之一是给出了"可容忍划分"构造：当少数 µtrace 承载大部分概率质量（低泄漏）而其余罕见 µtrace 高泄漏时，选概率最低的容忍 µtrace 为 ε-µtrace，ε 取其 PML，1−δ 为可容忍集概率和，得到 ε 与 δ 都较小的可解释保证。
- Web 证据：PML 论文（https://arxiv.org/abs/2205.04935）第 IV-D 节定义 tail-bound；Helium 论文 §IV-D 与 §VI-B 使用该定义并给出构造与 TracerSim 统计版本（Clopper-Pearson 95% 置信、Rule of Three）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 Helium pipeline 中 tail-bound 的计算过程（§VI-B）：
```
1. 用 Tracer 得到 µtrace 概率分布 {P_Y(y)}
2. 对每条 µtrace 计算 PML ℓ(y) = -log P_Y(y)
3. 若存在"可容忍划分"：
   把 µtrace 按 PML 排序，找到可容忍（低泄漏）集合
   ε-µtrace = 可容忍集合中概率最低的 µtrace
   ε = ℓ(ε-µtrace)          # 所有可容忍 µtrace 的 PML ≤ ε
   1-δ = Σ_{y ∈ 可容忍集} P_Y(y)   # 高泄漏(bad)概率 ≤ δ
4. 输出保证 P_Y[ℓ(Y) ≤ ε] ≥ 1-δ
   （TracerSim 下：ε 用 ε-µtrace 概率的 Clopper-Pearson 下界
    的 -log；1-δ 用 bad 概率上界对应的下界，95% 置信；
    仅单 µtrace 时用 Rule of Three 3/N）
```
- 具体例子（论文 §VII-D，Chacha20-Poly1305 mul64，TracerSim N=10,000/轮）：两轮各 10,000 trials 均只观察到单一 µtrace ⇒ Rule of Three 得未观测 µtrace 概率 <3/10000=0.0003 ⇒ 保守估计该 µtrace 概率 ≥1−0.0003 ⇒ ε=0.0004 bits（−log(0.9997)）、1−δ=0.9997 ⇒ 保证 P[ℓ≤0.0004]≥0.9997。对照 cs64 类别：多 µtrace、高泄漏，得弱保证 P[ℓ≤2.1461]≥0.9552——提示程序员 cs64 仍需缓解。
- Annotations：ε（PML 阈值，bits）与 δ（超过 ε 的观测总概率）成对出现；点 (ε,δ) 构成可行保证集合（论文图 7 的 Poly1305 曲线，x 轴候选 ε、y 轴累计高泄漏概率）；"可容忍划分"不总存在——Ed25519/Argon2id 多数类别所有 µtrace 都高泄漏，此时无意义保证、须缓解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TracerSym 精确给出 µtrace 概率（需秘密输入均匀分布假设，符合密码学 key 场景）；TracerSim 用两轮 Monte Carlo（N₁ 定 ε 防选择偏差、N₂ 定 δ）+ Clopper-Pearson 二项置信区间产生保守 (ε,δ)，N₁/N₂ 可调提供运行时-精度权衡，trials 可并行。
- 使用场景：程序员把 (ε,δ) 作为"安全-性能权衡"的决策输入——若可容忍泄漏风险（如 Chacha20-Poly1305 的 mul64/cs32），免去 cio 缓解的 2.31×/3.37× 开销；若泄漏不可容忍（Ed25519/Argon2id），保留缓解或只缓解泄漏重的函数（表 VII 的 sc25519_reduce、ge25519_scalarmult_base）。限制：Helium 不提供密码学证明（TracerSim 到密码级 N 不现实），tail-bound 针对单一固定公共输入计算，未覆盖攻击者自适应选择公共输入的场景（§VIII 列为未来工作）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
