## Pointwise Maximal Leakage（PML，逐点最大泄漏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PML 是 Saeidian、Cervia、Oechtering、Skoglund（KTH）在 ISIT 2022（arXiv 2205.04935，期刊版 IEEE TIT 2023, DOI 10.1109/TIT.2023.3304378）提出的信息论泄漏度量。它把"单个观测"的泄漏定义为：观测到 y 后攻击者猜秘密 x 的后验/先验最大乘性增益，$\ell_{P_{XY}}(X \to y) = \log \max_{x: P_X(x)>0} \frac{P_{X|Y=y}(x)}{P_X(x)}$。与平均类指标（互信息、最大泄漏）不同，PML 把泄漏视为随机变量（其分布由观测分布 P_Y 诱导），从而支持把隐私保证表达为泄漏分布的统计性质。Helium（ISCA 2026，Stanford）是第一个把 PML 应用于硬件侧信道泄漏量化并给出可计算方法的工作（论文引 [79] 即 PML 原文献）。
- 逻辑链：PML 的定义需要"观测"级别的后验 P_{X|Y=y}(x)——这要求知道程序级观测分布；Helium 用 µobs functions 建模指令级可观测执行、用 Tracer 计算 µtrace 概率分布，从而把 PML 落到可计算处；在确定性信道下（Helium 的默认威胁模型，泄漏函数是操作数的确定函数），PML 简化为 $\ell(y) = -\log P_Y(y)$——即观测概率越低、泄漏越大。例：32-bit 均匀秘密下优化 2 的 y₁（x=0）PML=log(2³²)=32（泄漏全部 32 bit），y₂（x≠0）PML=log(2³²/(2³²−1))≈3.36×10⁻¹⁰。
- Web 证据：https://arxiv.org/abs/2205.04935（ISIT 2022 论文）；https://dl.acm.org/doi/abs/10.1109/TIT.2023.3304378（TIT 期刊版）。PML 定义、动机（最大泄漏是平均保证、无法区分"个别观测完全泄露秘密"的通道）与隐私保证（把泄漏看作随机变量的统计性质）均与 Helium 论文描述一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 Helium 的泄漏量化 pipeline 中，PML 是"度量-建模-分析"三部分的最顶层输出标准。pipeline 计算流程（确定性信道）：
```
输入：程序 P、秘密输入分布 X、µobs functions F
1. Tracer 计算每个程序级观测（µtrace）y 的概率 P_Y(y)
   （TracerSym：符号执行+模型计数得精确概率；
     TracerSim：Monte Carlo 频率估计+Clopper-Pearson 保守界）
2. 每条 µtrace 的 PML：ℓ(y) = -log P_Y(y)        # 确定性信道简化式
3. 构造 tail-bound 保证：选"可容忍划分"中概率最低的 µtrace 为 ε-µtrace，
   ε = ℓ(ε-µtrace)，1-δ = 可容忍集概率和
输出：P_Y[ℓ(Y) ≤ ε] ≥ 1-δ
```
- 具体例子（论文 §VII-A Poly1305，zero-skip 乘法）：128-bit 均匀 key 下 TracerSym 得到 8 条 µtrace；概率最低（最不可容忍）的 µtrace 是"所有秘密相关乘法都非零"的情形，其概率 1−9.39×10⁻¹⁰，PML=−log(1−9.39×10⁻¹⁰)≈1.35×10⁻⁹ bit；其余更高泄漏的 µtrace 总概率 ≤9.39×10⁻¹⁰ ⇒ 保证 P[ℓ≤1.35×10⁻⁹]≥1−9.39×10⁻¹⁰。对照：digit-serial 乘法下高泄漏 µtrace 概率和达 0.49，得 P[ℓ≤0.97]≥0.51——同一程序、不同硬件优化，PML 分布完全不同。
- Annotations：ℓ(y) 对观测 y 逐点计算而非平均；ε 是"可容忍"泄漏上界（用户可接受的最高逐点泄漏），δ 是高泄漏观测总概率上界；ε-µtrace 的选择构造出"泄漏≤ε 的概率≥1−δ"这一最可解释的保证形式。确定性信道假设使 PML 只依赖观测概率，无需显式后验。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上需要"观测概率分布"，这正是 Helium Tracer 的产出：TracerSym 用 Angr 符号执行把秘密符号化、逐 transponder 与 µobs 约束合取、Ganak 模型计数得精确概率；TracerSim 用 Intel Pin 动态插桩逐 trial 记录 µtrace、频率/N 估计概率（支持任意输入分布，如非均匀秘密）。Helium 用它输出 tail-bound 隐私保证（§VI-B），并给出 Clopper-Pearson 95% 置信与 Rule of Three（单 µtrace 时未观测事件概率<3/N）的保守统计版本。
- 使用场景：硬件/软件设计者权衡"接受多少小概率泄漏以换取多少性能"——如论文 Case Study IV 中，Chacha20-Poly1305 接受 P[ℓ≤0.0004]≥0.9997 即可省去 cio 缓解的 mul64 2.31×、cs32 3.37× 开销；Ed25519 按函数分解（表 VII）选择性缓解。限制：TracerSym 受符号执行可扩展性限制（路径爆炸、密码哈希不可符号化）；TracerSim 无法达到密码级（≤2⁻⁸⁰）保证——Helium 明确不提供密码学证明，面向"愿牺牲绝对安全换性能"的设计空间。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
