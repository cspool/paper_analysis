## µtrace（微架构观测轨迹，microarchitectural observation trace）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- µtrace 是 Helium 的程序级攻击者观测抽象：victim 程序执行时，各指令与硬件侧信道交互、按 µobs functions 暴露指令级 µobs，这些 µobs 组成的序列（整个程序跨全部动态 transponder）即一条 µtrace。Helium 采用强攻击者模型：攻击者可观测完整 µtrace（cycle-accurate 的指令执行路径部分序），更弱攻击者（如只看总执行时间）是 µtrace 的非恒等函数（§VIII 用 observer function 后处理分组）。
- 逻辑链：程序语义决定指令操作数分布 → 操作数分布经 µobs functions 诱导指令级 µobs 概率 → µobs 概率组合成完整 µtrace 概率。计算 µtrace 概率分布正是 Helium Tracer 的任务：TracerSym 用符号执行+模型计数精确算，TracerSim 用 Monte Carlo 近似。µtrace 概率分布再输入 PML 计算（确定性信道下 ℓ(y)=−log P_Y(y)）。
- Web 证据：论文图 5/图 6 展示 µtrace 概率计算的两条路径（TracerSym 符号 µtrace 树 + 模型计数；TracerSim Monte Carlo）；Helium-Artifact 仓库实现。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件架构分析中，µtrace 把"硬件数据相关优化 + 程序"组合的泄漏行为折叠成可计数的观测空间。TracerSym 的构造流程（图 6）：每条控制流路径一棵 µtrace 树——根为秘密符号 ŝ 与 C₀={True}；遇到第 j 个秘密相关 transponder（如 MUL）时，把 C_{j-1} 中每个公式与 µobs function 的每条约束 c(O_j) 合取、SAT 检查（SAT 黑节点保留进 C_j、UNSAT 白节点丢弃）；路径终点 C_n 中每个公式是一条完整符号 µtrace（描述产生该具体 µtrace 的秘密输入值集合）；bit-blast + Ganak 模型计数得秘密输入个数，除以 2^{|ŝ|} 得概率。例：Poly1305 zero-skip 下 8 条 µtrace（8 个"哪些乘法取零值"的组合）；SVG 卷积 zero-skip 下 8 条 µtrace 各 PML=3（每条都全泄漏 3-bit 秘密）。
- 具体例子（Chacha20-Poly1305 mul64，TracerSim）：Intel Pin 插桩 72 条动态 MUL 指令，每 trial 记录各 MUL 操作数值 → 经 f^{µobs}_{MUL} 得该 trial 的 µtrace；10,000 trials 下仅见单一 µtrace ⇒ Rule of Three 保守界。
- Annotations：µtrace 是"逐动态 transponder 的 µobs 序列"，不是逐指令的完整指令 trace；µtrace 可能重叠/不相交/完全覆盖秘密的不同部分（§III-B 强调程序级分析优于逐指令分析的并集，避免高估泄漏）；TracerSym 对每控制流路径独立建树，秘密相关控制流下路径约束也并入 µtrace。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TracerSym 扩展 Angr 符号执行引擎（Bitwuzla/CSB 求解、Ganak 模型计数，概率 0.95）；TracerSim 用 Intel Pin 动态二进制插桩写 pintool，逐 trial 记录动态 transponder 操作数并生成具体 µtrace，频率/N 估计概率，两轮采样 + Clopper-Pearson 95% 置信（单 µtrace 用 Rule of Three 3/N）。存储可用碰撞抗性哈希与无损压缩降低（§VII-D）。
- 使用：µtrace 分布是 PML/tail-bound 的输入；也可后处理为弱攻击者观测（按端到端延迟分组 µtrace，§VIII）；Helium artifact 的四个 run_case_study_*.sh 脚本复现四组 µtrace 概率计算（Case Study I <6min、II <1min、III ~1.5h、IV <6h，Docker 内运行）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
