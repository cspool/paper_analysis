## 符号执行 + 模型计数（Symbolic Execution + Model Counting，TracerSym）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 符号执行（symbolic execution）：程序分析技术，用符号值（而非具体值）执行程序——维护 symbolic store（变量→符号表达式映射）与 path constraint（当前控制流路径分支条件的公式）；遇到符号操作数的条件分支时 fork 符号状态并分别探索可达路径（SAT 检查）。模型计数（model counting）：统计布尔/SMT 公式满足赋值的个数（#SAT）。两者组合可精确计算"每个可观测路径对应多少个秘密输入"。Helium 的 TracerSym 把两者结合用于侧信道泄漏量化：符号执行探索程序路径并收集 µtrace 约束，模型计数把每条符号 µtrace 的满足赋值数除以秘密输入空间大小 2^{|ŝ|} 得精确 µtrace 概率（需秘密输入均匀分布，如密码学 key）。
- 逻辑链：TracerSym 扩展 Angr 符号执行引擎（https://angr.io，论文引 [85]）；SMT 求解用 Bitwuzla（https://github.com/bitwuzla/bitwuzla，[69]）与 CSB（[84]，bit-vector 计数采样工具）；模型计数用 Ganak（https://github.com/meelgroup/ganak，[83]）——概率精确模型计数器（probabilistic exact model counter）：对 CNF 公式 F 与置信参数 δ 返回计数，Pr[|Solutions(F)|=count] ≥ 1−δ（论文配置概率 0.95，实际 1650 个 benchmark 全部正确）。相关先例：Bao et al. 用近似符号执行+模型计数量化 cache/控制流泄漏（Abacus [13]）；Saha et al. 用近似模型计数求泄漏界（[80]）。
- Web 证据：Ganak（Sharma, Roy, Soos, Meel, IJCAI 2019, DOI 10.24963/ijcai.2019/163，概率组件缓存 PCC、DPLL 式计数、组件分解+缓存）；Angr（Shoshitaishvili et al., IEEE S&P 2016）；Bitwuzla（Niemetz & Preiner, CAV 2023）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Helium 的二进制分析框架中，TracerSym 的流程（论文图 5/图 6）：
```
输入：二进制 P、秘密输入列表、µobs functions F={f^{µobs}_{op0},...}
1. 初始化秘密为符号位向量 ŝ，µtrace 约束集 C₀={True}
2. 符号执行（Angr）沿每条控制流路径：
   - 遇到第 j 个秘密相关 transponder（如 MUL）：
     取符号操作数 O_j，C_j = {φ ∧ c(O_j) | (φ,c) ∈ C_{j-1} × f^{µobs}_{op(j)},
                                 SAT(φ ∧ c(O_j))}        # UNSAT 丢弃（白节点）
   - 符号控制流分支：fork µtrace 约束集随符号状态
   - 路径终点：C_n 中每个 φ = 一条完整符号 µtrace
     （再与路径约束合取；论文 §VII 程序无秘密相关控制流）
3. 对每个 φ bit-blast 成布尔公式
4. Ganak 模型计数得满足赋值数 #(φ)
5. P(µtrace y) = #(φ_y) / 2^{|ŝ|}
输出：µtrace 概率分布 → PML → tail-bound (ε,δ)
```
- 具体例子（Case Study I Poly1305 zero-skip，64-bit 版 8 个 µobs）：128-bit key 符号化；逐秘密相关 MUL 与 f^{µobs}_{MUL}（第二操作数是否 0）合取、SAT 裁剪；8 条完整 µtrace；Ganak 计数得概率分布，ε=1.35×10⁻⁹、δ=9.39×10⁻¹⁰。运行时统计（表 III）：zero-skip 54s/225 个 SMT 查询/8 个 MC 查询；digit-serial 735s/36,618 SMT/660 MC。
- Annotations：bit-blast 把位向量约束转布尔 CNF 供 Ganak 计数；模型计数假设秘密均匀分布（对密码学 key 成立），非均匀输入需 TracerSim；SMT 查询数随插桩指令数与每 µobs function 的 µobs 数指数增长（Case Study III：最坏指数、µtrace 集饱和后近线性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TracerSym 作为 Angr 的扩展（Python），拦截 transponder 指令做 µobs 约束注入；SMT 求解（Bitwuzla/CSB）+ Ganak 模型计数流水线；artifact（https://github.com/samanthaarcher0/Helium-Artifact）Docker 化，预编译 workload（Libsodium Poly1305、Firefox SVG 卷积、图像变换 kernel），run_case_study_I.sh（<6min）/II.sh（<1min）/III.sh（~1.5h）。
- 使用与局限：精确但受符号执行可扩展性限制——大程序路径爆炸、复杂路径约束压垮引擎（§VI-C）；密码哈希输出作操作数时 SMT 求解等于破解密码原语（TracerSym 不可行，换 TracerSim）；模型计数可扩展性受符号 µtrace 复杂度限制。TracerSym 在可解时原则上可达密码级概率（≤2⁻⁸⁰），但 Helium 定位是"愿牺牲绝对安全换性能"的设计空间，不提供密码学证明。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
