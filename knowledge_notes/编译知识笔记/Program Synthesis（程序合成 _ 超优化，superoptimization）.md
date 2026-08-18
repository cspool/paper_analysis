## Program Synthesis（程序合成 / 超优化，superoptimization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
程序合成（Program Synthesis）是自动从规范（spec）构造满足该规范的程序/代码片段的研究领域，广泛应用于代码超优化（superoptimization）：系统性地搜索语义保持的程序变换，找到比编译器产出更优的实现。核心形式：合成指令序列 $S_{\mathrm{synth}}$ 使对所有输入状态 $\sigma$ 满足 $[S_{\mathrm{spec}}](\sigma) = [S_{\mathrm{synth}}](\sigma)$。三类主流方法（论文 §III-A）：①符号合成（symbolic，Sketch [69]/Rosette [72]）——把搜索编码为 SAT/SMT 约束，构造性证明"存在程序满足语义"，算术层级 $\Sigma_3^0$-complete（Kim, arXiv 2024），开销高；②随机/启发式合成（stochastic，STOKE [64]）——Monte Carlo/模拟退火探索程序空间，以最优性换可扩展性，能发现符号方法难触达的非显然重写；③枚举合成（enumerative）——有界语法枚举候选表达式，最坏指数但近年 pruning/version-space 技术使其对领域特定指令级重写实用 [3]。相关：反例引导归纳合成 CEGIS（Abate et al. CAV 2018）、Alive（Lopes et al. PLDI 2015，provably correct peephole optimizations）、Ruler [50]/Enumo [52]（重写规则推断，高层面代数域）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
æSIP 用程序合成自动发现 ISA 指令重写规则（而非手工编写，手工对 RV32IM 每条指令繁琐易错 [43]）——这是编译优化规则库的自动生成环节（Fig.5，一次性离线执行）：
```
① RISC-V 验证引擎（Greenthumb 数据结构上实现）：
   State = R × M（R: f:B^5→B^32 寄存器映射、M: f:B^32→B^8 内存映射）
   每条指令 [inst]: State→State；指令序列复合 [S] = [inst_n]∘...∘[inst_1]
   等价性：S_spec ≡ S_synth ⟺ ∀σ, [S_spec](σ) = [S_synth](σ)   （Z3 SMT 证明）
② 搜索算法（4 个）：
   - Greenthumb 继承：symbolic / stochastic / enumerative（简单算术分钟级出结果）
   - 新增 LLM-guided（Claude Opus 4.5，见"Neuro-symbolic 程序合成"条目）
   - 关键约束：候选指令集排除原指令（强制用替代序列合成原行为）；优先利用不同硬件单元的
     指令子集（保证原指令的硬件可安全移除）
③ 产出：51 条重写规则覆盖 26 条指令（Table II：I-type 8、Arithmetic 17、Branch 14、
   Mult/Div 8、Ld/St 4；如 mulh→Karatsuba/convolution 风格 mul 序列、blt→bge+jal、
   lh→lw+slli+srai）
```
规则随后供 equality saturation（e-graph）消费，实现"编译优化"：把 mul 等复杂指令从程序中重写掉，使对应硬件可从处理器中移除。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：æSIP 构建在 Greenthumb 超优化器框架（https://github.com/upenn-acg/greenthumb，论文 ref [56]）之上——它提供寄存器/内存/程序 flag/指令语义库与三种搜索算法；æSIP 在其上实现 RISC-V simulator+validator（Z3 验证）并增加 LLM 第四搜索算法。Rosette v1.1 + Racket 6.7（https://github.com/emina/rosette）用于符号合成。使用：对给定 ISA 一次性离线发现规则（论文 64 并行实例下 mulh/div 传统算法 12 小时超时，LLM 搜索数轮收敛）；对分支类规则（Greenthumb 不跟踪 PC/分支目标、无法形式验证）改用端到端仿真验证。局限与替代：Sail [6]/ILA [37] 提供更系统 ISA 建模但无内建搜索算法；Ruler/Enumo 有搜索策略但面向高层面代数域而非寄存器级 ISA 语义。可扩展性：换 ISA 只需实现对应 ISA 语义（Greenthumb 已支持 ARMv7 与 GA 18-bit stack ISA [32]）。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
