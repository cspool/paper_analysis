## µobs function（微架构观测函数，microarchitectural observation function）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- µobs function 是 Helium（ISCA 2026，Stanford）提出的统一形式化：把"硬件侧信道如何产生指令级攻击者观测"编码为函数/约束公式集。一个 µobs function 把一个或多个 transmitter 的不安全操作数映射到单个 transponder 的、可观测且互不相同的微架构执行（µobs）。Helium 聚焦 intrinsic transmitter（指令自身产生执行可变性、同时也是 transponder），故 µobs function 映射"一条指令自己的不安全操作数 → 它自己的可观测执行路径"。例如 zero-skip 优化的 MUL：任一操作数为 0 时走快速 µobs1，否则慢速 µobs2（论文图 1）。
- 逻辑链：µobs function 建立在 leakage functions（RTL2µpath 论文 [48]）之上——transponder 的全部泄漏函数（其执行可变性的所有实例）合并抽象成一个 µobs function，输出即 transponder 端到端微架构执行路径；每个 transponder 恰好一个 µobs function，捕获其全部 µobs。其形式是"发射机操作数上的约束公式集合"，每个公式对应一个 µobs（如 AND 的 f^{µobs}_{AND} = {op1=0∨op2=0∨op1=0xFF..F∨op2=0xFF..F, op1≠0∧op2≠0∧...}）。
- Web 证据：Helium-Artifact 仓库（https://github.com/samanthaarcher0/Helium-Artifact）与论文 PDF（https://cs.stanford.edu/~trippel/pubs/archer_ISCA26.pdf）描述该形式化；leakage functions 出自 RTL2µpath（Hsiao et al., MICRO 2024，论文引 [48]）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件架构中，µobs function 抽象掉与泄漏无关的微架构细节，只保留"操作数等价类 → 可观测执行路径"的映射。论文图 4 给出四个 µobs function（伪代码形式）：
```
µobs zero_skip(MUL i):
  if (i.op1 == 0 ∨ i.op2 == 0): return µobs1   // 快速路径
  else: return µobs2                            // 慢速路径
µobs zero_all_ones_skip(AND i):
  if (op1/op2 ∈ {0, 0xFF..F}): return µobs1; else: return µobs2
µobs zero_one_skip(MUL i):
  if (op1/op2 ∈ {0, 1}): return µobs1; else: return µobs2
µobs digit_serial(MUL i):          // 按乘数最高字节为 0 分档
  if (i.op2[8:31]==0): return µobs1
  if (i.op2[16:31]==0): return µobs2
  if (i.op2[24:31]==0): return µobs3
  else: return µobs4
µobs bit_serial(DIV i):            // 按两操作数前导零差分 65 个 µobs
  if (i.op2==0 ∨ i.op1<i.op2): return µobs1
  else: for d in 0..63: if ((i.op1>>d)≥i.op2 ∧ (i.op1>>(d+1))<i.op2): return µobs_{d+2}
```
- 运转流程：硬件设计早期，设计者/厂商为每个数据相关优化给出 µobs function（比 DIT/DOIT 硬件模式多一层信息：DIT/DOIT 只标"不安全指令/操作数对"，µobs function 把操作数划分为产生不同 µobs 的等价类）；µobs function 输入到 Helium Tracer，经程序级分析输出泄漏保证，从而在设计早期评估该优化对不同程序的安全影响。例：SVG 卷积下 digit-serial 乘法对单字节像素恒走同 µobs（PML=0、性能更好），而 Poly1305 下 digit-serial 泄漏高（P[ℓ≤0.97]≥0.51）。
- Annotations：µobs 是"可观测执行路径"（cycle-accurate 部分序），也可推广为端到端延迟、功耗等可观测属性（§V）；µobs function 是确定函数（Helium 默认威胁模型），非确定性信道需扩展为观测上的概率分布；其输出可标注不透明标识符（µobs0、µobs1）保护厂商微架构细节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：µobs function 作为约束公式集合输入 Tracer——TracerSym 在符号执行中拦截 transponder，把公式与路径约束合取做 SAT；TracerSim 用 Intel Pin 拦截 transponder 指令、按操作数值经 µobs function 计算 µobs。来源上，µobs function 可由 RTL 合成（SynthLC [48] 已能从 SystemVerilog 自动合成泄漏函数签名；精确函数映射预期可用符号仿真/模型检验扩展）。厂商也可直接发布 µobs functions（用不透明标签）。
- 使用：Helium artifact（https://github.com/samanthaarcher0/Helium-Artifact，MIT，Docker 化）内置四组 µobs functions（zero-skip、digit-serial、bit-serial 及 Case Study IV 的 mul64/cs64/cs32 类别，对应 cio 论文 [37] 的计算简化优化）；用户改 µobs function 即可评估不同微架构优化。局限（§VIII）：需要硬件厂商披露比现有硬件模式略多的信息；复杂优化反推 µobs function 困难。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
