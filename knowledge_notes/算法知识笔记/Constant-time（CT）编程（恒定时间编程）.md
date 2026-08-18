## Constant-time（CT）编程（恒定时间编程）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Constant-time 编程是侧信道防御的黄金标准：编写代码时避免把秘密值传给"不安全指令"的"不安全操作数"——历史上不安全操作数限于内存地址、分支操作数与少数可变时算术指令（如除法），因为这些指令在秘密相关时会产生数据相关的硬件资源占用（执行时间、cache 行为等）可被攻击者观测。Helium 论文（§I）指出：现代微架构采用越来越多的数据相关优化（计算简化 zero-skip、流水线/寄存器文件压缩、silent stores、计算复用、值预测、数据内存相关预取等），威胁使所有指令操作数都可能不安全，CT 编程变得不可能（Opening Pandora's Box, ISCA'21 的论点）。
- 逻辑链：CT 保证"零泄漏"（秘密永不经不安全操作数）→ 需要安全指令集合仍足够大以变换秘密值；随数据相关优化扩散，安全指令集合萎缩，CT 成本上升甚至不可行 → 催生替代策略：硬件 ISA 扩展（Arm DIT、Intel DOIT、RISC-V Zkt/Zkvt，粗粒度关闭优化）与软件细粒度变换（如 cio 的二进制码变换）；Helium 则提供"有界泄漏"（bounding leakage）路径——不追求零泄漏，而是量化泄漏概率供设计者权衡。
- Web 证据：常数时间编程为公开通用概念（Almeida et al., "Verifying Constant-Time Implementations", USENIX Security 16，论文引 [5]）；Intel/OpenSSL 均有官方 CT 指南。论文 §I 引 [37]（cio）报告软件侧 CT 变换开销可达 28×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在秘密处理程序的算法 pipeline 中，CT 编程是对"秘密数据流向不安全指令操作数"这一数据依赖的约束。伪代码层面的 CT 模式：
```
# 非 CT（泄漏）：用秘密作分支条件
if secret_bit == 1:
    y = table_a[x]
else:
    y = table_b[x]

# CT 变换：消除秘密依赖的地址/分支（位掩码选择）
mask = 0 - secret_bit        # secret_bit∈{0,1} ⇒ mask=0 或 0xFF..F
y = (table_a[x] & mask) | (table_b[x] & ~mask)   # 两条路径都执行
```
- Helium 论文的动机例子：zero-skip 优化使算术指令（历史上"安全"、CT 代码可放心传秘密）也变成 intrinsic transmitter——乘法器在操作数含 0 时走快速 µobs、否则慢速 µobs（图 1）；因此即使传统 CT 代码也可能在新微架构上泄漏。cio（论文 §VII-D 的 baseline）把不安全操作数变换为永不在不安全值集合取值（如 32-bit 减法：两操作数零扩展、第 33 位置 1、相减、取低 32 位），使所有指令恒走同一 µobs，但开销 2.31×–15.71×。
- Annotations：CT 的核心约束是"指令的时序/资源行为与秘密值统计独立"；位掩码选择示例消除地址/分支依赖但增加指令数与执行路径；zero-skip 例子说明"安全指令集合"随微架构演化而缩小——Helium 正是为评估这类新泄漏而设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现途径：① 手工编码纪律（crypto 库如 Libsodium 的既定做法）+ 工具验证（ctgrind、dudect、SideChannelMarvel 等）；② 编译器/语言级（FACT 语言、SynthCT 可移植 CT 代码合成）；③ 二进制码变换（cio，ASPLOS'24，github.com/counter-optimization，对 x86_64 与 libsodium 实现，处理寄存器溢出、地址计算与复杂指令微操作，发现首个微架构缓解组合安全问题的实例）；④ 硬件模式（DIT/DOIT/Zkt/Zkvt）。
- 使用场景：CT 用于需要绝对零泄漏的安全关键代码（密码学、密钥处理）；其成本随微架构数据相关优化增多而上升（Intel 警告 DOIT 未来处理器性能影响可能"显著更高"）；Helium 提供的替代路线是量化为 (ε,δ) 的有界泄漏——当程序员可接受极小概率的高泄漏（如 P[ℓ≤0.0004]≥0.9997）时可免去高开销缓解。局限（§III-A）：Helium 只覆盖非推测性泄漏（非 Spectre 类）与 intrinsic transmitter。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
