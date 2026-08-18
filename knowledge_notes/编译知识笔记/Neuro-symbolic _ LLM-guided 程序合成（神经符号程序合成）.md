## Neuro-symbolic / LLM-guided 程序合成（神经符号程序合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
神经符号程序合成（Neuro-symbolic Program Synthesis）是结合 LLM/神经网络（神经部分：提出候选结构/模式）与符号引擎（SMT/SAT 求解器：验证正确性）的程序合成方法。核心模式：LLM 基于训练先验提出语义上合理的候选（能捕捉难以枚举的结构化代数变换），符号求解器验证等价性，验证失败的反例（counterexample）喂回 LLM 形成反例引导闭环（CEGIS 式）。动机：程序合成理论上 $\Sigma_3^0$-complete [40]，纯符号/枚举搜索对长序列、非显然算法模式（如乘法分解）指数爆炸；近期工作显示 LLM 先验可大幅缩小搜索空间（Olympiad 不等式证明 [42]、Alphaevolve [51]、Olympiad geometry [73] 等，[54] 为早期神经符号范式）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
æSIP 把 LLM（Claude Opus 4.5）作为第四个搜索算法加入 Greenthumb 重写规则发现框架（§IV-A2，Fig.5 紫色框）：
```
① warm-start：用已有 RISC-V 软件仿真库（RV32I 的 mul/div/rem 例程）中的专家验证重写
   示例引导 LLM 泛化常见模式
② LLM 提出候选指令序列：利用已知算法知识（如 mulh 的 convolution 与 Karatsuba 分解
   [78]）——这类分解对 brute-force 枚举/符号搜索不实际
③ SMT 验证引擎（Z3）验证候选与原始指令序列的状态等价性（State = R × M）
④ 反例（counterexample）喂回 LLM → 新一轮候选 → 数轮内收敛
对比：传统三算法（symbolic/stochastic/enumerative）对简单算术分钟级出结果，但对 mulh/div
等复杂指令 64 并行实例下 12 小时超时；LLM 搜索是唯一能收敛的算法
```
规则按来源标记：G = Greenthumb 算法发现，L = LLM 发现（Table II，如 blt→bge+jal、mulh 序列、div 序列、lh 内联等）。局限：LLM 对训练数据少的 bespoke/novel ISA 效果较差；Greenthumb 语义库不跟踪 PC/分支目标，分支规则用端到端仿真验证而非形式证明。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LLM API（论文用 Claude Opus 4.5）+ Z3 SMT 求解器 + Greenthumb 框架集成；候选生成与验证交替迭代（CEGIS 闭环）。使用：离线一次性为 ISA 生成规则库，之后由 equality saturation 阶段消费（规则库 51 条覆盖 26 指令，一次合成全程复用）；相关替代：Ruler [50]/Enumo [52]（符号重写规则推断，高层面代数域）、Sail [6]/ILA [37]（系统 ISA 建模无搜索）、Alive（provably correct peephole optimizations，PLDI 2015，同一"合成+验证"思想用于编译器优化规则）。评估口径：规则正确性由 SMT 形式验证（分支类端到端仿真）保证，规则有效性由重写后 ASIP 面积/功耗下降衡量（几何平均面积 -17.0% vs PDAG）。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
