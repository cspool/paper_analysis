## Matsumoto-Amano（MA）Normal Form（单比特 Clifford+T 序列范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MA Normal Form 是任意单比特 Clifford+T 门序列的规范（canonical）表示，由 Matsumoto & Amano（arXiv:0806.3834）与 Giles & Selinger（arXiv:1312.6584）确立：任何单比特 Clifford+T 算子可唯一地写成 $$(T|\epsilon)\,(HT|SHT)^*\,C$$ 的形式——可选的初始 T 门、紧跟若干 HT 或 SHT 模式、最后以一个 Clifford 门 C 结尾。两个性质：①保证最小 T 门数（minimal T-count）；②对任意目标酉给出唯一分解。因此任意 Rz(θ) 合成出的单比特 Clifford+T 序列都能被"压平"成该范式，把优化空间从任意长度的门串收敛到有限模式串。
- 论文用途（TACO）：把电路中所有单比特门序列转成 MA Normal Form 后，Clifford 门消除被隔离成两个子问题——Toffoli 分解内的冗余 Clifford 与单比特序列内的 Clifford（S/H/Pauli），使 TACO 能在 O(n) 内逐模式化简而不破坏门级并行。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TACO 转译器中的运转流程（示例序列 T S H T H T S H T，式 2）：
```
① 转 MA Normal Form：T S H T H T S H T → 规范模式序列
② 消 Phase(S) 门：利用恒等式 TS = T†Z（逐矩阵验证：
   TS=diag(1, e^{iπ/4})·diag(1,i)=diag(1, e^{i3π/4})，
   T†Z=diag(1,e^{-iπ/4})·diag(1,-1)=diag(1, e^{i3π/4})），
   把每个 S 门替换成 "免费"虚拟 Z 门（Z 门可零成本执行/吸收进 Pauli frame）
③ 消 Pauli(Z) 门：用交换律 ZH=HX、ZT=TZ、ZT†=T†Z、XT=T†X、XT†=TX
   把所有 Pauli 门换到序列末尾合并（式 6）
④ 消 Hadamard(H) 门：引入硬件原生 Rx(π/4) 门，用 HT=Rx(π/4)H、
   HT†=Rx†(π/4)H、HH=I 从左到右扫描，把 H 推到末尾或与相邻 H 相消（式 7）
⑤ 输出：只剩 T/T† 与 Rx(π/4)/Rx†(π/4)（后者的实现代价与 T 相同，
   仅区别在于 T 与 magic state 在 Z 边做 lattice surgery、Rx 在 X 边做）
```
- 效果：18 比特 QFT 的 104,217 个 Clifford+T 门经此化简后 Clifford 门降 98.6%，且全部操作权重 ≤2（不产生 PBC 式的高权重旋转）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：作为转译器内单比特门序列重写 pass。对每个连续单比特门片段：①按 Clifford+T 群的单比特表示（T、S、H 生成）将序列规范化到 MA Normal Form（可用单比特 Clifford+T 的群论/表查找实现，n=1 情形群较小、可枚举）；②按上述三组恒等式逐模式消去 S/Pauli/H。TACO 中三步均为 O(n)，n 为电路门数；配合 FTQC 动态分解（先降 Rz 数）与 GridSynth 合成（ε=10⁻¹⁰）构成完整 Clifford+T 转译流水线，集成在 NWQEC（nwqec-cli）。使用场景：任何"先合成 Clifford+T、再消除多余 Clifford"的 FTQC 编译流程。

涉及论文标题：
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
