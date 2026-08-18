## 多项式交互式预言机证明（PIOP，Polynomial Interactive Oracle Proof）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PIOP 是 ZKP 三层视图的第二层：prover 把 witness 承诺为多项式集合，verifier 查询少量随机点上的求值，以高健全性检查低度关系、置换约束与积/零性质。prover 侧多项式计算两种方式：(1) 用数论变换 NTT 在系数域/求值域间切换，把多项式乘法变成逐元素操作；(2) 用布尔超立方上的多线性扩展（MLE）结合 sumcheck 协议，避免大 NTT。代表性 PIOP：Groth16 用线性 PCP（NTT 域）、HyperPlonk 用 MLE+sumcheck、Plonky2 用 PLONK 风格（NTT 域）+FRI。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PIOP prover 计算例子（多项式乘法域切换）：系数域多项式 a(x)、b(x) → NTT 转求值域 â、b̂ → 逐元素乘 ĉ=â⊙b̂ → 逆 NTT 回系数域得 c(x)=a·b。MLE+sumcheck 路径：把约束写成多线性多项式 g(x1..xn)，prover 对 g 在超立方上的和 ∑g 用 sumcheck 协议逐轮约简（见 sumcheck 条目）。
- Annotations：NTT 路径是计算密集型（logN 级蝴蝶）；MLE/sumcheck 路径是访存密集型（向量逐元素操作）。GenZA 中 NTT 与 sumcheck 都映射到同一批 PE 的不同模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Groth16（libsnark）、HyperPlonk（EspressoSystems/hyperplonk）、Plonky2（mir-protocol/plonky2）各自实现其 PIOP；硬件加速器把 PIOP 翻译成计算图（GenZA 手动翻译，可交给 ZKP 编译器）再调度到 PE。使用：决定 prover 的 kernel 构成（NTT vs sumcheck vs 多项式操作）与访存/计算特征，是加速器 kernel 映射设计的主要输入。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
