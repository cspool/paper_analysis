## Sumcheck 协议

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sumcheck 是证明布尔超立方上多线性多项式之和 ∑_{x∈{0,1}^n} g(x) = C 的交互协议（Lund-Fortnow-Karloff-Nisan 1992），是 MLE 类 PIOP（HyperPlonk、Spartan）的 prover 核心。协议 n 轮：第 1 轮 prover 算 g1(x1)=∑_{x2..xn} g(x1,...,xn)，verifier 检查 g1(0)+g1(1)=C0=C 并发挑战 r1 固定 x1；第 i 轮算 gi(xi)=∑_{x_{i+1}..xn} g(r1,...,r_{i-1},xi,...)，检查 gi(0)+gi(1)=C_{i-1} 并发 ri；Fiat-Shamir 后非交互。prover 计算本质是向量逐元素求和（访存密集型，算术强度低）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- prover 一轮的向量计算例子（g = qL·wa + qR·wb + ... 因子多项式之和）：逐轮对剩余变量求和，每轮把因子多项式逐元素相乘相加。GenZA 用两个优化压缩访存：(1) 延迟绑定（delayed binding）——稀疏 0/1 系数与稠密挑战分开存储，逐轮 on-the-fly 绑定（只在向量长度足够短后才物化稠密 gi），2^23 实例流量 2.9→0.7 GB；(2) 等号多项式空间压缩（eq-poly space reduction）——e~q(w,X) 用 O(√N) 工作存储 on-the-fly 求值替代 O(N) 物化，再省 1.3× 流量。
- Annotations：zkSpeed 的固定功能 sumcheck 单元把所有系数当稠密场元素处理、忽略稀疏性；NoCap 的 64-bit 向量单元使延迟绑定收益天然小（收益正比于场元素 bitwidth）。GenZA 首次在专用加速器中实现这些算法优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 库（HyperPlonk 官方实现、Speeding Up Sum-Check Proving [4]）；硬件按向量 PE 映射（逐元素加/乘、树式求和、分段并行 segmented-parallel 处理串行归约链）。使用：MLE 类 PIOP（HyperPlonk/Spartan）的 prover 主计算；访存受限，硬件需减少 off-chip 流量与提高 PE 利用率。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
