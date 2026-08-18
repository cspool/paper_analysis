## 零知识证明（ZKP）与 zk-SNARK / zk-STARK

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ZKP 是密码学协议：prover 向 verifier 证明某个关于私有 witness 的 statement 为真，而不泄露 witness 本身（零知识性）。现代应用使用 succinct 形式：zk-SNARK（Succinct Non-interactive Arguments of Knowledge，证明大小 polylog/常数、验证毫秒级、可非交互）与 zk-STARK（Scalable Transparent Arguments of Knowledge，透明设置、抗量子，但证明对数级更大、验证更慢）。证明生成（prover）计算量大，是硬件加速的目标。GenZA 采用标准三层视图描述现代 ZKP 证明生成：算术化（Arithmetization）→ 多项式交互式预言机证明（PIOP）→ 多项式承诺方案（PCS）；PCS 经 Fiat-Shamir 变换后协议变为非交互。
- 本论文角色：GenZA 面向三类代表性协议 Groth16/HyperPlonk/Plonky2 的证明生成阶段，按三层视图识别出 dominant kernels（NTT、MSM、sumcheck、Merkle tree、多项式运算、hash），并设计统一可重构硬件覆盖这些 kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 证明生成 pipeline（以一个电路 statement 为例）：
```
1) 算术化：把程序/电路展平成代数约束（R1CS 或 Plonkish 表格），生成 witness 向量 w
2) PIOP：把约束编码成多项式，prover 对 witness 多项式求值/承诺，
   与 verifier 交互若干轮（sumcheck/NTT 域切换）验证低度、置换、积/零性质
3) PCS：绑定多项式并打开选定点（KZG 用 MSM+配对，FRI 用 Merkle 树+哈希）
4) Fiat-Shamir：用哈希挑战替代交互，输出非交互证明
```
- Annotations：各协议只替换某层——Groth16 用 R1CS+NTT+MSM(KZG)，HyperPlonk 用 Plonkish+sumcheck+MSM(KZG)，Plonky2 用 Plonkish+NTT+Merkle/FRI。证明生成的时间占比（Table II）：MSM/Merkle 各占 59–70%，NTT/sumcheck/多项式各占 1–33%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 上用 libsnark/Jsnark（Groth16）、官方 HyperPlonk/Plonky2 库；GPU 用 GZKP/cuZK/plonky2-gpu 等；ASIC 用 PipeZK/SZKP/zkSpeed/UniZK/LegoZK/GenZA 等加速器。使用场景：隐私区块链/rollup（Zcash、zkEVM）、可验证云计算、ZKML、匿名投票、递归证明组合（内层 Plonky2 快协议 + 外层 Groth16 恒定大小验证）。硬件侧关键点：不同协议场/bitwidth/kernel 差异大（64-bit Goldilocks 到 768-bit EC 域），单一专用单元无法覆盖全部，需统一可重构架构。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
