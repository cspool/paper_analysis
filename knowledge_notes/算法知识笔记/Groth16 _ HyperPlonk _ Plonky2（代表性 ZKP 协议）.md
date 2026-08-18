## Groth16 / HyperPlonk / Plonky2（代表性 ZKP 协议）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 三协议覆盖现代 ZKP 的主要权衡维度（可信设置、证明大小、证明/验证时间），是 GenZA 的目标协议：(1) Groth16（EUROCRYPT 2016）——最优化 zk-SNARK，R1CS+线性 PCP+NTT+KZG，每电路可信设置，证明 3 个群元素、3 个配对验证，256/384/768-bit EC 域；(2) HyperPlonk（ePrint 2022/1355）——Plonkish+MLE/sumcheck+KZG，通用（电路无关）可信设置，近线性时间 prover，256/384-bit EC 域，无大 NTT；(3) Plonky2（Polygon Zero 2022）——Plonkish+PLONK 风格+FRI，透明设置，64-bit Goldilocks 域（p=2^64−2^32+1），面向快速递归，大证明但生成快。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 三协议 kernel 构成对比（Table II，2^20 门 mock circuit，80 线程 CPU 实测）：Groth16——MSM 69.73%、NTT 29.32%、多项式 0.96%；HyperPlonk——MSM 59.34%、sumcheck 33.49%、多项式 7.14%；Plonky2——Merkle 树 68.84%、多项式 14.17%、NTT 0.15%。位宽/场：Groth16/HyperPlonk 用 BN128/BLS12-381/MNT4-753（256–768 bit），Plonky2 用 Goldilocks（64 bit）。
- Annotations：三协议揭示"多样性"——硬件必须同时支持 64 到 768-bit 多 bitwidth、通用与特殊模、计算密集（MSM）与访存密集（NTT/sumcheck/多项式）kernel，这是 GenZA 统一架构设计动机的直接来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Groth16 用 libsnark/Jsnark（CPU）、GZKP（GPU）、PipeZK/SZKP/LegoZK（ASIC）；HyperPlonk 用 EspressoSystems/hyperplonk、zkSpeed（ASIC）；Plonky2 用 mir-protocol/plonky2、plonky2-gpu（GPU）、UniZK（ASIC）。GenZA 在一套 16×8 PE 阵列上运行全部三协议。使用场景：隐私币/rollup（Groth16 小证明）、可验证云计算/ZKML（HyperPlonk/Plonky2 快证明）、递归组合（内层 Plonky2+外层 Groth16）；云端需同时服务多协议客户，正是 GenZA 通用性价值所在。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
