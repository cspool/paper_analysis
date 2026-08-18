## Fiat-Shamir 变换

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Fiat-Shamir 把交互式证明变成非交互式：用对"当前协议状态（前几轮消息+statement）的哈希"作为 verifier 随机挑战，替代真实 verifier 的随机数。在 ZKP 中，PIOP/PCS 的每轮挑战（如 sumcheck 的 r_i、KZG 打开点）都由哈希（如 SHA-256、Poseidon）从已通信消息派生，使证明可在单条消息中发出（非交互），是 zk-SNARK/STARK 可落地的关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- sumcheck 中的 Fiat-Shamir 例子：prover 算 g1(x1) → 挑战 r1 = H(statement || g1) → 算 g2(x2) → r2 = H(statement || g1 || g2) → ... 取代"verifier 随机发 r_i"。硬件影响：哈希挑战形成严格串行点——后续 kernel 依赖该挑战，故 GenZA 的调度器把 Fiat-Shamir 变换（hash 计算状态生成下一挑战）识别为一类 cut 点，kernel 流水线必须在此处序列化。
- Annotations：GenZA 为 HyperPlonk 内置 SHA3 core（<1% 时间）做 Fiat-Shamir，Poseidon 哈希也可用于此；挑战的串行依赖限制了跨 kernel 流水线的自由度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：通用哈希（SHA-256/SHA-3）或 ZK 友好哈希（Poseidon）；在 libsnark/HyperPlonk/Plonky2 中以"transcript"对象累计消息并派生挑战。使用：所有非交互 ZKP 的标准组件；加速器上以专用小 hash core 或复用向量 PE 实现（GenZA 用 SHA3 core，面积 0.01mm² 级）。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
