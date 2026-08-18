## Poseidon 哈希函数

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Poseidon 是面向 ZKP 的 ARX 类哈希（Grassi-Khovratovich-Rechberger-Schofnegger, USENIX Security 2021）：把消息映射到有限域元素并用域算术（S-box + 线性层）做压缩，比 SHA-2 在电路规模/证明成本上低数量级，广泛用于 Merkle 树、Fiat-Shamir 挑战、zkEVM。GenZA 遵循 Plonky2 设置：64-bit Goldilocks 域、状态宽 t=12、x^7 S-box，含若干 full 与 partial rounds，底层原语为 S-box 与稠密/稀疏 MDS 矩阵-向量乘。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一轮计算例子：S-box 层把每个状态元素求 x^7（4 个乘法器的快速幂链，x^7=x^4·x^2·x）；稠密 MDS 矩阵-向量乘 t×t·state 拆成 t=12 个独立点积，每个点积长 12 由 4 个乘法器+归约链 3 轮完成，多点积流水；稀疏 MDS 乘拆成若干点积与逐元素乘。GenZA 把每 PE 分成 8 个 1D Poseidon 单元（各 4 个 64-bit 乘法器匹配 Goldilocks），对比 UniZK 的 2D systolic 在 full round 损失 25% 利用率，1D 向量更紧凑。
- Annotations：Poseidon 的 MDS 常数预装进 PE scratchpad；S-box 的乘法链充分使用 4 乘法器避免串行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Plonky2/starky 内置 Poseidon 参考实现；硬件用向量/systolic 阵列（UniZK 2D、GenZA 1D 向量）。使用：Plonky2/FRI 类协议的 Merkle 树哈希与 Fiat-Shamir（GenZA 为忠实匹配 HyperPlonk 另留 SHA3 core，但 PE 原生支持 Poseidon）；hash 类 kernel 算术强度高（191 modmul/元素）但需大量并行向量单元。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
