## 多项式承诺方案（PCS）：KZG 与 FRI

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PCS 是 ZKP 三层视图的第三层：把多项式绑定成承诺，之后用简洁证明打开选定的求值。两种主流实例：(1) 配对型 KZG（Kate 等）——基于椭圆曲线与配对，需可信设置，常数大小证明、极快验证，典型用 MSM kernel；(2) 哈希型 FRI——基于 Merkle 树承诺与低度测试，透明（无需可信设置），但证明对数级更大、验证哈希重。PCS 结合 Fiat-Shamir 使协议非交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- KZG 承诺例子：对多项式 f(x)=Σ f_i·x^i，承诺 C = Σ f_i·[τ^i]_1（τ 为 SRS 秘密，[]_1 为群元素）——这是对系数向量的 MSM；打开 f(α) 时验证配对等式 e(C−f(α)G, H)=e(w, τH−αH)。FRI 例子：把 f 的 Merkle 根作承诺，逐轮把多项式折叠并抽样承诺，最后做低度测试。
- Annotations：KZG 的 MSM 是 Groth16/HyperPlonk 的最大 kernel（占 prover 时间 59–70%）；FRI 的 Merkle 树+哈希是 Plonky2 的最大 kernel（68.84%）。GenZA 对 MSM 做动态 window sizing 与 window-major 映射，对 Merkle/哈希用向量 PE+树分片。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：KZG 用 libsnark/Bellman（zkcrypto）、MSM 加速用 PipeZK/SZKP/GenZA；FRI 用 Plonky2/starky、Merkle 哈希用 Poseidon（Plonky2）。使用场景：按应用权衡选择——加密货币/rollup 偏爱 KZG 小证明快验证（接受可信设置），可验证云计算/ZKML 偏爱 FRI 透明设置快证明。硬件：PCS 阶段 kernel 决定加速器的主要单元需求（EC 运算 vs 哈希/向量）。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
