## 静态调度器与 kernel 模板（分析成本模型 + 离线调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是面向空间加速器（CGRA/统一 PE 阵列）的软件侧调度机制：利用 ZKP 协议行为的确定性，把协议翻译成计算图，由静态调度器离线决定 kernel 的映射设置与执行方案。GenZA 分两阶段：(1) 把 ZKP 协议（参照官方实现 [21][58][69]）翻译成计算图（当前手动，可交给标准 ZKP 编译器）；(2) 调度器自动——用分析成本模型选 kernel 映射设置（NTT pipeline 长度、MSM window size）、给每个 kernel 分配 PE 并确定 PE 内数据流、生成描述跨 kernel 并行/流水线方案的静态执行调度。映射策略实现为手写 kernel 模板集（类比 GPU 库 cuDNN 的手调 kernel），每模板含若干参数由后端调度器全自动配置；运行时每 PE 收一个模式配置（kernel 类型、bitwidth、场模、数据流模式如 lane 映射/流水深度）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译/调度流程（HyperPlonk 2^23 电路为例）：
```
1) 协议 → 计算图（节点=sumcheck/MSM/NTT/多项式等 kernel，边=数据依赖）
2) 调度器对每 kernel 用成本模型定参数：
   - MSM：按曲线 bitwidth、N、片上 SRAM、带宽选 window c（如 BLS12-381 大 N 选 c≈16）
   - NTT：按 bitwidth/可用 SRAM 定管线长度 L 与折叠到几行 PE
3) 分配 PE：把 kernel 分到 16×8 阵列的 PE 组，确定数据流
4) 图切分：按三类 cut（数据转置/全局归约/Fiat-Shamir 串行点）把图切成子图，
   贪心合并 kernel 成空间流水线（roofline 模型估计收益），调各 kernel PE 比例匹配吞吐
5) 输出静态调度表 + 每 PE 的模式配置序列
```
- Annotations：成本模型输入包括曲线 bitwidth、MSM 大小 N、片上 SRAM 容量、off-chip 带宽约束（MSM window 选择依据 Figure 5 的 PADD 数 vs 内存流量权衡）。离线调度成本一次性摊销：Groth16 0.02 s、HyperPlonk 0.5 s、Plonky2 517.8 s（复杂计算图，收益最大——流量 9.5× 削减）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：GenZA 的静态调度器（分析模型+贪心+roofline 估计）+ 手写 kernel 模板（NTT/MDC 折叠、MSM Pippenger、Poseidon、sumcheck、多项式/向量模板），模板参数自动化；对比 GPU 生态的"手调 kernel + 运行时 JIT 调度"（cuDNN/cuBLAS 手调 kernel、Triton 编译器自动生成），GenZA 因 kernel 类型有界、协议确定而采用全离线静态调度。使用：任何"统一硬件 + 多 kernel 复用"的领域专用加速器（ZKP/FHE/密码学）可借鉴——把确定性计算图离线调度、把 kernel 参数化模板化、运行时仅下发配置。论文未明确说明调度器与模板是否开源。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
