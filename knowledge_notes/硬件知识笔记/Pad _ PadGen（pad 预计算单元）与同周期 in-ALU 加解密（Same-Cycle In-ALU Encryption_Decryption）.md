## Pad / PadGen（pad 预计算单元）与同周期 in-ALU 加解密（Same-Cycle In-ALU Encryption/Decryption）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- pad（密码学中常称 keystream/one-time-pad 片段）是加密/解密用的密钥流：明文 XOR pad = 密文、密文 XOR pad = 明文。IroKnight 中 pad = f(密钥 κ, 物理地址, 版本号)，即 pad 只依赖"数据放在哪、被写过几次"，与数据本身无关——这是"可由地址提前预计算"的根本前提。PadGen 是生成 pad 的硬件单元：把 128-bit 的 (地址, key, 版本号) 推进 10 轮 AES 类变换（key addition 用 AES-GCM 扩展的 round keys、S-box 非线性字节替换、行移位、列混合），每周期产出一个 128-bit pad（AES-GCM 的粒度）。同周期 in-ALU 加解密：因 pad 可提前算好，加解密被降为 ALU 输入/输出处的一行 bitwise XOR（pad 生成本身是长时延流水操作，但被 Run-Ahead 提前掩盖），明文只在 MACC/ALU 组合逻辑内瞬态出现，所有存储中都是密文。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - IroKnight 脉动阵列 PE 的运转流程（Fig.3）：原 Index Generator 改造为 Run-Ahead Index Generator，提前输出 WBUF 索引 → 索引同时送入 (a) 流水化 PadGen（因 WBUF 索引范围有限，PadGen 把索引与 tile 物理地址结合生成 pad）与 (b) Run-Ahead FIFO（缓冲同一批索引供 WBUF 正常读取，FIFO 深度 = PadGen 流水级数，保证 pad 与操作数同周期到达）→ PE 内 MACC ALU 输入处 XOR 解出明文（操作数来自加密的 WBUF/IBUF）、ALU 输出处再 XOR 加密 → 结果写回加密态。向量引擎同理：Run-Ahead Address Generator 把源/目的地址提前喂给 PadGens 与 Run-Ahead FIFOs，每个 memory-ALU 对的输入输出加 XOR。PadGen 粒度是 128-bit：脉动阵列中 8-bit 操作数时 1 个 PadGen 共享 16 个 PE；向量引擎中 2×32-bit 源 + 32-bit 输出（共 96 bit）×4 组 = 384 bit 由 3 个 PadGen（3×128=384）服务——PadGen:ALU 数量比随操作数位宽变化。pad 生成后立即被 ALU 的 XOR 消耗、不存在 pad 存储，因此 pad 本身也不可观测。版本号：每个 ⟨地址,值⟩ 对需唯一 pad，故每次写入递增 tile/pipe-register 的版本号（模型权重/激活只读、版本号不变；输出 tile 一个版本号；脉动阵列 pipe-register 共享版本号 + PE ID 个性化），防止重放/反演。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RTL（IroKnight 用 SystemVerilog，Synopsys Design Compiler 2023.09 + FreePDK45/OpenRAM 综合、DeepScaleTool 缩放 7nm）；PadGen = AES-like 10 轮 datapath（S-box 用 Canright 紧凑实现 [85]），每周期 128-bit 吞吐。使用要点：pad 预计算依赖"地址已知且规则"——LLM 细粒度 tiled/vectored 算子的仿射访问（MatMul tile 的 i/k 索引、SIMD 的 stride 扫描、vector reduction）使地址 = 基址的线性组合，编译器把 tile 基址/stride/offset/版本号喂给 PadGen。代价：仅 PadGen 流水线填充延迟（每个 LLM 算子一次，运行时开销 0.2%）；面积/TDP 贡献（加密变体总面积 1.79×、TDP 1.76×）。通用场景（不规则随机访问）无法提前算 pad，这是该设计只适用于规则神经计算的边界。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
