## Fully-State Encrypted Acceleration（全状态加密加速，IroKnight 范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 IroKnight 提出的新安全执行范式：明文只短暂存在于 ALU 内部线网与组合逻辑门中（纳秒级闪烁），而所有存储——片上 pipe-register、buffer、SRAM、scratchpad，以及片外 HBM、CPU、网络——中的值始终保持 AES-GCM 加密并带完整性保护。依据是数字逻辑测试中的可观测性原理：瞬态电信号若不被寄存器/存储锁存，则不可被观测。它填补 FHE（加密上计算、开销巨大）与 TEE（片上明文、不保有所有权）之间的空白，宣称达到"近 FHE 级加密所有权"而开销仅 3.3% 运行时/15.2% 能量。其可行性依赖两个 LLM 域特定洞察：规则访问使同周期 in-ALU 加解密可行（见 Pad/PadGen 条目），认证算术与 MACC 同构使执行中认证零开销（见 Mid-Execution Authentication 条目）。代数形式（Eq.1）：R̃_{i,j} = Σ_k ( (Õ_{i,k}⊕Pad^O_{i,k})·(W̃_{k,j}⊕Pad^W_{k,j}) ) ⊕ Pad^R_{i,j}——操作数在 ALU 内 XOR 出明文乘累加、结果 XOR 回密文再写回。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 以一个 MoE transformer 层的自注意力 QK^T MatMul tile 为例：编译框架按 tiling/fusion 生成 kernel 并把 tile 基址/stride/offset/版本号提供给 PadGens；Q、K^T 以密文驻留 IBUF/WBUF（KV cache 中缓存的 K^T 也保持密文）；脉动阵列每个 PE 的 Run-Ahead Index Generator 提前出索引 → PadGen 预计算 pad → 加密操作数读入 PE 后经 XOR 瞬态解密、MACC 计算、再 XOR 加密写回 pipe-register/OBUF——整个 tile 的每个中间值（部分和、pipe-register 内容、softmax 前后向量、输出 R）在存储中始终是密文；明文只存在于乘法器与加法器之间的组合线网。向量引擎处理非 GEMM 算子（缩放、softmax 的 exp/求和/除法、Top-K）时，每个 memory-ALU 对同样在输入输出加 XOR，地址由 Run-Ahead Address Generator 提前供给 PadGen。动态优化（KV caching、投机解码、token pruning）因不改变细粒度规则的仿射访问，都能维持 Fully-State Encrypted 执行。TCB = 仅 ALU + 片上 key/pad 生成电路；对端（Model Provider/Data Owner）经 RSA 私密通道分发 AES-GCM key。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：在 TPU-like NPU（128×128 systolic + 128-lane vector、1 GHz、INT8、18MB scratchpad、16GB HBM/1TB/s）上以 SystemVerilog 实现两个变体——仅加密（1.79× 面积/1.76× TDP）与加密+认证（1.95×/2.18×）；合成显示关键路径仍为 SRAM 读、不降频。使用/评估：开源 cycle-accurate 模拟器（Tandem Processor，GeneSys 项目 actlab-genesys.github.io）显式建模全部开销（in-ALU 加解密、pad 生成、认证、hash 片外读写），编译器功能不变但额外提供版本号/pad 信息；结果：8 个 LLM（GPT-OSS-120B、Llama4-Scout、Llama3-70B、OPT-66B、Llama2-34B、OPT-30B、Llama3-8B、Llama3-1B）加密运行时 0.2%、认证 3.3%，能量 ≤14%/≤18%（LLM 能量以 HBM 访存为主，pad 128-bit 5.4 pJ vs HBM 读 508.2 pJ）；小模型（BERT/ResNet-50 等）计算密集、能量 3.28×–4.68×。对比：FHE NPU 运行时 713×–1793×、能量 871×–7396×；NPU TEE（含 scratchpad 清除）3.3% vs 19.6% 且不保有所有权。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
