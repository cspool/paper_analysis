## qLDPC 稳定子测量电路（Quantum LDPC Stabilizer Measurement Circuit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- qLDPC（quantum low-density parity-check）码是量子纠错码（QEC）的一类，其校验矩阵稀疏，编码率远高于表面码（surface code），正从理论研究走向实验 FTQC 核心（编码效率优势使其成为超导平台上表面码之后的候选）。qLDPC 稳定子测量电路（stabilizer measurement circuit）是把 qLDPC 码的稳定子（stabilizer）测量实现为门电路的子程序：每个稳定子测量通常由若干 CNOT（把数据 qubit 与 ancilla qubit 纠缠）+ 测量 + 重置组成，其纠缠门作用在码的 Tanner 图上。由于 qLDPC 码常有长程交互（long-range interactions），在固定局域连接的超导处理器上执行时会产生显著路由开销，这正是 CANOPUS 研究的 FTQC 场景。
- 论文使用的码类型：generalized bicycle（GB）与 bivariate bicycle（BB）码（取自 [53][67]），在 2D heavy-hex 与 square 拓扑上编译稳定子测量电路；评估用标准 memory experiment 模拟。CX-iSWAP 组合 ISA 与此场景契合：稳定子测量中有大量 CNOT，把 SWAP 插入 piggyback 到 CX 上（复合等价 iSWAP）可零额外 2Q 门数完成路由。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 端到端评估 pipeline（CANOPUS 论文 V-B）：① 输入 qLDPC 码（GB/BB 码的校验矩阵）；② 生成稳定子测量逻辑电路（CX 或 CX-iSWAP ISA 表达的 CNOT 网络）；③ 编译/路由：SABRE 或 CANOPUS 把逻辑电路映射到 2D heavy-hex/square 耦合图，输出含 SWAP 的物理电路；④ 用 stim（https://github.com/quantumlib/Stim，Google 的高速稳定子电路模拟器）按文献[6]的电路级噪声模型模拟标准 memory experiment（大量重复的编码/纠错周期，每次测量生成 syndrome）；⑤ 所有 syndrome 用 BP-OSD decoder（belief propagation + ordered statistics decoding，文献[28][53]）解码，得到逻辑错误率（logical error rate）。
- 伪代码（评估逻辑）：
  ```
  for code in {GB, BB}:
      for isa in {CX, CX-iSWAP}:
          circ_logical = build_stabilizer_circuit(code, isa)   # 稳定子测量 CNOT 网络
          for compiler in {SABRE, CANOPUS}:
              circ_phys = compiler.route(circ_logical, coupling_map)  # 插入 SWAP
              pL = stim_memory_experiment(circ_phys, noise_model)     # stim 模拟
              pL = bp_osd_decode(syndromes)                            # BP-OSD 解码
  ```
- 结果（Fig.10）：CANOPUS vs SABRE 的逻辑错误率抑制——CX ISA 下 square 49.4%、heavy-hex 11.4%；CX-iSWAP 下 square 52.6%、heavy-hex 29.3%。差异来源：CANOPUS 编译出的电路 CX/iSWAP 门数与深度更少，尤其 CX-iSWAP 下大量 SWAP 被 CX 吸收。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：stim（Python/C++，Google Quantum AI 开源）是稳定子电路与噪声仿真的行业标准工具（含 Clifford tableau 模拟、syndrome 采样、`detector`/`observable` 定义）；BP-OSD 解码器（Roffe 等，开源实现如 ldpc 库）用于从 syndrome 估计逻辑错误。CANOPUS 实验套件在 ./experiments/eval_qldpc/（Makefile 中 make 准备、make canopus/baselines 运行）。
- 场景意义：qLDPC 稳定子测量电路是"ISA-aware 路由直接提升容错性能"的典型例子——路由开销（SWAP 数）转化为更多噪声门 → 更高逻辑错误率；CANOPUS 把 ISA 合成能力用于路由，直接压低 FTQC 逻辑错误率，是"NISQ 与 FTQC 双场景"的桥接验证。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
