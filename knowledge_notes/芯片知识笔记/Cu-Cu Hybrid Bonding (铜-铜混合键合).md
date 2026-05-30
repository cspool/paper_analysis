## Cu-Cu Hybrid Bonding (铜-铜混合键合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cu-Cu Hybrid Bonding 是一种 3D IC 芯片互联技术，将两个 dies 通过铜金属对铜金属（Cu-Cu）的直接键合实现垂直互联，同时介质层（dielectric）也形成键合，因此称为"混合"（hybrid）bonding。与 TSV 需要钻孔填充金属不同，hybrid bonding 在 wafer 表面直接形成微米级 Cu 触点（bonding pad），通过 face-to-face 或 face-to-back 方式 bonding。Stratum 中使用 face-to-face hybrid bonding，bonding pitch = 1μm，相比 HBM 的 TSV pitch = 10μm，垂直互联密度提升约 25×（面积比），非 I/O 面积比（因 TSV 还需要 keep-out zone）提升约 5× 的可用互联密度。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Stratum 中，hybrid bonding 连接 Mono3D DRAM die 和 logic die（NMP processor）：
1. DRAM die 底部的 Cu pads 与 logic die 顶部的 Cu pads 对准并 bond（face-to-face）。
2. 每个 bonding pad 区域 ≈ 1μm²，每 chip 可有数百万个 bonding points。
3. 相比 HBM 的 TSV（每 stack 仅 1024 个 TSV I/O），hybrid bonding 提供全芯片面积的互联密度，使逻辑 die 能在每个 DRAM bank 上放置 PE 并充分利用 DRAM 内部带宽（19-34 TB/s）。
4. 热方面：hybrid bonding 不需要 TSV 的填充金属热阻，thinner dies 使垂直热传导更好，有利于逻辑 die 的散热。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cu-Cu hybrid bonding 已在图像传感器（Sony, Samsung）和部分 3D NAND 产品（YMTC Xtacking）中商业化。在 DRAM 领域，IMEC 和三星等研究机构正在开发 sub-1μm pitch 的 hybrid bonding 工艺。实现步骤：(1) CMP（化学机械抛光）平整 wafer 表面；(2) plasma activation 激活 bonding 表面；(3) 室温下对准并接触（pre-bonding）；(4) 退火（~300°C）使 Cu 互相扩散形成永久键合。当前工艺限制：对 wafer 平整度要求极高（<1nm roughness），yield 依赖于 bonding alignment 精度。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
