## Near-SRAM 数据流架构（Near-SRAM Dataflow Architecture）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Near-SRAM 数据流架构指把计算单元（PE/神经核）紧邻片上 SRAM 布置、让绝大多数数据在片上近存完成的一种加速器组织方式：权重、激活状态（SNN 的膜电位、spike tracer）等驻留于片上 SRAM，计算单元从邻近 buffer 取数、避免远距离/片外访存。这是 TrueNorth、Darwin、MorphIC、PAICORE 等神经形态芯片与 Groq/Cerebras 等数据流芯片的共同基础（ELSA 论文 Related Work）。对 SNN 尤其合适：全部权重 + 膜 + tracer 需要 SRAM-only 存储（Tab.IV 中 SRAM Only=Yes 的弹性加速器），近存执行把片外 DRAM 访问压到仅输入 spike 的加载（ELSA：off-chip access negligible）。
- ELSA 在此基础上叠加 SNN 特化：addition-only 计算（无乘法）、事件驱动稀疏（跳过零 spike）、细粒度 spine/token 流水、BAER、mini-batch Gustavson（见各自术语）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一次 spike 在 ELSA 近存数据流中的完整路径：
```
输入 spike 从 DRAM 载入（唯一片外访问）→ NoC → 目标核路由 FIFO
→ PE 控制模块 → N-way 权重 SRAM 读权重行 + 膜 SRAM 读膜行（近存，本地 port）
→ 加法树累加 → fire 判激发 → 写回膜/tracer SRAM（仍在本 PE）
→ 输出 spike 打包 BAER 前传下一层
```
- 例：ResNet50 推理中膜电位行 [3,5,4,4] 的读、累加、写回全部在本 PE 的 SRAM 上完成，不跨核、不访 DRAM；只有每帧的首批输入 spike 从 DRAM 读入。ELSA 每核 4×102.4KB 权重 + 4×307.2KB 膜 + 4×102.4KB tracer SRAM（PE 面积 93.97% 被三类 SRAM 占据），整芯片 72MB 片上存储（Tab.IV）。
- Annotations：near-SRAM = 计算与存储在同一 tile 内通过本地端口访问；SRAM-only = 无片外权重重取；膜/tracer 是时间步间持久状态，必须常驻。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ELSA 6×6 神经核 2D-mesh（36 核、100.23 mm²、82490 µW、200-500MHz、28nm 商用工艺），每核 4 PE × 128 ST-BIF 电路 + 路由器；SRAM 用商业 memory compiler 生成，功耗/面积注入 cycle-level 模拟器。评价：Tab.IV 中 ELSA 25.55 TOPS/W（ResNet50）为弹性 SNN 加速器最高（PAICORE 20.89、TrueNorth 0.4），且相对非弹性 C-DNN（24.5 TOPS/W，LBL 免存膜）仍有小幅优势；相对大芯片（Jetson AGX Orin/A100/TPUv4/Groq）以 28nm/200MHz/100mm² 取得最高能效（25.55 vs Groq 3.125 TOPS/W）。Fig.22 消融：基线（near-memory + addition-only + 大 SRAM、无架构优化）已相对 Eyeriss 节能 9.2×（ResNet50）/2.1×（ViT-S）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
