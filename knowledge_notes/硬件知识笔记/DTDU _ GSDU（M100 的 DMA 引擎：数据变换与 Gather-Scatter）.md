## DTDU / GSDU（M100 的 DMA 引擎：数据变换与 Gather-Scatter）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TPB 内两类 DMA：① Data Transformation DMA Unit（DTDU）——像计算单元一样执行 TPB 指令，负责 HBSM 内数据搬运、矩阵转置等 tensor layout 变换、按地址范围填充初始化内存；② Gather-Scatter DMA Unit（GSDU）——由 cluster CPU 管理、不直接执行 TPB 指令，处理难用标准 TPB 指令编码的不规则数据移动：TPB 指令触发 CSU，CSU 启动 CPU 例程控制 GSDU，在本地 HBSM 与外部内存（其他 TPB HBSM、CCB SRAM、DDR）之间做 gather/scatter。CCB 级另有 2 个 DDR↔SRAM DMA（权重广播）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：规则搬运（连续搬移/转置/内存填充）→ 作为 TPB 指令由 DTDU 执行（与 TCU/CVU 一样参与指令队列派发）；不规则搬运（运行时才知道的非连续地址，如稀疏访问）→ TPB 指令触发 CSU 保存参数并中断 cluster CPU → CPU 服务例程经 VCIX 接口配置 GSDU → GSDU 完成本地↔远程 gather/scatter → 例程结束通知 CSU 标记指令完成。DTDU 的转置能力让 layout 变换在片上完成，避免额外访存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DMA 引擎 + 地址生成（DTDU 走 TPB 指令/TWU 类模式，GSDU 走 CPU 下发的不连续地址列表）+ VCIX 接口。使用：编译器生成 DTDU 搬运指令；GSDU 供 CPU 处理不规则访问。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
