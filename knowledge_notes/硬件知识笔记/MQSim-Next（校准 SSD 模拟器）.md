## MQSim-Next（校准 SSD 模拟器）

术语解释
- MQSim-Next 是论文在开源 MQSim（CMU-SAFARI，FAST 2018）基础上扩展的 SSD 模拟器，现代化了 NAND 后端（SCA 协议、独立多平面读、transfer-sense 重叠、两级级联 ECC、更多 I/O 队列），用于在 Storage-Next 高随机 IOPS regime 下以真实保真度刻画设备级 IOPS/延迟、验证解析模型并支持敏感性分析。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MQSim 是 CMU-SAFARI 开发的快速准确 SSD 模拟器，忠实建模现代多队列（NVMe）与 SATA SSD：多队列协议、稳态条件（含 GC）、请求端到端延迟，误差在真实 SSD 的 6%-18% 内（网络来源：FAST 2018 论文、https://github.com/CMU-SAFARI/MQSim，C++/MIT）。MQSim-Next 保留其已验证基础（PCIe/TLP 与 FTL/cache 时序、请求获取控制、稳态预处理），新增：①SCA 通道协议（命令/地址开销 100-200ns）；②独立多平面读；③显式 transfer-sense 重叠（一个请求的 sensing/programming 与另一请求的命令/数据搬运并发）；④read-prioritized、plane-aware 后端仲裁（短读与长 program 重叠、SCA burst 与数据搬运交错）；⑤显式可配置两级级联 ECC（512B BCH 内码 + 跨 8 扇区 LDPC 外码，BCH 失败升级 4KB LDPC，可调 p_BCH）；⑥更多 I/O 队列以提取满随机 IOPS。
- 从硬件架构角度拆解术语：MQSim-Next 是设备级（SSD 内部）微架构/物理建模工具，模拟从主机请求进入（PCIe/TLP）→ FTL 地址翻译 → 通道/plane 调度 → NAND sensing/program 时序 → ECC 解码 → 数据返回的完整数据路径，输出设备级 IOPS 与延迟。其与解析模型的关系：解析模型（Sec. III-B）给出闭式 IOPS，MQSim-Next 逐事件模拟验证（两者紧密吻合，模拟略高因模型保守 Φ_WA=3），并承接解析框架无法覆盖的效应（GC 竞争、ECC 失败升级、通道带宽变化）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MQSim 使用方式（网络来源）：配置 SSD 参数（通道/FTL/GC/平面分配/调度策略）→ 加载 I/O trace → 运行模拟 → 输出 IOPS/延迟统计；论文未给出 MQSim-Next 复现命令，按其描述用 Table I 参数 + Gen7 ×8 PCIe 配置。用途：①验证解析 IOPS 模型（Fig. 7：不同读写比/通道带宽/ECC 失败率的模型-模拟对照）；②作为 KV/ANN 案例研究的设备特征来源（MQSim-Next 提供峰值 IOPS 与延迟行为，解析框架计算可用 IOPS 与可行性）。开源：MQSim-Next 未见公开仓库（联网搜索无法确认），基座 MQSim 开源。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
