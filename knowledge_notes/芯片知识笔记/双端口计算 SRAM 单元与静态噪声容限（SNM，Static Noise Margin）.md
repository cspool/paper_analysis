## 双端口计算 SRAM 单元与静态噪声容限（SNM，Static Noise Margin）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
双端口计算 SRAM 单元是 PipeIMC 为实现流水化存内计算而设计的存储单元：在普通计算 SRAM 单元上额外增加一组 bitline/wordline（类似 true dual-port SRAM），使同一个 SRAM 阵列能同时被 memory port（取数/写回，空闲时也可计算）与 calculation port（只执行计算 phase）访问；两个端口各自支持多行访问（同时激活两条 wordline 做位线计算）并配有独立计算外围电路（8 组 1-bit logic/add/shift/writeback 四层外围电路跨 8 条 bitline 组成 8-bit 外围电路）与微码 sequencer。静态噪声容限（SNM）是 SRAM 单元稳定性的核心指标：在互补存储节点 Q 与 QB 上通过可编程电压源 V1/V2 注入受控噪声，测量单元能保持状态的最大噪声幅度，通常以 N 曲线（蝴蝶曲线）内切正方形的边长表示。论文用 Cadence Virtuoso 全定制实现 256×256 dual-port 计算 SRAM 阵列（TSMC 40nm、1.1V 标称电压），netlist 集成到 Cadence Spectre（TT corner、25°C）仿真；功能正确性通过注入多组随机输入、打印关键节点信号波形验证（覆盖 SRAM 阵列与外围电路）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
SNM 测量流程（论文 Fig.7）：在 TT corner、25°C、1.1V 下对 dual-port 计算 SRAM cell 的三种工作模式测 N 曲线——(1) 单端口读：红/蓝曲线为 V1 随 V2 与 V2 随 V1 的变化，绿色方块边界即噪声容限，测得 192.9mV；(2) 双端口同时读：测得 134.6mV（双端口同时读入更多电流扰动，SNM 下降）；(3) 写：测得 409.3mV。结果说明单元在典型条件下稳定性强。同时测量电路开销：面积相对 vanilla SRAM +55.7%、相对 dual-port vanilla SRAM +18.2%；静态功耗相对单端口计算 SRAM +48.1%；多行访问能耗相对读/写 +54.7%；多行访问频率相对 vanilla 读/写慢 2%（多行访问降低 wordline 电压以对抗数据破坏，代价是频率略降）。三端口阵列相对双端口面积 +19.6%、静态功耗 +23%——这是选择双端口而非三端口的芯片级依据。论文还讨论布线拥塞（额外布线/外围电路）可由成熟 EDA 流程管理（援引商业 dual-port 计算 SRAM 流片成功案例），可用 bitline 多路复用缓解，并采用 1:1 bitline-to-SA 比例。架构级面积/能耗用 GPUWattch + McPAT + 原论文数据 + 综合评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：全定制电路设计流程——Virtuoso 画 cell 版图与外围电路 → Spectre 仿真（1.1V、TT、25°C）提取延迟/能量/SNM → 为 cycle-approximate 模拟器提供计算 phase 周期表（add 9、mul 105–634、div 145–1174 cycles）与 GPUWattch/McPAT 能耗面积参数。使用：作为 CPU cache 阵列中计算 bank 的存储单元（每 IMC 执行单元一个 8KB 256×256 slice）；SNM 定量分析用于流片前验证单元稳定性，面积/功耗/频率开销用于权衡端口数（双端口 vs 三端口）。Web 证据：SNM 是 SRAM 设计的标准稳定性指标（N 曲线法，Seevinck 1987）；双端口/多端口 SRAM 单元与 CIM 宏在 ISSCC/JSSC 有大量流片报道，EVE（HPCA 2023）的 S-CIM 阵列用 OpenRAM 生成 28nm 布局。Vault 笔记（omnisearch 无命中；text 检索命中知识库芯片笔记均为 DRAM/封装方向）无本术语专门笔记证据。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
