## SCA（Separate Command/Address）NAND 通道协议

术语解释
- SCA 是 JEDEC/ONFI 系的 NAND 接口新协议（JESD230G 等），把命令/地址（CA）流量与数据流量分离到独立通道，命令/地址走专用串行 CA 引脚与时钟，数据走 DQ 总线，从而允许命令/地址与数据传输并发进行，显著缩短每命令的通道占用时间。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 传统 NAND（ONFI 1.0-5.1）把命令、地址、数据在 DQ[7:0] 上时分复用：命令/地址以异步单端低速发送、数据以同步差分高速发送，命令开销大。SCA 增加专用 CA 通道（CA[1:0] 串行命令/地址线 + CA_CLK 时钟），命令/地址信息与数据搬运可同时进行，提升接口带宽利用率并支持更强的命令交错/并行（网络来源：JEDEC 2024 年 11 月发布 JESD230G，接口速率最高 4800MT/s，SCA 为关键新特性）。论文（Sec. III-B）用 τ_CMD 表示每命令的通道占用时间：常规 NAND 8-bit 共享命令/数据总线时 τ_CMD≈1.2μs，采用 SCA 后降到 100-200ns。通道读吞吐 = 1/(τ_CMD + l_blk/B_CH)，小块下 τ_CMD 越小有效带宽越接近 B_CH/l_blk。
- 从芯片设计角度拆解术语：通道(channel)是控制器与一组 NAND die 之间的共享总线，通道命令时间 τ_CMD 直接决定通道级峰值 IOPS（IOPS_CH = R_r·1/(τ_CMD+l_blk/B_CH) + R_w·1/((l_blk/l_PG)·τ_CMD+l_blk/B_CH)）。SCA 通过把命令/地址从数据总线剥离，使 512B 随机读的每请求通道占用从"命令+数据串行"变为"命令并发、数据独占"，是 Storage-Next SSD 达到 50M-class 小块 IOPS 的关键使能之一。MQSim-Next 也新增 SCA 支持以对齐现代设备实践。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- SCA 在 NAND 芯片引脚层实现：为命令/地址增设专用串行化引脚（CA[1:0]）与时钟（CA_CLK），在发送数据期间可并行下发下一命令/地址，实现流水化。JEDEC 与 ONFI、Toggle 阵营协作制定（JESD230G，免费下载）。SSD 控制器侧需支持 SCA 时序（命令/地址发送后立即释放 DQ 给数据）。论文在解析模型与 MQSim-Next 中均以 τ_CMD=100-200ns（基准 150ns）参数化 SCA 效果，并在敏感性分析中扫描 τ_CMD∈{100,150,200}ns 验证结论稳健（512B 峰值 IOPS 39.4M-79.3M）。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
