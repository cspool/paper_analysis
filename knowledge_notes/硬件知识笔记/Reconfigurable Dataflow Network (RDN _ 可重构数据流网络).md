## Reconfigurable Dataflow Network (RDN / 可重构数据流网络)

术语是什么？
Reconfigurable Dataflow Network (RDN) 是 SN40L RDU tile 的片上可编程互联网络，以 2D mesh of non-blocking switches 连接所有 PCU、PMU 和 AGCUs。RDN 包含三种物理 fabric：(1) Vector fabric — packet-switched，tensor 数据主通道，支持 credit-based per-hop flow control 和 end-to-end flow control；(2) Scalar fabric — packet-switched，用于传输 metadata（地址、标量参数）和少量控制数据；(3) Control fabric — circuit-switched，由单比特线束（bundle of single bit wires）组成，用于分布式 coarse-grain flow control 和集体编排（如 PCU counter done events 作为 control tokens）。RDN 的路由支持两种模式：动态 2-D dimension order routing 和软件配置的 static flow routing（通过 flow ID 字段解码 + multicast）。Vector packet 携带 sequence ID metadata field，支持 many-to-one traffic 的数据重排序（PMU 使用 sequence ID 计算写地址将乱序 packet 归位）。

从硬件架构角度拆解：
RDN 支持的通信模式：
- One-to-many (multicast): flow ID based → fan-out paths to program-decided destinations
- Many-to-one: sequence ID + PMU address generation → 重排序合并
- Pipeline (PCU→PMU→PCU): credit-based flow control 防止生产者溢出消费者
- Data Reordering: 多源 many-to-one 场景，sequence ID 保证逻辑顺序恢复
- Bandwidth Management: 可编程 packet throttling 减少 bursty traffic 导致的拥塞

术语一般如何实现？如何使用？
RDN 的路由表、flow ID 分配、multicast group 配置由编译器的 Place-and-Route (PnR) 层自动生成。硬件 performance counter 在 switches 中计数 stall cycles 帮助识别 RDN 拥塞热点。Packet throttling 由软件可编程控制，缓解 bursty traffic。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
