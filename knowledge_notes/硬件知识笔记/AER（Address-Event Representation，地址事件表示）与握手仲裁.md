## AER（Address-Event Representation，地址事件表示）与握手仲裁

术语解释
AER 是异步事件传感器/神经形态系统的标准事件通信协议：事件以（地址, 极性, 时间戳）编码，经 Req/Ack 握手仲裁在共享总线上传输。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AER 把"像素发生了什么、在哪里、何时"编码为短包：像素产生事件 → 置 Req 请求总线 → 仲裁器（树形/列行优先）选中后回 Ack → 事件地址与数据送上共享总线 → 接收端解码。无全局时钟、事件驱动：总线仅在事件传输时消耗动态功耗，天然匹配稀疏数据。DESSCam 把 AER 粒度从像素提升到 patch：16×16 patch 事件计数超阈值才握手，AER 包 = addrX(5 bit) + addrY(5 bit) + 512 bit 事件（16×16×2 bit 极性）+ 32 bit 时间戳（346×260 分 22×17 个 patch）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
标准 DVS（per-pixel 握手）：像素事件 -> Req -> 行/列仲裁器 -> Ack -> 事件总线（高容负载，~120ns 延迟）
DESSCam（patch 级握手）：16×16 像素 -> SDP SRAM 锁存 -> 加法树计数 -> 超阈值 -> ReqX/ReqY -> Ack -> SRAM 读出 -> ping-pong 行 buffer -> 输出 FIFO -> MIPI CSI-2
```
patch 级握手省掉逐像素仲裁链（复杂仲裁逻辑），共享握手单元把握手逻辑面积从像素 21% 降到 18%；ping-pong 行 buffer 把读出与新 patch 采样流水重叠；AER 总线仅在事件活跃时耗动态功耗。延迟链：模拟前端检测 + eventification/采样 + 加法树传播 + 握手/AER 打包 + MIPI 传输（数据量/2.5 Gbps）+ NPU 推理，六段求和。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：异步树仲裁器（如 adaptive priority toggle asynchronous tree arbiter）、行/列 Req-Ack 四相握手、AER 编码器/解码器；商用 DVS（DAVIS/Prophesee/Sony）广泛采用；数据率 = 事件率 × 包大小。使用方式：与 MIPI CSI-2/USB3 配合把事件流送 host；设计权衡：per-pixel 握手保精度但仲裁复杂延迟高，patch/行级聚合降延迟与功耗但引入聚合粒度取舍。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
