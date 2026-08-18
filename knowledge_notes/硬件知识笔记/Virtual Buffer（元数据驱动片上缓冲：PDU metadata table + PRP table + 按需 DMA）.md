## Virtual Buffer（元数据驱动片上缓冲：PDU metadata table + PRP table + 按需 DMA）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Virtual buffer = NTI 提出的技术：用"每连接数 KB 级片上元数据 + 按需 DMA"取代常规 NVMe/TCP 所需的"每连接大 TCP send/receive buffer"，使 I/O 数据永不穿越板载片外内存。动机（§III-B 内存瓶颈）：常规实现把所有 TCP payload 先 stage 进大 TCP 缓冲再找 PDU 边界，数据在片外内存双向全速流动一遍；且小对象（命令/完成/PDU 头）+ 随机访问模式（乱序、重传）把有效带宽进一步拉低；而纯片上缓冲又不可行——SRAM 容量远小于数百 MB 的连接缓冲需求。
- 两张元数据表：PDU metadata table（每 NVMe/TCP 队列记录 {qid, virtual offset, PDU header}，其中 virtual offset = 概念性 per-connection TCP 发送缓冲中下一 TCP payload 应放置的位置）；PRP table（每条 NVMe 命令关联的 PRP 条目，供按需 DMA 时取主机数据缓冲地址；条目直接是地址则直返，是 PRP list 则执行 list walking）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TX/PDU 生成（Fig.7a）：Header Generator 产生 PDU 头 → Table handler 按 {qid, virtual offset} 存入 PDU metadata table（①）→ 同时 TOE 被要求发包，随后按 TCP 字节序从"TX virtual buffer"读 payload（②）→ Table handler 拦截读请求（③）、用 metadata table 找出与该字节区间重叠的 PDU（④）→ PDU payload receiver 判读区间是否含 payload → 含则按 PRP table 地址从主机内存 DMA 取 payload（⑤）→ 头+payload 现场拼接交给 TOE（⑥）。重传任意历史字节区间、PDU 跨包/多 PDU 同包都由元数据区间查表解决，无需把 PDU 实物存进缓冲。
- RX/PDU 解析（Fig.7b）：TOE 把某连接的 TCP payload 交给 RX virtual buffer（①）→ PDU splitter 拦截并顺序扫描、即时切分 PDU 头与 payload（②）→ 头送 PDU header handler（③）按 opcode 判定类型（④）→ 完成 PDU 转 Decapsulator（⑤）；数据 PDU 则查 PRP table 得目的地址（⑥）→ splitter 按该地址 DMA 直写主机 NVMe data buffer（⑦）。全程无中间 stage。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：元数据规模小（每连接数 KB SRAM），支撑 50M IOPS 与 1.6 Tbps 扩展；资源优化（§V-C）：PRP list 留宿主内存、片上只存头指针 + 地址预取；完成命令的 PDU 元数据实时失效降驻留。Fault handler 在 Parser/Stitcher 与 NHI/TOE 边界监控错误。
- 使用场景：一切"字节流 transport × 消息协议"的硬件卸载（NVMe/TCP、iSCSI 等）中消除中间缓冲；与零拷贝方案的区别：ANO/XLIO 只做单方向且依赖软件回退，Virtual buffer 是双向、纯硬件、无回退路径。信息缺口：论文未给出虚拟缓冲可支撑的最大连接数/队列数。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
