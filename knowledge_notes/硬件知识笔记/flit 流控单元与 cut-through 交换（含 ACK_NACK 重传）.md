## flit 流控单元与 cut-through 交换（含 ACK/NACK 重传）

术语解释
flit（flow control digit）是 NoC 链路流控的最小单位；cut-through（直通交换）指 flit 无需等整个包到达即可逐 flit 转发；ACK/NACK 是接收端逐 flit 反馈确认/否认、发送端仅重发失败 flit 的可靠性机制。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NoC 中包（packet）被切成定长 flit 传输：典型 flit 128-bit（gem5 Garnet 默认 16B），控制包 1 个 flit、数据包 5–6 个 flit（HEAD/BODY/TAIL）。交换技术谱系：store-and-forward（整包收齐再转发，缓冲大延迟高）→ virtual cut-through（有整包空间即转发）→ wormhole/cut-through（按 flit 级缓冲推进，包可横跨多路由器，延迟最低）。流控机制谱系：credit-based（上游计数下游空闲缓冲 = credits，flit 发出扣、返回归还）与 ACK/NACK 重传（保留 flit 副本直至收到 ACK，NACK 则重发；Xpipes 的 Go-Back-N 变体丢弃后续 flit 直至出错 flit 重达）。DICE 在 chiplet 边界的 PHY 路由器采用"flit 级 cut-through + ACK/NACK"组合（Web 证据：NoC 教科书、Xpipes 设计、Onur Mutlu 课程讲义）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE PHY 路由器的运转流程（Fig.11）：发送侧 Router A 的 inter-die 输出单元 = FEC 编码器 + 发送缓冲（flit 大小条目、按包粒度预留：控制包 1 条、数据包 6 条）+ 调制仲裁器 + 调制器；flit 编码后入缓冲并经仲裁串行调制上链路。接收侧 Router B 收软采样入接收缓冲 → LLR + FEC 解码 → 成功则转发 flit 并回 ACK，失败回 NACK、发送端仅重调制重传该 flit。设计要点：① 包的全部 flit 均被 ACK 后才释放其缓冲预留；② 多包可共存于发送缓冲，一包停顿不影响他包经调制仲裁推进——避免 head-of-line blocking、保持链路利用率。硬件架构含义：流控粒度（flit vs 包）决定重传代价与缓冲面积（flit 级恢复局部化、重传量小）；ACK/NACK 的往返延迟直接加入链路尾延迟（DICE 中重传是尾延迟 104 vs 61 cycles 差距的来源之一）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：发送缓冲 + 缓冲分配器（packet 粒度预留/flit 粒度条目）、逐 flit credit/ACK 反馈通道、重传用 flit 副本（DICE 因有 FEC 只在解码失败时 NACK）；HeteroGarnet 等无 PHY 建模的模拟器则用 credit-based 限流近似 SerDes。使用方式：作为 PHY 流控的建模模板——把"FEC、调制、串行化、重传"合成进路由器微架构，使 FEC 诱导的 backpressure 进入端到端包时序；DICE 证明其改变了包延迟构成（FEC/SerDes/EC 占比显著）并影响 IPC（平均 6.8%）。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
