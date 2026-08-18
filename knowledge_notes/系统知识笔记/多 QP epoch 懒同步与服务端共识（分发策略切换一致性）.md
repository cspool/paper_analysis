## 多 QP epoch 懒同步与服务端共识（分发策略切换一致性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Fusa 保证分发策略动态切换正确性的机制组合，分客户端/服务端两侧。客户端懒同步：每 QP 维护 64-bit 状态（1-bit running + 63-bit epoch），驱动在每次 post 时被动 epoch++（不主动观察策略变化）；收到新策略后 Fusa-Agent 先用 CAS 原子切换策略指针，再执行 Wait_sync 阻塞等待：每个 QP 满足"epoch 已推进 或 退出 running"即视为同步（此后该 QP 的下一次请求必然读到新策略）。服务端共识：切换期间 Fusa-Server 计算新旧策略的逐组 XOR 一致位，位=1（策略变更组）的请求被 reject（返回消息带 reject 标志，客户端库透明重传），杜绝"部分客户端用旧策略、部分用新策略"对同一地址的并发执行；RNIC→CPU 方向的切换还需等待该组 in-flight 计数归零（CQE 轮询递减）。正确性用 TLA+ 建模验证：两个不变量（同一地址不被 RNIC 与 CPU 同时执行、同组请求始终由单一后端执行）在 >400 亿状态内恒成立。机制灵感来自 epoch-based reclamation 的 quiescent 概念。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 切换时序（Exp#4 实测，总 48 µs）：① Fusa-Server 推送新策略（10 µs）→ ② 各 Fusa-Agent 接收并 CAS 应用本地策略位（10 µs）→ ③ 各客户端 Wait_sync 等全部 QP epoch 推进/退出 running（28 µs）→ ④ 检查变更组 inflight==0（CPU→RNIC 方向 server 直接 reject 在途请求；RNIC→CPU 方向等 inflight 归零）→ 恢复处理。论文图 9 给出客户端共识伪代码（passive 路径 lines 4–26 更新 epoch/元数据，active 路径 lines 28–56 检查 QP 共识）。切换期间未变更组（一致位=0）请求不受影响继续执行。
- 为什么不用全量隔离：策略更新频率低（1 s/stage）且变更组数量少，只对变更组做一致性协调即可，避免像 [5][75] 那样对所有请求强制隔离的代价。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 同类机制（Web 证据）：EBR（Concurrency Kit ck_epoch、FreeBSD epoch(9)、libqsbr）——读者发布所在 epoch/静止状态，写者等 grace period 后回收内存；Fusa 借其"quiescent 检测"做策略切换而非内存回收。TLA+（Lamport 的 Temporal Logic of Actions）用于对切换协议做穷举模型检查，是与实现并行的验证手段。实现细节：策略指针切换用一次 CAS；reject 标志挂在 Fusa-Server 返回消息上，客户端检测后自动重发，应用无感知。信息缺口：论文未给出 Wait_sync 期间的吞吐损失曲线（仅报告 48 µs 共识时间与 Exp#4 的 1 s 内恢复）。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
