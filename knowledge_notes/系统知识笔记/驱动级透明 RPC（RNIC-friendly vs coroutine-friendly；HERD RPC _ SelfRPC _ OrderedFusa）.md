## 驱动级透明 RPC（RNIC-friendly vs coroutine-friendly；HERD RPC / SelfRPC / OrderedFusa）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Fusa 把被卸载的原子请求从客户端传输到服务端 CPU 的 RPC 机制，实现于 RNIC 驱动层（而非应用层）以换取透明性。两个候选实现：(a) RNIC-friendly RPC——移植自 SelfRPC，流程为分配缓冲 → RDMA WRITE（one-sided）→ 主动轮询结果缓冲；RNIC 资源占用低（不产生服务端 CQE），但自旋等待阻塞协程切换，与现代协程架构（RACE、Sherman、SMART 等）不兼容。(b) coroutine-friendly RPC——移植自 HERD RPC，先 post RECV、再 WRITE 请求、控制权立即交还应用，靠 SEND 的 CQE 异步完成；等 CQE 期间其他协程继续执行，吞吐更高，代价是多耗 RNIC 资源（two-sided verb）。实验（每线程 2 协程、均匀分布、更新比 25%–100%）显示 (b) 各配置全面胜出，故 Fusa-RPC 默认采用 coroutine-friendly。OrderedFusa 变体在 WRITE 后追加 RDMA WAIT verb（WAIT 在 RECV 完成前不放行后续请求），恢复 per-QP 线性化，供严格保序应用使用。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- coroutine-friendly 路径（图 11(b)）：客户端收到原子请求 → post RECV 到 RQ → 发 WRITE 把 (address, current, new) 写入 server 请求缓冲 → 控制权立即交还应用（协程可切换）→ server CPU 线程出队解析、执行原子 → server SEND 结果 → 客户端 poll CQ 等到 CQE → 协程继续。原始 SQ {X, ATOMIC, Y} 被改写为 SQ' {X, WRITE, Y} + RQ' {RECV}；Y 可能在原子完成前执行（Fusa 接受的语义松弛）。OrderedFusa 的 SQ' 为 {X, WRITE, WAIT, Y}，WAIT 由预 post 的 RECV 完成释放，保证原子先于 Y。
- 为什么选 (b)：轮询式 (a) 把协程卡死在驱动层，等同把并发模型打回"阻塞调用"；异步式 (b) 用 CQE 通知天然融入 poll CQ 循环，协程间可交错，隐藏同步延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Web 证据：HERD（Kalia 等，SIGCOMM'14）即用 UC WRITE 写请求 + UD SEND 回响应（客户端预 post RECV）的混合设计，后续演进为 FaSST、eRPC；Fusa 是把该模式下沉到驱动层、叠加分发策略与 reject/重传。reject 标志与自动重传复用同一 RPC 返回路径。信息缺口：论文未给出 (a)/(b) 的具体吞吐数字（仅图 12 显示 coroutine-friendly 各配置胜出），SelfRPC 的独立出处未在参考文献列表直接给出（正文 [45] 指向 Octopus，SelfRPC 为其组件）。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
