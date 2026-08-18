## libibverbs / rdma-core 用户态驱动与 LD_PRELOAD 透明拦截

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- libibverbs（rdma-core，https://github.com/linux-rdma/rdma-core）是 Linux 标准用户态 RDMA 编程库：统一 ibv_* API + 可插拔 provider（libmlx5 等按 sysfs PCI-ID 经 dlopen 加载）。控制路径（创建 QP/CQ、注册内存、建连）走内核 ioctl（/dev/infiniband/uverbsX），数据路径（ibv_post_send、ibv_poll_cq）纯用户态：provider 把 WQE 写进队列内存并敲 MMIO doorbell。Fusa 修改 mlx5 provider 的 mlx5_create_qp()（每 QP 分配元数据区）与 mlx5_post_send()（按策略拦截并重写请求），并以 LD_PRELOAD 覆盖 libibverbs 导出符号，使未修改的 RDMA 应用透明接入 Fusa 分发。Web 证据补充：ibv_post_send 本身是 verbs.h 中 static inline，实际执行 qp->context->ops.post_send() 回调——论文改造的正是该 provider 回调层，而非替换 libibverbs 符号。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一个 CAS 请求穿过驱动层的流程：应用调 ibv_post_send → inline 转发 context ops.post_send（= mlx5_post_send）→ Fusa-Driver 计算 group_id = address % 8192、组计数++、置 QP running、epoch++、读组 strategy bit → bit=0：inflight++，原样下发 RNIC；bit=1：转 Fusa-RPC（WRITE+RECV）写 server 请求缓冲 → 敲 doorbell → RNIC 执行 → 驱动 poll CQ 拿 CQE、从 WR_ID 提取 13-bit group_id 递减 inflight。Fusa-Agent 是独立控制线程，经 Fusa-SHM（65 KB 共享内存）与驱动交换统计/策略位，周期（1 s stage）更新策略。QP 创建时预留的元数据：每 QP 64-bit（1-bit running + 63-bit epoch），每客户端 ≤32 QP（32 线程），开销 256 B。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：Fusa-SHM 以 8,192 组 ×（64-bit 计数 + 1-bit 标志）管理；13-bit group_id 复用 WR_ID 字段（CQE 回传时无额外存储）；驱动开销实测 ibv_post_send 123→141 ns（+18 ns，相对 ~2 µs RTT 可忽略）。使用方式：LD_PRELOAD 使 RACE、DrTM 等系统零修改运行在 Fusa 之上（Exp#8/#9）。信息缺口：论文未逐一列出被覆盖的 libibverbs 符号；Web 证据表明 fast path 因 static inline 无法靠符号替换拦截，实际拦截点在 provider ops 回调（论文开源仓库确认改动集中在 providers/mlx5/）。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
