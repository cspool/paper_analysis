## Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic (Fusa)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Fusa 的运行时原子传输调度——在驱动运行时把 RDMA Atomic（CAS/FAA 传输类操作）按 group 粒度动态调度到两个异构后端：RNIC 硬件路径（锁定表槽 + PCIe RMW）与服务端 CPU 软件路径（cache-coherence 原子）；用 group 请求计数器、WR_ID 嵌入 13-bit group_id、CQE 轮询递减 inflight 计数实现在途请求跟踪，保证策略切换时新旧后端互不混用；Fusa-RPC 提供 coroutine-friendly 的 WRITE+RECV 异步传输（依赖 SEND 的 CQE 异步完成，等待期间切换协程），对比 RNIC-friendly RPC（SelfRPC 移植：WRITE + 主动轮询缓冲，阻塞协程切换，吞吐更低）。
  - 实验比较：RNIC-Only（纯硬件后端）、HERD（纯 CPU 后端）、Static（静态各半）、Fusa（动态混合）；另测 Fusa-PA（PCIe Atomic 启用，PCIe Atomic Completer Engine 作为第三硬件后端）与 OrderedFusa（加 WAIT 保序）。指标：原子吞吐（Mops/s）、平均/P50/P99 延迟、后端切换共识时间。
- 后端平台是什么，配置是什么。
  - RNIC 后端：100 Gbps Mellanox ConnectX-6（512 槽锁定表、多 PU、PCIe RMW）；对比 ConnectX-5。CPU 后端：Intel Xeon Silver 4314（单线程约 2.5 Mops/s；DDIO 使 RNIC 直写 LLC，缩短 CPU 原子路径）。Intel Xeon Gold 5420 / AMD EPYC 7281 复验 PCIe Atomic 瓶颈（Atomic Completer Engine 吞吐仅锁定表 34.0%；CX-6 stride 8B 42.3 Mops/s vs CX-6-PA 平均 14.4 Mops/s、EPYC 7281 上 27.6 Mops/s）。
- 评估性能的软件/脚本是什么。修改了什么。
  - 微基准：YCSB（Zipfian θ=0.99 默认；YCSB-A 50% 更新、YCSB-B 5% 更新、U40R60/U30R70/U20R80/U10R90/U100；更新用 RDMA CAS、读用 RDMA READ，YCSB key 作访问地址）；128 线程 RDMA_CAS + stride 8B–8,192B 控制锁定表槽分布；DrTM 锁 trace（ShiftLock [23] 生成，W50R50/W5R95/R100）。修改：rdma-core fork（providers/mlx5/ 下改 qp.c、verbs.c，新增 recorder.{c,h} 与 fusioncas 实验框架）；fusioncas 提供 agent/test/scripts 与 run_ycsb.py 编排。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/xmusys/fusa（已确认；rdma-core fork，含最小示例 test_rdma 与 run_ycsb.py）。
  - 评估原理与全过程：客户端多线程按 YCSB 分布生成 CAS/READ 请求 → Fusa-Driver 按 group bit 分派 → RNIC 后端：槽哈希 + PCIe RMW；CPU 后端：RPC 缓冲 + server 线程原子执行 → 驱动轮询 CQ 收集 CQE 计时 → 输出吞吐（Mops/s）与 P50/P99。关键数字：θ=0 均匀时 RNIC-Only 比 HERD 高 4.7×（P50/P99 低 78.9%/90.1%），θ=0.99 时 HERD 反超 RNIC-Only 1.4×；Fusa 在 θ=0.99 下 4.8×、≥4 server 线程时 4.8–7.0×；Fusa-PA 在 YCSB-A Zipfian 下 +36.4% 吞吐、P99 -92.0%；热点全切换时策略共识仅 48 µs。
