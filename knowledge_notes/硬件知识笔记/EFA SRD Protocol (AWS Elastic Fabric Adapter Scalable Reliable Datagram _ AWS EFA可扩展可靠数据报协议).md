## EFA SRD Protocol (AWS Elastic Fabric Adapter Scalable Reliable Datagram / AWS EFA可扩展可靠数据报协议)

术语是什么？
EFA (Elastic Fabric Adapter) 是 AWS 的云原生 RDMA 网络接口，运行 SRD (Scalable Reliable Datagram) 传输协议。与 InfiniBand RC (Reliable Connected) 的严格保序不同，SRD 提供可靠但无序的报文交付——数据保证到达但不保证顺序。SRD 设计为多路径传输，在网络层利用 ECMP 和自适应路由跨所有可用路径分发报文，避免单路径拥塞但导致 packet reordering。EFA 不提供硬件 RDMA atomics（如 compare-and-swap、fetch-and-add）。

从硬件架构角度拆解术语：
```
// EFA SRD vs IB RC 的网络层行为差异:
// IB RC (InfiniBand Reliable Connected):
//   - 固定路径: 同一 QP 消息经同一网络路径
//   - 硬件保证 ordering: write(A)→write(B)→atomic(C) 严格按序到达
//   - 硬件 atomics: NIC 在接收端直接执行 fetch-and-add 等操作

// EFA SRD:
//   - packet spraying: 将同一消息的不同报文经不同路径发送
//   - 无 ordering 保证: 到达顺序 ≠ 发送顺序
//   - 无硬件 atomics: 需软件模拟或上层处理
//
//   UCCL-EP 在 EFA 上的 ordering emulation:
//   sender: write(A, imm=seq1) → write(B, imm=seq2) → write_atomic(C, imm=seq3)
//   各报文可能经不同路径 (ECMP/自适应路由), 到达顺序未知
//   receiver CPU proxy:
//     poll(CQ) 获取 completed WQEs (含 immediate data)
//     若收到 seq3 但 seq1/seq2 的 writes 未完成:
//       暂存 atomic to control buffer
//     若收到 seq1 标记完成:
//       检查 control buffer: seq3 现在可以 apply?
//       → 尚未 (seq2 未到)
//     若收到 seq2 标记完成:
//       检查 control buffer: seq3 现在可以 apply?
//       → 是 → apply atomic C (更新 ring buffer tail)
```

术语一般如何实现？如何使用？
EFA 通过 libfabric 或 libibverbs 接口在 AWS p4d/p5/p5en 实例上使用。每 H200 GPU 配备 2×200G EFAv3 NICs（B200 + EFAv4 400G）。主要限制：(1) 无硬件 RDMA atomics，需软件模拟；(2) unordered delivery 需应用层自行处理 ordering；(3) EFA firmware 对小消息（~7KB token activation）处理速率有限。UCCL-EP 是首个在 EFA 上实现 GPU-initiated token-level EP 通信的系统（DeepEP 因 IBGDA 依赖无法运行），通过 immediate data + CPU control buffer + 软件 atomics 克服了 EFA 的上述限制。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
