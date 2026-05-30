## RDMA immediate data (ordering emulation / RDMA 即时数据)

术语是什么？
RDMA immediate data 是 RoCEv2 协议包头中的 32-bit 字段，RDMA write/send 操作可在发送数据的同时携带该字段。接收端 CPU 通过 polling completion queue (CQ) 获取该 32-bit 值，无需访问远端 GPU memory。UCCL-EP 利用此字段嵌入 sequence number 和 expert index，在 CPU proxy 中实现不支持硬件 ordering 的 NIC 上的 delivery semantics 模拟。

从kernel调度角度拆解术语：
```
// EFA SRD 协议: 可靠但无序 (unordered delivery)
// 问题: GPU kernel 假设 write→atomic 严格顺序
//       但 EFA 可能让 atomic (用于 ring buffer tail update)
//       先于对应的 data writes 到达，导致 GPU 读到 stale data

// UCCL-EP solution via immediate data:
// 发送端:
//   RDMA_write(dst_addr, data_payload, imm_data = seq_num | expert_idx)
//   RDMA_write_atomic(dst_addr, tail_update, imm_data = atomic_seq)

// 接收端 CPU proxy:
//   cqe = poll_cq()
//   seq = cqe.immediate_data & SEQ_MASK
//   if cqe is atomic and seq > last_applied_write_seq:
//       将 atomic 暂存到 control buffer (unordered arrival)
//   elif cqe is write:
//       标记 write_seq 已到达
//       检查 control buffer 中是否有 pending atomic 现在可 apply
//   if all prior writes done:
//       apply atomic (更新 ring buffer tail)
```

术语一般如何实现？如何使用？
Immediate data 字段在 RoCEv2 标准包头中定义，几乎所有 RDMA NIC 都支持（包括 EFA、ConnectX、Broadcom）。在 UCCL-EP 中：(a) LL mode：接收端 CPU 用 immediate data 中的 expert index 做 conditional check——仅当特定 expert 的 X 个 writes 完成后才 apply atomic；(b) HT mode：per-channel 的 sequence number 用于保证 ring buffer head/tail 更新不引入 race condition；(c) EFA 上模拟 atomics：将 atomic 值打包进 immediate data 的 RDMA write，接收端 CPU 更新 host memory counter。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
