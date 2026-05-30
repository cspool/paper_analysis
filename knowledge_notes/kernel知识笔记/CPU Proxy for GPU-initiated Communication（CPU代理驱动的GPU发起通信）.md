## CPU Proxy for GPU-initiated Communication（CPU代理驱动的GPU发起通信）

术语是什么？
CPU proxy 是 UCCL-EP 的核心架构组件：一组在 CPU 上运行的多线程代理，接收 GPU 通过 FIFO channel 发送的 TransferCmd（128-bit 紧凑命令），解析后通过 libibverbs（可移植 RDMA 库）发出 GPUDirect RDMA 操作，并负责强制执行 delivery semantics（ordering、completion fence、barrier）。每个 GPU 分配 1 个 CPU proxy（含 4 worker threads），不同 threads 无共享状态、无需同步。

从kernel调度角度拆解术语：
```
// CPU proxy thread 执行流程:
while True:
    // 1. Poll assigned FIFO channels
    for each channel in my_channels:
        cmd = Poll(channel)  // 读取但不弹出

    // 2. 根据 cmd.type 执行
    switch cmd.type:
        Write:
            wr = build_rdma_write(cmd.dst_rank, cmd.dst_offset,
                                  cmd.length, cmd.seq_num)
            imm = pack(cmd.seq_num, cmd.expert_idx)
            ibv_post_send(qp, wr, ibv_send_flags | IBV_SEND_SIGNALED, imm)
            Pop(channel)  // 可靠传输下入队后立即弹出

        Atomics:
            // EFA: 模拟 via immediate data → host memory counter
            // CX7: 使用硬件 RDMA atomics
            ibv_post_send_atomic(qp, ...)

        Drain:
            drain_cq_until(cq, cmd.idx)  // 等待所有 in-flight 完成

        Barrier:
            hierarchical_barrier(shm, leader)  // 节点内→跨节点同步

    // 3. Poll completion queue (non-blocking)
    for cqe in poll_cq():
        if IMM in cqe:
            seq = extract_seq(cqe.imm_data)
            if out_of_order(seq):
                buffer_in_control_buffer(cqe)  // 暂存, 顺序 apply
            else:
                apply_immediately(cqe)
```

术语一般如何实现？如何使用？
UCCL-EP CPU proxy 通过 libibverbs 抽象 NIC 差异，支持 CX7 EFA Broadcom 等 NIC。每 thread 管理一组 QPs（包括 QP load balancing across NICs），负责 polling sender CQ（确认发送完成）和 receiver CQ（处理到达的消息 + ordering enforcement）。CPU 利用率从 8%（无 UCCL-EP）升至 ~22%（4 threads），远低于 GPU 集群中 CPU 的可用核数（128-192 cores）。与 CPU-assisted IBGDA 的区别：UCCL-EP 使用多线程 proxy + multi-FIFO channels 实现 small-message scalability + heterogeneous NIC ordering emulation。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
