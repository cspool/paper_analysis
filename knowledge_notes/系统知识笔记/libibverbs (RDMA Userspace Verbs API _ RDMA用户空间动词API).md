## libibverbs (RDMA Userspace Verbs API / RDMA用户空间动词API)

术语是什么？
libibverbs 是 Linux RDMA 子系统的用户空间 API 库，由 Linux 社区和 NIC 厂商共同维护。它提供了一套标准的、厂商无关的接口用于创建 Queue Pairs (QPs)、提交 RDMA work requests (send/write/read/atomic)、polling completion queues (CQs) 和管理 memory regions。通过 libibverbs，应用程序可以以一种可移植的方式使用 RDMA，无需关心底层是 InfiniBand、RoCE、iWARP 还是 AWS EFA。

从系统架构角度拆解术语：
```
// libibverbs 在 UCCL-EP CPU proxy 中的作用:
// 可移植 RDMA 通信路径:

// 初始化 (CPU proxy thread):
ctx = ibv_open_device(nic_device)        // 打开 NIC (CX7/EFA/Broadcom 皆可)
pd = ibv_alloc_pd(ctx)                    // 分配 protection domain
mr = ibv_reg_mr(pd, gpu_mem_base, size,   // 注册 GPU memory for GPUDirect RDMA
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC)
qp = ibv_create_qp(pd, init_attr)        // 创建 Queue Pair
ibv_modify_qp(qp, ...)                    // 迁至 RTS (Ready To Send) 状态

// RDMA write (per TransferCmd of type Write):
wr.wr_id = cmd_id
wr.opcode = IBV_WR_RDMA_WRITE             // GPUDirect RDMA write
wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE
wr.imm_data = htonl(seq_num | expert_idx) // 32-bit immediate data
sge.addr = src_gpu_addr + cmd.src_offset  // GPU memory address
sge.length = cmd.length
wr.sg_list = &sge
wr.wr.rdma.remote_addr = dst_gpu_addr + cmd.dst_offset
ibv_post_send(qp, &wr, &bad_wr)

// Poll completion:
ibv_poll_cq(cq, num_wc, wc)
if wc.status == IBV_WC_SUCCESS:
    imm = ntohl(wc.imm_data)              // 提取远端 immediate data
    process_imm_data(imm)                 // ordering enforcement
```

术语一般如何实现？如何使用？
libibverbs 由 rdma-core 包提供（https://github.com/linux-rdma/rdma-core），是 Linux 发行版的标准组件。UCCL-EP 通过 libibverbs 实现对 NVIDIA CX7 (InfiniBand/RoCE)、AWS EFA (SRD)、Broadcom Thor-2 等 NIC 的统一支持——CPU proxy 的 NIC 适配代码只需写一次，即可在所有支持 libibverbs 的 NIC 上运行。这是 UCCL-EP 实现 O(m+n) 而非 O(m×n) 可移植性的关键：GPU 侧 kernel 需为不同 GPU 写一次（每种 GPU 一次），但 CPU-NIC 侧通过 libibverbs 对所有 NIC 通用。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
