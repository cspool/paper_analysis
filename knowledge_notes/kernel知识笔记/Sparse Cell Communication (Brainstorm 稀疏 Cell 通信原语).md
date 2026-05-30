## Sparse Cell Communication (Brainstorm 稀疏 Cell 通信原语)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Cell Communication 是 Brainstorm 框架实现的多 GPU 间 Cell 路由通信原语，用一组点对点（point-to-point）send/recv 操作替代传统 all-to-all collective。其动机是动态网络中 Router 的 Cell 分发不均匀——某些 (src_gpu, dst_gpu) pair 间传输的 Cell 数远小于其他 pair。传统 all-to-all 需要将所有 GPU pair 的传输量 padding 到 equal size，导致大量冗余通信。而 sparse communication 按实际 Cell 数量逐对传输，避免 padding。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 输入: routes[N_cells] — 每个 Cell 的目标 (gpu_id, branch_id)
//       cells[N_cells] — Cell 数据
// 输出: 每个 dst GPU 收到其负责的 Cells，按 branch 组织

// Step 1: 统计每个 (src, dst) pair 的 Cell 数量
for cell_id in 0..N_cells-1:
    dst_gpu = routes[cell_id].gpu_id
    send_counts[src_gpu][dst_gpu]++

// Step 2: All-to-all 交换 send_counts 得到 recv_counts
// (仅交换元数据，数据量极小)

// Step 3: 生成 point-to-point send/recv 计划
for dst_gpu in 0..num_gpus-1:
    if send_counts[src_gpu][dst_gpu] > 0:
        pack_cells_for_dst(dst_gpu)  // 将发往同一 GPU 的 Cell 打包
        schedule_nccl_send(buf, send_counts * cell_size, dst_gpu)
    if recv_counts[dst_gpu][src_gpu] > 0:
        schedule_nccl_recv(buf, recv_counts * cell_size, dst_gpu)

// Step 4: 执行所有 point-to-point 传输（可并行）
execute_all_scheduled_transfers()

// vs. 传统 All-to-All: 
// 每个 src→dst pair 都传输 max_count 个 Cell，总传输量为 N_gpus^2 * max_count
// Sparse: 总传输量为 sum(actual_counts)，节省 sum(padding_counts)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 用 ~3,000 LOC C++/CUDA 实现。在 CUDA 层面使用 NCCL 的 point-to-point send/recv API（ncclSend/ncclRecv），而非 ncclAllToAll。Pack 操作用 custom GPU kernel 完成 Cell 的 gather-scatter 重排列。Micro-benchmark 显示：1024 Cells（512 float32 each），4 branch/GPU，2 GPU 加速 2.13×，8 GPU 加速 2.66× vs NCCL all-to-all。加速随着 branch 数和 Cell 大小增加而放大（更多 padding 被省去）。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
