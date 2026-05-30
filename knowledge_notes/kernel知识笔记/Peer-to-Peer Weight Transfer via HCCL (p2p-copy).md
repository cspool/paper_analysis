## Peer-to-Peer Weight Transfer via HCCL (p2p-copy)

术语是什么？

p2p-copy 是 ElasticMoE 的跨 NPU 高速 P2P 张量传输原语，在缩放时用于将模型权重从已有 NPU 传输到新增 NPU。与从磁盘加载（最慢链路）不同，p2p-copy 通过 Ascend HCCL 集合通信库经 Unified Bus 或 RDMA 链路直接进行 device-to-device 传输，绕过 host memory，比磁盘 I/O 快约一个数量级。核心 API：HCCL `isend`/`irecv`/`broadcast` + CANN `aclrtMemcpyAsync`。

从 kernel 调度角度拆解术语：

```
p2p-copy 操作伪代码：

// 初始化（一次性）
init_process_group(npus=all_devices, backend="hccl")

// 缩放时 P2P 传输
void p2p_copy_weight(src_npu, dst_npu, tensor_name, partition):
    stream = aclrtCreateStream()  // 可选独立 stream 避免阻塞计算
    src_tensor = hmm.get_tensor(tensor_name, partition, src_npu)
    dst_tensor = aclrtMalloc(shape, dtype, dst_npu)
    aclrtMemcpyAsync(dst_tensor, src_tensor, size,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream)
    // 或使用 HCCL: isend/irecv 异步 P2P
    aclrtSynchronizeStream(stream)
```

传输路径：NPU A HBM → Ascend Unified Bus (intra-node) 或 RDMA (cross-node) → NPU B HBM。整个链路不经过 CPU host memory。

术语一般如何实现？如何使用？

基于 HCCL (Huawei Collective Communication Library) + CANN runtime API。Ablation 表明禁用 HCCL P2P 后 scale-up 延迟从 3.14s 升至 10.42s（约 3.3× 变慢，回退到磁盘 I/O）。在 CUDA 生态中等效使用 NCCL `ncclSend`/`ncclRecv` + `cudaMemcpyDeviceToDevice`。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
