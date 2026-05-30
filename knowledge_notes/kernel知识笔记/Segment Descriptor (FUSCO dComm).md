## Segment Descriptor (FUSCO dComm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Segment Descriptor 是 FUSCO 的 Data-Fused Communication Engine (dComm) 的核心抽象，灵感来源于操作系统虚拟内存管理中的段描述符机制。在 FUSCO 中，Segment Descriptor 是一个 {memory_address, size_in_bytes} 对，描述通信 payload（如 MoE token）在 GPU 内存中的一段连续区域的地址和大小。一个 descriptor list（descriptor 数组，连续存放于 GPU global memory）描述一次通信的所有 segments，发送端 descriptor 指定从哪些非连续内存位置 gather 数据，接收端 descriptor 指定将收到的数据 scatter 到哪些目标位置。通过这种统一元数据，dComm 可在单次传输中完成端到端的 structured data layout transformation——将 expert-major layout 的 token 直接转换为通信所需的 device-major layout，无需额外 permute kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// descriptor_list 连续存放于 GPU global memory
// 每个 descriptor: {uint64_t addr, uint32_t size}
// dComm GPU Producer Kernel (per-slice launch):

__global__ void dcomm_gather_kernel(
    Descriptor* desc_list, int num_descs,
    char* ring_buffer, int slice_id, int slice_size)
{
    int slice_start = slice_id * slice_size;
    int bytes_copied = 0, cumsum = 0;
    
    for (int i = 0; i < num_descs && bytes_copied < slice_size; i++) {
        if (cumsum + desc_list[i].size <= slice_start) {
            cumsum += desc_list[i].size;
            continue;
        }
        int offset_in_seg = max(0, slice_start - cumsum);
        int to_copy = min(desc_list[i].size - offset_in_seg,
                          slice_size - bytes_copied);
        // GPU copy: non-contiguous segments → contiguous ring buffer
        // Layout transformation inline during this copy
        cudaMemcpyAsync(ring_buffer + bytes_copied,
                        (char*)desc_list[i].addr + offset_in_seg,
                        to_copy, cudaMemcpyDeviceToDevice);
        bytes_copied += to_copy;
        cumsum += desc_list[i].size;
    }
    __threadfence_system();
    *slice_ready_flag = 1;  // 通知 NIC consumer: slice 就绪
}
```

接收端 scatter 逻辑镜像上述：`desc.addr` 指向 expert activation tensor 的最终目标偏移，数据从 receive buffer 直接写入最终 layout，无需二次重排。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FUSCO 的 Communication Planner 构建两级 descriptor：
- **Node-Level**：基于 token-node 矩阵，每个 destination node 仅一份 token 拷贝（deduplication），发送端 descriptor 指向原始 token 地址，接收端 forwarder descriptor 指向 receive buffer 偏移。
- **Expert-Level**：基于 token-expert 矩阵，将 forwarder 上已收 token 的 local address 映射到各 expert GPU 上 expert activation tensor 的 exact offset。

Descriptor 数组通过累计已传输字节数定位当前 active segment（O(1)），无需端点间协调。Slice 将多个 segment 打包为较大传输单元（远大于 4-14KB 的单个 token），amortize descriptor 处理开销并确保持续填充 NIC。

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
