## Virtual Memory-based Expert Management (vpage-remap)

术语是什么？

vpage-remap 是 ElasticMoE 的虚拟内存抽象，用于在 MoE EP 重配置时高效管理 expert 权重的物理布局。Expert 权重在 NPU HBM 中以非连续物理页存储（每个 expert 对应独立的 `aclrtMallocPhysical` 物理页），但通过 `aclrtReserveMemAddress` 预留的连续虚拟地址范围映射为逻辑连续张量，满足 GEMM kernel 的对齐要求。当 EP 度变化需要重新分配 expert 时，仅更新虚拟→物理映射表，无需重新分配大缓冲区或全量拷贝 expert 权重。

从 kernel 调度角度拆解术语：

```
vpage-remap 操作伪代码：

// 初始化
void init_expert_vmem(npuid, expert_list):
    total_size = sum(e.size for e in expert_list)
    va_base = aclrtReserveMemAddress(total_size)  // 预留连续 VA
    offset = 0
    for expert in expert_list:
        phys_page = aclrtMallocPhysical(expert.size)  // 非连续物理页
        aclrtMapMem(va_base + offset, expert.size, phys_page)
        offset += expert.size
    // GEMM kernel: torch::from_blob(va_base) → 视为连续张量

// EP 重配置时 (如 EP=4→EP=6)
void remap_experts(npuid, new_expert_list):
    for expert in new_expert_list:
        if expert 新到达:
            received_page = p2p_copy(expert)  // HCCL 接收
            aclrtMapMem(va_base + offset, expert.size, received_page)
        // 旧映射保持活跃直到新实例接管
    // 接管后: aclrtUnmapMem(old_pages) + aclrtFreePhysical(old_pages)
```

术语一般如何实现？如何使用？

基于 Ascend ACL API：`aclrtMallocPhysical` / `aclrtReserveMemAddress` / `aclrtMapMem` / `aclrtUnmapMem` / `aclrtFreePhysical`。在 CUDA 生态中等效使用 CUDA Virtual Memory API：`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemUnmap`。关键收益：避免 EP 重配置时重新分配大连续缓冲区和全量拷贝，降低 peak memory 和延迟。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
