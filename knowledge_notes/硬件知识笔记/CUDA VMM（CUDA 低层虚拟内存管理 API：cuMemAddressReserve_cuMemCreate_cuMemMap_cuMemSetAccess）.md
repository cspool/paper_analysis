## CUDA VMM（CUDA 低层虚拟内存管理 API：cuMemAddressReserve/cuMemCreate/cuMemMap/cuMemSetAccess）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA VMM 是 CUDA Driver API 的虚拟内存管理接口（官方文档 group__CUDA__VA，CUDA 11 引入），把 GPU 虚拟地址预留与物理内存提交解耦：cuMemAddressReserve 预留 VA 范围（不占 HBM）；cuMemCreate 按 CUmemAllocationProp 分配物理页并返回 CUmemGenericAllocationHandle；cuMemMap 把 handle 映射进预留 VA（当前 offset 须为 0）；cuMemSetAccess 设置各设备访问权限（映射后默认不可访问）；cuMemUnmap/cuMemRelease/cuMemAddressFree 回收。分配粒度由 cuMemGetAllocationGranularity 查询（设备内存默认最小/推荐 2 MB；vAttention 修改开源 UVM 驱动支持 64/128/256 KB 小页）。关键语义：同一物理页可映射到多个 VA 范围（别名映射，ConServe 前缀缓存共享的基础）；cuMemMap 之后必须 cuMemSetAccess 才可访问；不能对已映射范围的子区间单独 unmap。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VMM 调用直接驱动 GPU 内存管理硬件：map/unmap 更新 GPU 页表条目、触发 TLB 失效（shootdown）——这是"重映射有硬件代价"的来源。ConServe 的用法：resize 时在 layer ℓ 完成后对新 slice 该层段 cuMemMap/cuMemSetAccess 绑到旧 slice 的同一物理页——只改页表、不搬 KV 字节；旧映射保留到迭代结束，在飞 kernel 观察稳定 VA 镜像；批量 cuMemUnmap + TLB invalidation 延迟到空闲窗口执行，翻译失效完全移出关键路径。微基准延迟（2 MB 页）：cuMemAddressReserve 2 µs、cuMemCreate 29.2 µs、cuMemMap 1.9 µs、cuMemSetAccess 36.8 µs、cuMemUnmap 34.3 µs、cuMemRelease 24 µs、cuMemAddressFree 1.6 µs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：GMLake（训练显存 stitching 去碎片）、vAttention（KV 按需映射）、ConServe（conversation 级 slice 弹性增长）等；CUDA 内存池 cudaMallocAsync/cudaMemPool 是其在运行时库上的高层封装。使用注意：分配/映射是驱动调用、存在驱动内部锁竞争，须与 GPU 计算重叠（ConServe 重叠窗口约 (L−1)/L·T_iteration，94.7% resize 全隐藏）；跨进程共享可用 cuMemExportToShareableHandle/cuMemImportFromShareableHandle。Web 证据：NVIDIA Driver API 文档（https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html）与官方 blog（https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/）确认 API 语义与 2 MB 粒度。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
