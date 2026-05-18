## CMA (Contiguous Memory Allocator) for TEE Secure Memory

术语是什么？通过联网搜索让回答具体和精准。

CMA（Contiguous Memory Allocator）是Linux内核的连续物理内存分配器，用于为大块连续物理内存需求（设备DMA缓冲区、GPU/NPU framebuffer等）提供服务。在TZ-LLM中，CMA被用来为TZASC分配secure memory——TZASC要求被保护的DRAM区域必须是连续物理地址（region descriptor以物理地址起始+大小为参数），而Linux buddy allocator在系统运行后很难提供GB级连续物理页。CMA在系统启动时预留一块大连续物理内存区域（cmdline cma=配置），运行时从此区域分配；非CMA分配时这些页可作为movable页被page cache/用户页使用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TZ-LLM中CMA用于TEE secure memory的分配/释放流程：

```
// REE侧: CMA分配 (TZ driver, +197 LoC)
def extend_allocated(op_id, size):
    pages = cma_alloc(size, align=4KB)
    // cma_alloc内部: 
    //   1. CMA bitmap中查找足够连续空闲页
    //   2. 有movable页占用时调用migrate_pages()搬走
    //   3. 返回连续物理页的起始PFN
    return pages  // [pa_start, pa_end)，此时Non-Secure

// TEE侧: 升级为Secure
def extend_protected(op_id, pa_start, size):
    tzasc_add_region(pa_start, size, SECURE_ONLY)
    tee_map_pages(ta_vm_space, pa_start, size)
    // 此后REE无法访问此内存

// 释放流程 (inference后按reverse topological order)
def shrink_allocated(op_id, pa_start, size):
    tzasc_remove_region(pa_start, size)
    tee_unmap_pages(ta_vm_space, pa_start, size)
    cma_release(pa_start, size)  // REE侧归还CMA pool
```

CMA分配性能特性（与TZ-LLM pipeline的关系）：
- **分配延迟**：Llama-3-8B (7.9GB) CMA allocation约4.182s（memory stress下），取决于碎片和需migrate的page量
- **Pipeline隐藏**：allocation被pipeline化——operator 0的CMA完成后立即开始computation，后续operator的CMA在computation后台完成
- **Memory pressure**：stress-ng模拟REE memory pressure（四个模型对应13/11/10/6GB压力），增加CMA migration开销
- **碎片影响**：pipeline-aware extend/shrink利用LLM参数first-in-last-out模式减缓碎片（顺序分配逆序释放减少碎片空洞）

Buddy vs CMA：Buddy 4KB页粒度无法保证GB级连续（碎片化后）；CMA可保证GB级连续但需预留区域且migrate有开销。TZASC的连续内存需求是移动TEE保护大模型的根本限制——长期运行碎片严重时即使是CMA也可能无法分配数GB连续内存。Geekbench评估CMA分配对REE应用性能干扰最高约6.7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CMA是Linux kernel标准特性（CONFIG_CMA），通过cmdline（cma=size）或device tree配置。TZ-LLM通过REE TZ driver封装cma_alloc/cma_release供TEE远程调用。TZ-LLM通过transient allocation + pipeline overlapping缓解CMA延迟和碎片问题。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

