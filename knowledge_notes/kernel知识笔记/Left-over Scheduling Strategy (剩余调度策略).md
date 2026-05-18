## Left-over Scheduling Strategy (剩余调度策略)

术语是什么？

Left-over Scheduling（剩余调度策略）是NVIDIA GPU hardware dispatch unit采用的block-to-SM调度策略：当一个新的block需要被调度时，dispatch unit检查所有SM的剩余资源（主要是available threads），选择第一个满足资源需求的SM将block分配进去。这种策略简单、硬件实现成本低，但它是导致stacked co-location的直接原因——因为同kernel的所有block需求相同，它们会被连续分配到相同的SM集合。"Left-over"一词指调度决策仅基于SM的剩余容量（left-over capacity），不考虑block之间的资源互补性。μShare正是利用了这一策略的可预测行为：通过设置half-plus blocksize使SM的"left-over capacity"对同kernel的第二个block不满足条件，从而"欺骗"硬件scheduler将blocks散布。

从kernel调度角度拆解术语：

Left-over scheduling的简化逻辑：

```
// GPU dispatch unit的block分配逻辑 (simplified from observation)
function dispatch_block(block, blocksize):
    for each SM in GPU_SMs:
        if SM.available_threads >= blocksize
            AND SM.available_shared_memory >= block.smem
            AND SM.available_registers >= block.regs:
            assign block to SM
            SM.available_threads -= blocksize
            return SM.id
    // 如果没有SM有足够资源 → block进入pending queue
    // 当某个SM的active block完成释放资源后 → retry
    return PENDING

// 单kernel场景 (blocksize=512, A40 SM=1536 threads):
//   SM0: 512 → 1024 left → block_1: 512 → 512 left → block_2: 512 → 0 left
//   SM1: 512 → 1024 left → ...
//   → stacked co-location (同kernel的3个block在SM0内)

// Half-plus场景 (blocksize=800):
//   SM0: 800 → 736 left → block_1: 800 > 736 → skip SM0 → SM1
//   SM1: 800 → 736 left → block_2: 800 > 736 → skip SM0,SM1 → SM2
//   → scattered: 每个SM仅1个half-plus block
//   → 736 left in SM0 可放 roll block (512) → co-location
```

术语一般如何实现？如何使用？

Left-over scheduling是NVIDIA GPU的硬件实现，闭源不可修改。μShare的贡献在于反向利用这一策略：
1. **正向利用**：设置half-plus blocksize → 使同kernel blocks不能满足left-over条件 → forced scattering
2. **Timer-shifted launching补充**：对unmodifiable kernel，通过延迟其launch timing使SM的left-over capacity刚好被互补kernel block利用
3. **局限性**：left-over scheduling缺乏全局优化（不考虑block间资源互补、不进行负载均衡），μShare只能在launch stage间接影响，无法在dispatch stage直接干预

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

