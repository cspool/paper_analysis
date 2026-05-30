## SC-HRF Memory Model / Acquire-Release Semantics for GPU

术语是什么？
SC-HRF (Sequentially Consistent Heterogeneous Race Free) 是AMD GPU的内存一致性模型，由HSA Foundation标准化并实现在AMDGPU LLVM后端。该模型类比C++内存模型，但扩展了GPU特有的memory scope层次。SC-HRF支持标准C++ memory orderings (relaxed/acquire/release/acq_rel/seq_cst)，并定义四级同步scope：
- wavefront (warp级): 64线程内可见
- workgroup (block级): 一个CTA内可见  
- agent (device级): 整个GPU可见
- system: 跨所有GPU和CPU全系统可见

对于多GPU通信，Iris使用agent-scope（单节点内跨GPU）和system-scope（需要更广可见性），配合acquire/release语义保证跨GPU操作的正确顺序。

从硬件架构角度拆解术语：
Acquire/Release在Iris workgroup specialization模式中的作用：
```
GEMM Worker (pid 0-255):                  COMM Worker (pid 256-303):
  // 产出tile后发信号                          // 等待信号后取数据
  tl.store(C + offset, c)                  while tl.atomic_cas(
  tl.atomic_cas(                              locks + tile_id, 1, 0,
    locks + tile_id, 0, 1,                    sem="acquire",
    sem="release",  // ← release               scope="gpu"
    scope="gpu"       // 确保C的store           ) == 0:
  )                    // 在cas之前可见        pass  // spin-lock
                                          // acquire确保看到C的最新值
                                          iris.put(C, remote_rank, ...)
```
- release语义: GEMM worker的tl.store(C) must happen-before tl.atomic_cas(release)，即C写操作在cas之前对同一scope的其他agent可见
- acquire语义: COMM worker的tl.atomic_cas(acquire)成功后，之后的iris.put must see the latest value of C
- scope="gpu": 同步在agent(device)级可见，足以覆盖同节点内所有GPU（通过Infinity Fabric coherence协议）

术语一般如何实现？如何使用？
在Iris的Triton kernel中，通过Triton的atomic API直接使用：`tl.atomic_cas(ptr, cmp, val, sem="release", scope="gpu")`。Triton将这些语义lower到AMDGPU的buffer_atomic指令（如buffer_atomic_cmpswap with sc0/sc1 bits controlling scope），LLVM AMDGPU后端根据scope参数设置缓存flush/invalidation范围。开发者无需理解底层缓存一致性协议。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
