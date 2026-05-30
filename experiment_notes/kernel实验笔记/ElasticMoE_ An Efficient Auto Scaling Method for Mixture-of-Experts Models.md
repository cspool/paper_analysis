## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **ElasticMoE HMM Data Plane 中的五个低层 C++ primitives**，运行在 Ascend 910C NPU 上，为弹性缩放提供高效的设备内存管理和跨 NPU 数据传输基础：(1) **IpcSafeAllocator**：覆盖 PyTorch 默认 `TorchCachingAllocator`，用 `torch.ones()`、`torch.empty()`、`torch.full()` 等核心分配函数来分配 IPC 兼容的物理内存区域（Ascend IPC API 标记），使得张量可直接跨进程共享而不需额外拷贝；(2) **disk-copy**：按张量名称/partition index（TP rank）/layer type 选择性从磁盘加载权重到目标 NPU，避免同一张量在不同 NPU 上被多次从磁盘读取，最小化最慢的磁盘→NPU 链路使用；(3) **p2p-copy**：通过 Ascend HCCL 集合通信库（isend/irecv/broadcast）经 Unified Bus 或 RDMA 链路进行 NPU 间异步 P2P 传输，使用 `aclrtMemcpyAsync` API 直接 device-to-device 传输绕过 host memory，可选独立 stream 避免阻塞当前 NPU 计算上下文，比磁盘 I/O 快一个数量级；(4) **zero-copy**：通过 `rtIpcSetMemoryName()` 注册内存句柄 + `rtSetIpcMemPid()` 白名单目标进程 + UNIX domain socket 传输句柄 + `rtIpcOpenMemory()` 导入 + `torch::from_blob()` 封装，实现跨进程零拷贝张量共享，避免数据实际传输；(5) **vpage-remap**：通过 `aclrtMallocPhysical` 分配非连续物理页 → `aclrtReserveMemAddress` 预留连续虚拟地址范围 → `aclrtMapMem` 映射物理页到虚拟地址空间，使 kernel 将 expert 权重视为连续张量（满足 GEMM kernel 要求），而底层物理放置灵活可重映射，缩放时只需更新映射而无需重新分配和拷贝整个缓冲区。实验通过 Ablation 量化各组件贡献（Table 1/Table 3）：逐步禁用 IPCAlloc→HCCL→PreInit→ZeroCopy，测量 DP3↔DP4 缩放时间和 peak memory。ElasticMoE full：scale-up 2.43s / 0 downtime / 275.2 GB peak memory。

- 后端平台是什么，配置是什么。
  **Ascend 910C NPU**（每颗 64 GB HBM），部署在 Huawei CloudMatrix384 supernode 中。使用 **Huawei CANN (Compute Architecture for Neural Networks)** API 进行设备内存管理、HCCL (Huawei Collective Communication Library) 进行集合通信和 P2P 传输。Ascend Unified Bus (UB) 提供 NPU 间 non-blocking all-to-all 高带宽互联，也支持 RDMA 跨节点链路。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 ElasticMoE HMM data plane 的 C++/PyBind11 实现，通过 PyBind 暴露给 Python 层调用。核心评估方式为 Ablation study：在 DP3→DP4 scale-up 上逐步禁用各 primitive，重复 3 次取均值。评估使用以下自定义组件：(a) IpcSafeAllocator 覆盖 PyTorch 默认分配器；(b) p2p-copy 使用 HCCL `init_process_group` 建立跨设备通信域，`isend/irecv/broadcast` 进行异步传输，`aclrtMemcpyAsync` 进行设备间拷贝；(c) zero-copy 使用 Ascend `rtIpcSetMemoryName/rtIpcOpenMemory` + `rtSetIpcMemPid` 白名单机制；(d) vpage-remap 使用 `aclrtMallocPhysical/aclrtReserveMemAddress/aclrtMapMem` 进行虚拟内存映射；(e) disk-copy 使用 CANN API 按 filter（name/partition/layer）选择性磁盘加载。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供独立开源代码。ElasticMoE 的 kernel 级别 primitives 评估原理和执行流程：

  ```
  === Ablation 实验流程 (Table 1: DP3→DP4 scale-up) ===
  
  Step 1: IPC 兼容内存分配 (IpcSafeAllocator)
    输入: PyTorch tensor shape, dtype
    执行: 拦截 torch.ones/empty/full → 调用 CANN aclrtMalloc (IPC-compatible flag)
    → 返回物理内存地址
    输出: 可跨进程共享的张量
    禁用效果: scale time 2.43→3.14s (+29%), peak mem 275.2→290.0 GB (+5.4%)
  
  Step 2: P2P 数据传输 (p2p-copy)
    输入: source NPU tensor handle, target NPU id, tensor size
    执行: 初始化 HCCL domain → target NPU 通过 aclrtMalloc 分配目标张量
    → aclrtMemcpyAsync 异步传输 (Unified Bus/RDMA)
    → 可选独立 stream 避免阻塞计算
    输出: target NPU 上的张量副本
    禁用效果: scale time +IPCAlloc 3.14→+HCCL 10.42s (+232%, 慢一个数量级)
    替代方案: 退化为从磁盘逐张量加载
  
  Step 3: Zero-Copy 跨进程张量共享 (zero-copy)
    输入: IpcSafeAllocator 分配的张量, source process PID, target process PID
    执行: 
      (a) HMM 通过 rtIpcSetMemoryName(tensor_ptr, name) 注册内存句柄
      (b) HMM 通过 rtSetIpcMemPid(pid) 白名单目标进程
      (c) 通过 UNIX domain socket (ZMQ) 传输句柄名称到 IMM
      (d) IMM 通过 rtIpcOpenMemory(name) 导入物理指针
      (e) IMM 通过 torch::from_blob(ptr, shape, dtype) 封装为 PyTorch tensor
    输出: 两个进程引用同一物理内存（零数据拷贝）
    禁用效果: scale time +HCCL+PreInit 62.78→+ZeroCopy 67.40s, 引入 67.40s downtime
  
  Step 4: 虚拟 Expert 管理 (vpage-remap)
    输入: expert 权重张量列表, 每个 expert 的物理页大小, 目标虚拟地址布局
    执行:
      (a) aclrtMallocPhysical 为每个 expert 分配独立非连续物理页
      (b) aclrtReserveMemAddress 预留连续虚拟地址空间
      (c) aclrtMapMem 将各物理页绑定到虚拟地址对应偏移
      (d) 缩放时: 更新虚拟→物理映射指向新页 (本地分配或 p2p-copy 接收)
      (e) 旧映射保持活跃直到新推理实例接管
      (f) 过渡完成后: 解绑旧物理页, 释放
    输出: kernel 视角为连续张量（满足 GEMM 对齐要求），物理内存灵活可重映射
    收益: 避免 EP 重配置时的大缓冲区重新分配和全量拷贝，降低 peak memory 和延迟
  
  === 性能输出 (Ablation Table 1) ===
  完整 ElasticMoE:       scale-up 2.43s, downtime 0, peak mem 275.2 GB
  - IPCAlloc:            scale-up 3.14s, downtime 0, peak mem 290.0 GB
  - IPCAlloc - HCCL:     scale-up 10.42s, downtime 0, peak mem 290.0 GB
  - IPCAlloc - HCCL - PreInit: scale-up 62.78s, downtime 0, peak mem 290.0 GB
  - All disabled (no ZeroCopy): scale-up 67.40s, downtime 67.40s, peak mem 290.0 GB
  ```

  关键结论：(a) ZeroCopy 对消除 downtime 最关键——无 ZeroCopy 时 downtime=scale time；(b) HCCL P2P 比磁盘加载快约一个数量级（10.42s vs 62.78s）; (c) PreInit（IMM 预热实例）贡献最大延迟改善，从 62.78s 降到 10.42s；(d) IPCAlloc 主要降低 peak memory 而非延迟（-5.4% peak mem）；(e) 全部四个机制联合作用才能实现低延迟、零停机缩放。
