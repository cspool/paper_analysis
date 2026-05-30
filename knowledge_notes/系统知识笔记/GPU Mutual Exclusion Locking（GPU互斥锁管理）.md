## GPU Mutual Exclusion Locking（GPU互斥锁管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Mutual Exclusion Locking 是 GPUSync (Elliott et al., RTSS 2013) 及其扩展提出的一类 GPU 实时管理方法。核心思想：将 GPU 的各 engine（compute engine、copy engine）视为独立可锁资源，GPU-using task 在使用前必须 acquire 对应 engine 的 mutual-exclusion lock，使用后 release。这使得 GPU 调度问题转化为经典的实时资源管理问题——已有的大量实时 locking protocol 和 response-time analysis 可直接应用于 GPU-using task。该方法的优势是实现简单、分析成熟；缺点是 capacity loss（task 持锁期间 GPU engine 可能未被充分利用）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

基于 GPUSync 方法和 Bakita & Anderson 的修正：

```
原GPUSync方法 (基于错误假设——copy engine独立):
  系统初始化:
    CUDA报告2个copy engine → 创建:
      Lock_CPU_to_GPU: 对应 Copy Engine 0
      Lock_GPU_to_CPU: 对应 Copy Engine 1
      Lock_Compute:     对应 Compute Engine
  
  Task1 (CPU→GPU copy + graphics):
    acquire(Lock_CPU_to_GPU)  # 获取copy engine 0
    acquire(Lock_Compute)      # 获取compute engine
    // 执行...
    release(Lock_Compute)
    release(Lock_CPU_to_GPU)
  
  Task2 (GPU→CPU copy):
    acquire(Lock_GPU_to_CPU)  # 获取copy engine 1
    // 两个task的lock无交集 → 被认为可并行执行
    // 但实际RTX 6000 Ada上(R8): 
    //   GRCE0(处理CPU→GPU copy)可能共享LCE2(处理GPU→CPU copy)的PCE
    //   → 一个PCE被竞争 → copy时间翻倍 → execution time bound被打破

Bakita & Anderson修正 (基于R6, R8):
  系统初始化:
    通过nvdebug检查PCE-LCE-GRCE映射(Fig.11) → 发现PCE共享
    仅创建实际独立的copy lock数目
    
  正确的lock分配 (基于硬件PCE独立性):
    Lock_PCE0: 对应PCE0上的所有LCE和GRCE
    Lock_PCE1: 对应PCE1上的所有LCE (如果存在且未被GRCE共享)
    Lock_Compute: 对应Compute Engine
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

正确实现 GPU mutual exclusion locking 的前提：(i) 通过 nvdebug 的 device_info 和 lce_for_pce / shared_lce_for_grce 接口获取正确的 engine 独立性映射；(ii) 在实时系统启动时而非运行时检查硬件配置（因为 PCE-LCE 映射被认为是硬编码常量）；(iii) 考虑 runlist 共享导致的非独立调度（R6, 尤其 Runlist 0 的 compute+copy 共 runlist 可能引入额外干扰）；(iv) 在高 end GPU 上（如 RTX 6000 Ada, 17 runlists + 5 PCE），per-engine locking 可能是安全的——但前提是逐一验证 engine 独立性。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
