## Driver-Level Placeholder Kernel Slots

术语是什么？通过联网搜索让回答具体和精准。
Driver-Level Placeholder Kernel Slots是Infera绕过CUDA driver kernel module加载限制的技术：NVIDIA GPU执行kernel前需通过cuModuleLoad将CUDA binary加载到module，该函数会触发global host-device synchronization，不适用于推理时的低延迟动态kernel发射。Infera在GPU memory中预留placeholder kernels（空壳kernel），运行时通过driver-level修改直接覆盖其code section为fused kernel的实际binary code，从而跳过cuModuleLoad，实现低延迟动态kernel加载。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Placeholder kernel slot的工作流程：
```
编译期（离线）:
1. 预留N个placeholder CUDA kernels
2. 编译为标准CUDA binary
3. cuModuleLoad加载到GPU → 分配kernel memory slots
4. 记录每个slot的device address和size

推理期（在线）:
1. FuseKernels生成fused kernel binary
2. 选择size足够的空闲placeholder slot
3. GDRCopy gdr_copy_to_mapping(slot_addr, fused_binary, size)
   → 直接覆盖slot的code section
4. 更新launch config (gridDim, blockDim, sharedMem)
5. DKQ入队(kernel_ptr=slot_addr, args_ptr, launch_config)
6. Daemon kernel cudaLaunchDevice(slot_addr, ...)
   → GPU执行覆盖后的fused kernel code
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera利用NVIDIA GPU kernel memory管理的特性：cuModuleLoad后kernel code section在GPU memory中固定，不会动态重定位。通过driver-level API（论文引用[21]但未详述具体API）获取slot物理地址并直接写入。关键约束：(1) fused kernel code size ≤ placeholder slot size；(2) slot数量需足够容纳并发fused kernel的峰值；(3) 覆盖操作需与daemon kernel launch正确同步（slot被占用时不可覆盖）。论文未说明driver-level修改的具体API名称，仅标注引用[21]。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
