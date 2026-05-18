## Cut-and-Patch Instruction Scheduling

术语是什么？通过联网搜索让回答具体和精准。
Cut-and-Patch Instruction Scheduling是Infera compiler提出的SASS级别指令调度优化技术：关闭nvcc优化生成CUDA binary→反汇编为SASS→切出mainloop computation segment→用list scheduling算法最小化stall cycles→每64条指令插入yield flag平衡warp进度→将优化后的segment插回原binary。该技术实现精准的ILP控制，因为ILP同时依赖resource控制（register/shared memory）和instruction pattern，而后者需要在machine-level assembly上直接操作。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Cut-and-patch instruction scheduling的运转流程：
```
1. nvcc编译（关闭优化）
   CUDA C++ → PTX → SASS binary
   (关闭nvcc优化: loop unrolling, instruction aggregation等)
        ↓
2. dsass反汇编
   SASS binary → SASS assembly text
        ↓
3. Cut: 切出mainloop segment
   识别on-chip computation的核心循环体
   保留epilogue/prologue部分不变
        ↓
4. List Scheduling优化
   输入: mainloop指令序列 + 指令延迟表 + 寄存器依赖图
   算法: priority-based list scheduling [36]
   目标: 最小化total stall cycles
   输出: 重排后的指令序列
        ↓
5. Insert Yield Flags
   每64条指令插入yield flag [16]
   目的: 平衡warp进度，防止单warp超前过多卡在barrier
        ↓
6. Patch: 插回优化segment
   将优化后的mainloop替换回SASS binary
        ↓
7. 最终binary
   优化的CUDA binary → static library
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera使用NVIDIA的dsass工具反汇编CUDA binary，list scheduling算法基于经典compiler backend技术（如[36]所述）。关键实现细节：(1) 关闭nvcc优化防止loop unrolling破坏tile size精心构造的ILP/TLP balance；(2) yield flag每64指令插入基于GPU warp scheduler的round-robin特性，防止某warp远远超前在barrier处stall；(3) cut-and-patch仅在mainloop computation segment操作，不改变global memory access pattern和synchronization structure。与直接编写SASS相比，cut-and-patch降低了错误率（利用nvcc处理复杂的binary格式），同时保持对instruction pattern的精细控制。Infera compiler生成的kernel平均性能比Ansor/MetaSchedule/Roller/cuDNN至少高5%。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
