## ILP/TLP/Arithmetic Intensity Trade-off in DNN Kernels

术语是什么？通过联网搜索让回答具体和精准。
ILP (Instruction-Level Parallelism)、TLP (Thread-Level Parallelism) 和 Arithmetic Intensity 是GPU DNN kernel优化的三维trade-off空间。ILP控制单warp内独立指令的并行度（通过register file中tile size和instruction scheduling控制），TLP控制SM内并发active warp数量（主要受register和shared memory per-thread使用量约束），Arithmetic Intensity = #arithmetic_ops / #bytes_accessed，控制计算与访存的比值（通过tile size调节数据复用度）。三者竞争有限的register file和shared memory资源：更高ILP需要更多register→降低TLP；更高intensity需要更大tile→更多shared memory→可能降低TLP。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Infera的形式化分析（Eq.1-4）：
```
// 执行时间 = 指令数 / IPC
#inst ≈ #inst_mem + #inst_math

// 内存指令数估算（Eq.2+3）
#inst_mem ≈ a × #bytes + b × #ops
         = #ops × (a/I + b)
// I = #ops / #bytes = arithmetic intensity

// Trade-off约束（Eq.4）
min #inst  s.t. ILP constraints, TLP constraints, intensity constraints
  ILP ↑ → #inst ↓ but register ↑ → TLP ↓
  Intensity ↑ → #inst ↓ but shared memory ↑ → TLP ↓
  TLP depends on: register/thread × thread/SM ≤ RF_size/SM
                   shared_memory/block ≤ SMEM_size/SM
```

**具体例子（GEMM kernel on A100）**：
- Low ILP, High TLP (register=64/thread): 更多active warps但每warp内指令串行依赖多
- High ILP, Low TLP (register=128/thread): 每warp内更多independent指令但occupancy降低
- 峰值性能出现在green box (Figure 4): ILP, TLP, intensity三者平衡处

**Infera的tile配置实现trade-off**：
- Register level: 64/96/128 registers/thread → ILP vs TLP
- Shared memory level: 48/80/112/144 KiB/block → tile size控制intensity
- Pipeline stages: 2/3/4 → async copy overlap
- 通过spatial vs reduction axis tile size ratio进一步调节ILP vs intensity

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera compiler在生成multi-version kernels时系统化覆盖trade-off空间：register 3种×shared memory 4种×pipeline 3种=36种基础配置组合。每种配置通过top-down tile size derivation从register level向下传播到shared memory和global memory level。Cut-and-patch instruction scheduling进一步在SASS level优化ILP（通过list scheduling减少stall cycles）。推理时SelectKernels根据当前GPU occupancy状态和kernel hazard分析选择最优配置（通过online regression model估计IPC）。该trade-off分析同时指导了Infera compiler的warp specialization设计（4 mainloop + 4 copy warps固定分配，GPU scheduler将每组4连续warp map到同一SM的4 SMSP）。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

---

