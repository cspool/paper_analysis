## Dual-MMA Packed Layout

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-MMA Packed Layout是LiquidGEMM提出的W4A8 weight矩阵内存布局优化。核心洞察：单个WGMMA操作每个thread需要16个UINT4元素，但LDS.128指令可一次加载32个UINT4元素。将两个连续MMA操作所需元素打包存储，使每个thread用单条LDS.128加载全部32元素。与QServe的2D layout不同，Dual-MMA采用1D layout消除shared memory bank conflict，无需swizzle或复杂packing。离线变换无运行时开销。

从kernel调度角度拆解术语：
```
传统单MMA layout:
  Thread需要16 UINT4 → LDS.32加载浪费50%数据
  ldmatrix不可用 (设计为1B/element, 4-bit不兼容)

Dual-MMA Packed Layout:
  Thread需要32 UINT4 (16 MMA_0 + 16 MMA_1) → LDS.128一次加载
  元素交错排列使MMA_0/MMA_1所需数据相邻存储
  1D layout: 连续排列, 无bank conflict, 支持8路并发LDS.128
  GMEM与SMEM layout一致 (LDG.128, 离线变换)
```

术语一般如何实现？如何使用？
离线量化阶段完成layout变换写入checkpoint。CUTLASS data layout abstraction配置TMA descriptor的block shape/stride。K_tile size必须≥64（2×32）以保证双MMA打包有效。不适用：activation端（动态在线量化，无法预排列）。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
