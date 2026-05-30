## TileLang JIT Compiler Pipeline

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TileLang JIT Compiler Pipeline 是 TileLang 的即时编译流水线，将 Python-embedded tile 程序编译为可在目标硬件上执行的高性能 binary。编译流水线包含五个阶段：Parser → IR Builder → Optimization → Codegen → Runtime。核心特点是支持多后端（NVIDIA CUDA、AMD HIP、LLVM IR/CPU），且每个阶段的优化 pass 可组合扩展。编译器约 90% 的编译时间用于 CUTLASS 模板展开（通过 NVCC 12.8 trace tool 验证），TileLang 计划未来用 self-hosting Tile Library 替代 CUTLASS 依赖以降低编译时间。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

五阶段完整流程（以 FlashAttention on H100 为例）：

```
Stage 1 — Parser:
  Python 程序 → Python AST → TileLang AST
  关键: 类型标注解析 (T.Tensor shape/dtype → 确定 device code bitwidth)

Stage 2 — IR Builder:
  TileLang AST → TVM Tensor IR
  T.alloc_shared([block_H, dim], f16) → memory scope="shared" buffer
  T.alloc_fragment([block_H, block_N], f32) → memory scope="local" buffer (register)
  T.Pipelined(loop_range, num_stages=2) → loop IR with pipeline annotation

Stage 3 — Optimization:
  a) Layout Inference: Q_shared, KV_shared → InferLayout → thread binding + vectorization
  b) Fragment Layout: acc_s, acc_o, scores_max → MakeMMASTMatrixLayout 自动分配
  c) Pipeline Derivation (Hopper): 
     - 分析 buffer 使用确定 producer/consumer 角色
     - Producer threads: TMA copy KV → shared memory
     - Consumer threads: wgmma.mma_async(Q_shared, KV_shared, acc_s)
     - Live Variable Analysis → 插入 mbarrier 同步点
  d) Warp Specialization: 按 threadIdx 分离 producer/consumer 执行路径

Stage 4 — Codegen:
  优化后 IR → CUDA C with PTX intrinsics
  - TMA: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier
  - WGMMA: wgmma.fence + wgmma.commit_group + wgmma.mma_async + wgmma.wait_group
  - mbarrier: mbarrier.arrive / mbarrier.try_wait

Stage 5 — Runtime:
  NVCC 编译 → .cubin → CUDA driver API 加载 → Kernel launch
  Kernel cache 管理: 相同配置的 kernel 避免重复编译
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

使用方式: `tilelang.compile(program, target="cuda")` 或 `@tilelang.jit` decorator。target 参数支持 "cuda"、"hip"、"llvm"。JIT 编译结果缓存在进程内，相同配置的 kernel 无需重新编译。对于 kernel library 开发者，TileLang 支持动态参数（如动态 shape）的 kernel 编译，自动应用 loop tail splitting 处理非对齐尺寸。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems
