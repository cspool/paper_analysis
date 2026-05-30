## Attention Runtime (MetaAttention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Runtime（注意力运行时）是 MetaAttention 编译框架的执行层，将 two-layer scheduling policy 生成的 scheduling plan（IntermediateTensors 的 tile size + memory location + pipeline stage 配置）翻译为可执行的 GPU kernel。Attention Runtime 的核心职责：(1) **Kernel Template Selection**——根据 attention pattern（Parallel/Recurrent）选择对应的预定义 kernel 模板；(2) **Template Instantiation**——根据 scheduling plan 配置模板参数（tile sizes, memory buffer sizes, pipeline stages）；(3) **Code Inlining**——将 lowering 后的 customizable function code（elementwise SIMT + row-reduce warp reduction）inline 到 kernel 模板的固定注入点；(4) **Backend-specific Code Generation**——通过 backend framework (CUTE for NVIDIA CUDA, TileLang for NVIDIA/AMD) 生成最终 CUDA/ROCm kernel。

Runtime 内置通用 attention 优化：(1) Parallel pattern——实现 online row-wise normalization（行归一化的 tile-by-tile 在线计算），避免 attention score/weight 矩阵的 HBM materialization；(2) Recurrent pattern——实现 chunk parallelism [Yang et al. 2024]，将 sequence 分块并行处理，块内 recurrent update，块间传递 hidden state；(3) Multi-backend mapping——NVIDIA: TMA (cp.async.bulk) + wgmma (Tensor Core)；AMD: async copy + Matrix Core。

从编译框架角度拆解：Runtime 的执行管线：
```
SchedulingPlan(tile_sizes, mem_locations, pipeline_stages)
  → KernelTemplateSelector(pattern=Parallel/Recurrent)
    → Template Instantiation: 填充 tile sizes, mem buffers
      → CustomFuncInliner: fuse Mod/RowNorm code into mainloop
        → BackendLowering(CUTE | TileLang)
          → CUDA kernel (TMA + wgmma) | ROCm kernel (async copy + Matrix Core)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 实现两套 attention runtime backend：CUTE (NVIDIA CUTLASS 的 CUDA template library) 和 TileLang (composable tiled programming model)。CUTE backend 直接生成 CUDA C++ kernel（利用 Hopper wgmma/TMA 硬件特性），TileLang backend 通过 TileLang compiler 间接生成 CUDA/ROCm kernel（跨硬件可移植）。Runtime 的 kernel launch 对用户透明——用户调用 compiled kernel function 即可，内部自动处理 device memory allocation、stream management、kernel launch configuration。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
