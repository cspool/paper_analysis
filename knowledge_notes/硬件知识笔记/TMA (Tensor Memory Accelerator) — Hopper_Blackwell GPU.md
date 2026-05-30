## TMA (Tensor Memory Accelerator) — Hopper/Blackwell GPU

术语解释
Tensor Memory Accelerator (TMA) 是 NVIDIA Hopper (H100) GPU 引入的专用硬件拷贝引擎，支持 GMEM-HBM 与 SMEM 之间的异步批量张量数据传输，无需经过寄存器文件。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TMA 支持 1D-5D tensor 双向传输，核心特性：(1) 单线程发起——一个线程发出 TMA transfer，其余线程可继续计算；(2) 无寄存器中转——数据直接在 HBM↔SMEM 移动（vs cp.async 需经过 SMEM buffer）；(3) 硬件加速 mbarrier——SM 硬件专门加速 barrier wait；(4) multicast——从 HBM 同时拷贝到同一 thread block cluster 内多个 SM 的 SMEM；(5) swizzling——自动优化数据布局避免 SMEM bank conflict；(6) descriptor-based——通过 host 端 cuTensorMapEncodeTiled API 创建 tensor map descriptor。SonicMoE 重度依赖 TMA：(a) TMA load: forward/backward GEMM prologue 加载权重和输入；(b) TMA store: forward down-proj Y kernel 和 backward up-proj dX~ kernel 使用异步 TMA store 代替 st.global scatter（避免同步 store 阻塞 MMA next tile）；(c) 异步 TMA load H: backward dH kernel epilogue 异步加载 cached pre-activation H 以计算 dSwiGLU。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SonicMoE 中 TMA store vs st.global scatter 的对比（H100 forward down-proj Y kernel）：

```
// 方案A: TMA store (SonicMoE) — 异步
wgmma(tile_i) → accumulator ready
tma_store(Y_expert[tile_i], desc)    // 异步发起，不阻塞
// MMA warp 立即处理 tile_{i+1}    // TMA engine 后台写入 HBM
// 结果: MMA 和 store 完全 overlap

// 方案B: st.global + scatter fusion (ScatterMoE) — 同步
wgmma(tile_i) → accumulator ready
for each token in tile:
    st.global(Y[scatter_idx[t]], tile_result[t])  // 同步操作
// MMA warp 等待所有 store 完成才处理 tile_{i+1}
// 结果: 同步 store 成为 bottleneck, ~20% TFLOPS 下降
```

SonicMoE 选择方案 A (TMA store)，然后在 expert aggregation O kernel 中用 TMA gather 收集各 expert 的 Y_e。Figure 16 显示 TMA store 方案比 st.global scatter 快 20.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA 中通过 `cuda::memcpy_async` + tensor map descriptor 使用。CUTLASS 3.x+ (CuTe-DSL) 完整支持 TMA。Triton 实验性支持 `tl._experimental_descriptor_load/store`。仅 H100+ (Hopper, Blackwell) 支持。使用前提：数据访问模式需能用规则 tensor 坐标描述（非随机访问），需预先创建 tensor map descriptor（有 host-device overhead）。SonicMoE 在 CuTe-DSL 中直接使用 TMA PTX 指令。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
