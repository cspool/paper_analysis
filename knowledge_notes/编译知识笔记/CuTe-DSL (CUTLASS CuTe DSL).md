## CuTe-DSL (CUTLASS CuTe DSL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CuTe-DSL 是 NVIDIA CUTLASS 库（CUDA Templates for Linear Algebra Subroutines）的 C++ template-based domain-specific language，用于编写高性能 GPU GEMM kernel。CuTe 的核心抽象是将矩阵运算分解为 tile、atom 和 copy 操作，通过模板参数静态调度 warp specialization、TMA load/store、WGMMA/UMMA 指令和 pipeline synchronization。不同于 Triton（Python DSL → MLIR → PTX），CuTe 直接生成 PTX 和 SASS，给开发者对异步硬件操作的完全控制。SonicMoE 完全基于 CuTe-DSL 编写 8 个 MoE kernel，利用其 warp-specialized pipeline、TMA descriptor、Ping-Pong scheduling 和 persistent tile scheduler 等特性。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
CuTe-DSL kernel 的编写和编译流程（以 SonicMoE varlen-M Grouped GEMM 为例）：

```
// 1. CuTe C++ 编写 (kernel 模板参数化)
template <typename Config>
__global__ void grouped_gemm_kernel(GemmParams params) {
    // Config 包含: tile shape, warp specialization 策略, TMA descriptor 配置
    
    // 2. CuTe 抽象映射到 PTX 指令
    // TiledMMA → WGMMA (SM90) 或 UMMA (SM100)
    TiledMMA mma = make_tiled_mma<Config::MMA>();
    // TiledCopy → TMA load / cp.async
    auto tma_load = make_tiled_copy_A<Config::CopyA>();
    
    // 3. Pipeline stages（异步 warp specialization）
    Pipeline pipe = make_pipeline<Config::Stages>();
    
    // Producer warp: TMA load
    if (thread_idx == producer) {
        copy(tma_load, gmem_tensor, smem_tensor);
        pipe.commit();
    }
    // Consumer warp 0: WGMMA + epilogue
    if (thread_idx == consumer_0) {
        pipe.wait();
        warpgroup_arrive();
        mma(acc, smem_A, smem_B);
        warpgroup_commit();
        // Epilogue: TMA store
        copy(tma_store, acc_tensor, gmem_output);
    }
}

// 4. 编译链: CuTe C++ → NVCC → PTX → SASS
// nvcc -arch=sm_90a -std=c++20 grouped_gemm.cu -o kernel.ptx
```

CuTe 对比 Triton：Triton 简化开发（Python 语法，auto-tuning），但受限于 MLIR 中间表示的抽象层——无法表达 warp-specialized 异步调度、Ping-Pong pipeline、cluster-level synchronization（mbarrier cluster scope）等底层硬件特性。SonicMoE 选择 CuTe 正是因为细粒度 MoE 的 IO-heavy 特性要求精确控制异步操作的 overlap timing。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装：CUTLASS 3.x (https://github.com/NVIDIA/cutlass)，包含在 `/include/cute/` 头文件中。CuTe 的核心 API：(1) `TiledMMA` — 定义 MMA tile 大小和指令类型；(2) `TiledCopy` — 定义数据加载策略（TMA/cp.async/ldmatrix）；(3) `make_tensor` / `make_layout` — 定义多维 tensor 的 shape 和 memory layout；(4) `Pipeline` — 构建异步 producer-consumer pipeline。SonicMoE 提供 PyTorch 封装：在 Python 侧通过 `pybind11` 绑定 CuTe kernel，用户调用 `SonicMoELayer.forward(x, routing_weights)` 即可。适合需要精确控制 GPU 异步操作延迟和 memory hierarchy 的高性能 kernel 开发，开发门槛显著高于 Triton。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
