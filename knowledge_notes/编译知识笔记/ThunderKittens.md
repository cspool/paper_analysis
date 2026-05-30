## ThunderKittens

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ThunderKittens (TK) 是一个针对 NVIDIA Hopper 架构 (SM90+) 的 GPU kernel 编程框架，由加州大学伯克利分校开发（与 FlashAttention-3 相关研究项目）。它提供比 Triton 更底层但比手写 CUDA PTX/汇编更易用的抽象层，允许开发者利用 Hopper 特有的硬件特性：TMA（Tensor Memory Access，异步数据加载引擎）、warp-group specialization（warpgroups 分工）、mbarrier（异步同步屏障）、FP8 tensor core 指令等。TK 的编程模型基于 C++ template，通过 `tk::kernel`、`tk::warpgroup`、`tk::tma` 等 API 抽象 Hopper 硬件特性。在 FlashMHF 论文中，TK 用于实现 SRAMFFN kernel 的 Hopper 优化版本（Algorithm 4-5），利用 warp-group specialization 实现 producer-consumer 异步流水线——1 个 producer warpgroup 通过 TMA 预取 K/U/V tiles 到 SRAM stage buffer，多个 consumer warpgroups 并行执行不同 sequence partition 的 FFN 计算。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

ThunderKittens 在 FlashMHF 中的使用和编译流程：

```
高层算法设计 (FlashMHF Module)
    │
    ▼
PyTorch 调用层: 将 FlashMHF FFN forward 替换为 TK kernel
    │  输入: Q/R/K/U/V tensors (PyTorch tensors on GPU)
    │
    ▼
ThunderKittens Kernel (C++ template, 编译为 CUDA binary)
    │  ┌─────────────────────────────────────────┐
    │  │ Kernel 结构 (Algorithm 4):               │
    │  │                                          │
    │  │ Grid: x=[L/(B_S·C_W)], y=H, z=B         │
    │  │                                          │
    │  │ 1 Producer Warpgroup:                    │
    │  │   - TMA prefetch Q tiles per consumer    │
    │  │   - TMA prefetch R for current subnet    │
    │  │   - TMA prefetch K/U/V into ring buffer  │
    │  │   - Manage NUM_STAGES pipeline stages    │
    │  │                                          │
    │  │ C Consumer Warpgroups (C≥2):             │
    │  │   - Each processes different x-block     │
    │  │   - MMA (tensor core): M=Q·K^T, N=Q·U^T │
    │  │   - CUDA core: SiLU, element-wise ops    │
    │  │   - MMA (tensor core): O+=S·V            │
    │  │   - Signal producer via mbarrier         │
    │  └─────────────────────────────────────────┘
    │
    ▼
nvcc / CUDA Toolkit
    │  编译 TK C++ template → SM90 PTX → SASS
    │  (TK 利用 TMA/MMA/mbarrier 等 SM90 指令)
    ▼
NVIDIA H100 GPU 执行
```

关键设计: TK 的 ring buffer pipeline — NUM_STAGES 个 K/U/V tile buffer 被轮转复用。Producer 在第 s 个 stage 被 consumer 释放后立即填入下一个 tile，consumer 在下一个 tile iteration 中处理该 stage。当 tile boundary 恰好对齐 sub-network boundary 时（d_e mod BLOCK_INTER = 0），router R 的更新与 K/U/V 更新天然同步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ThunderKittens 以 header-only C++ 库形式提供。开发者编写 `.cu` 文件，包含 TK headers，定义 kernel launch configuration 和 warpgroup 逻辑。核心 API：(1) `tk::kernel` — 定义 grid/block 维度和 entry function；(2) `tk::warpgroup` — 将 block 内 warps 分组，支持异步 barrier (mbarrier)；(3) `tk::tma` — TMA 异步 load/store 的 C++ wrapper；(4) `tk::mma` — 抽象 Tensor Core MMA 指令（支持 FP16/BF16/FP8）。对比 Triton：Triton 不暴露 warp-group 和 TMA（当前版本），因此无法在 Hopper 上达到 peak performance；TK 提供更直接的控制但需要更多 CUDA 专业知识。FlashMHF 论文同时提供了 Triton 版本（用于 consumer GPU 和开发）和 TK 版本（用于 H100 生产部署）的 SRAMFFN kernel。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- FlashAttention-3 (Shah et al., 2024) — TK 框架的来源项目
