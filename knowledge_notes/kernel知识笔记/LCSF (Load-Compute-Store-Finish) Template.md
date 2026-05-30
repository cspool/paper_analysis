## LCSF (Load-Compute-Store-Finish) Template

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LCSF (Load-Compute-Store-Finish) Template 是 ThunderKittens 提出的统一异步 GPU kernel 编程模型，基于经典生产者-消费者范式。它将 thread block 内的 warp 划分为两种角色——load/store worker（负责 HBM ↔ shared memory 数据搬运）和 compute worker（负责 register/shared memory 内计算）——并将 kernel 分解为四个阶段函数：Load（指定从 HBM 异步加载哪些 tile 到 shared memory pipeline buffer，通过 arrive barrier 通知 compute worker）、Compute（用 tile 操作原语执行 mma/softmax 等计算，完成后通过 arrive 通知 load worker 可覆盖已消费 buffer）、Store（将结果 tile 从 shared memory 异步写回 HBM）、Finish（退出前保存最终状态）。LCSF 是 TK 对 block 级并行性的核心抽象，用户只需填充这四个函数，框架自动管理 multi-stage pipeline buffer、同步 barriers 和 TMA descriptor 创建。对比 FlashAttention-3 的"ping-pong scheduler"（手动管理两个 buffer 轮换），LCSF 用统一的 pipeline buffer 抽象替代，将 attention 实现从 2325 行 CUTLASS 代码缩减到 217 行 TK 代码。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LCSF attention kernel 执行流程（H100, d=64, 2-stage pipeline）：
```
struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12, INPUT_PIPE_STAGES = 2;

    // Producer: load worker (warp_id==0)
    static void load(args) {
        tma::expect(inputs_arrived, block.k, block.v);
        tma::load_async(block.k, globals.K, {batch, head, iter, 0});
        tma::load_async(block.v, globals.V, {batch, head, iter, 0});
    }
    // Consumer: compute warpgroups (3 warpgroups × 4 warps)
    static void compute(args) {
        warpgroup::mm_ABt(att, q_reg, block.k);     // Q @ K^T via WGMMA
        warpgroup::mma_async_wait();
        sub_row(att, att, max_vec);   // online softmax
        exp(att, att);  div_row(att, att, norm_vec);
        copy(att_bf16, att);           // fp32 → bf16
        warpgroup::mma_AB(o_reg, att_bf16, block.v); // att @ V
        warpgroup::mma_async_wait();
        arrive(inputs_finished);       // 通知 load worker 该 input buffer 可覆盖
    }
    static void store(args)  { tma::store_async 结果 → HBM }
    static void finish(args) { div_row 最终归一化 + store }
};
```
时间线：load worker (warp 0) 通过 TMA 异步加载 K/V tile 到 2-stage buffer slot[0]→arrive→compute 在 slot[0] 上执行 while load worker 预取 slot[1]→compute arrive(inputs_finished) 释放 slot[0]→循环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 通过 C++ template metaprogramming 实现：(1) 用户定义 LCSF template struct，填充 Load/Compute/Store/Finish 四个静态函数；(2) 设置编译时常量 NUM_CONSUMER_WARPS（compute worker 数量）和 INPUT_PIPE_STAGES（pipeline 深度）；(3) 使用 TK tile 原语（warpgroup::mma_ABt, tma::load_async, arrive）编写逻辑；(4) 调用 kittens::prototype::lcsf::kernel<template> 启动。通过 NUM_CONSUMER_WARPS 调节 occupancy——更多 worker 增加并行度但减少每 worker 的寄存器配额。LCSF 已验证通用性：用于 GEMM (40行)、attention (217行)、long convolution (131行)、Mamba-2 (192行)、linear attention (282-316行)、rotary (101行)、fused layernorm (146行) 等多种 workload，全部匹配或超过 state-of-the-art 性能。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels
