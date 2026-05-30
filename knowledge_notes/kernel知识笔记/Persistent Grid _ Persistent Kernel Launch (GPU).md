## Persistent Grid / Persistent Kernel Launch (GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Persistent Grid（持久化网格）是一种 GPU kernel 调度策略：launch 恰好等于 GPU 物理 SM 数量的 thread block（H100 为 132 个），让每个 block 常驻在其 SM 上，通过 task iteration 循环处理多个 tile 任务。每个 block 完成当前 task 后不退出，而是加载下一个 task 坐标继续执行。ThunderKittens 用 persistent grid：(1) 消除重复 block launch 的 setup/teardown 开销；(2) 利用 load worker 在 finish 阶段预取下一个 task 的数据，消除 pipeline bubble；(3) 配合 block order scheduling 最大化 L2 cache reuse。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Host: 只 launch 132 个 block（== H100 SM 数量）
dim3 grid(132); // PERISISTENT_GRID=true
// Device 端伪代码:
while (task_id < total_tasks) {
    common_setup: task_id = task_iter * 132 + blockIdx.x;
    // Load→Compute→Store→Finish 流水线处理当前 task
    for (iter over K tiles) { tma::load + warpgroup::mma }
    store output to HBM;
    // 预取下一个 task 的输入 (overlap with current finish)
    task_iter++;
}
```
TK 实验（GEMM, M=N=4096）：persistent vs non-persistent——K=64: 108 vs 93 TFLOPS (+16%); K=128: 184 vs 161 (+14%); K=256: 309 vs 271 (+14%); K=512: 450 vs 414 (+9%); K=1024: 600 vs 565 (+6%)。大 K 时优势递减，因 compute 时间增长使 launch 开销占比减小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 通过 `kittens::prototype::lcf::kernel<template>` 的 `PERISISTENT_GRID` 模板参数实现——true 时 host grid 固定为 132，device 端自动在 common_setup 中用 while 循环 + task_id 映射替代一次性遍历。需配合 block order scheduling 最大化 L2 reuse。局限性：总 task 数少于 SM 数时部分 SM 闲置；极计算密集的 kernel 收益递减。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels
