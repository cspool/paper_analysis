## Megakernel 与 Persistent Thread Blocks（持久线程块调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Megakernel（巨核）是把多个算子合并进一次 kernel launch 的持久化执行模式：固定数量的持久线程块（persistent TB）常驻 SM，从任务列表循环取"任务"执行，用软件调度替代硬件 TB 调度器。机制源自 Rammer（OSDI'20，NNFusion 编译器）：rOperator 分解为 rTask（逻辑调度单元，原 TB 粒度），vDevice 抽象为 vEU（GPU 上映射到 SM），rTask 内用逻辑 rtask_id 替代 blockIdx 寻址。优势：消除 kernel launch/上下文准备开销、支持跨算子的细粒度并行与动态依赖门控（Rammer 为编译期静态计划，DySHARP 为动态 readiness 门控）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
__global__ void megakernel(task_list_t *tasks):
    while (tid = next_task(tasks)) != -1:      # 持久 TB 循环取任务
        switch tasks[tid].type:                # 原 TB → 任务
            case GEMM1_TB:   gemm1_row(tasks[tid])
            case DISPATCH_TB: dymultimem.st(...)
            case COMBINE_TB:  dymultimem.ld_reduce(...)
```
Annotations：持久 TB 数量 = 各 SM 组内可驻留 TB 数；"向 SM 发射 TB"被模拟为持久 TB 取任务。DySHARP 变体：取任务前先查 token tracker 表的 readiness 位（TS/OR 表），未就绪则用专用 load 指令 spin-wait，形成 readiness-gated 调度；SM 分 4 组（Dispatch/GEMM-1/GEMM-2/Combine），GEMM-1/GEMM-2 可共享 SM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Rammer（OSDI'20，约 52k 行 C++，输入 TF frozen graph/TorchScript/ONNX）在编译期生成 rTask 调度计划，运行期按计划静态映射执行；CUDA 层以 persistent kernel + 软件任务队列实现。DySHARP 的用法：megakernel 内嵌 token tracker 轮询，实现 token 级动态依赖流水（区别于 Rammer 的静态计划），用于 MoE 层四算子融合。适用：算子多、粒度细、依赖动态的负载；代价是失去硬件 TB 调度器的抢占/负载均衡，需要软件分组与容量管理。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
