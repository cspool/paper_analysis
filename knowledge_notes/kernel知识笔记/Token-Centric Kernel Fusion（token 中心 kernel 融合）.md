## Token-Centric Kernel Fusion（token 中心 kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
本论文提出的通信感知调度技术：把 MoE 层的 Dispatch→GEMM-1→GEMM-2→Combine 四个算子重组为 token-paced pipeline。核心洞察：算子间依赖可以在 token/tile 粒度判定，因此某 token（或 tsize=128 个 token 的 tile）的输入一就绪即可执行对应操作，无需等待算子级全局完成。通过显式跟踪 token 级依赖并在 readiness 边界调度，Dispatch（GPU→交换机方向为主）与 Combine（交换机→GPU 方向为主）并发执行，把 in-switch 多播/归约造成的互补非对称流量合并，双向带宽利用率提升——从而把 dynamic multimem addressing 削减的流量转化为真实加速（消融：仅流量削减因方向不对称不加速；仅 kernel 融合不超过 COMET；二者缺一不可）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# megakernel 内持久 TB 循环（SM 分 4 组：A=Dispatch, B=GEMM-1, C=GEMM-2, D=Combine）
while (task = fetch_task(task_list)) != NULL:
    switch task.type:
      case DISPATCH: dymultimem.st(token_tile)                      # SM 组 A
      case GEMM1:    if TS_Table.row_ready(tile): gemm1(tile)       # SM 组 B
      case GEMM2:    if TS_Table.row_ready(tile): gemm2(tile)       # SM 组 C（可共享）
      case COMBINE:  if OR_Table.nReady(token) == topk:
                         dymultimem.ld_reduce(token)                # SM 组 D
```
Annotations：GEMM-1/GEMM-2 无就绪 TB 时可共享 SM；ready 检查用专用 load 指令 spin-poll tracker 表；tile 尺寸 128 = GEMM tile 尺寸。时间轴效果：同一时刻 tile i 的 Dispatch 在发、tile i-2 在 GEMM、tile i-4 的 Combine 在归约——双向链路同时被使用，互补的非对称流量互相填补。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 = megakernel + persistent TB（Rammer rTasks 机制）+ token tracker 三表（TS/TID/OR）+ 软件 readiness 轮询。适用场景：任何"动态多目标通信 + 依赖可 token 化"的 MoE 训练/推理；必须与 in-switch computing 绑定使用（单独 kernel fusion 不带来加速）。对比：FasterMoE/Tutel 是算子级粗粒度重叠，CCFuser/COMET 是 Dispatch-GEMM / GEMM-Combine 细粒度重叠但两通信算子仍隔离，DySHARP 是四个算子全链路 token 级流水。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
