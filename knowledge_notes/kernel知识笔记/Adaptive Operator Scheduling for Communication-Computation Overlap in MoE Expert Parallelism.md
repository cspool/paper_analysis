## Adaptive Operator Scheduling for Communication-Computation Overlap in MoE Expert Parallelism

术语是什么？
Adaptive Operator Scheduling 是 ScMoE 中实现专家并行通信-计算重叠的运行时调度策略。核心思想：在 ScMoE 架构解耦通信与计算的顺序依赖后，将 MoE stream 中的算子（gate routing、encode、All-to-All dispatch、expert computation、All-to-All combine、decode）自适应插入到 shared expert stream（attention → shared expert computation）的适当位置，最大化通信与计算的重叠时间。关键调度决策是为 expert computation 选择最优插入位置——在 shared expert stream 的4个候选位置（①②③④）中选择使总时间最小的位置 K*。

从kernel调度角度拆解术语：
自适应调度器选择 expert computation 位置的优化目标和伪代码：

```
# 性能模型输入（通过profiling获得）
T_disp    = All-to-All Dispatch通信时间
T_comb    = All-to-All Combine通信时间
COMP[1:4] = shared expert stream中4个计算算子各自的执行时间

# 优化目标：选择最优expert computation插入位置K
K* = argmin_K ( |Σ_{i=1}^{K-1} COMP_i - T_disp| + |Σ_{i=K+1}^{4} COMP_i - T_comb| )

# 调度执行（双CUDA stream并行）
# Stream 0 (Shared Expert):  [COMP_1][COMP_2]...[COMP_K-1]...[COMP_K+1][COMP_4]
#                             <-- Overlap with Dispatch -->   <-- Overlap with Combine -->
# Stream 1 (MoE):  [Gate][Encode][Async Dispatch]...[Expert Comp at pos K]...[Async Combine][Decode]

# 总时间边界：
# 下界: T_overall >= |(ΣCOMP) - (T_disp + T_comb)|
# 上界: T_overall <= (ΣCOMP) + (T_disp + T_comb)
# 当通信时间 ≤ overlap_window = ΣCOMP 时，T_overall = ΣCOMP (100%通信隐藏)
```

三种shortcut位置对应的overlap窗口：
- Pos-1: overlap_window = T_Atten + T_SE
- Pos-2: overlap_window = T_Atten + T_SE + T_MLP
- Pos-3: overlap_window = 2*T_Atten + T_SE + T_MLP

Pipeline Augmentation：当 T_disp + T_comb > overlap_window 时，ScMoE 的自适应调度与 pipeline 策略可组合使用。先用 ScMoE 的扩展窗口隐藏部分通信，剩余无法隐藏的部分通过 token 分 chunk 的 pipeline 进一步隐藏。第5条 timeline（图7）展示了这种组合。

与纯 pipeline 策略的核心区别：
- Pipeline: 将tokens等分为M个chunks，chunks间通信与计算交错。但第1个chunk的dispatch（prologue）和第M个chunk的combine（epilogue）无法被隐藏。bubble = T_disp/M + T_comb/M。
- ScMoE自适应调度: 通信的"窗口期"从时间线前端（Block-MLP计算期间）自然延伸到后端（shared expert计算期间），无prologue/epilogue概念。当 T_disp+T_comb ≤ overlap_window 时实现0%通信暴露。

术语一般如何实现？如何使用？
基于 PyTorch CUDA stream API 实现：(1) 通过 `torch.cuda.Stream()` 创建独立的 MoE stream；(2) 在训练开始前 profiling 各算子的执行时间（T_disp, T_comb, COMP_1..4），构建性能模型；(3) 每个 iteration 在 CPU 侧计算 min_K 目标函数确定 K*；(4) 在 MoE stream 上提交 gate → encode → async dispatch 操作；(5) 在主 stream 的 K*-1 位置后插入 expert computation（通过 `torch.cuda.current_stream().wait_stream(moe_stream)` 同步）；(6) expert computation 完成后在主 stream 恢复后续计算，同时在 MoE stream 提交 async combine → decode。ScMoE 在 8×A30-PCIe 场景重叠 70% 通信，8×A800-NVLink 场景完全重叠通信。

涉及论文标题：
- Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts
