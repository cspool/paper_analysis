## Shape-Specific Tiling (Kernel Generation)

术语是什么？
Shape-Specific Tiling是KernelEvolve自动生成的kernel优化策略：针对production deployment的特定tensor shape distribution定制tile尺寸和layout，最大化SRAM utilization和data reuse。与compiler的generic autotuning（标准化tile sizes和heuristics）和手动开发（expert基于经验选择tile尺寸）不同，agent在生成阶段就incorporate production input ranges并通过search自动发现最优tiling。

从kernel调度角度拆解术语：
PFFN kernel的tiling行为分析（Figure 15）:
```
Production config: B=1024, N∈[150,400], D∈[96,256], K∈[96,256]

Speedup vs D (fixed B=1024, K=256):
  D≤100: 1.6-1.9×  → tile comfortably fits SRAM → effective fusion
  D≈200: 1.1-1.2×  → tile near SRAM limit → partial spilling
  D>200: 1.2-1.4×  → adaptive strategy → recovery via tile resizing
```

Non-monotonic behavior源于tile size和SRAM capacity的复杂交互——human expert需要extensive trial-and-error找optimal tiling，agent通过系统化搜索自动发现。

Conv1d kernel的specialization trade-off：production shape (2048,96,96,200)上2.30× speedup，但out-of-distribution shapes (64×768×768×1024)上仅0.49-0.63×——specialization以generality为代价。Deployment用shape-aware dispatch：target shapes用generated kernel，其他用vendor library fallback。

术语一般如何实现？如何使用？
KernelEvolve通过search每步evaluate tile configurations on production shapes（via get_inputs()），fitness feedback automatic ranking tiling策略。Expanded autotuning探索20+ configurations (BLOCK_M/N/K + num_warps + num_stages + pipeline stages)，keyed to input dimensions (key=["N"])在shape变化时re-autotune。Cross-operation tile reuse进一步利用SRAM——同一tile的loaded data完成整个operator chain后才写回HBM。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
