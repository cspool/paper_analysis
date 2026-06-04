# NetMoE ILP Device Placement 详解

- 导出时间：2025-06-04
- 来源：当前 Claude Code session 对话上下文
- 保存路径：human_notes/NetMoE_ILP_Device_Placement_详解.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（从 review_draft.md 第 111 行选中）：
| **图优化** | NetMoE ILP Device Placement* | 将token-to-expert映射建模为ILP问题：变量SmpDev(i)∈[I]*_n (token i分配到节点n的expert集合中)，目标minimize Σ_i[t_intra+t_inter]，约束每个token恰好top-k节点、节点上expert已预放置、显存不溢出。Per-layer分解+贪心预热→1000 GPU规模下<1s编译。**运行时placement_plan查询O(1)** | **ILP求解在编译时offline完成**；**运行时各GPU独立CUDA stream并发发射expert FFN** | Q3.3, paper_secs NetMoE (889.4) |

教我

### 002 Claude
（对 NetMoE ILP Device Placement 的完整讲解，涵盖问题背景、ILP 建模、两阶段拆解+KM 算法求解、系统实现流程、性能对比与在 MoE Serving/Training 图优化中的定位）
