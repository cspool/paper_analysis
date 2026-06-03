# Infera Shepherd Operator 与 TEU 协同编译调度

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/Infera_Shepherd_Operator_TEU_协同编译调度.md
- 范围：仅用户输入与 Claude 最终输出
- 涉及论文：Automated End-to-End Model Serving with Cooperative Compilation and Scheduling (EuroSys 2026)

## 对话记录

### 001 User
（选中 L1_horizon_summary.md 第118行：「**算子融合** | Shepherd Operator | Micro-operator→virtual operator 合并; per-shepherd-operator 调度降级 | 消除 micro-operator scheduling overhead | Q1.4 |」）

教我

### 002 Claude
（详细解释 Shepherd Operator 的概念、设计动机、在 Infera 编译流水线中的位置、与其他算子融合/并发方案的对比关系，以及局限性）

### 003 User
（选中 L1_horizon_summary.md 第120行：「**Memory Planning** | Welder Tile-Graph | Tile propagation 反向推导 + SetConnect memory level 选择 (L0/L1/L2) + traffic cost model | Inter-layer independence 解耦优化 | Q1.6 |」）

'TEU 调度流水线（per micro-operator）: SelectKernels → FuseKernels → LaunchKernel → Execute' 这是什么？根据运行时资源的动态编译吗？

### 004 Claude
（详细解释 TEU (Task Execution Unit) 三阶段流水线：SelectKernels 的两阶段选择机制（data block 选择 + kernel 版本选择）、FuseKernels 的 SASS binary 级别 warp 水平融合、LaunchKernel 的 HKQ→GDRCopy→DKQ→CDP daemon kernel 多级发射流水线。核心回答了「不是纯动态编译，而是编译期静态准备多版本 kernel + 运行时动态选择 + binary 级拼装」的协同编译与调度范式，并给出离线/在线两阶段的完整架构图。）
