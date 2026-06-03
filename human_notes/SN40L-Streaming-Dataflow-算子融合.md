# SN40L Streaming Dataflow 算子融合

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/SN40L-Streaming-Dataflow-算子融合.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（用户选中了 `learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md` 第 116 行：）

> **算子融合** | SN40L Streaming Dataflow | 硬件原生 streaming: PCU SA/SIMD + PMU composable mem；Gated FFN 全融合为单 spatial pipeline | 中间结果永不物化到 off-chip | Q1.6 |

教我

### 002 Claude
（对算子融合从 GPU Kernel Fusion 到 SN40L 硬件原生 Streaming Dataflow 的完整讲解，涵盖：1. 算子融合的定义与动机；2. GPU 上的 Vertical Fusion 及其限制；3. SN40L 硬件原生 Streaming Dataflow 架构详解（PCU SA/SIMD + PMU + RDN）；4. 空间融合 vs 时间融合的范式差异；5. Gated FFN 全融合为单 Spatial Pipeline 的具体例子与伪代码；6. "中间结果永不物化到 off-chip" 的量化分析；7. 编译器 Place-and-Route 工作流程；8. GPU vs SN40L 全维度对比总结表。）
