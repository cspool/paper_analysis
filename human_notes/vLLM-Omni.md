# vLLM-Omni Stage Graph 多模态 Serving 调度

- 导出时间：2026-06-04
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/vLLM-Omni.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
用户选中 `draft/review_draft.md` 第 31 行：
```
| 多模态调度 | vLLM-Omni Stage Graph? | 复杂Any-to-Any多模态模型分解为独立stage（Thinker LLM→Talker LLM→DiT Vocoder），每stage由独立execution engine服务；unified connector (NCD/shared memory/Mooncake RDMA)传输中间数据；Qwen2.5-Omni RTF降低61.4%，Qwen3-Omni RTF降低90.7% | Stage Disaggregation + Unified Connector传输 | Q2.2, vault: paper_secs/vLLM-Omni (2910.6) |
```
并说：教我

### 002 Claude
（详细讲解了 vLLM-Omni 的 Stage Graph 抽象、Stage Disaggregation 机制、Unified Connector 数据传输、以及 Qwen-Omni 系列模型的 RTF 大幅降低原因。内容涵盖：Any-to-Any 模型的架构复杂性挑战、Stage Graph 的 Node/Edge 编程模型、每个 Stage 独立 Execution Engine 的资源分配策略、Shared Memory 与 Mooncake RDMA 两种传输模式的性能对比、Streaming Stage Output 流水线重叠、以及与 ModServe/EPD-Serve/EEVEE 等同类方法的对比分析。全文使用中文，包含 Mermaid 流程图、表格对比和关键要点总结。）
