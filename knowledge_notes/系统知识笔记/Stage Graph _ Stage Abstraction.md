## Stage Graph / Stage Abstraction

术语是什么？

Stage Graph 是 vLLM-Omni 提出的 any-to-any 多模态模型的编程抽象。传统 LLM serving 框架（vLLM、SGLang）使用 **step-centric abstraction**——将模型推理抽象为单个 `forward()` 函数，框架内部封装 iteration loop 和 KV cache management。这种抽象只能表达单 AR decoding stage，无法表达多 stage pipeline（如 Qwen3-Omni 的 Thinker→Talker→Vocoder）。Stage Graph 将 any-to-any 模型定义为有向图 G=(V,E)：

- **Node (Stage)**：每个 node 是独立的模型 stage（AR LLM、DiT、CNN 等），需实现两个函数：
  - `forward()`：step-centric batched forward，兼容 vLLM 的 iteration loop 优化
  - `preprocess()`：每 iteration 调用，修改 stage input 以 incorporate upstream data
- **Edge (Stage-Transfer Function)**：控制 request states 和 intermediate data 如何在 stage 间变换和路由，仅在 stage transition 时调用一次

从系统架构角度拆解术语：

以 Qwen2.5-Omni 的 Stage Graph 定义（3-node, 2-edge DAG）为例：
```
Stage Graph G:
  Node Thinker (AR LLM):
    forward: thinker.forward() — 标准 step-centric AR decode
    preprocess: mm_encode() — 将 audio/image/video embeddings concat 到 text embeddings
  Node Talker (AR LLM):
    forward: talker.forward() — AR decode for audio codec tokens
    preprocess: process_input() — 每 iteration concat Thinker hidden states + Talker embeddings
  Node Vocoder (DiT):
    forward: dit_decode() — DiT denoising for waveform synthesis
    preprocess: 无特殊逻辑

  Edge Thinker → Talker:
    Thinker2Talker(): 提取 Thinker hidden states → 转为 Talker input embeddings
  Edge Talker → Vocoder:
    Talker2Vocoder(): 提取 Talker codec tokens → 转为 Vocoder input

执行流程:
  Request 到达 Orchestrator
  → Orchestrator 沿 Stage Graph 路由:
    Thinker: preprocess → AR decode generate text + hidden states
    → Thinker2Talker transform → Unified Connector → Talker device
    Talker: preprocess (concat Thinker hidden states each step) → AR decode codec tokens
    → Talker2Vocoder transform → Unified Connector → Vocoder device
    Vocoder: DiT denoising → audio waveforms
  → Return final text + audio response
```

关键洞察：Stage Graph 将复杂 multi-stage pipeline 从"application-level manual orchestration"下沉为"framework-level programmable abstraction"。开发者仅需实现 per-node forward/preprocess 和 per-edge transfer 函数，vLLM-Omni 的 backend 自动 handling batching、scheduling、data transfer 和 resource allocation。

术语一般如何实现？如何使用？

在 vLLM-Omni 中，Stage Graph 通过 Python API 编程。开发者：
1. 继承 `OmniStage` 基类实现每个 stage 的 `forward()` 和可选的 `preprocess()`
2. 定义 transfer function（普通 Python 函数）
3. 构建 `StageGraph` 对象，添加 nodes 和 edges
4. 将 `StageGraph` 提交给 vLLM-Omni backend——Orchestrator 自动初始化 engines、加载 models、建立 connectors 并开始 serving

vLLM-Omni 使用 Pydantic 数据模型定义 input/output schemas，使 stages 可以是 type-safe connectors。Unified Connector 抽象（ZeroMQ/shared memory/Mooncake）被封装为 backend 细节，对开发者透明。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models
