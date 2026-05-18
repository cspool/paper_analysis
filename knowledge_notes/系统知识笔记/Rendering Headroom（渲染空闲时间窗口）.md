## Rendering Headroom（渲染空闲时间窗口）

术语是什么？通过联网搜索让回答具体和精准。

Rendering Headroom 在 LEGO 中指游戏渲染任务执行期间 GPU 未被使用的计算时间窗口。在 60 FPS 游戏场景中，每帧有 16.6ms deadline，但实际渲染任务并不总是占满整段时间，未被使用的 GPU cycle 即为 rendering headroom，可供 LLM 推理利用。LEGO 将 rendering headroom 分为两类：(1) Inter-rendering headroom：连续两帧渲染任务之间的空闲时段；(2) Intra-rendering headroom：单帧内部因游戏引擎的 batch rendering 优化产生的空闲子时段（rendering subtasks 之间，或 rendering subtask 与 auxiliary subtask 之间）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

两类 headroom 的特性与利用方式（基于论文在 RTX 4090 上的 Nsight Systems profiling）：

**Inter-rendering Headroom（帧间 headroom）：**
- 产生：一帧的所有 rendering subtasks 完成后到下一帧开始前
- 时长：较大，可容纳多个 Transformer layer 的推理
- 利用方式：coarse-grained LLM subtask scheduling（多个 transformer layers 一次提交）
- 传统方法（PilotFish）仅利用此类 headroom

**Intra-rendering Headroom（帧内 headroom）：**
- 产生：单帧内部的 rendering subtask 之间，或 rendering subtask 与 auxiliary subtask 之间
- 时长分布：平均 0.24ms/gap，90% < 0.73ms，单帧总计平均 1.39ms，最大 3.1ms
- 利用方式：fine-grained LLM subtask scheduling（decode: 单个 Transformer layer ~0.4ms; prefill: self-attention ~0.5ms 或 FFN sublayer ~1.0ms）
- 发现来源：game engine 通过 batching similar objects 优化渲染 → 一个 rendering task 包含多个 rendering subtasks（使用 GPU）和 auxiliary subtasks（不使用 GPU）
- 传统方法未利用此类 headroom

**调度机制：**
```
while LLM inference not complete:
    if rendering_subtask just completed:
        if next_rendering_subtask not started:
            dispatch fine-grained LLM subtask (≤ T_minimal)
        else:
            wait()
    elif entire rendering frame completed:
        calculate inter-rendering headroom size
        dispatch coarse-grained LLM subtask (multiple layers)
```

**Safety constraint：**
- T_subtasks ≤ T_minimal（T_minimal = 游戏所有 rendering tasks 中最小的 inter-rendering headroom）
- 保证利用 intra-rendering headroom 不会阻塞下一个 rendering subtask → 不会导致 FPS drop

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **Profiling 工具**：使用 NVIDIA Nsight Systems 进行 GPU trace 分析，测量每个 rendering subtask 的起止时间和 GPU idle 分布
- **Headroom 测量**：收集 30 分钟的游戏数据，计算长期 GPU time slice reservation（BlackMyth 60.8%、FFXVI 54.8%、RDR2 47.6%）
- **Headroom usage 指标**：LEGO 相比 SmallModel 在 100/200/300 APM 下分别提升 rendering headroom usage 25.2%/28.6%/18.8%
- **Adaptive rendering 关系**：论文在最高画质设置 + DRS (Dynamic Resolution Scaling) 开启下测试，RTX 4090 上未触发 adaptive workload reduction → headroom 基于固定 workload
- **通用性**：不同游戏在不同 GPU 上 headroom 分布不同，需对每个 Game-GPU 组合重新 profiling

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU
