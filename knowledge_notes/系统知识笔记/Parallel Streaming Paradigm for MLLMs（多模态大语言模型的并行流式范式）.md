## Parallel Streaming Paradigm for MLLMs（多模态大语言模型的并行流式范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parallel Streaming Paradigm 是 Speak While Watching 论文提出的 MLLM 实时视频理解范式，核心创新是打破传统位置编码的全局连续性约束，使**视觉感知（prefill）和文本生成（decode）可以同时执行**，而非交替串行。传统 Interleaved Streaming 中，每个新帧的视觉 token 位置必须紧跟上一轮文本生成的位置，导致 prefill 必须等待 decode 完成→延迟累加。Parallel Streaming 通过重新设计位置 ID 分配规则（GDPE/OSPE/GIPE），使视觉和文本拥有独立位置空间，消除 prefill-decode 串行依赖。理论加速比最高 2×（当感知和生成负载均衡时）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
┌─────────────────────────────────────────────────────────┐
│        Parallel Streaming System Pipeline               │
├─────────────────────────────────────────────────────────┤
│  Video Stream (frames arrive at 2fps)                   │
│       │                                                  │
│       ├──► Vision Encoder (ViT) ──► MLP Projector       │
│       │         │                       │                │
│       │    visual tokens (m_i per frame)                 │
│       │         │                       │                │
│       │         ▼                       ▼                │
│       │    ┌─────────────────────────────────┐          │
│       │    │  Parallel Position Assignment   │          │
│       │    │  ┌───────────┐ ┌─────────────┐  │          │
│       │    │  │ Visual Grp│ │  Text Group │  │          │
│       │    │  │ pos_v=0.. │ │ pos_a=0..   │  │          │
│       │    │  │ (indep)   │ │ (indep)     │  │          │
│       │    │  └─────┬─────┘ └──────┬──────┘  │          │
│       │    └────────┼──────────────┼─────────┘          │
│       │             │              │                     │
│       │             ▼              ▼                     │
│       │    ┌────────────┐  ┌──────────────┐             │
│       │    │ Prefill    │  │  Decode      │             │
│       │    │ (GPU 0)    │  │  (GPU 1)     │             │
│       │    │ process V_i│  │  generate A_j│             │
│       │    └─────┬──────┘  └──────┬───────┘             │
│       │          │                │                      │
│       │          └────────┬───────┘                     │
│       │                   │                              │
│       │              KV Cache                            │
│       │          (shared, unified)                       │
│       │                   │                              │
│       ▼                   ▼                              │
│  Streaming Output: A_1, A_2, ..., A_N                   │
│  (text tokens generated in real-time)                    │
│                                                          │
│  Latency per step:                                       │
│    Interleave: T = m_i/R_v + k_i/R_t (serial)           │
│    Parallel:   T = max(m_i/R_v, k_i/R_t) (overlapped)   │
└─────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现层面，Parallel Streaming 分为两个层次：(1) **算法层**——修改位置编码和 causal mask，使模型接受独立的位置组输入。代码仓库（https://github.com/EIT-NLP/Speak-While-Watching）实现了此层。(2) **系统层**——在双 GPU 或双 CUDA Stream 上同时运行 prefill 和 decode kernel。论文指出了这一可能性但当前开源代码未实现物理并行。实际部署需：配置两个 GPU/计算流，一个专用于 vision encoder + prefill，另一个专用于 text decode；通过共享 KV cache pool 同步两流状态。应用场景：实时导航辅助、手语翻译、直播视频解说等需要"边看边说"的流式视频理解任务。

涉及论文标题：
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
