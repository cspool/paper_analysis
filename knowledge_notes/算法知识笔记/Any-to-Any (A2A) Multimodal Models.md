## Any-to-Any (A2A) Multimodal Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Any-to-Any（A2A）多模态模型是一类新兴的多模态模型，能够接受文本和多种模态数据（图像、视频、音频）的任意组合作为输入，并生成任意组合的模态输出。截至 2026 年 3 月，Hugging Face 上有超过 11,000 个 A2A 模型变体。代表性模型包括：Qwen Omni 系列（接受 T/I/V/A 输入，生成 T/A 输出）、InternVL 3（T/I/V→T）、DeepSeek Janus（T/I→T/I）、LTX-2（T/I→V/A）、Qwen Image（T→I）。传统 text-only LLM 或仅生成图像/视频的 Diffusion 模型是 A2A 的特例——所有请求沿同一线性 pipeline 遍历所有 component。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

A2A 模型的计算图结构（以 Qwen 3 Omni 为例）：

```
Component Graph (DAG):
  E_img (Vision Encoder) ──┐
  E_vid (Video Encoder)  ──┼──► L_th (Thinker LLM) ──► L_ta (Talker LLM) ──► G_aud (Vocoder)
  E_aud (Audio Encoder)  ──┘         │                        │                    │
                                      ▼                        ▼                    ▼
                                   text output             audio tokens          audio waveform

Request Types (不同输入/输出组合遍历不同子图):
  ① T+I → T:     E_img → L_th → text output
  ② T+I+V → T:   E_img → E_vid → L_th → text output
  ③ T+I → A:     E_img → L_th → L_ta → G_aud → audio output
  ④ T+I+V+A → A: E_img → E_vid → E_aud → L_th → L_ta → G_aud → audio output

关键特性:
  - 不同 request type 遍历不同子图 → 各 component 面临不同 request rate
  - 不同 component 的计算特性差异极大:
    Qwen 3 Omni on A100: E_aud 21.43 req/s vs G_aud 0.12 req/s (178× 差异)
    Thinker LLM 2.15 req/s vs Talker LLM 0.12 req/s (18× 差异)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

A2A 模型的实现通常基于：(1) 预训练的 modality-specific encoder（Vision Transformer for image/video, Whisper-style encoder for audio），将多模态输入编码为 unified embedding；(2) 核心 LLM（如 Qwen 系列）进行跨模态理解和推理；(3) modality-specific generator（如 Diffusion Transformer for image, autoregressive + vocoder for audio）。Serving 时，不同 executor type 处理不同的 component 集合：encoder executor（处理多模态输入→embedding）、LLM executor（autoregressive 生成）、DiT executor（扩散去噪）、vocoder executor（token→waveform）。vLLM-Omni 和 SGLang-Omni 提供通用的 component-wise disaggregation 机制，Cornfigurator 在此基础上增加自动规划。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
