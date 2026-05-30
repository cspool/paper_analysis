## MaxFPS (Maximum Sustainable Frame Rate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MaxFPS（最大可持续帧率）是流式视频理解评估中量化视觉编码吞吐上限的核心指标。定义为：在流式推理 pipeline 中，视觉编码和记忆更新模块能够持续处理的最高输入帧率（fps），超过此阈值将导致帧积压（frame backlog）和延迟不稳定。MaxFPS 直接反映模型的 visual encoding + memory update pipeline 的实时吞吐能力。不同于离散的单次推理测量，MaxFPS 测量的是系统在持续、不中断的流式输入下能够稳定运行的最大帧率。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 StreamingEval 的三进程异步 pipeline（Frame Player → Encoder-Memory Updater → Responder）中测量。对每帧 $v_i$，记录视觉编码 $z_i = g_{\theta}(v_i)$ + projector + memory update 的总 wall-clock 时间，取倒数得瞬时 fps，取长期稳定运行时的微分值为可持续 fps。StreamingEval 实验中大多数模型 MaxFPS > 1（满足 1fps 最低实时要求），但 VideoChatOnline-4B 的 MaxFPS 仅 0.14，因为其复杂的 cross-frame memory maintenance 机制严重降低了编码吞吐。在真实部署中 MaxFPS < 输入帧率意味着系统无法跟上视频流，将导致持续积压和累积延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
测量需在与实际部署一致的条件下进行：统一帧预处理 pipeline、同一 GPU（如 StreamingEval 使用 RTX 4090 48GB BF16）、同一输入帧率（1fps default）。MaxFPS 是判断流式模型是否满足实时性的首要指标——如果 MaxFPS < 目标帧率，无论 accuracy 多高模型都无法部署。StreamingEval 通过设置统一帧率观察各模型的 MaxFPS 瓶颈，发现编码效率瓶颈主要来自复杂的 memory-update 和 cross-frame maintenance 机制。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding
