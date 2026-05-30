## Synthetic Video QnA Generation with Long-Context VLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Synthetic Video QnA Generation 是 SAGE 提出的利用长上下文 VLM（Gemini-2.5-Flash）一次性处理完整长视频（up to 2 hours）直接生成覆盖全时间跨度的 QnA pairs 的数据合成 pipeline。与传统 bottom-up pipeline（将视频切割为 10-30s subclip，分别处理后再聚合）不同，SAGE 利用 Gemini-2.5-Flash 的长上下文能力（支持数小时视频）在 single pass 中生成 10-20 个 QnA pairs。关键技术是 prompt 中的 **percent_video_parsed** 字段——要求模型为每个 QnA pair 计算已处理视频百分比，并强制至少一个问题的 percent_video_parsed ≥ 90%，确保问题覆盖视频全部时间范围。该方法成本约为人工标注的 1/100（约 $0.30/video vs $30/video），速度约为 subclip pipeline 的 10×（约 2 min/video vs 20 min/video）。人工验证 1700+ 样本仅 5% 错误率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Synthetic QnA Generation Pipeline
# Stage 1: QnA Pair Generation (single-pass长上下文处理)
def generate_qna_pairs(video_path):
    video_frames = sample(video_path, fps=1)  # 1 FPS全视频
    prompt = build_qna_prompt(video_duration)
    # prompt 要求: (1) 10-20 QnAs; (2) 混合 visual/verbal/both modality;
    #   (3) 混合 open-ended/MCQ; (4) 混合 easy/medium/hard;
    #   (5) 每个QnA含percent_video_parsed字段;
    #   (6) 至少1个QnA的percent ≥ 90%
    
    response = Gemini_2_5_Flash(video_frames + prompt + audio)
    # response 为JSON: [{index, type, difficulty, modality, 
    #   answer, question, options, start_timestamp, end_timestamp,
    #   percent_video_parsed, ...}, ...]
    
    qna_pairs = parse_json(response)  # 10-20 QnA pairs
    return qna_pairs

# Stage 2: Tool Call Trajectory Generation
def generate_tool_trajectories(qna_pairs, video_path):
    # 使用 SAGE 系统（Gemini-2.5-Flash 作为 SAGE-MM）
    # 为每个 QnA 生成 4 条 tool call trajectories
    trajectories = []
    for q in qna_pairs:
        for _ in range(4):
            traj = SAGE_system(video_path, q.question, orchestrator=Gemini)
            trajectories.append(traj)
    # 从 input-action pairs 提取 unique trajectories 构建 SFT 数据
    return trajectories
```

关键数据特征：(1) 99.1k 训练问题来自 6659 个视频；(2) 417.7k state-action pairs 用于 SFT；(3) 13 个 YouTube 频道覆盖体育、美食、喜剧、教育、旅游等 genre；(4) RL 数据 7.68k 样本（half tool-calls, half single-turn）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要求：(a) 支持超长上下文（≥2小时视频）的 VLM（如 Gemini-2.5-Flash）；(b) Carefully designed prompt 强制 percent_video_parsed 字段确保全时覆盖，否则模型倾向于仅覆盖视频开头；(c) 人工验证子集确认质量（5% 错误率已足够低）；(d) QnA pairs 可直接用于 SFT（DIRECT training），而 tool call trajectories 用于 SFT（AGENT training）。该 pipeline 可泛化到其他视频领域（论文仅用于娱乐视频，但在体育、教育视频上同样有效）。局限性：依赖闭源 Gemini-2.5-Flash 的 API 访问；开源替代品（如 Qwen3-VL）目前尚无足够的 long-context video understanding 能力来替代该 role。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning
