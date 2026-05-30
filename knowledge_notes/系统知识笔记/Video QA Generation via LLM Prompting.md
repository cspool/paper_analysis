## Video QA Generation via LLM Prompting

术语是什么？
Video QA Generation via LLM Prompting 是指利用大型语言模型（LLM），通过精心设计的prompts，从视频内容和结构化metadata（如bounding box trajectories）中自动生成多选问答对的过程。FoundationMotion使用GPT-4o-mini作为QA generator，输入包括：(1) 视频帧（2fps采样）、(2) motion caption（由前一阶段生成）、(3) structured motion data JSON（归一化bbox轨迹含object_type和interactions）。输出为多选QA list，每个包含question、4个options（正确答案在A位，后续随机打乱避免position bias）、answer和category label。5类QA覆盖：Motion Recognition、Action Order、Motion-related Objects、Location-based Motion、Repetition Count。

从系统架构角度拆解术语：
QA generation的系统交互：
```
Input → GPT-4o-mini:
  - Video frames (2fps, up to ~20 images per 10s clip)
  - Motion caption (7-dimension, from caption stage)
  - Prompt template with 5 categories and format spec

Processing:
  1. GPT-4o-mini generates Q&A in free-text format
     "Q1: What action... A1: The person..."
  2. For each Q&A, GPT-4o-mini generates 3 distractors from caption content
  3. Correct answer at position A, then shuffled

Output:
  [{Q, A, B, C, D, answer, category}]
  # ~10 QAs per video, 467K total from 46.7K videos
```

Quality control：distractors must be "distinctive from the correct answer" and "no ambiguity with any other choice"，通过prompt约束实现。Ablation证明video+bbox JSON比video-only提升Overall QA Quality从6.3→8.6（GPT-4评分，0-10）。

术语一般如何实现？如何使用？
通过OpenAI API或其他LLM API实现。核心是prompt engineering：设计category-specific prompt templates，指定输出格式（JSON list of strings），要求distractors来源于caption但不与正确答案ambiguous。配合视频帧和structured metadata（bbox轨迹JSON）共同输入效果最优。使用时注意token limit（帧数×分辨率影响token数）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
