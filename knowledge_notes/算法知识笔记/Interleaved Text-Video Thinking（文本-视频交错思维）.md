## Interleaved Text-Video Thinking（文本-视频交错思维）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Text-Video Thinking 是一种长视频推理范式，将时序搜索重新定义为文本推理与视频片段检索交错的思维过程。与传统 "先选帧后推理" 不同，模型在每个推理步 k 生成文本推理 T_k；若含搜索指令，video environment 检索视频片段 V_k 追加到 CoT，形成 "thinking → searching → thinking → ..." 循环直至输出答案或达到预算（最多 8 轮搜索/8 帧）。形式化：C_k ≜ {(T_1,V_1), ..., (T_k,V_k)}，整个推理链分解为 P_θ(A,C|Ṽ,Q) = P_θ(C|Ṽ,Q) · P_θ(A|C,Ṽ,Q)（时序搜索概率 × 答案预测概率）。该范式是 "Thinking with Images" 向长视频域的扩展（空间搜索→时间搜索），使模型从数据中端到端学习最优搜索策略。

从算法pipeline角度拆解术语，给出具体例子。
推理格式示例：
```
<think>The video shows a living room. I need to find when cooking 
starts. Let me search around 120s-300s.</think>
<tool_call>{"name":"seek_video_frames","arguments":{"query":"person 
cooking in kitchen","start_time":120,"end_time":300,"num_frames":8}}
</tool_call>
[8 frames at: 155.2s, 178.6s, 203.1s, ...]  ← 搜索返回
<think>Frames at 178.6s-224.8s show vegetable chopping. Let me 
check for earlier steps.</think>
<tool_call>...</tool_call>
[more frames]
<think>Preparation starts at 178.6s, cooking at 245.0s.</think>
<answer>B</answer>
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 两阶段训练：SFT (GPT-4o 生成交错 CoT) → RL (GRPO-CSV)；(2) 搜索用 SigLIP-400M 编码帧/query + DPP 选择帧；(3) 帧带绝对时间戳 token ("12.3s") 保持时间定位；(4) 与 "Interleaved Multimodal Reasoning" (Mirage) 区别：TimeSearch-R 处理显式视频帧检索，Mirage 处理隐式 latent token 生成。适用场景：长视频理解、视频问答、视频动作定位。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
