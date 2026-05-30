## Temporal Embedding Grouping (TEG)

术语是什么？
Temporal Embedding Grouping (TEG) 是 OmniVinci 提出的时序嵌入分组机制，用于在 omni-modal 嵌入序列中编码视觉和音频信号的**相对时序关系**。核心思想：按固定时间窗口 $T_G$ 将时间轴划分为多个 chunk，根据每个视觉帧和音频采样点的时间戳，将其对应的嵌入分配到相应的时序组中，然后按时间顺序交叠排列各组，形成 $[G_v^1, G_a^1, G_v^2, G_a^2, ...]$ 的 omni-modal 嵌入序列。

具体例子：假设 $T_G$ 为某时长，4 个视觉帧时间戳为 $t_v^1 < t_v^2 < T_G < t_v^3 < t_v^4 < 2T_G$，4 个音频采样时间戳为 $t_a^1 < t_a^2 < T_G < t_a^3 < t_a^4 < 2T_G$。TEG 将嵌入分组为 $G_v^1 = \{\mathbf{e}_v^{t_v^1}, \mathbf{e}_v^{t_v^2}\}, G_v^2 = \{\mathbf{e}_v^{t_v^3}, \mathbf{e}_v^{t_v^4}\}, G_a^1 = \{\mathbf{e}_a^{t_a^1}, \mathbf{e}_a^{t_a^2}\}, G_a^2 = \{\mathbf{e}_a^{t_a^3}, \mathbf{e}_a^{t_a^4}\}$，最终序列为 $[\mathbf{e}_v^{t_v^1}, \mathbf{e}_v^{t_v^2}, \mathbf{e}_a^{t_a^1}, \mathbf{e}_a^{t_a^2}, \mathbf{e}_v^{t_v^3}, \mathbf{e}_v^{t_v^4}, \mathbf{e}_a^{t_a^3}, \mathbf{e}_a^{t_a^4}]$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
TEG 在视觉/音频嵌入完成 projection 之后、送入 LLM 之前执行，是对 token 序列顺序的确定性重排操作，不涉及可学习参数。计算流程：
```
输入: visual_embeds {e_v_i, t_v_i}, audio_embeds {e_a_j, t_a_j}, group_duration T_G

groups = {}
for each embed in visual_embeds + audio_embeds:
    g_idx = floor(t / T_G)  # 根据时间戳确定所属组
    groups[g_idx].append(embed)

# 按组索引升序排列
sorted_seq = []
for g_idx in sorted(groups.keys()):
    sorted_seq.extend(groups[g_idx])  # 组内视觉嵌入在前，音频在后

输出: omni_modal_sequence = sorted_seq
```
TEG 的核心假设是：LLM 的 position embedding 将序列位置隐式编码为相对时序信息，同一时间窗口内的视觉和音频嵌入在序列中相邻，LLM 的 self-attention 能更好地捕获跨模态时间对应关系。

术语一般如何实现？如何使用？
TEG 是确定性算法，无需训练，在数据预处理/批处理阶段执行。实现要点：(1) $T_G$ 的选择需平衡时序粒度——太小导致组过多、序列碎片化，太大导致时序区分度不足，OmniVinci 论文中 $T_G$ 为超参数通过消融确定；(2) 组内排列顺序（视觉→音频或交替）影响 LLM attention pattern，OmniVinci 采用视觉嵌入在前、音频嵌入在后的固定顺序；(3) 与 CRTE 互补——TEG 提供相对顺序，CRTE 提供绝对时间戳。消融实验中 TEG 使平均得分从 45.51 (Token Concatenation Baseline) 提升至 47.72 (+2.21)，Dailyomni 增益最显著 (+6.44)。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
