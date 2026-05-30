## Iterative Reward-Guided Selection (迭代奖励引导选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Iterative Reward-Guided Selection 是 DIG 提出的无参数帧选择方法，替代需要预设 K 的 Top-K selection。核心思想：给定 r-frames 的奖励集 {R_j}，迭代式 mean-thresholding 自动确定"显著高于平均"的相关帧：(1) 计算当前奖励均值 R̄；(2) 低于均值的置零：R'_j = max(R_j - R̄, 0)；(3) 检查正值集合是否与前一轮一致，一致则终止。优势：(1) 无预设参数——不需指定 K；(2) 自适应——不同查询的 reward 分布不同时自然产生不同选择数量；(3) 单调收敛保证——每次迭代至少移除低于均值的元素。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Iterative Reward-Guided Selection
R = {R_1, ..., R_{N-1}}; prev = None
while True:
    R_mean = mean(R)
    R_new = [max(r - R_mean, 0) for r in R]
    positives = {j | R_new[j] > 0}
    if positives == prev: break
    prev = positives; R = R_new
return S = positives  # 最终选中的 r-frame indices
```
数值示例：rewards [85,60,45,30,20,10,5] → Round 1 mean=36.4 → positives [0,1,2] → Round 2 mean=26.9 → positives [0] → Round 3 all zero → 收敛，最终选 r-frame[0]（reward=85 的帧）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 DIG 中，Iterative Selection 作为 Video Refinement 的第一步执行，之后对被选 r-frames 进行窗口合并构建 refined video。通常 2-5 轮迭代收敛。对比 Top-K：Top-K 需预设 K（5? 10? 20?），optimal K 随视频/查询/分布变化，Iterative 方法自动适应无需调参。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
