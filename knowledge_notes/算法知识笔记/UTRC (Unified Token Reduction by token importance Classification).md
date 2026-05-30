## UTRC (Unified Token Reduction by token importance Classification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UTRC是Rethinking Token Reduction for SSMs论文提出的面向SSM（Mamba系列）的统一后训练token reduction方法。核心流程为6步：(1) 从Mamba block的SSM隐藏状态y计算token重要性 `S = Σ_d max(0, y_{:,d}) / D'`（使用ReLU clip保留正向激活通道）。(2) 按重要性将N个token两等分为集合M_A（低重要性N/2个）和M_B（高重要性N/2个）。(3) 为M_A中每个token a_i计算其到M_B中最相似token的连接：`f_i = argmax_{b_j∈M_B} cosine_sim(a_i, b_j)`，得到最大相似度g_i。(4) 按g_i降序排序所有连接，保留最相似的top-p%连接对。(5) 对保留连接执行UTR：q比例的连接执行pruning（删除M_A中的token），(1-q)比例的连接执行merging（`f_i = (a_i + f_i)/2`），q=0.5时效果最优。(6) 重新组装M_B和缩减后的M_A。设计空间：hidden states上使用hybrid（q=0.5），residual connections上仅使用merging以保护残差信息完整性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# UTRC per-layer pipeline:
y = SSM(A, B, C)(x)                       # hidden states ∈ R^{B×N×D'}
S_i = sum(max(0, y[i,:,:])) / D'          # importance ∈ R^{B×N×1}
sorted_idx = argsort(S, descending=True)
M_B = sorted_idx[:, :N//2]                 # 高重要性
M_A = sorted_idx[:, N//2:]                 # 低重要性
for a_i in M_A:
    sims = [cosine_sim(a_i, b_j) for b_j in M_B]
    f_i = M_B[argmax(sims)]
    g_i = max(sims)
num_keep = int(p * N/2)
keep = sort_by_g({(a_i, f_i, g_i)})[:num_keep]
mid = int(0.5 * num_keep)
for (a_i, f_i) in keep[:mid]:             # PRUNE
    M_A.remove(a_i)
for (a_i, f_i) in keep[mid:]:             # MERGE
    T[f_i] = (T[a_i] + T[f_i]) / 2
    M_A.remove(a_i)
output = reassemble(M_B, M_A_reduced)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/wuyushuwys/ToR_SSM。基于PyTorch + HuggingFace Transformers，作为hook注入Mamba block的SSM输出处（在Linear投影和残差加法之前），不修改模型权重。层次化应用：从第10~12层开始，每5层执行一次（如Mamba-2-2.7B在layers [12,17,22,27,32,37,42]），使用固定压缩率。p值由目标FLOPS reduction反推。评估适配：token数减少后PPL/Accuracy在调整后的logits上计算（取前(1-m%)个token对应标签）。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
