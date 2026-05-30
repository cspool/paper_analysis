## Activation-aware Grid Alignment (AGA, 激活感知网格对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation-aware Grid Alignment（AGA，激活感知网格对齐）是 PT²-LLM 在 ITF 之后应用的第二阶段优化。ITF 仅最小化权重层面误差 E_w = ||W - Ŵ||²，但 LLM 实际输出取决于 ŴX 而非 Ŵ 本身。AGA 将优化目标切换为 E_x = ||WX - ŴX||²，利用校准数据 X 的激活统计量（协方差矩阵 C = Σ_b Σ_l X_bl X_bl^T）以闭式解更新三值网格参数 (α, μ)。关键设计：AGA 仅更新连续参数 (α, μ)，冻结离散 T 不更新——论文实验表明在 AGA 阶段更新 T 会导致严重过拟合（模型在少量校准样本上 E_x 下降但泛化差）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AGA 伪代码 (ITF 收敛后执行)
# 输入: W (FP16), T (冻结), X (校准数据 B×L×m)
C = Σ_b Σ_l X[b,l,:] X[b,l,:]^T             # 激活协方差 (m,m)
d = 1^T C 1                                  # 标量
v = T C 1                                     # (n,)
α* = (d*(W∘T)S1 - v∘(WS1)) / (d*T²S1 - v²)   # 闭式解 (Eq. 13)
μ* = (T²S1∘(WS1) - v∘((W∘T)S1)) / (d*T²S1 - v²)
Ŵ = α* T + μ*                                 # T 保持 ITF 输出不变
```
效果：LLaMA-2-7B 上 Avg Acc 从 38.12%（ITF only）提升至 43.33%（ITF+AGA），输出误差 E_x 在 AGA 后急剧下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AGA 实现要点：(1) 校准数据——WikiText2 128个 2048-token 片段；(2) C 矩阵——若 m 较大可通过增量累积 ΣXX^T 避免存储全部激活；(3) 与 AWQ 的区别——AWQ 通过 per-channel scaling + MSE 网格搜索，AGA 通过协方差矩阵闭式解直接求解；(4) T 冻结是防止过拟合的关键设计——三值空间仅有 3^m 种可能赋值，少量校准样本下搜索 T 会严重过拟合。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models
