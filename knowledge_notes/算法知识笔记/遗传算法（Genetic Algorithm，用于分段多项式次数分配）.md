## 遗传算法（Genetic Algorithm，用于分段多项式次数分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 遗传算法（GA）是受自然选择启发的元启发式优化：种群（一组候选解/个体）经选择（按适应度）、交叉、变异迭代演化，收敛到近似最优。适合大规模/组合/不可微的搜索空间；随机性使其不一定最优，但迭代充分时逼近最优。在 LoRA 中用于分段逼近的次数分配：个体=各段多项式次数分配 (k_seg1,...,k_#seg)，breakpoints 由三种分段策略确定，适应度=整函数平均 MSE，选平均 MSE 最小的个体。
- 为什么需要：高次多项式在每段都可能导致过拟合，且各段最优次数不同；穷举 6 段×6 项=6^6=46656 种次数分配一周内不可行（论文 VIII-A），GA 在迭代充足时更合适。经验设置：#gen=10、#pop=16、equal-error 容差 ξ=1.5e−5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（LoRA 算法级）：
  ```
  初始化种群 P = 随机次数分配 {k_seg1..k_segN}×#pop
  for gen in 1..#gen:
    对每个个体: breakpoints = 分段策略(seg 数); 逐段最小二乘求系数; 计算平均 MSE
    适应度 = 1/平均MSE; 按适应度选择父代; 交叉/变异生成新种群
  返回平均 MSE 最小的个体
  ```
- 算法级评估结论（Fig.7）：ξ 越小误差分布越均匀、越接近近优但 runtime 越大；#gen 越多越好但收益递减；GA 优于穷举（时间可行性）。
- 其它用法（vault 中广泛出现）：ScaleMoE 用 GA 做专家重映射（coverage×bandwidth 矩阵下最小化通信）、Cocco 用 GA 做硬件映射-内存配置协同探索（比贪心/DP 更稳）、AutoFHE 用多目标 GA 做 FHE 加速器设计空间探索（NSGA-II）、TileFlow 用 GA 做 tiling 调度、GAMMA/Magma 用 GA 自动映射 DNN 到加速器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 PiecewiseChebFitter（Python）内置 GA；标准实现需定义编码（个体表示）、适应度函数、选择/交叉/变异算子与终止条件；可用库如 DEAP、PyGAD。超参（#gen/#pop/交叉率/变异率）需按问题调。
- 使用场景：与分段策略/最小二乘联合构成"算法-硬件协同"的离线配置生成（LoRA）；以及分布式训练专家放置、硬件映射、调度等组合优化问题。局限：随机性导致结果不稳定（Cocco 论文指出 GA 比 DP 不稳定，需多次运行取优）；论文明确穷举不可行时才用启发式。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
