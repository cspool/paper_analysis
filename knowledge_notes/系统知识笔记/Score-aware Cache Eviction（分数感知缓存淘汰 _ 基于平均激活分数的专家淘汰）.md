## Score-aware Cache Eviction（分数感知缓存淘汰 / 基于平均激活分数的专家淘汰）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Score-aware Cache Eviction 是 SMoE 的专家缓存淘汰策略，用于回答"GPU 缓存该保留哪些专家"：相比传统 LRU（按最近使用时间）或 LFU（按使用频率），它按专家最近 n 次解码迭代的平均 activation score（router gate score）淘汰最低分专家——对 expert i 在第 j 次 token 生成时的分数记为 S_{i,j}，淘汰 argmin_i (Σ_{k=max(1,j-n)}^{j} S_{i,k})/(窗口长度)。动机（论文 Fig.7）：当前迭代分数高的专家在接下来 3 次迭代中作为 top-k 或替换候选被复用的概率显著更高，分数高反映其历史重要性；按分数保留可同时保住两类专家——将作为 top-k 被激活的专家，以及分数接近 low-score 专家、可作为替换候选（E_s）的 inactive 专家。相比 LRU 平均降低 TPOT 8%、提升 GPU cache ratio 11%（+CE 消融），且单独优于 LRU/LFU 等传统缓存（Fig.19）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 每 token 解码后维护每个专家的分数历史
for expert i in all_experts:
    avg_score_i = mean(S_{i, j-n..j})          # 近 n 次迭代窗口平均
# 需要为新加载专家腾空间时：
evicted = argmin_i avg_score_i                  # 淘汰平均分最低者（Eq.3）
# 保护机制（protection shield）：
# 被 expert-cache router 选中用于当前层计算的专家临时提升淘汰优先级
# （标记为不可淘汰），该层计算完成后自动解除，恢复按分数淘汰
```
运行流程：cache eviction 与 protection shield 更新都安排在"Gating 与 attention 之后"触发（数据依赖），与 GPU 专家计算、PCIe 加载重叠执行，不产生额外关键路径延迟；淘汰候选池包含所有访问过的专家（含 inactive 可替换候选），确保 GPU 保留"既相关又可能用于未来计算"的专家。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SMoE 运行时（https://github.com/goingshr/SMoE）中由 config 的 window_size 字段控制观察窗口（null 表示退化为 LRU），if_replace 开关替换功能。它支撑 expert-cache router 的替换候选池 E_s（扩大 GPU 驻留的近似分数专家），是专家替换取得高命中率的前提之一。与经典策略对比：LRU 只考虑最近性、LFU 只考虑频率，均未利用"gate score 反映的输出重要性"这一 MoE 特有信号；论文实验显示 score-aware 策略在保留高分专家方面显著优于两者。通用可迁移点：任何"按重要性而非按时间淘汰"的缓存策略，重要性信号来自路由分数。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
