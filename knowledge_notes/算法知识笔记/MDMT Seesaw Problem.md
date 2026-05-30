## MDMT Seesaw Problem

术语是什么？
MDMT Seesaw Problem（多域多任务跷跷板问题）是 M3oE 论文提出的新概念，描述了在多域多任务推荐中同时出现的两类跷跷板效应的叠加。具体表现为：(1) 同一多域信息传递方法不能泛化到不同任务——例如用域间迁移提升点击率的方法可能损害点赞率；(2) 同一多任务优化平衡策略不能泛化到不同域——例如在域 A 有效的任务权重分配在域 B 可能失效。这一问题综合了已知的 domain seesaw（域间跷跷板，提升一个域的性能可能损害其他域）和 task seesaw（任务间跷跷板，提升一个任务的性能可能损害其他任务，如 PLE 论文所述）。M3oE 是首个明确定义并系统解决此问题的工作。

从算法pipeline角度拆解术语：
以视频平台的例子说明 MDMT seesaw 的具体表现：
- Domain seesaw 场景：用户在 TV 上观看 Sci-Fi 的偏好如何迁移到 Tablet 域
- Task seesaw 场景：用户"观看"行为与"点赞"行为之间的关系建模
- MDMT seesaw 场景：用户在 TV 上"观看"Sci-Fi 的偏好如何迁移并增强 Tablet 域"点赞"的预测——这涉及跨域×跨任务的双重信息传递

M3oE 解决此问题的策略是解耦：shared experts 学习跨域跨任务的共同模式，domain experts 维护域特定信息，task experts 维护任务特定信息，通过 AutoML 自适应两级融合权重为每个 (d,t) 对找到最优的信息组合方式。

术语一般如何实现？如何使用？
解决 MDMT seesaw 的核心思路是解耦（disentanglement）+ 自适应融合（adaptive fusion）。解耦确保不同类型的信息不会互相干扰，自适应融合确保每个 domain-task pair 能按需获取合适比例的信息。其他相关工作如 PEPNet 通过个性化先验信息注入来缓解此问题，MTKDN 通过对比解耦机制分离共享和任务特定表征。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---
