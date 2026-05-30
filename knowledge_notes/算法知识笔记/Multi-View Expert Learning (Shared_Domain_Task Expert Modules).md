## Multi-View Expert Learning (Shared/Domain/Task Expert Modules)

术语是什么？
Multi-View Expert Learning（多视角专家学习）是 M3oE 框架的核心设计，通过三种类型的专家模块从输入中提取不同视角的信息：(1) Shared Expert Module（共享专家模块）——N 个专家网络处理所有域的输入，通过 D×T 个独立 gate 为每对域-任务生成加权组合，捕获跨域跨任务的共性模式（如通用用户兴趣）；(2) Domain Expert Module（域专家模块）——D 个域专属专家，每个专家关联一个特定域，通过带偏置权重 β_d 的融合策略整合域特定视角和其他域的增强信息；(3) Task Expert Module（任务专家模块）——T 个任务专属专家，每个专家关联一个特定任务，通过 β_t 控制当前任务视角和其他任务视角的融合。三类专家共享相同的网络结构（单层 MLP + LayerNorm + ReLU），但具有独立的可学习参数。

从算法pipeline角度拆解术语：
```
输入: h_d (域d的表示)
// Shared Expert Module (N个专家, D×T个gate)
S_{d,t}(h_d) = softmax(W_gate_{d,t} @ h_d) · [expert_1(h_d), ..., expert_N(h_d)]

// Domain Expert Module (D个专家, 偏置融合)
d_out = β_d·expert_d(h_d) + (1-β_d)/(D-1)·Σ_{k≠d} expert_k(h_d)

// Task Expert Module (T个专家, 偏置融合)
t_out = β_t·expert_t(h_d) + (1-β_t)/(T-1)·Σ_{k≠t} expert_k(h_d)

// 两级融合 (Level-2: 模块间平衡)
h̄_d = S_{d,t}(h_d) + α_d·t_out + α_t·d_out
```
其中 α_d, α_t, β_d, β_t ∈ (0,1) 由 AutoML 自适应学习。

术语一般如何实现？如何使用？
所有专家网络共享相同结构但参数独立，可用 PyTorch ModuleList 实现。Shared expert 的每个 gate 是一个线性层后接 softmax；Domain/Task expert 的融合通过标量权重加权求和实现（无需额外网络）。β_d 接近 1 表示仅依赖域自身专家、忽略其他域信息；β_d 接近 0 表示更多依赖其他域知识传递。T-SNE 可视化（论文 Figure 3）验证了该设计确实产生了解耦的嵌入表示——domain expert 的融合嵌入与对应域嵌入分布相似（域特定专家占主导），而 task expert 的融合嵌入在 β_t 最优时取得多个专家间的平衡分布。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---
