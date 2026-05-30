## M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 M3oE 框架，通过三个 Mixture-of-Experts 模块（共享专家 S、域专家 D、任务专家 T）以解耦方式学习 common/domain-aspect/task-aspect 用户偏好，并使用两级融合机制（第一级：域间/任务间融合；第二级：三类专家间融合）实现精确的信息聚合控制，再通过 AutoML（Bi-Level Optimization）自适应优化融合权重 α_d, α_t, β_d, β_t。
  - 实验比较 M3oE 与四类 baseline：(a) 单域单任务 MLP；(b) 多任务方法（ShBot-MTL, PLE-MTL, MMoE-MTL, AdaTT, AdaTT-sp）；(c) 多域方法（ShBot-MDL, MMoE-MDL, PLE-MDL, STAR）；(d) 多域多任务方法（ShBot-MDMT, MMoE-MDMT, PLE-MDMT, M2M）。评估指标为 AUC 和 LogLoss。
  - 消融实验：w/o AutoML、Concat modules、Fully gated modules、w/o domain module、w/o task module、w/o domain&task module。
  - 可视化：T-SNE 分析解耦嵌入和融合嵌入。
  - 超参数分析：learning rate、shared expert 数量 N。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的 GPU 或 CPU 硬件配置。
