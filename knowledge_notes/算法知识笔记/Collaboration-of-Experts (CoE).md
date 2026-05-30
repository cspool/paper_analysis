## Collaboration-of-Experts (CoE)

术语解释
Collaboration-of-Experts (CoE) 是一种多专家模型协作推理范式：将多个独立训练的专家模型（experts）通过路由模块（Routing Module）集成，协同完成推理任务。每个 expert 是独立训练的模型（可以架构不同），Routing Module 决定输入应由哪个/哪些 expert 处理以及 expert 间的调用顺序。

术语是什么？
CoE 与 MoE（Mixture of Experts）是两类不同的多专家模型范式：
1. **训练方式**：CoE 的 experts 各自独立训练/微调，Routing Module 可手动配置；MoE 的 experts 和 router 需联合训练。
2. **路由可分析性**：CoE 的路由规则可离线分析——用户可预定义路由规则，从而提前计算每个 expert 的使用概率和依赖关系。MoE 的 router 在推理时动态输出，无法提前获知。
3. **专家管理**：CoE 可独立增删 expert，更灵活；MoE 的 experts 在训练时固定。
4. **精度优势**：CoE 通过集成多个专业化 expert 可达比单一模型更高的精度。例如电路板检测从单模型 92%→CoE 99.9%。

CoE 推理流程（以电路板缺陷检测为例）：
```
输入: 组件图像 I
1. Routing Module(I) → 选择分类 expert E_class
2. E_class(I) → 输出: (缺陷类型, 是否需要进一步检测)
3. if 需要进一步检测:
     Routing(output) → 选择目标检测 expert E_detect
4.   E_detect(I) → 最终结果: (对齐点, 焊接方向)
```

从算法pipeline角度拆解术语：
CoE 的算法 pipeline 特点：
- 与 MoE token-level routing（每个 token 独立选 expert）不同，CoE 是 request-level 路由——一个请求整体被路由到一系列 expert 组成的 pipeline
- Expert 之间存在依赖链（后续 expert 依赖前置 expert 的输出）
- 这种依赖关系可在推理前通过路由规则分析获取，为系统优化提供了 MoE 不具备的先验信息

术语一般如何实现？如何使用？
- 实现框架：PyTorch（CoServe）、SambaNova SN40L 数据流架构（Samba-CoE）
- Routing Module：用户预定义规则（如组件类型→expert 映射）或独立训练
- Expert 模型：可使用多种架构（ResNet、YOLO、Llama 等）
- 典型应用：电路板缺陷检测（300+ experts, 60GB）、Qihoo 360 CoE（多领域 LLM 协作）
- 局限：论文验证仅限智能制造成本场景；CoE 需提供 routing module 和 expert models

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
