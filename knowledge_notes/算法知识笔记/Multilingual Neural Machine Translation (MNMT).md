## Multilingual Neural Machine Translation (MNMT)

术语解释
MNMT 是用单一神经网络模型同时翻译多个语言对的范式，本质上是 multi-task learning 问题。参数共享的程度决定正迁移（positive transfer）的程度，过度共享则导致任务干扰（task interference）因容量瓶颈。

术语是什么？
在 MNMT 中，参数可以在不同语言对之间完全共享（如 Johnson et al., 2017 的 Google Multilingual NMT），也可以部分共享、部分专用。MoE 模型天然适合 MNMT：不同 experts 可以学习不同语言的专业知识，router 根据输入语言动态分配计算资源。Kudugunta et al. (2021) 的 Task-MoE 利用 MNMT 的 task boundary 天然先验，将 "翻译到 French" 和 "翻译到 German" 定义为不同 task。

从算法pipeline角度拆解术语。
MNMT 中参数共享的程度谱系：
- 完全共享（all-shared）：单一 encoder-decoder，所有语言使用相同参数 → 最大 transfer 但容量瓶颈
- 语言特定（language-specific）：每语言独立 encoder/decoder → 无 transfer 但无干扰
- MoE（本论文）：共享非 expert 参数 + task-specific experts → 在 transfer 和 specialization 之间平衡
- Task-level MoE：task 级 expert 选择 → 推理时可提取 task-specific sub-network

实际设置：15-102 种语言，to/from English，温度采样（T=5）处理数据不平衡（150k—64M 句对）。

术语一般如何实现？如何使用？
- SentencePiece 共享词汇表（64k tokens），源句前 prepend `<2xx>` token 指示目标语言
- BLEU (SacreBLEU) 评估
- 数据采样策略（温度 T 控制高/低资源语言平衡）
- Adafactor optimizer，inverse sqrt LR schedule

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---
