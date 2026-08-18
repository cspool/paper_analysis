## Fine-grained MoE（细粒度专家混合架构 / fine-grained expert segmentation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-grained MoE 是 DeepSeekMoE（Dai et al., 2024）提出的 MoE 架构变体：在总参数量不变的前提下，把传统 MoE 的少量大专家拆分为大量小专家（专家数量 N 增大、每个专家 FFN 中间维度缩小），并增大 top-k 使每个 token 激活更多专家，从而提升专家专业化（specialization）、允许更丰富的知识子域划分。核心数量关系：总参数量 ≈ N_experts × d_expert_size，细粒度通过增大 N、减小 d 保持参数量不变。与 vanilla MoE 的典型对比：Mixtral 式 E=8、top-2、d_ff=4h vs DeepSeekMoE 式 E=64、top-8、d_ff=h/4。DeepSeek-V2 扩展到 2 shared + 160 routed（top-6 routed），Qwen2-57B-A14B、XVERSE-MoE-A4.2B 均采用该设计。SMoE 论文正是以 fine-grained MoE 为研究对象：因为共享专家吸收通用知识 + 细粒度产生高度特化的非共享专家，激活专家中 gate score 高度不均（只有少数 top-score 专家显著影响输出），这构成了"专家替换"的算法前提。fine-grained MoE 的代价：top-k 更大使 dispatch/All-to-All 通信量随 top_k 线性增长（BigMac 表 1：top_k=8 时 All-to-All 占训练 91.8%、推理 90.6%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 同一 MoE 层，vanilla vs fine-grained 配置对比
# Vanilla（如 Mixtral）：E=8, top_k=2, d_ff=5632
# Fine-Grained（如 DeepSeekMoE / Qwen2-57B-A14B）：E=64, top_k=8, d_ff=704

# Fine-Grained MoE 前向（token 粒度）
x = input_token                 # [h]
logits = x @ W_gate             # [N] 全部专家打分
# SMoE 视角：logits 排序后高度不均——前几名（top-score）主导输出，
# 其余被激活专家（low-score）分数与未激活专家相当
topk_idx, topk_w = TopK(SoftMax(logits), k)
output = Σ_i topk_w[i] * Expert_i(x)     # 激活的 k 个专家 FFN（可选加 shared expert 项）
```
从系统角度，fine-grained MoE 使每个专家更小，单次加载一个专家的 PCIe 传输量更小、cache 可容纳更多专家（利于替换候选池），但每 token 激活专家数多导致未命中时加载次数多——这正是 SMoE 用专家替换把 low-score 专家加载量消掉的原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DeepSeekMoE 首次系统化提出（2 shared + 64 routed，top-6 routed）；DeepSeek-V2 扩展到 160 routed + 2 shared；Qwen2-57B-A14B（107GB，S3 设置）与 XVERSE-MoE-A4.2B 直接继承该设计。在 HuggingFace Transformers 中由 MoE 层 config（num_experts、num_experts_per_tok、expert intermediate size、shared_expert_intermediate_size）表达。相关系统工作：BigMac（DCCA 低维通信）、IFMoE（细粒度 MoE 推理框架）、X-MoE（HPC 上 DeepSeek 风格 expert-specialized MoE）、FasterMoE（fine-grained MoE 推理分析）、Scaling Laws for Fine-Grained MoE。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
