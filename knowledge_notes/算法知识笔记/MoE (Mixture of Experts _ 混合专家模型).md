## MoE (Mixture of Experts / 混合专家模型)

术语是什么？
MoE (Mixture of Experts) 是一种神经网络架构模式，在LLM中将标准Transformer的FFN层替换为多个并行"专家"子网络（每个expert是一个独立FFN），由一个可学习的Gate Network（门控网络/路由器）为每个token选择top-k个expert进行计算。核心特征是sparse activation：每个token只激活少数expert（如top-1或top-2），因此模型总参数量可以很大但每个token计算量保持可控。代表性模型：Mixtral-8x7B（8 experts, top-2, 47B total/13B active）、Qwen1.5-MoE、Phi-3.5-MoE、DeepSeek-V3（256 experts, top-8, 671B total/37B active）。MoE在decoder-only LLM每层有8到256个expert，未被激活的expert参数称为inactive parameters（占72%-84%总参数）。

从算法pipeline角度拆解术语：
在decoder-only LLM的每个Transformer block中，MoE层替代标准FFN：
```
# 第l层，token i：
h = input_hidden_state  # [d_model]
g = GateNetwork(h)      # [num_experts], Softmax归一化的概率分布
topk_idx, topk_prob = TopK(g, k)  # 选概率最高的k个
output = sum(prob * expert_ffn(h) for idx, prob in zip(topk_idx, topk_prob))
```
流程：hidden state→Gate Network计算所有expert概率→Top-K选择→所选expert各自计算FFN→加权累加。稀疏激活使每token计算量≈dense model×active/total parameter ratio。

术语一般如何实现？
常用HuggingFace Transformers `MixtralSparseMoeBlock`实现。Gate Network为线性层(d_model→num_experts) + Softmax。训练时加load-balancing loss鼓励expert均匀使用。推理时Expert Parallelism将不同expert分布到不同GPU。Expert offloading将不活跃expert放CPU memory、按需加载到GPU。Gate Network输出完整概率分布（不仅是top-k选中集），在FineMoE中被用于iteration-level expert map和probability-aware prefetch/cache决策。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

