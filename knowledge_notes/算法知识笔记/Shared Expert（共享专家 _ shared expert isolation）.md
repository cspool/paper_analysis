## Shared Expert（共享专家 / shared expert isolation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Expert 是 MoE 架构的一种设计：在 MoE 层中设置一组始终激活、不参与路由的专家，所有 token 都必须经过它们计算，与 router 选择性激活的 routed experts 并行；其输出与 routed 输出合并。动机：传统 MoE 中通用知识（common knowledge）被冗余存储于各专家，shared expert 把通用知识集中承载，让 routed experts 专注特化域，提升参数效率。典型配置：DeepSeekMoE 2 shared + 64 routed；DeepSeek-V2 2 shared + 160 routed；Qwen1.5-MoE 4 shared + 60 routed；Qwen2-57B-A14B、XVERSE-MoE-A4.2B 继承。SMoE 论文中 shared experts 有两个作用：(1) 常驻 GPU 的共享专家提供稳定的通用表示，使"用共享专家+缓存专家预测下一层 top-score 专家"的预取打分更准确；(2) 因 shared expert 吸收通用知识，非共享专家中 low-score 与 top-score 分化更明显，强化了专家替换的合理性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
h = input_hidden_state                     # 当前 token
# Shared Experts（始终激活，对所有 token，不经过 router）
shared_output = Σ_{i=1}^{n_shared} FFN_shared_i(h)
# Routed Experts（router 选择性激活）
logits = h @ W_gate
topk_vals, topk_idx = TopK(SoftMax(logits), K)
routed_output = Σ_i gate_weights[i] * FFN_routed_i(h)
output = shared_output + routed_output     # 合并
```
SMoE 的具体用法：解码每一层时 shared experts 与 attention、gate 同属常驻 GPU 的 common parameters；预取预测阶段用"GPU 中未共享专家（已缓存）+ 共享专家"生成 hidden state → 走下一层 attention（用下一层 KV cache）→ 计算下一层 gate 分数，从而在真实 router 运行前预测出下一层 top-score 专家并提前 PCIe 预取。命中率约 82%（95% 概率为 active）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在模型 config 中以独立参数组存在（如 Qwen 的 shared_expert_intermediate_size），HuggingFace Transformers 的 MoE forward 先算 shared experts 再算 routed experts 后合并。DeepSeekMoE 论文首次系统提出，Qwen/DeepSeek 系列采用；MoLE 中 shared expert 保持标准 FFN 计算、与 attention 一起常驻 VRAM 不参与 offload。研究还发现 shared experts 会降低局部路由一致性（bypass effect + 缩小 expert combination space），offloading 场景下需权衡（Not All Models Suit Expert Offloading）。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
