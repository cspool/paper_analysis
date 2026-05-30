## Shared Experts in MoE

术语是什么？
Shared Experts（共享专家）是MoE架构的一种变体：在MoE layer中设置一组始终激活的expert（shared experts），所有token都必须经过它们计算，再结合一组通过router选择性激活的routed experts。典型配置：Qwen1.5-MoE每个MoE layer有60个routed experts（每token选top-4）+ 4个shared experts；DeepSeek-R1有256个routed experts（每token选top-8）+ 1个shared expert。Shared experts的引入使得MoE层既保持了sparsity的计算优势，又通过shared experts保证了基础表示能力——所有token都经过shared部分处理，避免了某些token因routing不当而完全遗漏重要特征。

从算法pipeline角度拆解术语：
含Shared Experts的MoE layer计算流程：
```
h = input_hidden_state                    # 当前token

# Shared Experts（始终激活，对所有token）
shared_output = Σ_{i=1}^{n_shared} FFN_shared_i(h)

# Routed Experts（选择性激活）
logits = h @ W_g                         # Router计算
probs = Softmax(logits)
topk_vals, topk_indices = TopK(probs, K)  # K=4 routed experts

routed_output = 0
for each selected expert i:
    routed_output += gate_weights[i] * FFN_routed_i(h)

output = shared_output + routed_output    # 合并
```
从S-MBU角度：shared experts对应的𝟙[l,i]恒为1（i=1..4），routed experts的𝟙[l,i]需通过profiler追踪。因此vanilla MBU的高估程度在含shared experts的模型上相对较低（batch size=1时高估约1.5×而非3×），因为shared部分始终计入S_model。

**MoLE 中的 Shared Expert：** MoLE 的 shared expert 保持标准 FFN 计算（不接受 embedding tokens 输入，接受中间特征 h），推理时执行标准 SwiGLU 计算（FLOPs = 4dD_s）。MoLE 的 shared expert 承担了 routed experts 被 LUT 化后缺失的"上下文相关"计算能力——因为 LUT-based routed experts 的输入不含上下文信息（仅 input_ids），shared expert 仍从中间特征 h 中提取上下文信息。这种"shared expert（有计算）+ routed LUT experts（无计算）"的组合实现了 FLOPs 等同于同大小 dense model（FLOPs_MoLE = 4dD_s = FLOPs_dense）。

术语一般如何实现？
Shared experts在模型config中以独立参数组存在（如Qwen的`shared_expert_intermediate_size`），HuggingFace Transformers在MoE layer forward中先计算shared experts再计算routed experts，最后合并。DeepSeek-MoE论文[10]首次系统提出shared experts设计，后续Qwen1.5-MoE[4]和DeepSeek-R1采用。MoLE codebase (https://github.com/JieShibo/MoLE) 中 shared expert 即为标准 MLP(config)，与 attention 权重一同常驻 VRAM，不参与 LUT offloading。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
- Mixture of Lookup Experts
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**共享专家对局部路由一致性的影响** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文发现 Shared Experts 是降低 MoE 模型局部路由一致性（Local Routing Consistency）的重要因素。在所有 REAL 模型中，高 SRP 的 Group 1 和 Group 2 模型均不使用 shared experts；TOY 模型的 1ShrExp/2ShrExp 变体在相近 PPL 水平下 SRP 显著低于 Baseline。双重机制：(1) Bypass effect——更多信息由 shared 处理，使 routed expert 相对不重要；(2) 减小 expert combination space——从 C(64,8) 降至 C(62,6) 约 72×，限制了 router 在相邻 token 间做局部调整的能力。论文结论：若目标部署场景涉及 expert offloading，架构设计时应权衡 shared experts 的好处（保留通用能力）与 local routing consistency 的下降。

**Nexus 中 Shared Expert 的使用**：Nexus 将 seed model 的原始 FFN 作为 shared expert（而非像 BTX 那样作为普通 routed expert），目的是"更好地保留 upcycling 前 seed model 的通用语言能力"。在 Nexus 的 470M MoE 中：1 shared expert + 6 routed experts（top-1）= 每 token 激活 2 experts；2.8B MoE 中：1 shared + 4 routed。Shared expert 始终激活确保模型不会因 routing 失误而丢失基础能力——这在扩展新 Code expert 时尤为关键：shared expert 保留了非 Code 域的知识，防止 catastrophic forgetting。实验显示 Nexus 扩展 Code expert 后通用任务性能仅下降 1.9%（相对），而 shared expert 是实现这一稳定性的关键组件。
