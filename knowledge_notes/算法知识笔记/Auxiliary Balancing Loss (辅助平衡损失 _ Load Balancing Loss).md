## Auxiliary Balancing Loss (辅助平衡损失 / Load Balancing Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Auxiliary Balancing Loss（辅助平衡损失）是 MoE 模型训练中的一种正则化项，源于 GShard（Lepikhin et al., ICLR 2021），用于防止 Router（路由器）"坍塌"——即 Router 将所有 tokens 分配给少数几个专家，导致其余专家不被训练、模型退化为 Dense 模型。该损失鼓励所有专家获得大致相等的 token 分配量，从而保证模型容量被充分利用。

GShard 中的典型形式：$L_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i$，其中 $f_i = \frac{1}{T}\sum_{t} \mathbf{1}\{\text{token}_t \text{ routed to expert } i\}$ 是 expert i 实际接收的 token 比例，$P_i = \frac{1}{T}\sum_{t} p_{t,i}$ 是 Router 分配给 expert i 的平均概率。$L_{\text{aux}}$ 在 $f_i$ 与 $P_i$ 不均衡时增大，通过梯度引导 Router 均匀分配。

从算法pipeline角度拆解术语：

Uni-MoE 中的平衡损失实验流程（表 8-9）：

```
# MoE 训练阶段的标准 forward + 辅助平衡损失
for each batch:
    x = input_tokens                     # shape: T x d
    # Router 计算
    logits = x @ W_router                # W_router in R^{d x M}
    probs = softmax(logits, dim=-1)      # probs in R^{T x M}
    # Top-K 选择
    topk_probs, topk_idx = top_k(probs, k=2)
    # Expert FFN 计算 (normal forward)
    output = sum_{i in top_k} topk_probs_i * Expert_FFN_i(x)
    # 辅助平衡损失（如果启用）
    f_i = (1/T) * sum_{t} indicator(token_t -> expert_i)
    P_i = (1/T) * sum_{t} probs[t, i]
    L_aux = alpha * sum_i (f_i * P_i)
    # 总损失
    L_total = L_CE + L_aux
```

Uni-MoE 的关键发现：
1. **Mixture MoE（预训练多样化专家）**：不加 aux loss 时 Avg. 49.2%，加了 aux loss 降至 48.5%（表 8 a vs a'）——因为专家已在阶段二各自发展出模态偏好，Router 天然学会将不同模态 tokens 分配给对应专家，aux loss 反而干扰了自然分化。
2. **Pure MoE（相同初始专家）**：不加 aux loss 时 Avg. 47.5%，加了 aux loss 升至 48.4%（表 8 b vs b'）——相同初始专家缺乏差异化，aux loss 强制 Router 探索不同专家组合。
3. **扩展到 8 专家时**：aux loss 的作用增强（表 9），因为路由搜索空间从 C(4,2)=6 增至 C(8,2)=28 种组合，aux loss 帮助优化专家组合选择。

术语一般如何实现？如何使用？

在 HuggingFace Transformers 的 MoE 实现中（如 Mixtral、Switch Transformers），auxiliary loss 通常以 `load_balancing_loss` 或 `router_z_loss` 的形式内置在 MoE 模块的 forward 中。典型做法：$L_{\text{aux}} = \text{num\_experts} \cdot \sum_i (f_i \cdot P_i)$，超参数 $\alpha$ 通常取 0.01~0.1（平衡主任务损失和平衡损失的量级）。Uni-MoE 的实验表明该损失不是万能的：是否使用需根据专家是否有预训练差异化来决定——当专家已通过预训练发展出明确模态偏好时，aux loss 可能适得其反。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
