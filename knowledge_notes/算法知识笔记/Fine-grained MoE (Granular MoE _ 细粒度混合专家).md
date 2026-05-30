## Fine-grained MoE (Granular MoE / 细粒度混合专家)

术语是什么？

Fine-grained MoE（细粒度混合专家）是 MoE 架构的一个子类，通过缩小每个 expert 的 FFN hidden size（乘以因子 1/G）并增加激活的 expert 数量（乘以因子 G），在保持总计算 FLOPs 不变的前提下，使用更多但更小的 expert。由三个关键超参数定义：(1) E（Expansion Rate）：MoE 层总参数量是 dense MLP 的 E 倍；(2) G（Granularity）：expert hidden size 是 dense FFN hidden size 的 1/G；(3) T（TopK）：每 token 路由到的 expert 数量。总 expert 数 N = E × G。

与 coarse-grained MoE（如 Mixtral 8x7B，E=8, G=1, T=2）相比，fine-grained MoE（如 E8G8T8，64 experts 每个 1/8 hidden size）的理论优势在于：更多 expert 提供更细粒度的路由选择，理论上可实现更精准的 expert 专业化。

从算法pipeline角度拆解：

Fine-grained MoE 层的计算（E8G8T8 为例）：

```
# 参数规模：
# dense FFN: W1(d,h), W2(h,d) 其中 h = intermediate hidden (如 4*d)
# fine-grained expert: W1_i(d, h/G), W2_i(h/G, d)

# 前向传播
x = attention_output   # (S, d)
gate = softmax(x @ W_r)  # (S, N), N=64
topk_val, topk_idx = topk(gate, T=8)  # 选 8 个 expert

# 每个 expert 计算 (hidden smaller by G=8)
output = zeros(d)
for (val, idx) in zip(topk_val, topk_idx):
    h_i = activation(x @ W1_{idx})          # (h/8,)
    output += val * (h_i @ W2_{idx})        # (d,)
```

实际挑战：
1. **Upcycling 困难**：不能简单复制 dense MLP 权重，因为 expert 尺寸不匹配。Virtual Group Init 解决了这个问题。
2. **输出缩放**：多个小 expert 的加权输出需要缩放补偿。Weight Scaling 解决。
3. **系统开销**：更多 expert 意味着更多 all-to-all 通信轮次和更小的 GEMM 尺寸，降低 GPU FLOP utilization。

论文实验结论（Nemotron-2B 和 Nemotron-4 15B）：增加 granularity 到 G=8 有收益，但继续增加到 G=16/32 有 diminishing returns。64 experts (E8G8T8) 在 0.1T tokens 消融中最优，但在 1T+ tokens 大规模训练中 coarse-grained (E8G1T2) 和 fine-grained (E8G8T8) 最终 loss 趋同。Fine-grained MoE 的 scaling law 详见 Krajewski et al., 2024 "Scaling Laws for Fine-Grained Mixture of Experts"。

术语一般如何实现？

Megatron-LM 和 NeMo 框架支持 fine-grained MoE 的配置和训练。关键实现要点：(1) expert 权重按 G 因子缩小 intermediate dimension；(2) 需要 Virtual Group Init + Weight Scaling 来稳定 upcycling 训练；(3) 使用 scattermoe (Tan et al., 2024) 等优化来降低 fine-grained MoE 的内存和通信开销。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
