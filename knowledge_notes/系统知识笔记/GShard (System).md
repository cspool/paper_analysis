## GShard (System)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GShard（Lepikhin et al., 2020）是 Google 提出的 MoE 分布式训练系统，首次提出 expert parallelism 概念。基于 Mesh TensorFlow 实现，支持将 MoE 模型的 expert 分布到数千 TPU 上进行训练（最大 600B 参数，2048 TPUs）。GShard 使用 top-2 gate 选择 expert，并通过 auxiliary load balancing loss 来均衡各 expert 的 token 分配。FasterMoE 将其作为修改 expert selection 的对比 baseline——GShard 每次 iteration 延迟最低（1000ms），但因 load balancing 机制改变 expert 选择，需要更多 iteration 才能收敛。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# GShard 的关键设计:
# 1. Auxiliary Loss for Load Balancing:
#    L_aux = α · Σ_e (f_e · P_e)
#    其中 f_e = expert e 接收的 token 比例
#         P_e = gate 分配给 expert e 的平均概率
#    α: 平衡系数

# 2. Expert Capacity:
#    capacity = (tokens_per_batch / num_experts) × capacity_factor
#    超出的 tokens 被丢弃 ("token overflowing")

# 3. Top-2 Gating with Random Routing:
#    gate_output = softmax(W_gate · x + noise)
#    top-2 experts selected, output = weighted sum
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GShard 在 FasterMoE 中的复现：基于 FastMoE 的 custom gate 机制实现 GShard 的 load balancing policy。FasterMoE 对比实验（MoE-GPT, johnny 集群）：GShard 每次 iteration 仅 1000ms（最快），但需 30.9k iterations 达到 LM-loss=4.0，而 FasterMoE w/ topology-aware gate 仅需 15.3k iterations（1471ms/iter），总收敛时间快 1.37×。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
