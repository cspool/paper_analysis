## Loss-Free Load Balancing (in MoE-based LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 采用的 MoE expert 负载均衡策略 (基于 DeepSeek-V3 auxiliary-loss-free 策略)。维护 expert-wise bias d ∈ R^r，训练期间手动更新 d_i ← d_i + u·sign(ē_i - c_i)，其中 ē_i 为期望分配频率 (均匀 1/r)、c_i 为实际计数、u 为小步长。当 expert i 过度使用时 d_i 减小 (抑制)，使用不足时 d_i 增大 (鼓励)。bias 在 top-k 前加到 Ax 上 (Eq. 10)：I_topk = argtopk(Ax + d, k)，直接改变 expert 选择而非通过 loss 间接影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Loss-Free Load Balancing (Eq. 9-10):
// d ∈ R^r: bias, 初始化为 0; u ≈ 1e-3

每训练步更新:
  expected = total_tokens * (1/r)
  for i in 0..r-1:
    actual[i] = count(I_topk == i)
    d[i] += u * sign(expected - actual[i])

Forward 中使用:
  I_topk = argtopk(Ax + d, k)  // bias 影响 top-k

// 消融 (Table 3, MMLU, Llama-3.1-8B):
// Loss-Free:       40.88±1.61 (默认)
// Loss-Controlled: 40.59±0.51
// No Balancing:    37.56±2.87 (↓3.32, 方差大)
//
// 对比 Loss-Controlled (Switch Transformer):
// L_aux = α·N·Σ_i f_i·P_i, L_total = L_task + L_aux
// 需额外超参数 α, 与主 loss 梯度竞争
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 来源：DeepSeek-V3 (Liu et al. 2024) 首次提出，FlyLoRA 适配到 MoE-based LoRA
- d 不参与梯度计算——手动更新；u 适当设置避免 bias 震荡
- 开销：r=32 时仅 32 个 float，r 步 O(r) 操作，可忽略
- 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
