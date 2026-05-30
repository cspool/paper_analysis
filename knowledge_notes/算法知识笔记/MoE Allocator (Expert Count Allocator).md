## MoE Allocator (Expert Count Allocator)

术语解释
MoE Allocator 是 Ada-K 路由中引入的轻量级可学习模块，负责为每个 token 动态决定应激活的最优专家数量，与原始 router 协同工作实现自适应专家分配。

术语是什么？
Allocator 是一个可训练的线性层 W_alloc ∈ R^{C×N}（C = hidden_dim, N = num_experts），输入 token 的 hidden state，输出该 token 应激活 1 到 N 个专家的概率分布。结构上与 MoE router 完全同构——两者都是线性投影 + SoftMax，但功能不同：router 决定"激活哪些专家"，allocator 决定"激活多少个专家"。

与 router 的关键区别：
- Router: 输出在 N 个专家上的概率分布 P_router ∈ R^N，Top-K 选择哪 K 个专家
- Allocator: 输出在 N 个可能的 k 值上的概率分布 P_alloc ∈ R^N，采样决定 K = k*
- Allocator 先于 router 执行：k* = sample(Softmax(W_alloc · x_i)) → Router 再执行 TopK(P_router, k*)

从算法pipeline角度拆解术语。
```
# Allocator = 一个小型线性层 + SoftMax + 采样
class MoEAllocator(nn.Module):
    def __init__(self, d_model, num_experts):
        self.linear = nn.Linear(d_model, num_experts)  # W_alloc
        # 输出维度 = num_experts (每个可能的 k 值一个 logit)
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.linear(x)              # [batch, seq, N]
        probs = F.softmax(logits, dim=-1)     # 概率分布 over k=1..N
        k_star = torch.multinomial(probs, 1)  # 采样 (不可微分!)
        log_prob = torch.log(probs.gather(-1, k_star))
        return k_star, log_prob

# Allocator 集成到 MoE Layer
class MoELayerWithAllocator(nn.Module):
    def __init__(self, d_model, num_experts):
        self.router = nn.Linear(d_model, num_experts)   # 冻结
        self.allocator = MoEAllocator(d_model, num_experts)  # 可训练
        self.experts = nn.ModuleList([FFN() for _ in range(num_experts)])  # 冻结
    
    def forward(self, x):
        k_star, log_prob = self.allocator(x)  # 先 allocator
        router_probs = F.softmax(self.router(x), dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, k_star, dim=-1)
        # ... expert computation ...
```

术语一般如何实现？如何使用？
- 实现为一个简单的 `nn.Linear(d_model, num_experts)` + SoftMax + 多项分布采样
- 训练时：采样操作的前向不可微分，需通过 PPO/REINFORCE 类 RL 算法优化
- 推理时：可直接取 argmax (k* = argmax(P_alloc))，无需采样
- 训练参数量极小：Mixtral-8x22B 仅 2.75M 可训练参数（每层 allocator ~49k 参数）
- 可选择性部署：可根据 layer ratio 决定在多少层插入 allocator（实验显示 ratio=1.0 即每层部署最优）
- 训练无关数据域：使用 10k pretrain 或 SFT 数据均可获得相近效果

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---
