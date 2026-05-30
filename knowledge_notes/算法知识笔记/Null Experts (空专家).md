## Null Experts (空专家)

术语解释
Null Experts（空专家）是 AdaMOE 提出的核心机制，指在 MoE layer 的 expert set 中引入的固定数量的"空操作"专家。Null expert 定义为一个消耗 **零 FLOPs** 的空操作（默认 zero mapping: E_null(x) = 0，也可选 identity mapping: E_null(x) = x），在 top-k 路由中被选中时不执行任何计算。通过将 null experts 与 true experts 混合，并增大 top-k 的 k 值，使不同 token 可选择不同数量的 true experts，实现 token-adaptive routing。

术语是什么？
Null experts 的关键特性：
1. **零 FLOPs**: E_null(x) = 0（constant zero mapping），不消耗任何计算资源。可选 identity mapping E_null(x) = x 也消耗零 FLOPs，但论文未探索此方案。
2. **等质无差别**: 所有 m 个 null experts 在功能上完全相同，因此在 load balancing loss 中不对 null experts 之间做负载均衡区分。
3. **token bypass 能力**: 若某 token 在 top-k 中全部选中 null experts，该 token 完全绕过此 MoE layer，实现类似 Mixture-of-Depths (MoD) 的 layer skipping 效果。
4. **计算预算可控**: 通过调整 m（null expert 数量）和 k（top-k 值），可精确控制平均 true expert 负载。Load = k × (n/(n+m))（无 load balancing loss 时的理论值）。

从算法pipeline角度拆解术语。
```
# Null Expert 定义
class NullExpert:
    """空专家: 零 FLOPs 空操作"""
    def forward(self, x):
        # Zero mapping (default): output = 0, 0 FLOPs
        return torch.zeros_like(x)
        # Identity mapping (alternative): output = x, 0 FLOPs
        # return x

# AdaMOE Layer with Null Experts
class AdaMOELayer:
    def __init__(self, n_true_experts, m_null_experts, k):
        self.true_experts = [FFN() for _ in range(n_true_experts)]  # E_1...E_n
        self.null_experts = [NullExpert() for _ in range(m_null_experts)]  # E_{n+1}...E_{n+m}
        self.router = Linear(d_model, n_true_experts + m_null_experts)  # W_g
        self.k = k  # top-k selection, k > vanilla MoE's k
    
    def forward(self, x):  # x: [d_model]
        # Step 1: Router 计算所有 expert (含 null) 的 logits
        logits = self.router(x)  # [n+m]
        
        # Step 2: Top-K 选择
        top_logits, top_indices = topk(logits, self.k)
        
        # Step 3: 分离 true experts 和 null experts
        true_mask = top_indices < self.n_true_experts
        null_mask = ~true_mask
        
        if true_mask.sum() == 0:
            # 全部选中 null experts → token bypass this layer
            return torch.zeros_like(x)
        
        # Step 4: 仅对 true experts 做 Softmax (option 2)
        true_logits = top_logits[true_mask]
        true_weights = softmax(true_logits)
        
        # Step 5: 仅 true experts 贡献计算，null experts 贡献 0
        output = sum(
            true_weights[i] * self.true_experts[idx](x)
            for i, idx in enumerate(top_indices[true_mask])
        )
        # null experts: weight * 0 = 0, 无 FLOPs
        return output
```

术语一般如何实现？如何使用？
- **对 vanilla LLM (Mo-LoRA)**: 在 Mo-LoRA 架构中，每个 layer 的 LoRA experts 作为 true experts（n=4），添加 m=5~9 个 null experts（实现为不执行任何操作的占位符），k=2~4。使用 mola-moe 框架实现。
- **对 MoE-LLM (Mixtral)**: 原始 router gate 输出 n=8 → 新增 gate2 module 输出 m=8~48 → 拼接为 n+m 维 router → k=3~8。gate2 参数可从 gate 复制推导。
- **Load Balancing**: ℓ_null = α·(n+m)·[Σ_{i≤n} f_i·P_i + Σ_{j>n} (avg_f_null)·P_j]，不对 null experts 间做负载均衡。
- **Training**: α annealing: epoch 1 用大 α (0.02) 建立负载均衡，epoch 2 用小 α (0.0001) 释放 token 自由度。
- **关键结果**: Mixtral-8x7B fine-tuning: Load 从 2.00→1.66，FLOPs ↓14.5%，accuracy +1.69% on ARC-C。

涉及论文标题：
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models
