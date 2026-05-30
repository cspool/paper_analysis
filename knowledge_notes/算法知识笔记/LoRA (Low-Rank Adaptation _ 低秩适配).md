## LoRA (Low-Rank Adaptation / 低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LoRA（Low-Rank Adaptation）是一种参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，通过在预训练模型的权重矩阵旁添加低秩分解矩阵来实现任务适配，而无需更新原始权重。核心思想：对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将权重更新表示为低秩分解 $\Delta W = B \cdot A$，其中 $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$。前向传播时：$h = W_0 x + \Delta W x = W_0 x + B A x$。训练时仅更新 $A$ 和 $B$（加 adapter 输出），而 $W_0$ 冻结。$A$ 通常用随机高斯初始化，$B$ 用零初始化，使训练开始时 $\Delta W = 0$，不破坏预训练权重。

从算法pipeline角度拆解术语：

Uni-MoE 中 LoRA 的应用——在不同阶段使用不同配置：

```
# 阶段二（训练模态特定专家）LoRA
r = 64, alpha = 16
仅应用于 LLM 中 MLP 层的 LoRA 微调
# 前向：
x = input_tokens
h_original = Expert_FFN(x)      # 冻结专家参数
h_lora = B @ A @ x               # B in R^{d x 64}, A in R^{64 x k}
output = h_original + (alpha/r) * h_lora

# 阶段三（MoE 联合训练）LoRA
r = 8, alpha = 16
应用于所有专家 + self-attention 层
# 对于每个 token 被 router 分配给 expert e1：
h_e1 = e1(X_E1)                   # 冻结的专家 FFN
h_e1_LoRA = LoRA-e1(X_E1)         # LoRA 适配器
  = W_0 @ X_E1 + (B @ A) @ X_E1   # 式(19)-(20)
h_e1 = h_e1 + h_e1_LoRA            # 式(21)
```

核心：LoRA 使 Uni-MoE 能在不更新全部专家参数（最多 37B 总参数）的情况下高效微调，阶段三仅需更新少量 LoRA 参数 + Router + 投影层，训练成本显著降低。

术语一般如何实现？如何使用？

通过 HuggingFace PEFT 库或手动实现：对目标线性层（nn.Linear）注册 forward hook 或替换为 LoRA 包装类，定义 `self.lora_A = nn.Linear(in_features, r, bias=False)` 和 `self.lora_B = nn.Linear(r, out_features, bias=False)`。常用配置：对 attention 层的 Q/K/V/O 投影和 FFN 的 W1/W2/W3 应用 LoRA，r 取 4~64，alpha 取 8~32。LoRA 权重可与预训练权重合并（merge）进行无额外开销的推理：$W = W_0 + \frac{\alpha}{r}BA$。在 Uni-MoE 的 MoE 场景中，LoRA 不 merge——Router 控制哪些 token 激活哪些专家，LoRA 参数始终在线适配。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
