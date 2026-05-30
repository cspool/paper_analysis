## FarSkip-Collective

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FarSkip-Collective 是一种修改 MoE Transformer 模型残差连接（residual connectivity）的架构方法，通过在通信进行期间使用"过时"（outdated）或"部分"（partial）激活值作为下一子块的输入，消除分布式执行中的阻塞通信模式。核心思想是：标准 Transformer 中 `o_k = o_{k-1} + f_k(o_{k-1})`，即下一子块 `f_{k+1}` 必须等待 `f_k` 的完整输出（包括其通信结果）。FarSkip 改为 `o_k = o_0 + f_1(o_0) + f_2(o_1^*) + ... + f_k(o_{k-1}^*)`，其中 `o_k^*` 是不依赖当前子块通信结果的可用激活值。

两种 `o_k^*` 选择：
- **(8a) Outdated**：`o_k^* = o_{k-1}`，使用上一层的完整输出
- **(8b) Partial**：`o_k^* = o_{k-1} + f_k^*(o_{k-1}^*)`，使用当前子块中不依赖通信的部分计算结果

对于 MoE 层的具体应用：
- Attention 子块输入（partial）：`attn-in_k = o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}`（省略 routed-exp-out_{k-1}），使 Combine 通信可与 Attention 重叠
- MLP 子块输入（outdated）：`mlp-in_k = o_{k-1}`，使 Dispatch 通信可与 Attention 重叠

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip-Collective 修改后的 MoE 层前向执行（训练，EP=8）：

```
# 原始 MoE 层前向:
# attn_out = Attention(LN(o_{k-1}))
# o_k_attn = o_{k-1} + attn_out
# gate = Router(LN(o_k_attn))
# dispatched = AllToAllDispatch(o_k_attn, gate)  ← 阻塞通信气泡
# routed_out = RoutedExperts(dispatched)
# combined = AllToAllCombine(routed_out)           ← 阻塞通信气泡
# o_k = o_k_attn + SharedExperts(o_k_attn) + combined

# FarSkip-Collective 前向:
# 1. MLA q,k,v 准备 (attn-in_k 使用 partial activation)
q, k, v = MLA_prepare(o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1})

# 2. 同步上一层的 Combine (此时 Combine 已被重叠)
WaitCombineHandle(prev_combine_handle)

# 3. MoE gating
gate = Router(LN(o_{k-1}))

# 4. 异步 Dispatch (async_op=True, 立即返回)
dispatch_handle = AllToAllDispatchAsync(tokens, gate)

# 5. Core attention + output projection (与 Dispatch 重叠!)
attn_out = MLA_core(q, k, v)

# 6. 同步 Dispatch
WaitDispatchHandle(dispatch_handle)

# 7. Routed experts
routed_out = RoutedExperts(dispatched_tokens)

# 8. 异步 Combine (与 shared experts 重叠)
combine_handle = AllToAllCombineAsync(routed_out)
shared_out = SharedExperts(o_{k-1})  # 与 Combine 并行
```

重叠窗口条件（Eq. 9）：`T_Dispatch + T_Combine ≤ T_layer - (T_RoutedExperts + T_Gate)`

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **训练侧**：在 Megatron-LM 中实现，使用 `torch.dist.all_to_all(async_op=True)` 异步启动通信，通过 backward hook 和 Sequence Number hijacking 实现反向传播的通信重叠。前向重叠率 87.6%-92.9%，反向重叠率 84.1%-89.0%。
- **推理侧**：在 vLLM/SGLang 中实现，将 EP all-reduce 改为 `async_op` 模式，通过 CUDA Stream 分离通信与计算，使用 PyNCCL 兼容 CUDA graphs。All-reduce 重叠率 95.3%-97.6%。
- **适用范围**：任何 MoE 模型（训练和推理），不改变模型参数形状，仅修改连接性。与 TP、PP、DP 正交兼容。
- **限制**：routed experts 和 gating 的计算不可重叠（它们依赖通信的输入/输出），是重叠窗口的下界。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
