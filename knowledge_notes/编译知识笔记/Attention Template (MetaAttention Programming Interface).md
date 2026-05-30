## Attention Template (MetaAttention Programming Interface)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Template（注意力模板）是 MetaAttention 编程接口的核心抽象——将 attention 机制的结构性骨架（relevance scoring + aggregation）固定为不可修改的模板，同时暴露 customizable functions 的注入点（scores_Mod、scores_RowNorm/Rownorm_Online、Q/K/V/output_Mod、h_mod）。模板按两种计算模式实例化：Parallel Pattern（全局上下文，matmul-based relevance scoring and aggregation）和 Recurrent Pattern（压缩 state，iterative matmul-based relevance and aggregation update）。用户在模板中选择 pattern、定义输入 tensor shapes、编写 customizable functions，即可定义任意 attention 变体。

该设计与 FlexAttention (score_mod/mask_mod) 和 FlashInfer (variant functor) 的区别：FlexAttention 仅支持 parallel-style softmax attention（score_mod + mask_mod callable 注入 Triton kernel 模板），不支持 recurrent attention（Mamba2/RetNet）或非标准 shapes（MLA 的 dimqk≠dimv, head_kv≠head）；FlashInfer 通过 JIT CUDA compilation 支持更广的变体空间但仍限于 parallel pattern。MetaAttention 的 attention template 同时覆盖 parallel 和 recurrent 两类 attention pattern，且 customizable functions 不限定于特定 normalization（不预设 softmax）。

从编译框架角度拆解：Template 的定义和实例化流程：
1. 用户声明 pattern（Parallel/Recurrent）+ tensor shapes + customizable functions
2. 编译框架解析 template → 生成 computation graph（IntermediateTensors + customizable function DAG nodes）
3. Two-layer scheduling policy 生成 optimal execution plan
4. Attention Runtime 选择对应 pattern 的 kernel template（parallel 含 online normalization mainloop；recurrent 含 chunk-based mainloop）
5. Customizable function code inline 到 kernel template 的固定注入点
6. Backend-specific lowering（CUTE/TileLang → CUDA/ROCm kernel）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention template API 为 Pythonic DSL。使用示例——RetNet parallel attention（完整定义约 22 行）：
```python
pattern = "Parallel"
inputs = {"Query": [B,H,S,256], "Key": [B,H,Skv,256], "Value": [B,H,Skv,512]}
customizable_functions = {
    "scores_Mod": lambda s: s * mask,
    "scores_RowNorm": RetNetNorm()  # 含 online_prologue/forward/epilogue
}
```
用户调用 `MetaAttention.compile(template, target_device)` 触发 scheduling + code generation，返回 callable kernel。Default template 不带 customizable functions 即为 standard scaled dot-product attention (QK^T/√d + softmax + PV)。Template 的 extensibility 体现于 customizable functions 的无限制组合——同一 attention 变体可通过不同 Mod/RowNorm 组合表达（如 causal attention = scores_Mod with causal mask + identity RowNorm）。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
