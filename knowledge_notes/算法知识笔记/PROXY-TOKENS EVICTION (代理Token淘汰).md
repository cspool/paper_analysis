## PROXY-TOKENS EVICTION (代理Token淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PROXY-TOKENS EVICTION 是 NACL 提出的 KV Cache 淘汰评分策略。在 attention matrix A ∈ R^{p×p} 中，仅使用输入末尾的一小部分 token（proxy tokens P，~10% tokens，对应用户问题部分）的 attention scores 来评估所有 prefix token 的重要性，而非 H2O 的累加全部 token 的 attention 或 MSRNN 的仅用当前 token。

评分函数：F_score = Σ_{x_p∈P} Softmax(A(x_p, *))。F_score[j] 表示 token j 对所有 proxy tokens 的综合重要性。淘汰建模为组合优化：S_t = argmax_{S⊂R} Σ_{x∈S} F_score(A, C_p) ∪ P，其中 R = x_prompt \ P，proxy tokens P 默认保留。

直觉：proxy tokens（用户问题）的 attention pattern 反映了"哪些 prefix token 对回答当前问题有用"，比全量累加更精准。0% proxy = MSRNN（信息不足），100% = H2O（冗余干扰），~10% 最优。

从算法pipeline角度拆解术语：

```
输入: A ∈ R^{p×p}, P = {p*0.9,...,p-1}, C_p

Step 1: F_score = Σ_{x_p∈P} Softmax(A[x_p, :])     # column-wise sum over proxy rows
Step 2: R = {0,...,p-1} \ P                          # exclude proxy tokens
Step 3: S_proxy = TopK(F_score[R], C_p)              # top-C_p non-proxy tokens
Step 4: S_keep = S_proxy ∪ P                         # proxy tokens always kept
```

术语一般如何实现？如何使用？

NACL 开源实现基于 PyTorch + FlashAttention-2。encoding 阶段收集 proxy tokens 的 attention scores（通过 FlashAttention-2 backward 方式重算或仅对 proxy tokens 重算），column-wise sum 后 top-k。默认 proxy tokens 设在输入末尾（用户问题处），无法区分时默认末尾 token。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---
