## Speculative Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding (SD, Leviathan et al., 2023) 是一种 LLM 推理加速技术。核心思想：用一个较小/较快的 draft model 自回归生成多个候选 token，然后由原始大模型（target model）在一次前向中并行验证这些 token。如果 draft token 与 target model 的预测一致（acceptance rate 高），则等效于一次前向生成了多个 token，实现 wall-clock 加速。

标准 SD 流程：(1) Draft model 自回归生成 γ 个候选 token；(2) Target model 一次前向输入 [prefix + γ 个候选 token]，输出 γ+1 个 logits；(3) 逐 token 比较 draft 和 target 的预测分布，通过 rejection sampling 接受匹配的 token，在第一个不匹配处截断并重新采样；(4) 接受的 token 追加到输出，继续下一轮。

IFMoE 的 Self-Draft 变体不同于标准 SD：(1) Draft model 和 target model 是同一个 fine-grained MoE 模型，区别在于激活的 expert 数（draft 用 2 experts，target 用 6 experts）；(2) 不做逐 token rejection sampling，接受所有 draft token；(3) 通过 KV-cache revision（用全量 experts 重算 KV）补偿 draft 阶段的信息损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

标准 Speculative Decoding 算法：

```
# Standard SD (Leviathan et al. 2023)
Input: prefix p, draft model M_q, target model M_p, draft length γ
while not EOS:
    # Draft phase (auto-regressive, small model)
    draft_tokens = []
    for i in 1..γ:
        q_i ~ M_q(p + draft_tokens)
        draft_tokens.append(q_i)
    
    # Verification phase (parallel, large model)
    p_1..p_{γ+1} = M_p(p + draft_tokens)  # single forward
    
    # Rejection sampling
    for i in 1..γ:
        if random() < min(1, p_i(x_i)/q_i(x_i)):
            output.append(draft_tokens[i])  # accept
        else:
            output.append(resample from p_i - q_i)  # reject, break
            break
    p = p + output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SD 的有效性依赖：(1) Draft model 的 acceptance rate 足够高（通常 >70% 才有加速收益）；(2) Draft model 推理速度显著快于 target model；(3) Target model 的验证前向（并行处理 γ 个 token）比 draft model 的 γ 次自回归前向更快。典型配置：draft model 参数量约为 target 的 1/10-1/100。

IFMoE 的 self-draft 方法不需要额外 draft model，通过减少激活 expert 数（6→2）自然获得约 3× 的草稿加速，无需额外模型部署和内存开销。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
