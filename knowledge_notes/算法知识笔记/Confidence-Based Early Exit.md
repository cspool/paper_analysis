## Confidence-Based Early Exit

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Confidence-Based Early Exit（基于置信度的早停）是 VideoAuto-R1 在推理时使用的规则化决策机制。模型首先生成初始答案 $a_1$（通常 <10 tokens），计算其 length-normalized mean log-probability 作为置信度分数 $s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid t_{<\ell}, q)$。若 $s(a_1) \geq \log \tau$（默认 $\tau = 0.97$），则接受 $a_1$ 并提前终止解码（等效 direct answering）；否则继续生成推理链 $r$ 和审查答案 $a_2$。此机制的关键特性：(1) 仅需 $a_1$ 的 log-probability，无需额外校准器或分类头；(2) 决策完全由 test-time 信号驱动，训练时未显式优化 confidence calibration；(3) $\tau$ 提供连续可控的精度-效率 trade-off knob。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
推理流程（Algorithm 1）：

```
Require: p_θ, v, q, τ=0.97, fallback_string f

# Phase 1: Generate until first <think> tag
tokens_prefix = p_θ.greedy_decode(v, q, stop_on="<think>")
a_1 = parse_first_boxed(tokens_prefix)  # extract \boxed{...}
L = len(tokenize(a_1))

# Phase 2: Confidence computation
if a_1 == f:
    s = -1e6  # fallback forces full CoT
else:
    # Length-normalized mean log probability
    s = (1/L) * sum(logprobs_of(a_1_tokens))  # log p_θ(t_ℓ | context)

# Phase 3: Decision
if s >= log(τ):     # τ=0.97 → threshold ≈ -0.0305
    return a_1       # EARLY EXIT (~10 tokens)
else:
    remaining = p_θ.continue_decode(max_new=4096)
    r = parse_between(remaining, "<think>", "</think>")
    a_2 = parse_last_boxed(remaining)
    return a_2       # FULL CoT (~91 tokens)
```

$\tau$ 的影响（Figure 3）：
- 推理密集 benchmark (VideoMMMU): τ 从 0.86→0.98，accuracy 从 57.5%→58.7%，think ratio 从 29%→55%
- 感知导向 benchmark (VideoMME): accuracy 始终不变（diminishing returns from CoT），think ratio 仍随 τ 增加
- 默认 τ=0.97 为鲁棒选择，无需 per-dataset 调参

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上检测到 `<think>` tag 时暂停生成，提取 `\boxed{...}` 中的 tokens 计算 log-probability。由于 $a_1$ 通常仅包含答案字母/数字/短文本（<10 tokens），confidence 计算开销可忽略。$s(a_1)$ 使用标准自回归 log-probability（由 greedy decoding 的 softmax 输出），无需额外 forward pass。此机制依赖于 token-level confidence 与答案正确性的相关性（Liao et al. 2025 首次系统证明），VideoAuto-R1 在视频域验证了该相关性（Table 8: MVBench/MMVU 上 recall of think-needed samples = 100%，VideoMMMU = 94%）。局限性：(1) 训练时未显式优化 confidence calibration；(2) τ=0.97 为经验最优值，泛化到其他模型家族可能需要重新校准。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
