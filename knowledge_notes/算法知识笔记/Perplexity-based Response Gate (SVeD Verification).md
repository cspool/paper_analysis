## Perplexity-based Response Gate (SVeD Verification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perplexity-based Response Gate 是 SVeD (Streaming Verification Decoding) 推理框架中的核心决策机制。它通过监控当前字幕 [Dec] 在 incoming frame 上下文下的 perplexity 变化来决定是否触发新字幕生成。具体地，在时刻 t_j 收到新帧后，通过单次 forward pass 计算 PPL^{t_j}([Dec]) = sqrt[N]{1/P([Dec] | [Ctx^{≤t_j}], [Frm^{t_j}])}。若 PPL^{t_j} > α · PPL^{t_i}（α 默认 1.03），说明新帧的视觉内容与当前字幕语义不匹配（perplexity 上升），应激活解码 gate 生成更新字幕；否则保持沉默。这种方法将"响应vs沉默"的决策从离散的 EOS 分类问题转化为连续的 perplexity 变化检测问题，避免了 EOS token 的词汇表污染和全帧解码开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在推理 pipeline 中的位置：每个 incoming frame → Vision Encoding → PPL Verification → Gate Decision → (if activated) Full Decoding / (else) Silent Pass。

```
# PPL Verification伪代码
def verify_and_gate(Frm^{t_j}, Ctx, Dec, alpha=1.03):
    # 1. 追加新帧到上下文
    Ctx_new = Ctx + [Frm^{t_j}]
    
    # 2. 单次forward pass计算perplexity
    # model.forward() 输出: logits for all positions
    logits = model.forward(Ctx_new)[-len(Dec):]  # 仅取Dec位置
    # PPL = exp(-1/N Σ log P(token_i))
    log_probs = log_softmax(logits, dim=-1)
    token_log_probs = gather(log_probs, Dec_token_ids)
    PPL_new = exp(-mean(token_log_probs))
    
    # 3. Gate Decision
    if PPL_new > alpha * PPL_reference:
        return ACTIVE_DECODE  # 激活解码
    else:
        return SILENCE  # 保持沉默
```

perplexity 计算仅需一次 forward pass（约 1ms），而完整 decoding 需要逐 token 生成（约 10-50ms per token）。在 1 分钟视频 @3fps 含 5 个语义变化段的场景中，180 帧仅触发 5 次完整 decoding + 180 次 verification passes，比 EOS-based 方法（180 次完整 decoding）减少约 97% 的 decoding 开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Alpha 阈值调优：在 OmniStar-RNG 上 α ∈ [1.0, 1.1] 区间搜索，发现 1.02-1.04 最优，选 1.03 为最终配置。α 越大 → 更频繁触发解码（更敏感），TimDiff↓ 但 TimRedun↑；α 越小 → 更保守（更少解码），TimRedun↓ 但 TimDiff↑。实现时需要在 SVeD 循环中维护 PPL_cache（每次成功解码后更新 reference PPL），swap_last_two(Ctx) 操作保持沉默帧的 Dec 在上下文末尾以维持时间一致性。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
