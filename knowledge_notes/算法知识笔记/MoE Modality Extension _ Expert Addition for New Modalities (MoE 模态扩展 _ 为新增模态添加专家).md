## MoE Modality Extension / Expert Addition for New Modalities (MoE 模态扩展 / 为新增模态添加专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE Modality Extension 是一种将预训练的 MoE LLM（原仅处理文本模态）扩展到新模态（如视觉、语音）的方法。核心思想是：**不修改原有 MoE 模型的任何参数（包括 expert FFN、router、attention 层），仅在部分 MoE 层中新增 expert 和对应的 router 参数，仅训练这些新增部分来学习新模态知识**。这与 LLaVA 式的全参数微调形成根本区别——全参数微调修改所有 MLP/FFN 权重来桥接模态间隙，而 MoE Modality Extension 通过为不同模态提供专用 expert，让原有专家继续服务文本模态，新增专家专门处理新模态 token。

在 MoExtend 中的具体实现：为 Mixtral 8x7B 的 50% MoE 层（由 Extender 选出的层）各添加 1 个新 expert FFN_{m+1}，router 拓展为 W_new = [W; v_new] ∈ R^{D×(m+1)}，每 token 仍选 top-2 expert。

从算法pipeline角度拆解术语：
扩展前后的 MoE 层前向计算对比：

**扩展前（原始 MoE，m experts）：**
```
# Router 计算
logits = H @ W_g     # [B, m]
probs = softmax(logits)  
# Top-K selection (K=2)
weights, indices = top_k(probs, K)  
# Expert 加权输出
MoE(x) = Σ_{j=1}^{K} s(x)_j · FFN_{idx_j}(x)
```

**扩展后（MoE + 新模态 expert，m+1 experts）：**
```
# Router 扩展为 m+1 列
W_new = concat(W_g, v_new)  # W_g [D, m] → W_new [D, m+1]
logits = H @ W_new           # [B, m+1]
probs = softmax(logits)
weights, indices = top_k(probs, K)  # 仍选 K=2
# Calibration 校正后加权输出
MoE(x) = Σ_{j=1}^{K} s(x)_j · [1 + s_c(x)] · FFN_{idx_j}(x)
```

关键洞察：新 expert 权重初始化为复制该层原有最活跃 expert（对视觉数据响应最大的），router 列 v_new 同理。这使得新 expert 从"最接近新模态理解"的参数空间出发训练，避免"冷启动"导致的选中概率过低问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **训练成本**：MoExtend 仅训练新增参数（~3B），全参数冻结（~46.7B），训练时间约 45 小时（Alignment ~15h + Fine-tuning ~30h），对比全参数微调约 200 小时，加速约 6 倍（8×A800-80G）。
- **推理成本**：无额外开销——推理时 MoE layer 仍执行 top-K routing，仅 router 从 m 列扩为 m+1 列（对 Mixtral 从 8 选 2 变为 9 选 2），新增参数仅在扩展层存在（50% 层）。
- **防遗忘机制**：原有 expert FFN 参数完全冻结，文本 token 仍优先被原有 expert 选中处理，因此文本性能几乎不降（Avg. drop 仅 0.41 vs 全参数微调 3.30）。
- **扩展性**：方法不限于视觉模态——替换 vision encoder 为语音/其他模态 encoder 即可扩展，也不限于 MoE LLM 文本→视觉扩展场景。
- **已知局限**：1）视觉任务外未验证（论文因 GPU 资源限制仅验证视觉模态）；2）需要足够的专家容量——如果原有 expert 已经过度专门化，简单复制可能不够。

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension
