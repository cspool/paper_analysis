## Input-side Adaptation（输入侧自适应）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Input-side Adaptation（输入侧自适应）是 ResAdapt 提出的一种视频 MLLM 效率范式，将视觉 token 预算的分配时机从"编码后"（model-side compression）或"推理迭代中"（output-side agentic reasoning）前移到"编码前"。核心思想：传统方法接受编码器的全分辨率输入作为固定成本，在编码后才进行 token 剪枝或合并（model-side），或通过多轮检索缩放恢复覆盖（output-side）。Input-side Adaptation 则通过一个轻量级 Allocator 在编码前预测每帧的分辨率分配，让 backbone 只处理被缩放的像素——保存的像素预算可 reinvest 为更多帧的时间覆盖。该范式完全兼容 FlashAttention、vLLM 和 SGLang，无需定制 kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Input-side Adaptation 的三阶段对比伪代码：
```
# ===== Model-side (编码后压缩) =====
V_raw = load_video(n_frames=T, resolution=HxW)  # [T, 3, H, W]
tokens = vision_encoder(V_raw)                    # [T*H*W/P², D]  全量计算
tokens_pruned = token_prune(tokens)               # [K, D]  K << T*H*W/P²
answer = llm_backbone(tokens_pruned, query)       # 证据已丢失无法恢复

# ===== Output-side (迭代检索) =====
V_coarse = load_video(n_frames=T/4, resolution=HxW)
tokens_coarse = vision_encoder(V_coarse)
hint = llm_backbone(tokens_coarse, query)          # 第1次 backbone 调用
V_fine = crop_video(hint.spans)                    # 检索细粒度帧
tokens_fine = vision_encoder(V_fine)
answer = llm_backbone(concat(tokens_coarse, tokens_fine), query)  # 第2次调用

# ===== Input-side Adaptation (ResAdapt) =====
V_raw = load_video(n_frames=T, resolution=HxW)    # [T, 3, H, W]
f_coarse = lightweight_encoder(V_raw)              # SmolVLM, frozen, [T, D_coarse]
scales = allocator(f_coarse, query)                # st ∈ [0.2, 1.8] per frame
V_resized = [resize(V_raw[t], scales[t]) for t]    # 编码前缩放
tokens = vision_encoder(V_resized)                 # 仅处理缩放后像素 → 节省 token
answer = llm_backbone(tokens, query)               # 单次 backbone 调用
```

关键计算：Token Retention Ratio ρ = Σ s_t²/T。在 ρ≈0.11 时，attention FLOPs 降低为 ρ²≈0.012（约 83×）。Allocator 基于 SmolVLM (Lpred=4, Dpred=1024)，占 <3% 总 FLOPs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Allocator 使用 Beta 分布参数化连续动作空间：每帧 t，Alpha 头输出 (α_t, β_t) 通过 softplus 确保正值，at ~ Beta(α_t, β_t)，st = smin + at · (smax − smin) where smin=0.2, smax=1.8。分配策略通过 GRPO + CAPO 训练，backbone 保持 frozen（ResAdapt）或联合微调（ResAdapt-RL）。训练框架 VeRL + DeepSpeed ZeRO + vLLM。推理时取 at 的期望值代替采样。代码：https://github.com/Xnhyacinth/ResAdapt。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning
