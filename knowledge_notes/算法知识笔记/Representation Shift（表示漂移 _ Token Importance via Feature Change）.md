## Representation Shift（表示漂移 / Token Importance via Feature Change）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Representation Shift 是一种训练无关（training-free）、模型无关（model-agnostic）的 token 重要性度量方法，由 ICCV 2025 论文 "Representation Shift: Unifying Token Compression with FlashAttention" 提出。核心公式为 s = Δx = ||F(x) - x||₂，其中 F(·) 为某个层的变换函数（优选 MLP），Δx 量化每个 token 经过该层后的表示变化量。直观理解：对任务关键的 token 会被网络"强调"——其表示经过 MLP 后发生较大变化（大 representation shift）；冗余 token 几乎不变（小 shift）。因此可以通过剪除低 representation shift 的 token 来减少计算量。

与 attention-based token importance（如 EViT 使用 s = Softmax(q_cls K^T/√C)）的本质区别：
1. **不依赖 attention map**：可直接与 FlashAttention 配合使用。FlashAttention 为避免 HBM I/O 不构建完整 attention map，attention-based 方法因此失效。
2. **信号更可靠**：MLP 逐 token 独立操作（per-token independent），产生的 representation shift 比全局 attention（cross-token information exchange，transformation 更 diffuse）更具判别性。
3. **模型无关**：可扩展到 CNN（各 stage 后计算 feature map 变化）、SSM（替换激活值基分数）。

关键消融发现：
- 操作选择：MLP > Attention > Entire Block（Figure 5a），因 MLP 逐 token 独立使信号更具判别性
- 距离度量：L2 > L1 > Cosine（Figure 5b），L2 在所有深度上最一致；Cosine 在深层失效
- 可靠性验证：top 50% vs bottom 50% token 准确率差 26.3%（Table 8）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Representation Shift-based Token Pruning
# 超参: drop_layers = [0,1,2], drop_ratio = 0.2

for layer_idx in range(num_layers):
    # Step 1: LayerNorm + Attention (FlashAttention)
    x_norm1 = LayerNorm(x)
    x_attn = FlashAttention(x_norm1)     # 不暴露 attention map
    x = x + x_attn
    
    # Step 2: LayerNorm + MLP
    x_norm2 = LayerNorm(x)
    x_mlp = MLP(x_norm2)                 # [N, C]
    
    if layer_idx in drop_layers:
        # Step 3: 计算 representation shift (L2 distance)
        delta_x = ||x_mlp - x||_2, dim=-1  # [N]
        
        # Step 4: Top-K 保留
        num_keep = int(N * (1 - drop_ratio))
        keep_idx = topk(delta_x, k=num_keep)
        
        # Step 5: 对 token 维度剪枝
        x = x[keep_idx]                   # [N*(1-r), C]
        x_mlp = x_mlp[keep_idx]
    
    # Step 6: 残差连接
    x = x + x_mlp
```

张量计算流程（以 UMT-B, 12 frames × 224² 为例）：
- 输入 tokens x ∈ R^(12×14×14=2352, C)
- Layer 0: FlashAttention → Δ = ||MLP(LN(x')) - x'||₂ → top-80% → x ∈ R^(1881, C)
- Layer 1: 同样流程 → x ∈ R^(1505, C) 
- Layer 2: 同样流程 → x ∈ R^(1204, C)
- Layer 3-11: 1204 tokens 不变，正常 Transformer

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：加载预训练模型后直接应用，无需额外训练（training-free）。在指定层的 MLP 后插入 L2 norm 计算 + token pruning 模块。L2 norm 计算开销为 O(N × C)，可忽略（<1% total FLOPs）。使用 FlashAttention 的 fused kernel 作为标准 self-attention 后端。开源实现：https://github.com/mlvlab/Representation-Shift（MIT License），使用 `main.py --eval --use_flash True --drop_r [...]`。

配置（论文实验）：
- Video (UMT): drop_layers=[0,1,2], drop_ratio=0.2 (retrieval) / 0.1 (QA), FlashAttention enabled
- Image (DeiT): drop_layers=[1,4,7], drop_ratio=0.2
- CNN (ResNet): line-wise/token-wise pruning after stage 1/2
- SSM (ViM): 替换 ToP-ViM 的激活值基分数

累积加速效果：FlashAttention ~2.7× + pruning ~2× → 总计 5.5× (UMT-L video-text retrieval)。

涉及论文标题：
- Representation_Shift__Unifying_Token_Compression_with_FlashAttention
