## 2D Bilinear Token Pooling / 二维双线性Token池化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D Bilinear Token Pooling 是 Multimodal LLM 中压缩视觉 token 数量的空间保持型池化方法。Vision encoder (CLIP ViT) 输出 24×24 = 576 个 patch tokens / image。为减少输入 LLM 的视觉序列长度，使用 2×2 bilinear pooling 将 token 网格从 24×24 压缩到 12×12 = 144 tokens（75% 压缩率）。与 1D pooling（直接平均池化到 144 tokens，丢失 2D 位置信息）不同，2D pooling 保持 12×12 spatial layout，使压缩 token 仍编码空间关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 2D Bilinear Token Pooling: 576 → 144 tokens
def bilinear_pool_2d(H_v):  # H_v: [576, D]
    H_grid = H_v.reshape(24, 24, D)        # [24, 24, D]
    # 2x2 avg pooling preserving spatial layout
    H_pooled = avg_pool2d(H_grid.permute(2,0,1),
                          kernel_size=2, stride=2)  # [D, 12, 12]
    return H_pooled.permute(1,2,0).reshape(144, D) # [144, D]

# Contrast 1D Pooling:
def avg_pool_1d(H_v):  # drops spatial info
    return H_v.reshape(144, 4, D).mean(dim=1)
```

Annotations: CLIP ViT-B/32 gives patch grid 24×24=576 tokens；2×2 pooling → 12×12=144；kernel_size=2, stride=2；`avg_pool2d` = bilinear downsampling with spatial awareness。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA Table 6 消融：No pooling (576 tokens) GQA 63.2/Mile 38.2；1D (144) 60.4/36.2；2D (144) 61.3/37.7。2D 在 GQA/SEED/Mile 上均优于 1D。Token compression 对细粒度任务的负面影响通过 Image Partitioning (pad to multiple of 168, split into 168×168 blocks) 缓解：V* 上 49.6%→68.5%。144 tokens/image 使单卡 A100 80GB 可处理 ~1000 张图像。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
