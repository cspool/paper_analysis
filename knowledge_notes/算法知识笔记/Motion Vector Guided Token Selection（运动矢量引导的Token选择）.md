## Motion Vector Guided Token Selection（运动矢量引导的Token选择）

术语是什么？
利用 HEVC 解码时暴露的运动矢量（motion vectors）和预测残差（prediction residuals）作为 patch 级信息熵代理的 token 筛选策略。HEVC 的 P-frame 每个 coding unit (4×4~64×64) 关联运动矢量 d∈R² 和残差信号。运动矢量大小 ||d||₂ 反映局部运动强度，残差能量 |res| 反映外观变化不可预测性。两者相加作为 patch saliency score，从密集 P-frames 中筛选最具信息量的 patches。优势：motion vectors 是 HEVC bitstream 解码的"免费"副产品，无需额外模型推理。

从算法pipeline角度拆解术语：

```
# CU级 → Pixel级 → Patch级
mv_field[pixels] = broadcast(CU_motion_vectors)    # [H,W,2]
for each patch(i,j):
    motion_score = sum(||mv_field[patch]||₂)        # 运动强度
    residual_score = sum(|res[patch]|)               # 残差能量
    saliency[i,j] = motion_score + residual_score

global_topk_indices = topk(all_P_saliency, k=B_P)   # 跨所有P-frames
```

运动矢量需 camera motion compensation（去除全局相机运动）。saliency = motion + residual（等权相加，无学习参数）。选择是全局 Top-K（跨所有P-frames排序，非 per-frame）。

术语一般如何实现？如何使用？
FFmpeg with `-flags2 +export_mvs` 导出运动矢量，或 libx265 API 直接读取。残差信号从 YUV Y-channel 解码。CPU 上计算 saliency → 选中 patch indices → GPU tensor → ViT forward。适用场景：内容自适应的视频 token 分配；推广到 H.264/AV1/VVC 等编码标准。限制：非编码视频需先转码。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
