## SAM2 (Segment Anything Model 2)

术语是什么？
SAM2（Segment Anything Model 2）是Meta AI于2024年7月发布的promptable视觉分割基础模型，继承自SAM但扩展到视频领域。SAM2使用Hiera-based hierarchical vision transformer作为图像编码器（四个尺寸：Tiny 38.9M、Small 46M、Base+ 80.8M、Large 224.4M），核心创新是**streaming memory module**——存储过去帧的object-aware memory context来condition当前帧预测，实现实时视频处理而无需回溯所有历史帧。输入可以是单张图像或视频帧序列，支持point/box/mask三种prompt方式指定分割目标。输出是跨帧的masklet（时空一致性分割mask序列）。2024年9月发布SAM2.1改进版，2024年12月更新支持更好的multi-object tracking和torch.compile加速。

从算法pipeline角度拆解术语：
在FoundationMotion的Object Detection & Tracking阶段（Sec 3.2.3），SAM2作为时序tracking backbone：
```
# SAM2 Two-Stage Tracking in FoundationMotion
M_0 = SAM2VideoPredictor.init_state(video[0], prompts=B_init)
# B_init = 初始帧检测的所有person + object bboxes作为prompts

for t in 1..T:
    M_t = SAM2VideoPredictor.propagate_in_video(M_{t-1})
    # SAM2内部：memory encoder计算当前帧特征 → memory bank存储
    # → memory attention跨帧condition → mask decoder输出当前帧mask
    
    if t % 5 == 0:  # keyframe refinement
        B_new = Hands23.detect(video[t])  # 重新检测手部
        SAM2VideoPredictor.add_new_prompts(B_new)  # 注入新prompts纠正drift
```

ID分配：persons ID∈[0,99]，left_hand=ID×10+1，right_hand=ID×10+4，objects ID≥1000。

术语一般如何实现？如何使用？
通过`SAM2VideoPredictor`类使用。初始化时调用`init_state(video_frame, prompts)`，prompts包含正负点坐标、bbox坐标或mask。之后循环调用`propagate_in_video()`获取每帧mask。支持中途调用`add_new_prompts()`或`remove_objects()`动态增删跟踪目标。官方仓库：https://github.com/facebookresearch/sam2，支持torch.compile加速VOS。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
