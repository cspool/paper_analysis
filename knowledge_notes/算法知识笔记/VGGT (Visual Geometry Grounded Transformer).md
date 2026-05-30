## VGGT (Visual Geometry Grounded Transformer)

术语是什么？
VGGT（Visual Geometry Grounded Transformer）是Oxford VGG + Meta AI开发的feed-forward神经网络，获CVPR 2025 Best Paper Award。它从1张至数百张任意视角图像中，在数秒内直接推断场景的全部3D属性：相机位姿（外参+内参）、深度图、点云图和3D点轨迹，无需任何后处理优化（如bundle adjustment）。使用transformer-based aggregator处理多视图图像，然后专门的camera_head预测pose encoding并解码为旋转+平移+内参矩阵（OpenCV convention）。与DUSt3R/MASt3R需要visual geometry optimization不同，VGGT直接输出准确参数。

从算法pipeline角度拆解术语：
在FoundationMotion的Video Preprocessing阶段（Sec 3.1），VGGT用于**相机运动过滤**：
```
# FoundationMotion中VGGT的使用
frames = sample_frames(video_clip, stride=5)  # 采样关键帧
poses = VGGT.infer_poses(frames)  # 推断每帧相机位姿：R_t, T_t

# 计算相机运动分数
delta_t = mean(||T_{i+1} - T_i||)   # 平均位移变化
delta_r = mean(||R_{i+1} - R_i||)   # 平均旋转变化
motion_score = alpha*delta_t + beta*delta_r + gamma*max(delta_t) + delta*max(delta_r)

if motion_score > 0.3:  # 阈值过滤
    skip_video()  # 相机运动过大→tracking质量差→丢弃
```

作用：过滤相机大幅运动的视频，因为此时物体运动+camera motion耦合使人类都难以描述其真实运动轨迹，tracking和标注精度严重退化。

术语一般如何实现？如何使用？
通过官方GitHub仓库使用，输入多视图图像，输出camera poses、depth maps、point maps。适用于3D重建、pose estimation、multi-view geometry等场景。也支持COLMAP格式bundle adjustment（2025年6月后）。官方：https://github.com/chengwei920412/vggt-3dgs。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
