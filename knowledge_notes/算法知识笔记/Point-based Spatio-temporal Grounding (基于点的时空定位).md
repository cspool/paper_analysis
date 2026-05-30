## Point-based Spatio-temporal Grounding (基于点的时空定位)

术语解释
VLM通过生成带时间戳+空间坐标的point来定位视频中的object/action/event，而非仅输出文本/bbox。包括Video Pointing（一次性标注多个时空点，obj_id for counting）和Video Tracking（连续标注object轨迹with consistent IDs across frames）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将image grounding的2D (x,y)点标注扩展到视频的3D (x,y,t)域。核心格式：`<points coords="ts obj_id x y;...">` 或 `<tracks coords="ts obj_id x y;...">`，ts=秒（1 decimal），obj_id=unique sequential ID, (x,y)=0-1000 normalized coords。Pointing→Counting: 先point→max(obj_id)得count（"point then count" strategy）。Tracking: same obj_id跨多帧→track trajectory, HOTA评估association accuracy。Molmo2-VideoPoint: 650K human queries (8 categories, avg 2.3 points/query, 280K videos)。Molmo2-VideoTrack: 15K queries (avg 2.28 objects/query, 3.6K clips from diverse VOS + bbox datasets via SAM 2 point extraction)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Pointing output: <points coords="2.5 1 320 450;5.0 2 680 320">dogs</points>
# → Dog1 at 2.5s(x=320,y=450), Dog2 at 5.0s(x=680,y=320), count=2

# Tracking output: <tracks coords="0.0 1 635 522;0.5 1 606 490;1.0 1 515 164">person</tracks>
# → Person (obj_id=1) track: moves from (635,522)→(606,490)→(515,164) over 1s

# Evaluation: 
# Pointing F1 = point in GT mask? (2 fps sampling, window-based tolerance)
# Tracking HOTA = sqrt(DetA × AssA), DetA=binary mask hit, AssA=ID consistency
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
点坐标归一化至0-1000（image-resolution-independent）。格式选择HTML-like (token-efficient) vs JSON。Tracking数据pipeline: segmentation/bbox tracks→SAM 2 point extraction (alpha-weighted centroid+boundary distance score)→Human text query annotation+validation。训练: upsampled high-count examples + auxiliary tasks (first/last frame only, single-point tracking)。Molmo2 video pointing F1=38.4 (vs Gemini 3 Pro 20.0), tracking HOTA=57.5 (vs Gemini 3 Pro 29.1)。适用于视频搜索、机器人、安防等需要pixel级时空定位的应用。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding
