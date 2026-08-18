## UniAD（端到端自动驾驶模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- UniAD（Planning-Oriented Autonomous Driving，CVPR 2023 最佳论文，Hu et al.）是端到端自动驾驶算法：统一框架集成自动驾驶两大核心任务——感知（object detection/tracking：BEVFormer、TrackFormer、MapFormer）与预测（motion forecast/occupancy prediction：MotionFormer、OccFormer），各模块均为 transformer 架构，经大量 query token 连接（如 TrackFormer 900 个 query），提供丰富并行机会。M100 基准版本用 RegNet 替换 ResNet-101 主干，以更贴近理想汽车实际部署的 AD 算法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（M100 评估的 MAC 分布，Table II）：CNN 主干 RegNet+FPN（30M 参数、2381.6 GFLOPs，占大部分算力，来自高分辨率图像密集卷积）→ BEVFormer（85.6M、1492.9）→ TempFusion（0.2M、49.0）→ TrackFormer（8.5M、97.17）→ MapFormer（6M、105.94）→ MotionFormer（22.6M、266.55）→ OccFormer（46.2M、687.62）→ Planner（3.5M、220.75）。感知模块（BEVFormer/TrackFormer/MapFormer）通常以高于预测模块的帧率运行，计算需求更大，因此分析聚焦 CNN 主干与感知 transformer。感知推理一帧 ≈ 并行执行上述模块链，query token 之间的并行是数据流架构的主要收益来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 开源实现（OpenDriveLab/UniAD，CVPR 2023），图像+LiDAR 输入。使用：作为 AD 推理 benchmark 对比 M100（8/14 cluster）与 NVIDIA Thor-U——RegNet 13.1ms vs 57.4ms（4.4×）、FPN 4.23 vs 5.1（1.2×）、BEVFormer 7.92 vs 32.83（4.1×）、TempFusion 4.47 vs 17（3.8×）、TrackFormer 1.27 vs 7.95（6.3×）、MapFormer 1.46 vs 6.14（4.2×）；perception 30 FPS vs 7.9 FPS（3.8×，同功率预算，满足高速自动驾驶实时要求而 Thor-U 未达）。剩余 6 cluster 保留给座舱功能，验证多域隔离。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
