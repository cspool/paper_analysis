## 事件相机眼动追踪数据集（EVBEYE / ini-30）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVBEYE（Angelopoulos 等，TVCG 2021）是首个近眼事件相机 gaze 数据集：27 个 subject、DAVIS346 传感器（346×260）同步采集左右眼事件流（events.aerdat）与约 25 fps 灰度帧，两个实验范式对应两类眼动（random saccades 与 smooth pursuits），刺激显示在 40 英寸 1920×1080 屏、40 cm 阅读距离；开源于 https://github.com/aangelopoulos/event_based_gaze_tracking。ini-30（Bonazzi 等，Retina 工作，CVPRW 2024）是首个在传感器上标注 pupil 中心的事件相机眼动数据集：两台 DVXplorer（640×480）镜架式采集，受试者自由观看收集自然眼动。两者分别支撑 gaze 估计与 pupil 追踪任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# EVBEYE 用法（DESSCam）：
清理标签（剔除 stop/pause、每次 saccade 前 15 个标签；左右眼不区分）
随机 22/27 subject -> 生成 ESS 掩码（A100 约 2 分钟）
全部 subject -> 训练 Robust ViT（batch 64、500 epochs）
全部 subject -> 测试，AE 逐 inference 平均
# ini-30 用法（泛化验证）：
5-fold 交叉验证 ESS + ViT，50× 压缩率，pixel error 2.76±0.15 vs Retina 3.24±0.79
```
注意点：掩码生成 subject 与训练测试数据分离（unseen 验证泛化性）；AE 定义为预测/真值 3D gaze 向量夹角 arccos(v_pred·v_gt/(|v_pred||v_gt|))，v=(x,y,L0)、L0 为受试者到屏幕距离。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
EVBEYE 仓库提供 conda 环境（ebv-eye.yml）、setup.sh 下载脚本与 visualize.py 可视化；被 Papers with Code 收录、广泛用作事件相机 gaze 基准（gaze 精度 0.45°–1.75°@45°–98° FOV）。ini-30 随 Retina 工作发布（Speck 芯片上 5 mW 运行）。使用场景：事件相机眼动追踪算法评测、稀疏采样/去噪/剪枝算法消融（DESSCam 以二者分别作为主评估与跨任务泛化评估）。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
