## Epitopological Sparse Sampling（ESS，表观拓扑稀疏采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ESS 是 DESSCam 提出的像素级注意力引导稀疏采样机制：用训练集事件帧构造 Sample Matrix（M×N，M=H×W 像素、N 帧），计算 M×M Pearson 相关矩阵（每对像素的全局相关性），逐行求和得 M×1 Feature Importance Matrix（每个像素与其余像素的相关性和，代表其对该任务的重要性），再用稀疏阈值 TH 二值化得到 binary Mask Matrix，写入像素阵列的 1-bit SCtrl SRAM 使能约 2% 的像素做事件化（50× 下采样）。其灵感来自 brain-inspired 的 epitopological learning（ICLR 2024，Zhang/Cannistraci 等，"Epitopological learning and Cannistraci-Hebb network shape intelligence"）：该方法把全局相关矩阵/连接预测用于稀疏化全连接层（ESML + CH3-L3 Cannistraci-Hebb 规则，约 1% 连接保留时在 VGG16/ResNet 上超过全连接网络）。DESSCam 把"全局相关矩阵稀疏化"的思想从网络层迁移到像素阵列，用相关性高的像素保留全局数据结构，抑制冗余/热像素。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ESS 在眼动追踪 pipeline 中位于最前端（离线掩码生成 + 传感器内在线应用）：
```
# 离线（A100 约 2 分钟，22/27 subject）：
Sample = stack(frames)                # M×N, M=H×W
rho(i,j) = cov(i,j)/(sigma_i*sigma_j) # M×M Correlation Matrix
importance(i) = sum_j rho(i,j)        # M×1 Feature Importance
mask(i) = 1 if importance(i) > TH else 0
# 在线（传感器内）：
for pixel i: if mask(i): enable eventification   # 仅 2% 像素使能
events -> 16×16 patch -> count>2 激活 -> ViT 推理
```
TH 可调控制压缩率（50× 压缩率 = 仅 2% 像素采样）；掩码离线生成、跨 subject 复用、无需用户特定重校准。效果：50× 压缩率下 AE 0.5°，同压缩率下 PAC-only（无 ESS）为 4.7°；迁移到 ini-30 瞳孔追踪（50× 压缩）pixel error 2.76±0.15，优于无稀疏的 Retina（3.24±0.79）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对比三种稀疏采样实现：① BlissCam 随机稀疏采样——每帧刷新 10-bit in-pixel SRAM 随机数，只控制总体稀疏度、不指定有效像素位置；② MESA 像素级动态注意力——实时逐像素自适应遗忘因子，硬件代价大；③ ESS——预计算全局注意力系数 + 1-bit SRAM 掩码，硬件友好且保留全局相关性。ESS 还可以替代事件相机的去噪预处理（背景活动噪声过滤、热像素抑制）与 ROI 分割（后两者每 inference 需数千万 MAC）。使用前提：掩码由特定硬件设置采集的训练数据初始化；论文指出未在无约束环境验证。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
