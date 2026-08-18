## DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为三层协同的稀疏算法栈：(1) ESS（Epitopological Sparse Sampling）——受 epitopological learning 方法 [141] 启发，基于 Pearson 相关系数离线生成全局相关性注意力掩码，对像素阵列做 attention 引导的 50× 像素下采样（仅掩码有效坐标的像素参与 eventification），把"重要性高的像素保留、冗余像素抑制"的全局结构先验固化进像素阵列；(2) PAC（Patch Activation）——把稀疏事件按 16×16 分组为 patch，仅当 patch 内累计事件数超过阈值（实验取 2）才被激活读出，实现 in-sensor token pruning，最多减少 61% 的 host 端算法 MAC；(3) Robust ViT——conv stem（两层 depthwise-separable 卷积，输出 128 维）+ conv enhancement（两层 3×3 卷积替代标准 ViT 位置嵌入，引入跨 patch 局部交互）+ 3 个 transformer encoder（8 head、128 维多头自注意力）+ 平均池化 + 检测头（两层 FC + sigmoid），在 50× 压缩率（仅 2% 像素使能）下达到 0.5° 角误差。
  - 实验比较：算法精度对比两种 SOTA 事件相机 gaze 算法（一种在 EVBEYE [120] 上报告、一种用 OpenEDS 的 RitNet backbone [28] + 本文检测头）；稀疏采样方法对比三种：BlissCam 的随机稀疏采样 [47]、像素并行 DVS 稀疏方法 [67]、event transformer [98]，以及 event-density based denoising [45]；消融 PAC-only（无 ESS）。指标：角误差 AE（压缩率 1×–50× 扫描）、pixel error（ini-30 瞳孔追踪）。
- 硬件平台是什么，配置是什么。
  - 算法训练：NVIDIA A100 GPU（ESS 掩码离线生成约 2 分钟，用 27 个 subject 中随机 22 个生成、其余 5 个 unseen 验证）；ViT 训练 batch size 64、500 epochs。模型部署评估：STM32N6x7 处理器（16 nm，Arm Cortex-M55 + Neural-ART NPU，执行 MobileNet v2 基准 6 mJ/inference），LSQ INT8 量化、ONNX 导出、STM32Cube.AI 异构部署（卷积/线性层在 Neural-ART NPU、LayerNorm/Softmax 等非线性层在 Cortex-M55）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Robust ViT（conv stem + conv enhancement + 3 transformer encoder + detector head），无参数量/FLOPs 明确报告。数据集/bench：EVBEYE [17]（首个事件相机 gaze 数据集，27 subject、random saccades + smooth pursuits 两类眼动任务，40 英寸 1920×1080 屏、40 cm 阅读距离，剔除 stop/pause 标签与每次 saccade 前 15 个标签，训练/测试集不区分左右眼，AE 全数据集逐 inference 平均）；泛化验证 ini-30 [22]（DVXplorer 事件相机 640×480 镜架式采集、pupil 中心标注，5-fold 交叉验证，对比 SNN 方法 Retina）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明 DESSCam 代码开源；联网搜索未找到其公开仓库。基座组件开源：EVBEYE 数据集与代码（https://github.com/aangelopoulos/event_based_gaze_tracking）、LSQ、ONNX、STM32Cube.AI 工具链。
  - 算法 pipeline 执行例子（H×W 像素阵列、50× 压缩率的 gaze 估计）：
    ① 离线 ESS 掩码生成：把训练集 N 个 event frame 展平成 M×1 向量（M=H×W）并按列堆叠为 M×N Sample Matrix → 用 Pearson 公式 ρ_{X,Y}=cov(X,Y)/(σ_X·σ_Y) 对每对特征求相关，得 M×M Correlation Matrix → 逐行求和得 M×1 Feature Importance Matrix（每个像素与其他像素的相关性和，即该像素对 eye-tracking 任务的重要性）→ 以稀疏阈值 TH 二值化（>TH 置 1）得 binary Mask Matrix，写入像素阵列 SCtrl SRAM 使能对应像素的 eventification。TH 可调以控制下采样率（50× 压缩率 → 仅 2% 像素使能）。
    ② 事件累积：使能像素中 Vdiff 越过 VH/VL 阈值时生成 ON/OFF 事件，按 16×16 patch 聚合，事件数 > 2 的 patch 被 PAC 激活读出。
    ③ ViT 前向（每次 gaze 估计使用 12 个激活 patch）：conv stem 对稀疏事件帧做 depthwise-separable 卷积 → conv enhancement 两个 3×3 卷积替代位置嵌入、在 token 序列形成前做跨 patch 交互 → 输出特征图应用同一 ESS 掩码得到稀疏 token → 3 个 transformer encoder（8 head、128 维）捕捉长程依赖与全局相关性 → 平均池化 → 两层 FC + sigmoid 输出 (x_pred, y_pred)，AE = arccos(v_pred·v_gt / (|v_pred|·|v_gt|))，其中 v=(x, y, L0)、L0 为受试者到屏幕距离。
    ④ 部署：LSQ 量化为 INT8 → 导出 ONNX → STM32Cube.AI 按算子切分（卷积/线性 → Neural-ART NPU，非线性 → Cortex-M55）。
  - 效果：50× 压缩率下 AE 0.5°（同压缩率下 PAC-only 无 ESS 为 4.7°，AR/VR 不可接受）；压缩率 1×–50× 全程 AE 保持在 2° 以内；ini-30 上 50× 压缩率 pixel error 2.76±0.15，优于无稀疏的 Retina（3.24±0.79）；PAC 最多减少 61% host MAC。
