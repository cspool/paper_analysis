## ORB-SLAM3（视觉-惯性 SLAM 框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ORB-SLAM3 是 University of Zaragoza 的 Carlos Campos 等人提出的开源视觉-惯性 SLAM 框架（IEEE T-RO 2021），支持单目/双目/RGB-D/惯性（IMU）配置，被广泛用于机器人与 AR/VR 的 marker-free inside-out 6DoF 位姿跟踪。它由三线程组成：①Tracking（跟踪）线程——对每帧做 ORB 特征检测（图像金字塔多尺度）、描述子匹配、PnP 局部地图跟踪（LM tracking）、IMU 预积分、位姿估计与关键帧决策，逐帧执行，其每帧延迟决定跟踪频率上限；②Local Mapping（局部建图）线程——关键帧插入时处理共视/生成地图点、局部 Bundle Adjustment（惯性模式下滑动窗口）；③Loop Closing（回环）线程——BoW2 词袋重定位检测回环、Sim(3)/SE(3) 几何验证、位姿图优化。后两者异步、仅在特定条件触发，不阻塞实时跟踪。ECHO 论文用 ORB-SLAM3 作为商业 VR 跟踪管线的代表做延迟剖析与优化对象。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在 ECHO 中 ORB-SLAM3 是位姿估计（T_P）主路径，每帧流程：输入 480×640 单目灰度帧（AEA 数据集 10Hz）+ IMU（1000Hz）→ ①构建 8 级图像金字塔（迭代下采样，尺度不变性）；②每层按 35×35 像素 cell 划分，FAST 角点检测器逐像素扫描，比较中心像素强度与 16 邻域像素，n≥12 个连续邻域显著更亮/更暗即角点，再用强度矩为关键点分配主方向（旋转不变性）——ORB 提取约占 tracking 时间 23.92ms；③BRIEF 描述子匹配建立当前帧 2D 关键点与 3D 地图点对应，RANSAC 剔除离群点；④IMU 预积分 + 位姿预测初始化位姿，LM tracking 以最小化重投影误差做 40 次 Gauss-Newton 迭代优化（每迭代投影全部对应并算 Jacobian）——约 24.09ms；⑤关键帧决策。ORB 提取与 LM tracking 合计占 >95% tracking 时间（总 ~50ms），是 ECHO 优化与硬件加速的两个靶点。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：开源仓库 https://github.com/UZ-SLAMLab/ORB_SLAM3（C++，g2o 后端优化，DBoW2 重定位）；可编译运行于桌面 CPU 与嵌入式平台（ECHO 用 Nvidia Jetson Orin NX）。用法：提供相机标定（含鱼眼 Kannala-Brandt 模型参数）、IMU 标定与数据集（如 AEA、TUM VI）即可跑通；评估位姿精度常用 evo toolkit 算 ATE（绝对平移误差，m）与 RRE（相对旋转误差，°）。ECHO 在其上叠加低精度量化、点过滤、RNN 高频位姿与硬件加速器，将每帧 tracking 延迟降到 11.0ms（TUM VI），且 ATE/RRE 与全精度相当。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
