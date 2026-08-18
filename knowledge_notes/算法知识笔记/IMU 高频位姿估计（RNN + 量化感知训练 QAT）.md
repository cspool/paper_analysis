## IMU 高频位姿估计（RNN + 量化感知训练 QAT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ECHO 提出的 IO（Inertial-Only）模式位姿估计：因 SLAM 相机仅 10-30Hz，帧间捕获间隔 T_IN 大、运动检测滞后，而 IMU 可达 1000Hz。ECHO 用轻量循环神经网络（RNN）在两次 SLAM 位姿更新之间做高频插值：以当前 IMU 数据 + 最新 SLAM 优化位姿/速度/传感器偏置为输入，输出 100Hz 的 7D 位姿（3D 平移 + 4D 四元数）。为降低推理开销，权重做 per-channel INT4 量化、激活用 FP8，并用量化感知训练（QAT）保持精度。RNN 只做短期预测（被后续 SLAM 优化输出周期性纠正，不作为独立跟踪器），跨数据集泛化研究显示其训练集选择几乎不影响精度（Combined/AEA-only/TUM-only 训练 ATE 0.0332/0.0338/0.0337 vs 默认 0.0326）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 运行流程（IO 模式，Fig.11b）：①ECHO 加速器直接读 IMU 测量（Step 1）；②计算引擎跑量化 RNN：输入串联最新 IMU 读数（多步窗口）与 SLAM 提供的位姿/速度/偏置，RNN 单元（INT4 权重 × FP8 激活）在 8×8 weight-stationary 脉动阵列上执行矩阵乘，SFU 提供 ReLU/Tanh 非线性，输出 100Hz 7D 位姿（Step 2）；③位姿交 CPU 做声学注视聚类（Step 3）、GPU 渲染（Step 4）、DAC 输出（Step 5）。RNN 在 MI 帧之间以更短间隔产出位姿，使 T_IN 从相机周期（10Hz→100ms）降到 100Hz 的 10ms 期望（ECHO 建模取 T_IN=5ms）。训练：每数据集分别 QAT 训练（AEA 用 4 测试 + 10 训练/验证序列；TUM VI 每序列 20% 测试/80% 训练验证）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：QAT——训练时在前向传播中插入 INT4/FP8 量化器（直通估计器）让模型感知量化噪声，推理时用纯整数/混合精度计算；per-channel 权重量化把缩放因子按输出通道分组，比 per-tensor 精度高。硬件落地：权重离线量化后驻留片上 buffer，运行时载入脉动阵列；输入激活经 FP8 量化单元实时转换。使用收益：压缩 T_IN（motion-to-sound 延迟公式中的采样延迟），与 MI 模式交错（Hybrid mode）使整体 T_m-s=max(T_m-s^MI, T_m-s^IO)，且 RRE 比 ORB-SLAM3 更低（1.014° vs 1.194°，混合模式平均），方位误差小直接有利于声学注视聚类。注意点：RNN 位姿是短期插值，长期漂移靠 SLAM 纠正，不能独立长时跟踪。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
