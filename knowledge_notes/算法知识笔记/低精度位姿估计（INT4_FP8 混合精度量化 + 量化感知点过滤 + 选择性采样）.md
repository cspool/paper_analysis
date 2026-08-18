## 低精度位姿估计（INT4/FP8 混合精度量化 + 量化感知点过滤 + 选择性采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ECHO 提出的位姿估计低精度化方案，目标是削减 LM tracking 中坐标变换（x_i^c=R·x_i+t）的算术开销：该步是大批量矩阵-向量乘法，baseline 在 CPU 上以 FP64 执行。低精度方案：①旋转矩阵 R 元素 ∈[-1,1] 且正交，乘缩放因子 8 后四舍五入成 4-bit 有符号整数（Q_INT4(r)=clamp[round(8r), -8, 7]），选 8 而非常见 7 是因为除 8 可通过浮点指数位调整实现（硬件友好，免完整乘法）；②3D 地图点 x_i 用 FP8 E4M3（Q_FP8，范围 -448~+448，跨度 896m，覆盖室内场景）；③坐标变换变为 x_i^c=Q_INT4(R)·Q_FP8(x_i)/8+t；④因鱼眼投影非线性高、对数值敏感，π 与其导数全程 FP32。配套点过滤（量化感知）：按 FP8 量化误差 E1=||x_i-Q_FP8(x_i)||>α 剔点，一次性稳定性检查 E2^q=||u_i-π(Q_INT4(R)·Q_FP8(x_i)/8+t)||²>β 丢弃不稳定对应，选择性采样（被拒比例<r1 时随机丢 r2 剩余对应）避免近最优位姿时的多余优化。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 计算过程（量化 + 过滤 + 采样三步）：
  ```
  def low_precision_lm_track(R, t, correspondences):
      R4 = Q_INT4(R)                       # 缩放 8 → [-8,7] 整数（指数位调整）
      pts = [x for (x,u) in correspondences if ||x - Q_FP8(x)|| <= alpha]   # E1 过滤
      stable = []
      for x, u in pts:
          x_c = R4 @ Q_FP8(x) / 8 + t      # 混合精度坐标变换
          if ||u - pi_FP32(x_c)||^2 <= beta: stable.append((x,u))            # E2^q 检查
      if rejected_ratio(stable, pts) < r1:  # 位姿已准
          stable = random_sample(stable, keep=1-r2)                          # 选择性采样
      for iter in range(40):                # Gauss-Newton，复用 R4/Q_FP8
          ... # 每迭代在 INT4×FP8 脉动阵列上算 x_c，投影/Jacobian 用 FP32 PJ 模块
  ```
  消融证据：FP16/FP32 替换低精度模块只带来微小 RRE 增益（1.133 vs 1.153°），ATE 无差别（0.030 vs 0.030/0.029）→ 低精度几乎无损；去掉点过滤（No F）导致跟踪发散（无有效位姿输出），只留 E1（QAF）则 ATE/RRE 劣化到 0.042/1.264 → 稳定性检查+选择性采样必要。超参敏感性：α=0.1、β=120、r1=5%、r2=40% 为最优（α 过小 0.01 跟踪失败、过大 1 精度降；r2=80% 跟踪失败）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：软件侧在 CPU 上做量化和过滤（缩放 8 的 INT4 量化、FP8 转换），迭代优化中坐标变换与投影/Jacobian 卸载到 ECHO 加速器计算引擎（INT4 量化单元在线量化动态更新的 R——位姿在迭代中变化不能离线量化；FP8 单元转换 3D 点与 RNN 激活）。除 8 的除法在硬件上通过 FP 指数位 -3 实现。使用收益：平均丢弃 ~75% 点（point filtering 单独贡献 1.26× 延迟降低）、低精度额外 1.10×（相对 FP32 加速器），能量较无过滤方案省 2.39×。迁移性：该方法针对通用 VIO/SLAM 前端的坐标变换+重投影内核，可迁移到 VINS-Fusion/OKVIS/HybVIO 等。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
