## Local Map Tracking（LM 跟踪）与重投影误差位姿优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LM tracking 是 ORB-SLAM3 跟踪线程中把当前帧特征与 3D 地图点匹配、并优化相机（头）位姿的阶段。流程：IMU 预积分与位姿预测给出初始位姿 → 选邻近关键帧建立 2D-3D 对应（BRIEF 描述子匹配）→ RANSAC 剔除离群 → 通过最小化重投影误差迭代优化 6DoF 位姿（旋转 R∈R^{3×3}、平移 t∈R^3）。目标函数为 min_{R,t} Σ_i ||u_i - π(R·x_i + t)||²，其中 x_i 是第 i 个世界系 3D 地图点、u_i 是其 2D 关键点、π(·) 是相机投影函数。ECHO 剖析显示该阶段约占 ORB-SLAM3 tracking 时间的一半（24.09ms），是两大瓶颈之一。VR 中 SLAM 相机多用鱼眼镜头（强径向畸变），π 采用 Kannala-Brandt（KB）非线性模型。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 优化过程（ECHO 论文给出细节）：KB 投影对相机系点 x^c=[x,y,z]^T 定义 ρ=√(x²+y²)、θ=arctan(ρ/z)、φ=arctan2(y,x)，径向畸变 d(θ)=θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹，投影 π(x^c)=[fx·d(θ)cosφ+cx, fy·d(θ)sinφ+cy]^T，含三角函数与 9 阶多项式、开销高；位姿优化用 Gauss-Newton（论文引用 [104]）在流形上迭代 ~40 次，每次迭代对每个 3D 点执行坐标变换 x_i^c=R·x_i+t（大批量矩阵-向量乘，FP64）并计算投影/残差与 Jacobian。伪代码：
  ```
  R, t = init_from_imu_preintegration()
  for iter in range(40):
      J, r = [], []
      for (x_i, u_i) in correspondences:          # 2D-3D 对应
          x_c = R @ x_i + t                        # 坐标变换（主要瓶颈）
          r_i = u_i - project_KB(x_c)              # 重投影残差
          J_i = jacobian_KB(x_c)                   # 鱼眼投影 Jacobian
          J.append(J_i); r.append(r_i)
      delta = solve_gauss_newton(J, r)             # 法方程
      R, t = exp_map(R, t, delta)                  # 流形更新
  ```
  ECHO 的加速手段：①低精度——R 量化为 INT4（缩放 8）、x_i 用 FP8 E4M3，x_i^c=Q_INT4(R)·Q_FP8(x_i)/8+t 在混合精度下计算（除 8 靠指数位调整），投影/Jacobian 保持 FP32；②点过滤——按 FP8 量化误差 E1>α 与量化重投影误差 E2^q>β 剔除低质量/不稳定对应，位姿已准（拒绝率<r1）时随机丢弃 r2 比例（默认 α=0.1、β=120、r1=5%、r2=40%），平均削 75% 点；③硬件——计算引擎 8×8 脉动阵列算坐标变换、PJ 模块单遍算投影+Jacobian，T_P^MI 平均降 3.4×。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：LM tracking 是通用 VIO/SLAM 前端模式（VINS-Fusion、OKVIS、HybVIO 也有类似角点+重投影管线，ECHO 的优化可迁移）。落地方式：在 CPU 上跑优化循环（ORB-SLAM3 C++），或把坐标变换/投影/Jacobian 这类可并行、可复用的计算卸载到硬件（ECHO 加速器）；精度评估用 evo 工具算 ATE/RRE（RMSE），数据集用 AEA/TUM VI。使用注意：鱼眼模型的 trig/高阶多项式对数值敏感，量化时投影与 Jacobian 必须留 FP32 防精度劣化（ECHO 消融显示 INT4/FP8 量化相对 FP32/FP16 几乎无损：ATE 0.030 vs 0.030/0.029，RRE 1.153 vs 1.133/1.133）。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
