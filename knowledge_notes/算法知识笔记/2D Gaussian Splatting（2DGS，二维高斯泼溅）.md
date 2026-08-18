## 2D Gaussian Splatting（2DGS，二维高斯泼溅）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 2DGS（Huang et al., SIGGRAPH 2024）由 3DGS 退化而来：把 3D 椭球的 z 方向缩放分量 s_z 置零，3D 椭球退化为平面椭圆（surface-aligned 的 2D 原语），天然适合表达表面几何与法线估计，视图一致性好，常用于几何敏感任务。与传统在质心近似深度不同，2DGS 显式计算像素 (x,y) 视线与 2D 椭圆平面的精确交点 (u,v)（式 4：u = (h_u[2]h_v[4]-h_u[4]h_v[2])/(h_u[1]h_v[2]-h_u[2]h_v[1])，其中 h_u=[-1,0,0,x]·P、h_v=[0,-1,0,y]·P，P∈R^{4×4} 为椭圆空间到世界空间的变换矩阵），保证视图一致的深度评估。GauTracer 论文把 2DGS 用于 ray tracing（2DGRT/IRGS [28]）：射线与 2D 椭圆的交点是显式平面交点（变换后 z 轴垂直于 2D 平面，t_hit = 变换后射线原点的 z 分量 / 变换后射线方向的 z 分量）。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：2DGS 推理 pipeline = 高斯参数加载 →（每像素）变换到椭圆局部空间 → 求视线与 z=0 平面的交点 (u,v)（式 4 的射线-椭圆求交）→ 在该 (u,v) 处评估 2D 高斯响应（比 3D 投影-积分更精确）→ 得到 alpha → 深度排序 alpha 混合。伪代码示意：
  ```
  # 每条像素射线与 2D 椭圆平面求交
  t_hit = -O.z / D.z          # 变换空间中，z=0 平面交点
  P_uv  = O + t_hit * D       # 交点坐标
  u, v  = P_uv.x, P_uv.y      # 椭圆局部 2D 坐标
  alpha = opacity * exp(-0.5*(u^2+v^2))   # 在 (u,v) 处的高斯响应
  ```
  GauTracer 的 RGIU 在 2DGS 模式下就是执行这一求交（z 分量除法 + 沿射线前进取 (u,v)），与 3DGS 模式（原点投影）共享点积/MAC 单元，仅由模式开关切换。
- 术语一般如何实现？如何使用？：开源实现为 2DGS（https://github.com/hbb1/2d-gaussian-splatting）与 2DGRT/IRGS（C. Gu et al., CVPR 2025；相关仓库 fudan-zvg/gtracer 与 fudan-zvg/gaussian-raytracing 为 OptiX 实现）。GauTracer 论文把 2DGRT 作为第二种评估模式：RGIU 无需结构性改动即可支持 2DGS（单 bit 模式开关 + 复用计算单元），2DGRT 模式下平均加速 7.3×（硬件 shader 贡献 5.6×）。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
