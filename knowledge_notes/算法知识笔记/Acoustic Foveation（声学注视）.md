## Acoustic Foveation（声学注视）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 声学注视（Acoustic Foveation）类比视觉注视渲染（foveated rendering）：人类听觉空间分辨率并非均匀，朝向中央方位（frontal）时最灵敏、偏离中央方位（peripheral）时显著下降，因此处于低感知重要性区域的声源可被分组并合并为单一源（把它们的源音频求和），从而降低 SS 渲染计算量，同时在最关键区域保持空间精度。ECHO 论文指出先验工作（IEEE VR 2025 "Perceptually-Guided Acoustic Foveation"）基于该感知特性；论文将其扩展为"鲁棒声学注视"（robust acoustic foveation），即考虑位姿估计误差的注视聚类。核心心理物理依据：最小可听角度（MAA）随方位角单调递增（正前方 ~3°、侧向 ~90° 处接近 40°）；距离感知不是精确值而是区间，距离判断标准差超过源距离的 20%，故相距小于该阈值的源在感知深度上不可区分。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - ECHO 的鲁棒声学注视算法流程（对应 Fig.8）：①把 3D 房间沿高度切分为若干水平层，把问题降为若干 2D 子问题；②每层内按听者朝向（Ori.）计算每个源的方位角 θ，方位角差 < MAA 阈值的源并入角向组；③角向组内按径向细化：距听者距离差 < 最远源距离的 20% 的源视为感知等价并入簇；④位姿误差鲁棒化：设角向跟踪误差 Δθ^r 与平移误差 Δt，源相对听者的角偏差下界为 θ_eff = θ - Δθ^r - Δθ^t（Δθ^t≈||Δt||/r，r 为源距离），MAA 在 θ_eff 处取值，从而实施更严格的聚类阈值，保证即使位姿有误差也不破坏感知有效性；⑤每个簇用位于原源质心的单一虚拟源替代，进入后续 BRIR 生成与可听化。伪代码示意：
  ```
  def acoustic_foveation(sources, pose, layers, MAA_fn, dist_thresh=0.2):
      for layer in layers:
          for src in sources_in(layer):
              src.theta = azimuth(src, pose.orientation)
              src.theta_eff = src.theta - pose.err_rot - pose.err_trans/src.dist
          angular_groups = group_by_MAA(sources_in(layer), MAA_fn(src.theta_eff))
          for g in angular_groups:
              cluster(g) by radial distance within dist_thresh of farthest src
      return [virtual_source(centroid(c)) for c in clusters]
  ```
  ECHO 把 MAA 实现为对 [77] 感知曲线的分段方位函数；用户研究（17 人，544+288 trial，2IFC）显示注视渲染与全渲染在感知上无显著差异（p≈0.25/0.82），证明注视保留空间音频保真度。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：聚类在渲染前完成，源数从 N 降到簇数 K（K≪N），T_R1（每簇一次 ISM+BRIR）与 T_R2（每簇一次卷积）随之下降。与声源空间分布强相关：Poisson Cluster Process（PD，空间聚簇）比均匀随机分布（UD）产生更少簇（256 源时 70 vs 129），注视收益更大（PD 在 256 源仍 <50ms，UD 接近 70ms）。依赖高质量 head pose（聚类以位姿为条件），位姿误差会破坏 MAA——ECHO 用 θ_eff 保守化解决；其 RRE 更低（1.014° vs ORB-SLAM3 1.194°）也直接支撑注视有效性。使用场景：多声源 VR 场景（8-256 源）、大型展厅等；注意远场 HRTF 假设（源距 ≥1m）。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
