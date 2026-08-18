## Ray-Gauss Intersection Unit（RGIU，射线-高斯求交单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RGIU 是 GauTracer 提出的 RTA 固定功能单元，把软件 intersection shader 的"射线-高斯求交 + 响应评估"搬进硬件。Gaussian 叶节点存逆变换矩阵 T'（式 6：T' = [[M^{-1}, -M^{-1}μ],[0,1]]，M^{-1}=S^{-1}R^T，仅 12 元素 48B），把世界空间射线变换到以高斯为中心、主轴对齐的单位球空间。在该归一化空间：3DGS 的"交点"定义为高斯响应沿射线的最大点 = 坐标原点在射线上的投影（t_hit^g = O·(d/|d|)，再乘 |d| 还原世界距离；r_hit² 由勾股定理得）；2DGS 的交点是射线与 z=0 平面的显式交点（t_hit = O_z/D_z，再沿射线取 (u,v)）。响应评估 3D/2D 共用同一流程：r_hit²/2 → 指数激活（分段线性近似，LUT 存斜率/偏置，限制区间外输入直接剪枝：≤0 数值不稳定、超上界必低于阈值）→ ×不透明度 → 与阈值 1/255 比较判定命中，alpha 直接前送 Hit Gauss Buffer 避免重复计算。设计上复用现有 TRAN（读 T' 做两次矩阵向量乘变换 ray origin/direction），RGIU 仅需 3 个 vec3 点积 + 1 个倒数 + 1 个标量乘法（合成浮点除法）+ 1 个 MAC，通过单 bit 模式开关的轻量 MUX 重定向数据流同时支持 3D/2D 模式。面积/延迟：27 cycles、34710.2 µm²（28nm 综合，为 baseline RBIU+RTIU+TRAN 组的 21.8%）。
- 从硬件架构角度拆解术语，给出运转流程具体例子：RTA 遍历到 Gaussian 叶 → operation arbiter 解码 2-bit 类型 → 顺序派发 TRAN→RGIU：TRAN 读叶节点 T'，两次 4×4 矩阵向量乘把 ray 变换到单位球空间 → RGIU 按模式求 t_hit 与 r_hit²（3D：dot(O,d/|d|)·|d| 与勾股；2D：O_z/D_z 与前进取 (u,v)）→ r_hit²/2 → 分段线性 exp → ×opacity → 与 1/255 比较 → 命中则把 (t_hit, alpha, GID) 送入 AGHU。相比 baseline 的软件 intersection shader（ALU 指令占 80.3%、RTX3090 实机 38% shader 开销），RGIU 削减 shader 指令 14.7×，单独带来 2.3~2.6× 加速。
- 术语一般如何实现？如何使用？：RGIU 是论文新增硬件单元，无商用对应物（商用 RT Core 支持 box/triangle/LSS 等，不支持 Gaussian）；实现路径为在 Vulkan-Sim 的 operation unit 模型中加入 RGIU 延迟/面积模型（延迟从 Agner Fog 指令表推导，面积用 28nm 商用标准单元库综合）。可重构 3D/2D 双模式设计是论文贡献点（Motivation 2：避免两套独立求交硬件管线导致面积/功耗翻倍）。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
