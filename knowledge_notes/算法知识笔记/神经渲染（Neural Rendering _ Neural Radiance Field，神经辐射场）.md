## 神经渲染（Neural Rendering / Neural Radiance Field，神经辐射场）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 神经渲染指用神经网络/可学习表示替代传统显式几何（网格、点云、体素）来做 3D 场景表示与图像合成的技术族（参考 Tewari et al., "Advances in neural rendering", CGF 2022）。按 NeRArch-Sim 论文的分类，主流神经渲染 pipeline 分三类：(1) MLP-based（如 NeRF [Mildenhall 2020]）——用 MLP 隐式建模辐射场，低内存占用；(2) Grid-based（如 Instant-NGP 多分辨率哈希编码、voxel grid）——用离散化空间结构存预计算场景特征，渲染质量高；(3) Primitive-based（如 3D Gaussian Splatting）——用显式几何原语（三角形/3D 高斯）走光栅化，渲染速度快，可视为"零层网络"。此外还有 hybrid（如 grid+3DGS 混合）管线。NeRArch-Sim 论文把神经渲染 pipeline 统一分解为四个阶段：Field Sampler（沿相机射线采样 3D 点/做视锥剔除）→ Encoding（位置编码，RFF/哈希编码）→ Field Computation（MLP/球谐算颜色密度）→ Blending（体积渲染/排序+alpha 混合）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 NeRF 推理 pipeline 为例（MLP-based）：`for pixel: 发射相机射线 r → 沿射线采样 N 个 3D 点 {x_i} → RFF(x_i) 位置编码 → MLP 输出 (σ_i, c_i)（密度+颜色）→ 体积渲染积分 C(r) = Σ T_i·α_i·c_i → 得像素颜色`。NeRArch-Sim 中同一 pipeline 用统一算子接口搭建：`g = OperatorGraph(); s = UniformSampler(dim, graph=g); e = HashEncoding(dim, num_levels=16, graph=g); m = MLP(dim, in_dim=e.out_dim, num_layers=4, graph=g)`，即把采样/编码/MLP 映射为分类学算子，再由插桩框架自动提取成算子图供硬件模拟。MLP/grid/primitive 三种 pipeline 在 NeRArch-Sim 中被建模成不同算子图（vanilla-nerf、instant-ngp、splatfacto 三种 Nerfstudio 模型），是"跨 pipeline 公平基准"的算法侧基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现层面：算法框架用 Nerfstudio（NeRF 开发框架）、GauStudio、Kaolin Wisp 等；NeRArch-Sim 利用与这些框架分类学相似性，通过运行时钩子插桩 Nerfstudio（checkout 指定 commit，注入 tracing.py/eval.py）用 `ns-eval --enable-trace` 渲染一帧输出 execution_dag.pkl 算子图。加速器实现方面，NeRArch-Sim 复现的 ICARUS（NeRF 专用）、NeuRex（Instant-NGP 类）、CICERO、SRender、GSCore、GS Processor、GBU、Uni-Render 等即各类 pipeline 的硬件实现，用 SystemC/Catapult HLS 按同一分类学模块化建模。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
