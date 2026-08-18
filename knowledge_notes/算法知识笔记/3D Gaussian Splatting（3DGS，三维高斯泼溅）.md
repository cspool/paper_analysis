## 3D Gaussian Splatting（3DGS，三维高斯泼溅）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 3DGS（Kerbl et al., SIGGRAPH 2023）把场景表示为大量各向异性三维高斯椭球原语的集合，替代传统显式网格或隐式神经场（NeRF）。每个原语由中心 μ、3D 协方差矩阵 Σ（分解为缩放矩阵 S∈R^{3×3} 与旋转矩阵 R∈SO(3)，存紧凑形式：四元数 q∈R^4 + 缩放向量 s∈R^3）与不透明度 o 参数化，高斯响应为 G(x) = o·exp(-½(x-μ)^T Σ^{-1} (x-μ))（式 1），Σ = RSS^T R^T（式 2）。渲染时把 3D 高斯经 world-to-camera 变换后投影到像平面（局部仿射近似，丢弃变换后协方差第 3 行第 3 列得到 2D 泼溅），按深度从前到后做体积 alpha 混合（式 3：C = Σ T_i α_i c_i）。参数通过光度损失训练拟合 ground-truth 图像。GauTracer 论文中 3DGS 是 ray-oriented 渲染（3DGRT）的算法基础：3D 高斯在单位球空间中的"射线-高斯交点"定义为高斯响应沿射线最大点（原点在射线上的投影）。
- 从算法pipeline角度拆解术语，给出伪代码或具体计算过程例子：3DGS 推理 pipeline = 高斯参数加载 →（每像素/tile）把高斯按 view matrix 投影 → 计算 2D 协方差 → 计算不透明度贡献（2D 高斯泼溅在像素上的积分）→ 深度排序 → front-to-back alpha 混合累加颜色与透射率 → 提前终止（透射率低于阈值）。伪代码示意：
  ```
  for tile in image:
      gaussians = tile 内投影覆盖的高斯（由 3D AABB 剔除）
      sort(gaussians, by depth)
      T, C = 1.0, 0
      for g in gaussians:
          alpha = g.opacity * splat(g, pixel)   # 2D 高斯积分值
          C += T * alpha * g.color
          T *= 1 - alpha
          if T < threshold: break
  ```
  训练侧：photometric loss（L1 + SSIM）反向传播更新 μ, q, s, o 与颜色（SH 系数），并做自适应密度控制（clone/split）。GauTracer 论文按式 1-3 给出响应与混合定义，并指出 3DGS 是 tile-based 光栅化管线，天生受限无法直接支持 ray-oriented 渲染（畸变相机、二次光线）。
- 术语一般如何实现？如何使用？：主流开源实现为原始 3DGS（https://github.com/graphdeco-inria/gaussian-splatting，diff-gaussian-rasterization CUDA 光栅化 kernel），2DGS（https://github.com/hbb1/2d-gaussian-splatting），以及 ray-oriented 的 3DGRT/3DGUT（https://github.com/nv-tlabs/3dgrut，OptiX+Slang）。GauTracer 论文用 3DGS 的 ray-tracing 变体（3DGRT [27]）作为算法 baseline，BVH 用 Intel Embree 构建（分支因子 6），NeRF-Synthetic 数据集 8 场景 30,000 次训练迭代的点云做质量评估。硬件侧（GScore [29]、GCC [33]、GausPU [36] 等）研究 rasterization 的软硬件协同，GauTracer 则把 3DGS 作为一等原语做 RTA 硬件扩展。

NeRArch-Sim 补充视角（ISCA'26，3DGS 作为 primitive-based pipeline 的模拟对象）：NeRArch-Sim 把 3DGS pipeline 按分类学分解为 Field Sampler（视锥剔除）→（无 Encoding，原语自带属性）→ Field Computation（SH 颜色）→ Blending（排序 + alpha 混合），作为 splatfacto（Nerfstudio 3DGS 模型）算子图被插桩提取；复现的 3DGS 加速器包括 GSCore（tile 化排序-光栅化重叠，FPS 190→182.2，误差 4.1%）、GBU（180→172）、Uni-Render（65→63，3DGS pipeline 评估 PSNR 33.0/33.0）、GS Processor（373→343，与流片芯片对比延迟误差 8.0%）。内存侧（表 VIII）：GSCore 每 tile 从 DRAM 流式读 78.8MB 高斯特征（Gaussian In FIFO），排序缓冲 12.4MB SRAM 读写，Pixel Out Buffer 写 3.1MB；bank conflict 开销仅 0.01%（顺序 tile 处理几乎无争用）。DSE 案例即在 GSCore 上用模拟退火调 (Culling Conversion Units, Quick Sorting Units, Bitonic Sorting Units, VRCs, Buffer sizes) 五参数，最优 (16,8,4,32,4) 相对原配置 (4,8,4,64,8) 取得 1.3× energy-delay product 与 1.6× 面积下降。

3DGS 加速器补充视角（ISCA'26，算法-硬件 co-design 的光栅化/排序优化）：论文 profiling（Jetson Orin Nano + MipNeRF-360）显示 projection/sorting/rasterization 各占 14.2%/25.3%/60.5% 延迟；α-computation 每像素/Gaussian 需 8 MUL+4 ADD+1 EXP（α=o·exp(-½(p-μ)ᵀΣ⁻¹(p-μ))，Σ⁻¹ 圆锥矩阵用 a,b,c 参数化），α-blending 需 5 MUL+4 ADD。两项优化：(1) axis-shared rasterization——把 α 指数分解为 X/Y 二次项+交叉项，X-PE/Y-PE 线预计算共享项广播给 16×16 PE 阵列，α-computation 摊销降至 2.31 MUL+2.13 ADD/PE（-63%），总 MAC 减 38%；(2) MLP-based OIT——2 层 10 参数 MLP（输入深度 d_i+视角 (x,y,z)，推理仅 6 MAC，指数输出激活）直接预测透射率 F(d_i) 替代显式排序，渲染式改顺序无关 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i，PSNR 26.90 vs 排序 baseline 27.21（-0.3）、优于 weight-sum[18]（25.43），训练约 30 分钟/场景（预训练 checkpoint 初始化，MLP lr=0.005、GS lr×0.01，关闭 cloning/splitting）。统一可重构 PE 阵列加速器 3.85mm²/1.64W（28nm、1 GHz、256 PE、96KB SRAM、DDR5-4800），相对 GSCore/MetaSapiens/GBU 1.33~1.88×、相对 edge GPU 端到端 4.0~5.5× 加速与 16.2~31.9× 能耗节省（150+ FPS）；GPU 实现基于 gsplat（https://github.com/nerfstudio-project/gsplat），代码开源 https://github.com/WangZhican/ISCA26_3DGS_Acc。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
