## 射线行进（Ray Marching，射线步进 / 射线采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ray marching 是沿相机发出的射线按一定步长/采样策略离散地探测 3D 场景（采样点、求交）的渲染技术，是 NeRF/Instant-NGP 等隐式神经渲染的核心采样步骤，区别于传统显式 ray tracing 的 BVH 求交。NeRArch-Sim 论文把 Field Sampler 定义为"沿相机射线采样物体（3D 点）定义感兴趣区域"，MLP-/grid-based 管线用 uniform 或 PDF-based 采样，primitive-based（3DGS）则用视锥剔除（frustum culling）丢弃目标 2D 区域外的原语。论文指出通用 NN 加速器模拟器（Timeloop/SCALE-Sim 类）不支持 ray marching、spatial sampling 等图形学专用算子，是构建神经渲染专用模拟器的动机之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 NeRArch-Sim 中 ray marching 是 Field Sampler 阶段算子，且与硬件调度直接耦合：表 VIII 显示 ICARUS 与 NeuRex 都在 Field Sampler 阶段做 ray marching——每帧把射线采样坐标从 DRAM 流式读入（ICARUS 1.4GB、NeuRex 469MB position streaming），体现"采样→编码→MLP"的流水。伪代码：
```
for pixel (u,v):
    r = camera.generate_ray(u, v)         # 生成射线
    for i in range(num_samples):           # ray marching
        x_i = r.sample(depth_i, strategy)  # uniform/PDF 采样 3D 点
        features.append(encode(x_i))       # 交给 Encoding 阶段
```
- 采样策略差异：NeRArch-Sim 分类学里 Sampling 阶段参数含 num_samples 与 sampling_strategy（uniform / PDF / frustum culling）。算子级优化中，"restricted hashing（在 subgrid 内处理射线）"（NeuRex）是 region-level reuse 优化、与 ray marching 的采样组织相关。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：Nerfstudio/Instant-NGP 的 ray sampler（uniform、PDF 采样器）；硬件：专用 sampling unit、Culling & Conversion Unit（GSCore）等。NeRArch-Sim 硬件库 Sampling 类含 Culling conversion unit、Skipping controller、Sampling unit。GauTracer 论文则是把 ray tracing 硬件（RTA）扩展用于高斯表示——ray-gen shader 沿像素发射光线、BVH 遍历（与之对应的是显式 ray tracing 而非 ray marching）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
