## LumiBench（硬件光线追踪基准套件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LumiBench 是 UBC Aamodt 组（Liu et al., IISWC 2023）发布的面向硬件光线追踪的基准套件，配套 Vulkan-Sim 2.0 与 RayTracingInVulkan 应用（Peter Shirley《Ray Tracing In One Weekend》的 Vulkan + NVIDIA RTX 扩展实现）使用。提供 16 个几何复杂度递增的 3D 场景（wknd/ship/bunny/spnza/chsnt/bath/ref/crnvl/fox/party/sprng/lands/frst/park/car/robot），BVH 树大小从 0.2MB（wknd）到 1721.3MB（robot），深度 7-18；每个场景的 BVH 由开源 Intel Embree 库构建（每叶节点 1 primitive）。支持三种 ray tracing 负载：path tracing（closest-hit 主光线+次级光线，最重，每像素 4-16 条弹跳光线）、Ambient Occlusion（最近命中点后向随机方向 4 条光线）、Shadow（向光源 2 条 any-hit 光线）。默认 128x128 分辨率、1 sample/像素，可调分辨率与采样数。TTP 论文用它做主要评估集：15/16 场景在 128x128 完成（park 72h 超时改 64x64），并测 256x256/64x64/32x32 分辨率；chsnt 不支持 AO/SH。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- LumiBench 的 ray tracing pipeline 流程（以 path tracing 为例）：①raygen shader 每像素生成主光线（origin/direction）→ ②trace ray 指令交给 RT unit 做 BVH 遍历（Embree 构建的树），closest-hit 语义找最近命中 → ③命中点处生成次级光线（反射/折射，4-16 条/像素）→ ④每条次级光线再次 trace ray 遍历 BVH → ⑤递归或迭代累加颜色直至收敛。AO/SH 负载：先 trace 一条主光线找最近命中点，再分别向 4 个随机方向（AO，评估环境光遮蔽）或 2 个光源方向（SH，any-hit 提前终止，评估阴影）trace 次级光线。每场景 × 负载 × 分辨率 × 采样数构成实验矩阵；TTP 论文用 path tracing 为主、AO/SH 为辅评估硬件预取器。运行时命令：./RayTracer --scene 20 --width 32 --height 32 --samples 1 > ship_pt.log（--scene 选场景索引）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RayTracingInVulkan（github.com/ubc-aamodt-group/RayTracingInVulkan）用 Vulkan ray tracing pipeline（VK_KHR_ray_tracing）实现各 shader，BVH 用 Embree（embree.org）构建后上传为加速结构；LumiBench 场景文件随套件分发（Zenodo，配套 Vulkan-Sim v2.0）。使用：作为硬件光线追踪架构研究的标准负载集——TTP 论文在 Vulkan-sim 上运行全部 16 场景（park 因 72h 超时降分辨率），对比 DFS/BFS、TTP/Treelet/无预取；关键统计包括每射线访问节点数（表 I）、RT read miss 构成、MPKI、DRAM 带宽等。用途：量化 BVH 遍历的内存瓶颈（表 I 显示 DFS 平均每射线 49.0 节点 vs BFS 70.0），支撑预取/缓存/近存等架构优化的动机与评估。

涉及论文标题：
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
