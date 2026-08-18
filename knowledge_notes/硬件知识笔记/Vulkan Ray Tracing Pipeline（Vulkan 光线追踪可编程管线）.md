## Vulkan Ray Tracing Pipeline（Vulkan 光线追踪可编程管线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Vulkan Ray Tracing（VK_KHR_ray_tracing，2020 [39]）是 Khronos 标准化的可编程光线追踪执行模型，与 NVIDIA OptiX [40]、DirectX Raytracing（DXR）[41] 并列。定义 5 种 shader 阶段：Ray Generation（每像素生成主射线、初始化 ray 属性、发起 traceRayEXT）、Intersection（候选图元被访问时执行自定义求交，用于非内置 procedural 类型）、Any-Hit（可见性/命中筛选，可拒绝或记录命中）、Closest-Hit（最近命中后的着色）、Miss（未命中时执行）。BVH 遍历由 RTA 固定功能处理，可编程 shader 在 SM 上执行。GauTracer 用 Vulkan 实现 Gaussian ray tracing（§III）：Gaussian 注册为 procedural 原语（一对一映射到 BLAS 叶），ray-gen/intersection/any-hit/closest-hit 四个 shader 用 SSBO（Gauss Param Buffer、Closest Hit Buffer、Hit Count Buffer）管理参数与命中。
- 从硬件架构角度拆解术语，给出运转流程具体例子（GauTracer §III-C + Alg. 1-3）：ray-gen shader（一线程一射线）while(Ray.trans > threshold && Ray.thit > 0) 循环内 traceRayEXT(BVH, origin, direction, tMin, tMax)，结束条件为 miss 或透射率低于阈值；每次 traceRay 后 origin += Ray.thit × direction（沿命中距离前进避免重复访问）。intersection shader 内嵌 any-hit 逻辑：确认命中后按 t_hit 比较插入 Closest Hit Buffer（保留最近 K）。closest-hit shader 读 N_hit 遍历混合更新颜色/透射率，并把 Ray.thit 设为 buffer 最远命中。Vulkan-Sim 采用延迟调度：intersection 任务在遍历期间记入 task table，warp 内所有线程遍历完成后逐个执行，从而解耦遍历与 shader 执行延迟（便于细粒度分析）。
- 术语一般如何实现？如何使用？：实现 = Vulkan API 应用 + GLSL shader + 加速结构（TLAS/BLAS），运行在支持 VK_KHR_ray_tracing 的 GPU 上（NVIDIA RTX、AMD RDNA2+）；学术模拟 = Vulkan-Sim 显式映射 Vulkan ray tracing pipeline 到 GPU（MICRO 2022 [44]）。GauTracer 用它做 baseline 实现（忠实遵循 3DGRT 算法流程，不做算法改动），并用 Vulkan-Sim profiling 量化软件 shader 的瓶颈（ALU 指令占 80.3%、shader 侧访存约占 BVH 遍历的 21%、带 treelet 时 shader 占 72.9% 总延迟），支撑硬件扩展动机；RTX3090 上用 OptiX 部署 3DGRT 交叉验证（procedural 高斯比 icosahedron 硬件事 1.6× 慢，shader 开销 38%，与模拟器 43% 一致）。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
