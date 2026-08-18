## Ray Tracing Accelerator（RTA，光线追踪加速器 / RT Core）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RTA 是集成在现代 GPU（NVIDIA RTX [5]、Imagination PowerVR Photon [3]）中的专用固定功能硬件，与通用 shader core（SM）协同完成光线追踪两大功能：(1) 加速结构（BVH）遍历——高效行走场景树；(2) 内置几何（box 与 triangle）求交测试，由专用固定功能运算单元完成。GauTracer 论文采用 Vulkan-Sim [44] 建模的典型 RTA 微架构：每个 SM 配 1 个 RTA，光线按 warp 分组（与 SM warp 对齐），RTA 内部含 warp buffer、内部 warp 调度器、遍历栈、operation arbiter（节点解码 + 目标表）与运算单元（RBIU 射线-盒求交、RTIU 射线-三角形求交、TRAN 射线变换）。
- 从硬件架构角度拆解术语，给出运转流程具体例子：一次 traceRay 的 RTA 控制流（GauTracer II-B）：①SM 调度器发出 traceRay 指令 → 射线元数据（origin/direction）入队 RTA 的 warp buffer；②每周期内部 warp 调度器选活跃 warp，从遍历栈取顶部节点；③内存响应后节点数据被 operation arbiter 解码，按类型转发到对应运算单元（instance 节点→TRAN、内部节点→RBIU、mesh 叶→RTIU）；④结果（命中点/距离）写回更新射线状态，或派发到通用核做着色。GauTracer 在该流程中新增 Gaussian 叶类型（descriptor 1 bit→2 bit：triangle/procedural/3D Gauss/2D Gauss），把 TRAN 复用为"射线到单位球空间变换"，再顺序派发到新单元 RGIU，随后 AGHU 处理命中，无需新增仲裁阶段。
- 术语一般如何实现？如何使用？：商用实现为 NVIDIA RT Core（Turing 起，加速 box/triangle 求交 + BVH 遍历，LSS、Micro-Mesh 等非网格原语扩展）与 AMD RDNA 的 ray accelerator；学术模拟实现为 Vulkan-Sim（https://github.com/ubc-aamodt-group/vulkan-sim，基于 GPGPU-Sim + Mesa 的 cycle-accurate 模拟器）。GauTracer 在 Vulkan-Sim 的 RTA 上做微架构扩展（RGIU/AGHU/PRUNE），用 RTX2060 配置（30 SM、每 SM 1 RT 单元、RT warp buffer 4）评估，并外推到 RTX3090 级（82 SM、L1/L2 加倍）验证扩展性。

TTP 补充视角（ISCA'26，RT unit 微架构与硬件预取）：TTP 论文使用的 RT unit 模型：warp buffer 存每线程遍历栈与射线元数据；每周期选一个 warp 并服务该 warp 的一个内存请求，同 warp 内重复请求 coalescing 后插入内存访问队列送 L1；内存响应进 response FIFO，数据喂给 math units（ray-box/ray-triangle 求交、坐标变换）；warp buffer 依求交结果更新并把新节点地址压栈。BVH 遍历由单条 CISC 式 trace ray 指令在 RT unit 内完成。TTP 的修改仅在此 warp buffer 上：每线程加 2-bit FSM + 预取指针 + 比较器，监控栈 push/pop 生成预取（见 TTP 条目）。
涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
