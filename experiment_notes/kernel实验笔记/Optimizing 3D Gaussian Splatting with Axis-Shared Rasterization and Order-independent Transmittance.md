## Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：两项 GPU 端 kernel 实现，用于评估专用加速器之外的性能边界：(1) GPU-implemented axis-shared rasterization——在 NVIDIA RTX 3090 上用 CUDA 把轴共享光栅化三阶段（共享项计算→广播→组合）映射：每个 16×16 tile 分配一个 thread block，块内显式共享项计算阶段由线程协同计算 X/Y 轴共享项存入 shared memory，同步后所有线程复用共享项并行执行组合阶段；(2) GPU-implemented MLP-based OIT——用 cuBLAS [37] 以 GEMM 形式实现 MLP 推理（推理前把视角折入 bias），替换 Gsplat [51] 的 Radix sort [31] kernel。
  - 实验比较：(1) axis-shared GPU kernel vs 原始 Gsplat 实现，按等效面积预算归一化比较延迟，几何均值加速 22%（远低于专用加速器的近 60%）；(2) MLP-OIT GPU 实现（cuBLAS）vs Gsplat Radix sort baseline，几何均值延迟为 baseline 的 1.59×（更慢）。结论：光栅化是非 GEMM 负载、Tensor Core 用不上且 GPU 固定 FMA 流水与均衡乘加比不适应其算术结构；MLP 推理算术强度低（1 深度参数仅 6 MAC vs 光栅化每 GS 256×6 MAC，约 30 倍差）、GPU 上 memory-bound，因此两项优化在通用 GPU 收益有限，论证专用可重构 PE 阵列 + interleaved pipeline 的必要性。
- 后端平台是什么，配置是什么。
  - NVIDIA RTX 3090 桌面 GPU（8nm、628mm²、350W、936 GB/s、10496 CUDA cores）；软件栈：Gsplat 渲染库（CUDA kernel 与 Radix sort）、cuBLAS；对照 edge 平台为 Jetson Orin Nano（8nm、200mm²、~15W、68.2 GB/s、1024 CUDA cores）。
- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：Gsplat（开源 3DGS 渲染库，https://github.com/nerfstudio-project/gsplat）。修改：在 Gsplat CUDA rasterization/排序 kernel 基础上新增 axis-shared rasterization kernel（每 tile 一个 thread block、shared memory 存共享项）与 cuBLAS 版 MLP-OIT kernel；baseline 为 Gsplat 原始实现（含 Radix sort）。论文代码仓库 https://github.com/WangZhican/ISCA26_3DGS_Acc 含 MLP-based_OIT 的 CUDA 实现（9 commits）。评估脚本细节论文未明确说明。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：算法/GPU 代码开源于 https://github.com/WangZhican/ISCA26_3DGS_Acc（MLP-based_OIT 目录）；基础库 gsplat 开源。评估原理与全过程（以 GPU MLP-OIT vs Radix sort 为例）：输入为预训练 3DGS 场景与相机位姿 → Gsplat 管线在 GPU 上做 projection → baseline 路径按 tile 对深度做 Radix sort 得有序 Gaussian 列表 → 逐 tile 排序 α-blending；OIT 路径把 MLP 推理实现为 cuBLAS GEMM（深度向量 × 权重 → 指数激活）输出 F(d_i)，跳过排序直接按 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i 光栅化 → 输出逐场景渲染延迟（ms），比较两者几何均值（1.59×）。axis-shared kernel 例子：每个 16×16 tile 启动 256-thread block → 阶段 1 前 16 线程算 X 轴共享项、另 16 线程算 Y 轴共享项写入 shared memory → __syncthreads() 同步 → 阶段 2 全部 256 线程从 shared memory 读所属行/列共享项合成指数并乘 opacity → 输出该 tile 的 α/颜色，延迟相对原 Gsplat kernel 按等效面积归一化比较。
