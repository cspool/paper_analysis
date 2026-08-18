## 体积渲染（Volume Rendering，体绘制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 体积渲染是沿视线方向对采样点按光学模型（吸收/发射）累积颜色与透射率的渲染技术（经典参考 Max, "Optical models for direct volume rendering" 2002）。NeRF/Instant-NGP 等 MLP/grid-based 神经渲染用其把每点预测的密度 σ 与颜色 c 合成像素颜色：C(r) = Σ_{i} T_i (1−exp(−σ_i δ_i)) c_i，T_i = exp(−Σ_{j<i} σ_j δ_j)，δ_i 为采样间距。NeRArch-Sim 论文把 Blending 阶段定义为"聚合场景属性产生最终像素颜色"，其中 MLP-/grid-based 管线用体积渲染，primitive-based（3DGS）用排序后 alpha 混合。3DGS 论文说明其 α-blending 即 NeRF 式体积模型（复用经典图形学 alpha 术语）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 体积渲染在 NeRArch-Sim 中是 Blending 分类学阶段的核心算子，硬件库有 3 个 VRU（Volume Rendering Unit）变体（分别对应 ICARUS/CICERO/GSCore 的实现），表 VI 给出 ICARUS 的 VRU 延迟 192/192 cycle、面积 4755/4960 µm²、功率 1917/2110 µW（NeRArch-Sim vs 全 ASIC flow）。伪代码：
```
C, T = 0, 1
for i in sorted_samples:            # 沿射线从前到后
    alpha_i = 1 - exp(-sigma_i * delta_i)
    C += T * alpha_i * c_i          # 累加颜色
    T *= 1 - alpha_i                # 更新透射率
    if T < eps: break               # 提前终止（early termination）
```
- 内存侧（NeRArch-Sim 表 VIII）：ICARUS/NeuRex 在 Field Sampler 阶段做 ray marching，采样坐标从 DRAM 经 Input FIFO 流入；ICARUS 每帧 1.4GB 采样坐标、权重 1.9MB 只加载一次复用约 10 万次、激活值片上 ping-pong。GSCore 不做 ray marching，每 tile 从 DRAM 流式读 79MB 高斯特征，排序与光栅化中间量全片上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：Nerfstudio 的 nerfstudio field 组件逐点预测后做体积渲染积分；硬件实现：专用 VRU（如 ICARUS 的 volume rendering unit、CICERO 的 NRU、GSCore 的 VRU），NeRArch-Sim 硬件库含 VRU_v1/v2/v3 变体并支持配置累计方式（accumulation_type）。算子级优化上，"per-ray early termination（基于累计不透明度）"是体积渲染特有的 element-level skip 优化（NeRArch-Sim 调度优化库）；相关硬件研究如 VR-Pipe（HPCA 2025）专门流水化硬件图形管线做体积渲染。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
