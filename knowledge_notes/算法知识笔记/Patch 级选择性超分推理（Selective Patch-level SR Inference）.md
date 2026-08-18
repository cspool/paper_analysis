## Patch 级选择性超分推理（Selective Patch-level SR Inference）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把视频帧划分为等尺寸非重叠 patch，逐 patch 决定上采样策略（SR 推理 / 跨帧复用 / 像素插值），只对信息量最大的区域执行昂贵的 SR 模型推理，从而在满足实时预算（30FPS/33ms）的前提下把 SR 计算量降到最低。核心动机是 SR 增益在空间上高度选择性：实测 44.1% 的 patch 上 SR 相对插值无增益甚至负增益（Fig.4），背景/平坦区域 SR 与插值视觉等价；patch 级高频残差占比与 PSNR 增益正相关、大 MV 区域增益小甚至为负（Fig.5/7）。SLICE 默认参数：patch 16×16、推理面积比 k=35%（按 score TopK 选出）、intra 帧做全帧 SR、inter 帧三路选择。效果：2.72× 帧率提升、62.57% 能量节省、PSNR 仅降 0.35dB（对比无复用变体 SLICE-noreuse 的 0.78dB，说明复用的质量贡献）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 伪代码：
```
for each frame f in video:
    if INTRACODED(f):            # 每个 GOP 开头的 I 帧，占比小
        F^SR ← FULLFRAMEInference(f)
    else:                        # 占绝大多数的 inter 帧
        (M^reuse, M^SR) ← PATCHANALYSIS(f)        # codec 元数据驱动，全 GPU
        P^HR ← PATCHWISEUPSCALE(f, M^SR, M^reuse) # reuse 直拷 / SR 推理 / 插值
        F^SR ← MERGEPATCHES(P^HR)
```
例子（270p 帧，P=16 → 30×17 patch）：MV 与像素域残差均为 0 的静态 patch → 复用；score=0.9·hf_ratio+0.1·(1−clip(mv/10)) 前 35% 的 patch → EDSR 推理；其余 → bicubic 插值。patch 大小权衡（Fig.16）：32×32 接受野大、单 patch 质量高，但 patch 变大导致复用率下降、复用收益变小；16×16 只牺牲少量接受野而显著提升复用机会，综合质量最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 PyTorch GPU 管线：unfold 把 patch 网格转为紧凑张量，按 M^SR gather 出需推理的 patch 组成 batch 做一次或少数几次 EDSR(FP16) forward；复用 patch 从 GPU 常驻 HR cache 直拷；其余 patch GPU 插值；按行分带（row-wise banded）合并写 framebuffer。硬件平台为 NVIDIA Jetson AGX Orin，能量用 Tegrastats 测。与模型级高效 SR（APE 的 patch 级 early exit、轻量/量化 SR 模型）正交，可叠加。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
