## Volumetric Alpha Blending（体积 alpha 混合 / 体积泼溅合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 体积 alpha 混合是 Gaussian splatting 的最终合成步骤：把沿视线/射线方向排好序的高斯颜色按"不透明度 alpha + 累积透射率 T"从前到后（front-to-back）累加，得到像素最终颜色。标准公式（式 3）：C = Σ_{i=1..N} T_i α_i c_i，T_i = Π_{j<i} (1-α_j)。T_i 是到达第 i 个高斯前的剩余透射率（前面高斯的遮挡积累），α_i c_i 是第 i 个高斯对光线的贡献，本质是离散化的体渲染（ray marching）积分。GauTracer 论文用该混合更新射线颜色与透射率，并在 closest-hit shader 中做 front-to-back 混合、以透射率低于阈值作为 ray-gen 循环的提前终止条件。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：ray-gen 循环（GauTracer Alg. 1/3）：每条射线迭代 traceRayEXT，每次收集 K 个命中高斯，closest-hit shader 把 K 个命中按深度排序后 front-to-back 混合：
  ```
  # front-to-back（baseline，式 3）
  T, C = 1.0, 0
  for entry in ClosestHit[rayID][0..N_hit]:
      alpha = entry.alpha; color = GaussParam[entry.GID].color
      C += T * alpha * color
      T *= 1 - alpha
  ```
  GauTracer 的 AGHU 输出 far-to-near 序列（max-heap 弹出根=最远），为避免再排序采用 back-to-front 混合（式 7）：C^p_{i+1} = α_i c_i + (1-α_i) C^p_i，C^p_0=0，再以射线透射率缩放合入像素：C += T·C^p；透射率按 round 级更新 T *= Π(1-α_i)，实现 round 级（而非逐高斯）提前终止。该近似只影响透射率权重低于 0.001 的高斯，PSNR 几乎无损（33.40→33.32 dB）。
- 术语一般如何实现？如何使用？：软件上在 closest-hit shader（Vulkan）或 OptiX 内核（3DGRT）中循环混合；光栅化侧在 fragment shader 或专用光栅化 kernel 中按 tile 深度排序混合。硬件上，GauTracer 把"排序+混合"拆给 AGHU（排序）与 closest-hit shader（混合）；相关硬件工作（Gaussian Blending Unit [30]、GScore [29]）也在硬件中做高斯混合。混合正确性依赖命中排序，这正是 OIT（order-independent transparency）类近似（Local-GS [71]）在 ray tracing 中不可行的原因——ray tracing 需要精确命中距离决定二次光线生成位置。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 加速器补充视角（ISCA'26，OIT 用 MLP 预测透射率替代排序混合）：本论文重新审视混合公式 C=ΣT_iα_ic_i、T_i=Π_{j<i}(1-α_j)，指出排序的唯一目的是算正确的累积透射率 T_i（深度递增则 T 递减的衰减因子），并把 α-blending 与图像合成（"over" 算子，3DGS 前到后混合 == 图像合成后到前）类比，引入顺序无关透射率（OIT）渲染式 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i（式 5），F(d_i) 由 2 层 10 参数 MLP（输入深度+视角方向，指数输出激活，推理 6 MAC）预测；质量 PSNR 26.90 vs 27.21、SSIM 0.8263 vs 0.8309、LPIPS 0.1739 vs 0.2017（略优），且优于 handcrafted depth-function 的 weight-sum[18]（25.43）。硬件上 α-blending 在可重构 PE 中实现：M-3 乘 F(d_i)α_i、A-3 累加分母，M-4-{1~3} 乘 RGB、A-4-{1~3} 累加 RGB 分子，除法阵列归一化；相对 32 并行 bitonic 排序网络 21.1~32.4× 加速。与 GauTracer 的结论（OIT 类近似在 ray tracing 中因需精确命中距离而不可行）形成对照：本论文证明 tile 光栅化场景下 OIT 可行且质量损失小。
