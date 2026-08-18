## Order-Independent Transmittance（OIT，顺序无关透射率）/ MLP-based OIT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OIT 在本论文中指一种替代显式深度排序的算法：直接计算/预测每个 Gaussian 的透射率（衰减因子）F(d_i)，使最终颜色可按与顺序无关的方式合成，从而省掉排序环节。动机链：3DGS 的 α-blending 本质是图像合成（"over" 算子），而 "over" 非交换 → 传统需要按深度排序；计算机图形学为绕开排序发展了 order-independent transparency（OIT）技术（A-buffer、stochastic transparency、weighted OIT 等），其中 weighted OIT 用深度单调递减权重 F(d_i) 做加权合成 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i，质量损失可忽略。本论文把这一思想引入 3DGS：观察到排序的唯一目的是算正确的累积透射率 T_i（随深度递减的衰减因子），于是提出直接用轻量 MLP 预测 F(d_i)。关键扩展：3DGS 是视角相关渲染，同一深度在不同视角下贡献不同，因此输入为 (深度 d_i, 归一化视角方向 (x,y,z))——视角信息在推理时对同一相机位姿恒定、可折入 MLP bias；MLP 为 2 层 10 参数（Leaky ReLU(1/8) + 指数输出激活），推理仅 6 MAC，训练约 30 分钟/场景。最终渲染式 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i（式 5），与 baseline 排序 3DGS 相比 PSNR 仅降 0.3（26.90 vs 27.21）、SSIM 几乎不变、LPIPS 略优，且优于 handcrafted depth-function 的 sort-free weight-sum 渲染[18]（25.43 PSNR，LC-WSR 最优变体）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
训练与推理 pipeline（本论文 IV-B 章 + Fig.7）：
```
# 训练（RTX 3090，每场景约 30 min）
1) 用原始排序 3DGS 预训练 7000 epochs 得 checkpoint（初始化，加速收敛）
2) 对每个相机位姿 projection 得深度 d_i，构造样本 (d_i, 视角(x,y,z))
3) 前向：MLP → F(d_i)；按 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i 渲染，与 GT 算 loss（原 3DGS 设置）
4) 反向：MLP lr=0.005（大步长快速收敛），Gaussian lr×0.01（小步长慢精修）；
   关闭 cloning/splitting 保持 Gaussian 数恒定、训练稳定
# 推理（加速器）
1) 相机位姿给定 → 视角 (x,y,z) 恒定，预计算并融合进 bias（b_i=b'_i+c_i·view）
2) 对每 tile 内 Gaussian 深度 d_i 做 6 MAC 前向得 F(d_i)（可重构 PE 阵列 MLP 模式，
   每周期处理 256 个深度值，写回 depth buffer）
3) α-blending 用广播寄存器中的 F(d_i) 累加分子分母 → 除法阵列归一化输出像素
```
动态场景扩展：Neu3D + 4DGS baseline，每 30 帧更新一次 10 参数 MLP（300 帧序列共 10 组），PSNR 仅降 0.45。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 实现（本论文 VI-E 章）：用 cuBLAS 以 GEMM 形式做 MLP 推理替换 Gsplat 的 Radix sort，因 MLP 算术强度低（1 深度参数仅 6 MAC vs 光栅化每 GS 256×6 MAC，~30 倍差）而 memory-bound，几何均值延迟为 baseline 排序的 1.59×（更慢）——说明 GPU 上 MLP-OIT 不划算，需专用硬件。加速器实现：复用光栅化 PE 阵列的 MAC/EXP 单元做 MLP 推理（可重构仅 +5% 面积/+6% 功耗），相对 32 并行 bitonic 排序网络 21.1~32.4× 加速，配合 fine-grained interleaved pipeline 隐藏 memory-bound 延迟（见硬件架构层对应条目）。对比：weight-sum[18] 是手工深度函数（无视角信息），硬件上需额外除法单元（+0.363mm²/+341mW），而本 MLP 复用现有除法阵列，仅 +0.147mm²/+88mW。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
