## Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现有两项：(1) Axis-shared rasterization（轴共享光栅化）——把每个 tile（16×16 像素）内 α-computation 的指数项分解为 X 轴二次项、Y 轴二次项、交叉项三部分，沿 X/Y 轴预计算共享项并广播给各 PE 复用，消除同行/同列像素间的冗余计算，使 α-computation 的 MAC 从每像素 8 MUL+4 ADD 降至摊销后 2.31 MUL+2.13 ADD/PE，完整 rasterization（含 α-blending）MAC 减少约 38%，且保持像素级全并行（不引入 GBU/FastSplat 的行序差分依赖）。(2) MLP-based order-independent transmittance（MLP 顺序无关透射率，OIT）——用轻量 2 层 10 参数 MLP 直接预测每个 Gaussian 的衰减因子 F(d_i) 替代显式深度排序：输入为 Gaussian 深度 d_i 与归一化视角方向 (x,y,z)（推理时视角恒定、其贡献折入 MLP bias），第一层 Leaky ReLU(系数 1/8)，输出层指数函数（复用光栅化 EXP 单元），推理仅 6 MAC；渲染公式由按深度序的 C=ΣT_iα_ic_i 改为顺序无关的 C=ΣF(d_i)α_ic_i / ΣF(d_i)α_i。
  - 实验比较：算法精度对比 (i) 原始排序 3DGS baseline，(ii) sort-free weight-sum 渲染 [18]（最优变体 LC-WSR），及 OIT 仅深度输入（OIT+d）与含视角输入（OIT+d+view）的消融，指标 PSNR/SSIM/LPIPS；动态场景（Neu3D + 4DGS）比较 baseline vs OIT 的 PSNR/SSIM。硬件消融 BS/BS+AR/BS+AR+OIT/BS+AR+OIT+IP 四变体比较吞吐（1×/1.37×/2.16×/2.27×，几何均值）。

- 硬件平台是什么，配置是什么。
  - 算法训练平台：NVIDIA RTX 3090 GPU。先按 [22] 训练 7000 epochs 得到 checkpoint（原始排序算法预训练模型），再以其初始化额外训练 10000 epochs：MLP 学习率 0.005，Gaussian 学习率按 0.01 缩放；关闭 Gaussian cloning/splitting 保证训练稳定；每场景约 30 分钟。动态场景每 30 帧更新一次 10 参数 MLP（300 帧序列共 10 组参数）。GPU 推理评估平台：Jetson Orin Nano edge GPU（8nm、200mm²、~15W、68.2 GB/s、1024 CUDA cores）与 RTX 3090（8nm、628mm²、350W、936 GB/s、10496 CUDA cores）。加速器硬件：TSMC 28nm、1 GHz、3.85mm²、1.64W、96KB on-chip SRAM、DDR5-4800 38.4 GB/s、256 PEs（16×16 可重构阵列）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3D Gaussian Splatting 场景表示（每个 Gaussian 59 参数：3 均值 + 3 scale + 4 rotation + 1 opacity + 48 SH 颜色系数），外加 2 层 10 参数 MLP（Leaky ReLU(1/8) + 指数输出）预测透射率。动态场景用 4DGS [50] 建模（per-frame 渲染与静态 3DGS 相同）。
  - 数据集与 bench：静态用 MipNeRF-360 [1] 的 7 个真实场景 garden、bicycle、stump、bonsai、counter、kitchen、room（validation 集），指标 PSNR/SSIM/LPIPS；动态用 Neu3D 数据集 [27]（2704×2028 高分辨率、300 帧/10 秒，Cook Spinach、Cut Beef、Flame Steak），指标 PSNR/SSIM；GPU profiling 用 MipNeRF-360（7k 训练 checkpoint）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/WangZhican/ISCA26_3DGS_Acc（含 MLP-based_OIT 目录，CUDA 实现，9 commits，1 contributor）；算法与 GPU 实现基于开源库 gsplat [51]（https://github.com/nerfstudio-project/gsplat）。sort-free weight-sum 基线见 arXiv 2410.18931。
  - 算法 pipeline 解释（以 MipNeRF-360 一个场景为例）：(1) 预训练：用原始 3DGS（含排序）训练 7000 epochs 得 Gaussian checkpoint；(2) OIT 训练：加载 checkpoint，对每个相机位姿投影 Gaussian 得 2D 均值 μ、2D 协方差 Σ 与深度 d_i，构造样本（d_i, 视角(x,y,z)）→ 前向：MLP 输出 F(d_i)，按 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i 渲染，与 GT 图像计算 loss（原 3DGS 设置）→ 反向：MLP 大步长（lr=0.005）快速收敛，Gaussian 小步长（lr×0.01）缓慢精修；(3) OIT 推理：给定相机位姿，视角 (x,y,z) 对所有 Gaussian 相同，预计算并融合进 bias（b_i=b'_i+c_i·view 贡献），对每 tile 内 Gaussian 深度 d_i 做 6 MAC 前向得 F(d_i)，再按上述公式 α-blending；(4) 全程无显式排序。张量计算例子：单 Gaussian α 计算 α=o·exp(-½(p-μ)ᵀΣ⁻¹(p-μ))，指数分解为 -(½a(x-μx)² + ½b(y-μy)² + c(x-μx)(y-μy))；axis-shared 版本把 x 项、x² 项（X-PE 线）与 y 项、y² 项（Y-PE 线）各算一次广播到 16×16 阵列，每像素 PE 仅需 1 乘法 (x·y) + 2 加法合成指数，再乘 opacity o 过 EXP。
