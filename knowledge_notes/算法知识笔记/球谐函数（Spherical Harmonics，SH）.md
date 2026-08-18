## 球谐函数（Spherical Harmonics，SH）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spherical Harmonics 是球面上的正交基函数族，图形学里用它紧凑编码方向相关量（如随视角变化的颜色/光照）。NeRArch-Sim 论文指出 Field Computation 阶段（"基于编码特征计算采样物体的场景属性（颜色/密度）"）"typically using MLPs or Spherical Harmonics"——3DGS 等 primitive-based 管线用 SH 系数存高斯的方向相关颜色。3DGS 论文参数化中：位置 3 + 旋转四元数 4 + 不透明度 1 + SH 系数 16×3=48 参数（三阶 SH），即每个 3D 高斯的外观由 SH 系数决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 3DGS 渲染中，像素颜色按视角方向 d 从 SH 基重构：`color(d) = Σ_{l=0}^{L-1} Σ_{m=-l}^{l} c_{lm} · Y_{lm}(d)`，其中 Y_{lm} 为 SH 基、c_{lm} 为学习到的系数（3DGS 默认 3 阶 16 组 × RGB 3 通道 = 48 参数）。NeRArch-Sim 中此计算是 Field Computation/Blending 阶段的颜色相关算子，其分类学参数含 encoding_type 等；硬件侧 GSCore/GS Processor 的光栅化/混合路径需访问高斯外观（GauTracer 论文把 SH/颜色存为纹理数据经 Gaussian ID 访问）。NeRArch-Sim 表 XII 显示 GS Processor 有 Feature Computing Unit（计算高斯特征），即此类方向相关颜色计算硬件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：3DGS（graphdeco-inria/gaussian-splatting）用 SH 系数 + CUDA 光栅化 kernel 计算视角相关颜色；Nerfstudio 的 splatfacto 模型即 3DGS 类（NeRArch-Sim 支持其 trace）。硬件：NeRArch-Sim 硬件库 Field Comp/Blending 类含 MLP 引擎与加法树等，可建模 SH 求值；SRender 的 interpolation units 等处理方向相关插值。SH 阶数（degree）是可配置精度参数，与"low bit 优化"（SRender 的 sensitivity-aware 动态精度）相关。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 加速器补充视角（ISCA'26，SH 作为 GS 特征存储与加速器访存）：本论文把投影后 RGB（SH 求值结果）作为 9 参数 GS 特征之一存入 GS-feature cache（每 cache line 28-bit GS ID tag + 4-bit 记录该 GS 相交 tile 数，32-bit 对齐），光栅化时经广播寄存器广播 16 次供同线 PE 的 α-blending 使用；59 参数中 SH 占 48（16×3），是参数占比最大的部分。SH 求值在 GPU 侧 projection 阶段完成，加速器不重复计算——体现"投影留 GPU、光栅化/混合上专用阵列"的分工。
