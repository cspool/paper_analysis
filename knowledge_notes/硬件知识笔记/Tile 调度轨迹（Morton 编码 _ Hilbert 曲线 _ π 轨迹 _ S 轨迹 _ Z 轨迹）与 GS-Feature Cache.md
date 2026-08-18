## Tile 调度轨迹（Morton 编码 / Hilbert 曲线 / π 轨迹 / S 轨迹 / Z 轨迹）与 GS-Feature Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tile 调度轨迹指加速器处理屏幕上各 16×16 tile 的遍历顺序（tile schedule），决定 GS-feature cache 的命中率与 off-chip 访存。本论文对比五种轨迹：(1) baseline 行序（row-by-row，S 轨迹的反向变体）——只利用水平局部性，不能复用竖直对齐 tile（如 2×1、3×1）的 GS；(2) S 轨迹——每行末尾反向，微改进；(3) Z 轨迹——按 Morton 编码（Z-order，把 y/x 位交织，如 y1y0 与 x1x0 交织成 y1x1y0x0）递增调度，保 2D 局部性但有对角线跳变；(4) π 轨迹——在 Z 轨迹基础上引入 Gray code 连续性（对应 1891 年 Hilbert 曲线，具有层级局部性，蓝色箭头示 2×2 tile 层次复用），连续性好；(5) 广义 π 轨迹——仅在各 8×8 tile block 内部用 π 轨迹，block 间用 S 轨迹；tile 数不能被 8 整除时剩余 tile 用行序 S 轨迹。配套 GS-feature cache：每 cache line 用 28-bit GS ID 打 tag + 4 bit 记录该 GS 相交的 tile 数（32-bit 对齐），替换时优先淘汰低重要性 GS，利用 GS 跨 tile 的空间局部性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（本论文 V-C 章 + Fig.11）：坐标生成器 CG 按轨迹生成 tile 地址 → 光栅化某 tile 前查 cache：命中则复用 GS 特征（μ、圆锥参数、颜色、opacity），未命中从 DRAM 载入该 GS 的 9 参数并更新 cache（28-bit GS ID + 4-bit tile 计数做替换决策）→ 处理完当前 tile 按轨迹跳到下一 tile，利用轨迹保的 2D 局部性提高相邻 tile 的共享 GS 命中率。轨迹编码例子：Morton 把 (x,y) 位交织得到一维码（(x=1,y=2)→bits y1x1y0x0）；Hilbert/π 轨迹在 2×2 与 4×4 粒度上都有空间连续的子块（自相似层级局部性），使 8×8 block 内轨迹连续、block 间 S 轨迹。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CG 硬件单元内置轨迹逻辑（π 轨迹在 8×8 block 内、block 间 S 轨迹），cache 控制器按 28-bit ID 查表。效果（本论文 VI-C 章 Fig.15）：平均 cache 命中率 baseline 43% → Z 轨迹 55% → π 轨迹 62%；off-chip 访问能耗相对"无 cache"配置：π 轨迹 2.56× 节省，相对 baseline 轨迹 1.51×、相对 Z 轨迹 1.23×。Morton 编码与 Hilbert 曲线是空间填充曲线（space-filling curve）经典方法，广泛用于空间索引/纹理遍历/GPU tile 调度；本论文的 π 轨迹是其面向 GS 特征复用的硬件实现变体。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
