## 随机傅里叶特征（Random Fourier Features，RFF / 位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Random Fourier Features（Rahimi & Recht 2007）把输入映射到随机傅里叶特征空间以近似核函数；NeRF（Mildenhall 2020）采用其思想作为位置编码：对 3D 采样点 x 按不同频率的正弦/余弦（高频分量）升维，使 MLP 能表示高频细节。NeRArch-Sim 论文明确 MLP-based 方法（NeRF 类）"sample 3D points along camera rays, encode them via Random Fourier Features (RFF)，再 query MLP 预测密度与颜色"，并指出 RFF 与 hash encoding 是 MLP-/grid-based 管线的代表 Encoding 方案（primitive-based 如 3DGS 跳过此阶段，原语自带场景属性）。注意本仓库中"Random Feature Attention（RFA，Mamba 线性注意力用 RFF 近似 softmax 核）"是同一数学工具在注意力上的不同应用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- RFF 在 NeRArch-Sim 中属于 Encoding 分类学阶段，与 Position encoding unit（ICARUS 用）等价。伪代码：
```
def rff(x):                            # x ∈ R^3
    return [sin(2^0 π B x), cos(2^0 π B x),
            sin(2^1 π B x), cos(2^1 π B x), ...]   # 多频段
# 之后: MLP(concat(x, rff(x))) → (σ, c)
```
- 硬件侧（NeRArch-Sim 表 VI）：ICARUS 的 Pos Encoding Unit（PEU）延迟 130/130 cycle、面积 6714/5200 µm²、功率 305/330 µW（NeRArch-Sim vs 全 ASIC flow）；图 12 显示 ICARUS 中 PEU 利用率持续高企、MLP 是主导瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：Nerfstudio 的 vanilla-nerf 模型（`ns-eval` 可跑）与 Nerfstudio 的 `PositionEncoding`；实现通常用预计算正弦/余弦表或即时三角函数。硬件：ICARUS 的 Position Encoding Unit、NeRArch-Sim Encoding 硬件库中的 Position encoding unit，支持 CORDIC/分段线性等 exp/三角实现选择（表 IV：Implementation = CORDIC, piecewise linear）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
