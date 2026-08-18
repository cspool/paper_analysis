## Projection with Jacobian（PJ，投影与雅可比模块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PJ（Projection with Jacobian）模块是 ECHO 计算引擎中专用于加速鱼眼相机投影 π 及其导数（Jacobian）计算的硬件单元，服务于 LM tracking 的 Gauss-Newton 优化（每次迭代都要对每个 2D-3D 对应计算投影残差与 Jacobian，是第二大瓶颈 24.09ms 的主要构成）。由于投影与 Jacobian 共享变换后 3D 点 x^c 与中间量 ρ=√(x²+y²)、θ=arctan(ρ/z)、多项式畸变项 d(θ)，ECHO 设计统一硬件模块单遍计算两者，复用数据通路与共享表达式，最小化冗余计算与硬件成本。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 运转流程（针对 KB 鱼眼模型）：输入 3D 点 x^c=[x,y,z]^T → ①计算 ρ=√(x²+y²)（平方根用 fast inverse square root 方法近似，免除法）；②θ=arctan(ρ/z) 用 sign-aware 多项式近似（域归一化 + 低阶多项式）；③畸变 d(θ)=θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹ 用 Horner 法（减少乘法次数）；④投影 π(x^c)=[fx·d(θ)cosφ+cx, fy·d(θ)sinφ+cy]^T 中 cosφ=x/ρ、sinφ=y/ρ（消除显式 arctan2 与 trig，division-free）；⑤Jacobian 路径：共享中间量由流水线一次生成，仅对共享量做增量链式求导（∂θ/∂x、∂θ/∂y、∂θ/∂z、畸变项导数）组装雅可比，避免重复计算。PJ 全程 FP32（投影/Jacobian 对量化敏感），配合脉动阵列输出的低精度坐标变换结果使用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Verilog RTL（带中间量复用流水线）+ 综合（45nm/1GHz）；用多项式近似表/系数、Horner 求值器、reciprocal-square-root 近似器。使用：CPU 在 Gauss-Newton 迭代中调用加速器（Step 6，Fig.11a）请求批量投影+Jacobian，加速器单遍算完返回，CPU 完成法方程求解与流形更新。算术优化目标：把 KB 模型的 trig/9 阶多项式高开销计算变为 division-free 低延迟流水，T_P^MI 平均降 3.4×（配合点过滤与低精度）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
