## CORDIC（Coordinate Rotation Digital Computer，坐标旋转数字计算机）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CORDIC 是 1959 年 Volder 提出的迭代算法：基于所选坐标系（圆/线性/双曲）与旋转/向量模式，用简单的移位+加法迭代逼近三角函数、双曲函数、对数、指数、开方等。优点：硬件开销低（只需 add/shift）。缺点：精度靠迭代次数，迭代越多延迟越大；支持输入范围有限，某些版本需旋转前后额外处理（LoRA 论文 II-B 背景）。
- 在 LoRA 中的作用：作为非线性函数硬件实现的基线之一——huicore [10] 是 CORDIC 通用复杂函数加速器（28nm、153k µm²、≥20 cycle/次迭代数，支持 2GHz）；XCore 与其对比：更低延迟（4/7 cycle）与硬件开销（40nm、71.7–78.4k µm²），支持可编程定点格式，且能单步逼近复合函数（CORDIC 级联方式做不到）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（旋转模式算 sin/cos）：初始化角度累加器与 (x,y)，每轮 i 做 x'=x−σ_i·2^(−i)·y、y'=y+σ_i·2^(−i)·x、z'=z−σ_i·arctan(2^(−i))，σ_i 按 z 的符号选（±1）；迭代 n 轮后 (x,y) 收敛到 (cos(z0), sin(z0))·K（K 为增益常数）。精度↑⇔迭代数↑⇔延迟↑。
- LoRA 对比点：huicore（CORDIC 级联）逼近复合函数需串联多次旋转/向量操作，而 XCore 把复合函数（tanh(x)+1、sin(x)+cos(x)、ln(sin(x))）作为一个多项式直接逼近，一个 XCore 节点完成；且 CORDIC 只支持有限输入范围，XCore 支持用户定义任意范围。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件为移位器+加法器迭代阵列（可用展开流水线提高吞吐）；软件库/FPGA IP 核常用（Xilinx CORDIC IP）；嵌入式 DSP（STM32 CMSIS-DSP、FFT 多旋转 CORDIC）广泛使用。
- 使用场景：三角函数/双曲/对数/开方的低成本硬件实现；LoRA 论文把它归类为"迭代式（iterative-based）"三类非线性硬件方案之一（另两类：LUT 式、多项式式），与 Flex-SFU、PACE 等多项式式方案对比时作为精度参考（XCore 逼近精度与 CORDIC 目标相当甚至更好：arcsinh [−19.4,19.4] XCore-A AAE=1.67e−5 vs CORDIC [10] 8.91e−6，sin [±π/2] XCore 与 [10] 同量级）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
