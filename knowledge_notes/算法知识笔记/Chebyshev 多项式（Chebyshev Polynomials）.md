## Chebyshev 多项式（Chebyshev Polynomials）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Chebyshev 多项式（第一类，T_n(x)）是 [-1,1] 上一族正交多项式，由递推 T_0(x)=1、T_1(x)=x、T_n(x)=2x·T_(n−1)(x)−T_(n−2)(x) 定义，等价于 T_n(x)=cos(n·arccos(x))。相比 Taylor 级数：在整个区间上极小化最大误差（minimax 性质）、收敛更快、达到同样 sup-norm 误差所需次数更低、避免 Runge 振荡、数值条件更好（B.-Chebyshev-Polynomials 论文依据）。正交性使最小二乘系数求解稳定。用于函数逼近时，f(x)≈Σc_i·T_i(x)，系数 c_i 由采样点最小二乘解得。
- 在 LoRA（ISCA'26）中的作用：作为分段逼近的基函数——对用户给定输入范围 [a,b] 的非线性函数，先做区间变换 x'=(2x−(b+a))/(b−a) 映射到 [-1,1]（Chebyshev 自然定义域，正交性在 [-1,1] 成立），构造 Chebyshev 矩阵 V（第 j 列=T_j(x_i')），用最小二乘 min Σ(f(x_i')−Σc_j·T_j(x_i'))² 解系数 c，再用递推把 Σc_i·T_i(x') 展开回标准多项式 Σp_i·x'^i、逆区间变换得到 Σp_i·x^i，供 XCore 硬件以 LNS 高效执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（单段 6 项逼近，n=5）：
  1. 采样 m 点 {x_i, f(x_i)}（LoRA 用 curvature-based：均匀采样后数值微分估曲率、高曲率区插点）；
  2. 区间变换 x_i'=(2x_i−(b+a))/(b−a)∈[−1,1]；
  3. 构造 V=[T_0(x_i')...T_5(x_i')]（m×6 矩阵），解最小二乘 V·[c_0..c_5]^T≈[f(x_i')]^T；
  4. 用递推把 Σc_i·T_i(x') 展开为 Σp_i'·x'^i，逆变换得 Σp_i·x^i；
  5. 利用奇偶性：奇函数 V 偶数列置 0（如 [0,T1,0,T3,0,T5]），同 6 项可支持 9 次多项式（x^{1,3,5,7,9}），更高次数→更高精度与数值稳定性（cos 考虑奇偶性使 MAE 降 148×）。
- 示例（cos(x) 逼近，[−π/2,π/2]）：变换到 [-1,1] 采样 → 因 cos 是偶函数只用偶阶项（T_0,T_2,T_4...）→ 最小二乘得系数 → 硬件 XCore 在 LNS 中算 Σp_i·x^i（每项 c_i·x^(k_i)=2^(log2(c_i)+k_i·log2(x))）。结果：XCore-A/C 在 [−π/2,π/2] 上 AAE≈1.03e−6、MSE≈2.18e−12，与软件双精度逼近同数量级。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 PiecewiseChebFitter（Python，开源于 COFFA 仓库 LoRA-ISCA-AE 分支）实现完整流程，`python3 run_func.py <gelu|sigmoid|softplus|swish|tanh>` 对每函数（config 文件夹配参数）输出 breakpoints/系数/次数到 fig/ 与 result/；遗传算法负责各段次数分配、三种分段策略确定 breakpoints（见分段逼近条目）。
- 使用场景：误差容忍的硬件非线性函数实现——AI 激活函数（Sigmoid/Tanh/GELU/Swish/Softplus/Swiglu）、DSP（sin、sqrt、arcsinh）、LLM（Softmax、GELU）；与 PACE（同为 Chebyshev 分段逼近，3-term）对比，LoRA 利用奇偶性+equal-error 分段精度更高（EfficientNet/MobileNetV3 误差 0.002%/0.006% < PACE 最小 0.01%）。其它用法：图神经网络（Chebyshev 卷积近似，NeurIPS 2022）、DS-TPU 用 Chebyshev 多项式建模非线性节点交互（另见 vault 中 DS-ISA、Oracle-MoE 论文对 Chebyshev 不等式/逼近的引用）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
