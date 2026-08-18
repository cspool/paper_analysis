## LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Chebyshev 分段逼近算法（Python，PiecewiseChebFitter），为任意用户给定输入范围 [a,b] 的非线性函数生成 {breakpoints, 每段多项式系数, 次数} 配置，供 XCore 硬件执行。算法流程（Fig.3）：①采样（curvature-based：先均匀采样，数值微分估曲率，高曲率区插入更多样本点）；②无分段整体逼近，考虑最大项数（决定允许的多项式次数）、函数奇偶性（代数性质：偶/奇函数只用偶/奇阶项，6 项下非对称函数支持 5 次 x^0~5、奇函数支持 9 次 x^1,3,5,7,9，同项数更高次数→更高精度与更好数值稳定性）、定点硬件约束 Q_(m,n)^max（不等式 |p_i||x^i|_max<|Q_(m,n)^max| 并入最小二乘求解防溢出）；③从 2 段起逐段递增到最大支持段数 N，每段：区间变换 x'=(2x-(b+a))/(b-a)→[-1,1]（Chebyshev 自然定义域）→ 构造 Chebyshev 矩阵 V（第 j 列=T_j(x_i')）→ 最小二乘 min Σ_i(f(x_i')-Σ_j c_j T_j(x_i'))² 解系数 → 用递推 T_0=1,T_1=x',T_n=2x'T_{n-1}-T_{n-2} 把 Σc_iT_i(x') 展开回标准多项式 Σp_i'x'^i 再逆变换得 Σp_ix^i；④用遗传算法（个体=各段次数分配 k_seg1..k_#seg，设 #gen=10、#pop=16）优化次数分配防高次过拟合，breakpoint 由三种分段策略确定（见下）；选平均 MSE 最小的个体/段数作为最优解，输出 breakpoints+系数+次数存入 XCore LUT（Fig.3 红色参数）。算法也可探索 >6 项多项式（由多个 XCore 计算）。
  - 三种分段策略（图 3/图 4）：(1) uniform——等宽分段；(2) curvature-based——按累计曲率 W_k=Σ_{i=1..k} κ(x_i)Δx 均分，找 breakpoints 使每段累计曲率 = W_m/N，高曲率区段更密；(3) equal-error——从 curvature 分段出发迭代优化，新 breakpoint x_e 在 [x_(s-1), x_(s+1)] 间用 Brent 法 [9] 解 MAE(x_(s-1),x)-MAE(x,x_(s+1))<ξ（ξ=容差，默认 1.5e-5）使左右段 MAE 接近，直到各段 MAE 方差低于阈值，使误差分布更均匀、接近最优精度。
  - 实验比较什么：(1) 算法级——遗传算法设置（#gen、#pop、ξ 容差）对逼近 runtime 与最终精度的影响：ξ 越小误差分布越均匀、越接近近优但 runtime 越大；迭代越多越好但有收益递减；穷举 6 段×6 项=6^6 次分配一周内不可行 → 遗传算法更合适。(2) 单元级精度——XCore-A/B/C 与 LoRA-SW（软件逼近结果）在 Sigmoid/Tanh/GELU/Swish/Softplus/arcsinh/Sin/sqrt（±8、±4、[0,63.75]、[±π/2] 等范围）上与先前工作 [4][10][25][32][76][79][80] 比 AAE/sq-AAE/MSE：XCore 逼近误差与软件 LoRA-SW 同数量级（sin 因 log 转换器精度 ~1e-5 上限例外）；奇偶性使 cos 的 MAE 降 148×；与 PACE 比（同为 Chebyshev 分段逼近）：XCore 充分利用代数性质+equal-error 更稳定，3-term XCore 在 EfficientNet/MobileNetV3 上误差 0.002%/0.006% < PACE 最小 0.01%。(3) 端到端精度——DCT（cos，量化步长 0.1/0.75/1.5，比 MSE/PSNR/压缩率，baseline=AMD Ryzen 9 7945HX，LoRA 与 PICACHU-4th 持平或更好，故后续 Taylor 阶数取 ≥4）、DNN（SE-ResNet/EfficientNet/MobileNetV3 在 ImageNet，baseline=RTX 4090 FP32 输出，LoRA 精度差 ≤±0.008%，部分反超 baseline）、LLM（GPT2-XL/Mistral-7B/Mistral-7B-v0.3/DeepSeek-7B 在 ARC-e/HellaSwag/CommonsenseQA/COPA，baseline=A100 FP32 输出，Q16,16 定点，LoRA 精度差约 ±1% 内且部分任务反超 baseline；HardSwish 无指数但含多项式也由 XCore 直接算）。
- 硬件平台是什么，配置是什么。
  - 算法级/单元级/端到端精度评估为软件：DNN 用 NVIDIA GeForce RTX 4090 GPU，LLM 用 NVIDIA A100 GPU，DCT baseline 用 AMD Ryzen 9 7945HX CPU（FP32）。硬件目标平台是 LoRA CGRA SoC（Chisel 建模、TSMC 40nm 综合、~475MHz）；XCore 为 40nm、~510/485MHz、6 段（XCore-C）/7 段（XCore-A/B）配置，FP32+可编程定点格式。
- 模型是什么。数据集和bench分别是什么。
  - 模型：DNN——SE-ResNet、EfficientNet、MobileNetV3（ImageNet 数据集，FP32 与 Q16,16 定点）；LLM——GPT2-XL、Mistral-7B、Mistral-7B-v0.3、DeepSeek-7B（激活函数 GELU/Softmax/Swish）。数据集：ImageNet（DNN）；ARC-Easy、HellaSwag、CommonsenseQA、COPA（LLM 的 4 个 NLP 任务）。bench：8 个非线性函数（Sigmoid/Tanh/GELU/Swish/Softplus/arcsinh/Sin/sqrt）用于单元级；8 个系统级 benchmark（Mish、Logsigmoid、Softmax、Softplus、Swiglu、Svm、DCT、KNN，共 11 个 loop kernel），激活函数 benchmark 源自 CGRA-Nonlinear-Benchmark [55] 与 PICACHU [56]。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：算法部分开源（COFFA 仓库 LoRA-ISCA-AE 分支 https://github.com/Dai-dirk/COFFA/tree/LoRA-ISCA-AE，Docker docker.cnb.cool/fudaneda/docker/chipyard 内 /root/PiecewiseChebFitter，Zenodo DOI https://doi.org/10.5281/zenodo.19447155，BSD 3-Clause）。使用例子（算法级复现）：`cd /root/PiecewiseChebFitter && python3 run_func.py <gelu|sigmoid|softplus|swish|tanh>`，每函数配置在 config 文件夹，结果输出到 fig/ 与 result/ 文件夹。
  - 算法 pipeline 伪代码：
    ```
    # 输入: 目标函数 f(x), 区间 [a,b], 最大项数(默认6), 最大段数 N, 定点上限 Q_max
    # 1. 采样（curvature-based）
    {x_i} = uniform_sample(a, b, m)
    κ(x_i) = 数值微分估计曲率; 在 κ 大的区域插入额外样本 → {x_i, f(x_i)}
    # 2. 无分段整体逼近（决定每段允许的最大次数）
    x' = (2x - (b+a)) / (b-a)                    # 区间变换 → [-1,1]
    若 f 为奇函数: V 的偶数列置 0 (如 [0,T1,0,T3,0,T5])   # 奇偶性 → 同项数更高次数
    # 3. 分段搜索
    for seg in 2..N:
      breakpoints = 分段策略(seg)   # uniform | curvature (累计曲率 W_m/N 均分) | equal-error (Brent 迭代)
      best_deg = 遗传算法(个体=各段次数分配 k_seg1..k_segN, #gen=10, #pop=16)
      对每段: min_c Σ_i (f(x_i) - Σ_j c_j T_j(x_i'))²  s.t. |p_j||x^j|_max < Q_max   # 最小二乘+约束
      记录平均 MSE
    选平均 MSE 最小的 (段数, breakpoints, 系数, 次数)
    # 4. 输出: breakpoints, 每段多项式系数 c_i/log2(c_i) 与次数 k_i → 存入 XCore LUT
    ```
  - 张量计算例子（单段 6 项逼近）：采样 m 点后构造 6 列 Chebyshev 矩阵 V（第 j 列 = T_j(x_i')，按递推 T_0=1、T_1=x'、T_n=2x'T_{n-1}-T_{n-2}），解最小二乘 V·[c_0..c_5]^T ≈ [f(x_i')]^T 得系数 c；再把 Σc_iT_i(x') 展开为标准多项式 Σp_i'x'^i，经逆区间变换 x=(x'+1)(b-a)/2+a 得 Σp_ix^i。XCore 执行时每项 c_i·x^(k_i) 在 LNS 中算为 2^(log2(c_i) + k_i·log2(x))（log 转换 → 30b×6b 乘法（5 项并行，乘法器分解 30b 为 5×6b）→ 反log），Output 级加法树（CPA 可配定点/浮点加）求和，7 cycle 输出。示例：tanh(x)+1 或 sin(x)+cos(x) 等复合函数作为单一 XCore 节点直接逼近，无需分解。
