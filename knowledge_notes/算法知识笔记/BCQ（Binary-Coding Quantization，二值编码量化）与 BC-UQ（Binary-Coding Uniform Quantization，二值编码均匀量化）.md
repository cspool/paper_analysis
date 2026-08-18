## BCQ（Binary-Coding Quantization，二值编码量化）与 BC-UQ（Binary-Coding Uniform Quantization，二值编码均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BCQ（Binary-Coding Quantization）是一种非均匀量化方法（源自 [70] Xu et al. 2018 的 alternating multi-bit quantization，以及 ARB-LLM 等二值化工作）：把全精度向量 W 分解为高精度缩放因子向量 a 与二值编码矩阵 B∈{-1,+1}，重建为 W_i ≈ Σ_{j=1}^k B_{ij}·a_j（k 个二值基、B_ij 为二值系数、a_j 为缩放因子）。它是"非均匀"的——量化等级由数据分布自适应分配，对含 outlier 的数据比均匀量化误差更小。Omni-LUT（ISCA 2026）为兼容 LUT-based GEMM 加速器，把 BCQ 扩展为带 offset 与结构化 power-of-2 scaler 的变体：不学习缩放因子而是用固定 power-of-2 基 {α_u·2^{-1}, α_u·2^0, ..., α_u·2^{b-2}} 整体乘以 uniform scaler α_u，并加 zero-point z_bcq=z_u−(2^b−1)/2——即 BC-UQ（Binary-Coding Uniform Quantization），把均匀量化表达成 binary-coding 格式。BCQ 本身则从校准数据用交替优化（Algorithm 1）学最优缩放因子：GREEDY_INIT 初始化 → 每轮 LEAST_SQUARES（固定 B 解最小二乘 α）+ BST（固定 α 用二分搜索树求最优 B），共 R 轮，低位宽更准但离线校准更重；Key 校准一次每模型，LLaMA2-13B 校准 <10 分钟（H200 GPU）。bit-plane 表示上每平面 B_i∈{-1,+1} 存 1 bit/元素（packed binary），数学本质 Ŵ=Σ α_i B_i（B_i 是"方向"、α_i 是"幅度"），LUT-GEMM 直接对 {±1} 平面做加减 + 查表，无需非均匀量化 centroid index 的 bit-transpose。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - BCQ 离线校准（Omni-LUT Algorithm 1，Key cache）：输入 Key 校准数据 K_cal、位宽 B、轮数 R；GREEDY_INIT(K_cal,B) 得初始 α,B；for r=1..R：α ← LEAST_SQUARES(B,K_cal,α)；B ← BST(K_cal,α)；输出 α*,B*。BC-UQ 在线编码（BEA 贪心残差，Value/Key 通用）：r^(0)=x−zp；for i=1..q：B_i=sign(r^(i-1))；r^(i)=r^(i-1)−B_i·α_i；得 x≈zp+Σ_{i=1}^q α_i⊙B_i。张量计算例子（q=4，d=128 head）：设 token 的 Value 向量 x∈R^128，TSE 求 x_min/x_max → zp_v、δ_v → α={4δ_v,2δ_v,δ_v,0.5δ_v} → BEA 贪心得 4 个 ±1 bit-plane → 每个 bit-plane 由 LUT PE 按 4 激活组查表累加（与 32 个量化权重并行 RAC）→ 各 plane 结果乘 α_i 累加得最终点积。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：BCQ 在算法侧用交替优化/PyTorch 校准；硬件侧由 Omni-LUT 的 BQU（BEA 贪心编码器）在线实现、LGU/PE 消费 bit-plane。已开源的同类参考：NAVER LUT-GEMM（github.com/naver-aics/lut-gemm，支持 BCQ+uniform）、AnyBCQ（BCQ 的多精度扩展，支持直接 bit-plane 运算与按需加载前 p 个平面）。用途：把量化权重/激活表达成硬件友好的二值平面，使 mpGEMM 变成查表+加减；多精度推理可按需只加载所需平面，p=2 比 p=4 少 50% 数据。在 Omni-LUT 中 BCQ（Key per-channel 离线）+ BC-UQ（Value per-token 在线）共同构成 KV cache 量化，实现 KV4 平均 PPL 仅增 0.17、KV3 增 0.75。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
