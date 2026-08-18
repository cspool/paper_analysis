## FPMA（Floating-Point Multiplication Approximation，浮点乘法近似）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FPMA 是一种近似浮点乘法技术：基于 Mitchell 对数近似 log2(1+M)≈M（1962），把 IEEE 浮点数的对数值近似为指数与尾数的直接拼接/加法，从而把浮点乘法变成整数加法。具体推导：对 x=(−1)^Sx·2^(Ex−B)·(1+Mx)，有 log2(|x|)=Ex−B+log2(1+Mx)≈Ex−B+Mx；于是乘积 r=x·y 的对数近似为 (Ex+Mx)+(Ey+My)−2B，等价于在拼接的 exponent-mantissa 域上做整数加法 R≈X+Y−B（X=Ex+Mx、Y=Ey+My、R=Er+Mr）。由于消除了乘法器（部分积 O(n²)），FPMA 的硬件成本随位宽近似线性 O(n)，这是它在 UNICORE（S-FPMA）、AxCore（mpFPMA）等加速器中被用作可扩展计算原语的根本原因。局限：对数近似在低比特/含 subnormal 输入时误差大（log2(1+M)≈M 对无前导 1 的 subnormal 不成立），需要 subnormal 归一化与误差补偿配合。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - UNICORE 中 FPMA 乘法的计算过程（W4 例）：操作数先被归一化为内部 E3M2 正常数（无 subnormal）→ 取 X=EW+MW、Y=EA+MA（拼接的 exponent-mantissa 域）→ 整数加法 R=X+Y−B（一次加法即一次"乘法"）→ 双路径补偿：FG 细粒度补偿 C_fg(M_A,M_W)（LUT 预存残差 bit-pattern）拼接到 R 的 LSB 侧恢复低位、CG 粗粒度 1-bit 进位注入修正高位 → 得到近似乘积，转 sign-magnitude 后与部分和累加。示例（FP4 E1M2）：FPMA 对精确积 36 输出 32，CG 补偿无法在 2-bit 尾数粒度下表达修正（结果仍 32），FG 补偿拼接 "01" 后恢复为 36，与全精度乘法一致；FP8（E4M3）例：精确积 66，FPMA 输出 60，CG 补偿调高位到 64、FG 拼接 "01" 恢复 66。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RTL 中用整数加法器网络替代浮点乘法器（UNICORE 的 S-FPMA 把 FPMA 分解为统一 4-bit 加法 slice、进位链级联成更宽精度；AxCore 用 mpFPMA 支持 W4A16 定宽混合精度）；补偿用小型 LUT（按 (M_A,M_W) 索引）+ 进位注入。使用：量化 LLM GEMM 加速器（AxCore、April、UNICORE）的计算核心，把乘法主导的 GEMM 变成加法主导、位宽可线性扩展的计算；低比特模式必须配合 subnormal 归一化 + FG/CG 补偿（无 FG 时 UNICORE FP4 PPL 崩坏到 1.1E+4–4.9E+6，加补偿后 11.15 与原始 FP4 相同）。开源参考：UNICORE https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
