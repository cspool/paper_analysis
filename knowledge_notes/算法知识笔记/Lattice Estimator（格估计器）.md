## Lattice Estimator（格估计器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lattice Estimator 是 Albrecht、Player、Scott 等维护的开源密码学工具（https://github.com/malb/lattice-estimator，Sage 实现），用于估计基于格的问题（LWE、NTRU、SIS 等）在给定参数下的实际破解成本（比特安全强度），把主流攻击算法（BKZ 格基约化、primal/dual 攻击、Meißner 等）的复杂度估计出来。它由论文 "On the concrete hardness of Learning with Errors"（J. Math. Cryptology, 2015）提出，是 FHE 社区选安全参数的事实标准。
- 在 TFHE 参数选择中，Lattice Estimator 用于验证：(1) 给定 LWE 维数 n、GLWE 度 N、模数等，估计的安全级是否 ≥ 目标（如 128-bit）；(2) 结合噪声传播分析（保证 PBS 错误概率 p_err<2^-14），在"安全-噪声-位宽"三维空间找可行参数点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FlashTFHE 论文用 Lattice Estimator 映射"安全级别 vs 常用参数"（Figure 7）：对每个消息位宽（2–10 bit），搜索满足 128-bit 安全与 p_err≤2^-14 的 (n, N, l_b) 组合，绘制 log2(N) 与 n 随位宽的增长曲线，得到结论"位宽增大 → n 与 N 必须同步增大"（10-bit 需要 N=2^16、n>1000），并据此标注 Morphling 支持的 6-bit 上限与 FlashTFHE 的 10-bit 目标。流程：候选参数 → Lattice Estimator 估安全级 → 若 ≥128-bit 且噪声估计 ≤2^-14 则接受 → 否则增大 n/N/l_b 重试。
- 具体到一次评估：给定 (n=1070, N=65536, k=1, l_b=8, 位宽9)，Lattice Estimator 估计该 LWE 实例的 BKZ 成本对应的安全 bit 数；同时用 gadget 分解与多项式乘的噪声增长公式估计自举后噪声，两者都达标才进入 Table II 的可用参数集。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Sage 库 + 攻击复杂度模型（BKZ 成本用 sieve/enumeration 模型），命令行或 Python API 调用，输出每攻击路径的估计成本与最小安全位。使用：FHE 库（如 Concrete Optimizer、OpenFHE、TFHE-rs 参数搜索）内嵌 Lattice Estimator 自动选参；研究者用它批量扫描参数空间生成曲线（本论文 Figure 7 即一例）。注意点：它是"估计"工具，不同版本/攻击模型假设会改变结果，论文通常固定版本与攻击集以保证可复现。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
