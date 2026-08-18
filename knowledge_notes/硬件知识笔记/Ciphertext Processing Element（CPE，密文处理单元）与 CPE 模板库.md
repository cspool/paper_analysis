## Ciphertext Processing Element（CPE，密文处理单元）与 CPE 模板库

术语解释
- FHE 加速器中的基本密文运算单元：普通 PE 的加密对应物，但内嵌 FHE 特有部件（bootstrapping、key-switching、多项式运算）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CPE 是领域专用 FHE 加速器（Strix/MATCHCHA/PPGNN 等）里直接处理密文的功能单元。它不是普通 PE 的简单加密版：以 Strix 的 CPE 为例，除逐元素函数单元外还含 bootstrapping 单元（FFT/IFFT + 多项式乘 + 分解 + vector MAC）与 key-switching 单元，微架构与 FHE 算法紧耦合（Barrier ➊）。AutoFHE 把 CPE 参数化为四类模板：ciphertext arithmetic 模板（并行乘法器/加法器，参数 PoC，做密文加/乘/逻辑，密文x密文走 external product）；bootstrapping 模板（多项式缩放 PoV、分解 PoD、FFT/IFFT 的 BFU/IBFU butterfly 数、external product PoE，内置 (2^r-1) unrolling 阵列）；key-switching 模板（向量单元+累加器，参数 PoK，向量-矩阵乘）；HMUX 模板（由 bootstrapping 模板实例化）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 数据路径例子（一次加密乘加 + 噪声刷新）：密文进入 arithmetic CPE 的并行乘加阵列 → 输出密文噪声升高 → 流入 bootstrapping CPE：GadgetDecomp（PoD lane 分解）→ FFT（BFU butterfly）→ external product（PoE MAC）→ IFFT（IBFU）→ 输出 → key-switching CPE 向量-矩阵乘切回密钥域 → 结果写回。AutoFHE 按 TFHE 协议自动把 bootstrapping 单元实例化在 arithmetic 下游。参数空间：bootstrapping 单元 5 个独立维度各约 8 个离散候选 → ≥8^5≈10^4 倍设计空间膨胀（Difficulty 2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 手工实现难点：FFT/IFFT 占自举延迟约 80%（MATCHA 用 38-bit DVQTF 整数近似 FFT——TFHE 逐门自举刷新噪声故可容忍近似，BGV/CKKS 则需精确 NTT；FPT 用单 CMUX 流水 PE；OFHE 用光电 FFT 引擎）。AutoFHE 用法：设计者不写 CPE，只写明文 Chisel + Secure 注解，CPE 模板由框架实例化互连。面积代价：单 CPE 资源比普通 PE 高数量级（这是 CPE 虚拟化的动因）。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
