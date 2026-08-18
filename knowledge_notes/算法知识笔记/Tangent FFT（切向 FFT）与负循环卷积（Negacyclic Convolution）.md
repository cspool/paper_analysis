## Tangent FFT（切向 FFT）与负循环卷积（Negacyclic Convolution）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tangent FFT（Bernstein 2007）是把环 $\mathcal{R}_q=\mathbb{Z}_q[X]/(X^N+1)$ 上的多项式乘（即负循环卷积 negacyclic convolution）归约为一个 N/2 点标准复数 FFT 的专用变换族：正变换 $\mathrm{TFFT}[\mathbf{a}]=\mathrm{FFT}_{N/2}[\mathbf{b}]$，其中 $b_j=(a_j-i\cdot a_{j+N/2})\cdot\omega^j$、$\omega=e^{-i\pi/N}$ 为 2N 次本原单位根；逆变换 $\mathbf{b}=(\mathrm{IFFT}_{N/2}[\mathbf{c}])^*$、$a_j=\operatorname{Re}(b_j\cdot\omega^j)$、$a_{j+N/2}=\operatorname{Im}(b_j\cdot\omega^j)$。相比把多项式长度翻倍到 2N 的常规做法，Tangent FFT 的辅助运算（前后处理）更简单且完全可并行，天然适合 GPU。负循环卷积是格基 FHE 中多项式乘的核心：因环模为 $X^N+1$（而非 $X^N-1$），标准 FFT 的循环卷积（模 $X^N-1$）不能直接使用，必须用带符号折叠的专用变换。两多项式 u、v 的负循环积 $c=\mathrm{ITFFT}[\mathrm{TFFT}[\mathbf{u}]\circ\mathrm{TFFT}[\mathbf{v}]]$（∘ 为逐元素乘），把复杂度从 $O(N^2)$ 降到 $O(N\log N)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 TFHE 盲旋转中，负循环卷积出现于每次外部乘积的多项式乘（ACC 与 BSK 分量的乘积），是 PBS 每迭代的核心计算。伪代码（N=512 的 Tangent FFT，即 N/2=256 点 FFT）：
```
# 正变换（Forward Tangent FFT）
for j in 0..N/2-1:
    b[j] = (a[j] - i*a[j + N/2]) * omega^j     # omega = e^{-i*pi/N}，预计算
c = FFT_256(b)                                  # 256 点复数 FFT

# 负循环卷积：c = ITFFT[ TFFT[u] ∘ TFFT[v] ]
cu = TFFT(u); cv = TFFT(v)                      # 两个正变换
c_pt = cu ∘ cv                                  # 逐元素乘（Fourier 域）
c = ITFFT(c_pt)                                 # 逆变换，含共轭与 omega^j 复原
```
- Annotations：`omega^j` 是旋转因子（precomputation factor）；正/逆变换分别使用 `omega^j` 与其共轭版本（MNEMOS 跨迭代融合即利用这一共轭对称性）；MNEMOS 在 GPU 上把 N/2 点复数 FFT 映射到 FP64 Tensor Core（WMMA 8×8×4）执行，且精度分析表明该 FFT 需 FP64（≥30 小数位）才能保证 PBS 解密正确。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Bernstein 原版 C++、TFHE-rs/Concrete（Zama 生产库，CPU/GPU）、Lattigo（Go）、cuFHE 等 FHE 库内置；加速器（MATCHA/Morphling/Strix/FlashTFHE/CASCADE）的 FFT 单元即实现该变换（或等价的双实数 FFT/负循环卷积归约）。MNEMOS 在 GPU 上以 CUDA kernel 实现：b 向量构造与 omega^j 乘加在寄存器完成，256/512/1024 点 FFT 用四步 FFT 递归分解到 8 点 WMMA 基例，Fourier 矩阵运行时片上生成。使用场景：所有环为 $\mathbb{Z}[X]/(X^N+1)$ 的 TFHE/类 TFHE 方案的多项式乘加速；N/2 点 FFT 也显著减小变换规模（相对 2N 做法省一半点数）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
