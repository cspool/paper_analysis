## NTT / INTT（数论变换 / 逆变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NTT（Number-Theoretic Transform，数论变换）是 FFT 在有限域（模素数）上的对应，把次数 N 的多项式从系数域变换到求值（槽/NTT）域，复杂度 O(N log N) 次蝶形运算；INTT 是逆变换。CKKS 中多项式乘法（密文乘法、BConv、IP 等）利用 NTT 把系数卷积变成逐点乘法：先 NTT 到求值域逐点乘，再 INTT 回系数域。NTT/INTT 是计算密集型算子（算术强度 0.89 ops/byte），其可并行性与数据访问模式决定硬件 NTT 单元（NTTU）的微架构。
- 在 keyswitch 的 ModUp/ModDown 中，每个密文 group 沿 INTT→BConv→NTT 流水执行：INTT 把系数域多项式转到 BConv 需要的求值形式、BConv 完成基转换、NTT 转回系数域。NTT 域（求值域）密文利于多项式乘（PMul/CMul），INTT 域密文仅出现在 BConv 之前——这一不平衡是 HE² 的 INTT-Resident/NTT-Resident 自适应密文格式管理策略的出发点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次多项式乘（p = a·b mod (X^N+1)）的 NTT 计算过程：
```
A = NTT(a);  B = NTT(b)       # 系数域 → 求值域（各 O(N log N) 蝶形）
C = A ⊙ B                     # 逐点相乘（N 次模乘）
c = INTT(C)                   # 求值域 → 系数域（O(N log N)）
```
- Annotations：NTT/INTT 蝶形访问同一多项式内的不同系数（高多项式内并行），而 BConv 同时处理来自多个多项式的系数（高多项式间并行）——两类单元的并行模式错配导致难以重叠，HE² 用可配置迭代 radix-2 NTTU 与 tree-based BConvU 做吞吐匹配解决（见"NTTU/BConvU"与"双级流水 xPU"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库用迭代 radix-2/radix-4 蝶形 + twiddle 因子表；硬件 NTTU 用蝶形阵列 + 冲突无关访存（HE² 采用 Mu et al. 的可配置迭代 radix-2 NTTU，NTT/INTT 动态共享）。使用：CKKS 每次多项式乘（PMul/CMul/keyswitch 的 IP/BConv）都含 NTT/INTT；HE² 中 NTTU 按 dnum 组均分到 BConv 所需 limb 上保证并行供数，NTTU allocator 在 INTT-Resident 流水两条并行路径间动态平衡负载（HE² xPU 的 NTTU 平均吞吐 768 w/ns，对比 SHARP 1024 w/ns）。
- HyperDrive 补充视角（ISCA'26，GPU TCU 上的分层 NTT 分解）：采用 Bailey 4-step NTT 递归分解（Alg. 1）把 N=2^13~2^16 的多项式写成 N=N1·N2、N1=8·8·N13、N2=8·8·N23，递归直到 radix-64 基例（匹配 FP64 TCU 的 8×4×8 MMA 维度，即 PTX m8n8k4），复杂度保持 O(N log N)（Inner-NTT 开销 C_NTT=Nk(t+1)、Hadamard C_HP=t·O(N)，N=k^(t+1)）；(N13,N23) 按 (2,1)/(4,1)/(2,4)/(4,4) 对应 N=2^13/2^14/2^15/2^16。NTT 被拆成 Inner-NTT（radix-64 基例，全片上执行）与 Outer-NTT（EWMult、Residual NTT、转置、GMEM/片上搬运），并把 32-bit 字长在 FP64 TCU 上用轻量 MPA 处理（单次 32-bit 乘法仅 2 次 FP64 乘法，对比 INT8 方案的 16 次）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
