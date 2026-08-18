## cio（Countering Instruction-Centric Optimizations，二进制码变换缓解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- cio 是 Flanders、Sharma、Michael、Grossman、Kohlbrenner（UW）在 ASPLOS 2024 提出的零泄漏软件缓解工具（"Avoiding Instruction-Centric Microarchitectural Timing Channels via Binary-Code Transformations"，DOI 10.1145/3620665.3640400，Zenodo artifact https://zenodo.org/records/10594315，仓库 https://github.com/counter-optimization）：针对未来指令中心微架构优化（computation simplification、silent stores 等）产生的时序侧信道，在汇编/二进制级做变换，使秘密相关不安全操作数永不取触发"快速路径"的值，从而消除数据相关行为。它解决源码级 CT 无法处理的挑战：寄存器溢出、地址计算、复杂指令微操作；并展示微架构缓解组合安全问题的首个具体实例。Helium（ISCA 2026）把它作为 Case Study IV 的 baseline，用 TracerSim 评估"接受小概率泄漏"可避免的 cio 开销。
- 逻辑链：cio 覆盖的计算简化类别（Helium 表 V）——ADD（两操作数不安全值 {0}）、SUB（第二操作数 {0}）、MUL（两操作数 {0,1}）、OR/AND/XOR（两操作数 {0,0xFFF}）、SHL/SHR/SAL（两操作数 {0}，SAR 第一操作数 {0,0xFFF}、第二 {0}）；类别 mul64（仅 64-bit 乘法）、cs64（64-bit 除乘法外全部）、cs32（32-bit 含乘法）。变换示例（论文 §VII-D）：32-bit 减法强制第二操作数非零——两操作数零扩展、第 33 位置 1、减法、取低 32 位；语义等价但操作数恒不含不安全值。
- Web 证据：https://dl.acm.org/doi/abs/10.1145/3620665.3640400；论文 PDF https://homes.cs.washington.edu/~dkohlbre/papers/cio-asplos24.pdf；Zenodo artifact DOI 10.5281/zenodo.10594315。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译/二进制变换框架中，cio 的运转（对已编译二进制做指令级重写）：
```
输入：x86_64 二进制（libsodium），不安全指令/操作数/值集合（表 V 类别）
1. 定位不安全指令：识别其不安全操作数（可能来自寄存器溢出/地址计算的复杂来源）
2. 逐指令变换为语义等价的无泄漏序列：
   例 32-bit SUB（强制第二操作数≠0）：
     a' = zero_extend_33(a); b' = zero_extend_33(b)
     b' = b' | (1 << 32)          # 第 33 位置 1 ⇒ 永不为 0
     r' = a' - b'                 # 恒走慢速 µobs（无 zero-skip 快速路径）
     r = r'[31:0]                 # 取低 32 位（语义等价）
3. 输出变换后二进制（所有不安全指令恒呈现同一 µobs）
```
- 全栈例子（Chacha20-Poly1305 mul64）：每条 64-bit MUL 被变换（零扩展→第 33 位置 1→乘法→截断），使操作数永不含 0/1、zero-skip 快速路径永不触发 ⇒ 时序与秘密统计独立（零泄漏），总开销 2.31×（Helium 表 VI 引 [37]）。其他类别开销：cs64 2.67×、cs32 3.37×；AES-GCM 因无 64-bit MUL/少 32-bit 算术而零开销；Ed25519 mul64 8.79×、cs64 6.51×；Argon2id mul64 15.71×。
- Annotations：变换针对"不安全值集合"（触发快速路径的值）而非全部秘密路径——语义等价性保证正确性；每变换指令数增加（平均多 3–4 条指令）⇒ 开销随不安全指令密度上升；cio 只针对"指令中心"（instruction-centric）优化（操作数值依赖），不覆盖其它类型侧信道（如 cache 末级共享）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：x86_64 汇编/二进制级重写工具（非 LLVM pass），对 libsodium 评估；处理寄存器溢出、地址计算与复杂指令微操作；Zenodo artifact 含 Docker 镜像与评估脚本（ASPLOS 2024 AEC）。仓库 https://github.com/counter-optimization。
- 使用：作为"零泄漏"baseline 与 Helium 的有界泄漏路线对比——Helium 用 TracerSim 在未修改的 Libsodium 1.0.18-RELEASE 二进制上算 tail-bound：Chacha20-Poly1305 mul64 得 P[ℓ≤0.0004]≥0.9997 ⇒ 可安全省去 2.31× 开销；cs32 得 P[ℓ≤0.0062]≥0.9947 ⇒ 省 3.37×；cs64 得弱保证 P[ℓ≤2.1461]≥0.9552 ⇒ 建议保留缓解；Ed25519 按函数分解（表 VII）支持选择性缓解（只保护 sc25519_reduce、ge25519_scalarmult_base，避免 8.79×）。局限：Helium 明确 cio 类缓解仍必要当程序高泄漏概率高时（Ed25519/Argon2id 多数类别"all µtraces high leakage"）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
