## Multi-bit TFHE（多位 TFHE：整数密文扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Multi-bit TFHE 是 TFHE（Fast FHE over the Torus）的整数加密扩展（Chillotti-Joye-Paillier ASIACRYPT'21 等）：不再每个 ciphertext 只加密 1 bit，而是把多个 bit 打包进单个 LWE ciphertext 加密一个整数（如 3–10 bit），从而把可编程自举（PBS）的成本摊薄到大量廉价线性运算上。Boolean TFHE 每个 bit 一次操作都要一次 PBS（每个 gate 一次 bootstrapping，几百万 gate）；multi-bit TFHE 只有两类原语：线性运算（加法、明文标量乘法，LWE-native、无需自举、极快）与 LUT（查任意非线性函数，每次需一次 PBS）。因此程序执行画像从"百万次 PBS 主导"变为"海量廉价线性运算 + 少量昂贵 PBS"，且位宽越宽，两次 PBS 之间可连续执行的本征线性运算链越长，PBS 频率按数量级下降（FlashTFHE 论文 profiled 七个真实 workload 验证：DNN 类 ~99.5% 运算是 LWE-native，PBS 仅占 ~0.5%）。
- 代价：更宽位宽需要更大的密码参数集（LWE 维数 n、GLWE 度 N、gadget 分解深度 l_b 都要随位宽增大才能维持 128-bit 安全与 p_err<2^-14 的噪声界）。例如 10-bit ciphertext 需要 N=2^16、n≈1000+，BSK key 达数 GB（GPT-2 decoder layer 的 key 有 4.7GB），远超 Boolean TFHE 的几 MB。参数变大的后果正是 FlashTFHE 的动机：现有空间加速器（Morphling 只支持 6-bit/N≤4096、Strix N≤16384）被带宽、利用率、面积三重瓶颈卡住，FlashTFHE 用时间域密钥复用把支持推到 10-bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个 8-bit 整数加法的两种实现对比（论文 Figure 5）：
```
# Boolean TFHE：6-bit 加法 = 逐 gate 拆解，每 gate 一次 PBS
for each bit b in 0..5:
    out[b] = Gate(FA(...))   # 每个 full-adder gate 都含一次 PBS
# 253 ms（每 gate 11ms，加法需 23 个 gate）
```
```
# Multi-bit 8-bit TFHE：整个整数在一个 ciphertext 里
acc = c1 + c2                # LWE-native 向量加，无 PBS
# 0.008 ms；5-bit 分段版因 carry 需 bivariate LUT 反而要 1 次 PBS（47ms）
```
- 程序级 pipeline（以 GPT-2 decoder layer，7-bit 量化/6-bit rounding）：量化权重与激活 → 逐层 matmul/逐元素加（LWE-native，零 PBS）→ 非线性（GELU/softmax 等）编码进 LUT → 每 LUT 一次 PBS 刷新噪声 → 继续下一层。PBS 频率 = 非线性激活/函数个数而非算子个数，这是 multi-bit 相对 Boolean 端到端加速（论文实测 8-bit 相对 4-bit 模拟加速 6.8–8.1×，DNN 类）的根本来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：Zama 的 Concrete/Concrete-ML/TFHE-rs 库原生支持多 bit 整数密文与参数搜索（Concrete Optimizer），Concrete-ML 对 PyTorch/scikit-learn 模型做 PTQ 后自动生成 multi-bit TFHE 程序；FlashTFHE 论文所有 workload（CNN-20/50、KNN、XGBoost、Decision Tree、GPT-2）均由 Concrete-ML v1.6.1 + Concrete Compiler v2.7.0 生成。硬件实现：FlashTFHE 加速器（BRU 做外部乘积/盲旋转、LPU 做 key-switching 与 LWE-native 运算）。使用注意：位宽选择要在参数代价（N/n/l_b 增大、key 变大）与 PBS 频率降低之间权衡，需 Lattice Estimator/Concrete Optimizer 联合搜索安全参数。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
