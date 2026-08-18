## FHE（Fully Homomorphic Encryption，全同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FHE 是一种加密范式：允许第三方在不解密的情况下对密文直接执行任意计算，解密结果与对明文做同样计算一致（Rivest 等 1978 提出概念，Gentry 2009 首次构造）。形式化 = 公钥加密方案 + 同态求值算法 Eval：对任意电路 C 与密文集合，Eval 输出 Enc(C(m)) 而不解密。安全基于 LWE/Ring-LWE 困难问题；同态性 = 加法与乘法都可在密文域完成并可组合成任意电路。
- 核心机制是噪声：每个密文携带误差 e，同态加法噪声近似相加，同态乘法增长更快，噪声超过阈值则解密失败——因此计算深度受限，需 bootstrapping（同态执行自身解密函数）刷新噪声实现任意深度。性能代价被称 "performance tax"：20 层网络隐私推理比明文推理慢约 10^4 倍；bootstrapping 占 FHE 成本主体（部分分析 >90%，应用运行时通常 50–80%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FHE 在算法 pipeline 中把明文应用的每个算子替换为同态算子。以本论文加密 NPU 隐私推理为例，一层网络的同态计算过程：
```
for each layer l in network:
    enc_out = HomMul(enc_act, plaintext_weight)   # 密文x明文权重
    enc_out = HomAdd(enc_out, plaintext_bias)
    enc_out = ProgrammableBootstrap(enc_out, f)   # 噪声刷新 + 同态激活查表
return enc_out
```
- Annotations：enc_act 是 TFHE 密文；HomMul/HomAdd 由密文算术 CPE 执行；ProgrammableBootstrap 是 FHE 特有开销（FFT/IFFT + external product），是 pipeline 主要延迟来源；bootstrapping 插入时机由 FHE 算法设计者决定（噪声预算耗尽前必须刷新）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 方案族：BGV/BFV（整数域、SIMD 打包）、CKKS（近似实数）、FHEW/TFHE（快速逐 bit 自举 + 可编程自举）。软件库：TFHE-rs（Zama）、OpenFHE、HElib、SEAL、concrete；GPU 库 cuFHE、nuFHE。硬件：Strix/MATCHA/PPGNN/Trinity/Poseidon 等专用加速器把 FFT/NTT、external product 做成硬件单元。使用场景：隐私推理（PI）、私密信息检索（PIR）、加密数据库查询、隐私图处理（本论文三个案例域）。在 FEnc² 中，FHE 还被赋予新的系统视角：应用级密文打包布局（数据布局）是降低 HE 工作负载的一级设计维度，与底层 NTT/keyswitch 加速器优化正交互补——FEnc² 通过减少旋转/keyswitch/NTT 的数量来重塑暴露给硬件的同态负载。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving

IroKnight 补充视角（ISCA'26，FHE 作为"加密所有权"的理论金标准 vs 工程不可行）：IroKnight 把 FHE NPU 当作所有权保有的理论上限与对比基线，用乐观建模（基于 SHARP [9] 设计：专用功能单元 + 180MB 片上 SRAM 存中间 FHE 密文；权重保持 FHE 明文多项式、不密文化，因密文化更贵；非线性算子 softmax/layernorm/ReLU 只能近似）。开销来源：每次同态乘加积累噪声、需反复 bootstrapping（IroKnight 估计占 FHE 延迟主体），以及权重明文多项式/激活密文多项式的大量片外搬移。8 个 LLM 上 FHE 运行时 713x-1793x（Llama3-70B 单 query 10.9 小时）、能量 871x-7396x（0.59 kWh）；小网络（BERT/ResNet-50/ViT）607x-1745x 延迟、11904x-35564x 能量。对比：IroKnight 同时加密激活与模型参数且防篡改，运行时仅 0.2%（加密）/3.3%（认证）、LLM 能量 <=14%/>=<18%。结论：等待实用 FHE 不可行，IroKnight 以"明文仅瞬态存在于 ALU、存储全加密"的新设计点逼近 FHE 所有权而避开其开销。
