## KeySwitch（密钥切换 / Key Switching，含 ModUp-ModDown）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KeySwitch（密钥切换）是 FHE 中把密文从一把密钥下转换到另一把密钥下的过程（不泄露明文）：HMult 后 relinearization 把三元密文压回两元，Rot（automorphism）后把旋转密钥换回原密钥。RNS-CKKS 采用 hybrid KeySwitch [20]：先把密文末分量在 R_{Q_ℓ} 中分解为 β 个 digit（β=⌈ℓ/α⌉，α 为 special prime 模数个数、dnum 为分解数），ModUp（NTT + BConv）把 digits 提升到扩展环 R_{PQ_ℓ}，与求值密钥 evk（R_{PQ_ℓ}^{2×β} 中）做 inner product（IP），再经 ModDown（NTT + BConv）回到 R_{Q_ℓ}，最后加到原密文其余分量上。
- KeySwitch 是 CKKS 操作级（HMult/Rot）的主要性能瓶颈：它频繁执行 NTT、BConv、IP 三个高开销子操作，且每次调用伴随大量跨多项式、跨 limb 的高维数据搬运。论文通过 Nsight Compute 剖析发现 IP kernel 的 stall long scoreboard 占比达 74.6%（off-chip 访存），BConv 为 60.6%，是 COOP 融合优化的直接目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- hybrid KeySwitch 的完整流水（β 个 digit，输入密文 ct=(c0,c1)∈R_{Q_ℓ}²）：
```
# 分解：把 c1 按特殊素数结构拆成 β 个 digit
for d in 0..β-1:  digit_d = Decompose(c1, d)          # 各 digit 在 α'_d 个 limb 上
# ModUp：digit 从 R_{Q_ℓ} 提升到 R_{PQ_ℓ}
for each digit_d:  NTT(digit_d);  BConv(Q→PQ, digit_d) # 变换到扩展环（hoisting 可复用）
# IP：与求值密钥点积累加（输出两分量）
IP = Σ_d  digit_d ⊙ evk_d      # evk 尺寸约为输入 2 倍，GMEM 访存密集
# ModDown：结果从 R_{PQ_ℓ} 降回 R_{Q_ℓ}
NTT(IP);  BConv(PQ→Q, IP);  ModDown 归约
# 合并：加到 c0 上
ct_out = (c0 + IP_downtoQ, 0)
```
- Annotations：ModUp/ModDown 内部即 NTT+BConv 交替（BConv 必须在非 NTT 域执行）；double hoisting [7] 把 PMAC 中间量全程保留在 R_{PQ_ℓ}，省掉大部分 ModDown；hyperdrive 的 COOP 把 KeySwitch 的两处 NTT-跨多项式边界融合为 (BConv2-NTT1) 与 (NTT2-IP)（Alg. 2/3），并把 IP 的 evk 在 NTT 计算期间预取到片上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件库（SEAL/OpenFHE/TenSEAL）中作为 relinearization/rotation 的内部过程，evk 在密钥生成阶段预计算；GPU 库（TensorFHE/WarpDrive/Neo/Cheddar/HyperDrive）把 KeySwitch 拆成多个 CUDA kernel（NTT/INTT、BConv、IP、EWAdd），并做 kernel fusion（[22] 的 intra/inter-operation fusion；HyperDrive 的 COOP）与操作重排（ModDown 后接 ModUp 时提前 EWSub）。使用场景：任何 HMult（relinearization）与 Rot（密钥基恢复）；论文实验 KeySwitch（N=2^16、L=32、β=2）中 COOP 相对 NTT+ 提速 1.24×，(BConv2-NTT1) 相对分开执行 1.36×、(NTT2-IP) 1.32×；H100 上 KeySwitch 延迟 414 μs（Set-E，A100 为 710 μs）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
