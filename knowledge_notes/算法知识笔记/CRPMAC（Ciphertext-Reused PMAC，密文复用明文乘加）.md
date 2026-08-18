## CRPMAC（Ciphertext-Reused PMAC，密文复用明文乘加）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PMAC（Plaintext-Multiply-and-Add）组合 PMult（明文乘密文）与 HAdd（密文加），计算一批密文与明文系数的线性组合 Σ_i pt_i·ct_i，是 CKKS bootstrapping 的 CtS/StC 相位中 PCMM（明文-密文矩阵向量乘）的核心原语，在高 level（ℓ 大）下 PMAC 可能主导成本。CRPMAC（Ciphertext-Reused PMAC）是 HyperDrive 提出的 PMAC 变体：在 BSGS PCMM 中把所有 GS-Rots 推迟到末尾、按 GS 方向批处理，使单个 baby-step 密文被一批明文复用（一次 GMEM 读、多次乘加），消除 [22] 批处理 PMAC 每轮重复加载同一 baby-step 密文的冗余 GMEM 读。
- 背景：prior work [22] 只把 PMult 与 HAdd 融合进每轮 GS 密文生成，且按 BS 方向批处理（图 7）；HyperDrive 的 CRPMAC 按 GS 方向批处理（图 14），内存足迹更小，并对所有 EWArith kernel 施加向量化访存缓解带宽压力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BSGS PCMM 中 CRPMAC 的批处理计算过程（bs 个 baby-step 密文 ct_i、gs 个 giant-step 明文对角 pt_{i,j}，i=1..bs、j=1..gs）：
```
# BS 相位：只做一次预旋转（ModUp），得到 bs 个 baby-step 密文
for i in 1..bs:  ct_i = BS_Rot(ct, offset_i)     # 复用同一 ModUp 输出（hoisting）
# CRPMAC：推迟全部 GS-Rots，按 GS 方向批处理（一次读 ct_i 复用 gs 次）
for i in 1..bs:                                  # 外层遍历 baby-step 密文
    ct_i ← GMEM-Load 一次                        # 关键：每 i 只读一次，跨 j 复用
    for j in 1..gs:  acc_j = acc_j + PMult(ct_i, pt_{i,j})   # 批量乘加
# 末尾统一做 GS-Rots，得到 ct'_j 输入给后续
for j in 1..gs:  ct'_j = GS_Rot(acc_j, offset'_j)
```
- Annotations：对照 [22] 的 BS 方向批处理（外层 j、内层 i）每轮 j 都要重读同一批 ct_i，GMEM 读冗余；CRPMAC 把密文读从 O(bs·gs) 降为 O(bs)。与 hoisting 场景结合时在 GS 方向批处理使内存足迹更小。论文消融：CRPMAC 使 CtS/StC 的 EWArith 相对 batched PMAC 提速 1.34×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 bootstrapping 的 CtS/StC 相位把 PCMM 的 PMAC 阶段重排为"按 baby-step 密文外层循环 + 按 GS 方向内层批处理"的 CUDA kernel，配合向量化（vectorized）EWArith 访存；与 hyperdrive 的 COOP kernel（BConv2-NTT1/NTT2-IP）集成进 bootstrapping 流程（论文 §V）。使用场景：BSGS/double-hoisting 的 CKKS bootstrapping；论文在 Set-E 上 Bootstrap 39.49 ms，HyperDrive-CORE 相对 BASE 的 bootstrap 提速 1.85×（含 NTT+ 1.49×、COOP 1.21×、CRPMAC 对 CtS/StC 的 1.34× 贡献）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
