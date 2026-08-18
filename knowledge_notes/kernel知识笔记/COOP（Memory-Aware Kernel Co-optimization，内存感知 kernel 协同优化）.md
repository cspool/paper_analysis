## COOP（Memory-Aware Kernel Co-optimization，内存感知 kernel 协同优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- COOP 是 HyperDrive 的操作级 kernel 融合优化：把跨多项式（cross-poly）kernel（BConv、IP）与 NTT 融合成协同 kernel，让高维中间数据驻留片上、隐藏内存延迟、避免 off-chip 往返。此前融合不可行的根因是维度冲突：NTT kernel 以"多项式内"多 pad 方式访问（2D 足迹 N1×#pad），而 cross-poly kernel 需"多项式间"数据交换（3D 足迹 N1×#pad×#poly），融合会耗尽寄存器/SMEM 并降低并行度。RowMaj 消除多 pad 约束后，每 block 数据维度降低，融合成为可能。
- 具体融合：(1) (BConv2-NTT1)——BConv 的矩阵乘法阶段与 NTT Stage-1 融合，每 thread block 处理单个 limb i、BConv 约减沿 α' 维（GMEM→Reg→SMEM），SMEM 中间系数直接喂 NTT Stage-1（SMEM→Reg→GMEM），避免物化 L+α-α' 维（Alg. 2）；(2) (NTT2-IP)——NTT Stage-2 与 IP 融合，NTT 中间结果保持寄存器内直接做 IP 累加（Alg. 3）；(3) INTT2-BConv1——BConv1（EWMult 阶段）与前置 INTT2 融合；BConv1 与 BConv2 不融合（BConv2 的矩阵乘结构会导致重复计算）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- (NTT2-IP) 融合 kernel 的伪代码结构（Alg. 3 简化）：
```
for i in 0..(ℓ+α)-1:                 # 逐 limb
  for k1 in 0..N1-1:
    # NTT Stage-2 内层：寄存器内完成内层 NTT
    for k2 in 0..N2-1:  t[k2] = InnerNTT(c_partial[i][k1][k2])   # 寄存器驻留
    # 外层循环做 IP：与 evk 做 MAC（放弃 lazy reduction，每乘即约减）
    for d in 0..β-1:
      evk_d ← GMEM 预取（在 NTT 计算期间预取到片上，隐藏延迟）
      acc += t[k2] * evk[i][k1][k2][d] mod q_i      # 立即约减，寄存器减半
    out[i][k1][k2] = acc
```
- Annotations：数据 prefetching 的关键是 IP 的 evk（尺寸为输入 2 倍、访存密度高、算术强度低）——融合后 evk 在 NTT 执行阶段预取，与 NTT 计算重叠；寄存器压力缓解：放弃 lazy reduction（每乘即约减）把中间结果位宽减半、寄存器需求减半，NTT-IP kernel 寄存器 72/线程（35.3% occupancy）仍优于分开执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：改写 KeySwitch/bootstrapping 的 kernel 流水，把 (BConv2-NTT1)、(NTT2-IP)、(INTT2-BConv1) 写成融合 kernel，并用 evk 预取与即时约减控制寄存器；配合操作重排（ModDown 后紧跟 ModUp 时提前 EWSub，使 ModDown/ModUp 的 INTT 可批处理合并，图 13）。效果：KeySwitch 相对 NTT+ 提速 1.24×，(BConv2-NTT1) 1.36×、(NTT2-IP) 1.32×；BConv 的 stall long scoreboard 从 60.6%、IP 的 74.6% 显著下降；bootstrap 中 COOP 再贡献 1.21×（KeySwitch 1.24×、Rescale 1.21×）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
