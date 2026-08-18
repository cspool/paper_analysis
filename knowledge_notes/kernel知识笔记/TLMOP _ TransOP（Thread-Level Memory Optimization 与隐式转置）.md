## TLMOP / TransOP（Thread-Level Memory Optimization 与隐式转置）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TLMOP（Thread-Level Memory OPtimization）是 HyperDrive 的 Inner-NTT 优化：用细粒度寄存器数据访问替换 TCU MMA 的自动 SMEM fragment 填充。自动填充下，每步 MMA 前后 fragment 都要经 SMEM 读写（单次 N=2^16 Inner-NTT 产生 6.13MB SMEM 流量、单次 KeySwitch 达数 GB）。TLMOP 先分析 FP64 MMA（8×4×8，PTX m8n8k4）fragment 在 32 线程间的数据分布（每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 D 均分），据此把 64 点 Inner-NTT 的计算流与 lane 映射设计成纯寄存器流水：每个 warp 执行 4 次 MMA + Bit-Merge + ModRed + EWMult，SMEM 只在读全局输入与写最终结果时访问一次。
- TransOP（Transpose OPtimization）解决 4-step NTT 内嵌的转置：常规转置是 SMEM 读-写全周期的内存操作，HyperDrive 利用 MMA 的 Fragment A/B/D 跨线程分布实现隐式转置——MMA1/2 把数据矩阵当 Fragment B、MMA3/4 当 Fragment A 并交替选取奇/偶列，使数据无需跨线程交换即可从 MMA1/2 结果直接进入 MMA3/4，只有 Bit-Merge 等线程本地操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单个 warp 完成一个 64 点 Inner-NTT 的寄存器内流水（图 8A/9）：
```
# 数据布局：warp 的 32 线程持有 8×8 数据矩阵的 fragment 分布
# ① 两级 radix-8 的第一级：两次 MMA（乘 TFM 高/低 16-bit 分量）
D1 = MMA1(A_data, B_tfm_hi);  D2 = MMA2(A_data, B_tfm_lo)   # 8 个 radix-8 NTT
# ② 位合并与约减
r = BitMerge(D1, D2);  r = ModRed(r);  r = EWMult(r, twiddle)
# ③ 第二级：数据变 Fragment A、twiddle 变 Fragment B，隐式转置
D3 = MMA3(A=r[奇列], B=tfm);  D4 = MMA4(A=r[偶列], B=tfm)
# ④ 最终位合并与约减
out = ModRed(BitMerge(D3, D4))
```
- Annotations：关键点——(1) MMA1/2 的 fragment 分布与 MMA3/4 相同，转置由"数据在 Fragment A/B 之间换角色 + 奇偶列选取"完成，无 SMEM 往返；(2) 多 Inner-NTT 在一个 block 内并行执行，采用多 warp/多 block 并行 + warp 内串行循环的混合策略平衡并行度与片上资源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：手写 CUDA kernel，用内联 PTX mma.sync.aligned.m8n8k4.f64（或 WMMA API）并手动管理寄存器 fragment 的线程映射，替代 auto-filled fragment 的 SMEM 中转。使用：任何在 FP64/其他 TCU 上执行 NTT 的 GPU FHE 库；效果——SMEM 相关 stall -33.2%（加 TransOP 累计 -38.9%）、scheduler stall cycle -44.2%（累计 -50%）、occupancy 从 55.0%/62.3% 提到 76.8%/77.0%（NTT+ 达 92.5%）、单次 Inner-NTT SMEM 流量大幅下降；TLMOP 是消融中贡献最大的单项优化（图 15）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
