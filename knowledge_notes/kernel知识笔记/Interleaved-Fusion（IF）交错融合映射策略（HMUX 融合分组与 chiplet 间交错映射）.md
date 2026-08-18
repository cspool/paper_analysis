## Interleaved-Fusion（IF）交错融合映射策略（HMUX 融合分组与 chiplet 间交错映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- IF 是 CASCADE 的 HMUX 任务映射策略（论文 IV 节）：把 BSP 的 n 个 HMUX 划分为若干"连续 HMUX 的融合组"，再把这些组按循环时空顺序交错映射到 C 个 HMUX Chiplet（HC）。两步：(1) 划分——把 n 个 HMUX 分成 k 个连续组 (G_0…G_k)；(2) 交错——组按 t mod C 循环交错到 chiplet（G_0→C_0、G_1→C_1……），用二维时空矩阵 f(t,c) 表示。动机：naive 把每个 HMUX 单独跨 chiplet 交错最大化并行时，ICT（中间密文传输）使 D2D 通信成为瓶颈（D2D 时延 > HMUX 计算时间 → HC 严重欠利用），且 inter-HC batching 按比例增加跨 chiplet ICT 总量、无济于事；IF 把连续 HMUX 融合在本地执行，使组内 ICT 留在单个 chiplet（降低 D2D 通信频率），同时组间交错保持流水并行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合组在 HC 内的执行（Fig.8，组内 2 个 HMUX）：RLWE1 遍历 Rotation→Decomp→FFT→VMA→IFFT 完成 HMUX_1，输出回馈 HC 输入再遍历同一功能单元完成 HMUX_2——因功能单元是 PCG 系数粒度流水，两次遍历在时间上重叠（一个 HMUX 时延≈最长流水级）；为避免功能单元气泡，注入多个 RLWE（intra-HC batching）让不同密文的计算重叠。
- 映射调度伪代码（f(t,c) 的构建，n=17、C=4 示例）：
```
# 目标：Σ|f(t,c)| = 17，最小化 T_task = T_run + T_bubble
f = 2D 矩阵 (t=0..T-1, c=0..3)
t0: f(0,0)=H0,H1   f(0,1)=H2,H3   f(0,2)=H4,H5   f(0,3)=H6,H7   # 组尺寸 2
t1: f(1,0)=H8,H9   f(1,1)=H10,H11 f(1,2)=H12,H13 f(1,3)=H14,H15
t2: f(2,0)=H16,H17 f(2,1)=NA     f(2,2)=NA      f(2,3)=NA       # 3 个空槽（empty-slot）
# OIFS 改为可变尺寸：t1 用 H8,H9 | H10,H11 | H12-H14 | H15-H17 消除空槽
```
- Annotations：|f(t,c)| 为组内 HMUX 数，T_exe(t,c)=max(T_comp×|f(t,c)|, T_comm)；融合变大隐藏 T_comm（D2D 时延），但过粗融合增大 bubble（启动/排空）；空槽（|f|=0）浪费时间槽。权衡由 OIFS 的 DP 求解（见 编译框架 库 OIFS 条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为调度策略由 OIFS 离线生成 f(t,c) 并写入硬件配置；HC 端支持"组内回馈"（输出重入本地输入执行下一 HMUX）与 intra-HC batching（多 RLWE 注入）。使用场景：TFHE 自举流水线的跨 chiplet 部署——解决"流水并行 vs D2D 通信"的折中；评估中相对两种基线策略（SHM 均匀分段、FFM 固定融合尺寸）在 DeepCNN-50/XG-Classifier 上总执行时间最低、HC 利用率与 D2D 带宽利用率最高（DeepCNN-50 参数集 I：95.9%/76.8%）。注意：IF 是映射/调度策略（决定 HMUX 到 chiplet/时隙的分配），与 OIFS（离线求最优配置的调度器+成本模型+DP）是"策略-求解器"关系。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
