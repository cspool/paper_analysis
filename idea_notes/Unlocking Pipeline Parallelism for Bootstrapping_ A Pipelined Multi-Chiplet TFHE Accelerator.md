## Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

- baseline方法是什么？
  - baseline 是现有 TFHE 加速器（MATCHA/Strix/Morphling 等 ASIC、FPT 等 FPGA、GPU 方案）对可编程自举（BSP）中 n 次 HMUX（盲旋转）迭代的处理方式：由于每个 HMUX_i 依赖前一个 HMUX 输出的中间密文 ACC（ICT），且每个 HMUX 需要访问独立的高阶多项式 BSK（GGSW，L×(k+1)×(k+1) 多项式矩阵），先前的加速器一律把 n 次 HMUX 严格串行执行。Morphling 用集中式多级内存层次（片上 buffer + HBM）+ batching 提高 BSK 复用、用 transform-domain reuse（FFT 域复用）减少域变换，但仍强制 bootstrapping 内部串行访问 BSK 以避免带宽崩溃；Strix 用两级 ciphertext batching 流式架构但 n 次迭代仍整体串行。其吞吐受 Thp_seq ≈ 1/(n×t_HMUX) 硬约束，n 是安全参数（128-bit 需 n≈592、N=2048），越大越慢。集中式 HBM 无法支撑跨 HMUX 流水线所需的并发 BSK 访问带宽：Morphling 单 HBM stack 约 30W（接近加速器 die 功耗的 56%），为匹配流水线带宽需更多 HBM stack，成本/功耗不可接受。实测：CPU 上 BSP 中 n 次 HMUX 占执行时间 79%，BSK 搬运占总数据搬运 80%，n-HMUX 算术强度远低于 A100 的平衡点（440 GOPS/s）。
  - baseline 全栈执行例子（Morphling 执行一次参数集 III 的 bootstrapping，n=592）：
    ```
    算法pipeline层：可编程自举 BSP——LWE 密文 c_in=(a_1..a_n,b) 放大 2N/q 倍，ACC←X^(-b)·c_T，
               然后 n 次 HMUX（Line 5: BSK_i←(X^(-a_i)-1)·BSK_i；Line 6: ACC_i←BSK_i⊡ACC_{i-1}），
               最后 SampleExtract + key-switching（KSK 标量乘）；BSK 为 GGSW 多项式矩阵
    系统框架层：论文未明确说明（加速器论文，无 serving 框架；CPU baseline 用 Concrete 软件库）
    编译框架层：论文未明确说明（硬件映射无编译框架，Morphling 依赖离线配置）
    kernel调度层：Morphling 用 batching 提高 BSK 复用 + transform-domain reuse 减少域变换，
               但 bootstrapping 内部 n 次 HMUX 严格串行——同一时刻只处理一个 HMUX，BSK 从
               HBM 顺序取回避免带宽崩溃（若使能流水并行，集中式内存带宽饱和，利用率仅 14.3%）
    硬件架构层：集中式多级内存（片上 SRAM + HBM），一次 HMUX = 域变换(FFT) → 逐系数乘加(外积) →
               逆变换(IFFT)，ACC 写回/读回 buffer；执行时延 ≈ n×t_HMUX（n=592、参数集 III 时
               BSP 时延 0.38 ms、吞吐 41,850 BSP/s；DeepCNN-100 上为 Morphling 基线）
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 CASCADE 通过三项协同设计把 n 次 HMUX 从串行改为流水线并行（理论吞吐 Thp_pipe ≈ 1/t_HMUX，最高 n× 提升）：
    ① 多芯粒流水线架构 + BSK-distributed 分布式 SRAM 层次：把全部 BSK（126 MB，足以容纳 128-bit 安全参数）驻留在 12 个 HC 的私有 SRAM（每 HC 10.5 MB + 1 MB 缓冲），消除 BSK 片外搬运，并把并发 BSK 访问分散并限制在各 chiplet 内；配合 intra-HC 与 inter-HC 多项式系数粒度（PCG）两级流水最大化资源利用率（对比：MP-PP 集中式带宽饱和利用率仅 14.3%，MP-PP-HBM 需 8 个 HBM stack 且每瓦效率低 3.7×）。
    ② Interleaved-Fusion（IF）映射策略：把 n 个 HMUX 划分成可变大小的连续融合组、按二维时空矩阵 f(t,c) 循环交错映射到 C 个 chiplet——组内 ICT 留在本地 chiplet（降低 D2D 通信频率），组间交错保持流水并行；解决 naively-interleaving 每个 HMUX 都跨 chiplet 时 D2D 时延 > HMUX 计算时间导致的严重欠利用。
    ③ OIFS 离线调度器：用 Interleaved-Fusion Cost Model（IFCM：T_task = T_run + T_bubble，T_exe(t,c) = max(T_comp×|f(t,c)|, T_comm)）与动态规划（DP[j][r] 状态 + S_max/k_min 剪枝）对不同加密参数 n 求全局最优 f(t,c)，权衡 empty-slot penalty（固定融合组数留下的空槽浪费，如 n=17、C=4 时 3 个空槽）与 bubble penalty（过粗融合的启动/排空气泡），最小化总执行时间。
  - 对应解决 baseline 缺陷：串行 n×t_HMUX 吞吐天花板 → 跨 HMUX 流水线并行（参数集 I 吞吐 2,133,624 BSP/s，为 Morphling 的 ~14.5×）；集中式 HBM 并发 BSK 访问带宽崩溃 → 分布式 SRAM 驻留 + 本地化访问（BSK-distributed）；ICT 频繁导致 D2D 通信成为瓶颈 → 融合组内 ICT 本地化 + 组间交错隐藏 D2D 时延（OIFS 后 4.1× 额外提升）；固定融合映射的空槽/气泡 → OIFS 可变融合粒度 DP 全局最优（相对 SHM 减 bubble、相对 FFM 减 empty-slot，HC 利用率 90.7%→95.9%）。
  - 论文方法全栈执行例子（CASCADE 执行 DeepCNN-50，参数集 I，n=500，C=12）：
    ```
    算法pipeline层：同一 BSP 算法（LWE → n 次 HMUX → SampleExtract → key-switching），但 n 次 HMUX
               改为流水线并行；BSK（GGSW，参数集 I：n=500/N=1024/L=2/k=1）总量 ≤126 MB 全部驻留片上
    系统框架层：论文未明确说明（无 serving 框架；应用层 BSP 任务图由 OIFS 解析构造）
    编译框架层：OIFS——解析应用构造 BSP 计算图（同层并行、跨层串行）→ IFCM 建模 → DP 求最优 f(t,c)
               （500 个 HMUX 分成可变大小融合组、t mod 12 循环交错到 12 个 HC，T_exe(t,c)=max(T_comp×|f|, T_comm)）
               → 生成执行调度并指导 BSK 放置（离线、编译期完成，无运行时开销）
    kernel调度层：每 HC 内 Rotation→Decomposition→FFT→VMA→IFFT 系数粒度流水；融合组内 HMUX 输出
               回馈本地输入、intra-HC batching 注入多个 RLWE 避免气泡；HC 间经 UCIe D2D（1024 Gbps）
               环形传输 ACC，双缓冲隐藏 D2D 时延；n>C 时 RLWE 在环上多次循环；key-switching 由
               HC0 的 VPU 与 HMUX 流水并行执行
    硬件架构层：12 HC（4×3 网格、环形拓扑、UCIe 16 GT/s 1024 Gbps D2D、TSMC 28nm @1.2 GHz、
               单 HC 92.5 mm²/29.91 W、全系统 1170.1 mm²/372.72 W），每 HC 10.5 MB BSK SRAM
               本地驻留（BSK-stationary），一个 HMUX 时延≈最长流水级，稳态吞吐≈1/t_HMUX；
               DeepCNN-50 平均 2201.5×/48.5× 相对 CPU/Morphling 加速，Speedup/Area 30.5×/15.6×/3.1×
               vs MATCHA/Strix/Morphling，HC 利用率 95.9%、D2D 带宽利用率 76.8%
    ```
