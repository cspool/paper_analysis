---
tags: [调研报告, DiT, 多模态, kernel, 体系结构, 软硬件协同]
date: 2026-09-21
updated: 2026-09-22
version: v4（并入六层深潜 survey：181 条新增、43 条矛盾经对抗验证存活 14 条；D1 在性能轴关闭、D3 新颖性被 DiTPA 收窄、C3 主线迁往形态 C、全局改为 memory-bound 默认）
---

# DiT/多模态推理：性能优先的三类贡献模板 —— 可做点调研报告

> **v3 与 v1.x 的关键差异**：v1.x 的方向排序建立在网络文献与推断之上。v3 用 11 路 vault 扫描 + 154 条主张的对抗性验证（149 条确认、5 条部分推翻）重算了每个方向的 G1/C1/C2/C3。**四个方向的结论被本地数字改写，其中 D6 被证伪、D1 的核心前提被发现从未验证。**
>
> **v4 与 v3 的关键差异**：v4 并入六层深潜 survey（L1 算法 → L2 Serving → L3 编译 → L4 kernel → L5 硬件 → L6 芯片，36 问 × 一手 vault 证据）。35 个合并 agent 提出 **181 条新增结论与 43 条对 v3 的矛盾**，经对抗性挑战后 **14 条存活**（全部为 PARTIAL —— 收窄口径，无一条整体推翻 v3 的方向裁决）。**净效果：D1 从"待实验裁决"降为"性能轴关闭"、D3 保住首推但新颖性被一篇已发表 ASIC 论文收窄、D6 的关闭理由被换掉（原理由本身无一手证据）、C2 的论证口径全局改为 memory-bound + S-MFU/S-MBU、C3 主线从形态 A 迁往形态 C。** 凡与 v3 冲突处，以 §0.10 为准。

## 0. 论文模板（加速是核心评判）

> **因果链：较新的{任务 / 模型架构 / 硬件特性} = 原生加速潜力 → 魔改模型制造机会（C1）→【闸门 G1：潜力必须巨大，且必须用"简单实现改造后模块的原型 kernel"初测（原 kernel 是为原结构优化的，直接套用测不出新结构的收益）】→ 才继续谈精度几乎不降（G2，红线 ≤5%）与软件适配（G3：端到端、编译、运行时）→ 必要时硬件适配（形态B）。**

**闸门式评审（从便宜到贵逐级淘汰；加速是唯一的先决判据）**：

| 闸门 | 判据 | 评估成本 | 不过闸的处置 |
|---|---|---|---|
| **G1 加速潜力**（先决） | 理论加速**巨大**；**验证必做：为改造后的模块简单实现一个原型 kernel 并 microbench**——原型不求最优，量级对了即过闸 | 纸面核算 + 原型 microbench，天级 | 弃案或降级为组件（D6 即此下场） |
| **G2 精度几乎不降** | 微调/蒸馏后 **≤5%**，报精度-加速 Pareto | 训练卡时，周级 | 回改 C1 重过 G1 |
| **G3 软件适配** | 端到端仍达标：编译（图改写/JIT/CUDA Graph）、运行时/serving 集成 | 工程，周~月级 | 补形态A；GPU 吃不满则升形态B |

**卡数无关原则**：魔改收益是结构性的，单卡多卡都应体现；多卡是放大器。实验平台 = **Ada RTX 4090（现役）+ Blackwell RTX 5090（即将）**。⚠️ 见 §0.9 的硬件常量警告——**本报告中任何以 SMEM 容量定 tile、或以 vault 峰值算 MFU 分母的论证，在真卡实测前都不可写进论文。**

**"新近性 = 潜力"选题原则**：新硬件特性（NVFP4、TMA、96MB L2）、新任务（长序列视频生成、流式理解）、新模型架构（DiT、MoE-VLM、hybrid Mamba）——"新"即未开采潜力。单轴新=有空位；两轴新=潜力最大竞争最少；三轴全旧=只剩组件级贡献（D6 印证）。

**三类贡献**：C1 模型魔改+训练兜底（过 G1、G2 ≤5%）；C2 对应 kernel**必须论证最优**（compute-bound 报 MFU 对标 FA3/cuBLAS，memory-bound 报 HBM/PCIe 带宽利用率）；C3 软硬件协同（§0.5）。

> **⚠️ v4 全局改口径（§0.10.1）**：单请求 batch=1 时**算术强度 CI ≈ eff_bs ≈ 1，低于任何屋脊点** → **六个方向的 C2 一律按 memory-bound 论证**，v3 中"compute-bound 报 MFU"只在明确跑 batch/CFG 折叠的段落才成立。两条连带后果：(a) **SM 分区对 memory-bound 恒无效**（HBM/L2/crossbar 共享）→ 所有"门控与主干并发共跑"方案作废；(b) 稀疏/门控路径（D3 gate、D4 experts）必须用 **S-MFU/S-MBU**（激活参数口径），常规 MFU 在 batch>1 时高估资源利用 **1.5–3×**（MoE-CAP，探针开销 ≤2.7%）。另：单屋顶 roofline 在极端算力/带宽比的硬件上 MAPE 达 **127%（H800）vs 11%（H20）**，**4090/5090 正属该类** → 必须多流水线屋顶或同卡实测 SoL，**禁用纸面峰值**。

## 0.5 C3 的三种形态

- **形态 A｜运行时 + 编译集成**（OSDI/ATC/MLSys/EuroSys）：显存/缓存驻留、预取与 pipeline 重叠、SLO 调度、vLLM/SGLang/xDiT 集成；编译侧=图改写、JIT 特化、CUDA Graph 捕获。
- **形态 B｜专用体系结构**（ISCA/MICRO/HPCA/DAC）：新数据通路/PE 异构/近存 PIM，报 perf/area/energy。本地对标：DiTPA（ISCA'26）、DASA、RainFusion2.0、Stratum。
- **形态 C｜硬件感知闭环**（两条路线都必备）：硬件 cost model **反解** C1 的结构超参。本地已备三个现成反解式：D4 的 `ρ ≈ 2× active experts`、D5 的 `T_forget = 5.172·N_S − 4.469 (R²>0.999)`、D2 的"同 FLOPs 下 pre-fusion 层数 vs token 数"曲线。

**写作配方：C1 + C2 + 形态C（必备闭环）+ 形态A或B（按会场二选一）。**

> **v4 对三形态的重新定价（§0.10.1）**
> - **形态 A 有硬天花板 ≈1.2×**：Bullet 把 TTFT 做到 **54.9×**、SM 利用率 86.2%，**端到端仅 1.21×**。**任何方向若收益全在形态 A，即已被本地数据证伪**；形态 A 只能写成"可与 C1/C2 复合"，不得单独作 headline。
> - **形态 B 的承诺区间按瓶颈类型绑定**：片内重编排 **1.1–1.2×**／近存 **2–6×**／专用加速器 14–387×。**memory-bound 的方向（v4 后是全部）做形态 B 只能承诺 2–6×**，v3 引 DiTPA 386.93× 作暗示是跨类比较，须删。评审底线固定为 **iso-compute-area + DC/PrimeTime/CACTI**。
> - **形态 C 是本轮暴露的最干净空位**：全库**没有** DiT 侧统一 cost model（无 step-level PME、无 DiT roofline breakpoint、无任何 Timeloop/Accel-Sim 评估 DiT mapping 的笔记）。**建议把 C3 主线从 A 迁到 C。** D4 另补一个现成模板：DIAMoND 的 `OU: H=min(ρ_in,d_min), W=min(ρ_out,d_min·QB)` 一行反解结构超参。

## 0.6 Baseline 坐标系方法论

**论文坐标 =（任务, 模型, 硬件平台）三轴，都难改**；自由变量是"魔改+适配"。文献按重合度分四档：**三同**=主对比 baseline + 实验起点（fork 其代码）；**同任务·异模型**=借模型改法；**异任务·同架构**=借"任务特点→改法"的映射；**同结构·异硬件**=kernel 映射互鉴 + 形态B参照。差两轴以上只作背景引用。

## 0.7 Survey v2 合并结论（六层 Web 证据，2026-09-21）

`job_20260921T091901Z` 的六层（算法→Serving→编译→kernel→硬件→芯片）产出。**注意：该轮 L4/L5/L6 为 0 条本地证据**，原因是 Obsidian 服务中断 + 查询纪律问题（见 §0.9），非知识不存在。**→ v4 已用修复后的栈重跑该三层并拿到一手证据（L4 143 / L5 221 / L6 106 条），结论并入 §0.10；本节保留作对照。** 可用结论：

| 环节 | 硬证据 | 用处 |
|---|---|---|
| G1（方向1/3） | STA：HunyuanVideo 5s/720P 注意力 800s/945s≈**85%**；decode 实测 **1,490 launch/token、54 ms/token API 编排税、SM 利用率 4%** | 注意力占比与 launch 税量化 |
| C2 论证素材 | FA3 740 TFLOPS/75%、FP8 1.3P；**SFU exp 仅 3.9 TFLOPS（matmul 的 1/256）**；**FP32 累加使 FP16 吞吐腰斩 1929→965**；B200 FP4 7702 TFLOPS（仅作上界） | SOL 论证与精度-吞吐权衡 |
| C3-A 编译约束 | **Conditional CUDA Graph 与 cooperative/persistent kernel 互斥**；PDL 在 GB200 +15/+33% 但 **Ada 上 −5%**（须 arch-gate）；CUDA Graph ~200MB/张 | 编译适配设计约束 |
| 时间相干性同构先例 | **TIDE：扩散去噪相邻 step 的专家路由高度稳定，按 τ 刷新 +1.4–1.5×** | 方向1的同构证据（专家维成立） |

## 0.8 专业评审（依据 `human_notes/生成式模型原理/`）

1. **adaLN-Zero 结构**：每 block 一个 SiLU+Linear 从同一 t+c 嵌入回归 **6×hidden** 的 γ/β/α，且调制对所有 token 同函数 → 可折进 GEMM epilogue（D6 的 C1/C2 依据，但见 D6 裁决）。
2. **patchify 杠杆**：p 减半 → token 数 4×、FLOPs ≥4×，G1 核算必须把 patch size / latent 压缩比写进坐标固定项。
3. **⚠️ 最重要修正**：v1.x 写的"与少步蒸馏正交可叠乘"**表述过强**。DDIM 的 τ 子序列跳步说明：S 越小、相邻 α_τ 间隔越大 → 跨步相干性单调下降；少步蒸馏本身就在吃同一份时间冗余。改为"**部分正交，收益随 S 衰减，必须 50 步与 4–8 步双 regime 分别报告**"。（§0.9 证实：该衰减曲线 vault 里完全没有，是必测项。）
4. **CFG 双口径**：CFG 使 batch 翻倍的前提是未做 guidance distillation；G1 profile 必须分 CFG on/off 两种口径，并 pin `guidance_scale=1.0` 避免 server 默认双跑。
5. **训练网格对齐**：Wan 推理用 sigma shift，微调若按训练默认分布采样 t，学到的路由/门控会在推理 σ 网格上失配。

---

## 0.9 本地证据恢复（v3 核心，2026-09-22）

**方法**：11 路 vault 扫描（6 方向 + 5 横切）→ 每条主张由独立 agent 对抗性验证（默认 REFUTED，必须亲自打开笔记确认路径与引文）→ 按方向合成。**173 agent、154 条主张、149 条 CONFIRMED、5 条部分推翻、94 条自述缺口。**

### 0.9.1 三条全局性修正（影响所有方向）

**① ⚠️ 硬件常量表不可信 —— 本报告 v1.x 的 Blackwell 表需挂警告。**
本地只支持 **H100 228KB / A100 164KB** 两个 SMEM 数字。4090 的 128KB 出自 SageAttention 派生笔记；**5090 的 ~99KB 只出现在一份自身标注"OCR 需验证"的行里**；"数据中心→消费级递减"是合成趋势，不是笔记结论。两处未决冲突：(a) **sm_120 是否有 tcgen05/TMEM** —— survey 说无，QuTLASS/Bridging-the-Gap 说 RTX 5090 走 `tcgen05.mma` scale layout；(b) **4090 峰值 165 vs 330 TFLOPS 相差一倍**。
**处置：任何以 SMEM 定 tile、以 vault 峰值做 MFU 分母的论证，必须先在真卡上建常量表。**

> **v4 收紧（两处，见 §0.10.2 全表）**：
> 1. **"4090 SMEM=128KB"不是弱证据，是张冠李戴。** 逐笔核实：vault 中唯一写「L1 Data Cache / Shared Memory: 128KB」的笔记（`硬件知识笔记/Streaming Multiprocessor (SM…).md`）**上下文明写「Ampere 架构，RTX 3090 为例 / GA102 / SM ×82 / L2 6144KB」—— 这是 RTX 3090 的参数**。应从常量表**彻底删除，不得降级保留**。vault 中 **4090 的 SMEM 与 L2 容量各为零条**。
> 2. **连"参照点"本身也不唯一**：H100 的 SMEM 在 vault 里同时存在 **228 / 227 / 224 KB** 三个值，A100 同时存在 **164 / 168 / 192 KB** 三个值，全部出自 A 级一手笔记。**「消费级比数据中心小多少」这个相对论证从起点就是浮的**，引用前必须先选定口径并注明来源。

**② 上一轮"kernel/硬件层 0 证据"是双重误判。**
真实故障有二：Obsidian 渲染进程崩溃（已修，见附录），**以及 omnisearch 响应体超 token 上限（150KB）时工具报错——agent 把它当成了"证据不存在"**。实际该 query 返回 totalCount=50、truncated=true。纪律：加引号短语 + `path:`/`ext:` 过滤，或 `mode:text` 后取 filename。**不要再把检索失败当作知识缺失。**

**③ 六个方向在抢同一份冗余预算，此前无人核账。**
D1（路由缓存）、D3（跨步 rank-r 增量）、少步蒸馏**吃同一份跨步冗余**；D2（token 预算）与 D5（O(1) state）**吃同一份 token 数**。vault 里唯一相关实测是**警告**：QuantSparse 在稀疏+量化上叠 TeaCache，1.83→2.47× 但 **VQA 90.58→86.24、Delta-FScore 恶化 5.4×**（远超 G2 5% 红线）。**结论：六方向不能同时立项，必须先做冗余预算记账。**

### 0.9.2 一次实验结算四个缺口（最高优先级行动）

**实验**：Wan2.1-1.3B（480p, 5s）单次 instrumented 推理，逐步逐头 dump 池化后的 block 级注意力分数 P_c，计算 top-5% block 索引集在 step t 与 t+lag（lag=1,2,4,8）的 **Jaccard 重叠**，并在 **50 步与 10/4 步两种 schedule 各跑一遍**；同 run 开 torch profiler/nsys。

一次跑动结算：
- **(a) D1 的全部前提** —— 跨步路由 mask 重叠率。**低于 ~70% 则 D1 直接毙。**
- **(b) §0.8-3 的衰减曲线** —— 相似度随 schedule 长度变化（4 步 vs 50 步同 backbone），vault 完全没有。
- **(c) 路由 cache 显存足迹** —— N≈30K 时 M_c 为 469×469/头/层 × 30 层 × 12 头，直接算出能否驻留 4090 的 L2。
- **(d) 免费副产品** —— DiT block 逐算子延迟分解 + 每步 kernel 数，这正是 **D6 的生死判据**与 launch-bound 比例，并给 D1/D3 一个本地 attention 占比（Amdahl 上限）。

**代价：一个下午，无需训练、无需 5090、无需多卡。没有第二个实验能以同样成本解锁这么多方向。**

### 0.9.3 六方向裁决速查（**已被 §0.10.3 取代，保留作 v3 记录**）

| # | 方向 | v1.x 判定 | **v3 本地证据裁决** | 新优先级 |
|---|---|---|---|---|
| 1 | 跨 step 路由缓存 SLA++ | ★★★ 首推，E2E 2.5-3× | **G1 头条崩塌**（SLA 已吃掉 61.5%→残余 15.3%，路由清零仅 ~1.06× e2e）；**复用轴错了**（>92.4% 重叠是空间的，跨步 Jaccard vault 里没有）；新颖性被 InfiniteHiP+SSAR 夹击 | ★★（待实验裁决） |
| 2 | 弹性视觉 token 压缩块 | ★★☆ | **G1/G2 本地最强**（77% FLOPs、1/4/16/64 token 内 <1 点）**但创新点被证伪**（VisionSelector 已发表单权重多档、AIM 训练无关 7 档 40×）；只剩机制差异+C2 | ★★☆ |
| 3 | block 级三态门控 | ★★☆ | **三根支柱齐全**（SLA 证三态融合单 kernel 可行且在 5090；LoRunner 低秩分支融进 quant+GEMM 仅 +5%；FlashInfer 让逐步门控对 CUDA Graph 安全）；**无一条本地证据反驳方向本身** | **★★★ 首推** |
| 4 | 模态分组 MoE + 预取 | ★★☆ | G1 势能真实（0.1% 利用率、88% 时间在 expert fetch）；**但"prefill 已知工作集"和"命中率"两条卖点被占满**；**必须删 shared 档**、改以"传输字节"为战场 | ★★ |
| 5 | 视觉历史层蒸馏 Mamba | ★☆☆ 二期 | **动机被削弱**（8 FPS 是 StreamingVLM 自己的吞吐而非滑窗天花板）；**1:7 混合在 100K token 上内存仅 79.4→79.1GB**；SSM 上 token reduction 灾难性失败 | ★☆☆ 二期 |
| 6 | AdaLN 低秩共享+融合 | ☆ 组件（G1 不过） | **本地证据证伪，且方向性相反**：attention 占 92–95%、移除整条调制路径只买 29% 参数且无延迟数字、torch.compile 已自动融合、最相似实例 HATA 仅 7.6% e2e | 结案 |

---

## 0.10 六层深潜合并（v4 核心，2026-09-22）

**来源**：六层 survey（L1 算法 → L2 Serving → L3 编译 → L4 kernel → L5 硬件 → L6 芯片，36 问）跑在修复后的 Obsidian/MCP 栈上，**首次拿到 L4/L5/L6 的一手 vault 证据**（此前三层为 0；本轮各层一手引用 L1 51 / L2 175 / L3 60 / **L4 143 / L5 221 / L6 106**，且**无任何一条答案再声称"无 note evidence"**）。随后 35 个合并 agent 按 7 个 scope（D1–D6 + 横切）做"新增 + 找矛盾 + 对抗验证"：**181 条新增结论、43 条声称与 v3 矛盾、经挑战后 14 条存活，全部为 PARTIAL**（收窄口径，无一条整体推翻 v3 的方向裁决）。

### 0.10.1 五条全局裁决（横切 scope，改写 v3 表述）

1. **C2 的 memory-bound 化从局部升为全局**（原只对 D1/D5 要求）：batch=1 → CI≈1 → 全部六方向按 memory-bound 论证；分母换 S-MFU/S-MBU + 同卡实测带宽；删除所有依赖 SM 分区 / 并发共跑的方案。详见 §0 的 v4 全局改口径框。
2. **A100 带宽必须按 SKU 进 roofline 分母**：v3 把数据中心 L2/HBM 常数的分歧当"矛盾"，实为 A100/H100 口径混写。正确写法：**A100-40GB 1555 GB/s → 屋脊 ≈208 FLOP/Byte；A100-80GB 2039 GB/s → 屋脊 ≈153**。CI 阈值写「208 (40GB) / 153 (80GB)」，**不写单一 208**。
3. **batch=1 单卡前提保留，但须标注外推**：芯片级本地无 batch=1 专用实验环境，最近证据是 Duplex 交叉点（**B<32 PIM 优、B>64 GPU 优**），Raptor 的硅后建模把稳态 batch 定在 **<32、平衡点 32**（Poisson 110 req/s）。**batch=1 是我们框架里支撑最弱的一环**，写作时必须标为"单请求在线推理"场景而非普适设定。
4. **CUDA Graph 在少步 DiT 下是必需项而非可选**：launch 5–20 µs/kernel → Graph replay 2–5 µs（省 90–98%），但**每张图约 200 MB 显存** → 24GB 4090 上**可捕获的门控组合数必须先数**（这是 D3 的直接约束）。
5. **launch overhead 不能扛 G1**：本地无一手的 launch 占比数据，且此类增益天然落在 1.1–1.2×（形态 A 天花板）。**只能作 C3-A 组件，不能作 G1 论据。**

**实验协议（写死三条，全方向适用）**：(a) 计时的 `synchronize` 必须在计时区**内**，否则 CUDA Graph 会报出 ~1000× 的假加速；(b) 四层消融 A/B/C/D 必须**分离 graph-capture 增益与 fusion 增益**；(c) **50 步与 4–8 步两套数并报，`guidance_scale=1.0`**（否则加速与"打开 CFG batching"不可区分）。

### 0.10.2 可信度分层硬件常量表（4090 / 5090）

证据等级：**A=笔记显示**（vault 一手正文直接写出）｜**B=可推断**（两条以上笔记推出，无笔记直接陈述）｜**C=派生/Web**（上一轮 job 输出或其引用网页，自标"无 note evidence"）｜**D=无证据**。

**RTX 4090（现役）**

| 项目 | 值 | 等级 | 备注 |
|---|---|---|---|
| SM 数 | **128**（AD102，boost ~2520 MHz） | A | 两处独立笔记一致（MxMoE tile 调度交叉印证） |
| threads/SM | **1536** | A | μShare blocksize shaping |
| 显存带宽 | **1008 GB/s** | A | 两处一致；但笔记误写作 "HBM"（实为 GDDR6X），需 bandwidthTest 复核 |
| 显存容量 | **24 GB** | A | 多篇 offloading 笔记一致 |
| 峰值算力 | **165 TFLOPS（BF16）vs 330 TFLOPS（FP16）—— 冲突未决** | A/A | 见下「冲突 B」 |
| FP8 支持 | **有**（`mma.f16.f8.f8.f16` 自 SM 8.9 起）；FP8+FP16 累加 4×、+FP32 累加仅 2× | A | SageAttention2++ 的 P×V 路径 |
| Sparse Tensor Core | **有** | A | Samoyeds Table 1 |
| NVLink | **无**（token 交换须经 PCIe + host） | A | 直接决定多卡形态 |
| FP4 / NVFP4 | **无正面证据**（所有 FP4 笔记的硬件列表只含 Blackwell，从不列 4090；亦无明确否定句） | D | 沉默证据 |
| **shared memory / SM** | **vault 中零条** | **D** | ⚠️ 流传的 128KB 是 **RTX 3090/GA102** 参数被错挂 |
| **L2 容量** | **vault 中零条** | **D** | ⚠️ §0.9.2-(c)「路由 cache 能否驻留 4090 L2」**目前没有分母** |

**RTX 5090（即将到位）**

| 项目 | 值 | 等级 | 备注 |
|---|---|---|---|
| 架构 | Blackwell **SM120** | A | QuTLASS 评测卡（与 B200 SM100 并列） |
| 显存容量 | **32 GB**（笔记实为 **5090D** 中国特供版） | A（卡型有别） | LOGART 量化实验平台 |
| FP4 实测 | 单层 MXFP4 ≈**6×**（理想 8×）、端到端近 **4×**（vLLM 内测，Llama-3.3-70B 层形状，batch 1–256） | A | QuTLASS，独立于下一行 |
| FP4 峰值 | ≈**1600 TOPS**（FP16 ≈200 TOPS）；SageAttention3 实测 1038 TOPS | A（载体笔记已证伪一处） | 同页把 5090 错标为 SM100 → 数字不可单独取用 |
| 端到端锚点 | **SLA：Wan2.1-1.3B / 5090 / 95% 稀疏，注意力 97s→11s（8.8×）、e2e 2.2×**；FW 13.7× / BW 6.8× vs FA2；超参 k_h=5%, b_q=b_kv=64；微调 2000 步 × batch 64 @480p | A | **这是全表最可直接复用的口径** |
| kernel 占比锚点 | fused reorder-and-quantize 仅占总 kernel 时间 **7.9%–17.0%**（seqlen 128→4096），GEMM 占 83.0%–92.1% | A | MicroMix |
| tcgen05 / TMEM | **冲突未决**（见下「冲突 A」） | — | — |
| shared memory / SM | 传闻 **99 KB**（另处 101,376 B，147KB 配置报 `OutOfResources`） | **C** | 自标"无 note evidence"，且实测卡是 **5070 Ti / PRO 6000 / GB10，不是 5090** |
| SM 数 / 带宽 / L2 | **vault 中各为零条** | **D** | — |

**冲突 A：sm_120 上是否有 tcgen05 / TMEM。** 主张"有"为 **A 级**、直接点名 RTX 5090+SM120（QuTLASS 用 Triton kernel 为 `tcgen05.mma` 做 scale 重排），但**该证据自身有裂缝**：笔记从未逐卡拆分 kernel 路径（"matmul 支持 CUTLASS/FlashInfer 多后端插拔"），**无法排除 tcgen05 路径只走 B200**。主张"无"为 **C 级**且卡型不对（实测 5070 Ti / PRO 6000 / GB10，`INVALID_PTX`、退回 `mma.sync.m16n8k32`）。另有三条笔记把 5090 归进 **SM100**，与 SM120 直接矛盾 → 该族笔记在"消费级 vs 数据中心 Blackwell"上没有分卡，**两边都不能用**。**唯一出路：真卡上跑一条 `tcgen05` PTX 探针。**

**冲突 B：4090 峰值 165 vs 330 TFLOPS。** 三条 A 级文本互斥：①「peak BF16 165 TFLOPS」（ZipServ roofline，据此算屋脊 164 FLOP/Byte）；②「FP16 330 TFLOPS non-sparse / INT8 660 TOPS」（AD102 理论规格）；③「FlashAttention2 在 FP16 峰值 330 下**仅达 165 = 50%**」（headdim=64 **实测**）。两条重建路径都只到 B 级、**不得升格**：**α 累加器差异**（FP16+FP16 累加 512 FMA/SM/cycle 是 FP16+FP32 累加 256 的 2×，则 165/330/660 三数自洽）；**β 把实测当峰值**（③ 明写 165 是 achieved 值，与 ① 的 peak 用法字面冲突）。旁证：同为 SageAttention 派生的两篇笔记对 INT8:FP16 比值给出 **4× 与 2×** 两个答案；5090 笔记的「FP16 ≈200 TOPS」**低于** 4090 的 330，跨代不单调 → 指向"两代笔记用的不是同一累加器口径"。

**⚠️ 13 格 UNRESOLVED —— 写任何 tile/MFU 论证前必须实测**
*4090（今天就能测）*：① SMEM/SM 上限（`cudaDevAttrMaxSharedMemoryPerBlockOptin`，**全表最危险的一格，它直接约束 tile 尺寸**）② L2 容量 ③ 峰值 TFLOPS 的**六格口径矩阵**（BF16/FP32-acc、FP16/FP16-acc、FP16/FP32-acc、FP8/FP16-acc、FP8/FP32-acc、INT8）④ FP4 是否可用 ⑤ 实际带宽。
*5090（到货第一天）*：⑥ tcgen05/TMEM 探针 ⑦ SMEM/SM（含 147KB 是否真 OOR）⑧ SM 数 ⑨ 带宽与容量 ⑩ L2 ⑪ 各 dtype 峰值 ⑫ cluster multicast 是否退化为 1×1×1（决定 TMA multicast 类优化在本平台是否整体失效）。
*跨卡*：⑬ 统一 H100（228/227/224）与 A100（164/168/192）的口径。
**可用的临时替代**：A40 实测 SMEM=102400 B、QuantCache 实调 48 KB/SM、PCIe 硬线 32–64 GB/s；**原型实现走 TileLang 最省**（已在 4090 验证，~70 行达 FA3 的 98%）。

### 0.10.3 六方向 v4 裁决（替代 §0.9.3）

| # | 方向 | v3 判定 | **v4 净改动** | 优先级 |
|---|---|---|---|---|
| 1 | 跨 step 路由缓存 SLA++ | ★★（待实验裁决） | **性能轴关闭**：同一 SLA 实验给出更硬的上界——**把整段 attention 归零，e2e 仅 ≤1.18×**，连带否掉其形态 B 分支。技术资产转移给 D3 | **结案**（资产转 D3） |
| 2 | 弹性视觉 token 压缩块 | ★★☆ | 硬件轴也被占（**Focus** 已把 token 重要性+top-k 做成加速器原语）；标杆上移到 **MoDES 2.03× prefill**；map-free 由偏好升为 **G1 硬门**；删掉省显存叙事 | ★★☆ |
| 3 | block 级三态门控 | ★★★ 首推 | **保住首推，但新颖性被 DiTPA（ISCA'26，开源）收窄**——它已发表 per-step {full｜skip｜reuse} 三态门控；基线由 4.12× 抬到 **6.72×**；支柱三（FlashInfer graph-safe）必须改写；**C3 主线迁往形态 C** | **★★★ 首推** |
| 4 | 模态分组 MoE + 预取 | ★★ | 目标函数首次闭式化：**min a_max**（T_moe=β·a_max+c_e）；"删 shared 档"**限定为架构级共享专家**，运行期常驻层在字节轴上是**收益不是代价**（STEP −50%/步） | ★★ |
| 5 | 视觉历史层蒸馏 Mamba | ★☆☆ 二期 | **换主线**：SSM kernel 的 C2 头寸≈0（Inductor 10.57 vs 手写 10.81），改押 **StreamingVLM 式 sink+非对称短视觉窗+Contiguous RoPE**；G1 分母须先扣掉 **~40% 预处理** | ★☆☆ 二期 |
| 6 | AdaLN 低秩共享+融合 | 结案 | **结论不变，但理由被换掉**：v3 的"torch.compile 已折进 epilogue"**在本地无一手证据**，必须删；新理由 = 带宽饱和上界 + **FG1–FG4 融合组已公开** | 结案 |

---

## 方向 3（新首推）：DiT block 级可训练三态门控 {重算 | rank-r 增量 | 复用}

**坐标**：视频/图像生成 / Wan2.1、FLUX / Ada 4090 → Blackwell 5090。**新轴**：架构新——U-Net 时代的跨步缓存栈需为 DiT 重做。

**为什么升为首推**：这是六个方向里**唯一没有任何一条本地证据反驳方向本身**的；被反驳的是两种朴素替代路径。三根支柱与三道硬约束都已量化。

**G1（门槛被抬高，不是 1.0×）**
- 支柱：图像 DiT 相邻步相似度**只在近对角高、非对角迅速降为 0**（`knowledge_notes/算法知识笔记/交替去噪特征复用…md`）→ 二态缓存的有效窗口天然短，**rank-r 第三态的作用正是把缓存寿命延长到近对角窗口之外**。
- **硬门槛**：QuantCache 叠加阶梯（Open-Sora 1.2、100 steps、A800）1.00× → **+HLC 4.12×** → +AIGQ 6.33× → +SRAP 6.72×。**D3 的原型必须在同等设置下超过 4.12×，而不是超过 1.0×。**
- 预算分配：**SRAP 只值 +0.39×（6.33→6.72）**，步内层剪枝不该进 D3 堆栈，预算全押"把跨步决策做得更好"。

**C1（结构被一条已发表的反面理由钳死）**
- **rank-r 增量不能穿过非线性**：DiTPA 明确以"细粒度控制开销 + 对 GELU 等非线性不兼容"区分于 Cambricon-D/Ditto 差分与 EXION 细粒度稀疏。**必须二选一并写清**：(a) 增量只加在线性子块（QKV/FFN GEMM），非线性重算；(b) 让可训练门控吸收非线性误差。
- 现成的可替换 baseline：HLC 是**三级 refresh 决策**（τ_max/τ_mid/τ_min）但计算模式仍是二值 {recompute|reuse}，阈值 δ₁/δ₂ 固定、不训练。**D3 的新颖性正是 (a) 第三种计算模式而非第三个间隔，(b) 用学习替代手工阈值。**
- 门控输入信号现成且便宜：SRAP 的相邻层特征余弦相似度 runtime 可算。
- **最强 pitch**：QuantSparse Tab.16 证明"在量化+稀疏之上叠固定阈值缓存"代价惨重（1.83→2.47× 换来 VQA 90.58→86.24、Delta-FScore 5.4× 恶化）。**D3 = 在同一 2.47× 上把质量收回来。**

**C2（被两个数字收紧为可证伪命题）**
- Roofline（A40 BF16，breakpoint ᾱ≈215）：规则 rank-r 把 FLOPs 34G→17G 而 **α≈978 仍远在 breakpoint 之上**（对比 Monarch 123、BLAST 64 已掉进 memory-bound）→ **"rank-r 增量优于不规则列稀疏 delta"有 SOL 级依据**。
- 但崩点是 **b×n×r 中间张量物化**：C2 的最优性**只在 Δ 不回写 HBM 时成立**。LoRunner 已给实现路径（3 launch→1、激活读取 3×→1×）与代价表（rank=16 时 +5%）。
- **可证伪目标**：在所选 r 下，三分支融合 kernel 相对 full recompute 净收益 **>5%**，且 profiling 须显示 Δ 全程驻留 shared memory/寄存器。
- 控制开销预算：MixFusion 给出 **<2ms/block**（SD3 每 step 40–50ms、24 blocks）→ 门控只有约 1/24 step 预算，**必须批量化、不能 host 往返**。

**C3（形态A 成立，且唯一安全实现已知）**
- 逐块逐步决策 = 数据相关控制流，常规写法会改 graph 拓扑触发完整重 capture（本地记录 **Brax 的 DAG 构建占 47% 执行时间**）。
- **唯一干净解**（FlashInfer 模式，可原样照搬）：CPU 侧 plan（<1ms/step，在 Graph 外）写**固定 workspace 偏移**，固定 grid 的 persistent kernel 内按 flag 走三分支 —— launch 参数恒定，Graph 只 capture 参数不 capture 数据。
- 最近先例 SLA 同时为融合 kernel 与训练回退提供模板：三种复杂度融进同一 forward kernel + 一个 backward kernel，微调 **2000 steps × batch 64（<0.1% 预训练成本）**，平台正是 **RTX 5090**。差异需辩护：SLA 门控单步内 attention 的 block pair，**D3 门控跨步的整个 DiT block —— 轴不同，属可建之基**。
- **D3 不受 SLA 的 Amdahl 上限约束**：SLA 8.8× attention 只换 2.2× 端到端，说明纯 attention 优化饱和；D3 在 block 粒度覆盖 attention+FFN+norm。**这是"块级而非注意力级门控"的具体论据。**

**本地缺口**：Chipmunk/TaylorSeer/ToCa/DisCa/LearniBridge **全 vault 零命中** —— D3 的核心论断没有本地对手可比，必须自行补抓并复现。

### 方向 3 · v4 增量（六层深潜）

**三条必须改写的 v3 句子**

1. **⚠️ 新颖性被一篇已发表论文收窄（最重要的一条）**：v3 写"无一条本地证据反驳 D3 方向本身"——**已不成立**。**DiTPA（ISCA'26，开源，28nm / 4.37 mm² / 1.05 W）的控制结构就是逐去噪步的三态判定 {完整推理 | 跳过整轮 | 复用上轮输出}**，叠列稀疏、跑在可重构 PE 阵列上（S1 1.74× → +S2 2.90× → +S3 32.60×，均相对其自身 ASIC 基线）。差异是它做的是**动作规划 DiT，不是文生视频**，且是 ASIC 不是 GPU。**D3 的新颖性因此只剩三条，必须在 intro 里逐条挑明**：(a) **中间态是 rank-r 增量，而非特征复用**；(b) **门控可训练**，而非固定阈值；(c) 在 **GPU 文生视频**同设置下超过现有栈。**好消息**：DiTPA 的门控硬件成本仅 **0.23% 面积 / 0.05% 功耗、4 cycle 判定** → v3 担心的"门控面积异议"消失，形态 B 可行；其结构规则"沿门控**不改变**的维度（hidden size）优先 tiling"可直接借用。
2. **支柱三（FlashInfer 让逐步门控对 CUDA Graph 安全）必须改写**：graph 安全性**不来自 persistent kernel 本身**，而来自「**plan/run 分离 + 固定 grid + workspace 固定偏移**」——数据依赖的 `attn.plan()` 在 **CPU 上执行且刻意不被捕获**（1×H100 / Llama-3.1-8B / CUDA 12.4 + PT 2.4.0）。CUDA Graph 的硬约束是**没有条件 kernel**、grid/block/SMEM 全固定、多形状需多份图实例；MoE 动态路由触发的重建代价 **40–100 ms/次**（ACS：input-dependent 图上 DAG 构建占总执行 47%）。**推论：把三态实现成三个分别 launch 的 per-state kernel 按步选择，不是 graph-safe 的。** D3 的门控合法当且仅当：host 预算好经预分配 device buffer 喂入，**或**单个固定 grid 的 persistent kernel 内做数据依赖分支。
3. **"graph-safe" ≠ "变长工作代价可忽略"**：DiTPA 实测**同类纯软件 GPU 实现只有 2.3×**，并点名两项代价——**动态 token 长度使利用率降 25.1%、多模态数据管理增 35.4% 时延**（其硬件协同版 PE 利用率 98.36% vs GPU <20%）。这是 D3 在 GPU 上最现实的风险。

**新增结论**

| 槽位 | 结论（数字 + 条件） | 作用 |
|---|---|---|
| G1 | QuantCache 全消融 1.00 → HLC 4.12 → +AIGQ 6.33 → **+SRAP 6.72×**（单 A800-80G，Open-Sora 1.2，512²×64f，100 步，W4A6，含 VAE） | **基线从 4.12× 抬到 6.72×**；AIGQ/SRAP 两条轴已被占 |
| G1 | Flux-dev 12B 每步读 ~24 GB 权重，H100 3.35 TB/s → **7.2 ms/步纯权重**；CFG batch 折叠把 TC 利用率 50%→80%+ | C2 的 memory-bound 分母；这 ~2× 必须在测 G1 **之前**先吃掉 |
| C2 | **HyTiS**：多模式共存于一个 persistent kernel，**共享同一 SMEM layout、仅 tile 几何不同、双射 offset 切换**，零 sync / 零 workspace；1.10–1.19× vs cuBLAS（H100/A100，4624 个 GEMM） | 三态融合的**第二个独立可行性证据 + 现成设计规则**；也给出"模式切换本身"的诚实收益量级 |
| C2 | FusedGemmAdd 在 epilogue 内累加 vs 独立 add kernel：**3.12–3.86×** | "delta 不得落 HBM"首次有数字支撑 |
| C2 | 双缓冲占**一半**片上容量；FA3 三级流水因寄存器溢出**劣于**二级（256 KB/SM） | **r 的上界直接减半**，且三态必须共用一套寄存器预算 |
| C2 | 核内数据依赖门控的成本：MPK 占 4 SM（~3%）；FlashMoE 1 block 仍保 93.17% SM 利用；Batched Load Scheduling 的决策态仅 512 B SMEM | 门控可负担，**但预算是个位数 %，紧贴 >5% 净收益门槛** |
| C2 | **Ada 无 thread-block-cluster DSM** → FlashFuser 的 reg→SMEM→DSM 溢出层在 4090 **不存在**；Triton 3.5.0 小批仅 0.67×/0.66× | H100 融合论文的 C2 论证**不可直接移植**；实现优先 TileLang/CuTe |
| C3-A | Difflow 编译侧 DiT 基线 1.58× 均值 / 2.13× max（vs PyTorch-Inductor）；**batch=1 下只剩常量折叠 + 循环不变量外提**；属性特化不分区则 2ⁿ 引擎（14 输入 = 16384 引擎 / 11 天） | C3-A 必须正交于此；三态应走 **Brainstorm 式"编译期枚举 3 变体 + 运行时查表"** |
| C3-A | TetriServe 调度粒度扫描：**τ=1 步"开销过大"，5 步为稳健最优**（FLUX.1-dev, 8×H100） | 门控要么完全在 kernel/graph 内不碰调度器，**要么按 ~5 步粘滞**（则 C1 设计随之改变） |
| C1 | **FlashPS**：扩散连续批内**同一步所有请求走相同块数**，join/leave 只在步边界 | 门控必须是 **batch 级单掩码**，否则退化为 batch=1 |
| C1 | 本地唯一被 kernel 实测的低秩增量是 **Q-VDiT TQE，rank=1**，且承载的是量化误差修正；校准 10k/15k iter，W8A8 单项 **12.5–12.9 GPU-h** | 训练回退须预算 **≥12.5 h/配置**，并论证 **r>1 为何可负担** |
| C1 | DiTPA：**60.1% 外存访问是重复权重加载**（动作规划 DiT，10–50 步，单 die） | 提示第二条复用轴——只管激活的门控会漏掉这 60.1% |
| C3-C | 全库**无** DiT 侧统一 cost model（无 step-level PME、无 DiT roofline breakpoint、无 Timeloop/Accel-Sim 的 DiT mapping） | **本轮暴露的最干净空位 → C3 主线迁到这里** |

**对 v3 的净改动**
- **基线与协议**：D3 的 >5% 净增益改成对 **6.72×**（而非 4.12×）计；测 G1 前先 pin `guidance_scale=1.0`、打开 CFG batch 折叠，并分别报 **50 步与 4–8 步**两档。采用 **Open-Sora 1.2 标准设置**以对齐四篇已发表数字。
- **C2 的 r 上界重算**：可用片上预算 = SMEM 的**一半**（双缓冲）∩ 三态共享的同一寄存器集（FA3 三级反例）∩ Ada 无 DSM 溢出层 —— **先算出 r_max，再谈 α≈978 vs 屋脊 215**。从 Triton 起步意味着 0.67× 的起跑线，且须按 ARGUS 口径**一次性叠齐 pipelining + warp specialization + 指令调度**，否则原型会被测成负收益。
- **C3 主攻改向**：不得再宣称"使动态 per-step 决策 CUDA-Graph-safe"（PROBE 五件套已发表），只能论证"**DiT 门控提前一整步可知，结构上比 MoE routing 更易**"；真正空缺的是 **C3-C（DiT 硬件 cost model 反解 C1 超参）**。

## 方向 1（v4 结案：性能轴关闭）：视频 DiT 跨 step 路由缓存 SLA++

**坐标**：视频生成 / Wan2.1-1.3B / **RTX 5090（SLA 自己的评测平台，与我们即将到位的卡一致）**。

**G1 头条崩塌（v1.x 的 2.5-3× 是错的）**
- SLA 实测（Wan2.1-1.3B、95% 稀疏、RTX 5090）：注意力 **97s→11s（8.8×）**、e2e **2.2×** → 推算总时长 ≈157.7s，**注意力占比 SLA 前 ≈61.5%、SLA 后仅 11/71.7≈15.3%**。
- 把 routing 开销折进这 15.3%（Twilight 实测 TokenSel~15%+Pruner~20%≈**35%**；MInference Block-Sparse 索引构建 **25%**）：**即使 routing 开销全部消除，e2e 仅 1/(1−0.153×0.35)≈1.06×**；把注意力整段清零也只有 ≤1.18×。
- **处置**：G1 只能以"注意力 kernel 上 ~1.5×（35%→0）"立论，**原型必须对着 52.75T→2.73T 的单步绝对预算测，不要报 e2e**。**替代出口**：用省下的 routing 预算把稀疏度推过 95%，收益重新变成结构性的 C1 论点，不再受 Amdahl 封顶。

**C1 的轴错了 —— 必须从步轴改到 query 轴**
- **反驳跨步**：图像/视频 DiT 相邻步相似度只在对角附近高、非对角降为 0；>98% 全矩阵高是**动作规划 DiT** 的性质，不可迁移。
- **vault 中唯一的硬复用数字在 query 轴**（DSV Obs 5）：同一 **2×2×2 3D latent cube 内相邻 query 与锚 query 的 critical-KV 索引重叠 >92.4%**，跨 block 均值 80.1%。
- **⚠️ 关键发现：跨去噪步的 top-k block mask Jaccard，vault 里根本没有。D1 一直在靠一个从未验证的前提。** → §0.9.2 的实验就是为它设计的。
- 可辩护形态应改为：**每个 2×2×2 latent cube 只算一份 P_c 并 per-head 共享**，跨步缓存降级为可选的小 interval 增量项。
- 三态容差不对称（支持"陈旧决策优雅退化进线性补偿分支"）：仅 8.1% 权重大于均值 1/N、约 45% 低于 1/(100N)；丢弃最小 45% 相对 L1 误差 **<3%**，仅保留最大 8.1% 误差升至 **≈33%**。

**新颖性被双向夹击**
- InfiniteHiP **已实现**分 stage refresh interval 的 mask 缓存：默认 n_refresh=(16,8,4)、256K 上下文 **110 µs/token**、命中率 Stage1 **71.67%** / Stage1&2 **98.75%**。
- DiTFastAttn/SSAR **已占**跨步注意力图残差复用：SSAR 二阶残差、τ=5、SVD rank 16、开销 +0.2% 时间/+8–11% 显存；Wan2.1-14B W4A8 PSNR 14.16→17.08→**18.68**→18.72。
- **剩下的窄缝**：(a) 轴从 token/decoding 换到 diffusion-step × 3D-cube 位置；(b) 陈旧路由**优雅退化进线性补偿分支**而非丢 token（由上述容差不对称支撑）。

**C2 必须改成 memory-bound 叙事**
- 线性分支已 **<0.5% 全注意力开销** → 缓存其 KV 状态在 FLOPs 上买不到东西，只能主张省掉 φ(K)ᵀV 预计算的 HBM 流量。
- DiTPA 佐证收益本质偏访存：**60.1% 重复外部访存**。
- SOL 主轴 = **HBM 带宽利用率**；MFU vs FA3 只作 2.73T 稀疏主体那段的辅证。正确性论证落在 `Block-Sparse Attention Pattern` 笔记的 **MeanPool(Q)·MeanPool(K)ᵀ ≈ MeanPool(QKᵀ)** 恒等式。
- ⚠️ **DSV 里没有可借用的 fused routing kernel**（只有 3 步概念伪代码）→ **D1 的 C2 必须从零写**。

**最危险的对手不是"没人做过"，而是 routing 成本为 0 的静态 mask**：SSAR 自己的 mask 就是 SVG 的静态时空模式。DSV 的"critical KV 无局部性"（仅 15.1% 在 5-token 半径内、48.5% >10 token）是我们反驳它的唯一弹药，但**vault 里没有把 SVG-static 与 computed-routing 直接对齐的质量数字 → D1 必须自带这条对照实验**。误差纪律沿用 DiTPA：每 20 次跳过插入一次完整去噪复位累积误差。

### 方向 1 · v4 裁决：与 D6 同样在性能轴关闭

**首要裁决**：v3 把 D1 定为"headline 崩塌、但可另寻角度、待实验裁决"。**v4 改为性能轴关闭。** 理由是同一个 SLA 实验（Wan2.1-1.3B / RTX 5090 / attention 97s→11s = 8.8×、占比 61.5%→15.3%）除 v3 已引的"归零 routing 开销 ≈1.06× e2e"外，还给出更硬的上界：**把整段 attention 归零，e2e 也只有 ≤1.18×**。这意味着**任何 D1 家族方案——包括其形态 B 专用 attention 加速器分支——在 SLA 基线上都过不了 G1**。形态 B 分支同时被第二条证据否掉：本地**无任何 video-DiT 加速器评测范式**（Timeloop / SCALE-Sim / NoC 零命中），方法学本身就是净新增工作量。

**连带修正（PARTIAL，收窄但未推翻）**：三态稀疏+线性 routing kernel 在访存轴上的"最优"论证目前**仍是解析性的**。唯一本地测量（XY-Serve，Ascend 910B）显示前缀复用使 score 矩阵由方阵变矩形后 **torch-npu MFU 53% → 47% → 30%**，论文明言**理论算力节省被 kernel 效率损失抵消**；GPU 侧无同类多态 routing kernel 测量。另一条同向反例：block 顺序 **{N,H,B} 得 600 TFLOPS @ 213 GB/s，而 {B,H,N} 得 494 TFLOPS @ 2390 GB/s** —— 实测算力与带宽**朝相反方向移动**。**结论：v3 的"融合后 routing 开销归零"必须实测，不能假设。**

**应转移给 D3 的技术资产（D1 唯一的存活价值）**

| 约束 | 内容 |
|---|---|
| 索引形态 | DSV：top10% KV 贡献 >90% 分数，但仅 15.1% 落在 5-token 半径、48.5% 距离 >10 → **必须索引表 gather，不可假设连续** |
| top-k 形态 | in-kernel 迭代 top-k 需全局同步 → 须改写为 **KV 序列维归约**（InfiniteHiP / FA2 不 split-K） |
| 粒度契约 | routing block size **必须整除融合 kernel 的 tile**（NSA=64 对齐 FA2）——是编译期契约，不是自由变量 |
| 掩码形态 | **TMA 不支持非仿射访问**，稀疏 BSR 会退回 `cp.async` → 掩码须保持仿射块结构 |
| 合格线 | 访存型 kernel：**A100 83–86% / H100 92.3–94.2% HBM 带宽**；寄存器溢出会跌到 40.3% → 最优性须过 ~85% |
| 分母纪律 | 92–95% 的 attention 占比是 **8×H100、200K token、前向+反向训练**值，**不是单卡推理** → 引用必须标注条件，单卡分母须自测 |
| 对比基线 | FlashPS 对 TeaCache **6×**；每步 batch formation **1.2 ms** → serving 场景下的每步预算 <1 ms |

## 方向 2：多模态 LLM 弹性视觉 token 压缩块

**坐标**：图文/视频理解 / LLaVA-OneVision 或 Qwen2.5-VL / 4090+host。

**G1/G2 是六方向中最强的一环，但也正因此成了创新性风险**
- G1：LLaVA-Mini 摘要 **FLOPs −77%、<40ms、24GB 单卡 >10,000 帧**；token 预算算术 **32 帧×576=18,432 → Adaptive Pooling ~2304**；**FLOPs ∝ (N_v+M)²** → 8..256 弹性区间是二次型杠杆。
- G2：336 分辨率下 **1/4/16/64 token 的 VQAv2 = 77.6/77.7/78.1/78.5**，对比 576-token 的 78.5 —— **整段落在 1 点以内，远低于 5% 闸门**。第二条独立 Pareto（SCOPE）：192 token（−66.7%）保 99.5%、64 token（−88.9%）保 96.0%。
- **⚠️ 适用面被严格界定**：CroAttn 类 LMM 的 image encoding 占 TTFT **79%（Llama3.2-11B）/65%（90B）**，DecOnly 为 25%/54%，connector <0.4%。**D2 必须声明只针对 decoder-only 拼接架构（可攻 46–75% TTFT），不得把编码器份额算进加速上限。**

**C1 的创新点被两条本地证据证伪**
- **VisionSelector 已实现 D2 的核心主张**："单套 12.85M 权重、固定 20% 预算训练、推理期适配 30/20/10%"，保持 97.20%/94.83%/87.75%。
- **AIM 甚至无需训练**：7 个配置覆盖 **2.51→99.63 T FLOPs（约 40×）**，VideoMME 最低档 50.9、峰值 58.5（基线 58.2，整体 <13% 降幅），仅靠 r_merge 与 pruning scheduler 调档。
- **→ "可学习 + 运行时可调 + 单套权重"不能再作为卖点。** 只剩：机制差异（cross-attn latent **重采样** vs Top-K **选择**）+ 在 8–32 token 端压过 AIM 的 Pareto + C2/C3。

**C1 的结构被三处硬性界定**
1. **压缩块本身买不到 Pareto**：无 pre-fusion 时 1 token 掉 **6.1 点**（78.5→72.4），144 token 也只有 76.9；固定 1 token 增加 pre-fusion 层则 72.4→74.8→76.0→76.9→77.6 —— **同等 FLOPs 下增加 pre-fusion 层收益大于增加 token 数**。pre-fusion 层数必须作为第二结构旋钮。
2. **插入位置由 VisiPruner 定**：layer 1 全掩视觉 token 72.6→65.2（需视觉 attention sink），layer 2–7 由 system prompt 顶替**无损**；真正融合在 **9–15 层**（掩 top-10% GQA 61.95→54.09，掩 bottom-10% 无影响）→ **8-token 档在"一个 latent 充当 sink"的前提下可辩护**。
3. **低端必须文本条件化**：HiPrune @64 token 92.7% vs HiPrune++ 96.1%、POPE **73.0% vs 84.3%** → <64 token 的自变量是指令条件而非预算本身。

**C2 有可量化靶心，且设计空间被 FlashAttention 定死**
- 靶心：VisionSelector 在 ~5× FLOPs 削减下只拿到 **1.86× prefill / 1.74× E2E**、显存 17.57 vs 25.97GB（−32.3%）—— **这段"FLOPs 削减→墙钟加速"的落差就是 gather+GEMM kernel 的贡献位**。
- **硬约束**：FlashAttention 不物化 attention map → 任何从 map 读重要性的打分都会丧失 FA3。**D2 的打分必须 map-free**（可学习 latent、QKᵀ 打分均满足）。本地反证：Representation Shift 吞吐 5.5×/4.4×、R@1 比基于注意力的剪枝高 +7.2%。
- ModServe 指明视觉前段应按 **compute-bound** 立论（~100% SM 活跃、DRAM <30%）。

**C3**：只有形态C 一条本地线索（"同 FLOPs 下 pre-fusion 层数 vs token 数"曲线 + AIM 的 r_merge/(l1,l2) + VisiPruner 平均 23.9 层 vision exit）→ 可构成"给定单卡算力/显存反解 (token 数, pre-fusion 层数, exit 层)"的成本模型骨架。**形态A/B 在视觉 token 侧本地零证据。**

⚠️ G2 报告必须**逐方法**给 Pareto 而非取平均：同为 −97.6% 时 VisiPruner 61.3、SparseVLM 58.2、**PDrop 崩到 46.6**。

### 方向 2 · v4 增量（六层深潜）

**两条必须改写的 v3 句子**

1. **"新颖性只在算法轴被驳" → 硬件轴同样被占。** v3 说 D2 的算法轴新颖性被 VisionSelector（单套权重服务 30/20/10%）与 AIM（免训练 7 配置跨 40× FLOPs）驳回，硬件侧还空着——**不成立**。**Focus** 已在**原语级**抢先：**SEC**（在 attention softmax 处做 token 重要性 + top-k + offset 打分）与 **SIC**（在 GEMM tile 输出处做片上 tile 内相似度 gather/scatter）都已实现为加速器单元，28nm 开源，**+2.7% 面积 / +0.9% 功耗**，仿真（SCALE-Sim v2 + DRAMsim3）**4.47× vs 密集脉动阵列、7.90× vs A100、但仅 2.37× vs GPU+FrameFusion**、DRAM 流量 4.9×。**幸存贡献只剩 C2 + decoder-only concat 的 scoping**；可攻击面是 Focus 用**固定剪枝策略、无预算弹性打分、无 encoder/decoder 异构划分**。注意它对 GPU 软件基线只赢 2.37× —— **这同时说明 D2 走形态 B 的天花板很低**。
2. **"打分必须 map-free" → 这是 GPU 层约束，不是根本约束。** 在本就显式产生 attention map 的专用加速器上该约束消失：Focus 的 SEC 流式排序耗 M·a·k cycles，被 image-attention GEMM 的 M·(M+T)·h·n/(a·b) cycles **完全掩盖**（"SEC 不在 critical path 上"，面积 1.9%）。**但这是设计期解析估算而非实测，且仍付 O(M²) map 物化代价** → 在我们的 GPU 平台上，map-free 仍是硬约束，且 v4 把它**从设计偏好升级为 G1 通过条件**（见下表）。

**新增结论**

| 槽位 | 结论（数字 + 条件） | 作用 |
|---|---|---|
| G1 | decoder-only concat 下图像编码仅占 TTFT **25–54%**（cross-attn 才 65–79%；ModServe，vLLM 0.7.2 / 128×A100）；connector <0.4% TTFT | **表头腰斩**（v3 写 46–75%），且给出打分器 <0.4% 的预算 |
| G1 | 视觉 token = prefill FLOPs 上限 **77%**；VisiPruner 实测 **53.9% FLOPs↓**（LLaVA-1.5/InternVL2，中层仅 ~10/576 token 驱动融合，免训练）；LLaVA-OV-7B 32 帧×196 = 6272 token → **41.4 T** | 绝对分母 + 已被实现的值 |
| G1 | **map 依赖的代价是双向的**：回退非融合注意力后**峰值内存上升** —— FastV +3.7%、SparseVLM video **+54.8%** | **map-free 由偏好升为 G1 硬门** |
| G1 | batch=1 时 KV 仅占显存 **7–35%**（batch=64 才 80–97%） | 单卡单请求**禁止讲省显存**，只讲延迟/FLOPs |
| C1 | 跨帧时序合并 VideoMME **47.4** vs 帧内空间 **52.3**（AIM, r=3.1%）；AIM 自身开销 88.25 GFLOPs = **0.6%** | **弹性只能在帧内变动**；0.6% 是打分器开销上限 |
| C2 | 竞品标杆远高于 v3：**MoDES 单卡 H200 2.03× prefill**（Qwen3-VL-MoE-30B，branch-free 掩码 + 哨兵 <1% 开销）；**RESONATOR 纯 runtime 4.9× E2E**、TTFT 5.1×（8×A100，分辨率作自变量） | **v3 的 1.86× 目标失效** |
| C2 | 弹性 token 数的 kernel 解**不是"任意形状"**：cuBLAS wave 量化在 M 变 64 时掉 36%/21%；SmoothGEMM 片上虚拟 padding +14.6%；XY-Serve 收敛到**有限固定形状集**离线调优（重排 <1% 延迟）；SonicMoE varlen-M 持久调度去掉 M_tile 整除约束 | C2 必须证明 **tile 边界零代价** |
| C2 | **GQA 陷阱**：KV 先复制到 query head 数，(r−1)/r 为冗余 → **压缩比须 > 组大小**才净省；modality-aware 拆 cross/self-KV 比均匀多省 30–50% | 新增门槛 + 新增一条可做轴 |
| C3-A | 最低成本落点 = **LExI 式 `moe_layers[j].topk=k_j`**（不动调度器/显存管理/kernel）；BrownoutServe 规则 = 结构参数 k 固定、阈值每迭代滞回调（1.58–2.07×，SLO 违约 73.68%→7.14%，**但精度损失约 5%，正压 G2 红线**）；Cornserve 证明 L_t ∝ 该类型算力成本时各类约束等紧；**Dynamo 在数据相关分支上必 graph break** | C3-A 蓝图成型；弹性**必须暴露为编译期枚举的 variant index**，不能是 Python 控制流 |
| C3-C | 帧数已被显存反解：同模型 int8 **48 帧** / fp16 **22 帧** / 双卡 25 帧 | form-C 的现成模板 |
| C3-A | 传输隐藏轴已封顶：EPD-Serve overlap 15.27%→**98.78%**，纯 runtime +57~69% 吞吐 | 打分模块 offload **无空间** |

**对 v3 的净改动**：重算 G1 分母（decoder-only 的 **25–54% TTFT** × 视觉 token **77%** prefill FLOPs 上限，对照 VisiPruner 已达的 53.9%）；把 map-free 改为 G1 通过条件；**删掉任何显存节省叙事**；C2 标杆从 VisionSelector 1.86× 换成 **MoDES 2.03× prefill**，并须在 RESONATOR 类调度器**之上**报增量；**C3-B 已被 Focus 占**，改押 C3-A（LExI 整数 knob + BrownoutServe 的"结构固定/阈值每迭代"分离 + Cornserve 准入代数）或 C3-C（按帧数反解）；**压缩轴锁定帧内空间**，打分器开销须 <0.6%。

## 方向 4：多模态 MoE 模态分组专家 + 预取

**坐标**：多模态理解 / MoE-VLM / 4090+大 DRAM。

**G1 势能真实，但基线被换掉**
- 塌陷数字（DeepSeek-V2 236B、单卡 A5000 24GB + 512GB host）：decode 期每专家平均 **0.3–0.4 token**、GPU FLOPs 利用率 **0.1%**、**1 tok/s**；batch=1 的 MoE offloading GPU 利用率仅 **0.76%**。headroom ~100×。
- **但 G1 的对照物不再是 naive offloading**：module-based batching 已从同一塌陷拿回 **31×**（Bsz 75、41% 利用率、31 tok/s）；ProMoE 拿回 1.34–2.07×；同层 pre-attention 预测拿到 **93.03%** 命中。**D4 的 headline 必须是相对这三者的增量。**
- 纯预取天花板：expert fetch 占 **~88%**、gating+compute+aggregation ~12% → **完美隐藏也只有 ≈1/0.12 ≈ 8×**。超过该数的收益必须来自**减少传输量本身**。

**两处实质反驳（必须正面处理）**
1. **"prefill 时工作集已知"不是新颖点**：Mixtral-8×7B/TruthfulQA 上 prefill 与 decode 专家激活分布余弦相似度全层平均 **0.89**，据此的 placement 已达 **8.7× decode 吞吐、0.13% 精度损失**。→ D4 必须论证**模态段信号强于这个 0.89 的上下文信号**。
2. **命中率赛道已饱和**：93.03%（cloud 98.65%/edge 98.85%），0.15ms CPU 预测器藏在 0.74–1.13ms self-attention 下且无 bootstrap 问题 → 命中率维度只剩 ~7% 边际。**战场必须换成"传输字节"。**

**C1 被两条证据收紧**
- **删掉 {vision/text/shared} 的 shared 一档**：shared expert 是与**低**局部路由一致性相关性最强的架构因素（REAL 模型 SRP 前两组均不用 shared expert）；且 local load balance 是同等显著因素，**可预测性正则会与负载均衡损失直接对抗**，必须在 C1 中显式预算。
- 层间亲和性**本已是预训练模型的固有且 OOD 稳定属性**（~1000/3000 token 即可估出，Pile→C4/Dolma/Yelp 行归一化 0.989–1.005）→ 可预测性正则必须证明它在 router 已有可预测性**之上**还有增量。唯一支持训练期干预的本地证据是 **gate input-sensitivity**（token-based 预测仅 58.3%、skip-based 在 QW-2 上 66.9%）。
- 模态冗余有机制而非仅相关性：vision token 与 FFN 权重更正交（角度→90°），FFN 对 vision token 的改变小于 text token；MoDES 双模态阈值 **τ_v > τ_t** 是直接先例。

**C2 钉在 memory-bound 侧**：SOL 论证用 PCIe/HBM 带宽利用率与"隐藏字节数"，**不是** MFU vs FA3/cuBLAS。G1 原型 kernel 应度量为**有效带宽利用率 / 被隐藏字节比例**。

**C3 落在形态C，现成靶标已备**：**ρ ≈ 2× active experts** 是 20 个 MoE LLM 上的折中常数，可直接作为预取缓冲容量；SRP/SCH 指标与 `moe-lrc` harness 可让模态分组超参被反解而非发明（注意 SCH 是 oracle 上界，不可当 LRU/LFU 实测代理）。

**⚠️ 本地缺口 = D4 的前提**：MoDES 只给"每模态冗余度"（定性），**没有"模态间专家集合不相交程度"**。→ **立项前置测量：vision 与 text top-k 专家集的 Jaccard。**

### 方向 4 · v4 增量（六层深潜）

**口径修正（PARTIAL）**：v3 的"**删掉 shared 一档**"只对**架构级、训练进去的 shared expert**（DeepSeek-V2/V3、SharedFusedMoE 那类）成立——它确实仍是路由一致性下降的首要架构成因。**但它不适用于运行期常驻/预取层，那里结论相反**：STEP 的**窗口票选临时共享专家**把每层每步 H2D 从 `k×expert_size` 降到 `(k−c)×expert_size`（k=2/c=1 即**减半**），命中常驻集则整段跳过 H2D；Qwen3-30B-A3B / A100 / INT8 上 **expert 取数占执行时间 ~88%**，prefetch 命中 **85.5–98.8%（CNN/DM）、72.1–95.6%（LongBench）**。**改口径：不要把"删共享层"读成"禁止任何常驻专家层"——在字节轴上常驻层是收益不是代价。**（注意这些数字全部实测于**文本 LLM decode**，非 MoE-DiT / Video。）

**新增结论**

| 槽位 | 结论（数字 + 条件） | 作用 |
|---|---|---|
| C2 | **T_moe = β·a_max + c_e**：延迟线性于**不同激活专家数**，与 token 数几乎无关（B 64→512 延迟基本不变，JANUS 离线 profiling）；B_min≈**9,400 tok**（H100/DSv3, n=256,k=8）→ 在线 B<100 **铁定 memory-bound**；decode b=1 强度 ~23 FLOPs/B，利用率 <1% | **v3 缺的目标函数：min a_max**；且**加 FLOPs 近乎免费** |
| C1 | EPLB 1.5× 复制 → 激活专家数 **+30%**、decode 延迟 **−14%** | 减字节不免费，**必须同时报 a_max** |
| C1 | 前置门控/预取在专家常驻显存时"**退化为无收益的额外 router**"；batch 级激活异质 15–20% vs 聚合级 1.2%（Lynx）；ExFlow 离线 profiling 8.5/45 min | D4 **必须先锁定 offload 设定**，并按 per-batch 论证 |
| G1 | 完全隐藏 HtoD 需 expert batch > **2^11 tok**；M100（N5A 399.8 mm²）LLaMA2-7B W4A16 decode 21.34 vs Thor-U 20 ms = **0.94×**，同受 273 GB/s DDR 限 | 单卡**不可能靠"隐藏"立论**；**iso-bytes 下架构收益为零** |
| C2 | 字节轴的既有基线：STEP **−50%/步**、MoE-Infinity **>80%**、MoE-SpeQ **96–99.85%** 命中；SonicMoE **已把 gather 融入 prologue** | 这些是必须越过的线，且 **gather 融合不可再作为贡献** |
| C2 | 单卡与多卡目标**相反**（ARGUS 内聚 / CUCo 拆分）；单卡残余瓶颈 = **fp8 requantize + 中间张量落 HBM**；fused MoE SoL 437–567 vs Triton 225–341 TFLOPS | **全部 all-to-all 文献出 D4 范围** |
| C2 | MoE 必须用 **S-MFU/S-MBU**，常规 MFU 在 batch>1 时高估 1.5–3× | 否则自缩 headroom |
| C3-A | 实测天花板：LLEP e2e **1.88–2.2×**（MoE 层 6.11× 塌缩到 e2e）、PROBE 1.32×/1.26×；JANUS 每 batch 重写路由 **<90 µs@4096** | **决策开销可忽略**，但须报 e2e 且 **>2×** 才成立 |
| C3-B | ATU+PDU：hop **−213×** 但吞吐仅 6.63×、**增量仅 +1.1×**；M100 的 GSDU 由共享 RISC-V CPU **串行仲裁** | **不走形态 B**，走 A 或 C |
| C3-C | **DIAMoND**：OU 尺寸 `H=min(ρ_in,d_min)`、`W=min(ρ_out,d_min·QB)`，一行反解 C1 结构超参；NAND endurance ~10³ → **专家权重必须 write-once** | 首个可复用的 C3-C 模板；**动态专家重分组被物理排除** |

**对 v3 的净改动**
1. **把"fight on bytes"换成可优化的闭式目标：min a_max**（T_moe=β·a_max+c_e）。C1 的成败标准从命中率/字节量改为"**是否降低不同激活专家数**"，并强制同时报 a_max 与 e2e；因 b=1 时强度仅 ~23，**C1 可以加算力换更少专家**（这是 v3 完全没有的设计自由度）。
2. **先承诺 offload 设定再谈 D4**：专家常驻 4090/5090 显存时 D4 的 C1 是 no-op；须以 **PCIe 字节/秒**为瓶颈口径，并**放弃"隐藏传输"叙事**（2^11 tok 阈值不可达），只论减字节 + 抬 compute-per-byte（如投机解码，A30 2.5×）。
3. **门槛与形态双双上调**：C3 选 **A 或 C**；e2e 须 **>2×** 才越过 LLEP 的纯运行时基线；roofline 一律用 S-MFU/S-MBU。

## 方向 5：视觉历史层蒸馏替换 Mamba（二期，风险最高）

**坐标**：流式/长视频理解 / Qwen2.5-VL 改造 / 4090。

**G1 headline 必须重写**
- 基线不是全注意力，而是 **XStreamVGGT 的 4.42× 内存 / 5.48× 加速**（KV 压缩已银行入账）。
- **最关键的一条反驳**：LongLLaVA @100K tokens，**1:7 混合比下内存 79.4GB → 79.1GB 近乎为零**（吞吐 2.6× 但内存不动）—— **存活的全注意力层仍持 O(L) KV，"O(L)→O(1)" headline 直接失效**。
- 量级锚点：TimeViper vanilla **128 帧即 OOM** → +ToMe ~5K 帧 → +ToMe+TransV **10K+ 帧**（不是层比单独的功劳）。
- **G1 原型必须在递增帧数下测 state 常量性，单点吞吐数无法过闸。**

**C1 必须改写**
- **Samba 反驳"保留几层全注意力"**：16K 时**即使仅 1 层全注意力也无法外推**（PPL 10.29→13.66），而 SWA 可外推至 1M。→ 定稿形态：**视觉历史层→SSM，保留层→sliding-window / sink+window**（可直接借 StreamingVLM 的 Tsink=512 + Twindow=512 + Vwindow=16s 三类预算）。
- keep-ratio 区间由 TimeViper 的 **7.1%（4/56，从零训练）**与 MambaInLlama/M1 的 **21%（6/28，蒸馏）**夹出。
- **两条互斥性必须在方法节显式声明**：(1) **SSM state 压缩 vs vision-token reduction 二择一** —— token reduction 在 SSM 上失效有机制（pruning 的不可恢复损失沿递推**逐步放大**）和量级（**LTMP 在 Mamba-2-2.7B 上 PPL 4.10 → 4670.71**、Avg acc 63.8%→41.3%）双重证据，而 TimeViper 的 TransV 正是丢 95% vision tokens；(2) **SSM state 就地更新不可回滚** → 流式 seek-back/重答/分支需要显式 state-checkpoint pool —— 这是 C3-form-A 真正的承重贡献点。

**C2 整列搬到 memory-bound**：SSM 侧削 **30% FLOPS 只换 1.35× 吞吐**（Mamba-2-1.3B）→ 不能用 MFU 立论，必须用 HBM 带宽利用率对定长 state 的 scan/update kernel 做 SOL 论证。**必须跨过的两条线**：LiveStar 纯 runtime caching **1.53×**、XStreamVGGT 纯 KV 压缩 **5.48×**。

**C3-form-C 反解式现成，同时给出最严重的 G2 风险**：**T_forget = 5.172·N_S − 4.469（R²>0.999）**，Mamba-2 的 N_S=256d → 设计规则 T_train > 5.172·N_S；370M Mamba-2 已需 **T_train > 66.7K tokens**；按 16 vision tokens/frame，10K 帧 = **160K tokens** 蒸馏上下文。叠加 Passkey 精度随训练 token 增长而下降、**780M 比小模型更差** → **G2 的 ≤5% 必须在长流检索类任务上以 Pareto 报告，不能只报短片段指标**。
可行性支撑：attention upcycling **70 GPU-hours vs 370K（约 5000× 成本差）**；层选择可用 MEDA 的 **E_CM^l = −(E_TV^l + E_VT^l)**（prefill 一次，O(n_T·n_V)/层）。

**⚠️ 动机被削弱（critic）**：StreamingVLM 的 **8 FPS 是它自己在单 H100 上的吞吐，不是滑窗方案的天花板**；去掉 sink 或 text window 只是 win-rate 73.64→69.68/66.76 的**退化而非崩塌**。→ **D5 不是在修一个坏掉的方案，而是要正面打赢一个已可用的 SOTA。** 且多模态 SSM 蒸馏质量本地零覆盖。

### 方向 5 · v4 增量（六层深潜）：换主线

**口径收窄（PARTIAL）**：C2（注意力→SSM 替换）在通用 GPU 的**常驻容量轴**上仍弱——30% FLOPs 削减只换 1.35×、1:7 混合 100K token 下 79.4→79.1 GB，两项测量不变；**但该结论现明确限定于"通用硬件 + 权重/KV 常驻容量 + 长上下文"口径**。HLX/URSC 在**另一条内存轴（片上中间数据与 kernel DRAM 流量）**上确有收益：PipeFlash 把 score/prob 矩阵 **128 KB → 1 KB**（4.8×）、PipeSSD 中间数据 **642 KB → 58.5 KB**（11×）、DRAM 访问 −6.8×。**但条件把它锁死**：自研 URSC 加速器（TPU 半芯片模拟，2×128×128 MXU）+ Hybrid-2.7B，**TPU DRAM 单 batch >32K 即 OOM，从未进入 100K 区间**；量纲也不同（每 block KB 级缓冲 vs GB 级端到端常驻）；且 4.8× 出自作用于 FA-2 的 PipeFlash（**注意力侧**）而非 SSM 数据通路。HLX 反而佐证 memory-bound：SSD 在 A100 利用率 **~26.9%**，TPU 融合后仅 **~11%**。

**新增结论**

| 槽位 | 结论（数字 + 条件） | 作用 |
|---|---|---|
| C3-A | **StreamingVLM**：sink 512 + 文本窗 512 + **视觉窗仅 16s**、视觉先驱逐 + **Contiguous RoPE**，单 H100 bf16 持 8 FPS；训练用 **Overlapped-Chunk Full-Attention**（W=24s/O=12s）保证训推一致；训练 ~128 H100-day | 给出 v3"保滑窗/sink"的**已测、单卡可行**替代路线，并**补上 D5 本要自己发明的训推一致机制** |
| G1 | LiveStar 类分块流式 W=64/S=32（50% 重叠），KV 增长 O(N²T)→O(W²+K·logT)，严格因果不可跨未来帧并行；**预处理（resize/normalize/H2D）可占推理时间 40%**，可与文本解码重叠 | **40% 不在模型内，封顶所有模型侧优化**；本身就是一个便宜的 C3-A |
| C2 | Mamba2 scan 单体融合 4 kernel/3 次 HBM → 1 Triton kernel，但"融合只是壳，性能来自**块内并行前缀扫描 + 极短串行段**"；**TorchInductor 10.57 vs 手写 PIKE-B 10.81** | **SSM kernel 侧 C2 头寸 ≈ 0（差 ~2%）** |
| C3-B | HLX/URSC：利用率近 100% **仅当 block_size = d_head = d_state**，5 个 Mamba-2 配置（130M–2.7B）间波动 <2%；FA-2/SSD 异构模式使**瓶颈跨层漂移** | 唯一的 d_state 硬件选型规则；瓶颈漂移即"1:7 混合问题"的硬件回声 |
| C2 | Stratum PPA 约束形式：`P_dram+P_compute+P_misc ≤ 45 W`、面积 ≤ 63%×121 mm²（36 MB SRAM、128 TFLOPS、内部带宽 19–34 TB/s），**跨 interposer token I/O 仅 819 GB/s** | 状态尺寸/分层论证须以此**不等式形式**呈现；每步递归状态流量应对标 **819 GB/s** 量级而非 HBM 峰值 |

**对 v3 的净改动**
- **换主线**：D5 不再以"蒸馏进 Mamba/SSM"为主张——**SSM kernel 已无 C2 空间**（10.57 vs 10.81）。改以 **StreamingVLM 式 sink + 非对称短视觉窗 + Contiguous RoPE** 作为 C1/C3-A；v3 的"保滑窗"从建议升级为**可落地方案**。
- **重算 G1 分母**：任何 D5 端到端加速上限须**先扣掉 ~40% 预处理**；同时把"预处理与解码重叠"列为**独立的 C3-A 收益项**。
- **若坚持形态 B**：改以中间数据/DRAM 流量为 C2 内存轴，并锁 `d_state = d_head = block_size`；**但必须声明其仅在专用加速器且 ≤32K 序列成立**，不可外推到 100K 常驻容量结论。

## 方向 6：AdaLN 低秩共享 + 调制链融合（**结案：G1 不通过**）

v1.5 判它"G1 不过闸"是对的；v3 用 14 条本地证据把它**证伪到方向性相反**：

- **Amdahl 封顶**：DiT 原文自证 adaLN/adaLN-Zero 的 Gflops 增量**最小/可忽略**（唯一 ~15% overhead 的是 cross-attention）；DSV 实测 **attention 占端到端 92%/93%（1.3B/3B @200K）到最高 95%**。
- **收益上限**：完全移除调制路径，唯一可查收益是 ELF-B **148M→105M（≈29% 参数）且无任何延迟数字**；两条笔记都把它写成**参数故事，从不是速度故事**。
- **C1 已是既有基线**：DSV Table 1 显示 **AdaLayerNormSingle 在 0.8B/2.7B/30B 三档 MovieGen-like 架构中原样沿用** —— 跨块共享调制表不是新 hack。
- ~~**C2 的目标已被编译器免费产出**：torch.compile 已自动把 BlockNorm + Hard Swish + 量化反量化折进 GEMM epilogue~~ → **v4 删除此条，理由本身无一手证据**（见下方 v4 增量①）。
- **融合本身也被切开**：只有 **token 广播的 scale/shift/gate 能自由折叠**；LayerNorm 沿特征维的全局统计无法在 tile 内完成（LoKA 必须改成块级 RMS 的 BlockNorm 才可行）。要吃下整链就得动归一化本身 —— 那是另一个更大的 C1。
- **同构实测的诚实预期**：HATA 四次 launch 合一仅 **7.6% 端到端**；FlashFuser **58% 访存下降 + 3.3×/4.1× kernel 加速只换来 1.24× 端到端**。
- **C1 的经济学也不利**：Basis Sharing 在 LLaMA-7B 20% 压缩下 PPL 5.68→7.74，收益来自带宽而非算力，>50% 误差爆炸 → 会把 G2 精度预算花在 DiT 原文称 Gflops 可忽略的路径上。

**唯一残留缝隙**：笔记同时记录"新低精度 kernel 接入 torch.compile 可能需人工介入、动态精度切换触发重编译"。
**⚠️ 同时也是最大的证据缺口**：**没有任何笔记测过 adaLN/modulation 自身占 DiT block 延迟的比例** —— §0.9.2 实验的 (d) 项会免费给出这个数。若该比例意外很高，D6 可重开。

### 方向 6 · v4 增量：结论不变，但关闭理由被换掉

**① v3 的关闭理由之一不成立（PARTIAL：理由被推翻、结论不变）。** v3 用"torch.compile 已把 BlockNorm+激活+量化折进 GEMM epilogue"论证 D6 的 C2 无可争之地。**六层深潜里没有任何一手笔记记载 Inductor 做 GEMM epilogue 融合**——一手记录只到 pattern-based 融合（select/pointwise/reduction）+ 布局传播 + Triton codegen。**所有实测的 epilogue 收益都出自手写 CUTLASS/CuTe-DSL**：LongCat-Flash 的 fused GroupedGemmAdd **3.12×–3.86×**；MoEBlaze SwiGLU epilogue 最高 **4×** 激活内存下降、2×–6.2× 训练加速；SonicMoE 的自定义 CuTe-DSL epilogue functor（Hopper 需 ping-pong 双缓冲、Blackwell 需 TMEM 2-stage + UMMA 才守住 TFLOPS）。形似 v3 说法的"DiT FG1–FG4 + 120s 冷启/12s 预热"时间线，已被本层自评为**前序 agent 的模拟值**（可推断，弱）。**→ 这句话不许在任何方向里再复用。**

**② D6 的新关闭理由（比原理由更强，且可写进 related-work）**
- **带宽饱和上界**：adaLN modulation 属"统计量可在 tile 内算出"的可融合类，而 Fused Add RMSNorm 这类 **memory-bound 融合一旦显存带宽饱和即触顶**。FlashInfer-Bench 实证 **RMSNorm 是唯一 agent 能追平/超过人类的 case-study kernel**（memory-bound，写对就接近上限）——**GEMM/GQA 才是需要 pipelining 与 tiling 的战场**。
- **融合组已被公开认领**：DiT 的 **FG1–FG4** 已公开——FG: LN→QKV→Attn→Residual 纵向；gate+up 以 `W=[W_gate|W_up]` 横向拼接；SiLU→Mul→down 纵向。**没有未被认领的 epilogue 可做 → C2 无新意。** 纵向融合上界另由"单 CTA 时分复用对 A100 192 KB SMEM"的判据框定。
- **反向可用**：若日后要写手写 epilogue kernel（非 D6），**3.12×–6.2× 是可引用的 kernel 级收益上限**，但 D6 的 **1.24× e2e 天花板仍在**。

---

## 推荐路线（v4 改写）

1. **先做 §0.9.2 的单次实验**（仍是第一优先，但**目的变了**）—— D1 已在性能轴关闭，该实验不再是"D1 的生死判据"，而是结算：**(b) 相似度随 schedule 长度的衰减曲线**（D3 的 6.72× 基线要分 50 步 / 4–8 步两档报，这条曲线是分档的依据）、**(c) 中间态张量的显存足迹**、**(d) DiT block 逐算子延迟分解**（D6 的复活条件 + D3 的本地 Amdahl 分母）。**在拿到 (d) 之前不要写任何以"attention 占 92–95%"为分母的句子**——那个数是 8×H100/200K/前向+反向训练值。
2. **主线 = 方向 3，但 intro 必须重写**：新颖性不再能说"无人做过三态门控"（**DiTPA ISCA'26 已发表 {full|skip|reuse}**），只能说三条差异：**rank-r 增量作为中间态 / 门控可训练 / GPU 文生视频**。起点 = fork QuantCache，在 **Open-Sora 1.2 标准设置**下先复现到 **6.72×**（不是 4.12×），再加 rank-r 第三态；kernel 走 **TileLang/CuTe**（Ada 无 DSM，H100 融合论文不可移植；Triton 起跑线 0.67×），形态按 **HyTiS 模式**（一个 persistent kernel、共享 SMEM layout、仅 tile 几何不同、双射 offset 切换）。
3. **方向 1 结案，资产转移**：把它的七条 C2 约束（索引表 gather / 序列维归约 top-k / block size 编译期对齐 / 仿射掩码保 TMA / ≥85% HBM 带宽合格线 / 分母标注纪律 / 每步 <1ms 预算）整体搬进 D3，**不再单独立项**。
4. **系统味支线 = 方向 2 / 4**：D2 靶心从 VisionSelector 1.86× 上移到 **MoDES 2.03× prefill**，且形态 B 已被 Focus 占（它对 GPU 软件基线只赢 2.37×）→ 走 C3-A/C；D4 先做"vision/text 专家集 Jaccard"前置测量，**并先承诺 offload 设定**（专家常驻显存时 D4 的 C1 是 no-op），目标函数改为 **min a_max**。
5. **方向 5 二期但换主线**（StreamingVLM 式 sink+短视觉窗，不再是 Mamba 蒸馏）；**方向 6 结案**（新理由：带宽饱和 + FG1–FG4 已公开）。
6. **冗余预算记账**：D3/少步蒸馏抢同一份跨步冗余，D2/D5 抢同一份 token —— 立项前必须算清，否则会重演 QuantSparse 的"1.83→2.47× 换 VQA −4.8%"。
7. **形态选择的硬约束（v4 新增）**：形态 A 单独的天花板 ≈**1.21×**（Bullet），形态 B 在 memory-bound 下只能承诺 **2–6×** —— **任何方向若收益全在形态 A，即已被本地数据证伪**；真正空缺的是**形态 C**（全库无 DiT cost model）。

## 下一步行动（按闸门顺序）

- [ ] **P0：§0.9.2 单次 instrumented 实验**（Wan2.1-1.3B，4090，一个下午）—— 结算 (b)(c)(d)；(a) 的用途由"D1 生死"改为"D3 的跨步冗余预算"
- [ ] **P0：真卡硬件常量表 —— 13 格（§0.10.2）**。4090 侧 5 格今天就能测（**SMEM/SM 与 L2 容量在 vault 里各为零条**，流传的 128KB 是 RTX 3090 参数错挂）；5090 到货第一天测 7 格（首测 `tcgen05` PTX 探针）；另统一 H100/A100 的口径。**在此之前不写任何 tile/MFU 论证。**
- [ ] **P0（新）：读完 DiTPA（ISCA'26，开源）全文并逐条对齐** —— 它是 D3 的直接先例（三态门控 + 0.23% 面积 / 4 cycle 判定 + "沿门控不变维度优先 tiling" + 60.1% 重复权重加载）。**D3 的 intro 在读完它之前不要动笔。**
- [ ] P1：方向 3 起步 —— fork QuantCache，在 Open-Sora 1.2 / 100-step / W4A6 下复现 **6.72×** 全消融阶梯，再加 rank-r 第三态；**先按"SMEM 一半（双缓冲）∩ 三态共享寄存器集"算出 r_max**
- [ ] P1：补抓本地零命中的对照组 —— Chipmunk / TaylorSeer / ToCa / DisCa / LearniBridge / TeaCache / AdaCache（D3 的对手全缺）
- [ ] P1（新）：**数一数 24GB 4090 上可捕获的 CUDA Graph 组合数** —— 每图约 200 MB，这是 D3 三态门控的直接容量约束
- [ ] P2：方向 4 前置测量 —— vision vs text top-k 专家集 Jaccard；**并同时报 a_max**（不是只报字节量）
- [ ] P2：微调可行性核账 —— SLA 配方是 2000 steps × **batch 64 @480p**，单张 4090 显然不成立；**Q-VDiT TQE 的 rank=1 校准就要 12.5–12.9 GPU-h/配置**，须据此定 r>1 的预算
- [ ] P2：**反向传播覆盖** —— SageBwd 已示前向 2×/反向仅 1.2–1.6× 的不对称，且学术工作与 TensorRT **均未演示融合反传**；凡需微调的方向都要配反向 kernel
- [ ] P3：评测协议与成本 —— 质量指标目前无共同轴（VBench/Delta-FScore/win-rate/CosSim 各说各话）；须算一次可发表视频质量评测在单卡上的卡时
- [ ] P3：**消费卡锁频协议** —— SOL-ExecBench 给了 B200 锁 1500MHz 的协议，**GeForce 上没有**，boost 波动本身能吞掉 10% kernel 收益
- [ ] P3：新颖性/并发工作检索 —— 前两轮只问"vault 里有没有"，从未问"2026 年是否已被做过"。**DiTPA 这条正是这个缺口的代价**，应尽快补一次 2026 年并发工作检索

## 附录：本地检索基础设施（2026-09-21/22 修复）

上一轮 survey 丢失三层证据的根因与修复（全部已实测验证）：

| 问题 | 根因 | 修复 |
|---|---|---|
| agent 的 Obsidian 调用全部被拒 | `learning_scheduler.ts` 用 `--permission-mode acceptEdits`，非交互 agent 的 MCP 调用需授权 | 改 `bypassPermissions`（与其他 job 一致）；实测 denied 0、83 次调用成功 |
| Obsidian 渲染进程反复崩溃 | **21,252 个 md / 306MB 被索引进 Omnisearch 的单个 V8 堆**，其中 `papers_md`(106MB)+`repos`(40.7MB) 根本不在 `OBSIDIAN_READ_PATHS` 里 | 按 skill 的六目录口径设 `userIgnoreFilters`（索引 306→151MB）；`hideExcluded=true`（否则仍索引）；`useCache=true`；渲染进程堆 12GB |
| 并发 agent 压垮服务 | 桥无并发控制，2 个 agent 的全库搜索即可打爆 | 桥 adapter 加 `tools/call` 并发闸（2 槽、60s 队列）。实测 **3 波×8 并发 = 24/24 成功**（此前第 4 个即断连） |
| 崩溃后无人恢复 | systemd 只看主进程，渲染进程死了仍是 active | `obsidian-rest-watchdog.timer`（20s 探测，双探测确认后重启）；REST 恢复时间 25-60s → **10s** |
| 客户端过早放弃 | `OBSIDIAN_REQUEST_TIMEOUT_MS=30000`，退化期搜索可达 67s | 提到 90s（compose 默认值，随仓库留存） |
| 重跑要从零开始 | 调度器只支持同目录 checkpoint，跨 job 无复用 | `--reuse-from` + 内容判据 + 级联失效 + `--dry-run`；MCP 侧 `reuse_from_job_id` / `reuse_skip` / `reuse_exclude_regex` / `reuse_require_regex` / `reuse_dry_run`。**实测：6/6 问题空间 + 11 答案复用，25 答案重做（含 Q2.6/L2 —— 它正好跑在中断窗口内），5 个 horizon 与纵向总结级联失效** |

**给后续 survey 的查询纪律**（§0.9.1-② 的直接产物）：omnisearch 响应体超限会报错而非返回空——**必须加引号短语 + `path:`/`ext:` 过滤**，并且**不要把检索失败记为"证据不存在"**。

**v4 补记：复用闸门是必要的，不是多余的保险。** 在**已修复**的那轮里，`--reuse-from` 仍然**拒绝了 3 份携带合法 `[ANSWER_AGENT_DONE]` 信号的答案**（`Q3.6/L3`、`Q4.2/L4`、`Q4.3/L4`，全部"命中 --reuse-exclude：均无/全无 note evidence"），并按级联规则让 L3/L4 两个 horizon 与纵向总结失效重算。**若只按"有结果就跳过"实现，这 3 份空答案会被当成有效证据带进 v4。** 另一处实战教训：Phase 3/4 四个 horizon agent 全部撞上 `Reached maximum budget ($5)`（输入 188K token + 3.6M cache read），**其中两个在被切断前已写出 DONE 标记**——这正是"内容判据优先于完成信号"的第二个理由；改 `max_budget_usd_per_agent: 20` 后经复用重跑 19m34s 完成。

## 网络文献索引

SANA [2410.10629](https://arxiv.org/abs/2410.10629) · VSA [2505.13389](https://arxiv.org/abs/2505.13389) · SVG2 [2505.18875](https://arxiv.org/abs/2505.18875) · Chipmunk [2506.03275](https://arxiv.org/abs/2506.03275) · MOHAWK [2408.10189](https://arxiv.org/abs/2408.10189) · Zebra-Llama [2505.17272](https://arxiv.org/abs/2505.17272) · LLaVA-Mini [2501.03895](https://arxiv.org/abs/2501.03895) · SVDQuant/Nunchaku（[HAN Lab](https://hanlab.mit.edu/blog/svdquant-nvfp4)）· PixArt-α [2310.00426](https://arxiv.org/abs/2310.00426) · RainFusion2.0 [2512.24086](https://arxiv.org/abs/2512.24086) · SLA（本地 + [thu-ml/SLA](https://github.com/thu-ml/SLA)）
