---
tags: [调研报告, DiT, 多模态, kernel, 体系结构, 软硬件协同, 当前版]
date: 2026-09-22
status: 当前结论（合订版，无迭代痕迹）
迭代记录: "[[调研报告-DiT多模态结构改动-训练保精度-kernel协同-20260921]]"
---

# DiT/多模态推理：可做点调研 —— 当前结论

> **本文件是唯一的现行版本。** 所有结论已按最新证据筛过一遍：被推翻的论据直接删除、被取代的口径直接换成新的，不保留"某版曾经怎么说"。逐版演变过程见 [[调研报告-DiT多模态结构改动-训练保精度-kernel协同-20260921]]。
>
> **证据基础**：11 路 vault 扫描（154 条主张对抗性验证，149 条确认）+ 六层深潜 survey（L1 算法 → L2 Serving → L3 编译 → L4 kernel → L5 硬件 → L6 芯片，36 问，一手引用 L1 51 / L2 175 / L3 60 / L4 143 / L5 221 / L6 106 条）+ 35 agent 合并（181 条新增、43 条矛盾经挑战后 14 条成立）。凡标注**「笔记显示」**者为 vault 一手正文；标**「派生」**者为上一轮 job 输出或其引用网页，**不可直接写进论文**。

---

## 一句话现状

**六个候选方向，两个已结案（D1、D6），D3 与 D4 现并列待决，支线 D2，二期 D5。** 2026-09-24 读完 D3 的 7 篇对照组后，**D3 的新颖性防线从三条塌到一条（仅剩"决策可训练"）**，另新增一条设计级反对（两篇独立工作都主张只缓存最后一层，反对逐 block）。**D4 的新颖性此刻高于 D3**，但 D4 整体挂在一个尚未测量的数上（模态间专家集 Jaccard）——**这两件事都是一个下午能测完的，测完再定主线**（§11 的两个 P0）。**所有 kernel 论证在真卡常量表建立之前都不能动笔**（§3）。

| # | 方向 | 状态 | 一句话理由 |
|---|---|---|---|
| **3** | **DiT block 级可训练三态门控 {重算｜rank-r 增量｜复用}** | **★★☆ 降级（原首推）** | 工程路线仍最完整，但读完 7 篇对照组后**三条新颖性防线只剩"决策可训练"一条**，且 DisCa/LearniBridge 两篇独立反对逐 block 缓存（§4.1、§4.1.1） |
| 2 | 多模态 LLM 弹性视觉 token 压缩块 | ★★☆ 支线 | G1/G2 最强（77% FLOPs 上限、64 token 掉 <1 点），但算法轴与硬件轴的创新点**都已被占**，只剩 C2 + scoping |
| 4 | 多模态 MoE 模态分组专家 + 预取 | ★★ 支线 | 势能真实（0.1% 利用率），目标函数已闭式化（**min a_max**）；但必须先承诺 offload 设定，否则 C1 是 no-op |
| 5 | 视觉历史层流式改造 | ★☆☆ 二期 | 主线已从"蒸馏进 Mamba"改为"sink + 非对称短视觉窗"；SSM kernel 侧无 C2 空间 |
| 1 | 视频 DiT 跨 step 路由缓存 | **结案** | **性能轴关闭**：把整段 attention 归零，e2e 也只有 ≤1.18×，连形态 B 分支一并否掉 |
| 6 | AdaLN 低秩共享 + 调制链融合 | **结案** | 带宽饱和上界 + FG1–FG4 融合组已公开，C2 无未认领的空位 |

---

## 1. 论文模板（加速是核心评判）

> **因果链**：较新的 {任务 / 模型架构 / 硬件特性} = 原生加速潜力 → 魔改模型制造机会（C1）→【**闸门 G1**：潜力必须巨大，且必须用"简单实现改造后模块的原型 kernel"初测 —— 原 kernel 是为原结构优化的，直接套用测不出新结构的收益】→ 才继续谈精度几乎不降（G2，红线 ≤5%）与软件适配（G3）→ 必要时硬件适配（形态 B）。

| 闸门 | 判据 | 成本 | 不过闸的处置 |
|---|---|---|---|
| **G1 加速潜力**（先决） | 理论加速**巨大**；**验证必做**：为改造后的模块简单实现一个原型 kernel 并 microbench。原型不求最优，**量级对了即过闸**；但必须为改动模块**重新调过**，否则测量在构造上就无效 | 纸面核算 + 原型 microbench，天级 | 弃案或降级为组件 |
| **G2 精度几乎不降** | 微调/蒸馏后 **≤5%**，报精度-加速 Pareto | 训练卡时，周级 | 回改 C1 重过 G1 |
| **G3 软件适配** | 端到端仍达标：编译（图改写/JIT/CUDA Graph）、运行时/serving 集成 | 工程，周~月级 | 补形态 A；GPU 吃不满则升形态 B |

**三类贡献**
- **C1** 模型魔改 + 训练兜底（过 G1、G2 ≤5%）
- **C2** 对应 kernel，**必须论证最优**
- **C3** 软硬件协同（三形态见 §1.2）

**"新近性 = 潜力"选题原则**：新硬件特性（NVFP4、TMA）、新任务（长序列视频生成、流式理解）、新模型架构（DiT、MoE-VLM、hybrid Mamba）—— "新"即未开采潜力。单轴新=有空位；两轴新=潜力最大竞争最少；**三轴全旧=只剩组件级贡献**（D6 即此下场）。

**卡数无关原则**：魔改收益是结构性的，单卡多卡都应体现，多卡是放大器。实验平台 = **Ada RTX 4090（现役）+ Blackwell RTX 5090（即将）**。

### 1.1 Baseline 坐标系

**论文坐标 =（任务, 模型, 硬件平台）三轴，都难改**；自由变量是"魔改 + 适配"。文献按重合度分四档：

| 档位 | 用法 |
|---|---|
| **三同** | 主对比 baseline + 实验起点（fork 其代码） |
| **同任务 · 异模型** | 借模型改法 |
| **异任务 · 同架构** | 借"任务特点 → 改法"的映射 |
| **同结构 · 异硬件** | kernel 映射互鉴 + 形态 B 参照 |

差两轴以上只作背景引用。

### 1.2 C3 的三种形态及其定价

| 形态 | 内容 | **收益上限（本地实测）** |
|---|---|---|
| **A｜运行时 + 编译** | 显存/缓存驻留、预取与 pipeline 重叠、SLO 调度、vLLM/SGLang/xDiT 集成；编译侧 = 图改写、JIT 特化、CUDA Graph 捕获 | **≈1.2×**。Bullet 把 TTFT 做到 54.9×、SM 利用率 86.2%，**端到端仅 1.21×**。**任何方向若收益全在形态 A，即已被证伪**；形态 A 只能写成"可与 C1/C2 复合" |
| **B｜专用体系结构** | 新数据通路 / PE 异构 / 近存 PIM，报 perf/area/energy | **按瓶颈类型绑定**：片内重编排 1.1–1.2× ／ 近存 2–6× ／ 专用加速器 14–387×。**memory-bound 的方向只能承诺 2–6×**。评审底线 = iso-compute-area + DC/PrimeTime/CACTI |
| **C｜硬件感知闭环** | 硬件 cost model **反解** C1 的结构超参 | **本轮暴露的最干净空位**：全库**没有** DiT 侧统一 cost model（无 step-level PME、无 DiT roofline breakpoint、无 Timeloop/Accel-Sim 的 DiT mapping 评估） |

**现成的反解式（形态 C 模板）**：D4 的 `ρ ≈ 2× active experts`、D4 的 DIAMoND `OU: H=min(ρ_in,d_min), W=min(ρ_out,d_min·QB)`、D5 的 `T_forget = 5.172·N_S − 4.469 (R²>0.999)`、D2 的"同 FLOPs 下 pre-fusion 层数 vs token 数"曲线与显存反解帧数。

**写作配方：C1 + C2 + 形态 C（必备闭环）+ 形态 A 或 B（按会场二选一）。**

---

## 2. 全局论证纪律（六方向共用）

### 2.1 C2 一律按 memory-bound 论证

单请求 batch=1 时**算术强度 CI ≈ eff_bs ≈ 1，低于任何屋脊点** → 六个方向的 C2 全部走 memory-bound；"compute-bound 报 MFU 对标 FA3/cuBLAS"只在明确跑 batch / CFG 折叠的段落才成立。三条连带后果：

1. **SM 分区对 memory-bound 恒无效**（HBM/L2/crossbar 共享）→ **所有"门控与主干并发共跑"的方案作废**。
2. 稀疏 / 门控路径（D3 gate、D4 experts）必须用 **S-MFU / S-MBU**（激活参数口径）。常规 MFU 在 batch>1 时高估资源利用 **1.5–3×**（MoE-CAP；探针开销 ≤2.7%）—— 用错分母会自缩 headroom。
3. **单屋顶 roofline 在极端算力/带宽比硬件上 MAPE 达 127%（H800）vs 11%（H20）**，**4090/5090 正属该类** → 必须多流水线屋顶或同卡实测 SoL，**禁用纸面峰值**。

**屋脊点口径**：A100 必须按 SKU 分——**40GB 1555 GB/s → 屋脊 ≈208 FLOP/Byte；80GB 2039 GB/s → 屋脊 ≈153**。CI 阈值写「208 (40GB) / 153 (80GB)」，不写单一值。

**访存型 kernel 的合格线**：A100 **83–86%**、H100 **92.3–94.2%** HBM 带宽利用率；寄存器溢出会跌到 40.3%。**最优性主张须过 ~85%。**

### 2.2 实验协议（写死，全方向适用）

- **计时的 `synchronize` 必须在计时区内** —— 否则 CUDA Graph 会报出 ~1000× 的假加速。
- **四层消融 A/B/C/D 必须分离 graph-capture 增益与 fusion 增益。**
- **50 步与 4–8 步两套数并报，`guidance_scale=1.0`** —— 否则加速与"打开 CFG batching"不可区分。
- **CFG 双口径**：CFG 使 batch 翻倍的前提是未做 guidance distillation；G1 profile 必须分 CFG on/off 报。
- **训练网格对齐**：Wan 推理用 sigma shift，微调若按训练默认分布采样 t，学到的路由/门控会在推理 σ 网格上失配。
- **消费卡锁频**：SOL-ExecBench 给了 B200 锁 1500 MHz 的协议，**GeForce 上没有** —— boost 波动本身能吞掉 10% 的 kernel 收益，必须先定协议。

### 2.3 编译 / 运行时的硬约束

- **CUDA Graph 在少步 DiT 下是必需项而非可选**：launch 5–20 µs/kernel → replay 2–5 µs（省 90–98%）。**但每张图约 200 MB 显存** → 24GB 4090 上可捕获的变体组合数**必须先数**。
- **CUDA Graph 没有条件 kernel**：grid/block/SMEM 全固定，多形状需多份图实例，数据依赖的控制流变更触发重建（MoE 动态路由实测 **40–100 ms/次**；ACS 记录 input-dependent 图上 DAG 构建占总执行 **47%**）。
- **唯一干净的动态决策解**（FlashInfer 模式）：**plan / run 分离** —— 数据依赖的 `plan()` 在 **CPU 上执行且刻意不被捕获**，写固定 workspace 偏移；固定 grid 的 persistent kernel 内按 flag 走分支。**把 N 个状态实现成 N 个分别 launch 的 kernel 按步选择，不是 graph-safe 的。**
- **`torch.compile` / Inductor 不做 GEMM epilogue 融合** —— vault 中一手记录只到 pattern-based 融合（select/pointwise/reduction）+ 布局传播 + Triton codegen。**所有实测的 epilogue 收益都出自手写 CUTLASS/CuTe-DSL。** 不要再用"编译器已经免费做了"作为任何论据。
- **Dynamo 在数据相关分支上必 graph break**（FX 非全程序 IR，guard 重编译随分布漂移上升）→ 弹性/门控**必须暴露为编译期枚举的 variant index**，不能是 Python 控制流。
- **launch overhead 不能扛 G1**：本地无一手的 launch 占比数据，且此类增益天然落在 1.1–1.2×。只能作 C3-A 组件。

### 2.4 Ada 平台的移植性警告

**Ada 无 thread-block-cluster DSM** → FlashFuser 那条 reg→SMEM→DSM 的溢出层在 4090 **根本不存在**；**H100 融合论文的 C2 论证不可直接移植**。Triton 3.5.0 在小批上只有 **0.67×/0.66×** → **原型实现优先 TileLang/CuTe**（TileLang 已在 4090 验证，~70 行达 FA3 的 98%）。按 ARGUS 口径，pipelining + warp specialization + 指令调度**必须一次性叠齐**，否则原型会被测成负收益。

---

## 3. 硬件常量表（⚠️ 动笔前必读）

**纪律：下列 13 格全部落定之前，不写任何以 SMEM 定 tile、或以 vault 峰值做 MFU 分母的句子。**

证据等级：**A=笔记显示**（vault 一手正文）｜**B=可推断**（两条以上笔记推出）｜**C=派生/Web**｜**D=无证据**。

### 3.1 RTX 4090（现役）

| 项目 | 值 | 等级 |
|---|---|---|
| SM 数 | **128**（AD102，boost ~2520 MHz），两处独立笔记一致 | A |
| threads/SM | **1536** | A |
| 显存带宽 | **1008 GB/s**（笔记误写作 "HBM"，实为 GDDR6X，需 bandwidthTest 复核） | A |
| 显存容量 | **24 GB** | A |
| 峰值算力 | **165（BF16）vs 330（FP16）TFLOPS —— 冲突未决**，见 §3.3 | A/A |
| FP8 | **有**（`mma.f16.f8.f8.f16` 自 SM 8.9 起）；FP8+FP16 累加 4×、+FP32 累加仅 2× | A |
| Sparse Tensor Core | **有** | A |
| NVLink | **无** → token 交换须经 PCIe + host（PCIe 硬线 32–64 GB/s） | A |
| FP4 / NVFP4 | **无正面证据**（所有 FP4 笔记的硬件列表只含 Blackwell，从不列 4090） | D |
| **shared memory / SM** | **vault 中零条** ⚠️ | **D** |
| **L2 容量** | **vault 中零条** ⚠️ | **D** |

> **⚠️ 必须记住的一条**：流传的「4090 SMEM = 128 KB」是**张冠李戴**——该数字所在笔记的上下文明写「**Ampere 架构，RTX 3090 为例 / GA102 / SM ×82 / L2 6144KB**」，是 **RTX 3090** 的参数。**不是弱证据，是错挂，已从常量表彻底删除。**
> **可用的临时替代**：A40 实测 SMEM = 102400 B；QuantCache 实调 48 KB/SM。

### 3.2 RTX 5090（即将到位）

| 项目 | 值 | 等级 |
|---|---|---|
| 架构 | Blackwell **SM120**（QuTLASS 评测卡，与 B200 SM100 并列） | A |
| 显存容量 | **32 GB**（笔记实为 **5090D** 中国特供版） | A（卡型有别） |
| FP4 实测 | 单层 MXFP4 ≈**6×**（理想 8×）、端到端近 **4×**（vLLM 内测，Llama-3.3-70B 层形状，batch 1–256） | A |
| FP4 峰值 | ≈**1600 TOPS**（FP16 ≈200 TOPS）；SageAttention3 实测 1038 TOPS。**同页把 5090 错标为 SM100 → 数字不可单独取用** | A（载体已证伪一处） |
| **端到端锚点（最可直接复用）** | **SLA：Wan2.1-1.3B / 5090 / 95% 稀疏 —— 注意力 97s→11s（8.8×）、e2e 2.2×**；FW 13.7× / BW 6.8× vs FA2；超参 `k_h=5%, b_q=b_kv=64, φ=softmax`；微调 **2000 步 × batch 64 @480p**（<0.1% 预训练成本） | A |
| kernel 占比锚点 | fused reorder-and-quantize 仅占总 kernel 时间 **7.9%–17.0%**（seqlen 128→4096），GEMM 占 83.0%–92.1% | A |
| tcgen05 / TMEM | **冲突未决**，见 §3.3 | — |
| shared memory / SM | 传闻 **99 KB**（另处 101,376 B；147 KB 配置报 `OutOfResources`）—— **实测卡是 5070 Ti / PRO 6000 / GB10，不是 5090** | **C** |
| SM 数 / 带宽 / L2 | **vault 中各为零条** | **D** |

### 3.3 两处硬冲突（原样并置，不裁决）

**冲突 A｜sm_120 上是否有 tcgen05 / TMEM。**
- **主张"有"**：A 级，直接点名 RTX 5090 + SM120（QuTLASS 用 Triton kernel 为 `tcgen05.mma` 做 scale 重排）。**但该证据自身有裂缝**：笔记从未逐卡拆分 kernel 路径（"matmul 支持 CUTLASS/FlashInfer 多后端插拔"），**无法排除 tcgen05 路径只走 B200**。
- **主张"无"**：C 级且卡型不对（实测 5070 Ti / PRO 6000 / GB10，加载 tcgen05 模块得 `INVALID_PTX`，退回 `mma.sync.m16n8k32`）。
- **污染项**：另有三条笔记把 5090 归进 **SM100**，与 SM120 直接矛盾 → 该族笔记在"消费级 vs 数据中心 Blackwell"上没有分卡，**两边都不能用**。（TMEM 的"每 SM 256KB / 128 rows × 512 cols"规格出自 SonicMoE 在 **B300**，与消费卡无关。）
- **唯一出路**：真卡上跑一条 `tcgen05` PTX 探针。

**冲突 B｜4090 峰值 165 vs 330 TFLOPS。** 三条 A 级文本互斥：① 「peak BF16 **165** TFLOPS」（ZipServ roofline，据此算屋脊 164 FLOP/Byte）；② 「FP16 **330** TFLOPS non-sparse / INT8 660 TOPS」（AD102 理论规格）；③ 「FlashAttention2 在 FP16 峰值 330 下**仅达 165 = 50%**」（headdim=64 **实测**）。两条重建路径都只到 B 级、**不得升格**：**α 累加器差异**（FP16+FP16 累加 512 FMA/SM/cycle 是 FP16+FP32 累加 256 的 2×，则 165/330/660 自洽）；**β 把实测当峰值**（③ 明写 165 是 achieved 值，与 ① 的 peak 用法字面冲突）。旁证：两篇同族笔记对 INT8:FP16 比值给出 **4× 与 2×** 两个答案；5090 的「FP16 ≈200 TOPS」**低于** 4090 的 330，跨代不单调。

### 3.4 参照点本身也不干净

| 卡 | vault 中并存的值 | 
|---|---|
| H100 SMEM | **228 / 227 / 224 KB**（全部 A 级一手笔记） |
| A100 SMEM | **164 / 168 / 192 KB**（全部 A 级一手笔记） |

→ **"消费级比数据中心小多少"这个相对论证从起点就是浮的**，引用前必须先选定口径并注明来源。

### 3.5 13 格 UNRESOLVED

**4090（今天就能测）**：① **SMEM/SM 上限**（`cudaDevAttrMaxSharedMemoryPerBlockOptin`，**全表最危险的一格，直接约束 tile 尺寸**）② **L2 容量** ③ 峰值 TFLOPS 的**六格口径矩阵**（BF16/FP32-acc、FP16/FP16-acc、FP16/FP32-acc、FP8/FP16-acc、FP8/FP32-acc、INT8）④ FP4 是否可用 ⑤ 实际带宽。
**5090（到货第一天）**：⑥ `tcgen05`/TMEM 探针 ⑦ SMEM/SM（含 147 KB 是否真 OOR）⑧ SM 数 ⑨ 带宽与容量 ⑩ L2 ⑪ 各 dtype 峰值 ⑫ cluster multicast 是否退化为 1×1×1（决定 TMA multicast 类优化是否整体失效）。
**跨卡**：⑬ 统一 H100 / A100 的口径。

---

## 4. 【首推】方向 3：DiT block 级可训练三态门控 {重算 | rank-r 增量 | 复用}

**坐标**：视频/图像生成 ／ Wan2.1、FLUX ／ Ada 4090 → Blackwell 5090。**新轴 = 架构新**：U-Net 时代的跨步缓存栈需为 DiT 重做。

### 4.1 新颖性防线（2026-09-24 读完 7 篇对照组后重判：**三条只剩一条**）

**先例一：DiTPA（ISCA'26，开源，28nm / 4.37 mm² / 1.05 W）** 的控制结构就是逐去噪步的三态判定 {完整推理 | 跳过整轮 | 复用上轮输出}，叠列稀疏、跑在可重构 PE 阵列上（S1 1.74× → +S2 2.90× → +S3 32.60×，相对其自身 ASIC 基线；PE 利用率 98.36% vs GPU <20%）。差异：**动作规划 DiT，非文生视频**；ASIC 非 GPU。

**先例二：LearniBridge（ICML 2026）—— 直接吃掉防线 (a)。**
- 它定义的正是**跨去噪步残差** `e^l_{t→t−k} ≜ f^l(x^l_{t−k}) − f^l(x^l_t)`，闭式解 `ΔW^l = E^l_{t→t−k}(X^l_t)†`；对 X^l_t 做 SVD 发现奇异值快速衰减 → `rank(ΔW^l) ≤ rank(X^l_t) ≤ r`，据此断言"**最优修正天然低秩**"。
- Eq(12) `F^l(x^l_{t−k}; W) ≈ F^l(x^l_t; W + ΔW^l)` —— **用低秩适配器从缓存特征重建被跳过时间步的表示**，这就是 D3 所说的"rank-r 增量中间态"。
- **我原先设想的退路已被堵死**：不是"每步单独拟合"，而是**一套 LoRA 在所有 (t, t−k) 步对上共享**（loss 对所有训练步对求和），**且跨 prompt 通用**（100 prompt 分 20 组各解最优 ΔW，两两主子空间夹角一致地小）。训练只要 **3–5 条 prompt**。
- 实现：LoRA 只加在**最后一个 block g_L** 的 Q/K/V/O/FFN1/FFN2，基座冻结；推理时每 N 步全算一次，其余步把缓存的 x_t^L 直接喂进 LoRA 增强的 g_L，**跳过 g₁..g_{L−1}**。
- 数字（FLUX.1-dev / DrawBench）：~4.4× 档 ImageReward **0.9590**（TaylorSeer 0.9359、TeaCache 0.8975、ToCa 0.8352；原模型 0.9885）；~6.2× 档 0.8308（TaylorSeer 0.8041）。另报 **Wan2.1 4.10×、HunyuanVideo 5.75×、FLUX 5.87×**。

**先例三：DisCa（CVPR 2026，腾讯混元，开源）—— 不吃 (a)/(b)，但另辟一路。** "可学"的是**轻量神经预测器**（替代 TaylorSeer 的手工泰勒公式），用 MSE 500 iter 初始化 + **GAN 对抗训练** 1000 iter（预测器 lr 1e-4 / 判别器 1e-2，λ=1.0）。消融（HunyuanVideo，VBench 语义/质量/总分 69.3/81.1/78.7）：去掉可学预测器 **−2.9% 语义 / −0.7% 质量**；去掉 Restricted MeanFlow −5.9% 语义；去掉 GAN −1.2% 语义。最高 **11.8×**。

**→ 重判后的三条防线**

| 防线 | 裁决 | 依据 |
|---|---|---|
| (a) 中间态是 rank-r 增量 | **❌ 已被占** | LearniBridge 的全部机制就是跨步低秩修正，且跨步对+跨 prompt 双重共享 |
| (b) **门控/决策可训练** | **✅ 存活，且是唯一一条** | 7 篇**无一篇学习决策本身**：LearniBridge 固定间隔 N=5/6/8、DisCa 固定最大间隔 N=2/3/4、TeaCache 手设阈值 δ=0.8/1/1.4、ToCa/AdaCache/Chipmunk 完全 training-free。"gating"一词在全批语料里只出现在 Chipmunk 讨论 MoE 的相关工作段 |
| (c) GPU 文生视频 | **❌ 不成立** | LearniBridge §4.3 文生视频（Wan2.1）；DisCa 做 HunyuanVideo / HunyuanVideo 1.5；Chipmunk 在 H100 上做 HunyuanVideo(118k) + WAN2.1(76k) + FLUX.1-dev(4.5k) |

### 4.1.1 ⚠️ 比新颖性更硬的一条：两篇独立工作都反对"逐 block 缓存"

**DisCa 与 LearniBridge 各自独立收敛到"只缓存最后一层"（single-final-layer-cache），且 DisCa 明确论证多层缓存更差**：缓存结构"不仅决定显存开销，也显著影响并行环境下的计算效率……显然，只复用最后一层的 cache、而非复杂的多层 cache 结构，才是更合适的"。DisCa 因此声称其预测器计算**高度并行、访存压力低**，实测加速比与理论值之差"在误差范围内"。

**D3 的设计正是逐 block 三态门控 = 他们明确拒绝的那个"复杂多层缓存结构"。** 这不是新颖性问题，是**设计合理性问题**：D3 必须正面回答"为什么逐 block 值得付那份显存与并行代价"，否则 reviewer 会直接引这两篇。D3 现有的答复只有一条（v3 写的"block 粒度覆盖 attention+FFN+norm，不受 SLA 的纯 attention Amdahl 上限约束"）——**这条现在必须用真机数据撑起来，不能再只是论证。**

**两条可直接借用的 DiTPA 结论**：门控硬件成本仅 **0.23% 面积 / 0.05% 功耗、4 cycle 判定**（→ 形态 B 的面积异议消失）；结构规则"**沿门控不改变的维度（hidden size）优先 tiling**"。
**一条必须正视的风险**：DiTPA 实测**同类纯软件 GPU 实现只有 2.3×**，点名两项代价——**动态 token 长度使利用率降 25.1%、多模态数据管理增 35.4% 时延**。

### 4.2 G1（门槛是 6.72×，不是 1.0×）

- **硬基线**：QuantCache 全消融阶梯（单 A800-80G，Open-Sora 1.2，512²×64f，100 步，W4A6，含 VAE）：1.00 → **HLC 4.12** → +AIGQ 6.33 → **+SRAP 6.72×**。**D3 的原型必须在同等设置下超过 6.72×。** 采用 Open-Sora 1.2 标准设置以对齐四篇已发表数字。
- **预算分配**：**SRAP 只值 +0.39×**（6.33→6.72）→ 步内层剪枝不该进 D3 堆栈，预算全押"把跨步决策做得更好"。
- **必须先吃掉的 ~2×**：Flux-dev 12B 每步读 ~24 GB 权重，H100 3.35 TB/s → **7.2 ms/步纯权重**；**CFG batch 折叠把 TC 利用率从 50% 抬到 80%+**。不先吃掉，G1 的测量会把它算成自己的收益。
- **机会的物理来源**：图像 DiT 相邻步相似度**只在近对角高、非对角迅速降为 0** → 二态缓存的有效窗口天然短，**rank-r 第三态的作用正是把缓存寿命延长到近对角窗口之外**。

### 4.3 C1（结构被一条已发表的反面理由钳死）

- **rank-r 增量不能穿过非线性**：DiTPA 明确以"细粒度控制开销 + 对 GELU 等非线性不兼容"区分于 Cambricon-D/Ditto 差分与 EXION 细粒度稀疏。**必须二选一并写清**：(a) 增量只加在线性子块（QKV/FFN GEMM），非线性重算；(b) 让可训练门控吸收非线性误差。
- **可替换的现成 baseline**：HLC 是**三级 refresh 决策**（τ_max/τ_mid/τ_min），但计算模式仍是二值 {recompute | reuse}，阈值 δ₁/δ₂ 固定不训练。**D3 的新颖性正是 (a) 第三种计算模式而非第三个间隔，(b) 用学习替代手工阈值。**
- **门控输入信号现成且便宜**：SRAP 的相邻层特征余弦相似度 runtime 可算。
- **门控必须是 batch 级单掩码**：FlashPS 记录扩散连续批内**同一步所有请求走相同块数**，join/leave 只在步边界 —— 否则退化为 batch=1。
- **训练回退的预算**：本地唯一被 kernel 实测的低秩增量是 **Q-VDiT TQE，rank=1**，承载的是量化误差修正；校准 10k/15k iter，W8A8 单项 **12.5–12.9 GPU-h**。→ **须按 ≥12.5 h/配置预算，并论证 r>1 为何可负担。**
- **最强 pitch**：QuantSparse Tab.16 证明"在量化+稀疏之上叠固定阈值缓存"代价惨重（1.83→2.47× 换来 **VQA 90.58→86.24、Delta-FScore 恶化 5.4×**，远超 G2 红线）。**D3 = 在同一 2.47× 上把质量收回来。**
- **第二条复用轴（提示）**：DiTPA 记录 **60.1% 外存访问是重复权重加载** → 只管激活的门控会漏掉这 60.1%。

### 4.4 C2（可证伪命题 + 已被收紧的设计空间）

**最优性论证**
- Roofline（A40 BF16，breakpoint ᾱ≈215）：规则 rank-r 把 FLOPs 34G→17G 而 **α≈978 仍远在 breakpoint 之上**（对比 Monarch 123、BLAST 64 已掉进 memory-bound）。
- **⚠️ 但"规则 rank-r 优于不规则列稀疏 delta"这一断言已被 Chipmunk 正面反驳，不能再只靠 Monarch/BLAST 的类比**：Chipmunk 的列稀疏注意力 kernel 在 **93% 稀疏度下比 FlashAttention-3（ThunderKittens）快 9.3×**（基线为 H100-SXM5 上 650 TFLOP 的 FA3，CUDA 12.8 + PyTorch 2.5.0），且**列稀疏的近似误差比块稀疏低 2×**。它绕开"不规则模式喂不饱 tensor core"的办法是：把非连续行的稀疏 K/V **gather 成 shared memory 里的稠密 tile**（tensor core 需 ≥64×64 才达峰值吞吐，最大 tile 64×256）。**→ D3 的 C2 必须与这两个数正面比较：(i) 同稀疏度下 rank-r 的近似误差是否优于列稀疏的"块稀疏 2× 优势"，(ii) 三态融合 kernel 是否打得过 9.3× vs FA3。**
- **崩点是 `b×n×r` 中间张量物化**：C2 的最优性**只在 Δ 不回写 HBM 时成立**。LoRunner 给了实现路径（3 launch→1、激活读取 3×→1×）与代价表（rank=16 时 +5%）；FusedGemmAdd 在 epilogue 内累加 vs 独立 add kernel 实测 **3.12–3.86×**。
- **可证伪目标**：在所选 r 下，三分支融合 kernel 相对 full recompute 净收益 **>5%**，且 profiling 须显示 **Δ 全程驻留 shared memory / 寄存器**。

**r 的上界（先算 r_max，再谈 α）**
`r_max` = SMEM 的**一半**（双缓冲占一半片上容量）∩ **三态共享的同一寄存器集**（FA3 三级流水因寄存器溢出**劣于**二级，256 KB/SM）∩ Ada 无 DSM 溢出层。**⚠️ 4090 的 SMEM 容量目前是空格（§3.1），这一步现在算不了。**

**形态（有现成模板）**
- **HyTiS 模式**：多模式共存于**一个 persistent kernel**，**共享同一 SMEM layout、仅 tile 几何不同、双射 offset 切换**，零 sync、零 workspace；1.10–1.19× vs cuBLAS（H100/A100，4624 个 GEMM）。这既是三态融合的第二个独立可行性证据，也给出"模式切换本身"的诚实收益量级。
- **门控开销预算是个位数 %**：MPK 占 4 SM（~3%）；FlashMoE 1 block 仍保 93.17% SM 利用；Batched Load Scheduling 的决策态仅 512 B SMEM。**紧贴 >5% 的净收益门槛。**
- **控制开销上限**：MixFusion 给出 **<2 ms/block**（SD3 每步 40–50 ms、24 blocks）→ 门控只有约 **1/24 step** 预算，**必须批量化、不能 host 往返**。

**从 D1 继承的七条约束（D1 结案后唯一的存活价值）**

| 约束 | 内容 |
|---|---|
| 索引形态 | top10% KV 贡献 >90% 分数，但仅 **15.1%** 落在 5-token 半径、**48.5%** 距离 >10 → **必须索引表 gather，不可假设连续** |
| top-k 形态 | in-kernel 迭代 top-k 需全局同步 → 须改写为 **KV 序列维归约**（InfiniteHiP / FA2 不 split-K） |
| 粒度契约 | routing block size **必须整除融合 kernel 的 tile**（NSA=64 对齐 FA2）—— 编译期契约，非自由变量 |
| 掩码形态 | **TMA 不支持非仿射访问**，稀疏 BSR 会退回 `cp.async` → 掩码须保持仿射块结构 |
| 合格线 | 访存型 kernel 须过 **~85%** HBM 带宽利用率 |
| 分母纪律 | 92–95% 的 attention 占比是 **8×H100 / 200K token / 前向+反向训练**值，**不是单卡推理** → 单卡分母须自测 |
| 每步预算 | FlashPS 每步 batch formation **1.2 ms**；对 TeaCache **6×** 是 serving 下的对比基线 |

**⚠️ 一条未实测的假设**：唯一本地测量（XY-Serve，Ascend 910B）显示 score 矩阵由方阵变矩形后 **torch-npu MFU 53% → 47% → 30%**，论文明言**理论算力节省被 kernel 效率损失抵消**；另有 block 顺序反例——**{N,H,B} 得 600 TFLOPS @ 213 GB/s，而 {B,H,N} 得 494 TFLOPS @ 2390 GB/s**（实测算力与带宽朝相反方向移动）。**"融合后门控开销归零"必须实测，不能假设。**

### 4.5 C3（主线走形态 C）

- **形态 A 只能这么写**：不得宣称"使动态 per-step 决策 CUDA-Graph-safe"（PROBE 五件套已发表），只能论证"**DiT 门控提前一整步可知，结构上比 MoE routing 更易**"。实现走 plan/run 分离（§2.3）。
- **必须正交于已有编译收益**：Difflow 编译侧 DiT 基线 **1.58× 均值 / 2.13× max**（vs PyTorch-Inductor，A100/H100）；**batch=1 下只剩常量折叠 + 循环不变量外提**；属性特化不分区则 2ⁿ 引擎（14 输入 = 16384 引擎 / 11 天）→ 三态应走 **Brainstorm 式"编译期枚举 3 变体 + 运行时查表"**。
- **调度粒度**：TetriServe 扫描显示 **τ=1 步"开销过大"，5 步为稳健最优**（FLUX.1-dev，8×H100）→ 门控要么完全在 kernel/graph 内不碰调度器，**要么按 ~5 步粘滞**（则 C1 设计随之改变）。
- **形态 C = 主攻方向**：全库无 DiT 侧统一 cost model → **"硬件 cost model 反解 D3 的 r、门控粒度、粘滞步长"是最干净的空位。**
- **训练与 kernel 的联合模板**：SLA 把三种复杂度融进**同一 forward kernel + 一个 backward kernel**，微调 2000 步 × batch 64，平台正是 RTX 5090。差异需辩护：**SLA 门控单步内 attention 的 block pair，D3 门控跨步的整个 DiT block —— 轴不同，属可建之基。**
- **D3 不受 SLA 的 Amdahl 上限约束**：SLA 用 8.8× 的 attention 只换 2.2× 端到端，说明纯 attention 优化已饱和；**D3 在 block 粒度覆盖 attention + FFN + norm** —— 这是"块级而非注意力级门控"的具体论据。

---

## 5. 【支线】方向 2：多模态 LLM 弹性视觉 token 压缩块

**坐标**：图文/视频理解 ／ LLaVA-OneVision 或 Qwen2.5-VL ／ 4090 + host。

### 5.1 G1 / G2（六方向中最强，也因此成了创新性风险）

- **适用面须声明**：**只针对 decoder-only 拼接架构**。图像编码占 TTFT —— decoder-only **25–54%**，cross-attn 类才 65–79%（ModServe，vLLM 0.7.2 / 128×A100）；connector <0.4%。**不得把编码器份额算进加速上限。**
- **绝对分母**：视觉 token = prefill FLOPs 上限 **77%**；VisiPruner 已实测 **53.9% FLOPs↓**（LLaVA-1.5/InternVL2，免训练，中层仅 ~10/576 token 驱动融合）；LLaVA-OV-7B 32 帧×196 = 6272 token → **41.4 T**。LLaVA-Mini：FLOPs −77%、<40 ms、24GB 单卡 >10,000 帧。
- **G2 极宽松**：336 分辨率下 **1/4/16/64 token 的 VQAv2 = 77.6/77.7/78.1/78.5**，对比 576-token 的 78.5 —— **整段落在 1 点以内**。第二条独立 Pareto（SCOPE）：192 token（−66.7%）保 99.5%、64 token（−88.9%）保 96.0%。
- **G1 硬门：打分必须 map-free。** FlashAttention 不物化 attention map，任何从 map 读重要性的打分都会丧失 FA3，而回退非融合注意力会让**峰值内存上升**——FastV **+3.7%**、SparseVLM video **+54.8%**。本地正面反证：Representation Shift 吞吐 5.5×/4.4×、R@1 比基于注意力的剪枝高 **+7.2%**。
- **禁止讲省显存**：batch=1 时 KV 仅占显存 **7–35%**（batch=64 才 80–97%）→ 单卡单请求只讲延迟/FLOPs。

### 5.2 创新点已被两个轴同时占住

- **算法轴**：**VisionSelector** 已实现"单套 12.85M 权重、固定 20% 预算训练、推理期适配 30/20/10%"（保持 97.20%/94.83%/87.75%）；**AIM 甚至无需训练**（7 配置覆盖 2.51→99.63 T FLOPs ≈40×，VideoMME 最低 50.9 / 峰值 58.5，基线 58.2）。→ **"可学习 + 运行时可调 + 单套权重"不能再作卖点。**
- **硬件轴**：**Focus** 已在原语级抢先——**SEC**（attention softmax 处做 token 重要性 + top-k + offset 打分）与 **SIC**（GEMM tile 输出处做片上 tile 内相似度 gather/scatter）都已是加速器单元，28nm 开源，**+2.7% 面积 / +0.9% 功耗**，仿真 4.47× vs 密集脉动阵列、7.90× vs A100、**但仅 2.37× vs GPU+FrameFusion**。→ **形态 B 天花板很低，不走。**（注：在 Focus 上 map-free 约束消失——SEC 的 M·a·k cycles 被 image-attention GEMM 的 M·(M+T)·h·n/(a·b) 完全掩盖，"不在 critical path 上"；但这是设计期解析估算，且仍付 O(M²) map 物化代价。**在我们的 GPU 平台上 map-free 仍是硬约束。**）

**幸存贡献**：C2 + decoder-only concat 的 scoping + 8–32 token 端压过 AIM 的 Pareto + 机制差异（cross-attn latent **重采样** vs Top-K **选择**）。可攻击面：Focus 用**固定剪枝策略、无预算弹性打分、无 encoder/decoder 异构划分**。

### 5.3 C1 的三处硬性界定

1. **压缩块本身买不到 Pareto**：无 pre-fusion 时 1 token 掉 **6.1 点**（78.5→72.4），144 token 也只有 76.9；固定 1 token 增加 pre-fusion 层则 72.4→74.8→76.0→76.9→77.6 —— **同等 FLOPs 下增加 pre-fusion 层收益大于增加 token 数**。**pre-fusion 层数必须作为第二结构旋钮。**
2. **插入位置由 VisiPruner 定**：layer 1 全掩视觉 token 72.6→65.2（需视觉 attention sink），layer 2–7 由 system prompt 顶替**无损**，真正融合在 **9–15 层**（掩 top-10% GQA 61.95→54.09，掩 bottom-10% 无影响）→ **8-token 档在"一个 latent 充当 sink"的前提下可辩护。**
3. **低端必须文本条件化**：HiPrune @64 token 92.7% vs HiPrune++ 96.1%、POPE **73.0% vs 84.3%** → <64 token 的自变量是**指令条件**而非预算本身。

**压缩轴锁定帧内空间**：跨帧时序合并 VideoMME **47.4** vs 帧内空间 **52.3**（AIM, r=3.1%）。**打分器开销上限 0.6%**（AIM 自身 88.25 GFLOPs）。

### 5.4 C2（标杆已上移）

- **靶心不再是 VisionSelector 的 1.86× prefill**，而是 **MoDES 单卡 H200 2.03× prefill**（Qwen3-VL-MoE-30B，branch-free 掩码 + 哨兵 <1% 开销）；且必须在 **RESONATOR 类调度器之上**报增量（其纯 runtime 已 4.9× E2E、TTFT 5.1×）。
- **弹性 token 数的 kernel 解不是"任意形状"**：cuBLAS wave 量化在 M 变 64 时掉 **36%/21%**；SmoothGEMM 片上虚拟 padding **+14.6%**；XY-Serve 收敛到**有限固定形状集**离线调优（重排 <1% 延迟）；SonicMoE varlen-M 持久调度去掉 M_tile 整除约束。**C2 必须证明 tile 边界零代价。**
- **GQA 陷阱**：KV 先复制到 query head 数，(r−1)/r 为冗余 → **压缩比须 > 组大小**才净省；modality-aware 拆 cross/self-KV 比均匀多省 **30–50%**（新增一条可做轴）。
- ModServe 指明视觉前段本身是 compute-bound（~100% SM 活跃、DRAM <30%）—— 但在 batch=1 单请求下仍按 §2.1 的全局口径论证。

### 5.5 C3

- **形态 A（最低成本落点）**：**LExI 式 `moe_layers[j].topk=k_j`** —— 不动调度器/显存管理/kernel；**BrownoutServe** 的规则分离（结构参数 k 固定、阈值每迭代滞回调；1.58–2.07×，SLO 违约 73.68%→7.14%，**但精度损失约 5%，正压 G2 红线**）；**Cornserve** 证明 L_t ∝ 该类型算力成本时各类约束等紧。**弹性必须暴露为编译期枚举的 variant index**，否则 Dynamo graph break。
- **形态 C（现成模板）**：显存已反解帧数——同模型 int8 **48 帧** / fp16 **22 帧** / 双卡 25 帧；加上"同 FLOPs 下 pre-fusion 层数 vs token 数"曲线 + AIM 的 r_merge/(l1,l2) + VisiPruner 平均 23.9 层 vision exit → 可构成"给定单卡算力/显存反解 (token 数, pre-fusion 层数, exit 层)"的成本模型。
- **传输隐藏轴已封顶**：EPD-Serve overlap 15.27%→**98.78%**，纯 runtime +57~69% 吞吐 → **打分模块 offload 无空间。**

**⚠️ G2 报告必须逐方法给 Pareto 而非取平均**：同为 −97.6% 时 VisiPruner 61.3、SparseVLM 58.2、**PDrop 崩到 46.6**。

---

## 6. 【支线】方向 4：多模态 MoE 模态分组专家 + 预取

**坐标**：多模态理解 ／ MoE-VLM ／ 4090 + 大 DRAM。

### 6.1 前置条件（不满足则整个方向是 no-op）

**必须先承诺 offload 设定。** 专家常驻 4090/5090 显存时，前置门控/预取"**退化为无收益的额外 router**"。瓶颈口径必须是 **PCIe 字节/秒**。

### 6.2 G1（势能真实，但基线已被换掉三次）

- **塌陷数字**（DeepSeek-V2 236B、单卡 A5000 24GB + 512GB host）：decode 期每专家平均 **0.3–0.4 token**、GPU FLOPs 利用率 **0.1%**、**1 tok/s**；batch=1 的 MoE offloading GPU 利用率仅 **0.76%**。headroom ~100×。
- **但对照物不再是 naive offloading**：module-based batching 已拿回 **31×**（Bsz 75、41% 利用率、31 tok/s）；ProMoE 1.34–2.07×；同层 pre-attention 预测 **93.03%** 命中。**headline 必须是相对这三者的增量。**
- **放弃"隐藏传输"叙事**：完全隐藏 HtoD 需 expert batch > **2^11 tok**，单卡不可达。纯预取天花板本身也只有 ≈**8×**（expert fetch 占 ~88%、gating+compute+aggregation ~12%）。**超过该数的收益必须来自减少传输量本身。**
- **iso-bytes 下架构收益为零**：M100（N5A 399.8 mm²）LLaMA2-7B W4A16 decode **21.34 vs Thor-U 20 ms = 0.94×**，同受 273 GB/s DDR 限。

### 6.3 目标函数（本方向最重要的一条）

**T_moe = β · a_max + c_e** —— 延迟线性于**不同激活专家数**，与 token 数几乎无关（B 64→512 延迟基本不变，JANUS 离线 profiling）。

→ **优化目标 = min a_max。** 三条推论：
- C1 的成败标准从"命中率 / 字节量"改为"**是否降低不同激活专家数**"，且**必须同时报 a_max 与 e2e**（EPLB 1.5× 复制 → 激活专家数 **+30%**、decode 延迟 **−14%**：减字节不免费）。
- **加算力近乎免费**：decode b=1 强度仅 **~23 FLOPs/Byte**、利用率 <1%，B_min≈**9,400 tok**（H100/DSv3, n=256,k=8）→ 在线 B<100 铁定 memory-bound。**C1 可以加算力换更少专家**（如投机解码，A30 2.5×）。
- 论证必须 **per-batch**：batch 级激活异质 15–20% vs 聚合级 1.2%（Lynx）。

### 6.4 C1（两处正面反驳 + 一条口径修正）

**必须正面处理的两条反驳**
1. **"prefill 时工作集已知"不是新颖点**：Mixtral-8×7B/TruthfulQA 上 prefill 与 decode 专家激活分布余弦相似度全层平均 **0.89**，据此的 placement 已达 **8.7× decode 吞吐、0.13% 精度损失**。→ **必须论证模态段信号强于这个 0.89 的上下文信号。**
2. **命中率赛道已饱和**：93.03%（cloud 98.65% / edge 98.85%），0.15 ms CPU 预测器藏在 0.74–1.13 ms self-attention 下且无 bootstrap 问题 → 命中率维度只剩 ~7% 边际。**战场必须换成"传输字节"与 a_max。**

**shared 档的正确口径（易错）**
- **架构级、训练进去的 shared expert 要删**（DeepSeek-V2/V3、SharedFusedMoE 那类）——它是与**低**局部路由一致性相关性最强的架构因素（REAL 模型 SRP 前两组均不用 shared expert）。
- **但运行期常驻/预取层相反，是收益不是代价**：STEP 的**窗口票选临时共享专家**把每层每步 H2D 从 `k×expert_size` 降到 `(k−c)×expert_size`（k=2/c=1 即**减半**），命中常驻集则整段跳过 H2D；prefetch 命中 **85.5–98.8%（CNN/DM）、72.1–95.6%（LongBench）**，expert 取数占执行时间 **~88%**（Qwen3-30B-A3B / A100 / INT8，**文本 LLM decode，非 MoE-DiT**）。
- **可预测性正则会与负载均衡损失直接对抗**，必须在 C1 中显式预算。

**其他约束**
- 层间亲和性**本已是预训练模型的固有且 OOD 稳定属性**（~1000/3000 token 即可估出，Pile→C4/Dolma/Yelp 行归一化 0.989–1.005）→ **可预测性正则必须证明它在 router 已有可预测性之上还有增量**。唯一支持训练期干预的本地证据是 **gate input-sensitivity**（token-based 预测仅 58.3%、skip-based 在 QW-2 上 66.9%）。
- 模态冗余有机制而非仅相关性：vision token 与 FFN 权重更正交（角度→90°），FFN 对 vision token 的改变小于 text token；**MoDES 双模态阈值 τ_v > τ_t 是直接先例**。

### 6.5 C2 / C3

**C2 必须越过的既有线**：STEP **−50%/步**、MoE-Infinity **>80%**、MoE-SpeQ **96–99.85%** 命中；**SonicMoE 已把 gather 融入 prologue → gather 融合不可再作为贡献**。单卡残余瓶颈 = **fp8 requantize + 中间张量落 HBM**（fused MoE SoL 437–567 vs Triton 225–341 TFLOPS）。单卡与多卡目标**相反**（ARGUS 内聚 / CUCo 拆分）→ **全部 all-to-all 文献出本方向范围**。roofline 一律用 **S-MFU/S-MBU**。

**C3 选 A 或 C，不走 B**
- 形态 A 天花板：LLEP e2e **1.88–2.2×**（MoE 层 6.11× 塌缩到 e2e）、PROBE 1.32×/1.26×；JANUS 每 batch 重写路由 **<90 µs@4096**（决策开销可忽略）→ **e2e 须 >2× 才算越过纯运行时基线。**
- 形态 B 被否：ATU+PDU 把 hop 降 **−213×** 但吞吐仅 6.63×、**增量只有 +1.1×**；M100 的 GSDU 由共享 RISC-V CPU **串行仲裁**。
- **形态 C（推荐）**：**DIAMoND** 的 `OU: H=min(ρ_in,d_min), W=min(ρ_out,d_min·QB)` 一行反解 C1 结构超参；`ρ ≈ 2× active experts`（20 个 MoE LLM 上的折中常数）可直接作预取缓冲容量。**⚠️ NAND endurance ~10³ → 专家权重必须 write-once，动态专家重分组被物理排除。**

**⚠️ 立项前置测量**：MoDES 只给"每模态冗余度"（定性），**没有"模态间专家集合不相交程度"** → **必须先测 vision 与 text top-k 专家集的 Jaccard，并同时报 a_max。**

---

## 7. 【二期】方向 5：视觉历史层流式改造

**坐标**：流式/长视频理解 ／ Qwen2.5-VL 改造 ／ 4090。

### 7.1 主线（已换）

**不再以"蒸馏进 Mamba/SSM"为主张** —— SSM kernel 侧无 C2 空间（见 7.3）。现主线 = **StreamingVLM 式结构**：**sink 512 + 文本窗 512 + 视觉窗仅 16s、视觉先驱逐 + Contiguous RoPE**，单 H100 bf16 持 **8 FPS**；训练用 **Overlapped-Chunk Full-Attention**（W=24s / O=12s）保证训推一致（这正是 D5 本来要自己发明的机制）；训练成本 ~128 H100-day。

### 7.2 G1（分母必须先扣两项）

- **预处理占推理时间可达 40%**（resize/normalize/H2D，LiveStar 类分块流式）—— **不在模型内，封顶所有模型侧优化**；但它可与文本解码重叠，本身就是一个便宜的 C3-A 收益项。
- **基线不是全注意力**：XStreamVGGT 的 **4.42× 内存 / 5.48× 加速**已银行入账；LiveStar 纯 runtime caching **1.53×**。
- **"O(L)→O(1)" headline 已失效**：LongLLaVA @100K tokens，**1:7 混合比下内存 79.4 GB → 79.1 GB 近乎为零**（吞吐 2.6× 但内存不动）—— 存活的全注意力层仍持 O(L) KV。
- 量级锚点：TimeViper vanilla **128 帧即 OOM** → +ToMe ~5K 帧 → +ToMe+TransV **10K+ 帧**。**G1 原型必须在递增帧数下测 state 常量性，单点吞吐数无法过闸。**
- KV 增长口径：分块流式 W=64/S=32（50% 重叠）→ O(N²T) → O(W²+K·logT)；严格因果，不可跨未来帧并行。

### 7.3 C1 / C2

- **Samba 反驳"保留几层全注意力"**：16K 时**即使仅 1 层全注意力也无法外推**（PPL 10.29→13.66），而 SWA 可外推至 1M。→ 定稿形态：**视觉历史层 → 有界窗口，保留层 → sliding-window / sink+window。**
- keep-ratio 区间由 TimeViper 的 **7.1%（4/56，从零训练）**与 MambaInLlama/M1 的 **21%（6/28，蒸馏）**夹出。
- **两条互斥性必须在方法节显式声明**：(1) **state 压缩 vs vision-token reduction 二择一** —— token reduction 在 SSM 上失效有机制（pruning 的不可恢复损失沿递推**逐步放大**）和量级（**LTMP 在 Mamba-2-2.7B 上 PPL 4.10 → 4670.71**、Avg acc 63.8%→41.3%）双重证据；(2) **SSM state 就地更新不可回滚** → 流式 seek-back/重答/分支需要显式 **state-checkpoint pool** —— 这是 C3 形态 A 真正的承重贡献点。
- **SSM kernel 的 C2 头寸 ≈ 0**：Mamba2 scan 单体融合 4 kernel / 3 次 HBM → 1 Triton kernel，但"融合只是壳，性能来自**块内并行前缀扫描 + 极短串行段**"；**TorchInductor 10.57 vs 手写 PIKE-B 10.81（差 ~2%）**。另：SSM 侧削 **30% FLOPs 只换 1.35× 吞吐**（Mamba-2-1.3B）。**必须跨过的两条线**：LiveStar 1.53×、XStreamVGGT 5.48×。

### 7.4 若坚持形态 B

改以**中间数据 / DRAM 流量**为 C2 内存轴（**不是**常驻容量轴）：PipeFlash 把 score/prob 矩阵 **128 KB → 1 KB**（4.8×）、PipeSSD 中间数据 **642 KB → 58.5 KB**（11×）、DRAM 访问 **−6.8×**。
**但条件把它锁死**：自研 URSC 加速器（TPU 半芯片模拟，2×128×128 MXU）+ Hybrid-2.7B，**TPU DRAM 单 batch >32K 即 OOM，从未进入 100K 区间**；量纲也不同（每 block KB 级缓冲 vs GB 级端到端常驻）；4.8× 出自作用于 FA-2 的 PipeFlash（**注意力侧**）而非 SSM 数据通路。**不可外推到 100K 常驻容量结论。**
**硬件选型规则**：利用率近 100% **仅当 `block_size = d_head = d_state`**（5 个 Mamba-2 配置 130M–2.7B 间波动 <2%）；FA-2/SSD 异构模式会使**瓶颈跨层漂移**。
**PPA 约束须以不等式形式呈现**（Stratum）：`P_dram + P_compute + P_misc ≤ 45 W`、面积 ≤ 63%×121 mm²（36 MB SRAM、128 TFLOPS、内部带宽 19–34 TB/s）；**跨 interposer token I/O 仅 819 GB/s** —— 每步递归状态流量应对标这个量级，而非 HBM 峰值。

### 7.5 形态 C 的反解式 + 最严重的 G2 风险

`T_forget = 5.172 · N_S − 4.469`（R²>0.999）→ 设计规则 `T_train > 5.172 · N_S`。Mamba-2 的 N_S=256d；370M Mamba-2 已需 **T_train > 66.7K tokens**；按 16 vision tokens/frame，10K 帧 = **160K tokens** 蒸馏上下文。
**⚠️ G2 风险**：Passkey 精度随训练 token 增长而**下降**，且 **780M 比小模型更差** → **≤5% 必须在长流检索类任务上以 Pareto 报告，不能只报短片段指标。**
可行性支撑：attention upcycling **70 GPU-hours vs 370K（约 5000× 成本差）**；层选择可用 MEDA 的 `E_CM^l = −(E_TV^l + E_VT^l)`（prefill 一次，O(n_T·n_V)/层）。
**⚠️ 动机须诚实陈述**：StreamingVLM 的 8 FPS 是**它自己**在单 H100 上的吞吐，不是滑窗方案的天花板；去掉 sink 或 text window 只是 win-rate 73.64→69.68/66.76 的**退化而非崩塌**。**D5 不是在修一个坏掉的方案，而是要正面打赢一个已可用的 SOTA。** 且多模态 SSM 蒸馏质量本地零覆盖。

---

## 8. 【结案】方向 1：视频 DiT 跨 step 路由缓存

**结论：性能轴关闭，不单独立项。**

- **Amdahl 上界**：SLA 实测（Wan2.1-1.3B、95% 稀疏、RTX 5090）注意力 97s→11s（8.8×）、e2e 2.2× → 注意力占比 SLA 前 ≈61.5%、**SLA 后仅 ≈15.3%**。把 routing 开销折进这 15.3%（Twilight TokenSel~15% + Pruner~20%≈35%；MInference Block-Sparse 索引构建 25%）：**即使 routing 开销全部消除，e2e 仅 ≈1.06×；把整段 attention 归零也只有 ≤1.18×。**
- **形态 B 分支同时被否**：本地**无任何 video-DiT 加速器评测范式**（Timeloop / SCALE-Sim / NoC 零命中），方法学本身就是净新增工作量；且 memory-bound 下形态 B 只能承诺 2–6×（§1.2）。
- **复用轴本身也是错的**：vault 中唯一的硬复用数字在 **query 轴**（同一 **2×2×2 3D latent cube** 内相邻 query 与锚 query 的 critical-KV 索引重叠 **>92.4%**，跨 block 均值 80.1%）—— 这是**空间重叠**；**跨去噪步的 top-k block mask Jaccard，vault 里根本没有。**
- **新颖性被双向夹击**：InfiniteHiP 已实现分 stage refresh interval 的 mask 缓存（n_refresh=(16,8,4)、256K 上下文 110 µs/token、命中率 Stage1 71.67% / Stage1&2 98.75%）；DiTFastAttn/SSAR 已占跨步注意力图残差复用（SSAR 二阶残差、τ=5、SVD rank 16、开销 +0.2% 时间/+8–11% 显存）。

**唯一存活价值**：七条 C2 约束已转移至 D3（§4.4）。另两条可复用的观察：三态容差不对称（仅 8.1% 权重大于均值 1/N、约 45% 低于 1/(100N)；丢弃最小 45% 相对 L1 误差 **<3%**，仅保留最大 8.1% 误差升至 **≈33%**）支持"陈旧决策优雅退化进线性补偿分支"；误差纪律沿用 DiTPA —— **每 20 次跳过插入一次完整去噪**复位累积误差。

---

## 9. 【结案】方向 6：AdaLN 低秩共享 + 调制链融合

**结论：G1 不通过，且证据方向性相反。**

- **Amdahl 封顶**：DiT 原文自证 adaLN/adaLN-Zero 的 Gflops 增量**最小/可忽略**（唯一 ~15% overhead 的是 cross-attention）；实测 attention 占端到端 **92%/93%（1.3B/3B @200K）到最高 95%**（⚠️ 该数是 8×H100/200K/前向+反向条件，引用须标注）。
- **收益上限**：完全移除调制路径，唯一可查收益是 ELF-B **148M→105M（≈29% 参数）且无任何延迟数字** —— 两条笔记都把它写成**参数故事，从不是速度故事**。
- **C1 已是既有基线**：DSV Table 1 显示 **AdaLayerNormSingle 在 0.8B/2.7B/30B 三档 MovieGen-like 架构中原样沿用** —— 跨块共享调制表不是新 hack。
- **C2 无未认领的空位**：DiT 的融合组 **FG1–FG4 已公开**（LN→QKV→Attn→Residual 纵向；gate+up 以 `W=[W_gate|W_up]` 横向拼接；SiLU→Mul→down 纵向）。且 adaLN modulation 属"统计量可在 tile 内算出"的可融合类，而这类 **memory-bound 融合一旦显存带宽饱和即触顶**（FlashInfer-Bench 实证 RMSNorm 是唯一 agent 能追平/超过人类的 case-study kernel —— 写对就接近上限；GEMM/GQA 才是需要 pipelining 与 tiling 的战场）。
- **融合本身也被切开**：只有 **token 广播的 scale/shift/gate 能自由折叠**；LayerNorm 沿特征维的全局统计无法在 tile 内完成（LoKA 必须改成块级 RMS 的 BlockNorm 才可行）。要吃下整链就得动归一化本身 —— 那是另一个更大的 C1。
- **同构实测的诚实预期**：HATA 四次 launch 合一仅 **7.6% 端到端**；FlashFuser **58% 访存下降 + 3.3×/4.1× kernel 加速只换来 1.24× 端到端**。

**反向可用**：若日后要写手写 epilogue kernel（非本方向），**LongCat-Flash fused GroupedGemmAdd 3.12×–3.86×、MoEBlaze SwiGLU epilogue 最高 4× 激活内存下降 / 2×–6.2× 训练加速**是可引用的 kernel 级收益上限，但 **1.24× 的 e2e 天花板仍在**。
**唯一的复活条件**：**没有任何笔记测过 adaLN/modulation 自身占 DiT block 延迟的比例** —— §10 实验的 (d) 项会免费给出这个数。若该比例意外很高，可重开。

---

## 10. 最高优先级实验：一次跑动结算四个缺口

**实验**：Wan2.1-1.3B（480p, 5s）单次 instrumented 推理，逐步逐头 dump 池化后的 block 级注意力分数 P_c，计算 top-5% block 索引集在 step t 与 t+lag（lag=1,2,4,8）的 **Jaccard 重叠**，并在 **50 步与 10/4 步两种 schedule 各跑一遍**；同 run 开 torch profiler / nsys。

| 结算项 | 用途 |
|---|---|
| **(a) 跨步 mask 重叠率** | D3 的跨步冗余预算（D1 已结案，此项不再是"某方向的生死判据"） |
| **(b) 相似度随 schedule 长度的衰减曲线** | **D3 的基线要分 50 步 / 4–8 步两档报，这条曲线是分档依据**。⚠️ **已被 DisCa 定性回答，且结论对缓存类方法不利**：未蒸馏时相邻步相似、传统 reuse/插值可用；**经步蒸馏后步间差异显著变大，training-free 缓存失效**——DisCa 整篇的立论就是"少步区间必须改用可学预测器"。仍需自测定量曲线，但**不要再假设少步区间还有可观的跨步冗余** |
| **(c) 中间态/路由 cache 的显存足迹** | N≈30K 时 M_c 为 469×469/头/层 × 30 层 × 12 头 —— ⚠️ **能否驻留 4090 L2 目前算不了，L2 容量是空格（§3.1）** |
| **(d) DiT block 逐算子延迟分解 + 每步 kernel 数** | **D6 的复活条件** + launch-bound 比例 + **D3 的本地 Amdahl 分母**（在拿到它之前，不要写任何以"attention 占 92–95%"为分母的句子） |

**代价：一个下午，无需训练、无需 5090、无需多卡。没有第二个实验能以同样成本解锁这么多方向。**

---

## 11. 行动清单（按闸门顺序）

- [ ] **P0：§10 单次 instrumented 实验**（Wan2.1-1.3B，4090，一个下午）
- [ ] **P0：真卡硬件常量表 —— 13 格（§3.5）**。4090 侧 5 格今天就能测；5090 到货第一天测 7 格（首测 `tcgen05` PTX 探针）；另统一 H100/A100 口径。**在此之前不写任何 tile / MFU 论证。**
- [ ] **P0：读完 DiTPA（ISCA'26，开源）全文并逐条对齐** —— 它是 D3 的直接先例。**D3 的 intro 在读完它之前不要动笔。**
- [ ] P1：D3 起步 —— fork QuantCache，在 Open-Sora 1.2 / 100-step / W4A6 下复现 **6.72×** 全消融阶梯，再加 rank-r 第三态；先按"SMEM 一半（双缓冲）∩ 三态共享寄存器集"算出 **r_max**
- [x] ~~P1：补抓本地零命中的对照组~~ **已完成（2026-09-22 下载 / 09-24 入库分析）**，7/7 在 `papers_pdf/paper_dit_cache_comparators/` 与 `paper_secs/secs_dit_cache_comparators/`，清单见 [[20260922-dit_cache_comparators]]。裁决见 §4.1 —— **D3 三条防线塌到一条**
- [ ] **P0（新，主线抉择）：D3 与 D4 二选一的两个前置测量** —— (i) D4 的 vision/text 专家集 Jaccard；(ii) D3 唯一存活防线的可行性：**"可训练决策"相对 LearniBridge 的固定间隔 N 能拿到多少增量**。在同一 backbone/步数下，把 LearniBridge 的固定 N 换成一个学出来的逐 block 决策，若净增益 <5%，**D3 的最后一条防线也不成立**
- [ ] **P1（新）：回答"为什么要逐 block"** —— DisCa/LearniBridge 都只缓存最后一层并明确论证多层更差（显存 + 并行效率）。需用真机数据证明逐 block 的收益盖过这份代价，否则 D3 的结构选择站不住（§4.1.1）
- [ ] P1：**数 24GB 4090 上可捕获的 CUDA Graph 组合数**（每图约 200 MB）—— D3 三态门控的直接容量约束
- [ ] P2：D4 前置测量 —— vision vs text top-k 专家集 Jaccard，**并同时报 a_max**
- [ ] P2：微调可行性核账 —— SLA 配方是 2000 步 × **batch 64 @480p**，单张 4090 显然不成立；**Q-VDiT TQE 的 rank=1 校准就要 12.5–12.9 GPU-h/配置**，须据此定 r>1 的预算
- [ ] P2：**反向传播覆盖** —— SageBwd 已示前向 2× / 反向仅 1.2–1.6× 的不对称，且学术工作与 TensorRT **均未演示融合反传**；凡需微调的方向都要配反向 kernel
- [ ] P2：**冗余预算记账** —— D3 与少步蒸馏抢同一份跨步冗余，D2 与 D5 抢同一份 token。立项前必须算清，否则会重演 QuantSparse 的"1.83→2.47× 换 VQA −4.8%"
- [ ] P3：评测协议与成本 —— 质量指标目前无共同轴（VBench / Delta-FScore / win-rate / CosSim 各说各话）；须算一次可发表视频质量评测在单卡上的卡时
- [ ] P3：**消费卡锁频协议**（GeForce 上没有现成协议，boost 波动能吞掉 10% kernel 收益）
- [ ] P3：**新颖性 / 并发工作检索** —— 前两轮只问"vault 里有没有"，从未问"2026 年是否已被做过"。**DiTPA 这条正是这个缺口的代价。**

---

## 12. 检索纪律（避免重演证据丢失）

1. **omnisearch 响应体超 token 上限（150KB）时会报错而非返回空** —— 必须加**引号短语 + `path:` / `ext:` 过滤**，或 `mode:text` 后取 filename。**绝不要把检索失败记为"证据不存在"**（前一轮 kernel/硬件/芯片三层 100% Web 证据就是这个误判造成的）。
2. **复用闸门必须按内容判据，不能按完成信号。** 在**已修复**的那轮里，`--reuse-from` 仍拒绝了 3 份携带合法 `[ANSWER_AGENT_DONE]` 标记的答案（全部"命中 --reuse-exclude：均无/全无 note evidence"）——若只按"有结果就跳过"实现，这 3 份空答案会被当成有效证据带进结论。
3. **给 agent 的预算要足**：Phase 3/4 四个 horizon agent 曾全部撞上 `Reached maximum budget ($5)`（输入 188K token + 3.6M cache read），**其中两个在被切断前已写出 DONE 标记** —— 这是"内容判据优先于完成信号"的第二个理由。
4. **Obsidian 索引必须限目录**（vault 全量 21,252 md / 306 MB 会压垮渲染进程）：按 skill 的六目录口径设 `userIgnoreFilters` + `hideExcluded=true`（否则被排除的文件仍会被索引）。

---

## 网络文献索引

SANA [2410.10629](https://arxiv.org/abs/2410.10629) · VSA [2505.13389](https://arxiv.org/abs/2505.13389) · SVG2 [2505.18875](https://arxiv.org/abs/2505.18875) · Chipmunk [2506.03275](https://arxiv.org/abs/2506.03275) · MOHAWK [2408.10189](https://arxiv.org/abs/2408.10189) · Zebra-Llama [2505.17272](https://arxiv.org/abs/2505.17272) · LLaVA-Mini [2501.03895](https://arxiv.org/abs/2501.03895) · SVDQuant/Nunchaku（[HAN Lab](https://hanlab.mit.edu/blog/svdquant-nvfp4)）· PixArt-α [2310.00426](https://arxiv.org/abs/2310.00426) · RainFusion2.0 [2512.24086](https://arxiv.org/abs/2512.24086) · SLA（本地 + [thu-ml/SLA](https://github.com/thu-ml/SLA)）
