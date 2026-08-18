## Stitched Dies / Reticle Stitching（reticle 拼接制程）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reticle stitching：光刻单次曝光的 reticle 尺寸有上限（ConBin 引用 ~858mm²/die），超大芯片通过把同一 reticle 级版图多次曝光拼接实现——多个 reticle 级 die 在晶圆上相邻排列，互连金属线跨过 die 之间的 scribe line（常规切割死区）实现 die-to-die 电气连通，最终在晶圆上制造出"近晶圆尺寸"的大芯片。ConBin 的建模：WSC = 多个 stitched die 组成的 2D mesh，每 die 含 8×8 核阵列（每核 8MB SRAM）、die 尺寸 16.66×22.17mm；四种规模 [5×6, 8×9, 12×13, 16×17] dies = [40×48, 64×72, 96×104, 128×136] 核（默认 128×136）。Web 证据：Cerebras 与 TSMC 合作用 cross-reticle stitching 使互连跨 scribe line 形成整晶圆单芯片（WSE 系）；Apple/Intel 等专利描述 stitch routing（BEOL 布线跨 die 区域、金属密封环）与 double interconnects for stitched dies；Tesla Dojo 走晶圆级 chiplet 路线。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
stitching 决定 WSC 的物理拓扑与故障影响面。ConBin 假设所有 die 完全相同且冗余互连模板全片统一（保证可制造性，R_max=6 条/核）；跨 die 边界相邻核直连、与 die 内通信同延迟（与 Cerebras 公开实施例 [39] 一致）；多跳冗余链直接连接相距若干 mesh hop 的 router，延迟按跨越 hop 数成比例；stitched-die 阵列规模扫描 [5×6..16×17] 验证扩展性（F_norm 保持 86.43%–88.19%、设计时间 1.31–6.28 min）。故障跨 die 分布（聚簇 + 随机），修复需在不规则拓扑上保全局连通。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
版图级——同一 reticle 版图重复曝光，拼接边界预留跨 scribe 布线通道（专利中常见 pre-formed die routing + stitch routing，金属密封环保护）；物理约束——>50mm 长线显著增大误码率（ConBin 保守限 25mm 等效线长）、工艺布线支持 ≤6 hop 长链 [7]；建模使用——stitched-die 结构是 ConBin 故障注入与修复仿真的基座（Gaussian 簇 + 随机噪声故障图 [48]，每模式 512 实例）。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
