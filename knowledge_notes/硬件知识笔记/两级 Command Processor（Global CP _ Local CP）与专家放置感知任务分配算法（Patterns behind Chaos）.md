## 两级 Command Processor（Global CP / Local CP）与专家放置感知任务分配算法（Patterns behind Chaos）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Command Processor（CP，命令处理器）是 GPU 中负责把 kernel 解析成任务并分配到各 SM（Streaming Multiprocessor）的硬件模块：传统 GPU 把 CPU（SoC 内嵌）或专用前端作为 CP，把所有 SM 一视同仁地调度。本文（ISCA 2026，UCSD/Samsung/NVIDIA）针对未来 wafer-scale 多 chiplet GPU 的 MoE serving，把 CP 重构为两级层次化结构：(1) Global CP——wafer 级（整晶圆 1 个），维护系统级专家选择与放置信息，运行任务分配算法与数据驱动预测器；(2) Local CP——每个 die 内 1 个（Dojo/TSMC-SoW 25/24 个），接收 Global CP 派发的 sub-kernel，把任务分配到本 die 的 SM 并配置本 die D2D controller 的预测表。动机：当前 GPU 的 CP "oblivious" 到 SM 的物理位置与数据放置，在 wafer 上会制造大量 D2D 流量且无视 MoE 专家选择偏斜导致负载不均。Global CP 维护两类数据结构：Expert Distribution Table（每专家初始 die ID + n-bit 分布状态码，指示该专家当前存在于哪些 die，4.5KB/72-bit，wafer 级 1 份）与 cross-token heatmap（跨 token 专家激活历史，50MB 存 Global CP DRAM，0.5MB on-chip cache 一次缓冲一层）。任务分配算法（Algorithm 1，启发式，NP-hard 的近似）：输入 expert_reqs_dict（每个专家的请求数）+ expert_die_map（专家动态分布）；按请求数升序排序专家（小专家先分）；对每个专家生成候选 die 列表（存专家权重的 die + 相邻 die），按负载排序后限制到 max_split_num 个；请求以 50 个/块的粒度按 cost model（DRAM 访问 + 计算 + D2D 通信）选最优 die 分配并更新负载；最后合并同 die 的块成最终分配计划。面积评估（Table II）：Global CP 用 ARM A76 估算（~1.1 mm² / ~1000 mW）、Local CP 用 A72 估算（~7.5 mm² / ~7000 mW，每 die），整体 <0.04% 开销。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（kernel 启动到执行）：
```
1 kernel launch（MoE 层）
2 Global CP: 读 Expert Distribution Table + cross-token heatmap
3 Global CP: 任务分配算法(Algorithm 1) → per-die sub-kernels（按请求块分配到本地/相邻 die）
4 Global CP: 数据驱动预测器 → 下一 token 热门专家 → cp_en 位
5 Global CP → 各 Local CP: 下发 sub-kernel + 预测信息
6 Local CP: 分配任务到本 die 的 SMs
7 Local CP: 配置本 die D2D controller 的 PDU Prediction Table（is_local/cp_en）
8 各 die 计算专家 GEMM
9 Local CP 汇总专家复制统计 → 回传 Global CP 更新 Expert Distribution Table
```
Annotations：第 3 步是负载均衡的核心——把"哪个 die 算什么专家、算多少请求"由 Global CP 集中决定，而传统 GPU 由各 SM 抢任务（不感知位置）；第 4 步的预测结果直接决定第 7 步哪些远程专家要被本地复制；第 9 步形成反馈环使专家分布表随时间演化。该设计在 single-GPU-like 编程模型下成立：软件把整 wafer 当一个 GPU 编程，全部跨 die 决策由硬件（Global/Local CP）完成，类似 HDPAT/Hecton 的抽象，与 NVIDIA Blackwell/Rubin 的商业多 chiplet GPU 对齐（业界不暴露 die 级控制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Global/Local CP 用 ARM 核（A76/A72）实现（面积用 ARM core 数据估算），寄存器用 Yosys 综合、SRAM 用 CACTI 建模，均缩放到 5nm 匹配 H100 制程。使用：作为自研 Python 多 chiplet GPU 事件驱动模拟器（开源 https://github.com/zhongkaiyu/waferscale_gpu_moe_sim，DOI 10.5281/zenodo.19617713，Apache-2.0，CPU 可跑，main_ae.py 自动下载 HuggingFace 上的 MoE expert-selection traces 并复现 Figure 12）的一部分被模拟，输入真实专家选择 trace + 专家-到-die 分布 + die 拓扑/带宽/延迟参数，输出 MoE decode 吞吐与 hop count。效果：Allo+Pred 平均 6.6x 吞吐提升（Deepseek 7.0x、Kimi 8.2x、Llama 7.3x、Qwen 4.1x；Dojo 6.0x、TSMC-SoW 7.5x），hop 数减少 213x，batch 16384 时比 EP 快 1.44x；若任务分配改跑 host CPU，开销在 Dojo-Enhanced 下高达 19.3%-51.6%（PCIe 每 MoE 层传 Expert Distribution Table 与分配结果），说明算法下沉到 GPU CP 的必要性。局限：任务分配是近似启发式（候选裁剪 + 50 块粒度），块越小越准但开销越大。

涉及论文标题：
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
