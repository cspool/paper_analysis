## Wafer-Scale Multi-Chiplet GPU for MoE Serving (晶圆级多芯粒GPU的MoE推理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wafer-Scale Multi-Chiplet GPU 是将多个 GPU compute die 和 HBM die 通过先进封装技术（如 TSMC CoWoS/SoW, Samsung X-Cube, Intel EMIB）集成在单个 wafer/interposer 上的 GPU 架构。论文研究的两种具体拓扑：(1) **Tesla Dojo**：5×5 2D mesh，25 个 die，每 die 1000 TFLOPS FP16，80GB HBM，3.35 TB/s HBM BW；(2) **TSMC SoW (System-on-Wafer)**：8×3 2D mesh，24 个 die，同样 H100-like 配置。各 die 通过 D2D links 互联（1.7 TB/s per adjacent die pair，200ns link latency），使用 XY routing。这类架构将整个 MoE 模型（200B-1000B）容纳在单个芯片上，支持 batch size >10,000，total HBM >3 TB，PFLOPS-level 计算能力。

从硬件架构角度拆解术语：
Wafer-scale GPU 上的 MoE 推理执行流程（baseline，无优化）：

```
1. 模型部署: 所有层的所有 experts 分布在 wafer 的 HBM 中
   Expert Placement: EP-like, 每 die 均匀分配 E/num_dies 个 experts

2. MoE Kernel Launch:
   Host CPU → Global CP: "执行 layer l 的 MoE computation"
   Global CP: uniform 分配任务到所有 SMs (忽略 die 物理位置)

3. Per-SM 执行:
   for each token assigned to this SM:
       selected_experts = gate_output[token]  // top-k expert IDs
       for each expert e in selected_experts:
           die_of_expert = expert_location[e]
           if die_of_expert == current_die:
               read expert weights from local HBM (300ns)
           else:
               // D2D read: multi-hop XY routing
               request traverses N hops × (200ns/hop)
               + remote HBM access (300ns)
               + return path (N hops × 200ns/hop)
           execute GEMM (gate_proj, up_proj, down_proj)

4. 瓶颈: Expert selection skewness → 热门 expert 所在 die 成为 bottleneck
   冷门 expert 所在 die 空闲。D2D traffic 占据大部分执行时间。
```

论文提出的架构扩展（Global CP + Local CP + ATU + PDU）在此基础上增加了：(a) Global CP 的任务分配算法——将请求重定向到负载较轻的邻居 die；(b) Hardware-managed HBM——自动将远程热门 expert 缓存到本地 HBM；(c) Data-driven predictor——预测下一 token 的热门 expert 并预缓存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Wafer-scale GPU 的驱动力：single-die size 受限于 photomask dimensions (800-1000 mm²)，先进封装技术使多 die 集成成为必然。AMD MI300 (8 compute chiplets)、NVIDIA Blackwell (2 chiplets)、Rubin (4 chiplets) 是当前代表。
- 论文的模拟器实现：自研 Python event-driven simulator（开源），建模各 die 的 LLC (64 MB, 100ns hit), HBM (80 GB, 3.35 TB/s), compute units, D2D links (1.7 TB/s, 200ns/hop, XY routing)，以及 central resource manager 捕获 contention。
- 论文提出的架构修改面积/功耗 overhead <0.04%（6.13 mm² / 8.59 W per 25-die wafer, 5nm process）。
- 未来演进：TSMC SoW 技术支持 up to 24 compute dies + 96 HBM dies on single wafer (>200,000 mm²)。论文也评估了 Dojo-Enhanced 配置（B300-like die: 4500 TFLOPS, 180GB HBM, 8 TB/s BW），GPU 性能 outpace interconnect bandwidth，更凸显 on-GPU-command-processor 实现的必要性。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
