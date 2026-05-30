## Two-Level Command Processor (两级命令处理器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Level Command Processor (CP) 是论文为 wafer-scale multi-chiplet GPU 提出的架构扩展，将传统 GPU 的单层 Command Processor 重新设计为两级层次结构：(1) **Global CP**（wafer 级）——维护全系统 expert selection 和 placement 信息，运行 Task Allocation Algorithm 和 Data-Driven Predictor，负责将 MoE kernel 拆分为 per-die 子 kernel；(2) **Local CP**（每 die 内）——接收 Global CP 的子 kernel 和 prediction 配置，分配任务到本 die 的 SM，并将 prediction table 配置到 D2D controller。传统 GPU 的 Command Processor 将所有 SM 视为均等资源（忽略物理位置和 data placement），在 wafer-scale 场景下这导致大量不必要的 D2D traffic 和严重的负载不均。

从硬件架构角度拆解术语：
两级 CP 的硬件组织和工作流：

```
Hardware Organization:
├── Global CP (wafer 级，1 个)
│   ├── A76-class ARM core (~1.1 mm², ~1 W at 5nm)
│   ├── Expert Distribution Table (4.5 KB SRAM)
│   │   └── 每 entry: expert_id | die_id | n-bit distribution bitmask
│   ├── Cross-token Heatmap Cache (0.5 MB on-chip SRAM)
│   │   └── 缓存一层 heatmap (512 experts × 512 experts, 512-bit width)
│   └── DRAM: full heatmap 50 MB
│
├── Local CP × 25 (每 die 1 个)
│   ├── A72-class ARM core (~0.3 mm² each, ~280 mW each at 5nm)
│   └── 功能: SM task dispatch + D2D controller config
│
└── D2D Controller (每 die 1 个，含 ATU + PDU)
    ├── ATU (4.25 KB SRAM, 68-bit entries)
    │   └── 翻译远程 HBM 地址到本地地址
    └── PDU (128 B register file, 16-bit entries)
        └── Prediction Table: [expert_id → cp_en (1 bit), is_local (1 bit)]

Kernel Launch 工作流:
1. Host CPU → Global CP: "Launch MoE kernel for layer l"
2. Global CP:
   a. 读取 expert_reqs_dict (每 expert 的请求数)
   b. 运行 Task Allocation Algorithm → allo_plan
   c. 运行 Predictor → cp_en bits per die
   d. 打包: sub-kernel descriptor + prediction info
3. Global CP → Local CPs (via D2D): 发送 sub-kernels
4. 每个 Local CP:
   a. 分配 sub-kernel 任务到本 die 的 SMs
   b. 配置 D2D controller 的 PDU prediction table
   c. 等待 SM completion
5. Local CPs → Global CP: 报告 expert duplication statistics
6. Global CP: 更新 Expert Distribution Table
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Global CP 基于 A76-class ARM core（~1.1 mm², ~1 W at 5nm），Local CP 基于 A72-class ARM core（~0.3 mm², ~280 mW each）。总面积/功耗 overhead <0.04%。
- 设计可支持 100 layers × 512 experts per layer（远超当前 SOTA Kimi K2: 61 layers, 384 experts）。
- 硬件 synthesis：register files 用 Yosys synthesis，SRAM 用 CACTI modeling，均 scaled to 5nm（H100 process node）。
- 关键设计决策：Global CP 的 Expert Distribution Table 使用 n-bit bitmask（而非 single die ID）——bitmask 中每位表示该 expert 是否存在于对应 die，支持 expert 被复制到多个 die 的场景。
- 替代实现：若未来编程模型演化为 multi-GPU-like（die-level 控制），这些算法可在 host CPU 软件层实现而无需硬件修改——但 host CPU 实现 overhead 在 Dojo-Enhanced 上可达 42-51.6%（vs Global CP 实现几乎零 overhead）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
