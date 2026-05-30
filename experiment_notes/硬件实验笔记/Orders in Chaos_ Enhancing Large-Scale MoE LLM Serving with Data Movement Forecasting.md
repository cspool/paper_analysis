## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现核心：针对未来 **wafer-scale multi-chiplet GPU** 架构，提出两项硬件架构扩展：(1) **两级 Command Processor 架构**——Global CP（wafer 级）维护全系统 expert selection 和 placement 信息，Local CP（每 die 内）负责本 die 的 SM 任务分配；(2) **扩展 D2D Controller**——集成 Address Translation Unit (ATU) 和 Prediction Unit (PDU)，实现 hardware-managed HBM 方案，自动将远程 HBM 中的热门 expert 缓存到本地 HBM，减少 die-to-die 通信。
  - 实验比较：(1) Base（EP-like data placement + 无视 expert placement 的均匀任务分配）vs EP（所有 expert 计算分配到其所在 die，无 D2D 通信但负载严重不均）vs Allo Only（仅 task allocation 算法）vs Pred Only（仅 data-driven predictor）vs Allo+Pred（两者结合）；(2) 两种 chiplet 拓扑（Tesla Dojo 5×5 mesh vs TSMC SoW 8×3 mesh）；(3) 四种模型（DeepSeek V3, Kimi K2, Llama4 Maverick, Qwen3-235B）；(4) Dojo-Enhanced 配置（B300-like die，host CPU 开销评估）；(5) area/power overhead 分析。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - **自研 Python 事件驱动 multi-chiplet GPU simulator**。论文说明现有工具（Gem5, gpgpusim, mgpusim 为 cycle-accurate 但太慢；ASTRA-sim 不支持 single-GPU-like programming model）不适用，因此自研。
  - 开源链接：https://github.com/zhongkaiyu/waferscale_gpu_moe_sim，DOI: 10.5281/zenodo.19617713。
  - 验证方式：使用 8×H100 DGX server 的实测数据验证 simulator——(a) 单 GPU 执行一个 MoE expert（3 GEMMs），不同 batch size；(b) 两 GPU P2P 数据传输（4KB-4GB payload）。验证误差 < 5%。

- 模拟器模拟什么的性能，修改了什么。
  - 模拟目标：wafer-scale multi-chiplet GPU 上 MoE decode 阶段的 throughput（token/s）。
  - 建模组件：各 die 的 LLC、HBM、compute unit、D2D links，以及 central resource manager 捕获 contention 和 congestion。
  - 模拟的硬件配置：Dojo (5×5 mesh, 25 dies), TSMC SoW (8×3 mesh, 24 dies)。每 die 为 H100-like（1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s local HBM BW, 1.7 TB/s D2D BW）。Dojo-Enhanced 每 die 为 B300-like（4500 TFLOPS FP16, 180GB HBM, 8 TB/s BW, 2 TB/s D2D BW）。
  - 修改/新增的架构模块：
    1. **Global CP**（wafer 级）：含 Expert Distribution Table（每 expert 的 die ID + n-bit 分布 bitmask）、Cross-token Heatmap Cache（0.5 MB on-chip，缓存一层 heatmap）。运行 Task Allocation Algorithm 和 Data-Driven Predictor。
    2. **Local CP**（每 die 内）：接收 Global CP 的子 kernel 和 prediction 信息，分配任务到 SM，配置 D2D controller 的 PDU prediction table。
    3. **ATU (Address Translation Unit)**：4.25 KB，68-bit entry，将远程 HBM 地址翻译为本地地址（当远程数据已缓存到本地 HBM 时）。
    4. **PDU (Prediction Unit)**：128 B prediction table，含 cp_en（是否应缓存）和 is_local（是否已缓存）字段。在远程数据读取返回时做 duplication 决策。
  - Area/power overhead：总计 6.13 mm² / 8588.61 mW（< 0.04% of 25-die wafer）。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - **开源**：Case Study 1 代码仓库 https://github.com/zhongkaiyu/waferscale_gpu_moe_sim（Apache-2.0），expert selection traces（>150 GB JSON）https://huggingface.co/datasets/core12345/MoE_expert_selection_trace。
  - **模拟器全流程（以 DeepSeek V3 在 Dojo 5×5 配置上，batch size 16384 为例）**：
    1. **输入**：(a) Expert selection traces（JSON，从 HuggingFace 下载）——记录每层每个 token 的 expert 选择；(b) Hardware config（Table I）——topology、HBM BW、D2D BW、LLC latency/size 等；(c) MoE 模型配置——expert 数量、hidden size、intermediate size、GEMM 尺寸。
    2. **Simulator 初始化**：创建 25 个 die 对象，每个含 LLC model（64 MB，100ns hit latency）、HBM model（80 GB，3.35 TB/s BW，300ns local access latency）、D2D link model（1.7 TB/s BW，200ns latency per hop，XY routing）。Central resource manager 建模所有 die 间的 bandwidth contention。
    3. **Expert 初始布局**：Base 配置采用 EP-like placement——expert 均匀分配到各 die。Global CP 维护 Expert Distribution Table 记录每个 expert 的 die ID。
    4. **每层 MoE kernel 执行循环**：
       - (a) **Global CP 运行 Task Allocation Algorithm (Alg. 1)**：输入 expert_reqs_dict（当前 batch 中每个 expert 的请求数）+ expert_die_map（expert 所在 die）。按请求数升序遍历 expert → 生成候选 die 列表（含本地 die + 距离 1 的邻居 die）→ 按 block size 50 贪心分配请求到 cost 最小的 die → 合并同 die 任务生成分配计划 allo_plan。
       - (b) **Global CP 运行 Data-Driven Predictor**：用当前 token 的 expert selection 查 cross-token heatmap，取每行 top-n 预测下一 token 的热门 expert → 生成 cp_en bits 写入各 die PDU prediction table。
       - (c) **任务分发**：Global CP 将子 kernel + prediction 信息通过 D2D 网络发送到各 Local CP。
       - (d) **各 die 执行**：Local CP 分配子 kernel 到 SM → SM 请求 expert 数据。(i) 本地数据：直接从 LLC/HBM 读取；(ii) 远程非复制数据：D2D controller 通过 XY routing 跨 die 读取 → 返回时 PDU 检查 prediction table → 若需复制，写入本地 LLC + HBM，更新 ATU 和 is_local bit；(iii) 远程已复制数据：ATU 将远程地址翻译为本地地址，直接从本地 LLC/HBM 读取。
    5. **输出**：(a) MoE layer 总执行时间 → 计算 throughput (tokens/s)；(b) Hop count（所有 cross-die 通信的 Manhattan 距离之和）；(c) DRAM access breakdown（local reads / remote reads / local writes）。
    6. **关键结果**：Allo+Pred 在 DeepSeek V3 on Dojo 上实现 7.0× throughput 提升，hop count 降低 213×。Allo Only 已降低 hop 142×，证明大部分请求已分配到本地 die；Pred Only 额外降低 remote DRAM reads。
