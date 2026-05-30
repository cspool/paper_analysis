## Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

- baseline方法是什么？
  - Baseline 方法包括：
    - **通信调度**：RCS（Random Communication Scheduling，随机通信调度）和 SJF（Shortest Job First，最短作业优先调度）。
    - **GPU 分配**：RGA（Random GPU Assignment，随机 GPU 分配）。
    - **专家共置**：Lina（将同模型的最热和 coldest expert 共置在同一 GPU），REC（Random Expert Colocation，随机跨模型专家共置）。
  - 全栈执行例子（以 Colocating+Heterogeneous 的 Lina baseline 为例，2 个 MoE 模型各 8 expert，异构 GPU 集群）：
    - **算法层**：MoE 模型（B/16, B/32，来自 Google LiMoE，每模型 8 expert，4 层），Gate 网络使用 top-k routing 为每个 token 选择 expert。FFN 处理分配的 token，Aggregation 合并输出。
    - **系统框架层**：Lina 将同模型的最热门和最冷门 expert 打包在同一 GPU。因同模型 expert 受到同步 all-to-all 通信约束——所有 GPU 必须等通信完成后才能开始计算——即使同一 GPU 上有两个 expert，一个 expert 计算完毕也必须等待另一个的通信完成。通信使用 NCCL 集体通信库，采用同步 all-to-all 实现。GPU 分配为随机。两个模型的 all-to-all 通信时间直接叠加（无交错）。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch eager mode）。
    - **Kernel层**：论文未明确说明（使用标准 GPU kernel，通信通过 NCCL）。
    - **硬件架构层**：异构 GPU 集群通过 big switch 无阻塞网络互联。同构集群带宽统一 100 Gbps；异构集群包含 100/80/50/40 Gbps 四种带宽 GPU。SJF 调度优先发送流量最小的 token，但在 all-to-all 场景下无法减少带宽竞争（与 RCS 效果相当）。
  - Baseline 缺陷：
    1. **同步 all-to-all 通信导致 GPU 空闲**：同模型 expert 共置时，计算（FFN/Aggregation）必须等待所有 GPU 的通信完成，无法交错利用 GPU 资源，GPU 利用率极低（低于 20%）。
    2. **通信时间次优**：随机或 SJF 的 token 传输顺序未考虑接收端带宽竞争——两个 GPU 同时向同一目标 GPU 发送 token 会导致带宽争用，延长通信时间。
    3. **GPU 分配不考虑性能差异**：在异构集群中随机分配 expert，冷门 expert 可能占用高性能 GPU，而热门 expert 运行在低性能 GPU 上。
    4. **缺乏理论指导**：现有方法依赖经验方法，没有理论推导出最小推理时间的下界。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Aurora 系统，通过联合优化模型部署和 all-to-all 通信调度，在所有四种 GPU 集群场景下最小化 MoE 推理时间：
    (1) **通信调度（§4.2）**：Theorem 4.2 证明通过避免接收端带宽竞争的 token 传输顺序，可将通信时间压缩至由最大发送/接收 GPU 流量决定的理论下界 b_max。Alg. 1 提供构建最优传输顺序的多项式算法。
    (2) **跨模型专家共置（§6）**：将来自不同模型的专家放在同一 GPU，打破同步 all-to-all 约束——两个模型的计算和通信可完全交错（Model a 计算时 Model b 通信，反之亦然）。通过 bottleneck matching 找到最优 expert 配对（Case I: Theorem 6.2 交替热门-冷门；Case II: bottleneck matching + Hopcroft-Karp）。
    (3) **性能排序 GPU 分配（§5.1）**：Theorem 5.1 证明按 expert token 负载降序分配给性能降序 GPU 是最优的。
    (4) **完整理论分析**：覆盖四种场景，前三种给出最优解，第四种（Colocating+Heterogeneous, NP-hard）给出仅偏离 1.07× 的次优多项式解。
  - 全栈执行例子（以 Aurora Colocating+Heterogeneous 为例）：
    - **算法层**：同 baseline，MoE 模型本身不变。Aurora 在算法之下做系统级优化。
    - **系统框架层**：Aurora 接收 traffic matrix D_N、D_C 和计算时间作为输入。(a) 先解耦 3D 匹配为两个 bottleneck matching：先确定 Model a 和 Model b 的 expert 配对（最小化聚合通信时间的最大列/行和），再将配对后的 expert 通过 bottleneck matching 分配给 GPU（按性能排序）。(b) 通信阶段：对每个 all-to-all，Aurora 通过 Alg. 1 确定 token 传输顺序——以瓶颈 GPU 为起点，剩余 GPU 按流量降序排列，避免向同一目标 GPU 同时发送。(c) 执行时：同一 GPU 上的两个 expert（来自不同模型）交替使用计算和通信资源，完全消除同步等待——Model a 的第一个 all-to-all 与 Model b 的第二个 all-to-all 可在时间上重叠。
    - **编译框架层**：论文未明确说明。通信调度可通过在计算操作的 buffer 层调用 NCCL 集体库按指定顺序执行实现。
    - **Kernel层**：论文未明确说明（标准 GPU kernel + NCCL）。
    - **硬件架构层**：同 baseline 的 big switch 网络。Aurora 的最优调度将通信时间压缩至 b_max——由最大列/行和除以带宽决定的理论下界。在 Colocating+Homo 场景下推理时间加速达 2.38×，Colocating+Hetero 达 3.54×。

- baseline方法是什么？
  - Baseline 是在 CPU 集群上部署 MoE 模型进行推理（如 2×64-core AMD EPYC + 512GB DRAM），所有 expert 均分同一集群资源，按固定周期（月/小时）计费，空闲资源仍需付费。
  - 全栈执行例子（GPT2 MoE，10240 tokens，text generation）：
    - **算法层**：MoE 模型（GPT2 改造，12 MoE layers × 4 experts），gating network 使用 top-k routing 将每个 token 路由到最相关的 expert。token ID 是唯一直接可用的 token 特征。
    - **系统框架层**：使用 expert parallelism，每个 expert 分配到一个 CPU 设备。所有 expert 同时执行，通过 all-to-all 通信完成 scatter（gating→experts）和 gather（experts→next layer）。非 MoE 层通过 model parallelism 分配。资源按固定粗粒度周期计费，空闲资源（如冷门 expert 所在设备）仍产生成本。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch eager mode）。
    - **Kernel层**：论文未明确说明（使用标准 PyTorch CPU kernel）。
    - **硬件架构层**：CPU cluster 节点间通过数据中心网络通信。所有 expert 的模型参数常驻内存（512GB DRAM），无冷启动问题，但每个设备无论负载高低均占用固定资源。
  - Baseline 缺陷：
    1. **计费模式不灵活**：CPU 集群按月/小时计费，即使 expert 在推理间隔期间空闲也产生成本。
    2. **无法差异化资源配置**：所有 expert 分配相同资源，冷门 expert 资源浪费，热门 expert 可能资源不足。
    3. **无 serverless 通信优化**：scatter-gather 通信在 CPU 集群上的 pipelining 设计（如 PipeMoE、MpipeMoE）依赖 GPU 集群硬件架构，无法直接迁移到 serverless 平台。
    4. **缺乏专家热度预测**：CPU 集群可在推理过程中动态调整资源分配，但在 serverless 平台上函数部署需要数分钟，无法动态调整；需要预先知道 expert popularity 来配置函数内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：在 AWS Lambda serverless 平台上构建端到端 MoE 推理系统，包含三个核心模块：
    (1) **贝叶斯专家选择预测器**：使用 token ID + position ID + attention ID 三维特征，通过重新设计的后验概率计算方法（Bayes 定理 + 两重积分引入 P'(f2) 和 P'(f3)），在推理前预测每个 expert 的 token 负载。
    (2) **三种 serverless scatter-gather 通信方法**：pipelined indirect transfer（通过 S3，minibatch pipeline 重叠下载/上传/计算）、non-pipelined indirect transfer、direct function invocation。支持按 MoE 层混合选择，在 payload size 限制下灵活权衡。
    (3) **BO + ODS 全局优化框架**：将 MoE 部署形式化为 MIQCP 问题，ODS 算法（O(|E|) 时间）联合决策 communication method、memory config、replica count；多维度 ε-GS 迭代调整 key-value table 优化预测准确性，在 billed cost 反馈下收敛到近优部署。
  - 对比 baseline 全栈执行例子（GPT2 MoE，10240 tokens，AWS Lambda）：
    - **算法层**：相同 MoE 模型结构，gating network 不做修改。但增加了**前置专家预测**：从 profiled data 中统计 (token_id, position_id, attention_id) → expert 的映射频次；对新 token，用 token ID 查表，position ID 假设均匀分布，attention ID 用最高 attention score token ID 近似，通过两重积分计算 P(N_{e,i}|f1') = ∫∫ P*(N_{e,i}|f1',f2,f3) · P*(f1',f2,f3)·P'(f3)/P*(f1',f2) · P*(f1',f2)·P'(f2)/P*(f1') df3 df2，取 argmax 得预测专家。
    - **系统框架层**：
      - 每个 expert 部署为一个独立的 AWS Lambda 函数，内存按预测 popularity 差异化配置（热门 expert 3008MB，冷门 128MB）。
      - 热门 expert 可复制多份（最多 8 副本），每副本处理一部分 token。
      - Scatter-gather 通信由 ODS 算法逐层选择最优方法：
        - 小 batch（256 tokens）：direct transfer（a^e=3），gating 函数直接调用 expert 函数传输数据。
        - 大 batch（2560 tokens）：pipelined indirect transfer（a^e=1），gating 函数将 minibatch 写入 S3 → expert 函数从 S3 下载+计算，同时 S3 上传前一批处理结果 → 非 MoE 层从 S3 下载聚合。
      - 按毫秒粒度按量计费（GBs），无请求时零成本。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：论文未明确说明（CPU serverless 函数使用 PyTorch 标准 kernel）。
    - **硬件架构层**：AWS Lambda 自动分配 vCPU（内存越大 vCPU 越多），每个函数独立执行。S3 bucket 作为中间存储 relay 数据。函数冷启动延迟通过预热（warm start）缓解。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（粗粒度计费）→ serverless per-GBs 计费，按需调用，无空闲成本。billed cost 降低至少 75.67% vs CPU cluster。
    - 缺陷2（均分资源）→ 基于预测 popularity 的差异化内存配置 + 专家副本机制。热门 expert 多资源，冷门省资源。
    - 缺陷3（通信无优化）→ 三种 serverless 专用通信方法（含 pipeline 设计），按 MoE 层 + token 量灵活选择，兼顾 payload size 约束。
    - 缺陷4（无预先预测）→ 三维 token 特征的贝叶斯预测 + BO 框架迭代优化。比 Lina（仅用 token ID）预测更准确，比 LSTM 方法更轻量。
