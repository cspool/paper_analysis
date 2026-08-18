## Hardware-Managed HBM（硬件管理 HBM：ATU + PDU 自动复制缓存远程专家权重）（Patterns behind Chaos）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hardware-Managed HBM 是本文为未来 wafer-scale 多 chiplet GPU 提出的本地 HBM 自动管理机制：让每个 die 的 D2D controller 自动把"预测将被频繁访问的远程专家权重"复制到本地 HBM，并把后续远程读翻译成本地读，从而消除跨 die 流量。传统 GPU 把全部 HBM（本地 + 远程）视为统一地址空间，不区分本地/远程，产生大量不必要 D2D 传输。实现由 D2D controller 中新增两个单元完成：(1) Address Translation Unit（ATU，4.25KB/68-bit）——维护远程地址→本地地址映射，当数据已被复制到本地时把远程读重定向到本地 LLC/HBM；(2) Prediction Unit（PDU，Prediction Table 128B/16-bit）——每个专家一项，含 cp_en 位（Global CP 计算的"是否应本地缓存"，下发到各 die）与 is_local 位（该专家是否已在本地 HBM 缓存）。配合的数据驱动预测器运行在 Global CP：从 cross-token heatmap 中取当前 kernel 已选择专家对应的行，取每行 top-n 专家作为下一 token 的预测结果，据此为各 die 生成 cp_en 指导（预测器只复制当前已在读的热门远程专家，如示例中 die 读专家 1/4，预测下一 token 用 2/4/6，则只复制专家 4）。论文称为 hardware-managed HBM，别名 autonomous caching / hardware-managed memory。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
两类数据流：
```
# A. 非复制远程读（远程数据不在本地 HBM）
1 SM 读远程数据（D2D controller 命中但 is_local=0）
2 D2D controller 按常规路由请求到远程 die（绿色路径）
3 数据返回；PDU 查 Prediction Table 决定是否复制
4 无论是否复制都先把数据送 SM（不阻塞计算）
5 若需复制（cp_en=1）：写 LLC + 本地 HBM，ATU 记录远程→本地映射，PDU 置 is_local=1

# B. 复制后本地读（远程数据已在本地 HBM）
1 SM 读远程地址（ATU 命中）
2 ATU 翻译远程地址→本地地址，重定向到本地 LLC
3 LLC + 内存控制器取本地 HBM 数据返回 SM（蓝色路径，无 D2D）
```
Annotations：第 4 步"数据先送 SM 再后台复制"使复制不阻塞首次访问；第 5 步是自学习——第一次远程读触发复制，后续访问全走本地；is_local 位避免重复复制。预测器（Global CP）在 kernel 启动时从 heatmap 生成 cp_en，heatmap 是跨 token 的历史激活模式（Insight 1/2：专家选择在 token 级、layer 级可预测）。效果（DRAM 访问 breakdown，Figure 14）：baseline 大部分读是远程读；Pred Only / Allo Only / Allo+Pred 把远程读转成本地读，Allo+Pred 只剩"极热门专家"需跨 die 计算，并靠本地缓存进一步减少远程读。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：ATU/PDU 都是小表（4.25KB/128B），在 D2D controller 内实现，面积功耗开销 <0.04%（Table II：ATU 0.0048 mm²/334.25mW×25 die、Prediction Table 0.002 mm²/55.75mW×25 die，5nm 工艺，寄存器用 Yosys、SRAM 用 CACTI 建模）。使用：在自研多 chiplet GPU 模拟器（https://github.com/zhongkaiyu/waferscale_gpu_moe_sim）中模拟——模拟器建模 LLC、HBM、计算单元、D2D 链路及中央资源管理器（竞争/拥塞），用真实 8xH100 DGX 的 MoE 层 GEMM 与 P2P 传输（4KB-4GB）实测校准，误差 <5%。该机制与任务分配算法正交可叠加：Pred Only 降 hop 4.5x（性能 3.0x）、Allo Only 降 142x（6.3x）、Allo+Pred 降 213x（6.63x）——说明缓存复制先消除通信瓶颈，再叠加分配算法后瓶颈转向负载均衡。局限：预测基于历史 heatmap，需足够的层内/跨 token 相关性（论文实测 top 20% 下一专家候选覆盖 47%-80% 条件概率质量，Llama4 最强、DeepSeek 最弱）。

涉及论文标题：
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
