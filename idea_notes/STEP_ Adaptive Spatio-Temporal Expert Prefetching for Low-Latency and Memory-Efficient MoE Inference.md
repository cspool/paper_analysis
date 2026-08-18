## STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

- baseline方法是什么？
  - baseline 是内存受限单卡下 MoE offloading 的两类主流方法：①启发式/静态预取类——llama.cpp（静态 layer-to-device 映射、无动态适应）、Mixtral-Offloading（按使用频率缓存专家 + 混合精度）、MoE-Infinity（activation-aware 预取，按序列级激活模式预测）、AdapMoE/HybriMoE（GPU-CPU 调度 + 缓存层级管理）、DAOP（按 per-sequence 激活模式在 CPU/GPU 间分配 + 预测预计算）、APTMoE（亲和感知流水线 + 层级加载）、MoE-Lightning（CPU-GPU-I/O 流水线 + 层级 roofline）；②压缩类——MoE-I2（专家剪枝 + 低秩分解），不重训则精度损失 5–10%。这些方法对每个 MoE 层一律按固定 top-k 激活专家、预取策略基于固定假设（启发式或静态），无法捕捉动态路由行为，prefetch 准确率低、带宽利用率差；且不利用专家贡献的空间不均（低排名专家权重 ≤0.05 仍被加载）与专家选择的时间连续性。
  - baseline 全栈执行例子（Mixtral-8x7B top-2，一个 decode token 穿过一层 MoE）：
    - 算法pipeline：gating 输出全部 8 个 routed expert 的 score，固定取 top-2（即使第 2 名权重 ≤0.05），两层之间无层内剪枝、每层都加载相同数量专家。
    - 系统框架：llama.cpp 把部分层静态映射到 CPU；MoE-Infinity 类系统按历史激活启发式预取，预取窗口/候选固定，命中率低（较 STEP 低约 10–30 个百分点）；无 batch 级 Serving 调度（batch=1 实时推理）。
    - 编译框架：论文未明确说明。
    - kernel调度：每层严格串行 gating→PCIe 取专家权重→GPU 计算→聚合；profiling（Qwen3-30B-A3B、A100 INT8）显示 expert fetching 占总执行时间 ~88%，传输与计算无重叠或重叠不足；专家常驻率低（低 CER 下多数选中专家在显存外）导致反复加载卸载。
    - 硬件架构：GPU HBM 只缓存少量专家（CER 25–75%），其余经 PCIe 4.0（64 GB/s）从 CPU 主存取；无 peer-GPU 缓存利用。
  - Baseline 缺陷：① 固定 top-k 加载低贡献专家造成冗余计算与 PCIe 流量（T_load = S·Σ(k_l − p_l·R_l)·t_expert 中 k_l 过大）；② 预取策略静态/启发式、命中率低（R_l 低），传输无法被计算隐藏（offloading 时间常超过计算时间，Fig.2a）；③ 专家激活模式跨任务/层/序列变化大（Fig.4/5），固定窗口策略要么上下文不足、要么过度预取浪费带宽。
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 STEP：混合静态-动态框架，用三个设计逐条对应 baseline 缺陷：① spatial-aware expert allocation（离线）——用校准数据收集每层 top-k routing score，按归一化阈值 θ 剪掉持续低贡献专家、按层动态决定 k_l（平均每层减 1–2 个），剩余权重重归一化，直接对应缺陷①（减小 k_l 即减小 T_load 与计算/带宽）；② temporal-aware prefetching / cached temporary shared experts（在线）——窗口内投票选 top-c 高频专家为"临时共享专家"提前常驻 GPU，使每步动态加载从 k 降到 k−c，配合独立 CUDA stream 异步传输 + CUDA event 把传输与计算重叠，直接对应缺陷②（提高 R_l 并隐藏 p_l 传输）；③ token-aware adaptive window（在线）——按每层实际 prefetch 准确率动态翻倍/减半/保持窗口大小 d_i（th_s=75%、th_f=40%、τ=3/4），窗口=1 时停用预取防浪费，直接对应缺陷③（把预取范围对齐任务/层特定模式）。三步对应 T_load 公式的三个因子 (k_l, p_l, R_l) 全部优化。
  - 论文方法全栈执行例子（同一 Mixtral top-2 decode token，STEP 把 k 从 2 减到 1.75 平均、窗口 8）：
    - 算法pipeline：校准后部分层 k_l 减到 1，低权重专家不再被激活；窗口内 top-2k=top-4 投票选 1 个临时共享专家，每层结构变为 1 shared + 1 routed；gating 仍对全部专家算 score 保证统计一致性。
    - 系统框架：基于 Hugging Face Transformers 实现，batch=1 实时推理；与 expert parallelism 正交——每个 EP group 独立维护热专家缓存并运行自适应窗口，NVLink/NVSwitch 下可用 peer GPU HBM 作二级缓存 P2P 取专家。
    - 编译框架：论文未明确说明。
    - kernel调度：当选临时共享专家在计算前由独立 CUDA stream 异步预取（cudaMemcpyAsync H2D）常驻 GPU；routed 专家 GEMM 与传输在不同 stream 重叠执行（CUDA 非抢占 kernel 执行）；每步序列最后一个预取 kernel 后 cudaEventRecord，CPU 用 cudaEventQuery 非阻塞查询，避免同步阻塞；窗口末按准确率更新 d_i。
    - 硬件架构：GPU HBM 在固定 cache 预算下用临时共享专家替换低使用率专家（不增显存）；低 CER(25%) 下因 k_l 下降与高命中率预取，TTFT 相对 llama.cpp 达 3.12×；decode 平均 1.54–2.22× 于各 baseline；V100/A100/H20 均 ≥1.3×。
  - 对应解决：① 空间分配按层剪掉低贡献专家，减少冗余计算与 PCIe 流量且保持精度（Table II–IV：平均专家数减 1 时 Accuracy 几乎不变；MMLU Mixtral 77.3→77.0）；② 临时共享专家 + 异步流预取把命中率提到 85.5–98.8%（CNN/DM）并隐藏传输延迟，decode 加速显著（消融：spatial 1.46× → +prefetch 1.52× → +adaptive window 2.22× → 全量 3.12×）；③ 自适应窗口按层/任务动态调窗，避免过早预取与过度预取（Fig.14b：早期 token 少预取、后期专家复用稳定后积极预取），与压缩（MoE-I2）与 offloading（APTMoE）均正交可叠加（Table V/VI：MoE-I2+STEP 24.1 tok/s，APTMoE+STEP 21.3 tok/s）。
