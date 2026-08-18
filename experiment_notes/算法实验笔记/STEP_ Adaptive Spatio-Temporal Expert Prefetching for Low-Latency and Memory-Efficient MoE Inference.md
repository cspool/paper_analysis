## STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是面向内存受限单卡 MoE 推理的混合静态-动态优化框架 STEP，核心为三项算法：(1) 离线 spatial-aware expert allocation（空间感知专家分配）：用校准数据集逐层收集 top-k routing score，按归一化权重阈值 θ 识别低贡献专家并削减该层 routed 专家数 k_l（例如 top-4 权重 0.62/0.21/0.13/0.04 → 分配 3 个专家；0.72/0.18/0.08/0.02 → 分配 2 个），剩余专家权重重新归一化保持输出一致；默认 θ：Mixtral 0.25、Qwen 0.13、DeepSeek 0.07；(2) 在线 temporal-aware prefetching / cached temporary shared experts（缓存临时共享专家）：把输出序列切成 token 窗口，窗口内每步跟踪 top-2k 专家（而非 top-k）并投票，窗口结束时按票数选出 top-c 专家作为下个窗口的"临时共享专家"，使每层有效结构从 j shared + k routed 变为 j+c shared + k-c routed，动态加载从每步 k 个降到 k−c 个；临时共享专家常驻 GPU 但仅被 gating 选中时才计算，且在固定 cache 预算下替换低使用率专家、不增加显存；(3) token-aware adaptive candidate window selection（Token 感知自适应候选窗口）：为每层维护奖励分 r_i 与窗口大小 d_i，按 prefetch 准确率自适应调窗——准确率 > th_s(75%) 时 r_i+1、累计达 τ(3 或 4) 则窗口翻倍；准确率 < th_f(40%) 时窗口减半；介于两者之间窗口不变；窗口缩到 1 时禁用实际 prefetch 但继续统计投票。总目标最小化专家加载时间 T_load = S·Σ_l(k_l − p_l·R_l)·t_expert（Eq.1）。
  - 实验比较：① 准确率/生成质量与 prefetch hit rate——以平均每层激活专家数（Avg. #Experts，由 θ 控制）与窗口长度（Window Size）为自变量，对比 Origin（未优化）与不同 (Avg. #Experts, Window Size) 组合（Mixtral 2→1.75/1.5；Qwen 4→3/2.5/2；DeepSeek 6→5/4/3），指标为 MMLU/Arc-e/PIQA/WinoGrande Accuracy、CNN/DM 与 LongBench Rouge-L + Prefetch Hit Rate（Table II–IV）；② 端到端 prefill（TTFT）与 decode（TPOT/tok/s）速度对比 llama.cpp、AdapMoE、HybriMoE、DAOP、APTMoE、MoE-Lightning 六个 baseline，prefill 平均几何平均加速 3.12×/1.97×/1.52×/1.07×/1.07×/1.03×，decode 为 1.54×/2.22×/1.39×/1.15×/1.10×/1.25×（Fig.10/11）；③ 消融：spatial allocation 单独 1.46×、+prefetch 1.52×、+adaptive window 2.22×、全量 3.12×（Fig.13）；④ 与 MoE-I2 压缩、APTMoE offloading 的正交性（Table V/VI：MoE-I2+STEP decode 24.1 tok/s、TTFT 470.3ms；APTMoE+STEP 21.3 tok/s、TTFT 531.2ms）；⑤ 批量 1–8、硬件 V100/A100/H20 敏感性分析（Fig.18/19，STEP 始终 ≥1.3×）。
- 硬件平台是什么，配置是什么。
  - 4× NVIDIA A100 80GB GPU，经 PCIe 4.0、64 GB/s switch 互联；AMD EPYC 7542 32-core CPU，512GB 主存；GPU-GPU 与 GPU-CPU 通信均走 PCIe（实验刻意不用 NVLink peer-GPU 共享以保证公平）。硬件敏感性实验另覆盖 NVIDIA V100 与 H20。
- 模型是什么。数据集和bench分别是什么。
  - 模型（3 个代表性 MoE，见表 I）：Mixtral-8x7B-Instruct（32 层、0 shared、8 routed、top-2、激活 13B/总量 46.7B、无 shared expert）；DeepSeek-V2-Lite-Chat（26 层、2 shared、64 routed、top-6、激活 2.7B/总量 14.3B，routed expert (2048,1408)）；Qwen1.5-MoE-A2.7B（24 层、4 shared、60 routed、top-4、激活 2.4B/总量 16B）。这些模型专家参数总量超过单 GPU 显存，必须 offloading。
  - 数据集/bench：常识推理 ARC、PIQA、WinoGrande（Accuracy），MMLU（Accuracy），摘要生成 CNN/DM 与 Longbench(Summarization)（Rouge-L + Prefetch Hit Rate）；延迟评估从多个数据集采样定长 trace 保证跨模型/策略可比。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方开源链接，联网检索（2026-08）未能确认 STEP 的公开代码仓库；实现基于 Hugging Face Transformers 库，batch size=1 模拟实时推理。算法 pipeline 执行例子（伪代码级，以 DeepSeek-V2-Lite top-6、校准后分配 4 个 routed + 2 个临时 shared 为例）：
    ```
    # == 离线阶段（spatial-aware allocation，per layer l）==
    # 1) 校准数据集前向，收集每层 top-k routing score 分布
    # 2) 对第 l 层按归一化权重阈值 θ 剪枝：score/Σscore < θ 的低贡献专家被移除，
    #    有效 routed 数 k_l 下降（如 6→4），剩余权重在 softmax 后重新归一化
    #    （y_routed = Σ w_i^r · E_i(x)，权重重归一保证输出一致性）
    # == 在线阶段（decode，per window of size d_i）==
    # 3) 每 decode step：gating 对全部专家算 score（含已当选的临时 shared），
    #    记录每步 top-2k 专家并投票（出现即得 1 票，反映频率与选择强度）
    # 4) 窗口结束：按票数选 top-c（如 c=2）专家为下一窗口"临时共享专家"，
    #    结构变为 (j+c) shared + (k−c) routed；shared 全部在计算开始前预取到 GPU，
    #    每步动态加载从 k 降到 k−c
    # 5) 窗口末评估 prefetch 准确率 → 更新 r_i/d_i（>75% 累计 τ 次翻倍窗口、
    #    <40% 减半、否则不变）；d_i=1 时暂停 prefetch 仅统计
    # 6) 计算 y = y_shared(含临时 shared 的平均/加权和) + y_routed(top-(k−c))
    # 张量化示例：一个 decode 步 → gating 输出 (64,) score 向量 → top-2k=top-12 投票
    #   → 2 个临时 shared 权重 (2048,1408) 已常驻 GPU，其余被选 routed 专家经 PCIe 加载
    #   → 计算 4 个 routed 专家 GEMM（每专家 (2048,1408)×(1408,) 激活向量）+
    #   2 shared GEMM，全部输出按 softmax 权重聚合
    ```
    （论文未涉及编译框架、硬件架构 RTL、芯片设计层次。）
