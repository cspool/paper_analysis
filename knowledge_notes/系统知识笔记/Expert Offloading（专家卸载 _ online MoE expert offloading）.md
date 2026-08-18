## Expert Offloading（专家卸载 / online MoE expert offloading）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Offloading 是把 MoE 模型中的专家参数从 GPU 显存（HBM）卸载到更慢但容量更大的存储（CPU DRAM/SSD），使超出显存容量的 MoE 模型能在受限 GPU 上运行的技术。利用 MoE 的稀疏激活特性（每 token 只激活 k 个专家），大部分专家大部分时间不被使用，可常驻慢速存储；典型方案把 attention、token embedding、router 等 common parameters 常驻 GPU，所有专家权重放 CPU 内存，被激活时经 PCIe 加载到 GPU 或直接在 CPU 上计算。按是否依赖预先知道的 workload 分 online（MoE-Infinity、HybriMoE、SMoE，动态调度）与 offline（MoE-lightning、expert pruning，预先优化）两类。SMoE 论文的核心动机：边缘 GPU 显存放不下全部专家时，非驻留专家的 PCIe 加载（比 GPU 计算慢 10–100×）与 CPU 计算（慢约 1 个数量级）主导 TPOT——Qwen/A6000 上 low-score 专家加载占 TPOT 42%、top-score 加载 29%、GPU 计算 29%。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Expert Offloading 在 MoE 推理中的执行流程（以 SMoE 在 A6000 48GB 上跑 Qwen2-57B-A14B 107GB 为例，需约 150GB CPU 内存）：①初始化：common params（attention/LN/router/shared experts）驻留 GPU，全部专家存 CPU DRAM，GPU 缓存 k 个专家/层；②每层 gate 打分 → 检查 expert residency：命中则 GPU 直接计算；未命中则二选一（PCIe 加载到 GPU 或 CPU 直接计算）；③decode 阶段逐 token 重复，专家换入换出成为 memory-bound 解码主要延迟来源；④优化手段：缓存策略（LRU/score-aware eviction）、prefetching（预取下一层专家与当前层计算重叠）、CPU 计算与 PCIe 加载流水线、residency-aware routing。相关系统：MoE-Infinity（activation-aware 预取+缓存、EAMC 请求级激活矩阵预测）、HybriMoE（CPU-GPU 调度+缓存管理）、FineMoE（概率感知细粒度缓存）、DeepSpeed-Inference（layer-wise 加载，非 expert-aware）、llama.cpp（纯 CPU）。关键 trade-off：offload 越多显存需求越低、但传输延迟越高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：dvmazur/mixtral-offloading（Mixtral-8x7B on T4 16GB，quantization+LRU+speculative prefetch）、MoE-Infinity（开源，activation-aware expert offloading）、HybriMoE（基于 ktransformers 的 CPU-GPU offloading）。SMoE 的 online expert offloading 实现在 https://github.com/goingshr/SMoE：Python 3.13 no-GIL 运行时，config 控制 replaceScoreRatio/window_size/if_prefetch/if_usecpu/if_replace，GPU_MEM 参数给定显存预算，模型权重经 HuggingFace 下载到 parameters/ 目录、数据集自动下载。使用例子（运行 S3）：`MODEL_NAME=qwenmoe GPU_MEM=43 CONFIG_PATH=configs/qwen2moe_config.json bash run.sh`，脚本自动跑 Gaokao/triviaqa/WiC/Race-mid/gsm8k 五个数据集并STEP 补充视角（ISCA'26，混合静态-动态专家卸载）：STEP 把 expert offloading 从"启发式/静态预取"升级为"空间剪枝 + 时间预取 + 自适应窗口"三件套，直接优化卸载加载时间 T_load = S·Σ_l(k_l − p_l·R_l)·t_expert（Eq.1）的三个因子：k_l（离线层内按归一化路由权重阈值 θ 剪低贡献专家）、p_l/R_l（窗口投票选举临时共享专家 + 命中率自适应调窗）。系统侧差异：STEP 在固定 cache 预算下用当选临时 shared 替换低使用率专家（不增加显存），预取与计算经独立 CUDA stream 重叠（见 kernel 调度层条目），且与 EP 正交（每 EP group 独立维护热专家缓存、peer GPU HBM 可作二级缓存）。评估：batch=1 实时推理、CER 25/50/75% 控制显存约束，prefill TTFT 相对 llama.cpp/AdapMoE/HybriMoE/DAOP/APTMoE/MoE-Lightning 几何平均 3.12×/1.97×/1.52×/1.07×/1.07×/1.03×，decode 1.54×/2.22×/1.39×/1.15×/1.10×/1.25×；与 MoE-I2（压缩）和 APTMoE（调度）正交可叠加（Table V/VI）。

解析日志得到 TPOT 与 GPU cache hit ratio（对应论文 Fig.12/13）。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
