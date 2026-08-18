## STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现为 MoE 推理中的运行时专家预取调度（expert prefetch scheduling）：在 CPU-GPU 异构平台上，把 CPU 主存中的专家权重经 PCIe 预取到 GPU 并与专家计算重叠，隐藏专家加载延迟。核心机制：(1) 临时共享专家预取——每个 token 窗口内经投票当选的 top-c 专家在窗口计算开始前整体预取并常驻 GPU，使每步动态加载的 routed 专家从 k 降到 k−c；(2) CUDA 异步流调度——prefetch 被实现为独立 GPU stream 上的异步数据传输 kernel，在专家计算 kernel 之前发起，利用 CUDA 非抢占式 kernel 执行实现传输与计算重叠；(3) CUDA event 同步——在每个解码步序列的最后一个 prefetch kernel 后插入 CUDA event 记录完成时间，供 CPU 高效查询（避免阻塞式同步），保证数据可用性的同时最大化流间重叠；(4) token-aware 自适应窗口——按每层 prefetch 准确率动态调整投票/预取窗口大小（th_s=75%、th_f=40%、τ=3/4），准确率过低（窗口=1）时禁用预取防带宽浪费；(5) EP 扩展——每个 expert-parallel group 独立维护热专家本地缓存并运行自适应预取，NVLink/NVSwitch 高带宽环境下可用 peer GPU HBM 作二级缓存直接 P2P 取专家。
  - 实验比较：以 Cached Expert Ratio（CER = GPU 可用专家槽位数 / 模型专家总数，取 25%/50%/75%）为内存约束控制变量，比较 prefill 阶段 TTFT 与 decode 阶段 TPOT/tok/s（batch=1，输入 512 token 定长 trace），baseline 为 llama.cpp（静态 layer-to-device 映射）、AdapMoE、HybriMoE、DAOP、APTMoE、MoE-Lightning；prefill 平均几何加速 3.12×/1.97×/1.52×/1.07×/1.07×/1.03×，decode 1.54×/2.22×/1.39×/1.15×/1.10×/1.25×；decode 延迟分解（Fig.12）显示 STEP 相对标准 MoE 显著压缩 expert offloading 占比；batch 1–8 与 V100/A100/H20 敏感性分析（Fig.18/19）。
- 后端平台是什么，配置是什么。
  - NVIDIA A100 80GB（4 卡，PCIe 4.0、64 GB/s switch，GPU-CPU 通信走 PCIe）+ AMD EPYC 7542 32-core CPU、512GB 主存；硬件敏感性实验另覆盖 V100、H20。
- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：基于 Hugging Face Transformers 库自研实现（STEP 及全部 baseline 均在其上实现）；batch size=1 模拟实时推理，CER 控制显存约束，指标为 TTFT/TPOT。修改了什么：在 Transformers 推理路径上加入离线校准（收集每层 top-k score 定 θ）、窗口投票选举临时共享专家、独立 CUDA stream 异步预取 kernel 与 CUDA event 同步、自适应窗口调参；未修改任何开源 Serving 框架（llama.cpp 等仅作 baseline 对比）。
  - 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未提供开源链接，联网检索（2026-08）未能确认 STEP 公开仓库；可复现部件为 Hugging Face Transformers（https://github.com/huggingface/transformers）与三个开源模型（Mixtral-8x7B-Instruct、DeepSeek-V2-Lite-Chat、Qwen1.5-MoE-A2.7B，HuggingFace Hub 可下载），STEP 自身的预取/窗口逻辑需按论文实现。评估原理与 kernel 输入→性能输出全过程（decode 阶段一个 token 穿过预取调度）：
    ```
    # 输入：当前层 gating 的 top-k 选择结果 + 当前窗口已当选的 c 个临时共享专家
    # 1) 计算开始前：对当选的临时共享专家在独立 CUDA stream 上发起异步 cudaMemcpyAsync
    #    （H2D，CPU→GPU PCIe），若已常驻 GPU 则跳过（prefetch hit）
    # 2) 计算 stream：GPU 上执行被选 routed 专家（k−c 个，若不在显存则同样走异步加载）与
    #    shared 专家的 GEMM kernel，与 stream 1 的传输重叠（CUDA 非抢占 kernel 执行 +
    #    多 stream 并发）
    # 3) 每个解码步序列最后一个 prefetch kernel 后插入 cudaEventRecord；CPU 以
    #    cudaEventQuery 非阻塞轮询，数据就绪后开始专家计算，避免同步阻塞
    # 4) 窗口结束：统计该窗口 prefetch 命中率 → 按 th_s/th_f 更新 r_i/d_i →
    #    确定下窗口窗口大小并举行新选举
    # 5) 输出：TTFT/TPOT 由 trace 计时给出；prefetch hit rate（CNN/DM 85.5–98.8%、
    #    LongBench 72.1–95.6%，Table II–IV）衡量预取调度质量
    ```
    （论文未涉及编译框架、硬件架构 RTL/模拟器、芯片设计层次。）
