## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoESys 的 inference 阶段包含两大核心组件：（1）**Graph Optimization Pipeline**——6 步流程（Graph Fusion → Distillation & Compression → Graph Conversion → Graph Segmentation → IR Optimization → Deployment），将训练的动态图转为静态图后做 kernel fusion、分布式切分并部署；（2）**Ring Memory Offloading**——当 MoE 模型超过单 GPU 显存时，将 CPU-GPU 内存构建为环形内存区，每个 decoder layer 的 expert 参数在 CPU 上存 N 份副本、GPU 上缓存 K 份副本，计算第 i 层时释放第 i 层参数并异步加载第 (K+i) 层参数，形成 "计算-释放-加载" 的流水线。
  - 实验比较：（1）MoE inference throughput：对比 DeepSpeed，不同参数规模（10B/106.5B/209.6B）和 GPU 数（1/8/16）下的 tokens/s；（2）Ring Memory Offloading：48.2B 参数 32-expert MoE 模型在 16×A100(40G) 上，有/无 overlapping offloading 的耗时对比及 GPU memory 节省比例。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB（大模型 inference），A100 40GB（ring memory offloading 实验，16 GPU）。
  - CPU: 论文未明确说明型号，用于存储 expert 参数副本并提供 copy engine。
  - Storage: SSD 存储模型参数文件，CPU memory 缓存 expert 参数。

- 开源Serving框架是什么。修改了什么。
  - 框架：PaddlePaddle / PaddleFleetX（https://github.com/PaddlePaddle/PaddleFleetX）。MoESys 作为上层系统集成。
  - 修改内容：
    - Graph Optimization：实现 6 步推理图优化 pipeline——Graph Fusion（合并分布式策略消除参数冗余）、Distillation & Compression（teacher→student 减少 expert 数）、Graph Conversion（动态图→静态图，使用 PaddlePaddle JIT `paddle.jit.to_static`）、Graph Segmentation（手动或自动选择分布式策略切分子图）、IR Optimization（kernel fusion 等 pass）、Deployment。
    - Ring Memory Offloading：为 MoE 模型设计 ring memory 调度器，管理 CPU↔GPU expert 参数传输，利用多个 CUDA stream 实现 compute 与 H2D copy 的部分重叠。
    - Kernel 层：Fused Multi-head Attention（来自 NVIDIA BERT MLPerf 实现）、Custom H2D/D2H kernels（CUDA Pinned Memory）、Custom AlltoAll communication。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX，论文称代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - MoESys Serving 全流程（以 209.6B MoE 模型在 16×A100 上做 text generation 为例）：
    1. **输入**：用户 query tokens 进入 MoESys inference server。
    2. **Graph Optimization（离线）**：
       a. Graph Fusion：原始动态图 + 分布式策略 → 合并冗余参数。
       b. Distillation & Compression：teacher MoE（多 expert）→ student MoE（少 expert），通过 Mixture-of-Students 方式。
       c. Graph Conversion：`paddle.jit.to_static` 将动态图转静态图。
       d. Graph Segmentation：根据可用 GPU 资源自动或手动选择 expert parallelism + tensor slicing 策略，将静态图切分为分布式子图，插入必要的通信 op。
       e. IR Pass Optimization：应用 kernel fusion（如 fused MHA）提升子图推理性能。
       f. Deployment：优化后的子图部署到各 GPU。
    3. **在线推理（Ring Memory Offloading）**：
       a. 模型含 N 个 decoder layer，每个 layer 的 expert 参数在 CPU memory 存 N 份副本。
       b. GPU memory 划出 ring buffer，容量为 K 份 expert 参数副本 + dense 参数 buffer。
       c. 初始：从 CPU 加载前 K 层的 expert 参数到 GPU ring buffer。
       d. 计算第 i 层：GPU 执行 attention + FFN（含 MoE routing + expert FFN），耗时 T_compute。
       e. 释放：第 i 层计算完成后，释放 Pi 占据的 ring buffer slot。
       f. 异步加载：CPU→GPU copy engine 开始将第 (K+i) 层 expert 参数从 CPU memory 传输到刚释放的 slot（使用独立 CUDA stream），耗时 T_copy。
       g. T_copy 与 T_compute 部分重叠 —— 当 K 足够大且 decoder layer 数足够多时，重叠率极高。
    4. **输出**：生成的 token 序列返回用户。
    5. 效果：GPU memory 消耗降低 ≥30%，推理速度不受 CPU offloading 明显影响（图 12 显示重叠 offloading 的 compute time 仅略高于无 offloading）。
