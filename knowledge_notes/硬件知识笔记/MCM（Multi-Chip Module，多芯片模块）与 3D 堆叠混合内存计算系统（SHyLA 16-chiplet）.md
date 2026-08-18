## MCM（Multi-Chip Module，多芯片模块）与 3D 堆叠混合内存计算系统（SHyLA 16-chiplet）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MCM（多芯片模块）把多个 die 封装进同一基板/封装，通过片间互连组成大系统。SHyLA 是 16-chiplet MCM：每个 chiplet = 7nm 计算 die + 3D 堆叠的 4-Hi DRAM 与 4-Hi PCM NVM 混合内存（并行堆叠、信号经 buffer die 重路由）。计算 die 上 DRAM/NVM 内存 bank 按 plane 组织，每 plane 与下方计算 tile 配对（DTile/NTile）。chiplet 间以 pipeline/tensor/expert 并行组成同构 MCM（避免数据并行以消除 Weight 复制）；通用处理器管理片间链路、类 NVIDIA Bluefield 网络处理器处理跨板通信与归约。比较：8× H800 / 4× MI300X GPU 集群按计算面积匹配（H800 8 die 814mm²、MI300X 4 die）作为 GPU baseline，vLLM 驱动。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：一个 LLM 请求 → 按 (pp=2, pt=8) 切分到 16 chiplet（QKV/FFN1 Weight 按宽切、AttnOut/FFN2 按高切 + all-reduce）→ 每个 chiplet 内 20 个 tile 并行执行分块 GEMM/GEMV → 数据从 3D 堆叠混合内存（Weight/KVCache 在 NVM plane、IA 在 DRAM plane）经 tile group 内专用链路/跨组 AXI 供给 MAC 阵列 → 输出部分和在 Attention Output/FFN2 经 ICNT（429GB/s）跨 chiplet 归约 → 逐 Transformer block 流水。DSE 在 (PD, pp, pt, pe, b) 部署空间选吞吐最优配置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 以 GPGPU-Sim 仿真（chiplet 跑相同分区并行、片间通信单独计算）+ CACTI-3DD 推导内存带宽/容量 + 3D-ICE 热分析 + 自研解析模型评估；架构参数（Table IV）：计算 die 7nm 400mm²、20 tile、1GHz、FP16 MAC 32×32、每 tile 4 MAC tree、全局输入 buffer 4MB、weight buffer/tile 1.2MB、output buffer/tile 100KB、ICNT 429GB/s。SHyLA 本体未开源（联网未找到仓库）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
