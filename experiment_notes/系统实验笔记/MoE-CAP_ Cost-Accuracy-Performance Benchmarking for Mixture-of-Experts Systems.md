## MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems

- 属于Serving调度的实现是什么？实验比较什么？
  实现了一套自动化MoE系统评测流水线，在SGLang和HuggingFace Transformers中植入轻量级expert activation profiler，在每层MoE layer的路由器附近插入probe记录前向传播中的expert激活模式。基于vLLM、SGLang、MoE-Infinity、K-Transformers等serving框架评测CAP三维度（Cost/Accuracy/Performance）的权衡。实验比较：(1) 不同serving系统（SGLang vs K-Transformers vs MoE-Infinity）在Qwen3-30B-A3B上的解码延迟、硬件成本和准确率权衡（CAP雷达图）；(2) 量化vs offloading方法（SGLang-FP8, SGLang-AWQ vs MoE-Infinity）在Qwen3-235B-A22B上的吞吐、功耗和准确率权衡；(3) batch size（1-64）对expert稀疏性和实际带宽需求的定量影响（DeepSeek-V2-Lite, Qwen1.5-MoE, DeepSeek-R1）；(4) S-MBU在多节点推理（2节点×8 H20, InfiniBand 400 GB/s）的精度验证；(5) batch-size骤增压力测试（Microsoft Azure请求trace重放，Poisson分布）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80G-SXM4（1×, 2×, 8×）, A100-80G-PCIe（1×, 4×, 8×）, H20（8×）, A6000（4×, Ada 300W）, A5000, RTX 4090（450W）；Apple M3 Max；NVIDIA Orin AGX, Orin NX；DGX-H100（10200W）。多节点环境：2节点各配备8×NVIDIA H20 GPU，400 GB/s InfiniBand互联。CPU能耗参考：AMD 777X峰值280W。

- 开源Serving框架是什么。修改了什么。
  评测覆盖六个框架：vLLM, SGLang, MoE-Infinity, K-Transformers, HuggingFace Transformers, Accelerate。核心修改：(1) 在SGLang和HuggingFace Transformers的每层MoE layer路由器附近植入轻量级probe，记录forward pass中每个expert的激活状态（布尔变量𝟙[l,i]），以此计算S_activated = n_layer × S_attn + Σ_l Σ_i 𝟙[l,i] × S_expert；(2) 构建自动化评测流水线，用户提供系统和硬件详情即可自动完成模型加载、数据集评测和CAP指标计算，基于HuggingFace leaderboard设计；(3) 探针兼容CUDA graph编译以最小化性能干扰，最大overhead仅2.7%（TTFT +8ms, TPOT +4ms）；(4) 激活模式数据持久化为activation sheet以便后续复用，避免重复profiling。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/Auto-CAP/MoE-CAP。提供预构建Docker镜像和FastAPI-based CAP分析服务。使用流程：(1) 用户指定MoE模型（如Qwen3-30B-A3B）、serving框架（如SGLang）、硬件（如4×A6000）、数据集（如GSM8K）和batch size；(2) MoE-CAP自动加载模型到指定serving框架，挂载数据集和模型存储卷；(3) 推理每个forward pass时，CAP profiler（POST /cap-profiler）在每层MoE router后记录：router输出的top-k expert索引、每个expert的激活布尔值𝟙[l,i]、当前batch size、解码延迟——这些信息用于计算S_activated；(4) 从activation sheet计算精确的S-MBU = (S_activated + S_KV) / (TPOT × B_peak)，S-MFU = (T_token × (F_attn + 2N_router + 2k_expert × N_expert)) / F_peak；(5) 同时采集硬件成本（C_hardware = C_GPU + C_CPU + C_Motherboard + C_DRAM + C_SSD，覆盖所有异构资源）和能耗成本（C_energy = (P_GPU + P_CPU + P_C2M + P_PCIe + P_NVLink) × R），合成per-token cost C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)；(6) 所有请求结束后，GET /cap-results获取包含CAP雷达图的最终报告，展示Cost（$/token或W）、Accuracy（exact match/F1/win rate）、Performance（TPOT/吞吐/S-MBU/S-MFU）三维权衡对比。
