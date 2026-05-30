## MoE Heterogeneous Cost Model (MoE异构成本模型)

术语是什么？
MoE-CAP提出的覆盖全部异构资源的MoE系统成本模型，不同于现有benchmark仅按GPU使用计费。成本分为硬件购买成本和能耗成本两部分。硬件成本 C_hardware = C_GPU + C_CPU + C_Motherboard + C_DRAM + C_SSD，其中选择GPU变体隐含了HBM大小(C_HBM)、NVLink带宽(C_NVLink)和芯片内互联(C_C2M)的选择；选择CPU变体也限制了C_C2M；选择主板变体覆盖了PCIe成本。能耗成本 C_energy = [(P_GPU + P_CPU) + (P_C2M + P_PCIe + P_NVLink)] × R，其中R为服务器运行时长，覆盖计算功耗（GPU+CPU）、通信功耗（芯片内互联+PCIe+NVLink）。最终per-token成本 C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)。

从系统架构角度拆解术语：
成本模型在CAP评测流水线中的计算流程：(1) 硬件购买成本从硬件规格和市场价格获取，GPU-only部署下云厂商通常配套4× DRAM容量（如8×A100-80GB配2TB DDR5）和最新CPU（如AMD 9004系列），约$176,000/服务器；若选用低配CPU和减配内存可在GPU-only场景节省约$20,000/服务器；(2) 对MoE offloading系统需额外计算CPU能耗——AMD 777X峰值280W与A6000 Ada 300W可比，忽略CPU能耗会严重低估总成本；(3) 通信功耗覆盖CPU-to-DRAM、PCIe（GPU-CPU-SSD）和NVLink（GPU-GPU）；(4) per-token成本根据部署周期R分摊硬件成本+能耗成本。该模型揭示：CA型系统（如MoE-Infinity offloading）虽然硬件成本低（仅需消费级GPU+大容量DRAM），但以更高延迟换取低成本和高精度。

术语一般如何实现？
MoE-CAP的FastAPI服务在评测过程中自动采集硬件规格和功耗数据，计算C_hardware和C_energy，合成C_token。CAP雷达图的Cost轴即基于此模型的归一化值。未来工作包括在serverless端点[18]、弹性基础设施[43,45,34]和spot-instance定价[35,36,42]等真实云部署场景下验证。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
