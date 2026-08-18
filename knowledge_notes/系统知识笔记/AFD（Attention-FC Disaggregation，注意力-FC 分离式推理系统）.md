## AFD（Attention-FC Disaggregation，注意力-FC 分离式推理系统）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AFD 把 LLM 推理拆成两类特性不同的算子并部署到不同设备：memory-bound 的 decoding attention 与 KV cache 卸载到内存丰富的"加速器"（CPU 主机内存、HBM-PIM、GDDR-PIM、DIMM-PIM），compute-bound 的 FC（QKV Generation、投影、FFN）留在 GPU/NPU 上批处理。动机：decode 阶段的 attention 是带宽密集且不随 batch 复用（每个请求的 KV cache 独立），而 FC 受益于 batching；两类算子的最优硬件不同。跨设备数据经 PCIe 传输（prefill 产生的 K/V、每步 decode 的 QKV 与 attention 输出）。代表系统：NeuPIMs（NPU-PIM）、AttAcc（HBM-PIM）、PAPI、NEO（CPU 卸载）、CENT（GDDR6-PIM）、CHIME（DIMM-PIM）。CHIME 观察到"单纯增强加速器不一定提升吞吐"：给 DGX-A100 的 HBM 配 1×→16× 带宽的 PIM，GPT-175B+OpenR1 吞吐仅 +<1%，因为容量成为瓶颈——这是其 DRM 模型的出发点。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME 的 AFD 执行流（一次迭代、两个 sub-batch）：每个 sub-batch 的 QKV Generation 在 GPU 批处理 → decoding attention 在 CHIME-PIM 以 rank 粒度执行、prefill attention 在 GPU 并发 → CHIME 聚合所有请求 attention 输出经 PCIe 返回 GPU → GPU 批处理投影/FFN 等 FC。单个 batch 内 GPU 必须等 attention 完成后才能开始 FC（不可并行），因此 AFD 系统普遍用 sub-batch 调度让两个子批的 attention 与 FC 在两设备上并行。系统吞吐 = 两设备吞吐的较低者（DRM 推论 I）：GPU 侧受最大 batch（KV 容量）限制，加速器侧受带宽限制。AFD 的两类固有同步开销：数据同步（PCIe 传输 QKV/输出）与进度同步（两设备执行时间不等产生空闲气泡）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器选请求组子批（sub-batch 调度）、加速器存 KV cache 并执行 attention、GPU 批处理 FC、PCIe 异步传输 + 通信计算重叠（CHIME 的 rankset 粒度重叠）。加速器选型按"带宽与容量均衡"（DRM/Liebig's Law）：CPU 容量足带宽不足、HBM-PIM 带宽足容量不足、DIMM-PIM 较均衡且可经 DIMM 接口/CXL 扩展。局限：跨设备同步与通信是主要工程难点；batch 拆小会降低 GPU 利用率（CHIME 实测 HBM-PIM sub-batch 在 OPT-66B+Dolphin 上低于 GPU-only）。

Raptor 补充视角（ISCA'26，AFD 即 Attention-FFN Disaggregation，同族概念）：Raptor 提出把单个 MoE 层拆到 GPU 与 Raptor 两类设备——attention 放 HBM-class GPU（KV cache 流量主导、HBM 容量够）、FFN/expert 块放 Raptor（专家权重加载随 batch 与 top-k 缩放、吃最高带宽，~100TB/s）。与 CHIME 的"attention 卸载到 PIM 加速器"相反，这里把带宽密集的 FFN 放到高带宽 Raptor、把 KV 密集的 attention 留在容量大的 GPU；对 DeepSeek-V3 类 MoE，激活交换可融进专家并行的 all-to-all collectives（与 MegaScale-Infer、Step-3 一致），边界几乎免费。配套异构方向：投机解码的"draft 在 Raptor（100TB/s 压低串行 draft 关键路径延迟）+ verify 在 HBM GPU（计算受限、tensor core 友好）"，Raptor 前代 Corsair 已有该配对的规模化端到端加速部署先例，NVIDIA Vera Rubin（Rubin GPU + Groq 3 LPX）也采用类似异构组合。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
