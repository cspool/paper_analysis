## Epoch AI 加速器（EVAS/奕行智能，ME/VE/DE 异构执行单元）

术语解释
EVAS Intelligence（奕行智能）的商用吞吐型 AI 加速器，32 核、1 GHz、已流片量产（2025 年 8 月发布、规模化量产），是论文 TISA 动态调度框架的主要评估平台与 TISA 二进制编码的目标硬件。每核集成三个异构引擎：ME（Matrix Engine 矩阵/张量引擎）、VE（Vector Engine 向量引擎）、DE（Data Engine 数据/DMA 引擎），配每核硬件调度器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文（Section VII-A）：Epoch 是吞吐型 AI 加速器，经高带宽互连从主机 CPU offload 计算密集型 kernel，与主机共享 48 GB DDR 内存；芯片 1 GHz 成功流片（tape-out），论文全部 Epoch 性能结果在真硅片上测量（W=8）。组织：32 核，每核 1.5 MB 本地内存，核间经片上共享 SRAM 通信（支持核间 tile 复用），片上 NoC 互连，参数/激活经 48 GB 全局 DDR 与主机交换。表 V 平台规格：32 核、峰值 256T FP16、48 GB GDDR6、1 TB/s 带宽。
- 异构单元结构对标表 I 的业界映射（NVIDIA Tensor/TMA+CUDA cores、AMD Matrix/SIMD+SDMA、TPU MXU/V-ALU+async DMA、Ascend Cube/VU+on-chip DMA、Tenstorrent SFPU/FPU+on-chip DMA）：ME 做张量算术、VE 做元素级与规约、DE 做 DMA/异步数据搬移。
- Web 佐证：奕行智能（EVAS Intelligence，北京）为 RISC-V 云端 AI 算力芯片企业（2022 年成立，CEO 刘珲），Epoch 为旗下首款 AI 算力芯片，走"RISC-V 开放指令集 + 类 TPU 架构 + VISA 虚拟指令集"路线，原生支持 FP8/NVFP4/MXFP4 与 PyTorch/vLLM/SGLang 等主流框架；ELink 高速互连（单芯片 3.2T 带宽）；2026 年 WAIC 发布首个 RISC-V AI 算力超节点（单柜 64–128 颗 Epoch、全对全 Scale-Up、10 万卡集群扩展）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TISA 是 Epoch 的软件-硬件契约：每核集成硬件调度器消费 TISA 描述符，无显式软件屏障地编排异构执行。ME 增加自定义 block 张量/矩阵指令、VE 扩展 tile 友好向量操作、DE 暴露 DMA 风格异步非阻塞描述符，全部遵守 TISA 接口（OpType/UnitMap/TileMem 供调度器做合法性检查与动态重叠）。
- 运转流程例子（FA3 一个 tile）：编译器生成的 TISA 二进制把每 tile 描述符（OpType=GEMM、UnitMap=(ME,1,affinity)、TileMem=(s_P 区间, Local)）交给 Epoch 每核调度器 → 路由到 ME 的 WQ → 与 ME 的 F_u 检查后提升 IQ → ME 执行 block 矩阵乘；同时 DE 的 tisa::load 描述符在另一 WQ 等待，VE 的 softmax 描述符同理，三引擎按就绪状态重叠执行。多核：编译期空间划分把独立 tile 组（attention head、batch 维）静态分配核，每核本地调度器独立，核间用共享 SRAM bank 更新触发的轻量 NoC 信号同步。
- 内存层级：每核 1.5 MB 本地内存 + 片上共享 SRAM（L2-local 级）+ 48 GB DDR（HBM 级作用）；TileMem 的 scope ∈ {Private, Local, Shared} 对应此层级。评估结果：Epoch(TISA) 比 A100 TensorRT 平均延迟低 1.46×，比 H100 FA3 在 head dim 128 硬件利用率高 26.4%（尽管 Epoch 带宽仅 H100 的 1/3.35）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：真硅片（非模拟器），32 核 + 每核 ME/VE/DE + 硬件调度器，1 GHz，W=8 验证；调度器 RTL 综合面积/功耗见表 IV（W=8 时 1.5M gates/0.25mm²/100mW）。
- 使用：主机 CPU 经高带宽互连 offload kernel，共享 48 GB DDR；软件栈 = TISA 语义算子库（tile 粒度、shape-parametric、双缓冲流水）+ 语义保持编译器自动生成 TISA 指令。移植到其他加速器需添加硬件调度器与 TISA（论文明确）。
- 开源情况：Epoch 为商用芯片，论文未给出 RTL/工具链开源链接，联网搜索未见公开仓库（web 新闻仅确认产品发布与量产状态）。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
