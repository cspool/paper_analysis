## 神经渲染加速器（Neural Rendering Accelerator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 神经渲染加速器是为神经渲染 pipeline（NeRF/Instant-NGP/3DGS 等）定制的专用硬件，把射线采样、位置编码、MLP/球谐场计算、体积渲染/alpha 混合等步骤做成专用计算单元与存储层级以满足实时 3D 应用（AR/VR、机器人、数字孪生）。NeRArch-Sim 论文（ISCA 2026）汇总的代表性加速器（表 III）：MLP-based 的 ICARUS（0.017 FPS、40nm、16.5mm²、282.8mW）与 MetaVRain（110 FPS、28nm、20.25mm²、899mW）；Grid-based 的 NeuRex（19.72 FPS、28nm、1GHz、21.37mm²、6.1W）、CICERO、SRender；Primitive-based 的 GSCore（190 FPS、28nm、3.95mm²、870mW）、GS Processor（373 FPS、28nm、700MHz、2.43mm²、664mW）。每个加速器原来都是"单设计点"手工定制（one-off specialization），评估口径不一，这正是 NeRArch-Sim 统一模拟要解决的痛点。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 按 NeRArch-Sim 的统一分类学，神经渲染加速器由四类硬件模块构成：Sampling（采样/剔除单元）、Encoding（位置/哈希编码、地址生成、树归约、索引单元）、Field Computation（MLP 引擎 4 变体：SSA/MONB/SONB/Systolic array；加法树）、Blending（体积渲染单元 3 变体、Bitonic/Quick 排序、PE 阵列 3 变体、NRU/LuminCache 等），配可配置 SRAM（容量/组/端口/延迟）+ Ramulator 建模的 DRAM。运转流程例（NeuRex）：Field Sampler 把射线采样坐标从 DRAM 流式读入（469MB/frame）→ Encoding 的 hash 地址生成+Grid Cache/Subgrid Buffer 查表（不规则查找，bank conflict 开销 2.25%）→ MLP（systolic array 37 cycle 延迟、面积 5.4×10⁵ µm²）→ 体积渲染。图 12 的资源利用率分析显示：ICARUS 以 MLP 为瓶颈（PEU/MLP 持续高利用率、访存低），NeuRex 由 IGU 与 systolic array 主导，GSCore 先做预处理后由 VRU 主导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：专用 ASIC（ICARUS/NeuRex/CICERO/GSCore/GBU/GS Processor 等，多为 ISSCC/MICRO/ASPLOS/HPCA 论文）、或模拟/流片验证（GS Processor 为真实流片芯片，NeRArch-Sim 与之对比延迟误差 8.0%）。复现/设计工具：NeRArch-Sim（SystemC + Catapult HLS，模块化建模 20+ 模块，HLS 阶段 PPA 估计几分钟 vs 全 ASIC flow 数小时，面积/功率相对误差 4.72%~9.33%）；软件侧用 Nerfstudio 插桩取算子图，调度器 C++ 实现，28nm HPCP 工艺节点 + DeepScaleTool 归一化基线。NeRArch-Sim 已复现 11 个加速器（表 VII 端到端：面积平均误差 6.5%、FPS 3.4%、PSNR 2.6%；表 XIII 扩展案例：SRender 面积误差 5.3%、Lumina FPS 误差 7.8%、GS Processor 8.0%、Instant-3D 9.1%、GSArch 6.1%）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
