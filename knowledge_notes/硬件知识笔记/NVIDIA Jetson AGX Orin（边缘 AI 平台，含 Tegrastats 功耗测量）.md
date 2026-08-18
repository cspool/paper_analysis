## NVIDIA Jetson AGX Orin（边缘 AI 平台，含 Tegrastats 功耗测量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA Jetson AGX Orin 是面向边缘 AI 的嵌入式平台：Ampere 架构 GPU、12 核 Arm Cortex-A78AE CPU、统一内存架构、自带硬件视频解码/编码引擎（VIC）与 Tegrastats 功耗监测工具。它是 SLICE 的实测平台：端到端测量 SR 管线的吞吐（FPS）、质量（PSNR）与整机能量，而非仿真。Tegrastats 是 Jetson Linux 自带的功耗读取工具（读 SoC 内部电源管理寄存器），SLICE 对评估区间内的功率读数取平均估算能量消耗。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SLICE 在 Jetson AGX Orin 上的运行流程：① SoC 硬件解码器（VIC）解码 H.264 bitstream；② GPU 执行 patch 分析（AvgPool2D 生成 PSM、GPU TopK 选 patch）；③ GPU 跑 EDSR(FP16) 对选中 patch 组成的 batch 做 forward（unfold 聚批避免 CPU 往返）；④ GPU 上按行分带合并写 framebuffer；⑤ Tegrastats 测整机功耗。论文强调即使在统一内存架构上，CPU 与 GPU 逻辑内存空间之间仍存在数据移动，因此 SLICE 把聚合/推理/合并全放在 GPU 侧以最小化传输。测量结果：EDSR 全帧推理延迟为 bicubic 的 120.7×（基线动机）、SLICE 相对 per-frame 推理降 62.57% 能量（SR 推理能量降 78.56%、附加四阶段仅占约 15.86%）、intra 帧全帧处理 77.4ms（12.9 FPS）需 playout buffer 吸收。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Jetson AGX Orin 是商用开发板/模组（NVIDIA 官方文档 https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/），运行 Jetson Linux（Ubuntu 基座）与 CUDA/PyTorch；Tegrastats 是官方自带命令行工具（Jetson Linux Developer Guide，r35.5.0 文档）。SLICE 用它做真实设备测量（论文 [25][33] 引用），属于真实硬件部署验证而非模拟。论文未给出 Jetson 的具体型号/显存/功率档配置细节（记为论文未明确说明）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
