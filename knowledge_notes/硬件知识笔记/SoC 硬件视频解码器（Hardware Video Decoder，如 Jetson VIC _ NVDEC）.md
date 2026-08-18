## SoC 硬件视频解码器（Hardware Video Decoder，如 Jetson VIC / NVDEC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
硬件视频解码器是 SoC/GPU 上的专用解码引擎（Jetson 的 VIC、桌面/数据中心 NVIDIA GPU 的 NVDEC、移动 SoC 的 VPU），负责解析标准视频 bitstream 并重建像素帧：熵解码、解析运动矢量、运动补偿、反变换（IDCT 类）、像素域残差叠加、去块滤波等。相比软件解码，硬件解码器能量效率高出一个数量级以上，是移动/边缘设备视频播放的标准路径。SLICE 的架构前提是"消费标准 bitstream、不改码流"，因此能完整使用 SoC 硬件解码器；而 server-orchestrated 类方案因在码流中注入自定义元数据（SR 帧标记、HR 运动矢量）被迫回退到软件解码，硬件解码器闲置。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SLICE 的客户端管线（Fig.9）：① 硬件解码器解码 bitstream 重建帧（标准流程：解析 MV → 运动补偿 → 反变换得像素域残差 → 叠加重建）；② patch 分析（利用解码过程中本就要产出的 MV/残差，运行时开销可忽略）；③ patch 级 SR 推理（GPU）；④ 合并写 HR framebuffer。由于 bitstream 未修改，解码全走硬件，能量高效。注意：Jetson 硬件解码器并不对外暴露 MV/残差这些码流侧信号，SLICE 用扩展版 Compressed Video Reader（补丁化 FFmpeg）在主机端仿真提取这些信号，仅替换"引导信号的来源"，解码流程本身与硬件解码器兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为芯片内专用硬件模块（Jetson AGX Orin 的 VIC 视频图像合成器、NVIDIA GPU 的 NVDEC 引擎等），由厂商驱动（NVDEC 见 NVIDIA Video Codec SDK / FFmpeg hwaccel）调用。SLICE 论文（Discussion）指出其设计不依赖移动专用硬件：桌面/数据中心 GPU 的 NVDEC 同样消费运动矢量与残差等码流元数据，因此同一 codec 引导的 SR 管线可迁移到桌面工作站与服务器 GPU。论文未明确说明对 NVDEC 类引擎暴露元数据的依赖程度（记为信息缺口）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
