## FFmpeg 与 Compressed Video Reader（CVR，补丁化 FFmpeg 元数据提取工具）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FFmpeg 是广泛使用的开源多媒体框架（libavcodec/libavformat 等库，支持解码/编码/转码/滤镜/封装），提供完整软件解码管线。Compressed Video Reader（CVR，https://github.com/Yaojie-Shen/Compressed-Video-Reader，MIT）是基于 FFmpeg 构建的 H.264 码流元数据读取工具：通过给 FFmpeg 源码打补丁、重新编译，在 H.264 解码过程中导出每块的运动矢量与残差（频域系数，并可做反变换得到像素域残差）。它源自 Motion Vector Extractor（LukasBommes/mv-extractor）并修改。SLICE 用它解决"SoC 硬件解码器不暴露码流侧信号"的问题：硬件解码器照常标准解码，CVR 在主机端平行地模拟提取 MV/残差网格作为 patch 分析的引导信号——只替换信号来源，不改 bitstream、不改解码流程。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
CVR 的构建与使用流程（论文 [34] + 仓库 README）：
```
# 构建：下载 FFmpeg 源码 → 打补丁（新增元数据导出接口）→ configure/编译 FFmpeg → 构建 reader
# 使用（Python API）：
import compressed_video_reader as cv_reader
data = cv_reader.read_video(video_path="clip.mp4", with_residual=True)
# 输出每帧的运动矢量网格与残差网格（H.264 解码过程中导出）
# CLI 等价：cv_reader <video> <output>
```
在 SLICE 中，CVR 输出的 G^mv / G^pix / G^hf / G^t 网格即 Algorithm 2 中 Patch Statistics Maps 的输入；它只做"引导信号来源的仿真替代"，与片上硬件解码器兼容，剩余的 patch 分析/推理/合并全部在 GPU（PyTorch）完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现机制：修改 FFmpeg 的解码器源码（在 H.264 解码循环中导出宏块 MV 与残差变换系数），编译为定制 FFmpeg 库并由 Python 包装调用（安装脚本自动下载/打补丁/编译，CMake + C++ 构建）。SLICE 论文在 Jetson 主机端用它模拟硬件解码器未暴露的码流侧信号；论文未说明 CVR 提取的残差网格在部署到真实硬件解码器时的对应实现路径（记为信息缺口）。同类替代还有 NVIDIA NVDEC 的码流侧元数据（论文 Discussion 提及桌面 GPU 可用同一类元数据）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
