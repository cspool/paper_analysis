## DFVG: FPGA-Based Multi-Branch Speculative Decoding

- 属于硬件架构的实现是什么？实验比较什么？
  实现为FPGA上部署的Multi Compute Core Overlay Processor，用Verilog HDL设计，在Xilinx Vivado 2024.1中综合，运行于300MHz。Overlay Processor包含：(1) Multi-Core PE阵列：systolic PE array执行矩阵乘法，多个core的partial sum经parallel adder tree融合后累积到output buffer；(2) PE微架构：支持branch concatenation（多个weight buffer + 额外连线使PE可按branch选择正确路径）和DSP packing（每个DSP支持两个BF16×BF16乘法，利用BF16仅7-bit mantissa特性，计算吞吐翻倍）；(3) HBM通道全利用：weights和activations从HBM加载，计算和加载重叠（KERload = PE_Num × Data_width / Bandwidth，IFMload = KERload + CAS_Latency）；(4) KV-Cache管理：on-chip temp buffer按branch临时存储K/V，verified acceptance后prune无效分支，accepted tokens的KV通过contiguous allocation最大化利用on-chip RAM，block-based批量eviction；(5) Dynamic Token Management模块：监控GPU执行状态切换draft streams，Branch Management模块根据token confidence score决定下一轮draft数量。设计占V80 FPGA约89.6% LUT、90.9% FF、8192 DSP、18MB BRAM、67MB URAM。实验比较：(1) 资源利用率布局（PE占55.9% LUT）；(2) FPGA operator执行效率（矩阵乘法loading和computation均达86.2%-97.5% efficiency）；(3) 不同设备组合（U200/V80 FPGA + RTX4090/A100 GPU）的性能对比；(4) ablation study中MCore-Acc贡献（3.08× speedup）。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  未使用模拟器。FPGA实现使用Xilinx Vivado 2024.1综合+布局布线，部署在AMD V80 FPGA（HBM）和U200 FPGA（DDR）实卡上。GPU侧使用NVIDIA RTX 4090和A100实卡。功耗测量使用Xilinx xbutil和nvidia-smi每1ms采样。

- 模拟器模拟什么的性能，修改了什么。
  不适用（未使用模拟器，均为实卡部署）。FPGA bitstream通过Vivado生成，GPU kernel通过CUDA 12.1编译，跨设备通信使用PCIe Gen4×16（64GB/s）+ shared host CPU memory的ping-pong buffering机制。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/ShaoqiangLu/DFVG（MIT License）。FPGA Overlay Processor使用流程：
  1. 环境准备：Ubuntu 20.04，安装Xilinx Vivado 2024.1和XRT 2024.1：`source /opt/xilinx/xrt/setup.sh`
  2. 编译FPGA bitstream：`cd fpga/ && make synthesize && make implement`（约2-4小时）
  3. 编译GPU kernels：`cd gpu/ && make all`
  4. 下载模型：`python scripts/download_models.py`
  5. 运行实验：`python scripts/run_experiments.py --config configs/llama7b.yaml`
  6. 收集结果：`python scripts/collect_results.py --output results/`
  Overlay Processor在FPGA上执行draft model的forward pass，利用multi-branch并行生成多个候选token序列，通过PE array的branch concatenation和DSP packing最大化FPGA计算资源利用率。例如在V80上部署LLaMA-160M draft model，multi-core并行生成多个speculative branches，运行时功耗仅75W（远低于TDP 190W）。PCIe Gen4×16提供64GB/s带宽用于token传输，通过DMA和interrupt-driven coordination实现与GPU的异步通信。
