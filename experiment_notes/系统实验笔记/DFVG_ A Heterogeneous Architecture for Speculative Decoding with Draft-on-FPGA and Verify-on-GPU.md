## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于Serving调度的实现是什么？实验比较什么？
  提出Cross-Device Compact Pipeline Scheduling（跨设备紧密耦合流水线调度），实现FPGA draft generation和GPU verification的overlapped execution。核心机制：(1) Stage-Decoupled Scheduling：FPGA持续生成draft tokens（包含多分支动态长度），GPU同时验证上一轮候选tokens——当GPU验证时FPGA不等待而继续生成新draft，当FPGA未完成时GPU从prefix继续forward生成新tokens；(2) 五阶段流水线：FPGA生成候选→GPU立即验证→GPU返回决策→若FPGA未完成则GPU继续forward→若token被rejected则FPGA rollback同时GPU从prefix forward；(3) interrupt-driven coordination：FPGA写入新draft token IDs到shared host buffer（BAR空间），更新状态寄存器，通过interrupt通知CPU；CPU触发DMA使GPU从同一buffer读取并验证；(4) lightweight cross-device alignment：仅传输compact token IDs + status metadata，通信占wall time的1.08%-3.2%，不成为瓶颈。实验比较：(1) end-to-end speedup vs AR、SpS、DuoDecoding、SpecInfer（2.44×-3.26×）；(2) wall time breakdown（draft 92%-96%，verify 96%-98%，communication 1.08%-3.2%）；(3) energy efficiency（4.33×-5.79×）；(4) framework对比（vs vLLM、LLaMA.cpp、GPT-Fast）。

- 硬件平台是什么，配置是什么。
  服务器：Intel Xeon 4310 CPU + NVIDIA RTX 4090 GPU (512 Tensor Cores, 2230 MHz, 24GB DRAM, 1008 GB/s BW, 330 FP16 TOPS, 450W TDP) + AMD V80 FPGA (300 MHz, 10848 DSPs, 43MB SRAM, 64GB DRAM, 76 GB/s BW, 225W TDP)。附加配置：NVIDIA A100 GPU (432 Tensor Cores, 1410 MHz, 80GB, 1935 GB/s, 312 FP16 TOPS, 400W TDP) + AMD U200 FPGA (6480 DSPs, 84MB SRAM, 32+32GB DRAM, 51+820 GB/s DDR/HBM BW, 190W TDP)。互联：PCIe Gen4×16 (64GB/s)。

- 开源Serving框架是什么。修改了什么。
  不是基于现有Serving框架修改，而是从零构建的自定义heterogeneous runtime。CPU host controller使用C++编写，支持non-blocking draft和verify streams，通过interrupt和DMA协调FPGA-GPU流水线。FPGA侧使用Verilog HDL实现的custom micro-architecture + 扩展的compiler支持communication/dynamic draft configuration/rollback recovery指令。GPU侧使用CUDA 12.1实现TreeSort-Verify framework，含unified KV-cache management和multi-GPU synchronization via NCCL。Xilinx Runtime (XRT) 2024.1管理FPGA-PCIe通信。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/ShaoqiangLu/DFVG（MIT License）。使用流程：
  1. 环境：Ubuntu 20.04，CUDA 12.1，Xilinx Vivado 2024.1，XRT 2024.1
  2. Setup FPGA：`source /opt/xilinx/xrt/setup.sh`
  3. 编译FPGA+GPU：`cd fpga/ && make synthesize && make implement && cd ../gpu/ && make all`
  4. 下载模型：`python scripts/download_models.py`
  5. 运行实验：`python scripts/run_experiments.py --config configs/llama7b.yaml`
  Pipeline调度作用：FPGA draft model生成多分支候选tokens（在hardware budget约束下），GPU verify model以batch方式并行验证所有candidates并做acceptance decision，两者通过PCIe ping-pong buffer + interrupt实现tightly overlapped execution。例如Qwen3-8B/V80+RTX4090配置下，draft阶段占wall time的92%-96%，verify阶段占96%-98%（两者overlap执行，不串行相加），通信开销仅1.08%-3.2%。Pipeline通过Pipe-Overlap ablation达到3.26× speedup（相比无overlap的3.08×）。

